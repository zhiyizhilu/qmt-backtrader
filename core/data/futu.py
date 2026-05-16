"""富途数据处理器 - 从本地 .cache/FutuData 目录读取预存的行情数据

数据来源: 富途OpenD API (futu-api)，支持自动增量下载。
存储格式: .cache/FutuData/market/{symbol}/{year}_{period}.parquet
          .cache/FutuData/market_raw/{symbol}/{year}_{period}.parquet

当数据缺失时，会自动检测富途OpenD服务是否可用：
  - 如果服务已开启 → 自动增量下载缺失数据并缓存到本地
  - 如果服务未开启 → 报错并退出
"""
import os
import sys
import time
import logging
import threading
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

from core.data.base import DataProcessor
from core.cache import cache_manager


class FutuRateLimiter:
    """富途API请求频率限制器

    富途OpenD限制: 每30秒最多60次请求。
    使用滑动窗口算法控制请求频率，预留安全余量。
    """

    def __init__(self, max_requests: int = 50, window_seconds: float = 30.0):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._timestamps: List[float] = []

    def wait(self):
        """等待直到可以发送下一个请求

        清理过期的时间戳，如果窗口内请求数已达上限则等待。
        """
        with self._lock:
            now = time.time()
            # 清理超过滑动窗口的旧时间戳
            cutoff = now - self._window_seconds
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) >= self._max_requests:
                # 窗口已满，需要等待最早的请求过期
                oldest = self._timestamps[0]
                wait_time = oldest + self._window_seconds - now + 0.1
                if wait_time > 0:
                    self._lock.release()
                    try:
                        time.sleep(wait_time)
                    finally:
                        self._lock.acquire()
                    now = time.time()
                    cutoff = now - self._window_seconds
                    self._timestamps = [t for t in self._timestamps if t > cutoff]

            self._timestamps.append(time.time())


class FutuServiceError(Exception):
    """富途OpenD服务不可用异常"""
    pass


class FutuDataProcessor(DataProcessor):
    """富途数据处理器

    从本地 .cache/FutuData 目录读取预存的行情数据（parquet格式）。
    数据缺失时，自动检测富途OpenD服务并进行增量下载缓存。

    支持后复权（market/）和不复权（market_raw/）两种数据。
    """

    # 默认富途OpenD连接参数
    DEFAULT_HOST = '127.0.0.1'
    DEFAULT_PORT = 11111

    # 类级别共享限速器，所有实例共用，防止多线程突破频率限制
    _rate_limiter = FutuRateLimiter(max_requests=55, window_seconds=30.0)

    def __init__(self, fallback_to_simulated: bool = False, data_dir: str = '',
                 futu_host: str = '', futu_port: int = 0):
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        if data_dir:
            self._data_dir = data_dir
        else:
            base = os.environ.get('QMT_CACHE_DIR', os.path.join(os.getcwd(), '.cache'))
            self._data_dir = os.path.join(base, 'FutuData')

        self._futu_host = futu_host or self.DEFAULT_HOST
        self._futu_port = futu_port or self.DEFAULT_PORT

        self._raw_fetcher = FutuDataProcessor_Raw(self)

    # ── symbol 格式转换 ──

    @staticmethod
    def qmt_to_futu_code(symbol: str) -> Optional[str]:
        """将QMT格式 (601398.SH) 转换为富途格式 (SH.601398)"""
        if '.' not in symbol:
            return None
        parts = symbol.split('.')
        if len(parts) != 2:
            return None
        code, market = parts
        return f"{market}.{code}"

    @staticmethod
    def futu_to_qmt_code(futu_code: str) -> Optional[str]:
        """将富途格式 (SH.601398) 转换为QMT格式 (601398.SH)"""
        if '.' not in futu_code:
            return None
        parts = futu_code.split('.')
        if len(parts) != 2:
            return None
        market, code = parts
        return f"{code}.{market.upper()}"

    # ── 富途OpenD服务检测 ──

    def check_futu_service(self) -> bool:
        """检测富途OpenD服务是否可用

        尝试建立连接并获取全局状态，成功则返回True。
        """
        try:
            from futu import OpenQuoteContext, RET_OK
            ctx = OpenQuoteContext(host=self._futu_host, port=self._futu_port)
            try:
                ret, data = ctx.get_global_state()
                return ret == RET_OK
            finally:
                ctx.close()
        except ImportError:
            self.logger.error("futu-api 未安装，无法连接富途OpenD服务。请运行: pip install futu-api")
            return False
        except Exception as e:
            self.logger.debug(f"富途OpenD服务不可用: {e}")
            return False

    def _ensure_futu_service(self):
        """确保富途OpenD服务可用，不可用则抛出异常并退出"""
        if self.check_futu_service():
            return
        msg = (
            f"\n{'='*60}\n"
            f"错误: {self._futu_host}:{self._futu_port} 富途OpenD服务未开启！\n"
            f"数据缺失且无法自动下载，请执行以下操作后重试：\n"
            f"  1. 启动富途OpenD网关程序\n"
            f"  2. 确认OpenD监听地址为 {self._futu_host}:{self._futu_port}\n"
            f"{'='*60}"
        )
        self.logger.error(msg)
        raise FutuServiceError(msg)

    # ── 数据下载与缓存 ──

    # API调用超时时间（秒）
    API_TIMEOUT = 60
    # API调用最大重试次数
    API_MAX_RETRIES = 3

    def _download_kline(self, ctx, futu_code: str, start: str, end: str,
                        ktype, autype) -> Optional[pd.DataFrame]:
        """分页获取历史K线数据（带超时和重试）

        Args:
            ctx: OpenQuoteContext 实例
            futu_code: 富途格式代码 (如 SH.601398)
            start: 起始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            ktype: KLType 常量
            autype: AuType 常量 (HFQ/QFQ/NONE)

        Returns:
            原始DataFrame（含 time_key/open/high/low/close/volume 等列），失败返回 None
        """
        import concurrent.futures
        from futu import RET_OK

        all_dfs = []
        page_req_key = None

        while True:
            result = None
            last_error = None

            for attempt in range(1, self.API_MAX_RETRIES + 1):
                self._rate_limiter.wait()

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            ctx.request_history_kline,
                            code=futu_code,
                            start=start,
                            end=end,
                            ktype=ktype,
                            autype=autype,
                            max_count=1000,
                            page_req_key=page_req_key
                        )
                        result = future.result(timeout=self.API_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    last_error = f"API调用超时 ({self.API_TIMEOUT}秒)"
                    self.logger.warning(
                        f"富途API调用超时 ({attempt}/{self.API_MAX_RETRIES}): "
                        f"{futu_code} {start}~{end}"
                    )
                    if attempt < self.API_MAX_RETRIES:
                        time.sleep(2 * attempt)  # 递增等待
                    continue
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(
                        f"富途API调用异常 ({attempt}/{self.API_MAX_RETRIES}): {e}"
                    )
                    if attempt < self.API_MAX_RETRIES:
                        time.sleep(2 * attempt)
                    continue

                break  # 成功，跳出重试循环

            if result is None:
                self.logger.error(f"富途API请求失败（已重试{self.API_MAX_RETRIES}次）: {last_error}")
                return None

            ret, data, page_req_key = result

            if ret != RET_OK:
                # 区分"未知股票"（富途不支持退市股票）和其他错误
                err_msg = str(data)
                if '未知股票' in err_msg or '不存在的股票' in err_msg:
                    self.logger.debug(f"富途不支持该股票: {futu_code} - {err_msg}")
                    return None
                self.logger.error(f"富途API请求失败: {data}")
                return None

            if data is not None and not data.empty:
                all_dfs.append(data)

            if page_req_key is None:
                break

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True)

    def _save_kline_to_parquet(self, raw_df: pd.DataFrame, symbol: str,
                               year: int, period: str, sub_dir: str = 'market'):
        """将富途API返回的K线数据保存为parquet文件

        Args:
            raw_df: 富途API原始返回的DataFrame
            symbol: QMT格式代码
            year: 年份
            period: 周期后缀 (如 '1d', '1m')
            sub_dir: 'market' (后复权) 或 'market_raw' (不复权)
        """
        if raw_df is None or raw_df.empty:
            return

        # 转换 time_key 为 datetime 索引
        df = raw_df.copy()
        df['datetime'] = pd.to_datetime(df['time_key'])
        df = df.set_index('datetime')

        # 只保留标准列
        keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        df = df[keep_cols]

        for col in df.columns:
            df[col] = df[col].astype(float)

        df = df.sort_index()

        # 过滤只保留该年份的数据
        df = df[df.index.year == year]

        if df.empty:
            return

        # 保存
        symbol_dir = os.path.join(self._data_dir, sub_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        fpath = os.path.join(symbol_dir, f'{year}_{period}.parquet')
        df.to_parquet(fpath, index=True)
        self.logger.info(f"已缓存: {fpath} ({len(df)} 行)")

    def _get_missing_years(self, symbol: str, start_date: str, end_date: str,
                            period: str, sub_dir: str = 'market') -> List[int]:
        """获取缺失数据的年份列表

        检查指定范围内哪些年份的parquet文件不存在。
        """
        try:
            start_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year
        except Exception:
            return []

        period_suffix = self._map_period(period)
        symbol_dir = os.path.join(self._data_dir, sub_dir, symbol)

        missing_years = []
        for year in range(start_year, end_year + 1):
            file_path = os.path.join(symbol_dir, f"{year}_{period_suffix}.parquet")
            if not os.path.exists(file_path):
                missing_years.append(year)

        return missing_years

    def _download_missing_data(self, symbol: str, start_date: str, end_date: str,
                               period: str, sub_dir: str = 'market') -> bool:
        """增量下载缺失数据并缓存到本地

        对于后复权数据，采用重叠校验策略：
            1. 如果有缺失年份且是后复权数据
            2. 先检查是否有缺失年份在已有年份前面 → 全量重下载
            3. 如果只是后面有缺失，先下载一个重叠范围检查调整因子
            4. 如果重叠部分价格一致 → 只追加缺失年份
            5. 如果价格不一致 → 全量重下载所有年份

        Args:
            symbol: QMT格式代码
            start_date: 起始日期
            end_date: 结束日期
            period: 周期
            sub_dir: 'market' (后复权) 或 'market_raw' (不复权)

        Returns:
            True 如果下载成功（或有数据可读），False 如果失败
        """
        from futu import OpenQuoteContext, KLType, AuType

        # 检查服务
        self._ensure_futu_service()

        # symbol 格式转换
        futu_code = self.qmt_to_futu_code(symbol)
        if not futu_code:
            self.logger.error(f"无法将 {symbol} 转换为富途代码格式")
            return False

        # 确定缺失年份
        missing_years = self._get_missing_years(symbol, start_date, end_date, period, sub_dir)

        if not missing_years:
            self.logger.debug(f"{symbol}: 无需增量下载，所有年份文件已存在")
            return True

        self.logger.info(f"{symbol}: 检测到缺失数据，自动增量下载年份 {missing_years}")

        # 映射 period 到 KLType 和 autype
        ktype = self._map_period_to_kltype(period)
        autype = AuType.HFQ if sub_dir == 'market' else AuType.NONE
        period_suffix = self._map_period(period)

        # 对于后复权数据，采用重叠校验策略
        if sub_dir == 'market':
            try:
                return self._download_hfq_with_overlap_check(
                    symbol, futu_code, start_date, end_date, period, period_suffix,
                    ktype, autype, missing_years
                )
            except Exception as e:
                self.logger.error(f"{symbol}: 后复权重叠校验失败，回退到全量重下载: {e}")

        # 不复权数据，或者后复权重叠校验失败，按原有逻辑下载
        ctx = OpenQuoteContext(host=self._futu_host, port=self._futu_port)
        try:
            stock_unknown = False  # 标记该股票是否被富途识别为"未知股票"
            for year in missing_years:
                if stock_unknown:
                    # 已确认富途不支持该股票，后续年份直接跳过
                    self.logger.debug(f"  {symbol} {year}年 - 跳过（富途不支持该股票）")
                    continue

                year_start = f'{year}-01-01'
                year_end = f'{year}-12-31'

                self.logger.info(f"  下载 {symbol} {year}年 {sub_dir} 数据...")
                raw_df = self._download_kline(ctx, futu_code, year_start, year_end, ktype, autype)

                if raw_df is not None and not raw_df.empty:
                    self._save_kline_to_parquet(raw_df, symbol, year, period_suffix, sub_dir)
                else:
                    # 检查是否是"未知股票"导致的失败
                    # _download_kline 对未知股票会返回 None（非超时/非其他错误）
                    # 尝试首次请求即失败，很可能是富途不支持该股票
                    self.logger.warning(f"  {symbol} {year}年无数据（可能未上市或已退市）")
                    # 如果是第一个缺失年份就失败，标记为未知股票
                    if year == missing_years[0]:
                        stock_unknown = True
                        self.logger.info(f"  {symbol} - 富途不支持该股票，跳过剩余年份")
        except Exception as e:
            self.logger.error(f"增量下载 {symbol} 失败: {e}")
            return False
        finally:
            ctx.close()

        return True

    def _download_hfq_with_overlap_check(self, symbol: str, futu_code: str,
                                          start_date: str, end_date: str,
                                          period: str, period_suffix: str,
                                          ktype, autype,
                                          missing_years: list) -> bool:
        """下载后复权数据，采用重叠校验策略"""
        from futu import OpenQuoteContext, KLType, AuType

        ctx = OpenQuoteContext(host=self._futu_host, port=self._futu_port)
        try:
            # 获取已有的年份
            req_years = []
            try:
                start_year = pd.Timestamp(start_date).year
                end_year = pd.Timestamp(end_date).year
                req_years = list(range(start_year, end_year + 1))
            except Exception:
                pass

            existing_years = sorted(set(req_years) - set(missing_years))
            all_years = sorted(set(existing_years) | set(missing_years))

            # 检查是否有缺失年份在已有年份前面，是的话全量重下载
            missing_before_existing = False
            if existing_years:
                min_existing = min(existing_years)
                missing_before_existing = any(y < min_existing for y in missing_years)

            if missing_before_existing:
                self.logger.info(f"[{symbol}] 检测到缺失年份在已有年份前面，全量重下载...")
                return self._download_hfq_full_range(
                    symbol, futu_code, start_date, end_date, period, period_suffix,
                    ktype, autype, all_years
                )

            # 只有后面有缺失年份，尝试重叠校验
            if existing_years:
                try:
                    # 获取重叠检查的起始日期（已有最后一年的最后一个月）
                    max_existing = max(existing_years)
                    overlap_start = f"{max_existing}-12-01"
                    overlap_end = f"{max_existing}-12-31"

                    # 加载已有的重叠部分数据
                    existing_df = self._load_from_parquet(
                        symbol, overlap_start, overlap_end, period, sub_dir='market'
                    )

                    if existing_df is not None and not existing_df.empty:
                        # 下载包含重叠部分的新数据（从重叠检查起始到需要的结束）
                        new_overlap_start = overlap_start
                        new_overlap_end = end_date
                        self.logger.info(f"[{symbol}] 检查后复权调整因子: {new_overlap_start}~{new_overlap_end}")

                        overlap_raw_df = self._download_kline(ctx, futu_code, new_overlap_start, new_overlap_end, ktype, autype)
                        if overlap_raw_df is not None and not overlap_raw_df.empty:
                            # 提取重叠的部分
                            overlap_df = overlap_raw_df.copy()
                            overlap_df['datetime'] = pd.to_datetime(overlap_df['time_key'])
                            overlap_df = overlap_df.set_index('datetime')
                            overlap_df = overlap_df[(overlap_df.index >= overlap_start) & (overlap_df.index <= overlap_end)]
                            overlap_df = overlap_df[['close']]

                            # 对比收盘价
                            if not overlap_df.empty and not existing_df.empty:
                                common_idx = overlap_df.index.intersection(existing_df.index)
                                if len(common_idx) > 0:
                                    overlap_close = overlap_df.loc[common_idx, 'close'].sort_index()
                                    existing_close = existing_df.loc[common_idx, 'close'].sort_index()

                                    # 检查是否一致
                                    import numpy as np
                                    close_diff = np.abs(overlap_close - existing_close)
                                    max_diff = close_diff.max()

                                    if max_diff < 0.01:
                                        self.logger.info(f"[{symbol}] 调整因子一致，只追加缺失年份")
                                        # 重叠校验通过，只下载缺失年份
                                        stock_unknown = False
                                        for year in missing_years:
                                            if stock_unknown:
                                                continue
                                            year_start = f'{year}-01-01'
                                            year_end = f'{year}-12-31'
                                            self.logger.info(f"  下载 {symbol} {year}年数据...")
                                            raw_df = self._download_kline(ctx, futu_code, year_start, year_end, ktype, autype)
                                            if raw_df is not None and not raw_df.empty:
                                                self._save_kline_to_parquet(raw_df, symbol, year, period_suffix, 'market')
                                            else:
                                                self.logger.warning(f"  {symbol} {year}年无数据")
                                                if year == missing_years[0]:
                                                    stock_unknown = True
                                        return True
                                    else:
                                        self.logger.warning(f"[{symbol}] 调整因子变化 ({max_diff:.4f}>)，全量重下载")
                                        # 调整因子变了，全量重下载
                                        return self._download_hfq_full_range(
                                            symbol, futu_code, start_date, end_date, period, period_suffix,
                                            ktype, autype, all_years
                                        )
                except Exception as e:
                    self.logger.warning(f"[{symbol}] 重叠校验异常，回退到全量重下载: {e}")

            # 没有已有数据，或者重叠校验失败，全量重下载
            self.logger.info(f"[{symbol}] 全量重下载: {start_date}~{end_date}")
            return self._download_hfq_full_range(
                symbol, futu_code, start_date, end_date, period, period_suffix,
                ktype, autype, all_years
            )
        finally:
            ctx.close()

    def _download_hfq_full_range(self, symbol: str, futu_code: str,
                                  start_date: str, end_date: str,
                                  period: str, period_suffix: str,
                                  ktype, autype, years: list) -> bool:
        """全量下载后复权数据，覆盖所有年份"""
        from futu import OpenQuoteContext, KLType, AuType

        # 先删除旧的年份文件
        for year in years:
            fpath = os.path.join(self._data_dir, 'market', symbol, f"{year}_{period_suffix}.parquet")
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    self.logger.info(f"  删除旧文件: {fpath}")
                except Exception as e:
                    self.logger.warning(f"  删除失败 {fpath}: {e}")

        # 下载整个范围
        self.logger.info(f"  全量下载 {symbol} 后复权数据: {start_date}~{end_date}")
        ctx = OpenQuoteContext(host=self._futu_host, port=self._futu_port)
        try:
            raw_df = self._download_kline(ctx, futu_code, start_date, end_date, ktype, autype)
            if raw_df is not None and not raw_df.empty:
                # 按年份拆分并保存
                df = raw_df.copy()
                df['datetime'] = pd.to_datetime(df['time_key'])
                for year in years:
                    year_df = df[df['datetime'].dt.year == year].copy()
                    if not year_df.empty:
                        self._save_kline_to_parquet(year_df, symbol, year, period_suffix, 'market')
                return True
            else:
                self.logger.warning(f"  {symbol} 无数据返回")
                return False
        except Exception as e:
            self.logger.error(f"  全量下载 {symbol} 失败: {e}")
            return False
        finally:
            ctx.close()

    @staticmethod
    def _map_period_to_kltype(period: str):
        """映射周期字符串到富途KLType常量"""
        try:
            from futu import KLType
            mapping = {
                '1d': KLType.K_DAY,
                'day': KLType.K_DAY,
                'daily': KLType.K_DAY,
                '1m': KLType.K_1M,
                '1min': KLType.K_1M,
                'minute': KLType.K_1M,
                '5m': KLType.K_5M,
                '5min': KLType.K_5M,
                '15m': KLType.K_15M,
                '15min': KLType.K_15M,
                '30m': KLType.K_30M,
                '30min': KLType.K_30M,
                '60m': KLType.K_60M,
                '60min': KLType.K_60M,
            }
            return mapping.get(period, KLType.K_DAY)
        except ImportError:
            return None

    # ── 数据获取主入口 ──

    def get_data(self, symbol: str, start_date: str, end_date: str,
                 period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取后复权行情数据

        从 .cache/FutuData/market/{symbol}/ 目录按年份读取parquet文件并合并。
        如果本地数据缺失，自动检测富途OpenD服务并进行增量下载。

        Args:
            symbol: 股票代码，QMT格式如 '601398.SH'
            start_date: 起始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            period: 周期 '1d', '1m' 等

        Raises:
            FutuServiceError: 数据缺失且富途OpenD服务未开启
            ValueError: 数据缺失且无法下载
        """
        # 1. 尝试从本地缓存读取
        df = self._load_from_parquet(symbol, start_date, end_date, period, sub_dir='market')
        if df is not None and not df.empty:
            return df

        # 2. 本地缺失，尝试自动增量下载
        self.logger.info(f"{symbol}: 本地数据缺失，尝试自动增量下载...")
        try:
            success = self._download_missing_data(symbol, start_date, end_date, period, sub_dir='market')
        except FutuServiceError:
            raise  # 服务不可用，直接抛出
        except Exception as e:
            self.logger.error(f"{symbol}: 自动增量下载失败: {e}")
            raise ValueError(f"{symbol} 数据缺失且自动下载失败: {e}")

        if not success:
            raise ValueError(f"{symbol} 数据缺失且自动下载失败")

        # 3. 重新从本地读取
        df = self._load_from_parquet(symbol, start_date, end_date, period, sub_dir='market')
        if df is not None and not df.empty:
            return df

        raise ValueError(f"{symbol} 在 FutuData 中没有 {start_date}~{end_date} 的数据")

    def get_raw_data(self, symbol: str, start_date: str, end_date: str,
                     period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取不复权行情数据"""
        return self._raw_fetcher.get_data(symbol, start_date, end_date, period, **kwargs)

    def _load_from_parquet(self, symbol: str, start_date: str, end_date: str,
                           period: str, sub_dir: str = 'market') -> Optional[pd.DataFrame]:
        """从parquet文件加载数据

        目录结构: {data_dir}/{sub_dir}/{symbol}/{year}_{period}.parquet
        """
        symbol_dir = os.path.join(self._data_dir, sub_dir, symbol)
        if not os.path.isdir(symbol_dir):
            self.logger.debug(f"目录不存在: {symbol_dir}")
            return None

        try:
            start_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year
        except Exception:
            return None

        period_suffix = self._map_period(period)

        dfs = []
        for year in range(start_year, end_year + 1):
            file_path = os.path.join(symbol_dir, f"{year}_{period_suffix}.parquet")
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    if df is not None and not df.empty:
                        dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"读取失败: {file_path}, {e}")

        if not dfs:
            return None

        df = pd.concat(dfs)

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                self.logger.warning(f"无法解析日期索引: {symbol}")
                return None

        df = df.sort_index()

        df = df[(df.index >= start_date) & (df.index <= end_date)]

        if df.empty:
            return None

        keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        if keep_cols:
            df = df[keep_cols]

        return self.preprocess_data(df)

    @staticmethod
    def _map_period(period: str) -> str:
        """映射周期格式: '1d' -> '1d', '1m' -> '1m' 等"""
        mapping = {
            '1d': '1d', 'day': '1d', 'daily': '1d',
            '1m': '1m', '1min': '1m', 'minute': '1m',
            '5m': '5m', '5min': '5m',
            '15m': '15m', '15min': '15m',
            '30m': '30m', '30min': '30m',
            '60m': '60m', '60min': '60m',
        }
        return mapping.get(period, period)

    def get_stock_list(self, sector: str = '沪深A股') -> list:
        """FutuData不支持获取股票列表，返回空列表"""
        self.logger.warning("FutuDataProcessor 不支持获取股票列表")
        return []

    def get_industry_mapping(self, level: int = 1, stock_pool=None) -> dict:
        """FutuData不支持获取行业映射"""
        self.logger.warning("FutuDataProcessor 不支持获取行业映射")
        return {}


class FutuDataProcessor_Raw:
    """不复权行情数据获取器（FutuData 数据源）"""

    def __init__(self, processor: FutuDataProcessor):
        self._processor = processor

    def get_data(self, symbol: str, start_date: str, end_date: str,
                 period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取不复权行情数据

        本地缺失时自动增量下载（同 FutuDataProcessor.get_data 逻辑）。

        Raises:
            FutuServiceError: 数据缺失且富途OpenD服务未开启
            ValueError: 数据缺失且无法下载
        """
        # 1. 尝试从本地缓存读取
        df = self._processor._load_from_parquet(
            symbol, start_date, end_date, period, sub_dir='market_raw'
        )
        if df is not None and not df.empty:
            return df

        # 2. 本地缺失，尝试自动增量下载
        self._processor.logger.info(f"{symbol}: 本地不复权数据缺失，尝试自动增量下载...")
        try:
            success = self._processor._download_missing_data(
                symbol, start_date, end_date, period, sub_dir='market_raw'
            )
        except FutuServiceError:
            raise
        except Exception as e:
            self._processor.logger.error(f"{symbol}: 不复权数据自动增量下载失败: {e}")
            raise ValueError(f"{symbol} 不复权数据缺失且自动下载失败: {e}")

        if not success:
            raise ValueError(f"{symbol} 不复权数据缺失且自动下载失败")

        # 3. 重新从本地读取
        df = self._processor._load_from_parquet(
            symbol, start_date, end_date, period, sub_dir='market_raw'
        )
        if df is not None and not df.empty:
            return df

        raise ValueError(f"{symbol} 在 FutuData 中没有不复权数据 ({start_date}~{end_date})")
