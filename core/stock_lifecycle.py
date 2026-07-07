"""股票生命周期管理器 - 管理上市/退市时间

核心功能：
1. 批量获取股票的上市/退市时间
2. 判断股票是否已退市、晚上市
3. 计算有效日期范围（避免对退市/未上市股票的无效数据请求）
4. 持久化缓存，避免重复获取

数据源优先级：
1. QMT get_instrument_detail → OpenDate + ExpireDate（仅上市股）
2. 腾讯财经 HTTP API → 行情+K线（推算上市/退市时间）
3. akshare → 兜底备选（仅腾讯失败时使用）
4. QMT get_instrument_detail 返回 None → 确认退市
"""

import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class StockLifecycleManager:
    """股票生命周期管理器"""

    CACHE_FILE = 'stock_lifecycle.json'
    CACHE_MAX_AGE_DAYS = 30  # 缓存超过30天的条目需要更新

    def __init__(self, cache_dir: str = '.cache/lifecycle', xtdata=None):
        self._cache_dir = Path(cache_dir)
        self._cache_file = self._cache_dir / self.CACHE_FILE
        self._data: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._xtdata = xtdata
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self._load()

    def set_xtdata(self, xtdata):
        """设置QMT xtdata实例"""
        self._xtdata = xtdata

    # ── 公共查询接口 ──

    def get_list_date(self, symbol: str) -> Optional[str]:
        """获取上市日期，格式 'YYYY-MM-DD'，未知返回None"""
        info = self._data.get(symbol)
        return info.get('list_date') if info else None

    def get_delist_date(self, symbol: str) -> Optional[str]:
        """获取退市日期，格式 'YYYY-MM-DD'，未退市或日期未知返回None"""
        info = self._data.get(symbol)
        if not info:
            return None
        dd = info.get('delist_date')
        return None if dd == 'unknown' else dd

    def is_delisted(self, symbol: str) -> bool:
        """判断股票是否已退市"""
        info = self._data.get(symbol)
        if not info:
            return False
        dd = info.get('delist_date')
        # "unknown" 表示确认退市但日期未知，仍应视为已退市
        return dd is not None

    def is_listed_after(self, symbol: str, date: str) -> bool:
        """判断股票是否在指定日期之后上市"""
        list_date = self.get_list_date(symbol)
        if not list_date:
            return False
        try:
            return list_date > date[:10]
        except Exception:
            return False

    def is_delisted_before(self, symbol: str, date: str) -> bool:
        """判断股票是否在指定日期之前退市"""
        delist_date = self.get_delist_date(symbol)
        if not delist_date:
            return False
        try:
            return delist_date < date[:10]
        except Exception:
            return False

    def get_effective_date_range(self, symbol: str, start_date: str, end_date: str) -> Tuple[Optional[str], Optional[str]]:
        """获取股票在回测区间内的有效日期范围

        Returns:
            (effective_start, effective_end)
            如果股票在回测区间内无有效数据，返回 (None, None)
        """
        list_date = self.get_list_date(symbol)
        delist_date = self.get_delist_date(symbol)

        effective_start = start_date
        effective_end = end_date

        if list_date and list_date > start_date:
            effective_start = list_date

        if delist_date and delist_date < end_date:
            effective_end = delist_date

        if effective_end <= effective_start:
            return (None, None)

        return (effective_start, effective_end)

    # ── 批量更新 ──

    def batch_update(self, stock_list: List[str]) -> None:
        """批量更新股票生命周期数据

        流程：
        1. 从缓存加载已有数据
        2. 过滤出需要更新的股票（缓存中没有的，或缓存超过30天的）
        3. 先用 QMT get_instrument_detail 批量获取（快速，但退市股返回None）
        4. 对QMT返回None的股票，用腾讯财经补充（主数据源）
        5. 腾讯失败时，用akshare兜底
        6. 持久化到JSON文件
        """
        need_update = self._filter_stale_stocks(stock_list)
        if not need_update:
            self.logger.info(f"生命周期数据全部有效，无需更新")
            return

        self.logger.info(f"需要更新生命周期数据: {len(need_update)}/{len(stock_list)} 只股票")

        # 阶段1: QMT批量获取（快速）
        qmt_success = 0
        qmt_delisted_unknown = []  # QMT返回None的股票（可能是退市股）
        for symbol in need_update:
            if self._update_from_qmt(symbol):
                qmt_success += 1
            else:
                qmt_delisted_unknown.append(symbol)

        self.logger.info(
            f"QMT获取完成: {qmt_success} 只成功, "
            f"{len(qmt_delisted_unknown)} 只QMT无数据(可能是退市股)"
        )

        # 阶段2: 腾讯财经补充退市股（主数据源）
        if qmt_delisted_unknown:
            tencent_success = 0
            tencent_fail = 0
            for i, symbol in enumerate(qmt_delisted_unknown, 1):
                if self._update_from_tencent(symbol):
                    tencent_success += 1
                else:
                    tencent_fail += 1
                    # 腾讯失败 → akshare兜底
                    if self._update_from_akshare(symbol):
                        tencent_success += 1
                        tencent_fail -= 1
                    else:
                        # 腾讯+akshare均失败
                        # 注意: 无论是否已在缓存中都要刷新 update_time,
                        # 否则已存在的失败条目时间戳永远不过期刷新,
                        # 会导致每个回测都重复拉取这些查不到的退市股 (死循环)
                        existing = self._data.get(symbol, {})
                        old_list = existing.get('list_date')
                        old_delist = existing.get('delist_date')
                        # 若缓存中已有真实有效的上市/退市数据, 仅刷新时间戳, 不降级覆盖
                        if old_list or (old_delist and old_delist != 'unknown'):
                            self._data[symbol] = {
                                'list_date': old_list,
                                'delist_date': old_delist,
                                'source': existing.get('source', 'qmt_none'),
                                'update_time': datetime.now().strftime('%Y-%m-%d'),
                            }
                        else:
                            # 确实查不到 → 标记为退市但日期未知
                            self._data[symbol] = {
                                'list_date': old_list,
                                'delist_date': 'unknown',
                                'source': 'qmt_none',
                                'update_time': datetime.now().strftime('%Y-%m-%d'),
                            }
                if i % 50 == 0 or i == len(qmt_delisted_unknown):
                    self.logger.info(
                        f"腾讯财经补充进度: {i}/{len(qmt_delisted_unknown)} "
                        f"(成功 {tencent_success}, 失败 {tencent_fail})"
                    )

            self.logger.info(
                f"腾讯财经补充完成: 成功 {tencent_success}, 失败 {tencent_fail}"
            )

        self._save()

    # ── QMT数据获取 ──

    def _update_from_qmt(self, symbol: str) -> bool:
        """从QMT获取上市/退市时间

        Returns:
            True=成功获取到信息, False=QMT无此股票数据
        """
        if not self._xtdata:
            return False

        try:
            detail = self._xtdata.get_instrument_detail(symbol)
            if detail is None:
                return False

            # QMT正常上市股: ExpireDate=99999999 表示未退市
            open_date = detail.get('OpenDate')
            expire_date = detail.get('ExpireDate')

            list_date = None
            delist_date = None

            if open_date and str(open_date) != '0':
                try:
                    list_date = str(open_date)
                    # 格式化: 19910403 -> 1991-04-03
                    if len(list_date) == 8 and list_date.isdigit():
                        list_date = f"{list_date[:4]}-{list_date[4:6]}-{list_date[6:8]}"
                except Exception:
                    list_date = None

            if expire_date and str(expire_date) != '99999999' and str(expire_date) != '0':
                try:
                    delist_date = str(expire_date)
                    if len(delist_date) == 8 and delist_date.isdigit():
                        delist_date = f"{delist_date[:4]}-{delist_date[4:6]}-{delist_date[6:8]}"
                except Exception:
                    delist_date = None

            self._data[symbol] = {
                'list_date': list_date,
                'delist_date': delist_date,
                'source': 'qmt',
                'update_time': datetime.now().strftime('%Y-%m-%d'),
            }
            return True

        except Exception as e:
            self.logger.debug(f"QMT获取 {symbol} 生命周期数据失败: {e}")
            return False

    # ── 腾讯财经数据获取（主数据源） ──

    def _update_from_tencent(self, symbol: str) -> bool:
        """从腾讯财经获取上市/退市时间（直接HTTP请求，不依赖akshare）

        数据源:
        1. qt.gtimg.cn — 快速获取股票名称和状态
        2. web.ifzq.gtimg.cn — 历史K线，推算上市日/退市日

        Returns:
            True=成功获取到信息, False=获取失败
        """
        try:
            import requests as _req
        except ImportError:
            self.logger.debug("requests未安装，无法从腾讯财经获取数据")
            return False

        # 转换代码格式: 000005.SZ -> sz000005
        code, market = symbol.split('.') if '.' in symbol else (symbol, '')
        if market == 'SZ':
            prefix = 'sz'
        elif market == 'SH':
            prefix = 'sh'
        elif code.startswith(('6', '5', '9')):
            prefix = 'sh'
        else:
            prefix = 'sz'
        qt_code = f"{prefix}{code}"

        list_date = None
        delist_date = None
        source = 'tencent'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://finance.qq.com/',
        }

        # Step 1: 获取实时行情（检查股票状态和名称）
        stock_name = ''
        is_active = False
        try:
            url = f"https://qt.gtimg.cn/q={qt_code}"
            resp = _req.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                text = resp.text.strip()
                # 格式: v_sz000005="51~name~code~..."
                if '=' in text and '~' in text:
                    value_part = text.split('="', 1)[1].rstrip('";\n')
                    fields = value_part.split('~')
                    if len(fields) > 1:
                        stock_name = fields[1]
                        is_active = bool(stock_name and stock_name.strip())
                        # 字段38附近是上市日期(YYYYMMDD格式)，尝试提取
                        for idx in [38, 30, 39]:
                            if len(fields) > idx and fields[idx] and fields[idx].isdigit() and len(fields[idx]) == 8:
                                list_date = self._normalize_date(fields[idx])
                                break
            time.sleep(0.15)
        except Exception as e:
            self.logger.debug(f"腾讯行情获取 {symbol} 失败: {e}")

        # Step 2: 获取历史K线数据
        has_history = False
        try:
            url = (
                f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                f"?param={qt_code},day,1990-01-01,2099-12-31,100000,qfq"
            )
            resp = _req.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                kline_data = None
                if 'data' in data and qt_code in data['data']:
                    stock_data = data['data'][qt_code]
                    for key in ['qfqday', 'day']:
                        if key in stock_data:
                            kline_data = stock_data[key]
                            break

                if kline_data and len(kline_data) > 0:
                    has_history = True
                    # 上市日期 = 第一条K线的日期
                    first_date = str(kline_data[0][0])
                    if not list_date:
                        list_date = self._normalize_date(first_date)

                    # 退市判断
                    last_date = str(kline_data[-1][0])
                    normalized_last = self._normalize_date(last_date)

                    # 名称含"退" → 退市股
                    if stock_name and '退' in stock_name:
                        delist_date = normalized_last
                    elif not is_active and normalized_last:
                        # 不在当前行情 + 有历史 → 可能退市
                        last_dt = datetime.strptime(normalized_last, '%Y-%m-%d')
                        days_inactive = (datetime.now() - last_dt).days
                        if days_inactive > 30:
                            delist_date = normalized_last
            time.sleep(0.15)
        except Exception as e:
            self.logger.debug(f"腾讯K线获取 {symbol} 失败: {e}")

        if list_date or delist_date or has_history:
            existing = self._data.get(symbol, {})
            self._data[symbol] = {
                'list_date': list_date or existing.get('list_date'),
                'delist_date': delist_date or existing.get('delist_date'),
                'source': source,
                'update_time': datetime.now().strftime('%Y-%m-%d'),
            }
            return True

        return False

    # ── akshare数据获取（兜底备选） ──

    def _update_from_akshare(self, symbol: str) -> bool:
        """从akshare获取上市/退市时间（兜底备选，仅腾讯失败时使用）

        Returns:
            True=成功获取到信息, False=获取失败
        """
        try:
            import akshare as ak
        except ImportError:
            self.logger.debug("akshare未安装，无法补充退市股生命周期数据")
            return False

        list_date = None
        delist_date = None
        source = 'akshare'

        # 1. 获取上市时间
        try:
            ak_code = symbol.split('.')[0] if '.' in symbol else symbol
            info_df = ak.stock_individual_info_em(symbol=ak_code)
            if info_df is not None and not info_df.empty:
                for _, row in info_df.iterrows():
                    item = str(row.iloc[0]) if len(row) > 0 else ''
                    value = str(row.iloc[1]) if len(row) > 1 else ''
                    if '上市时间' in item and value and value != '-':
                        list_date = self._normalize_date(value)
                        break
            time.sleep(0.3)
        except Exception as e:
            self.logger.debug(f"akshare获取 {symbol} 上市时间失败: {e}")

        # 2. 尝试获取退市时间（通过历史行情最后交易日）
        try:
            ak_code = symbol.split('.')[0] if '.' in symbol else symbol
            hist_df = ak.stock_zh_a_hist(
                symbol=ak_code, period='daily',
                start_date='19900101', end_date='20991231', adjust=''
            )
            if hist_df is not None and not hist_df.empty:
                name_col = None
                for col in hist_df.columns:
                    if '股票' in col or '名称' in col:
                        name_col = col
                        break

                if name_col:
                    names = hist_df[name_col].dropna().astype(str)
                    for name in names:
                        if '退' in name:
                            last_date = str(hist_df.iloc[-1]['日期'] if '日期' in hist_df.columns else '')
                            if last_date:
                                delist_date = self._normalize_date(last_date)
                            break

                if not delist_date and not name_col:
                    date_col = '日期' if '日期' in hist_df.columns else hist_df.columns[0]
                    last_date_str = str(hist_df.iloc[-1][date_col])
                    last_dt = self._normalize_date(last_date_str)
                    if last_dt:
                        last_dt_obj = datetime.strptime(last_dt, '%Y-%m-%d')
                        days_inactive = (datetime.now() - last_dt_obj).days
                        if days_inactive > 90:
                            delist_date = last_dt

            time.sleep(0.3)
        except Exception as e:
            self.logger.debug(f"akshare获取 {symbol} 退市时间失败: {e}")

        if list_date or delist_date:
            existing = self._data.get(symbol, {})
            self._data[symbol] = {
                'list_date': list_date or existing.get('list_date'),
                'delist_date': delist_date or existing.get('delist_date'),
                'source': source,
                'update_time': datetime.now().strftime('%Y-%m-%d'),
            }
            return True

        return False

    # ── 辅助方法 ──

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """将各种日期格式标准化为 'YYYY-MM-DD'"""
        if not date_str or date_str == '-' or date_str == 'None':
            return None

        date_str = str(date_str).strip()

        if len(date_str) == 10 and date_str[4] == '-':
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except ValueError:
                pass

        if len(date_str) == 8 and date_str.isdigit():
            try:
                dt = datetime.strptime(date_str, '%Y%m%d')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        if len(date_str) == 10 and date_str[4] == '/':
            try:
                dt = datetime.strptime(date_str, '%Y/%m/%d')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        try:
            dt = datetime.strptime(date_str, '%Y年%m月%d日')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

        try:
            parsed = datetime.strptime(date_str[:19], '%Y-%m-%dT%H:%M:%S')
            return parsed.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            pass

        return None

    def _filter_stale_stocks(self, stock_list: List[str]) -> List[str]:
        """过滤出需要更新的股票"""
        need_update = []
        now_str = datetime.now().strftime('%Y-%m-%d')

        for symbol in stock_list:
            info = self._data.get(symbol)
            if not info:
                need_update.append(symbol)
                continue

            update_time = info.get('update_time', '')
            if not update_time:
                need_update.append(symbol)
                continue

            try:
                age_days = (datetime.now() - datetime.strptime(update_time, '%Y-%m-%d')).days
                if age_days > self.CACHE_MAX_AGE_DAYS:
                    need_update.append(symbol)
            except Exception:
                need_update.append(symbol)

        return need_update

    # ── 持久化 ──

    def _load(self) -> None:
        """从JSON文件加载缓存"""
        if not self._cache_file.exists():
            return

        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            self.logger.info(f"生命周期缓存已加载: {len(self._data)} 只股票")
        except Exception as e:
            self.logger.warning(f"加载生命周期缓存失败: {e}")
            self._data = {}

    def _save(self) -> None:
        """保存到JSON文件"""
        with self._lock:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                tmp_file = self._cache_file.with_suffix('.json.tmp')
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                tmp_file.replace(self._cache_file)
                self.logger.info(f"生命周期缓存已保存: {len(self._data)} 只股票")
            except Exception as e:
                self.logger.warning(f"保存生命周期缓存失败: {e}")

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        total = len(self._data)
        delisted = sum(1 for v in self._data.values() if v.get('delist_date'))
        has_list_date = sum(1 for v in self._data.values() if v.get('list_date'))
        return {
            'total': total,
            'delisted': delisted,
            'has_list_date': has_list_date,
            'active': total - delisted,
        }


# 模块级单例
_lifecycle_manager: Optional[StockLifecycleManager] = None
_lifecycle_lock = threading.Lock()


def get_lifecycle_manager(cache_dir: str = '.cache/lifecycle', xtdata=None) -> StockLifecycleManager:
    """获取全局生命周期管理器实例"""
    global _lifecycle_manager
    with _lifecycle_lock:
        if _lifecycle_manager is None:
            _lifecycle_manager = StockLifecycleManager(cache_dir=cache_dir, xtdata=xtdata)
        elif xtdata is not None:
            _lifecycle_manager.set_xtdata(xtdata)
        return _lifecycle_manager
