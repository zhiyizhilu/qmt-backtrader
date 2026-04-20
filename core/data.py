import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
import time


class DataProcessor(ABC):
    """数据处理器基类"""
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """获取数据"""
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        critical_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in data.columns]
        data = data.dropna(subset=critical_cols)
        return data


class QMTDataProcessor(DataProcessor):
    """QMT数据处理器
    
    当QMT不可用时，自动降级为模拟数据，方便策略测试。
    可通过 fallback_to_simulated=False 禁用降级行为。
    """
    
    FINANCIAL_TABLES = [
        'Balance', 'Income', 'CashFlow', 'Capital',
        'Holdernum', 'Top10holder', 'Top10flowholder', 'Pershareindex',
    ]

    def __init__(self, fallback_to_simulated: bool = False):
        try:
            from xtquant import xtdata
            self.xtdata = xtdata
        except ImportError:
            self.xtdata = None
            logging.getLogger(__name__).warning("xtquant not installed, using simulated data")
        self._fallback_to_simulated = fallback_to_simulated
        self._financial_data_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    def check_connection(self) -> bool:
        """检查QMT数据服务是否可用"""
        if not self.xtdata:
            return False
        try:
            self.xtdata.get_financial_index('000001.SZ')
            return True
        except Exception:
            return False

    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """从QMT获取数据，QMT不可用时降级为模拟数据"""
        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning(f"xtquant 未安装，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        try:
            start_time = start_date.replace('-', '')
            end_time = end_date.replace('-', '')

            self.xtdata.download_history_data(stock_code=symbol, period=period, start_time='', end_time='', incrementally=True)

            history_data = self.xtdata.get_market_data_ex(
                [], [symbol], period=period,
                start_time=start_time, end_time=end_time,
                count=-1, dividend_type="front"  # 前复权，与akshare/baostock保持一致
            )

            if symbol in history_data:
                df = history_data[symbol]
                
                # Check if data is empty
                if df.empty:
                    if self._fallback_to_simulated:
                        self.logger.warning(f"{symbol} 数据为空，使用模拟数据")
                        return self._generate_simulated_data(start_date, end_date, symbol)
                    raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据，请检查QMT数据是否同步")

                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S')
                    except (ValueError, TypeError):
                        try:
                            df.index = pd.to_datetime(df.index, format='%Y%m%d')
                        except (ValueError, TypeError):
                            if self._fallback_to_simulated:
                                self.logger.warning(f"{symbol} 索引转换失败，使用模拟数据")
                                return self._generate_simulated_data(start_date, end_date, symbol)
                            raise ValueError(f"{symbol} 索引转换失败")

                df = df[(df.index >= start_date) & (df.index <= end_date)]

                if df.empty:
                    if self._fallback_to_simulated:
                        self.logger.warning(f"{symbol} 数据为空，使用模拟数据")
                        return self._generate_simulated_data(start_date, end_date, symbol)
                    raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据，请检查QMT数据是否同步")

                return self.preprocess_data(df)
            else:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} 未在返回数据中，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 未在返回数据中，请检查QMT数据是否同步")
        except RuntimeError:
            raise
        except ValueError:
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"获取QMT数据失败: {e}，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError(f"获取QMT数据失败: {e}，请检查QMT是否已启动并登录")
    
    def download_financial_data(self, stock_list: List[str],
                                table_list: Optional[List[str]] = None,
                                start_time: str = '', end_time: str = '') -> None:
        """下载财务数据到本地缓存

        Args:
            stock_list: 股票代码列表，如 ['000001.SZ', '600000.SH']
            table_list: 财务报表列表，如 ['Balance', 'Income']，为空则下载全部
            start_time: 起始时间，如 '20230101'
            end_time: 结束时间，如 '20240101'
        """
        if not self.xtdata:
            self.logger.warning("xtquant 未安装，无法下载财务数据")
            return

        tables = table_list or self.FINANCIAL_TABLES

        try:
            if len(stock_list) <= 5:
                for stock in stock_list:
                    self.xtdata.download_financial_data(stock, tables)
            else:
                self.xtdata.download_financial_data2(
                    stock_list, tables,
                    start_time=start_time, end_time=end_time,
                )
            self.logger.info(f"财务数据下载完成: {len(stock_list)} 只股票, {tables}")
        except Exception as e:
            self.logger.error(f"财务数据下载失败: {e}")

    def get_financial_data(self, stock_list: List[str],
                           table_list: Optional[List[str]] = None,
                           start_time: str = '', end_time: str = '',
                           report_type: str = 'announce_time') -> Dict[str, Any]:
        """获取财务数据

        Args:
            stock_list: 股票代码列表
            table_list: 财务报表列表，为空则获取全部
            start_time: 起始时间
            end_time: 结束时间
            report_type: 报表筛选方式
                'report_time' - 按报告期（截止日期）
                'announce_time' - 按披露日期（回测时推荐，避免未来数据）

        Returns:
            dict: { stock1: { table1: DataFrame, ... }, ... }
        """
        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning("xtquant 未安装，返回空财务数据")
                return {}
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        tables = table_list or self.FINANCIAL_TABLES

        try:
            data = self.xtdata.get_financial_data(
                stock_list, tables,
                start_time=start_time, end_time=end_time,
                report_type=report_type,
            )
            return data
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"获取财务数据失败: {e}，返回空数据")
                return {}
            raise RuntimeError(f"获取财务数据失败: {e}")

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        """获取板块成分股列表

        Args:
            sector: 板块名称，如 '沪深A股', '上证50', '沪深300'

        Returns:
            股票代码列表
        """
        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning("xtquant 未安装，返回模拟股票列表")
                return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        try:
            stock_list = self.xtdata.get_stock_list_in_sector(sector)
            if stock_list:
                return stock_list
            try:
                self.xtdata.download_sector_data()
            except Exception:
                pass
            stock_list = self.xtdata.get_stock_list_in_sector(sector)
            return stock_list if stock_list else []
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"获取板块成分股失败: {e}，返回模拟列表")
                return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
            raise RuntimeError(f"获取板块成分股失败: {e}")

    def get_sector_list(self) -> List[str]:
        """获取所有板块列表"""
        if not self.xtdata:
            return []
        try:
            result = self.xtdata.get_sector_list() or []
            if result:
                return result
            try:
                self.xtdata.download_sector_data()
            except Exception:
                pass
            return self.xtdata.get_sector_list() or []
        except Exception:
            return []

    def get_industry_mapping(self, level: int = 1,
                             stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        """获取申万行业分类映射

        Args:
            level: 行业级别，1=一级行业，2=二级行业，3=三级行业
            stock_pool: 股票池，为空则使用沪深A股

        Returns:
            { stock_code: industry_name, ... } 映射字典
        """
        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning("xtquant 未安装，返回模拟行业映射")
                return {
                    '000001.SZ': '银行', '000002.SZ': '房地产',
                    '600000.SH': '银行', '600036.SH': '银行',
                }
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        try:
            sector_list = self.get_sector_list()
            if not sector_list:
                try:
                    client = self.xtdata.get_client()
                    client.down_all_sector_data()
                except Exception:
                    try:
                        self.xtdata.download_sector_data()
                    except Exception:
                        pass
                sector_list = self.xtdata.get_sector_list() or []
            sw_prefix = f'SW{level}'
            sw_sectors = [s for s in sector_list
                          if s.upper().startswith(sw_prefix.upper())
                          and '加权' not in s
                          and not s.upper().startswith('1000SW')
                          and not s.upper().startswith('300SW')
                          and not s.upper().startswith('500SW')
                          and not s.upper().startswith('HKSW')
                          and not s.upper().startswith('转债SW')]

            if stock_pool is None:
                stock_pool = self.get_stock_list('沪深A股')

            stock_set = set(stock_pool)
            mapping: Dict[str, str] = {}

            for sw_sector in sw_sectors:
                industry_name = sw_sector[len(sw_prefix):]
                members = self.xtdata.get_stock_list_in_sector(sw_sector) or []
                for stock in members:
                    if stock in stock_set:
                        mapping[stock] = industry_name

            self.logger.info(f"申万{level}级行业映射加载完成: {len(sw_sectors)} 个行业, {len(mapping)} 只股票")
            return mapping
        except RuntimeError:
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"获取行业映射失败: {e}，返回模拟数据")
                return {
                    '000001.SZ': '银行', '000002.SZ': '房地产',
                    '600000.SH': '银行', '600036.SH': '银行',
                }
            raise RuntimeError(f"获取行业映射失败: {e}")

    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        """获取股票分红数据

        Args:
            stock_list: 股票代码列表

        Returns:
            { stock_code: DataFrame(columns=[time, interest, ...]), ... }
            interest 列为每股派息金额
        """
        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning("xtquant 未安装，返回空分红数据")
                return {}
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        result = {}
        for stock in stock_list:
            try:
                df = self.xtdata.get_divid_factors(stock)
                if df is not None and not df.empty:
                    result[stock] = df
            except Exception:
                continue

        self.logger.info(f"分红数据加载完成: {len(result)}/{len(stock_list)} 只股票有数据")
        return result

    def get_instrument_detail(self, stock_code: str) -> Optional[Dict]:
        """获取合约基础信息

        Args:
            stock_code: 股票代码，如 '000001.SZ'

        Returns:
            合约信息字典
        """
        if not self.xtdata:
            return None
        try:
            return self.xtdata.get_instrument_detail(stock_code)
        except Exception:
            return None

    def _generate_simulated_data(self, start_date: str, end_date: str, symbol: str = None) -> pd.DataFrame:
        """生成模拟数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        seed = 42
        if symbol:
            seed = hash(symbol) % 10000
        rng = np.random.default_rng(seed)
        
        base_price = 10.0 + rng.random() * 5.0
        prices = []
        for i in range(len(date_range)):
            price_change = rng.normal(0, 0.02)
            base_price *= (1 + price_change)
            prices.append(base_price)
        
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': rng.integers(10000, 100000, len(date_range)),
        }, index=date_range)
        
        return self.preprocess_data(df)


class CSVDataProcessor(DataProcessor):
    """CSV数据处理器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    def get_data(self, symbol: str, start_date: str, end_date: str, file_path: str, **kwargs) -> pd.DataFrame:
        """从CSV文件获取数据

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV文件不存在: {file_path}")

        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV文件为空: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV文件格式错误: {file_path}, {e}")
        except Exception as e:
            raise ValueError(f"读取CSV文件失败: {file_path}, {e}")

        required_cols = {'open', 'high', 'low', 'close'}
        missing = required_cols - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"CSV文件缺少必要列: {missing}, 文件: {file_path}")

        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return self.preprocess_data(df)


def _convert_symbol_to_qmt(symbol: str) -> str:
    """将各种格式的股票代码转换为QMT格式 (如 000001.SZ)

    支持输入格式:
    - QMT格式: 000001.SZ, 600000.SH (直接返回)
    - 纯数字: 000001, 600000
    - baostock格式: sz.000001, sh.600000
    """
    if '.' in symbol and not symbol.startswith(('sz.', 'sh.', 'SZ.', 'SH.')):
        return symbol  # 已经是QMT格式

    # baostock格式 -> 纯数字+市场
    if symbol.lower().startswith(('sz.', 'sh.')):
        parts = symbol.split('.')
        market = parts[0].upper()
        code = parts[1]
        return f"{code}.{market}"

    # 纯数字 -> 根据代码规则判断市场
    code = symbol.replace('.', '')
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def _convert_symbol_to_akshare(symbol: str) -> str:
    """将QMT格式股票代码转换为akshare格式 (纯数字, 如 000001)"""
    return symbol.split('.')[0]


def _convert_symbol_to_baostock(symbol: str) -> str:
    """将QMT格式股票代码转换为baostock格式 (如 sz.000001)"""
    parts = symbol.split('.')
    if len(parts) == 2:
        code, market = parts
        return f"{market.lower()}.{code}"
    # 纯数字
    code = symbol
    if code.startswith(('6', '9')):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


class AKShareDataProcessor(DataProcessor):
    """AKShare数据处理器

    基于东方财富数据源，支持获取A股历史行情数据，数据范围远超QMT的一年限制。
    股票代码统一使用QMT格式 (000001.SZ)，内部自动转换。

    注意：akshare仅提供行情数据，财务/行业/分红等数据仍依赖QMT。
    """

    # 指数代码映射 (QMT格式 -> 中证指数代码)
    INDEX_CODE_MAP = {
        '000300.SH': '000300',  # 沪深300
        '000905.SH': '000905',  # 中证500
        '000852.SH': '000852',  # 中证1000
        '000016.SH': '000016',  # 上证50
    }

    PERIOD_MAP = {
        '1d': 'daily',
        'day': 'daily',
        'daily': 'daily',
        '1w': 'weekly',
        'week': 'weekly',
        'weekly': 'weekly',
        '1M': 'monthly',
        'month': 'monthly',
        'monthly': 'monthly',
    }

    def __init__(self, fallback_to_simulated: bool = True):
        try:
            import akshare as ak
            self._ak = ak
        except ImportError:
            self._ak = None
            logging.getLogger(__name__).warning("akshare not installed, pip install akshare")
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """从AKShare获取行情数据

        Args:
            symbol: QMT格式股票代码, 如 '000001.SZ'
            start_date: 起始日期 '2016-01-01'
            end_date: 结束日期 '2026-04-17'
            period: 周期, 仅支持日线 '1d'
        """
        if not self._ak:
            if self._fallback_to_simulated:
                self.logger.warning(f"akshare 未安装，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("akshare 未安装，请 pip install akshare")

        try:
            ak_code = _convert_symbol_to_akshare(symbol)
            ak_period = self.PERIOD_MAP.get(period, 'daily')
            start_dt = start_date.replace('-', '')
            end_dt = end_date.replace('-', '')

            self.logger.info(f"AKShare获取数据: {symbol} ({ak_code}), {start_date} ~ {end_date}")

            # 带重试的数据获取
            max_retries = 3
            df = None
            for retry in range(max_retries):
                try:
                    df = self._ak.stock_zh_a_hist(
                        symbol=ak_code,
                        period=ak_period,
                        start_date=start_dt,
                        end_date=end_dt,
                        adjust="qfq",  # 前复权，回测推荐
                    )
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = 2 * (retry + 1)
                        self.logger.warning(f"AKShare请求失败(第{retry+1}次): {e}, {wait_time}秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise

            if df is None or df.empty:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} AKShare数据为空，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

            # 中文列名映射为英文
            col_map = {
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
                '成交额': 'amount', '振幅': 'amplitude', '涨跌幅': 'pct_change',
                '涨跌额': 'change', '换手率': 'turnover',
            }
            df = df.rename(columns=col_map)

            # 设置日期索引
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                df = df.set_index('datetime')
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 过滤日期范围
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            if df.empty:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} 数据过滤后为空，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

            # 确保必要列存在且类型正确
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 只保留回测需要的列
            keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[keep_cols]

            # 限频：akshare建议每次调用间隔
            time.sleep(1.0)

            return self.preprocess_data(df)

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"AKShare获取数据失败: {e}，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError(f"AKShare获取数据失败: {e}")

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        """获取板块成分股列表

        支持沪深300、中证500、上证50等主要指数成分股。
        其他板块名称尝试通过akshare获取。
        """
        if not self._ak:
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

        try:
            # 常见板块名映射到指数代码
            sector_map = {
                '沪深300': '000300',
                '中证500': '000905',
                '中证1000': '000852',
                '上证50': '000016',
            }
            index_code = sector_map.get(sector)
            if index_code:
                df = self._ak.index_stock_cons_csindex(symbol=index_code)
                if df is not None and not df.empty:
                    # 成分股代码列可能是 '品种代码' 或 '股票代码'
                    code_col = None
                    for col_name in ['品种代码', '股票代码', 'code', '成分券代码']:
                        if col_name in df.columns:
                            code_col = col_name
                            break
                    if code_col:
                        codes = df[code_col].astype(str).tolist()
                        result = [_convert_symbol_to_qmt(c) for c in codes]
                        self.logger.info(f"AKShare获取 {sector} 成分股: {len(result)} 只")
                        return result

            # 如果不是已知指数，尝试获取全部A股
            if sector in ('沪深A股', 'A股', '全部A股'):
                df = self._ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    code_col = None
                    for col_name in ['代码', '股票代码', 'code']:
                        if col_name in df.columns:
                            code_col = col_name
                            break
                    if code_col:
                        codes = df[code_col].astype(str).tolist()
                        result = [_convert_symbol_to_qmt(c) for c in codes]
                        self.logger.info(f"AKShare获取 {sector}: {len(result)} 只")
                        return result

            self.logger.warning(f"AKShare不支持板块 '{sector}'，返回默认列表")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        except Exception as e:
            self.logger.warning(f"AKShare获取板块成分股失败: {e}")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

    def _generate_simulated_data(self, start_date: str, end_date: str, symbol: str = None) -> pd.DataFrame:
        """生成模拟数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        seed = 42
        if symbol:
            seed = hash(symbol) % 10000
        rng = np.random.default_rng(seed)
        base_price = 10.0 + rng.random() * 5.0
        prices = []
        for i in range(len(date_range)):
            price_change = rng.normal(0, 0.02)
            base_price *= (1 + price_change)
            prices.append(base_price)
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': rng.integers(10000, 100000, len(date_range)),
        }, index=date_range)
        return self.preprocess_data(df)


class BaoStockDataProcessor(DataProcessor):
    """BaoStock数据处理器

    免费开源证券数据平台，支持1990年至今的A股历史数据。
    股票代码统一使用QMT格式 (000001.SZ)，内部自动转换。

    注意：baostock仅提供行情数据，财务/行业/分红等数据仍依赖QMT。
    """

    PERIOD_MAP = {
        '1d': 'd', 'day': 'd', 'daily': 'd',
        '1w': 'w', 'week': 'w', 'weekly': 'w',
        '1M': 'm', 'month': 'm', 'monthly': 'm',
        '5m': '5', '5min': '5',
        '15m': '15', '15min': '15',
        '30m': '30', '30min': '30',
        '60m': '60', '60min': '60',
    }

    def __init__(self, fallback_to_simulated: bool = True):
        self._bs = None
        self._logged_in = False
        try:
            import baostock as bs
            self._bs = bs
        except ImportError:
            logging.getLogger(__name__).warning("baostock not installed, pip install baostock")
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def _ensure_login(self):
        """确保baostock已登录"""
        if not self._bs:
            return False
        if not self._logged_in:
            return self._do_login()
        return True

    def _do_login(self):
        """执行baostock登录"""
        try:
            # 先尝试登出（防止重复登录）
            try:
                self._bs.logout()
            except Exception:
                pass
            lg = self._bs.login()
            if lg.error_code == '0':
                self._logged_in = True
                return True
            else:
                self.logger.error(f"baostock登录失败: {lg.error_msg}")
                self._logged_in = False
                return False
        except Exception as e:
            self.logger.error(f"baostock登录异常: {e}")
            self._logged_in = False
            return False

    def _reconnect(self):
        """重新连接baostock（连接超时后调用）"""
        self._logged_in = False
        return self._do_login()

    def __del__(self):
        """析构时登出baostock"""
        if self._bs and self._logged_in:
            try:
                self._bs.logout()
                self._logged_in = False
            except Exception:
                pass

    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """从BaoStock获取行情数据

        Args:
            symbol: QMT格式股票代码, 如 '000001.SZ'
            start_date: 起始日期 '2016-01-01'
            end_date: 结束日期 '2026-04-17'
            period: 周期, 支持 '1d', '1w', '1M', '5m', '15m', '30m', '60m'
        """
        if not self._bs:
            if self._fallback_to_simulated:
                self.logger.warning(f"baostock 未安装，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("baostock 未安装，请 pip install baostock")

        if not self._ensure_login():
            if self._fallback_to_simulated:
                self.logger.warning(f"baostock 登录失败，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("baostock 登录失败")

        try:
            bs_code = _convert_symbol_to_baostock(symbol)
            bs_freq = self.PERIOD_MAP.get(period, 'd')

            # 日线字段
            if bs_freq in ('d', 'w', 'm'):
                fields = "date,open,high,low,close,volume,amount,turn,pctChg"
            else:
                fields = "date,time,open,high,low,close,volume,amount"

            self.logger.info(f"BaoStock获取数据: {symbol} ({bs_code}), {start_date} ~ {end_date}, freq={bs_freq}")

            # 带重连重试的数据查询
            max_retries = 2
            rs = None
            for retry in range(max_retries + 1):
                rs = self._bs.query_history_k_data_plus(
                    code=bs_code,
                    fields=fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=bs_freq,
                    adjustflag="2",  # 前复权
                )
                if rs.error_code == '0':
                    break
                # 检测登录过期错误，自动重连
                if '未登录' in str(rs.error_msg) or 'login' in str(rs.error_msg).lower():
                    if retry < max_retries:
                        self.logger.warning(f"BaoStock连接超时，正在重连... (第{retry+1}次)")
                        time.sleep(1)
                        if self._reconnect():
                            continue
                # 其他错误直接退出
                break

            if rs.error_code != '0':
                if self._fallback_to_simulated:
                    self.logger.warning(f"BaoStock查询失败: {rs.error_msg}，使用模拟数据: {symbol}")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise RuntimeError(f"BaoStock查询失败: {rs.error_msg}")

            # 收集数据
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} BaoStock数据为空，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

            df = pd.DataFrame(data_list, columns=rs.fields)

            # 类型转换 (baostock返回的都是字符串)
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 处理日期索引
            if 'date' in df.columns:
                if 'time' in df.columns:
                    # 分钟线: 合并 date + time
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].str[:6].apply(
                        lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}" if len(x) >= 6 else x
                    ), errors='coerce')
                else:
                    df['datetime'] = pd.to_datetime(df['date'])
                df = df.set_index('datetime')

            # 去除空行
            df = df.dropna(subset=['close'])

            if df.empty:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} 数据清洗后为空，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 数据清洗后为空")

            # 只保留回测需要的列
            keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[keep_cols]

            # 限频
            time.sleep(0.1)

            return self.preprocess_data(df)

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"BaoStock获取数据失败: {e}，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError(f"BaoStock获取数据失败: {e}")

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        """获取板块成分股列表

        支持沪深300、中证500等主要指数。
        """
        if not self._bs:
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

        if not self._ensure_login():
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

        try:
            # 常见板块名映射到baostock指数代码
            sector_map = {
                '沪深300': 'sh.000300',
                '上证50': 'sh.000016',
                '中证500': 'sh.000905',
                '中证1000': 'sh.000852',
            }

            bs_index_code = sector_map.get(sector)
            if bs_index_code:
                # baostock没有通用的指数成分股查询，只有特定接口
                rs = None
                if sector == '沪深300':
                    rs = self._bs.query_hs300_stocks()
                elif sector == '上证50':
                    rs = self._bs.query_sz50_stocks()
                elif sector == '中证500':
                    rs = self._bs.query_zz500_stocks()

                # 检测登录过期并重连
                if rs and '未登录' in str(rs.error_msg):
                    self.logger.warning("BaoStock获取成分股时连接超时，正在重连...")
                    if self._reconnect():
                        if sector == '沪深300':
                            rs = self._bs.query_hs300_stocks()
                        elif sector == '上证50':
                            rs = self._bs.query_sz50_stocks()
                        elif sector == '中证500':
                            rs = self._bs.query_zz500_stocks()

                if rs and rs.error_code == '0':
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        code_col = 'code' if 'code' in df.columns else rs.fields[0]
                        codes = df[code_col].tolist()
                        result = [_convert_symbol_to_qmt(c) for c in codes]
                        self.logger.info(f"BaoStock获取 {sector} 成分股: {len(result)} 只")
                        return result

            # 全部A股
            if sector in ('沪深A股', 'A股', '全部A股'):
                rs = self._bs.query_stock_basic()
                if rs and rs.error_code == '0':
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        code_col = 'code' if 'code' in df.columns else rs.fields[0]
                        codes = df[code_col].tolist()
                        result = [_convert_symbol_to_qmt(c) for c in codes]
                        self.logger.info(f"BaoStock获取 {sector}: {len(result)} 只")
                        return result

            self.logger.warning(f"BaoStock不支持板块 '{sector}'，返回默认列表")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        except Exception as e:
            self.logger.warning(f"BaoStock获取板块成分股失败: {e}")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

    def _generate_simulated_data(self, start_date: str, end_date: str, symbol: str = None) -> pd.DataFrame:
        """生成模拟数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        seed = 42
        if symbol:
            seed = hash(symbol) % 10000
        rng = np.random.default_rng(seed)
        base_price = 10.0 + rng.random() * 5.0
        prices = []
        for i in range(len(date_range)):
            price_change = rng.normal(0, 0.02)
            base_price *= (1 + price_change)
            prices.append(base_price)
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': rng.integers(10000, 100000, len(date_range)),
        }, index=date_range)
        return self.preprocess_data(df)


def create_data_processor(data_source: str = 'qmt', fallback_to_simulated: bool = True) -> DataProcessor:
    """数据处理器工厂函数

    Args:
        data_source: 数据源，可选 'qmt', 'akshare', 'baostock'
        fallback_to_simulated: 数据获取失败时是否降级为模拟数据

    Returns:
        DataProcessor 实例
    """
    if data_source == 'akshare':
        return AKShareDataProcessor(fallback_to_simulated=fallback_to_simulated)
    elif data_source == 'baostock':
        return BaoStockDataProcessor(fallback_to_simulated=fallback_to_simulated)
    elif data_source == 'qmt':
        return QMTDataProcessor(fallback_to_simulated=fallback_to_simulated)
    else:
        raise ValueError(f"不支持的数据源: {data_source}，可选: 'qmt', 'akshare', 'baostock'")
