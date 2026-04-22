import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging


class FinancialDataCache:
    """财务数据缓存

    预加载所有需要的财务数据，按披露日期排序，
    支持快速查询指定日期前已披露的最新财报。

    数据结构:
        _data: {
            stock1: {
                table1: pd.DataFrame (index=日期, columns=字段),
                table2: pd.DataFrame,
            },
            ...
        }
    """

    def __init__(self, financial_data: Dict[str, Any] = None):
        self._data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._loaded = False
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        if financial_data:
            self.load(financial_data)

    def load(self, financial_data: Dict[str, Any]) -> None:
        """加载财务数据

        Args:
            financial_data: xtdata.get_financial_data 返回的原始数据
                格式: { stock1: { table1: DataFrame, ... }, ... }
        """
        for stock_code, tables in financial_data.items():
            if stock_code not in self._data:
                self._data[stock_code] = {}

            if not isinstance(tables, dict):
                continue

            for table_name, df in tables.items():
                if df is None:
                    continue
                if isinstance(df, pd.DataFrame):
                    df = self._ensure_datetime_index(df, stock_code, table_name)
                    if not df.empty:
                        sorted_df = df.sort_index()
                        self._data[stock_code][table_name] = sorted_df
                    else:
                        self._data[stock_code][table_name] = df
                else:
                    try:
                        converted = pd.DataFrame(df)
                        converted = self._ensure_datetime_index(converted, stock_code, table_name)
                        if not converted.empty:
                            converted = converted.sort_index()
                        self._data[stock_code][table_name] = converted
                    except Exception:
                        continue

        self._loaded = True
        stock_count = len(self._data)
        self.logger.info(f"财务数据缓存加载完成: {stock_count} 只股票")

    def _ensure_datetime_index(self, df: pd.DataFrame,
                                stock_code: str = '', table_name: str = '') -> pd.DataFrame:
        """确保 DataFrame 的索引是 DatetimeIndex

        如果索引不是 DatetimeIndex，尝试从已知的日期列中构建。
        这对从 parquet 缓存加载的旧格式数据尤其重要。
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # 尝试从常见日期列构建 DatetimeIndex
        # 优先使用公告日期列（避免未来数据），其次使用报告期列
        for col in ['announce_date', 'm_anntime', 'pubDate', '公告日期',
                     'report_date', 'm_timetag', 'statDate', '报告期', 'index']:
            if col in df.columns:
                try:
                    dt_values = pd.to_datetime(df[col], errors='coerce')
                    valid = dt_values.notna()
                    if valid.any():
                        df = df[valid].copy()
                        df.index = dt_values[valid]
                        # 删除已用作索引的日期列，避免重复
                        # 但保留非索引用途的日期列（如 announce_date 仅用于索引时删除，
                        # 但如果数据源本来就将其作为数据列，则保留）
                        # 安全策略：仅删除 parquet 还原时产生的 'index' 列
                        if col == 'index':
                            df = df.drop(columns=['index'], errors='ignore')
                        return df
                except Exception:
                    continue

        # 尝试将整数索引转为日期（如 20231231 格式）
        if len(df) > 0 and df.index.dtype in ('int64', 'object'):
            try:
                dt_values = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
                if dt_values.notna().any():
                    valid = dt_values.notna()
                    df = df[valid].copy()
                    df.index = dt_values[valid]
                    return df
            except Exception:
                pass

        # 所有尝试失败，记录警告
        if stock_code and table_name:
            self.logger.debug(
                f'[_ensure_datetime_index] {stock_code}.{table_name} '
                f'无法构建DatetimeIndex, index类型={type(df.index).__name__}'
            )

        return df

    def get_stocks(self) -> List[str]:
        """获取已缓存的股票列表"""
        return list(self._data.keys())

    def get_tables(self, stock_code: str) -> List[str]:
        """获取指定股票已缓存的报表列表"""
        stock_data = self._data.get(stock_code, {})
        return list(stock_data.keys())

    def get_latest(self, stock_code: str, table_name: str,
                   date: datetime.date, field: Optional[str] = None) -> Any:
        """获取指定日期前已披露的最新财报数据

        关键：只返回 date 之前已披露的数据，避免未来数据（look-ahead bias）。

        Args:
            stock_code: 股票代码
            table_name: 报表名称，如 'Balance', 'Income'
            date: 查询日期，只返回此日期之前已披露的数据
            field: 字段名，为空则返回整行

        Returns:
            字段值或整行Series，无数据返回None
        """
        stock_data = self._data.get(stock_code, {})
        table_df = stock_data.get(table_name)

        if table_df is None or table_df.empty:
            return None

        if not isinstance(table_df.index, pd.DatetimeIndex):
            try:
                table_df.index = pd.to_datetime(table_df.index)
                self._data[stock_code][table_name] = table_df
            except Exception as e:
                self.logger.debug(
                    f'[get_latest] {stock_code}.{table_name} 索引转DatetimeIndex失败: {e}, '
                    f'index类型={type(table_df.index).__name__}, '
                    f'前3个值={table_df.index[:3].tolist() if len(table_df) > 0 else "empty"}'
                )
                return None

        mask = table_df.index.date <= date
        available = table_df[mask]

        if available.empty:
            # 调试：输出数据范围与查询日期的关系
            if not table_df.empty:
                self.logger.debug(
                    f'[get_latest] {stock_code}.{table_name} 无可用数据: '
                    f'查询日期={date}, 数据日期范围={table_df.index[0].date()}~{table_df.index[-1].date()}, '
                    f'数据行数={len(table_df)}'
                )
            return None

        latest_row = available.iloc[-1]

        if field:
            if field in latest_row.index:
                val = latest_row[field]
                if pd.isna(val):
                    return None
                return val
            return None

        return latest_row

    def get_latest_multi_fields(self, stock_code: str, table_name: str,
                                date: datetime.date, fields: List[str]) -> Dict[str, Any]:
        """获取指定日期前已披露的最新财报的多个字段

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            date: 查询日期
            fields: 字段名列表

        Returns:
            { field1: value1, field2: value2, ... }
        """
        result = {}
        stock_data = self._data.get(stock_code, {})
        table_df = stock_data.get(table_name)

        if table_df is None or table_df.empty:
            return {f: None for f in fields}

        if not isinstance(table_df.index, pd.DatetimeIndex):
            try:
                table_df.index = pd.to_datetime(table_df.index)
                self._data[stock_code][table_name] = table_df
            except Exception:
                return {f: None for f in fields}

        mask = table_df.index.date <= date
        available = table_df[mask]

        if available.empty:
            return {f: None for f in fields}

        latest_row = available.iloc[-1]

        for field in fields:
            if field in latest_row.index:
                val = latest_row[field]
                result[field] = None if pd.isna(val) else val
            else:
                result[field] = None

        return result

    def get_history(self, stock_code: str, table_name: str,
                    date: datetime.date, count: int = 4,
                    field: Optional[str] = None) -> Any:
        """获取指定日期前已披露的最近N期财报数据

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            date: 查询日期
            count: 期数
            field: 字段名，为空则返回整行

        Returns:
            field非空时返回list，否则返回DataFrame
        """
        stock_data = self._data.get(stock_code, {})
        table_df = stock_data.get(table_name)

        if table_df is None or table_df.empty:
            return [] if field else pd.DataFrame()

        if not isinstance(table_df.index, pd.DatetimeIndex):
            try:
                table_df.index = pd.to_datetime(table_df.index)
                self._data[stock_code][table_name] = table_df
            except Exception:
                return [] if field else pd.DataFrame()

        mask = table_df.index.date <= date
        available = table_df[mask].tail(count)

        if available.empty:
            return [] if field else pd.DataFrame()

        if field:
            if field in available.columns:
                vals = available[field].tolist()
                return [v for v in vals if not pd.isna(v)]
            return []

        return available


class FinancialDataAdapter:
    """财务数据适配器 - 提供策略层面的财报数据访问

    核心功能：
    1. 时间对齐：确保回测中只使用已披露的财报数据
    2. 筛选接口：支持基于财务条件的股票筛选
    3. 与 StrategyLogic 集成：通过 set_financial_data_adapter 注入

    使用方式：
        cache = FinancialDataCache(raw_data)
        adapter = FinancialDataAdapter(cache)
        strategy.set_financial_data_adapter(adapter)

        # 在策略中
        pe = self.get_financial_field('000001.SZ', 'Pershareindex', 'eps_diluted')
        selected = self.screen_stocks(lambda s: adapter.get_financial_field(s, 'Income', 'total_operate_income') > 1e10)
    """

    def __init__(self, cache: FinancialDataCache):
        self._cache = cache
        self._current_date: Optional[datetime.date] = None
        self._industry_mapping: Dict[str, str] = {}
        self._dividend_data: Dict[str, pd.DataFrame] = {}
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    @property
    def cache(self) -> FinancialDataCache:
        return self._cache

    def set_current_date(self, date: datetime.date) -> None:
        """设置当前回测日期（由框架在每个bar调用）"""
        self._current_date = date

    def get_current_date(self) -> Optional[datetime.date]:
        return self._current_date

    def set_industry_mapping(self, mapping: Dict[str, str]) -> None:
        """设置行业分类映射

        Args:
            mapping: { stock_code: industry_name, ... }
        """
        self._industry_mapping = mapping or {}

    def get_industry(self, stock_code: str) -> Optional[str]:
        """获取指定股票的行业分类

        Args:
            stock_code: 股票代码

        Returns:
            行业名称，无数据返回None
        """
        return self._industry_mapping.get(stock_code)

    def get_industry_mapping(self) -> Dict[str, str]:
        """获取完整的行业分类映射"""
        return dict(self._industry_mapping)

    def set_dividend_data(self, dividend_data: Dict[str, pd.DataFrame]) -> None:
        """设置分红数据

        Args:
            dividend_data: { stock_code: DataFrame(columns=[time, interest, ...]), ... }
        """
        self._dividend_data = dividend_data or {}

    def _ensure_dividend_datetime_index(self, df: pd.DataFrame,
                                         stock_code: str = '') -> Optional[pd.DataFrame]:
        """确保分红数据 DataFrame 的索引是 DatetimeIndex

        支持多种分红数据来源：
        - BaoStock: 索引已经是 DatetimeIndex 或 YYYYMMDD 字符串
        - QMT: time 列是毫秒时间戳，索引是 RangeIndex
        - AKShare: 索引可能是其他格式
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # QMT 格式：time 列是毫秒时间戳
        if 'time' in df.columns and df['time'].dtype in ('float64', 'int64'):
            try:
                dt_values = pd.to_datetime(df['time'], unit='ms', errors='coerce')
                valid = dt_values.notna()
                if valid.any():
                    df = df[valid].copy()
                    df.index = dt_values[valid]
                    df = df.sort_index()
                    return df
            except Exception:
                pass

        # BaoStock 格式：索引是 YYYYMMDD 字符串或整数
        try:
            dt_values = pd.to_datetime(df.index, format='%Y%m%d', errors='coerce')
            valid = dt_values.notna()
            if valid.any():
                df = df[valid].copy()
                df.index = dt_values[valid]
                return df
        except Exception:
            pass

        # 通用格式：直接尝试 to_datetime
        try:
            dt_values = pd.to_datetime(df.index, errors='coerce')
            valid = dt_values.notna()
            if valid.any():
                df = df[valid].copy()
                df.index = dt_values[valid]
                return df
        except Exception:
            pass

        if stock_code:
            self.logger.debug(
                f'[_ensure_dividend_datetime_index] {stock_code} '
                f'无法构建DatetimeIndex, index类型={type(df.index).__name__}'
            )
        return None

    def get_latest_dvps(self, stock_code: str,
                        date: Optional[datetime.date] = None) -> Optional[float]:
        """获取指定日期前最近一次每股派息金额

        Args:
            stock_code: 股票代码
            date: 查询日期，默认使用当前回测日期

        Returns:
            每股派息金额，无数据返回None
        """
        query_date = date or self._current_date
        if query_date is None:
            return None

        df = self._dividend_data.get(stock_code)
        if df is None or df.empty:
            return None

        try:
            df = self._ensure_dividend_datetime_index(df, stock_code)
            if df is None:
                return None
            self._dividend_data[stock_code] = df

            mask = df.index.date <= query_date
            available = df[mask]
            if available.empty:
                return None

            latest = available.iloc[-1]
            interest = latest.get('interest')
            if interest is not None and not pd.isna(interest) and interest > 0:
                return float(interest)
            return None
        except Exception:
            return None

    def get_dvps_history(self, stock_code: str, count: int = 3,
                         date: Optional[datetime.date] = None) -> List[float]:
        """获取指定日期前最近N次每股派息金额

        Args:
            stock_code: 股票代码
            count: 期数
            date: 查询日期

        Returns:
            每股派息金额列表，按时间升序
        """
        query_date = date or self._current_date
        if query_date is None:
            return []

        df = self._dividend_data.get(stock_code)
        if df is None or df.empty:
            return []

        try:
            df = self._ensure_dividend_datetime_index(df, stock_code)
            if df is None:
                return []
            self._dividend_data[stock_code] = df

            mask = df.index.date <= query_date
            available = df[mask].tail(count)

            if available.empty:
                return []

            result = []
            for _, row in available.iterrows():
                interest = row.get('interest')
                if interest is not None and not pd.isna(interest) and interest > 0:
                    result.append(float(interest))
            return result
        except Exception:
            return []

    def get_financial_field(self, stock_code: str, table_name: str,
                            field: str, date: Optional[datetime.date] = None) -> Any:
        """获取指定股票的最新已披露财务字段值

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            field: 字段名
            date: 查询日期，默认使用当前回测日期

        Returns:
            字段值，无数据返回None
        """
        query_date = date or self._current_date
        if query_date is None:
            return None
        return self._cache.get_latest(stock_code, table_name, query_date, field)

    def get_financial_fields(self, stock_code: str, table_name: str,
                             fields: List[str], date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """获取指定股票的最新已披露财务多个字段值

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            fields: 字段名列表
            date: 查询日期

        Returns:
            { field1: value1, field2: value2, ... }
        """
        query_date = date or self._current_date
        if query_date is None:
            return {f: None for f in fields}
        return self._cache.get_latest_multi_fields(stock_code, table_name, query_date, fields)

    def get_financial_history(self, stock_code: str, table_name: str,
                              field: str, count: int = 4,
                              date: Optional[datetime.date] = None) -> List[Any]:
        """获取指定股票最近N期的财务字段值

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            field: 字段名
            count: 期数
            date: 查询日期

        Returns:
            字段值列表，按时间升序
        """
        query_date = date or self._current_date
        if query_date is None:
            return []
        return self._cache.get_history(stock_code, table_name, query_date, count, field)

    def screen_stocks(self, condition: Callable[[str], bool],
                      stock_pool: Optional[List[str]] = None) -> List[str]:
        """基于财务条件筛选股票

        Args:
            condition: 筛选条件函数，参数为股票代码，返回bool
            stock_pool: 股票池，默认使用缓存中的全部股票

        Returns:
            满足条件的股票代码列表
        """
        pool = stock_pool or self._cache.get_stocks()
        result = []
        for stock in pool:
            try:
                if condition(stock):
                    result.append(stock)
            except Exception:
                continue
        return result

    def rank_stocks(self, score_func: Callable[[str], Optional[float]],
                    stock_pool: Optional[List[str]] = None,
                    ascending: bool = False,
                    top_n: Optional[int] = None) -> List[tuple]:
        """基于财务指标对股票排序

        Args:
            score_func: 评分函数，参数为股票代码，返回数值（None表示排除）
            stock_pool: 股票池
            ascending: 是否升序
            top_n: 返回前N名

        Returns:
            [(stock_code, score), ...] 排序后的列表
        """
        pool = stock_pool or self._cache.get_stocks()
        scored = []
        for stock in pool:
            try:
                score = score_func(stock)
                if score is not None and not (isinstance(score, float) and (np.isnan(score) or np.isinf(score))):
                    scored.append((stock, score))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=not ascending)

        if top_n:
            scored = scored[:top_n]

        return scored

    def compute_growth_rate(self, stock_code: str, table_name: str,
                            field: str, periods: int = 1,
                            date: Optional[datetime.date] = None) -> Optional[float]:
        """计算财务字段的同比增长率

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            field: 字段名
            periods: 增长期数（1=同比，即最近1期vs前1期）
            date: 查询日期

        Returns:
            增长率，如 0.15 表示增长15%，无数据返回None
        """
        query_date = date or self._current_date
        if query_date is None:
            return None

        history = self._cache.get_history(stock_code, table_name, query_date, periods + 1, field)
        if len(history) < periods + 1:
            return None

        current = history[-1]
        previous = history[-(periods + 1)]

        if previous is None or previous == 0:
            return None

        return (current - previous) / abs(previous)
