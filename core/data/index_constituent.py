import ast
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class IndexConstituentManager:
    """指数历史成分股管理器

    基于聚宽下载的CSV文件获取历史成分股数据，
    当回测日期超出CSV文件范围时，自动从QMT获取最新成分股并更新文件。

    CSV文件格式:
        date,codes
        2016-04-28,"['000001.SZ', '000002.SZ', ...]"
        2016-06-13,"['000001.SZ', '000008.SZ', ...]"

    每行代表成分股发生变更的日期，两个日期之间的成分股不变。
    查询时取 <= 查询日期 的最近一行。
    """

    DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / '.cache' / 'JQData' / 'index_constituent'

    SECTOR_TO_INDEX = {
        '沪深300': '000300.SH',
        '中证500': '000905.SH',
        '中证1000': '000852.SH',
        '上证50': '000016.SH',
    }

    INDEX_TO_SECTOR = {v: k for k, v in SECTOR_TO_INDEX.items()}

    def __init__(self, data_dir: Optional[str] = None, xtdata=None):
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR
        self.xtdata = xtdata
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load_csv(self, index_code: str) -> Optional[pd.DataFrame]:
        if index_code in self._cache:
            return self._cache[index_code]

        csv_path = self.data_dir / f"{index_code}.csv"
        if not csv_path.exists():
            self.logger.debug(f"成分股CSV文件不存在: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df['codes'] = df['codes'].apply(self._parse_codes)
            self._cache[index_code] = df
            self.logger.info(f"加载 {index_code} 成分股: {len(df)} 条变更记录, "
                             f"日期范围 {df['date'].min().strftime('%Y-%m-%d')} ~ "
                             f"{df['date'].max().strftime('%Y-%m-%d')}")
            return df
        except Exception as e:
            self.logger.error(f"加载 {index_code} 成分股CSV失败: {e}")
            return None

    @staticmethod
    def _parse_codes(codes_str) -> List[str]:
        if isinstance(codes_str, list):
            return codes_str
        try:
            result = ast.literal_eval(codes_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
        if isinstance(codes_str, str):
            cleaned = codes_str.strip("[]").replace("'", "").replace('"', '')
            if cleaned:
                return [c.strip() for c in cleaned.split(',') if c.strip()]
        return []

    def get_constituent_stocks(self, index_code: str, date: str) -> List[str]:
        """获取指定日期的指数成分股

        查询逻辑:
        1. 从CSV文件中查找 <= date 的最近一条记录
        2. 如果 date 超出CSV最新日期，尝试从QMT获取最新成分股
        3. 若QMT返回的成分股与CSV最新记录不同，追加新行到CSV

        Args:
            index_code: 指数代码，如 '000300.SH'
            date: 日期，格式 'YYYY-MM-DD'

        Returns:
            股票代码列表
        """
        df = self._load_csv(index_code)
        target_date = pd.Timestamp(date)

        if df is not None and not df.empty:
            mask = df['date'] <= target_date
            if mask.any():
                latest_row = df[mask].iloc[-1]
                latest_date = latest_row['date']

                if target_date > latest_date:
                    updated = self._try_update_from_qmt(index_code, date, df)
                    if updated:
                        self._cache.pop(index_code, None)
                        df = self._load_csv(index_code)
                        if df is not None:
                            mask = df['date'] <= target_date
                            if mask.any():
                                return list(df[mask].iloc[-1]['codes'])

                return list(latest_row['codes'])

        updated = self._try_update_from_qmt(index_code, date, df)
        if updated:
            self._cache.pop(index_code, None)
            df = self._load_csv(index_code)
            if df is not None and not df.empty:
                mask = df['date'] <= target_date
                if mask.any():
                    return list(df[mask].iloc[-1]['codes'])

        return []

    def _try_update_from_qmt(self, index_code: str, date: str,
                              existing_df: Optional[pd.DataFrame] = None) -> bool:
        """尝试从QMT获取最新成分股并更新CSV文件

        Returns:
            是否更新了CSV文件
        """
        if not self.xtdata:
            self.logger.debug(f"QMT不可用，无法更新 {index_code} 成分股")
            return False

        sector = self.INDEX_TO_SECTOR.get(index_code)
        if not sector:
            self.logger.warning(f"未知的指数代码: {index_code}，无法映射到板块名称")
            return False

        try:
            stock_list = self.xtdata.get_stock_list_in_sector(sector)
            if not stock_list or len(stock_list) < 10:
                self.logger.warning(f"QMT返回 {sector} 成分股数量异常: {len(stock_list) if stock_list else 0}")
                return False

            if existing_df is not None and not existing_df.empty:
                latest_row = existing_df.iloc[-1]
                latest_codes = set(latest_row['codes'])
                new_codes = set(stock_list)

                if new_codes == latest_codes:
                    self.logger.debug(f"{index_code} 成分股无变化，无需更新")
                    return False

                diff_in = new_codes - latest_codes
                diff_out = latest_codes - new_codes
                self.logger.info(f"{index_code} 成分股有变化: "
                                 f"新纳入 {len(diff_in)} 只, 剔除 {len(diff_out)} 只")

            today = datetime.now().strftime('%Y-%m-%d')
            self._append_to_csv(index_code, today, stock_list)
            return True

        except Exception as e:
            self.logger.warning(f"从QMT获取 {index_code} 成分股失败: {e}")
            return False

    def _append_to_csv(self, index_code: str, date: str, stock_list: List[str]) -> None:
        csv_path = self.data_dir / f"{index_code}.csv"
        sorted_stocks = sorted(stock_list)
        codes_str = str(sorted_stocks)

        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            last_date = pd.to_datetime(existing['date'].iloc[-1])
            new_date = pd.Timestamp(date)
            if new_date <= last_date:
                self.logger.debug(f"跳过追加: {date} 不晚于CSV最新日期 {last_date.strftime('%Y-%m-%d')}")
                return

            new_row = pd.DataFrame({'date': [date], 'codes': [codes_str]})
            combined = pd.concat([existing, new_row], ignore_index=True)
            combined.to_csv(csv_path, index=False)
        else:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            new_row = pd.DataFrame({'date': [date], 'codes': [codes_str]})
            new_row.to_csv(csv_path, index=False)

        self.logger.info(f"已更新 {csv_path.name}: 追加 {date}, {len(stock_list)} 只成分股")

    def get_all_constituent_stocks_in_range(self, index_code: str,
                                             start_date: str,
                                             end_date: str) -> List[str]:
        """获取指定时间范围内所有历史成分股的并集

        指数成分股会定期调整，回测期间涉及的股票数量远超单次成分股数量。
        此方法收集时间范围内所有变更记录的成分股并集，
        确保回测时能获取到所有曾经属于该指数的股票数据。

        Args:
            index_code: 指数代码，如 '000300.SH'
            start_date: 起始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'

        Returns:
            去重的股票代码列表
        """
        df = self._load_csv(index_code)
        if df is None or df.empty:
            return self.get_constituent_stocks(index_code, start_date)

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # 取时间范围内的变更记录
        mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
        period_df = df[mask]

        # 还需要取起始日期之前的最近一条记录
        mask_before = df['date'] < start_ts
        if mask_before.any():
            before_row = df[mask_before].iloc[-1]
            all_stocks = set(before_row['codes'])
        else:
            all_stocks = set()

        # 加上时间范围内所有变更记录的成分股
        change_count = 0
        for _, row in period_df.iterrows():
            codes = row['codes']
            if isinstance(codes, list):
                all_stocks.update(codes)
                change_count += 1

        self.logger.info(f"{index_code} 在 {start_date}~{end_date} 期间: "
                         f"{change_count} 次成分股变更, 共涉及 {len(all_stocks)} 只股票")

        return sorted(list(all_stocks))

    @classmethod
    def sector_to_index_code(cls, sector: str) -> Optional[str]:
        return cls.SECTOR_TO_INDEX.get(sector)

    @classmethod
    def index_code_to_sector(cls, index_code: str) -> Optional[str]:
        return cls.INDEX_TO_SECTOR.get(index_code)
