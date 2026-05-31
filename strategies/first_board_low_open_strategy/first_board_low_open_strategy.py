from typing import List
from core.stock_selection import StockSelectionStrategy
from core.data_adapter import get_limit_ratio
from strategies import register_strategy


@register_strategy('first_board_low_open',
                   default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 200000, 'commission': 0.0003,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中证全指'})
class FirstBoardLowOpenStrategy(StockSelectionStrategy):
    """首板低开策略 - 选取低位、非连板、涨停后次日低开的股票

    来源: 聚宽社区 https://www.joinquant.com/view/community/detail/44901
    作者: wywy1995

    核心逻辑：选取"低位、非连板、涨停后次日低开"的股票，开盘买入，
    第二日上午如果有盈利就卖出，没有就拿到尾盘无论盈利与否都卖出。

    选股逻辑：
    1. 筛选昨日涨停的股票（收盘价等于涨停价）
    2. 排除连板股票（前日也涨停的为连板，排除）
    3. 计算相对位置，只保留60日内相对位置<=50%的低位股
    4. 筛选今日低开3%-4%的股票（开盘价/昨收在0.96-0.97之间）

    调仓规则：
    - 每日调仓，等权重持仓
    - 持仓股次日若未涨停则卖出（通过日度调仓自动实现）

    注意：
    - 日线回测模式下，买入价格为当日收盘价（非开盘价），与原始策略有差异
    - 原始策略为日内交易（开盘买、盘中/尾盘卖），日线模式近似为隔日交易
    """

    params = (
        ('rebalance_freq', 'daily'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('rp_period', 60),
        ('rp_threshold', 0.5),
        ('open_pct_low', 0.96),
        ('open_pct_high', 0.97),
        ('filter_kcbj', True),
        ('filter_cyb', False),
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()

        if self.params.filter_kcbj:
            pool = [s for s in pool if not self._is_kcbj(s)]

        if self.params.filter_cyb:
            pool = [s for s in pool if not self._is_cyb(s)]

        pool = [s for s in pool if not self.is_suspended(s)]

        limit_up_stocks = self._filter_limit_up_yesterday(pool)
        if not limit_up_stocks:
            self.log(f'选股结果: {len(pool)} -> 涨停0')
            return []

        first_board_stocks = self._filter_first_board(limit_up_stocks)
        if not first_board_stocks:
            self.log(f'选股结果: {len(pool)} -> 涨停{len(limit_up_stocks)} -> 首板0')
            return []

        low_position_stocks = self._filter_low_position(first_board_stocks)
        if not low_position_stocks:
            self.log(f'选股结果: {len(pool)} -> 涨停{len(limit_up_stocks)} -> '
                     f'首板{len(first_board_stocks)} -> 低位0')
            return []

        low_open_stocks = self._filter_low_open(low_position_stocks)

        self.log(f'选股结果: {len(pool)} -> 涨停{len(limit_up_stocks)} -> '
                 f'首板{len(first_board_stocks)} -> 低位{len(low_position_stocks)} -> '
                 f'低开{len(low_open_stocks)}')

        return low_open_stocks

    @staticmethod
    def _is_kcbj(symbol: str) -> bool:
        code = symbol.split('.')[0] if '.' in symbol else symbol
        if len(code) < 2:
            return False
        if code[0] in ('4', '8'):
            return True
        if code[:2] == '68':
            return True
        return False

    @staticmethod
    def _is_cyb(symbol: str) -> bool:
        code = symbol.split('.')[0] if '.' in symbol else symbol
        if len(code) >= 3 and code[:3] in ('300', '301'):
            return True
        return False

    def _is_limit_up_on_day(self, symbol: str, day_offset: int) -> bool:
        ohlcv = self.get_ohlcv_data(symbol, day_offset + 2)
        if len(ohlcv) < day_offset + 2:
            return False
        target_close = ohlcv[-(day_offset + 1)]['close']
        prev_close = ohlcv[-(day_offset + 2)]['close']
        if prev_close <= 0:
            return False
        limit_ratio = get_limit_ratio(symbol)
        high_limit = round(prev_close * (1 + limit_ratio), 2)
        return target_close >= high_limit - 0.005

    def _filter_limit_up_yesterday(self, pool: List[str]) -> List[str]:
        result = []
        for symbol in pool:
            if self._is_limit_up_on_day(symbol, 1):
                result.append(symbol)
        return result

    def _filter_first_board(self, limit_up_stocks: List[str]) -> List[str]:
        result = []
        for symbol in limit_up_stocks:
            if self._is_limit_up_on_day(symbol, 2):
                continue
            result.append(symbol)
        return result

    def _filter_low_position(self, stocks: List[str]) -> List[str]:
        rp_period = self.params.rp_period
        rp_threshold = self.params.rp_threshold
        result = []
        for symbol in stocks:
            ohlcv = self.get_ohlcv_data(symbol, rp_period)
            if len(ohlcv) < rp_period:
                continue
            closes = [d['close'] for d in ohlcv]
            highs = [d['high'] for d in ohlcv]
            lows = [d['low'] for d in ohlcv]
            current_close = closes[-1]
            highest = max(highs)
            lowest = min(lows)
            if highest == lowest:
                continue
            rp = (current_close - lowest) / (highest - lowest)
            if rp <= rp_threshold:
                result.append(symbol)
        return result

    def _filter_low_open(self, stocks: List[str]) -> List[str]:
        open_pct_low = self.params.open_pct_low
        open_pct_high = self.params.open_pct_high
        result = []
        for symbol in stocks:
            ohlcv = self.get_ohlcv_data(symbol, 2)
            if len(ohlcv) < 2:
                continue
            yesterday_close = ohlcv[-2]['close']
            today_open = ohlcv[-1]['open']
            if yesterday_close <= 0:
                continue
            open_pct = today_open / yesterday_close
            if open_pct_low <= open_pct <= open_pct_high:
                result.append(symbol)
        return result
