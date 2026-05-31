import datetime as dt_module
from typing import List, Optional, Dict
from core.strategy_logic import StrategyLogic, BarData
from core.data_adapter import get_limit_ratio, validate_trade_volume
from strategies import register_strategy

SELECTED_STOCKS = [
    '000007.SZ', '000065.SZ', '000078.SZ', '000498.SZ', '000520.SZ',
    '000530.SZ', '000554.SZ', '000565.SZ', '000600.SZ', '000608.SZ',
    '000629.SZ', '000710.SZ', '000711.SZ', '000715.SZ', '000793.SZ',
    '000797.SZ', '000798.SZ', '000809.SZ', '000812.SZ', '000830.SZ',
    '000868.SZ', '000936.SZ', '001231.SZ', '001236.SZ', '001308.SZ',
    '001309.SZ', '001896.SZ', '002041.SZ', '002060.SZ', '002084.SZ',
    '002094.SZ', '002096.SZ', '002105.SZ', '002155.SZ', '002160.SZ',
    '002162.SZ', '002165.SZ', '002173.SZ', '002188.SZ', '002194.SZ',
    '002205.SZ', '002206.SZ', '002227.SZ', '002237.SZ', '002246.SZ',
    '002253.SZ', '002256.SZ', '002263.SZ', '002269.SZ', '002274.SZ',
    '002278.SZ', '002285.SZ', '002301.SZ', '002305.SZ', '002331.SZ',
    '002361.SZ', '002377.SZ', '002419.SZ', '002455.SZ', '002460.SZ',
    '002467.SZ', '002512.SZ', '002514.SZ', '002516.SZ', '002518.SZ',
    '002538.SZ', '002542.SZ', '002565.SZ', '002568.SZ', '002596.SZ',
    '002599.SZ', '002611.SZ', '002622.SZ', '002631.SZ', '002647.SZ',
    '002663.SZ', '002667.SZ', '002679.SZ', '002688.SZ', '002716.SZ',
    '002719.SZ', '002723.SZ', '002752.SZ', '002762.SZ', '002771.SZ',
    '002778.SZ', '002797.SZ', '002799.SZ', '002825.SZ', '002830.SZ',
    '002856.SZ', '002861.SZ', '002881.SZ', '002883.SZ', '002888.SZ',
    '002922.SZ', '002928.SZ', '002952.SZ', '002962.SZ', '003001.SZ',
    '003007.SZ', '003021.SZ', '003025.SZ', '003040.SZ', '300157.SZ',
    '300158.SZ', '300159.SZ', '300209.SZ', '300369.SZ', '300427.SZ',
    '300464.SZ', '300546.SZ', '300643.SZ', '300716.SZ', '300720.SZ',
    '300793.SZ', '300807.SZ', '300822.SZ', '300829.SZ', '300875.SZ',
    '300889.SZ', '300892.SZ', '300921.SZ', '301027.SZ', '301125.SZ',
    '301230.SZ', '301302.SZ', '301313.SZ', '301319.SZ', '301370.SZ',
    '600053.SH', '600082.SH', '600119.SH', '600137.SH', '600173.SH',
    '600207.SH', '600243.SH', '600249.SH', '600257.SH', '600265.SH',
    '600280.SH', '600318.SH', '600337.SH', '600353.SH', '600403.SH',
    '600421.SH', '600520.SH', '600521.SH', '600602.SH', '600604.SH',
    '600623.SH', '600657.SH', '600661.SH', '600682.SH', '600683.SH',
    '600721.SH', '600732.SH', '600735.SH', '600738.SH', '600756.SH',
    '600798.SH', '600800.SH', '600822.SH', '600825.SH', '600986.SH',
    '601099.SH', '601208.SH', '601528.SH', '601919.SH', '603002.SH',
    '603004.SH', '603065.SH', '603098.SH', '603103.SH', '603118.SH',
    '603214.SH', '603223.SH', '603290.SH', '603359.SH', '603383.SH',
    '603486.SH', '603507.SH', '603536.SH', '603595.SH', '603728.SH',
    '603729.SH', '603755.SH', '603778.SH', '603856.SH', '603860.SH',
    '603900.SH', '603918.SH', '605086.SH', '605133.SH', '605199.SH',
    '605588.SH',
]


@register_strategy('first_board_low_open_1m',
                   default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 200000, 'commission': 0.0003,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1m', 'data_source': 'qmt'})
class FirstBoardLowOpen1MStrategy(StrategyLogic):
    """首板低开策略 - 分钟线版本

    来源: 聚宽社区 https://www.joinquant.com/view/community/detail/44901
    作者: wywy1995

    核心逻辑：选取"低位、非连板、涨停后次日低开"的股票，开盘买入，
    第二日上午如果有盈利就卖出，没有就拿到尾盘无论盈利与否都卖出。

    分钟线交易规则：
    - 09:31 开盘买入选中的股票
    - 11:28 检查持仓，有盈利则卖出
    - 14:50 尾盘卖出所有剩余持仓（无论盈亏）

    选股逻辑（与日线版一致）：
    1. 筛选昨日涨停的股票
    2. 排除连板股票
    3. 计算相对位置，只保留60日内相对位置<=50%的低位股
    4. 筛选今日低开3%-4%的股票

    注意：分钟线模式下 get_ohlcv_data 返回的日线数据中，
    ohlcv[-1] 是最近一个完整交易日（昨天），今天的数据尚未完成。
    """

    params = (
        ('t_plus_1', True),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('rp_period', 60),
        ('rp_threshold', 0.5),
        ('open_pct_low', 0.96),
        ('open_pct_high', 0.97),
        ('filter_kcbj', True),
        ('filter_cyb', False),
        ('buy_time', '09:31'),
        ('morning_sell_time', '11:28'),
        ('afternoon_sell_time', '14:50'),
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._current_holdings: Dict[str, int] = {}
        self._buy_prices: Dict[str, float] = {}
        self._selected_stocks: List[str] = []
        self._selection_done_today: bool = False
        self._buy_done_today: bool = False
        self._morning_sell_done: bool = False
        self._afternoon_sell_done: bool = False
        self._last_selection_date: Optional[str] = None
        self._today_open_prices: Dict[str, float] = {}

    def get_symbols(self) -> List[str]:
        pool = getattr(self.params, 'stock_pool', None)
        if pool:
            return list(pool)
        return list(SELECTED_STOCKS)

    def _get_stock_pool(self) -> List[str]:
        pool = getattr(self.params, 'stock_pool', None)
        if pool:
            return list(pool)
        return list(SELECTED_STOCKS)

    def on_bar(self, bar: BarData):
        current_dt = self.get_current_datetime()
        if current_dt is None:
            return

        if not hasattr(current_dt, 'strftime'):
            return

        date_str = current_dt.strftime('%Y-%m-%d')
        time_str = current_dt.strftime('%H:%M')

        if date_str != self._last_selection_date:
            self._selection_done_today = False
            self._buy_done_today = False
            self._morning_sell_done = False
            self._afternoon_sell_done = False
            self._selected_stocks = []
            self._today_open_prices = {}
            self._last_selection_date = date_str

        if not self._selection_done_today and time_str >= self.params.buy_time:
            self._capture_today_open()
            self._do_selection()
            self._selection_done_today = True

        if not self._buy_done_today and self._selected_stocks:
            if time_str >= self.params.buy_time:
                self._do_buy()
                self._buy_done_today = True

        if not self._morning_sell_done and self._current_holdings:
            if time_str >= self.params.morning_sell_time:
                self._do_morning_sell()
                self._morning_sell_done = True

        if not self._afternoon_sell_done and self._current_holdings:
            if time_str >= self.params.afternoon_sell_time:
                self._do_afternoon_sell()
                self._afternoon_sell_done = True

    def _capture_today_open(self):
        pool = self._get_stock_pool()
        for symbol in pool:
            if symbol in self._today_open_prices:
                continue
            price = self.get_current_price(symbol)
            if price and price > 0:
                self._today_open_prices[symbol] = price

    def _do_selection(self):
        pool = self._get_stock_pool()

        if self.params.filter_kcbj:
            pool = [s for s in pool if not self._is_kcbj(s)]
        if self.params.filter_cyb:
            pool = [s for s in pool if not self._is_cyb(s)]

        pool = [s for s in pool if not self.is_suspended(s)]

        limit_up_stocks = self._filter_limit_up_yesterday(pool)
        if not limit_up_stocks:
            return

        first_board_stocks = self._filter_first_board(limit_up_stocks)
        if not first_board_stocks:
            return

        low_position_stocks = self._filter_low_position(first_board_stocks)
        if not low_position_stocks:
            return

        low_open_stocks = self._filter_low_open(low_position_stocks)

        max_stocks = getattr(self.params, 'max_stocks', 10)
        self._selected_stocks = low_open_stocks[:max_stocks]

        self.log(f'选股结果: {len(pool)} -> 涨停{len(limit_up_stocks)} -> '
                 f'首板{len(first_board_stocks)} -> 低位{len(low_position_stocks)} -> '
                 f'低开{len(low_open_stocks)} -> 买入{len(self._selected_stocks)}')

    def _do_buy(self):
        if not self._selected_stocks:
            return

        cash = self.get_cash()
        position_value = 0.0
        for symbol, volume in self._current_holdings.items():
            price = self.get_current_price(symbol)
            if price and price > 0:
                position_value += price * volume

        total_assets = cash + position_value
        position_ratio = getattr(self.params, 'position_ratio', 0.95)
        investable = total_assets * position_ratio
        per_stock = investable / len(self._selected_stocks) if self._selected_stocks else 0

        for symbol in self._selected_stocks:
            price = self.get_current_price(symbol)
            if not price or price <= 0:
                continue

            volume = int(per_stock / price / 100) * 100
            if volume <= 0:
                continue

            is_valid, _ = validate_trade_volume(symbol, volume)
            if not is_valid:
                continue

            result = self.buy(symbol, price, volume)
            if result is not None:
                self._current_holdings[symbol] = self._current_holdings.get(symbol, 0) + volume
                self._buy_prices[symbol] = price

    def _do_morning_sell(self):
        symbols_to_sell = []
        for symbol in list(self._current_holdings.keys()):
            buy_price = self._buy_prices.get(symbol, 0)
            if buy_price <= 0:
                continue
            current_price = self.get_current_price(symbol)
            if current_price and current_price > buy_price:
                symbols_to_sell.append(symbol)

        for symbol in symbols_to_sell:
            self._sell_position(symbol, '上午盈利卖出')

    def _do_afternoon_sell(self):
        for symbol in list(self._current_holdings.keys()):
            self._sell_position(symbol, '尾盘清仓')

    def _sell_position(self, symbol: str, reason: str = ''):
        volume = self._current_holdings.get(symbol, 0)
        if volume <= 0:
            return

        sellable = self.get_sellable_volume(symbol)
        if sellable <= 0:
            return

        price = self.get_current_price(symbol)
        if not price or price <= 0:
            return

        sell_vol = min(volume, sellable)
        result = self.sell(symbol, price, sell_vol)
        if result is not None:
            remaining = volume - sell_vol
            if remaining > 0:
                self._current_holdings[symbol] = remaining
            else:
                del self._current_holdings[symbol]
            self._buy_prices.pop(symbol, None)
            self.log(f'{reason}: {symbol}, 数量: {sell_vol}, 价格: {price:.2f}')

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

    def _filter_limit_up_yesterday(self, pool: List[str]) -> List[str]:
        result = []
        for symbol in pool:
            ohlcv = self.get_ohlcv_data(symbol, 2)
            if len(ohlcv) < 2:
                continue
            yesterday = ohlcv[-1]
            prev_day = ohlcv[-2]
            prev_close = prev_day['close']
            if prev_close <= 0:
                continue
            limit_ratio = get_limit_ratio(symbol)
            high_limit = round(prev_close * (1 + limit_ratio), 2)
            if yesterday['close'] >= high_limit - 0.005:
                result.append(symbol)
        return result

    def _filter_first_board(self, limit_up_stocks: List[str]) -> List[str]:
        result = []
        for symbol in limit_up_stocks:
            ohlcv = self.get_ohlcv_data(symbol, 3)
            if len(ohlcv) < 3:
                result.append(symbol)
                continue
            day_before_yesterday = ohlcv[-2]
            two_days_before = ohlcv[-3]
            prev_close = two_days_before['close']
            if prev_close <= 0:
                result.append(symbol)
                continue
            limit_ratio = get_limit_ratio(symbol)
            high_limit = round(prev_close * (1 + limit_ratio), 2)
            if day_before_yesterday['close'] >= high_limit - 0.005:
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
            ohlcv = self.get_ohlcv_data(symbol, 1)
            if not ohlcv:
                continue
            yesterday_close = ohlcv[-1]['close']
            if yesterday_close <= 0:
                continue
            today_open = self._today_open_prices.get(symbol)
            if today_open is None:
                continue
            open_pct = today_open / yesterday_close
            if open_pct_low <= open_pct <= open_pct_high:
                result.append(symbol)
        return result
