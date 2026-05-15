import datetime as dt_module
from typing import Dict, List
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('jq_small_cap',
                   default_kwargs={'max_stocks': 5},
                   backtest_config={'cash': 1000000, 'commission': 0.0003,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中小综指'})
class JqSmallCapStrategy(StockSelectionStrategy):
    """聚宽小市值策略 - 按流通市值升序选股，每日调仓

    克隆自聚宽文章: https://www.joinquant.com/post/25496
    原始策略: 中小板流通市值最小5只股票，每日14:40调仓

    选股逻辑：
    1. 从股票池中获取所有股票
    2. 过滤停牌、涨停、跌停股票
    3. 按流通市值升序排列
    4. 选取前N只股票等权持仓

    调仓规则：
    - 每日调仓，等权重持仓
    - 基类 rebalance_to 自动处理卖出不在目标池的股票、等权买入新股票
    """

    params = (
        ('rebalance_freq', 'daily'),
        ('max_stocks', 5),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('max_market_cap', None),
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        self.log(f'股票池: {len(pool)} 只')

        filtered = self._filter_tradeable(pool)
        if not filtered:
            self.log('过滤后无股票')
            return []

        market_caps = self._calc_market_caps(filtered)
        if not market_caps:
            self.log('无法计算市值')
            return []

        if self.params.max_market_cap is not None:
            cap_limit = self.params.max_market_cap * 1e8
            before = len(market_caps)
            market_caps = {s: v for s, v in market_caps.items() if v <= cap_limit}
            self.log(f'市值上限过滤: {before} -> {len(market_caps)} 只 (上限{self.params.max_market_cap}亿)')

        if not market_caps:
            self.log('过滤后无股票')
            return []

        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1])
        max_stocks = self.params.max_stocks
        selected = [stock for stock, _ in sorted_stocks[:max_stocks]]

        self.log(f'按市值排序选股: {len(pool)} -> {len(filtered)} -> {len(selected)} 只')
        for stock, mc in sorted_stocks[:max_stocks]:
            self.log(f'  {stock} | 市值: {mc / 1e8:.2f}亿')

        return selected

    def _filter_tradeable(self, pool: List[str]) -> List[str]:
        result = []
        for stock in pool:
            if self.is_suspended(stock):
                continue
            if self.is_limit_up(stock):
                continue
            if self.is_limit_down(stock):
                continue
            result.append(stock)
        self.log(f'过滤(停牌/涨停/跌停): {len(pool)} -> {len(result)} 只')
        return result

    def _calc_market_caps(self, stocks: List[str]) -> Dict[str, float]:
        result = {}
        no_price = 0
        no_cap_data = 0

        for stock in stocks:
            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                no_price += 1
                continue

            total_equity = self.get_financial_field(stock, 'Balance', 'total_equity')
            bps = self.get_financial_field(stock, 'Pershareindex', 's_fa_bps')

            if total_equity and total_equity > 0 and bps and bps > 0:
                total_shares = total_equity / bps
                market_cap = total_shares * price
                result[stock] = market_cap
            elif total_equity and total_equity > 0:
                result[stock] = total_equity
            else:
                no_cap_data += 1

        if no_price > 0 or no_cap_data > 0:
            self.log(f'市值计算: {len(stocks)} 只 -> {len(result)} 只有有效数据 '
                     f'(无价格={no_price}, 无股本数据={no_cap_data})')

        return result
