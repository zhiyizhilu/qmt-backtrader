from typing import Dict, List
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('small_cap', default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2016-01-01', 'end_date': '2026-04-17'})
class SmallCapStrategy(StockSelectionStrategy):
    """小市值策略 - 纯市值排序

    选股逻辑：
    1. 计算市值，按升序排列，取前N只
    2. 等权重持仓，月度调仓
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('max_market_cap', None),
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()

        market_caps = self._calc_market_caps(pool)
        if not market_caps:
            self.log('无法计算市值')
            return []
        self.log(f'市值计算: {len(market_caps)} 只有有效市值数据')

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

        self.log(f'按市值排序选股: 选中 {len(selected)} 只')
        for stock, mc in sorted_stocks[:max_stocks]:
            self.log(f'  {stock} | 市值: {mc / 1e8:.2f}亿')

        return selected

    def _calc_market_caps(self, stocks: List[str]) -> Dict[str, float]:
        """计算市值 = 总股本 × 当前股价

        优先使用 总股本 = 所有者权益合计 / 每股净资产
        回退方案：使用所有者权益作为市值代理
        """
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
