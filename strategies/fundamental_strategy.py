from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('fundamental_roe', default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2024-01-01', 'end_date': '2026-04-17'})
class ROEFundamentalStrategy(StockSelectionStrategy):
    """ROE基本面选股策略 - 基于净资产收益率筛选优质股票

    选股逻辑：
    1. 从股票池中筛选 ROE > 阈值 的股票
    2. 按 ROE 降序排列
    3. 取前 N 只股票等权重持仓
    4. 每月调仓一次
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('min_roe', 0.10),
        ('min_eps', 0.3),
    )

    def select_stocks(self):
        pool = self.get_stock_pool()

        def filter_fn(stock):
            roe = self.get_financial_field(stock, 'Pershareindex', 'roe_diluted')
            if roe is None or roe < self.params.min_roe:
                return False

            eps = self.get_financial_field(stock, 'Pershareindex', 'eps_diluted')
            if eps is None or eps < self.params.min_eps:
                return False

            return True

        filtered = self.screen_stocks(filter_fn, pool)

        def roe_score(stock):
            roe = self.get_financial_field(stock, 'Pershareindex', 'roe_diluted')
            return roe

        ranked = self.rank_stocks(roe_score, stock_pool=filtered, top_n=self.params.max_stocks)

        return [stock for stock, _ in ranked]


@register_strategy('fundamental_growth', default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2024-01-01', 'end_date': '2026-04-17'})
class GrowthFundamentalStrategy(StockSelectionStrategy):
    """成长性选股策略 - 基于营收和利润增长率筛选成长股

    选股逻辑：
    1. 筛选营收同比增长 > 阈值 的股票
    2. 筛选净利润同比增长 > 阈值 的股票
    3. 按综合增长率评分排序
    4. 取前 N 只股票等权重持仓
    5. 每季度调仓一次
    """

    params = (
        ('rebalance_freq', 'quarterly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('min_revenue_growth', 0.10),
        ('min_profit_growth', 0.10),
    )

    def select_stocks(self):
        pool = self.get_stock_pool()
        candidates = []

        for stock in pool:
            rev_growth = self.compute_growth_rate(stock, 'Income', 'total_operate_income')
            if rev_growth is None or rev_growth < self.params.min_revenue_growth:
                continue

            profit_growth = self.compute_growth_rate(stock, 'Income', 'net_profit_incl_min_int_inc')
            if profit_growth is None or profit_growth < self.params.min_profit_growth:
                continue

            eps = self.get_financial_field(stock, 'Pershareindex', 'eps_diluted')
            if eps is None or eps <= 0:
                continue

            score = rev_growth * 0.4 + profit_growth * 0.6
            candidates.append((stock, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        return [stock for stock, _ in candidates[:self.params.max_stocks]]
