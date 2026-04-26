from collections import defaultdict
from typing import Dict, List, Optional
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('high_dividend', default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2016-01-01', 'end_date': '2026-04-17'})
class HighDividendStrategy(StockSelectionStrategy):
    """高股息行业均仓策略 - 规避高股息陷阱，行业分散选股

    选股逻辑：
    1. 基本面过滤：ROE > 0、归母净利润增速 > 0、经营现金流 > 0
    2. 行业分散：申万一级行业各选股息率最高的1只
    3. 优中选优：从行业代表中取前N只
    4. 等权重持仓，月度调仓

    规避高股息陷阱：
    - ROE > 0：确保公司盈利，非亏损吃老本分红
    - 归母净利润增速 > 0：确保业绩增长，分红可持续
    - 经营现金流 > 0：确保分红来自真金白银，非数字游戏
    - 行业分散：避免重仓单一周期行业
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('min_roe', 0.0),
        ('min_profit_growth', 0.0),
        ('min_operate_cashflow', 0.0),
        ('use_avg_dividend', True),
        ('skip_fundamental_if_missing', True),  # 财务数据缺失时跳过基本面过滤
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()

        filtered = self._filter_fundamentals(pool)

        if not filtered:
            self.log(f'基本面过滤后无股票')
            return []

        self.log(f'基本面过滤: {len(pool)} -> {len(filtered)} 只')

        dividend_scores = self._calc_dividend_yield(filtered)

        if not dividend_scores:
            self.log(f'无法计算股息率')
            return []

        industry_best = self._pick_industry_best(dividend_scores)

        if not industry_best:
            self.log(f'行业分散选股后无股票')
            return []

        industry_best.sort(key=lambda x: x[1], reverse=True)
        max_stocks = self.params.max_stocks
        selected = [stock for stock, _ in industry_best[:max_stocks]]

        self.log(f'行业分散选股: {len(industry_best)} 个行业 -> 选中 {len(selected)} 只')
        for stock, dy in industry_best[:max_stocks]:
            industry = self.get_industry(stock) or '未知'
            self.log(f'  {stock} | 行业: {industry} | 股息率: {dy:.2%}')

        return selected

    def _filter_fundamentals(self, pool: List[str]) -> List[str]:
        """基本面三重过滤：ROE、归母净利润增速、经营现金流"""
        result = []
        debug_logged = 0
        missing_count = 0

        for stock in pool:
            roe = self.get_financial_field(stock, 'Pershareindex', 'du_return_on_equity')
            profit_growth = self.get_financial_field(
                stock, 'Pershareindex', 'inc_net_profit_rate'
            )
            ocf = self.get_financial_field(stock, 'Pershareindex', 's_fa_ocfps')
            
            self.log(f'  {stock} ROE={roe}% 归母净利润增速={profit_growth}% 经营现金流={ocf}')

            # 记录前3条缺失调试信息
            if debug_logged < 3 and roe is None:
                self.log(f'[DEBUG] {stock} ROE=None (Pershareindex.du_return_on_equity)')
                debug_logged += 1

            # 统计缺失情况
            if roe is None and profit_growth is None and ocf is None:
                missing_count += 1


            # 正常过滤逻辑
            if roe is None or roe <= self.params.min_roe:
                continue
            if profit_growth is None or profit_growth <= self.params.min_profit_growth:
                continue
            if ocf is None or ocf <= self.params.min_operate_cashflow:
                continue

            result.append(stock)

        # 降级逻辑：如果全部股票都缺少财务数据，且允许跳过，则返回整个池子
        if not result and missing_count == len(pool) and self.params.skip_fundamental_if_missing:
            self.log(f'[WARN] 所有股票财务数据缺失({missing_count}/{len(pool)})，跳过基本面过滤')
            return pool

        return result

    def _calc_dividend_yield(self, stocks: List[str]) -> Dict[str, float]:
        """计算股息率 = 每股分红 / 当前股价

        支持 use_avg_dividend 模式：取近3年股息率均值，提升稳定性
        """
        result = {}
        for stock in stocks:
            dy = self.get_dividend_yield(stock, use_avg=self.params.use_avg_dividend)
            if dy is not None and dy > 0:
                result[stock] = dy

        return result

    def _pick_industry_best(self, dividend_scores: Dict[str, float]) -> List[tuple]:
        """每个申万一级行业选股息率最高的1只，返回 [(stock, dividend_yield), ...]"""
        industry_stocks: Dict[str, List[tuple]] = defaultdict(list)

        for stock, dy in dividend_scores.items():
            industry = self.get_industry(stock)
            if industry:
                industry_stocks[industry].append((stock, dy))
            else:
                industry_stocks['未知'].append((stock, dy))

        result = []
        for industry, stocks in industry_stocks.items():
            stocks.sort(key=lambda x: x[1], reverse=True)
            result.append(stocks[0])

        return result
