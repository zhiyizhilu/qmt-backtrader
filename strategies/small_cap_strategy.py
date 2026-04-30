from collections import defaultdict
from typing import Dict, List, Optional
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('small_cap', default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2016-01-01', 'end_date': '2026-04-17'})
class SmallCapStrategy(StockSelectionStrategy):
    """小市值优化策略 - 基本面过滤+行业分散+动量确认

    选股逻辑：
    1. 基本面过滤：ROE > 0、营收增速 > 0、经营现金流 > 0、资产负债率 < 阈值
    2. 行业分散：申万一级行业各选市值最小的1只
    3. 动量确认：近N日涨幅 > 0
    4. 按市值升序排列，取前N只
    5. 等权重持仓，月度调仓

    降低回撤机制：
    - 基本面过滤：排除"垃圾小盘"，剔除亏损、高负债、现金流断裂的公司
    - 行业分散：避免重仓单一周期行业
    - 动量确认：避免在下跌趋势中接飞刀
    - 建议配合 RiskController 使用（单股止损8%、组合最大回撤15%）
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('min_roe', 0.0),
        ('min_revenue_growth', 0.0),
        ('min_operate_cashflow', 0.0),
        ('max_debt_ratio', 0.6),
        ('momentum_period', 20),
        ('min_momentum', 0.0),
        ('max_market_cap', None),
        ('skip_fundamental_if_missing', True),
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()

        filtered = self._filter_fundamentals(pool)
        if not filtered:
            self.log('基本面过滤后无股票')
            return []
        self.log(f'基本面过滤: {len(pool)} -> {len(filtered)} 只')

        market_caps = self._calc_market_caps(filtered)
        if not market_caps:
            self.log('无法计算市值')
            return []
        self.log(f'市值计算: {len(market_caps)} 只有有效市值数据')

        if self.params.max_market_cap is not None:
            cap_limit = self.params.max_market_cap * 1e8
            before = len(market_caps)
            market_caps = {s: v for s, v in market_caps.items() if v <= cap_limit}
            self.log(f'市值上限过滤: {before} -> {len(market_caps)} 只 (上限{self.params.max_market_cap}亿)')

        momentum_filtered = self._filter_momentum(list(market_caps.keys()))
        if not momentum_filtered:
            self.log('动量过滤后无股票，使用全部市值有效股票')
        else:
            before = len(market_caps)
            momentum_set = set(momentum_filtered)
            market_caps = {s: v for s, v in market_caps.items() if s in momentum_set}
            self.log(f'动量过滤: {before} -> {len(market_caps)} 只')

        if not market_caps:
            self.log('过滤后无股票')
            return []

        industry_best = self._pick_industry_smallest(market_caps)
        if not industry_best:
            self.log('行业分散选股后无股票')
            return []

        industry_best.sort(key=lambda x: x[1])
        max_stocks = self.params.max_stocks
        selected = [stock for stock, _ in industry_best[:max_stocks]]

        self.log(f'行业分散选股: {len(industry_best)} 个行业 -> 选中 {len(selected)} 只')
        for stock, mc in industry_best[:max_stocks]:
            industry = self.get_industry(stock) or '未知'
            self.log(f'  {stock} | 行业: {industry} | 市值: {mc / 1e8:.2f}亿')

        return selected

    def _filter_fundamentals(self, pool: List[str]) -> List[str]:
        """基本面四重过滤：ROE、营收增速、经营现金流、资产负债率

        过滤原则：
        - 有数据的字段正常过滤
        - 缺失的字段跳过该条检查（不因数据缺失而排除）
        - 仅当所有字段全部缺失时才排除
        """
        result = []
        all_missing_count = 0
        filter_stats = {'roe_fail': 0, 'rev_fail': 0, 'ocf_fail': 0, 'debt_fail': 0}

        for stock in pool:
            roe = self.get_financial_field(stock, 'Pershareindex', 'du_return_on_equity')
            rev_growth = self.get_financial_field(stock, 'Pershareindex', 'inc_revenue_rate')
            ocf = self.get_financial_field(stock, 'Pershareindex', 's_fa_ocfps')

            if roe is None and rev_growth is None and ocf is None:
                all_missing_count += 1
                continue

            if roe is not None and roe <= self.params.min_roe:
                filter_stats['roe_fail'] += 1
                continue

            if rev_growth is not None and rev_growth <= self.params.min_revenue_growth:
                filter_stats['rev_fail'] += 1
                continue

            if ocf is not None and ocf <= self.params.min_operate_cashflow:
                filter_stats['ocf_fail'] += 1
                continue

            if self.params.max_debt_ratio is not None and self.params.max_debt_ratio < 1.0:
                total_assets = self.get_financial_field(stock, 'Balance', 'total_assets')
                total_liabilities = self.get_financial_field(stock, 'Balance', 'total_liabilities')
                if total_assets and total_assets > 0 and total_liabilities is not None:
                    debt_ratio = total_liabilities / total_assets
                    if debt_ratio > self.params.max_debt_ratio:
                        filter_stats['debt_fail'] += 1
                        continue

            result.append(stock)

        self.log(f'基本面过滤统计: {len(pool)} -> {len(result)} 只 '
                 f'(全缺={all_missing_count}, ROE不达标={filter_stats["roe_fail"]}, '
                 f'营收不达标={filter_stats["rev_fail"]}, 现金流不达标={filter_stats["ocf_fail"]}, '
                 f'负债率不达标={filter_stats["debt_fail"]})')

        if not result and all_missing_count == len(pool) and self.params.skip_fundamental_if_missing:
            self.log(f'[WARN] 所有股票财务数据缺失({all_missing_count}/{len(pool)})，跳过基本面过滤')
            return pool

        return result

    def _calc_market_caps(self, stocks: List[str]) -> Dict[str, float]:
        """计算市值 = 总股本 × 当前股价

        优先使用 总股本 = 所有者权益合计 / 每股净资产
        回退方案：使用所有者权益作为市值代理
        """
        result = {}
        no_price = 0
        no_cap_data = 0

        for stock in stocks:
            price = self.get_current_price(stock)
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

    def _filter_momentum(self, stocks: List[str]) -> List[str]:
        """动量过滤：近N日涨幅 > 阈值，避免在下跌趋势中接飞刀"""
        result = []
        period = self.params.momentum_period
        min_momentum = self.params.min_momentum

        for stock in stocks:
            prices = self.get_close_prices(stock, period + 1)
            if len(prices) < period + 1:
                continue

            if prices[0] <= 0:
                continue

            momentum = (prices[-1] - prices[0]) / prices[0]
            if momentum > min_momentum:
                result.append(stock)

        return result

    def _pick_industry_smallest(self, market_caps: Dict[str, float]) -> List[tuple]:
        """每个申万一级行业选市值最小的1只，避免行业集中暴露"""
        industry_stocks: Dict[str, List[tuple]] = defaultdict(list)

        for stock, mc in market_caps.items():
            industry = self.get_industry(stock)
            if industry:
                industry_stocks[industry].append((stock, mc))
            else:
                industry_stocks['未知'].append((stock, mc))

        result = []
        for industry, stocks in industry_stocks.items():
            stocks.sort(key=lambda x: x[1])
            result.append(stocks[0])

        return result
