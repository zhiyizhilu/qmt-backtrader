import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
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
        ('max_volatility', 0.04),
        ('volatility_period', 20),
        ('min_avg_volume_ratio', None),
        ('volume_period', 20),
        ('use_multi_factor_score', False),
        ('use_relative_momentum', False),
        ('use_industry_momentum', False),
        ('stop_loss_pct', 0.08),
        ('min_market_cap', None),
        ('use_quality_score', False),
        ('max_turnover_ratio', None),
        ('prev_selected_stocks', None),
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._prev_selected: List[str] = []
        self._entry_prices: Dict[str, float] = {}

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

        if self.params.min_market_cap is not None:
            cap_floor = self.params.min_market_cap * 1e8
            before = len(market_caps)
            market_caps = {s: v for s, v in market_caps.items() if v >= cap_floor}
            self.log(f'市值下限过滤: {before} -> {len(market_caps)} 只 (下限{self.params.min_market_cap}亿)')

        if getattr(self.params, 'max_volatility', None) is not None:
            market_caps = self._filter_volatility(market_caps)

        if getattr(self.params, 'min_avg_volume_ratio', None) is not None:
            market_caps = self._filter_volume(market_caps)

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

        if getattr(self.params, 'use_quality_score', False):
            market_caps = self._apply_quality_score(market_caps)

        if getattr(self.params, 'use_multi_factor_score', False):
            selected = self._select_multi_factor(market_caps)
        elif getattr(self.params, 'use_industry_momentum', False):
            selected = self._pick_industry_momentum_weighted(market_caps)
        else:
            industry_best = self._pick_industry_smallest(market_caps)
            if not industry_best:
                self.log('行业分散选股后无股票')
                return []

            industry_best.sort(key=lambda x: x[1])
            max_stocks = self.params.max_stocks
            selected = [stock for stock, _ in industry_best[:max_stocks]]

        if getattr(self.params, 'stop_loss_pct', None) is not None:
            selected = self._apply_stop_loss(selected)

        if getattr(self.params, 'max_turnover_ratio', None) is not None:
            selected = self._apply_turnover_control(selected)

        self.log(f'最终选股: {len(selected)} 只')
        for stock in selected:
            mc = market_caps.get(stock, 0)
            industry = self.get_industry(stock) or '未知'
            self.log(f'  {stock} | 行业: {industry} | 市值: {mc / 1e8:.2f}亿')

        self._prev_selected = list(selected)
        for stock in selected:
            price = self.get_current_price(stock)
            if price and price > 0:
                self._entry_prices[stock] = price

        return selected

    def _filter_fundamentals(self, pool: List[str]) -> List[str]:
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
        period = self.params.momentum_period
        min_momentum = self.params.min_momentum

        if getattr(self.params, 'use_relative_momentum', False):
            return self._filter_relative_momentum(stocks, period, min_momentum)

        result = []
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

    def _filter_relative_momentum(self, stocks: List[str], period: int, min_momentum: float) -> List[str]:
        benchmark_prices = None
        for symbol in self.get_symbols():
            benchmark_prices = self.get_close_prices(symbol, period + 1)
            if len(benchmark_prices) >= period + 1:
                break

        benchmark_momentum = 0.0
        if benchmark_prices and len(benchmark_prices) >= period + 1 and benchmark_prices[0] > 0:
            benchmark_momentum = (benchmark_prices[-1] - benchmark_prices[0]) / benchmark_prices[0]

        result = []
        for stock in stocks:
            prices = self.get_close_prices(stock, period + 1)
            if len(prices) < period + 1:
                continue
            if prices[0] <= 0:
                continue
            momentum = (prices[-1] - prices[0]) / prices[0]
            relative_momentum = momentum - benchmark_momentum
            if relative_momentum > min_momentum:
                result.append(stock)

        self.log(f'相对动量过滤: 基准动量={benchmark_momentum:.4f}, 通过={len(result)} 只')
        return result

    def _filter_volatility(self, market_caps: Dict[str, float]) -> Dict[str, float]:
        max_vol = self.params.max_volatility
        period = getattr(self.params, 'volatility_period', 20)
        result = {}
        for stock, mc in market_caps.items():
            prices = self.get_close_prices(stock, period + 1)
            if len(prices) < period + 1:
                continue
            returns = []
            for i in range(1, len(prices)):
                if prices[i - 1] > 0:
                    returns.append(prices[i] / prices[i - 1] - 1)
            if not returns:
                continue
            daily_vol = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5
            if daily_vol <= max_vol:
                result[stock] = mc

        self.log(f'波动率过滤: {len(market_caps)} -> {len(result)} 只 (上限{max_vol})')
        return result

    def _filter_volume(self, market_caps: Dict[str, float]) -> Dict[str, float]:
        min_ratio = self.params.min_avg_volume_ratio
        period = getattr(self.params, 'volume_period', 20)
        result = {}

        all_volumes = []
        stock_volumes = {}
        for stock in market_caps:
            prices = self.get_close_prices(stock, period + 1)
            if len(prices) < 2:
                continue
            avg_vol = sum(abs(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices)) if prices[i - 1] > 0)
            avg_vol /= max(len(prices) - 1, 1)
            stock_volumes[stock] = avg_vol
            all_volumes.append(avg_vol)

        if not all_volumes:
            return market_caps

        pool_avg = sum(all_volumes) / len(all_volumes)
        threshold = pool_avg * min_ratio

        for stock, mc in market_caps.items():
            vol = stock_volumes.get(stock, 0)
            if vol >= threshold:
                result[stock] = mc

        self.log(f'成交量过滤: {len(market_caps)} -> {len(result)} 只 (阈值={threshold:.6f})')
        return result

    def _apply_quality_score(self, market_caps: Dict[str, float]) -> Dict[str, float]:
        scored = {}
        for stock, mc in market_caps.items():
            score = 0.0
            count = 0

            roe = self.get_financial_field(stock, 'Pershareindex', 'du_return_on_equity')
            if roe is not None:
                score += min(roe / 0.15, 1.0)
                count += 1

            rev_growth = self.get_financial_field(stock, 'Pershareindex', 'inc_revenue_rate')
            if rev_growth is not None:
                score += min(max(rev_growth, 0) / 0.3, 1.0)
                count += 1

            ocf = self.get_financial_field(stock, 'Pershareindex', 's_fa_ocfps')
            if ocf is not None:
                score += 1.0 if ocf > 0 else 0.0
                count += 1

            total_assets = self.get_financial_field(stock, 'Balance', 'total_assets')
            total_liabilities = self.get_financial_field(stock, 'Balance', 'total_liabilities')
            if total_assets and total_assets > 0 and total_liabilities is not None:
                debt_ratio = total_liabilities / total_assets
                score += max(1.0 - debt_ratio / 0.6, 0.0)
                count += 1

            if count > 0:
                final_score = score / count
                if final_score >= 0.3:
                    scored[stock] = mc

        self.log(f'质量评分过滤: {len(market_caps)} -> {len(scored)} 只')
        return scored

    def _select_multi_factor(self, market_caps: Dict[str, float]) -> List[str]:
        scores: Dict[str, float] = {}
        for stock, mc in market_caps.items():
            score = 0.0

            if mc > 0:
                cap_score = max(0, 1.0 - mc / (50 * 1e8))
                score += cap_score * 0.3

            period = self.params.momentum_period
            prices = self.get_close_prices(stock, period + 1)
            if len(prices) >= period + 1 and prices[0] > 0:
                momentum = (prices[-1] - prices[0]) / prices[0]
                mom_score = min(max(momentum, 0) / 0.3, 1.0)
                score += mom_score * 0.3

            roe = self.get_financial_field(stock, 'Pershareindex', 'du_return_on_equity')
            if roe is not None and roe > 0:
                roe_score = min(roe / 0.15, 1.0)
                score += roe_score * 0.2

            rev_growth = self.get_financial_field(stock, 'Pershareindex', 'inc_revenue_rate')
            if rev_growth is not None and rev_growth > 0:
                rev_score = min(rev_growth / 0.3, 1.0)
                score += rev_score * 0.2

            scores[stock] = score

        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        max_stocks = self.params.max_stocks
        selected = [stock for stock, _ in sorted_stocks[:max_stocks]]

        self.log(f'多因子评分选股: {len(market_caps)} 只 -> 选中 {len(selected)} 只')
        return selected

    def _pick_industry_smallest(self, market_caps: Dict[str, float]) -> List[tuple]:
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

    def _pick_industry_momentum_weighted(self, market_caps: Dict[str, float]) -> List[str]:
        industry_stocks: Dict[str, List[tuple]] = defaultdict(list)
        period = self.params.momentum_period

        for stock, mc in market_caps.items():
            industry = self.get_industry(stock) or '未知'
            industry_stocks[industry].append((stock, mc))

        industry_momentum: Dict[str, float] = {}
        industry_candidates: Dict[str, List[tuple]] = {}

        for industry, stocks in industry_stocks.items():
            stocks.sort(key=lambda x: x[1])
            industry_candidates[industry] = stocks

            mom_values = []
            for stock, _ in stocks[:3]:
                prices = self.get_close_prices(stock, period + 1)
                if len(prices) >= period + 1 and prices[0] > 0:
                    mom_values.append((prices[-1] - prices[0]) / prices[0])
            if mom_values:
                industry_momentum[industry] = sum(mom_values) / len(mom_values)
            else:
                industry_momentum[industry] = 0.0

        sorted_industries = sorted(industry_momentum.items(), key=lambda x: x[1], reverse=True)

        max_stocks = self.params.max_stocks
        selected = []
        for industry, _ in sorted_industries:
            if industry in industry_candidates and industry_candidates[industry]:
                selected.append(industry_candidates[industry][0][0])
                if len(selected) >= max_stocks:
                    break

        self.log(f'行业动量加权选股: {len(industry_stocks)} 个行业 -> 选中 {len(selected)} 只')
        return selected

    def _apply_stop_loss(self, selected: List[str]) -> List[str]:
        stop_pct = self.params.stop_loss_pct
        result = []
        stopped = []

        for stock in selected:
            entry_price = self._entry_prices.get(stock)
            current_price = self.get_current_price(stock)
            if entry_price and entry_price > 0 and current_price and current_price > 0:
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct < -stop_pct:
                    stopped.append(stock)
                    continue
            result.append(stock)

        if stopped:
            self.log(f'止损剔除: {stopped} (阈值{-stop_pct * 100:.0f}%)')

        return result

    def _apply_turnover_control(self, selected: List[str]) -> List[str]:
        max_ratio = self.params.max_turnover_ratio
        if not self._prev_selected:
            return selected

        prev_set = set(self._prev_selected)
        new_set = set(selected)

        new_stocks = new_set - prev_set
        total = len(selected)

        if total == 0:
            return selected

        turnover_ratio = len(new_stocks) / total

        if turnover_ratio <= max_ratio:
            return selected

        keep_count = max(1, int(total * max_ratio))
        keep_from_prev = [s for s in selected if s in prev_set]
        keep_from_new = [s for s in selected if s in new_stocks]

        result = list(keep_from_prev)
        remaining_slots = total - len(result)
        result.extend(keep_from_new[:remaining_slots])

        self.log(f'换手率控制: 原换手率={turnover_ratio:.2f}, 控制后={len(set(result) - prev_set) / max(len(result), 1):.2f}')
        return result[:total]
