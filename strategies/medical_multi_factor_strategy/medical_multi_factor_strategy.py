import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('medical_multi_factor',
                   default_kwargs={'max_stocks': 20, 'use_ic_abs_weight': True, 'min_quality_score': 2},
                   backtest_config={'cash': 10000000, 'commission': 0.0001,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中证全指',
                                    'benchmark': '000985.SH'})
class MedicalMultiFactorStrategy(StockSelectionStrategy):
    """医疗行业多因子选股策略 - 基于rank IC赋权的多因子模型

    核心逻辑（复现聚宽社区策略）：
    1. 股票池：中证全指成分股中非ST股
    2. 因子池（显著性0.2）：PB值、市值对数、换手率、ROE
    3. 因子权重：学习周期内各因子的rank IC均值
    4. 合成组合因子，选取分数最高的N只股票
    5. 等权持仓，月度调仓

    因子说明：
    - PB值：市净率，低PB倾向价值股（反向因子）
    - 市值对数：小市值效应
    - 换手率：低换手率倾向更高收益（反向因子）
    - ROE：盈利能力因子，高ROE倾向更高收益

    数据预处理：
    - 去极值：中位数去极值法（MAD）
    - 标准化：Z-score标准化
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 20),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('learning_period', 12),
        ('turnover_window', 20),
        ('ic_threshold', 0.02),
        ('mad_scale', 1.4826),
        ('mad_threshold', 3.0),
        ('max_volatility', None),
        ('stop_loss', None),
        ('max_industry_stocks', None),
        ('min_momentum', None),
        ('momentum_days', 20),
        ('min_roe', None),
        ('use_ic_abs_weight', True),
        ('min_quality_score', 2),
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._ic_history: Dict[str, List[float]] = {}
        self._factor_weights: Dict[str, float] = {}

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        if not pool:
            self.log('股票池为空')
            return []

        pool = self._filter_st(pool)
        if not pool:
            self.log('过滤ST后无股票')
            return []

        if self.params.max_volatility is not None:
            pool = self._filter_volatility(pool)
            if not pool:
                self.log('波动率过滤后无股票')
                return []

        if self.params.min_roe is not None:
            pool = self._filter_roe(pool)
            if not pool:
                self.log('ROE过滤后无股票')
                return []

        if self.params.min_quality_score is not None and self.params.min_quality_score > 0:
            pool = self._filter_quality(pool)
            if not pool:
                self.log('质量评分过滤后无股票')
                return []

        factor_data = self._calc_factors(pool)
        if not factor_data:
            self.log('无法计算因子')
            return []

        valid_stocks = [s for s in pool if s in factor_data and factor_data[s]]
        if len(valid_stocks) < 5:
            self.log(f'有效因子数据不足: {len(valid_stocks)} 只')
            return []

        self.log(f'因子计算: {len(pool)} -> {len(valid_stocks)} 只有完整数据')

        returns_data = self._calc_period_returns(valid_stocks)
        self._update_ic_history(valid_stocks, factor_data, returns_data)

        self._update_factor_weights()

        if not self._factor_weights:
            self.log('因子权重为空，使用等权')
            factor_names = list(factor_data[valid_stocks[0]].keys())
            self._factor_weights = {f: 1.0 / len(factor_names) for f in factor_names}

        normalized = self._normalize_factors(valid_stocks, factor_data)

        composite_scores = self._calc_composite_scores(valid_stocks, normalized)

        if self.params.min_momentum is not None:
            composite_scores = self._apply_momentum_filter(composite_scores)

        if self.params.stop_loss is not None:
            composite_scores = self._apply_stop_loss(composite_scores)

        if not composite_scores:
            self.log('过滤后无股票可选')
            return []

        max_stocks = self.params.max_stocks
        ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)

        if self.params.max_industry_stocks is not None:
            selected = self._apply_industry_limit(ranked, self.params.max_industry_stocks, max_stocks)
        else:
            selected = [s for s, _ in ranked[:max_stocks]]

        self.log(f'选股结果: {len(pool)} -> {len(valid_stocks)} -> {len(selected)} 只')
        for stock in selected[:10]:
            score = composite_scores.get(stock, 0)
            industry = self.get_industry(stock) or '未知'
            self.log(f'  {stock} | 行业: {industry} | 综合得分: {score:.4f}')

        self.log(f'因子权重: {self._factor_weights}')

        return selected

    def _filter_st(self, pool: List[str]) -> List[str]:
        result = []
        for stock in pool:
            name = ''
            try:
                from core.cache import cache_manager
                info = cache_manager.index_manager.get_stock_info(stock)
                if info:
                    name = info.get('name', '')
            except Exception:
                pass
            if 'ST' in name or 'st' in name:
                continue
            result.append(stock)
        return result if result else list(pool)

    def _filter_volatility(self, pool: List[str]) -> List[str]:
        threshold = self.params.max_volatility
        result = []
        for stock in pool:
            closes = self.get_close_prices(stock, 21)
            if len(closes) < 2:
                continue
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]
            if not returns:
                continue
            daily_vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
            if daily_vol <= threshold:
                result.append(stock)
        self.log(f'波动率过滤: {len(pool)} -> {len(result)} 只 (阈值={threshold})')
        return result

    def _filter_roe(self, pool: List[str]) -> List[str]:
        min_roe = self.params.min_roe
        pershare_data = self.get_financial_fields_batch(pool, 'Pershareindex', ['du_return_on_equity'])
        result = []
        for stock in pool:
            stock_pershare = pershare_data.get(stock, {})
            roe = stock_pershare.get('du_return_on_equity')
            if roe is not None and roe >= min_roe:
                result.append(stock)
        self.log(f'ROE过滤: {len(pool)} -> {len(result)} 只 (阈值={min_roe})')
        return result

    def _filter_quality(self, pool: List[str]) -> List[str]:
        min_score = self.params.min_quality_score
        fields = ['du_return_on_equity', 'inc_net_profit_rate', 's_fa_ocfps']
        pershare_data = self.get_financial_fields_batch(pool, 'Pershareindex', fields)
        result = []
        for stock in pool:
            stock_data = pershare_data.get(stock, {})
            roe = stock_data.get('du_return_on_equity')
            profit_growth = stock_data.get('inc_net_profit_rate')
            ocfps = stock_data.get('s_fa_ocfps')
            score = 0
            if roe is not None and roe > 0:
                score += 1
            if profit_growth is not None and profit_growth > 0:
                score += 1
            if ocfps is not None and ocfps > 0:
                score += 1
            if score >= min_score:
                result.append(stock)
        self.log(f'质量评分过滤: {len(pool)} -> {len(result)} 只 (最低分={min_score})')
        return result

    def _apply_momentum_filter(self, scores: Dict[str, float]) -> Dict[str, float]:
        min_mom = self.params.min_momentum
        days = self.params.momentum_days
        result = {}
        for stock, score in scores.items():
            closes = self.get_close_prices(stock, days + 1)
            if len(closes) >= 2 and closes[0] > 0:
                mom = (closes[-1] - closes[0]) / closes[0]
                if mom >= min_mom:
                    result[stock] = score
        self.log(f'动量过滤: {len(scores)} -> {len(result)} 只 (最低动量={min_mom})')
        return result

    def _apply_stop_loss(self, scores: Dict[str, float]) -> Dict[str, float]:
        stop_loss = self.params.stop_loss
        holdings = self.get_current_holdings()
        result = dict(scores)
        for stock, volume in holdings.items():
            if volume > 0 and stock in result:
                lookback = 22
                closes = self.get_close_prices(stock, lookback + 1)
                if len(closes) >= 2 and closes[0] > 0:
                    loss = (closes[-1] - closes[0]) / closes[0]
                    if loss <= stop_loss:
                        del result[stock]
                        self.log(f'止损剔除: {stock}, {lookback}日亏损={loss*100:.1f}%')
        return result

    def _apply_industry_limit(self, ranked: List[Tuple[str, float]],
                               max_per_industry: int, max_total: int) -> List[str]:
        industry_count: Dict[str, int] = defaultdict(int)
        selected = []
        for stock, score in ranked:
            industry = self.get_industry(stock) or '未知'
            if industry_count[industry] < max_per_industry:
                selected.append(stock)
                industry_count[industry] += 1
            if len(selected) >= max_total:
                break
        self.log(f'行业分散: {len(ranked)} -> {len(selected)} 只 (每行业最多{max_per_industry}只)')
        return selected

    def _calc_factors(self, pool: List[str]) -> Dict[str, Dict[str, float]]:
        pb_data = self._calc_pb(pool)
        market_cap_data = self._calc_market_cap(pool)
        turnover_data = self._calc_turnover(pool)
        roe_data = self._calc_roe(pool)

        result = {}
        for stock in pool:
            factors = {}
            if stock in pb_data:
                factors['pb'] = pb_data[stock]
            if stock in market_cap_data:
                factors['log_market_cap'] = math.log(market_cap_data[stock]) if market_cap_data[stock] > 0 else None
            if stock in turnover_data:
                factors['turnover'] = turnover_data[stock]
            if stock in roe_data:
                factors['roe'] = roe_data[stock]

            if len(factors) >= 3:
                clean = {k: v for k, v in factors.items() if v is not None and math.isfinite(v)}
                if len(clean) >= 3:
                    result[stock] = clean

        return result

    def _calc_pb(self, pool: List[str]) -> Dict[str, float]:
        pershare_data = self.get_financial_fields_batch(pool, 'Pershareindex', ['s_fa_bps'])
        result = {}
        for stock in pool:
            price = self.get_current_price(stock)
            if price is None or price <= 0:
                continue
            stock_pershare = pershare_data.get(stock, {})
            bps = stock_pershare.get('s_fa_bps')
            if bps and bps > 0:
                result[stock] = price / bps
        return result

    def _calc_market_cap(self, pool: List[str]) -> Dict[str, float]:
        balance_data = self.get_financial_fields_batch(pool, 'Balance', ['total_equity'])
        pershare_data = self.get_financial_fields_batch(pool, 'Pershareindex', ['s_fa_bps'])
        result = {}
        for stock in pool:
            price = self.get_current_price(stock)
            if price is None or price <= 0:
                continue
            stock_balance = balance_data.get(stock, {})
            stock_pershare = pershare_data.get(stock, {})
            total_equity = stock_balance.get('total_equity')
            bps = stock_pershare.get('s_fa_bps')
            if total_equity and total_equity > 0 and bps and bps > 0:
                total_shares = total_equity / bps
                result[stock] = total_shares * price
            elif total_equity and total_equity > 0:
                result[stock] = total_equity
        return result

    def _calc_turnover(self, pool: List[str]) -> Dict[str, float]:
        window = self.params.turnover_window
        balance_data = self.get_financial_fields_batch(pool, 'Balance', ['total_equity'])
        pershare_data = self.get_financial_fields_batch(pool, 'Pershareindex', ['s_fa_bps'])
        result = {}
        for stock in pool:
            ohlcv = self.get_ohlcv_data(stock, window + 1)
            if len(ohlcv) < window:
                continue
            recent = ohlcv[-window:]
            total_volume = sum(bar.get('volume', 0) for bar in recent)
            avg_volume = total_volume / window

            stock_balance = balance_data.get(stock, {})
            stock_pershare = pershare_data.get(stock, {})
            total_equity = stock_balance.get('total_equity')
            bps = stock_pershare.get('s_fa_bps')

            if total_equity and total_equity > 0 and bps and bps > 0:
                total_shares = total_equity / bps
                if total_shares > 0:
                    turnover_rate = avg_volume / total_shares
                    if math.isfinite(turnover_rate):
                        result[stock] = turnover_rate
        return result

    def _calc_roe(self, pool: List[str]) -> Dict[str, float]:
        pershare_data = self.get_financial_fields_batch(pool, 'Pershareindex', ['du_return_on_equity'])
        result = {}
        for stock in pool:
            stock_pershare = pershare_data.get(stock, {})
            roe = stock_pershare.get('du_return_on_equity')
            if roe is not None and math.isfinite(roe):
                result[stock] = roe
        return result

    def _calc_period_returns(self, pool: List[str]) -> Dict[str, float]:
        result = {}
        for stock in pool:
            closes = self.get_close_prices(stock, 22)
            if len(closes) >= 2 and closes[0] > 0:
                ret = (closes[-1] - closes[0]) / closes[0]
                if math.isfinite(ret):
                    result[stock] = ret
        return result

    def _update_ic_history(self, stocks: List[str],
                           factor_data: Dict[str, Dict[str, float]],
                           returns_data: Dict[str, float]):
        common_stocks = [s for s in stocks if s in returns_data]
        if len(common_stocks) < 10:
            return

        factor_names = set()
        for s in common_stocks:
            factor_names.update(factor_data.get(s, {}).keys())

        for fname in factor_names:
            pairs = []
            for s in common_stocks:
                fval = factor_data.get(s, {}).get(fname)
                rval = returns_data.get(s)
                if fval is not None and rval is not None and math.isfinite(fval) and math.isfinite(rval):
                    pairs.append((fval, rval))

            if len(pairs) < 10:
                continue

            ic = self._spearman_rank_correlation(pairs)
            if ic is not None and math.isfinite(ic):
                if fname not in self._ic_history:
                    self._ic_history[fname] = []
                self._ic_history[fname].append(ic)
                max_history = self.params.learning_period
                if len(self._ic_history[fname]) > max_history:
                    self._ic_history[fname] = self._ic_history[fname][-max_history:]

    def _spearman_rank_correlation(self, pairs: List[Tuple[float, float]]) -> Optional[float]:
        n = len(pairs)
        if n < 5:
            return None

        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]

        x_ranks = self._rank(x_vals)
        y_ranks = self._rank(y_vals)

        d_sq_sum = sum((x_ranks[i] - y_ranks[i]) ** 2 for i in range(n))

        denominator = n * (n ** 2 - 1)
        if denominator == 0:
            return None

        return 1 - (6 * d_sq_sum) / denominator

    @staticmethod
    def _rank(values: List[float]) -> List[float]:
        n = len(values)
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    def _update_factor_weights(self):
        if not self._ic_history:
            return

        weights = {}
        total_abs_ic = 0.0

        use_abs = self.params.use_ic_abs_weight

        for fname, ic_list in self._ic_history.items():
            if not ic_list:
                continue
            avg_ic = sum(ic_list) / len(ic_list)
            if abs(avg_ic) >= self.params.ic_threshold:
                if use_abs:
                    weights[fname] = abs(avg_ic)
                else:
                    weights[fname] = avg_ic
                total_abs_ic += abs(avg_ic)

        if total_abs_ic > 0:
            self._factor_weights = {f: w / total_abs_ic for f, w in weights.items()}
        else:
            self._factor_weights = {}

    def _normalize_factors(self, stocks: List[str],
                           factor_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        factor_names = set()
        for s in stocks:
            factor_names.update(factor_data.get(s, {}).keys())

        factor_values = {}
        for fname in factor_names:
            vals = []
            for s in stocks:
                v = factor_data.get(s, {}).get(fname)
                if v is not None and math.isfinite(v):
                    vals.append(v)
            if len(vals) < 5:
                continue
            factor_values[fname] = vals

        result = {}
        for s in stocks:
            result[s] = {}

        for fname, vals in factor_values.items():
            median_val = self._median(vals)
            mad = self._mad(vals)
            mad_scale = self.params.mad_scale
            mad_threshold = self.params.mad_threshold

            trimmed_vals = []
            for v in vals:
                if mad > 0:
                    z = mad_scale * abs(v - median_val) / mad
                    if z <= mad_threshold:
                        trimmed_vals.append(v)
                else:
                    trimmed_vals.append(v)

            if not trimmed_vals:
                continue

            mean_val = sum(trimmed_vals) / len(trimmed_vals)
            std_val = (sum((v - mean_val) ** 2 for v in trimmed_vals) / len(trimmed_vals)) ** 0.5

            for s in stocks:
                v = factor_data.get(s, {}).get(fname)
                if v is not None and math.isfinite(v):
                    if mad > 0:
                        z = mad_scale * abs(v - median_val) / mad
                        if z > mad_threshold:
                            v = median_val + mad_threshold * (1 if v > median_val else -1) * mad / mad_scale
                    if std_val > 0:
                        result[s][fname] = (v - mean_val) / std_val
                    else:
                        result[s][fname] = 0.0

        return result

    @staticmethod
    def _median(values: List[float]) -> float:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        return sorted_vals[n // 2]

    @staticmethod
    def _mad(values: List[float]) -> float:
        med = MedicalMultiFactorStrategy._median(values)
        deviations = [abs(v - med) for v in values]
        return MedicalMultiFactorStrategy._median(deviations)

    def _calc_composite_scores(self, stocks: List[str],
                               normalized: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        result = {}
        for s in stocks:
            score = 0.0
            norm_factors = normalized.get(s, {})
            for fname, weight in self._factor_weights.items():
                fval = norm_factors.get(fname)
                if fval is not None:
                    if fname == 'turnover':
                        score -= abs(weight) * fval
                    elif fname == 'pb':
                        score -= abs(weight) * fval
                    else:
                        score += weight * fval
            result[s] = score
        return result

    def on_backtest_end(self):
        pass
