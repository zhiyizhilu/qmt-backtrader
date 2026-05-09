import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('ivff3', default_kwargs={'max_stocks': 50},
                   backtest_config={'cash': 30000000, 'commission': 0.0001,
                                    'start_date': '2015-01-01', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中证1000'})
class IVFF3Strategy(StockSelectionStrategy):
    """特质波动率因子策略 - 基于Fama-French三因子模型的特质波动率选股

    核心逻辑（复现东方证券研报）：
    1. 构建Fama-French三因子（MKT, SMB, HML）
    2. 对每只股票回归日收益率得到残差
    3. 残差的年化标准差即为特质波动率IVFF3
    4. 低特质波动率股票未来预期收益更高（特质波动率之谜）
    5. 选择IVFF3最低的股票构建等权组合

    研报结论：IVFF3在IC和分组回测两方面均优于IVCAPM、IVCARHART、IVFF5
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 50),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('regression_window', 20),
        ('min_regression_window', 15),
        ('annualize_factor', 243),
        ('n_groups', 10),
        ('target_group', 1),
        ('skip_fundamental_if_missing', True),
        ('min_roe', 0.0),
        ('min_profit_growth', None),
        ('max_debt_ratio', None),
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._factor_cache: Dict[str, Dict] = {}

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        if not pool:
            self.log('股票池为空')
            return []

        filtered = self._filter_stocks(pool)
        if not filtered:
            self.log('过滤后无股票')
            return []

        self.log(f'股票过滤: {len(pool)} -> {len(filtered)} 只')

        market_caps = self._calc_market_caps(filtered)
        if not market_caps:
            self.log('无法计算市值')
            return []

        bm_ratios = self._calc_bm_ratios(list(market_caps.keys()))

        daily_returns = self._calc_daily_returns(list(market_caps.keys()))
        if not daily_returns:
            self.log('无法计算日收益率')
            return []

        valid_stocks = [s for s in market_caps if s in daily_returns and s in bm_ratios]
        if len(valid_stocks) < 20:
            self.log(f'有效股票不足: {len(valid_stocks)} 只')
            return []

        self.log(f'有效股票: {len(valid_stocks)} 只 (有市值/BM/收益率数据)')

        factors = self._construct_ff3_factors(valid_stocks, market_caps, bm_ratios, daily_returns)
        if factors is None:
            self.log('无法构建Fama-French三因子')
            return []

        ivff3_scores = self._calc_ivff3(valid_stocks, daily_returns, factors)
        if not ivff3_scores:
            self.log('无法计算IVFF3')
            return []

        ivff3_ranked = sorted(ivff3_scores.items(), key=lambda x: x[1])

        group_size = max(1, len(ivff3_ranked) // self.params.n_groups)
        target_end = group_size * self.params.target_group
        target_start = group_size * (self.params.target_group - 1)
        selected = [s for s, _ in ivff3_ranked[target_start:target_end]]

        max_stocks = self.params.max_stocks
        selected = selected[:max_stocks]

        self.log(f'IVFF3选股: {len(ivff3_scores)} 只中选 {len(selected)} 只 (第{self.params.target_group}组, 共{self.params.n_groups}组)')
        for stock in selected[:10]:
            iv = ivff3_scores.get(stock, 0)
            industry = self.get_industry(stock) or '未知'
            mc = market_caps.get(stock, 0) / 1e8
            self.log(f'  {stock} | 行业: {industry} | IVFF3: {iv:.4f} | 市值: {mc:.1f}亿')

        return selected

    def _filter_stocks(self, pool: List[str]) -> List[str]:
        result = []
        need_fundamental = (self.params.min_roe > 0 or
                           self.params.min_profit_growth is not None or
                           self.params.max_debt_ratio is not None)

        if not need_fundamental:
            return list(pool)

        fields = ['du_return_on_equity']
        if self.params.min_profit_growth is not None:
            fields.append('inc_net_profit_rate')

        batch_data = self.get_financial_fields_batch(pool, 'Pershareindex', fields)

        need_debt = self.params.max_debt_ratio is not None and self.params.max_debt_ratio < 1.0
        balance_data = {}
        if need_debt:
            balance_fields = ['total_assets', 'total_liabilities']
            balance_data = self.get_financial_fields_batch(pool, 'Balance', balance_fields)

        missing_count = 0
        for stock in pool:
            stock_data = batch_data.get(stock, {})

            roe = stock_data.get('du_return_on_equity')
            if roe is None:
                missing_count += 1

            if self.params.min_roe > 0:
                if roe is None or roe <= self.params.min_roe:
                    continue

            if self.params.min_profit_growth is not None:
                profit_growth = stock_data.get('inc_net_profit_rate')
                if profit_growth is not None and profit_growth <= self.params.min_profit_growth:
                    continue

            if need_debt:
                stock_balance = balance_data.get(stock, {})
                total_assets = stock_balance.get('total_assets')
                total_liabilities = stock_balance.get('total_liabilities')
                if total_assets and total_assets > 0 and total_liabilities is not None:
                    debt_ratio = total_liabilities / total_assets
                    if debt_ratio > self.params.max_debt_ratio:
                        continue

            result.append(stock)

        if not result and missing_count == len(pool) and self.params.skip_fundamental_if_missing:
            self.log(f'[WARN] 所有股票财务数据缺失({missing_count}/{len(pool)})，跳过基本面过滤')
            return list(pool)

        return result

    def _calc_market_caps(self, stocks: List[str]) -> Dict[str, float]:
        result = {}
        balance_fields = ['total_equity']
        balance_data = self.get_financial_fields_batch(stocks, 'Balance', balance_fields)
        pershare_fields = ['s_fa_bps']
        pershare_data = self.get_financial_fields_batch(stocks, 'Pershareindex', pershare_fields)

        for stock in stocks:
            price = self.get_current_price(stock)
            if price is None or price <= 0:
                continue

            stock_balance = balance_data.get(stock, {})
            stock_pershare = pershare_data.get(stock, {})
            total_equity = stock_balance.get('total_equity')
            bps = stock_pershare.get('s_fa_bps')

            if total_equity and total_equity > 0 and bps and bps > 0:
                total_shares = total_equity / bps
                market_cap = total_shares * price
                result[stock] = market_cap
            elif total_equity and total_equity > 0:
                result[stock] = total_equity

        return result

    def _calc_bm_ratios(self, stocks: List[str]) -> Dict[str, float]:
        result = {}
        pershare_fields = ['s_fa_bps']
        pershare_data = self.get_financial_fields_batch(stocks, 'Pershareindex', pershare_fields)

        for stock in stocks:
            price = self.get_current_price(stock)
            if price is None or price <= 0:
                continue

            stock_pershare = pershare_data.get(stock, {})
            bps = stock_pershare.get('s_fa_bps')

            if bps and bps > 0:
                bm = bps / price
                result[stock] = bm

        return result

    def _calc_daily_returns(self, stocks: List[str]) -> Dict[str, np.ndarray]:
        window = self.params.regression_window + 1
        result = {}
        for stock in stocks:
            prices = self.get_close_prices(stock, window)
            if len(prices) < self.params.min_regression_window + 1:
                continue
            prices_arr = np.array(prices, dtype=float)
            valid_mask = np.isfinite(prices_arr) & (prices_arr > 0)
            if valid_mask.sum() < self.params.min_regression_window + 1:
                continue
            returns = np.diff(prices_arr) / prices_arr[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) >= self.params.min_regression_window:
                result[stock] = returns[-self.params.regression_window:]
        return result

    def _construct_ff3_factors(self, stocks: List[str],
                               market_caps: Dict[str, float],
                               bm_ratios: Dict[str, float],
                               daily_returns: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        min_len = min(len(daily_returns[s]) for s in stocks if s in daily_returns)
        if min_len < self.params.min_regression_window:
            return None

        n_days = min_len
        n_stocks = len(stocks)

        cap_median = np.median([market_caps[s] for s in stocks])
        bm_values = [bm_ratios.get(s, 0) for s in stocks]
        bm_30 = np.percentile(bm_values, 30) if bm_values else 0
        bm_70 = np.percentile(bm_values, 70) if bm_values else 0

        size_groups = {}
        for s in stocks:
            size_groups[s] = 'S' if market_caps[s] <= cap_median else 'B'

        bm_groups = {}
        for s in stocks:
            bm = bm_ratios.get(s, 0)
            if bm <= bm_30:
                bm_groups[s] = 'L'
            elif bm >= bm_70:
                bm_groups[s] = 'H'
            else:
                bm_groups[s] = 'M'

        returns_matrix = np.zeros((n_stocks, n_days))
        for i, s in enumerate(stocks):
            ret = daily_returns[s][-n_days:]
            returns_matrix[i, :] = ret

        mkt_factor = np.mean(returns_matrix, axis=0)

        smb_factor = np.zeros(n_days)
        hml_factor = np.zeros(n_days)

        for t in range(n_days):
            sl_rets = [returns_matrix[i, t] for i, s in enumerate(stocks)
                       if size_groups[s] == 'S' and bm_groups[s] == 'L']
            sm_rets = [returns_matrix[i, t] for i, s in enumerate(stocks)
                       if size_groups[s] == 'S' and bm_groups[s] == 'M']
            sh_rets = [returns_matrix[i, t] for i, s in enumerate(stocks)
                       if size_groups[s] == 'S' and bm_groups[s] == 'H']
            bl_rets = [returns_matrix[i, t] for i, s in enumerate(stocks)
                       if size_groups[s] == 'B' and bm_groups[s] == 'L']
            bm_rets = [returns_matrix[i, t] for i, s in enumerate(stocks)
                       if size_groups[s] == 'B' and bm_groups[s] == 'M']
            bh_rets = [returns_matrix[i, t] for i, s in enumerate(stocks)
                       if size_groups[s] == 'B' and bm_groups[s] == 'H']

            small_avg = np.mean(sl_rets + sm_rets + sh_rets) if (sl_rets + sm_rets + sh_rets) else 0
            big_avg = np.mean(bl_rets + bm_rets + bh_rets) if (bl_rets + bm_rets + bh_rets) else 0
            smb_factor[t] = small_avg - big_avg

            high_avg = np.mean(sh_rets + bh_rets) if (sh_rets + bh_rets) else 0
            low_avg = np.mean(sl_rets + bl_rets) if (sl_rets + bl_rets) else 0
            hml_factor[t] = high_avg - low_avg

        return {
            'MKT': mkt_factor,
            'SMB': smb_factor,
            'HML': hml_factor,
        }

    def _calc_ivff3(self, stocks: List[str],
                    daily_returns: Dict[str, np.ndarray],
                    factors: Dict[str, np.ndarray]) -> Dict[str, float]:
        mkt = factors['MKT']
        smb = factors['SMB']
        hml = factors['HML']
        n_days = len(mkt)

        X = np.column_stack([np.ones(n_days), mkt, smb, hml])

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            self.log('因子矩阵奇异，无法计算IVFF3')
            return {}

        result = {}
        for s in stocks:
            ret = daily_returns[s][-n_days:]
            if len(ret) != n_days:
                continue

            y = ret
            beta = XtX_inv @ (X.T @ y)
            residuals = y - X @ beta

            iv = np.std(residuals, ddof=1) * np.sqrt(self.params.annualize_factor)
            if np.isfinite(iv) and iv > 0:
                result[s] = iv

        if result:
            iv_values = np.array(list(result.values()))
            mean_iv = np.mean(iv_values)
            std_iv = np.std(iv_values, ddof=1)
            if std_iv > 0:
                z_scores = {s: (v - mean_iv) / std_iv for s, v in result.items()}
                return z_scores

        return result

    def on_backtest_end(self):
        pass
