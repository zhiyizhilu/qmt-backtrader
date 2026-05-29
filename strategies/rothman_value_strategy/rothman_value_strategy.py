from typing import Dict, List, Optional, Set, Any
import pandas as pd
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('rothman_value',
                   default_kwargs={'max_stocks': 20},
                   backtest_config={'cash': 1000000, 'commission': 0.0013,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中证全指'})
class RothmanValueStrategy(StockSelectionStrategy):
    """霍华·罗斯曼审慎致富价值精选策略 - 穿越牛熊基业长青

    忠实复现聚宽原版策略逻辑，6大条件均采用动态阈值（市场平均值）：
    1. 总市值 ≧ 市场平均值（动态阈值）
    2. 最近一季流动比率 ≧ 市场平均值（动态阈值）
    3. 近四季ROE ≧ 市场平均值（多期连续性检验）
    4. 近五年自由现金流量均为正值（FCF = 经营现金流 - 投资现金流，与聚宽一致）
    5. 近四季营收成长率介于6%至30%
    6. 近四季净利润增速介于8%至50%

    排序：按市值从大到小（偏好大型股）

    已知偏差（vs聚宽原版）：
    - 市场平均值在股票池内计算，非全A股（当pool≠沪深A股时）
    - ROE多期检验使用当期市场平均值，非各期独立市场平均值（框架限制）
    - FCF按年度分组汇总（OCF-ICF），与聚宽原版逻辑一致
    - 条件6使用净利润增速替代EPS比值（修正原版注释与代码不一致的bug）
    - 市值使用总市值（总股本×价格），原版使用流通市值

    原版代码bug修正：
    - 条件4（FCF）：原版交集逻辑被覆盖，仅检查最后一年，已修正为5年全检
    - 条件6：原版注释为"盈余成长率8%-50%"但代码用EPS绝对值0.08-0.5，
      已修正为净利润增速8%-50%
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 20),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('min_revenue_growth', 6),
        ('max_revenue_growth', 30),
        ('min_profit_growth', 8),
        ('max_profit_growth', 50),
        ('roe_periods', 4),
        ('revenue_periods', 4),
        ('profit_periods', 4),
        ('fcf_years', 5),
        ('skip_fundamental_if_missing', True),
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        if not pool:
            self.log('股票池为空')
            return []

        balance_data = self._get_balance_data(pool)
        pershare_data = self._get_pershare_data(pool)
        market_caps = self._calc_market_caps(pool, balance_data, pershare_data)

        if not market_caps:
            self.log('无法计算市值')
            return []

        avg_market_cap = self._compute_average(market_caps)
        current_ratios = {s: bd['current_ratio'] for s, bd in balance_data.items()
                          if bd.get('current_ratio') is not None}
        avg_current_ratio = self._compute_average(current_ratios)
        roe_values = {s: pd['du_return_on_equity'] for s, pd in pershare_data.items()
                      if pd.get('du_return_on_equity') is not None}
        avg_roe = self._compute_average(roe_values)

        self.log(f'动态阈值: 平均市值={avg_market_cap / 1e8:.1f}亿, '
                 f'平均流动比率={avg_current_ratio:.2f}, '
                 f'平均ROE={avg_roe:.2f}%')

        l1 = {s for s in pool if market_caps.get(s, 0) >= avg_market_cap} if avg_market_cap else set()
        l2 = {s for s in pool if current_ratios.get(s, 0) >= avg_current_ratio} if avg_current_ratio else set()
        l3_single = {s for s in pool if roe_values.get(s, 0) >= avg_roe} if avg_roe else set()
        l5_single = {s for s in pool
                     if pershare_data.get(s, {}).get('inc_revenue_rate') is not None
                     and self.params.min_revenue_growth
                     <= pershare_data[s]['inc_revenue_rate']
                     <= self.params.max_revenue_growth}
        l6_single = {s for s in pool
                     if pershare_data.get(s, {}).get('inc_net_profit_rate') is not None
                     and self.params.min_profit_growth
                     <= pershare_data[s]['inc_net_profit_rate']
                     <= self.params.max_profit_growth}

        self.log(f'条件1(市值>=市场平均): {len(l1)} 只')
        self.log(f'条件2(流动比率>=市场平均): {len(l2)} 只')
        self.log(f'条件3(ROE>=市场平均): {len(l3_single)} 只')
        self.log(f'条件5(营收增速{self.params.min_revenue_growth:.0f}%'
                 f'-{self.params.max_revenue_growth:.0f}%): {len(l5_single)} 只')
        self.log(f'条件6(利润增速{self.params.min_profit_growth:.0f}%'
                 f'-{self.params.max_profit_growth:.0f}%): {len(l6_single)} 只')

        candidates = l1 & l2 & l3_single & l5_single & l6_single
        self.log(f'单期条件交集: {len(candidates)} 只')

        if not candidates:
            if self.params.skip_fundamental_if_missing:
                self.log('[WARN] 单期条件交集为空，回退到市值排序')
                sorted_by_cap = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
                return [s for s, _ in sorted_by_cap[:self.params.max_stocks]]
            return []

        l3_multi = self._check_roe_multi_period(candidates, avg_roe)
        l4 = self._check_fcf_multi_year(candidates)
        l5_multi = self._check_revenue_growth_multi_period(candidates)
        l6_multi = self._check_profit_growth_multi_period(candidates)

        self.log(f'条件3-多期(近{self.params.roe_periods}季ROE>=市场平均): {len(l3_multi)} 只')
        self.log(f'条件4(近{self.params.fcf_years}年FCF为正): {len(l4)} 只')
        self.log(f'条件5-多期(近{self.params.revenue_periods}季营收增速达标): {len(l5_multi)} 只')
        self.log(f'条件6-多期(近{self.params.profit_periods}季利润增速达标): {len(l6_multi)} 只')

        result = l3_multi & l4 & l5_multi & l6_multi
        self.log(f'全部条件交集: {len(result)} 只')

        if not result:
            if self.params.skip_fundamental_if_missing:
                self.log('[WARN] 全部条件交集为空，回退到单期条件交集')
                candidates_list = sorted(candidates,
                                         key=lambda s: market_caps.get(s, 0), reverse=True)
                return candidates_list[:self.params.max_stocks]
            return []

        result_list = sorted(result, key=lambda s: market_caps.get(s, 0), reverse=True)
        selected = result_list[:self.params.max_stocks]

        self.log(f'选股结果: {len(pool)} -> {len(result)} -> {len(selected)} 只')
        for stock in selected[:10]:
            pd = pershare_data.get(stock, {})
            mc = market_caps.get(stock, 0)
            parts = [f'市值:{mc / 1e8:.0f}亿']
            if pd.get('du_return_on_equity') is not None:
                parts.append(f'ROE:{pd["du_return_on_equity"]:.2f}%')
            if pd.get('inc_revenue_rate') is not None:
                parts.append(f'营收增速:{pd["inc_revenue_rate"]:.2f}%')
            if pd.get('inc_net_profit_rate') is not None:
                parts.append(f'利润增速:{pd["inc_net_profit_rate"]:.2f}%')
            cr = balance_data.get(stock, {}).get('current_ratio')
            if cr is not None:
                parts.append(f'流动比率:{cr:.2f}')
            self.log(f'  {stock} | ' + ' | '.join(parts))

        return selected

    def _calc_market_caps(self, stocks: List[str],
                          balance_data: Dict[str, Dict],
                          pershare_data: Dict[str, Dict]) -> Dict[str, float]:
        result = {}
        no_price = 0
        no_cap_data = 0

        for stock in stocks:
            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                no_price += 1
                continue

            total_equity = balance_data.get(stock, {}).get('total_equity')
            bps = pershare_data.get(stock, {}).get('s_fa_bps')

            if total_equity and total_equity > 0 and bps and bps > 0:
                total_shares = total_equity / bps
                market_cap = total_shares * price
                result[stock] = market_cap
            else:
                no_cap_data += 1

        if no_price > 0 or no_cap_data > 0:
            self.log(f'市值计算: {len(stocks)} 只 -> {len(result)} 只有有效数据 '
                     f'(无价格={no_price}, 无股本数据={no_cap_data})')
        else:
            self.log(f'市值计算: {len(stocks)} 只有有效数据')

        return result

    def _get_balance_data(self, stocks: List[str]) -> Dict[str, Dict]:
        fields = ['total_current_assets', 'total_current_liability', 'total_equity']
        batch_data = self.get_financial_fields_batch(stocks, 'Balance', fields)

        result = {}
        for stock in stocks:
            stock_data = batch_data.get(stock, {})
            tca = stock_data.get('total_current_assets')
            tcl = stock_data.get('total_current_liability')

            current_ratio = None
            if tca is not None and tcl is not None and tcl > 0:
                current_ratio = tca / tcl

            result[stock] = {
                'current_ratio': current_ratio,
                'total_equity': stock_data.get('total_equity'),
            }
        return result

    def _get_pershare_data(self, stocks: List[str]) -> Dict[str, Dict]:
        fields = ['du_return_on_equity', 'inc_revenue_rate', 'inc_net_profit_rate',
                   's_fa_bps']
        batch_data = self.get_financial_fields_batch(stocks, 'Pershareindex', fields)

        result = {}
        for stock in stocks:
            stock_data = batch_data.get(stock, {})
            result[stock] = {
                'du_return_on_equity': stock_data.get('du_return_on_equity'),
                'inc_revenue_rate': stock_data.get('inc_revenue_rate'),
                'inc_net_profit_rate': stock_data.get('inc_net_profit_rate'),
                's_fa_bps': stock_data.get('s_fa_bps'),
            }
        return result

    def _compute_average(self, values: Dict[str, float]) -> float:
        valid = sorted([v for v in values.values() if v is not None])
        if not valid:
            return 0.0
        n = len(valid)
        lo = int(n * 0.05)
        hi = int(n * 0.95)
        trimmed = valid[lo:n - hi] if hi > lo else valid
        return sum(trimmed) / len(trimmed) if trimmed else 0.0

    def _check_roe_multi_period(self, candidates: Set[str], avg_roe: float) -> Set[str]:
        result = set()
        periods = self.params.roe_periods
        for stock in candidates:
            history = self.get_financial_history(
                stock, 'Pershareindex', 'du_return_on_equity', count=periods)
            if len(history) >= periods and all(
                    roe is not None and roe >= avg_roe for roe in history[-periods:]):
                result.add(stock)
        return result

    def _check_fcf_multi_year(self, candidates: Set[str]) -> Set[str]:
        result = set()
        years_needed = self.params.fcf_years
        query_date = self.get_current_date()
        if query_date is None:
            return result

        for stock in candidates:
            df = self._get_cashflow_df_full(stock, query_date)
            if df is None or df.empty:
                continue

            ocf_col = 'net_cash_flows_oper_act'
            icf_col = 'net_cash_flows_inv_act'
            if ocf_col not in df.columns or icf_col not in df.columns:
                continue

            df_fcf = df[[ocf_col, icf_col]].dropna()
            if df_fcf.empty:
                continue

            df_fcf['fcf'] = df_fcf[ocf_col] - df_fcf[icf_col]
            df_fcf['year'] = df_fcf.index.year

            yearly_fcf = df_fcf.groupby('year')['fcf'].sum()

            current_year = query_date.year
            target_years = [current_year - y for y in range(1, years_needed + 1)]
            fcf_values = [yearly_fcf.get(y) for y in target_years]

            if all(v is not None and v > 0 for v in fcf_values):
                result.add(stock)
        return result

    def _get_cashflow_df(self, stock: str, query_date) -> Optional[Any]:
        if not self._financial_data_adapter:
            return None
        cache = self._financial_data_adapter.cache
        cache._ensure_table_loaded(stock, 'CashFlow')
        stock_data = cache._data.get(stock, {})
        table_df = stock_data.get('CashFlow')
        if table_df is None or table_df.empty:
            return None
        if not isinstance(table_df.index, pd.DatetimeIndex):
            try:
                table_df.index = pd.to_datetime(table_df.index)
                cache._data[stock]['CashFlow'] = table_df
            except Exception:
                return None
        return table_df[table_df.index <= pd.Timestamp(query_date)]

    def _get_cashflow_df_full(self, stock: str, query_date) -> Optional[pd.DataFrame]:
        from core.cache import cache_manager as _cm

        namespace = 'QMTDataProcessor_Financial'
        table_suffix = 'CashFlow_announce_time'

        available_years = _cm.index_manager.get_available_financial_years(stock, table_suffix)
        if not available_years:
            available_years = _cm.disk_cache.list_yearly_files(namespace, stock, table_suffix)

        if not available_years:
            return None

        current_year = query_date.year if query_date else 2023
        needed_years = [y for y in available_years if y <= current_year]
        if not needed_years:
            return None

        df = _cm.disk_cache.get_yearly_range(namespace, stock, sorted(needed_years), table_suffix)
        if df is None or df.empty:
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                return None

        return df[df.index <= pd.Timestamp(query_date)]

    def _check_revenue_growth_multi_period(self, candidates: Set[str]) -> Set[str]:
        result = set()
        periods = self.params.revenue_periods
        for stock in candidates:
            history = self.get_financial_history(
                stock, 'Pershareindex', 'inc_revenue_rate', count=periods)
            if len(history) >= periods and all(
                    r is not None
                    and self.params.min_revenue_growth <= r <= self.params.max_revenue_growth
                    for r in history[-periods:]):
                result.add(stock)
        return result

    def _check_profit_growth_multi_period(self, candidates: Set[str]) -> Set[str]:
        result = set()
        periods = self.params.profit_periods
        for stock in candidates:
            history = self.get_financial_history(
                stock, 'Pershareindex', 'inc_net_profit_rate', count=periods)
            if len(history) >= periods and all(
                    r is not None
                    and self.params.min_profit_growth <= r <= self.params.max_profit_growth
                    for r in history[-periods:]):
                result.add(stock)
        return result

    def on_backtest_end(self):
        pass
