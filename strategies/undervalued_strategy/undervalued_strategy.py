from typing import Dict, List, Optional
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('undervalued',
                   default_kwargs={'max_stocks': 50},
                   backtest_config={'cash': 1000000, 'commission': 0.0013,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '沪深300'})
class UndervaluedStrategy(StockSelectionStrategy):
    """低估价值选股策略 - 迈克尔·普莱斯与本杰明·格雷厄姆价值选股法

    选股逻辑：
    1. PB < 1.8（股价与每股净值比小于1.8，严格低估）
    2. 资产负债率 > 市场均值（负债比例高于市场平均，逆向思维：市场给予低估值）
    3. 流动比率 >= 1.2（流动资产至少是流动负债的1.2倍，确保短期偿债能力）
    4. 动量过滤：20日跌幅不超过-8%（排除处于下跌趋势的股票）

    调仓规则：
    - 月度调仓，等权重持仓（优化后，原季度调仓）

    数据来源：
    - PB: 价格 / 每股净资产(Pershareindex.s_fa_bps)
    - 资产负债率: Balance.total_liabilities / Balance.total_assets
    - 流动比率: Balance.total_current_assets / Balance.total_current_liability
    - 动量: 20日收益率
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 50),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('max_pb', 1.8),
        ('min_current_ratio', 1.2),
        ('use_momentum_filter', True),
        ('momentum_days', 20),
        ('min_momentum', -0.08),
        ('skip_fundamental_if_missing', True),
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        if not pool:
            self.log('股票池为空')
            return []

        pb_data = self._calc_pb_ratios(pool)
        balance_data = self._get_balance_data(pool)

        avg_debt_ratio = self._calc_avg_debt_ratio(pool, balance_data)

        filtered = self._apply_filters(pool, pb_data, balance_data, avg_debt_ratio)

        if not filtered:
            self.log('筛选后无股票')
            return []

        if self.params.use_momentum_filter:
            filtered = self._apply_momentum_filter(filtered)

        if not filtered:
            self.log('过滤后无股票')
            return []

        scored = self._score_stocks(filtered, pb_data)

        max_stocks = self.params.max_stocks
        selected = [stock for stock, _ in scored[:max_stocks]]

        self.log(f'选股结果: {len(pool)} -> {len(filtered)} -> {len(selected)} 只')
        for stock, score in scored[:10]:
            pb = pb_data.get(stock)
            bd = balance_data.get(stock, {})
            debt_ratio = bd.get('debt_ratio') or 0
            cr = bd.get('current_ratio') or 0
            self.log(f'  {stock} | PB: {pb:.2f} | 负债率: {debt_ratio:.2%} | 流动比率: {cr:.2f}')

        return selected

    def _calc_pb_ratios(self, pool: List[str]) -> Dict[str, float]:
        pershare_fields = ['s_fa_bps']
        pershare_data = self.get_financial_fields_batch(pool, 'Pershareindex', pershare_fields)

        result = {}
        for stock in pool:
            price = self.get_current_price(stock)
            if price is None or price <= 0:
                continue
            stock_pershare = pershare_data.get(stock, {})
            bps = stock_pershare.get('s_fa_bps')
            if bps and bps > 0:
                pb = price / bps
                result[stock] = pb
        return result

    def _get_balance_data(self, pool: List[str]) -> Dict[str, Dict]:
        balance_fields = ['total_assets', 'total_liabilities', 'total_current_assets', 'total_current_liability']
        balance_data = self.get_financial_fields_batch(pool, 'Balance', balance_fields)

        result = {}
        for stock in pool:
            stock_balance = balance_data.get(stock, {})
            total_assets = stock_balance.get('total_assets')
            total_liabilities = stock_balance.get('total_liabilities')
            total_current_assets = stock_balance.get('total_current_assets')
            total_current_liability = stock_balance.get('total_current_liability')

            debt_ratio = None
            if total_assets and total_assets > 0 and total_liabilities is not None:
                debt_ratio = total_liabilities / total_assets

            current_ratio = None
            if total_current_liability and total_current_liability > 0 and total_current_assets is not None:
                current_ratio = total_current_assets / total_current_liability

            result[stock] = {
                'debt_ratio': debt_ratio,
                'current_ratio': current_ratio,
            }
        return result

    def _calc_avg_debt_ratio(self, pool: List[str], balance_data: Dict[str, Dict]) -> Optional[float]:
        ratios = []
        for stock in pool:
            debt_ratio = balance_data.get(stock, {}).get('debt_ratio')
            if debt_ratio is not None:
                ratios.append(debt_ratio)

        if not ratios:
            return None
        avg = sum(ratios) / len(ratios)
        self.log(f'股票池平均资产负债率: {avg:.2%} ({len(ratios)}/{len(pool)} 有数据)')
        return avg

    def _apply_filters(self, pool: List[str], pb_data: Dict[str, float],
                       balance_data: Dict[str, Dict], avg_debt_ratio: Optional[float]) -> List[str]:
        result = []
        missing_count = 0

        for stock in pool:
            pb = pb_data.get(stock)
            bd = balance_data.get(stock, {})
            debt_ratio = bd.get('debt_ratio')
            current_ratio = bd.get('current_ratio')

            if pb is None and debt_ratio is None and current_ratio is None:
                missing_count += 1

            if pb is None or pb >= self.params.max_pb:
                continue

            if avg_debt_ratio is not None:
                if debt_ratio is None or debt_ratio <= avg_debt_ratio:
                    continue

            if current_ratio is None or current_ratio < self.params.min_current_ratio:
                continue

            result.append(stock)

        if not result and missing_count == len(pool) and self.params.skip_fundamental_if_missing:
            self.log(f'[WARN] 所有股票财务数据缺失({missing_count}/{len(pool)})，跳过基本面过滤')
            return list(pool)

        return result

    def _score_stocks(self, stocks: List[str], pb_data: Dict[str, float]) -> List[tuple]:
        scored = []
        for stock in stocks:
            pb = pb_data.get(stock, float('inf'))
            score = -pb
            scored.append((stock, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _apply_momentum_filter(self, stocks: List[str]) -> List[str]:
        result = []
        for stock in stocks:
            closes = self.get_close_prices(stock, self.params.momentum_days + 1)
            if len(closes) < 2:
                continue
            ret = (closes[-1] / closes[0]) - 1
            if ret >= self.params.min_momentum:
                result.append(stock)
        self.log(f'动量过滤: {len(stocks)} -> {len(result)} 只 (最低动量:{self.params.min_momentum:.2%})')
        return result

    def on_backtest_end(self):
        pass
