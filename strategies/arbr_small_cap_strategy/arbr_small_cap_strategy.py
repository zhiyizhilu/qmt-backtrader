import datetime as dt_module
from typing import Dict, List, Optional
import numpy as np
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('arbr_small_cap',
                   default_kwargs={'max_stocks': 3},
                   backtest_config={'cash': 1000000, 'commission': 0.0003,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中证1000'})
class ARBRSmallCapStrategy(StockSelectionStrategy):
    """ARBR因子+小市值选股策略 - 基于ARBR情绪因子筛选+小市值排序

    克隆自聚宽文章: https://www.joinquant.com/view/community/detail/44699
    原始策略: 10年52倍，年化59%，全新因子方法超稳定

    选股逻辑：
    1. 从全A股中过滤：剔除创业板/科创板/北交所/ST/停牌/退市/涨跌停/次新股
    2. 计算ARBR因子值，标准化后筛选(-1, 1)范围内的股票
    3. 按流通市值升序排列
    4. 过滤EPS<=0的股票
    5. 取前N只

    调仓规则：
    - 每周调仓
    - 4月5日-4月30日空仓
    - 持仓中昨日涨停的股票，若今日打开涨停则卖出

    已知偏差（vs聚宽原版）：
    - 原版使用聚宽jqfactor的ARBR因子，本版自行计算ARBR指标
    - 原版使用valuation.circulating_market_cap（流通市值），本版用cap_stk*price（总市值）近似
    - 原版使用indicator.eps，本版用Pershareindex.eps_diluted
    - 原版因子预处理含行业均值填充NaN，本版简化为直接剔除NaN
    - 原版涨停检查使用1分钟数据，本版使用日线数据近似判断
    """

    params = (
        ('rebalance_freq', 'weekly'),
        ('max_stocks', 3),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        # 策略特有参数
        ('arbr_period', 26),              # ARBR计算周期
        ('arbr_low', -1.0),               # ARBR标准化值下限
        ('arbr_high', 1.0),               # ARBR标准化值上限
        ('min_ipo_days', 375),            # 上市天数下限（约252个交易日对应自然日约375天）
        ('skip_date_start', '04-05'),     # 空仓起始日期（月-日）
        ('skip_date_end', '04-30'),       # 空仓结束日期（月-日）
        ('stoploss_limit', 0.09),         # 个股止损线
        ('take_profit_ratio', 2.0),       # 止盈线（成本价的倍数）
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._high_limit_list: List[str] = []
        self._lifecycle_manager = None

    def _get_lifecycle_manager(self):
        if self._lifecycle_manager is None:
            try:
                from core.stock_lifecycle import get_lifecycle_manager
                self._lifecycle_manager = get_lifecycle_manager()
            except Exception:
                pass
        return self._lifecycle_manager

    def on_bar(self, bar):
        # 每日检查涨停和止损
        self._prepare_high_limit_list()
        self._check_limit_up()
        self._check_stoploss()
        super().on_bar(bar)

    def select_stocks(self) -> List[str]:
        current_date = self.get_current_date()
        if current_date is None:
            return []

        # 检查是否为空仓期（4月5日-4月30日）
        if self._is_skip_period(current_date):
            self.log(f'空仓期({self.params.skip_date_start}~{self.params.skip_date_end})，清仓')
            return []

        pool = self.get_stock_pool()
        self.log(f'股票池: {len(pool)} 只')

        # 步骤1: 基本过滤
        filtered = self._filter_basic(pool)
        self.log(f'基本过滤: {len(pool)} -> {len(filtered)} 只')

        if not filtered:
            self.log('基本过滤后无股票')
            return []

        # 步骤2: ARBR因子筛选
        filtered = self._filter_by_arbr(filtered)
        self.log(f'ARBR因子筛选: -> {len(filtered)} 只')

        if not filtered:
            self.log('ARBR因子筛选后无股票')
            return []

        # 步骤3: EPS过滤 + 市值排序
        selected = self._select_by_market_cap(filtered)
        self.log(f'选股结果: {len(pool)} -> {len(selected)} 只')

        for stock in selected:
            mc = self._get_market_cap(stock)
            price = self.get_current_price(stock)
            mc_str = f'{mc / 1e8:.2f}亿' if mc else 'N/A'
            price_str = f'{price:.2f}' if price else 'N/A'
            self.log(f'  {stock} | 市值: {mc_str} | 价格: {price_str}')

        return selected

    # ================================================================
    # 空仓期判断
    # ================================================================

    def _is_skip_period(self, current_date) -> bool:
        """判断当前日期是否在空仓期内（4月5日-4月30日）"""
        if isinstance(current_date, dt_module.datetime):
            current_date = current_date.date()
        mm_dd = current_date.strftime('%m-%d')
        return self.params.skip_date_start <= mm_dd <= self.params.skip_date_end

    # ================================================================
    # 过滤方法
    # ================================================================

    def _filter_basic(self, stock_list: List[str]) -> List[str]:
        """基本过滤：科创北交、创业板、ST/退市、次新股、停牌、涨跌停"""
        result = []
        holdings = set(self._current_holdings.keys())

        for stock in stock_list:
            # 过滤创业板(30)、科创板(68)、北交所(8开头/4开头)
            code = stock.split('.')[0] if '.' in stock else stock
            if code.startswith('3') or code.startswith('68') or code.startswith('8') or code.startswith('4'):
                continue

            # 过滤ST和退市
            lifecycle = self._get_lifecycle_manager()
            if lifecycle is not None:
                info = lifecycle._data.get(stock)
                if info and info.get('name'):
                    name = info['name']
                    if 'ST' in name or '*' in name or '退' in name:
                        continue

            # 过滤次新股
            if lifecycle is not None:
                list_date_str = lifecycle.get_list_date(stock)
                if list_date_str:
                    try:
                        list_date = dt_module.date.fromisoformat(list_date_str[:10])
                        current_date = self.get_current_date()
                        if current_date:
                            if isinstance(current_date, dt_module.datetime):
                                current = current_date.date()
                            else:
                                current = current_date
                            if (current - list_date).days < self.params.min_ipo_days:
                                continue
                    except (ValueError, TypeError):
                        pass

            # 过滤停牌
            if self.is_suspended(stock):
                continue

            # 过滤涨停（持仓中的不过滤）
            if stock not in holdings and self.is_limit_up(stock):
                continue

            # 过滤跌停（持仓中的不过滤）
            if stock not in holdings and self.is_limit_down(stock):
                continue

            result.append(stock)

        return result

    def _filter_by_arbr(self, stock_list: List[str]) -> List[str]:
        """ARBR因子筛选：计算ARBR值，标准化后筛选指定范围内的股票"""
        if not stock_list:
            return []

        # 计算所有股票的ARBR值
        arbr_values = {}
        period = self.params.arbr_period

        for stock in stock_list:
            arbr = self._calc_arbr(stock, period)
            if arbr is not None:
                arbr_values[stock] = arbr

        if not arbr_values:
            self.log('无法计算任何股票的ARBR值，跳过因子筛选')
            return stock_list

        # 标准化（z-score）
        values = list(arbr_values.values())
        arr = np.array(values)

        # 去极值：MAD法
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad > 0:
            scale = 5  # 与原版winsorize_med的scale=5一致
            lower = median - scale * 1.4826 * mad
            upper = median + scale * 1.4826 * mad
            arr = np.clip(arr, lower, upper)

        # 标准化
        mean = np.mean(arr)
        std = np.std(arr)
        if std > 0:
            arr = (arr - mean) / std

        # 筛选标准化值在指定范围内的股票
        result = []
        for i, stock in enumerate(arbr_values.keys()):
            z_score = arr[i]
            if self.params.arbr_low <= z_score <= self.params.arbr_high:
                result.append(stock)

        self.log(f'ARBR筛选: {len(arbr_values)} -> {len(result)} 只 (范围: [{self.params.arbr_low}, {self.params.arbr_high}])')
        return result

    def _get_ohlcv_for_arbr(self, stock: str, period: int):
        """获取OHLCV数据用于ARBR计算，优先使用lazy feed作为回退"""
        # 先尝试主数据源
        ohlcv = self.get_ohlcv_data(stock, period + 1)
        if ohlcv is not None and len(ohlcv) >= period + 1:
            return ohlcv

        # 回退到lazy feed
        try:
            df = self.get_lazy_daily_data(stock, n_days=period + 5)
            if df is not None and len(df) >= period + 1:
                result = []
                for idx, row in df.iterrows():
                    result.append({
                        'open': float(row.get('open', 0)),
                        'high': float(row.get('high', 0)),
                        'low': float(row.get('low', 0)),
                        'close': float(row.get('close', 0)),
                        'volume': float(row.get('volume', 0)),
                    })
                return result[-(period + 1):]
        except Exception:
            pass

        return None

    def _calc_arbr(self, stock: str, period: int = 26) -> Optional[float]:
        """计算ARBR指标值

        AR = SUM(HIGH - OPEN, N) / SUM(OPEN - LOW, N) * 100
        BR = SUM(MAX(HIGH - PREV_CLOSE, 0), N) / SUM(MAX(PREV_CLOSE - LOW, 0), N) * 100

        Returns:
            ARBR = AR - BR 的差值，用于衡量市场情绪偏离
        """
        data = self._get_ohlcv_for_arbr(stock, period)
        if data is None or len(data) < period + 1:
            return None

        # 取最近period+1天的数据（需要前一天收盘价计算BR）
        data = data[-(period + 1):]

        sum_ho = 0.0  # SUM(HIGH - OPEN)
        sum_ol = 0.0  # SUM(OPEN - LOW)
        sum_hpc = 0.0  # SUM(MAX(HIGH - PREV_CLOSE, 0))
        sum_pcl = 0.0  # SUM(MAX(PREV_CLOSE - LOW, 0))

        for i in range(1, len(data)):
            high = data[i].get('high', 0)
            open_ = data[i].get('open', 0)
            low = data[i].get('low', 0)
            prev_close = data[i - 1].get('close', 0)

            if open_ <= 0 or prev_close <= 0:
                continue

            sum_ho += max(high - open_, 0)
            sum_ol += max(open_ - low, 0)
            sum_hpc += max(high - prev_close, 0)
            sum_pcl += max(prev_close - low, 0)

        # 计算AR和BR
        ar = (sum_ho / sum_ol * 100) if sum_ol > 0 else None
        br = (sum_hpc / sum_pcl * 100) if sum_pcl > 0 else None

        if ar is None or br is None:
            return None

        # 返回AR与BR的差值作为因子值
        return ar - br

    def _select_by_market_cap(self, stock_list: List[str]) -> List[str]:
        """按市值升序排列，过滤EPS<=0，取前N只"""
        if not stock_list:
            return []

        # 批量获取EPS（QMT字段名为s_fa_eps_diluted）
        eps_data = self.get_financial_fields_batch(stock_list, 'Pershareindex', ['s_fa_eps_diluted'])

        # 计算市值并过滤
        market_caps = {}
        no_eps_count = 0
        no_mc_count = 0
        for stock in stock_list:
            # EPS过滤
            eps = eps_data.get(stock, {}).get('s_fa_eps_diluted')
            if eps is None or eps <= 0:
                no_eps_count += 1
                continue

            # 计算市值
            mc = self._get_market_cap(stock)
            if mc and mc > 0:
                market_caps[stock] = mc
            else:
                no_mc_count += 1

        self.log(f'EPS<=0: {no_eps_count}只, 无市值数据: {no_mc_count}只, 有效: {len(market_caps)}只')

        if not market_caps:
            self.log('无符合市值条件的股票')
            return []

        # 按市值升序排列
        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1])

        # 取前N只
        max_stocks = int(self.params.max_stocks)
        selected = [stock for stock, _ in sorted_stocks[:max_stocks]]

        return selected

    def _get_market_cap(self, stock: str) -> Optional[float]:
        """获取总市值 = 总股本 × 当前价格"""
        # 优先使用get_current_price（支持lazy feed回退）
        price = self.get_current_price(stock)
        if price is None or price <= 0:
            # 回退到unadjusted price
            price = self.get_unadjusted_price(stock)
        if price is None or price <= 0:
            return None

        # 优先使用财报中的总股本(cap_stk)
        cap_stk = self.get_financial_field(stock, 'Balance', 'cap_stk')
        if cap_stk and cap_stk > 0:
            return cap_stk * price

        # 回退：使用 total_equity/bps 估算总股本
        total_equity = self.get_financial_field(stock, 'Balance', 'total_equity')
        bps = self.get_financial_field(stock, 'Pershareindex', 's_fa_bps')
        if total_equity and total_equity > 0 and bps and bps > 0:
            total_shares = total_equity / bps
            return total_shares * price
        return None

    # ================================================================
    # 涨停检查
    # ================================================================

    def _prepare_high_limit_list(self):
        """检查持仓中昨日涨停的股票"""
        holdings = self.get_current_holdings()
        self._high_limit_list = []

        for symbol in holdings:
            ohlcv = self.get_ohlcv_data(symbol, 2)
            if ohlcv and len(ohlcv) >= 2:
                prev = ohlcv[-2]
                prev_close = prev.get('close', 0)
                prev_high_limit = prev.get('high_limit', 0)
                if prev_high_limit > 0 and prev_close >= prev_high_limit - 0.005:
                    self._high_limit_list.append(symbol)

    def _check_limit_up(self):
        """昨日涨停今天开板的卖出"""
        if not self._high_limit_list:
            return

        for symbol in self._high_limit_list[:]:
            if not self.is_limit_up(symbol):
                pos_size = self._current_holdings.get(symbol, 0)
                if pos_size > 0:
                    sellable = self.get_sellable_volume(symbol)
                    if sellable > 0:
                        price = self.get_current_price(symbol)
                        if price and price > 0 and not self.is_limit_down(symbol):
                            self.sell(symbol, price, sellable)
                            if symbol in self._current_holdings:
                                del self._current_holdings[symbol]
                            self.log(f'涨停打开，卖出: {symbol}')
            else:
                self.log(f'涨停继续持有: {symbol}')

    # ================================================================
    # 止损止盈
    # ================================================================

    def _check_stoploss(self):
        """个股止损止盈检查"""
        for symbol in list(self._current_holdings.keys()):
            pos_size = self._current_holdings.get(symbol, 0)
            if pos_size <= 0:
                continue

            price = self.get_current_price(symbol)
            if not price or price <= 0:
                continue

            avg_cost = self._get_avg_cost(symbol)
            if avg_cost is None or avg_cost <= 0:
                continue

            # 止盈
            if price >= avg_cost * self.params.take_profit_ratio:
                sellable = self.get_sellable_volume(symbol)
                if sellable > 0 and not self.is_limit_down(symbol) and not self.is_suspended(symbol):
                    self.sell(symbol, price, sellable)
                    if symbol in self._current_holdings:
                        del self._current_holdings[symbol]
                    self.log(f'收益{self.params.take_profit_ratio - 1:.0%}止盈，卖出: {symbol}')

            # 止损
            elif price < avg_cost * (1 - self.params.stoploss_limit):
                sellable = self.get_sellable_volume(symbol)
                if sellable > 0 and not self.is_limit_down(symbol) and not self.is_suspended(symbol):
                    self.sell(symbol, price, sellable)
                    if symbol in self._current_holdings:
                        del self._current_holdings[symbol]
                    self.log(f'止损(亏损{self.params.stoploss_limit:.0%})，卖出: {symbol}')

    def _get_avg_cost(self, symbol: str) -> Optional[float]:
        """获取持仓均价"""
        try:
            if self.executor and hasattr(self.executor, 'get_avg_cost'):
                cost = self.executor.get_avg_cost(symbol)
                if cost and cost > 0:
                    return cost
        except Exception:
            pass
        return None

    def on_backtest_end(self):
        pass
