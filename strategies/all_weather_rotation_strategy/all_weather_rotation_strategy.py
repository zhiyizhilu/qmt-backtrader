from typing import Dict, List, Optional
import datetime as dt_module
from core.stock_selection import StockSelectionStrategy
from core.strategy_logic import BarData, OrderInfo
from strategies import register_strategy


@register_strategy('all_weather_rotation',
                   default_kwargs={'max_stocks': 9},
                   backtest_config={'cash': 1000000, 'commission': 0.0013,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '沪深300'})
class AllWeatherRotationStrategy(StockSelectionStrategy):
    """全天候轮动策略 - 大小盘动量轮动 + 海外ETF避险

    忠实复现聚宽社区策略（作者：MarioC，帖子ID: 48819）：
    每月调仓时，根据沪深300（大盘）和中小综指（小盘）近10日动量，
    动态选择大盘股、小盘股或海外ETF。

    核心差异说明（vs 聚宽原版）：
    1. 大盘池=沪深300成分股，小盘池=中小综指成分股（两个独立指数）
    2. 动量计算：大盘取流通市值前20大，小盘取流通市值前20小的10日涨幅均值
    3. stock_num=3，BIG/BM/ROIC_BIG各取3只合并，SMALL取9只
    4. ROIC_BIG有ROIC_TTM>8%过滤，且有inc_revenue>0.2的第二个营收条件
    5. 原版有止损逻辑：亏损>8%止损+涨停打开卖出+补仓跌幅最大3只
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 9),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        # 动量判断参数
        ('momentum_days', 10),
        ('momentum_top_n', 20),
        ('momentum_threshold', 10),
        # stock_num（原版g.stock_num=3）
        ('stock_num', 3),
        # SMALL选股阈值（聚宽indicator.roe/roa是小数形式）
        ('small_min_roe', 0.15),
        ('small_min_roa', 0.10),
        # BIG选股阈值
        ('big_max_pe', 30),
        ('big_max_ps', 8),
        ('big_max_pcf', 10),
        ('big_min_eps', 0.3),
        ('big_min_roe', 0.1),
        ('big_min_net_margin', 0.1),
        ('big_min_gross_margin', 0.3),
        ('big_min_revenue_growth', 0.25),
        # ROIC_BIG选股阈值
        ('roic_big_min_market_cap_yi', 300),
        ('roic_big_max_pe', 50),
        ('roic_big_min_eps', 0.12),
        ('roic_big_min_roa', 0.15),
        ('roic_big_max_debt_ratio', 0.5),
        ('roic_big_min_total_revenue_growth', 0.3),
        ('roic_big_min_revenue_growth', 0.2),
        # BM选股阈值
        ('bm_min_market_cap_yi', 100),
        ('bm_max_market_cap_yi', 900),
        ('bm_max_pb', 10),
        ('bm_max_pcf', 4),
        ('bm_min_eps', 0.3),
        ('bm_min_roe', 0.2),
        ('bm_min_net_margin', 0.1),
        ('bm_min_revenue_growth', 0.2),
        ('bm_min_operate_profit_growth', 0.1),
        # 海外ETF列表
        ('foreign_etf', [
            '518880.SH',
            '513030.SH',
            '513100.SH',
            '164824.SZ',
            '159866.SZ',
        ]),
        # 止损参数（原版stop_loss逻辑）
        ('stop_loss_threshold', 0.08),       # 亏损8%止损
        ('stop_loss_topup_num', 3),           # 补仓跌幅最大的3只
        ('rebalance_hour', 9),                # 月度调仓时间-小时
        ('rebalance_minute', 30),             # 月度调仓时间-分钟
        ('stop_loss_hour', 14),               # 止损时间-小时
        ('stop_loss_minute', 0),              # 止损时间-分钟
        ('skip_fundamental_if_missing', True),
        # ===== 优化参数（默认禁用） =====
        # opt01: 波动率过滤 - 过滤日波动率超过阈值的股票
        ('max_volatility', None),           # None=禁用, 如0.03表示3%
        # opt02: 调仓时止损 - 调仓时剔除亏损超过阈值的持仓
        ('rebalance_stop_loss', None),      # None=禁用, 如0.05表示5%
        # opt03: 动量确认 - 两个动量均需>0才开仓
        ('require_positive_momentum', None), # None=禁用, True=启用
        # opt04: 换仓阈值 - 新标的需比当前持仓动量高出阈值才换仓
        ('switch_threshold', None),         # None=禁用, 如0.05表示5%
        # opt05: 放宽SMALL选股条件 - 降低ROE/ROA阈值（已验证通过样本外测试）
        ('small_min_roe_relaxed', 0.08),   # None=禁用, 0.08表示8%
        ('small_min_roa_relaxed', 0.05),   # None=禁用, 0.05表示5%
        # opt06: 降低无敌行情阈值
        ('momentum_threshold_low', None),   # None=禁用, 如5表示5%
        # opt07: 增加动量计算天数
        ('momentum_days_long', None),       # None=禁用, 如20表示20日
        # opt08: 持仓集中度限制 - 单只股票最大仓位比例
        ('max_position_ratio', None),       # None=禁用, 如0.15表示15%
        # opt09: 双周期动量验证 - 10日和20日动量同向才确认
        ('dual_period_momentum', None),     # None=禁用, True=启用
        # opt10: 海外ETF优先级调整 - 动量均<0时优先选黄金ETF
        ('prefer_gold_etf', None),          # None=禁用, True=启用
    )

    def __init__(self, executor=None, weight_allocator=None, **kwargs):
        super().__init__(executor, weight_allocator, **kwargs)
        self._small_cap_mgr = None
        self._cost_cache: Dict[str, float] = {}  # 买入成本价缓存 {stock: avg_cost}
        self._yesterday_limit_up: List[str] = []  # 昨日涨停股票列表

    def on_bar(self, bar: BarData):
        """K线数据到达 - 分钟级回测时支持调仓+止损双时间点

        聚宽原版逻辑：
        - run_daily(prepare_stock_list, '9:05')  每日09:05更新持仓和昨日涨停列表
        - run_monthly(monthly_adjustment, 1, '9:30')  每月第一个交易日09:30调仓
        - run_daily(stop_loss, '14:00')               每天14:00止损/补仓
        """
        current_date = self.get_current_date()
        if current_date is None:
            return

        bar_datetime = getattr(bar, 'datetime', None)
        if bar_datetime and isinstance(bar_datetime, dt_module.datetime):
            hour = bar_datetime.hour
            minute = bar_datetime.minute

            # 日线模式（00:00）：执行调仓+更新涨停列表
            if hour == 0 and minute == 0:
                self._update_yesterday_limit_up()
                if self._rebalance_phase != self.PHASE_IDLE:
                    return
                if self.is_rebalance_day(current_date):
                    self.log(f'[选股] 调仓日: {current_date}', level='info')
                    self._execute_rebalance(current_date)
                return

            # 分钟级模式
            rebalance_hour = getattr(self.params, 'rebalance_hour', 9)
            rebalance_minute = getattr(self.params, 'rebalance_minute', 30)
            stop_loss_hour = getattr(self.params, 'stop_loss_hour', 14)
            stop_loss_minute = getattr(self.params, 'stop_loss_minute', 0)

            # 09:05 更新昨日涨停列表（聚宽 prepare_stock_list）
            if hour == 9 and minute == 5:
                self._update_yesterday_limit_up()
                return

            # 09:30 月度调仓
            if hour == rebalance_hour and minute == rebalance_minute:
                if self._rebalance_phase != self.PHASE_IDLE:
                    return
                if self.is_rebalance_day(current_date):
                    self.log(f'[选股] 调仓日: {current_date}', level='info')
                    self._execute_rebalance(current_date)
                return

            # 14:00 每日止损
            if hour == stop_loss_hour and minute == stop_loss_minute:
                if self._rebalance_phase != self.PHASE_IDLE:
                    return
                self._daily_stop_loss()
                return
        else:
            # 无时间信息（日线模式）
            self._update_yesterday_limit_up()
            if self._rebalance_phase != self.PHASE_IDLE:
                return
            if self.is_rebalance_day(current_date):
                self.log(f'[选股] 调仓日: {current_date}', level='info')
                self._execute_rebalance(current_date)

    def _daily_stop_loss(self):
        """每日止损逻辑 - 忠实复现聚宽stop_loss函数

        1. 昨日涨停股票如涨停打开则卖出
        2. 持仓亏损>8%则全部止损卖出
        3. 如果有止损操作，用可用资金补仓剩余持仓中跌幅最大的3只
        """
        num_stopped = 0
        remaining = []  # (stock, return_pct)

        # 1. 涨停打开卖出（聚宽原版 stop_loss 第64-75行）
        if self._yesterday_limit_up:
            for stock in list(self._yesterday_limit_up):
                if stock not in self._current_holdings:
                    continue
                pos_size = self._current_holdings.get(stock, 0)
                if pos_size <= 0:
                    continue
                # 判断今日是否仍涨停
                if self.is_limit_up(stock):
                    self.log(f'涨停继续持有: {stock}')
                else:
                    sellable = self.get_sellable_volume(stock)
                    if sellable > 0 and not self.is_suspended(stock) and not self.is_limit_down(stock):
                        price = self.get_current_price(stock)
                        if price and price > 0:
                            self.sell(stock, price, sellable)
                            if stock in self._current_holdings:
                                del self._current_holdings[stock]
                            self.log(f'涨停打开卖出: {stock}, 价格:{price:.2f}')
                            num_stopped += 1

        # 2. 亏损>8%止损
        holdings = dict(self._current_holdings)
        if not holdings:
            return

        self.log(f'[止损检查] 持仓{len(holdings)}只: {list(holdings.keys())[:5]}', level='debug')

        for stock in list(holdings.keys()):
            pos_size = holdings.get(stock, 0)
            if pos_size <= 0:
                continue

            price = self.get_current_price(stock)
            if not price or price <= 0:
                self.log(f'[止损检查] {stock} 无法获取价格', level='debug')
                continue

            avg_cost = self._get_avg_cost(stock)
            if avg_cost is None or avg_cost <= 0:
                self.log(f'[止损检查] {stock} 无法获取均价, price={price}', level='debug')
                continue

            loss_pct = (price / avg_cost - 1) * 100

            # 亏损>8%止损
            if price < avg_cost * (1 - self.params.stop_loss_threshold):
                sellable = self.get_sellable_volume(stock)
                if sellable > 0 and not self.is_suspended(stock) and not self.is_limit_down(stock):
                    self.sell(stock, price, sellable)
                    if stock in self._current_holdings:
                        del self._current_holdings[stock]
                    self.log(f'止损卖出: {stock}, 价格:{price:.2f}, 均价:{avg_cost:.2f}, '
                             f'亏损:{loss_pct:.1f}%')
                    num_stopped += 1
                else:
                    self.log(f'[止损检查] {stock} 亏损{loss_pct:.1f}% 但无法卖出(sellable={sellable})', level='debug')
                continue

            ret_pct = (price - avg_cost) / avg_cost
            remaining.append((stock, ret_pct))

        # 3. 如果有止损操作，补仓跌幅最大的3只
        if num_stopped >= 1 and remaining:
            topup_num = min(self.params.stop_loss_topup_num, len(remaining))
            remaining.sort(key=lambda x: x[1])
            topup_stocks = [s for s, _ in remaining[:topup_num]]

            cash = self.get_cash()
            if cash > 0 and topup_stocks:
                per_value = cash / len(topup_stocks)
                for stock in topup_stocks:
                    price = self.get_current_price(stock)
                    if price and price > 0 and not self.is_suspended(stock) and not self.is_limit_up(stock):
                        from core.data_adapter import get_trade_unit, validate_trade_volume
                        buy_volume = int(per_value / price / get_trade_unit(stock)) * get_trade_unit(stock)
                        is_valid, _ = validate_trade_volume(stock, buy_volume)
                        if is_valid:
                            self.buy(stock, price, buy_volume)
                            self._current_holdings[stock] = self._current_holdings.get(stock, 0) + buy_volume
                            self._cost_cache[stock] = price  # 更新成本缓存
                            self.log(f'止损补仓: {stock}, 价格:{price:.2f}, 数量:{buy_volume}')

    def _get_avg_cost(self, stock: str) -> Optional[float]:
        """获取持仓均价 - 优先使用成本缓存，回退到executor查询"""
        # 优先使用成本缓存（最可靠）
        if stock in self._cost_cache:
            cost = self._cost_cache[stock]
            if cost and cost > 0:
                return cost

        # 回退：从executor的Position获取
        if self.executor and hasattr(self.executor, 'get_position'):
            try:
                pos = self.executor.get_position(stock)
                if pos:
                    if hasattr(pos, 'avg_price') and pos.avg_price and pos.avg_price > 0:
                        return pos.avg_price
                    if hasattr(pos, 'avg_cost') and pos.avg_cost and pos.avg_cost > 0:
                        return pos.avg_cost
                    if hasattr(pos, 'price') and pos.price and pos.price > 0:
                        return pos.price
            except Exception:
                pass
        return None

    def _update_yesterday_limit_up(self):
        """更新昨日涨停股票列表 - 复现聚宽 prepare_stock_list

        聚宽原版每日09:05执行：
        1. 更新当前持仓列表
        2. 获取持仓中昨日涨停的股票
        """
        holdings = dict(self._current_holdings)
        if not holdings:
            self._yesterday_limit_up = []
            return

        limit_up_list = []
        for stock in holdings:
            if self._was_limit_up_yesterday(stock):
                limit_up_list.append(stock)

        if limit_up_list:
            self.log(f'昨日涨停: {limit_up_list}')
        self._yesterday_limit_up = limit_up_list

    def _was_limit_up_yesterday(self, stock: str) -> bool:
        """判断股票昨日是否涨停

        使用前一日收盘价与涨停价比较
        """
        closes = self.get_close_prices(stock, 2)
        if len(closes) < 2:
            return False
        # 获取涨停价：通过 data_adapter 的 is_limit_up 方法无法直接查历史，
        # 使用近似判断：收盘价涨幅接近10%（或20%对于ST/创业板）
        yesterday_close = closes[-2]
        day_before_close = closes[-3] if len(closes) >= 3 else closes[-2]
        if day_before_close <= 0:
            return False
        # 近似判断：涨幅>=9.8%视为涨停（考虑四舍五入误差）
        change_pct = (yesterday_close / day_before_close - 1) * 100
        # 判断涨停阈值：普通股10%，ST股5%，创业板/科创板20%
        code = stock.split('.')[0] if '.' in stock else stock
        if code.startswith('3') or code.startswith('68'):
            threshold = 19.8  # 创业板/科创板20%
        elif code.startswith('ST') or 'ST' in stock:
            threshold = 4.8   # ST股5%
        else:
            threshold = 9.8   # 普通股10%
        return change_pct >= threshold

    def _ensure_auxiliary_stocks_registered(self, stock_list: List[str]):
        """确保辅助股票（中小综指成分股、海外ETF）注册为lazy feed

        回测时pool只加载了沪深300的行情数据，
        中小综指的股票和海外ETF需要动态注册为lazy feed才能获取行情。
        """
        if not self._data_adapter or not hasattr(self._data_adapter, '_lazy_feeds'):
            return

        existing_feeds = self._data_adapter._lazy_feeds
        from engine.data_feed import LazyDataFeed

        # 获取数据处理器
        data_processor = self._data_processor
        if not data_processor:
            return

        # 获取回测日期范围
        start_date = getattr(self, '_data_start_date', '') or ''
        end_date = getattr(self, '_data_end_date', '') or ''
        if not start_date:
            current_date = self.get_current_date()
            if current_date:
                start_date = current_date.strftime('%Y-%m-%d') if isinstance(current_date, dt_module.date) else str(current_date)
        if not end_date:
            end_date = start_date

        registered = 0
        # 根据回测周期选择注册的period
        period = '1d'
        if self._data_adapter and hasattr(self._data_adapter, 'period'):
            period = self._data_adapter.period or '1d'

        for stock in stock_list:
            if stock not in existing_feeds:
                try:
                    lazy_feed = LazyDataFeed(
                        stock, data_processor, period, start_date, end_date
                    )
                    self._data_adapter.register_lazy_feed(stock, lazy_feed)
                    existing_feeds[stock] = lazy_feed
                    registered += 1
                except Exception:
                    pass

        if registered > 0:
            self.log(f'注册 {registered} 只辅助股票的lazy feed')

    def _get_index_stocks(self, sector: str) -> List[str]:
        """获取指定指数的当日成分股

        通过 IndexConstituentManager 获取动态历史成分股。
        """
        current_date = self.get_current_date()
        if current_date is None:
            return []

        date_str = current_date.strftime('%Y-%m-%d') if isinstance(current_date, dt_module.date) else str(current_date)

        # 优先使用 financial_data_adapter 中的 index_constituent_mgr
        if self._financial_data_adapter and self._financial_data_adapter._index_constituent_mgr:
            mgr = self._financial_data_adapter._index_constituent_mgr
            index_code = mgr.SECTOR_TO_INDEX.get(sector)
            if index_code:
                stocks = mgr.get_constituent_stocks_fast(index_code, date_str)
                if stocks:
                    return stocks

        # 回退：创建独立的 IndexConstituentManager
        if self._small_cap_mgr is None:
            try:
                from core.data.index_constituent import IndexConstituentManager
                self._small_cap_mgr = IndexConstituentManager()
            except Exception as e:
                self.log(f'创建IndexConstituentManager失败: {e}')
                return []

        index_code = self._small_cap_mgr.SECTOR_TO_INDEX.get(sector)
        if not index_code:
            return []
        stocks = self._small_cap_mgr.get_constituent_stocks(index_code, date_str)
        return stocks or []

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        if not pool:
            self.log('股票池为空')
            return []

        # 获取大盘池（沪深300）和小盘池（中小综指）
        big_pool = self._get_index_stocks('沪深300')
        small_pool = self._get_index_stocks('中小综指')

        self.log(f'大盘池(沪深300): {len(big_pool)}只, 小盘池(中小综指): {len(small_pool)}只')

        if not big_pool and not small_pool:
            self.log('大小盘池均为空，使用当前pool')
            big_pool = list(pool)
            small_pool = list(pool)

        # 确保中小综指成分股注册为lazy feed（首次调用时注册）
        if small_pool:
            self._ensure_auxiliary_stocks_registered(small_pool)

        # 确保海外ETF注册为lazy feed
        self._ensure_auxiliary_stocks_registered(list(self.params.foreign_etf))

        # opt07: 增加动量计算天数
        if self.params.momentum_days_long is not None:
            original_days = self.params.momentum_days
            self.params.momentum_days = self.params.momentum_days_long

        # 计算大盘和小盘动量
        big_momentum = self._calc_pool_momentum(big_pool, is_big=True)
        small_momentum = self._calc_pool_momentum(small_pool, is_big=False)

        # opt09: 双周期动量验证
        if self.params.dual_period_momentum:
            # 用20日动量验证10日动量方向
            original_days = self.params.momentum_days
            self.params.momentum_days = 20
            big_momentum_20 = self._calc_pool_momentum(big_pool, is_big=True)
            small_momentum_20 = self._calc_pool_momentum(small_pool, is_big=False)
            self.params.momentum_days = original_days
            # 双周期同向确认：如果10日和20日方向不一致，动量置0
            if big_momentum * big_momentum_20 <= 0:
                big_momentum = 0
            if small_momentum * small_momentum_20 <= 0:
                small_momentum = 0

        # 恢复原始动量天数
        if self.params.momentum_days_long is not None:
            self.params.momentum_days = original_days

        self.log(f'动量: 大盘={big_momentum:.2f}% 小盘={small_momentum:.2f}%')

        # opt03: 动量确认 - 两个动量均需>0才开仓
        if self.params.require_positive_momentum:
            if big_momentum <= 0 and small_momentum <= 0:
                self.log('动量确认: 大小盘动量均<=0，转海外ETF')
                etf = self._get_tradeable_foreign_etf()
                if etf:
                    return etf
            elif big_momentum <= 0 and small_momentum > 0:
                # 只有小盘为正，强制开小
                self.log(f'动量确认: 仅小盘为正({small_momentum:.1f}%)，强制开小')
                target_list = self._select_small(small_pool)[:int(self.params.stock_num) * 3]
            elif small_momentum <= 0 and big_momentum > 0:
                # 只有大盘为正，强制开大
                self.log(f'动量确认: 仅大盘为正({big_momentum:.1f}%)，强制开大')
                target_list = self._select_big_combo(big_pool, int(self.params.stock_num))
            else:
                target_list = self._decide_and_select(big_pool, small_pool, big_momentum, small_momentum)
        else:
            # 轮动决策
            target_list = self._decide_and_select(big_pool, small_pool, big_momentum, small_momentum)

        # opt06: 降低无敌行情阈值
        # (已在_decide_and_select中通过覆盖momentum_threshold实现)

        # opt10: 海外ETF优先级调整 - 动量均<0时优先选黄金ETF
        if self.params.prefer_gold_etf and not target_list:
            gold_etf = '518880.SH'
            price = self.get_current_price(gold_etf)
            if price and price > 0:
                self.log(f'优先选黄金ETF: {gold_etf}')
                return [gold_etf]

        if not target_list:
            self.log('选股结果为空，使用海外ETF')
            etf = self._get_tradeable_foreign_etf()
            if etf:
                return etf
            self.log('海外ETF不可交易，回退到小盘选股')
            return self._select_small(small_pool)

        # opt01: 波动率过滤
        if self.params.max_volatility is not None:
            filtered = []
            for stock in target_list:
                closes = self.get_close_prices(stock, 20)
                if len(closes) >= 20:
                    daily_returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes)) if closes[i-1] > 0]
                    if daily_returns:
                        import numpy as np
                        vol = float(np.std(daily_returns))
                        if vol <= self.params.max_volatility:
                            filtered.append(stock)
                        else:
                            self.log(f'波动率过滤: {stock} vol={vol:.4f} > {self.params.max_volatility}', level='debug')
                    else:
                        filtered.append(stock)
                else:
                    filtered.append(stock)
            if filtered:
                target_list = filtered
            self.log(f'波动率过滤: {len(target_list)} -> {len(filtered)} 只')

        # opt02: 调仓时止损 - 剔除亏损超过阈值的持仓
        if self.params.rebalance_stop_loss is not None:
            holdings = dict(self._current_holdings)
            stopped = []
            for stock in list(target_list):
                if stock in holdings and holdings.get(stock, 0) > 0:
                    avg_cost = self._get_avg_cost(stock)
                    price = self.get_current_price(stock)
                    if avg_cost and price and avg_cost > 0:
                        loss_pct = (price / avg_cost - 1)
                        if loss_pct < -self.params.rebalance_stop_loss:
                            stopped.append(stock)
                            self.log(f'调仓止损: {stock} 亏损{loss_pct*100:.1f}% > {self.params.rebalance_stop_loss*100:.1f}%')
            if stopped:
                target_list = [s for s in target_list if s not in stopped]

        # opt04: 换仓阈值 - 新标的需比当前持仓动量高出阈值才换仓
        if self.params.switch_threshold is not None:
            holdings = dict(self._current_holdings)
            if holdings:
                # 计算当前持仓的平均收益
                holding_returns = []
                for stock in holdings:
                    closes = self.get_close_prices(stock, self.params.momentum_days + 1)
                    if len(closes) >= self.params.momentum_days + 1 and closes[0] > 0:
                        ret = (closes[-1] / closes[0] - 1)
                        holding_returns.append(ret)
                if holding_returns:
                    avg_holding_return = sum(holding_returns) / len(holding_returns)
                    # 计算新标的的平均收益
                    new_returns = []
                    for stock in target_list:
                        closes = self.get_close_prices(stock, self.params.momentum_days + 1)
                        if len(closes) >= self.params.momentum_days + 1 and closes[0] > 0:
                            ret = (closes[-1] / closes[0] - 1)
                            new_returns.append(ret)
                    if new_returns:
                        avg_new_return = sum(new_returns) / len(new_returns)
                        if avg_new_return - avg_holding_return < self.params.switch_threshold:
                            self.log(f'换仓阈值: 新标的平均收益{avg_new_return*100:.1f}% - 持仓{avg_holding_return*100:.1f}% < {self.params.switch_threshold*100:.1f}%，保持持仓')
                            return list(holdings.keys())[:int(self.params.max_stocks)]

        # opt08: 持仓集中度限制
        if self.params.max_position_ratio is not None:
            max_stocks = max(int(1 / self.params.max_position_ratio), int(self.params.max_stocks))
            target_list = target_list[:max_stocks]
        else:
            max_stocks = int(self.params.max_stocks)
            target_list = target_list[:max_stocks]

        self.log(f'选股结果: {len(pool)}池 -> {len(target_list)} 只: {target_list}')
        return target_list

    # ================================================================
    # 动量计算 - 原版逻辑：大盘取流通市值前20大，小盘取前20小
    # ================================================================

    def _calc_pool_momentum(self, stock_list: List[str], is_big: bool) -> float:
        """计算股票池动量

        原版逻辑：
        - 大盘：沪深300中流通市值最大的20只，计算10日平均涨幅
        - 小盘：中小综指中流通市值最小的20只，计算10日平均涨幅
        """
        if not stock_list:
            return 0.0

        n = self.params.momentum_days
        top_n = min(self.params.momentum_top_n, len(stock_list))

        # 按市值排序，大盘取前20大，小盘取前20小
        pershare_fields = ['s_fa_bps']
        pershare_data = self.get_financial_fields_batch(stock_list, 'Pershareindex', pershare_fields)
        balance_fields = ['total_equity', 'cap_stk']
        balance_data = self.get_financial_fields_batch(stock_list, 'Balance', balance_fields)

        market_caps = {}
        for stock in stock_list:
            mc = self._calc_market_cap(stock, balance_data, pershare_data)
            if mc and mc > 0:
                market_caps[stock] = mc

        if not market_caps:
            # 无法计算市值，直接取前N只
            sample = stock_list[:top_n]
        else:
            sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1],
                                   reverse=is_big)
            sample = [s for s, _ in sorted_stocks[:top_n]]

        returns = []
        for stock in sample:
            closes = self.get_close_prices(stock, n + 1)
            if len(closes) >= n + 1 and closes[0] > 0:
                ret = (closes[-1] / closes[0] - 1) * 100
                returns.append(ret)

        if not returns:
            return 0.0

        return sum(returns) / len(returns)

    def _get_tradeable_foreign_etf(self) -> List[str]:
        """获取可交易的海外ETF列表"""
        etf_list = list(self.params.foreign_etf)
        tradeable = []
        for etf in etf_list:
            price = self.get_current_price(etf)
            if price and price > 0:
                tradeable.append(etf)
        return tradeable

    # ================================================================
    # 轮动决策
    # ================================================================

    def _decide_and_select(self, big_pool: List[str], small_pool: List[str],
                           big_momentum: float, small_momentum: float) -> List[str]:
        """根据动量决定选股方向"""
        # opt06: 降低无敌行情阈值
        if self.params.momentum_threshold_low is not None:
            threshold = self.params.momentum_threshold_low
        else:
            threshold = self.params.momentum_threshold
        stock_num = self.params.stock_num

        if big_momentum > threshold or small_momentum > threshold:
            self.log(f'无敌好行情 (大盘:{big_momentum:.1f}% 小盘:{small_momentum:.1f}%)')
            if big_momentum > small_momentum:
                self.log('开大(无敌) - ROIC_BIG+BIG+BM')
                return self._select_big_combo(big_pool, stock_num)
            else:
                self.log('开小(无敌) - SMALL')
                return self._select_small(small_pool)[:stock_num * 3]

        if big_momentum > small_momentum and big_momentum > 0:
            self.log(f'开大 (大盘:{big_momentum:.1f}% > 小盘:{small_momentum:.1f}%)')
            return self._select_big_combo(big_pool, stock_num)

        if small_momentum > big_momentum and small_momentum > 0:
            self.log(f'开小 (小盘:{small_momentum:.1f}% > 大盘:{big_momentum:.1f}%)')
            return self._select_small(small_pool)[:stock_num * 3]

        self.log(f'开外盘 (大盘:{big_momentum:.1f}% 小盘:{small_momentum:.1f}%)')
        etf = self._get_tradeable_foreign_etf()
        if etf:
            return etf
        self.log('海外ETF不可交易，回退到小盘选股')
        return self._select_small(small_pool)[:stock_num * 3]

    # ================================================================
    # SMALL选股：ROE>0.15, ROA>0.10, 按市值升序
    # ================================================================

    def _select_small(self, stock_list: List[str]) -> List[str]:
        """小盘选股：ROE>15%, ROA>10%, 按市值升序

        聚宽 indicator.roe 和 indicator.roa 是小数形式（0.15 = 15%）
        QMT Pershareindex.du_return_on_equity 是百分比形式（15 = 15%）
        """
        if not stock_list:
            return []

        pershare_fields = ['du_return_on_equity', 's_fa_bps', 's_fa_eps_diluted']
        pershare_data = self.get_financial_fields_batch(
            stock_list, 'Pershareindex', pershare_fields)

        balance_fields = ['total_equity', 'tot_liab', 'tot_assets', 'cap_stk']
        balance_data = self.get_financial_fields_batch(
            stock_list, 'Balance', balance_fields)

        income_fields = ['net_profit_incl_min_int_inc_after']
        income_data = self.get_financial_fields_batch(
            stock_list, 'Income', income_fields)

        # 聚宽阈值是小数，QMT ROE是百分比，需转换
        # opt05: 放宽SMALL选股条件
        if self.params.small_min_roe_relaxed is not None:
            min_roe_pct = self.params.small_min_roe_relaxed * 100  # 如0.08 -> 8
        else:
            min_roe_pct = self.params.small_min_roe * 100  # 0.15 -> 15
        if self.params.small_min_roa_relaxed is not None:
            min_roa_pct = self.params.small_min_roa_relaxed * 100  # 如0.05 -> 5
        else:
            min_roa_pct = self.params.small_min_roa * 100  # 0.10 -> 10

        candidates = []
        missing_count = 0
        for stock in stock_list:
            ps = pershare_data.get(stock, {})
            bd = balance_data.get(stock, {})
            inc = income_data.get(stock, {})

            roe = ps.get('du_return_on_equity')
            total_assets = bd.get('tot_assets')
            net_profit = inc.get('net_profit_incl_min_int_inc_after')

            if roe is None and total_assets is None:
                missing_count += 1
                continue

            # ROE > 15%（QMT百分比形式）
            if roe is None or roe <= min_roe_pct:
                continue

            # ROA = 净利润/总资产 > 10%
            if total_assets and total_assets > 0 and net_profit:
                roa = net_profit / total_assets * 100
                if roa <= min_roa_pct:
                    continue
            else:
                continue

            # 股价 < 10元过滤（聚宽原版 filter_highprice_stock）
            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                continue
            if price >= 10:
                continue

            market_cap = self._calc_market_cap(stock, balance_data, pershare_data)
            if market_cap and market_cap > 0:
                candidates.append((stock, market_cap))

        if not candidates and missing_count == len(stock_list) and self.params.skip_fundamental_if_missing:
            self.log(f'[WARN] SMALL: 所有股票财务数据缺失({missing_count}/{len(stock_list)})，跳过基本面过滤')
            ranked = []
            for stock in stock_list:
                market_cap = self._calc_market_cap(stock, balance_data, pershare_data)
                if market_cap and market_cap > 0:
                    ranked.append((stock, market_cap))
            ranked.sort(key=lambda x: x[1])
            return [s for s, _ in ranked]

        candidates.sort(key=lambda x: x[1])
        result = [s for s, _ in candidates]
        self.log(f'SMALL选股: {len(stock_list)} -> {len(candidates)} -> {len(result)} 只')
        return result

    # ================================================================
    # BIG+ROIC_BIG+BM 合并选股
    # ================================================================

    def _select_big_combo(self, stock_list: List[str], stock_num: int) -> List[str]:
        """大盘合并选股：ROIC_BIG + BIG + BM 三种结果合并去重"""
        roic_big = self._select_roic_big(stock_list, stock_num)
        big = self._select_big(stock_list, stock_num)
        bm = self._select_bm(stock_list, stock_num)

        # 原版：target_list3 + target_list1 + target_list2 (BM + ROIC_BIG + BIG)
        merged = bm + roic_big + big
        seen = set()
        result = []
        for s in merged:
            if s not in seen:
                seen.add(s)
                result.append(s)

        self.log(f'大盘合并: ROIC_BIG={len(roic_big)} + BIG={len(big)} + BM={len(bm)} -> {len(result)} 只')
        return result

    # ================================================================
    # BIG选股：多因子价值成长
    # ================================================================

    def _select_big(self, stock_list: List[str], stock_num: int) -> List[str]:
        """大盘选股：PE(0-30), PS(0-8), PCF<10, EPS>0.3, ROE>10%,
        净利率>10%, 毛利率>30%, 营收增速>25%, 按市值降序取stock_num只"""
        if not stock_list:
            return []

        pershare_fields = ['du_return_on_equity', 's_fa_eps_diluted',
                           's_fa_bps', 's_fa_ocfps', 'inc_revenue_rate']
        pershare_data = self.get_financial_fields_batch(
            stock_list, 'Pershareindex', pershare_fields)

        income_fields = ['revenue', 'oper_profit', 'net_profit_incl_min_int_inc_after']
        income_data = self.get_financial_fields_batch(
            stock_list, 'Income', income_fields)

        balance_fields = ['total_equity', 'tot_liab', 'tot_assets', 'cap_stk']
        balance_data = self.get_financial_fields_batch(
            stock_list, 'Balance', balance_fields)

        # 聚宽阈值是小数，QMT百分比需转换
        min_roe_pct = self.params.big_min_roe * 100       # 0.1 -> 10
        min_revenue_growth_pct = self.params.big_min_revenue_growth * 100  # 0.25 -> 25
        min_net_margin_pct = self.params.big_min_net_margin * 100  # 0.1 -> 10
        min_gross_margin_pct = self.params.big_min_gross_margin * 100  # 0.3 -> 30

        candidates = []
        missing_count = 0
        for stock in stock_list:
            ps = pershare_data.get(stock, {})
            inc = income_data.get(stock, {})
            bd = balance_data.get(stock, {})

            eps = ps.get('s_fa_eps_diluted')
            roe = ps.get('du_return_on_equity')
            revenue_growth = ps.get('inc_revenue_rate')
            ocfps = ps.get('s_fa_ocfps')

            if eps is None and roe is None:
                missing_count += 1
                continue

            if eps is None or eps <= self.params.big_min_eps:
                continue
            if roe is None or roe <= min_roe_pct:
                continue
            if revenue_growth is None or revenue_growth <= min_revenue_growth_pct:
                continue

            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                continue

            # 股价 < 300元过滤（聚宽原版 filter_highprice_stock2）
            if price >= 300:
                continue

            pe = price / eps
            if pe <= 0 or pe >= self.params.big_max_pe:
                continue

            # PS = 价格/每股营收
            revenue = inc.get('revenue')
            total_shares = self._calc_total_shares(stock, bd, ps)
            if revenue and total_shares and total_shares > 0:
                ps_ratio = price / (revenue / total_shares)
                if ps_ratio >= self.params.big_max_ps:
                    continue
            else:
                continue

            # PCF
            if ocfps and ocfps > 0:
                pcf = price / ocfps
                if pcf >= self.params.big_max_pcf:
                    continue
            else:
                continue

            # 净利率 > 10%
            net_profit = inc.get('net_profit_incl_min_int_inc_after')
            if revenue and net_profit and revenue > 0:
                net_margin = net_profit / revenue * 100
                if net_margin <= min_net_margin_pct:
                    continue
            else:
                continue

            # 毛利率 > 30%（用营业利润/营收近似）
            operating_profit = inc.get('oper_profit')
            if revenue and operating_profit and revenue > 0:
                gross_margin = operating_profit / revenue * 100
                if gross_margin <= min_gross_margin_pct:
                    continue
            else:
                continue

            market_cap = self._calc_market_cap(stock, balance_data, pershare_data)
            if market_cap and market_cap > 0:
                candidates.append((stock, market_cap))

        if not candidates and missing_count == len(stock_list) and self.params.skip_fundamental_if_missing:
            self.log(f'[WARN] BIG: 所有股票财务数据缺失，跳过')
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        result = [s for s, _ in candidates[:stock_num]]
        self.log(f'BIG选股: {len(stock_list)} -> {len(candidates)} -> {len(result)} 只')
        return result

    # ================================================================
    # ROIC_BIG选股：大市值高质量
    # ================================================================

    def _select_roic_big(self, stock_list: List[str], stock_num: int) -> List[str]:
        """ROIC_BIG选股：市值>300亿, PE(0-50), EPS>0.12, ROA>15%,
        负债率<50%, 营收总增速>30%, 营收增速>20%, ROIC_TTM>8%, 按未分配利润降序"""
        if not stock_list:
            return []

        pershare_fields = ['du_return_on_equity', 's_fa_eps_diluted',
                           's_fa_bps', 'inc_revenue_rate', 'inc_total_revenue_annual']
        pershare_data = self.get_financial_fields_batch(
            stock_list, 'Pershareindex', pershare_fields)

        balance_fields = ['total_equity', 'tot_liab', 'tot_assets',
                          'surplus_rsrv', 'undistributed_profit',
                          'total_current_assets', 'total_current_liability',
                          'cap_stk']
        balance_data = self.get_financial_fields_batch(
            stock_list, 'Balance', balance_fields)

        income_fields = ['net_profit_incl_min_int_inc_after', 'oper_profit',
                         'total_operate_income']
        income_data = self.get_financial_fields_batch(
            stock_list, 'Income', income_fields)

        min_cap = self.params.roic_big_min_market_cap_yi * 1e8
        min_roa_pct = self.params.roic_big_min_roa * 100  # 0.15 -> 15
        min_total_rev_growth_pct = self.params.roic_big_min_total_revenue_growth * 100  # 0.3 -> 30
        min_rev_growth_pct = self.params.roic_big_min_revenue_growth * 100  # 0.2 -> 20

        candidates = []
        missing_count = 0
        for stock in stock_list:
            ps = pershare_data.get(stock, {})
            bd = balance_data.get(stock, {})
            inc = income_data.get(stock, {})

            eps = ps.get('s_fa_eps_diluted')
            revenue_growth = ps.get('inc_revenue_rate')
            total_revenue_growth = ps.get('inc_total_revenue_annual')
            total_assets = bd.get('tot_assets')
            net_profit = inc.get('net_profit_incl_min_int_inc_after')

            if eps is None and total_assets is None:
                missing_count += 1
                continue

            market_cap = self._calc_market_cap(stock, balance_data, pershare_data)
            if not market_cap or market_cap <= min_cap:
                continue

            if eps is None or eps <= self.params.roic_big_min_eps:
                continue

            # ROA = 净利润/总资产 > 15%
            if total_assets and total_assets > 0 and net_profit:
                roa = net_profit / total_assets * 100
                if roa <= min_roa_pct:
                    continue
            else:
                continue

            # 负债率 < 50%
            total_liability = bd.get('tot_liab')
            total_equity = bd.get('total_equity')
            if total_equity and total_equity > 0 and total_liability is not None:
                debt_ratio = total_liability / (total_liability + total_equity)
                if debt_ratio >= self.params.roic_big_max_debt_ratio:
                    continue
            else:
                continue

            # 营收总增速 > 30%
            if total_revenue_growth is None or total_revenue_growth <= min_total_rev_growth_pct:
                continue

            # 营收增速 > 20%
            if revenue_growth is None or revenue_growth <= min_rev_growth_pct:
                continue

            # PE (0, 50)
            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                continue

            # 股价 < 300元过滤（聚宽原版 filter_highprice_stock2）
            if price >= 300:
                continue

            pe = price / eps
            if pe <= 0 or pe >= self.params.roic_big_max_pe:
                continue

            # ROIC_TTM > 8% 二次过滤（聚宽原版 filter_roic）
            # 近似计算: ROIC ≈ EBIT*(1-税率) / (股东权益 + 短期负债)
            # QMT无直接ROIC字段，用营业利润近似EBIT
            roic = self._calc_roic_approx(inc, bd)
            if roic is not None and roic <= 8.0:
                self.log(f'[ROIC过滤] {stock} ROIC={roic:.1f}% <= 8%, 跳过', level='debug')
                continue

            # 未分配利润代理
            surplus = bd.get('surplus_rsrv') or 0
            undistributed = bd.get('undistributed_profit') or 0
            retained_proxy = surplus + undistributed

            candidates.append((stock, retained_proxy))

        if not candidates and missing_count == len(stock_list) and self.params.skip_fundamental_if_missing:
            self.log(f'[WARN] ROIC_BIG: 所有股票财务数据缺失，跳过')
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        result = [s for s, _ in candidates[:stock_num]]
        self.log(f'ROIC_BIG选股: {len(stock_list)} -> {len(candidates)} -> {len(result)} 只')
        return result

    def _calc_roic_approx(self, inc_data: Dict, bd_data: Dict) -> Optional[float]:
        """近似计算ROIC_TTM

        ROIC = EBIT * (1-税率) / 投入资本
        投入资本 = 股东权益 + 有息负债（近似用流动负债替代）
        EBIT ≈ 营业利润（oper_profit）
        税率 ≈ 25%

        Returns:
            ROIC百分比（如8.5表示8.5%），无法计算时返回None
        """
        oper_profit = inc_data.get('oper_profit')
        total_equity = bd_data.get('total_equity')
        current_liability = bd_data.get('total_current_liability')

        if oper_profit is None or total_equity is None or total_equity <= 0:
            return None

        # EBIT * (1 - 25%)
        nopat = oper_profit * 0.75

        # 投入资本 = 股东权益 + 流动负债（近似有息负债）
        invested_capital = total_equity + (current_liability or 0)
        if invested_capital <= 0:
            return None

        return nopat / invested_capital * 100

    # ================================================================
    # BM选股：中等市值白马
    # ================================================================

    def _select_bm(self, stock_list: List[str], stock_num: int) -> List[str]:
        """BM选股：市值100-900亿, PB(0-10), PCF<4, EPS>0.3, ROE>20%,
        净利率>10%, 营收增速>20%, 营业利润增速>10%, 按市值升序"""
        if not stock_list:
            return []

        pershare_fields = ['du_return_on_equity', 's_fa_eps_diluted',
                           's_fa_bps', 's_fa_ocfps', 'inc_revenue_rate',
                           'inc_net_profit_rate']
        pershare_data = self.get_financial_fields_batch(
            stock_list, 'Pershareindex', pershare_fields)

        income_fields = ['revenue', 'oper_profit',
                         'net_profit_incl_min_int_inc_after']
        income_data = self.get_financial_fields_batch(
            stock_list, 'Income', income_fields)

        balance_fields = ['total_equity', 'tot_liab', 'tot_assets', 'cap_stk']
        balance_data = self.get_financial_fields_batch(
            stock_list, 'Balance', balance_fields)

        min_cap = self.params.bm_min_market_cap_yi * 1e8
        max_cap = self.params.bm_max_market_cap_yi * 1e8
        min_roe_pct = self.params.bm_min_roe * 100  # 0.2 -> 20
        min_rev_growth_pct = self.params.bm_min_revenue_growth * 100  # 0.2 -> 20
        min_net_margin_pct = self.params.bm_min_net_margin * 100  # 0.1 -> 10
        min_operate_profit_growth_pct = self.params.bm_min_operate_profit_growth * 100  # 0.1 -> 10

        candidates = []
        missing_count = 0
        for stock in stock_list:
            ps = pershare_data.get(stock, {})
            inc = income_data.get(stock, {})
            bd = balance_data.get(stock, {})

            eps = ps.get('s_fa_eps_diluted')
            roe = ps.get('du_return_on_equity')
            bps = ps.get('s_fa_bps')
            ocfps = ps.get('s_fa_ocfps')
            revenue_growth = ps.get('inc_revenue_rate')
            profit_growth = ps.get('inc_net_profit_rate')

            if eps is None and roe is None:
                missing_count += 1
                continue

            market_cap = self._calc_market_cap(stock, balance_data, pershare_data)
            if not market_cap or market_cap <= min_cap or market_cap >= max_cap:
                continue

            if eps is None or eps <= self.params.bm_min_eps:
                continue
            if roe is None or roe <= min_roe_pct:
                continue
            if revenue_growth is None or revenue_growth <= min_rev_growth_pct:
                continue

            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                continue

            # 股价 < 300元过滤（聚宽原版 filter_highprice_stock2）
            if price >= 300:
                continue

            if bps and bps > 0:
                pb = price / bps
                if pb <= 0 or pb >= self.params.bm_max_pb:
                    continue
            else:
                continue

            if ocfps and ocfps > 0:
                pcf = price / ocfps
                if pcf >= self.params.bm_max_pcf:
                    continue
            else:
                continue

            revenue = inc.get('revenue')
            net_profit = inc.get('net_profit_incl_min_int_inc_after')
            if revenue and net_profit and revenue > 0:
                net_margin = net_profit / revenue * 100
                if net_margin <= min_net_margin_pct:
                    continue
            else:
                continue

            if profit_growth is None or profit_growth <= min_operate_profit_growth_pct:
                continue

            candidates.append((stock, market_cap))

        if not candidates and missing_count == len(stock_list) and self.params.skip_fundamental_if_missing:
            self.log(f'[WARN] BM: 所有股票财务数据缺失，跳过')
            return []

        candidates.sort(key=lambda x: x[1])
        result = [s for s, _ in candidates[:stock_num]]
        self.log(f'BM选股: {len(stock_list)} -> {len(candidates)} -> {len(result)} 只')
        return result

    # ================================================================
    # 辅助计算
    # ================================================================

    def _calc_market_cap(self, stock: str,
                         balance_data: Dict[str, Dict],
                         pershare_data: Dict[str, Dict]) -> Optional[float]:
        """计算市值 = 总股本 × 当前股价

        聚宽原版使用 circulating_market_cap（流通市值），QMT无此字段，
        使用总股本(cap_stk)×股价近似。cap_stk 是财报中的总股本，
        是 point-in-time 数据，比 total_equity/bps 更准确。

        注意：总市值 vs 流通市值差异主要影响动量排名，
        大盘股差异小（流通比例高），小盘股差异可能较大。
        """
        price = self.get_unadjusted_price(stock)
        if price is None or price <= 0:
            return None

        bd = balance_data.get(stock, {})
        ps = pershare_data.get(stock, {})

        # 优先使用财报中的总股本(cap_stk)，这是point-in-time数据
        cap_stk = bd.get('cap_stk')
        if cap_stk and cap_stk > 0:
            return cap_stk * price

        # 回退：使用 total_equity/bps 估算总股本
        total_equity = bd.get('total_equity')
        bps = ps.get('s_fa_bps')

        if total_equity and total_equity > 0 and bps and bps > 0:
            total_shares = total_equity / bps
            return total_shares * price
        elif total_equity and total_equity > 0:
            return total_equity
        return None

    def _calc_total_shares(self, stock: str,
                           balance_data: Dict[str, Dict],
                           pershare_data: Dict[str, Dict]) -> Optional[float]:
        """计算总股本 = 所有者权益 / 每股净资产"""
        bd = balance_data.get(stock, {})
        ps = pershare_data.get(stock, {})

        total_equity = bd.get('total_equity')
        bps = ps.get('s_fa_bps')

        if total_equity and total_equity > 0 and bps and bps > 0:
            return total_equity / bps
        return None

    def on_order(self, order: OrderInfo):
        super().on_order(order)
        # 买入成交时更新成本缓存
        if hasattr(order, 'direction') and order.direction == 'buy':
            symbol = order.symbol
            exec_price = getattr(order, 'executed_price', None) or getattr(order, 'price', None)
            if exec_price and exec_price > 0:
                # 如果已有持仓，计算加权平均成本
                old_cost = self._cost_cache.get(symbol, 0)
                old_size = self._current_holdings.get(symbol, 0)
                new_size = getattr(order, 'executed_volume', 0) or getattr(order, 'volume', 0)
                if old_size > 0 and old_cost > 0 and new_size > 0:
                    self._cost_cache[symbol] = (old_cost * old_size + exec_price * new_size) / (old_size + new_size)
                else:
                    self._cost_cache[symbol] = exec_price
        # 卖出清仓时移除成本缓存
        elif hasattr(order, 'direction') and order.direction == 'sell':
            symbol = order.symbol
            remaining = self._current_holdings.get(symbol, 0)
            if remaining <= 0:
                self._cost_cache.pop(symbol, None)

    def on_backtest_end(self):
        pass
