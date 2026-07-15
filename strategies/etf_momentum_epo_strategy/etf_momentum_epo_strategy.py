import math
import datetime as dt_module
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import solve

from core.strategy_logic import StrategyLogic, BarData, OrderInfo, TradeInfo
from strategies import register_strategy
from .config import ETF_POOL


@register_strategy('etf_momentum_epo',
                   backtest_config={'cash': 100000, 'commission': 0.0002,
                                    'open_commission': 0.0002,
                                    'close_commission': 0.0002,
                                    'close_tax': 0.0,
                                    'min_commission': 5.0,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d'})
class ETFMomentumEPOStrategy(StrategyLogic):
    """多品种ETF动量轮动+EPO优化策略 - 基于年化收益和R²打分的动量因子轮动，EPO+风险平价混合权重

    原始策略来源: 聚宽社区 openhe https://www.joinquant.com/post/47208

    策略逻辑：
    1. 每月首个交易日调仓
    2. 动量打分：对ETF池中每只ETF，用近N日收盘价做对数线性回归，
       年化收益率 × 判定系数R² 作为动量分数
    3. 过滤掉动量分数<=0的ETF，取前M只
    4. 权重计算：60% EPO优化权重 + 40% 风险平价权重
       - EPO：用anchored方法计算持仓权重（锚定为逆方差权重）
       - 风险平价：逆波动率等风险贡献权重
       - 混合权重取两者线性加权，兼顾信号驱动与风险分散
    5. 按权重分配资金买入

    调仓规则：
    - 月度调仓，每月首个交易日
    - 卖出不在目标列表中的持仓
    - 买入目标ETF，按混合权重分配资金
    """

    params = (
        ('momentum_days', 34),       # 动量参考天数
        ('top_n', 3),                # 持有动量排名前N只ETF
        ('epo_lambda', 10),          # EPO风险厌恶系数，越高越保守
        ('epo_w', 0.2),              # EPO模糊化参数/收缩权重，越高越分散
        ('epo_lookback', 1200),      # EPO优化回看天数（约5年）
        # --- v2优化参数（默认禁用）---
        ('multi_period_weights', None),  # v2-A: 多周期融合权重, 如 "0.3,0.4,0.3" 对应 (20d,34d,60d)
        ('momentum_accel_bonus', 0.0),   # v2-B: 动量加速度奖励系数, score *= (1 + bonus * accel)
        ('volume_confirm', False),        # v2-C: 成交量确认, score *= volume_ratio
        ('market_breadth_sizing', False), # v2-D: 市场宽度仓位, 按正动量ETF占比缩放总仓位
        ('trend_strength_ref', None),     # v2-E: 趋势强度仓位参考值, exposure=min(1, avg_score/ref)
        ('correlation_threshold', None),  # v2-F: 低相关性优选阈值, 高于此相关的ETF替换
        ('use_risk_parity', False),       # v2-G: 使用风险平价替代EPO
        ('risk_parity_blend', 0.4),       # v2-J: EPO与风险平价混合比例, 0=纯EPO, 1=纯RP
        ('skip_stable_rank', False),      # v2-H: top_n集合不变时跳过调仓
        ('quarterly_rebalance', False),   # v2-I: 季频调仓(1/4/7/10月)
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)

        # ETF数据 {symbol: name}
        self.etf_data = {symbol: name for name, symbol in ETF_POOL.items()}

        # 当前持仓 {symbol: volume}
        self.current_holdings: Dict[str, int] = {}

        # 上次调仓月份 (year, month)
        self._last_rebalance_month: Optional[Tuple[int, int]] = None

        # v2-H: 上次调仓的top_n集合
        self._last_top_n_set: Optional[set] = None

        # ETF设为T+0
        for symbol in self.etf_data:
            self.set_t_plus_1(symbol, False)

    def get_state(self) -> dict:
        state = super().get_state()
        state['last_rebalance_month'] = (
            list(self._last_rebalance_month)
            if self._last_rebalance_month else None
        )
        state['last_top_n_set'] = (
            list(self._last_top_n_set)
            if self._last_top_n_set else None
        )
        return state

    def set_state(self, state: dict):
        super().set_state(state)
        month = state.get('last_rebalance_month')
        self._last_rebalance_month = tuple(month) if month else None
        top_n = state.get('last_top_n_set')
        self._last_top_n_set = set(top_n) if top_n else None

    def get_symbols(self):
        return list(self.etf_data.keys())

    def on_bar(self, bar: BarData):
        current_date = self.get_current_date()
        if current_date is None:
            return

        # 分钟级过滤：只处理日线或指定时间
        bar_datetime = getattr(bar, 'datetime', None)
        if bar_datetime and isinstance(bar_datetime, dt_module.datetime):
            hour = bar_datetime.hour
            minute = bar_datetime.minute
            if hour != 0 or minute != 0:
                if hour != 9 or minute != 30:
                    return

        # 判断是否为调仓日
        if not self._is_rebalance_day(current_date):
            return

        self._execute_rebalance(current_date)

    def _is_rebalance_day(self, current_date) -> bool:
        """判断是否为调仓日（支持月度/季频）"""
        year, month = current_date.year, current_date.month

        # v2-I: 季频调仓
        if self.params.quarterly_rebalance:
            if month not in (1, 4, 7, 10):
                return False

        # 同月不重复调仓
        if self._last_rebalance_month == (year, month):
            return False

        day = current_date.day
        if day <= 5:
            for symbol in list(self.etf_data.keys())[:1]:
                closes = self.get_close_prices(symbol)
                if len(closes) >= 2:
                    self._last_rebalance_month = (year, month)
                    return True
            self._last_rebalance_month = (year, month)
            return True

        return False

    def _execute_rebalance(self, current_date):
        """执行调仓"""
        current_date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)

        # 1. 动量排名
        ranked_list = self._get_momentum_rank()
        if not ranked_list:
            self.log(f'[{current_date_str}] 无正动量ETF，清仓')
            self._sell_all(current_date_str)
            self._last_top_n_set = None
            return

        # v2-F: 低相关性优选
        if self.params.correlation_threshold is not None:
            ranked_list = self._select_uncorrelated(ranked_list)

        # 取前N只
        target_list = ranked_list[:self.params.top_n]
        target_symbols = [s for s, _ in target_list]

        # v2-H: 排名稳定跳过
        if self.params.skip_stable_rank:
            current_set = set(target_symbols)
            if self._last_top_n_set == current_set:
                return
            self._last_top_n_set = current_set

        self.log(f'[{current_date_str}] 动量排名: {[(self.etf_data.get(s, s), f"{score:.4f}") for s, score in target_list]}')

        # 2. 卖出不在目标列表中的持仓
        for symbol in list(self.current_holdings.keys()):
            if symbol not in target_symbols:
                self._sell_position(symbol, current_date_str)

        # 3. 计算权重
        if self.params.use_risk_parity:
            weights = self._risk_parity_weights(target_symbols)
        elif self.params.risk_parity_blend > 0:
            # v2-J: EPO与风险平价混合
            epo_w = self._run_optimization(target_symbols)
            rp_w = self._risk_parity_weights(target_symbols)
            if epo_w is not None and rp_w is not None:
                blend = self.params.risk_parity_blend
                weights = (1 - blend) * epo_w + blend * rp_w
                total = np.sum(weights)
                if total > 0:
                    weights = weights / total
            elif epo_w is not None:
                weights = epo_w
            elif rp_w is not None:
                weights = rp_w
            else:
                weights = None
        else:
            weights = self._run_optimization(target_symbols)

        # v2-D/E: 仓位缩放
        exposure = self._get_exposure(ranked_list)
        if exposure < 1.0 and weights is not None:
            weights = weights * exposure
            self.log(f'[{current_date_str}] 仓位缩放: exposure={exposure:.2%}')

        # 4. 按权重买入
        if weights is not None and len(weights) > 0:
            self._buy_with_weights(target_symbols, weights, current_date_str)

        self._log_portfolio_status(current_date_str)

    def _calc_single_momentum(self, symbol: str, period: int) -> Optional[Tuple[float, float, float]]:
        """计算单个周期动量指标

        Returns:
            (annualized_return, r_squared, score) 或 None
        """
        closes = self.get_close_prices(symbol)
        if len(closes) < period + 1:
            return None

        recent_closes = closes[-(period + 1):]

        try:
            y = np.log(np.array(recent_closes, dtype=float))
            x = np.arange(len(y), dtype=float)

            slope, intercept = np.polyfit(x, y, 1)
            annualized_returns = math.pow(math.exp(slope), 250) - 1

            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.var(y, ddof=1) * (len(y) - 1)
            if ss_tot == 0:
                return None
            r_squared = 1 - ss_res / ss_tot

            score = annualized_returns * r_squared
            return (annualized_returns, r_squared, score)
        except (ValueError, RuntimeWarning, FloatingPointError):
            return None

    def _get_momentum_rank(self) -> List[Tuple[str, float]]:
        """基于年化收益和判定系数打分的动量因子排名

        支持多周期融合(v2-A)、动量加速度(v2-B)、成交量确认(v2-C)
        """
        m_days = self.params.momentum_days
        mp_weights = self.params.multi_period_weights
        # 支持字符串格式的多周期权重, 如 "0.3|0.4|0.3" 或 "0.3,0.4,0.3"
        if isinstance(mp_weights, str):
            try:
                sep = '|' if '|' in mp_weights else ','
                mp_weights = tuple(float(x.strip()) for x in mp_weights.split(sep))
            except (ValueError, AttributeError):
                mp_weights = None
        elif isinstance(mp_weights, (int, float)):
            # 如果被错误解析为单个数字，忽略
            mp_weights = None
        score_list = []

        for symbol in self.etf_data:
            if mp_weights is not None:
                # v2-A: 多周期融合动量
                periods = [20, m_days, 60]
                scores = []
                for p in periods:
                    result = self._calc_single_momentum(symbol, p)
                    if result is None:
                        break
                    scores.append(result[2])

                if len(scores) != len(periods):
                    # 数据不足，回退到单周期
                    result = self._calc_single_momentum(symbol, m_days)
                    if result is None:
                        continue
                    score = result[2]
                else:
                    score = sum(w * s for w, s in zip(mp_weights, scores))
            else:
                result = self._calc_single_momentum(symbol, m_days)
                if result is None:
                    continue
                score = result[2]

            # v2-B: 动量加速度
            if self.params.momentum_accel_bonus > 0:
                closes = self.get_close_prices(symbol)
                if len(closes) >= m_days * 2 + 2:
                    # 前半段斜率
                    first_half = closes[-(m_days * 2 + 2):-(m_days // 2)]
                    second_half = closes[-(m_days + 1):]

                    if len(first_half) >= 10 and len(second_half) >= 10:
                        try:
                            y1 = np.log(np.array(first_half, dtype=float))
                            x1 = np.arange(len(y1), dtype=float)
                            slope1, _ = np.polyfit(x1, y1, 1)

                            y2 = np.log(np.array(second_half, dtype=float))
                            x2 = np.arange(len(y2), dtype=float)
                            slope2, _ = np.polyfit(x2, y2, 1)

                            # 加速度 = 近期斜率变化率
                            if abs(slope1) > 1e-10:
                                accel = (slope2 - slope1) / abs(slope1)
                                score *= (1 + self.params.momentum_accel_bonus * accel)
                        except (ValueError, RuntimeWarning):
                            pass

            # v2-C: 成交量确认
            if self.params.volume_confirm:
                ohlcv = self.get_ohlcv_data(symbol, period=20)
                if ohlcv and len(ohlcv) >= 10:
                    volumes = [bar.get('volume', 0) for bar in ohlcv[-20:]]
                    volumes = [v for v in volumes if v > 0]
                    if len(volumes) >= 5:
                        recent_vol = sum(volumes[-5:]) / 5
                        avg_vol = sum(volumes) / len(volumes)
                        if avg_vol > 0:
                            vol_ratio = recent_vol / avg_vol
                            score *= vol_ratio

            score_list.append((symbol, score))

        # 按分数降序排列
        score_list.sort(key=lambda x: x[1], reverse=True)

        # 过滤掉分数<=0的ETF
        filtered = [(s, sc) for s, sc in score_list if sc > 0]
        return filtered

    def _select_uncorrelated(self, ranked_list: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """v2-F: 低相关性优选 - 从排名中选出不高度相关的ETF"""
        threshold = self.params.correlation_threshold
        top_n = self.params.top_n

        # 取足够多的候选
        candidates = ranked_list[:top_n * 3]
        if len(candidates) <= top_n:
            return ranked_list

        # 计算候选ETF间的相关系数
        returns_dict = {}
        for symbol, _ in candidates:
            closes = self.get_close_prices(symbol)
            if len(closes) < 60:
                continue
            closes_arr = np.array(closes[-60:], dtype=float)
            rets = np.diff(closes_arr) / closes_arr[:-1]
            returns_dict[symbol] = rets

        # 贪心选择：依次选入与已选不高度相关的
        selected = []
        selected_symbols = set()

        for symbol, score in candidates:
            if len(selected) >= top_n * 2:  # 扩展到2倍候选
                break
            if symbol not in returns_dict:
                continue

            is_correlated = False
            for sel_symbol in selected_symbols:
                if sel_symbol not in returns_dict:
                    continue
                min_len = min(len(returns_dict[symbol]), len(returns_dict[sel_symbol]))
                if min_len < 20:
                    continue
                corr = np.corrcoef(returns_dict[symbol][-min_len:],
                                   returns_dict[sel_symbol][-min_len:])[0, 1]
                if abs(corr) > threshold:
                    is_correlated = True
                    break

            if not is_correlated or len(selected) < top_n:
                selected.append((symbol, score))
                selected_symbols.add(symbol)

        return selected if selected else ranked_list

    def _get_exposure(self, ranked_list: List[Tuple[str, float]]) -> float:
        """v2-D/E: 计算仓位暴露度"""
        exposure = 1.0

        # v2-D: 市场宽度仓位
        if self.params.market_breadth_sizing:
            total_etfs = len(self.etf_data)
            positive_count = len([s for s, sc in ranked_list if sc > 0])
            breadth = positive_count / total_etfs if total_etfs > 0 else 0
            # 宽度>=50%满仓，<20%空仓，中间线性
            if breadth < 0.2:
                exposure = breadth / 0.2 * 0.5
            elif breadth < 0.5:
                exposure = 0.5 + (breadth - 0.2) / 0.3 * 0.5
            else:
                exposure = 1.0

        # v2-E: 趋势强度仓位
        if self.params.trend_strength_ref is not None:
            ref = self.params.trend_strength_ref
            top_scores = [sc for _, sc in ranked_list[:self.params.top_n]]
            if top_scores:
                avg_score = sum(top_scores) / len(top_scores)
                strength_exposure = min(1.0, avg_score / ref)
                exposure = min(exposure, strength_exposure)

        return exposure

    def _risk_parity_weights(self, target_symbols: List[str]) -> Optional[np.ndarray]:
        """v2-G: 风险平价权重 - 每个持仓贡献等额风险"""
        if not target_symbols:
            return None

        vols = []
        for symbol in target_symbols:
            closes = self.get_close_prices(symbol)
            if len(closes) < 20:
                return None
            closes_arr = np.array(closes[-60:] if len(closes) >= 60 else closes, dtype=float)
            rets = np.diff(closes_arr) / closes_arr[:-1]
            vol = np.std(rets)
            vols.append(vol if vol > 0 else 1e-10)

        # 逆波动率权重
        inv_vols = [1.0 / v for v in vols]
        total = sum(inv_vols)
        weights = np.array([iv / total for iv in inv_vols])
        return weights

    def _run_optimization(self, target_symbols: List[str]) -> Optional[np.ndarray]:
        """EPO优化计算持仓权重

        使用anchored方法，锚定为逆方差权重。
        收缩相关矩阵 → 重建协方差矩阵 → anchored EPO

        Returns:
            归一化后的权重数组（非负，和为1），失败返回None
        """
        if not target_symbols:
            return None

        lookback = self.params.epo_lookback
        lambda_ = self.params.epo_lambda
        w = self.params.epo_w

        # 获取收益率数据
        returns_dict = {}
        min_len = float('inf')
        for symbol in target_symbols:
            closes = self.get_close_prices(symbol)
            if len(closes) < 2:
                return None
            # 限制回看期
            if len(closes) > lookback + 1:
                closes = closes[-(lookback + 1):]
            closes_arr = np.array(closes, dtype=float)
            rets = np.diff(closes_arr) / closes_arr[:-1]
            returns_dict[symbol] = rets
            min_len = min(min_len, len(rets))

        if min_len < 20:
            self.log(f'EPO数据不足: 最短收益率序列长度={min_len}')
            return None

        # 对齐长度
        aligned_returns = {}
        for symbol in target_symbols:
            rets = returns_dict[symbol]
            aligned_returns[symbol] = rets[-min_len:]

        # 构建收益率矩阵
        n = len(target_symbols)
        returns_matrix = np.column_stack([aligned_returns[s] for s in target_symbols])

        try:
            # 计算协方差和相关矩阵
            vcov = np.cov(returns_matrix, rowvar=False)
            corr = np.corrcoef(returns_matrix, rowvar=False)

            # 对角标准差矩阵
            diag_var = np.diag(np.diag(vcov))
            std = np.sqrt(diag_var)

            # 收缩相关矩阵: shrunk_cor = (1-w) * corr + w * I
            I = np.eye(n)
            shrunk_cor = (1 - w) * corr + w * I

            # 重建协方差矩阵
            cov_tilde = std @ shrunk_cor @ std

            # 求逆
            inv_shrunk_cov = solve(cov_tilde, I)

            # 信号向量 = 均值收益
            signal = np.array([np.mean(aligned_returns[s]) for s in target_symbols])

            # 锚定权重 = 逆方差权重
            d = np.diag(vcov)
            if np.any(d <= 0):
                # 方差非正，回退到等权
                anchor = np.ones(n) / n
            else:
                inv_d = 1.0 / d
                anchor = inv_d / np.sum(inv_d)

            # Anchored EPO (endogenous)
            # gamma = sqrt(a' @ cov_tilde @ a) / sqrt(s' @ inv_shrunk_cov @ cov_tilde @ inv_shrunk_cov @ s)
            num = np.sqrt(anchor.T @ cov_tilde @ anchor)
            den = np.sqrt(signal.T @ inv_shrunk_cov @ cov_tilde @ inv_shrunk_cov @ signal)
            if den == 0:
                gamma = 1.0
            else:
                gamma = num / den

            # epo = inv_shrunk_cov @ ((1-w)*gamma*s + w*I @ V @ a)
            V = diag_var
            epo = inv_shrunk_cov @ ((1 - w) * gamma * signal + w * I @ V @ anchor)

            # 归一化：负权重置零
            epo = np.array([0 if a < 0 else a for a in epo])
            total = np.sum(epo)
            if total > 0:
                epo = epo / total
            else:
                # 所有权重非正，回退等权
                epo = np.ones(n) / n

            return epo

        except (np.linalg.LinAlgError, ValueError) as e:
            self.log(f'EPO优化失败: {e}，回退等权')
            return np.ones(n) / n

    def _sell_all(self, current_date_str: str):
        """卖出所有持仓"""
        for symbol in list(self.current_holdings.keys()):
            self._sell_position(symbol, current_date_str)

    def _get_trade_price(self, symbol: str) -> Optional[float]:
        """获取交易价格 - 使用开盘价（与聚宽run_monthly 9:30开盘交易一致）"""
        price = self.get_open_price(symbol)
        if price is not None and price > 0:
            return price
        # 开盘价不可用时回退到收盘价
        return self.get_current_price(symbol)

    def _sell_position(self, symbol: str, current_date_str: str):
        """卖出指定标的全部持仓"""
        pos_size = self.get_position_size(symbol)
        if pos_size <= 0:
            self.current_holdings.pop(symbol, None)
            return

        price = self._get_trade_price(symbol)
        if price is None or price <= 0:
            return

        sellable = self.get_sellable_volume(symbol)
        if sellable <= 0:
            return

        self.sell(symbol, price, sellable)
        self.log(f'[{current_date_str}] 卖出: {self.etf_data.get(symbol, symbol)} ({symbol}), '
                 f'价格: {price:.3f}, 数量: {sellable}')
        self.current_holdings.pop(symbol, None)

    def _buy_with_weights(self, target_symbols: List[str], weights: np.ndarray,
                          current_date_str: str):
        """按权重调整持仓 - 模拟聚宽order_target_value逻辑

        先卖出超权重部分，再买入不足部分，避免负现金。
        weights之和可能<1(仓位缩放时)，剩余部分保留为现金。
        """
        # 计算总资产
        total_value = self._get_total_value()
        if total_value <= 0:
            return

        # 第一轮：卖出超权重的部分（释放资金）
        for i, symbol in enumerate(target_symbols):
            target_value = total_value * weights[i]

            pos_size = self.get_position_size(symbol)
            if pos_size <= 0:
                continue

            price = self._get_trade_price(symbol)
            if price is None or price <= 0:
                continue

            current_pos_value = pos_size * price

            # 需要减仓的金额
            if current_pos_value > target_value + 100:
                sell_value = current_pos_value - target_value
                sell_size = int(sell_value / price / 100) * 100
                if sell_size > 0:
                    sellable = self.get_sellable_volume(symbol)
                    sell_size = min(sell_size, sellable)
                    if sell_size > 0:
                        self.sell(symbol, price, sell_size)
                        self.log(f'[{current_date_str}] 减仓: {self.etf_data.get(symbol, symbol)} ({symbol}), '
                                 f'价格: {price:.3f}, 数量: {sell_size}, 目标权重: {weights[i]:.2%}')

        # 第二轮：买入不足的部分
        for i, symbol in enumerate(target_symbols):
            target_value = total_value * weights[i]

            # 重新获取持仓信息（第一轮卖出后可能已变）
            pos_size = self.get_position_size(symbol)
            current_pos_value = 0
            if pos_size > 0:
                price = self._get_trade_price(symbol)
                if price and price > 0:
                    current_pos_value = pos_size * price

            # 需要加仓的金额
            diff_value = target_value - current_pos_value

            if diff_value <= 100:
                continue

            price = self._get_trade_price(symbol)
            if price is None or price <= 0:
                continue

            # 计算买入数量（ETF最小交易单位100股）
            buy_size = int(diff_value / price / 100) * 100
            if buy_size <= 0:
                continue

            # 检查资金是否足够（使用实时现金）
            cash = self.get_cash()
            buy_cost = price * buy_size
            if buy_cost > cash:
                buy_size = int(cash / price / 100) * 100
                if buy_size <= 0:
                    continue

            self.buy(symbol, price, buy_size)
            self.current_holdings[symbol] = self.current_holdings.get(symbol, 0) + buy_size
            self.log(f'[{current_date_str}] 买入: {self.etf_data.get(symbol, symbol)} ({symbol}), '
                     f'价格: {price:.3f}, 数量: {buy_size}, 目标权重: {weights[i]:.2%}')

    def _get_total_value(self) -> float:
        """计算总资产（现金+持仓市值）"""
        cash = self.get_cash()
        position_value = 0
        for symbol in self.etf_data:
            pos_size = self.get_position_size(symbol)
            if pos_size > 0:
                price = self.get_current_price(symbol)
                if price and price > 0:
                    position_value += pos_size * price
        return cash + position_value

    def _log_portfolio_status(self, current_date_str: str):
        """记录持仓状态"""
        cash = self.get_cash()
        total_position_value = 0
        holdings_info = []

        for symbol in self.etf_data:
            pos_size = self.get_position_size(symbol)
            if pos_size > 0:
                price = self.get_current_price(symbol)
                if price and price > 0:
                    pos_value = pos_size * price
                    total_position_value += pos_value
                    name = self.etf_data.get(symbol, symbol)
                    holdings_info.append(f'{name}:{pos_size}股@{price:.3f}(市值{pos_value:.0f})')

        total_value = cash + total_position_value
        holdings_str = ', '.join(holdings_info) if holdings_info else '空仓'
        self.log(f'[{current_date_str}] 资产: 现金{cash:.0f} + 持仓{total_position_value:.0f} = {total_value:.0f} | {holdings_str}')

    def on_order(self, order: OrderInfo):
        super().on_order(order)

    def on_trade(self, trade: TradeInfo):
        super().on_trade(trade)
