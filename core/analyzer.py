import pandas as pd
import numpy as np
import datetime as dt_module
from typing import Dict, List, Optional, Any

try:
    import backtrader as bt
except ImportError:
    bt = None

from core.models import (
    BacktestingResult,
    AccountInfo,
    BacktestConfig,
    TradeRecord,
    InstrumentData,
)


class PerformanceAnalyzer:
    def __init__(self):
        self._cerebro = None
        self._strategy = None
        self._initial_cash: float = 0.0

    def set_context(self, cerebro, strategy, initial_cash: float = 0.0):
        self._cerebro = cerebro
        self._strategy = strategy
        self._initial_cash = initial_cash

    def build_result(
        self,
        strategy_params: Optional[Dict[str, Any]] = None,
        show_kline: bool = True,
        trade_start_date: Optional[str] = None,
    ) -> BacktestingResult:
        if self._strategy is None:
            return BacktestingResult()

        account = self._build_account()
        df = self._build_equity_dataframe()
        trade_log = self._build_trade_log()
        klines = self._build_klines()
        instruments_data = self._build_instruments_data()
        instrument_close_prices = self._build_instrument_close_prices()

        result = BacktestingResult(
            account=account,
            config=BacktestConfig(show_kline=show_kline),
            strategy_params=strategy_params or {},
            klines=klines,
            trade_log=trade_log,
            df=df,
            instruments_data=instruments_data,
            trade_start_date=trade_start_date,
            instrument_close_prices=instrument_close_prices,
        )

        result.turnover = self._calc_turnover(trade_log)
        result.total_volume = sum(abs(t.volume) for t in trade_log if t.order_id != -1)

        return result

    def _build_account(self) -> AccountInfo:
        strat = self._strategy
        initial = self._initial_cash if self._initial_cash > 0 else strat.broker.startingcash
        final_value = strat.broker.getvalue()
        total_profit = final_value - initial
        rate = total_profit / initial if initial > 0 else 0.0
        fee = self._calc_total_fee(strat)
        return AccountInfo(
            initial_capital=initial,
            dynamic_rights=final_value,
            total_profit=total_profit,
            rate=rate,
            fee=fee,
        )

    def _build_equity_dataframe(self) -> Optional[pd.DataFrame]:
        strat = self._strategy
        equity_history = strat.get_equity_history() if hasattr(strat, 'get_equity_history') else getattr(strat, '_equity_history', None)
        trade_start_date = getattr(strat.params, 'trade_start_date', None)
        return self._build_equity_dataframe_impl(equity_history, trade_start_date)

    def _build_equity_dataframe_impl(self, equity_history, trade_start_date=None) -> Optional[pd.DataFrame]:
        if not equity_history or len(equity_history) == 0:
            return None

        filtered_history = []
        for dt, value in equity_history:
            if trade_start_date and hasattr(dt, 'isoformat'):
                if dt.isoformat() < trade_start_date:
                    continue
            filtered_history.append((dt, value))

        if not filtered_history:
            return None

        dates = []
        portfolio_values = []
        daily_pnls = []

        for i, (dt, value) in enumerate(filtered_history):
            dates.append(pd.Timestamp(dt))
            portfolio_values.append(value)
            if i == 0:
                daily_pnls.append(0.0)
            else:
                daily_pnls.append(value - filtered_history[i - 1][1])

        df = pd.DataFrame({
            "datetime": dates,
            "PortfolioValue": portfolio_values,
            "PnL": daily_pnls,
        })
        return df

    def _build_trade_log(self) -> List[TradeRecord]:
        strat = self._strategy
        logic = getattr(strat, '_strategy_logic', None)
        logic_orders = logic.get_orders() if logic and hasattr(logic, 'get_orders') else getattr(logic, '_orders', None)
        trade_records = strat.get_trade_records() if hasattr(strat, 'get_trade_records') else getattr(strat, '_trade_records', None) or []
        return self._build_trade_log_impl(logic_orders, trade_records)

    def _build_trade_log_impl(self, logic_orders, trade_records) -> List[TradeRecord]:
        trades: List[TradeRecord] = []

        if not logic_orders:
            return trades

        completed_orders = [
            o for o in logic_orders.values()
            if o.is_completed and (o.executed_volume > 0 or o.executed_price > 0)
        ]

        used_trade_indices = set()

        def _find_matching_trade(symbol: str, direction: str, fallback_idx: int):
            for i, tr in enumerate(trade_records):
                if i in used_trade_indices:
                    continue
                tr_symbol = tr.get("symbol", "")
                tr_direction = str(tr.get("direction", ""))
                if tr_symbol == symbol:
                    is_buy_trade = tr_direction in ("1", "买", "buy")
                    is_sell_trade = tr_direction in ("2", "卖", "sell", "-1")
                    order_is_buy = direction == "0"
                    if (order_is_buy and is_buy_trade) or (not order_is_buy and is_sell_trade):
                        used_trade_indices.add(i)
                        return tr
            if fallback_idx < len(trade_records) and fallback_idx not in used_trade_indices:
                used_trade_indices.add(fallback_idx)
                return trade_records[fallback_idx]
            return None

        positions = {}
        trade_rec_idx = 0

        for order_info in completed_orders:
            direction = "0" if order_info.is_buy else "1"
            exec_price = order_info.executed_price if order_info.executed_price > 0 else order_info.price
            exec_vol = order_info.executed_volume if order_info.executed_volume > 0 else order_info.volume

            current_pos = positions.get(order_info.symbol, 0)

            if order_info.is_buy:
                if current_pos < 0:
                    close_vol = min(exec_vol, abs(current_pos))
                    offset = "1" if close_vol > 0 else "0"
                else:
                    offset = "0"
                new_pos = current_pos + exec_vol
            else:
                if current_pos > 0:
                    close_vol = min(exec_vol, current_pos)
                    offset = "1" if close_vol > 0 else "0"
                else:
                    offset = "0"
                new_pos = current_pos - exec_vol

            positions[order_info.symbol] = new_pos

            memo = ""
            is_buy = direction == "0"
            if is_buy:
                if offset == "0":
                    memo = "建仓" if current_pos <= 0 else "加仓"
                else:
                    memo = "平仓"
            else:
                if offset == "1":
                    memo = "清仓" if new_pos == 0 else "减仓"
                else:
                    memo = "空头建仓"

            trade_datetime = getattr(order_info, 'datetime', None)
            pnl_val = 0.0
            fee_val = getattr(order_info, 'commission', 0.0)

            if trade_datetime is None:
                tr = _find_matching_trade(order_info.symbol, direction, trade_rec_idx)
                if tr is not None:
                    trade_datetime = tr.get("datetime")
                    pnl_val = tr.get("pnl_no_commission", 0.0)
                    fee_val = tr.get("commission", fee_val)
                trade_rec_idx += 1

            trade = TradeRecord(
                order_id=len(trades) + 1,
                trade_time=trade_datetime,
                instrument_id=order_info.symbol,
                direction=direction,
                offset=offset,
                volume=exec_vol,
                order_price=order_info.price,
                trade_price=exec_price,
                fee=fee_val,
                pnl=pnl_val,
                memo=memo,
            )
            trades.append(trade)

        return trades

    def _build_klines(self) -> List[Dict]:
        strat = self._strategy
        if not hasattr(strat, "datas") or not strat.datas:
            return []

        data = strat.datas[0]
        klines = []

        try:
            num_bars = len(data)
        except Exception:
            return []

        for i in range(num_bars):
            try:
                idx = -num_bars + 1 + i
                bar_dt = data.datetime.datetime(idx)
                if isinstance(bar_dt, dt_module.datetime) and bar_dt.hour == 0 and bar_dt.minute == 0:
                    bar_dt = bar_dt.replace(hour=15, minute=0)
                kline = {
                    "datetime": bar_dt,
                    "open": float(data.open[idx]),
                    "high": float(data.high[idx]),
                    "low": float(data.low[idx]),
                    "close": float(data.close[idx]),
                    "volume": float(data.volume[idx]),
                }
                klines.append(kline)
            except Exception:
                continue

        return klines

    def _build_instruments_data(self) -> Dict[str, InstrumentData]:
        strat = self._strategy
        symbols = []
        if hasattr(strat, "datas") and strat.datas:
            for data_feed in strat.datas:
                symbol = getattr(data_feed, "_name", "")
                if symbol:
                    symbols.append(symbol)
        return self._build_instruments_data_impl(symbols)

    def _build_instruments_data_impl(self, symbols) -> Dict[str, InstrumentData]:
        instruments_data: Dict[str, InstrumentData] = {}
        for symbol in symbols:
            instruments_data[symbol] = InstrumentData(volume_multiple=1.0)
        return instruments_data

    def _build_instrument_close_prices(self) -> Dict[str, Dict[str, float]]:
        strat = self._strategy
        if not hasattr(strat, "datas") or not strat.datas:
            return {}

        symbol_bars = {}
        for data_feed in strat.datas:
            symbol = getattr(data_feed, "_name", "")
            if not symbol:
                continue
            try:
                num_bars = len(data_feed)
            except Exception:
                continue

            bars = []
            for i in range(num_bars):
                try:
                    idx = -num_bars + 1 + i
                    bar_dt = data_feed.datetime.datetime(idx)
                    bars.append((bar_dt, float(data_feed.close[idx])))
                except Exception:
                    continue
            if bars:
                symbol_bars[symbol] = bars

        return self._build_instrument_close_prices_impl(symbol_bars)

    def _build_instrument_close_prices_impl(self, symbol_bars: Dict[str, List[tuple]]) -> Dict[str, Dict[str, float]]:
        result = {}
        for symbol, bars in symbol_bars.items():
            close_prices = {}
            for bar_dt, close_price in bars:
                date_str = bar_dt.strftime("%Y-%m-%d")
                close_prices[date_str] = float(close_price)
            if close_prices:
                result[symbol] = close_prices
        return result

    def _calc_total_fee(self, strat) -> float:
        total_fee = 0.0
        logic = getattr(strat, '_strategy_logic', None)
        if logic:
            orders = logic.get_orders() if hasattr(logic, 'get_orders') else getattr(logic, '_orders', {})
            for order_info in orders.values():
                total_fee += getattr(order_info, 'commission', 0.0)
        else:
            trade_records = strat.get_trade_records() if hasattr(strat, 'get_trade_records') else getattr(strat, '_trade_records', None)
            if trade_records:
                for tr in trade_records:
                    total_fee += tr.get("commission", 0.0)
        return total_fee

    def _calc_turnover(self, trade_log: List[TradeRecord]) -> float:
        turnover = 0.0
        for t in trade_log:
            if t.order_id == -1:
                continue
            price = t.trade_price if t.trade_price > 0 else t.order_price
            if price > 0:
                turnover += price * t.volume
        return turnover

    def build_result_from_engine(
        self,
        engine_result,
        strategy_params: Optional[Dict[str, Any]] = None,
        show_kline: bool = True,
        trade_start_date: Optional[str] = None,
        build_close_prices: bool = True,
    ) -> BacktestingResult:
        from engine.result import EngineResult

        account = self._build_account_from_engine(engine_result)
        df = self._build_equity_dataframe_from_engine(engine_result, trade_start_date)
        trade_log = self._build_trade_log_from_engine(engine_result)
        klines = self._build_klines_from_engine(engine_result)
        instruments_data = self._build_instruments_data_from_engine(engine_result)
        instrument_close_prices = (
            self._build_instrument_close_prices_from_engine(engine_result)
            if build_close_prices else {}
        )

        result = BacktestingResult(
            account=account,
            config=BacktestConfig(show_kline=show_kline),
            strategy_params=strategy_params or {},
            klines=klines,
            trade_log=trade_log,
            df=df,
            instruments_data=instruments_data,
            trade_start_date=trade_start_date,
            instrument_close_prices=instrument_close_prices,
        )

        result.turnover = self._calc_turnover(trade_log)
        result.total_volume = sum(abs(t.volume) for t in trade_log if t.order_id != -1)

        return result

    def _build_account_from_engine(self, engine_result) -> AccountInfo:
        initial = engine_result.initial_cash
        final_value = engine_result.final_value
        total_profit = final_value - initial
        rate = total_profit / initial if initial > 0 else 0.0
        fee = sum(getattr(o, 'commission', 0.0) for o in engine_result.orders)
        return AccountInfo(
            initial_capital=initial,
            dynamic_rights=final_value,
            total_profit=total_profit,
            rate=rate,
            fee=fee,
        )

    def _build_equity_dataframe_from_engine(self, engine_result, trade_start_date=None) -> Optional[pd.DataFrame]:
        equity_history = engine_result.equity_history
        return self._build_equity_dataframe_impl(equity_history, trade_start_date)

    def _build_trade_log_from_engine(self, engine_result) -> List[TradeRecord]:
        logic = engine_result.strategy_logic
        logic_orders = logic.get_orders() if logic and hasattr(logic, 'get_orders') else {}
        trade_records = engine_result.trade_records or []
        return self._build_trade_log_impl(logic_orders, trade_records)

    def _build_klines_from_engine(self, engine_result) -> List[Dict]:
        data_feeds = engine_result.data_feeds
        if not data_feeds:
            return []

        first_symbol = next(iter(data_feeds), None)
        if not first_symbol:
            return []

        feed = data_feeds[first_symbol]
        klines = []
        for i in range(feed.length):
            try:
                bar_dt = feed.get_datetime(i)
                if bar_dt is None:
                    continue
                close = feed.get_close(i)
                if close != close:
                    continue
                kline = {
                    "datetime": bar_dt,
                    "open": float(feed.get_open(i)),
                    "high": float(feed.get_high(i)),
                    "low": float(feed.get_low(i)),
                    "close": float(close),
                    "volume": float(feed.get_volume(i)),
                }
                klines.append(kline)
            except Exception:
                continue

        return klines

    def _build_instruments_data_from_engine(self, engine_result) -> Dict[str, InstrumentData]:
        return self._build_instruments_data_impl(list(engine_result.data_feeds.keys()))

    def _build_instrument_close_prices_from_engine(self, engine_result) -> Dict[str, Dict[str, float]]:
        symbol_bars = {}
        for symbol, feed in engine_result.data_feeds.items():
            bars = []
            for i in range(feed.length):
                try:
                    bar_dt = feed.get_datetime(i)
                    if bar_dt is None:
                        continue
                    close = feed.get_close(i)
                    if close != close:
                        continue
                    bars.append((bar_dt, float(close)))
                except Exception:
                    continue
            if bars:
                symbol_bars[symbol] = bars
        return self._build_instrument_close_prices_impl(symbol_bars)

    def calculate_metrics(self, cerebro) -> Dict[str, float]:
        """计算回测指标

        .. deprecated::
            此方法会重新执行 cerebro.run()，导致回测运行两次。
            请使用 from_strategy() 从已运行的策略实例获取指标。
        """
        import warnings
        warnings.warn(
            "calculate_metrics() 会重新执行回测，请使用 from_strategy() 代替",
            DeprecationWarning,
            stacklevel=2,
        )
        results = cerebro.run()
        strategy = results[0]
        metrics = {}
        total_return = (strategy.broker.getvalue() - strategy.broker.startingcash) / strategy.broker.startingcash
        metrics["total_return"] = total_return
        days = (strategy.datas[0].datetime.date(-1) - strategy.datas[0].datetime.date(0)).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        metrics["annual_return"] = annual_return
        return metrics
