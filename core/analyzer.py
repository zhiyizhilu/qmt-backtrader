import backtrader as bt
import pandas as pd
import numpy as np
import datetime as dt_module
from typing import Dict, List, Optional, Any

from core.models import (
    BacktestingResult,
    AccountInfo,
    BacktestConfig,
    TradeRecord,
    InstrumentData,
)


class PerformanceAnalyzer:
    def __init__(self):
        self._cerebro: Optional[bt.Cerebro] = None
        self._strategy = None
        self._initial_cash: float = 0.0

    def set_context(self, cerebro: bt.Cerebro, strategy, initial_cash: float = 0.0):
        self._cerebro = cerebro
        self._strategy = strategy
        self._initial_cash = initial_cash

    def build_result(
        self,
        strategy_params: Optional[Dict[str, Any]] = None,
        show_kline: bool = True,
    ) -> BacktestingResult:
        if self._strategy is None:
            return BacktestingResult()

        account = self._build_account()
        df = self._build_equity_dataframe()
        trade_log = self._build_trade_log()
        klines = self._build_klines()
        instruments_data = self._build_instruments_data()

        result = BacktestingResult(
            account=account,
            config=BacktestConfig(show_kline=show_kline),
            strategy_params=strategy_params or {},
            klines=klines,
            trade_log=trade_log,
            df=df,
            instruments_data=instruments_data,
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

        if not equity_history or len(equity_history) == 0:
            return None

        dates = []
        portfolio_values = []
        daily_pnls = []

        for i, (dt, value) in enumerate(equity_history):
            dates.append(pd.Timestamp(dt))
            portfolio_values.append(value)
            if i == 0:
                daily_pnls.append(0.0)
            else:
                daily_pnls.append(value - equity_history[i - 1][1])

        df = pd.DataFrame({
            "datetime": dates,
            "PortfolioValue": portfolio_values,
            "PnL": daily_pnls,
        })
        return df

    def _build_trade_log(self) -> List[TradeRecord]:
        strat = self._strategy
        trades: List[TradeRecord] = []

        logic = getattr(strat, '_strategy_logic', None)
        logic_orders = logic.get_orders() if logic and hasattr(logic, 'get_orders') else getattr(logic, '_orders', None)
        if not logic_orders:
            return trades

        trade_records = strat.get_trade_records() if hasattr(strat, 'get_trade_records') else getattr(strat, '_trade_records', None) or []
        completed_orders = [
            o for o in logic_orders.values()
            if o.is_completed and (o.executed_volume > 0 or o.executed_price > 0)
        ]

        # 跟踪持仓，用于判断开平仓
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
                    if close_vol > 0:
                        offset = "1"
                    else:
                        offset = "0"
                elif current_pos == 0:
                    offset = "0"
                else:
                    offset = "0"
                new_pos = current_pos + exec_vol
            else:
                if current_pos > 0:
                    close_vol = min(exec_vol, current_pos)
                    if close_vol > 0:
                        offset = "1"
                    else:
                        offset = "0"
                elif current_pos == 0:
                    offset = "0"
                else:
                    offset = "0"
                new_pos = current_pos - exec_vol

            positions[order_info.symbol] = new_pos

            trade_datetime = getattr(order_info, 'datetime', None)
            pnl_val = 0.0
            fee_val = getattr(order_info, 'commission', 0.0)

            if trade_datetime is None and trade_rec_idx < len(trade_records):
                tr = trade_records[trade_rec_idx]
                trade_datetime = tr.get("datetime")
                pnl_val = tr.get("pnl_no_commission", 0.0)
                fee_val = tr.get("commission", fee_val)
                trade_rec_idx += 1

            if trade_datetime is None and trade_rec_idx > 0 and trade_rec_idx <= len(trade_records):
                trade_datetime = trade_records[min(trade_rec_idx - 1, len(trade_records) - 1)].get("datetime")

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
                memo="",
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
                    bar_dt = bar_dt.replace(hour=14, minute=50)
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
        instruments_data: Dict[str, InstrumentData] = {}

        if hasattr(strat, "datas") and strat.datas:
            for data_feed in strat.datas:
                symbol = getattr(data_feed, "_name", "")
                if symbol:
                    instruments_data[symbol] = InstrumentData(volume_multiple=1.0)

        return instruments_data

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

    def calculate_metrics(self, cerebro: bt.Cerebro) -> Dict[str, float]:
        results = cerebro.run()
        strategy = results[0]
        metrics = {}
        total_return = (strategy.broker.getvalue() - strategy.broker.startingcash) / strategy.broker.startingcash
        metrics["total_return"] = total_return
        days = (strategy.datas[0].datetime.date(-1) - strategy.datas[0].datetime.date(0)).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        metrics["annual_return"] = annual_return
        return metrics
