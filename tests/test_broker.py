import math
import pytest
import pandas as pd
import numpy as np

from engine.broker import SimulatedBroker, Position, Order, Trade
from engine.data_feed import ArrayDataFeed


@pytest.fixture
def sample_df():
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'open': [10.0, 11.0, 12.0, 13.0, 14.0],
        'high': [10.5, 11.5, 12.5, 13.5, 14.5],
        'low': [9.5, 10.5, 11.5, 12.5, 13.5],
        'close': [10.0, 11.0, 12.0, 13.0, 14.0],
        'volume': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
    }, index=dates)


@pytest.fixture
def data_feed(sample_df):
    return ArrayDataFeed('TEST', sample_df)


@pytest.fixture
def broker():
    return SimulatedBroker(cash=100000.0, commission=0.0003, slippage=0.0)


class TestBuyBasic:
    def test_buy_success(self, broker, data_feed):
        order = broker.submit_buy('TEST', 100, data_feed, 0)
        assert order is not None
        assert order.is_completed
        assert order.direction == 'buy'
        assert order.executed_size == 100
        assert order.executed_price == 10.0

    def test_buy_updates_cash(self, broker, data_feed):
        initial_cash = broker.getcash()
        order = broker.submit_buy('TEST', 100, data_feed, 0)
        expected_cost = 10.0 * 100 + 10.0 * 100 * 0.0003
        assert broker.getcash() == pytest.approx(initial_cash - expected_cost)

    def test_buy_updates_position(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        pos = broker.get_position('TEST')
        assert pos.size == 100
        assert pos.avg_price == 10.0

    def test_buy_invalid_idx(self, broker, data_feed):
        order = broker.submit_buy('TEST', 100, data_feed, -1)
        assert order is None
        order = broker.submit_buy('TEST', 100, data_feed, 999)
        assert order is None


class TestSellBasic:
    def test_sell_success(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        order = broker.submit_sell('TEST', 50, data_feed, 1)
        assert order is not None
        assert order.is_completed
        assert order.direction == 'sell'
        assert order.executed_size == 50
        assert order.executed_price == 11.0

    def test_sell_updates_position(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        broker.submit_sell('TEST', 50, data_feed, 1)
        pos = broker.get_position('TEST')
        assert pos.size == 50
        assert pos.avg_price == 10.0

    def test_sell_all_position(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        broker.submit_sell('TEST', 100, data_feed, 1)
        pos = broker.get_position('TEST')
        assert pos.size == 0
        assert pos.avg_price == 0.0

    def test_sell_more_than_position(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        order = broker.submit_sell('TEST', 200, data_feed, 1)
        assert order is not None
        assert order.executed_size == 100


class TestCommission:
    def test_buy_commission(self, broker, data_feed):
        order = broker.submit_buy('TEST', 100, data_feed, 0)
        expected_comm = 10.0 * 100 * 0.0003
        assert order.commission == pytest.approx(expected_comm)

    def test_sell_commission(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        order = broker.submit_sell('TEST', 100, data_feed, 1)
        expected_comm = 11.0 * 100 * 0.0003
        assert order.commission == pytest.approx(expected_comm)

    def test_custom_commission(self, data_feed):
        broker = SimulatedBroker(cash=100000.0, commission=0.001)
        order = broker.submit_buy('TEST', 100, data_feed, 0)
        expected_comm = 10.0 * 100 * 0.001
        assert order.commission == pytest.approx(expected_comm)

    def test_zero_commission(self, data_feed):
        broker = SimulatedBroker(cash=100000.0, commission=0.0)
        order = broker.submit_buy('TEST', 100, data_feed, 0)
        assert order.commission == 0.0


class TestSlippage:
    def test_buy_slippage(self, data_feed):
        broker = SimulatedBroker(cash=100000.0, commission=0.0, slippage=0.01)
        order = broker.submit_buy('TEST', 100, data_feed, 0)
        expected_price = 10.0 * (1 + 0.01)
        assert order.executed_price == pytest.approx(expected_price)

    def test_sell_slippage(self, data_feed):
        broker = SimulatedBroker(cash=100000.0, commission=0.0, slippage=0.01)
        broker.submit_buy('TEST', 100, data_feed, 0)
        order = broker.submit_sell('TEST', 100, data_feed, 1)
        expected_price = 11.0 * (1 - 0.01)
        assert order.executed_price == pytest.approx(expected_price)

    def test_no_slippage(self, broker, data_feed):
        order = broker.submit_buy('TEST', 100, data_feed, 0)
        assert order.executed_price == 10.0


class TestMarginReject:
    def test_insufficient_funds_rejected(self, broker, data_feed):
        order = broker.submit_buy('TEST', 100000, data_feed, 0)
        assert order is not None
        assert order.status == Order.STATUS_REJECTED

    def test_insufficient_funds_no_position_change(self, broker, data_feed):
        broker.submit_buy('TEST', 100000, data_feed, 0)
        pos = broker.get_position('TEST')
        assert pos.size == 0

    def test_insufficient_funds_no_cash_change(self, broker, data_feed):
        cash_before = broker.getcash()
        broker.submit_buy('TEST', 100000, data_feed, 0)
        assert broker.getcash() == cash_before

    def test_check_submit_disabled(self, data_feed):
        broker = SimulatedBroker(cash=1000.0, commission=0.0)
        broker.set_checksubmit(False)
        order = broker.submit_buy('TEST', 1000, data_feed, 0)
        assert order is not None
        assert order.is_completed


class TestSellNoPosition:
    def test_sell_no_position(self, broker, data_feed):
        order = broker.submit_sell('TEST', 100, data_feed, 0)
        assert order is None

    def test_sell_zero_position(self, broker, data_feed):
        pos = broker.get_position('TEST')
        assert pos.size == 0
        order = broker.submit_sell('TEST', 100, data_feed, 0)
        assert order is None


class TestAvgPrice:
    def test_avg_price_single_buy(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        pos = broker.get_position('TEST')
        assert pos.avg_price == 10.0

    def test_avg_price_multiple_buys(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        broker.submit_buy('TEST', 100, data_feed, 1)
        pos = broker.get_position('TEST')
        expected_avg = (100 * 10.0 + 100 * 11.0) / 200
        assert pos.avg_price == pytest.approx(expected_avg)

    def test_avg_price_after_partial_sell(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        broker.submit_buy('TEST', 100, data_feed, 1)
        broker.submit_sell('TEST', 100, data_feed, 2)
        pos = broker.get_position('TEST')
        assert pos.size == 100
        assert pos.avg_price == pytest.approx((100 * 10.0 + 100 * 11.0) / 200)

    def test_avg_price_after_full_sell(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        broker.submit_sell('TEST', 100, data_feed, 1)
        pos = broker.get_position('TEST')
        assert pos.size == 0
        assert pos.avg_price == 0.0


class TestTradeRecording:
    def test_sell_generates_trade(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        broker.submit_sell('TEST', 100, data_feed, 1)
        trades = broker.get_trades()
        assert len(trades) == 1
        assert trades[0].direction == 'sell'
        assert trades[0].size == 100

    def test_trade_pnl(self, broker, data_feed):
        broker.setcommission(0.0)
        broker.submit_buy('TEST', 100, data_feed, 0)
        broker.submit_sell('TEST', 100, data_feed, 1)
        trades = broker.get_trades()
        expected_pnl = 100 * (11.0 - 10.0)
        assert trades[0].pnl == pytest.approx(expected_pnl)

    def test_trade_pnlcomm(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        broker.submit_sell('TEST', 100, data_feed, 1)
        trades = broker.get_trades()
        expected_pnl = 100 * (11.0 - 10.0)
        expected_comm = 11.0 * 100 * 0.0003
        assert trades[0].pnlcomm == pytest.approx(expected_pnl - expected_comm)


class TestGetValue:
    def test_getvalue_no_positions(self, broker):
        assert broker.getvalue() == 100000.0

    def test_getvalue_with_positions(self, broker, data_feed):
        broker.submit_buy('TEST', 100, data_feed, 0)
        value = broker.getvalue(data_feeds={'TEST': data_feed}, current_indices={'TEST': 0})
        expected = broker.getcash() + 100 * 10.0
        assert value == pytest.approx(expected)

    def test_startingcash(self, broker):
        assert broker.startingcash == 100000.0
