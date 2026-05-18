import math
import pytest
import pandas as pd
import numpy as np

from engine.data_feed import ArrayDataFeed
from core.strategy_logic import BarData


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


class TestNormalCreation:
    def test_length(self, data_feed):
        assert data_feed.length == 5

    def test_len(self, data_feed):
        assert len(data_feed) == 5

    def test_symbol(self, data_feed):
        assert data_feed.symbol == 'TEST'

    def test_closes(self, data_feed):
        assert data_feed.closes[0] == 10.0
        assert data_feed.closes[4] == 14.0

    def test_opens(self, data_feed):
        assert data_feed.opens[0] == 10.0

    def test_highs(self, data_feed):
        assert data_feed.highs[0] == 10.5

    def test_lows(self, data_feed):
        assert data_feed.lows[0] == 9.5

    def test_volumes(self, data_feed):
        assert data_feed.volumes[0] == 1000.0


class TestEmptyCreation:
    def test_empty_df(self):
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        feed = ArrayDataFeed('EMPTY', df)
        assert feed.length == 0
        assert len(feed) == 0

    def test_empty_get_close(self):
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        feed = ArrayDataFeed('EMPTY', df)
        assert math.isnan(feed.get_close(0))


class TestNaNData:
    @pytest.fixture
    def nan_df(self):
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        return pd.DataFrame({
            'open': [10.0, np.nan, 12.0],
            'high': [10.5, np.nan, 12.5],
            'low': [9.5, np.nan, 11.5],
            'close': [10.0, np.nan, 12.0],
            'volume': [1000.0, np.nan, 3000.0],
        }, index=dates)

    def test_nan_close(self, nan_df):
        feed = ArrayDataFeed('NAN', nan_df)
        assert math.isnan(feed.get_close(1))

    def test_nan_get_bar(self, nan_df):
        feed = ArrayDataFeed('NAN', nan_df)
        bar = feed.get_bar(1)
        assert bar.close == 0.0
        assert bar.open == 0.0

    def test_is_nan(self, nan_df):
        feed = ArrayDataFeed('NAN', nan_df)
        assert feed.is_nan(1) is True
        assert feed.is_nan(0) is False

    def test_is_volume_zero_or_nan(self, nan_df):
        feed = ArrayDataFeed('NAN', nan_df)
        assert feed.is_volume_zero_or_nan(1) is True


class TestGetBar:
    def test_get_bar_valid(self, data_feed):
        bar = data_feed.get_bar(0)
        assert isinstance(bar, BarData)
        assert bar.symbol == 'TEST'
        assert bar.open == 10.0
        assert bar.high == 10.5
        assert bar.low == 9.5
        assert bar.close == 10.0
        assert bar.volume == 1000.0

    def test_get_bar_out_of_range(self, data_feed):
        bar = data_feed.get_bar(-1)
        assert bar.close == 0.0
        bar = data_feed.get_bar(999)
        assert bar.close == 0.0


class TestGetClose:
    def test_get_close_valid(self, data_feed):
        assert data_feed.get_close(0) == 10.0
        assert data_feed.get_close(4) == 14.0

    def test_get_close_out_of_range(self, data_feed):
        assert math.isnan(data_feed.get_close(-1))
        assert math.isnan(data_feed.get_close(5))


class TestGetDate:
    def test_get_date_valid(self, data_feed):
        d = data_feed.get_date(0)
        assert d is not None
        assert d.year == 2024
        assert d.month == 1
        assert d.day == 1

    def test_get_date_out_of_range(self, data_feed):
        assert data_feed.get_date(-1) is None
        assert data_feed.get_date(5) is None


class TestGetDatetime:
    def test_get_datetime_valid(self, data_feed):
        dt = data_feed.get_datetime(0)
        assert dt is not None
        assert dt.year == 2024

    def test_get_datetime_out_of_range(self, data_feed):
        assert data_feed.get_datetime(-1) is None
        assert data_feed.get_datetime(5) is None


class TestGetOpenHighLowVolume:
    def test_get_open(self, data_feed):
        assert data_feed.get_open(0) == 10.0
        assert math.isnan(data_feed.get_open(-1))

    def test_get_high(self, data_feed):
        assert data_feed.get_high(0) == 10.5
        assert math.isnan(data_feed.get_high(5))

    def test_get_low(self, data_feed):
        assert data_feed.get_low(0) == 9.5
        assert math.isnan(data_feed.get_low(-1))

    def test_get_volume(self, data_feed):
        assert data_feed.get_volume(0) == 1000.0
        assert math.isnan(data_feed.get_volume(5))


class TestIsNan:
    def test_is_nan_valid(self, data_feed):
        assert data_feed.is_nan(0) is False

    def test_is_nan_out_of_range(self, data_feed):
        assert data_feed.is_nan(-1) is True
        assert data_feed.is_nan(5) is True


class TestIsVolumeZeroOrNan:
    def test_normal_volume(self, data_feed):
        assert data_feed.is_volume_zero_or_nan(0) == False

    def test_zero_volume(self):
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        df = pd.DataFrame({
            'open': [10.0], 'high': [10.5], 'low': [9.5],
            'close': [10.0], 'volume': [0.0],
        }, index=dates)
        feed = ArrayDataFeed('ZERO', df)
        assert feed.is_volume_zero_or_nan(0) == True

    def test_out_of_range(self, data_feed):
        assert data_feed.is_volume_zero_or_nan(-1) is True
        assert data_feed.is_volume_zero_or_nan(5) is True


class TestDatetimeColumn:
    def test_datetime_column_as_index(self):
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=3, freq='D'),
            'open': [10.0, 11.0, 12.0],
            'high': [10.5, 11.5, 12.5],
            'low': [9.5, 10.5, 11.5],
            'close': [10.0, 11.0, 12.0],
            'volume': [1000.0, 2000.0, 3000.0],
        })
        feed = ArrayDataFeed('DT', df)
        assert feed.length == 3
        d = feed.get_date(0)
        assert d is not None
        assert d.year == 2024
