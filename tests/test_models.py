import math
import pytest
import pandas as pd
import numpy as np

from core.models import BacktestingResult, AccountInfo


class TestSharpeRatio:
    def test_sharpe_positive_returns(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000, 101000, 102000, 103000, 104000],
            'PnL': [0, 1000, 1000, 1000, 1000],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        sr = result.sharpe_ratio()
        assert sr > 0

    def test_sharpe_zero_returns(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000, 100000, 100000, 100000],
            'PnL': [0, 0, 0, 0],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        sr = result.sharpe_ratio()
        assert sr == 0.0

    def test_sharpe_negative_returns(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000, 99000, 98000, 97000],
            'PnL': [0, -1000, -1000, -1000],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        sr = result.sharpe_ratio()
        assert sr < 0

    def test_sharpe_no_df(self):
        result = BacktestingResult()
        assert result.sharpe_ratio() == 0.0

    def test_sharpe_empty_df(self):
        result = BacktestingResult(df=pd.DataFrame())
        assert result.sharpe_ratio() == 0.0

    def test_sharpe_missing_columns(self):
        result = BacktestingResult(df=pd.DataFrame({'A': [1, 2, 3]}))
        assert result.sharpe_ratio() == 0.0

    def test_sharpe_with_risk_free_rate(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000, 101000, 102000, 103000, 104000],
            'PnL': [0, 1000, 1000, 1000, 1000],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        sr_no_rf = result.sharpe_ratio(risk_free_rate=0.0)
        sr_with_rf = result.sharpe_ratio(risk_free_rate=0.03)
        assert sr_with_rf < sr_no_rf

    def test_sharpe_too_few_data(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000],
            'PnL': [0],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        assert result.sharpe_ratio() == 0.0


class TestMaxDrawdown:
    def test_max_drawdown_no_drawdown(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000, 110000, 120000, 130000],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        dd = result.max_drawdown()
        assert dd == pytest.approx(0.0)

    def test_max_drawdown_simple(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000, 110000, 90000, 95000],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        dd = result.max_drawdown()
        expected = (90000 - 110000) / 110000
        assert dd == pytest.approx(expected)

    def test_max_drawdown_full_loss(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000, 50000],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        dd = result.max_drawdown()
        expected = (50000 - 100000) / 100000
        assert dd == pytest.approx(expected)

    def test_max_drawdown_no_df(self):
        result = BacktestingResult()
        assert result.max_drawdown() == 0.0

    def test_max_drawdown_empty_df(self):
        result = BacktestingResult(df=pd.DataFrame())
        assert result.max_drawdown() == 0.0

    def test_max_drawdown_missing_column(self):
        result = BacktestingResult(df=pd.DataFrame({'A': [1, 2, 3]}))
        assert result.max_drawdown() == 0.0

    def test_max_drawdown_multiple_peaks(self):
        df = pd.DataFrame({
            'PortfolioValue': [100000, 120000, 110000, 130000, 100000],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        dd = result.max_drawdown()
        expected = (100000 - 130000) / 130000
        assert dd == pytest.approx(expected)


class TestAnnualReturn:
    def test_annual_return_positive(self):
        df = pd.DataFrame({'PnL': [0] * 252})
        result = BacktestingResult(df=df)
        result.prepare_data()
        ar = result.annual_return(100000, 120000)
        assert ar > 0

    def test_annual_return_negative(self):
        df = pd.DataFrame({'PnL': [0] * 252})
        result = BacktestingResult(df=df)
        result.prepare_data()
        ar = result.annual_return(100000, 80000)
        assert ar < 0

    def test_annual_return_zero_initial(self):
        result = BacktestingResult()
        result.total_trading_days = 252
        ar = result.annual_return(0, 120000)
        assert ar == 0.0

    def test_annual_return_zero_days(self):
        result = BacktestingResult()
        ar = result.annual_return(100000, 120000)
        assert ar == 0.0

    def test_annual_return_exact_one_year(self):
        df = pd.DataFrame({'PnL': [0] * 252})
        result = BacktestingResult(df=df)
        result.prepare_data()
        ar = result.annual_return(100000, 110000)
        assert ar == pytest.approx(0.1)

    def test_annual_return_half_year(self):
        df = pd.DataFrame({'PnL': [0] * 126})
        result = BacktestingResult(df=df)
        result.prepare_data()
        ar = result.annual_return(100000, 110000)
        expected = (1 + 0.1) ** (1 / 0.5) - 1
        assert ar == pytest.approx(expected)


class TestPrepareData:
    def test_prepare_data_computes_pnl_days(self):
        df = pd.DataFrame({
            'PnL': [0, 100, -50, 200, -30],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        assert result.pnl_days == (2, 2)

    def test_prepare_data_total_trading_days(self):
        df = pd.DataFrame({
            'PnL': [0, 100, -50, 200],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        assert result.total_trading_days == 4

    def test_prepare_data_no_df(self):
        result = BacktestingResult()
        result.prepare_data()
        assert result.pnl_days == (0, 0)

    def test_prepare_data_empty_df(self):
        result = BacktestingResult(df=pd.DataFrame())
        result.prepare_data()
        assert result.pnl_days == (0, 0)

    def test_prepare_data_idempotent(self):
        df = pd.DataFrame({
            'PnL': [0, 100, -50],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        result.prepare_data()
        assert result.pnl_days == (1, 1)

    def test_prepare_data_all_positive(self):
        df = pd.DataFrame({
            'PnL': [100, 200, 300],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        assert result.pnl_days == (3, 0)

    def test_prepare_data_all_negative(self):
        df = pd.DataFrame({
            'PnL': [-100, -200, -300],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        assert result.pnl_days == (0, 3)

    def test_prepare_data_no_pnl_column(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
        })
        result = BacktestingResult(df=df)
        result.prepare_data()
        assert result.pnl_days == (0, 0)
        assert result.total_trading_days == 3


class TestDefaultValues:
    def test_default_account(self):
        result = BacktestingResult()
        assert isinstance(result.account, AccountInfo)

    def test_default_pnl_days(self):
        result = BacktestingResult()
        assert result.pnl_days == (0, 0)

    def test_default_total_trading_days(self):
        result = BacktestingResult()
        assert result.total_trading_days == 0
