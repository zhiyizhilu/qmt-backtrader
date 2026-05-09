import os
import sys
import json
import datetime
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'ivff3'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool='中证1000', start_date='2020-04-28', end_date='2026-04-28'):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1d'
    config['start_date'] = start_date
    config['end_date'] = end_date
    benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get(pool, '000300.SH')
    config.setdefault('benchmark', benchmark)

    merged_kwargs = dict(default_kwargs)
    if extra_params:
        merged_kwargs.update(extra_params)

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(**config)
    api.load_financial_data(sector=pool)
    api.add_stock_selection_strategy(strategy_class, **merged_kwargs)
    results = api.run()

    result = api.get_result()
    metrics = {}
    if result:
        sr = result.sharpe_ratio()
        dd = result.max_drawdown()
        acc = result.account
        metrics['initial_capital'] = acc.initial_capital
        metrics['final_value'] = acc.dynamic_rights
        metrics['total_return_pct'] = acc.rate * 100
        metrics['sharpe_ratio'] = sr
        metrics['max_drawdown_pct'] = dd * 100
        if result.df is not None and len(result.df) > 0:
            days = len(result.df)
            years = days / 252
            annual_ret = (1 + acc.rate) ** (1 / years) - 1 if years > 0 else 0
            if isinstance(annual_ret, complex):
                annual_ret = annual_ret.real
            metrics['annual_return_pct'] = float(annual_ret) * 100
            metrics['trading_days'] = days
        metrics['fee'] = getattr(result, 'total_fee', 0)
        metrics['turnover'] = getattr(result, 'turnover', 0)
        metrics['label'] = label
        metrics['extra_params'] = extra_params
        metrics['timestamp'] = datetime.datetime.now().isoformat()
    else:
        metrics['label'] = label
        metrics['error'] = 'No result'

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


if __name__ == '__main__':
    opt_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    optimizations = {
        0: ('baseline', None),
        1: ('opt01_volatility_filter', {'max_volatility': 0.04}),
        2: ('opt02_stop_loss', {'stop_loss_pct': 0.08}),
        3: ('opt03_industry_dispersion', {'max_industry_stocks': 3}),
        4: ('opt04_min_profit_growth', {'min_profit_growth': 0.0}),
        5: ('opt05_max_debt_ratio', {'max_debt_ratio': 0.6}),
        6: ('opt06_biweekly_rebalance', {'rebalance_freq': 'biweekly'}),
        7: ('opt07_extended_regression', {'regression_window': 60, 'min_regression_window': 40}),
        8: ('opt08_iv_percentile_filter', {'max_iv_percentile': 0.3}),
        9: ('opt09_min_roe', {'min_roe': 0.05}),
        10: ('opt10_position_sizing', {'position_ratio': 0.85}),
        11: ('opt11_combined_vol_stoploss', {'max_volatility': 0.04, 'stop_loss_pct': 0.08}),
    }

    if opt_num in optimizations:
        label, params = optimizations[opt_num]
        try:
            result = run_backtest_with_params(STRATEGY_NAME, params, label)
            print(f'\n--- {label} ---')
            print(f'Sharpe: {result.get("sharpe_ratio", "N/A")}')
            print(f'Total Return: {result.get("total_return_pct", "N/A")}%')
            print(f'Max Drawdown: {result.get("max_drawdown_pct", "N/A")}%')
            if 'annual_return_pct' in result:
                print(f'Annual Return: {result["annual_return_pct"]:.2f}%')
        except Exception as e:
            print(f'Error running {label}: {e}')
            traceback.print_exc()
    else:
        print(f'Unknown optimization number: {opt_num}')
        print(f'Available: {list(optimizations.keys())}')
