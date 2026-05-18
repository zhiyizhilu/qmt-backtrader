import os
import sys
import json
import datetime
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'undervalued'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

POOL = '中证1000'
START_DATE = '2020-04-28'
END_DATE = '2026-04-28'

FREQ_TESTS = [
    ('freq_biweekly', '双周调仓', {'rebalance_freq': 'biweekly'}),
    ('freq_weekly', '周度调仓', {'rebalance_freq': 'weekly'}),
    ('freq_daily', '日度调仓', {'rebalance_freq': 'daily'}),
]


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool=POOL, start_date=START_DATE, end_date=END_DATE):
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-from', type=int, default=0)
    args = parser.parse_args()

    baseline_sharpe = 0.677

    for i, (label, name, params) in enumerate(FREQ_TESTS):
        if i < args.start_from:
            continue
        print('=' * 60)
        print(f'Running: {name}')
        print(f'Params: {params}')
        print('=' * 60)
        try:
            result = run_backtest_with_params(extra_params=params, label=label)
            sharpe = result.get('sharpe_ratio', 0)
            improvement = (sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100
            print(f'Sharpe: {sharpe:.4f} (vs baseline {baseline_sharpe:.4f}, change: {improvement:+.1f}%)')
            print(f'Return: {result.get("total_return_pct", 0):.2f}%')
            print(f'MaxDD: {result.get("max_drawdown_pct", 0):.2f}%')
            print(f'Annual: {result.get("annual_return_pct", 0):.2f}%')
            print(f'Turnover: {result.get("turnover", 0):,.0f}')
            print()
        except Exception as e:
            print(f'ERROR: {e}')
            traceback.print_exc()
            print()

    print('All frequency tests completed!')
