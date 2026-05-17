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

STRATEGY_NAME = 'undervalued'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

POOL = '中证1000'
START_DATE = '2020-04-28'
END_DATE = '2026-04-28'

OPTIMIZATIONS = [
    ('opt01_volatility_5pct', '波动率过滤(5%)', {'max_volatility': 0.05}),
    ('opt02_monthly_rebalance', '月度调仓', {'rebalance_freq': 'monthly'}),
    ('opt03_pe_filter_20', 'PE过滤(<20)', {'max_pe': 20}),
    ('opt04_dividend_yield_2pct', '股息率过滤(>2%)', {'min_dividend_yield': 0.02}),
    ('opt05_roe_filter_10pct', 'ROE过滤(>10%)', {'min_roe': 0.10}),
    ('opt06_industry_limit_3', '行业集中度限制(每行业3只)', {'max_per_industry': 3}),
    ('opt07_liquidity_50m', '流动性过滤(日均>5000万)', {'min_avg_amount': 50000000}),
    ('opt08_composite_score', '估值综合评分(PB+股息率)', {'use_composite_score': True}),
    ('opt09_semi_annual_rebalance', '半年调仓', {'rebalance_freq': 'semi_annual'}),
    ('opt10_vol_dividend_combo', '波动率+股息率组合', {'max_volatility': 0.05, 'min_dividend_yield': 0.02}),
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
    parser.add_argument('--start-from', type=int, default=0, help='Start from optimization index (0-based)')
    args = parser.parse_args()

    if args.start_from == 0:
        print('=' * 60)
        print('Running BASELINE backtest...')
        print('=' * 60)
        baseline = run_backtest_with_params(label='baseline')
        print(json.dumps(baseline, indent=2, ensure_ascii=False))
        baseline_sharpe = baseline.get('sharpe_ratio', 0)
        print(f'\nBaseline Sharpe: {baseline_sharpe:.4f}')
        print()

    for i, (label, name, params) in enumerate(OPTIMIZATIONS):
        if i < args.start_from:
            continue
        print('=' * 60)
        print(f'Running opt{i+1:02d}: {name}')
        print(f'Params: {params}')
        print('=' * 60)
        try:
            result = run_backtest_with_params(extra_params=params, label=label)
            sharpe = result.get('sharpe_ratio', 0)
            baseline_sharpe_val = 0.677
            improvement = (sharpe - baseline_sharpe_val) / abs(baseline_sharpe_val) * 100
            print(f'Sharpe: {sharpe:.4f} (vs baseline {baseline_sharpe_val:.4f}, change: {improvement:+.1f}%)')
            print(f'Return: {result.get("total_return_pct", 0):.2f}%')
            print(f'MaxDD: {result.get("max_drawdown_pct", 0):.2f}%')
            print()
        except Exception as e:
            print(f'ERROR: {e}')
            traceback.print_exc()
            error_result = {
                'label': label,
                'error': str(e),
                'extra_params': params,
                'timestamp': datetime.datetime.now().isoformat(),
            }
            result_file = os.path.join(RESULTS_DIR, f'{label}.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            print()

    print('All optimizations completed!')
