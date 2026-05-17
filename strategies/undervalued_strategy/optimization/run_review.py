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
FULL_START = '2020-04-28'
FULL_END = '2026-04-28'


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool=POOL, start_date=FULL_START, end_date=FULL_END):
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


def run_out_of_sample_test():
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(FULL_START, '%Y-%m-%d')
    end_dt = datetime.strptime(FULL_END, '%Y-%m-%d')
    total_days = (end_dt - start_dt).days
    split_dt = start_dt + timedelta(days=int(total_days * 0.5))
    split_date = split_dt.strftime('%Y-%m-%d')
    print(f'样本外验证: IS={FULL_START}~{split_date}, OOS={split_date}~{FULL_END}')

    print('\n--- 样本内基线 ---')
    baseline_is = run_backtest_with_params(
        extra_params=None, label='review_baseline_is',
        start_date=FULL_START, end_date=split_date)

    print('\n--- 样本外基线 ---')
    baseline_oos = run_backtest_with_params(
        extra_params=None, label='review_baseline_oos',
        start_date=split_date, end_date=FULL_END)

    print('\n--- 样本内月度调仓 ---')
    opt_is = run_backtest_with_params(
        extra_params={'rebalance_freq': 'monthly'}, label='review_monthly_is',
        start_date=FULL_START, end_date=split_date)

    print('\n--- 样本外月度调仓 ---')
    opt_oos = run_backtest_with_params(
        extra_params={'rebalance_freq': 'monthly'}, label='review_monthly_oos',
        start_date=split_date, end_date=FULL_END)

    is_sharpe_base = baseline_is.get('sharpe_ratio', 0)
    oos_sharpe_base = baseline_oos.get('sharpe_ratio', 0)
    is_sharpe_opt = opt_is.get('sharpe_ratio', 0)
    oos_sharpe_opt = opt_oos.get('sharpe_ratio', 0)

    is_improvement = (is_sharpe_opt - is_sharpe_base) / abs(is_sharpe_base) * 100 if is_sharpe_base != 0 else 0
    oos_improvement = (oos_sharpe_opt - oos_sharpe_base) / abs(oos_sharpe_base) * 100 if oos_sharpe_base != 0 else 0
    decay_ratio = oos_improvement / is_improvement if is_improvement != 0 else 0

    result = {
        'test': 'out_of_sample',
        'is_sharpe_baseline': is_sharpe_base,
        'oos_sharpe_baseline': oos_sharpe_base,
        'is_sharpe_optimized': is_sharpe_opt,
        'oos_sharpe_optimized': oos_sharpe_opt,
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': decay_ratio,
        'split_date': split_date,
    }

    print(f'\n样本外验证结果:')
    print(f'  IS: 基线={is_sharpe_base:.4f}, 月度调仓={is_sharpe_opt:.4f}, 提升={is_improvement:+.1f}%')
    print(f'  OOS: 基线={oos_sharpe_base:.4f}, 月度调仓={oos_sharpe_opt:.4f}, 提升={oos_improvement:+.1f}%')
    print(f'  衰减比: {decay_ratio:.3f}')

    return result


def run_temporal_stability_test():
    from datetime import datetime
    start_year = datetime.strptime(FULL_START, '%Y-%m-%d').year
    end_year = datetime.strptime(FULL_END, '%Y-%m-%d').year

    yearly_results = []
    for year in range(start_year, end_year + 1):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        if year_start < FULL_START:
            year_start = FULL_START
        if year_end > FULL_END:
            year_end = FULL_END

        print(f'\n--- {year}年: 基线 ---')
        base_result = run_backtest_with_params(
            extra_params=None, label=f'review_baseline_{year}',
            start_date=year_start, end_date=year_end)

        print(f'\n--- {year}年: 月度调仓 ---')
        opt_result = run_backtest_with_params(
            extra_params={'rebalance_freq': 'monthly'}, label=f'review_monthly_{year}',
            start_date=year_start, end_date=year_end)

        opt_sharpe = opt_result.get('sharpe_ratio', 0)
        base_sharpe = base_result.get('sharpe_ratio', 0)
        improvement = opt_sharpe - base_sharpe

        yearly_results.append({
            'year': year,
            'opt_sharpe': opt_sharpe,
            'base_sharpe': base_sharpe,
            'improvement': improvement,
            'is_positive': improvement > 0,
        })
        print(f'  {year}: 基线={base_sharpe:.4f}, 月度调仓={opt_sharpe:.4f}, 差值={improvement:+.4f}')

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency_ratio = positive_years / total_years if total_years > 0 else 0

    result = {
        'test': 'temporal_stability',
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': consistency_ratio,
    }

    print(f'\n时间稳定性结果:')
    print(f'  正改进年数: {positive_years}/{total_years}')
    print(f'  一致性比率: {consistency_ratio:.2f}')

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['oos', 'temporal', 'all'], default='all')
    args = parser.parse_args()

    results = {}

    if args.test in ('oos', 'all'):
        print('=' * 60)
        print('检测1: 样本外验证')
        print('=' * 60)
        results['out_of_sample'] = run_out_of_sample_test()

    if args.test in ('temporal', 'all'):
        print('\n' + '=' * 60)
        print('检测2: 时间分段稳定性')
        print('=' * 60)
        results['temporal_stability'] = run_temporal_stability_test()

    review_file = os.path.join(RESULTS_DIR, 'review_summary.json')
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n审查结果已保存: {review_file}')
