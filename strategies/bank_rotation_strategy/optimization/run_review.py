"""深度审查脚本 - 对有效优化进行OOS、敏感性、时间稳定性测试"""
import os
import sys
import json
import datetime
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir

STRATEGY_NAME = 'bank_rotation'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
RESULTS_DIR = os.path.join(STRATEGY_DIR, 'optimization', 'review_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

EFFECTIVE_OPTS = {
    'deep01': {'name': '换仓阈值0.003', 'params': {'switch_threshold': 0.003}},
    'deep02': {'name': '换仓阈值0.004', 'params': {'switch_threshold': 0.004}},
    'combo01': {'name': '换仓0.003+最小持仓5', 'params': {'switch_threshold': 0.003, 'min_holding_bars': 5}},
    'combo05': {'name': '换仓0.003+持仓5+收盘过滤', 'params': {'switch_threshold': 0.003, 'min_holding_bars': 5, 'no_trade_close_start': '14:57'}},
}


def run_backtest(strategy_name=STRATEGY_NAME, extra_params=None,
                 start_date=TRAIN_START, end_date=TRAIN_END, label='test'):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1m'
    config['start_date'] = start_date
    config['end_date'] = end_date
    config.setdefault('benchmark', '000300.SH')

    merged_kwargs = dict(default_kwargs)
    if extra_params:
        merged_kwargs.update(extra_params)

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(**config)
    api.add_strategy(strategy_class, **merged_kwargs)
    results = api.run()

    result = api.get_result()
    metrics = {'label': label, 'extra_params': extra_params}
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
        metrics['timestamp'] = datetime.datetime.now().isoformat()
    else:
        metrics['error'] = 'No result'
    return metrics


def review_oos(opt_key, opt_info):
    """1. 样本外验证"""
    print(f"\n  [OOS] 样本外验证...")

    baseline_is = run_backtest(extra_params=None, start_date=TRAIN_START, end_date=TRAIN_END, label='baseline_is')
    baseline_oos = run_backtest(extra_params=None, start_date=VALID_START, end_date=VALID_END, label='baseline_oos')
    opt_is = run_backtest(extra_params=opt_info['params'], start_date=TRAIN_START, end_date=TRAIN_END, label=f'{opt_key}_is')
    opt_oos = run_backtest(extra_params=opt_info['params'], start_date=VALID_START, end_date=VALID_END, label=f'{opt_key}_oos')

    is_sharpe_base = baseline_is.get('sharpe_ratio', 0)
    oos_sharpe_base = baseline_oos.get('sharpe_ratio', 0)
    is_sharpe_opt = opt_is.get('sharpe_ratio', 0)
    oos_sharpe_opt = opt_oos.get('sharpe_ratio', 0)

    is_improve = (is_sharpe_opt - is_sharpe_base) / abs(is_sharpe_base) * 100 if is_sharpe_base != 0 else 0
    oos_improve = (oos_sharpe_opt - oos_sharpe_base) / abs(oos_sharpe_base) * 100 if oos_sharpe_base != 0 else 0
    decay = oos_improve / is_improve if is_improve != 0 else 0

    print(f"    Baseline IS:  Sharpe={is_sharpe_base:.4f}")
    print(f"    Baseline OOS: Sharpe={oos_sharpe_base:.4f}")
    print(f"    Opt IS:       Sharpe={is_sharpe_opt:.4f} (提升 {is_improve:+.1f}%)")
    print(f"    Opt OOS:      Sharpe={oos_sharpe_opt:.4f} (提升 {oos_improve:+.1f}%)")
    print(f"    衰减比:       {decay:.4f}")

    verdict = 'PASS' if oos_improve > 0 else 'FAIL'
    print(f"    判定:         {verdict}")

    return {
        'baseline_is_sharpe': is_sharpe_base,
        'baseline_oos_sharpe': oos_sharpe_base,
        'opt_is_sharpe': is_sharpe_opt,
        'opt_oos_sharpe': oos_sharpe_opt,
        'is_improve_pct': is_improve,
        'oos_improve_pct': oos_improve,
        'decay_ratio': decay,
        'baseline_is': baseline_is,
        'baseline_oos': baseline_oos,
        'opt_is': opt_is,
        'opt_oos': opt_oos,
        'verdict': verdict,
    }


def review_sensitivity(opt_key, opt_info):
    """2. 参数敏感性测试"""
    print(f"\n  [SENS] 参数敏感性测试...")

    param_name = list(opt_info['params'].keys())[0]
    param_value = opt_info['params'][param_name]

    perturbations = [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4]
    base_result = run_backtest(extra_params=opt_info['params'], label=f'{opt_key}_sens_base')
    base_sharpe = base_result.get('sharpe_ratio', 0)

    perturb_results = {}
    for delta in perturbations:
        perturbed = param_value * (1 + delta)
        if isinstance(param_value, int):
            perturbed = max(1, int(round(perturbed)))
        result = run_backtest(extra_params={param_name: perturbed}, label=f'{opt_key}_sens_{delta:+.0%}')
        sharpe = result.get('sharpe_ratio', 0)
        perturb_results[f'{delta:+.0%}'] = {'param_value': perturbed, 'sharpe': sharpe}
        print(f"    delta={delta:+.0%}: param={perturbed:.6f}, sharpe={sharpe:.4f}")

    sharpe_values = [r['sharpe'] for r in perturb_results.values()]
    sharpe_range = max(sharpe_values) - min(sharpe_values)
    sharpe_std = statistics.stdev(sharpe_values) if len(sharpe_values) >= 2 else 0
    sensitivity_ratio = sharpe_range / abs(base_sharpe) if base_sharpe != 0 else float('inf')

    verdict = 'PASS' if sensitivity_ratio < 0.5 else 'CAUTION' if sensitivity_ratio < 1.0 else 'FAIL'
    print(f"    Base Sharpe: {base_sharpe:.4f}")
    print(f"    Sharpe范围: {sharpe_range:.4f}, 敏感比: {sensitivity_ratio:.4f}")
    print(f"    判定: {verdict}")

    return {
        'base_param': param_value,
        'base_sharpe': base_sharpe,
        'perturb_results': perturb_results,
        'sharpe_range': sharpe_range,
        'sharpe_std': sharpe_std,
        'sensitivity_ratio': sensitivity_ratio,
        'verdict': verdict,
    }


def review_temporal(opt_key, opt_info):
    """3. 时间稳定性测试（逐年）"""
    print(f"\n  [TEMP] 时间稳定性测试...")

    yearly_results = []
    for year in range(2020, 2025):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        if year_start < TRAIN_START:
            year_start = TRAIN_START
        if year_end > TRAIN_END:
            year_end = TRAIN_END

        opt_result = run_backtest(extra_params=opt_info['params'], start_date=year_start, end_date=year_end, label=f'{opt_key}_temp_{year}')
        base_result = run_backtest(extra_params=None, start_date=year_start, end_date=year_end, label=f'baseline_temp_{year}')

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
        print(f"    {year}: opt={opt_sharpe:.4f}, base={base_sharpe:.4f}, improvement={improvement:+.4f} {'✓' if improvement > 0 else '✗'}")

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency = positive_years / total_years if total_years > 0 else 0

    verdict = 'PASS' if consistency >= 0.75 else 'CAUTION' if consistency >= 0.5 else 'FAIL'
    print(f"    一致性: {positive_years}/{total_years} = {consistency:.0%}")
    print(f"    判定: {verdict}")

    return {
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': consistency,
        'verdict': verdict,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Review specific optimization (e.g., deep01)')
    parser.add_argument('--all', action='store_true', help='Review all effective optimizations')
    args = parser.parse_args()

    targets = {}
    if args.opt:
        if args.opt in EFFECTIVE_OPTS:
            targets = {args.opt: EFFECTIVE_OPTS[args.opt]}
        else:
            print(f"Unknown: {args.opt}")
            sys.exit(1)
    elif args.all:
        targets = EFFECTIVE_OPTS
    else:
        print("Specify --opt or --all")
        sys.exit(1)

    all_reviews = {}

    for opt_key, opt_info in targets.items():
        print(f"\n{'='*80}")
        print(f"审查: {opt_key} - {opt_info['name']}")
        print(f"参数: {opt_info['params']}")
        print(f"训练期: {TRAIN_START} ~ {TRAIN_END}")
        print(f"验证期: {VALID_START} ~ {VALID_END}")
        print(f"{'='*80}")

        oos = review_oos(opt_key, opt_info)
        sens = review_sensitivity(opt_key, opt_info)
        temp = review_temporal(opt_key, opt_info)

        overall = 'PASS' if all(v['verdict'] == 'PASS' for v in [oos, sens, temp]) else \
                  'CAUTION' if any(v['verdict'] == 'FAIL' for v in [oos, sens, temp]) is False else 'FAIL'

        review = {
            'opt_key': opt_key,
            'opt_name': opt_info['name'],
            'params': opt_info['params'],
            'oos': oos,
            'sensitivity': sens,
            'temporal': temp,
            'overall_verdict': overall,
        }

        all_reviews[opt_key] = review

        review_file = os.path.join(RESULTS_DIR, f'{opt_key}_review.json')
        with open(review_file, 'w', encoding='utf-8') as f:
            json.dump(review, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"审查结果: {opt_key} - {opt_info['name']}")
        print(f"{'='*60}")
        print(f"  OOS验证:       {oos['verdict']} (IS提升{oos['is_improve_pct']:+.1f}%, OOS提升{oos['oos_improve_pct']:+.1f}%)")
        print(f"  参数敏感性:    {sens['verdict']} (敏感比{sens['sensitivity_ratio']:.4f})")
        print(f"  时间稳定性:    {temp['verdict']} (一致性{temp['consistency_ratio']:.0%})")
        print(f"  总体判定:      {overall}")
        print(f"  结果已保存: {review_file}")

    # 汇总
    print(f"\n{'='*80}")
    print("审查汇总")
    print(f"{'='*80}")
    for opt_key, review in all_reviews.items():
        print(f"  {opt_key} ({review['opt_name']}): {review['overall_verdict']}")
