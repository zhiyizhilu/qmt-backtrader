"""银行轮动策略优化脚本 - 分钟线模式"""
import os
import sys
import json
import datetime
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir

STRATEGY_NAME = 'bank_rotation'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              start_date=TRAIN_START, end_date=TRAIN_END):
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


def run_out_of_sample_test(strategy_name, extra_params, label):
    """使用固定验证集进行样本外测试"""
    in_sample = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=extra_params,
        label=f'{label}_is', start_date=TRAIN_START, end_date=TRAIN_END)

    out_sample = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=extra_params,
        label=f'{label}_oos', start_date=VALID_START, end_date=VALID_END)

    baseline_is = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=None,
        label='baseline_is', start_date=TRAIN_START, end_date=TRAIN_END)

    baseline_oos = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=None,
        label='baseline_oos', start_date=VALID_START, end_date=VALID_END)

    is_improvement = (in_sample.get('sharpe_ratio', 0) - baseline_is.get('sharpe_ratio', 0)) / abs(baseline_is.get('sharpe_ratio', 1)) * 100
    oos_improvement = (out_sample.get('sharpe_ratio', 0) - baseline_oos.get('sharpe_ratio', 0)) / abs(baseline_oos.get('sharpe_ratio', 1)) * 100

    return {
        'train_period': f'{TRAIN_START} ~ {TRAIN_END}',
        'valid_period': f'{VALID_START} ~ {VALID_END}',
        'in_sample_sharpe': in_sample.get('sharpe_ratio', 0),
        'out_sample_sharpe': out_sample.get('sharpe_ratio', 0),
        'baseline_is_sharpe': baseline_is.get('sharpe_ratio', 0),
        'baseline_oos_sharpe': baseline_oos.get('sharpe_ratio', 0),
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': oos_improvement / is_improvement if is_improvement != 0 else 0,
    }


def run_parameter_sensitivity_test(strategy_name, param_name, param_value,
                                    label, perturbations=[-0.2, -0.1, 0.1, 0.2]):
    results = {}
    base_result = run_backtest_with_params(
        strategy_name=strategy_name,
        extra_params={param_name: param_value},
        label=f'{label}_base')

    for delta in perturbations:
        perturbed_value = param_value * (1 + delta)
        if isinstance(param_value, int):
            perturbed_value = int(round(perturbed_value))
            if perturbed_value == 0:
                perturbed_value = 1
        result = run_backtest_with_params(
            strategy_name=strategy_name,
            extra_params={param_name: perturbed_value},
            label=f'{label}_perturb_{delta:+.0%}')
        results[f'perturb_{delta:+.0%}'] = {
            'param_value': perturbed_value,
            'sharpe_ratio': result.get('sharpe_ratio', 0),
        }

    base_sharpe = base_result.get('sharpe_ratio', 0)
    sharpe_values = [r['sharpe_ratio'] for r in results.values()]
    sharpe_std = (sum((s - base_sharpe) ** 2 for s in sharpe_values) / len(sharpe_values)) ** 0.5
    sharpe_range = max(sharpe_values) - min(sharpe_values)

    return {
        'base_sharpe': base_sharpe,
        'base_param': param_value,
        'perturbation_results': results,
        'sharpe_std': sharpe_std,
        'sharpe_range': sharpe_range,
        'sensitivity_ratio': sharpe_range / abs(base_sharpe) if base_sharpe != 0 else float('inf'),
    }


def run_temporal_stability_test(strategy_name, extra_params, label,
                                 full_start=TRAIN_START, full_end=TRAIN_END):
    from datetime import datetime
    start_year = datetime.strptime(full_start, '%Y-%m-%d').year
    end_year = datetime.strptime(full_end, '%Y-%m-%d').year

    yearly_results = []
    for year in range(start_year, end_year + 1):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        if year_start < full_start:
            year_start = full_start
        if year_end > full_end:
            year_end = full_end

        opt_result = run_backtest_with_params(
            strategy_name=strategy_name, extra_params=extra_params,
            label=f'{label}_{year}', start_date=year_start, end_date=year_end)

        base_result = run_backtest_with_params(
            strategy_name=strategy_name, extra_params=None,
            label=f'baseline_{year}', start_date=year_start, end_date=year_end)

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

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency_ratio = positive_years / total_years if total_years > 0 else 0

    return {
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': consistency_ratio,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true', help='Run baseline backtest')
    parser.add_argument('--opt', type=str, help='Run specific optimization (e.g., opt01)')
    parser.add_argument('--all', action='store_true', help='Run all optimizations')
    parser.add_argument('--review', type=str, help='Run review for specific optimization')
    parser.add_argument('--combined', type=str, help='Run combined optimization with params JSON')
    args = parser.parse_args()

    # 优化方案定义
    OPTIMIZATIONS = {
        'opt01': {'name': '换仓阈值0.008', 'params': {'switch_threshold': 0.008}},
        'opt01b': {'name': '换仓阈值0.010', 'params': {'switch_threshold': 0.010}},
        'opt02': {'name': '波动率过滤0.03', 'params': {'max_volatility': 0.03}},
        'opt02b': {'name': '波动率过滤0.02', 'params': {'max_volatility': 0.02}},
        'opt03': {'name': '最小持仓30分钟', 'params': {'min_holding_bars': 30}},
        'opt03b': {'name': '最小持仓60分钟', 'params': {'min_holding_bars': 60}},
        'opt04': {'name': '开盘过滤09:30-09:45', 'params': {'no_trade_start': '09:30', 'no_trade_end': '09:45'}},
        'opt04b': {'name': '开盘+收盘过滤', 'params': {'no_trade_start': '09:30', 'no_trade_end': '09:45', 'no_trade_close_start': '14:55'}},
        'opt05': {'name': '趋势过滤MA20', 'params': {'ma_period': 20}},
        'opt05b': {'name': '趋势过滤MA60', 'params': {'ma_period': 60}},
        'opt06': {'name': '自适应阈值0.5', 'params': {'adaptive_threshold': 0.5}},
        'opt06b': {'name': '自适应阈值1.0', 'params': {'adaptive_threshold': 1.0}},
        'opt07': {'name': '比率确认3分钟', 'params': {'confirm_bars': 3}},
        'opt07b': {'name': '比率确认5分钟', 'params': {'confirm_bars': 5}},
        'opt08': {'name': '每日最大换仓3次', 'params': {'max_daily_trades': 3}},
        'opt08b': {'name': '每日最大换仓5次', 'params': {'max_daily_trades': 5}},
        'opt09': {'name': '当日回撤限制1%', 'params': {'daily_drawdown_limit': 0.01}},
        'opt09b': {'name': '当日回撤限制2%', 'params': {'daily_drawdown_limit': 0.02}},
        'opt10': {'name': '扩展标的(交行+招行)', 'params': {'extra_stocks': {'交通银行': '601328.SH', '招商银行': '600036.SH'}}},
        # 深层优化
        'deep01': {'name': '换仓阈值0.003', 'params': {'switch_threshold': 0.003}},
        'deep02': {'name': '换仓阈值0.004', 'params': {'switch_threshold': 0.004}},
        'deep03': {'name': '最小持仓5分钟', 'params': {'min_holding_bars': 5}},
        'deep04': {'name': '最小持仓10分钟', 'params': {'min_holding_bars': 10}},
        'deep05': {'name': '开盘过滤09:30-09:35', 'params': {'no_trade_start': '09:30', 'no_trade_end': '09:35'}},
        'deep06': {'name': '收盘过滤14:57', 'params': {'no_trade_close_start': '14:57'}},
        'deep07': {'name': '每日最大换仓10次', 'params': {'max_daily_trades': 10}},
        'deep08': {'name': '当日回撤0.5%', 'params': {'daily_drawdown_limit': 0.005}},
        'deep09': {'name': '自适应阈值0.2', 'params': {'adaptive_threshold': 0.2}},
        'deep10': {'name': '比率确认1分钟', 'params': {'confirm_bars': 1}},
        'deep11': {'name': '比率确认2分钟', 'params': {'confirm_bars': 2}},
        # 组合优化
        'combo01': {'name': '换仓0.003+最小持仓5', 'params': {'switch_threshold': 0.003, 'min_holding_bars': 5}},
        'combo02': {'name': '换仓0.004+最小持仓5', 'params': {'switch_threshold': 0.004, 'min_holding_bars': 5}},
        'combo03': {'name': '换仓0.003+收盘过滤', 'params': {'switch_threshold': 0.003, 'no_trade_close_start': '14:57'}},
        'combo04': {'name': '最小持仓5+收盘过滤', 'params': {'min_holding_bars': 5, 'no_trade_close_start': '14:57'}},
        'combo05': {'name': '换仓0.003+最小持仓5+收盘过滤', 'params': {'switch_threshold': 0.003, 'min_holding_bars': 5, 'no_trade_close_start': '14:57'}},
        'combo06': {'name': '换仓0.004+最小持仓10+收盘过滤', 'params': {'switch_threshold': 0.004, 'min_holding_bars': 10, 'no_trade_close_start': '14:57'}},
    }

    if args.baseline:
        print("Running baseline backtest...")
        result = run_backtest_with_params(label='baseline')
        print(f"Baseline: Sharpe={result.get('sharpe_ratio', 0):.4f}, "
              f"Return={result.get('total_return_pct', 0):.2f}%, "
              f"Drawdown={result.get('max_drawdown_pct', 0):.2f}%")

    elif args.opt:
        opt_key = args.opt
        if opt_key in OPTIMIZATIONS:
            opt = OPTIMIZATIONS[opt_key]
            print(f"Running {opt_key}: {opt['name']}...")
            result = run_backtest_with_params(extra_params=opt['params'], label=opt_key)
            print(f"{opt_key}: Sharpe={result.get('sharpe_ratio', 0):.4f}, "
                  f"Return={result.get('total_return_pct', 0):.2f}%, "
                  f"Drawdown={result.get('max_drawdown_pct', 0):.2f}%")
        else:
            print(f"Unknown optimization: {opt_key}")

    elif args.all:
        # Run baseline first
        print("Running baseline backtest...")
        baseline = run_backtest_with_params(label='baseline')
        baseline_sharpe = baseline.get('sharpe_ratio', 0)
        print(f"Baseline: Sharpe={baseline_sharpe:.4f}, Return={baseline.get('total_return_pct', 0):.2f}%")

        # Run all optimizations
        results = {}
        for opt_key, opt in OPTIMIZATIONS.items():
            print(f"\nRunning {opt_key}: {opt['name']}...")
            try:
                result = run_backtest_with_params(extra_params=opt['params'], label=opt_key)
                sharpe = result.get('sharpe_ratio', 0)
                improvement = (sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
                results[opt_key] = {
                    'name': opt['name'],
                    'sharpe': sharpe,
                    'improvement_pct': improvement,
                    'return_pct': result.get('total_return_pct', 0),
                    'drawdown_pct': result.get('max_drawdown_pct', 0),
                }
                verdict = 'EFFECTIVE' if improvement >= 5 else 'INEFFECTIVE'
                print(f"  Sharpe={sharpe:.4f}, Improvement={improvement:+.1f}%, Verdict={verdict}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results[opt_key] = {'name': opt['name'], 'error': str(e)}

        # Summary
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        for opt_key, r in results.items():
            if 'error' in r:
                print(f"  {opt_key} ({r['name']}): ERROR - {r['error']}")
            else:
                verdict = 'EFFECTIVE' if r['improvement_pct'] >= 5 else 'INEFFECTIVE'
                print(f"  {opt_key} ({r['name']}): Sharpe={r['sharpe']:.4f}, "
                      f"Improvement={r['improvement_pct']:+.1f}%, Verdict={verdict}")

    elif args.review:
        opt_key = args.review
        if opt_key in OPTIMIZATIONS:
            opt = OPTIMIZATIONS[opt_key]
            print(f"Running review for {opt_key}: {opt['name']}...")

            # 1. Out-of-sample test
            print("\n1. Out-of-sample test...")
            oos_result = run_out_of_sample_test(STRATEGY_NAME, opt['params'], opt_key)
            print(f"  IS Sharpe: {oos_result['in_sample_sharpe']:.4f}")
            print(f"  OOS Sharpe: {oos_result['out_sample_sharpe']:.4f}")
            print(f"  IS Improvement: {oos_result['is_improvement_pct']:+.1f}%")
            print(f"  OOS Improvement: {oos_result['oos_improvement_pct']:+.1f}%")
            print(f"  Decay Ratio: {oos_result['decay_ratio']:.4f}")

            # 2. Parameter sensitivity
            print("\n2. Parameter sensitivity test...")
            param_name = list(opt['params'].keys())[0]
            param_value = opt['params'][param_name]
            sens_result = run_parameter_sensitivity_test(STRATEGY_NAME, param_name, param_value, opt_key)
            print(f"  Base Sharpe: {sens_result['base_sharpe']:.4f}")
            print(f"  Sensitivity Ratio: {sens_result['sensitivity_ratio']:.4f}")
            for pk, pv in sens_result['perturbation_results'].items():
                print(f"    {pk}: param={pv['param_value']}, sharpe={pv['sharpe_ratio']:.4f}")

            # 3. Temporal stability
            print("\n3. Temporal stability test...")
            temporal_result = run_temporal_stability_test(STRATEGY_NAME, opt['params'], opt_key)
            print(f"  Consistency Ratio: {temporal_result['consistency_ratio']:.2f}")
            for yr in temporal_result['yearly_results']:
                print(f"    {yr['year']}: opt={yr['opt_sharpe']:.4f}, base={yr['base_sharpe']:.4f}, "
                      f"improvement={yr['improvement']:+.4f}")

            # Save review summary
            review = {
                'oos': oos_result,
                'sensitivity': sens_result,
                'temporal': temporal_result,
            }
            review_file = os.path.join(RESULTS_DIR, f'{opt_key}_review.json')
            with open(review_file, 'w', encoding='utf-8') as f:
                json.dump(review, f, indent=2, ensure_ascii=False)
            print(f"\nReview saved to {review_file}")

    elif args.combined:
        import json as json_mod
        combined_params = json_mod.loads(args.combined)
        print(f"Running combined optimization with params: {combined_params}")
        result = run_backtest_with_params(extra_params=combined_params, label='combined')
        print(f"Combined: Sharpe={result.get('sharpe_ratio', 0):.4f}, "
              f"Return={result.get('total_return_pct', 0):.2f}%, "
              f"Drawdown={result.get('max_drawdown_pct', 0):.2f}%")
