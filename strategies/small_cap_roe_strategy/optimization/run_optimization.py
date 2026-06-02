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
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'small_cap_roe'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'
POOL = '中证全指'


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool=POOL, start_date=TRAIN_START, end_date=TRAIN_END):
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


def run_out_of_sample_test(strategy_name, extra_params, label):
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
                                 full_start, full_end):
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


OPTIMIZATIONS = [
    ('baseline', '基线策略', None),
    ('opt01_volatility_filter', '波动率过滤(4%)', {'max_volatility': 0.04}),
    ('opt02_stop_loss', '止损机制(15%)', {'stop_loss_pct': 0.15}),
    ('opt03_industry_diversify', '行业分散(同行业<=2)', {'max_same_industry': 2}),
    ('opt04_min_market_cap', '市值下限(10亿)', {'min_market_cap': 1e9}),
    ('opt05_keep_existing', '换手率控制', {'keep_existing': True}),
    ('opt06_momentum_confirm', '动量确认(20日)', {'momentum_period': 20}),
    ('opt07_biweekly_rebalance', '双周调仓', {'rebalance_freq': 'biweekly'}),
    ('opt08_drawdown_control', '回撤控制(20%)', {'max_drawdown_limit': 0.20}),
    ('opt09_debt_ratio', '负债率过滤(60%)', {'max_debt_ratio': 0.60}),
    ('opt10_combined_vol_stop', '波动率+止损组合', {'max_volatility': 0.04, 'stop_loss_pct': 0.15}),
]

OPTIMIZATIONS_R2 = [
    ('r2_opt01_stop_loss_10', '止损机制(10%)', {'stop_loss_pct': 0.10}),
    ('r2_opt02_stop_loss_20', '止损机制(20%)', {'stop_loss_pct': 0.20}),
    ('r2_opt03_vol_filter_05', '波动率过滤(5%)', {'max_volatility': 0.05}),
    ('r2_opt04_vol_filter_06', '波动率过滤(6%)', {'max_volatility': 0.06}),
    ('r2_opt05_drawdown_30', '回撤控制(30%)', {'max_drawdown_limit': 0.30}),
    ('r2_opt06_debt_ratio_50', '负债率过滤(50%)', {'max_debt_ratio': 0.50}),
    ('r2_opt07_stop_vol_10_05', '止损10%+波动率5%', {'stop_loss_pct': 0.10, 'max_volatility': 0.05}),
    ('r2_opt08_stop_draw_10_30', '止损10%+回撤30%', {'stop_loss_pct': 0.10, 'max_drawdown_limit': 0.30}),
    ('r2_opt09_vol_draw_05_30', '波动率5%+回撤30%', {'max_volatility': 0.05, 'max_drawdown_limit': 0.30}),
    ('r2_opt10_all_3', '止损10%+波动5%+回撤30%', {'stop_loss_pct': 0.10, 'max_volatility': 0.05, 'max_drawdown_limit': 0.30}),
]


OPTIMIZATIONS_R3 = [
    ('r3_opt02_debt_70', '负债率过滤(70%)', {'max_debt_ratio': 0.70}),
    ('r3_opt03_debt_80', '负债率过滤(80%)', {'max_debt_ratio': 0.80}),
    ('r3_opt04_debt_90', '负债率过滤(90%)', {'max_debt_ratio': 0.90}),
    ('r3_opt05_stop10_vol06', '止损10%+波动6%', {'stop_loss_pct': 0.10, 'max_volatility': 0.06}),
    ('r3_opt06_stop10_debt70', '止损10%+负债率70%', {'stop_loss_pct': 0.10, 'max_debt_ratio': 0.70}),
    ('r3_opt07_vol06_debt70', '波动6%+负债率70%', {'max_volatility': 0.06, 'max_debt_ratio': 0.70}),
    ('r3_opt08_stop10_vol06_debt70', '止损10%+波动6%+负债率70%', {'stop_loss_pct': 0.10, 'max_volatility': 0.06, 'max_debt_ratio': 0.70}),
]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'all', 'oos', 'sensitivity', 'temporal', 'combined', 'r3'], default='all')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--param-name', type=str, default=None)
    parser.add_argument('--param-value', type=float, default=None)
    args = parser.parse_args()

    if args.mode == 'all':
        print("=" * 60)
        print("运行所有单项优化回测")
        print("=" * 60)
        all_opts = OPTIMIZATIONS + OPTIMIZATIONS_R2
        for label, name, params in all_opts:
            print(f"\n>>> 运行: {name} ({label})")
            try:
                result = run_backtest_with_params(
                    strategy_name=STRATEGY_NAME,
                    extra_params=params,
                    label=label)
                sharpe = result.get('sharpe_ratio', 0)
                total_ret = result.get('total_return_pct', 0)
                max_dd = result.get('max_drawdown_pct', 0)
                print(f"    夏普: {sharpe:.4f}, 总收益: {total_ret:.2f}%, 最大回撤: {max_dd:.2f}%")
            except Exception as e:
                print(f"    错误: {e}")
                traceback.print_exc()

    elif args.mode == 'r3':
        print("=" * 60)
        print("运行第三轮优化（修复后）")
        print("=" * 60)
        for label, name, params in OPTIMIZATIONS_R3:
            print(f"\n>>> 运行: {name} ({label})")
            try:
                result = run_backtest_with_params(
                    strategy_name=STRATEGY_NAME,
                    extra_params=params,
                    label=label)
                sharpe = result.get('sharpe_ratio', 0)
                total_ret = result.get('total_return_pct', 0)
                max_dd = result.get('max_drawdown_pct', 0)
                print(f"    夏普: {sharpe:.4f}, 总收益: {total_ret:.2f}%, 最大回撤: {max_dd:.2f}%")
            except Exception as e:
                print(f"    错误: {e}")
                traceback.print_exc()

    elif args.mode == 'single':
        if args.label is None:
            print("请指定 --label")
            sys.exit(1)
        opt = next((o for o in OPTIMIZATIONS + OPTIMIZATIONS_R2 if o[0] == args.label), None)
        if opt is None:
            print(f"未找到优化: {args.label}")
            sys.exit(1)
        label, name, params = opt
        result = run_backtest_with_params(
            strategy_name=STRATEGY_NAME,
            extra_params=params,
            label=label)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'oos':
        if args.label is None:
            print("请指定 --label")
            sys.exit(1)
        opt = next((o for o in OPTIMIZATIONS if o[0] == args.label), None)
        if opt is None:
            print(f"未找到优化: {args.label}")
            sys.exit(1)
        label, name, params = opt
        result = run_out_of_sample_test(STRATEGY_NAME, params, label)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'sensitivity':
        if args.label is None or args.param_name is None or args.param_value is None:
            print("请指定 --label, --param-name, --param-value")
            sys.exit(1)
        result = run_parameter_sensitivity_test(
            STRATEGY_NAME, args.param_name, args.param_value, args.label)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'temporal':
        if args.label is None:
            print("请指定 --label")
            sys.exit(1)
        opt = next((o for o in OPTIMIZATIONS if o[0] == args.label), None)
        if opt is None:
            print(f"未找到优化: {args.label}")
            sys.exit(1)
        label, name, params = opt
        result = run_temporal_stability_test(
            STRATEGY_NAME, params, label, TRAIN_START, TRAIN_END)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'combined':
        if args.label is None:
            print("请指定 --label (逗号分隔的优化标签)")
            sys.exit(1)
        labels = args.label.split(',')
        combined_params = {}
        for lbl in labels:
            opt = next((o for o in OPTIMIZATIONS if o[0] == lbl), None)
            if opt and opt[2]:
                combined_params.update(opt[2])
        result = run_backtest_with_params(
            strategy_name=STRATEGY_NAME,
            extra_params=combined_params,
            label=f"combined_{'_'.join(labels)}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
