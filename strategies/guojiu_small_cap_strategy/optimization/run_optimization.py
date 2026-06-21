import os
import sys
import json
import datetime
import traceback

# 策略位于 strategies_my 下，比 strategies/ 多一层
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'guojiu_small_cap'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 固定的测试集/验证集时间边界
TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'    # 测试集结束 = 验证集开始
VALID_START = '2024-04-28'  # 验证集开始
VALID_END = '2026-04-28'    # 验证集结束
POOL = '中小综指'


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'all', 'single', 'oos', 'sensitivity', 'temporal'], default='baseline')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--param-name', type=str, default=None)
    parser.add_argument('--param-value', type=float, default=None)
    args = parser.parse_args()

    # 10项单项优化配置
    OPTIMIZATIONS = [
        ('opt01_volatility_filter', '波动率过滤(5%)', {'max_volatility': 0.05}),
        ('opt02_industry_diversify', '行业分散(同行业<=2)', {'max_same_industry': 2}),
        ('opt03_debt_ratio', '负债率过滤(60%)', {'max_debt_ratio': 0.60}),
        ('opt04_roe_filter', 'ROE过滤(8%)', {'min_roe': 0.08}),
        ('opt05_keep_existing', '换手率控制(保留已持有)', {'keep_existing': True}),
        ('opt06_biweekly_rebalance', '双周调仓', {'rebalance_freq': 'biweekly'}),
        ('opt07_strict_stoploss', '更严格止损(7%)', {'stoploss_limit': 0.07}),
        ('opt08_loose_stoploss', '更宽松止损(12%)', {'stoploss_limit': 0.12}),
        ('opt09_more_stocks', '增加持仓数量(6只)', {'max_stocks': 6}),
        ('opt10_revenue_growth', '营收增长过滤(正增长)', {'min_revenue_growth': 0.0}),
    ]

    if args.mode == 'baseline':
        print("=" * 60)
        print("运行基线回测 (测试集 2020-04-28 ~ 2024-04-28)")
        print("=" * 60)
        result = run_backtest_with_params(label='baseline')
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'all':
        print("=" * 60)
        print("运行所有单项优化回测 (测试集 2020-04-28 ~ 2024-04-28)")
        print("=" * 60)
        for label, name, params in OPTIMIZATIONS:
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
        opt = next((o for o in OPTIMIZATIONS if o[0] == args.label), None)
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
