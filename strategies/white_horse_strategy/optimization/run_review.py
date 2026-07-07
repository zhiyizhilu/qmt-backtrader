import os
import sys
import json
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'white_horse'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'
POOL = '沪深300'


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool=POOL, start_date=TRAIN_START, end_date=TRAIN_END):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1d'
    config['start_date'] = start_date
    config['end_date'] = end_date
    config['data_lookback_days'] = 400
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


def run_review():
    """阶段五：硬逻辑与过度拟合审查"""
    # 5.2.1 样本外验证
    print("\n=== 5.2.1 样本外验证 ===")
    # 测试集（样本内）
    in_sample = run_backtest_with_params(
        extra_params={'warm_min_profit_yoy': 5},
        label='opt09_is', start_date=TRAIN_START, end_date=TRAIN_END)

    # 验证集（样本外）
    out_sample = run_backtest_with_params(
        extra_params={'warm_min_profit_yoy': 5},
        label='opt09_oos', start_date=VALID_START, end_date=VALID_END)

    # 基线对比
    baseline_is = run_backtest_with_params(
        label='baseline_is_review', start_date=TRAIN_START, end_date=TRAIN_END)

    baseline_oos = run_backtest_with_params(
        label='baseline_oos_review', start_date=VALID_START, end_date=VALID_END)

    is_sharpe = in_sample.get('sharpe_ratio', 0)
    oos_sharpe = out_sample.get('sharpe_ratio', 0)
    baseline_is_sharpe = baseline_is.get('sharpe_ratio', 0)
    baseline_oos_sharpe = baseline_oos.get('sharpe_ratio', 0)

    is_improvement = (is_sharpe - baseline_is_sharpe) / abs(baseline_is_sharpe) * 100 if baseline_is_sharpe != 0 else 0
    oos_improvement = (oos_sharpe - baseline_oos_sharpe) / abs(baseline_oos_sharpe) * 100 if baseline_oos_sharpe != 0 else 0

    decay_ratio = oos_improvement / is_improvement if is_improvement != 0 else 0

    print(f"\n样本内(IS) 夏普: {is_sharpe:.4f} (基线: {baseline_is_sharpe:.4f}, 改进: {is_improvement:+.1f}%)")
    print(f"样本外(OOS) 夏普: {oos_sharpe:.4f} (基线: {baseline_oos_sharpe:.4f}, 改进: {oos_improvement:+.1f}%)")
    print(f"衰减比(OOS/IS): {decay_ratio:.2f}")

    oos_result = {
        'train_period': f'{TRAIN_START} ~ {TRAIN_END}',
        'valid_period': f'{VALID_START} ~ {VALID_END}',
        'in_sample_sharpe': is_sharpe,
        'out_sample_sharpe': oos_sharpe,
        'baseline_is_sharpe': baseline_is_sharpe,
        'baseline_oos_sharpe': baseline_oos_sharpe,
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': decay_ratio,
    }

    # 5.2.2 参数敏感性分析
    print("\n=== 5.2.2 参数敏感性分析 ===")
    perturbations = [3, 4, 6, 7, 10]  # warm_min_profit_yoy 的扰动值
    sensitivity_results = {}
    for val in perturbations:
        result = run_backtest_with_params(
            extra_params={'warm_min_profit_yoy': val},
            label=f'opt09_sens_yoy{val}')
        sensitivity_results[f'yoy={val}'] = {
            'param_value': val,
            'sharpe_ratio': result.get('sharpe_ratio', 0),
        }

    # 也在测试集区间上重新跑 yoy=5 的基线
    base_5 = run_backtest_with_params(
        extra_params={'warm_min_profit_yoy': 5},
        label='opt09_sens_base')
    base_sharpe = base_5.get('sharpe_ratio', 0)

    sharpe_values = [r['sharpe_ratio'] for r in sensitivity_results.values()]
    sharpe_range = max(sharpe_values) - min(sharpe_values)
    sensitivity_ratio = sharpe_range / abs(base_sharpe) if base_sharpe != 0 else float('inf')

    print(f"\n基线(yoy=5) 夏普: {base_sharpe:.4f}")
    for key, r in sensitivity_results.items():
        print(f"  {key}: 夏普={r['sharpe_ratio']:.4f}")
    print(f"夏普范围: {sharpe_range:.4f}")
    print(f"敏感度比率: {sensitivity_ratio:.4f}")

    # 5.2.3 时间分段稳定性
    print("\n=== 5.2.3 时间分段稳定性 ===")
    yearly_results = []
    for year in range(2020, 2025):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        if year_start < TRAIN_START:
            year_start = TRAIN_START
        if year_end > TRAIN_END:
            year_end = TRAIN_END

        opt_result = run_backtest_with_params(
            extra_params={'warm_min_profit_yoy': 5},
            label=f'opt09_{year}', start_date=year_start, end_date=year_end)

        base_result = run_backtest_with_params(
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
        print(f"  {year}: 优化夏普={opt_sharpe:.4f}, 基线夏普={base_sharpe:.4f}, 改进={improvement:+.4f} {'OK' if improvement > 0 else 'FAIL'}")

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency_ratio = positive_years / total_years if total_years > 0 else 0
    print(f"\n一致性比率: {positive_years}/{total_years} = {consistency_ratio:.2f}")

    # 汇总审查结果
    review_summary = {
        'opt09_warm_profit_yoy_5': {
            'logic_rating': 'B',
            'logic_check': {
                '因果链': '通过',
                '经济合理性': '通过',
                '逻辑独立性': '通过',
                '极端场景': '中等',
                '可解释性': '通过',
            },
            'oos_test': oos_result,
            'sensitivity': {
                'base_sharpe': base_sharpe,
                'results': sensitivity_results,
                'range': sharpe_range,
                'sensitivity_ratio': sensitivity_ratio,
            },
            'temporal_stability': {
                'yearly_results': yearly_results,
                'positive_years': positive_years,
                'total_years': total_years,
                'consistency_ratio': consistency_ratio,
            },
        }
    }

    # 综合审查结论
    oos_decay = decay_ratio
    sensitivity = sensitivity_ratio
    consistency = consistency_ratio

    if oos_decay >= 0.5 and sensitivity < 0.3 and consistency >= 0.7:
        conclusion = 'PASS: 强力通过，优先组合'
    elif oos_decay >= 0.2 and sensitivity < 0.6 and consistency >= 0.5:
        conclusion = 'CONDITIONAL: 有条件通过，组合时降低权重'
    else:
        conclusion = 'FAIL: 不通过，放弃该优化'

    print(f"\n综合审查结论: {conclusion}")
    print(f"  硬逻辑评级: B")
    print(f"  样本外衰减比: {oos_decay:.2f}")
    print(f"  参数敏感度: {sensitivity:.4f}")
    print(f"  时间稳定性: {consistency:.2f}")

    review_summary['opt09_warm_profit_yoy_5']['conclusion'] = conclusion
    review_summary['opt09_warm_profit_yoy_5']['final_rating'] = {
        'logic_rating': 'B',
        'decay_ratio': oos_decay,
        'sensitivity_ratio': sensitivity,
        'consistency_ratio': consistency,
    }

    # 保存审查结果
    review_file = os.path.join(RESULTS_DIR, 'review_summary.json')
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump(review_summary, f, indent=2, ensure_ascii=False)

    return review_summary


if __name__ == '__main__':
    run_review()
