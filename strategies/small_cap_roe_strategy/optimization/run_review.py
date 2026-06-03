"""阶段五：硬逻辑与过度拟合审查脚本
对R5最佳方案 (max_stocks=20, max_volatility=0.06) 进行三项检测：
1. 样本外验证（2024-04-28 ~ 2026-04-28）
2. 参数敏感性分析（max_stocks ±20%, max_volatility ±20%）
3. 时间分段稳定性（按年分段）
"""
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

# 最佳方案参数
BEST_PARAMS = {'max_stocks': 20, 'max_volatility': 0.06}


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


def test_1_out_of_sample():
    """样本外验证"""
    print("=" * 60)
    print("检测1: 样本外验证")
    print("=" * 60)

    # 测试集 - 优化方案
    print("\n>>> 测试集 - 优化方案 (stocks20+vol6%)")
    is_opt = run_backtest_with_params(
        extra_params=BEST_PARAMS, label='review_is_opt',
        start_date=TRAIN_START, end_date=TRAIN_END)

    # 验证集 - 优化方案
    print("\n>>> 验证集 - 优化方案 (stocks20+vol6%)")
    oos_opt = run_backtest_with_params(
        extra_params=BEST_PARAMS, label='review_oos_opt',
        start_date=VALID_START, end_date=VALID_END)

    # 测试集 - 基线
    print("\n>>> 测试集 - 基线")
    is_base = run_backtest_with_params(
        extra_params=None, label='review_is_base',
        start_date=TRAIN_START, end_date=TRAIN_END)

    # 验证集 - 基线
    print("\n>>> 验证集 - 基线")
    oos_base = run_backtest_with_params(
        extra_params=None, label='review_oos_base',
        start_date=VALID_START, end_date=VALID_END)

    is_sharpe_opt = is_opt.get('sharpe_ratio', 0)
    oos_sharpe_opt = oos_opt.get('sharpe_ratio', 0)
    is_sharpe_base = is_base.get('sharpe_ratio', 0)
    oos_sharpe_base = oos_base.get('sharpe_ratio', 0)

    is_improvement = (is_sharpe_opt - is_sharpe_base) / abs(is_sharpe_base) * 100
    oos_improvement = (oos_sharpe_opt - oos_sharpe_base) / abs(oos_sharpe_base) * 100
    decay_ratio = oos_improvement / is_improvement if is_improvement != 0 else 0

    result = {
        'test': 'out_of_sample',
        'is_sharpe_opt': is_sharpe_opt,
        'oos_sharpe_opt': oos_sharpe_opt,
        'is_sharpe_base': is_sharpe_base,
        'oos_sharpe_base': oos_sharpe_base,
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': decay_ratio,
        'verdict': 'PASS' if decay_ratio >= 0.5 else ('CONDITIONAL' if decay_ratio >= 0.2 else 'FAIL'),
    }

    print(f"\n  测试集夏普: 优化={is_sharpe_opt:.4f}, 基线={is_sharpe_base:.4f}, 提升={is_improvement:+.1f}%")
    print(f"  验证集夏普: 优化={oos_sharpe_opt:.4f}, 基线={oos_sharpe_base:.4f}, 提升={oos_improvement:+.1f}%")
    print(f"  衰减比: {decay_ratio:.2f}")
    print(f"  判定: {result['verdict']}")

    return result


def test_2_parameter_sensitivity():
    """参数敏感性分析"""
    print("\n" + "=" * 60)
    print("检测2: 参数敏感性分析")
    print("=" * 60)

    results = {}

    # max_stocks 敏感性
    print("\n--- max_stocks 敏感性 ---")
    stocks_values = [16, 18, 20, 22, 24]
    stocks_sharpes = {}
    for v in stocks_values:
        print(f"\n>>> max_stocks={v}")
        r = run_backtest_with_params(
            extra_params={'max_stocks': v, 'max_volatility': 0.06},
            label=f'review_sens_stocks_{v}')
        s = r.get('sharpe_ratio', 0)
        stocks_sharpes[v] = s
        print(f"    夏普: {s:.4f}")

    stocks_range = max(stocks_sharpes.values()) - min(stocks_sharpes.values())
    stocks_base = stocks_sharpes.get(20, 0)
    stocks_sensitivity = stocks_range / abs(stocks_base) if stocks_base != 0 else float('inf')

    results['max_stocks'] = {
        'values': stocks_sharpes,
        'range': stocks_range,
        'sensitivity_ratio': stocks_sensitivity,
        'verdict': 'PASS' if stocks_sensitivity < 0.3 else ('CONDITIONAL' if stocks_sensitivity < 0.6 else 'FAIL'),
    }
    print(f"\n  max_stocks 范围: {stocks_range:.4f}, 敏感度: {stocks_sensitivity:.2f}, 判定: {results['max_stocks']['verdict']}")

    # max_volatility 敏感性
    print("\n--- max_volatility 敏感性 ---")
    vol_values = [0.048, 0.054, 0.06, 0.066, 0.072]
    vol_sharpes = {}
    for v in vol_values:
        print(f"\n>>> max_volatility={v}")
        r = run_backtest_with_params(
            extra_params={'max_stocks': 20, 'max_volatility': v},
            label=f'review_sens_vol_{v:.3f}')
        s = r.get('sharpe_ratio', 0)
        vol_sharpes[v] = s
        print(f"    夏普: {s:.4f}")

    vol_range = max(vol_sharpes.values()) - min(vol_sharpes.values())
    vol_base = vol_sharpes.get(0.06, 0)
    vol_sensitivity = vol_range / abs(vol_base) if vol_base != 0 else float('inf')

    results['max_volatility'] = {
        'values': vol_sharpes,
        'range': vol_range,
        'sensitivity_ratio': vol_sensitivity,
        'verdict': 'PASS' if vol_sensitivity < 0.3 else ('CONDITIONAL' if vol_sensitivity < 0.6 else 'FAIL'),
    }
    print(f"\n  max_volatility 范围: {vol_range:.4f}, 敏感度: {vol_sensitivity:.2f}, 判定: {results['max_volatility']['verdict']}")

    return results


def test_3_temporal_stability():
    """时间分段稳定性"""
    print("\n" + "=" * 60)
    print("检测3: 时间分段稳定性")
    print("=" * 60)

    yearly_results = []
    for year in range(2020, 2025):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        if year_start < TRAIN_START:
            year_start = TRAIN_START
        if year_end > TRAIN_END:
            year_end = TRAIN_END

        print(f"\n>>> {year}年 - 优化方案")
        opt_result = run_backtest_with_params(
            extra_params=BEST_PARAMS,
            label=f'review_temp_opt_{year}',
            start_date=year_start, end_date=year_end)

        print(f">>> {year}年 - 基线")
        base_result = run_backtest_with_params(
            extra_params=None,
            label=f'review_temp_base_{year}',
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
        print(f"  {year}: 优化={opt_sharpe:.4f}, 基线={base_sharpe:.4f}, 差值={improvement:+.4f} ({'✓' if improvement > 0 else '✗'})")

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency_ratio = positive_years / total_years if total_years > 0 else 0

    result = {
        'test': 'temporal_stability',
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': consistency_ratio,
        'verdict': 'PASS' if consistency_ratio >= 0.7 else ('CONDITIONAL' if consistency_ratio >= 0.5 else 'FAIL'),
    }

    print(f"\n  正向年数: {positive_years}/{total_years}, 一致性: {consistency_ratio:.2f}, 判定: {result['verdict']}")

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['oos', 'sensitivity', 'temporal', 'all'], default='all')
    args = parser.parse_args()

    review_results = {}

    try:
        if args.test in ('oos', 'all'):
            review_results['out_of_sample'] = test_1_out_of_sample()
    except Exception as e:
        print(f"样本外验证错误: {e}")
        traceback.print_exc()
        review_results['out_of_sample'] = {'error': str(e)}

    try:
        if args.test in ('sensitivity', 'all'):
            review_results['parameter_sensitivity'] = test_2_parameter_sensitivity()
    except Exception as e:
        print(f"参数敏感性分析错误: {e}")
        traceback.print_exc()
        review_results['parameter_sensitivity'] = {'error': str(e)}

    try:
        if args.test in ('temporal', 'all'):
            review_results['temporal_stability'] = test_3_temporal_stability()
    except Exception as e:
        print(f"时间稳定性测试错误: {e}")
        traceback.print_exc()
        review_results['temporal_stability'] = {'error': str(e)}

    # 保存审查结果
    review_file = os.path.join(RESULTS_DIR, 'review_summary.json')
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump(review_results, f, indent=2, ensure_ascii=False)
    print(f"\n审查结果已保存: {review_file}")
