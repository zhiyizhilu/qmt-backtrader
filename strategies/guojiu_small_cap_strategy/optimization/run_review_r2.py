"""Round 2 硬逻辑与过度拟合审查 - 参数敏感性和时间稳定性测试"""
import os, sys, json, datetime, traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'guojiu_small_cap'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
RESULTS_DIR = os.path.join(STRATEGY_DIR, 'optimization', 'optimization_results')
TRAIN_START = '2020-04-28'; TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'; VALID_END = '2026-04-28'
POOL = '中小综指'

# 持仓8只对应的 ma_stock_nums
STOCKS8_PARAMS = {'max_stocks': 8, 'ma_stock_nums': (5, 6, 8, 9, 10)}

def run_backtest(extra_params=None, label='test', start_date=TRAIN_START, end_date=TRAIN_END):
    strategy_class = get_strategy(STRATEGY_NAME)
    default_kwargs = get_strategy_default_kwargs(STRATEGY_NAME)
    backtest_config = get_strategy_backtest_config(STRATEGY_NAME)
    config = dict(backtest_config)
    config['period'] = '1d'; config['start_date'] = start_date; config['end_date'] = end_date
    config.setdefault('benchmark', IndexConstituentManager.SECTOR_TO_INDEX.get(POOL, '000300.SH'))
    merged = dict(default_kwargs)
    if extra_params: merged.update(extra_params)
    api = BacktestAPI(); api.set_ai_mode(True); api.set_no_record(True)
    api.configure(**config); api.load_financial_data(sector=POOL)
    api.add_stock_selection_strategy(strategy_class, **merged)
    api.run()
    result = api.get_result()
    m = {}
    if result:
        m['sharpe_ratio'] = result.sharpe_ratio()
        m['max_drawdown_pct'] = result.max_drawdown() * 100
        m['total_return_pct'] = result.account.rate * 100
        m['label'] = label; m['extra_params'] = extra_params
    else:
        m['label'] = label; m['error'] = 'No result'
    with open(os.path.join(RESULTS_DIR, f'{label}.json'), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return m

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sensitivity', 'temporal', 'all'], default='all')
    args = parser.parse_args()

    if args.mode in ('sensitivity', 'all'):
        print("=" * 60)
        print("参数敏感性测试: max_stocks=8")
        print("=" * 60)
        # 扰动值: 8的 ±10%, ±20% → 6, 7, 9, 10
        # 对应的 ma_stock_nums 需要调整
        perturbations = [
            (-0.20, 6, (4, 5, 6, 7, 8)),   # 8 * 0.8 = 6.4 → 6
            (-0.10, 7, (5, 6, 7, 8, 9)),   # 8 * 0.9 = 7.2 → 7
            (0.00,  8, (5, 6, 8, 9, 10)),  # 基准
            (+0.10, 9, (6, 7, 9, 10, 11)), # 8 * 1.1 = 8.8 → 9
            (+0.20, 10, (6, 7, 10, 11, 12)), # 8 * 1.2 = 9.6 → 10
        ]
        results = {}
        for delta, stocks, ma_nums in perturbations:
            label = f'r2_sensitivity_stocks{stocks}'
            print(f"\n>>> max_stocks={stocks} (delta={delta:+.0%})")
            params = {'max_stocks': stocks, 'ma_stock_nums': ma_nums}
            r = run_backtest(extra_params=params, label=label)
            sharpe = r.get('sharpe_ratio', 0)
            results[stocks] = sharpe
            print(f"    夏普: {sharpe:.4f}")

        base_sharpe = results.get(8, 0)
        sharpe_values = list(results.values())
        sharpe_range = max(sharpe_values) - min(sharpe_values)
        sens_ratio = sharpe_range / abs(base_sharpe) if base_sharpe else float('inf')
        print(f"\n基准夏普(max_stocks=8): {base_sharpe:.4f}")
        print(f"夏普范围: {sharpe_range:.4f}")
        print(f"敏感度比率: {sens_ratio:.4f}")
        if sens_ratio < 0.3:
            print("判定: ✅ 参数鲁棒 (< 0.3)")
        elif sens_ratio < 0.6:
            print("判定: ⚠️ 参数较敏感 (0.3~0.6)")
        else:
            print("判定: ❌ 参数高度敏感 (> 0.6)")

    if args.mode in ('temporal', 'all'):
        print("\n" + "=" * 60)
        print("时间稳定性测试: 持仓8只 vs 基线 (按年)")
        print("=" * 60)
        from datetime import datetime
        start_year = datetime.strptime(TRAIN_START, '%Y-%m-%d').year
        end_year = datetime.strptime(TRAIN_END, '%Y-%m-%d').year
        yearly = []
        for year in range(start_year, end_year + 1):
            ys = f'{year}-01-01'; ye = f'{year}-12-31'
            if ys < TRAIN_START: ys = TRAIN_START
            if ye > TRAIN_END: ye = TRAIN_END
            print(f"\n>>> {year}年")
            opt = run_backtest(extra_params=STOCKS8_PARAMS, label=f'r2_temporal_opt_{year}',
                               start_date=ys, end_date=ye)
            base = run_backtest(extra_params=None, label=f'r2_temporal_base_{year}',
                                start_date=ys, end_date=ye)
            os = opt.get('sharpe_ratio', 0); bs = base.get('sharpe_ratio', 0)
            imp = os - bs
            pos = imp > 0
            yearly.append({'year': year, 'opt': os, 'base': bs, 'imp': imp, 'positive': pos})
            print(f"    持仓8只: {os:.4f}, 基线: {bs:.4f}, 差值: {imp:+.4f} ({'✅' if pos else '❌'})")

        positive = sum(1 for y in yearly if y['positive'])
        total = len(yearly)
        ratio = positive / total if total else 0
        print(f"\n正改进年数: {positive}/{total}")
        print(f"一致性比率: {ratio:.2f}")
        if ratio >= 0.7:
            print("判定: ✅ 时间稳定 (>= 0.7)")
        elif ratio >= 0.5:
            print("判定: ⚠️ 部分年份无效 (0.5~0.7)")
        else:
            print("判定: ❌ 多数年份无效 (< 0.5)")
