"""精简审查 - 只对最优优化做核心OOS+敏感性验证"""
import os, sys, json, datetime, statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config

STRATEGY_NAME = 'bank_rotation'
TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'


def run_bt(extra_params=None, start=TRAIN_START, end=TRAIN_END):
    strategy_class = get_strategy(STRATEGY_NAME)
    default_kwargs = get_strategy_default_kwargs(STRATEGY_NAME)
    backtest_config = get_strategy_backtest_config(STRATEGY_NAME)
    config = dict(backtest_config)
    config['period'] = '1m'
    config['start_date'] = start
    config['end_date'] = end
    config.setdefault('benchmark', '000300.SH')
    merged = dict(default_kwargs)
    if extra_params:
        merged.update(extra_params)
    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(**config)
    api.add_strategy(strategy_class, **merged)
    api.run()
    result = api.get_result()
    if result:
        acc = result.account
        sr = result.sharpe_ratio()
        dd = result.max_drawdown()
        days = len(result.df) if result.df is not None else 0
        years = days / 252 if days else 0
        ann = (1 + acc.rate) ** (1 / years) - 1 if years > 0 else 0
        if isinstance(ann, complex):
            ann = ann.real
        return {'sharpe': sr, 'return_pct': acc.rate * 100, 'drawdown_pct': dd * 100,
                'annual_pct': float(ann) * 100, 'final_value': acc.dynamic_rights, 'days': days}
    return {'sharpe': 0, 'error': True}


print("=" * 80)
print("精简审查: switch_threshold 优化")
print(f"训练期: {TRAIN_START} ~ {TRAIN_END}")
print(f"验证期: {VALID_START} ~ {VALID_END}")
print("=" * 80)

# Step 1: baseline IS + OOS
print("\n[1] Baseline 回测...")
base_is = run_bt(start=TRAIN_START, end=TRAIN_END)
base_oos = run_bt(start=VALID_START, end=VALID_END)
print(f"  Baseline IS:  Sharpe={base_is['sharpe']:.4f}, Return={base_is['return_pct']:.2f}%, DD={base_is['drawdown_pct']:.2f}%")
print(f"  Baseline OOS: Sharpe={base_oos['sharpe']:.4f}, Return={base_oos['return_pct']:.2f}%, DD={base_oos['drawdown_pct']:.2f}%")

# Step 2: switch_threshold sweep
print("\n[2] switch_threshold 扫描 (IS)...")
sweep_results = {}
for st in [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]:
    r = run_bt(extra_params={'switch_threshold': st}, start=TRAIN_START, end=TRAIN_END)
    sweep_results[st] = r
    delta = (r['sharpe'] - base_is['sharpe']) / abs(base_is['sharpe']) * 100 if base_is['sharpe'] else 0
    print(f"  st={st:.4f}: Sharpe={r['sharpe']:.4f}, Return={r['return_pct']:.2f}%, DD={r['drawdown_pct']:.2f}%, delta={delta:+.1f}%")

# Step 3: OOS for best params
print("\n[3] 最优参数 OOS 验证...")
best_st = max(sweep_results, key=lambda x: sweep_results[x]['sharpe'])
print(f"  最优 switch_threshold (IS) = {best_st}")
oos_results = {}
for st in [0.002, 0.003, 0.004, 0.005]:
    r = run_bt(extra_params={'switch_threshold': st}, start=VALID_START, end=VALID_END)
    oos_results[st] = r
    is_sharpe = sweep_results[st]['sharpe']
    oos_sharpe = r['sharpe']
    is_imp = (is_sharpe - base_is['sharpe']) / abs(base_is['sharpe']) * 100
    oos_imp = (oos_sharpe - base_oos['sharpe']) / abs(base_oos['sharpe']) * 100
    decay = oos_imp / is_imp if is_imp != 0 else 0
    print(f"  st={st:.4f}: IS={is_sharpe:.4f}(+{is_imp:.1f}%), OOS={oos_sharpe:.4f}(+{oos_imp:.1f}%), decay={decay:.2f}")

# Step 4: Sensitivity around best
print(f"\n[4] 参数敏感性 ({best_st} 附近)...")
sens_results = {}
for delta_pct in [-40, -20, -10, 0, 10, 20, 40]:
    st_val = best_st * (1 + delta_pct / 100)
    r = run_bt(extra_params={'switch_threshold': st_val}, start=TRAIN_START, end=TRAIN_END)
    sens_results[delta_pct] = {'st': st_val, 'sharpe': r['sharpe']}
    print(f"  delta={delta_pct:+3d}%: st={st_val:.6f}, Sharpe={r['sharpe']:.4f}")

sharpe_vals = [v['sharpe'] for v in sens_results.values()]
sharpe_range = max(sharpe_vals) - min(sharpe_vals)
sens_ratio = sharpe_range / abs(sens_results[0]['sharpe']) if sens_results[0]['sharpe'] else float('inf')
print(f"  Sharpe范围={sharpe_range:.4f}, 敏感比={sens_ratio:.4f}")
sens_verdict = 'PASS' if sens_ratio < 0.5 else 'CAUTION' if sens_ratio < 1.0 else 'FAIL'
print(f"  敏感性判定: {sens_verdict}")

# Save
all_results = {
    'baseline_is': base_is, 'baseline_oos': base_oos,
    'sweep_is': {str(k): v for k, v in sweep_results.items()},
    'oos_results': {str(k): v for k, v in oos_results.items()},
    'best_switch_threshold': best_st,
    'sensitivity': sens_results,
    'sensitivity_ratio': sens_ratio,
    'sensitivity_verdict': sens_verdict,
}
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'optimization', 'review_results')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'switch_threshold_review.json')
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

print(f"\n结果已保存: {save_path}")
print("\n" + "=" * 80)
print("审查汇总")
print("=" * 80)
print(f"  最优 switch_threshold (IS): {best_st}")
print(f"  Baseline IS:  Sharpe={base_is['sharpe']:.4f}")
for st in [0.002, 0.003, 0.004]:
    is_s = sweep_results[st]['sharpe']
    oos_s = oos_results[st]['sharpe']
    is_imp = (is_s - base_is['sharpe']) / abs(base_is['sharpe']) * 100
    oos_imp = (oos_s - base_oos['sharpe']) / abs(base_oos['sharpe']) * 100
    print(f"  st={st}: IS={is_s:.4f}(+{is_imp:.1f}%), OOS={oos_s:.4f}(+{oos_imp:.1f}%)")
print(f"  敏感性判定: {sens_verdict} (ratio={sens_ratio:.4f})")
