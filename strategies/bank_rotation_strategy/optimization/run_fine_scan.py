"""精细阈值扫描 - 修正后slippage=0.0007"""
import sys, os, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir

STRATEGY_NAME = 'bank_rotation'
SLIPPAGE = 0.0007
COMMISSION = 0.0002
TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

def run_bt(switch_threshold, spread_threshold, start, end):
    strategy_class = get_strategy(STRATEGY_NAME)
    default_kwargs = get_strategy_default_kwargs(STRATEGY_NAME)
    backtest_config = get_strategy_backtest_config(STRATEGY_NAME)

    config = dict(backtest_config)
    config['period'] = '1m'
    config['start_date'] = start
    config['end_date'] = end
    config['slippage'] = SLIPPAGE
    config.setdefault('benchmark', '000300.SH')

    extra = {'switch_threshold': switch_threshold}
    if spread_threshold is not None:
        extra['spread_threshold'] = spread_threshold
    kwargs = {**default_kwargs, **extra}

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(**config)
    api.add_strategy(strategy_class, **kwargs)
    results = api.run()

    result = api.get_result()
    if result:
        acc = result.account
        return {
            'sharpe_ratio': result.sharpe_ratio(),
            'total_return_pct': acc.rate * 100,
            'max_drawdown_pct': result.max_drawdown() * 100,
        }
    return {'sharpe_ratio': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0}

print('=' * 90)
print(f'精细阈值扫描 | slippage={SLIPPAGE} | commission={COMMISSION}')
print('=' * 90)
print(f'{"阈值":<10} {"测试集夏普":>10} {"测试集收益":>10} {"测试集回撤":>10} {"验证集夏普":>10} {"验证集收益":>10} {"验证集回撤":>10} {"综合评分":>10}')
print('-' * 90)

results = []
for st in [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]:
    for sp in [0.003, 0.004, 0.005]:
        is_r = run_bt(st, sp, TRAIN_START, TRAIN_END)
        oos_r = run_bt(st, sp, VALID_START, VALID_END)
        is_sharpe = is_r.get('sharpe_ratio', 0)
        oos_sharpe = oos_r.get('sharpe_ratio', 0)
        # 综合评分：样本内权重0.4 + 样本外权重0.6（更重视样本外）
        score = is_sharpe * 0.4 + oos_sharpe * 0.6
        results.append({
            'st': st, 'sp': sp,
            'is_sharpe': is_sharpe, 'is_ret': is_r.get('total_return_pct', 0), 'is_dd': is_r.get('max_drawdown_pct', 0),
            'oos_sharpe': oos_sharpe, 'oos_ret': oos_r.get('total_return_pct', 0), 'oos_dd': oos_r.get('max_drawdown_pct', 0),
            'score': score
        })
        print(f'st={st:.3f},sp={sp:.3f} {is_sharpe:>10.4f} {is_r.get("total_return_pct",0):>10.2f}% {is_r.get("max_drawdown_pct",0):>10.2f}% {oos_sharpe:>10.4f} {oos_r.get("total_return_pct",0):>10.2f}% {oos_r.get("max_drawdown_pct",0):>10.2f}% {score:>10.4f}')

print()
print('=' * 90)
print('按综合评分排序 (样本内0.4 + 样本外0.6):')
print('=' * 90)
for r in sorted(results, key=lambda x: x['score'], reverse=True)[:10]:
    print(f'  st={r["st"]:.3f}, sp={r["sp"]:.3f} | IS夏普={r["is_sharpe"]:.4f} OOS夏普={r["oos_sharpe"]:.4f} | IS收益={r["is_ret"]:.2f}% OOS收益={r["oos_ret"]:.2f}% | 评分={r["score"]:.4f}')

# 保存结果
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results', 'fine_scan_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\n结果已保存: {out_path}')
