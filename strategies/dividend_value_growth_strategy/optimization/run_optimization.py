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

STRATEGY_NAME = 'dividend_value_growth'
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

    try:
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
        else:
            metrics['error'] = 'No result'
    except Exception as e:
        metrics = {'error': str(e), 'traceback': traceback.format_exc()}

    metrics['label'] = label
    metrics['extra_params'] = extra_params
    metrics['timestamp'] = datetime.datetime.now().isoformat()

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    sharpe = metrics.get('sharpe_ratio', 0)
    ret = metrics.get('total_return_pct', 0)
    dd = metrics.get('max_drawdown_pct', 0)
    err = metrics.get('error')
    if err:
        print(f"  {label}: ERROR - {err[:80]}")
    else:
        print(f"  {label}: Sharpe={sharpe:.4f}, Return={ret:.2f}%, DD={dd:.2f}%")
    return metrics


# ================================================================
# 第二轮优化方案 - 中证全指股票池
# ================================================================
# 基线: Sharpe=1.4421, Return=285.89%, DD=-17.72% (中证全指 2020-2024)
# 第一轮教训: 硬阈值收紧无效，动量确认有害，月度调仓最优

ROUND2_OPTIMIZATIONS = [
    # B1: 改变排序逻辑 - 按PEG升序选股（低PEG优先）
    ('r2_b1_sort_peg', 'B1: PEG排序选股', {'sort_by_peg': True}),

    # C1+C2: 增加财务维度 - 现金流+负债率联合过滤
    ('r2_c1c2_ocf_debt', 'C1+C2: 现金流+负债率', {'min_ocf_ratio': 0.5, 'max_debt_ratio': 70}),

    # D2: 调仓逻辑 - 持仓续留优先（减少不必要换手）
    ('r2_d2_hold_priority', 'D2: 持仓续留优先', {'hold_priority': True}),

    # E1+E2: 主动卖出 - 估值过高/股息率衰减时卖出
    ('r2_e1e2_sell_signals', 'E1+E2: 主动卖出信号', {'sell_max_pe': 40, 'sell_min_dividend_yield': 0.015}),

    # G1: 综合评分排序 - 多因子加权替代硬阈值排序
    ('r2_g1_composite_score', 'G1: 综合评分排序', {'use_composite_score': True}),

    # H1: 分红连续性 - 要求至少3年分红历史
    ('r2_h1_dividend_years', 'H1: 分红连续性(3年)', {'min_dividend_years': 3}),

    # 组合1: G1 + E1E2 (评分排序 + 主动卖出)
    ('r2_combo_ge', '组合: 评分+卖出', {'use_composite_score': True, 'sell_max_pe': 40, 'sell_min_dividend_yield': 0.015}),

    # 组合2: G1 + H1 + D2 (评分排序 + 分红质量 + 持仓续留)
    ('r2_combo_ghd', '组合: 评分+分红质量+续留', {'use_composite_score': True, 'min_dividend_years': 3, 'hold_priority': True}),
]


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        label = sys.argv[1]
        if label == 'r2_baseline':
            print("Running baseline (中证全指 2020-2024)...")
            run_backtest_with_params(label='r2_baseline')
        else:
            # 查找第二轮优化
            for opt_label, opt_name, opt_params in ROUND2_OPTIMIZATIONS:
                if opt_label == label:
                    print(f"Running Round2: {opt_name}...")
                    run_backtest_with_params(extra_params=opt_params, label=opt_label)
                    break
            else:
                print(f"Unknown optimization: {label}")
    else:
        # 运行基线（中证全指 2020-2024）
        print("Running baseline (中证全指 2020-2024)...")
        baseline = run_backtest_with_params(label='r2_baseline')
        print()

        # 逐项运行第二轮优化
        for opt_label, opt_name, opt_params in ROUND2_OPTIMIZATIONS:
            print(f"Running {opt_name}...")
            run_backtest_with_params(extra_params=opt_params, label=opt_label)
            print()

        # 汇总
        print("\n" + "="*70)
        print("ROUND 2 OPTIMIZATION SUMMARY (中证全指)")
        print("="*70)
        baseline_sharpe = baseline.get('sharpe_ratio', 0)
        print(f"{'Label':<30} {'Sharpe':>8} {'Change':>8} {'Return':>10} {'DD':>8}")
        print("-"*70)
        print(f"{'r2_baseline':<30} {baseline_sharpe:>8.4f} {'---':>8} {baseline.get('total_return_pct',0):>9.2f}% {baseline.get('max_drawdown_pct',0):>7.2f}%")

        for opt_label, opt_name, opt_params in ROUND2_OPTIMIZATIONS:
            filepath = os.path.join(RESULTS_DIR, f'{opt_label}.json')
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'error' in data:
                    print(f"{opt_label:<30} {'ERROR':>8} {'---':>8} {'---':>10} {'---':>8}")
                    continue
                sharpe = data.get('sharpe_ratio', 0)
                change = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if baseline_sharpe else 0
                ret = data.get('total_return_pct', 0)
                dd = data.get('max_drawdown_pct', 0)
                verdict = "PASS" if change >= 5 else "FAIL"
                print(f"{opt_label:<30} {sharpe:>8.4f} {change:>+7.1f}% {ret:>9.2f}% {dd:>7.2f}% {verdict}")
