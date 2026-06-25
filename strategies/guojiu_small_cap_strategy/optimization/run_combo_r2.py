"""Round 2 组合优化测试"""
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

def run_backtest_with_params(extra_params=None, label='test', start_date=TRAIN_START, end_date=TRAIN_END):
    strategy_class = get_strategy(STRATEGY_NAME)
    default_kwargs = get_strategy_default_kwargs(STRATEGY_NAME)
    backtest_config = get_strategy_backtest_config(STRATEGY_NAME)
    config = dict(backtest_config)
    config['period'] = '1d'; config['start_date'] = start_date; config['end_date'] = end_date
    config.setdefault('benchmark', IndexConstituentManager.SECTOR_TO_INDEX.get(POOL, '000300.SH'))
    merged_kwargs = dict(default_kwargs)
    if extra_params: merged_kwargs.update(extra_params)
    api = BacktestAPI(); api.set_ai_mode(True); api.set_no_record(True)
    api.configure(**config); api.load_financial_data(sector=POOL)
    api.add_stock_selection_strategy(strategy_class, **merged_kwargs)
    api.run()
    result = api.get_result()
    metrics = {}
    if result:
        sr = result.sharpe_ratio(); dd = result.max_drawdown(); acc = result.account
        metrics['sharpe_ratio'] = sr; metrics['max_drawdown_pct'] = dd * 100
        metrics['total_return_pct'] = acc.rate * 100; metrics['final_value'] = acc.dynamic_rights
        if result.df is not None and len(result.df) > 0:
            days = len(result.df); years = days / 252
            annual_ret = (1 + acc.rate) ** (1 / years) - 1 if years > 0 else 0
            if isinstance(annual_ret, complex): annual_ret = annual_ret.real
            metrics['annual_return_pct'] = float(annual_ret) * 100; metrics['trading_days'] = days
        metrics['turnover'] = getattr(result, 'turnover', 0)
        metrics['label'] = label; metrics['extra_params'] = extra_params
    else:
        metrics['label'] = label; metrics['error'] = 'No result'
    with open(os.path.join(RESULTS_DIR, f'{label}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics

# 组合优化方案
COMBOS = [
    # 持仓8只 + 波动率3%
    ('r2_combo_vol3_stocks8', '组合: 波动率3%+持仓8只',
     {'max_volatility': 0.03, 'max_stocks': 8, 'ma_stock_nums': (5, 6, 8, 9, 10)}),
    # 持仓8只 + 换仓阈值10%
    ('r2_combo_switch10_stocks8', '组合: 换仓10%+持仓8只',
     {'switch_threshold': 0.10, 'max_stocks': 8, 'ma_stock_nums': (5, 6, 8, 9, 10)}),
    # 持仓8只 + 波动率3% + 换仓阈值10%
    ('r2_combo_all3', '组合: 波动率3%+换仓10%+持仓8只',
     {'max_volatility': 0.03, 'switch_threshold': 0.10, 'max_stocks': 8, 'ma_stock_nums': (5, 6, 8, 9, 10)}),
]

# 样本外验证方案 (持仓8只)
OOS_TESTS = [
    ('r2_opt08_max_stocks_8', '持仓8只',
     {'max_stocks': 8, 'ma_stock_nums': (5, 6, 8, 9, 10)}),
]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['combo', 'oos', 'all'], default='all')
    args = parser.parse_args()

    if args.mode in ('combo', 'all'):
        print("=" * 60)
        print("组合优化回测 (测试集)")
        print("=" * 60)
        base = json.load(open(os.path.join(RESULTS_DIR, 'r2_baseline.json'), 'r', encoding='utf-8'))
        base_sharpe = base.get('sharpe_ratio', 0)
        print(f"基线夏普: {base_sharpe:.4f}")
        for label, name, params in COMBOS:
            print(f"\n>>> {name}")
            try:
                r = run_backtest_with_params(extra_params=params, label=label)
                sharpe = r.get('sharpe_ratio', 0)
                change = (sharpe - base_sharpe) / base_sharpe * 100 if base_sharpe else 0
                print(f"    夏普: {sharpe:.4f} ({change:+.1f}%), 收益: {r.get('total_return_pct',0):.2f}%, 回撤: {r.get('max_drawdown_pct',0):.2f}%")
            except Exception as e:
                print(f"    错误: {e}"); traceback.print_exc()

    if args.mode in ('oos', 'all'):
        print("\n" + "=" * 60)
        print("样本外验证 (持仓8只)")
        print("=" * 60)
        for label, name, params in OOS_TESTS:
            print(f"\n>>> {name}")
            # 样本内
            is_result = run_backtest_with_params(extra_params=params, label=f'{label}_is')
            # 样本外
            oos_result = run_backtest_with_params(extra_params=params, label=f'{label}_oos',
                                                    start_date=VALID_START, end_date=VALID_END)
            # 基线样本内/外
            base_is = json.load(open(os.path.join(RESULTS_DIR, 'r2_baseline.json'), 'r', encoding='utf-8'))
            base_oos_path = os.path.join(RESULTS_DIR, 'r2_baseline_oos.json')
            if os.path.exists(base_oos_path):
                base_oos = json.load(open(base_oos_path, 'r', encoding='utf-8'))
            else:
                base_oos = run_backtest_with_params(extra_params=None, label='r2_baseline_oos',
                                                     start_date=VALID_START, end_date=VALID_END)

            is_sharpe = is_result.get('sharpe_ratio', 0)
            oos_sharpe = oos_result.get('sharpe_ratio', 0)
            bis = base_is.get('sharpe_ratio', 0)
            bos = base_oos.get('sharpe_ratio', 0)
            is_imp = (is_sharpe - bis) / abs(bis) * 100 if bis else 0
            oos_imp = (oos_sharpe - bos) / abs(bos) * 100 if bos else 0
            decay = oos_imp / is_imp if is_imp != 0 else 0
            print(f"    样本内: 夏普 {is_sharpe:.4f} (基线 {bis:.4f}, {is_imp:+.1f}%)")
            print(f"    样本外: 夏普 {oos_sharpe:.4f} (基线 {bos:.4f}, {oos_imp:+.1f}%)")
            print(f"    衰减比: {decay:.2f}")
