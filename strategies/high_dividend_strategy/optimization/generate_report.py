import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')

SUMMARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_report.json')

def main():
    results = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
            results[data.get('label', fname)] = data

    baseline = results.get('baseline', {})
    baseline_sharpe = baseline.get('sharpe_ratio', 0)

    rows = []
    for label, data in sorted(results.items(), key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True):
        sharpe = data.get('sharpe_ratio', 0)
        total_ret = data.get('total_return_pct', 0)
        max_dd = data.get('max_drawdown_pct', 0)
        annual_ret = data.get('annual_return_pct', 0)
        change_pct = ((sharpe - baseline_sharpe) / baseline_sharpe * 100) if baseline_sharpe > 0 else 0
        effective = 'YES' if change_pct >= 5 else 'NO'
        rows.append({
            'label': label,
            'sharpe': sharpe,
            'total_return_pct': total_ret,
            'max_drawdown_pct': max_dd,
            'annual_return_pct': annual_ret,
            'sharpe_change_pct': round(change_pct, 2),
            'effective': effective,
        })

    report = {
        'baseline_sharpe': baseline_sharpe,
        'optimization_results': rows,
        'effective_optimizations': [r for r in rows if r['effective'] == 'YES'],
        'best_combined': results.get('opt12_combined_biweekly_15', {}),
        'conclusion': {
            'applied_changes': [
                'rebalance_freq: monthly -> biweekly',
                'max_stocks: 20 -> 15',
            ],
            'sharpe_improvement': f"{((results.get('opt12_combined_biweekly_15', {}).get('sharpe_ratio', 0) - baseline_sharpe) / baseline_sharpe * 100):.2f}%",
            'total_return_improvement': f"{(results.get('opt12_combined_biweekly_15', {}).get('total_return_pct', 0) - baseline.get('total_return_pct', 0)):.2f}%",
            'max_drawdown_improvement': f"{(results.get('opt12_combined_biweekly_15', {}).get('max_drawdown_pct', 0) - baseline.get('max_drawdown_pct', 0)):.2f}%",
        }
    }

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('=' * 80)
    print('高股息策略优化报告')
    print('=' * 80)
    print(f'\n基线夏普比率: {baseline_sharpe:.4f}')
    print(f'\n{"优化项":<35} {"夏普":>8} {"总收益%":>10} {"最大回撤%":>10} {"年化%":>8} {"夏普变化%":>10} {"有效"}')
    print('-' * 95)
    for r in rows:
        print(f'{r["label"]:<35} {r["sharpe"]:>8.4f} {r["total_return_pct"]:>10.2f} {r["max_drawdown_pct"]:>10.2f} {r["annual_return_pct"]:>8.2f} {r["sharpe_change_pct"]:>10.2f} {r["effective"]}')

    print(f'\n有效优化 (夏普提升>=5%):')
    for r in report['effective_optimizations']:
        print(f'  - {r["label"]}: 夏普 {r["sharpe"]:.4f} (提升 {r["sharpe_change_pct"]:.2f}%)')

    print(f'\n最优组合 (opt12_combined_biweekly_15):')
    best = report['best_combined']
    print(f'  夏普: {best.get("sharpe_ratio", 0):.4f}')
    print(f'  总收益: {best.get("total_return_pct", 0):.2f}%')
    print(f'  最大回撤: {best.get("max_drawdown_pct", 0):.2f}%')
    print(f'  年化收益: {best.get("annual_return_pct", 0):.2f}%')

    print(f'\n结论:')
    for k, v in report['conclusion'].items():
        print(f'  {k}: {v}')

    print(f'\n报告已保存至: {SUMMARY_FILE}')

if __name__ == '__main__':
    main()
