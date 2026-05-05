import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


optimization_labels = {
    'baseline': '基线',
    'opt01_volatility_filter': '波动率过滤',
    'opt02_volume_confirm': '成交量确认',
    'opt03_multi_factor_score': '多因子评分',
    'opt04_biweekly_rebalance': '双周调仓',
    'opt05_relative_momentum': '相对动量',
    'opt06_industry_momentum_weight': '行业动量加权',
    'opt07_stop_loss': '止损机制',
    'opt08_min_market_cap': '市值下限',
    'opt09_financial_quality_score': '财务质量评分',
    'opt10_turnover_control': '换手率控制',
    'opt11_combined_vol_stoploss': '组合优化',
}

order = [
    'baseline',
    'opt01_volatility_filter',
    'opt02_volume_confirm',
    'opt03_multi_factor_score',
    'opt04_biweekly_rebalance',
    'opt05_relative_momentum',
    'opt06_industry_momentum_weight',
    'opt07_stop_loss',
    'opt08_min_market_cap',
    'opt09_financial_quality_score',
    'opt10_turnover_control',
    'opt11_combined_vol_stoploss',
]

results = {}
for key in order:
    fname = key + '.json'
    fpath = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            results[key] = json.load(f)

labels = []
sharpe_values = []
return_values = []
drawdown_values = []
annual_values = []
colors = []

baseline_sharpe = results.get('baseline', {}).get('sharpe_ratio', 1.2506)

for key in order:
    if key not in results:
        continue
    r = results[key]
    labels.append(optimization_labels.get(key, key))
    sharpe_values.append(r['sharpe_ratio'])
    return_values.append(r['total_return_pct'])
    drawdown_values.append(abs(r['max_drawdown_pct']))
    annual_values.append(r['annual_return_pct'])

    if key == 'baseline':
        colors.append('#888888')
    elif key == 'opt11_combined_vol_stoploss':
        colors.append('#FF6B35')
    elif r['sharpe_ratio'] >= baseline_sharpe * 1.05:
        colors.append('#2ECC71')
    elif r['sharpe_ratio'] < baseline_sharpe:
        colors.append('#E74C3C')
    else:
        colors.append('#F39C12')

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('小市值策略优化对比分析', fontsize=18, fontweight='bold', y=0.98)

ax1 = axes[0, 0]
bars1 = ax1.bar(range(len(labels)), sharpe_values, color=colors, edgecolor='white', linewidth=0.5)
ax1.axhline(y=baseline_sharpe, color='#888888', linestyle='--', alpha=0.7, label=f'基线夏普={baseline_sharpe:.2f}')
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('夏普比率', fontsize=12)
ax1.set_title('夏普比率对比', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
for i, v in enumerate(sharpe_values):
    ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2 = axes[0, 1]
bars2 = ax2.bar(range(len(labels)), return_values, color=colors, edgecolor='white', linewidth=0.5)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('总收益 (%)', fontsize=12)
ax2.set_title('总收益对比', fontsize=14, fontweight='bold')
for i, v in enumerate(return_values):
    ax2.text(i, v + 10, f'{v:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax3 = axes[1, 0]
bars3 = ax3.bar(range(len(labels)), drawdown_values, color=colors, edgecolor='white', linewidth=0.5)
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('最大回撤 (%)', fontsize=12)
ax3.set_title('最大回撤对比（绝对值）', fontsize=14, fontweight='bold')
for i, v in enumerate(drawdown_values):
    ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax4 = axes[1, 1]
change_pct = [(s / baseline_sharpe - 1) * 100 for s in sharpe_values]
bar_colors = ['#2ECC71' if c >= 5 else '#E74C3C' if c < 0 else '#F39C12' for c in change_pct]
bars4 = ax4.bar(range(len(labels)), change_pct, color=bar_colors, edgecolor='white', linewidth=0.5)
ax4.axhline(y=5, color='#2ECC71', linestyle='--', alpha=0.7, label='5%阈值')
ax4.axhline(y=0, color='#888888', linestyle='-', alpha=0.5)
ax4.set_xticks(range(len(labels)))
ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('夏普比率变化 (%)', fontsize=12)
ax4.set_title('夏普比率相对基线变化', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
for i, v in enumerate(change_pct):
    offset = 1 if v >= 0 else -3
    ax4.text(i, v + offset, f'{v:+.1f}%', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(RESULTS_DIR, 'optimization_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'图表已保存: {output_path}')
plt.close()
