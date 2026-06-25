"""Round 2 HTML优化报告生成"""
import json, os

STRATEGY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_report_r2.html')

def load(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if os.path.exists(path):
        return json.load(open(path, 'r', encoding='utf-8'))
    return None

ALL_RESULTS = [
    ('r2_baseline', '基线(持仓6只)', None),
    ('r2_opt01_volatility_3pct', '波动率过滤(3%)', {'max_volatility': 0.03}),
    ('r2_opt02_volatility_4pct', '波动率过滤(4%)', {'max_volatility': 0.04}),
    ('r2_opt03_switch_5pct', '换仓阈值(5%)', {'switch_threshold': 0.05}),
    ('r2_opt04_switch_10pct', '换仓阈值(10%)', {'switch_threshold': 0.10}),
    ('r2_opt05_take_profit_15x', '止盈50%(1.5x)', {'take_profit_ratio': 1.5}),
    ('r2_opt06_market_stoploss_3pct', '市场止损(3%)', {'market_stoploss': 0.03}),
    ('r2_opt07_ma_period_20', 'MA周期20日', {'ma_period': 20}),
    ('r2_opt08_max_stocks_8', '持仓8只', {'max_stocks': 8}),
    ('r2_opt09_min_market_cap_5', '最小市值5亿', {'min_market_cap': 5}),
    ('r2_opt10_skip_months_124', '空仓月份(1,2,4)', {'skip_months': (1, 2, 4)}),
    ('r2_combo_vol3_stocks8', '组合:波动率3%+持仓8只', {'max_volatility': 0.03, 'max_stocks': 8}),
    ('r2_combo_switch10_stocks8', '组合:换仓10%+持仓8只', {'switch_threshold': 0.10, 'max_stocks': 8}),
    ('r2_combo_all3', '组合:三项全部', {'max_volatility': 0.03, 'switch_threshold': 0.10, 'max_stocks': 8}),
]

results = []
for label, name, params in ALL_RESULTS:
    data = load(label)
    if data:
        results.append({
            'label': label, 'name': name, 'params': params,
            'sharpe': data.get('sharpe_ratio', 0),
            'total_return': data.get('total_return_pct', 0),
            'max_drawdown': data.get('max_drawdown_pct', 0),
            'annual_return': data.get('annual_return_pct', 0),
        })

baseline = results[0] if results else None
single_opts = results[1:11]
combo_opts = results[11:]

# Precompute chart data
labels_single = [r['name'] for r in single_opts]
sharpe_single = [round(r['sharpe'], 4) for r in single_opts]
return_single = [round(r['total_return'], 2) for r in single_opts]
sharpe_improvement = [round((s - baseline['sharpe']) / baseline['sharpe'] * 100, 1) for s in sharpe_single]
single_colors = ['#34d399' if s >= baseline['sharpe'] * 1.05 else '#f87171' for s in sharpe_single]
improve_colors = ['#34d399' if x >= 5 else '#f87171' for x in sharpe_improvement]

labels_combo = [r['name'] for r in combo_opts]
sharpe_combo = [round(r['sharpe'], 4) for r in combo_opts]
combo_colors = ['#60a5fa' if s >= baseline['sharpe'] * 1.05 else '#f87171' for s in sharpe_combo]

# Table rows
rows_single = ""
for i, r in enumerate(single_opts):
    delta = sharpe_improvement[i]
    badge = '<span class="badge eff">有效</span>' if delta >= 5 else '<span class="badge ineff">无效</span>'
    param_str = ", ".join([f"{k}={v}" for k, v in (r['params'] or {}).items()]) if r['params'] else "-"
    cls = "effective" if delta >= 5 else ""
    rows_single += f'<tr class="{cls}"><td>{badge} {r["name"]}</td><td>{param_str}</td>'
    rows_single += f'<td>{r["sharpe"]:.4f}</td><td class="{"up" if delta > 0 else "down"}">{delta:+.1f}%</td>'
    rows_single += f'<td>{r["total_return"]:.2f}%</td><td>{r["max_drawdown"]:.2f}%</td></tr>'

rows_combo = ""
for i, r in enumerate(combo_opts):
    delta = round((r['sharpe'] - baseline['sharpe']) / baseline['sharpe'] * 100, 1)
    badge = '<span class="badge eff">有效</span>' if delta >= 5 else '<span class="badge ineff">无效</span>'
    rows_combo += f'<tr><td>{badge} {r["name"]}</td><td>{r["sharpe"]:.4f}</td>'
    rows_combo += f'<td class="{"up" if delta > 0 else "down"}">{delta:+.1f}%</td>'
    rows_combo += f'<td>{r["total_return"]:.2f}%</td><td>{r["max_drawdown"]:.2f}%</td></tr>'

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>国九小市值策略 Round 2 优化报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
h1 { font-size: 2.2em; text-align: center; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { font-size: 1.4em; color: #60a5fa; margin: 40px 0 20px; border-bottom: 1px solid #1e293b; padding-bottom: 10px; }
.subtitle { text-align: center; color: #94a3b8; margin-bottom: 40px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin: 30px 0; }
.card { background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; }
.card-label { font-size: 0.85em; color: #94a3b8; text-transform: uppercase; }
.card-value { font-size: 1.8em; font-weight: 700; margin-top: 8px; }
.up { color: #34d399; } .down { color: #f87171; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }
.chart-box { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
.chart-box.full { grid-column: 1 / -1; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; background: #1e293b; border-radius: 12px; overflow: hidden; }
th { background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase; padding: 14px 16px; text-align: left; }
td { padding: 12px 16px; border-top: 1px solid #334155; font-size: 0.92em; }
tr.effective { background: rgba(52, 211, 153, 0.05); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; margin-right: 6px; }
.eff { background: rgba(52, 211, 153, 0.2); color: #34d399; }
.ineff { background: rgba(248, 113, 113, 0.15); color: #f87171; }
.insight { background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; margin: 16px 0; }
.insight ul { padding-left: 20px; } .insight li { margin: 8px 0; }
.insight code { background: #334155; padding: 2px 6px; border-radius: 4px; color: #60a5fa; }
</style>
</head>
<body>
<div class="container">
<h1>国九小市值策略 Round 2 优化报告</h1>
<p class="subtitle">测试集: 2020-04-28 ~ 2024-04-28 | 验证集: 2024-04-28 ~ 2026-04-28</p>

<h2>一、优化总览</h2>
<div class="cards">
    <div class="card"><div class="card-label">夏普比率</div><div class="card-value up">__BASE_SHARPE__ → 2.1938</div></div>
    <div class="card"><div class="card-label">总收益率</div><div class="card-value up">__BASE_RET__% → 373.2%</div></div>
    <div class="card"><div class="card-label">最大回撤</div><div class="card-value up">__BASE_DD__% → -12.4%</div></div>
    <div class="card"><div class="card-label">年化收益</div><div class="card-value up">__BASE_ANN__% → 49.8%</div></div>
</div>

<h2>二、策略说明</h2>
<div class="insight">
<h3>原始策略逻辑</h3>
<ul>
<li>从中小企业板指数成分股中，通过国九条财务指标过滤后，按总市值升序选股</li>
<li>每周调仓，动态持股数量（MA10调整），1月和4月空仓</li>
<li>持仓6只（Round 1优化结果），个股止损9%，止盈100%</li>
</ul>
<h3>Round 2 优化内容</h3>
<ul>
<li>新增 <code>switch_threshold</code> 换仓阈值参数（来自ETF轮动策略经验）</li>
<li>测试10项新方向，最终采纳 <strong>持仓8只</strong>（max_stocks 6→8）</li>
<li>同步调整 ma_stock_nums: (4,5,6,7,8) → (5,6,8,9,10)</li>
</ul>
</div>

<h2>三、单项优化回测结果</h2>
<table>
<thead><tr><th>优化方向</th><th>参数</th><th>夏普比率</th><th>夏普变化</th><th>总收益</th><th>最大回撤</th></tr></thead>
<tbody>__ROWS_SINGLE__</tbody>
</table>
<div class="chart-grid">
    <div class="chart-box"><canvas id="chartSharpe"></canvas></div>
    <div class="chart-box"><canvas id="chartImprove"></canvas></div>
</div>

<h2>四、硬逻辑与过度拟合审查</h2>
<div class="insight">
<h3>持仓8只 - 审查结果</h3>
<table>
<thead><tr><th>检查维度</th><th>结果</th><th>判定</th></tr></thead>
<tbody>
<tr><td>硬逻辑评级</td><td>A（强）- 分散化降低非系统性风险</td><td>通过</td></tr>
<tr><td>样本外衰减比</td><td>1.18（样本外+10.3% > 样本内+8.7%）</td><td>通过（>0.5）</td></tr>
<tr><td>参数敏感度</td><td>0.0814（max_stocks 6~10夏普范围0.18）</td><td>通过（<0.3）</td></tr>
<tr><td>时间稳定性</td><td>2/3正改进（2020负, 2021正, 验证集正）</td><td>有条件通过</td></tr>
<tr><td><strong>综合结论</strong></td><td><strong>通过</strong></td><td>通过</td></tr>
</tbody>
</table>
</div>

<h2>五、组合优化回测结果</h2>
<table>
<thead><tr><th>组合方案</th><th>夏普比率</th><th>夏普变化</th><th>总收益</th><th>最大回撤</th></tr></thead>
<tbody>__ROWS_COMBO__</tbody>
</table>
<div class="chart-box full"><canvas id="chartCombo"></canvas></div>

<h2>六、核心发现</h2>
<div class="insight">
<ul>
<li><strong>持仓数量是核心优化方向</strong>：Round 1(4→6, +6.5%)和Round 2(6→8, +8.7%)均有效，分散化持续降低风险</li>
<li><strong>波动率过滤不适合此策略</strong>：3%接近有效(+4.6%)但组合后拖累收益(373%→270%)，与small_cap策略表现不同</li>
<li><strong>换仓阈值无协同效应</strong>：单独+0.3%夏普，组合后反而低于持仓8只单项</li>
<li><strong>止盈/止损机制未生效</strong>：止盈50%和MA20均0%变化，说明avg_cost可能返回None导致机制未触发</li>
<li><strong>市场止损3%有害</strong>：-13.2%，过于敏感导致频繁清仓</li>
<li><strong>春节效应不存在</strong>：增加2月空仓反而-6.7%，2月是小市值策略的有效月份</li>
<li><strong>参数极鲁棒</strong>：max_stocks 6~10的夏普在2.02~2.20之间，选择8是保守的中间值</li>
</ul>
</div>

<h2>七、验证集检验 (2024-04-28 ~ 2026-04-28)</h2>
<div class="insight">
<table>
<thead><tr><th>指标</th><th>基线(持仓6只)</th><th>优化后(持仓8只)</th><th>变化</th></tr></thead>
<tbody>
<tr><td>夏普比率</td><td>1.2018</td><td>1.3257</td><td class="up">+10.3%</td></tr>
<tr><td>总收益</td><td>70.77%</td><td>78.77%</td><td class="up">+8.0pp</td></tr>
<tr><td>最大回撤</td><td>-19.86%</td><td>-18.05%</td><td class="up">+1.8pp</td></tr>
<tr><td>衰减比</td><td colspan="3">1.18（样本外效果优于样本内）</td></tr>
</tbody>
</table>
</div>

</div>
<script>
Chart.defaults.color = '#94a3b8';
var gridColor = 'rgba(148, 163, 184, 0.1)';
var baselineSharpe = __BASE_SHARPE__;

new Chart(document.getElementById('chartSharpe'), {
    type: 'bar',
    data: { labels: __LABELS_SINGLE__, datasets: [
        { label: 'Sharpe', data: __SHARPE_SINGLE__, backgroundColor: __SINGLE_COLORS__, borderWidth: 1 },
        { label: 'Baseline', data: Array(__N_SINGLE__).fill(baselineSharpe), type: 'line', borderColor: '#fbbf24', borderDash: [5,5], borderWidth: 2, pointRadius: 0, fill: false }
    ]},
    options: { responsive: true, plugins: { title: { display: true, text: 'Sharpe Ratio' } }, scales: { y: { grid: { color: gridColor } }, x: { grid: { display: false }, ticks: { maxRotation: 45 } } } }
});

new Chart(document.getElementById('chartImprove'), {
    type: 'bar',
    data: { labels: __LABELS_SINGLE__, datasets: [{ label: 'Sharpe Change %', data: __IMPROVE_SINGLE__, backgroundColor: __IMPROVE_COLORS__, borderWidth: 1 }]},
    options: { indexAxis: 'y', responsive: true, plugins: { title: { display: true, text: 'Sharpe Improvement %' } } }
});

new Chart(document.getElementById('chartCombo'), {
    type: 'bar',
    data: { labels: __LABELS_COMBO__, datasets: [
        { label: 'Sharpe', data: __SHARPE_COMBO__, backgroundColor: __COMBO_COLORS__, borderWidth: 1 },
        { label: 'Baseline', data: Array(__N_COMBO__).fill(baselineSharpe), type: 'line', borderColor: '#fbbf24', borderDash: [5,5], borderWidth: 2, pointRadius: 0, fill: false }
    ]},
    options: { responsive: true, plugins: { title: { display: true, text: 'Combination Optimization' } } }
});
</script>
</body>
</html>"""

# Replace placeholders
html = html.replace('__BASE_SHARPE__', f'{baseline["sharpe"]:.4f}')
html = html.replace('__BASE_RET__', f'{baseline["total_return"]:.1f}')
html = html.replace('__BASE_DD__', f'{baseline["max_drawdown"]:.1f}')
html = html.replace('__BASE_ANN__', f'{baseline.get("annual_return", 0):.1f}')
html = html.replace('__ROWS_SINGLE__', rows_single)
html = html.replace('__ROWS_COMBO__', rows_combo)
html = html.replace('__LABELS_SINGLE__', json.dumps(labels_single, ensure_ascii=False))
html = html.replace('__SHARPE_SINGLE__', json.dumps(sharpe_single))
html = html.replace('__SINGLE_COLORS__', json.dumps(single_colors))
html = html.replace('__IMPROVE_SINGLE__', json.dumps(sharpe_improvement))
html = html.replace('__IMPROVE_COLORS__', json.dumps(improve_colors))
html = html.replace('__N_SINGLE__', str(len(labels_single)))
html = html.replace('__LABELS_COMBO__', json.dumps(labels_combo, ensure_ascii=False))
html = html.replace('__SHARPE_COMBO__', json.dumps(sharpe_combo))
html = html.replace('__COMBO_COLORS__', json.dumps(combo_colors))
html = html.replace('__N_COMBO__', str(len(labels_combo)))

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Report generated: {OUTPUT_FILE}')
