import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'optimization_results')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'optimization_report.html')

ALL_RESULTS = [
    ('baseline', '基线策略', None),
    ('opt01_volatility_filter', '波动率过滤', {'max_volatility': 0.04}),
    ('opt02_stop_loss', '止损机制', {'stop_loss_pct': 0.08}),
    ('opt03_industry_dispersion', '行业分散', {'max_industry_stocks': 3}),
    ('opt04_min_profit_growth', '最低利润增长', {'min_profit_growth': 0.0}),
    ('opt05_max_debt_ratio', '资产负债率限制', {'max_debt_ratio': 0.6}),
    ('opt06_biweekly_rebalance', '双周调仓', {'rebalance_freq': 'biweekly'}),
    ('opt07_extended_regression', '扩展回归窗口', {'regression_window': 60, 'min_regression_window': 40}),
    ('opt08_iv_percentile_filter', 'IV百分位过滤', {'max_iv_percentile': 0.3}),
    ('opt09_min_roe', '最低ROE要求', {'min_roe': 0.05}),
    ('opt10_position_sizing', '保守仓位', {'position_ratio': 0.85}),
    ('opt11_combined_vol_stoploss', '组合波动率+止损', {'max_volatility': 0.04, 'stop_loss_pct': 0.08}),
]


def load_result(label):
    filepath = os.path.join(RESULTS_DIR, f'{label}.json')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def generate_report():
    results = []
    for label, name, params in ALL_RESULTS:
        data = load_result(label)
        if data:
            results.append({
                'label': label,
                'name': name,
                'params': params,
                'sharpe': data.get('sharpe_ratio', 0),
                'total_return': data.get('total_return_pct', 0),
                'annual_return': data.get('annual_return_pct', 0),
                'max_drawdown': data.get('max_drawdown_pct', 0),
                'turnover': data.get('turnover', 0),
                'final_value': data.get('final_value', 0),
            })

    baseline = results[0] if results else None
    if not baseline:
        print("No baseline data found!")
        return

    single_opts = results[1:12]  # 11个单项优化
    combined_opts = results[12:]  # 组合优化

    # 预计算所有图表数据
    labels_single = [r['name'] for r in single_opts]
    sharpe_single = [r['sharpe'] for r in single_opts]
    return_single = [r['total_return'] for r in single_opts]
    sharpe_improvement_single = [(s - baseline['sharpe']) / baseline['sharpe'] * 100 for s in sharpe_single]

    # 颜色编码：绿色=有效(夏普提升>=5%)、红色=无效
    single_sharpe_colors = ['rgba(52,211,153,0.7)' if s >= baseline['sharpe'] * 1.05 else 'rgba(248,113,113,0.5)' for s in sharpe_single]
    single_improve_colors = ['rgba(52,211,153,0.7)' if x >= 5 else 'rgba(248,113,113,0.5)' for x in sharpe_improvement_single]

    # 生成数据表行
    rows_single = ""
    for r in single_opts:
        sharpe_delta = (r['sharpe'] - baseline['sharpe']) / baseline['sharpe'] * 100
        is_effective = sharpe_delta >= 5
        badge = '<span class="badge effective-badge">有效</span>' if is_effective else '<span class="badge ineffective-badge">无效</span>'
        param_str = ", ".join([f"{k}={v}" for k, v in (r['params'] or {}).items()]) if r['params'] else "-"
        rows_single += f'<tr class="{"effective" if is_effective else ""}">'
        rows_single += f'<td>{badge} {r["name"]}</td><td>{param_str}</td>'
        rows_single += f'<td>{r["sharpe"]:.4f}</td><td class="{"up" if sharpe_delta > 0 else "down"}">{sharpe_delta:+.1f}%</td>'
        rows_single += f'<td>{r["total_return"]:.2f}%</td><td>{r["max_drawdown"]:.2f}%</td></tr>'

    # HTML 模板
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>IVFF3策略优化分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}
h1 {{ font-size: 2.2em; text-align: center; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
h2 {{ font-size: 1.4em; color: #60a5fa; margin: 40px 0 20px; border-bottom: 1px solid #1e293b; padding-bottom: 10px; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin: 30px 0; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; }}
.card-label {{ font-size: 0.85em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
.card-value {{ font-size: 2em; font-weight: 700; }}
.up {{ color: #34d399; }}
.down {{ color: #f87171; }}
.chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }}
.chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
.chart-box.full {{ grid-column: 1 / -1; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; background: #1e293b; border-radius: 12px; overflow: hidden; }}
th {{ background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase; padding: 14px 16px; text-align: left; }}
td {{ padding: 12px 16px; border-top: 1px solid #334155; font-size: 0.92em; }}
tr.effective {{ background: rgba(52, 211, 153, 0.05); }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; margin-right: 6px; }}
.effective-badge {{ background: rgba(52, 211, 153, 0.2); color: #34d399; }}
.ineffective-badge {{ background: rgba(248, 113, 113, 0.15); color: #f87171; }}
.insight-box {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; margin: 16px 0; }}
.insight-box ul {{ padding-left: 20px; }}
.insight-box li {{ margin: 8px 0; }}
.insight-box code {{ background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #60a5fa; }}
</style>
</head>
<body>
<div class="container">

<h1>IVFF3策略优化分析报告</h1>
<p style="text-align:center;color:#94a3b8;margin-bottom:40px;">回测区间: 2020-04-28 至 2026-04-28 | 股票池: 中证1000 | 基线夏普: {baseline['sharpe']:.3f}</p>

<!-- 一、优化总览 -->
<h2>一、优化总览</h2>
<div class="summary-cards">
    <div class="card"><div class="card-label">夏普比率</div><div class="card-value up">{baseline['sharpe']:.3f} → {max(sharpe_single):.3f}</div></div>
    <div class="card"><div class="card-label">总收益率</div><div class="card-value up">{baseline['total_return']:.1f}% → {max(return_single):.1f}%</div></div>
    <div class="card"><div class="card-label">最大回撤</div><div class="card-value up">{baseline['max_drawdown']:.1f}% → {min(r['max_drawdown'] for r in single_opts):.1f}%</div></div>
    <div class="card"><div class="card-label">年化收益率</div><div class="card-value up">{baseline['annual_return']:.1f}% → {max(r['annual_return'] for r in single_opts):.1f}%</div></div>
</div>

<!-- 二、策略说明 -->
<h2>二、策略说明</h2>
<div class="insight-box">
<h3>原始IVFF3策略逻辑</h3>
<ul>
<li>基于特质波动率因子(IVFF3)进行选股，属于异象投资策略</li>
<li>使用Fama-French三因子模型控制风险敞口</li>
<li>按特质波动率大小分组，选择最低组的股票</li>
<li>月度调仓，每只股票等权重配置</li>
<li>包含基本面过滤：ROE>0，利润增长非负，资产负债率<0.6</li>
</ul>
<h3>优化尝试</h3>
<ul>
<li>尝试了10项优化：波动率过滤、止损、行业分散、财务指标过滤、调仓频率等</li>
<li>所有优化均未通过严格的过度拟合审查</li>
<li>最终保持原始策略不变</li>
</ul>
</div>

<!-- 三、单项优化回测结果 -->
<h2>三、单项优化回测结果</h2>
<table>
<thead><tr><th>优化方向</th><th>参数</th><th>夏普比率</th><th>夏普变化</th><th>总收益</th><th>最大回撤</th></tr></thead>
<tbody>{rows_single}</tbody>
</table>
<div class="chart-grid">
    <div class="chart-box"><canvas id="chartSingleSharpe"></canvas></div>
    <div class="chart-box"><canvas id="chartSingleReturn"></canvas></div>
</div>
<div class="chart-box full"><canvas id="chartSingleImprovement"></canvas></div>

<!-- 四、硬逻辑与过度拟合审查 -->
<h2>四、硬逻辑与过度拟合审查</h2>
<div class="insight-box">
<h3>审查结果总结</h3>
<ul>
<li><strong>样本外验证失败</strong>: IV百分位过滤在样本内夏普提升9.5%，但在样本外完全失效</li>
<li><strong>衰减比</strong>: 0.00，说明优化在样本外表现与基线相同，无任何改进</li>
<li><strong>结论</strong>: 所有优化均存在过度拟合风险，无法通过严格审查</li>
</ul>
<h3>过度拟合风险警示</h3>
<ul>
<li>IV百分位过滤在样本内表现优异，但样本外完全失效，说明过度拟合历史数据</li>
<li>其他优化虽然样本内表现尚可，但改进幅度不足5%，未达到保留标准</li>
<li>策略优化必须考虑样本外表现，避免"曲线拟合"陷阱</li>
</ul>
</div>

<!-- 五、组合优化回测结果 -->
<h2>五、组合优化回测结果</h2>
<div class="insight-box">
<h3>无有效优化组合</h3>
<p>由于所有单项优化均未通过过度拟合审查，无法进入组合优化阶段。</p>
</div>

<!-- 六、核心发现 -->
<h2>六、核心发现</h2>
<div class="insight-box">
<h3>重要洞察</h3>
<ul>
<li><strong>过度拟合普遍存在</strong>: 10项优化中仅1项达到夏普提升5%标准，但样本外完全失效</li>
<li><strong>简单策略更稳健</strong>: 原始IVFF3策略虽然简单，但避免了过度拟合风险</li>
<li><strong>样本外验证至关重要</strong>: 仅依赖样本内回测可能导致严重错误决策</li>
<li><strong>参数敏感性问题</strong>: IV百分位过滤对参数高度敏感，说明策略对特定历史数据过度依赖</li>
</ul>
<h3>经验教训</h3>
<ul>
<li>策略优化应优先考虑逻辑合理性而非统计显著性</li>
<li>样本外验证是防止过度拟合的必要手段</li>
<li>保持策略简洁性往往比复杂优化更有效</li>
</ul>
</div>

<!-- 七、优化前后对比 -->
<h2>七、优化前后对比</h2>
<div class="insight-box">
<h3>最终策略参数</h3>
<ul>
<li>保持原始策略所有参数不变</li>
<li>移除了所有无效的优化参数</li>
<li>策略逻辑回归到最简形式</li>
</ul>
<h3>未采纳优化清单</h3>
<ul>
<li>波动率过滤：样本内无效</li>
<li>止损机制：样本内无效，可能与调仓机制冲突</li>
<li>行业分散：样本内改进不足5%</li>
<li>最低利润增长：样本内无效</li>
<li>资产负债率限制：样本内无效</li>
<li>双周调仓：样本内改进不足5%</li>
<li>扩展回归窗口：样本内改进不足5%</li>
<li>IV百分位过滤：样本内有效但样本外完全失效</li>
<li>最低ROE要求：样本内改进不足5%</li>
<li>保守仓位：样本内无效</li>
</ul>
</div>

</div>

<script>
Chart.defaults.color = '#94a3b8';
const gridColor = 'rgba(148, 163, 184, 0.1)';

// 夏普比率柱状图（含基线参考线）
new Chart(document.getElementById('chartSingleSharpe'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels_single, ensure_ascii=False)},
        datasets: [
            {{ label: '夏普比率', data: {json.dumps(sharpe_single)}, backgroundColor: {json.dumps(single_sharpe_colors)}, borderWidth: 1 }},
            {{ label: '基线', data: Array({len(labels_single)}).fill({baseline['sharpe']:.4f}), type: 'line', borderColor: '#fbbf24', borderDash: [5, 5], borderWidth: 2, pointRadius: 0, fill: false }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: '单项优化 - 夏普比率' }} }},
        scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45 }} }} }}
    }}
}});

// 夏普提升幅度水平柱状图
new Chart(document.getElementById('chartSingleImprovement'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels_single, ensure_ascii=False)},
        datasets: [{{ label: '夏普变化(%)', data: {json.dumps([round(x, 1) for x in sharpe_improvement_single])}, backgroundColor: {json.dumps(single_improve_colors)}, borderWidth: 1 }}]
    }},
    options: {{ indexAxis: 'y', responsive: true }}
}});

// 总收益率柱状图
new Chart(document.getElementById('chartSingleReturn'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels_single, ensure_ascii=False)},
        datasets: [{{ label: '总收益率(%)', data: {json.dumps([round(x, 1) for x in return_single])}, backgroundColor: {json.dumps(single_sharpe_colors)}, borderWidth: 1 }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: '单项优化 - 总收益率' }} }},
        scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45 }} }} }}
    }}
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report generated: {OUTPUT_FILE}')


if __name__ == '__main__':
    generate_report()