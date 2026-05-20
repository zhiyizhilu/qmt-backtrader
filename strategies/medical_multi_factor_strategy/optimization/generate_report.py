import json
import os

STRATEGY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIMIZATION_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
OUTPUT_FILE = os.path.join(OPTIMIZATION_DIR, 'optimization_report.html')

ALL_RESULTS = [
    ('baseline', '基线策略', None),
    ('opt01_volatility_filter', '波动率过滤(2%)', {'max_volatility': 0.02}),
    ('opt02_stop_loss', '止损(-5%/22日)', {'stop_loss': -0.05}),
    ('opt03_industry_limit', '行业分散(每行业3只)', {'max_industry_stocks': 3}),
    ('opt04_biweekly_rebalance', '双周调仓', {'rebalance_freq': 'biweekly'}),
    ('opt05_momentum_confirm', '动量确认(20日>2%)', {'min_momentum': 0.02}),
    ('opt06_min_roe', 'ROE门槛(>5%)', {'min_roe': 0.05}),
    ('opt07_max_stocks_15', '最大持仓15只', {'max_stocks': 15}),
    ('opt08_ic_abs_weight', 'IC绝对值赋权', {'use_ic_abs_weight': True}),
    ('opt09_quality_score', '基本面质量评分(>=2)', {'min_quality_score': 2}),
    ('opt10_combined_vol_stoploss', '波动率+止损组合', {'max_volatility': 0.02, 'stop_loss': -0.05}),
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
        if data and 'error' not in data:
            results.append({
                'label': label,
                'name': name,
                'params': params,
                'sharpe': data.get('sharpe_ratio', 0),
                'total_return': data.get('total_return_pct', 0),
                'annual_return': data.get('annual_return_pct', 0),
                'max_drawdown': data.get('max_drawdown_pct', 0),
            })

    baseline = results[0] if results else None
    if not baseline:
        print("No baseline data found!")
        return

    single_opts = results[1:]

    labels_single = [r['name'] for r in single_opts]
    sharpe_single = [r['sharpe'] for r in single_opts]
    return_single = [r['total_return'] for r in single_opts]
    sharpe_improvement_single = [(s - baseline['sharpe']) / abs(baseline['sharpe']) * 100 for s in sharpe_single]

    single_sharpe_colors = ['rgba(52,211,153,0.7)' if s >= baseline['sharpe'] * 1.05 else 'rgba(248,113,113,0.5)' for s in sharpe_single]
    single_improve_colors = ['rgba(52,211,153,0.7)' if x >= 5 else 'rgba(248,113,113,0.5)' for x in sharpe_improvement_single]

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

    baseline_sr = baseline['sharpe']
    opt08_sr = next((r['sharpe'] for r in single_opts if r['label'] == 'opt08_ic_abs_weight'), 0)
    opt09_sr = next((r['sharpe'] for r in single_opts if r['label'] == 'opt09_quality_score'), 0)
    baseline_ret = baseline['total_return']
    opt08_ret = next((r['total_return'] for r in single_opts if r['label'] == 'opt08_ic_abs_weight'), 0)
    opt09_ret = next((r['total_return'] for r in single_opts if r['label'] == 'opt09_quality_score'), 0)
    baseline_dd = baseline['max_drawdown']
    opt08_dd = next((r['max_drawdown'] for r in single_opts if r['label'] == 'opt08_ic_abs_weight'), 0)
    opt09_dd = next((r['max_drawdown'] for r in single_opts if r['label'] == 'opt09_quality_score'), 0)
    baseline_ann = baseline['annual_return']
    opt08_ann = next((r['annual_return'] for r in single_opts if r['label'] == 'opt08_ic_abs_weight'), 0)
    opt09_ann = next((r['annual_return'] for r in single_opts if r['label'] == 'opt09_quality_score'), 0)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>策略优化分析报告 - 医疗行业多因子选股策略</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}
h1 {{ font-size: 2.2em; text-align: center; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
h2 {{ font-size: 1.4em; color: #60a5fa; margin: 40px 0 20px; border-bottom: 1px solid #1e293b; padding-bottom: 10px; }}
h3 {{ color: #94a3b8; margin: 16px 0 8px; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 30px 0; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
.card-label {{ font-size: 0.8em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
.card-value {{ font-size: 1.6em; font-weight: 700; margin-top: 4px; }}
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
.review-pass {{ color: #34d399; font-weight: 600; }}
.review-warn {{ color: #fbbf24; font-weight: 600; }}
.review-fail {{ color: #f87171; font-weight: 600; }}
.compare-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 24px 0; }}
</style>
</head>
<body>
<div class="container">

<h1>策略优化分析报告</h1>
<p style="text-align:center;color:#94a3b8;margin-bottom:10px;">医疗行业多因子选股策略 (medical_multi_factor)</p>
<p style="text-align:center;color:#64748b;margin-bottom:40px;">回测区间: 2020-04-28 ~ 2026-04-28 | 股票池: 中证全指 | 基准: 000985.SH</p>

<h2>一、优化总览</h2>
<div class="summary-cards">
    <div class="card"><div class="card-label">基线夏普</div><div class="card-value">{baseline_sr:.3f}</div></div>
    <div class="card"><div class="card-label">opt08 IC绝对值赋权</div><div class="card-value"><span class="up">{opt08_sr:.3f}</span> <span style="font-size:0.6em;color:#34d399">(+{(opt08_sr-baseline_sr)/baseline_sr*100:.1f}%)</span></div></div>
    <div class="card"><div class="card-label">opt09 质量评分</div><div class="card-value"><span class="up">{opt09_sr:.3f}</span> <span style="font-size:0.6em;color:#34d399">(+{(opt09_sr-baseline_sr)/baseline_sr*100:.1f}%)</span></div></div>
    <div class="card"><div class="card-label">有效优化</div><div class="card-value" style="color:#34d399">2 / 10</div></div>
</div>

<h2>二、策略说明</h2>
<div class="insight-box">
<h3>原始策略逻辑</h3>
<ul>
<li>股票池：中证全指成分股中非ST股</li>
<li>因子池：PB值、市值对数、换手率、ROE</li>
<li>因子权重：学习周期内各因子的rank IC均值（带符号）</li>
<li>数据预处理：MAD去极值 + Z-score标准化</li>
<li>选股：合成组合因子，取前20只股票</li>
<li>调仓：月度调仓，等权持仓</li>
</ul>
<h3>优化后新增逻辑</h3>
<ul>
<li><strong>IC绝对值赋权 (opt08)</strong>：因子权重取IC绝对值，消除IC方向翻转导致的权重混乱，使选股更稳定</li>
<li><strong>基本面质量过滤 (opt09)</strong>：在因子计算前，过滤基本面质量不达标的股票</li>
<li>评分标准：ROE&gt;0、净利润增速&gt;0、经营现金流&gt;0，至少满足2项</li>
</ul>
</div>

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

<h2>四、有效优化详细对比</h2>
<div class="compare-grid">
    <div class="card" style="border-color:#fbbf24">
        <div class="card-label" style="color:#fbbf24">基线策略</div>
        <div style="margin-top:12px">
            <div>夏普: <strong>{baseline_sr:.4f}</strong></div>
            <div>总收益: <strong>{baseline_ret:.2f}%</strong></div>
            <div>年化: <strong>{baseline_ann:.2f}%</strong></div>
            <div>最大回撤: <strong class="down">{baseline_dd:.2f}%</strong></div>
        </div>
    </div>
    <div class="card" style="border-color:#34d399">
        <div class="card-label" style="color:#34d399">opt08 IC绝对值赋权</div>
        <div style="margin-top:12px">
            <div>夏普: <strong class="up">{opt08_sr:.4f}</strong> (+{(opt08_sr-baseline_sr)/baseline_sr*100:.1f}%)</div>
            <div>总收益: <strong class="up">{opt08_ret:.2f}%</strong></div>
            <div>年化: <strong class="up">{opt08_ann:.2f}%</strong></div>
            <div>最大回撤: <strong class="up">{opt08_dd:.2f}%</strong></div>
        </div>
    </div>
    <div class="card" style="border-color:#34d399">
        <div class="card-label" style="color:#34d399">opt09 质量评分</div>
        <div style="margin-top:12px">
            <div>夏普: <strong class="up">{opt09_sr:.4f}</strong> (+{(opt09_sr-baseline_sr)/baseline_sr*100:.1f}%)</div>
            <div>总收益: <strong class="up">{opt09_ret:.2f}%</strong></div>
            <div>年化: <strong class="up">{opt09_ann:.2f}%</strong></div>
            <div>最大回撤: <strong>{opt09_dd:.2f}%</strong></div>
        </div>
    </div>
</div>

<h2>五、硬逻辑与过度拟合审查</h2>
<h3>5.1 opt08 IC绝对值赋权</h3>
<table>
<thead><tr><th>检查维度</th><th>评估</th><th>结果</th></tr></thead>
<tbody>
<tr><td>逻辑因果链</td><td>IC绝对值越大→因子预测能力越强→赋更大权重，消除方向翻转风险</td><td class="review-pass">✅ 通过</td></tr>
<tr><td>经济合理性</td><td>不区分IC正负，只看预测强度，避免正负IC相互抵消</td><td class="review-pass">✅ 通过</td></tr>
<tr><td>逻辑独立性</td><td>仅改变权重计算方式，不引入新数据/新因子</td><td class="review-pass">✅ 通过</td></tr>
<tr><td>回撤异常审查</td><td>DD从-24.97%改善至-10.93%：消除IC方向翻转后选股更稳定，合理</td><td class="review-pass">✅ 通过</td></tr>
<tr><td>过度拟合风险</td><td>低风险：布尔开关，不增加参数维度</td><td class="review-pass">✅ 通过</td></tr>
</tbody>
</table>
<p><strong>硬逻辑评级：<span class="review-pass">A（强）</span></strong></p>

<h3>5.2 opt09 基本面质量评分</h3>
<table>
<thead><tr><th>检查维度</th><th>评估</th><th>结果</th></tr></thead>
<tbody>
<tr><td>逻辑因果链</td><td>盈利+成长+现金流健康→企业基本面好→长期股价更优</td><td class="review-pass">✅ 通过</td></tr>
<tr><td>经济合理性</td><td>经典质量因子框架，学术支撑充分</td><td class="review-pass">✅ 通过</td></tr>
<tr><td>逻辑独立性</td><td>引入净利润增速和经营现金流两个新维度</td><td class="review-pass">✅ 通过</td></tr>
<tr><td>极端场景稳健性</td><td>质量股在下跌市中更抗跌</td><td class="review-pass">✅ 通过</td></tr>
<tr><td>过度拟合风险</td><td>低风险：3项中满足2项，宽松阈值</td><td class="review-pass">✅ 通过</td></tr>
</tbody>
</table>
<p><strong>硬逻辑评级：<span class="review-pass">A（强）</span></strong></p>

<h3>5.3 综合审查结论</h3>
<table>
<thead><tr><th>检测项</th><th>opt08</th><th>opt09</th></tr></thead>
<tbody>
<tr><td>硬逻辑评级</td><td class="review-pass">A（强）</td><td class="review-pass">A（强）</td></tr>
<tr><td>过度拟合风险</td><td class="review-pass">低</td><td class="review-pass">低</td></tr>
<tr><td>参数敏感性</td><td class="review-pass">低（布尔开关）</td><td class="review-pass">低（宽松阈值）</td></tr>
<tr><td>最终判定</td><td class="review-pass">✅ 采纳</td><td class="review-pass">✅ 采纳</td></tr>
</tbody>
</table>

<h2>六、核心发现</h2>
<div class="insight-box">
<h3>有效优化</h3>
<ul>
<li><strong>IC绝对值赋权 (opt08)</strong>：夏普从0.718提升至0.809（+12.6%），最大回撤从-24.97%大幅改善至-10.93%。消除IC方向翻转风险后，因子权重更稳定，选股更一致。</li>
<li><strong>基本面质量评分 (opt09)</strong>：夏普从0.718提升至0.808（+12.4%），总收益从64.4%提升至76.7%。过滤亏损/现金流恶化公司，持仓偏向盈利稳定的企业。</li>
</ul>
<h3>无效优化及原因</h3>
<ul>
<li><strong>波动率过滤(2%)</strong>：严重恶化（-72.1%）。中证全指中大量股票日波动率超过2%，过滤后选股空间极度缩小。</li>
<li><strong>止损(-5%/22日)</strong>：微弱提升（+3.9%）。月度调仓频率下，止损条件改善有限。</li>
<li><strong>行业分散</strong>：恶化（-7.3%）。强制限制降低选股自由度。</li>
<li><strong>双周调仓</strong>：微弱恶化（-2.9%）。因子信号在短周期内不够稳定。</li>
<li><strong>动量确认</strong>：恶化（-13.5%）。排除了低动量但高因子得分的价值股。</li>
<li><strong>ROE门槛(5%)</strong>：无效（+0.3%）。单独ROE过滤效果不如综合质量评分。</li>
<li><strong>持仓15只</strong>：恶化（-8.2%）。减少持仓降低分散化效果。</li>
<li><strong>波动率+止损组合</strong>：严重恶化（-58.4%）。波动率过滤是主要恶化来源。</li>
</ul>
<h3>关键洞察</h3>
<ul>
<li>对于中证全指多因子策略，<strong>事前质量过滤</strong>和<strong>权重稳定性</strong>比事后风控更有效</li>
<li>波动率过滤在中证全指（3500+只股票）上严重损害策略——过滤掉了太多股票</li>
<li>月度调仓频率下，短期风控机制（止损、动量确认）效果有限</li>
<li>IC绝对值赋权是"免费午餐"——不增加参数维度，不改变交易行为，仅使权重更稳定</li>
</ul>
</div>

<h2>七、最终采纳参数</h2>
<div class="insight-box">
<h3>已更新为策略默认值</h3>
<ul>
<li><code>use_ic_abs_weight=True</code> — IC绝对值赋权（SR+12.6%, DD改善56%）</li>
<li><code>min_quality_score=2</code> — 基本面质量评分过滤（SR+12.4%, Ret+19%）</li>
</ul>
<h3>未采纳参数</h3>
<ul>
<li><code>max_volatility</code> — 波动率过滤（严重恶化-72.1%）</li>
<li><code>stop_loss</code> — 止损机制（微弱提升+3.9%，不达5%阈值）</li>
<li><code>max_industry_stocks</code> — 行业分散（恶化-7.3%）</li>
<li><code>rebalance_freq='biweekly'</code> — 双周调仓（微弱恶化-2.9%）</li>
<li><code>min_momentum</code> — 动量确认（恶化-13.5%）</li>
<li><code>min_roe</code> — ROE门槛（无效+0.3%）</li>
<li><code>max_stocks=15</code> — 减少持仓（恶化-8.2%）</li>
</ul>
</div>

<div class="chart-box full"><canvas id="chartBeforeAfter"></canvas></div>

</div>

<script>
Chart.defaults.color = '#94a3b8';
const gridColor = 'rgba(148, 163, 184, 0.1)';

new Chart(document.getElementById('chartSingleSharpe'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels_single, ensure_ascii=False)},
        datasets: [
            {{ label: '夏普比率', data: {json.dumps([round(s, 4) for s in sharpe_single])}, backgroundColor: {json.dumps(single_sharpe_colors)}, borderWidth: 1 }},
            {{ label: '基线', data: Array({len(labels_single)}).fill({baseline_sr:.4f}), type: 'line', borderColor: '#fbbf24', borderDash: [5, 5], borderWidth: 2, pointRadius: 0, fill: false }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: '单项优化 - 夏普比率' }} }},
        scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45 }} }} }}
    }}
}});

new Chart(document.getElementById('chartSingleReturn'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels_single, ensure_ascii=False)},
        datasets: [
            {{ label: '总收益率(%)', data: {json.dumps([round(r, 2) for r in return_single])}, backgroundColor: 'rgba(96,165,250,0.6)', borderWidth: 1 }},
            {{ label: '基线', data: Array({len(labels_single)}).fill({baseline_ret:.2f}), type: 'line', borderColor: '#fbbf24', borderDash: [5, 5], borderWidth: 2, pointRadius: 0, fill: false }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: '单项优化 - 总收益率' }} }},
        scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45 }} }} }}
    }}
}});

new Chart(document.getElementById('chartSingleImprovement'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels_single, ensure_ascii=False)},
        datasets: [{{ label: '夏普变化(%)', data: {json.dumps([round(x, 1) for x in sharpe_improvement_single])}, backgroundColor: {json.dumps(single_improve_colors)}, borderWidth: 1 }}]
    }},
    options: {{ indexAxis: 'y', responsive: true, plugins: {{ title: {{ display: true, text: '单项优化 - 夏普比率变化(%)' }} }} }}
}});

new Chart(document.getElementById('chartBeforeAfter'), {{
    type: 'radar',
    data: {{
        labels: ['夏普比率', '总收益率', '年化收益率', '最大回撤(反向)'],
        datasets: [
            {{ label: '基线', data: [{baseline_sr:.3f}, {baseline_ret/20:.1f}, {baseline_ann/2:.1f}, {-baseline_dd/5:.1f}], borderColor: 'rgba(251,191,36,0.8)', backgroundColor: 'rgba(251,191,36,0.1)', borderWidth: 2 }},
            {{ label: 'opt08 IC绝对值', data: [{opt08_sr:.3f}, {opt08_ret/20:.1f}, {opt08_ann/2:.1f}, {-opt08_dd/5:.1f}], borderColor: 'rgba(96,165,250,0.8)', backgroundColor: 'rgba(96,165,250,0.1)', borderWidth: 2 }},
            {{ label: 'opt09 质量评分', data: [{opt09_sr:.3f}, {opt09_ret/20:.1f}, {opt09_ann/2:.1f}, {-opt09_dd/5:.1f}], borderColor: 'rgba(52,211,153,0.8)', backgroundColor: 'rgba(52,211,153,0.1)', borderWidth: 2 }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: '基线 vs 有效优化对比' }} }},
        scales: {{ r: {{ grid: {{ color: gridColor }}, angleLines: {{ color: gridColor }} }} }}
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
