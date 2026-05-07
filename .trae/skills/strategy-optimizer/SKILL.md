---
name: "strategy-optimizer"
description: "系统性地通过回测优化量化交易策略。当用户需要优化、改进或调优策略性能指标（如夏普比率）时调用此技能。"
---

# 策略优化器

在 qmt_backtrader 框架中系统性优化量化交易策略。本技能提供结构化的工作流程：提出优化建议、独立实施、逐一回测、硬逻辑与过度拟合审查、仅保留有效且稳健的改进。

## 调用时机

当用户出现以下情况时调用此技能：
- 想要优化或改进交易策略
- 要求调优策略参数以提升性能
- 想要对比不同优化方案的效果
- 提及夏普比率、回撤降低、收益提升等目标
- 要求系统性测试策略改进方案

## 核心工作流

### 阶段一：理解策略

1. 读取策略文件，理解当前逻辑和参数
2. 读取策略目录下的 `readme.md` 文档
3. **识别策略基类**：判断是 `StockSelectionStrategy` 还是 `StrategyLogic`，两者回测模板不同
4. 运行基线回测，记录当前性能指标
5. 确定核心评估指标（默认：夏普比率）

### 阶段二：提出优化建议

提出10项具体优化建议，每项必须包含：
- **优化方向**：改进哪个方面
- **技术实现路径**：如何在代码中实现
- **预期改进目标**：量化的目标值

#### StockSelectionStrategy 子类的常见优化类别：
- 风险控制（波动率过滤、止损机制、回撤控制）
- 选股质量（基本面评分、多因子模型）
- 择时（调仓频率、动量确认）
- 组合构建（行业分散、仓位管理）
- 成本控制（换手率限制、交易成本意识）

#### StrategyLogic 子类（轮动/择时策略）的常见优化类别：
- 交易成本控制（换仓阈值、最小持仓天数）
- 风险过滤（波动率过滤、趋势过滤）
- 信号质量（动量确认、多周期验证）
- 避险机制（债券替代空仓、回撤控制）
- 止损机制（固定百分比止损、追踪止损）

### 阶段三：独立实施并回测每项优化

每项优化的实施步骤：

1. **在 `params` 元组中添加参数**，默认值设为禁用状态（如 `None`、`False`、`0`）
2. **实现逻辑**，作为独立方法或主逻辑中的条件分支
3. **运行回测**，使用项目的回测命令或程序化脚本
4. **记录结果**，保存为 JSON 文件到优化结果目录
5. **对比**基线夏普比率

#### 回测命令模板

```bash
python main.py --mode backtest --strategy <策略名> --period 1d --pool <股票池> --start <起始日期> --end <结束日期> --debug
```

#### 程序化回测模板（StockSelectionStrategy）

在策略目录下的 `optimization/` 目录中创建 `run_optimization.py` 脚本：

```python
import os
import sys
import json
import datetime
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = '<策略注册名>'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool='中证1000', start_date='2020-04-28', end_date='2026-04-28'):
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
        metrics['label'] = label
        metrics['extra_params'] = extra_params
        metrics['timestamp'] = datetime.datetime.now().isoformat()
    else:
        metrics['label'] = label
        metrics['error'] = 'No result'

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
```

#### 程序化回测模板（StrategyLogic）

StrategyLogic 基类策略（如 ETF 轮动）不使用 `load_financial_data` 和 `add_stock_selection_strategy`，改用 `add_strategy`：

```python
import os
import sys
import json
import datetime
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config

STRATEGY_NAME = '<策略注册名>'
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              start_date='2020-04-28', end_date='2026-04-28'):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1d'
    config['start_date'] = start_date
    config['end_date'] = end_date
    config.setdefault('benchmark', '000300.SH')

    merged_kwargs = dict(default_kwargs)
    if extra_params:
        merged_kwargs.update(extra_params)

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(**config)
    api.add_strategy(strategy_class, **merged_kwargs)
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
        metrics['label'] = label
        metrics['extra_params'] = extra_params
        metrics['timestamp'] = datetime.datetime.now().isoformat()
    else:
        metrics['label'] = label
        metrics['error'] = 'No result'

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
```

**关键差异**：
- StockSelectionStrategy：`api.load_financial_data(sector=pool)` + `api.add_stock_selection_strategy()`
- StrategyLogic：`api.add_strategy()` （无 load_financial_data）
- StrategyLogic 的 PROJECT_ROOT 多一层 `dirname`（因策略在 `strategies_for_vip/` 子目录下）
- 两种模板都必须处理 `annual_ret` 可能为 `complex` 类型的问题（`if isinstance(annual_ret, complex): annual_ret = annual_ret.real`）
- 两种模板都应调用 `api.set_no_record(True)` 避免优化回测写入正式结果目录

### 阶段四：评估与筛选

**保留标准**：夏普比率相对基线提升 >= 5%

| 结果 | 处理方式 |
|------|---------|
| 夏普提升 >= 5% | 保留并整合到策略中 |
| 0% <= 提升 < 5% | 放弃（收益不足） |
| 负向提升 | 放弃并记录失败原因 |

### 阶段五：硬逻辑与过度拟合审查

对阶段四中夏普提升 >= 5% 的有效优化，逐一进行硬逻辑强度和过度拟合风险审查。**任何一项审查不通过的优化，必须降级为"有条件有效"或直接放弃。**

#### 5.1 硬逻辑强度检查

评估优化的逻辑是否具有坚实的因果基础，而非仅仅是统计巧合。

| 检查维度 | 评估标准 | 不通过判定 |
|---------|---------|-----------|
| **逻辑因果链** | 优化是否有清晰的"条件→动作→结果"因果链？能否用一句话解释为什么这个优化应该有效？ | 无法给出合理解释，或解释依赖"历史数据就是这样" |
| **经济合理性** | 优化效果能否被金融理论解释（风险溢价、市场微观结构、行为金融等）？ | 改进仅在统计上显著，但无经济学逻辑支撑 |
| **逻辑独立性** | 优化是否提供了新的信息维度，而非已有逻辑的变相重复？ | 与已有参数高度相关（如同时有"5日动量"和"10日动量"） |
| **极端场景稳健性** | 逻辑在极端市场（暴涨暴跌、流动性枯竭）下是否仍然成立？ | 仅在常态市场有效，极端场景下逻辑失效或反向 |
| **可解释的交易行为** | 优化导致的交易行为变化是否可解释、可预期？ | 优化后产生了无法解释的频繁交易或异常持仓 |

**硬逻辑强度评级**：

| 评级 | 条件 | 处理方式 |
|------|------|---------|
| A（强） | 全部5项通过 | 正常进入组合优化阶段 |
| B（中） | 通过4项，仅极端场景或可解释性稍弱 | 进入组合优化，但在报告中标注风险 |
| C（弱） | 通过3项及以下 | 降级为"有条件有效"，需补充样本外验证才能进入组合 |

#### 5.2 过度拟合检测

通过实证方法检验优化是否对历史数据过度拟合。

**检测方法一：样本外验证（必做）**

将回测区间分为前后两段，在前半段（样本内）验证优化效果，在后半段（样本外）检验是否仍然有效：

```python
def run_out_of_sample_test(strategy_name, extra_params, label,
                            full_start, full_end, split_ratio=0.5):
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(full_start, '%Y-%m-%d')
    end_dt = datetime.strptime(full_end, '%Y-%m-%d')
    total_days = (end_dt - start_dt).days
    split_dt = start_dt + timedelta(days=int(total_days * split_ratio))
    split_date = split_dt.strftime('%Y-%m-%d')

    in_sample = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=extra_params,
        label=f'{label}_is', start_date=full_start, end_date=split_date)

    out_sample = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=extra_params,
        label=f'{label}_oos', start_date=split_date, end_date=full_end)

    baseline_is = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=None,
        label='baseline_is', start_date=full_start, end_date=split_date)

    baseline_oos = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=None,
        label='baseline_oos', start_date=split_date, end_date=full_end)

    is_improvement = (in_sample.get('sharpe_ratio', 0) - baseline_is.get('sharpe_ratio', 0)) / abs(baseline_is.get('sharpe_ratio', 1)) * 100
    oos_improvement = (out_sample.get('sharpe_ratio', 0) - baseline_oos.get('sharpe_ratio', 0)) / abs(baseline_oos.get('sharpe_ratio', 1)) * 100

    return {
        'in_sample_sharpe': in_sample.get('sharpe_ratio', 0),
        'out_sample_sharpe': out_sample.get('sharpe_ratio', 0),
        'baseline_is_sharpe': baseline_is.get('sharpe_ratio', 0),
        'baseline_oos_sharpe': baseline_oos.get('sharpe_ratio', 0),
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': oos_improvement / is_improvement if is_improvement != 0 else 0,
    }
```

**样本外判定标准**：

| 衰减比（OOS/IS） | 判定 | 处理方式 |
|-----------------|------|---------|
| >= 0.5 | 样本外仍保持至少50%的改进 | 通过，优化稳健 |
| 0.2 ~ 0.5 | 样本外效果大幅衰减 | 有条件通过，需降低参数权重或收紧阈值 |
| < 0.2 或为负 | 样本外几乎无效或反向 | **不通过**，高度疑似过度拟合，必须放弃 |

**检测方法二：参数敏感性分析（必做）**

对优化的核心参数进行 ±10%、±20% 的扰动测试，检验优化效果是否对参数值高度敏感：

```python
def run_parameter_sensitivity_test(strategy_name, param_name, param_value,
                                    label, perturbations=[-0.2, -0.1, 0.1, 0.2]):
    results = {}
    base_result = run_backtest_with_params(
        strategy_name=strategy_name,
        extra_params={param_name: param_value},
        label=f'{label}_base')

    for delta in perturbations:
        perturbed_value = param_value * (1 + delta)
        if isinstance(param_value, int):
            perturbed_value = int(round(perturbed_value))
            if perturbed_value == 0:
                perturbed_value = 1
        result = run_backtest_with_params(
            strategy_name=strategy_name,
            extra_params={param_name: perturbed_value},
            label=f'{label}_perturb_{delta:+.0%}')
        results[f'perturb_{delta:+.0%}'] = {
            'param_value': perturbed_value,
            'sharpe_ratio': result.get('sharpe_ratio', 0),
        }

    base_sharpe = base_result.get('sharpe_ratio', 0)
    sharpe_values = [r['sharpe_ratio'] for r in results.values()]
    sharpe_std = (sum((s - base_sharpe) ** 2 for s in sharpe_values) / len(sharpe_values)) ** 0.5
    sharpe_range = max(sharpe_values) - min(sharpe_values)

    return {
        'base_sharpe': base_sharpe,
        'base_param': param_value,
        'perturbation_results': results,
        'sharpe_std': sharpe_std,
        'sharpe_range': sharpe_range,
        'sensitivity_ratio': sharpe_range / abs(base_sharpe) if base_sharpe != 0 else float('inf'),
    }
```

**参数敏感性判定标准**：

| 敏感度比率（range/|base|） | 判定 | 处理方式 |
|--------------------------|------|---------|
| < 0.3 | 参数鲁棒，优化可信 | 通过 |
| 0.3 ~ 0.6 | 参数较敏感 | 有条件通过，建议选择参数区间的中间值而非最优值 |
| > 0.6 | 参数高度敏感，优化可能仅对特定值有效 | **不通过**，高度疑似过度拟合 |

**检测方法三：时间分段稳定性（必做）**

将回测区间按年分段，检验优化在各年度是否一致有效：

```python
def run_temporal_stability_test(strategy_name, extra_params, label,
                                 full_start, full_end):
    from datetime import datetime
    start_year = datetime.strptime(full_start, '%Y-%m-%d').year
    end_year = datetime.strptime(full_end, '%Y-%m-%d').year

    yearly_results = []
    for year in range(start_year, end_year + 1):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        if year_start < full_start:
            year_start = full_start
        if year_end > full_end:
            year_end = full_end

        opt_result = run_backtest_with_params(
            strategy_name=strategy_name, extra_params=extra_params,
            label=f'{label}_{year}', start_date=year_start, end_date=year_end)

        base_result = run_backtest_with_params(
            strategy_name=strategy_name, extra_params=None,
            label=f'baseline_{year}', start_date=year_start, end_date=year_end)

        opt_sharpe = opt_result.get('sharpe_ratio', 0)
        base_sharpe = base_result.get('sharpe_ratio', 0)
        improvement = opt_sharpe - base_sharpe

        yearly_results.append({
            'year': year,
            'opt_sharpe': opt_sharpe,
            'base_sharpe': base_sharpe,
            'improvement': improvement,
            'is_positive': improvement > 0,
        })

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency_ratio = positive_years / total_years if total_years > 0 else 0

    return {
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': consistency_ratio,
    }
```

**时间稳定性判定标准**：

| 一致性比率（正改进年数/总年数） | 判定 | 处理方式 |
|-------------------------------|------|---------|
| >= 0.7 | 优化效果时间稳定 | 通过 |
| 0.5 ~ 0.7 | 部分年份无效 | 有条件通过，需分析无效年份的市场特征 |
| < 0.5 | 多数年份无效，改进集中在少数年份 | **不通过**，优化可能仅对特定行情有效 |

#### 5.3 综合审查结论

对每项有效优化，汇总三项检测结果，给出最终审查结论：

| 硬逻辑评级 | 样本外衰减比 | 参数敏感度 | 时间稳定性 | 最终结论 |
|-----------|------------|-----------|-----------|---------|
| A | >= 0.5 | < 0.3 | >= 0.7 | ✅ 强力通过，优先组合 |
| A/B | >= 0.2 | < 0.6 | >= 0.5 | ⚠️ 有条件通过，组合时降低权重 |
| 任意 | < 0.2 或 敏感度>0.6 或 稳定性<0.5 | - | - | ❌ 不通过，放弃该优化 |

**重要**：即使优化在阶段四中夏普提升 >= 5%，如果在本阶段审查不通过，也必须放弃。硬逻辑和过度拟合审查是比夏普比率更根本的质量门槛。

### 阶段六：组合有效优化

1. 测试所有有效优化的组合效果
2. **检测参数冲突**：某些优化在单项测试中有效，但组合后可能冲突（详见下方"参数冲突检测"）
3. 验证组合夏普比率 >= 最佳单项优化
4. 用有效优化更新策略默认参数
5. 从策略文件中删除所有无效优化代码

#### 参数冲突检测

在组合优化前，必须分析有效优化之间是否存在逻辑冲突：

| 冲突类型 | 示例 | 原因 |
|---------|------|------|
| 换仓限制冲突 | 最小持仓天数 + 波动率过滤 | 波动率过滤需及时调仓，最小持仓天数阻止调仓 |
| 止损与空仓冲突 | 止损 + 轮动空仓机制 | 轮动策略已有空仓机制，止损造成双重卖出 |
| 过滤叠加冲突 | 多重动量确认 | 过多过滤条件消除有效信号 |

**冲突检测方法**：分析两个优化的触发条件和执行动作是否存在互斥场景。如果优化A要求"延迟行动"而优化B要求"立即行动"，则存在冲突。

### 阶段七：清理与文档

1. **清理策略代码**：删除所有无效优化的参数和方法
2. **更新策略目录下的 `readme.md`**：仅反映保留的优化内容
3. **生成 HTML 优化报告**：使用 Chart.js 生成交互式报告（详见下方"HTML报告生成模板"）
4. **运行验证回测**：清理代码后重新回测，确认结果与优化预期一致

#### HTML 报告生成模板

在策略目录下的 `optimization/` 目录中创建 `generate_report.py` 脚本，生成包含 Chart.js 交互式图表的 HTML 报告。

报告应包含以下7个章节：

1. **优化总览**：4张核心指标卡片（夏普变化、收益变化、回撤变化、年化变化）
2. **策略说明**：原始逻辑与优化后新增逻辑的说明
3. **单项优化回测**：详细数据表 + 夏普/收益/提升幅度3张柱状图
4. **硬逻辑与过度拟合审查**：硬逻辑评级表 + 样本外验证/参数敏感性/时间稳定性3项检测结果 + 综合审查结论表
5. **组合优化回测**：组合方案对比表 + 夏普/收益2张柱状图
6. **核心发现**：深度洞察（交易成本、协同效应、参数冲突、无效优化分析、过度拟合风险警示）
7. **优化前后对比**：雷达图综合对比 + 最终采纳/未采纳参数清单

```python
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'optimization_results')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'optimization_report.html')

ALL_RESULTS = [
    ('baseline', '基线策略', None),
    ('opt01_xxx', '优化1名称', {'param1': value1}),
    # ... 按实际优化方案填写
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

    single_opts = results[1:N]  # N = 单项优化数量 + 1
    combined_opts = results[N:]

    # 预计算所有图表数据（避免在 HTML 模板中做复杂计算）
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

    # HTML 模板 - 使用 Chart.js CDN
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>策略优化分析报告</title>
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

<h1>策略优化分析报告</h1>
<p style="text-align:center;color:#94a3b8;margin-bottom:40px;">回测区间与参数信息</p>

<!-- 一、优化总览 -->
<h2>一、优化总览</h2>
<div class="summary-cards">
    <div class="card"><div class="card-label">夏普比率</div><div class="card-value up">基线 → 优化后</div></div>
    <div class="card"><div class="card-label">总收益率</div><div class="card-value up">基线 → 优化后</div></div>
    <div class="card"><div class="card-label">最大回撤</div><div class="card-value up">基线 → 优化后</div></div>
    <div class="card"><div class="card-label">年化收益率</div><div class="card-value up">基线 → 优化后</div></div>
</div>

<!-- 二、策略说明 -->
<h2>二、策略说明</h2>
<div class="insight-box">
<h3>原始策略逻辑</h3><ul><li>...</li></ul>
<h3>优化后新增逻辑</h3><ul><li>...</li></ul>
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
<table>
<thead><tr><th>优化方向</th><th>硬逻辑评级</th><th>样本外衰减比</th><th>参数敏感度</th><th>时间稳定性</th><th>审查结论</th></tr></thead>
<tbody><!-- 按实际审查结果填写 --></tbody>
</table>
<div class="chart-grid">
    <div class="chart-box"><canvas id="chartOOSDecay"></canvas></div>
    <div class="chart-box"><canvas id="chartParamSensitivity"></canvas></div>
</div>
<div class="chart-box full"><canvas id="chartTemporalStability"></canvas></div>
<div class="insight-box">
<h3>过度拟合风险警示</h3><ul><!-- 标注高风险优化及原因 --></ul>
</div>

<!-- 五、组合优化回测结果 -->
<h2>五、组合优化回测结果</h2>
<!-- 类似结构 -->

<!-- 六、核心发现 -->
<h2>六、核心发现</h2>
<div class="insight-box"><!-- 深度洞察 --></div>

<!-- 七、优化前后对比 -->
<h2>七、优化前后对比</h2>
<!-- 雷达图 + 参数清单 -->

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
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report generated: {OUTPUT_FILE}')


if __name__ == '__main__':
    generate_report()
```

**HTML 报告生成注意事项**：
- 所有图表数据必须在 HTML 模板字符串之前预计算为 Python 变量，不能在 f-string 模板内做列表推导等复杂计算
- 使用 `json.dumps()` 将 Python 列表转为 JavaScript 数组，需设置 `ensure_ascii=False` 以支持中文标签
- Chart.js 通过 CDN 引入：`https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`
- 报告输出到 `optimization/optimization_report.html`（与优化脚本同目录），便于统一管理
- 颜色编码规则：绿色 `rgba(52,211,153,...)` = 有效，红色 `rgba(248,113,113,...)` = 无效，蓝色 `rgba(96,165,250,...)` = 最优组合，黄色 `#fbbf24` = 基线参考线

## 核心原则

1. **独立测试**：每项优化必须独立对比基线测试，不能与其他优化混合测试
2. **参数隔离**：新参数默认禁用（None/False/0），确保基线行为不变
3. **不可替代核心机制**：行业分散、基本面过滤等核心风控机制不应被优化替代，而应被增强
4. **简单优于复杂**：优先使用简单直接的风控手段，而非复杂评分模型
5. **风控优于选股**：对大多数策略而言，最大改进来自风险控制（波动率过滤、止损），而非更好的选股
6. **全程记录**：记录每项优化成功或失败的原因，包括具体失败原因
7. **冲突检测**：组合优化前必须分析参数间是否存在逻辑冲突，避免1+1<1
8. **验证回测**：清理代码后必须重新回测，确认结果与优化预期一致
9. **硬逻辑优先**：优化的逻辑因果链和经济合理性比统计显著性更重要，无法解释"为什么有效"的优化不可采纳
10. **过度拟合防范**：每项有效优化必须通过样本外验证、参数敏感性分析和时间稳定性测试，三项中任何一项不通过即放弃

## 策略文件结构

本项目中的策略遵循以下模式：

### StockSelectionStrategy 子类（选股策略）

```python
from collections import defaultdict
from typing import Dict, List
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('strategy_name', default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2016-01-01', 'end_date': '2026-04-17'})
class MyStrategy(StockSelectionStrategy):
    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        # ... 过滤步骤
        return selected
```

### StrategyLogic 子类（轮动/择时策略）

```python
from core.strategy_logic import StrategyLogic, BarData, OrderInfo, TradeInfo
from strategies import register_strategy
from .config import ETF_CODES


@register_strategy('etf_rotation',
                   backtest_config={'cash': 200000, 'commission': 0.0005,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28'})
class ETFRotationStrategy(StrategyLogic):
    params = (
        ('return_period', 20),
        ('max_volatility', 0.03),
        ('switch_threshold', 0.05),
    )

    def on_bar(self, bar: BarData):
        # K线到达时执行轮动逻辑
        pass

    def on_order(self, order: OrderInfo):
        # 委托状态变化处理
        pass

    def on_trade(self, trade: TradeInfo):
        # 成交回报处理
        pass
```

### 可用的基类方法

#### StockSelectionStrategy 方法

| 方法 | 说明 |
|------|------|
| `self.get_stock_pool()` | 获取股票池列表 |
| `self.get_financial_field(stock, table, field)` | 获取股票财务数据 |
| `self.get_current_price(stock)` | 获取当前价格 |
| `self.get_close_prices(stock, count)` | 获取最近N日收盘价 |
| `self.get_industry(stock)` | 获取行业分类 |
| `self.get_symbols()` | 获取所有可用标的 |
| `self.log(msg)` | 记录日志 |

#### StrategyLogic 方法

| 方法 | 说明 |
|------|------|
| `self.get_close_prices(symbol)` | 获取标的收盘价序列 |
| `self.get_current_price(symbol)` | 获取标的当前价格 |
| `self.get_position_size(symbol)` | 获取标的持仓数量 |
| `self.get_cash()` | 获取可用现金 |
| `self.get_current_date()` | 获取当前日期 |
| `self.buy(symbol, price, size)` | 买入 |
| `self.sell(symbol, price, size)` | 卖出 |
| `self.log(msg)` | 记录日志 |

## 输出目录结构

优化结果保存在策略目录下的 `optimization/` 子目录中：

```
strategies/<策略目录>/
├── <策略名>.py                  # 策略主文件
├── readme.md                    # 策略文档
├── backtest_results/            # 回测结果记录
│   └── YYYYMMDD_HHMMSS_<策略名>.json
└── optimization/                # 优化案例目录
    ├── run_optimization.py          # 自动化回测运行脚本
    ├── generate_report.py           # HTML报告生成脚本
    ├── optimization_report.html     # HTML优化报告
    └── optimization_results/        # 优化回测结果
        ├── baseline.json            # 基线结果
        ├── opt01_xxx.json           # 优化1结果
        ├── opt02_xxx.json           # 优化2结果
        ├── ...
        ├── optNN_combined.json      # 组合优化结果
        ├── opt01_xxx_oos.json       # 优化1样本外验证结果
        ├── opt01_xxx_is.json        # 优化1样本内验证结果
        ├── opt01_xxx_perturb_*.json # 优化1参数敏感性测试结果
        ├── opt01_xxx_YYYY.json      # 优化1年度稳定性测试结果
        └── review_summary.json      # 硬逻辑与过度拟合审查汇总
```

### 策略目录与注册名映射

| 策略注册名 | 策略目录 | 基类 |
|-----------|---------|------|
| `small_cap` | `strategies/small_cap_strategy/` | StockSelectionStrategy |
| `high_dividend` | `strategies/high_dividend_strategy/` | StockSelectionStrategy |
| `double_ma` | `strategies/example_strategy/` | StrategyLogic |
| `fundamental_roe` | `strategies/fundamental_strategy/` | StockSelectionStrategy |
| `fundamental_growth` | `strategies/fundamental_strategy/` | StockSelectionStrategy |
| `etf_rotation` | `strategies_for_vip/etf_rotation_strategy/` | StrategyLogic |

### 获取策略目录

通过 `get_strategy_dir()` 函数获取策略的绝对路径：

```python
from strategies import get_strategy_dir
strategy_dir = get_strategy_dir('small_cap')
# => 'E:\...\strategies\small_cap_strategy'
```

## 优化报告模板

报告应包含以下内容：

1. **优化概览**：策略名称、回测区间、股票池、核心指标、基线与优化后对比
2. **结果汇总表**：所有优化的夏普比率、变化百分比、总收益、最大回撤、结论
3. **详细分析**：每项优化的方向、实现方式、预期目标、实际结果、成功/失败原因
4. **硬逻辑与过度拟合审查**：每项有效优化的硬逻辑评级、样本外衰减比、参数敏感度、时间稳定性、综合审查结论
5. **组合优化结果**：基线与组合优化的对比表
6. **最终参数**：更新后的策略参数值
7. **经验总结**：优化过程中的关键收获

## 已验证的有效优化

### 选股策略（StockSelectionStrategy）

基于小市值策略优化经验，以下方案已证明具有一致的有效性：

| 优化方案 | 类型 | 典型影响 | 实现方式 |
|---------|------|---------|---------|
| 波动率过滤 | 事前风控 | 夏普+10~15% | 过滤日波动率超过阈值的股票 |
| 止损机制 | 事后风控 | 夏普+10~15% | 调仓时剔除亏损超过阈值的股票 |
| 波动率+止损组合 | 双层风控 | 夏普+20~25% | 以上两项组合使用 |

以下方案对小市值策略已验证无效：
- 成交量确认（消除了低流动性溢价）
- 多因子评分（替代了行业分散机制）
- 行业动量加权（追逐短期趋势导致追高）
- 宽松阈值的财务质量评分（无过滤效果）
- 月度调仓频率下的换手率控制（从未触发）

### 轮动策略（StrategyLogic）

基于 ETF 轮动策略优化经验，以下方案已证明有效：

| 优化方案 | 类型 | 典型影响 | 实现方式 |
|---------|------|---------|---------|
| 波动率过滤 | 事前风控 | 夏普+15% | 过滤日波动率超过3%的标的 |
| 换仓阈值 | 成本控制 | 夏普+28% | 新标的需比当前持仓收益高出阈值才换仓 |
| 波动率+换仓阈值组合 | 风控+成本 | 夏普+48% | 以上两项组合，存在协同效应 |

以下方案对 ETF 轮动策略已验证无效：
- 止损机制（5%/8%）：轮动策略已有空仓机制，止损造成双重卖出；5%止损反而有害，错过反弹
- 回撤控制（15%）：灾难性亏损，反复清仓再买入导致巨额交易成本
- 动量确认/双周期动量/动量加速：额外过滤条件消除有效信号，降低策略捕捉能力
- 趋势过滤MA60：改善微弱（+1.3%），不值得增加复杂度
- 国债避险：收益几乎不变，避险效果微弱
- 最小持仓天数：单独有效但与波动率过滤冲突，组合后效果大幅下降

### 跨策略通用经验

| 经验 | 说明 |
|------|------|
| 交易成本是最大敌人 | 所有能有效减少换仓的优化都显著提升夏普比率 |
| 波动率过滤普适有效 | 对选股和轮动策略均有效，是首选优化方向 |
| 止损效果因策略而异 | 对选股策略有效，对已有空仓机制的轮动策略无效甚至有害 |
| 组合优化需检测冲突 | 单项有效的优化组合后可能冲突，必须验证 |
| 简单阈值优于复杂模型 | 换仓阈值（5%）比动量加速等复杂模型效果更好 |
