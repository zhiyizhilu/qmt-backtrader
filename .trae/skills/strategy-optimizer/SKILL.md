---
name: "strategy-optimizer"
description: "系统性地通过回测优化量化交易策略。当用户需要优化、改进或调优策略性能指标（如夏普比率）时调用此技能。"
---

# 策略优化器

在 qmt_backtrader 框架中系统性优化量化交易策略。本技能提供结构化的工作流程：提出优化建议、独立实施、逐一回测、仅保留有效改进。

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
2. 读取 `example/` 目录下对应的策略文档
3. 运行基线回测，记录当前性能指标
4. 确定核心评估指标（默认：夏普比率）

### 阶段二：提出优化建议

提出10项具体优化建议，每项必须包含：
- **优化方向**：改进哪个方面
- **技术实现路径**：如何在代码中实现
- **预期改进目标**：量化的目标值

StockSelectionStrategy 子类的常见优化类别：
- 风险控制（波动率过滤、止损机制、回撤控制）
- 选股质量（基本面评分、多因子模型）
- 择时（调仓频率、动量确认）
- 组合构建（行业分散、仓位管理）
- 成本控制（换手率限制、交易成本意识）

### 阶段三：独立实施并回测每项优化

每项优化的实施步骤：

1. **在 `params` 元组中添加参数**，默认值设为禁用状态（如 `None`、`False`、`0`）
2. **实现逻辑**，作为独立方法或 `select_stocks()` 中的条件分支
3. **运行回测**，使用项目的回测命令
4. **记录结果**，保存为 JSON 文件到优化结果目录
5. **对比**基线夏普比率

#### 回测命令模板

```bash
python main.py --mode backtest --strategy <策略名> --period 1d --pool <股票池> --start <起始日期> --end <结束日期> --debug
```

#### 程序化回测模板

在结果目录中创建 `run_optimization.py` 脚本：

```python
import os
import sys
import json
import datetime
import traceback

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config
from core.data.index_constituent import IndexConstituentManager

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_backtest_with_params(strategy_name, extra_params=None, label='test',
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
            metrics['annual_return_pct'] = annual_ret * 100
            metrics['trading_days'] = days
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

### 阶段四：评估与筛选

**保留标准**：夏普比率相对基线提升 >= 5%

| 结果 | 处理方式 |
|------|---------|
| 夏普提升 >= 5% | 保留并整合到策略中 |
| 0% <= 提升 < 5% | 放弃（收益不足） |
| 负向提升 | 放弃并记录失败原因 |

### 阶段五：组合有效优化

1. 测试所有有效优化的组合效果
2. 验证组合夏普比率 >= 最佳单项优化
3. 用有效优化更新策略默认参数
4. 从策略文件中删除所有无效优化代码

### 阶段六：清理与文档

1. **清理策略代码**：删除所有无效优化的参数和方法
2. **更新策略文档**：仅反映保留的优化内容
3. **生成对比图表**：使用 matplotlib 可视化结果
4. **撰写优化报告**：包含所有结果、分析和经验总结

#### 可视化模板

```python
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 从 optimization_results/ 读取所有 JSON 结果
# 创建 2x2 子图：夏普比率、总收益、最大回撤、夏普变化百分比
# 颜色编码：绿色=有效、红色=无效、橙色=边际、灰色=基线
# 保存到 optimization_results/optimization_comparison.png
```

## 核心原则

1. **独立测试**：每项优化必须独立对比基线测试，不能与其他优化混合测试
2. **参数隔离**：新参数默认禁用（None/False/0），确保基线行为不变
3. **不可替代核心机制**：行业分散、基本面过滤等核心风控机制不应被优化替代，而应被增强
4. **简单优于复杂**：优先使用简单直接的风控手段，而非复杂评分模型
5. **风控优于选股**：对大多数策略而言，最大改进来自风险控制（波动率过滤、止损），而非更好的选股
6. **全程记录**：记录每项优化成功或失败的原因，包括具体失败原因

## 策略文件结构

本项目中的策略遵循以下模式：

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
        # ... 其他参数
    )

    def select_stocks(self) -> List[str]:
        # 主要选股逻辑
        pool = self.get_stock_pool()
        # ... 过滤步骤
        return selected
```

### 可用的基类方法

实现优化时可使用 `StockSelectionStrategy` 的以下方法：

| 方法 | 说明 |
|------|------|
| `self.get_stock_pool()` | 获取股票池列表 |
| `self.get_financial_field(stock, table, field)` | 获取股票财务数据 |
| `self.get_current_price(stock)` | 获取当前价格 |
| `self.get_close_prices(stock, count)` | 获取最近N日收盘价 |
| `self.get_industry(stock)` | 获取行业分类 |
| `self.get_symbols()` | 获取所有可用标的 |
| `self.log(msg)` | 记录日志 |

## 输出目录结构

```
example/<策略名>自动优化案例/
├── run_optimization.py          # 自动化回测运行脚本
├── plot_comparison.py           # 可视化脚本
├── optimization_comparison.png  # 对比图表
├── 优化报告.md                   # 完整优化报告
├── baseline.json                # 基线结果
├── opt01_xxx.json               # 优化1结果
├── opt02_xxx.json               # 优化2结果
├── ...
└── optNN_combined.json          # 组合优化结果
```

## 优化报告模板

报告应包含以下内容：

1. **优化概览**：策略名称、回测区间、股票池、核心指标、基线与优化后对比
2. **结果汇总表**：所有优化的夏普比率、变化百分比、总收益、最大回撤、结论
3. **详细分析**：每项优化的方向、实现方式、预期目标、实际结果、成功/失败原因
4. **组合优化结果**：基线与组合优化的对比表
5. **最终参数**：更新后的策略参数值
6. **经验总结**：优化过程中的关键收获

## 已验证的有效优化

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
