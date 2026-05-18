# 回测结果记录

框架支持自动记录每次回测的完整结果，便于追溯和对比。

## 核心功能

- **自动记录**：每次回测自动保存完整结果到本地 JSON 文件（指标、交易日志、净值曲线、基准曲线）
- **全局索引**：维护 `index.json` 索引，支持按策略名检索历史回测
- **结果对比**：支持多组回测结果对比分析，按夏普比率排序
- **HTML 报告**：生成基于 Plotly 的交互式 HTML 可视化报告（净值曲线、回撤曲线、指标对比、交易统计）

## 存储位置

回测结果优先保存到策略目录下的 `backtest_results/`，便于与策略代码统一管理：

```
strategies/<策略目录>/
├── <策略名>.py
└── backtest_results/
    └── 20260428_143000_small_cap.json
```

如果策略目录不可写，则回退到全局 `backtest_results/` 目录。

## 使用方式

```python
from utils.backtest_recorder import BacktestRecorder

recorder = BacktestRecorder()

# 记录回测结果
run_id = recorder.record(result, strategy_name='small_cap', config=config)

# 列举历史回测
records = recorder.list_records(strategy_name='small_cap')

# 对比多组回测
comparison = recorder.compare(['run_id_1', 'run_id_2'])

# 生成 HTML 可视化报告
recorder.generate_report(['run_id_1', 'run_id_2'])
```
