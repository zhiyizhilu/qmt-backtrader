# 架构设计

## 自研回测引擎

框架使用自研回测引擎替代 Backtrader，核心优化点：

1. **预加载 numpy 数组**：将 DataFrame 预加载为 numpy 数组，替代 Backtrader lines 对象，运行时通过索引直接访问
2. **预计算统一时间轴**：替代 Backtrader 的运行时逐 bar 对齐，将多数据源时间对齐从 O(N*M) 优化为 O(N)
3. **简化订单执行**：COC（Close-On-Close）立即成交模式，替代 Backtrader 的状态机

```
engine/
├── engine.py        # BacktestEngine 回测引擎核心
├── broker.py        # SimulatedBroker 模拟经纪商（Position/Order/Trade）
├── data_feed.py     # ArrayDataFeed 基于 numpy 数组的高性能数据源
├── timeline.py      # Timeline 统一时间轴（预计算索引映射）
├── adapter.py       # EngineDataAdapter / EngineExecutor（适配器/执行器）
└── result.py        # EngineResult 回测结果容器
```

**回测流程**：

```
BacktestEngine.run()
├── _prepare() → 初始化 Timeline / DataAdapter / Executor
├── 遍历 Timeline 全局索引
│   ├── DataAdapter.update(idx) → 更新当前索引和价格缓存
│   ├── Broker.getvalue() → 计算当前权益
│   ├── StrategyLogic.on_bar(bar) → 执行策略逻辑
│   └── Executor.execute_buy/sell() → 提交订单到 Broker
└── 收集 equity_history / trade_records → EngineResult
```

**支持日线和分钟线**：
- 日线模式：每个全局索引对应一个交易日，每天执行一次 `on_bar`
- 分钟线模式：每个全局索引对应一根分钟K线，每分钟执行一次 `on_bar`

## 策略架构

框架采用 **策略逻辑与执行环境解耦** 的架构设计：

```
StrategyLogic（策略逻辑基类）
├── on_bar()       → K线事件
├── on_order()     → 委托事件
├── on_trade()     → 成交事件
├── 数据适配器接口  → get_current_price / get_close_prices / ...
└── 执行器接口     → buy / sell / cancel_order / ...

    ├── EngineExecutor（回测适配层，桥接自研回测引擎）
    └── QMTExecutor（实盘适配层，桥接 QMT）
```

- **StrategyLogic**：纯策略逻辑，不依赖任何执行环境
- **EngineExecutor**：将 StrategyLogic 适配到自研回测引擎的回测适配层
- **StockSelectionStrategy**：选股策略基类，只需实现 `select_stocks()` 方法，框架自动处理调仓
  - 支持灵活的权重分配：等权（EqualWeightAllocator）、风险平价（RiskParityAllocator）、因子加权（FactorWeightAllocator）

## 数据架构

```
DataProcessor（数据处理器基类）
├── OpenDataProcessor（主数据源：腾讯财经/AkShare，历史数据丰富，包含行情、财务、QVIX 等）
├── QMTDataProcessor（补充数据源：QMT 数据补足 + 实时交易接口）
├── FutuDataProcessor（富途数据源：Futu OpenD API，支持自动增量下载）
└── CSVDataProcessor（CSV 文件数据源）
```

- **OpenData 为主**：历史行情数据丰富，覆盖完整回测区间
- **QMT 补充**：QMT 数据补足缺口，同时提供实时交易接口
- **富途补充**：富途 OpenD 提供高质量行情数据，支持自动增量下载和频率限制
- **历史成分股**：内置聚宽下载的指数和行业历史成分股 CSV，回测时使用对应时点的真实成分股

**数据源选择**：通过 `--data-source` 参数切换，可选 `qmt`（默认）、`open`、`futu`。

## 执行器架构

```
StrategyExecutor（执行器基类）
├── EngineExecutor（回测执行器，通过自研回测引擎执行）
│   └── SimulatedBroker → 模拟经纪商（COC 立即成交）
└── QMTExecutor（实盘执行器，通过 QMT 交易接口执行）
    ├── virtual_book → VirtualBook（虚拟持仓簿，策略级隔离）
    └── order_router → OrderRouter（订单路由，回调分发）
```

## 多策略实例架构

当同一账户下运行多个策略时，框架通过虚拟簿记实现策略级隔离：

```
QMTAPI（一个账户一个实例）
├── OrderRouter（订单路由：订单ID → 策略实例映射）
├── Strategy A
│   ├── QMTExecutor → VirtualBook A（独立持仓/资金视图）
│   └── on_order / on_trade ← OrderRouter 路由
├── Strategy B
│   ├── QMTExecutor → VirtualBook B（独立持仓/资金视图）
│   └── on_order / on_trade ← OrderRouter 路由
└── Reconciler（对账器：校验所有 VirtualBook 之和 = 账户实际）
```

**核心组件**：

| 组件 | 职责 |
|------|------|
| **VirtualBook** | 维护策略实例的独立持仓和资金视图，交易记录是持仓的唯一真相来源 |
| **OrderRouter** | 维护订单ID与策略实例的映射，将 QMT 委托/成交回调正确分发到对应策略 |
| **Reconciler** | 校验所有策略簿记之和与账户实际状态的一致性，偏差自动校准 |
| **StrategyInstanceManager** | 从 JSON 配置加载多策略实例，按账户分组初始化，协调对账 |

**数据流**：

```
下单: Strategy → QMTExecutor.execute_buy() → QMTTrader.buy()
                                          ↓
                              VirtualBook.on_buy_submitted()
                              OrderRouter.register_order()

回调: QMT → _on_qmt_trade() → OrderRouter.route_trade()
                                    ↓
                        VirtualBook.on_buy_filled()（更新簿记）
                        Strategy.on_trade()（通知策略）
```

**对账策略**：

| 场景 | 处理方式 |
|------|---------|
| 独占标的（仅一个策略持有） | 自动校准到实际值 |
| 共享标的 + 正偏差（如送股） | 按持仓比例分配 |
| 共享标的 + 负偏差 | 报警，需人工确认 |
| 无持有人但有实际持仓 | 报警，标记未归属 |

## 辅助模块

### 股票生命周期管理（StockLifecycleManager）

管理股票的上市/退市时间，避免对退市/未上市股票的无效数据请求：

- 数据源优先级：QMT → 腾讯财经 HTTP API → akshare
- 持久化缓存到 `.cache/lifecycle/stock_lifecycle.json`
- 缓存超过 30 天自动更新

### 权重分配器（WeightAllocator）

为选股策略提供灵活的权重分配方式：

| 分配器 | 说明 |
|--------|------|
| **EqualWeightAllocator** | 等权分配，每只股票权重相同 |
| **RiskParityAllocator** | 风险平价，基于历史波动率倒数分配权重 |
| **FactorWeightAllocator** | 因子加权，按因子得分分配权重 |

### Walk-Forward 验证（WalkForwardSplitter）

Walk-Forward 时间窗口分割器，用于策略样本外验证：

- **滚动模式**（anchor=False）：训练窗口固定长度向前滑动
- **锚定模式**（anchor=True）：训练窗口起点固定，终点向前扩展
- 输出 `(train_start, train_end, test_start, test_end)` 列表
