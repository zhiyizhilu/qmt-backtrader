# 架构设计

## 策略架构

框架采用 **策略逻辑与执行环境解耦** 的架构设计：

```
StrategyLogic（策略逻辑基类）
├── on_bar()       → K线事件
├── on_order()     → 委托事件
├── on_trade()     → 成交事件
├── 数据适配器接口  → get_current_price / get_close_prices / ...
└── 执行器接口     → buy / sell / cancel_order / ...

    ├── BaseStrategy（回测适配层，桥接 Backtrader）
    └── QMTExecutor（实盘适配层，桥接 QMT）
```

- **StrategyLogic**：纯策略逻辑，不依赖任何执行环境
- **BaseStrategy**：将 StrategyLogic 适配到 Backtrader 框架的回测适配层
- **StockSelectionStrategy**：选股策略基类，只需实现 `select_stocks()` 方法，框架自动处理调仓

## 数据架构

```
DataProcessor（数据处理器基类）
├── OpenDataProcessor（主数据源：腾讯财经/AkShare，历史数据丰富，包含行情、财务、QVIX 等）
├── QMTDataProcessor（补充数据源：QMT 数据补足 + 实时交易接口）
└── CSVDataProcessor（CSV 文件数据源）
```

- **OpenData 为主**：历史行情数据丰富，覆盖完整回测区间
- **QMT 补充**：QMT 数据补足缺口，同时提供实时交易接口
- **历史成分股**：内置聚宽下载的指数和行业历史成分股 CSV，回测时使用对应时点的真实成分股

## 执行器架构

```
StrategyExecutor（执行器基类）
├── BacktestExecutor（回测执行器，通过 Backtrader 执行）
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
