# 量化交易框架

一个基于 Backtrader 和 QMT 的量化交易框架。**核心目标：一套策略代码无缝切换回测、模拟交易和实盘交易**，无需为不同运行模式修改策略逻辑，极大降低从研究到上线的迁移成本。

## 功能特点

- **多模式支持**：回测、模拟交易、实盘交易、多策略实例
- **多策略实例隔离**：同账户多策略独立运行，虚拟簿记实现持仓/资金级隔离，互不干扰
- **高性能回测**：基于 Backtrader 引擎
- **QMT 集成**：支持通过 QMT 进行模拟和实盘交易
- **双数据源架构**：OpenData（腾讯财经/AkShare）为主数据源 + QMT 实时数据补充回测覆盖范围
- **历史成分股支持**：内置指数和申万行业历史成分股数据，回测时使用对应时点的真实成分股
- **智能缓存**：内存 + 磁盘双层缓存 + 缓存索引管理，支持 parquet 格式，中断后自动恢复
- **策略模板**：内置高股息策略、小市值策略等
- **参数优化**：支持网格搜索 + 策略优化工作流（自动提出优化建议、独立回测、筛选有效改进）
- **可视化**：提供基于 PyQt5 的回测结果展示、基于 Plotly 的 HTML 可视化报告、基于 Dash 的实时监控
- **选股策略**：支持基于基本面的股票选择策略，行业分散配置
- **防未来数据**：财务数据按公告日期索引，回测时只使用已披露数据
- **自动对账**：每日开盘前自动校验虚拟簿记与账户实际状态的一致性，偏差自动校准
- **回测结果记录**：自动记录每次回测的完整结果到本地 JSON，支持加载、列举、对比历史回测数据
- **AI 模式**：支持 AI 自动运行模式，跳过所有图形界面渲染，适用于自动化策略优化
- **数据预下载**：提供独立的行情/财务数据预下载脚本，支持批量下载和缓存预热

## 项目结构

```
qmt_backtrader/
├── README.md                    # 项目说明文档
├── main.py                      # 主入口文件
├── .gitignore                   # Git 忽略配置
├── download_market_data.py      # 行情数据预下载脚本（批量下载 + 缓存预热）
├── download_financial_data.py   # 财务数据预下载脚本（批量下载 + 缓存预热）
├── read_parquet.py              # Parquet 缓存文件查看工具
├── clean_old_logs.bat           # 清理过期日志文件脚本
├── 码上生财.jpg                  # 作者微信二维码
├── .cache/                      # 数据缓存目录（自动创建）
│   └── JQData/                  # 聚宽下载的历史数据
│       ├── index_constituent/   # 指数历史成分股 CSV
│       │   ├── 000016.SH.csv    # 上证50
│       │   ├── 000300.SH.csv    # 沪深300
│       │   ├── 000852.SH.csv    # 中证1000
│       │   └── 000905.SH.csv    # 中证500
│       └── industry_constituent/ # 申万一级行业历史成分股 CSV
│           ├── SW1银行.csv
│           ├── SW1电子.csv
│           └── ...              # 共31个行业
├── api/                         # API 接口
│   ├── __init__.py
│   ├── backtest_api.py          # 回测 API（基于 Backtrader）
│   ├── base_api.py              # API 基类
│   ├── instance_manager.py      # 策略实例管理器（多策略模式）
│   └── qmt_api.py               # QMT 交易 API
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── analyzer.py              # 回测分析器
│   ├── cache.py                 # 智能缓存系统（内存+磁盘+索引+增量合并）
│   ├── data/                    # 数据处理模块
│   │   ├── __init__.py
│   │   ├── base.py              # 数据处理器基类 + 缓存序列化工具
│   │   ├── csv.py               # CSV 数据处理器
│   │   ├── factory.py           # 数据处理器工厂
│   │   ├── index_constituent.py # 指数历史成分股管理器
│   │   ├── industry_constituent.py # 申万行业历史成分股管理器
│   │   ├── opendata.py          # OpenData 数据处理器（腾讯财经/AkShare，含行情、财务、QVIX 等）
│   │   └── qmt.py               # QMT 数据处理器（补充数据源 + 实时交易接口）
│   ├── data_adapter.py          # 数据适配器（回测/实盘统一接口）
│   ├── executor.py              # 执行器（回测/QMT 统一接口）
│   ├── financial_data.py        # 财务数据缓存（按需加载）
│   ├── models.py                # 数据模型
│   ├── order_router.py          # 订单路由（多策略回调分发）
│   ├── reconciler.py            # 对账器（虚拟簿记校验与校准）
│   ├── stock_selection.py       # 选股策略基类
│   ├── strategy.py              # 回测策略适配层（Backtrader 桥接）
│   ├── strategy_logic.py        # 策略逻辑基类（与执行环境解耦）
│   └── virtual_book.py          # 虚拟持仓簿（策略级持仓/资金隔离）
├── jqdata/                      # 聚宽数据下载工具
│   ├── 数据下载说明.md            # JQData 数据下载说明
│   ├── 获取指数成分股.ipynb       # 指数历史成分股下载
│   └── 获取行业成分股.ipynb       # 申万行业历史成分股下载
├── logs/                        # 日志目录
├── monitor/                     # 监控模块
│   ├── __init__.py
│   └── realtime_monitor.py      # 实时监控（基于 Dash）
├── strategies/                  # 策略目录
│   ├── __init__.py              # 策略注册与自动发现
│   ├── high_dividend_strategy/  # 高股息策略
│   │   ├── __init__.py
│   │   ├── high_dividend_strategy.py  # 策略主文件
│   │   ├── readme.md            # 策略文档
│   │   └── optimization/        # 优化案例
│   │       ├── run_optimization.py     # 自动化回测脚本
│   │       ├── generate_report.py      # HTML 报告生成
│   │       ├── plot_comparison.py      # 对比图生成
│   │       ├── 优化报告.md             # 优化报告文档
│   │       └── optimization_results/   # 优化回测结果
│   └── small_cap_strategy/      # 小市值策略
│       ├── __init__.py
│       ├── small_cap_strategy.py # 策略主文件
│       ├── readme.md            # 策略文档
│       └── optimization/        # 优化案例
│           ├── run_optimization.py     # 自动化回测脚本
│           ├── plot_comparison.py      # 对比图生成
│           ├── 优化报告.md             # 优化报告文档
│           └── optimization_results/   # 优化回测结果
├── backtest_results/            # 全局回测结果目录
│   └── index.json               # 回测结果索引
└── utils/                       # 工具函数
    ├── __init__.py
    ├── backtest_recorder.py     # 回测结果记录与管理（JSON + HTML 报告）
    ├── logger.py                # 日志管理（支持策略实例隔离）
    ├── parameter_optimizer.py   # 参数优化工具（网格搜索）
    ├── plotly_templates.py      # Plotly 图表模板（净值曲线、回撤、指标对比）
    ├── report.py                # 报告生成（PyQt5 可视化）
    ├── report_generator.py      # HTML 报告生成（Plotly 可视化）
    └── visualization.py         # 可视化工具（Matplotlib/Plotly）
```

## 环境搭建

1. **安装 Python**：建议使用 Python 3.9+

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **安装 QMT**（可选，用于实盘/模拟交易）：
   - 从券商获取 QMT 客户端
   - 安装并登录 QMT
   - 勾选"独立交易"进入 MiniQMT 模式
   - 从 QMT 官网下载并安装 XtQuant 库

## 使用方法

### 1. 运行回测

**高股息策略回测：**
```bash
python main.py --mode backtest --strategy high_dividend --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --debug
```

**小市值策略回测：**
```bash
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --debug
```

**AI 模式回测（跳过图形界面，适用于自动化优化）：**
```bash
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --ai-mode --no-record
```

**参数说明**：
- `--mode`：运行模式，可选值：`backtest`（回测）、`sim`（模拟交易）、`real`（实盘交易）、`instances`（多策略实例）
- `--strategy`：策略类型，可选值：`high_dividend`、`small_cap` 等
- `--period`：数据周期，可选值：`1d`（日线）、`1m`、`5m`、`15m`、`30m`、`60m`、`tick`
- `--pool`：股票池板块名称，如 `沪深300`、`沪深A股`、`上证50`、`中证500`、`中证1000`
- `--start`：回测起始日期，格式：`YYYY-MM-DD`
- `--end`：回测结束日期，格式：`YYYY-MM-DD`
- `--qmt-path`：QMT userdata_mini 路径（默认 `D:\qmt\userdata_mini`）
- `--account`：QMT 资金账号，不传则自动获取第一个
- `--instances`：策略实例配置文件路径（JSON），用于 `--mode instances` 模式
- `--cache-dir`：自定义缓存数据存储目录（默认项目根目录下 `.cache`）
- `--mem-limit`：内存缓存最大对象数量限制（默认 500）
- `--debug`：启用 DEBUG 日志模式，输出详细调试信息
- `--ai-mode`：启用 AI 自动运行模式，跳过所有图形界面渲染，适用于自动化策略优化
- `--no-record`：禁用回测结果自动记录到本地文件
- `--slippage`：滑点百分比，如 `0.001` 表示 0.1%，不传则使用策略默认值

### 2. 运行模拟交易

```bash
python main.py --mode sim --strategy high_dividend --qmt-path D:\qmt\userdata_mini
```

### 3. 运行实盘交易

```bash
python main.py --mode real --strategy high_dividend --qmt-path D:\qmt\userdata_mini --account 12345678
```

### 4. 运行多策略实例

当需要在同一账户下同时运行多个策略时，使用多策略实例模式。每个策略实例拥有独立的虚拟持仓簿，实现持仓和资金的策略级隔离。

**创建配置文件** `config/instances.json`：

```json
{
  "instances": [
    {
      "instance_id": "small_cap_sim",
      "strategy_name": "small_cap",
      "mode": "sim",
      "account_id": "12345678",
      "initial_capital": 500000,
      "claim_existing_positions": true,
      "kwargs": {}
    },
    {
      "instance_id": "high_div_sim",
      "strategy_name": "high_dividend",
      "mode": "sim",
      "account_id": "12345678",
      "initial_capital": 500000,
      "claim_existing_positions": true,
      "kwargs": {}
    }
  ]
}
```

**配置说明**：
- `instance_id`：策略实例唯一标识，用于日志隔离和订单路由
- `strategy_name`：策略名称，对应 `--strategy` 参数的可选值
- `mode`：运行模式，`sim`（模拟）或 `real`（实盘）
- `account_id`：QMT 资金账号，相同账户的策略共享一个 QMTAPI 实例
- `initial_capital`：策略初始资金，用于虚拟簿记
- `claim_existing_positions`：是否认领账户现有持仓（首次启动时按标的分配给策略）
- `kwargs`：策略参数，覆盖默认参数

**启动多策略**：

```bash
python main.py --mode instances --instances config/instances.json
```

> **注意**：单策略模式（`sim`/`real`）也默认启用虚拟簿记，无需额外配置即可享受策略级隔离。

### 5. 数据预下载

在运行回测前，可以预先下载行情和财务数据到本地缓存，避免回测时逐只下载导致等待过长：

**下载行情数据：**
```bash
python download_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-04-28
```

**下载财务数据：**
```bash
python download_financial_data.py --pool 中证1000 --start 2020-01-01
```

**查看缓存文件内容：**
```bash
python read_parquet.py
```

**清理过期日志：**
```bash
clean_old_logs.bat
```

## 架构设计

### 策略架构

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

### 数据架构

```
DataProcessor（数据处理器基类）
├── OpenDataProcessor（主数据源：腾讯财经/AkShare，历史数据丰富，包含行情、财务、QVIX 等）
├── QMTDataProcessor（补充数据源：QMT 数据补足 + 实时交易接口）
└── CSVDataProcessor（CSV 文件数据源）
```

- **OpenData 为主**：历史行情数据丰富，覆盖完整回测区间
- **QMT 补充**：QMT 数据补足缺口，同时提供实时交易接口
- **历史成分股**：内置聚宽下载的指数和行业历史成分股 CSV，回测时使用对应时点的真实成分股

### 执行器架构

```
StrategyExecutor（执行器基类）
├── BacktestExecutor（回测执行器，通过 Backtrader 执行）
└── QMTExecutor（实盘执行器，通过 QMT 交易接口执行）
    ├── virtual_book → VirtualBook（虚拟持仓簿，策略级隔离）
    └── order_router → OrderRouter（订单路由，回调分发）
```

### 多策略实例架构

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

## 智能缓存系统

框架内置了双层智能缓存系统，大幅提升数据加载速度：

### 缓存架构

- **内存缓存（MemCache）**：基于 OrderedDict 的线程安全 LRU 缓存，默认容量 500
- **磁盘缓存（DiskCache）**：支持 parquet 和 pickle 两种格式，按命名空间隔离
- **缓存索引（CacheIndexManager）**：维护行情/财报数据的年份索引，支持增量更新

### 缓存特性

1. **逐只缓存**：财务数据按股票代码单独缓存，中断后自动恢复，避免重复下载
2. **合并缓存**：全部下载完成后自动合并为总缓存，提升后续加载速度
3. **增量更新**：行情数据支持智能增量更新，只下载缺失部分
4. **格式优化**：默认使用 pyarrow parquet 格式，比 pickle 更快更紧凑
5. **原子写入**：使用临时文件 + 原子重命名，避免写入中断导致文件损坏
6. **缓存索引**：维护年份索引，快速判断缓存覆盖范围，避免重复下载

### 缓存目录

缓存文件默认存储在项目根目录的 `.cache/` 文件夹下：
```
.cache/
├── index/                           # 缓存索引
│   ├── market_index.json            # 行情数据索引
│   └── financial_index.json         # 财务数据索引
├── QMTDataProcessor/                # QMT 行情数据缓存
├── QMTDataProcessor_Financial/      # QMT 财务数据缓存
├── OpenData/                        # OpenData 行情数据缓存
│   ├── vix/                         # QVIX 隐含波动率缓存
│   │   └── QVIX_510500.SH.parquet
│   └── financial/                   # OpenData 财务数据缓存
│       ├── 000001.SZ_Balance.parquet
│       ├── 000001.SZ_CashFlow.parquet
│       ├── 000001.SZ_Income.parquet
│       └── 000001.SZ_Pershareindex.parquet
└── ...
```

### 环境变量

- `QMT_CACHE_DIR`：自定义缓存目录路径
- `QMT_MEM_CACHE_LIMIT`：内存缓存容量限制（默认 500）
- `QMT_LOG_LEVEL`：日志级别（默认 `INFO`，设为 `DEBUG` 输出详细调试信息）

## 回测结果记录

框架支持自动记录每次回测的完整结果，便于追溯和对比：

### 核心功能

- **自动记录**：每次回测自动保存完整结果到本地 JSON 文件（指标、交易日志、净值曲线、基准曲线）
- **全局索引**：维护 `index.json` 索引，支持按策略名检索历史回测
- **结果对比**：支持多组回测结果对比分析，按夏普比率排序
- **HTML 报告**：生成基于 Plotly 的交互式 HTML 可视化报告（净值曲线、回撤曲线、指标对比、交易统计）

### 存储位置

回测结果优先保存到策略目录下的 `backtest_results/`，便于与策略代码统一管理：

```
strategies/<策略目录>/
├── <策略名>.py
└── backtest_results/
    └── 20260428_143000_small_cap.json
```

如果策略目录不可写，则回退到全局 `backtest_results/` 目录。

### 使用方式

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

## 策略开发

### 创建自定义策略

1. 在 `strategies` 目录下创建新的策略子目录（推荐）或策略文件
2. 继承 `core.strategy_logic.StrategyLogic` 类（单标的策略）或 `core.stock_selection.StockSelectionStrategy` 类（多标的选股策略）
3. 使用 `@register_strategy` 装饰器注册策略
4. 实现相应的策略逻辑方法

> **注意**：策略文件放入 `strategies/` 目录后会自动被发现和注册，无需手动导入。

### 策略目录结构（推荐）

```
strategies/my_strategy/
├── __init__.py                  # 导出策略类
├── my_strategy.py               # 策略主文件
├── readme.md                    # 策略文档
├── backtest_results/            # 回测结果记录
└── optimization/                # 优化案例
    ├── run_optimization.py      # 自动化回测脚本
    ├── generate_report.py       # HTML 报告生成
    ├── plot_comparison.py       # 对比图生成
    ├── 优化报告.md               # 优化报告文档
    └── optimization_results/    # 优化回测结果
```

### 示例策略

#### 单标的策略示例

```python
from core.strategy_logic import StrategyLogic, BarData
from strategies import register_strategy

@register_strategy('my_strategy', default_kwargs={'fast_period': 5, 'slow_period': 20},
                   backtest_config={'cash': 100000, 'commission': 0.0001,
                                    'start_date': '2025-01-01', 'end_date': '2026-04-17'})
class MyStrategy(StrategyLogic):
    params = (
        ('fast_period', 5),
        ('slow_period', 20),
        ('symbol', '000001.SZ'),
        ('position_ratio', 0.9),
    )

    def on_bar(self, bar: BarData):
        symbol = self.params.symbol
        close_prices = self.get_close_prices(symbol)

        if len(close_prices) < self.params.slow_period:
            return

        fast_ma = sum(close_prices[-self.params.fast_period:]) / self.params.fast_period
        slow_ma = sum(close_prices[-self.params.slow_period:]) / self.params.slow_period

        pos_size = self.get_position_size(symbol)

        if fast_ma > slow_ma and pos_size == 0:
            price = self.get_current_price(symbol)
            cash = self.get_cash()
            volume = int(cash * self.params.position_ratio / price / 100) * 100
            if volume >= 100:
                self.buy(symbol, price, volume)
        elif fast_ma < slow_ma and pos_size > 0:
            price = self.get_current_price(symbol)
            self.sell(symbol, price, pos_size)
```

#### 选股策略示例

```python
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy

@register_strategy('my_selection_strategy', default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2016-01-01', 'end_date': '2026-04-17'})
class MySelectionStrategy(StockSelectionStrategy):
    """自定义选股策略"""

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
    )

    def select_stocks(self):
        """选股逻辑"""
        pool = self.get_stock_pool()
        filtered = self.screen_stocks(
            lambda s: (self.get_financial_field(s, 'Pershareindex', 'roe_diluted') or 0) > 0.10,
            pool
        )
        ranked = self.rank_stocks(
            lambda s: self.get_financial_field(s, 'Pershareindex', 'roe_diluted') or 0,
            stock_pool=filtered,
            top_n=self.params.max_stocks
        )
        return [stock for stock, _ in ranked]
```

## 内置策略

### 1. 高股息策略 (high_dividend)
- 基于股息率选股，行业分散配置
- 规避高股息陷阱，要求 ROE>0、归母净利润增速>0、经营现金流>0
- 支持近3年平均股息率模式，提升稳定性
- 双周调仓（`biweekly`），等权重持仓
- **防未来数据**：财务数据按公告日期索引，回测时只使用已披露数据
- 优化记录：波动率过滤、止损机制、ROE阈值、现金流阈值、双周调仓等

### 2. 小市值策略 (small_cap)
- 基本面过滤 + 行业分散 + 动量确认 + 波动率过滤 + 止损机制的小市值选股策略
- 四重基本面过滤：ROE>0、营收增速>0、经营现金流>0、资产负债率<阈值
- 每个申万一级行业选市值最小的1只，避免行业集中暴露
- 动量确认：近N日涨幅>0，避免在下跌趋势中接飞刀
- 波动率过滤：剔除日波动率过高的投机标的（事前风控）
- 止损机制：持仓亏损>=8%时在调仓时止损卖出（事后风控）
- 月度调仓，等权重持仓
- 优化记录：波动率过滤（夏普+14.5%）、止损机制（夏普+12.8%）、组合优化（夏普+22.5%）

## 策略优化工作流

框架内置了系统化的策略优化工作流，支持从提出优化建议到验证改进效果的全流程：

### 优化流程

1. **理解策略**：读取策略代码和文档，运行基线回测，确定核心评估指标
2. **提出优化建议**：提出10项具体优化建议，每项包含方向、实现路径、预期目标
3. **独立实施并回测**：每项优化独立实施（新参数默认禁用），运行回测，记录结果
4. **评估与筛选**：夏普比率相对基线提升 >= 5% 的优化予以保留
5. **组合有效优化**：测试所有有效优化的组合效果，检测参数冲突
6. **清理与文档**：删除无效优化代码，更新策略文档，生成 HTML 优化报告

### 已验证的优化经验

| 优化方案 | 适用策略类型 | 典型影响 | 说明 |
|---------|------------|---------|------|
| 波动率过滤 | 选股策略 | 夏普+10~15% | 过滤日波动率超过阈值的标的，首选优化方向 |
| 止损机制 | 选股策略 | 夏普+10~15% | 调仓时剔除亏损超过阈值的股票 |
| 波动率+止损组合 | 选股策略 | 夏普+20~25% | 两项优化互补叠加 |

### 优化输出目录

```
strategies/<策略目录>/optimization/
├── run_optimization.py          # 自动化回测运行脚本
├── generate_report.py           # HTML 报告生成脚本
├── plot_comparison.py           # 对比图生成脚本
├── 优化报告.md                   # 优化报告文档
├── optimization_report.html     # HTML 优化报告
└── optimization_results/        # 优化回测结果
    ├── baseline.json            # 基线结果
    ├── opt01_xxx.json           # 优化1结果
    └── ...
```

## 数据源架构

| 数据源 | 实时性 | 数据范围 | 使用门槛 | 角色 |
|--------|--------|----------|----------|------|
| OpenData | 延迟 | A股全市场（历史数据丰富，包含行情、财务、QVIX 等） | 免费，需 akshare | 主数据源 |
| QMT | 实时 | A股全市场（行情约1年） | 需开户+客户端 | 补充数据源 + 交易接口 |
| CSV | - | 自定义 | 无 | 自定义数据源 |

**数据获取策略**：OpenData 提供丰富的历史行情数据，覆盖完整回测区间；QMT 数据用于补充缺口，同时提供实时交易接口。

**历史成分股**：基于聚宽下载的 CSV 文件，支持沪深300、中证500、中证1000、上证50及31个申万一级行业的历史成分股查询，回测时使用对应时点的真实成分股，超出范围时自动从 QMT 获取最新数据并更新文件。

### 未下载聚宽数据的影响

聚宽历史成分股数据是回测中避免**幸存者偏差**的关键数据。如果未下载 `.cache/JQData/` 下的 CSV 文件，框架会按以下逻辑降级处理：

| 场景 | 降级行为 | 影响 |
|------|---------|------|
| 回测 + 无聚宽数据 + QMT 可用 | 从 QMT 获取**当前最新**成分股，并自动创建 CSV 文件 | 回测使用的是当前成分股而非历史成分股，存在幸存者偏差 |
| 回测 + 无聚宽数据 + QMT 不可用 | 返回空列表 | 股票池为空，策略无法运行 |
| 模拟/实盘 + 无聚宽数据 + QMT 可用 | 从 QMT 获取当前成分股，功能正常 | 无影响，实盘本身就用当前成分股 |
| 模拟/实盘 + 无聚宽数据 + QMT 不可用 | 返回空列表 | QMT 不可用时模拟/实盘也无法运行 |

**结论**：对于**回测场景**，强烈建议下载聚宽历史成分股数据，否则回测结果会因幸存者偏差而失真。对于模拟和实盘交易，不下载也不影响正常使用。

数据下载方式详见 `jqdata/数据下载说明.md`。

## 注意事项

1. **实盘交易风险**：实盘交易存在风险，请谨慎使用
2. **QMT 依赖**：使用 QMT 接口需要安装 QMT 客户端和 XtQuant 库
3. **数据质量**：回测结果依赖于数据质量，请确保数据的准确性
4. **策略优化**：参数优化可能导致过拟合，请谨慎使用优化结果
5. **缓存清理**：如遇到数据异常，可手动删除 `.cache/` 目录下的对应文件重新下载
6. **历史成分股**：未下载聚宽数据时，回测会使用当前成分股替代历史成分股，存在幸存者偏差。详见上方「未下载聚宽数据的影响」

## 后续计划

- [x] 多策略实例隔离（VirtualBook + OrderRouter + Reconciler）
- [x] 同账户多策略虚拟簿记与自动对账
- [x] 日志按策略实例隔离
- [x] 回测结果自动记录与 HTML 可视化报告
- [x] 策略优化工作流（自动提出优化建议、独立回测、筛选有效改进）
- [x] AI 自动运行模式
- [x] 数据预下载脚本（行情 + 财务）
- [ ] 支持更多交易接口（如 tushare、yahoo 等）
- [ ] 增加深度学习模型集成
- [ ] 开发 Web 界面
- [ ] 实现分布式回测和交易
- [ ] 增加更多因子和策略模板
- [ ] 优化回测性能
- [ ] 完善风险管理功能
- [ ] 添加实时告警系统

---

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。使用者应自行承担使用本项目的全部风险和责任。

## 风险提示

1. **市场风险**：金融市场存在不可预测的风险，过往表现不代表未来收益
2. **技术风险**：自动化交易系统可能存在技术故障、网络延迟等问题
3. **合规风险**：请确保您的交易行为符合当地法律法规要求
4. **数据风险**：免费数据源可能存在延迟或错误，实盘前请验证数据准确性
5. **资金风险**：实盘交易可能导致资金损失，请谨慎投资

**投资有风险，入市需谨慎！**

## 联系作者

如果您对项目有任何问题或建议，欢迎添加作者微信交流：

![微信二维码](码上生财.jpg)

扫描上方二维码添加好友，获取更多项目更新和技术支持。
