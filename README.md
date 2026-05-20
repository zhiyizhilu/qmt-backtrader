# 量化交易框架

一个基于自研回测引擎和 QMT 的量化交易框架。**核心目标：一套策略代码无缝切换回测、模拟交易和实盘交易**，无需为不同运行模式修改策略逻辑，极大降低从研究到上线的迁移成本。

## 功能特点

- **多模式支持**：回测、模拟交易、实盘交易、多策略实例
- **多策略实例隔离**：同账户多策略独立运行，虚拟簿记实现持仓/资金级隔离，互不干扰
- **高性能自研回测引擎**：基于 numpy 数组预加载 + 预计算统一时间轴，替代 Backtrader
- **QMT 集成**：支持通过 QMT 进行模拟和实盘交易
- **多数据源架构**：OpenData（腾讯财经/AkShare）+ QMT + 富途（Futu OpenD），支持 `--data-source` 切换
- **历史成分股支持**：内置指数和申万行业历史成分股数据，回测时使用对应时点的真实成分股
- **智能缓存**：内存 + 磁盘双层缓存 + 缓存索引管理，支持 parquet 格式，中断后自动恢复
- **策略模板**：内置小市值策略、特质波动率因子策略、低估价值策略等
- **参数优化**：支持网格搜索 + 策略优化工作流 + Walk-Forward 样本外验证
- **可视化**：提供基于 PyQt5 的回测结果展示、基于 Plotly 的 HTML 可视化报告、基于 Dash 的实时监控
- **Web 查看器**：基于 Flask + Vue3 + ECharts 的策略回测结果 Web 查看器
- **选股策略**：支持基于基本面的股票选择策略，行业分散配置，灵活权重分配
- **防未来数据**：财务数据按公告日期索引，回测时只使用已披露数据
- **自动对账**：每日开盘前自动校验虚拟簿记与账户实际状态的一致性，偏差自动校准
- **回测结果记录**：自动记录每次回测的完整结果到本地 JSON，支持加载、列举、对比历史回测数据
- **AI 模式**：支持 AI 自动运行模式，跳过所有图形界面渲染，适用于自动化策略优化
- **数据预下载**：提供独立的行情/财务数据预下载脚本，支持 QMT、OpenData、富途三种数据源
- **YAML 配置**：支持 YAML 配置文件，命令行参数覆盖配置
- **股票生命周期管理**：自动管理上市/退市时间，避免对退市/未上市股票的无效数据请求

## 项目结构

```
qmt_backtrader/
├── README.md                    # 项目说明文档
├── docs/                        # 详细文档
│   ├── architecture.md          # 架构设计
│   ├── usage.md                 # 使用方法
│   ├── cache.md                 # 智能缓存系统
│   ├── strategy-development.md  # 策略开发与优化
│   ├── data-download.md         # 数据源与下载
│   └── backtest-recorder.md     # 回测结果记录
├── main.py                      # 主入口文件
├── download_market_data.py      # 行情数据预下载脚本（OpenData）
├── download_financial_data.py   # 财务数据预下载脚本
├── download_qmt_data.py         # QMT 行情数据预下载脚本
├── download_futu_data.py        # 富途行情数据预下载脚本
├── api/                         # API 接口
│   ├── backtest_api.py          # 回测 API（基于自研回测引擎）
│   ├── base_api.py              # API 基类
│   ├── instance_manager.py      # 策略实例管理器
│   └── qmt_api.py               # QMT 交易 API
├── engine/                      # 自研回测引擎
│   ├── engine.py                # BacktestEngine 回测引擎核心
│   ├── broker.py                # SimulatedBroker 模拟经纪商
│   ├── data_feed.py             # ArrayDataFeed numpy 数组数据源
│   ├── timeline.py              # Timeline 统一时间轴
│   ├── adapter.py               # EngineDataAdapter / EngineExecutor
│   └── result.py                # EngineResult 回测结果容器
├── core/                        # 核心模块
│   ├── cache/                   # 智能缓存系统（子包）
│   │   ├── manager.py           # SmartCacheManager 缓存调度器
│   │   ├── mem_cache.py         # MemCache 内存缓存
│   │   ├── disk_cache.py        # DiskCache 磁盘缓存
│   │   └── index_manager.py     # CacheIndexManager 缓存索引
│   ├── data/                    # 数据处理模块
│   │   ├── opendata.py          # OpenData 数据处理器（腾讯财经/AkShare）
│   │   ├── qmt.py               # QMT 数据处理器
│   │   ├── futu.py              # 富途数据处理器（Futu OpenD）
│   │   ├── qvix.py              # QVIX 隐含波动率
│   │   ├── csv.py               # CSV 数据处理器
│   │   ├── index_constituent.py # 指数历史成分股管理器
│   │   └── industry_constituent.py # 申万行业历史成分股管理器
│   ├── data_adapter.py          # 数据适配器（回测/实盘统一接口）
│   ├── executor.py              # 执行器（回测/QMT 统一接口）
│   ├── financial_data.py        # 财务数据缓存
│   ├── models.py                # 数据模型
│   ├── order_router.py          # 订单路由（多策略回调分发）
│   ├── reconciler.py            # 对账器（虚拟簿记校验与校准）
│   ├── stock_lifecycle.py       # 股票生命周期管理器（上市/退市时间）
│   ├── stock_selection.py       # 选股策略基类
│   ├── strategy.py              # 回测策略适配层
│   ├── strategy_logic.py        # 策略逻辑基类（与执行环境解耦）
│   ├── virtual_book.py          # 虚拟持仓簿（策略级持仓/资金隔离）
│   └── weight_allocator.py      # 权重分配器（等权/风险平价/因子加权）
├── strategies/                  # 策略目录
│   ├── small_cap_strategy/      # 小市值策略
│   ├── ivff3_strategy/          # 特质波动率因子策略
│   ├── undervalued_strategy/    # 低估价值策略
│   └── jq_small_cap_strategy/   # 聚宽小市值策略
├── utils/                       # 工具函数
│   ├── backtest_recorder.py     # 回测结果记录与管理
│   ├── config.py                # YAML 配置加载
│   ├── logger.py                # 日志管理（支持策略实例隔离）
│   ├── parameter_optimizer.py   # 参数优化工具（网格搜索）
│   ├── plotly_templates.py      # Plotly 图表模板
│   ├── report.py                # 报告生成（PyQt5 可视化）
│   ├── report_generator.py      # HTML 报告生成（Plotly 可视化）
│   ├── visualization.py         # 可视化工具（Matplotlib/Plotly）
│   └── walk_forward.py          # Walk-Forward 时间窗口分割器
├── tests/                       # 单元测试
│   ├── test_broker.py
│   ├── test_cache.py
│   ├── test_data_feed.py
│   └── test_models.py
└── web/                         # Web 回测查看器
    ├── app.py                   # Flask 后端
    └── templates/index.html     # Vue3 + ECharts 前端
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

4. **安装富途 OpenD**（可选，用于富途数据源）：
   - 安装 futu-api：`pip install futu-api`
   - 下载并启动富途 OpenD 行情网关

## 快速开始

```bash
# 运行回测（QMT 数据源）
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --debug

# 运行回测（OpenData 数据源）
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --data-source open

# 运行回测（富途数据源）
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --data-source futu

# 使用 YAML 配置文件
python main.py --mode backtest --strategy small_cap --config config/backtest.yaml

# 运行模拟交易
python main.py --mode sim --strategy small_cap --qmt-path D:\qmt\userdata_mini

# 运行实盘交易
python main.py --mode real --strategy small_cap --qmt-path D:\qmt\userdata_mini --account 12345678
```

更多使用方法详见 [使用方法](docs/usage.md)。

## 详细文档

| 文档 | 说明 |
|------|------|
| [架构设计](docs/architecture.md) | 自研回测引擎、策略架构、数据架构、执行器架构、多策略实例架构 |
| [使用方法](docs/usage.md) | 回测、模拟交易、实盘交易、多策略实例、数据预下载、Web 查看器 |
| [智能缓存系统](docs/cache.md) | 缓存架构、缓存特性、缓存目录、环境变量 |
| [策略开发与优化](docs/strategy-development.md) | 自定义策略开发、内置策略、策略优化工作流、Walk-Forward 验证 |
| [数据源与下载](docs/data-download.md) | 多数据源架构、行情数据下载详解、完整性校验、常见问题 |
| [回测结果记录](docs/backtest-recorder.md) | 回测结果自动记录、对比分析、HTML 报告 |

## 注意事项

1. **实盘交易风险**：实盘交易存在风险，请谨慎使用
2. **QMT 依赖**：使用 QMT 接口需要安装 QMT 客户端和 XtQuant 库
3. **数据质量**：回测结果依赖于数据质量，请确保数据的准确性
4. **策略优化**：参数优化可能导致过拟合，建议使用 Walk-Forward 样本外验证
5. **缓存清理**：如遇到数据异常，可手动删除 `.cache/` 目录下的对应文件重新下载
6. **历史成分股**：未下载聚宽数据时，回测会使用当前成分股替代历史成分股，存在幸存者偏差。详见 [数据源与下载](docs/data-download.md)

## 后续计划

- [x] 多策略实例隔离（VirtualBook + OrderRouter + Reconciler）
- [x] 同账户多策略虚拟簿记与自动对账
- [x] 日志按策略实例隔离
- [x] 回测结果自动记录与 HTML 可视化报告
- [x] 策略优化工作流（自动提出优化建议、独立回测、筛选有效改进）
- [x] AI 自动运行模式
- [x] 数据预下载脚本（行情 + 财务）
- [x] 自研回测引擎（替代 Backtrader，基于 numpy 数组预加载）
- [x] 富途数据源支持（Futu OpenD）
- [x] Walk-Forward 样本外验证
- [x] 权重分配器（等权/风险平价/因子加权）
- [x] 股票生命周期管理（上市/退市时间）
- [x] YAML 配置文件支持
- [ ] 支持更多交易接口（如 tushare、yahoo 等）
- [ ] 增加深度学习模型集成
- [x] Web 回测查看器（Flask + Vue3 + ECharts）
- [ ] 实现分布式回测和交易
- [ ] 增加更多因子和策略模板
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
