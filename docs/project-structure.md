# 项目结构

```
qmt_backtrader/
├── README.md                        # 项目说明文档
├── main.py                          # 主入口文件
├── convert_all_stocks.py            # 批量股票数据转换脚本
├── convert_minute_hfq.py             # 分钟级后复权数据转换（聚宽 CSV → 后复权）
├── download_open_market_data.py      # 行情数据预下载脚本（OpenData）
├── download_qmt_financial_data.py    # QMT 财务数据预下载脚本
├── download_qmt_market_data.py       # QMT 行情数据预下载脚本
├── download_futu_market_data.py      # 富途行情数据预下载脚本
├── download_futu_capital_flow.py     # 富途资金流向数据下载脚本
├── clean_old_logs.bat               # 清理过期日志批处理
├── start_web.bat                    # 启动 Web 查看器批处理
├── docs/                            # 详细文档
│   ├── architecture.md              # 架构设计
│   ├── usage.md                     # 使用方法
│   ├── cache.md                     # 智能缓存系统
│   ├── strategy-development.md      # 策略开发与优化
│   ├── data-download.md             # 数据源与下载
│   ├── backtest-recorder.md         # 回测结果记录
│   ├── strategies_for_vip.md        # VIP 量化策略集
│   ├── project-structure.md         # 项目结构（本文件）
│   └── strategies_for_vip/          # VIP 策略宣传图
│       ├── etf_rotation_strategy/
│       ├── four_troublemaker_strategy/
│       ├── high_dividend_strategy/
│       └── vol_rotation_strategy/
├── api/                             # API 接口
│   ├── base_api.py                  # API 基类
│   ├── instance_manager.py          # 策略实例管理器
│   └── qmt_api.py                   # QMT 交易 API
├── engine/                          # 自研回测引擎
│   ├── engine.py                    # BacktestEngine 回测引擎核心
│   ├── broker.py                    # SimulatedBroker 模拟经纪商
│   ├── data_feed.py                 # ArrayDataFeed numpy 数组数据源
│   ├── timeline.py                  # Timeline 统一时间轴
│   ├── adapter.py                   # EngineDataAdapter / EngineExecutor
│   └── result.py                    # EngineResult 回测结果容器
├── core/                            # 核心模块
│   ├── cache/                       # 智能缓存系统（子包）
│   │   ├── manager.py               # SmartCacheManager 缓存调度器
│   │   ├── mem_cache.py             # MemCache 内存缓存
│   │   ├── disk_cache.py            # DiskCache 磁盘缓存
│   │   └── index_manager.py         # CacheIndexManager 缓存索引
│   ├── data/                        # 数据处理模块
│   │   ├── base.py                  # DataProcessor 数据处理器基类
│   │   ├── factory.py                # 数据处理器工厂
│   │   ├── opendata.py               # OpenData 数据处理器（腾讯财经/AkShare）
│   │   ├── qmt.py                   # QMT 数据处理器
│   │   ├── futu.py                   # 富途数据处理器（Futu OpenD）
│   │   ├── csv.py                    # CSV 数据处理器
│   │   ├── qvix.py                  # QVIX 隐含波动率
│   │   ├── index_constituent.py     # 指数历史成分股管理器
│   │   └── industry_constituent.py  # 申万行业历史成分股管理器
│   ├── analyzer.py                  # 回测分析器
│   ├── data_adapter.py              # 数据适配器（回测/实盘统一接口）
│   ├── executor.py                  # 执行器（回测/QMT 统一接口）
│   ├── financial_data.py            # 财务数据缓存
│   ├── models.py                    # 数据模型
│   ├── order_router.py              # 订单路由（多策略回调分发）
│   ├── reconciler.py                # 对账器（虚拟簿记校验与校准）
│   ├── stock_lifecycle.py           # 股票生命周期管理器（上市/退市时间）
│   ├── stock_selection.py           # 选股策略基类
│   ├── strategy.py                 # 回测策略适配层
│   ├── strategy_logic.py            # 策略逻辑基类（与执行环境解耦）
│   ├── virtual_book.py              # 虚拟持仓簿（策略级持仓/资金隔离）
│   └── weight_allocator.py          # 权重分配器（等权/风险平价/因子加权）
├── strategies/                      # 标准策略目录
│   ├── small_cap_strategy/          # 小市值策略
│   ├── small_cap_roe_strategy/      # 小市值 ROE/ROA 策略
│   ├── ivff3_strategy/              # 特质波动率因子策略
│   ├── undervalued_strategy/        # 低估价值策略
│   ├── jq_small_cap_strategy/       # 聚宽小市值策略
│   ├── bank_rotation_strategy/      # 银行轮动策略（四大行均值回归）
│   ├── dividend_value_growth_strategy/ # 高股息低市盈率高增长价投策略
│   ├── medical_multi_factor_strategy/  # 医药行业多因子选股策略
│   ├── twenty_eight_rotation_strategy/ # 二八轮动小市值策略
│   ├── first_board_low_open_strategy/  # 首板低开策略
│   └── rothman_value_strategy/      # 罗斯曼价值精选策略
├── strategies_my/                   # 个人策略目录（自动发现，用户自定义）
├── strategies_for_vip/              # VIP 策略目录（自动发现，付费策略）
├── strategies_for_svip/             # SVIP 策略目录（自动发现，付费策略）
├── monitor/                         # 实时监控
│   └── realtime_monitor.py          # 基于 Dash 的实时监控
├── utils/                           # 工具函数
│   ├── config.py                    # YAML 配置加载
│   ├── logger.py                    # 日志管理（支持策略实例隔离）
│   ├── parameter_optimizer.py       # 参数优化工具（网格搜索）
│   ├── plotly_templates.py         # Plotly 图表模板
│   ├── report.py                    # 报告生成（PyQt5 可视化）
│   ├── report_generator.py          # HTML 报告生成（Plotly 可视化）
│   ├── visualization.py             # 可视化工具（Matplotlib/Plotly）
│   └── walk_forward.py              # Walk-Forward 时间窗口分割器
├── jqdata/                          # 聚宽数据下载
│   ├── 下载股票历史分钟线数据.ipynb
│   ├── 获取指数成分股.ipynb
│   ├── 获取行业成分股.ipynb
│   └── 数据下载说明.md
└── web/                             # Web 回测查看器
    ├── app.py                       # Flask 后端
    └── templates/index.html         # Vue3 + ECharts 前端
```

## 策略目录说明

框架支持三个策略目录，均会被自动发现和注册：

| 目录 | 说明 |
|------|------|
| `strategies/` | 标准策略目录，包含框架内置的策略模板 |
| `strategies_my/` | 个人策略目录，用户自定义策略（自动发现） |
| `strategies_for_vip/` | VIP 付费策略目录（可选，自动发现） |
| `strategies_for_svip/` | SVIP 付费策略目录（可选，自动发现） |

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
