# 项目结构

```
qmt_backtrader/
├── README.md                    # 项目说明文档
├── docs/                        # 详细文档
│   ├── architecture.md          # 架构设计
│   ├── usage.md                 # 使用方法
│   ├── cache.md                 # 智能缓存系统
│   ├── strategy-development.md  # 策略开发与优化
│   ├── data-download.md         # 数据源与下载
│   ├── backtest-recorder.md     # 回测结果记录
│   ├── strategies_for_vip.md    # VIP 量化策略集
│   └── project-structure.md     # 项目结构（本文件）
├── main.py                      # 主入口文件
├── download_open_market_data.py  # 行情数据预下载脚本（OpenData）
├── download_qmt_financial_data.py   # QMT 财务数据预下载脚本
├── download_qmt_market_data.py   # QMT 行情数据预下载脚本
├── download_futu_market_data.py        # 富途行情数据预下载脚本
├── download_futu_capital_flow.py       # 富途资金流向数据下载脚本
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
├── strategies_for_vip/          # VIP 策略目录（付费）
│   ├── etf_rotation_strategy/   # ETF 轮动策略
│   ├── bank_rotation_strategy/  # 银行轮动策略
│   ├── high_dividend_strategy/  # 高股息行业均仓策略
│   └── vol_rotation_strategy/   # 高低波动 ETF 轮动策略
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
