# 数据源与下载

## 数据源架构

| 数据源 | 实时性 | 数据范围 | 使用门槛 | 角色 |
|--------|--------|----------|----------|------|
| OpenData | 延迟 | A股全市场（历史数据丰富，包含行情、财务、QVIX 等） | 免费，需 akshare | 主数据源 |
| QMT | 实时 | A股全市场（行情约1年） | 需开户+客户端 | 补充数据源 + 交易接口 |
| 富途 | 实时 | A股+港股+美股（需 OpenD 网关） | 需开户+futu-api | 补充数据源 |
| CSV | - | 自定义 | 无 | 自定义数据源 |

**数据源选择**：通过 `--data-source` 参数切换，可选 `qmt`（默认）、`open`、`futu`。

**数据获取策略**：OpenData 提供丰富的历史行情数据，覆盖完整回测区间；QMT 数据用于补充缺口，同时提供实时交易接口；富途 OpenD 提供高质量行情数据，支持自动增量下载。

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

## 行情数据下载功能详解（download_open_market_data.py）

`download_open_market_data.py` 是独立的行情数据预下载脚本，支持批量并发下载 A 股行情数据、完整性校验、缺失修复与交易日一致性检查。

### 数据来源

| 数据源 | 接口 | 认证方式 | 数据范围 | 调用方式 |
|--------|------|----------|----------|----------|
| **腾讯财经**（主） | akshare `stock_zh_a_hist_tx` | 无需认证，免费 | A 股全市场历史数据 | `OpenDataProcessor.get_data()` |
| **东方财富**（指数备用） | akshare `stock_zh_index_daily_em` | 无需认证，免费 | 指数日线数据 | `OpenDataProcessor._get_index_data()` |
| **新浪财经**（指数备用） | akshare `stock_zh_index_daily` | 无需认证，免费 | 指数日线数据 | `OpenDataProcessor._get_index_data()` |
| **QMT**（补充） | xtdata `download_history_data` + `get_market_data_ex` | 需开户 + 安装 MiniQMT 客户端 | A 股全市场（约 1 年历史） | `QMTDataProcessor.get_data()` |

**数据获取优先级**：OpenData（腾讯财经）→ QMT 补充。QMT 行情数据不足 1 年时，自动用 OpenData 补充早期数据并合并。

### 数据类型

| 类型 | 参数值 | 说明 | 复权方式 | 缓存目录 |
|------|--------|------|----------|----------|
| 后复权行情 | `adjusted` | 回测主力数据 | `hfq`（后复权） | `.cache/OpenData/market/{symbol}/` |
| 不复权行情 | `raw` | 股息率等需实际价格的场景 | 空（不复权） | `.cache/OpenData/market_raw/{symbol}/` |
| 全部 | `all` | 同时下载上述两种 | — | — |

### 请求参数

```bash
python download_open_market_data.py [OPTIONS]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--pool` | 否* | 沪深A股 | 股票池：`沪深300`、`中证500`、`中证1000`、`上证50`、`沪深A股` |
| `--stocks` | 否* | — | 手动指定股票代码，逗号分隔，如 `000001.SZ,600000.SH` |
| `--start` | 是 | — | 数据起始日期，格式 `YYYY-MM-DD` |
| `--end` | 是 | — | 数据结束日期，格式 `YYYY-MM-DD` |
| `--period` | 否 | `1d` | 数据周期：`1d`（日线）、`1w`（周线）、`1M`（月线） |
| `--type` | 否 | `all` | 数据类型：`adjusted`、`raw`、`all` |
| `--workers` | 否 | `5` | 并发线程数（建议不超过 10） |
| `--force` | 否 | `False` | 强制重新下载，忽略已有缓存 |
| `--verify` | 否 | `False` | 下载后校验数据完整性并修复缺失 |
| `--verify-only` | 否 | `False` | 仅校验已有数据，不执行新下载 |
| `--check` | 否 | `False` | 检查数据一致性（以沪深300交易日为基准） |
| `--fix` | 否 | `False` | 与 `--check` 配合，自动修复不一致数据 |
| `--dry-run` | 否 | `False` | 与 `--fix` 配合，仅报告不实际修复 |
| `--report` | 否 | 自动 | HTML 报告输出路径 |
| `--full-year` | 否 | 开启 | 自动将首尾年份扩展为全年（默认开启） |
| `--no-full-year` | 否 | — | 禁用全年扩展，严格按指定日期范围下载 |
| `--cache-dir` | 否 | `.cache` | 自定义缓存数据存储目录 |
| `-v` / `--verbose` | 否 | `False` | 启用详细日志，显示每个股票的下载状态 |
| `--log` | 否 | `False` | 将日志同时写入 `logs/` 目录下的文件 |
| `--log-file` | 否 | 自动 | 指定日志文件路径（需配合 `--log`） |

> *`--pool` 和 `--stocks` 二选一，都不指定则默认下载沪深A股全部股票。

### 下载流程

```
┌─────────────────────────────────────────────────────────────┐
│                     main() 入口                              │
├─────────────────────────────────────────────────────────────┤
│  1. 解析命令行参数                                            │
│  2. 初始化 OpenDataProcessor（主数据源）                       │
│  3. 尝试初始化 QMTDataProcessor（补充数据源，失败不退出）        │
│  4. 根据模式分流：                                           │
│     ├─ --check 模式 → run_check()                            │
│     ├─ --verify-only 模式 → run_verify()                    │
│     └─ 默认 → run_download() [+ 可选 run_verify()]           │
└─────────────────────────────────────────────────────────────┘
```

**核心下载流程**（`download_one` → `run_download`）：

1. **解析股票列表**：`resolve_stock_list()` 按优先级获取股票代码
   - 手动指定（`--stocks`）→ 直接使用
   - 指定板块（`--pool`）→ 优先 QMT → 回退 OpenData
   - 默认 → 沪深A股全市场

2. **全年扩展**（默认开启）：将 `--start 2020-04-28` 扩展为 `2020-01-01`，`--end 2026-04-28` 扩展为 `2026-12-31`

3. **缓存检查**：`_needs_download()` 判断是否需要下载
   - 检查索引中已缓存的年份
   - 检查已检查但无数据的年份（避免重复请求）
   - 检查缓存完整性（检测日期空洞，>15天间隔视为异常）
   - `--force` 跳过缓存检查，强制重新下载

4. **数据获取**：根据数据类型调用不同的处理器方法
   - `adjusted` → `processor.get_data()`（后复权，通过 `@smart_cache` 装饰器自动按年份分片缓存）
   - `raw` → `processor.get_raw_data()`（不复权，独立命名空间缓存）

5. **并发执行**：`ThreadPoolExecutor` 多线程并发下载，默认 5 线程

6. **进度追踪**：`DownloadProgress` 实时统计缓存命中/下载/失败/修复/停牌数量

7. **索引持久化**：每处理 5% 股票自动保存索引，防止中断丢失

### 认证方式

- **OpenData（腾讯财经/AkShare）**：无需认证，安装 akshare 后直接使用
- **QMT**：需安装 MiniQMT 客户端并登录，脚本通过 `xtdata` 本地接口访问数据，无需 API Key
- **富途**：需安装 futu-api 并启动富途 OpenD 行情网关，脚本通过 `FutuOpenD` 本地接口访问数据

### 数据格式转换

**原始数据格式**（akshare 返回）：

| 中文列名 | 英文列名 | 说明 |
|----------|----------|------|
| 日期 | Date | 交易日期 |
| 开盘 | Open | 开盘价 |
| 收盘 | Close | 收盘价 |
| 最高 | High | 最高价 |
| 最低 | Low | 最低价 |
| 成交量 | Volume | 成交量 |
| 成交额 | Amount | 成交额（腾讯数据用此列作为成交量单位：手） |

**转换流程**（`_process_akshare_data`）：
1. 列名重命名：中文 → 英文标准列名
2. 日期列转 DatetimeIndex
3. 按日期范围过滤
4. 数值类型转换（`pd.to_numeric`，异常值 coerce）
5. 统一保留列：`open, high, low, close, volume`
6. 腾讯数据：`amount` 映射为 `volume`（单位：手）
7. 请求间隔 0.5 秒，避免触发限频

### 存储方式

**目录结构**（按年份分片存储）：

```
.cache/OpenData/
├── market/                          # 后复权行情
│   ├── 000001.SZ/
│   │   ├── 2020_1d.parquet         # 2020年日线数据
│   │   ├── 2021_1d.parquet
│   │   └── ...
│   └── 600000.SH/
│       └── ...
├── market_raw/                      # 不复权行情
│   ├── 000001.SZ/
│   │   └── ...
│   └── ...
└── index/                           # 缓存索引
    ├── market_index.json            # 后复权索引
    ├── market_raw_index.json        # 不复权索引
    └── ...
```

**存储特性**：
- 格式：Parquet（pyarrow 引擎，snappy 压缩），需 `pip install pyarrow`
- 按年份分片：每只股票每年一个文件，如 `2020_1d.parquet`
- 原子写入：先写 `.tmp` 临时文件，完成后重命名，避免中断损坏
- 索引管理：`CacheIndexManager` 维护 JSON 索引，快速判断缓存覆盖范围
- 内存缓存：LRU 缓存（默认 500 条），加速重复访问

### 完整性校验（--verify / --verify-only）

校验以**沪深300（000300.SH）交易日**为基准，检查每只股票数据是否完整。

**校验流程**（`DataIntegrityChecker`）：

1. **获取交易日历**：`TradingDayCalendar` 从 OpenData 获取沪深300日线，提取交易日
2. **按股票逐年检查**：比对缓存数据与交易日历
3. **缺失分类**：
   - `CURRENT_YEAR`：当年数据未完结（正常）
   - `SUSPENSION`：停牌/未上市/退市导致的缺失（>8个交易日，正常）
   - `DATA_INCOMPLETE`：数据不完整需修复（≤8个交易日，异常）
   - `EXTRA_NON_TRADING`：缓存含非交易日数据（需清理）
   - `RAW_MARKET_MISMATCH`：不复权与后复权数据日期不一致

4. **自动修复**（`DataRepairEngine`）：
   - 删除缺失年份缓存 → 重新下载 → 再次校验
   - 修复成功 → 清除停牌标记
   - 数据源无数据 → 标记为停牌区间，跳过后续下载
   - 非交易日数据 → 直接从 Parquet 中删除对应行

### 交易日一致性检查（--check）

独立于下载流程的数据质量检查，生成 HTML 报告。

**检查内容**：
- 后复权/不复权数据是否含非交易日
- 后复权/不复权数据是否缺交易日
- 不复权与后复权数据日期是否一致

**修复策略**（`--fix`）：
1. **Step 1**：清理非交易日数据（直接从 Parquet 中删除）
2. **Step 2**：重新下载数据不完整的文件（先删后下载）
3. **`--dry-run`**：只报告不执行，用于预览修复计划

### 错误处理

| 错误场景 | 处理方式 |
|----------|----------|
| akshare 未安装 | 启动时警告，调用时抛出 `RuntimeError` |
| 网络请求失败 | 腾讯财经 → 东方财富 → 新浪财经三级降级 |
| QMT 初始化失败 | 警告但不退出，使用 OpenData 作为唯一数据源 |
| 富途 OpenD 未启动 | 抛出 `FutuServiceError`，回测终止 |
| 股票数据为空 | 记录为 `empty`，跳过并更新已检查索引（30天有效） |
| 下载异常 | 记录为 `failed`，不影响其他股票下载 |
| 缓存文件损坏 | 自动删除损坏文件，重新下载 |
| 并发写入冲突 | 线程锁 + 原子文件写入（tmp → rename） |
| 索引与磁盘不同步 | 自动检测并重新获取缺失年份 |
| 停牌股票重复下载 | 停牌区间标记后永久跳过 |

### 使用示例

```bash
# 1. 基础：下载沪深300后复权行情
python download_open_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --workers 5

# 2. 下载并校验数据完整性
python download_open_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type all --workers 5 --verify

# 3. 仅校验已有数据（不下载新数据）
python download_open_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --verify-only

# 4. 检查交易日一致性并生成 HTML 报告
python download_open_market_data.py --check --start 2020-01-01 --end 2026-12-31

# 5. 检查并修复，生成 HTML 报告
python download_open_market_data.py --check --fix --start 2020-01-01 --end 2026-12-31

# 6. dry-run 模式：预览修复计划但不执行
python download_open_market_data.py --check --fix --dry-run --start 2020-01-01 --end 2026-12-31

# 7. 强制重新下载（忽略已有缓存）
python download_open_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-01-01 --type raw --force

# 8. 指定股票代码下载
python download_open_market_data.py --stocks 000001.SZ,600000.SH,600519.SH --start 2020-01-01 --end 2026-01-01 --type all

# 9. 将日志写入文件（适合长时间运行）
python download_open_market_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-01-01 --type all --workers 20 --log --verify

# 10. 下载全市场数据并校验（推荐首次使用）
python download_open_market_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-12-31 --type all --workers 10 --log --verify
```

### 参数配置指南

#### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `QMT_CACHE_DIR` | `.cache` | 缓存数据存储目录 |
| `QMT_MEM_CACHE_LIMIT` | `500` | 内存缓存最大对象数量 |
| `QMT_LOG_LEVEL` | `INFO` | 日志级别（`DEBUG` 输出详细调试信息） |

#### 并发线程数建议

| 场景 | 建议 `--workers` |
|------|-------------------|
| 本地开发调试 | 1-3 |
| 日常增量下载 | 5（默认） |
| 首次全量下载 | 10-20 |
| 网络不稳定 | 3-5 |

> 注意：并发数过高可能触发 akshare 数据源限频，建议不超过 20。

#### 缓存配置

```bash
# 自定义缓存目录
python download_open_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --cache-dir D:/market_cache

# 或通过环境变量
set QMT_CACHE_DIR=D:/market_cache
python download_open_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01
```

### 常见问题

#### Q1: 下载报错 "akshare 未安装"
**解决**：安装依赖 `pip install akshare pyarrow`

#### Q2: QMT 初始化失败是否影响下载？
**不影响**。QMT 仅用于获取股票列表的补充数据源。行情数据由 OpenData（腾讯财经）提供，不依赖 QMT。

#### Q3: 如何清理缓存重新下载？
- 全部清理：删除 `.cache/OpenData/market/` 或 `.cache/OpenData/market_raw/` 目录
- 单只股票：删除 `.cache/OpenData/market/{symbol}/` 目录
- 使用 `--force` 参数强制重新下载

#### Q4: 下载中断后如何恢复？
直接重新运行相同命令即可。缓存系统会自动跳过已下载的年份，只获取缺失部分。

#### Q5: 为什么有些股票显示"数据为空"？
可能是以下原因：
- 股票在指定日期范围内尚未上市
- 股票已退市
- 数据源暂无该股票数据
- 系统会自动标记已检查无数据的年份，30天内不再重复请求

#### Q6: 校验时报告"含非交易日"怎么处理？
使用 `--check --fix` 自动清理非交易日数据，或用 `--check --fix --dry-run` 先预览修复计划。

#### Q7: 下载全市场数据需要多长时间？
以沪深A股约 5000 只股票、5 年日线数据为例，10 线程并发约需 30-60 分钟（取决于网络速度）。

#### Q8: 如何查看缓存文件内容？
```bash
python read_parquet.py
```

#### Q9: `--full-year` 和 `--no-full-year` 的区别？
- `--full-year`（默认）：将日期范围扩展到整年，如 `2020-04-28~2026-04-28` → `2020-01-01~2026-12-31`
- `--no-full-year`：严格按指定日期范围下载

#### Q10: 日志文件在哪里？
- 默认仅输出到控制台
- 使用 `--log` 参数，日志文件保存在 `logs/` 目录下，命名格式：`{时间戳}_download_open_market_data.log`
- 使用 `--log-file` 自定义日志文件路径

#### Q11: 富途数据源如何使用？
1. 安装依赖：`pip install futu-api`
2. 下载并启动富途 OpenD 行情网关
3. 使用 `download_futu_market_data.py` 下载数据（详见下方章节），或在回测时指定 `--data-source futu`
4. 富途 API 限制：每30秒最多60次请求，框架内置了 `FutuRateLimiter` 自动控制频率

#### Q12: QMT 数据如何预下载？
使用 `download_qmt_market_data.py` 脚本（详见下方章节）：
```bash
python download_qmt_market_data.py --pool 中证1000 --type all
```

## 分钟级后复权数据转换（convert_minute_hfq.py）

QMT 本地只保留约 1 年的分钟线历史数据，无法满足长期回测需求。`convert_minute_hfq.py` 通过线性变换法，将聚宽下载的不复权分钟数据转换为后复权数据，并与 QMT 分钟数据合并，补足历史缺失部分。

### 核心原理

QMT 后复权价格与不复权价格是线性关系：

```
后复权价格 = 不复权价格 × a + b
```

- **a**：在非除权区间恒定；送股/转增时跳变（乘以 1+送股+转增）
- **b**：在除权除息日跳变，非除权日不变
- **volume/amount** 等非价格字段无需调整

通过日线 OHLC 四个数据点做最小二乘拟合，即可得到每天的 a 和 b，然后应用到分钟级数据上。

### 数据合并策略

| 区间 | 不复权数据 | 后复权数据 |
|------|-----------|-----------|
| QMT 有分钟数据的区间 | 直接使用 QMT 原始数据 | 直接使用 QMT 后复权数据 |
| QMT 无分钟数据的区间 | 使用聚宽 CSV 不复权数据 | 聚宽不复权 × a + b 转换 |

合并时 QMT 优先，聚宽仅补足 QMT 缺失的历史部分，两者在交接点因子完全一致，无缝衔接。

### 聚宽 CSV 文件要求

文件放置在 `.cache/JQData/market_raw/` 目录下，命名格式为 `{股票代码}_1m_{起始日期}_{结束日期}.csv`，例如：

```
.cache/JQData/market_raw/
├── 000001_SZ_1m_1990-01-01_2026-05-19.csv
├── 601398_SH_1m_1990-01-01_2026-05-19.csv
├── 601288_SH_1m_1990-01-01_2026-05-19.csv
└── ...
```

CSV 列格式要求：`datetime,code,open,close,high,low,volume,money`（聚宽标准格式），脚本会自动处理：
- `money` 列重命名为 `amount`
- `volume` 列从股转换为手（÷100），与 QMT 单位对齐
- `code` 列自动移除

### 请求参数

```bash
python convert_minute_hfq.py [OPTIONS]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--stocks` | 是 | — | 股票代码，逗号分隔，如 `000001.SZ,600519.SH` |
| `--start` | 否 | `19900101` | 数据起始日期 |
| `--end` | 否 | `20991231` | 数据结束日期 |
| `--force` | 否 | `False` | 强制覆盖已有缓存 |
| `--jq` | 否 | `False` | 启用聚宽 CSV 补足历史数据 |
| `--jq-dir` | 否 | `.cache/JQData/market_raw` | 聚宽 CSV 文件目录 |
| `--info` | 否 | `False` | 仅查看缓存信息，不转换 |
| `-v` / `--verbose` | 否 | `False` | 启用详细日志 |

### 存储位置

转换后的数据保存到 QMT 缓存目录，与日线数据使用相同的按年份分片格式：

```
.cache/QMTData/
├── market/                          # 后复权分钟线
│   └── {symbol}/
│       ├── 2005_1m.parquet
│       ├── 2006_1m.parquet
│       └── ...
└── market_raw/                      # 不复权分钟线
    └── {symbol}/
        ├── 2005_1m.parquet
        ├── 2006_1m.parquet
        └── ...
```

### 使用示例

```bash
# 1. 仅使用 QMT 数据转换（不补足历史）
python convert_minute_hfq.py --stocks 000001.SZ

# 2. 使用聚宽 CSV 补足历史（推荐）
python convert_minute_hfq.py --stocks 000001.SZ --jq

# 3. 批量转换多只股票
python convert_minute_hfq.py --stocks 000001.SZ,601398.SH,601288.SH,601939.SH,601988.SH --jq

# 4. 强制覆盖已有缓存（重新生成）
python convert_minute_hfq.py --stocks 000001.SZ --jq --force

# 5. 指定聚宽 CSV 目录
python convert_minute_hfq.py --stocks 000001.SZ --jq --jq-dir /path/to/csv

# 6. 指定日期范围
python convert_minute_hfq.py --stocks 000001.SZ --jq --start 2020-01-01 --end 2026-05-20

# 7. 查看缓存信息
python convert_minute_hfq.py --stocks 000001.SZ --info

# 8. Python API 调用
python -c "from convert_minute_hfq import convert_minute_data; convert_minute_data('000001.SZ', jq=True, force=True)"
```

### 数据精度验证

经 12 只股票（含仅派息、送股转增、创业板、科创板等不同类型）约 70 万根分钟 K 线验证：

| 对比项 | 结果 |
|--------|------|
| QMT 数据区间后复权 vs QMT 直接获取 | 100% 精确匹配（误差在浮点精度级别） |
| 聚宽转换后复权 close vs QMT 后复权 | 100% 匹配（<0.01） |
| 聚宽转换后复权 open/high/low vs QMT | 99.5%+ 匹配，微小差异来源于数据源撮合精度不同 |
| volume（聚宽÷100后）vs QMT | 94%~97% 相同 |
| amount vs QMT | 87%~90% 相同 |
| 交接点因子一致性 | 差异为 0，无缝衔接 |

### 常见问题

#### Q1: 不使用聚宽 CSV 可以吗？
可以。不加 `--jq` 参数时，仅转换 QMT 自身的分钟数据（约 1 年历史），不补足历史。

#### Q2: 聚宽 CSV 的 volume 单位与 QMT 不同怎么办？
脚本已自动处理。聚宽 volume 单位是股，QMT 是手（1手=100股），读取时会自动 ÷100 转换。

#### Q3: 如何确认转换结果正确？
使用 `--info` 查看缓存年份范围，或在 Python 中读取缓存数据与 QMT 直接获取的数据对比。

#### Q4: 重新运行会覆盖已有数据吗？
默认跳过已有年份文件。使用 `--force` 强制覆盖所有年份。

#### Q5: 转换一只股票需要多长时间？
取决于数据量。约 120 万根 K 线（20年分钟数据）转换+写入约需 60~90 秒。

## QMT 数据下载（download_qmt_market_data.py）

`download_qmt_market_data.py` 用于从 QMT（迅投 MiniQMT）预下载行情和财务数据到本地缓存。QMT 提供实时行情和交易接口，但本地只保留约 1 年的分钟线历史，日线历史完整。

### 前置条件

1. 安装 MiniQMT 客户端并登录
2. 安装 Python 依赖：`pip install xtquant`

### 数据类型

| 类型 | 参数值 | 说明 | 缓存目录 |
|------|--------|------|----------|
| 后复权行情 | `market` | 日线/分钟线后复权数据 | `.cache/QMTData/market/{symbol}/` |
| 不复权行情 | `market_raw` | 日线/分钟线不复权数据 | `.cache/QMTData/market_raw/{symbol}/` |
| 财务数据 | `financial` | 利润表、资产负债表、现金流量表等 | `.cache/QMTData/financial/{symbol}/` |
| 全部 | `all` | 同时下载上述三种 | — |

### 请求参数

```bash
python download_qmt_market_data.py [OPTIONS]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--pool` | 否* | — | 股票池：`沪深300`、`中证500`、`中证1000`、`上证50`、`全部A股` |
| `--stocks` | 否* | — | 手动指定股票代码，逗号分隔，如 `000001.SZ,600000.SH` |
| `--start` | 否 | `1990-01-01` | 数据起始日期，格式 `YYYY-MM-DD` |
| `--end` | 否 | 当前日期 | 数据结束日期，格式 `YYYY-MM-DD` |
| `--type` | 否 | `all` | 数据类型：`market`、`market_raw`、`financial`、`all` |
| `--period` | 否 | `1d` | 行情周期：`1d`（日线）、`1m`（分钟线） |
| `--force` | 否 | `False` | 强制重新下载，忽略已有缓存 |
| `--info` | 否 | `False` | 仅显示缓存信息，不下载 |
| `-v` / `--verbose` | 否 | `False` | 启用详细日志 |

> *`--pool` 和 `--stocks` 二选一，都不指定则下载全部 A 股。

### 下载流程

1. **解析股票列表**：根据 `--pool` 或 `--stocks` 获取目标股票代码
2. **数据就绪检查**：`_check_data_ready()` 调用 `xtdata.download_history_data()` 确保数据已下载到本地
3. **逐年分片获取**：通过 `xtdata.get_market_data_ex()` 按年份获取数据
4. **缓存写入**：通过 `smart_cache` 装饰器自动按年份分片写入 Parquet 文件
5. **增量更新**：已有年份的缓存自动跳过，仅获取缺失部分

### 存储位置

```
.cache/QMTData/
├── market/                          # 后复权行情
│   └── {symbol}/
│       ├── 2020_1d.parquet
│       ├── 2020_1m.parquet
│       └── ...
├── market_raw/                      # 不复权行情
│   └── {symbol}/
│       └── ...
└── financial/                       # 财务数据
    └── {symbol}/
        ├── 2020_income.parquet
        ├── 2020_balance_sheet.parquet
        ├── 2020_cash_flow.parquet
        └── ...
```

### 使用示例

```bash
# 1. 下载沪深300全部数据（行情+财务）
python download_qmt_market_data.py --pool 沪深300 --type all

# 2. 仅下载后复权日线行情
python download_qmt_market_data.py --pool 中证1000 --type market --period 1d

# 3. 下载指定股票的分钟线数据
python download_qmt_market_data.py --stocks 000001.SZ,600519.SH --type market --period 1m

# 4. 仅下载财务数据
python download_qmt_market_data.py --pool 沪深300 --type financial

# 5. 强制重新下载
python download_qmt_market_data.py --stocks 000001.SZ --type all --force

# 6. 查看缓存信息
python download_qmt_market_data.py --stocks 000001.SZ --info

# 7. 指定日期范围
python download_qmt_market_data.py --pool 沪深300 --type market --start 2020-01-01 --end 2026-05-20
```

### 常见问题

#### Q1: QMT 客户端需要一直运行吗？
下载时需要 MiniQMT 客户端在线。下载完成后数据缓存在本地，回测时无需客户端。

#### Q2: QMT 分钟线数据只有约 1 年，如何获取更长历史？
配合 `convert_minute_hfq.py` 使用聚宽 CSV 补足历史（详见上方章节）。

#### Q3: 下载速度慢怎么办？
QMT 数据通过本地接口获取，速度取决于 MiniQMT 客户端的数据同步状态。首次下载可能较慢，后续增量更新较快。

#### Q4: 不复权数据如何用于回测？
策略在 `@register_strategy` 的 `default_kwargs` 中指定 `dividend_type='none'`，回测时框架自动从 `market_raw` 目录加载不复权数据。预下载不复权数据后，回测时 `smart_cache` 直接读取本地缓存，不会重复下载。详见 [策略开发文档](strategy-development.md#使用不复权数据回测)。

```bash
# 预下载不复权日线数据（回测前执行一次即可）
python download_qmt_market_data.py --pool 全部A股 --type market_raw --period 1d

# 预下载不复权分钟线数据
python download_qmt_market_data.py --pool 全部A股 --type market_raw --period 1m
```

## 财务数据下载（download_qmt_financial_data.py）

`download_qmt_financial_data.py` 用于从 OpenData（akshare）批量下载 A 股财务数据，包括利润表、资产负债表、现金流量表等，按年份分片缓存到本地。

### 数据来源

| 数据表 | akshare 接口 | 说明 |
|--------|-------------|------|
| 利润表 | `stock_profit_sheet_by_report_em` | 营收、净利润、EPS 等 |
| 资产负债表 | `stock_balance_sheet_by_report_em` | 资产、负债、股东权益等 |
| 现金流量表 | `stock_cash_flow_sheet_by_report_em` | 经营/投资/筹资现金流等 |
| 业绩预告 | `stock_profit_forecast_em` | 业绩预告数据 |
| 业绩快报 | `stock_profit_express_em` | 业绩快报数据 |

### 请求参数

```bash
python download_qmt_financial_data.py [OPTIONS]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--pool` | 否* | 沪深A股 | 股票池：`沪深300`、`中证500`、`中证1000`、`上证50`、`沪深A股` |
| `--stocks` | 否* | — | 手动指定股票代码，逗号分隔 |
| `--start` | 否 | `2015-01-01` | 数据起始日期，格式 `YYYY-MM-DD` |
| `--end` | 否 | 当前日期 | 数据结束日期，格式 `YYYY-MM-DD` |
| `--workers` | 否 | `3` | 并发线程数（建议不超过 5，避免触发限频） |
| `--force` | 否 | `False` | 强制重新下载，忽略已有缓存 |
| `--verify` | 否 | `False` | 下载后校验数据完整性 |
| `--verify-only` | 否 | `False` | 仅校验已有数据，不执行新下载 |
| `-v` / `--verbose` | 否 | `False` | 启用详细日志 |
| `--log` | 否 | `False` | 将日志同时写入文件 |

> *`--pool` 和 `--stocks` 二选一，都不指定则默认下载沪深A股全部股票。

### 存储位置

```
.cache/OpenData/financial/
└── {symbol}/
    ├── 2020_income.parquet          # 利润表
    ├── 2020_balance_sheet.parquet   # 资产负债表
    ├── 2020_cash_flow.parquet       # 现金流量表
    └── ...
```

### 使用示例

```bash
# 1. 下载沪深300全部财务数据
python download_qmt_financial_data.py --pool 沪深300 --start 2015-01-01

# 2. 下载指定股票的财务数据
python download_qmt_financial_data.py --stocks 000001.SZ,600519.SH --start 2020-01-01

# 3. 下载并校验
python download_qmt_financial_data.py --pool 中证1000 --start 2015-01-01 --verify

# 4. 仅校验已有数据
python download_qmt_financial_data.py --pool 沪深300 --start 2015-01-01 --verify-only

# 5. 强制重新下载
python download_qmt_financial_data.py --pool 沪深300 --start 2015-01-01 --force

# 6. 写入日志文件
python download_qmt_financial_data.py --pool 沪深A股 --start 2015-01-01 --workers 5 --log
```

### 常见问题

#### Q1: 财务数据下载报错 "akshare 未安装"
**解决**：安装依赖 `pip install akshare pyarrow`

#### Q2: 下载速度慢怎么办？
akshare 数据源有频率限制，建议 `--workers` 不超过 5。全市场下载可能需要数小时，建议配合 `--log` 使用。

#### Q3: 财务数据更新频率？
财务数据按季度发布，建议每季度末更新一次。使用 `--force` 可强制刷新。

#### Q4: QMT 财务数据和 OpenData 财务数据有什么区别？
两者来源不同（QMT 来自交易所原始数据，OpenData 来自东方财富），字段名称和口径可能略有差异。回测框架默认优先使用 QMT 财务数据，OpenData 作为补充。

## 富途数据下载（download_futu_market_data.py）

`download_futu_market_data.py` 用于从富途 OpenD 行情网关下载 A 股、港股、美股行情数据，支持自动增量下载和频率限制。

### 前置条件

1. 开通富途证券账户
2. 安装依赖：`pip install futu-api`
3. 下载并启动 [富途 OpenD 行情网关](https://www.futunn.com/download/openAPI)
4. OpenD 启动后默认监听 `127.0.0.1:33333`

### 数据类型

| 类型 | 参数值 | 说明 | 支持市场 |
|------|--------|------|----------|
| 后复权行情 | `adjusted` | 日线后复权数据 | A股、港股、美股 |
| 不复权行情 | `raw` | 日线不复权数据 | A股、港股、美股 |
| 全部 | `all` | 同时下载上述两种 | — |

### 请求参数

```bash
python download_futu_market_data.py [OPTIONS]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--pool` | 否* | — | 股票池：`沪深300`、`中证500`、`中证1000`、`上证50`、`港股成分`、`美股成分` |
| `--stocks` | 否* | — | 手动指定股票代码，逗号分隔，如 `HK.00700,US.AAPL` |
| `--start` | 否 | `2020-01-01` | 数据起始日期，格式 `YYYY-MM-DD` |
| `--end` | 否 | 当前日期 | 数据结束日期，格式 `YYYY-MM-DD` |
| `--type` | 否 | `all` | 数据类型：`adjusted`、`raw`、`all` |
| `--force` | 否 | `False` | 强制重新下载，忽略已有缓存 |
| `--host` | 否 | `127.0.0.1` | OpenD 网关地址 |
| `--port` | 否 | `33333` | OpenD 网关端口 |
| `-v` / `--verbose` | 否 | `False` | 启用详细日志 |

> *`--pool` 和 `--stocks` 二选一。

### 频率限制

富途 API 限制：每 30 秒最多 60 次请求。框架内置 `FutuRateLimiter` 自动控制请求频率，无需手动处理。

### 存储位置

```
.cache/FutuData/
├── market/                          # 后复权行情
│   └── {symbol}/
│       ├── 2020_1d.parquet
│       └── ...
└── market_raw/                      # 不复权行情
    └── {symbol}/
        └── ...
```

### 使用示例

```bash
# 1. 下载沪深300行情数据
python download_futu_market_data.py --pool 沪深300 --start 2020-01-01

# 2. 下载港股数据
python download_futu_market_data.py --stocks HK.00700,HK.09988 --start 2020-01-01

# 3. 下载美股数据
python download_futu_market_data.py --stocks US.AAPL,US.TSLA --start 2020-01-01

# 4. 指定 OpenD 地址
python download_futu_market_data.py --pool 沪深300 --start 2020-01-01 --host 192.168.1.100 --port 33333

# 5. 强制重新下载
python download_futu_market_data.py --stocks HK.00700 --start 2020-01-01 --force

# 6. 仅下载后复权数据
python download_futu_market_data.py --pool 中证1000 --start 2020-01-01 --type adjusted
```

### 常见问题

#### Q1: 报错 "FutuOpenD 连接失败"
确认富途 OpenD 行情网关已启动，默认监听 `127.0.0.1:33333`。可通过 `--host` 和 `--port` 指定地址。

#### Q2: 港股/美股代码格式？
- 港股：`HK.00700`（腾讯控股）、`HK.09988`（阿里巴巴）
- 美股：`US.AAPL`（苹果）、`US.TSLA`（特斯拉）

#### Q3: 富途数据与 QMT/OpenData 数据有什么区别？
- **数据范围**：富途支持 A股+港股+美股，QMT 仅 A股，OpenData 仅 A股
- **数据精度**：富途和 QMT 均来自交易所原始数据，精度高于 OpenData
- **使用门槛**：富途需开户+OpenD 网关，QMT 需开户+客户端，OpenData 免费

#### Q4: 下载速度慢怎么办？
富途 API 有频率限制（30秒60次），框架已内置限速器。大批量下载建议分批进行。

## 富途资金流向数据下载（download_futu_capital_flow.py）

`download_futu_capital_flow.py` 用于从富途 OpenD 行情网关批量下载 A 股资金流向数据（日线），支持增量下载、自动重试和频率限制。

### 前置条件

1. 开通富途证券账户
2. 安装依赖：`pip install futu-api`
3. 下载并启动 [富途 OpenD 行情网关](https://www.futunn.com/download/openAPI)
4. OpenD 启动后默认监听 `127.0.0.1:11111`

### 数据说明

- 数据类型：A 股日线资金流向（主力/中户/散户资金流入流出）
- 自动获取沪深两市全部 A 股列表（排除 ST 股票）
- 支持增量下载：已下载的股票自动跳过
- 自动重试：首次下载失败的股票会自动重试，内置频率限制处理

### 存储位置

```
.cache/FutuData/capital_flow/
├── download_index.json              # 下载索引（记录每只股票状态）
├── SH_600000/
│   └── capital_flow_day.parquet     # 日线资金流向数据
├── SZ_000001/
│   └── capital_flow_day.parquet
└── ...
```

### 使用方式

```bash
# 直接运行（下载全部 A 股资金流向数据）
python download_futu_capital_flow.py
```

该脚本无需命令行参数，运行后自动：
1. 连接富途 OpenD 服务
2. 获取沪深两市全部 A 股（排除 ST）
3. 逐只下载日线资金流向数据
4. 自动跳过已下载的股票
5. 首轮下载完成后自动重试失败的股票
6. 每 200 只股票自动保存下载索引

### 频率限制

- 请求间隔：0.35 秒/只
- 重试阶段：每 25 次请求后等待 32 秒
- 遇到频率限制自动等待 35 秒

### 下载索引

`download_index.json` 记录每只股票的下载状态：

| 状态 | 说明 |
|------|------|
| `ok` | 下载成功，含行数和时间范围 |
| `fail` | 下载失败，含错误信息 |
| `empty` | 数据为空 |

## 数据下载脚本总览

| 脚本 | 数据源 | 数据类型 | 支持市场 | 认证方式 |
|------|--------|----------|----------|----------|
| `download_open_market_data.py` | OpenData（腾讯财经）+ QMT 补充 | 日线行情（后复权/不复权） | A股 | 免费（akshare） |
| `download_qmt_market_data.py` | QMT（MiniQMT） | 日线/分钟线行情 + 财务数据 | A股 | 需开户+客户端 |
| `download_qmt_financial_data.py` | OpenData（东方财富） | 财务报表（利润表/资产负债表/现金流等） | A股 | 免费（akshare） |
| `download_futu_market_data.py` | 富途（OpenD） | 日线行情（后复权/不复权） | A股+港股+美股 | 需开户+OpenD |
| `download_futu_capital_flow.py` | 富途（OpenD） | 日线资金流向 | A股 | 需开户+OpenD |
| `convert_minute_hfq.py` | QMT + 聚宽CSV | 分钟线行情（后复权/不复权） | A股 | QMT需开户+客户端 |

### 典型使用场景

| 场景 | 推荐脚本 | 命令 |
|------|---------|------|
| 首次搭建日线回测环境 | `download_open_market_data.py` | `--pool 沪深300 --start 2020-01-01 --end 2026-12-31 --type all --verify` |
| 补充 QMT 实时数据 | `download_qmt_market_data.py` | `--pool 沪深300 --type all` |
| 获取分钟线历史数据 | `convert_minute_hfq.py` | `--stocks 000001.SZ --jq` |
| 下载财务数据 | `download_qmt_financial_data.py` | `--pool 沪深300 --start 2015-01-01` |
| 获取港股/美股数据 | `download_futu_market_data.py` | `--stocks HK.00700,US.AAPL --start 2020-01-01` |
| 下载资金流向数据 | `download_futu_capital_flow.py` | 直接运行 `python download_futu_capital_flow.py` |
