# 数据源与下载

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

## 行情数据下载功能详解（download_market_data.py）

`download_market_data.py` 是独立的行情数据预下载脚本，支持批量并发下载 A 股行情数据、完整性校验、缺失修复与交易日一致性检查。

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
python download_market_data.py [OPTIONS]
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
| 股票数据为空 | 记录为 `empty`，跳过并更新已检查索引（30天有效） |
| 下载异常 | 记录为 `failed`，不影响其他股票下载 |
| 缓存文件损坏 | 自动删除损坏文件，重新下载 |
| 并发写入冲突 | 线程锁 + 原子文件写入（tmp → rename） |
| 索引与磁盘不同步 | 自动检测并重新获取缺失年份 |
| 停牌股票重复下载 | 停牌区间标记后永久跳过 |

### 使用示例

```bash
# 1. 基础：下载沪深300后复权行情
python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --workers 5

# 2. 下载并校验数据完整性
python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type all --workers 5 --verify

# 3. 仅校验已有数据（不下载新数据）
python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --verify-only

# 4. 检查交易日一致性并生成 HTML 报告
python download_market_data.py --check --start 2020-01-01 --end 2026-12-31

# 5. 检查并修复，生成 HTML 报告
python download_market_data.py --check --fix --start 2020-01-01 --end 2026-12-31

# 6. dry-run 模式：预览修复计划但不执行
python download_market_data.py --check --fix --dry-run --start 2020-01-01 --end 2026-12-31

# 7. 强制重新下载（忽略已有缓存）
python download_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-01-01 --type raw --force

# 8. 指定股票代码下载
python download_market_data.py --stocks 000001.SZ,600000.SH,600519.SH --start 2020-01-01 --end 2026-01-01 --type all

# 9. 将日志写入文件（适合长时间运行）
python download_market_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-01-01 --type all --workers 20 --log --verify

# 10. 下载全市场数据并校验（推荐首次使用）
python download_market_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-12-31 --type all --workers 10 --log --verify
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
python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --cache-dir D:/market_cache

# 或通过环境变量
set QMT_CACHE_DIR=D:/market_cache
python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01
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
- 使用 `--log` 参数，日志文件保存在 `logs/` 目录下，命名格式：`{时间戳}_download_market_data.log`
- 使用 `--log-file` 自定义日志文件路径
