# 智能缓存系统

框架内置了双层智能缓存系统，大幅提升数据加载速度。

## 缓存架构

缓存系统已重构为子包结构（`core/cache/`），包含以下模块：

| 模块 | 类 | 职责 |
|------|-----|------|
| `manager.py` | SmartCacheManager | 缓存核心调度器（单例），支持内存+磁盘+增量合并+按年份分片 |
| `mem_cache.py` | MemCache | 基于 OrderedDict 的线程安全 LRU 内存缓存，默认容量 500 |
| `disk_cache.py` | DiskCache | 磁盘缓存，支持 parquet 和 pickle 两种格式，按命名空间隔离 |
| `index_manager.py` | CacheIndexManager | 缓存索引管理，维护行情/财报数据的年份索引，支持增量更新 |

## 缓存特性

1. **逐只缓存**：财务数据按股票代码单独缓存，中断后自动恢复，避免重复下载
2. **合并缓存**：全部下载完成后自动合并为总缓存，提升后续加载速度
3. **增量更新**：行情数据支持智能增量更新，只下载缺失部分
4. **格式优化**：默认使用 pyarrow parquet 格式，比 pickle 更快更紧凑
5. **原子写入**：使用临时文件 + 原子重命名，避免写入中断导致文件损坏
6. **缓存索引**：维护年份索引，快速判断缓存覆盖范围，避免重复下载
7. **按年份分片**：行情/财报数据按年份分片存储，一个股票一年一个文件
8. **向后兼容**：旧格式缓存文件仍可读取

## 缓存目录

缓存文件默认存储在项目根目录的 `.cache/` 文件夹下：
```
.cache/
├── index/                           # 缓存索引
│   ├── market_index.json            # 行情数据索引
│   └── financial_index.json         # 财务数据索引
├── lifecycle/                       # 股票生命周期缓存
│   └── stock_lifecycle.json         # 上市/退市时间数据
├── QMTDataProcessor/                # QMT 行情数据缓存
├── QMTDataProcessor_Financial/      # QMT 财务数据缓存
├── FutuData/                        # 富途行情数据缓存
│   ├── market/                      # 后复权行情
│   │   └── {symbol}/{year}_{period}.parquet
│   └── market_raw/                  # 不复权行情
│       └── {symbol}/{year}_{period}.parquet
├── OpenData/                        # OpenData 行情数据缓存
│   ├── market/                      # 后复权行情
│   │   └── {symbol}/{year}_{period}.parquet
│   ├── market_raw/                  # 不复权行情
│   │   └── {symbol}/{year}_{period}.parquet
│   ├── vix/                         # QVIX 隐含波动率缓存
│   │   └── QVIX_510500.SH.parquet
│   └── financial/                   # OpenData 财务数据缓存
│       ├── 000001.SZ_Balance.parquet
│       ├── 000001.SZ_CashFlow.parquet
│       ├── 000001.SZ_Income.parquet
│       └── 000001.SZ_Pershareindex.parquet
└── JQData/                          # 聚宽历史成分股数据
    ├── index_constituent/           # 指数历史成分股 CSV
    └── industry_constituent/        # 申万行业历史成分股 CSV
```

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `QMT_CACHE_DIR` | `.cache` | 缓存数据存储目录 |
| `QMT_MEM_CACHE_LIMIT` | `500` | 内存缓存最大对象数量 |
| `QMT_LOG_LEVEL` | `INFO` | 日志级别（`DEBUG` 输出详细调试信息） |

## YAML 配置

缓存配置也可通过 YAML 配置文件设置：

```yaml
cache:
  dir: .cache
  mem_limit: 500
  max_age_days: 30
```
