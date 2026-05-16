# 智能缓存系统

框架内置了双层智能缓存系统，大幅提升数据加载速度。

## 缓存架构

- **内存缓存（MemCache）**：基于 OrderedDict 的线程安全 LRU 缓存，默认容量 500
- **磁盘缓存（DiskCache）**：支持 parquet 和 pickle 两种格式，按命名空间隔离
- **缓存索引（CacheIndexManager）**：维护行情/财报数据的年份索引，支持增量更新

## 缓存特性

1. **逐只缓存**：财务数据按股票代码单独缓存，中断后自动恢复，避免重复下载
2. **合并缓存**：全部下载完成后自动合并为总缓存，提升后续加载速度
3. **增量更新**：行情数据支持智能增量更新，只下载缺失部分
4. **格式优化**：默认使用 pyarrow parquet 格式，比 pickle 更快更紧凑
5. **原子写入**：使用临时文件 + 原子重命名，避免写入中断导致文件损坏
6. **缓存索引**：维护年份索引，快速判断缓存覆盖范围，避免重复下载

## 缓存目录

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

## 环境变量

- `QMT_CACHE_DIR`：自定义缓存目录路径
- `QMT_MEM_CACHE_LIMIT`：内存缓存容量限制（默认 500）
- `QMT_LOG_LEVEL`：日志级别（默认 `INFO`，设为 `DEBUG` 输出详细调试信息）
