"""
从富途 request_history_kline 获取银行股后复权行情数据
保存到 .cache/FutuData/market/ 目录，格式与 OpenData 一致

目录结构: {symbol}/{year}_{period}.parquet
  - index: DatetimeIndex(name='datetime')
  - columns: ['open', 'high', 'low', 'close', 'volume']

注意：request_history_kline 返回字段含 code/name/time_key/open/close/high/low/
      pe_ratio/turnover_rate/volume/turnover/change_rate/last_close
      只取 open/high/low/close/volume 保存，与 OpenData 格式对齐
"""
import os
import sys
import pandas as pd
from futu import (
    OpenQuoteContext, KLType, AuType,
    RET_OK
)

# 银行股列表
BANKS = {
    'SH.601398': '601398.SH',   # 工商银行
    'SH.601939': '601939.SH',   # 建设银行
    'SH.601288': '601288.SH',   # 农业银行
    'SH.600036': '600036.SH',   # 招商银行
    'SH.601988': '601988.SH',   # 中国银行
}

CACHE_ROOT = 'e:/jupyter notebook/automatic/qmt_backtrader/.cache/FutuData/market'
YEARS = list(range(2020, 2027))


def fetch_history_kline(ctx, code, start, end, ktype, autype=AuType.HFQ):
    """分页获取历史K线数据，返回完整DataFrame。出错即中止。"""
    all_dfs = []
    page_req_key = None

    while True:
        ret, data, page_req_key = ctx.request_history_kline(
            code=code,
            start=start,
            end=end,
            ktype=ktype,
            autype=autype,
            max_count=1000,
            page_req_key=page_req_key
        )

        if ret != RET_OK:
            print(f"    获取失败: {data}")
            ctx.close()
            sys.exit(1)

        if not data.empty:
            all_dfs.append(data)
            print(f"    获取 {len(data)} 行 (page_req_key={'有' if page_req_key else '无'})")

        if page_req_key is None:
            break

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    return df


def save_to_parquet(df, symbol_dir, year, period):
    """将DataFrame保存为parquet，格式与OpenData一致"""
    if df.empty:
        return

    # 转换 time_key 为 datetime 索引
    df['datetime'] = pd.to_datetime(df['time_key'])
    df = df.set_index('datetime')
    df = df[['open', 'high', 'low', 'close', 'volume']]

    for col in df.columns:
        df[col] = df[col].astype(float)

    df = df.sort_index()

    fpath = os.path.join(symbol_dir, f'{year}_{period}.parquet')
    df.to_parquet(fpath, index=True)
    print(f"    => 保存 {fpath} ({len(df)} 行)")


# ============ 主流程 ============
ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

for futu_code, dir_name in BANKS.items():
    symbol_dir = os.path.join(CACHE_ROOT, dir_name)
    os.makedirs(symbol_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"处理 {futu_code} ({dir_name})")

    # ============ 日线 ============
    print(f"  获取后复权日线...")
    for year in YEARS:
        start = f'{year}-01-01'
        end = f'{year}-12-31'

        df = fetch_history_kline(ctx, futu_code, start, end, KLType.K_DAY, AuType.HFQ)
        if not df.empty:
            save_to_parquet(df, symbol_dir, year, '1d')
        else:
            print(f"    {year} 无日线数据")

    # ============ 1分钟线 ============
    print(f"  获取后复权1分钟线...")
    for year in YEARS:
        start = f'{year}-01-01'
        end = f'{year}-12-31'

        df = fetch_history_kline(ctx, futu_code, start, end, KLType.K_1M, AuType.HFQ)
        if not df.empty:
            save_to_parquet(df, symbol_dir, year, '1m')
        else:
            print(f"    {year} 无1分钟线数据")

ctx.close()

# ============ 验证 ============
print(f"\n{'='*60}")
print("验证保存的数据...")

for futu_code, dir_name in BANKS.items():
    symbol_dir = os.path.join(CACHE_ROOT, dir_name)
    if not os.path.exists(symbol_dir):
        continue
    print(f"\n{dir_name}:")
    for period in ['1d', '1m']:
        files = sorted([f for f in os.listdir(symbol_dir) if f.endswith(f'_{period}.parquet')])
        if files:
            print(f"  {period}: {files}")
            fpath = os.path.join(symbol_dir, files[-1])
            df = pd.read_parquet(fpath)
            print(f"    index.name={df.index.name}, columns={df.columns.tolist()}")
            print(f"    {df.head(2).to_string()}")

print(f"\n全部完成！数据保存在 {CACHE_ROOT}")
