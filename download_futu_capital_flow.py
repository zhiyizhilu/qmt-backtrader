import os
import sys
import time
import json
import pandas as pd
from futu import OpenQuoteContext, PeriodType, RET_OK, Market, SecurityType

HOST = '127.0.0.1'
PORT = 11111
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'FutuData', 'capital_flow')
INDEX_FILE = os.path.join(SAVE_DIR, 'download_index.json')

os.makedirs(SAVE_DIR, exist_ok=True)

quote_ctx = OpenQuoteContext(host=HOST, port=PORT)

try:
    print("=" * 60)
    print("富途A股资金流向数据下载")
    print("=" * 60)

    print("\n[1] 连接 OpenD ...")
    ret, data = quote_ctx.get_global_state()
    if ret != RET_OK:
        print(f"    连接失败: {data}")
        sys.exit(1)
    print(f"    连接成功!")

    print("\n[2] 获取A股股票列表 ...")
    ret_sh, data_sh = quote_ctx.get_stock_basicinfo(Market.SH, SecurityType.STOCK)
    ret_sz, data_sz = quote_ctx.get_stock_basicinfo(Market.SZ, SecurityType.STOCK)

    all_stocks = []
    if ret_sh == RET_OK:
        all_stocks.append(data_sh)
        print(f"    沪市股票: {len(data_sh)} 只")
    if ret_sz == RET_OK:
        all_stocks.append(data_sz)
        print(f"    深市股票: {len(data_sz)} 只")

    if not all_stocks:
        print("    获取股票列表失败!")
        sys.exit(1)

    stock_df = pd.concat(all_stocks, ignore_index=True)
    stock_df = stock_df[~stock_df['name'].str.contains('ST', case=False, na=False)]
    stock_codes = stock_df['code'].tolist()
    print(f"    去除ST后: {len(stock_codes)} 只")

    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            download_index = json.load(f)
        print(f"    已有下载记录: {len(download_index)} 只")
    else:
        download_index = {}

    print(f"\n[3] 开始下载资金流向数据 (日线, 约1年) ...")
    print(f"    保存目录: {SAVE_DIR}")
    print(f"    文件格式: 股票代码/capital_flow_day.parquet")

    success_count = 0
    fail_count = 0
    skip_count = 0
    total = len(stock_codes)

    for i, code in enumerate(stock_codes):
        stock_dir = os.path.join(SAVE_DIR, code.replace('.', '_'))
        parquet_file = os.path.join(stock_dir, 'capital_flow_day.parquet')

        if os.path.exists(parquet_file) and code in download_index:
            skip_count += 1
            continue

        ret, data = quote_ctx.get_capital_flow(code, period_type=PeriodType.DAY)

        if ret != RET_OK:
            fail_count += 1
            if i % 100 == 0:
                print(f"    [{i+1}/{total}] {code} 获取失败: {data}")
            download_index[code] = {'status': 'fail', 'error': str(data), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            time.sleep(0.3)
            continue

        if len(data) == 0:
            fail_count += 1
            download_index[code] = {'status': 'empty', 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            time.sleep(0.3)
            continue

        os.makedirs(stock_dir, exist_ok=True)
        data.to_parquet(parquet_file, index=False, engine='pyarrow')

        success_count += 1
        download_index[code] = {
            'status': 'ok',
            'rows': len(data),
            'start': str(data['capital_flow_item_time'].iloc[0]),
            'end': str(data['capital_flow_item_time'].iloc[-1]),
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"    [{i+1}/{total}] 成功={success_count}, 失败={fail_count}, 跳过={skip_count}")

        if (i + 1) % 200 == 0:
            with open(INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(download_index, f, ensure_ascii=False, indent=2)
            print(f"    [自动保存] 已保存下载索引 ({len(download_index)} 条)")

        time.sleep(0.35)

    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(download_index, f, ensure_ascii=False, indent=2)

    # ========== 自动重试失败股票 ==========
    fail_codes = [code for code, info in download_index.items() if info.get('status') in ('fail', 'empty')]
    if fail_codes:
        print(f"\n{'='*60}")
        print(f"开始自动重试失败股票 (共 {len(fail_codes)} 只, 限速: 每30秒25次)")
        print(f"{'='*60}")

        retry_success = 0
        retry_fail = 0
        batch_count = 0

        for i, code in enumerate(fail_codes):
            stock_dir = os.path.join(SAVE_DIR, code.replace('.', '_'))
            parquet_file = os.path.join(stock_dir, 'capital_flow_day.parquet')

            ret, data = quote_ctx.get_capital_flow(code, period_type=PeriodType.DAY)

            if ret != RET_OK:
                retry_fail += 1
                download_index[code] = {'status': 'fail', 'error': str(data)[:100], 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
                if '频率' in str(data) or 'frequency' in str(data).lower():
                    print(f"    [{i+1}/{len(fail_codes)}] {code} 频率限制, 等待35秒...")
                    time.sleep(35)
                    batch_count = 0
                    continue
            elif len(data) == 0:
                retry_fail += 1
                download_index[code] = {'status': 'empty', 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            else:
                os.makedirs(stock_dir, exist_ok=True)
                data.to_parquet(parquet_file, index=False, engine='pyarrow')
                retry_success += 1
                download_index[code] = {
                    'status': 'ok',
                    'rows': len(data),
                    'start': str(data['capital_flow_item_time'].iloc[0]),
                    'end': str(data['capital_flow_item_time'].iloc[-1]),
                    'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                }

            batch_count += 1
            if batch_count >= 25:
                print(f"    [{i+1}/{len(fail_codes)}] 已发25次请求, 等待32秒... 重试成功={retry_success}, 仍失败={retry_fail}")
                time.sleep(32)
                batch_count = 0
            else:
                time.sleep(1.2)

            if (i + 1) % 100 == 0:
                with open(INDEX_FILE, 'w', encoding='utf-8') as f:
                    json.dump(download_index, f, ensure_ascii=False, indent=2)
                print(f"    [{i+1}/{len(fail_codes)}] 自动保存索引, 重试成功={retry_success}, 仍失败={retry_fail}")

        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(download_index, f, ensure_ascii=False, indent=2)

        print(f"\n    重试结果: 成功={retry_success}, 仍失败={retry_fail}")

    # ========== 最终统计 ==========
    final_ok = sum(1 for v in download_index.values() if v.get('status') == 'ok')
    final_fail = sum(1 for v in download_index.values() if v.get('status') != 'ok')

    print(f"\n{'='*60}")
    print(f"全部完成!")
    print(f"    首次下载: 成功={success_count}, 失败={fail_count}, 跳过={skip_count}")
    print(f"    累计成功: {final_ok}")
    print(f"    累计失败: {final_fail}")
    print(f"    索引文件: {INDEX_FILE}")
    print(f"{'='*60}")

finally:
    quote_ctx.close()
