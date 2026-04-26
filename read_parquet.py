import pandas as pd

# 定义文件路径
files = {
    "资产负债表 (Balance)": "e:\\jupyter notebook\\automatic\\qmt_backtrader\\.cache\\OpenData\\financial\\000001.SZ_Balance.parquet",
    "现金流量表 (CashFlow)": "e:\\jupyter notebook\\automatic\\qmt_backtrader\\.cache\\OpenData\\financial\\000001.SZ_CashFlow.parquet",
    "利润表 (Income)": "e:\\jupyter notebook\\automatic\\qmt_backtrader\\.cache\\OpenData\\financial\\000001.SZ_Income.parquet",
    "每股指标 (Pershareindex)": "e:\\jupyter notebook\\automatic\\qmt_backtrader\\.cache\\OpenData\\financial\\000001.SZ_Pershareindex.parquet"
}

for name, path in files.items():
    print("=" * 80)
    print(f"【{name}】")
    print(f"文件路径: {path}")
    print("=" * 80)

    try:
        df = pd.read_parquet(path)

        print(f"\n数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"\n数据类型:\n{df.dtypes}")
        print(f"\n前5行数据:")
        print(df.head())
        print(f"\n数据概览 (describe):")
        print(df.describe())
        print("\n")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        print("\n")
