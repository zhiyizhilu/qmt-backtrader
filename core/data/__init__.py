# core.data 包 — 数据处理器模块
# 拆分自原来的 core/data.py，保持向后兼容的导入路径

from core.data.base import (
    DataProcessor,
    _merged_dict_to_parquet,
    _parquet_to_merged_dict,
    _restore_dataframe_index,
)
from core.data.qmt import QMTDataProcessor
from core.data.csv import CSVDataProcessor
from core.data.akshare import (
    AKShareDataProcessor,
    _convert_symbol_to_qmt,
    _convert_symbol_to_akshare,
    _convert_symbol_to_akshare_financial,
    _convert_symbol_to_baostock,
)
from core.data.baostock import BaoStockDataProcessor
from core.data.factory import create_data_processor

__all__ = [
    'DataProcessor',
    'QMTDataProcessor',
    'CSVDataProcessor',
    'AKShareDataProcessor',
    'BaoStockDataProcessor',
    'create_data_processor',
    '_merged_dict_to_parquet',
    '_parquet_to_merged_dict',
    '_restore_dataframe_index',
    '_convert_symbol_to_qmt',
    '_convert_symbol_to_akshare',
    '_convert_symbol_to_akshare_financial',
    '_convert_symbol_to_baostock',
]
