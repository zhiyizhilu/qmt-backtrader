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
from core.data.opendata import (
    OpenDataProcessor,
    _convert_symbol_to_qmt,
    _convert_symbol_to_opendata,
    get_qvix_data,
)
from core.data.futu import FutuDataProcessor, FutuServiceError
from core.data.factory import create_data_processor
from core.data.index_constituent import IndexConstituentManager
from core.data.industry_constituent import IndustryConstituentManager

__all__ = [
    'DataProcessor',
    'QMTDataProcessor',
    'CSVDataProcessor',
    'OpenDataProcessor',
    'FutuDataProcessor',
    'FutuServiceError',
    'IndexConstituentManager',
    'IndustryConstituentManager',
    'create_data_processor',
    'get_qvix_data',
    '_merged_dict_to_parquet',
    '_parquet_to_merged_dict',
    '_restore_dataframe_index',
    '_convert_symbol_to_qmt',
    '_convert_symbol_to_opendata',
]
