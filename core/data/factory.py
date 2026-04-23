from core.data.base import DataProcessor
from core.data.qmt import QMTDataProcessor
from core.data.akshare import AKShareDataProcessor
from core.data.baostock import BaoStockDataProcessor
from core.data.csv import CSVDataProcessor


def create_data_processor(data_source: str = 'qmt', fallback_to_simulated: bool = True) -> DataProcessor:
    """数据处理器工厂函数

    Args:
        data_source: 数据源，可选 'qmt', 'akshare', 'baostock'
        fallback_to_simulated: 数据获取失败时是否降级为模拟数据

    Returns:
        DataProcessor 实例
    """
    if data_source == 'akshare':
        return AKShareDataProcessor(fallback_to_simulated=fallback_to_simulated)
    elif data_source == 'baostock':
        return BaoStockDataProcessor(fallback_to_simulated=fallback_to_simulated)
    elif data_source == 'qmt':
        return QMTDataProcessor(fallback_to_simulated=fallback_to_simulated)
    else:
        raise ValueError(f"不支持的数据源: {data_source}，可选: 'qmt', 'akshare', 'baostock'")
