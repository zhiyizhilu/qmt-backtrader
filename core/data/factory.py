from core.data.base import DataProcessor
from core.data.qmt import QMTDataProcessor
from core.data.opendata import OpenDataProcessor


def create_data_processor(fallback_to_simulated: bool = False, proxy: str = '',
                         use_opendata: bool = True, data_source: str = 'qmt') -> DataProcessor:
    """数据处理器工厂函数

    Args:
        fallback_to_simulated: 数据获取失败时是否降级为模拟数据（默认False，直接抛异常）
        proxy: 代理地址，格式 'host:port'（已弃用，保留参数用于兼容性）
        use_opendata: 是否使用 OpenData 作为补充数据源，默认为 True
        data_source: 数据源类型，可选 'qmt'(默认), 'open'(OpenData), 'futu'(富途本地数据)

    Returns:
        DataProcessor 实例
    """
    if data_source == 'futu':
        from core.data.futu import FutuDataProcessor
        return FutuDataProcessor(fallback_to_simulated=fallback_to_simulated)

    if data_source == 'open':
        return OpenDataProcessor(fallback_to_simulated=fallback_to_simulated)

    return QMTDataProcessor(fallback_to_simulated=fallback_to_simulated, use_opendata=use_opendata)
