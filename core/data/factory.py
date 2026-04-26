from core.data.base import DataProcessor
from core.data.qmt import QMTDataProcessor


def create_data_processor(fallback_to_simulated: bool = True, proxy: str = '') -> DataProcessor:
    """数据处理器工厂函数

    默认使用 QMT 为主数据源，当 QMT 数据不足时自动用 OpenData 补充。

    Args:
        fallback_to_simulated: 数据获取失败时是否降级为模拟数据
        proxy: 代理地址，格式 'host:port'（已弃用，保留参数用于兼容性）

    Returns:
        DataProcessor 实例（QMT为主，OpenData补充）
    """
    return QMTDataProcessor(fallback_to_simulated=fallback_to_simulated)
