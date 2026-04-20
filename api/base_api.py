from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class BaseAPI(ABC):
    """API基类"""
    
    @abstractmethod
    def add_strategy(self, strategy_class, **kwargs):
        """添加策略"""
        pass
    
    @abstractmethod
    def run(self):
        """运行"""
        pass
    
    def set_cash(self, cash: float):
        """设置初始资金"""
        pass
    
    def set_commission(self, commission: float):
        """设置佣金"""
        pass
    
    def add_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d"):
        """添加数据"""
        pass
    
    def plot(self, **kwargs):
        """绘制结果"""
        pass
