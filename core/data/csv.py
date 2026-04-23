import pandas as pd
import logging

from core.data.base import DataProcessor


class CSVDataProcessor(DataProcessor):
    """CSV数据处理器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    def get_data(self, symbol: str, start_date: str, end_date: str, file_path: str, **kwargs) -> pd.DataFrame:
        """从CSV文件获取数据

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV文件不存在: {file_path}")

        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV文件为空: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV文件格式错误: {file_path}, {e}")
        except Exception as e:
            raise ValueError(f"读取CSV文件失败: {file_path}, {e}")

        required_cols = {'open', 'high', 'low', 'close'}
        missing = required_cols - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"CSV文件缺少必要列: {missing}, 文件: {file_path}")

        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return self.preprocess_data(df)
