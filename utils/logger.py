import logging
import os
from datetime import datetime


class Logger:
    """日志管理"""

    _logger_cache: dict = {}
    _global_file_handler: logging.FileHandler = None
    _global_log_file: str = None

    @staticmethod
    def get_logger(name: str, log_file: str = None) -> logging.Logger:
        """获取日志记录器（带缓存，避免重复创建）"""
        cache_key = f"{name}_{log_file or 'console'}"

        if cache_key in Logger._logger_cache:
            return Logger._logger_cache[cache_key]

        logger = logging.getLogger(cache_key)
        logger.setLevel(logging.DEBUG)

        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        Logger._logger_cache[cache_key] = logger
        return logger

    @staticmethod
    def get_default_logger(strategy_name: str = 'default') -> logging.Logger:
        """获取默认日志记录器"""
        log_file = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy_name}.log"
        return Logger.get_logger('quant_framework', log_file)

    @staticmethod
    def setup_global_file_handler(strategy_name: str = 'default', log_dir: str = 'logs') -> str:
        """设置全局文件日志 handler，确保所有模块的日志都输出到文件

        通过在根 logger 上添加 FileHandler，所有子 logger 的日志
        会通过传播机制自动写入文件。

        Args:
            strategy_name: 策略名称，用于文件命名
            log_dir: 日志文件目录

        Returns:
            日志文件路径
        """
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy_name}.log")

        # 如果已有全局文件 handler，先移除（避免重复写入）
        if Logger._global_file_handler is not None:
            logging.getLogger().removeHandler(Logger._global_file_handler)
            Logger._global_file_handler.close()

        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 创建文件 handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 添加到根 logger，所有子 logger 的日志会通过传播机制写入文件
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        # 确保根 logger 的级别足够低，不会过滤掉子 logger 的日志
        if root_logger.level > logging.DEBUG:
            root_logger.setLevel(logging.DEBUG)

        Logger._global_file_handler = file_handler
        Logger._global_log_file = log_file

        return log_file
