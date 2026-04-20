import logging
import os
from datetime import datetime


class Logger:
    """日志管理"""

    _logger_cache: dict = {}

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
        log_file = f"logs/{datetime.now().strftime('%Y-%m-%d')}_{strategy_name}.log"
        return Logger.get_logger('quant_framework', log_file)
