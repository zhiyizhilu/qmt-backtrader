import logging
import os
from datetime import datetime


class InstanceLogFilter(logging.Filter):
    """按 instance_id 过滤日志，为日志消息添加 instance_id 上下文"""

    def __init__(self, instance_id: str = None):
        super().__init__()
        self.instance_id = instance_id

    def filter(self, record):
        record.instance_id = self.instance_id or '-'
        return True


class InstanceFormatter(logging.Formatter):
    """支持 instance_id 的日志格式化器

    自动为没有 instance_id 属性的 log record 填充默认值，
    避免非实例化 logger 的日志格式化报 KeyError。
    """

    def __init__(self, fmt=None, datefmt=None, style='%'):
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - [%(instance_id)s] - %(levelname)s - %(message)s'
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        if not hasattr(record, 'instance_id'):
            record.instance_id = '-'
        return super().format(record)


class Logger:
    """日志管理"""

    _logger_cache: dict = {}
    _global_file_handler: logging.FileHandler = None
    _global_log_file: str = None
    _instance_filters: dict = {}

    @staticmethod
    def get_logger(name: str, log_file: str = None, instance_id: str = None) -> logging.Logger:
        """获取日志记录器（带缓存，避免重复创建）

        Args:
            name: logger 名称
            log_file: 日志文件路径
            instance_id: 策略实例ID，传入后日志中会包含实例标识
        """
        cache_key = f"{name}_{log_file or 'console'}_{instance_id or 'default'}"

        if cache_key in Logger._logger_cache:
            return Logger._logger_cache[cache_key]

        logger = logging.getLogger(cache_key)
        logger.setLevel(logging.DEBUG)

        logger.handlers.clear()

        if instance_id:
            instance_filter = InstanceLogFilter(instance_id)
            logger.addFilter(instance_filter)
            Logger._instance_filters[instance_id] = instance_filter

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = InstanceFormatter()
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
    def get_instance_logger(instance_id: str) -> logging.Logger:
        """获取策略实例专属日志记录器

        Args:
            instance_id: 策略实例ID

        Returns:
            带 instance_id 过滤的日志记录器
        """
        log_dir = os.path.join('logs', 'instances', instance_id)
        log_file = os.path.join(
            log_dir,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{instance_id}.log"
        )
        return Logger.get_logger(
            f'quant_framework.{instance_id}',
            log_file,
            instance_id=instance_id
        )

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

        if Logger._global_file_handler is not None:
            logging.getLogger().removeHandler(Logger._global_file_handler)
            Logger._global_file_handler.close()

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = InstanceFormatter()
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        if root_logger.level > logging.DEBUG:
            root_logger.setLevel(logging.DEBUG)

        Logger._global_file_handler = file_handler
        Logger._global_log_file = log_file

        return log_file
