import os
import copy

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

_DEFAULT_CONFIG = {
    'strategy': {
        'default_mode': 'backtest',
        'default_period': '1d',
        'default_pool': None,
    },
    'backtest': {
        'initial_capital': 1000000,
        'commission': 0.0001,
        'slippage': 0.001,
        'benchmark': '000300.SH',
        'data_source': 'qmt',
    },
    'cache': {
        'dir': '.cache',
        'mem_limit': 500,
        'max_age_days': 30,
    },
    'log': {
        'level': 'INFO',
        'dir': 'logs',
        'format': 'text',
    },
    'instance': {
        'heartbeat_interval': 60,
        'max_restart_attempts': 3,
        'restart_delay': 10,
    },
}


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path=None):
    if path and HAS_YAML:
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            return _deep_merge(_DEFAULT_CONFIG, yaml_config)
    elif path and not HAS_YAML:
        import logging
        logging.getLogger(__name__).warning(
            'pyyaml 未安装，无法加载 YAML 配置文件，使用默认配置'
        )
    return copy.deepcopy(_DEFAULT_CONFIG)


def get_default_config():
    return copy.deepcopy(_DEFAULT_CONFIG)
