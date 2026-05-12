import json
import os
import glob
import re

from flask import Flask, jsonify, send_from_directory, request

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_DIRS = [
    os.path.join(PROJECT_ROOT, 'strategies'),
    os.path.join(PROJECT_ROOT, 'strategies_for_vip'),
]

# 策略显示名称映射文件路径
_DISPLAY_NAMES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategy_display_names.json')


def _load_display_names():
    """加载策略显示名称映射"""
    if os.path.exists(_DISPLAY_NAMES_FILE):
        try:
            with open(_DISPLAY_NAMES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_display_names(names):
    """保存策略显示名称映射"""
    with open(_DISPLAY_NAMES_FILE, 'w', encoding='utf-8') as f:
        json.dump(names, f, indent=2, ensure_ascii=False)


def _extract_readme_title(strategy_path):
    """从 readme.md 第一行提取策略中文标题"""
    for fname in ['readme.md', 'README.md', 'Readme.md']:
        readme_path = os.path.join(strategy_path, fname)
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    # 提取 # 后面的标题文字，去掉括号内的英文部分
                    match = re.match(r'^#+\s+(.+)', first_line)
                    if match:
                        title = match.group(1).strip()
                        # 去掉括号内的英文部分，如 "(Bank Rotation Strategy)"
                        title = re.sub(r'\s*[\(（][^\)）]+[\)）]\s*', '', title).strip()
                        # 去掉破折号后面的副标题，如 " — 降低回撤、提升收益"
                        title = re.sub(r'\s*[—\-–]\s+.*$', '', title).strip()
                        return title
            except (IOError, UnicodeDecodeError):
                pass
    return None


def _discover_strategies():
    display_names = _load_display_names()
    strategies = {}
    for base_dir in STRATEGY_DIRS:
        if not os.path.isdir(base_dir):
            continue
        source = os.path.basename(base_dir)
        for entry in os.listdir(base_dir):
            strategy_path = os.path.join(base_dir, entry)
            bt_dir = os.path.join(strategy_path, 'backtest_results')
            if os.path.isdir(strategy_path) and os.path.isdir(bt_dir):
                json_files = glob.glob(os.path.join(bt_dir, '*.json'))
                if json_files:
                    # 优先使用用户自定义名称，其次从readme提取
                    custom_name = display_names.get(entry)
                    readme_title = _extract_readme_title(strategy_path)
                    strategies.setdefault(entry, {
                        'name': entry,
                        'source': source,
                        'path': strategy_path,
                        'backtest_dir': bt_dir,
                        'run_count': 0,
                        'runs': [],
                        'display_name': custom_name or readme_title or '',
                    })
                    runs = _load_run_summaries(bt_dir)
                    strategies[entry]['run_count'] = len(runs)
                    strategies[entry]['runs'] = runs
    return strategies


def _load_run_summaries(bt_dir):
    runs = []
    for json_file in glob.glob(os.path.join(bt_dir, '*.json')):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            meta = data.get('meta', {})
            metrics = data.get('metrics', {})
            config = data.get('config', {})
            runs.append({
                'run_id': meta.get('run_id', ''),
                'strategy_name': meta.get('strategy_name', ''),
                'timestamp': meta.get('timestamp', ''),
                'file': os.path.basename(json_file),
                'total_return_pct': metrics.get('total_return_pct'),
                'annual_return_pct': metrics.get('annual_return_pct'),
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'max_drawdown_pct': metrics.get('max_drawdown_pct'),
                'total_trading_days': metrics.get('total_trading_days'),
                'initial_capital': metrics.get('initial_capital'),
                'final_value': metrics.get('final_value'),
                'start_date': config.get('start_date', ''),
                'end_date': config.get('end_date', ''),
            })
        except (json.JSONDecodeError, IOError):
            continue
    runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return runs


def _find_strategy_dir(strategy_name):
    for base_dir in STRATEGY_DIRS:
        if not os.path.isdir(base_dir):
            continue
        strategy_path = os.path.join(base_dir, strategy_name)
        if os.path.isdir(strategy_path):
            return strategy_path
    return None


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/api/strategies')
def api_strategies():
    strategies = _discover_strategies()
    result = []
    for name, info in sorted(strategies.items()):
        result.append({
            'name': info['name'],
            'source': info['source'],
            'run_count': info['run_count'],
            'latest_run': info['runs'][0] if info['runs'] else None,
            'display_name': info.get('display_name', ''),
        })
    return jsonify(result)


@app.route('/api/strategies/<strategy_name>/runs')
def api_strategy_runs(strategy_name):
    strategy_path = _find_strategy_dir(strategy_name)
    if not strategy_path:
        return jsonify({'error': f'Strategy "{strategy_name}" not found'}), 404
    bt_dir = os.path.join(strategy_path, 'backtest_results')
    if not os.path.isdir(bt_dir):
        return jsonify({'error': 'No backtest results found'}), 404
    runs = _load_run_summaries(bt_dir)
    return jsonify(runs)


@app.route('/api/strategies/<strategy_name>/runs/<run_id>')
def api_strategy_run_detail(strategy_name, run_id):
    strategy_path = _find_strategy_dir(strategy_name)
    if not strategy_path:
        return jsonify({'error': f'Strategy "{strategy_name}" not found'}), 404
    bt_dir = os.path.join(strategy_path, 'backtest_results')
    file_path = os.path.join(bt_dir, f'{run_id}.json')
    if not os.path.exists(file_path):
        for json_file in glob.glob(os.path.join(bt_dir, '*.json')):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('meta', {}).get('run_id') == run_id:
                    return jsonify(data)
            except (json.JSONDecodeError, IOError):
                continue
        return jsonify({'error': f'Run "{run_id}" not found'}), 404
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except (json.JSONDecodeError, IOError) as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategies/<strategy_name>/runs/<run_id>', methods=['DELETE'])
def api_delete_run(strategy_name, run_id):
    strategy_path = _find_strategy_dir(strategy_name)
    if not strategy_path:
        return jsonify({'error': f'Strategy "{strategy_name}" not found'}), 404
    bt_dir = os.path.join(strategy_path, 'backtest_results')
    if not os.path.isdir(bt_dir):
        return jsonify({'error': 'No backtest results found'}), 404
    file_path = os.path.join(bt_dir, f'{run_id}.json')
    if not os.path.exists(file_path):
        for json_file in glob.glob(os.path.join(bt_dir, '*.json')):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('meta', {}).get('run_id') == run_id:
                    file_path = json_file
                    break
            except (json.JSONDecodeError, IOError):
                continue
    if not os.path.exists(file_path):
        return jsonify({'error': f'Run "{run_id}" not found'}), 404
    try:
        os.remove(file_path)
        return jsonify({'success': True, 'run_id': run_id})
    except OSError as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategies/<strategy_name>/readme')
def api_strategy_readme(strategy_name):
    strategy_path = _find_strategy_dir(strategy_name)
    if not strategy_path:
        return jsonify({'error': f'Strategy "{strategy_name}" not found'}), 404
    for fname in ['readme.md', 'README.md', 'Readme.md']:
        readme_path = os.path.join(strategy_path, fname)
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return jsonify({'content': content})
            except (IOError, UnicodeDecodeError) as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'content': ''})


@app.route('/api/strategies/<strategy_name>/display_name', methods=['PUT'])
def api_update_display_name(strategy_name):
    """更新策略的显示名称"""
    data = request.get_json()
    if not data or 'display_name' not in data:
        return jsonify({'error': 'display_name is required'}), 400
    display_names = _load_display_names()
    display_names[strategy_name] = data['display_name']
    _save_display_names(display_names)
    return jsonify({'success': True, 'display_name': data['display_name']})


@app.route('/api/strategies/<strategy_name>/compare')
def api_strategy_compare(strategy_name):
    strategy_path = _find_strategy_dir(strategy_name)
    if not strategy_path:
        return jsonify({'error': f'Strategy "{strategy_name}" not found'}), 404
    bt_dir = os.path.join(strategy_path, 'backtest_results')
    if not os.path.isdir(bt_dir):
        return jsonify({'error': 'No backtest results found'}), 404
    runs = _load_run_summaries(bt_dir)
    comparison = {
        'strategy_name': strategy_name,
        'runs': runs,
        'best_sharpe': max(runs, key=lambda x: x.get('sharpe_ratio') or -999) if runs else None,
        'best_return': max(runs, key=lambda x: x.get('total_return_pct') or -999) if runs else None,
        'lowest_drawdown': max(runs, key=lambda x: x.get('max_drawdown_pct') or -999) if runs else None,
    }
    return jsonify(comparison)


if __name__ == '__main__':
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Strategy dirs: {STRATEGY_DIRS}')
    strategies = _discover_strategies()
    print(f'Discovered {len(strategies)} strategies with backtest results:')
    for name, info in strategies.items():
        print(f'  - {name} ({info["source"]}): {info["run_count"]} runs')
    print('\nStarting web server at http://localhost:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)
