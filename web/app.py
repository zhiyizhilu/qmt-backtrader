import json
import os
import glob
import re
import pickle

from flask import Flask, jsonify, send_from_directory, request

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, '.cache')
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
        log_path = _find_log_for_run(strategy_name, run_id)
        os.remove(file_path)
        if log_path and os.path.exists(log_path):
            os.remove(log_path)
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


def _find_log_for_run(strategy_name, run_id):
    strategy_path = _find_strategy_dir(strategy_name)
    if not strategy_path:
        return None
    bt_dir = os.path.join(strategy_path, 'backtest_results')
    if not os.path.isdir(bt_dir):
        return None
    json_path = os.path.join(bt_dir, f'{run_id}.json')
    if not os.path.exists(json_path):
        for jf in glob.glob(os.path.join(bt_dir, '*.json')):
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                if d.get('meta', {}).get('run_id') == run_id:
                    json_path = jf
                    break
            except (json.JSONDecodeError, IOError):
                continue
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log_file = data.get('meta', {}).get('log_file', '')
        if log_file and os.path.isfile(log_file):
            return log_file
    except (json.JSONDecodeError, IOError):
        pass
    return None


@app.route('/api/logs/<strategy_name>/<run_id>')
def api_log_content(strategy_name, run_id):
    log_path = _find_log_for_run(strategy_name, run_id)
    if not log_path:
        return jsonify({'content': '', 'found': False})
    try:
        offset = request.args.get('offset', type=int)
        limit = request.args.get('limit', type=int)
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            if offset is not None or limit is not None:
                lines = f.readlines()
                total_lines = len(lines)
                start = offset if offset is not None else 0
                batch = limit if limit is not None else total_lines
                sliced = lines[start:start + batch]
                return jsonify({
                    'content': ''.join(sliced),
                    'found': True,
                    'file': os.path.basename(log_path),
                    'total_lines': total_lines,
                    'offset': start,
                    'limit': batch,
                    'returned': len(sliced),
                })
            else:
                content = f.read()
                return jsonify({'content': content, 'found': True, 'file': os.path.basename(log_path)})
    except IOError as e:
        return jsonify({'content': '', 'found': False, 'error': str(e)})


@app.route('/api/strategies/<strategy_name>/runs/batch_delete', methods=['POST'])
def api_batch_delete_runs(strategy_name):
    strategy_path = _find_strategy_dir(strategy_name)
    if not strategy_path:
        return jsonify({'error': f'Strategy "{strategy_name}" not found'}), 404
    bt_dir = os.path.join(strategy_path, 'backtest_results')
    if not os.path.isdir(bt_dir):
        return jsonify({'error': 'No backtest results found'}), 404

    data = request.get_json()
    if not data or 'run_ids' not in data:
        return jsonify({'error': 'run_ids is required'}), 400

    run_ids = data['run_ids']
    deleted = []
    errors = []

    for run_id in run_ids:
        file_path = os.path.join(bt_dir, f'{run_id}.json')
        if not os.path.exists(file_path):
            for json_file in glob.glob(os.path.join(bt_dir, '*.json')):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        jdata = json.load(f)
                    if jdata.get('meta', {}).get('run_id') == run_id:
                        file_path = json_file
                        break
                except (json.JSONDecodeError, IOError):
                    continue

        if os.path.exists(file_path):
            try:
                log_path = _find_log_for_run(strategy_name, run_id)
                os.remove(file_path)
                if log_path and os.path.exists(log_path):
                    os.remove(log_path)
                deleted.append(run_id)
            except OSError as e:
                errors.append({'run_id': run_id, 'error': str(e)})
        else:
            errors.append({'run_id': run_id, 'error': 'not found'})

    return jsonify({'deleted': deleted, 'errors': errors, 'deleted_count': len(deleted)})


@app.route('/api/cache/browse')
def api_cache_browse():
    rel_path = request.args.get('path', '')
    abs_path = os.path.normpath(os.path.join(CACHE_DIR, rel_path))
    if not abs_path.startswith(os.path.normpath(CACHE_DIR)):
        return jsonify({'error': 'Invalid path'}), 400
    if not os.path.isdir(abs_path):
        return jsonify({'error': 'Directory not found'}), 404

    entries = []
    try:
        for entry in sorted(os.listdir(abs_path)):
            full = os.path.join(abs_path, entry)
            entry_rel = os.path.relpath(full, CACHE_DIR).replace('\\', '/')
            if os.path.isdir(full):
                entries.append({'name': entry, 'type': 'dir', 'path': entry_rel})
            else:
                size = os.path.getsize(full)
                entries.append({
                    'name': entry,
                    'type': 'file',
                    'path': entry_rel,
                    'size': size,
                    'ext': os.path.splitext(entry)[1].lower(),
                })
    except OSError as e:
        return jsonify({'error': str(e)}), 500

    parent = os.path.relpath(os.path.dirname(abs_path), CACHE_DIR).replace('\\', '/') if rel_path else ''
    return jsonify({
        'path': rel_path,
        'parent': parent if parent != '.' else '',
        'entries': entries,
    })


@app.route('/api/cache/dates')
def api_cache_dates():
    rel_path = request.args.get('path', '')
    abs_path = os.path.normpath(os.path.join(CACHE_DIR, rel_path))
    if not abs_path.startswith(os.path.normpath(CACHE_DIR)):
        return jsonify({'error': 'Invalid path'}), 400
    if not os.path.isfile(abs_path):
        return jsonify({'error': 'File not found'}), 404

    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in ('.parquet', '.pkl'):
        return jsonify({'error': 'Not supported for this format'}), 400

    try:
        import pandas as pd
        if ext == '.parquet':
            df = pd.read_parquet(abs_path)
        else:
            with open(abs_path, 'rb') as f:
                data = pickle.load(f)
            if not isinstance(data, pd.DataFrame):
                return jsonify({'error': 'Not a DataFrame'}), 400
            df = data

        has_index = not isinstance(df.index, pd.RangeIndex)
        date_col = None
        if has_index and pd.api.types.is_datetime64_any_dtype(df.index):
            date_col = df.index.name or 'index'
            dates = df.index.to_series()
        else:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                    dates = df[col]
                    break

        if date_col is None:
            return jsonify({'error': 'No datetime column found'}), 400

        years = sorted(dates.dt.year.dropna().unique().astype(int).tolist())
        year_months = {}
        year_month_days = {}
        for y in years:
            mask = dates.dt.year == y
            months = sorted(dates.loc[mask].dt.month.dropna().unique().astype(int).tolist())
            year_months[str(y)] = months
            year_month_days[str(y)] = {}
            for m in months:
                mask2 = (dates.dt.year == y) & (dates.dt.month == m)
                days = sorted(dates.loc[mask2].dt.day.dropna().unique().astype(int).tolist())
                year_month_days[str(y)][str(m).zfill(2)] = days

        return jsonify({
            'path': rel_path,
            'date_column': date_col,
            'years': [str(y) for y in years],
            'year_months': {str(k): [str(m).zfill(2) for m in v] for k, v in year_months.items()},
            'year_month_days': {str(k): {str(k2).zfill(2): v for k2, v in v2.items()} for k, v2 in year_month_days.items()},
            'total_rows': len(df),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _filter_by_date(df, date_filter):
    import pandas as pd
    import re
    has_index = not isinstance(df.index, pd.RangeIndex)
    if has_index and pd.api.types.is_datetime64_any_dtype(df.index):
        dates = df.index.to_series()
    else:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dates = df[col]
                break
        else:
            return df
    m_full = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', date_filter)
    if m_full:
        year, month, day = int(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3))
        mask = (dates.dt.year == year) & (dates.dt.month == month) & (dates.dt.day == day)
        return df.loc[mask]
    m_month = re.match(r'^(\d{4})-(\d{2})$', date_filter)
    if m_month:
        year, month = int(m_month.group(1)), int(m_month.group(2))
        mask = (dates.dt.year == year) & (dates.dt.month == month)
        return df.loc[mask]
    m_year = re.match(r'^(\d{4})$', date_filter)
    if m_year:
        year = int(m_year.group(1))
        mask = dates.dt.year == year
        return df.loc[mask]
    return df


@app.route('/api/cache/view')
def api_cache_view():
    rel_path = request.args.get('path', '')
    offset = request.args.get('offset', type=int)
    limit = request.args.get('limit', type=int)
    date_filter = request.args.get('date', '')

    abs_path = os.path.normpath(os.path.join(CACHE_DIR, rel_path))
    if not abs_path.startswith(os.path.normpath(CACHE_DIR)):
        return jsonify({'error': 'Invalid path'}), 400
    if not os.path.isfile(abs_path):
        return jsonify({'error': 'File not found'}), 404

    ext = os.path.splitext(abs_path)[1].lower()
    try:
        if ext == '.json':
            with open(abs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({'format': 'json', 'data': data, 'path': rel_path})

        elif ext == '.parquet':
            try:
                import pandas as pd
                df = pd.read_parquet(abs_path)

                if date_filter:
                    df = _filter_by_date(df, date_filter)

                has_index = not isinstance(df.index, pd.RangeIndex)
                index_name = df.index.name or 'index'
                index_dtype = str(df.index.dtype) if has_index else None
                columns = list(df.columns)
                all_columns = ([index_name] + columns) if has_index else columns
                dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
                if has_index:
                    dtypes[index_name] = index_dtype

                if offset is None:
                    offset = 0
                if limit is None:
                    limit = 50

                slice_df = df.iloc[offset:offset + limit]
                slice_df = slice_df.reset_index() if has_index else slice_df.copy()

                for col in slice_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(slice_df[col]):
                        s = slice_df[col]
                        has_time = (s.dt.hour != 0).any() or (s.dt.minute != 0).any() or (s.dt.second != 0).any()
                        slice_df[col] = s.dt.strftime('%Y-%m-%d %H:%M:%S' if has_time else '%Y-%m-%d')

                return jsonify({
                    'format': 'parquet',
                    'path': rel_path,
                    'columns': columns,
                    'all_columns': all_columns,
                    'dtypes': dtypes,
                    'has_index': has_index,
                    'index_name': index_name if has_index else None,
                    'shape': list(df.shape),
                    'offset': offset,
                    'limit': limit,
                    'total': len(df),
                    'date_filter': date_filter,
                    'head': slice_df.to_dict(orient='records'),
                })
            except Exception as e:
                return jsonify({'format': 'parquet', 'error': str(e), 'path': rel_path})

        elif ext == '.pkl':
            try:
                import pandas as pd
                with open(abs_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, pd.DataFrame):
                    if date_filter:
                        data = _filter_by_date(data, date_filter)
                    has_index = not isinstance(data.index, pd.RangeIndex)
                    index_name = data.index.name or 'index'
                    index_dtype = str(data.index.dtype) if has_index else None
                    columns = list(data.columns)
                    all_columns = ([index_name] + columns) if has_index else columns
                    dtypes = {c: str(dt) for c, dt in data.dtypes.items()}
                    if has_index:
                        dtypes[index_name] = index_dtype
                    if offset is None:
                        offset = 0
                    if limit is None:
                        limit = 50
                    slice_df = data.iloc[offset:offset + limit]
                    slice_df = slice_df.reset_index() if has_index else slice_df.copy()
                    for col in slice_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(slice_df[col]):
                            s = slice_df[col]
                            has_time = (s.dt.hour != 0).any() or (s.dt.minute != 0).any() or (s.dt.second != 0).any()
                            slice_df[col] = s.dt.strftime('%Y-%m-%d %H:%M:%S' if has_time else '%Y-%m-%d')
                    return jsonify({
                        'format': 'pkl_dataframe',
                        'path': rel_path,
                        'columns': columns,
                        'all_columns': all_columns,
                        'dtypes': dtypes,
                        'has_index': has_index,
                        'index_name': index_name if has_index else None,
                        'shape': list(data.shape),
                        'offset': offset,
                        'limit': limit,
                        'total': len(data),
                        'date_filter': date_filter,
                        'head': slice_df.to_dict(orient='records'),
                    })
                else:
                    return jsonify({
                        'format': 'pkl_other',
                        'path': rel_path,
                        'type': str(type(data)),
                        'repr': repr(data)[:5000],
                    })
            except Exception as e:
                return jsonify({'format': 'pkl', 'error': str(e), 'path': rel_path})

        else:
            size = os.path.getsize(abs_path)
            if size < 50000:
                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    return jsonify({'format': 'text', 'content': content, 'path': rel_path})
                except Exception:
                    pass
            return jsonify({'format': ext or 'unknown', 'path': rel_path, 'size': size,
                            'message': '此格式暂不支持在线预览'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Strategy dirs: {STRATEGY_DIRS}')
    strategies = _discover_strategies()
    print(f'Discovered {len(strategies)} strategies with backtest results:')
    for name, info in strategies.items():
        print(f'  - {name} ({info["source"]}): {info["run_count"]} runs')
    print('\nStarting web server at http://localhost:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)
