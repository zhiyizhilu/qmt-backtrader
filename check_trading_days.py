"""兼容层：将 check_trading_days.py 的调用转发到 download_market_data.py --check

原有功能已合并到 download_market_data.py 中，此文件仅做参数转换和转发。
"""
import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]

    forward_args = ['--check']

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == '--fix':
            forward_args.append('--fix')
        elif arg == '--dry-run':
            forward_args.append('--dry-run')
        elif arg == '--verbose' or arg == '-v':
            forward_args.append('--verbose')
        elif arg == '--report':
            forward_args.append('--report')
            i += 1
            if i < len(args):
                forward_args.append(args[i])
        elif arg == '--start':
            i += 1
            if i < len(args):
                val = args[i]
                if len(val) == 4 and val.isdigit():
                    forward_args.extend(['--start', f'{val}-01-01'])
                else:
                    forward_args.extend(['--start', val])
        elif arg == '--end':
            i += 1
            if i < len(args):
                val = args[i]
                if len(val) == 4 and val.isdigit():
                    forward_args.extend(['--end', f'{val}-12-31'])
                else:
                    forward_args.extend(['--end', val])
        else:
            forward_args.append(arg)

        i += 1

    script = os.path.join(PROJECT_ROOT, 'download_market_data.py')
    cmd = [sys.executable, script] + forward_args

    print(f"[兼容层] 转发到: python download_market_data.py {' '.join(forward_args)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
