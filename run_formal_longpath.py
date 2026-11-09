"""用 Windows 扩展长度路径前缀运行 run_design_family_formal.py（见 design_family_formal_20260929/DEVIATIONS.md 的 D1）。

参数原样转发，例如：python run_formal_longpath.py --run --limit 5
在 Python 里设置环境变量，避免 MSYS/bash 改写 \\\\?\\ 前缀。非 Windows 平台不加前缀。
"""
import os
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).resolve().parent
target = PROJECT / 'unitree_heading_results' / 'design_family_formal_20260929'
os.environ['DF_FORMAL_OUT'] = ('\\\\?\\' + str(target)) if os.name == 'nt' else str(target)
sys.argv = [str(PROJECT / 'run_design_family_formal.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
