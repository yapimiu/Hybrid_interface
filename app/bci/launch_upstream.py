"""Separate interpreter entry point, without modifying the original program."""
import os
from pathlib import Path
import runpy
import sys


if __name__ == '__main__':
    if sys.platform != 'win32':
        os.setsid()
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(Path.cwd()))
    runpy.run_path(str(script), run_name='__main__')
