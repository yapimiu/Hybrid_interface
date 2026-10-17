"""Проверить импорт научного ядра и liblsl в собранном приложении без GUI/сети."""

import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, help='Check an alternate build without replacing a running executable')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    executable = (root / "dist" / "Hybrid Interface.app" / "Contents" / "MacOS" / "HybridInterface"
                  if sys.platform == "darwin" else root / "dist" / "HybridInterface.exe")
    if args.executable:
        executable = args.executable.resolve()
    with tempfile.TemporaryDirectory(prefix="hybrid-check-") as folder:
        report = Path(folder) / "check.json"
        env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": folder}
        subprocess.run([str(executable), "--self-check", "--report", str(report)],
                       check=True, timeout=60, env=env)
        result = json.loads(report.read_text(encoding="utf-8"))
        if result.get("ok") is not True:
            raise RuntimeError("Bundle self-check failed")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
