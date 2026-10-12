"""Точка входа приложения приёма и отображения LSL сигнала."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Проверить зависимости P300 без GUI и сети")
    parser.add_argument("--report", type=Path, help="JSON-отчёт автоматической проверки")
    args, _qt_args = parser.parse_known_args()
    if args.self_check:
        from app.p300.diagnostics import self_check
        report = self_check()
        if args.report:
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    from app.viewer import main as viewer_main
    viewer_main()


if __name__ == "__main__":
    main()
