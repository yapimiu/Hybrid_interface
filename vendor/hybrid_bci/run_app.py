import argparse
import os
import subprocess
import sys
from pathlib import Path


def _maybe_reexec_in_venv(root: Path) -> None:
    venv_dir = root / ".venv"
    vpy = venv_dir / "bin" / "python"
    if not vpy.exists():
        vpy = venv_dir / "Scripts" / "python.exe"
    if not vpy.exists():
        return

    cur = Path(sys.executable).resolve()
    target = vpy.resolve()
    if cur == target:
        return

    # Re-run under venv Python so imports/dependencies match.
    os.execv(str(target), [str(target), *sys.argv])


def main() -> None:
    root = Path(__file__).resolve().parent
    _maybe_reexec_in_venv(root)
    os.chdir(root)
    sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description="Запуск стимулятора плиток (PsychoPy) и опций.")
    parser.add_argument(
        "--auto-random-protocol",
        action="store_true",
        help="Без экранных кнопок: сразу trial, новая случайная цель после каждого trial_end.",
    )
    parser.add_argument(
        "--no-analyzer",
        action="store_true",
        help="Не поднимать отдельно scripts/p300_analyzer.py (для protocol_runner и т.п.).",
    )
    parser.add_argument(
        "--inter-trial-s",
        type=float,
        default=1.0,
        help="Пауза между trial в режиме --auto-random-protocol, сек.",
    )
    parser.add_argument(
        "--auto-calib-trials",
        type=int,
        default=0,
        help="DEPRECATED: используйте --auto-plan-trials/--auto-plan-target-*. Оставлено для совместимости.",
    )
    parser.add_argument(
        "--auto-calib-target-tile-id",
        type=int,
        default=4,
        help="DEPRECATED: используйте --auto-plan-trials/--auto-plan-target-*. Оставлено для совместимости.",
    )
    parser.add_argument(
        "--auto-plan-trials",
        type=int,
        default=3,
        help="Длина плана целей (первые trial) в авто-режиме; для протокола — калибровка (обычно 3).",
    )
    parser.add_argument(
        "--auto-plan-target-tile-id",
        type=int,
        default=4,
        help="Какая плитка (0..8) должна встретиться нужное число раз в плане.",
    )
    parser.add_argument(
        "--auto-plan-target-repeats",
        type=int,
        default=0,
        help="Сколько раз выбранная плитка должна появиться в первых --auto-plan-trials trial (не подряд). "
        "0 = посчитать автоматически по --auto-plan-target-epochs и sequences.",
    )
    parser.add_argument(
        "--auto-plan-target-epochs",
        type=int,
        default=12,
        help="Сколько target-эпох нужно набрать для шаблона (используется при --auto-plan-target-repeats=0).",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        default=None,
        help="Переопределить число sequences (раундов) в P300 trial (полезно для автопротокола).",
    )
    parser.add_argument(
        "--auto-max-trials",
        type=int,
        default=None,
        help="Ограничить число auto-trial (после достижения — стимулятор завершится).",
    )
    parser.add_argument(
        "--stim-control-dir",
        type=str,
        default="",
        help="Папка сессии с stim_control.json: P300 trial по команде протокола (после калибровки).",
    )
    args = parser.parse_args()

    analyzer_proc: subprocess.Popen[str] | None = None
    analyzer_script = root / "scripts" / "p300_analyzer.py"
    if not args.no_analyzer and analyzer_script.exists():
        analyzer_proc = subprocess.Popen([sys.executable, str(analyzer_script)])
    elif not args.no_analyzer:
        print(f"[run_app] Warning: analyzer script not found: {analyzer_script}")

    from app.main import main as app_main

    app_main(
        auto_random_trials=bool(args.auto_random_protocol),
        inter_trial_s=float(args.inter_trial_s),
        auto_plan_trials=int(args.auto_plan_trials) if int(args.auto_calib_trials) <= 0 else int(args.auto_calib_trials),
        auto_plan_target_tile_id=int(args.auto_plan_target_tile_id)
        if int(args.auto_calib_trials) <= 0
        else int(args.auto_calib_target_tile_id),
        auto_plan_target_repeats=int(args.auto_plan_target_repeats),
        auto_plan_target_epochs=int(args.auto_plan_target_epochs),
        sequences_override=int(args.sequences) if args.sequences is not None else None,
        auto_max_trials=int(args.auto_max_trials) if args.auto_max_trials is not None else None,
        stim_control_dir=str(args.stim_control_dir).strip() or None,
    )
    if analyzer_proc is not None and analyzer_proc.poll() is None:
        print(
            "[run_app] Stimulator closed. P300 analyzer keeps running in a separate window/process."
        )


if __name__ == "__main__":
    main()

