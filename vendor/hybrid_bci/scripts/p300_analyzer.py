#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Точка входа онлайн P300-анализатора (Qt). Логика — в пакете ``p300_analysis``."""

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt5.QtWidgets import QApplication

from p300_analysis.logging_config import configure_logging
from p300_analysis.qt_window import P300AnalyzerWindow

LOG = logging.getLogger("p300_analyzer")


def main() -> None:
    log_path = configure_logging()
    LOG.info("Старт P300 Analyzer, лог: %s", log_path)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = P300AnalyzerWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
