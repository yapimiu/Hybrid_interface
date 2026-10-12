"""Независимое от Qt и LSL вычислительное ядро P300.

Модули Qt/LSL импортируются только интеграционным слоем, не этим пакетом.
"""

from .config import P300Config
from .markers import Stimulus

__all__ = ["P300Config", "Stimulus"]
