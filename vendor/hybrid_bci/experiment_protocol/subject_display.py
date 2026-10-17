"""Тексты для экрана испытуемого (без Qt / без логики протокола)."""

from __future__ import annotations


def format_subject_experiment_title(
    *,
    index: int,
    total: int,
    calib: bool = False,
) -> str:
    """«Эксперимент 3 из 15» или «Калибровка 2 из 5»."""
    idx = max(1, int(index))
    tot = max(int(total), idx)
    head = "Калибровка" if calib else "Эксперимент"
    return f"{head} {idx} из {tot}"
