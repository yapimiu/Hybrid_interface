"""Tests for manual run export helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from p300_analysis.run_export import export_run_continuous_csv, export_run_data_all_stims, stim_indices_in_run


def test_stim_indices_in_run_sorted_unique() -> None:
    run = {
        "epoch_segments": [
            {"stim_key": "стимул_8"},
            {"stim_key": "стимул_0"},
            {"stim_key": "стимул_8"},
            {"stim_key": "стимул_3"},
        ]
    }
    assert stim_indices_in_run(run) == [0, 3, 8]


def test_stim_indices_in_run_empty() -> None:
    assert stim_indices_in_run({}) == []
    assert stim_indices_in_run({"epoch_segments": []}) == []
    assert stim_indices_in_run({"epoch_segments": [{"stim_key": "bad"}]}) == []


def test_export_run_data_all_stims_one_csv() -> None:
    run = {
        "summary": {"analysis_params": {"sampling_rate_hz": 500.0}},
        "eeg_samples": [[1.0, 2.0], [3.0, 4.0]],
        "epoch_segments": [
            {
                "stim_key": "стимул_0",
                "marker_ts": 10.0,
                "eeg_ts": [10.0, 10.002],
                "eeg_samples": [[1.1, 2.1], [1.2, 2.2]],
            },
            {
                "stim_key": "стимул_1",
                "marker_ts": 20.0,
                "eeg_ts": [20.0],
                "eeg_samples": [[5.0, 6.0]],
            },
        ],
    }
    with TemporaryDirectory() as td:
        out = Path(td) / "all.csv"
        paths = export_run_data_all_stims(run_data=run, output_path=out, file_format="csv")
        assert len(paths) == 1
        assert paths[0] == out
        with open(paths[0], encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0][:5] == ["segment_id", "stim_key", "marker_ts", "sample_idx", "ts"]
        assert len(rows) == 1 + 2 + 1  # header + 2 samples seg0 + 1 sample seg1
        assert float(rows[1][0]) == 0.0 and rows[1][1] == "стимул_0"
        assert float(rows[3][0]) == 1.0 and rows[3][1] == "стимул_1"


def test_export_run_data_all_stims_sorted_by_marker_ts() -> None:
    """Позже в списке, но раньше по времени маркера — выходит первым в файле."""
    run = {
        "eeg_samples": [[1.0, 2.0]],
        "epoch_segments": [
            {
                "stim_key": "стимул_9",
                "marker_ts": 200.0,
                "eeg_ts": [200.0],
                "eeg_samples": [[9.0, 9.1]],
            },
            {
                "stim_key": "стимул_1",
                "marker_ts": 100.0,
                "eeg_ts": [100.0, 100.002],
                "eeg_samples": [[1.0, 2.0], [1.1, 2.1]],
            },
        ],
    }
    with TemporaryDirectory() as td:
        paths = export_run_data_all_stims(
            run_data=run, output_path=Path(td) / "o.csv", file_format="csv"
        )
        with open(paths[0], encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert float(rows[1][0]) == 0.0 and rows[1][1] == "стимул_1"
        assert float(rows[3][0]) == 1.0 and rows[3][1] == "стимул_9"


def test_export_run_data_all_stims_empty_raises() -> None:
    with TemporaryDirectory() as td:
        with pytest.raises(RuntimeError, match="Нет сегментов"):
            export_run_data_all_stims(
                run_data={"epoch_segments": []},
                output_path=Path(td) / "x.csv",
                file_format="csv",
            )


def _read_ru_csv(path: Path) -> list[list[str]]:
    """Читает CSV в русском формате Excel: первая строка ``sep=;``, разделитель ``;``."""
    with open(path, encoding="utf-8", newline="") as f:
        first = f.readline().strip()
        assert first.startswith("sep=")
        reader = csv.reader(f, delimiter=";")
        return [row for row in reader]


def _num_ru(s: str) -> float:
    """Парсит число из русского формата Excel (запятая → точка)."""
    return float(s.replace(",", "."))


def test_export_run_continuous_csv_marker_and_in_epoch() -> None:
    run = {
        "summary": {"analysis_params": {"sampling_rate_hz": 500.0}},
        "eeg_ts": [100.0 + 0.002 * i for i in range(10)],
        "eeg_samples": [[float(i), float(i) * 2.0] for i in range(10)],
        "markers": [{"ts": 100.004, "value": "8|on"}],
        "epoch_segments": [],
        # Окно эпохи 10 мс → при 500 Гц это 5 сэмплов (включая сэмпл вспышки).
        "epoch_time_ms": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    }
    with TemporaryDirectory() as td:
        base = Path(td) / "run1.csv"
        p = export_run_continuous_csv(run_data=run, output_path=base)
        assert p.name == "run1_continuous.csv"
        # Если имя уже содержит _continuous — не добавляем второй раз.
        base2 = Path(td) / "runX_continuous.csv"
        p2 = export_run_continuous_csv(run_data=run, output_path=base2)
        assert p2.name == "runX_continuous.csv"
        rows = _read_ru_csv(p)
        assert rows[0][:5] == ["sample_idx", "t_rel_s", "ts", "ch_1", "ch_2"]
        assert rows[0][-3:] == ["marker", "in_epoch", "target_tile_id"]
        assert "stim_target" in rows[0]
        assert "window_x_ms" in rows[0]
        # строка 3 (sample_idx=2) — ближайшая к маркеру 100.004
        assert _num_ru(rows[3][0]) == 2.0
        assert _num_ru(rows[3][-3]) == 108.0   # marker = 100 + tile_id (8)
        assert _num_ru(rows[3][-2]) == 1.0   # in_epoch = 1
        assert _num_ru(rows[3][-1]) == -1.0  # target_tile_id = -1 (нет trial_start)
        # пауза до вспышки
        assert _num_ru(rows[1][-3]) == 0.0
        assert _num_ru(rows[1][-2]) == 0.0
        # t_rel_s от начала
        assert abs(_num_ru(rows[1][1]) - 0.0) < 1e-9
        assert abs(_num_ru(rows[3][1]) - 0.004) < 1e-9
        # В строках с числами должна стоять запятая (русский формат Excel),
        # кроме sample_idx/marker/in_epoch которые целые.
        assert "," in rows[3][1] or rows[3][1] in {"0", "0,0"}


def test_export_run_continuous_csv_marker_via_lsl_clock_mapping() -> None:
    """ЭЭГ-штамп идёт с шагом 1 с (эмулируем NeuroSpectrum), а маркер — в осях
    lsl_local_clock() со смещением marker_eeg_offset. Проверяем, что строка
    маркера определяется по частоте дискретизации и lsl_clock_at_buffer_end,
    а не по ``argmin`` сырого ts.
    """
    n = 1000
    srate = 500.0
    # Грубый ts: фактически все отсчёты попадают в 2 секунды.
    eeg_ts = [100.0 if i < 500 else 101.0 for i in range(n)]
    eeg_samples = [[0.0] for _ in range(n)]
    # Последний отсчёт соответствует lsl_clock_at_buffer_end = 12345.0
    last_lc = 12345.0
    # Маркер в сырой оси — 999.5; offset = last_lc - last_marker_lc не важен,
    # важно, чтобы (marker_ts + offset) = последняя секунда хвоста.
    # Пусть marker попал за 0.4 с до конца буфера → sample_idx = n-1 - round(0.4*500) = 999 - 200 = 799
    marker_ts_raw = 999.5
    offset = last_lc - (marker_ts_raw + 0.4)  # → marker_ts_raw+offset = last_lc - 0.4
    run = {
        "summary": {
            "analysis_params": {"sampling_rate_hz": srate},
            "marker_eeg_offset": offset,
            "lsl_clock_at_buffer_end": last_lc,
        },
        "eeg_ts": eeg_ts,
        "eeg_samples": eeg_samples,
        "markers": [{"ts": marker_ts_raw, "value": "3|on"}],
        "epoch_segments": [],
        "epoch_time_ms": [0.0, 10.0],
    }
    with TemporaryDirectory() as td:
        base = Path(td) / "runLC.csv"
        p = export_run_continuous_csv(run_data=run, output_path=base)
        rows = _read_ru_csv(p)
    expected_idx = n - 1 - int(round(0.4 * srate))
    body = rows[1:]
    # Без |off плитка считается горящей до конца записи — marker=103 от expected_idx до N-1.
    markers_nonzero = [i for i, r in enumerate(body) if _num_ru(r[-3]) != 0.0]
    assert markers_nonzero and markers_nonzero[0] == expected_idx
    assert markers_nonzero[-1] == n - 1
    assert all(_num_ru(body[k][-3]) == 103.0 for k in markers_nonzero)
    assert _num_ru(body[expected_idx][-2]) == 1.0


def test_export_run_continuous_uses_precomputed_sample_idx() -> None:
    """Если в ``markers[i]["sample_idx"]`` уже записан индекс отсчёта, используем его
    и игнорируем ``marker_ts`` (он может быть в любой чужой временной шкале).
    """
    n = 50
    srate = 250.0
    eeg_ts = [100.0 for _ in range(n)]  # нарочно «сломанный» грубый ts
    eeg_samples = [[0.0] for _ in range(n)]
    run = {
        "summary": {"analysis_params": {"sampling_rate_hz": srate}},
        "eeg_ts": eeg_ts,
        "eeg_samples": eeg_samples,
        "markers": [{"ts": 999999.0, "value": "5|on", "sample_idx": 17}],
        "epoch_segments": [],
        "epoch_time_ms": [0.0, 4.0],
    }
    with TemporaryDirectory() as td:
        base = Path(td) / "pre.csv"
        p = export_run_continuous_csv(run_data=run, output_path=base)
        rows = _read_ru_csv(p)
    body = rows[1:]
    # Без |off плитка горит до конца записи — marker=105 с sample_idx=17 до N-1.
    markers_nonzero_idx = [i for i, r in enumerate(body) if _num_ru(r[-3]) != 0.0]
    assert markers_nonzero_idx[0] == 17
    assert all(_num_ru(body[k][-3]) == 105.0 for k in markers_nonzero_idx)


def test_export_run_continuous_marker_fills_on_to_off_range() -> None:
    """Между |on и |off для одной и той же плитки marker = её номер на каждом отсчёте."""
    n = 20
    srate = 500.0
    run = {
        "summary": {"analysis_params": {"sampling_rate_hz": srate}},
        "eeg_ts": [100.0 + i * 0.002 for i in range(n)],
        "eeg_samples": [[0.0] for _ in range(n)],
        "markers": [
            {"ts": 100.004, "value": "7|on", "sample_idx": 2},
            {"ts": 100.010, "value": "7|off", "sample_idx": 5},
            {"ts": 100.020, "value": "3|on", "sample_idx": 10},
            {"ts": 100.030, "value": "3|off", "sample_idx": 15},
        ],
        "epoch_segments": [],
        "epoch_time_ms": [0.0, 2.0],
    }
    with TemporaryDirectory() as td:
        base = Path(td) / "range.csv"
        p = export_run_continuous_csv(run_data=run, output_path=base)
        rows = _read_ru_csv(p)
    body = rows[1:]
    # sample_idx 2..5 → marker=107, 6..9 → 0, 10..15 → marker=103, прочее — 0.
    for k in range(2, 6):
        assert _num_ru(body[k][-3]) == 107.0, f"expected marker=107 at {k}, got {body[k][-3]}"
    for k in range(6, 10):
        assert _num_ru(body[k][-3]) == 0.0, f"expected 0 at {k}, got {body[k][-3]}"
    for k in range(10, 16):
        assert _num_ru(body[k][-3]) == 103.0, f"expected marker=103 at {k}, got {body[k][-3]}"
    assert _num_ru(body[0][-3]) == 0.0
    assert _num_ru(body[19][-3]) == 0.0


def test_export_run_continuous_target_tile_id() -> None:
    """target_tile_id должен быть равен N из trial_start|target=N, -1 до начала и после trial_end."""
    n = 20
    srate = 500.0
    run = {
        "summary": {"analysis_params": {"sampling_rate_hz": srate}},
        "eeg_ts": [100.0 + i * 0.002 for i in range(n)],
        "eeg_samples": [[0.0] for _ in range(n)],
        "markers": [
            {"ts": 100.004, "value": "-1|trial_start|target=5", "sample_idx": 2},
            {"ts": 100.010, "value": "3|on", "sample_idx": 5},
            {"ts": 100.030, "value": "-2|trial_end", "sample_idx": 15},
        ],
        "epoch_segments": [],
        "epoch_time_ms": [0.0, 2.0],
    }
    with TemporaryDirectory() as td:
        p = export_run_continuous_csv(run_data=run, output_path=Path(td) / "tgt.csv")
        rows = _read_ru_csv(p)
    body = rows[1:]
    # До trial_start (sample 0,1) — target_tile_id = -1
    assert _num_ru(body[0][-1]) == -1.0
    assert _num_ru(body[1][-1]) == -1.0
    # После trial_start (sample 2..14) — target_tile_id = 5
    for k in range(2, 15):
        assert _num_ru(body[k][-1]) == 5.0, f"expected 5 at sample {k}, got {body[k][-1]}"
    # После trial_end (sample 15..19) — target_tile_id = -1
    for k in range(15, n):
        assert _num_ru(body[k][-1]) == -1.0, f"expected -1 at sample {k}, got {body[k][-1]}"


def test_export_run_continuous_xlsx_format() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    n = 6
    srate = 500.0
    run = {
        "summary": {"analysis_params": {"sampling_rate_hz": srate}},
        "eeg_ts": [100.0 + i * 0.002 for i in range(n)],
        "eeg_samples": [[float(i), -float(i)] for i in range(n)],
        "markers": [{"ts": 100.004, "value": "2|on"}],
        "epoch_segments": [],
        "epoch_time_ms": [0.0, 2.0],
    }
    with TemporaryDirectory() as td:
        base = Path(td) / "xl.csv"  # расширение не важно — функция сама поставит
        p = export_run_continuous_csv(run_data=run, output_path=base, file_format="xlsx")
        assert p.suffix == ".xlsx"
        wb = openpyxl.load_workbook(p)
        ws = wb.active
        header = [c.value for c in ws[1]]
        assert header[:5] == ["sample_idx", "t_rel_s", "ts", "ch_1", "ch_2"]
        assert header[-3:] == ["marker", "in_epoch", "target_tile_id"]
        assert "stim_target" in header
        assert "window_x_ms" in header
        # строка 4 (sample_idx=2) — вспышка
        row2 = [c.value for c in ws[4]]
        assert row2[0] == 2
        assert row2[-3] == 102    # marker (100 + tile_id)
        assert row2[-2] == 1    # in_epoch
        assert row2[-1] == -1   # target_tile_id (нет trial_start)
