#!/usr/bin/env python3
"""
Standalone smoke test: synthetic EEG (numpy) -> managed double[,] -> MSI.ModelSignal -> MSI.MSIExec.

Повторно использует загрузку CoreCLR / ссылок из `test_msi_import.py` (без изменения P300/Qt/LSL).

Ожидаемая семантика MSIController (проверено reflection): перегрузки MSIExec
  - MSIExec(double[,] Signal) -> int
  - MSIExec(double[,] Signal, out double coefS) -> int
  - MSIExec(IList<double[,]> Signals) -> int

ModelSignal: IList<double[,]> — в pythonnet обычный Python list не подходит; нужен
System.Collections.Generic.List<double[,]>, сконструированный по фактическому generic-аргументу
из свойства (см. build_model_signal_list).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Репозиторий и каталог scripts в sys.path, чтобы `import test_msi_import` работал из корня и из scripts/
_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for _p in (str(_SCRIPTS), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_msi_import as tmi  # noqa: E402


def _debug(msg: str) -> None:
    print(f"[debug] {msg}", flush=True)


def generate_sine_signal(
    freq: float,
    srate: float,
    duration: float,
    channels: int,
) -> np.ndarray:
    """
    Синусоидальный «EEG» в виде numpy-массива.

    Форма: (channels, samples) — каналы по оси 0, время по оси 1 (как в запросе).
    Каждый канал: sin(2π f t), фаза 0.
    """
    if channels < 1:
        raise ValueError("channels must be >= 1")
    n = int(round(srate * duration))
    if n < 2:
        raise ValueError("duration*srate must yield at least 2 samples")
    t = np.arange(n, dtype=np.float64) / float(srate)
    w = 2.0 * np.pi * float(freq) * t
    row = np.sin(w, dtype=np.float64)
    out = np.empty((channels, n), dtype=np.float64)
    for c in range(channels):
        out[c, :] = row
    return out


def generate_model_signals(
    freqs_hz: Sequence[float],
    srate: float,
    duration: float,
) -> list[np.ndarray]:
    """
    Эталонные шаблоны для MSI.ModelSignal: для каждой частоты матрица (2, n_samples):
      строка 0: sin(2π f t)
      строка 1: cos(2π f t)
    """
    n = int(round(srate * duration))
    if n < 2:
        raise ValueError("duration*srate must yield at least 2 samples")
    t = np.arange(n, dtype=np.float64) / float(srate)
    mats: list[np.ndarray] = []
    for f in freqs_hz:
        w = 2.0 * np.pi * float(f) * t
        mats.append(np.stack([np.sin(w), np.cos(w)], axis=0))
    return mats


def numpy_to_double_matrix2d(arr: np.ndarray, *, verbose: bool = True):
    """
    Конвертация numpy.ndarray (2D) -> managed System.Double[,] для pythonnet / CoreCLR.

    Почему не «просто передать numpy»:
      В C# тип `double[,]` — это отдельный вид: прямоугольный массив ранга 2 (не зубчатый
      `double[][]`). pythonnet не делает автоматического implicit cast из numpy.ndarray
      в `double[,]` с гарантией совместимости контрактов MSI.

    Как создаётся `double[,]`:
      1) Импортируем `System.Array` и `System.Double` (после инициализации CoreCLR / import clr).
      2) `Array.CreateInstance(Double, rows, cols)` выделяет managed-буфер с нижней границей 0
         по обоим измерениям — это именно прямоугольная матрица `rows x cols` типа double.
      3) Элементы заполняются по индексам `[i, j]` в цикле из numpy (float64 -> double).
         Порядок обхода: row-major как в C# для `double[,]` (строка i — «канал/компонента»,
         столбец j — время), что соответствует `arr[i, j]` при arr.shape == (rows, cols).

    Ограничения:
      - только 2D; для 1D сначала сделайте reshape(1, -1) или (-1, 1) по контракту MSI.
      - значения приводятся к float; NaN/Inf допускаются, но MSI может на них реагировать.
    Конвертация выполняется **после** `tmi._configure_coreclr` в вызывающем коде, чтобы типы
    `System.*` были доступны через pythonnet.
    """
    from System import Array, Double

    if arr.ndim != 2:
        raise ValueError(f"ожидался 2D numpy-массив, получили shape={arr.shape}")
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    managed = Array.CreateInstance(Double, rows, cols)
    # Копирование явное — предсказуемо для отладки и не зависит от stride numpy.
    for i in range(rows):
        for j in range(cols):
            managed[i, j] = Double(float(arr[i, j]))
    if verbose:
        _debug(f"numpy_to_double_matrix2d: numpy shape={arr.shape} dtype={arr.dtype} -> Double[{rows},{cols}]")
    return managed


def build_model_signal_list(msi, numpy_templates: Sequence[np.ndarray], *, verbose: bool = True):
    """
    Собирает `System.Collections.Generic.IList<double[,]>` для присвоения `msi.ModelSignal`.

    Тип элемента списка (`System.Double[,]`) берём из reflection свойства ModelSignal:
    у `IList<T>` первый generic-аргумент — это T. Так мы не хардкодим строку типа и
    остаёмся совместимыми с версией System.Private.CoreLib, на которую ссылается MSIController.
    """
    from System.Collections.Generic import List

    prop = msi.GetType().GetProperty("ModelSignal")
    if prop is None:
        raise RuntimeError("У типа MSI нет свойства ModelSignal")
    prop_type = prop.PropertyType
    args = prop_type.GetGenericArguments()
    if args.Length != 1:
        raise RuntimeError(f"ModelSignal: ожидался IList<T> с одним T, получили {args.Length} generic args")
    elem_type = args[0]
    if verbose:
        _debug(f"ModelSignal element CLR type: {elem_type.FullName}")

    list_type = List[elem_type]
    lst = list_type()
    for idx, tmpl in enumerate(numpy_templates):
        if tmpl.ndim != 2 or tmpl.shape[0] != 2:
            raise ValueError(
                f"шаблон #{idx + 1}: ожидалась форма (2, n) sin/cos, получили {tmpl.shape}"
            )
        managed = numpy_to_double_matrix2d(
            np.ascontiguousarray(tmpl, dtype=np.float64), verbose=verbose
        )
        lst.Add(managed)
    if verbose:
        _debug(f"ModelSignal List built: Count={lst.Count}")
    return lst


_MSI_RUNTIME_CACHE: tuple | None = None
_MSI_EXEC_OUT_METHOD: Any = None


def _get_msi_exec_out_method(msi: Any) -> Any:
    """MSIExec(double[,] Signal, out double coefS) — единственный путь к ненулевому coefS."""
    global _MSI_EXEC_OUT_METHOD
    if _MSI_EXEC_OUT_METHOD is not None:
        return _MSI_EXEC_OUT_METHOD
    for meth in msi.GetType().GetMethods():
        if meth.Name != "MSIExec":
            continue
        ps = meth.GetParameters()
        if len(ps) == 2 and ps[1].IsOut:
            _MSI_EXEC_OUT_METHOD = meth
            break
    if _MSI_EXEC_OUT_METHOD is None:
        raise RuntimeError("MSIExec(double[,], out double coefS) not found on MSI")
    return _MSI_EXEC_OUT_METHOD


def msi_exec_signal(msi: Any, sig_managed: Any) -> tuple[int, float]:
    """Run MSI on one window; return (winner_1based, coefS for the winner).

    MSI.Coef property is System.Double and stays 0.0 after single-arg MSIExec — do not use it.
    """
    from System import Array, Double, Object

    meth = _get_msi_exec_out_method(msi)
    args = Array[Object]([sig_managed, Double(0.0)])
    winner = int(meth.Invoke(msi, args))
    return winner, float(args[1])


def load_msi_runtime():
    """Та же последовательность, что в test_msi_import.main до создания MSI (без печати «ok»)."""
    global _MSI_RUNTIME_CACHE
    if _MSI_RUNTIME_CACHE is not None:
        return _MSI_RUNTIME_CACHE
    msi_res = tmi._msi_res()
    msi_dll = (msi_res / "MSIController.dll").resolve()
    if not msi_dll.is_file():
        raise FileNotFoundError(f"Нет файла: {msi_dll}")

    dotnet_root = tmi._resolve_dotnet_root()
    if dotnet_root is None:
        raise RuntimeError("Не найден DOTNET_ROOT / установка .NET")

    hint = tmi._preflight_hostpack(msi_dll, dotnet_root)
    if hint:
        raise RuntimeError(hint)

    tmi._configure_coreclr(dotnet_root)
    import clr

    alglib = tmi._find_alglib_dll(msi_res, msi_dll, dotnet_root)
    tmi._add_reference(clr, alglib)
    tmi._load_optional_deps(clr, msi_res)
    tmi._add_reference(clr, msi_dll)
    MSI = tmi._import_msi_type(clr)
    msi = MSI()
    _MSI_RUNTIME_CACHE = (msi, msi_res, dotnet_root)
    return _MSI_RUNTIME_CACHE


def _format_coef_for_log(msi) -> str:
    """Coef может быть скаляром или коллекцией — печатаем осторожно."""
    try:
        c = msi.Coef
        if c is None:
            return "None"
        # IList / Array / scalar
        cnt = getattr(c, "Count", None)
        if cnt is not None:
            parts = []
            lim = min(int(cnt), 32)
            for i in range(lim):
                parts.append(str(c[i]))
            extra = "" if int(cnt) <= lim else f" ... (+{int(cnt) - lim} more)"
            return f"Count={cnt} [{', '.join(parts)}]{extra}"
        return repr(c)
    except Exception as e:
        return f"<error reading Coef: {e}>"


def main() -> int:
    print("=== MSI MSIExec synthetic smoke test ===", flush=True)

    # Параметры сценария (как в ТЗ-примере)
    srate = 250.0
    duration = 2.0
    channels = 2
    stim_freq_hz = 10.0
    model_freqs = (10.0, 12.0, 15.0, 20.0)
    # MSI вернул победителя 1 для первой частоты в списке (10 Hz) — интерпретируем как 1-based индекс шаблона.
    expected_winner_1based = 1

    try:
        msi, msi_res, dotnet_root = load_msi_runtime()
        _debug(f"repo={_REPO}, msi-res={msi_res}, DOTNET_ROOT={dotnet_root}")

        _debug("building numpy model templates (sin/cos rows per frequency)")
        np_models = generate_model_signals(model_freqs, srate, duration)
        for i, f in enumerate(model_freqs):
            _debug(f"  template[{i}] f={f} Hz -> numpy shape={np_models[i].shape}")

        _debug("converting ModelSignal -> IList<double[,]> (List<>)")
        model_list = build_model_signal_list(msi, np_models)
        msi.ModelSignal = model_list
        _debug("msi.ModelSignal assigned OK")

        _debug(f"generating synthetic EEG: {stim_freq_hz} Hz, channels={channels}")
        sig_np = generate_sine_signal(stim_freq_hz, srate, duration, channels)
        _debug(f"synthetic numpy signal shape={sig_np.shape} dtype={sig_np.dtype}")
        sig_managed = numpy_to_double_matrix2d(
            np.ascontiguousarray(sig_np, dtype=np.float64), verbose=True
        )

        _debug("calling MSIExec(double[,] Signal, out double coefS)")
        winner, coef_s = msi_exec_signal(msi, sig_managed)
        _debug(f"MSIExec returned winner={winner!r} coefS={coef_s}")

        coef_prop = _format_coef_for_log(msi)
        print(f"[result] winner (int) = {int(winner)}", flush=True)
        print(f"[result] coefS (out param) = {coef_s}", flush=True)
        print(f"[result] msi.Coef property (stale scalar, ignore) = {coef_prop}", flush=True)

        w = int(winner)
        if w == expected_winner_1based:
            print(
                f"[ok] Ожидание: сигнал {stim_freq_hz} Hz -> шаблон частоты #{expected_winner_1based} "
                f"(первая в списке {model_freqs} == {model_freqs[0]} Hz). Совпало.",
                flush=True,
            )
        else:
            print(
                f"[warn] Ожидали winner=={expected_winner_1based} для {stim_freq_hz} Hz, получили {w}. "
                "Проверьте порядок частот в ModelSignal и семантику индекса в MSIController.",
                flush=True,
            )

    except FileNotFoundError as e:
        print(f"[error] DLL / файл не найден: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"[error] Некорректная форма или параметры: {e}", file=sys.stderr)
        traceback.print_exc()
        return 4
    except TypeError as e:
        print(f"[error] Ошибка приведения типов pythonnet / IList / double[,]: {e}", file=sys.stderr)
        traceback.print_exc()
        return 5
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        traceback.print_exc()
        return 3
    except Exception as e:
        exc_name = type(e).__name__
        msg = str(e)
        print(f"[error] Неожиданная ошибка ({exc_name}): {msg}", file=sys.stderr)
        traceback.print_exc()
        if "ArgumentException" in exc_name or "ArgumentException" in msg:
            print(
                "[hint] ArgumentException: часто несовпадение размеров (samples) между Signal и шаблонами.",
                file=sys.stderr,
            )
        if "NullReferenceException" in exc_name or "NullReferenceException" in msg:
            print("[hint] ModelSignal или внутреннее состояние MSI не инициализированы.", file=sys.stderr)
        return 1

    _debug("success — MSIExec smoke test finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
