#!/usr/bin/env python3
"""
Standalone smoke test: load MSIController.dll via pythonnet (CoreCLR) and construct MSI.

MSIController.dll в этом репозитории собран под **.NET 8** (`net8.0`). Для pythonnet на macOS/Linux
нужен установленный **Microsoft.NETCore.App 8.x** (host/runtime), иначе обёртка типа `MSI`
может падать на более старом runtime (например, только .NET 6).

Не трогает остальной код проекта. Ожидаемая раскладка:
  msi-res/MSIController.dll
  msi-res/alglib.net.3.19.0.nupkg  (для извлечения alglib.net.dll, если нет готового файла; в составе
  3.19.0 обычно **нет** `lib/net8.0` — максимум net7.0; для net8-ALGLIB нужен более новый пакет или свой DLL)
  msi-res/alglib.net.dll | alglibnet.dll  (опционально, если уже распаковано вручную)
  msi-res/deps/*.dll  (опционально: транзитивные managed-зависимости, например CommunityToolkit.Mvvm.dll)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _msi_res() -> Path:
    return _repo_root() / "msi-res"


def _debug(msg: str) -> None:
    print(f"[debug] {msg}", flush=True)


def _guess_msi_target_framework_major(msi_dll: Path) -> int | None:
    """Грубая эвристика по образу DLL (без System.Reflection): ищем .NETCoreApp,Version=vN."""
    try:
        blob = msi_dll.read_bytes()
    except OSError:
        return None
    majors = [int(m) for m in re.findall(rb"\.NETCoreApp,Version=v(\d+)", blob)]
    if not majors:
        return None
    return max(majors)


def _dotnet_exe_in_root(dotnet_root: Path) -> Path | None:
    """Путь к dotnet CLI внутри DOTNET_ROOT (dotnet.exe на Windows)."""
    for name in ("dotnet.exe", "dotnet"):
        candidate = dotnet_root / name
        if candidate.is_file():
            return candidate
    return None


def _installed_netcore_major_versions(dotnet_root: Path) -> set[int]:
    dotnet_exe = _dotnet_exe_in_root(dotnet_root)
    if dotnet_exe is None:
        return set()
    try:
        out = subprocess.run(
            [str(dotnet_exe), "--list-runtimes"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    majors: set[int] = set()
    for line in (out.stdout or "").splitlines():
        if not line.startswith("Microsoft.NETCore.App "):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        ver = parts[1]
        m = re.match(r"(\d+)\.", ver)
        if m:
            majors.add(int(m.group(1)))
    return majors


def _preflight_hostpack(msi_dll: Path, dotnet_root: Path) -> str | None:
    """
    Возвращает текст ошибки, если целевой major выше установленного Microsoft.NETCore.App.
    MSI_SKIP_TFM_CHECK=1 — пропустить проверку (на свой риск: возможен crash pythonnet).
    """
    if os.environ.get("MSI_SKIP_TFM_CHECK", "").strip() in ("1", "true", "yes"):
        return None
    need_major = _guess_msi_target_framework_major(msi_dll)
    if need_major is None:
        return None
    have = _installed_netcore_major_versions(dotnet_root)
    if need_major in have or any(m >= need_major for m in have):
        return None
    return (
        f"Сборка похожа на .NET {need_major}+, а в {dotnet_root} не найден runtime "
        f"Microsoft.NETCore.App {need_major}.x (или новее). Установите .NET {need_major} runtime "
        "с https://dotnet.microsoft.com/download/dotnet/ и проверьте: "
        f'"{dotnet_root / "dotnet"}" --list-runtimes'
    )


def _resolve_dotnet_root() -> Path | None:
    env = os.environ.get("DOTNET_ROOT")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    for candidate in (
        Path(r"C:\Program Files\dotnet"),
        Path(r"C:\Program Files (x86)\dotnet"),
        Path("/opt/homebrew/share/dotnet"),
        Path("/usr/local/share/dotnet"),
    ):
        if candidate.is_dir() and _dotnet_exe_in_root(candidate) is not None:
            return candidate
    try:
        import shutil

        which = shutil.which("dotnet")
        if which:
            root = Path(which).resolve().parent
            if root.is_dir():
                return root
    except Exception:
        pass
    return None


def _configure_coreclr(dotnet_root: Path) -> None:
    """Настроить CoreCLR один раз на процесс (pythonnet.set_runtime нельзя вызывать повторно)."""
    try:
        import pythonnet
    except ImportError as e:
        raise RuntimeError(
            "Не установлен pythonnet / clr_loader. Установите зависимости: pip install -r requirements.txt"
        ) from e
    if pythonnet._RUNTIME is not None or pythonnet._LOADED:
        _debug("CoreCLR runtime already configured; skipping set_runtime")
        import clr  # noqa: F401,WPS433
        return
    _debug(f"configuring CoreCLR runtime (DOTNET_ROOT={dotnet_root})")
    try:
        from clr_loader import get_coreclr
        from pythonnet import set_runtime
    except ImportError as e:
        raise RuntimeError(
            "Не установлен pythonnet / clr_loader. Установите зависимости: pip install -r requirements.txt"
        ) from e
    set_runtime(get_coreclr(dotnet_root=str(dotnet_root)))
    _debug("CoreCLR runtime set; importing clr")
    import clr  # noqa: F401,WPS433 — после set_runtime


def _alglib_net_tfms_in_nupkg(names: set[str]) -> list[str]:
    """Список TFM вида netX.Y, для которых в nupkg есть lib/<tfm>/alglib.net.dll."""
    found: list[str] = []
    for n in names:
        m = re.match(r"lib/(net\d+\.\d+)/alglib\.net\.dll\Z", n)
        if m:
            found.append(m.group(1))

    def _key(tfm: str) -> tuple[int, int]:
        mm = re.match(r"net(\d+)\.(\d+)", tfm)
        if not mm:
            return (0, 0)
        return (int(mm.group(1)), int(mm.group(2)))

    found.sort(key=_key)
    return found


def _find_alglib_dll(
    msi_res: Path,
    msi_dll: Path | None = None,
    dotnet_root: Path | None = None,
) -> Path:
    for name in ("alglibnet.dll", "alglib.net.dll"):
        p = msi_res / name
        if p.is_file():
            _debug(f"using bundled ALGLIB: {p}")
            return p.resolve()
    nupkg = msi_res / "alglib.net.3.19.0.nupkg"
    if not nupkg.is_file():
        raise FileNotFoundError(
            f"Не найден ALGLIB: ни {msi_res / 'alglib.net.dll'}, ни nupkg {nupkg}. "
            "Положите alglib.net.dll рядом с MSIController или добавьте alglib.net.3.19.0.nupkg."
        )
    msi_major = _guess_msi_target_framework_major(msi_dll) if msi_dll is not None else None
    host_majors = _installed_netcore_major_versions(dotnet_root) if dotnet_root is not None else set()
    host_best = max(host_majors) if host_majors else None
    effective_major: int | None
    if msi_major is None and host_best is None:
        effective_major = None
    else:
        effective_major = max(msi_major or 0, host_best or 0)
        if effective_major == 0:
            effective_major = None

    tfm_order = (
        "net8.0",
        "net7.0",
        "net6.0",
        "net5.0",
        "netstandard2.1",
        "netstandard2.0",
        "net48",
        "net47",
        "net45",
    )
    if effective_major == 6:
        tfm_order = ("net6.0", "net7.0", "net8.0", "net5.0", "netstandard2.1", "netstandard2.0", "net48", "net47", "net45")
    elif effective_major == 7:
        tfm_order = ("net7.0", "net8.0", "net6.0", "net5.0", "netstandard2.1", "netstandard2.0", "net48", "net47", "net45")
    member: str | None = None
    with zipfile.ZipFile(nupkg) as zf:
        names = set(zf.namelist())
        present_net = _alglib_net_tfms_in_nupkg(names)
        _debug(f"nupkg содержит ALGLIB для TFMs: {', '.join(present_net) or '(нет lib/net*.0)'}")
        want8 = "lib/net8.0/alglib.net.dll"
        if effective_major is not None and effective_major >= 8 and want8 not in names:
            _debug(
                "нужен net8.0 ALGLIB, но в этом nupkg нет lib/net8.0/alglib.net.dll — "
                "берём максимально доступный TFM из пакета (часто net7.0)"
            )
        for tfm in tfm_order:
            cand = f"lib/{tfm}/alglib.net.dll"
            if cand in names:
                member = cand
                break
    if member is None:
        raise FileNotFoundError(
            f"В {nupkg} не найдено ни одного из ожидаемых TFM ({', '.join(tfm_order)})."
        )
    tmp = Path(tempfile.mkdtemp(prefix="msi_alglib_"))
    with zipfile.ZipFile(nupkg) as zf:
        zf.extract(member, tmp)
    extracted = (tmp / member).resolve()
    _debug(f"extracted ALGLIB from nupkg ({member}) -> {extracted}")
    return extracted


def _add_reference(clr, path: Path) -> None:
    p = str(path.resolve())
    _debug(f"clr.AddReference({p})")
    clr.AddReference(p)


def _load_optional_deps(clr, msi_res: Path) -> None:
    deps_dir = msi_res / "deps"
    if not deps_dir.is_dir():
        _debug(f"optional deps dir missing: {deps_dir} (skip)")
        return
    dlls = sorted(deps_dir.glob("*.dll"))
    if not dlls:
        _debug(f"optional deps dir empty: {deps_dir} (skip)")
        return
    for dll in dlls:
        _debug(f"loading optional dependency DLL: {dll.name}")
        _add_reference(clr, dll)


def _describe_file_not_found(exc: BaseException) -> str:
    msg = str(exc)
    if "CommunityToolkit.Mvvm" in msg:
        return (
            "Сборка MSIController ссылается на CommunityToolkit.Mvvm. "
            f"Скопируйте CommunityToolkit.Mvvm.dll (netstandard2.0 подходит) в {_msi_res() / 'deps'} "
            "или добавьте другие недостающие managed-DLL туда."
        )
    return (
        "Не удалось разрешить managed-зависимость при загрузке типов MSIController. "
        f"Проверьте каталог {_msi_res() / 'deps'} и положите туда недостающие .dll."
    )


def _import_msi_type(clr):
    _debug("importing MSI (from MSIController import MSI)")
    try:
        from MSIController import MSI  # type: ignore[import-not-found]
    except Exception:
        _debug("fallback: import MSIController as module namespace")
        import MSIController  # type: ignore[import-not-found]

        MSI = getattr(MSIController, "MSI", None)
        if MSI is None:
            raise ImportError(
                "Модуль MSIController загружен, но тип MSI не найден. "
                f"Атрибуты: {sorted(dir(MSIController))}"
            ) from None
    return MSI


def _format_member(mi) -> str:
    try:
        return f"{mi.MemberType}:{mi.Name}"
    except Exception:
        return repr(mi)


def _list_public_members(msi_instance) -> list[str]:
    """Перечисление публичных членов по System.Type; у pythonnet класс MSI — не CLR Type."""
    from System.Reflection import BindingFlags

    flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public
    get_type = getattr(msi_instance, "GetType", None)
    if not callable(get_type):
        raise TypeError("Ожидался экземпляр managed-объекта с методом GetType()")
    clr_type = get_type()
    members = list(clr_type.GetMembers(flags))
    lines = [_format_member(mi) for mi in members]
    lines.sort()
    return lines


def main() -> int:
    print("=== MSI pythonnet smoke test ===", flush=True)
    msi_res = _msi_res()
    _debug(f"msi-res directory: {msi_res}")

    if not msi_res.is_dir():
        print(f"[error] DLL directory not found: {msi_res}", file=sys.stderr)
        return 2

    msi_dll = (msi_res / "MSIController.dll").resolve()
    if not msi_dll.is_file():
        print(f"[error] MSIController.dll not found: {msi_dll}", file=sys.stderr)
        return 2

    dotnet_root = _resolve_dotnet_root()
    if dotnet_root is None:
        print(
            "[error] Не найден .NET (CoreCLR). Задайте DOTNET_ROOT или установите .NET SDK/runtime "
            "(ожидаемые пути: /opt/homebrew/share/dotnet, /usr/local/share/dotnet).",
            file=sys.stderr,
        )
        return 3

    tfm_hint = _preflight_hostpack(msi_dll, dotnet_root)
    if tfm_hint:
        print(f"[error] {tfm_hint}", file=sys.stderr)
        return 6

    try:
        _configure_coreclr(dotnet_root)
    except RuntimeError as e:
        print(f"[error] pythonnet / runtime: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print("[error] Не удалось инициализировать CoreCLR для pythonnet.", file=sys.stderr)
        traceback.print_exc()
        return 3

    import clr

    try:
        _debug("loading alglib (alglib.net.dll / nupkg)")
        alglib = _find_alglib_dll(msi_res, msi_dll, dotnet_root)
        _add_reference(clr, alglib)

        _debug("loading optional deps from msi-res/deps")
        _load_optional_deps(clr, msi_res)

        _debug("loading MSIController")
        _add_reference(clr, msi_dll)

        _debug("importing MSI type")
        MSI = _import_msi_type(clr)

        _debug("creating MSI object")
        msi = MSI()

    except FileNotFoundError as e:
        print(f"[error] DLL not found: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"[error] OS error while loading DLL: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2
    except Exception as e:
        exc_type = type(e).__name__
        msg = str(e)
        if "BadImageFormatException" in exc_type or "BadImageFormatException" in msg:
            print(
                "[error] Architecture / image format mismatch (BadImageFormatException). "
                "Убедитесь, что Python и установленный .NET — одной разрядности (обычно x64) "
                "и что MSIController.dll собран под совместимый с установленным runtime TFM.",
                file=sys.stderr,
            )
            traceback.print_exc()
            return 4
        if "FileNotFoundException" in exc_type or "FileNotFoundException" in msg:
            print("[error] Missing managed dependency at load time.", file=sys.stderr)
            print(_describe_file_not_found(e), file=sys.stderr)
            traceback.print_exc()
            return 5
        if "DllNotFoundException" in exc_type or "DllNotFoundException" in msg:
            print(
                "[error] DllNotFoundException: не найден нативный хост libcoreclr / зависимость hostpolicy.",
                file=sys.stderr,
            )
            traceback.print_exc()
            return 3
        if "NullableContextAttribute" in msg or "InternalPythonnetException" in exc_type:
            print(
                "[error] Несовместимость версии .NET runtime с MSIController.dll "
                "(часто: сборка net8.0, а установлен только .NET 6). "
                "Установите Microsoft.NETCore.App 8.x и повторите запуск.",
                file=sys.stderr,
            )
            traceback.print_exc()
            return 6
        print(f"[error] Unexpected failure ({exc_type}): {msg}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("[ok] DLL загружены (ALGLIB, MSIController и опциональные deps).", flush=True)
    print("[ok] Объект MSI успешно создан.", flush=True)

    try:
        lines = _list_public_members(msi)
        print("[info] Публичные члены типа MSI:", flush=True)
        for line in lines:
            print(f"  - {line}", flush=True)
    except Exception:
        print("[warn] Не удалось перечислить члены MSI через reflection:", file=sys.stderr)
        traceback.print_exc()

    _debug("success — smoke test finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
