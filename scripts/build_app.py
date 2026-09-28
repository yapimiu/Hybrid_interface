import os
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    spec_path = os.path.join(repo_root, "hybrid_interface.spec")

    if not os.path.exists(spec_path):
        print(f"Spec not found: {spec_path}", file=sys.stderr)
        return 2

    os.chdir(repo_root)
    _run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", spec_path])
    print("\nOK. See dist/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
