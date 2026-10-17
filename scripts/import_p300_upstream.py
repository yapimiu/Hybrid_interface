"""Refresh the explicitly selected, unmodified upstream P300 modules."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / '.upstream-hybrid-bci'
MODULES = tuple(path.stem for path in sorted((SOURCE / 'p300_analysis').glob('*.py')))
TESTS = ('epoch_geometry', 'epoch_indexing', 'winner_metrics', 'calibration',
         'analysis_profiles', 'min_epochs_to_decide', 'p300_external_template',
         'run_export', 'exam_session_detail_logger', 'lsl_streams_probe')

def main():
    target = ROOT / 'p300_analysis'
    target.mkdir(exist_ok=True)
    hashes = {}
    for name in MODULES:
        relative = f'p300_analysis/{name}.py'
        shutil.copyfile(SOURCE / relative, ROOT / relative)
        hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    tests = ROOT / 'tests' / 'upstream'
    tests.mkdir(exist_ok=True)
    for name in TESTS:
        relative = f'tests/test_{name}.py'
        shutil.copyfile(SOURCE / relative, tests / f'test_{name}.py')
        hashes[f'tests/upstream/test_{name}.py'] = hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest()
    shutil.copyfile(SOURCE / 'LICENSE', target / 'LICENSE')
    # Original standalone stimulus and protocol entry points run in their own
    # Python environment, avoiding collisions between the two projects' app packages.
    standalone = ROOT / 'vendor' / 'hybrid_bci'
    standalone.mkdir(parents=True, exist_ok=True)
    for folder in ('app', 'core', 'gui', 'p300_analysis', 'runtime_core',
                   'experiment_protocol', 'ssvep_analysis', 'validation', 'scripts'):
        for source in (SOURCE / folder).rglob('*.py'):
            relative = source.relative_to(SOURCE)
            destination = standalone / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            hashes[destination.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ('run_app.py', 'run_stimulus.py', 'config.py', 'requirements.txt', 'LICENSE', 'settings.json', 'Settings.xml'):
        shutil.copyfile(SOURCE / name, standalone / name)
        hashes[(standalone / name).relative_to(ROOT).as_posix()] = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    for source in (SOURCE / 'msi-res').rglob('*'):
        if source.is_file():
            destination = standalone / source.relative_to(SOURCE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            hashes[destination.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ('__init__', 'synthetic_streams', 'p300_synthetic'):
        destination = ROOT / 'validation' / (name + '.py')
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(SOURCE / 'validation' / (name + '.py'), destination)
        hashes[destination.relative_to(ROOT).as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (target / 'UPSTREAM.json').write_text(json.dumps({
        'repository': 'https://github.com/SergPronin/hybrid-bci-p300-ssvep',
        'commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'sha256': hashes,
        'changes': 'None. Host-specific integration lives in app/bci.'
    }, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
