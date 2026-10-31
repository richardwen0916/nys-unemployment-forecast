"""Run offline with audited project I/O confined to this extracted package.

Python installation libraries and operating-system timezone data remain runtime
dependencies. No other project/home files or network connections are allowed.
"""
from pathlib import Path
import argparse
import json
import os
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / '.runtime'
(runtime / 'tmp').mkdir(parents=True, exist_ok=True)
(runtime / 'matplotlib').mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(runtime / 'tmp')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
tempfile.tempdir = str(runtime / 'tmp')
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('command', choices=['inspect','features','tests','reproduce'])
parser.add_argument('--output', default='results/reproduced')
args = parser.parse_args()

libraries = [Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(), Path('/usr/share/zoneinfo')]
opened = set()
blocked_probes = []


def audit(event, values):
    if event == 'socket.connect':
        raise PermissionError('Offline handoff: network connections disabled')
    if event != 'open' or not values or isinstance(values[0], int):
        return
    path = Path(os.fsdecode(values[0])).resolve()
    if path.is_relative_to(ROOT):
        opened.add(str(path.relative_to(ROOT)))
        return
    if path in (Path('/dev/null'), Path('/etc/localtime')) or any(path.is_relative_to(p) for p in libraries):
        return
    blocked_probes.append('blocked external file-open attempt')
    raise PermissionError('Package isolation blocked an external project/home file')


sys.addaudithook(audit)
try:
    if args.command == 'tests':
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / 'tests')))
        if not result.wasSuccessful() or result.skipped:
            raise SystemExit(1)
    else:
        sys.argv = ['src.handoff', args.command] + (['--output', args.output] if args.command == 'reproduce' else [])
        runpy.run_module('src.handoff', run_name='__main__')
finally:
    receipt = dict(command=args.command, blocked_external_file_open_attempts=len(blocked_probes),
        outside_project_reads_permitted=False,
        project_files_accessed=sorted(opened), network_connections_disabled=True,
        exception='Installed interpreter libraries and system timezone files are permitted runtime dependencies')
    (runtime / f'isolation_{args.command}.json').write_text(json.dumps(receipt, indent=2) + '\n')
