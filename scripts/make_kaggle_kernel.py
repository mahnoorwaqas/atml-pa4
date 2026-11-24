#!/usr/bin/env python3
"""Build a private Kaggle notebook for an allow-listed experiment."""
import argparse
import json
from pathlib import Path

OPTIONS = {"dataset-check": False, "attention-smoke": True, "spec-toy": False}
p = argparse.ArgumentParser()
p.add_argument("--owner", required=True)
p.add_argument("--run-id", required=True)
p.add_argument("--experiment", required=True, choices=OPTIONS)
p.add_argument("--sha", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if len(a.sha) != 40 or not all(c in "0123456789abcdef" for c in a.sha.lower()):
    raise ValueError("Expected a full Git commit SHA")
slug = f"atml-pa4-{a.run_id}"
if len(slug) > 50 or not all(c.islower() or c.isdigit() or c == "-" for c in slug):
    raise ValueError("Invalid run slug")
a.output.mkdir(parents=True, exist_ok=True)
metadata = {
    "id": f"{a.owner}/{slug}",
    "title": f"ATML PA4 {a.experiment} {a.run_id}",
    "code_file": "kernel.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": OPTIONS[a.experiment],
    "enable_internet": True, "dataset_sources": [], "competition_sources": [],
    "kernel_sources": [],
}
config = json.dumps({"experiment": a.experiment, "sha": a.sha, "run_id": a.run_id})
code = """
import json, pathlib, subprocess, sys, time, traceback
config = json.loads(CONFIG_JSON)
out = pathlib.Path('/kaggle/working/pa4_outputs')
out.mkdir(parents=True, exist_ok=True)
commands = []
started = time.monotonic()
status = {'config': config, 'commands': commands}
def run(argv, cwd=None):
    commands.append({'argv': argv, 'cwd': str(cwd) if cwd else None})
    print('+', ' '.join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, check=True)
try:
    data = pathlib.Path('/kaggle/working/staff-data')
    run(['git', 'clone', '--depth', '1',
         'https://github.com/AbDu11aHHH/atml-pa4-datasets.git', str(data)])
    exp = config['experiment']
    if exp == 'dataset-check':
        run([sys.executable, 'scripts/validate_assets.py'], cwd=data)
        run([sys.executable, 'scripts/verify_checksums.py'], cwd=data)
        run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=data)
    else:
        project = pathlib.Path('/kaggle/working/pa4')
        run(['git', 'clone', 'https://github.com/mahnoorwaqas/atml-pa4.git', str(project)])
        run(['git', 'checkout', config['sha']], cwd=project)
        if not (project / 'pa4/runner.py').is_file():
            raise RuntimeError('Upload the PA4 source package before running this experiment.')
        run([sys.executable, '-m', 'pip', 'install', '-e', '.'], cwd=project)
        if (project / 'requirements.txt').exists():
            run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], cwd=project)
        args = ([sys.executable, '-m', 'pa4.runner', 'attention', '--output',
                 str(out/'attention.json'), '--max-length', '1024'] if exp == 'attention-smoke'
                else [sys.executable, '-m', 'pa4.runner', 'spec-toy', '--output',
                      str(out/'markov.json'), '--trials', '100000'])
        run(args, cwd=project)
    status['result'] = 'success'
except Exception as exc:
    status.update(result='failed', error=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    status['elapsed_seconds'] = time.monotonic() - started
    (out/'run_manifest.json').write_text(json.dumps(status, indent=2))
"""
source = "CONFIG_JSON = " + repr(config) + "\n" + code
notebook = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
               "outputs": [], "source": source.splitlines(keepends=True)}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(a.output/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + "\n")
(a.output/'kernel.ipynb').write_text(json.dumps(notebook, indent=2) + "\n")
