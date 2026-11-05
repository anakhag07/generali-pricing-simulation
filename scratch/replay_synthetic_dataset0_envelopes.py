"""Replay requested synthetic reports with each study's original numerical imports."""

from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('projects', nargs='+', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker:
        project, = args.projects
        contract = json.loads((project/'contract.json').read_text())
        source = next(s for s in contract['contract']['sources']
                      if s['path'].endswith('/envelopes/polynomial_study.py'))
        sys.path.insert(0, str(Path(source['path']).parents[2]))
        spec = importlib.util.spec_from_file_location('experiments.envelopes.dataset0_reporting',
            root/'src/experiments/envelopes/polynomial_reporting.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        report = module.write_context_envelope_report(project)
        print(f"{project.name}: {report['evaluation_contexts']} contexts, "
              f"width parity {max(report['width_parity_max_abs'].values()):.3g}", flush=True)
        return
    index = root/'scratch/retired_experiments/synthetic_envelope_dataset0_20261105.json'
    reports = ({r['report']: r for r in json.loads(index.read_text())['reports']}
               if index.exists() else {})
    for project in args.projects:
        project = project.resolve()
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(project)], check=True)
        report_path = project/'envelope_width_dataset_0.json'
        report = json.loads(report_path.read_text())
        reports[str(report_path)] = {'report': str(report_path), 'relocations': report['relocations']}
    index.write_text(json.dumps({'reason': 'Dataset-zero function/envelope panels and width legend simplification.',
                                'reports': list(reports.values())}, indent=2)+'\n')


if __name__ == '__main__':
    main()
