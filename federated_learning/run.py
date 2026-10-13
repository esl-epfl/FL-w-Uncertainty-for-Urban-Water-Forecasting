"""Run one paper configuration or a sweep, saving each run's console output."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from federated_learning.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--config', type=Path)
    selection.add_argument('--sweep', choices=('homogeneous', 'heterogeneous', 'all'))
    parser.add_argument('--data-root', type=Path, default=Path('new_data'))
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--num-gpus', type=int, default=None)
    parser.add_argument('--client-cpus', type=int, default=4)
    args = parser.parse_args()
    if args.config:
        configs = [args.config.resolve()]
    else:
        sweeps = ('homogeneous', 'heterogeneous') if args.sweep == 'all' else (args.sweep,)
        configs = [path for sweep in sweeps for path in sorted((ROOT / 'configs' / sweep).glob('*.json'))]
    data_root = args.data_root.resolve()
    missing = sorted({str(data_root / load_config(path)['aquacast_data_path'])
                      for path in configs
                      if not (data_root / load_config(path)['aquacast_data_path']).is_file()})
    if missing:
        parser.error('Required external CSVs are missing:\n' + '\n'.join(missing))

    for path in configs:
        output_dir = args.results_dir.resolve() / f'{path.stem}-{time.time_ns()}'
        output_dir.mkdir(parents=True)
        command = [sys.executable, '-u', '-m', 'federated_learning.main',
                   '--config', str(path), '--data-root', str(data_root),
                   '--output-dir', str(output_dir), '--device', args.device,
                   '--client-cpus', str(args.client_cpus)]
        if args.num_gpus is not None:
            command += ['--num-gpus', str(args.num_gpus)]
        print(f'Running {path.name}\nResults: {output_dir}', flush=True)
        env = os.environ.copy()
        # Every client's timing is needed for the deployment-cost figure.
        env['RAY_DEDUP_LOGS'] = '0'
        with (output_dir / 'run.log').open('w') as log:
            with subprocess.Popen(command, cwd=ROOT.parent, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                code = process.wait()
        if code:
            raise SystemExit(f'{path.name} failed (exit {code}); see {output_dir / "run.log"}')


if __name__ == '__main__':
    main()
