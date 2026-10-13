"""Flower simulation entry point; use federated_learning.run to capture logs."""

import os

# Match the original driver and keep Ray workers from oversubscribing CPUs.
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import flwr as fl
import torch

from federated_learning.client import gen_client_fn
from federated_learning.config import load_config
from federated_learning.datasets import load_aquacast_datasets
from federated_learning.strategy import FedAvg, gen_evaluate_fn
from federated_learning.utils import get_parameters, get_server_model, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--num-gpus', type=int, default=None)
    parser.add_argument('--client-cpus', type=int, default=4)
    cli = parser.parse_args()
    cfg = load_config(cli.config)
    # This integration deliberately supports only the paper's quantile runs.
    required = {
        'model': 'AquaCast', 'dataset': 'synthesized', 'algorithm': 'Dense',
        'aquacast_quantile': True, 'aquacast_features': 'M',
        'aquacast_partition_mode': 'narrow', 'aquacast_server_eval_mode': 'partitioned',
        'exo': True, 'exo_future': True, 'optimizer': 'adam', 'use_round_fit_lr': False,
        'fraction_fit': 1.0, 'fraction_evaluate': 0.0, 'min_evaluate_clients': 0,
    }
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f'The paper runner requires {key}={value!r}.')
    if cfg['min_fit_clients'] != cfg['num_clients']:
        raise ValueError('The paper uses full client participation.')
    device = cli.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = (torch.cuda.device_count() if cli.num_gpus is None else cli.num_gpus) if device == 'cuda' else 0
    if device == 'cuda' and not 1 <= num_gpus <= torch.cuda.device_count():
        raise ValueError('--num-gpus must be between 1 and the number of visible GPUs.')
    if cli.client_cpus < 1:
        raise ValueError('--client-cpus must be positive.')

    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'config.json').exists():
        raise FileExistsError(f'Run directory already contains an experiment: {output_dir}')
    cfg.update(aquacast_root_path=str(Path(cli.data_root).resolve()),
               output_dir=str(output_dir), device=device, num_gpus=num_gpus,
               client_cpus=cli.client_cpus)
    args = SimpleNamespace(**cfg)
    set_seed(args.seed)
    trainloaders, valloaders, testloaders = load_aquacast_datasets(args, args.num_clients)
    model, _ = get_server_model(args)
    cfg['num_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg['model_mb'] = cfg['num_parameters'] * 4 / (1024 ** 2)
    cfg['round_comm_mb_per_client'] = 2 * cfg['model_mb']
    cfg['round_comm_mb_total'] = args.num_clients * cfg['round_comm_mb_per_client']
    dataset = trainloaders[0].dataset
    cfg['num_sensors'] = getattr(dataset, 'base_ds', dataset).data_x.shape[1] - 1
    (output_dir / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print(json.dumps(cfg, indent=2), flush=True)

    parameters = fl.common.ndarrays_to_parameters(get_parameters(model))
    del model
    strategy = FedAvg(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=args.num_clients, min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        initial_parameters=parameters,
        on_fit_config_fn=lambda server_round: {'round': server_round},
        evaluate_fn=gen_evaluate_fn(args, testloaders),
    )
    # FedAvg weights by the number of time windows. All sensor partitions
    # have the same windows, so aggregation is uniform, including Zipf runs.
    repo_root = str(Path(__file__).resolve().parents[1])
    pythonpath = os.pathsep.join(filter(None, [repo_root, os.environ.get('PYTHONPATH')]))
    history = fl.simulation.start_simulation(
        client_fn=gen_client_fn(trainloaders, valloaders, args),
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=3600),
        strategy=strategy,
        client_resources={'num_cpus': cli.client_cpus, 'num_gpus': 1 if device == 'cuda' else 0},
        ray_init_args={
            'num_gpus': num_gpus, 'include_dashboard': False, 'log_to_driver': True,
            'runtime_env': {'env_vars': {'PYTHONPATH': pythonpath}},
        },
    )
    (output_dir / 'history.json').write_text(json.dumps({
        'losses_centralized': history.losses_centralized,
        'metrics_centralized': history.metrics_centralized,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
