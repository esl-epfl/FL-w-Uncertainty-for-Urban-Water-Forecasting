"""Original FedAvg round configuration, timing, and per-partition evaluation."""

import json
import time
from pathlib import Path

import flwr as fl

from federated_learning.quantile_metrics import SCALAR_METRIC_KEYS
from federated_learning.train import test_aquacast_partitioned
from federated_learning.utils import get_server_model, set_parameters


class FedAvg(fl.server.strategy.FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        self._round_configure_t0 = time.perf_counter()
        print(f'[TIMING] round={server_round} stage=configure_fit_start', flush=True)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != self.min_fit_clients:
            raise RuntimeError(
                f'Round {server_round} did not receive all client updates '
                f'({len(results)} returned, {len(failures)} failed).'
            )
        start = time.perf_counter()
        aggregated = super().aggregate_fit(server_round, results, failures)
        elapsed = time.perf_counter() - start
        print(
            f'[TIMING] round={server_round} stage=aggregate_fit '
            f'num_clients={len(results)} num_failures={len(failures)} '
            f'aggregation_seconds={elapsed:.4f}', flush=True,
        )
        print(
            f'[TIMING] round={server_round} stage=round_total '
            f'round_seconds={time.perf_counter() - self._round_configure_t0:.4f}',
            flush=True,
        )
        return aggregated


def gen_evaluate_fn(args, testloaders):
    def evaluate(server_round, parameters, config):
        start = time.perf_counter()
        model, _ = get_server_model(args)
        model = model.to(args.device)
        set_parameters(model, parameters)
        set_params_dt = time.perf_counter() - start
        test_t0 = time.perf_counter()
        loss, metrics = test_aquacast_partitioned(model, testloaders, args)
        test_dt = time.perf_counter() - test_t0
        print(
            f"Server-side evaluation MSE {loss} / avg_client_mse {metrics['avg_client_mse']} / "
            f"max_node_mse {metrics['max_node_mse']} / client_mse {metrics['client_mse']}",
            flush=True,
        )
        parts = [f'round={server_round}']
        for key in SCALAR_METRIC_KEYS:
            parts.append(f"avg_client_{key}={metrics[f'avg_client_{key}']}")
        print('Server-side quantile metrics ' + ' / '.join(parts), flush=True)
        for key in SCALAR_METRIC_KEYS:
            print(
                f"Server-side quantile per-client round={server_round} "
                f"client_{key}={metrics[f'client_{key}']}", flush=True,
            )
        print(
            f'[TIMING] round={server_round} stage=server_eval mode=partitioned '
            f'num_partitions={len(testloaders)} set_params_seconds={set_params_dt:.4f} '
            f'test_seconds={test_dt:.4f} total_seconds={time.perf_counter() - start:.4f}',
            flush=True,
        )
        # Keep arrays in JSON/logs; Flower metrics must be scalar values.
        with (Path(args.output_dir) / 'metrics.jsonl').open('a') as output:
            output.write(json.dumps({'round': server_round, **metrics}) + '\n')
        scalars = {key: value for key, value in metrics.items() if not isinstance(value, list)}
        return loss, scalars

    return evaluate
