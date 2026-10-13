"""Parse the original per-client metric and timing log formats."""

import json
from pathlib import Path
import re
import statistics

import numpy as np

CLIENT_FIT_RE = re.compile(
    r"\[TIMING\] round=(\d+) stage=client_fit cid=(\S+) .*? "
    r"train_seconds=([\d.]+) get_params_seconds=([\d.]+) fit_total_seconds=([\d.]+)"
)
AGG_RE = re.compile(
    r"\[TIMING\] round=(\d+) stage=aggregate_fit "
    r"num_clients=(\d+) num_failures=(\d+) aggregation_seconds=([\d.]+)"
)
SERVER_EVAL_RE = re.compile(
    r"\[TIMING\] round=(\d+) stage=server_eval(?: mode=\S+)?(?: num_partitions=\d+)? "
    r"set_params_seconds=([\d.]+) test_seconds=([\d.]+) total_seconds=([\d.]+)"
)
ROUND_TOTAL_RE = re.compile(
    r"\[TIMING\] round=(\d+) stage=round_total round_seconds=([\d.]+)"
)


def quantiles_or_nan(xs):
    """Return (q25, q50, q75) over a list, NaN if empty."""
    xs = [x for x in xs if x is not None]
    if not xs:
        nan = float("nan")
        return nan, nan, nan
    if len(xs) == 1:
        v = xs[0]
        return v, v, v
    qs = statistics.quantiles(xs, n=4, method="inclusive")
    # statistics.quantiles with n=4 returns [Q1, Q2, Q3].
    return qs[0], qs[1], qs[2]


def parse_log(path: Path):
    """Return per-round timing samples extracted from one log file."""
    samples = {
        "client_fit_train_s": [],     # one entry per (round, cid)
        "client_fit_total_s": [],
        "agg_s": [],                  # one entry per round
        "server_eval_total_s": [],
        "round_total_s": [],
    }
    with path.open("r", errors="replace") as f:
        for line in f:
            m = CLIENT_FIT_RE.search(line)
            if m:
                samples["client_fit_train_s"].append(float(m.group(3)))
                samples["client_fit_total_s"].append(float(m.group(5)))
                continue
            m = AGG_RE.search(line)
            if m:
                samples["agg_s"].append(float(m.group(4)))
                continue
            m = SERVER_EVAL_RE.search(line)
            if m:
                samples["server_eval_total_s"].append(float(m.group(4)))
                continue
            m = ROUND_TOTAL_RE.search(line)
            if m:
                samples["round_total_s"].append(float(m.group(2)))
                continue
    return samples



def read_run(directory):
    """Return resolved settings, best-round MSE statistics, and timing samples."""
    config = json.loads((directory / 'config.json').read_text())
    # A completed simulation writes history last. Ignore interrupted runs.
    if not (directory / 'history.json').exists():
        return None
    best = None
    with (directory / 'metrics.jsonl').open() as metrics:
        for line in metrics:
            record = json.loads(line)
            values = np.asarray(record['client_mse'], dtype=float)
            if not values.size or not np.isfinite(values).all():
                continue
            mean = float(values.mean())
            if best is None or mean < best['mean']:
                best = {'mean': mean, 'std': float(values.std(ddof=0)),
                        'round': record['round']}
    if best is None:
        return None
    return {'config': config, 'best': best, 'timings': parse_log(directory / 'run.log'),
            'path': str(directory)}
