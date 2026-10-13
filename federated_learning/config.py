"""Load shared experiment settings without duplicating them in every JSON file."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_config(path):
    path = Path(path).resolve()
    with path.open() as source:
        config = json.load(source)
    parent = config.pop('extends', None)
    if parent is not None:
        config = {**load_config(path.parent / parent), **config}
    config.setdefault('aquacast_partition_sizes', None)
    return config
