"""Adapt the shared AquaCast metrics to the original FL log format."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils import metrics as _M


def _to_python_scalar(x):
    if isinstance(x, np.ndarray):
        return x.tolist() if x.ndim > 0 else float(x)
    if isinstance(x, np.generic):
        return x.item()
    return x


def _scalarize(x) -> float:
    """Coerce a metric that may be scalar or vector into a single float by
    averaging. Used for `corr`, which the reference returns per timestep."""
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return float("nan")
        return float(x.mean())
    if isinstance(x, np.generic):
        return float(x.item())
    return float(x)


def compute_quantile_metrics(
    pred_q: np.ndarray,
    true: np.ndarray,
    quantiles: List[float],
) -> Dict[str, float]:
    """Compute the quantile-metric suite using the shared utils.metrics module.

    pred_q: [N, T, V, Q] numpy array of predicted quantiles.
    true:   [N, T, V]    numpy array of ground truth.
    quantiles: list of Q quantile levels in (0, 1).

    Returns a dict with scalar metrics and per-variable arrays converted to
    plain Python lists (so the dict is JSON / log-line friendly).
    """
    if pred_q.ndim != 4:
        raise ValueError(f"pred_q must be 4D [N,T,V,Q], got shape {pred_q.shape}")
    if true.ndim != 3:
        raise ValueError(f"true must be 3D [N,T,V], got shape {true.shape}")
    if pred_q.shape[:3] != true.shape:
        raise ValueError(
            f"pred_q[:3]={pred_q.shape[:3]} does not match true.shape={true.shape}"
        )
    if pred_q.shape[-1] != len(quantiles):
        raise ValueError(
            f"pred_q has Q={pred_q.shape[-1]} quantiles but {len(quantiles)} "
            "were passed in."
        )

    pred_q = pred_q.astype(np.float64, copy=False)
    true = true.astype(np.float64, copy=False)

    (
        mae, mse, rmse, mape, mspe, rse, corr, R2, EVS, nse,
        mse_individual, mae_individual, rmse_individual, r2_individual,
        sql, sql_point, _uncertainty,
    ) = _M.metric(pred_q, true, quantiles=list(quantiles))

    # WQL is the un-scaled counterpart of SQL; useful to log for cross-dataset
    # comparison even though the reference `metric()` only returns SQL.
    wql = _M.WQL(pred_q, true, list(quantiles))

    return {
        "mae": _scalarize(mae),
        "mse": _scalarize(mse),
        "rmse": _scalarize(rmse),
        "mape": _scalarize(mape),
        "mspe": _scalarize(mspe),
        "rse": _scalarize(rse),
        "corr": _scalarize(corr),
        "r2": _scalarize(R2),
        "evs": _scalarize(EVS),
        "nse": _scalarize(nse),
        "wql": _scalarize(wql),
        "sql": _scalarize(sql) if sql is not None else float("nan"),
        "sql_point": _scalarize(sql_point) if sql_point is not None else float("nan"),
        "mse_individual": _to_python_scalar(mse_individual),
        "mae_individual": _to_python_scalar(mae_individual),
        "rmse_individual": _to_python_scalar(rmse_individual),
        "r2_individual": _to_python_scalar(r2_individual),
    }


SCALAR_METRIC_KEYS = (
    "mae", "mse", "rmse", "mape", "mspe", "rse", "corr",
    "r2", "evs", "nse", "wql", "sql", "sql_point",
)
