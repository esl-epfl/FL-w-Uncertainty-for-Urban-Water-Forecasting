import numpy as np

def _validate_quantile_inputs(pred_q, true, quantiles):
    """
    pred_q: [B, T, V, Q]
    true:   [B, T, V]
    """
    pred_q = np.asarray(pred_q, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    qs = np.asarray(quantiles, dtype=np.float64)

    if pred_q.ndim != 4:
        raise ValueError("pred_q must have shape [B, T, V, Q].")
    if true.shape != pred_q.shape[:-1]:
        raise ValueError(
            f"true shape {true.shape} must match "
            f"pred_q.shape[:-1] {pred_q.shape[:-1]}."
        )
    if qs.ndim != 1 or len(qs) != pred_q.shape[-1]:
        raise ValueError("quantiles must match the last dimension of pred_q.")
    if np.any((qs <= 0.0) | (qs >= 1.0)):
        raise ValueError("All quantiles must be strictly between 0 and 1.")
    if np.any(np.diff(qs) <= 0):
        raise ValueError("quantiles must be strictly increasing.")

    return pred_q, true, qs


def quantile_prediction(pred_q, quantiles, q, atol=1e-10):
    """
    Return the prediction at quantile q.

    Exact available quantiles are matched with numerical tolerance.
    Otherwise, linear interpolation is used.
    """
    qs = np.asarray(quantiles, dtype=np.float64)
    q = float(q)

    # Check approximate equality before checking the boundaries.
    exact = np.where(np.isclose(qs, q, atol=atol, rtol=0.0))[0]
    if len(exact):
        return pred_q[..., exact[0]]

    if q < qs[0] - atol or q > qs[-1] + atol:
        raise ValueError(
            f"Requested quantile {q:.12f} is outside the available "
            f"range [{qs[0]:.12f}, {qs[-1]:.12f}]."
        )

    # Protect against insignificant boundary-rounding differences.
    q = np.clip(q, qs[0], qs[-1])

    upper_idx = np.searchsorted(qs, q, side="right")
    lower_idx = upper_idx - 1

    q_lower = qs[lower_idx]
    q_upper = qs[upper_idx]
    weight = (q - q_lower) / (q_upper - q_lower)

    return (
        (1.0 - weight) * pred_q[..., lower_idx]
        + weight * pred_q[..., upper_idx]
    )


def quantile_calibration_metrics(pred_q, true, quantiles):
    """
    Quantile calibration and reliability-diagram data.

    empirical_coverage[k] estimates:
        P(Y <= predicted q_k quantile)

    A calibrated model should have:
        empirical_coverage[k] approximately equal to quantiles[k].
    """
    pred_q, true, qs = _validate_quantile_inputs(
        pred_q, true, quantiles
    )

    indicators = true[..., np.newaxis] <= pred_q  # [B,T,V,Q]

    # Reliability-diagram coordinates
    empirical_coverage = indicators.mean(axis=(0, 1, 2))  # [Q]
    calibration_error = empirical_coverage - qs           # [Q]

    # Equal-weight mean absolute calibration error
    mean_absolute_calibration_error = np.mean(
        np.abs(calibration_error)
    )

    # Quantile-weighted ECE over [0,1].
    # Voronoi weights also support nonuniform quantile grids.
    edges = np.concatenate(
        ([0.0], 0.5 * (qs[:-1] + qs[1:]), [1.0])
    )
    quantile_weights = np.diff(edges)

    quantile_ece = np.sum(
        quantile_weights * np.abs(calibration_error)
    )

    return {
        # Use these two arrays for the reliability diagram.
        "nominal_quantiles": qs,
        "empirical_coverage": empirical_coverage,

        "calibration_error": calibration_error,
        "absolute_calibration_error": np.abs(calibration_error),
        "mean_absolute_calibration_error": (
            mean_absolute_calibration_error
        ),
        "quantile_ece": quantile_ece,

        # Diagnostic breakdowns
        "empirical_coverage_by_horizon": indicators.mean(
            axis=(0, 2)
        ),  # [T,Q]
        "empirical_coverage_by_variable": indicators.mean(
            axis=(0, 1)
        ),  # [V,Q]
    }


def interval_metrics(
    pred_q,
    true,
    quantiles,
    central_levels=(0.8, 0.9),
):
    """
    Coverage, calibration, sharpness, and interval score for central
    prediction intervals.

    For level=0.8, the interval is [q_0.1, q_0.9].
    For level=0.9, the interval is [q_0.05, q_0.95].
    """
    pred_q, true, qs = _validate_quantile_inputs(
        pred_q, true, quantiles
    )

    results = {}

    for level in central_levels:
        if not 0.0 < level < 1.0:
            raise ValueError("Interval levels must be between 0 and 1.")

        alpha = 1.0 - level
        q_lower = alpha / 2.0
        q_upper = 1.0 - alpha / 2.0
        name = f"{100 * level:.0f}%"

        # Do not extrapolate beyond the modeled quantile range.
        atol = 1e-10

        if q_lower < qs[0] - atol or q_upper > qs[-1] + atol:
            results[name] = {
                "available": False,
                "reason": (
                    f"Requires quantiles {q_lower:.3f} and "
                    f"{q_upper:.3f}, but available range is "
                    f"[{qs[0]:.3f}, {qs[-1]:.3f}]."
                ),
            }
            continue

        lower = quantile_prediction(pred_q, qs, q_lower)
        upper = quantile_prediction(pred_q, qs, q_upper)

        interval_crossing = lower > upper
        inside = (true >= lower) & (true <= upper)
        width = upper - lower

        empirical_coverage = inside.mean()
        signed_coverage_error = empirical_coverage - level

        # Winkler/interval score: lower is better.
        interval_score_values = (
            width
            + (2.0 / alpha) * (lower - true) * (true < lower)
            + (2.0 / alpha) * (true - upper) * (true > upper)
        )

        results[name] = {
            "available": True,
            "nominal_coverage": level,

            # Empirical coverage probability / PICP
            "empirical_coverage": empirical_coverage,

            # Signed value shows under- versus overcoverage
            "signed_coverage_error": signed_coverage_error,
            "absolute_coverage_error": abs(signed_coverage_error),

            # Interval sharpness / MPIW
            "mean_interval_width": width.mean(),

            # Proper interval scoring rule
            "mean_interval_score": interval_score_values.mean(),

            # Diagnostics
            "interval_crossing_rate": interval_crossing.mean(),
            "coverage_by_horizon": inside.mean(axis=(0, 2)),  # [T]
            "width_by_horizon": width.mean(axis=(0, 2)),      # [T]
            "coverage_by_variable": inside.mean(axis=(0, 1)), # [V]
            "width_by_variable": width.mean(axis=(0, 1)),     # [V]
        }

    return results


def approximate_crps(pred_q, true, quantiles):
    """
    Approximate CRPS from a set of predicted quantiles.

    CRPS(F,y) = 2 * integral_0^1 pinball_loss(q) dq.

    Voronoi integration weights support both uniform and nonuniform
    quantile grids. The approximation becomes more representative as
    the quantile grid becomes denser and covers more of (0,1).
    """
    pred_q, true, qs = _validate_quantile_inputs(
        pred_q, true, quantiles
    )

    error = true[..., np.newaxis] - pred_q
    losses = np.maximum(
        qs * error,
        (qs - 1.0) * error,
    )  # [B,T,V,Q]

    edges = np.concatenate(
        ([0.0], 0.5 * (qs[:-1] + qs[1:]), [1.0])
    )
    quantile_weights = np.diff(edges)  # sums to 1

    crps_values = 2.0 * np.sum(
        losses * quantile_weights,
        axis=-1,
    )  # [B,T,V]

    return {
        "crps": crps_values.mean(),
        "crps_by_horizon": crps_values.mean(axis=(0, 2)),  # [T]
        "crps_by_variable": crps_values.mean(axis=(0, 1)), # [V]
    }


def quantile_crossing_rate(pred_q):
    """
    Fraction of adjacent predicted quantile pairs that cross.

    A crossing occurs when q_i prediction > q_(i+1) prediction.
    Ideally this value is zero.
    """
    pred_q = np.asarray(pred_q)
    return np.mean(pred_q[..., :-1] > pred_q[..., 1:])


def uncertainty_metrics(
    pred_q,
    true,
    quantiles,
    central_levels=(0.8, 0.9),
):
    """
    Consolidated uncertainty evaluation.
    """
    calibration = quantile_calibration_metrics(
        pred_q, true, quantiles
    )
    intervals = interval_metrics(
        pred_q,
        true,
        quantiles,
        central_levels=central_levels,
    )
    crps = approximate_crps(pred_q, true, quantiles)

    return {
        "calibration": calibration,
        "intervals": intervals,
        "crps": crps,
        "quantile_crossing_rate": quantile_crossing_rate(pred_q),
    }

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def R2_score(pred, true):
    return 1 - (np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2))

def R2_individual(pred, true):
    ss_res = np.sum((true - pred) ** 2, axis=(0, 1))
    ss_tot = np.sum((true - np.mean(true, axis=(0, 1), keepdims=False)) ** 2, axis=(0, 1))
    return 1 - (ss_res / ss_tot)

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MAE_individual(pred, true):
    return np.mean(np.abs(pred - true), axis=(0,1))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def MSE_individual(pred, true):
    return np.mean((pred - true) ** 2, axis=(0,1))


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def RMSE_individual(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=(0,1)))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def explained_variance_score(pred, true): #explained_variance_score
    return 1 - (np.var(true - pred, ddof=1) / np.var(true, ddof=1))

def pinball_loss(pred, true, q):
    """
    pred: [B, T, V] predictions for one quantile
    true: [B, T, V]
    q: scalar in (0,1)
    """
    e = true - pred
    return np.maximum(q * e, (q - 1.0) * e)  # same as q*relu(e) + (1-q)*relu(-e)

def WQL(pred_q, true, quantiles):
    """
    Weighted Quantile Loss (mean over quantiles, then mean over B,T,V)
    pred_q: [B, T, V, Q]
    true:   [B, T, V]
    """
    qs = np.asarray(quantiles, dtype=np.float64)
    assert pred_q.shape[-1] == len(qs)

    losses = []
    for i, q in enumerate(qs):
        losses.append(pinball_loss(pred_q[..., i], true, q))  # [B,T,V]
    losses = np.stack(losses, axis=-1)  # [B,T,V,Q]

    # average over quantiles then over B,T,V
    return losses.mean(axis=-1).mean()

def SQL(pred_q, true, quantiles, eps=1e-8):
    """
    Scaled Quantile Loss:
      SQL = WQL / mean(|true|)
    averaged over all forecast steps and batch items (and variables).
    """
    wql = WQL(pred_q, true, quantiles)
    scale = np.mean(np.abs(true)) + eps
    return wql / scale

def quantile_to_point(pred_q, quantiles, q_point=0.5):
    qs = np.asarray(quantiles)
    idx = int(np.argmin(np.abs(qs - q_point)))
    return pred_q[..., idx]  # [B,T,V]

def metric(pred, true, quantiles=None, interval_levels=(0.8, 0.9)):

    sql = None
    sql_point = None
    if quantiles is not None:
        # Here pred has shape [B,T,V,Q].
        pred_q = pred

        sql = SQL(pred_q, true, quantiles)

        pred_point = quantile_to_point(
            pred_q,
            quantiles,
            q_point=0.5,
        )
        sql_point = SQL(
            pred_point[..., np.newaxis],
            true,
            [0.5],
        )

        uncertainty = uncertainty_metrics(
            pred_q,
            true,
            quantiles,
            central_levels=interval_levels,
        )

        # Median forecast for deterministic metrics.
        pred = pred_point

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    R2 = R2_score(pred, true)
    EVS = explained_variance_score(pred, true)

    mse_individual = MSE_individual(pred, true)
    mae_individual = MAE_individual(pred, true)
    rmse_individual = RMSE_individual(pred, true)
    r2_individual = R2_individual(pred, true)

    nse = r2_individual.mean()  # Nash-Sutcliffe Efficiency, average of individual R2 scores across variables, r2_individual = nse_individual

    return mae, mse, rmse, mape, mspe, rse, corr, R2, EVS, nse, mse_individual, mae_individual, rmse_individual, r2_individual, sql, sql_point, uncertainty