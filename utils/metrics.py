import numpy as np


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

def metric(pred, true, quantiles=None):

    sql = None
    sql_point = None
    if quantiles is not None:
        sql = SQL(pred, true, quantiles)
        pred_q = quantile_to_point(pred, quantiles, q_point=0.5)
        sql_point = SQL(pred_q[..., np.newaxis], true, [0.5])
        pred = pred_q # use point forecast for other metrics to follow previous convention

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

    return mae, mse, rmse, mape, mspe, rse, corr, R2, EVS, mse_individual, mae_individual, rmse_individual, r2_individual, sql, sql_point