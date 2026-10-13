"""AquaCast training and per-partition evaluation ported from FederatedMonarch."""

import time

import numpy as np
import torch
from tqdm import tqdm


def _get_owned_cols(loader):
    """Return the output columns owned by this client, or None for C=1."""
    ds = loader.dataset
    owned = getattr(ds, 'owned_cols', None)
    if owned is None:
        owned = getattr(getattr(ds, 'dataset', None), 'owned_cols', None)
    return owned


def _prepare_aquacast_target(batch_y, output_dim, args):
    """
    Match native AquaCast target preparation for both raw centralized loaders
    and FL's rearranged zero-padded loaders.
    """
    features = getattr(args, 'aquacast_features', 'M')
    exo = getattr(args, 'exo', False)
    f_dim = -1 if features == 'MS' else 0

    target = batch_y[:, -args.pred_len:, f_dim:]

    # Raw centralized loaders keep the exogenous column in batch_y; remove it
    # exactly like AquaCast's native train/vali/test loops do.
    if features == 'M' and exo and target.shape[-1] == output_dim + 1:
        target_col = target[:, :, -1:]
        target = torch.cat((target[:, :, :-2], target_col), dim=-1)

    if target.shape[-1] != output_dim:
        raise ValueError(
            f"AquaCast target/output mismatch: target={target.shape}, output_dim={output_dim}, "
            f"features={features}, exo={exo}"
        )

    return target


def _split_aquacast_inputs(batch_x, args):
    """
    Unpack AquaCast batches when future exogenous inputs are enabled.
    """
    if not getattr(args, 'exo_future', False):
        return batch_x, None

    if isinstance(batch_x, (list, tuple)) and len(batch_x) == 2:
        return batch_x[0], batch_x[1]

    raise ValueError(
        "AquaCast is configured with exo_future=True, but the dataloader did not "
        "return [seq_x, exo_future]."
    )


def _aquacast_quantiles(args):
    qs = getattr(args, 'quantiles', None) or [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    if isinstance(qs, str):
        qs = [float(q) for q in qs.split(',') if q.strip()]
    return [float(q) for q in qs]


def _pinball_loss(outputs_q, target, quantiles, owned_cols=None):
    """
    Pinball (quantile) loss.
      outputs_q: [B, T, V, Q]
      target:    [B, T, V]
    Optionally restrict to `owned_cols` along V before averaging.
    """
    if owned_cols is not None:
        outputs_q = outputs_q[:, :, owned_cols, :]
        target = target[:, :, owned_cols]

    q = torch.tensor(quantiles, device=outputs_q.device, dtype=outputs_q.dtype)
    q = q.view(1, 1, 1, -1)
    target_e = target.unsqueeze(-1)
    under = torch.relu(target_e - outputs_q)
    over = torch.relu(outputs_q - target_e)
    loss = q * under + (1.0 - q) * over     # [B, T, V, Q]
    loss = loss.sum(dim=-1)                  # [B, T, V]
    return loss.mean()


def _median_quantile_index(quantiles):
    qs = list(quantiles)
    if 0.5 in qs:
        return qs.index(0.5)
    # Fall back to the closest quantile to 0.5.
    return min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))


@torch.no_grad()
def _evaluate_aquacast_loader(model, testloader, args, desc="AquaCast eval"):
    """
    Evaluate one AquaCast loader and return detailed MSE statistics.

    In quantile mode (args.aquacast_quantile=True), additionally accumulates
    the full per-quantile predictions and ground truth (restricted to the
    client's owned columns) and computes the full quantile-metric suite from
    utils/metrics.py. The extra metrics are returned under
    the 'quantile_metrics' key.
    """
    criterion = torch.nn.MSELoss()
    owned_cols = _get_owned_cols(testloader)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_points = 0
    node_sse = None

    f_dim = -1 if getattr(args, 'aquacast_features', 'M') == 'MS' else 0
    use_quantile = bool(getattr(args, 'aquacast_quantile', False))
    quantiles = _aquacast_quantiles(args) if use_quantile else None
    median_idx = _median_quantile_index(quantiles) if use_quantile else None

    pred_q_chunks: list = [] if use_quantile else None
    target_chunks: list = [] if use_quantile else None

    for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(testloader, desc=desc):
        batch_x, exo_future = _split_aquacast_inputs(batch_x, args)
        batch_x = batch_x.float().to(args.device)
        batch_y = batch_y.float().to(args.device)
        if exo_future is not None:
            exo_future = exo_future.float().to(args.device)

        raw_outputs = model(batch_x, exo_future) if exo_future is not None else model(batch_x)
        if use_quantile:
            outputs_q = raw_outputs[:, -args.pred_len:, f_dim:, :]
            outputs = outputs_q[..., median_idx]
        else:
            outputs = raw_outputs[:, -args.pred_len:, f_dim:]
        batch_y = _prepare_aquacast_target(batch_y, outputs.shape[-1], args)

        if owned_cols is not None:
            outputs_eval = outputs[:, :, owned_cols]
            batch_y_eval = batch_y[:, :, owned_cols]
        else:
            outputs_eval = outputs
            batch_y_eval = batch_y

        loss = criterion(outputs_eval, batch_y_eval)
        sq_err = (outputs_eval - batch_y_eval).pow(2)

        if node_sse is None:
            node_sse = torch.zeros(sq_err.shape[-1], dtype=torch.float64)
        node_sse += sq_err.sum(dim=(0, 1)).detach().cpu().to(torch.float64)

        bs = batch_x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        total_points += sq_err.shape[0] * sq_err.shape[1]

        if use_quantile:
            if owned_cols is not None:
                pred_q_eval = outputs_q[:, :, owned_cols, :]
            else:
                pred_q_eval = outputs_q
            pred_q_chunks.append(pred_q_eval.detach().cpu().to(torch.float32).numpy())
            target_chunks.append(batch_y_eval.detach().cpu().to(torch.float32).numpy())

    if node_sse is None:
        raise ValueError("The test split has no full batches; check the CSV length and batch size.")
    avg_loss = total_loss / max(1, total_samples)
    node_mse = node_sse / max(1, total_points)
    model.train()

    quantile_metrics = None
    if use_quantile and pred_q_chunks:
        from federated_learning.quantile_metrics import compute_quantile_metrics
        pred_q_all = np.concatenate(pred_q_chunks, axis=0)
        target_all = np.concatenate(target_chunks, axis=0)
        quantile_metrics = compute_quantile_metrics(pred_q_all, target_all, quantiles)

    return {
        "loss": avg_loss,
        "node_mse": node_mse,
        "max_node_mse": float(node_mse.max().item()) if node_mse.numel() > 0 else float("nan"),
        "quantile_metrics": quantile_metrics,
    }


def train_aquacast(model, optimizer, trainloader, epochs: int, args, theta=None):
    """
    Train AquaCast with pinball loss restricted to the owned sensors.

    The trainloader's batch_y is already rearranged to the n_pred_vars-dim
    model output space (exo column removed, target kept) by the
    NarrowClientDataset. The loss is computed only over the client's
    owned variable columns (or all columns if owned_cols is None).
    """
    criterion = torch.nn.MSELoss()
    owned_cols = _get_owned_cols(trainloader)  # None for non-partitioned loaders
    use_quantile = bool(getattr(args, 'aquacast_quantile', False))
    quantiles = _aquacast_quantiles(args) if use_quantile else None
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        epoch_t0 = time.perf_counter()
        batch_count = 0
        batch_compute_seconds = 0.0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(trainloader, desc=f"AquaCast Epoch {epoch+1}"):
            batch_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            batch_x, exo_future = _split_aquacast_inputs(batch_x, args)
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            if exo_future is not None:
                exo_future = exo_future.float().to(args.device)

            outputs = model(batch_x, exo_future) if exo_future is not None else model(batch_x)
            f_dim = -1 if getattr(args, 'aquacast_features', 'M') == 'MS' else 0

            if use_quantile:
                # outputs: [B, T, V, Q]
                outputs = outputs[:, -args.pred_len:, f_dim:, :]
                n_pred_var = outputs.shape[-2]
                target = _prepare_aquacast_target(batch_y, n_pred_var, args)
                loss = _pinball_loss(outputs, target, quantiles, owned_cols=owned_cols)
            else:
                outputs = outputs[:, -args.pred_len:, f_dim:]
                target = _prepare_aquacast_target(batch_y, outputs.shape[-1], args)
                if owned_cols is not None:
                    # Masked loss: only on this client's owned variable columns.
                    loss = criterion(outputs[:, :, owned_cols], target[:, :, owned_cols])
                else:
                    loss = criterion(outputs, target)

            loss.backward()

            if args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            batch_count += 1
            batch_compute_seconds += time.perf_counter() - batch_t0

        epoch_dt = time.perf_counter() - epoch_t0
        avg_loss = total_loss / max(1, total_samples)
        avg_batch_seconds = batch_compute_seconds / max(1, batch_count)
        print(f"AquaCast Epoch {epoch+1}: train loss {avg_loss:.6f}", flush=True)
        print(
            f"[TIMING] stage=train_epoch epoch={epoch+1} num_batches={batch_count} "
            f"num_samples={total_samples} epoch_seconds={epoch_dt:.4f} "
            f"avg_batch_seconds={avg_batch_seconds:.4f}",
            flush=True,
        )


@torch.no_grad()
def test_aquacast(model, testloader, args):
    """
    Evaluate AquaCast.  Returns (mse_loss, mse_loss) to match the (loss, metric) API.

    The testloader uses the raw (non-zero-padded) dataset.  The evaluation
    replicates the original central AquaCast logic (exp_main_exo.py vali/test):
      - slice outputs and batch_y to [-pred_len:, f_dim:]
      - when features='M' and exo=True, remove the exo column from batch_y
        (second-to-last) and keep the target column (last), so MSE is computed
        over the same columns the model is trained to predict.
    """
    stats = _evaluate_aquacast_loader(model, testloader, args, desc="AquaCast eval")
    return stats["loss"], stats["loss"]  # (loss, metric) – report MSE for both


@torch.no_grad()
def test_aquacast_partitioned(model, testloaders, args):
    """
    Evaluate AquaCast on each client partition separately, then average client
    MSEs and report the worst per-node MSE across all partitions.
    """
    if not isinstance(testloaders, list):
        raise ValueError("Partitioned AquaCast evaluation expects a list of testloaders.")

    partition_stats = []
    for idx, loader in enumerate(testloaders):
        partition_stats.append(
            _evaluate_aquacast_loader(model, loader, args, desc=f"AquaCast eval client {idx}")
        )

    client_mse = [float(stats["loss"]) for stats in partition_stats]
    avg_client_mse = float(np.mean(client_mse))
    max_node_mse = float(max(stats["max_node_mse"] for stats in partition_stats))

    metrics = {
        "accuracy": avg_client_mse,
        "avg_client_mse": avg_client_mse,
        "max_node_mse": max_node_mse,
        "client_mse": client_mse,
    }

    use_quantile = bool(getattr(args, 'aquacast_quantile', False))
    if use_quantile:
        from federated_learning.quantile_metrics import SCALAR_METRIC_KEYS
        per_client_q = [s.get("quantile_metrics") for s in partition_stats]
        per_client_q = [m for m in per_client_q if m is not None]
        if per_client_q:
            for key in SCALAR_METRIC_KEYS:
                vals = [m[key] for m in per_client_q]
                # nan-safe mean: ignore non-finite values when averaging.
                finite = [v for v in vals if np.isfinite(v)]
                metrics[f"avg_client_{key}"] = float(np.mean(finite)) if finite else float("nan")
                metrics[f"client_{key}"] = [float(v) for v in vals]

    return avg_client_mse, metrics


