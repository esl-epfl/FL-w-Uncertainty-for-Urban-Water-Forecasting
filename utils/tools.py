import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from numpy import trapz

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

# def visual_rain(true, preds, rain, error_name=None, error=None, name='./pic/test.pdf', rain_name='Rain'):
#     """
#     Results visualization with consistent font sizes and in-frame legend.
#     """
#     if error is None or error_name is None:
#         pred_err = "Prediction"
#     else:
#         pred_err = f"Prediction {error_name}:{error:.4f}"

#     fig, ax1 = plt.subplots(figsize=[15, 6])

#     # Plot water height predictions
#     ax1.plot(preds, label=pred_err, color='green', linestyle='--')
#     ax1.plot(true, label='Ground Truth', color='orange')
#     ax1.set_xlabel('Time Step', fontsize=18)
#     ax1.set_ylabel('Water Height', fontsize=18)
#     ax1.tick_params(axis='both', labelsize=18)  # Apply fontsize to both x and y axes

#     handles, labels = ax1.get_legend_handles_labels()

#     # Plot rain if provided
#     if rain is not None:
#         ax2 = ax1.twinx()
#         ax2.set_ylabel('Precipitation', color='tab:blue', fontsize=18)
#         ax2.fill_between(np.arange(len(rain)), rain, alpha=0.3, label=rain_name, color='tab:blue')
#         ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=18)

#         # Add a patch for rain to combined legend
#         rain_patch = plt.Line2D([0], [0], color='tab:blue', alpha=0.3, lw=10, label=rain_name)
#         handles.append(rain_patch)
#         labels.append(rain_name)

#     # Add legend inside the plot frame
#     ax1.legend(handles, labels, fontsize=18, loc='best', frameon=True)

#     plt.tight_layout()
#     plt.savefig(name, bbox_inches='tight')
#     plt.close()

def visual_rain(true, preds, rain, error_name=None, error=None,
                name='./pic/test.pdf', rain_name='Rain', quantiles=None):
    """
    Results visualization with consistent font sizes and in-frame legend.
    Supports both point predictions and quantile forecast outputs.

    Adds:
    - Vertical "Now" line at timestep 96 (history length boundary)
    - Larger font sizes for ax1 and ax2
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.lines as mlines

    is_quantile = quantiles is not None and np.array(preds).ndim == 2

    # ---- Config ----
    history_len = 96          # vertical split point
    main_font = 22            # axis label font size
    tick_font = 20            # tick font size
    legend_font = 18          # legend font size
    now_line_width = 2.0

    # ---- Determine prediction label ----
    if is_quantile:
        pred_label = "Median Forecast (q=0.50)"
    elif error is None or error_name is None:
        pred_label = "Prediction"
    else:
        pred_label = f"Prediction {error_name}:{error:.4f}"

    fig, ax1 = plt.subplots(figsize=[15, 6])

    # x-axis for plotting
    if is_quantile:
        preds = np.array(preds)   # [T, n_quantiles]
        quantiles = list(quantiles)
        T = preds.shape[0]
        x = np.arange(T)

        # ---- Sort quantiles ----
        sorted_idx = np.argsort(quantiles)
        sorted_q = [quantiles[i] for i in sorted_idx]
        sorted_p = preds[:, sorted_idx]

        # ---- Confidence interval shading ----
        lo_indices = [i for i, q in enumerate(sorted_q) if q < 0.5]
        hi_indices = [i for i, q in enumerate(sorted_q) if q > 0.5]

        pairs = list(zip(lo_indices, reversed(hi_indices)))
        n_pairs = len(pairs)

        for k, (li, hi) in enumerate(pairs):
            alpha = 0.15 + 0.20 * (k / max(n_pairs - 1, 1))

            q_lo, q_hi = sorted_q[li], sorted_q[hi]

            ax1.fill_between(
                x,
                sorted_p[:, li],
                sorted_p[:, hi],
                alpha=alpha,
                color='green',
                label=f"CI [{q_lo:.2f}–{q_hi:.2f}]"
            )

        # ---- Median line ----
        if 0.5 in sorted_q:
            median_col = sorted_q.index(0.5)
        else:
            median_col = len(sorted_q) // 2

        ax1.plot(
            x,
            sorted_p[:, median_col],
            label=pred_label,
            color='green',
            linestyle='--',
            linewidth=2
        )

    else:
        x = np.arange(len(preds))

        ax1.plot(
            x,
            preds,
            label=pred_label,
            color='green',
            linestyle='--',
            linewidth=2
        )

    # ---- Ground truth ----
    ax1.plot(
        np.arange(len(true)),
        true,
        label='Ground Truth',
        color='crimson',
        linewidth=2
    )

    # ---- Vertical "Now" line ----
    ax1.axvline(
        x=history_len,
        color='black',
        linestyle=':',
        linewidth=now_line_width,
        label='Now'
    )

    ax1.text(
        history_len + 1,
        ax1.get_ylim()[1] * 0.95,
        "Now",
        fontsize=18,
        color='black',
        verticalalignment='top'
    )

    # ---- Axis styling ----
    ax1.set_xlabel('Time Step', fontsize=main_font)
    ax1.set_ylabel('Water Height', fontsize=main_font)
    ax1.tick_params(axis='both', labelsize=tick_font)

    handles, labels = ax1.get_legend_handles_labels()

    # ---- Rain overlay ----
    if rain is not None:
        ax2 = ax1.twinx()

        ax2.set_ylabel(
            'Precipitation',
            color='tab:blue',
            fontsize=main_font
        )

        ax2.fill_between(
            np.arange(len(rain)),
            rain,
            alpha=0.3,
            label=rain_name,
            color='tab:blue'
        )

        ax2.tick_params(
            axis='y',
            labelcolor='tab:blue',
            labelsize=tick_font
        )

        rain_patch = mlines.Line2D(
            [0], [0],
            color='tab:blue',
            alpha=0.3,
            lw=10,
            label=rain_name
        )

        handles.append(rain_patch)
        labels.append(rain_name)

    # ---- Legend ----
    ax1.legend(
        handles,
        labels,
        fontsize=legend_font,
        loc='best',
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {name}")


def visual_plot(train, valid, test, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(train, label='Train', linewidth=2)
    plt.plot(valid, label='Valid', linewidth=2)
    plt.plot(test, label='Test', linewidth=1)
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.savefig(name, bbox_inches='tight')

def visual_acc(mse_threshold, acc_i, path):
    # AUC Calculation
    auc = trapz(acc_i, mse_threshold)
    auc_normalized = auc / (mse_threshold[-1] - mse_threshold[0])
    # Plot
    plt.figure() 
    plt.plot(mse_threshold, acc_i)
    plt.ylabel('Accuracy')
    plt.xlabel('Error Threshold')
    plt.title(f"AUC (raw): {auc:.4f}, AUC (normalized): {auc_normalized:.4f}")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return auc, auc_normalized

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# ── Reproducibility : Visual Rain Quantiles ───────────────────────────────────────────────────────────
np.random.seed(42)
T = 96

# ── Ground truth: smooth sinusoidal water-height signal ───────────────────────
t = np.linspace(0, 4 * np.pi, T)
true = 2.5 * np.sin(t) + 0.5 * np.sin(3 * t) + 5.0

# ── Quantile forecasts ────────────────────────────────────────────────────────
# Median tracks the truth with a small lag + bias
median = 2.4 * np.sin(t - 0.15) + 0.48 * np.sin(3 * t - 0.1) + 5.05
median += np.random.normal(0, 0.08, T)          # tiny noise on median

# Spread grows slightly in the second half (heteroscedastic uncertainty)
spread = np.linspace(0.2, 0.6, T)

quantile_preds = np.column_stack([
    median - 2.0 * spread,   # q05
    median - 1.2 * spread,   # q25
    median,                  # q50
    median + 1.2 * spread,   # q75
    median + 2.0 * spread,   # q95
])

# ── Precipitation: sporadic rain events ───────────────────────────────────────
rain = np.zeros(T)
rain[10:18]  = np.random.uniform(2, 5,  8)
rain[35:42]  = np.random.uniform(4, 9,  7)
rain[60:70]  = np.random.uniform(1, 6, 10)
rain[82:90]  = np.random.uniform(3, 7,  8)

# ── Run visualization ─────────────────────────────────────────────────────────
visual_rain(
    true           = true,
    preds          = quantile_preds,
    rain           = rain,
    error_name     = 'CRPS',
    error          = 0.0318,
    name           = './test_quantile_96.pdf',
    rain_name      = 'Precipitation (mm)',
    quantiles      = [0.05, 0.25, 0.5, 0.75, 0.95],
)

print("Plot saved to ./test_quantile_96.pdf")