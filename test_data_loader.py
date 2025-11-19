import os
import torch
from torch.utils.data import DataLoader

# import Dataset_Custom_real_forecast from data_loader
from data_provider.data_loader import Dataset_Custom_real_forecast


def build_datasets(root_path):
    """
    Build train / val / test datasets using your Dataset_Custom_real_forecast.
    Adjust file names to your actual CSV paths.
    """
    endo_path = "Hydrique_WWTP_total.csv"              # e.g. 'ETTh1.csv' but with [date, Node1, Node2, ...]
    exo_measured_path = "measure_timeseries_dt_measure.csv"      # columns: [time, value]
    exo_forecast_path = "measure_timeseries_dt_forecast.csv"      # columns: [issued_at, forecast_time, value]

    size = [96, 0, 24]  # [seq_len, label_len, pred_len] -> adjust to your setup

    train_set = Dataset_Custom_real_forecast(
        root_path=root_path,
        flag='train',
        size=size,
        features='M',                   # or 'S' / 'MS' depending on what you want
        endo_path=endo_path,
        exo_measured_path=exo_measured_path,
        exo_forecast_path=exo_forecast_path,
        target='STEP_Debit entrée total',                 # change to your actual target column
        scale=True,
        timeenc=0,
        freq='h'
    )

    val_set = Dataset_Custom_real_forecast(
        root_path=root_path,
        flag='val',
        size=size,
        features='M',
        endo_path=endo_path,
        exo_measured_path=exo_measured_path,
        exo_forecast_path=exo_forecast_path,
        target='STEP_Debit entrée total',
        scale=True,
        timeenc=0,
        freq='h'
    )

    test_set = Dataset_Custom_real_forecast(
        root_path=root_path,
        flag='test',
        size=size,
        features='M',
        endo_path=endo_path,
        exo_measured_path=exo_measured_path,
        exo_forecast_path=exo_forecast_path,
        target='STEP_Debit entrée total',
        scale=True,
        timeenc=0,
        freq='h'
    )

    return train_set, val_set, test_set


def build_dataloaders(train_set, val_set, test_set, batch_size=32, num_workers=0):
    """
    Wrap datasets into PyTorch DataLoaders.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader


def sanity_check_batch(test_set):
    """
    Take a single sample from the test set and print shapes + basic info.
    This is the quick 'unit test' for your Dataset.
    """
    from pprint import pprint

    idx = 0  # first sample
    (seq_x, exo_future_vals), seq_y, seq_x_mark, seq_y_mark = test_set[idx]

    print("=== Single sample from test_set[0] ===")
    print(f"seq_x shape        : {seq_x.shape}")          # expected [seq_len, feat_dim]
    print(f"seq_y shape        : {seq_y.shape}")          # expected [label_len+pred_len, feat_dim]
    print(f"seq_x_mark shape   : {seq_x_mark.shape}")     # expected [seq_len, time_feat_dim]
    print(f"seq_y_mark shape   : {seq_y_mark.shape}")     # expected [label_len+pred_len, time_feat_dim]")
    print(f"exo_future_vals shape: {exo_future_vals.shape}") # should be >= pred_len (with your current code)

    # Optionally: convert to torch tensors here if your model expects tensors
    seq_x_t = torch.tensor(seq_x, dtype=torch.float32)
    exo_future_t = torch.tensor(exo_future_vals, dtype=torch.float32)
    seq_y_t = torch.tensor(seq_y, dtype=torch.float32)

    print("\nExample torch dtypes:")
    print("seq_x_t     :", seq_x_t.dtype)
    print("exo_future_t:", exo_future_t.dtype)
    print("seq_y_t     :", seq_y_t.dtype)


def sanity_check_loader(test_loader):
    """
    Take one batch from the test_loader and print shapes.
    """
    (seq_x_batch, exo_future_batch), seq_y_batch, seq_x_mark_batch, seq_y_mark_batch = next(iter(test_loader))

    print("\n=== One batch from test_loader ===")
    print(f"seq_x_batch shape        : {seq_x_batch.shape}")        # [B, seq_len, feat_dim]
    print(f"exo_future_batch shape   : {exo_future_batch.shape}")   # [B, ?] or [B, pred_len] depending on your collate
    print(f"seq_y_batch shape        : {seq_y_batch.shape}")        # [B, label_len+pred_len, feat_dim]
    print(f"seq_x_mark_batch shape   : {seq_x_mark_batch.shape}")
    print(f"seq_y_mark_batch shape   : {seq_y_mark_batch.shape}")


if __name__ == "__main__":
    # Adjust this to the folder where your CSVs are stored
    root_path = "/home/abgo/Data/Hydrique_Lausanne/code/"

    train_set, val_set, test_set = build_datasets(root_path)
    train_loader, val_loader, test_loader = build_dataloaders(train_set, val_set, test_set,
                                                              batch_size=4, num_workers=0)

    # Check a single sample from the test set
    sanity_check_batch(test_set)

    # Check a full batch from the test loader
    sanity_check_loader(test_loader)
