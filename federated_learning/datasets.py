"""Narrow sensor partitions from FederatedMonarch, using the shared CSV loader."""

import numpy as np
from torch.utils.data import DataLoader, Dataset

from data_provider.data_loader import Dataset_Custom_exo


class NarrowClientDataset(Dataset):
    """Map owned sensors to local slots, pad to max_owned, and append rain."""

    def __init__(self, base_ds, owned_raw_cols, exo_col, max_owned):
        self.base_ds = base_ds
        self.owned_raw_cols = np.asarray(owned_raw_cols, dtype=np.int64)
        self.exo_col = exo_col
        self.max_owned = max_owned
        self.n_owned = len(owned_raw_cols)
        self.owned_cols = list(range(self.n_owned))

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        (seq_x, exo_future), seq_y, seq_x_mark, seq_y_mark = self.base_ds[idx]
        x_pad = np.zeros((seq_x.shape[0], self.max_owned + 1), dtype=np.float32)
        x_pad[:, :self.n_owned] = seq_x[:, self.owned_raw_cols]
        x_pad[:, self.max_owned] = seq_x[:, self.exo_col]
        y_pad = np.zeros((seq_y.shape[0], self.max_owned), dtype=np.float32)
        y_pad[:, :self.n_owned] = seq_y[:, self.owned_raw_cols]
        return [x_pad, exo_future], y_pad, seq_x_mark, seq_y_mark


def load_aquacast_datasets(args, num_clients):
    """Keep the original 70/10/20 time split, scaling, and C=1 data path.

    Every client has the same time windows and a disjoint set of sensors.
    The shared loader moves the target last, leaving rain second to last.
    Larger federations use the original narrow, padded input construction;
    C=1 deliberately retains the original centralized column ordering.
    """
    splits = [
        Dataset_Custom_exo(
            root_path=args.aquacast_root_path,
            # The shared loader accepts a list of CSV paths.
            data_path=[args.aquacast_data_path],
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.aquacast_features,
            target=args.aquacast_target,
            timeenc=0,
            freq=args.aquacast_freq,
        )
        for flag in ('train', 'val', 'test')
    ]
    if any(len(ds) < args.batch_size for ds in splits):
        raise ValueError('Each time split must contain at least one full batch.')

    raw_width = splits[0].data_x.shape[1]
    n_sensors = raw_width - 1
    sizes = args.aquacast_partition_sizes
    if sizes is None:
        base, remainder = divmod(n_sensors, num_clients)
        sizes = [base + (i < remainder) for i in range(num_clients)]
    if (len(sizes) != num_clients or sum(sizes) != n_sensors
            or any(not isinstance(n, int) or n <= 0 for n in sizes)):
        raise ValueError('Partition sizes must be positive integers, one per client, summing to the sensor count.')
    max_owned = max(sizes)
    if args.enc_in != max_owned + 1:
        raise ValueError(f'enc_in={args.enc_in} must equal max(partition_sizes) + 1 = {max_owned + 1}.')

    def loader(dataset, training=False):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                          num_workers=1, drop_last=True)

    if num_clients == 1:
        return ([loader(splits[0], True)], [loader(splits[1])], [loader(splits[2])])

    exo_col = raw_width - 2
    sensor_cols = list(range(exo_col)) + [raw_width - 1]
    trainloaders, valloaders, testloaders = [], [], []
    start = 0
    for size in sizes:
        owned = sensor_cols[start:start + size]
        start += size
        client_splits = [NarrowClientDataset(ds, owned, exo_col, max_owned) for ds in splits]
        trainloaders.append(loader(client_splits[0], True))
        valloaders.append(loader(client_splits[1]))
        testloaders.append(loader(client_splits[2]))
    return trainloaders, valloaders, testloaders
