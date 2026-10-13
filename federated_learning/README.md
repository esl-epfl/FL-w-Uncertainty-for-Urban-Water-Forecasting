# Federated learning

FedAvg experiments for the paper, reusing the repository's AquaCast model,
data loader, and metrics. Includes 16 equal-split runs (1, 10, 50, 100 clients)
and four Zipf-1 runs (10 clients), each with 100 rounds and one local epoch.

Place these external CSVs in `new_data/` at the repository root:

| Setting | File | Sensors |
| --- | --- | ---: |
| S1 | `synthesized_v2_1_1.csv` | 100 |
| S2 | `synthesized_v2_2_1.csv` | 100 |
| S3 | `synthesized_v2_1_1_1000.csv` | 1000 |
| S4 | `synthesized_v2_2_1_1000.csv` | 1000 |

Each CSV needs `date`, sensor columns including `node_8711_output`, and a final
`rain` column. Data is not included.

After installing the main requirements, run from the repository root:

```bash
pip install -r federated_learning/requirements.txt
python -m federated_learning.run --sweep all --data-root new_data/
python -m federated_learning.plot_results
```

Use `--sweep homogeneous` or `--sweep heterogeneous` for either group, or
`--config` with a JSON file from `federated_learning/configs/` for one experiment.
Shared settings are in `federated_learning/configs/base.json`. GPUs are detected automatically; `--device cpu`
selects CPU execution.

Logs and metrics are saved under `federated_learning/results/`; PDFs for
Figures 2, 3, and 5 and CSV summaries are written to its `plots/` subfolder.
