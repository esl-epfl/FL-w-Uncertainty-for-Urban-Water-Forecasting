# Federated Learning with Uncertainty for Urban Water Dynamics

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

## Dataset Style
Dataset structure for time-series forecasting in CSV columns' format

1. only endogenous time-series

    date, node1, node2, ...

2. Including endogenous time-series history

    date, node1, node2, ..., rain

3. using perfect exogenous forecast

    a. AquaCast

        date, node1, node2, ..., rain

    b. PatchTST (rain_forecast_steps should be equal to history steps)
    
        date, node1, node2, ..., rain, rain_forecast_steps



## Federated learning

The paper's sensor-disjoint FedAvg experiments are in [`federated_learning/`](federated_learning/README.md), reusing the repository's AquaCast model, data loaders, and metrics. They include client scaling (1, 10, 50, 100 clients) and heterogeneous Zipf-1 sensor ownership.

```bash
pip install -r federated_learning/requirements.txt
python -m federated_learning.run --sweep all --data-root new_data/
```

See the [FL instructions](federated_learning/README.md) for the four required external CSVs, individual runs, and paper figures.

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST
