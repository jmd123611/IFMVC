# IFMVC

Implementation of IFMVC with an Mfeat reproduction example.

## Environment

Python 3.10 or newer is recommended. Install a PyTorch build suitable for your system, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, follow the official PyTorch installation instructions for the CUDA version supported by your machine.

## Data

Place the Mfeat MATLAB file at `data/mfeat.mat`. It must contain `X1` through `X6`.

## Run

Run one complete seed with the default settings:

```bash
python main.py --seeds 0
```

Run the five common reproduction seeds with:

```bash
python main.py --seeds 0 1 2 3 4
```

The default device is `cuda:0` when CUDA is available and CPU otherwise. Results are written to `results/`.

The learning rate is defined once in `main.py` as `1e-4`.

## Method details

Algorithm 1 uses KMeans to obtain view-specific partitions. Algorithm 2 computes coordinate-wise median representatives from these partitions and constructs the decision tree.

An empty cluster raises an error during center computation.

## Quick verification

Run the smoke test with:

```bash
python smoke_test.py
```
