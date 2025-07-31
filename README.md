# neversim: NETSIM + Nevergrad Optimisation Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`neversim` Python toolkit for performing parameter optimisation on [NETSIM]([https://www.netsim.com/](https://github.com/mullerlab/NETSIM)) models. It uses the gradient-free optimisation algorithms from the [Nevergrad](https://facebookresearch.github.io/nevergrad/) library to efficiently search complex parameter spaces.

---

## Core Features

* **Parallel Execution:** Runs multiple NETSIM simulations in parallel using a process pool to maximize the use of available CPU cores.
* **Checkpoint & Resume:** Automatically saves optimiser progress and can resume an interrupted run from the last checkpoint.
* **Extensible Metrics:** Custom metrics to guide the optimisation. Comes with built-in metrics for firing rate (`fr`), coefficient of variation (`cv`), and pairwise spike correlations.
* **Live Visualisation:** Generates and updates plots in real-time during the optimisation run, including parameter heatmaps and dimensionality reduction projections (PCA, t-SNE, UMAP).
* **Dual Interface:** Can be used as a simple command-line tool or as a Python library for more programmatic control.

---

## Installation

1.  **Prerequisites:**
    * Python 3.9+
    * A working installation of `pyNetsim`. `neversim` calls `pyNetsim`, so it must be installed and available in your `PYTHONPATH`.

2.  **Install `neversim`:**
    Clone the repository and install it in editable mode using `pip`. This is recommended as it allows you to easily modify the code.

    ```bash
    git clone <your-repo-url>
    cd neversim
    pip install -e .
    ```

    This will also install all required dependencies listed in `pyproject.toml`: `nevergrad`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.

---

## Usage

You can use `neversim` either as a Python library or via its command-line interface.

### As a Command-Line Tool (CLI)

The CLI is the easiest way to start an optimisation run. It exposes two main commands: `optimize` and `plot`.

#### `neversim optimize`

This command starts or resumes an optimisation.

**Example:**

```bash
neversim optimize run_01 \
    --params-file my_model/default.parameters \
    --netsim-dir /path/to/pyNetsim_package \
    --param ge log 2e-10 1e-9 \
    --param gi log 2e-9 1e-8 \
    --metric cv 1.0 --metric fr 6.0 \
    --optimizer TwoPointsDE --budget 500 --workers 16
```

**Key Arguments:**
* `run_folder`: A directory to store all outputs (logs, history, plots, checkpoints).
* `--params-file`: Path to the base NETSIM parameter file.
* `--netsim-dir`: Path to the directory containing the `pyNetsim` package.
* `--param`: Defines a parameter to search. Takes 4 values: `NAME TYPE LOWER_BOUND UPPER_BOUND`. Can be repeated for multiple parameters. `TYPE` can be `log`, `float`, or `int`.
* `--metric`: Defines a target metric. Takes at least 2 values: `NAME TARGET [WEIGHT]`. Can be repeated.
* `--optimizer`: The Nevergrad optimiser to use (e.g., `CMA`, `TwoPointsDE`, `RandomSearch`).
* `--budget`: The total number of simulations to run.
* `--workers`: The number of parallel simulations to run.

#### `neversim plot`

This command generates plots from a completed or in-progress run.

```bash
# Generate plots from the data in run_01
neversim plot run_01
```

### As a Python Library

Using `neversim` as a library gives you more control and is ideal for integration into larger workflows.

**Example:**

```python
from neversim import Optimizer, Metric

# 1. Define the parameter space to search
#    Format: { "param_name": (type, lower_bound, upper_bound) }
param_space = {
    "ge": ("log", 2e-10, 1e-9),
    "gi": ("log", 2e-9, 1e-8)
}

# 2. Define the target metrics
#    Format: Metric(name, target, weight)
metrics = [
    Metric("cv", target=1.0, weight=1.0),
    Metric("fr", target=6.0, weight=1.5)
]

# 3. Configure and create the Optimizer
opt = Optimizer(
    run_dir="./my_first_run",
    params_file="path/to/default.parameters",
    netsim_dir="/path/to/pyNetsim_package",
    param_space=param_space,
    metrics=metrics,
    optimizer="TwoPointsDE",
    budget=500,
    num_workers=16
)

# 4. Run the optimisation
#    This will block until the budget is exhausted or you interrupt it.
#    Progress is printed to the console.
history_df = opt.run()

# 5. Get the best parameters found
best_params = opt.best_parameters()
print("\nOptimization finished!")
print("Best parameters found:")
print(best_params)

# The full history is available in the returned pandas DataFrame
print("\nFull history:")
print(history_df.head())
```

---

## Extending `neversim`

### Adding a Custom Metric

It's easy to add your own custom metric.

1.  Open `neversim/metrics.py`.
2.  Write a Python function that takes a `stats` dictionary as input and returns a single float value.
3.  Decorate it with `@register_metric("your_metric_name")`.
4.  Use the shared `_ensure_basic_stats(stats)` helper if you need common statistics like firing rates or CVs.

**Example:**
Let's add a metric for the skewness of the excitatory firing rate distribution.

```python
# In neversim/metrics.py

# ... (imports)
from scipy.stats import skew # Make sure scipy is installed or add to pyproject.toml

# ... (other metrics)

@register_metric("fr_skew")
def _fr_skew(stats, *, subset: str = "exc", **kwargs):
    """
    Firing-rate skewness.
    """
    _ensure_basic_stats(stats, **kwargs) # ensures 'frs_exc' is computed
    data = stats[f"frs_{subset}"]
    # Remove NaNs before calculating skew
    valid_data = data[~np.isnan(data)]
    return float(skew(valid_data)) if len(valid_data) > 0 else float("nan")

```

You can now use `--metric fr_skew 0.5` in the CLI or `Metric("fr_skew", target=0.5)` in your Python code.
