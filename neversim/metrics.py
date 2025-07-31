"""
neversim.metrics
================

Registry + helper dataclass turning arbitrary spike-train statistics into
Nevergrad loss contributions.

Key points
----------
1.  Register a metric function with @register_metric("name").
2.  Create Metric("name", target, weight, kwargs) objects for optimisation.
3.  Heavy `pyNetsim.utility` imports are delayed until first use, so this
    module imports even when pyNetsim is not yet on PYTHONPATH.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List

import numpy as np

# ----------------------------------------------------------------------
# Internal registry
# ----------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[[Dict[str, Any]], float]] = {}


def register_metric(name: str) -> Callable[[Callable], Callable]:
    """Decorator: store *func* in the global metric registry under *name*."""
    key = name.lower()

    def _decorator(func: Callable[[Dict[str, Any]], float]) -> Callable:
        _REGISTRY[key] = func
        return func

    return _decorator


def available_metrics() -> List[str]:
    """Alphabetically sorted list of all registered metric names."""
    return sorted(_REGISTRY.keys())


# ----------------------------------------------------------------------
# Metric dataclass
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Metric:
    """
    Bundle a registered compute function with optimisation targets.

    Parameters
    ----------
    name      : registry key
    target    : desired value
    weight    : multiplicative weight on squared error
    kwargs    : forwarded to the compute function (exposes tunables)
    """
    name: str
    target: float
    weight: float = 1.0
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # ------------- helpers -------------------------------------------
    def compute(self, stats: Dict[str, Any]) -> float:
        key = self.name.lower()
        if key not in _REGISTRY:
            raise KeyError(
                f"Metric '{self.name}' is not registered. "
                f"Available: {', '.join(available_metrics())}"
            )
        return _REGISTRY[key](stats, **self.kwargs)

    def loss(self, stats: Dict[str, Any]) -> float:
        value = self.compute(stats)
        if math.isnan(value):
            return float("inf")
        scale = self.kwargs.get("scale", abs(self.target) if self.target else 1.0)
        return self.weight * ((value - self.target) / scale) ** 2


# ----------------------------------------------------------------------
# Shared cache helper (expensive statistics computed once)
# ----------------------------------------------------------------------
def _ensure_basic_stats(
    stats: Dict[str, Any],
    *,
    bin_size: int = 50,
    subsample_pct: float = 0.001,
    dt: float = 1e-4,
) -> None:
    """
    Populate stats[...] with firing-rates, CVs, pairwise correlations.

    Runs exactly once per SimulationResult and stores the arrays so other
    metrics can reuse them.
    """
    if "frs_all" in stats:        # already done
        return

    # Lazy import – only when first metric actually needs these utilities
    from pyNetsim.utility import (
        firing_rate,
        coefficients_of_variation,
        pairwise_spike_correlation,
    )

    spike_data = stats["spike_data"]
    spk_ids = spike_data["spk_ids"]
    spk_times = spike_data["spk_times"]

    total_N = int(spk_ids.max() + 1) if spk_ids.size else 0
    exc_N = int(total_N * 0.8)
    exc_ids = np.arange(exc_N)

    # Firing rate
    T = float(np.max(spk_times)) if spk_times.size else 1.0
    frs_all = firing_rate(spike_data, T)
    stats["frs_all"] = frs_all
    stats["frs_exc"] = frs_all[:exc_N]

    # CV
    cvs_all = coefficients_of_variation(spike_data)
    stats["cvs_all"] = cvs_all
    stats["cvs_exc"] = cvs_all[:exc_N]

    # Pairwise correlations
    stats["corr_exc"] = pairwise_spike_correlation(
        spike_data,
        bin_size=bin_size,
        subsample_pct=subsample_pct,
        dt=dt,
        neuron_ids_to_include=exc_ids,
    )
    stats["corr_all"] = pairwise_spike_correlation(
        spike_data,
        bin_size=bin_size,
        subsample_pct=subsample_pct,
        dt=dt,
        neuron_ids_to_include=None,
    )


# ----------------------------------------------------------------------
# Built-in convenience metrics (override / extend as you like)
# ----------------------------------------------------------------------
@register_metric("fr")
def _fr(stats, *, subset: str = "exc", reducer: str = "median", **kwargs):
    """
    Firing-rate statistic.

    subset  : "exc" or "all"
    reducer : "median" (default) or "mean"
    """
    _ensure_basic_stats(stats, **kwargs)
    data = stats[f"frs_{subset}"]
    func = np.nanmean if reducer == "mean" else np.nanmedian
    return float(func(data))


@register_metric("cv")
def _cv(stats, *, subset: str = "exc", reducer: str = "median", **kwargs):
    _ensure_basic_stats(stats, **kwargs)
    data = stats[f"cvs_{subset}"]
    func = np.nanmean if reducer == "mean" else np.nanmedian
    return float(func(data))


@register_metric("corr_median")
def _corr_median(stats, *, subset: str = "exc", **kwargs):
    _ensure_basic_stats(stats, **kwargs)
    return float(np.nanmedian(stats[f"corr_{subset}"]))


@register_metric("corr_std")
def _corr_std(stats, *, subset: str = "exc", **kwargs):
    _ensure_basic_stats(stats, **kwargs)
    return float(np.nanstd(stats[f"corr_{subset}"]))


@register_metric("corr_skew")
def _corr_skew(stats, *, subset: str = "exc", **kwargs):
    _ensure_basic_stats(stats, **kwargs)
    from scipy.stats import skew  # local import (optional dep)
    data = stats[f"corr_{subset}"]
    return float(skew(data)) if len(data) else float("nan")