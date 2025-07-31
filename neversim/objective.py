"""
neversim.objective
==================

“Objective function” layer that the Optimizer submits to the worker pool:

    1. Runs one NETSIM simulation.
    2. Evaluates every Metric the user supplied.
    3. Sums the weighted squared-error terms into a single loss value.

Nothing is cached or pre-computed here; each Metric implementation is
fully responsible for whatever statistics it needs and may expose *all*
its tunables through the Metric.kwargs mechanism.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import pandas as pd

from .config import RuntimeConfig
from .metrics import Metric
from .simulation import run_simulation, SimulationResult


# ----------------------------------------------------------------------
# Dataclass returned by `evaluate_candidate`
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ObjectiveResult:
    params: Dict[str, float]                 # parameters explored
    loss: float                              # scalar fed to Nevergrad
    metric_values: Dict[str, float]          # raw metric outputs
    status: int                              # copy of SimulationResult.status
    error_msg: Optional[str] = None          # None on success

    # Convenience ------------------------------------------------------
    def to_series(self) -> pd.Series:
        data: Dict[str, Any] = {
            **{f"p_{k}": v for k, v in self.params.items()},
            **self.metric_values,
            "loss": self.loss,
            "status": self.status,
        }
        if self.error_msg:
            data["error"] = self.error_msg
        return pd.Series(data)


# ----------------------------------------------------------------------
# Driver – called by the Optimizer
# ----------------------------------------------------------------------
def evaluate_candidate(
    params_file: str,
    param_updates: Dict[str, float],
    metrics: List[Metric],
    cfg: RuntimeConfig,
    *,
    netsim_dir: str | None = None,
) -> ObjectiveResult:
    """
    Run one simulation + metric evaluation; return ObjectiveResult.

    This function is purposely small so it can be submitted directly to a
    ProcessPoolExecutor without additional pickling headaches.
    """
    # ------------------------------------------------ simulation ------
    sim: SimulationResult = run_simulation(
        params_file=params_file,
        param_updates=param_updates,
        cfg=cfg,
        netsim_dir=netsim_dir,
    )

    if not sim.ok():
        return ObjectiveResult(
            params=param_updates,
            loss=math.inf,
            metric_values={},
            status=sim.status,
            error_msg=sim.error_msg,
        )

    # Unified stats structure handed to Metric.compute -----------------
    # The only guaranteed key is "spike_data"; Metric functions may add
    # derived arrays to this dict so later metrics can reuse them.
    stats: Dict[str, Any] = {"spike_data": sim.spike_data}

    metric_vals: Dict[str, float] = {}
    total_loss = 0.0
    for m in metrics:
        try:
            val = m.compute(stats)     # may mutate stats
            metric_vals[m.name] = val
            m_loss = m.loss(stats)     # uses the same value internally
            total_loss += m_loss
        except Exception as exc:       # metric blew up -> invalidate point
            logging.warning(
                "Metric '%s' raised %s for params %s",
                m.name, exc, param_updates
            )
            return ObjectiveResult(
                params=param_updates,
                loss=math.inf,
                metric_values={},
                status=-4,
                error_msg=f"metric {m.name}: {exc}",
            )

    if not math.isfinite(total_loss):
        total_loss = math.inf

    return ObjectiveResult(
        params=param_updates,
        loss=total_loss,
        metric_values=metric_vals,
        status=sim.status,
    )


# ----------------------------------------------------------------------
# Convenience shortcut
# ----------------------------------------------------------------------
__all__ = ["ObjectiveResult", "evaluate_candidate"]