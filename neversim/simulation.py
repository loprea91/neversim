"""
neversim.simulation
===================

Minimal wrapper around `pyNetsim.netsim` that performs *exactly one*
simulation and returns the raw spike data.

• No statistics are calculated here.
• No hidden parameters are introduced.
• Metric functions are fully responsible for any analysis;
  they receive the spike data and can expose **all** tunables to the user.
"""
from __future__ import annotations

import logging
import os
import resource
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .config import RuntimeConfig

# ---------------------------------------------------------------------
# Load NETISM
# ---------------------------------------------------------------------
_NETSIM_LOADER = {"ready": False}

def _ensure_netsim_loaded(netsim_dir: str | None):
    """Idempotent, per-process import of pyNetsim."""
    if _NETSIM_LOADER["ready"]:
        return
    import sys, importlib, logging
    if netsim_dir and str(netsim_dir) not in sys.path:
        sys.path.insert(0, str(netsim_dir))
    try:
        mod = importlib.import_module("pyNetsim.netsim")
    except ModuleNotFoundError as err:
        raise ImportError(
            "pyNetsim could not be imported; check --netsim-dir or PYTHONPATH"
        ) from err

    logging.getLogger("pyNetsim").setLevel(logging.INFO)
    _NETSIM_LOADER["mod"] = mod
    _NETSIM_LOADER["ready"] = True

# ---------------------------------------------------------------------
# Utility: per-process memory limit
# ---------------------------------------------------------------------
def _limit_memory(mb: int) -> None:
    if sys.platform.startswith(("linux", "darwin")):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mb * 2**20, mb * 2**20))
        except (ValueError, OSError) as exc:  # pragma: no cover
            logging.warning("Could not set memory limit: %s", exc)


# ---------------------------------------------------------------------
# Dataclass returned by run_simulation
# ---------------------------------------------------------------------
@dataclass(slots=True)
class SimulationResult:
    """
    Everything produced by a single NETSIM run.

    Attributes
    ----------
    params :
        dict of the parameter updates explored by Nevergrad.
    run_dir :
        Folder containing NETSIM output files.
    status :
        0  => success,
        >0 => non-zero NETSIM return code,
        <0 => Python-level exception captured.
    spike_data :
        Dict with numpy arrays 'spk_ids' and 'spk_times'.
    error_msg :
        Human-readable exception text if status < 0.
    """
    params: Dict[str, float]
    run_dir: Path
    status: int
    spike_data: Dict[str, np.ndarray] = field(default_factory=dict)
    error_msg: Optional[str] = None

    # Convenience ------------------------------------------------------
    def ok(self) -> bool:
        """Return True if the simulation finished without errors."""
        return self.status == 0 and self.error_msg is None


# ---------------------------------------------------------------------
# Helper to create a unique sub-folder for the run
# ---------------------------------------------------------------------
def _make_output_folder(root: Path, updates: Dict[str, float]) -> Path:
    tag = "_".join(f"{k}_{v:.3e}" for k, v in sorted(updates.items()))
    folder = root / "sim" / tag
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------------
# Main entry point used by the Optimizer
# ---------------------------------------------------------------------
def run_simulation(
    params_file: str | os.PathLike,
    param_updates: Dict[str, float],
    cfg: RuntimeConfig,
    *,
    netsim_dir: str | os.PathLike | None = None,
) -> SimulationResult:
    """
    Execute one NETSIM simulation with *param_updates* applied.
    """

    _ensure_netsim_loaded(str(netsim_dir) if netsim_dir else None)
    _netsim_mod = _NETSIM_LOADER["mod"]         # type: ignore                                     

    _limit_memory(cfg.memory_limit_mb)
    out_dir = _make_output_folder(cfg.run_dir, param_updates)

    result = SimulationResult(params=param_updates, run_dir=out_dir, status=-1)

    try:
        net = _netsim_mod.netsim(
            params=str(params_file),
            output_dir=str(out_dir),
            netsim_dir=str(netsim_dir) if netsim_dir else None,
            compile_on_run=False,
            verbose=False,
        )

        # Apply Nevergrad-proposed parameters
        for key, val in param_updates.items():
            net.set_param(key, val)
        net.write_paramfile(net.params)

        rc = net.run()
        result.status = rc.returncode

        if rc.returncode != 0:
            logging.info("NETSIM exited with code %s for %s", rc.returncode, param_updates)
            return result

        # ---------------- raw spike data ----------------
        raw = net.get_results()
        result.spike_data = {
            "spk_ids": np.asarray(raw["spikeids"]),
            "spk_times": np.asarray(raw["spiketimes"]),
        }

    except MemoryError:
        result.status = -2
        result.error_msg = "MemoryError"
    except Exception as exc:  # pragma: no cover
        result.status = -3
        result.error_msg = f"{type(exc).__name__}: {exc}"

    return result


__all__ = ["SimulationResult", "run_simulation"]