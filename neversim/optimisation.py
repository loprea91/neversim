"""
neversim.optimisation
=====================

High-level orchestration of

    • Nevergrad optimiser
    • worker pool  (ProcessPoolExecutor)
    • checkpoint / resume
    • CSV history + rudimentary live plotting hooks

The heavy lifting per candidate is delegated to
:func:`neversim.objective.evaluate_candidate`.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import pickle
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple, List, Any

import nevergrad as ng
import numpy as np
import pandas as pd
import os
import psutil

from .viz import LivePlotter
from .config import RuntimeConfig, new_run_directory
from .metrics import Metric
from .objective import evaluate_candidate, ObjectiveResult

# ----------------------------------------------------------------------
# Helper — build Nevergrad parametrization from user spec
# ----------------------------------------------------------------------
def _build_parametrization(
    space: Dict[str, Tuple],
    seed: int,
) -> ng.p.Instrumentation:

    kwargs: Dict[str, Any] = {}
    for name, spec in space.items():
        ptype, *bounds = spec
        ptype = ptype.lower()
        if ptype == "log":
            kwargs[name] = ng.p.Log(lower=bounds[0], upper=bounds[1])
        elif ptype == "float":
            kwargs[name] = ng.p.Scalar(lower=bounds[0], upper=bounds[1])
        elif ptype == "int":
            kwargs[name] = ng.p.Scalar(lower=bounds[0], upper=bounds[1]).set_integer_casting()
        else:
            raise ValueError(f"Unknown parameter type '{ptype}' for {name}")
    inst = ng.p.Instrumentation(**kwargs)
    inst.random_state.seed(seed)
    return inst

def _workers_memory_gb(self) -> float:
    """Total RSS (GB) of all active worker children."""
    import psutil, os
    me = psutil.Process(os.getpid())
    total = 0.0
    for child in me.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass  # worker already finished
    return total / 2**30


# ----------------------------------------------------------------------
# Optimizer class ------------------------------------------------------
# ----------------------------------------------------------------------
class Optimizer:
    """
    Drive a Nevergrad search while taking care of parallel workers,
    checkpoints, history CSV and basic logging.
    """

    # -------------------- construction -------------------------------
    def __init__(
        self,
        *,
        params_file: str,
        param_space: Dict[str, Tuple],
        metrics: List[Metric],
        run_dir: str | Path | None = None,
        optimizer: str = "TwoPointsDE",
        budget: int = 200,
        num_workers: int = 1,
        seed: int = 42,
        netsim_dir: str | Path | None = None
    ):
        # 0.  run directory / runtime config --------------------------
        self.run_dir: Path = Path(run_dir) if run_dir else new_run_directory()
        self.cfg = RuntimeConfig(self.run_dir, seed=seed)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(self.cfg.log_file), logging.StreamHandler()],
        )
        self.params_file = str(Path(params_file).expanduser().resolve())
        self.metrics = metrics
        self.budget = budget
        self.num_workers = max(1, num_workers)
        self.seed = seed
        self.netsim_dir = str(netsim_dir) if netsim_dir else None

        # 1.  parametrization ----------------------------------------
        self.parametrization = _build_parametrization(param_space, seed)

        # 2.  instantiate / load Nevergrad optimiser -----------------
        self._build_or_resume_ng(optimizer)

        # 3.  history CSV --------------------------------------------
        self._init_history()

        # 4. Plotter
        self.plotter = LivePlotter(
            self.cfg.history_file,      # history_csv
            self.cfg.fig_dir,           # fig_dir
            refresh_every=self.cfg.plot_refresh,
            reduce="pca",
        )

        total_cores = os.cpu_count() or 1
        mem = psutil.virtual_memory()
        banner = f"""
                ── neversim Optimizer ───────────────────────────────────────────
                Run directory : {self.run_dir}
                Optimiser     : {optimizer}
                Budget        : {self.budget}
                Parallel      : {self.num_workers} workers  (machine has {total_cores})
                Memory        : {mem.available / 2**30:.1f} / {mem.total / 2**30:.1f} GB available
                Parameters    : {', '.join(sorted(param_space.keys()))}
                Metrics       : {', '.join(m.name for m in metrics)}
                ──────────────────────────────────────────────────────────────────
                """
        logging.info(banner.strip())

    # ---------------------------------------------------------------
    # private helpers
    # ---------------------------------------------------------------
    def _settings_fingerprint(self, optimiser_name: str) -> str:
        """
        Build a tiny hash so we can detect when the user changes optimiser
        type or parameter space but points to an existing run_dir.
        """
        h = hashlib.md5()
        h.update(optimiser_name.encode())
        h.update(repr(sorted(self.parametrization.kwargs)).encode())
        h.update(str(self.netsim_dir).encode())
        return h.hexdigest()

    def _build_or_resume_ng(self, optimiser_name: str) -> None:
        """
        Create a new Nevergrad optimiser or resume from checkpoint if
        compatible.
        """
        fp_file = self.run_dir / ".fingerprint"
        checkpoint = self.cfg.checkpoint_file

        want_fp = self._settings_fingerprint(optimiser_name)
        have_fp = fp_file.read_text().strip() if fp_file.exists() else None

        if checkpoint.exists() and have_fp == want_fp:
            logging.info("Resuming optimiser from checkpoint %s", checkpoint)
            with checkpoint.open("rb") as fh:
                self.ng = pickle.load(fh)  # type: ignore
            # Update runtime-mutable fields
            self.ng.parametrization = self.parametrization
            self.ng.budget = self.budget
            self.ng.num_workers = self.num_workers
        else:
            if checkpoint.exists():
                logging.warning(
                    "Checkpoint exists but settings changed; starting a fresh run."
                )
            opt_cls = getattr(ng.optimizers, optimiser_name, None)
            if opt_cls is None:
                raise ValueError(f"Nevergrad optimiser '{optimiser_name}' not found.")
            self.ng = opt_cls(
                parametrization=self.parametrization,
                budget=self.budget,
                num_workers=self.num_workers,
            )
            # ensure budget is at least the number of already asked points
            if self.budget < self.ng.num_ask:
                logging.warning(
                    "Requested budget (%d) is smaller than progress so far (%d). "
                    "Using %d instead.",
                    self.budget, self.ng.num_ask, self.ng.num_ask
                )
                self.budget = self.ng.num_ask
            self.ng.budget = self.budget
            fp_file.write_text(want_fp)

    def _init_history(self) -> None:
        """
        Create history CSV with header if empty; otherwise leave untouched.
        """
        self.history_file = self.cfg.history_file
        if self.history_file.exists() and self.history_file.stat().st_size:
            return
        header = [f"p_{n}" for n in sorted(self.parametrization.kwargs.keys())]
        header += [m.name for m in self.metrics]
        header += ["loss", "status", "error"]
        pd.DataFrame(columns=header).to_csv(self.history_file, index=False)

    def _append_history(self, res: ObjectiveResult) -> None:
        res.to_series().to_frame().T.to_csv(
            self.history_file, mode="a", header=False, index=False
        )

    def _checkpoint(self) -> None:
        """Serialise the current Nevergrad state to disk."""
        with self.cfg.checkpoint_file.open("wb") as fh:
            pickle.dump(self.ng, fh)
        logging.info("Checkpoint saved to %s (eval %d)",
                     self.cfg.checkpoint_file, self.ng.num_tell)

    # ---------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Launch / continue the optimisation loop; returns the full history.
        """
        logging.info("Starting optimisation at evaluation %d", self.ng.num_ask + 1)

        # Prepare partial function for workers
        job = functools.partial(
            evaluate_candidate,
            self.params_file,
            metrics=self.metrics,
            cfg=self.cfg,
            netsim_dir=self.netsim_dir,
        )

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                futures = {}

                # Kick-start up to num_workers jobs
                start_jobs = min(self.num_workers, self.budget - self.ng.num_ask)
                for _ in range(start_jobs):
                    cand = self.ng.ask()
                    fut = pool.submit(job, cand.kwargs)
                    futures[fut] = cand

                # Main loop ------------------------------------------
                while futures:
                    for fut in as_completed(list(futures.keys())):
                        cand = futures.pop(fut)
                        res: ObjectiveResult = fut.result()

                        self.ng.tell(cand, res.loss)
                        self._append_history(res)
                        self.plotter.update()

                        # logging every evaluation
                        metric_str = " | ".join(f"{k}={v:.3g}" for k, v in res.metric_values.items())
                        param_str = ", ".join(f"{k}={v:.2e}" for k, v in cand.kwargs.items())
                        logging.info(
                            "Eval %3d/%d: %s -> loss=%.4g | %s",
                            self.ng.num_tell, self.budget, param_str, res.loss, metric_str
                        )

                        # schedule next job if budget not exhausted
                        if self.ng.num_ask < self.budget:
                            nxt = self.ng.ask()
                            fut_nxt = pool.submit(job, nxt.kwargs)
                            futures[fut_nxt] = nxt

                        # periodic checkpoint
                        if (
                            self.ng.num_tell % self.cfg.checkpoint_every == 0
                            or self.ng.num_tell == self.budget
                        ):
                            self._checkpoint()
                            logging.info("Workers RSS memory: %.2f GB", self._workers_memory_gb())

                        break  # leave inner loop so as_completed() list is updated
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt - checkpointing and shutting down.")
        finally:
            self._checkpoint()

        logging.info("Optimisation finished - best candidate: %s", self.best_parameters())
        self.plotter.close()
        return pd.read_csv(self.history_file)

    # ---------------------------------------------------------------
    # convenience getters
    # ---------------------------------------------------------------
    def best_parameters(self) -> Dict[str, float]:
        """Return kwargs of the current best recommendation."""
        rec = self.ng.provide_recommendation()
        return rec.kwargs if rec.kwargs else {}


__all__ = ["Optimizer"]