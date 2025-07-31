"""
Centralised configuration and convenience helpers for *neversim*.

Everything that represents a *tunable but not-often-changed* constant
lives here so that the rest of the codebase can import it from a single
place.  Users can still override most entries at run-time via the CLI or
the Optimizer constructor.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# Default, rarely touched settings
# ---------------------------------------------------------------------
DEFAULT_MEMORY_LIMIT_MB: int = 15 * 1024           # limit for each worker
DEFAULT_CHECKPOINT_EVERY: int = 5                  # evaluations
DEFAULT_PLOT_REFRESH: int = 3                      # evaluations
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_RUN_ROOT: Path = Path("./runs").resolve()


# ---------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------
@dataclass(slots=True)
class RuntimeConfig:
    """
    A small *immutable* record that bundles all paths and housekeeping
    parameters for a single optimisation run.
    """
    run_dir: Path
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    plot_refresh: int = DEFAULT_PLOT_REFRESH
    seed: int = DEFAULT_RANDOM_SEED

    # Derived paths ----------------------------------------------------
    checkpoint_file: Path = field(init=False)
    history_file: Path = field(init=False)
    log_file: Path = field(init=False)
    fig_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.run_dir / "optimizer.pkl"
        self.history_file = self.run_dir / "history.csv"
        self.log_file = self.run_dir / "run.log"
        self.fig_dir = self.run_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self).items()}

    def to_json(self, file: Optional[Path] = None, *, indent: int = 2) -> str:
        data = json.dumps(self.to_dict(), indent=indent)
        if file is not None:
            file.write_text(data)
        return data


# ---------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------
def new_run_directory(name: str | None = None,
                       root: Path = DEFAULT_RUN_ROOT) -> Path:
    """
    Create a unique run directory under *root*.

    If *name* is given we return *root/name* (creating it if necessary).
    Otherwise we auto-generate a monotonically increasing “run_XXXX”.
    """
    root.mkdir(parents=True, exist_ok=True)
    if name:
        run_dir = root / name
        run_dir.mkdir(exist_ok=True)
        return run_dir

    # auto-generate
    for i in range(1, 10_000):
        candidate = root / f"run_{i:04d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError("Could not create a new run directory - too many attempts.")


# ---------------------------------------------------------------------
# Simple YAML/JSON loader for power users
# ---------------------------------------------------------------------
def load_config(path: str | os.PathLike) -> RuntimeConfig:
    """
    Load a previously saved configuration (YAML or JSON).
    """
    p = Path(path).expanduser()
    txt = p.read_text()
    if p.suffix in {".yaml", ".yml"}:
        import yaml  # lazy import
        data = yaml.safe_load(txt)
    else:
        data = json.loads(txt)
    # Paths were serialised as strings -> convert back
    data["run_dir"] = Path(data["run_dir"])
    return RuntimeConfig(**data)