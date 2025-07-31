"""
neversim
========
High-level interface to the NETSIM + Nevergrad optimisation toolkit.

Typical usage
-------------
>>> from neversim import Optimizer, Metric
>>> opt = Optimizer(
...     params_file="default.parameters",
...     param_space={"ge": ("log", 2e-10, 1e-9),
...                  "gi": ("log", 2e-9, 1e-8)},
...     metrics=[Metric("cv", target=1.0),
...              Metric("fr", target=6.0)],
...     budget=400,
... )
>>> history = opt.run()

The same functionality is also exposed by the CLI installed with the
package:  ``neversim optimize ...``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Version handling – use PEP 621 metadata if the package is installed,
# fall back to a dev tag when running from a checkout.
# ---------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version  # Python ≥3.8
except ImportError:  # pragma: no cover
    from importlib_metadata import version as _pkg_version  # type: ignore

try:
    __version__: str = _pkg_version("neversim")
except Exception:  # package not installed or metadata missing
    __version__ = "0.0.0+dev"


# ---------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------
# Optimiser orchestration
from .optimisation import Optimizer                                  # noqa: E402
# Metric registry utilities
from .metrics import Metric, register_metric, available_metrics      # noqa: E402

__all__ = [
    "__version__",
    # core classes
    "Optimizer",
    "Metric",
    # metric registry helpers
    "register_metric",
    "available_metrics",
    # convenience wrapper
    "optimize",
]


# ---------------------------------------------------------------------
# Thin convenience wrapper so users can do `neversim.optimize(**kwargs)`
# instead of constructing the Optimizer themselves.
# ---------------------------------------------------------------------
def optimize(**kwargs):
    """
    Convenience function that instantiates ``Optimizer`` with *kwargs*
    and immediately runs it.

    Parameters
    ----------
    **kwargs
        Forwarded verbatim to :class:`neversim.optimisation.Optimizer`.

    Returns
    -------
    pandas.DataFrame
        History of the optimisation (one row per evaluation).
    """
    opt = Optimizer(**kwargs)
    return opt.run()