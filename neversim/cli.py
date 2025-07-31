"""
Command-line interface for the *neversim* package.

Installation automatically exposes the console-script entry-point
`neversim`, so after `pip install -e .` you can do for example:

    $ neversim optimize run1 \
        --params-file default.parameters \
        --param ge log 2e-10 1e-9 \
        --param gi log 2e-9 1e-8 \
        --metric cv 1.0 --metric fr 6 \
        --optimizer TwoPointsDE --budget 500 --workers 16
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, Tuple, List

from . import Optimizer, Metric
from .viz import create_pairplots  # will be implemented later


# ----------------------------------------------------------------------
# Helpers for parsing repeated --param and --metric flags
# ----------------------------------------------------------------------
def _parse_param(flag_values: List[str]) -> Tuple[str, Tuple]:
    """
    Convert ["ge", "log", "2e-10", "1e-9"] -> ("ge", ("log", 2e-10, 1e-9))
    """
    if len(flag_values) != 4:
        raise argparse.ArgumentTypeError(
            "--param expects 4 values: NAME TYPE LOWER UPPER"
        )
    name, ptype, lo, hi = flag_values
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Bounds must be numeric: {err}") from err
    return name, (ptype, lo_f, hi_f)


def _parse_metric(flag_values: List[str]) -> Metric:
    """
    --metric NAME TARGET [WEIGHT] [key=value]…

    Examples
    --------
    --metric cv 1
    --metric corr_median 0 bin_size=25 subsample_pct=0.002
    """
    if len(flag_values) < 2:
        raise argparse.ArgumentTypeError(
            "--metric expects at least NAME TARGET"
        )
    name, target_str, *rest = flag_values
    try:
        target = float(target_str)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"TARGET must be numeric: {err}") from err

    # optional weight (first token that is pure number)
    weight = 1.0
    kwargs_tokens = []
    if rest and "=" not in rest[0]:
        weight = float(rest[0])
        kwargs_tokens = rest[1:]
    else:
        kwargs_tokens = rest

    # parse key=value pairs
    kwargs: Dict[str, Any] = {}
    for tok in kwargs_tokens:
        if "=" not in tok:
            raise argparse.ArgumentTypeError(f"Bad kwarg '{tok}', expected key=value.")
        k, v = tok.split("=", 1)
        try:
            v_val: Any = float(v)
        except ValueError:
            v_val = v  # keep as string for metric to interpret
        kwargs[k] = v_val

    return Metric(name, target=target, weight=weight, kwargs=kwargs)


# ----------------------------------------------------------------------
# Sub-command implementations
# ----------------------------------------------------------------------
def optimise_cli(args: argparse.Namespace) -> None:
    # Build param_space dict
    param_space: Dict[str, Tuple] = dict(args.param or [])
    # Build metrics list
    metrics = args.metric or []

    run_dir = pathlib.Path(args.run_folder).expanduser().resolve()

    opt = Optimizer(
        run_dir=run_dir,
        netsim_dir=args.netsim_dir,
        params_file=args.params_file,
        param_space=param_space,
        metrics=metrics,
        optimizer=args.optimizer,
        budget=args.budget,
        num_workers=args.workers,
        seed=args.seed,
    )
    history = opt.run()

    # Real-time plots are handled by the Optimizer internally; here we only
    # drop out with the final recommendation.
    if not history.empty:
        best_row = history.loc[history["loss"].idxmin()]
        print("\nBest parameters so far:")
        print(best_row.to_string())


def plot_cli(args: argparse.Namespace) -> None:
    run_dir = pathlib.Path(args.run_folder).expanduser().resolve()
    csv_path = run_dir / "history.csv"
    if not csv_path.exists():
        sys.exit(f"Cannot find history file: {csv_path}")
    create_pairplots(csv_path, out_dir=run_dir, reduce=args.reduce)
    print(f"Plots written to {run_dir}")


# ----------------------------------------------------------------------
# Top-level parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neversim",
        description="Optimise NETSIM parameters with Nevergrad."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # OPTIMIZE ----------------------------------------------------------
    p_opt = sub.add_parser("optimize", help="start or resume an optimisation")
    p_opt.add_argument("run_folder", help="directory where checkpoints & logs live")
    p_opt.add_argument("--params-file", required=True, help="NETSIM parameter file")
    p_opt.add_argument(
        "--param",
        nargs=4,
        action="append",
        metavar=("NAME", "TYPE", "LOWER", "UPPER"),
        help="parameter to vary (repeatable)",
        type=_parse_param,
    )
    p_opt.add_argument(
        "--metric",
        nargs="+",
        action="append",
        metavar=("NAME", "TARGET", "[WEIGHT]"),
        help="metric target (repeatable)",
        type=_parse_metric,
    )
    p_opt.add_argument("--optimizer", default="TwoPointsDE",
                       help="Nevergrad optimiser class name")
    p_opt.add_argument("--budget", type=int, default=200,
                       help="total number of evaluations")
    p_opt.add_argument("--workers", type=int, default=max(1, (sys.cpu_count() or 2) - 1),
                       help="parallel workers")
    p_opt.add_argument("--seed", type=int, default=42, help="random seed")
    p_opt.set_defaults(func=optimise_cli)

    # PLOT --------------------------------------------------------------
    p_plot = sub.add_parser("plot", help="generate plots from an existing run")
    p_opt.add_argument("--netsim-dir", help="Path containing pyNetsim package")
    p_plot.add_argument("run_folder", help="folder created by the optimize command")
    p_plot.add_argument("--reduce", choices=("pca", "tsne", "umap"),
                        default="pca", help="dimensionality-reduction method")
    p_plot.set_defaults(func=plot_cli)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)  # dispatch


# Entry point for `python -m neversim`
if __name__ == "__main__":
    main()