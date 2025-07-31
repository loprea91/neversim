"""
neversim.viz
========================

• One legacy heat-map per ordered parameter pair (X-axis earlier name).
• red  x  marks all attempted points (including inf/NaN loss).
• Colour is log-scaled loss (viridis_r).
• Global DR panel (PCA/t-SNE/UMAP) refreshed into 'dr_plot.png'.

Call `LivePlotter.update()` after each evaluation; it spawns a daemon
process that runs `update_figures(history_csv, fig_dir, reduce, ...)`.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Literal, Optional, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

Reduction = Literal["none", "pca", "tsne", "umap"]


# ----------------------------------------------------------------------
# Internal helper that remembers how much of the CSV was processed
# ----------------------------------------------------------------------
class _HistoryCache:
    def __init__(self, csv: Path):
        self.csv = csv
        self.rows = 0
        self.df_full = pd.DataFrame()

    def read(self) -> pd.DataFrame:
        if not self.csv.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.csv)
        if len(df) == self.rows:
            return self.df_full              # nothing new
        self.rows = len(df)
        self.df_full = df
        return df

# ----------------------------------------------------------------------
# Public API -----------------------------------------------------------
# ----------------------------------------------------------------------
def update_figures(
    history_csv: str | os.PathLike,
    fig_dir: str | os.PathLike,
    *,
    reduce: Reduction = "pca",
    max_points_dr: int = 20_000,
) -> None:
    """
    Refresh (overwrite) heat-maps + DR panel from history.csv.

    Called automatically by LivePlotter but can be used stand-alone.
    """
    csv = Path(history_csv)
    figs = Path(fig_dir)
    figs.mkdir(parents=True, exist_ok=True)

    cache = _HistoryCache(csv)
    df = cache.read()
    if df.empty:
        return

    param_cols = sorted(c for c in df.columns if c.startswith("p_"))
    if len(param_cols) < 2:
        return

    # finite-loss subset for colour; attempted subset for red crosses
    finite = df[np.isfinite(df["loss"])]
    attempted = df

    # One heat-map per ordered pair
    for i in range(len(param_cols)):
        for j in range(i + 1, len(param_cols)):
            xcol, ycol = param_cols[i], param_cols[j]
            _plot_pair(
                finite, attempted,
                xcol, ycol,
                figs / f"{xcol[2:]}_{ycol[2:]}_heat.png"
            )

    # Dimensionality-reduction panel
    if reduce != "none" and len(param_cols) > 2:
        _plot_dr(finite, param_cols, figs / "dr_plot.png", reduce, max_points_dr)


# ----------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------
def _plot_pair(df: pd.DataFrame, attempted: pd.DataFrame,
               xcol: str, ycol: str, out: Path) -> None:
    vmin = max(df["loss"].min(), 1e-9)
    vmax = df["loss"].max()
    if vmin >= vmax:
        vmax = vmin * 1.1 + 1e-9

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    # show attempted points in red (no colour scaling)
    ax.scatter(
        attempted[xcol], attempted[ycol],
        color="red", marker="x", s=20, lw=0.8, alpha=0.4, label="attempted"
    )

    # overlay finite-loss points with coloured dots
    sc = ax.scatter(
        df[xcol], df[ycol],
        c=df["loss"],
        cmap="viridis_r",
        s=60,
        alpha=0.9,
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        label="completed",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xcol[2:])
    ax.set_ylabel(ycol[2:])
    ax.legend(loc="best", markerscale=1.2, frameon=True)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("loss")

    ax.set_title(f"{xcol[2:]} vs {ycol[2:]}  ({len(df)} evals)")

    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logging.debug("Saved heat-map %s", out)


def _plot_dr(df: pd.DataFrame, param_cols: List[str],
             out: Path, method: Reduction, max_points: int) -> None:
    import warnings
    from sklearn.decomposition import PCA
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)

    X = df[param_cols].to_numpy()
    if method == "pca":
        emb = PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, init="random", learning_rate="auto").fit_transform(X)
    elif method == "umap":
        try:
            import umap
            emb = umap.UMAP(n_components=2).fit_transform(X)
        except ImportError:
            warnings.warn("umap-learn not installed; skipping DR plot")
            return
    else:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=df["loss"],
        cmap="viridis_r",
        s=40,
        alpha=0.9,
        norm=matplotlib.colors.LogNorm(
            vmin=max(df["loss"].min(), 1e-9),
            vmax=df["loss"].max()
        )
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("loss")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.set_title(f"{method.upper()} projection ({len(df)} pts)")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logging.debug("Saved DR plot %s", out)


# ----------------------------------------------------------------------
# LivePlotter – unchanged public API
# ----------------------------------------------------------------------
class LivePlotter:
    """
    Daemon process that calls `update_figures` every *refresh_every*
    evaluations so Optimizer remains non-blocking.
    """

    def __init__(
        self,
        history_csv: str | os.PathLike,
        fig_dir: str | os.PathLike,
        *,
        refresh_every: int = 5,
        reduce: Reduction = "pca",
    ):
        self.history_csv = str(history_csv)
        self.fig_dir = str(fig_dir)
        self.refresh_every = refresh_every
        self.reduce = reduce
        self._counter = 0
        self._proc: Optional[mp.Process] = None

    # -----------------------------------------------------------------
    def _spawn(self):
        if self._proc and self._proc.is_alive():
            return
        p = mp.Process(
            target=update_figures,
            args=(self.history_csv, self.fig_dir),
            kwargs=dict(reduce=self.reduce),
            daemon=True,
        )
        p.start()
        self._proc = p

    # -----------------------------------------------------------------
    def update(self):
        self._counter += 1
        if self._counter % self.refresh_every == 0:
            self._spawn()

    # -----------------------------------------------------------------
    def close(self):
        if self._proc and self._proc.is_alive():
            self._proc.join(timeout=1)


__all__ = ["LivePlotter", "update_figures"]