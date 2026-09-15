"""Plots for one-dimensional (binary) phases."""

from __future__ import annotations

import numpy as np

from discopt.calphad.ops import NUM

__all__ = ["lower_convex_hull", "plot_gx"]


def lower_convex_hull(x, g):
    """Lower convex hull of the points (x, g), with x sorted ascending (monotone chain)."""
    hull: list[tuple[float, float]] = []
    for p in zip(np.asarray(x, float), np.asarray(g, float)):
        while len(hull) >= 2:
            (x1, g1), (x2, g2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - g1) - (g2 - g1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx, hg = zip(*hull)
    return np.array(hx), np.array(hg)


def plot_gx(phase, result=None, ax=None, n: int = 4001, show_hull: bool = True):
    """Plot G/RT against x for a binary phase with one site fraction.

    If ``result`` is given, the plot also shows the μ tangent line and the
    composition sets. ``show_hull`` adds the lower convex hull of a dense grid.
    The hull is a reference drawn from sampled points, not solver output.
    """
    import matplotlib.pyplot as plt

    if phase.n_y != 1:
        raise ValueError("plot_gx supports phases with one site fraction")
    ax = ax or plt.gca()
    lo, hi = phase.y_bounds[0]
    ys = np.linspace(lo, hi, n)
    xs = np.array([float(phase.amounts([y])[0]) / sum(float(a) for a in phase.amounts([y])) for y in ys])
    gs = np.array([float(phase.gibbs([y], NUM)) for y in ys])
    ax.plot(xs, gs, color="C0", lw=2, label=f"G/RT ({phase.name})")

    spin = phase.spinodal() if hasattr(phase, "spinodal") else None
    if spin:
        ax.axvspan(*spin, color="0.9", zorder=0, label="spinodal region")
    if show_hull:
        hx, hg = lower_convex_hull(xs, gs)
        ax.plot(hx, hg, ls=":", color="k", lw=1.2, label="lower convex hull (grid reference)")
    if result is not None:
        tangent = np.array([sum(m * float(a) for m, a in zip(result.mu, phase.amounts([y]))) for y in ys])
        ax.plot(xs, tangent, ls="--", color="C3", lw=1.2, label="μ tangent (solver)")
        for cs in result.composition_sets:
            ax.plot(cs.x[0], float(cs.phase.gibbs(cs.y, NUM)), "o", color="C3", ms=7, zorder=5)
        ax.plot([], [], "o", color="C3", label="composition sets")
    ax.set_xlabel(f"x({phase.elements[0]})")
    ax.set_ylabel("G / RT")
    ax.legend(fontsize=8)
    return ax
