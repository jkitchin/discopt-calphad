"""Plots for binary systems."""

from __future__ import annotations

import numpy as np

from discopt.calphad.ops import NUM

__all__ = ["lower_convex_hull", "gx_curves", "plot_gx"]


def lower_convex_hull(x, g):
    """Lower convex hull of points (x, g), using a monotone chain; the points are sorted by x first."""
    x, g = np.asarray(x, float), np.asarray(g, float)
    order = np.lexsort((g, x))
    x, g = x[order], g[order]
    # keep only the lowest g at each x; otherwise a higher point at the last x leaves a vertical edge
    first = np.concatenate(([True], np.diff(x) > 0))
    hull: list[tuple[float, float]] = []
    for p in zip(x[first], g[first]):
        while len(hull) >= 2:
            (x1, g1), (x2, g2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - g1) - (g2 - g1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx, hg = zip(*hull)
    return np.array(hx), np.array(hg)


def gx_curves(phase, T=None, n: int = 4001, grid_points: int = 250_000):
    """Mole fraction of the first element and G/RT per atom, sampled over a binary phase.

    A phase with one site fraction gives ``n`` points, and a compound gives one
    point. With several site fractions (a CEF phase), about ``grid_points`` points
    are sampled on a grid. For each of ``n`` bins in x the lowest G is kept,
    which is the curve a common-tangent construction sees.
    """
    if phase.n_y == 0:
        a = np.array(phase.amounts(), float)
        return np.array([a[0] / a.sum()]), np.array([float(phase.gibbs((), NUM, T)) / a.sum()])
    if phase.n_y == 1:
        lo, hi = phase.y_bounds[0]
        ys = np.linspace(lo, hi, n)
        a = np.array([[float(v) for v in phase.amounts([y])] for y in ys])
        g = np.array([float(phase.gibbs([y], NUM, T)) for y in ys])
        atoms = a.sum(axis=1)
        return a[:, 0] / atoms, g / atoms
    per_dim = max(3, int(round(grid_points ** (1.0 / phase.n_y))))
    axes = [np.linspace(lo, hi, per_dim) for lo, hi in phase.y_bounds]
    y = [v.ravel() for v in np.meshgrid(*axes, indexing="ij")]
    a = np.array([np.broadcast_to(np.asarray(v, float), y[0].shape) for v in phase.amounts(y)])
    g = np.broadcast_to(np.asarray(phase.gibbs(y, NUM, T), float), y[0].shape)
    atoms = a.sum(axis=0)
    keep = atoms > 0
    x, g = a[0][keep] / atoms[keep], g[keep] / atoms[keep]
    bins = np.minimum((x * n).astype(int), n - 1)
    order = np.lexsort((g, bins))
    first = np.concatenate(([True], np.diff(bins[order]) > 0))
    sel = order[first]
    sel = sel[np.argsort(x[sel])]
    return x[sel], g[sel]


def plot_gx(phases, result=None, T=None, ax=None, n: int = 4001, show_hull: bool = True):
    """Plot G/RT per atom against x for one or more binary phases.

    If ``result`` is given, the plot also shows the μ tangent line and the
    composition sets. ``show_hull`` adds the lower convex hull of all sampled
    points. The hull is a reference drawn from sampled points, not solver output.
    """
    import matplotlib.pyplot as plt

    phases = list(phases) if isinstance(phases, (list, tuple)) else [phases]
    if T is None and result is not None:
        T = result.T
    ax = ax or plt.gca()
    allx, allg = [], []
    for i, ph in enumerate(phases):
        xs, gs = gx_curves(ph, T, n)
        allx.append(xs)
        allg.append(gs)
        if ph.n_y == 0:
            ax.plot(xs, gs, "s", color=f"C{i}", ms=7, label=ph.name)
        else:
            ax.plot(xs, gs, color=f"C{i}", lw=2, label=ph.name)
        spin = ph.spinodal() if hasattr(ph, "spinodal") else None
        if spin:
            ax.axvspan(*spin, color="0.92", zorder=0, label=f"{ph.name} spinodal")
    if show_hull:
        hx, hg = lower_convex_hull(np.concatenate(allx), np.concatenate(allg))
        ax.plot(hx, hg, ls=":", color="k", lw=1.2, label="lower convex hull (grid reference)")
    if result is not None:
        xs = np.linspace(0, 1, 201)
        ax.plot(
            xs, result.mu[0] * xs + result.mu[1] * (1 - xs), ls="--", color="C3", lw=1.2, label="μ tangent (solver)"
        )
        for cs in result.composition_sets:
            ax.plot(
                cs.x[0],
                float(cs.phase.gibbs(cs.y, NUM, T)) / cs.atoms,
                "o",
                mfc="none",
                mec="k",
                ms=10,
                mew=1.5,
                zorder=6,
            )
        ax.plot([], [], "o", mfc="none", mec="k", label="composition sets")
    names = phases[0].elements
    ax.set_xlabel(f"x({names[0]})")
    ax.set_ylabel("G / RT per atom")
    ax.legend(fontsize=8)
    return ax
