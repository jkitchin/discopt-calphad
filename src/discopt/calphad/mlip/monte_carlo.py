"""Short-range order: canonical Monte Carlo on a cluster expansion, run by icet's mchammer.

A random alloy (SQS, random decorations, or the random-alloy prediction of a
cluster expansion) has no short-range order. At finite temperature, though,
the atoms of an alloy prefer like or unlike neighbours, which lowers both the
enthalpy and the entropy of mixing. Here mchammer samples the canonical
ensemble of a :class:`~discopt.calphad.mlip.cluster_expansion.ClusterExpansionFit`
in a supercell at each temperature, and reports per atom:

- ⟨ΔH_mix⟩(T), with the statistical error from mchammer's ``analyze_data``
  (correlation length taken into account);
- the Warren–Cowley parameters α of each pair shell within ``sro_radius``
  (negative: more unlike neighbours than at random, positive: fewer);
- with ``integrate=True``, the configurational free energy and entropy of
  mixing by thermodynamic integration in β = 1/kT from infinite temperature,
  where the alloy is random and its entropy ideal:

      βΔG(β) = s_mix (y ln y + (1 − y) ln(1 − y)) + ∫₀^β ⟨ΔH⟩(β′) dβ′,
      ΔS = (⟨ΔH⟩ − ΔG) / T,

  with s_mix the mixing sites per atom. At β = 0, ⟨ΔH⟩ is the random-alloy
  value of the expansion, exactly; elsewhere it comes from Monte Carlo on a
  β grid (trapezoid rule).

The composition must fit the supercell; it is the nearest one that does (the
observations record the composition actually sampled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from discopt.calphad.mlip.phonons import K_B

__all__ = ["MonteCarloOptions", "SROResult", "supercell", "sample", "short_range_order"]


@dataclass(frozen=True)
class MonteCarloOptions:
    """Settings for canonical Monte Carlo.

    ``temperatures`` (K) are where values are reported. The supercell repeats
    the expansion's primitive cell n×n×n, with the smallest n that gives at
    least ``min_atoms`` atoms. Each run makes ``sweeps`` × (atoms) trial swaps
    and discards the first ``equilibration`` fraction. ``sro_radius`` (in
    nearest-neighbour distances) sets the pair shells reported. With
    ``integrate=True``, ⟨ΔH⟩ is also sampled on ``n_beta`` evenly spaced β
    between 0 and that of the lowest temperature, for ΔG_mix and ΔS_mix.
    """

    temperatures: tuple[float, ...] = (600.0, 900.0, 1200.0)
    min_atoms: int = 256
    sweeps: int = 200
    equilibration: float = 0.25
    sro_radius: float = 1.1
    integrate: bool = True
    n_beta: int = 12
    seed: int = 0


@dataclass
class SROResult:
    """Monte Carlo averages at one composition and temperature, per atom (eV, eV/K)."""

    y: float
    T: float
    #: ⟨ΔH_mix⟩ and its statistical error
    H: float
    H_error: float
    #: Warren–Cowley α by observer name (``"sro_Al_Cu_1"`` is the first shell)
    alpha: dict = field(default_factory=dict)
    #: ΔG_mix and ΔS_mix by thermodynamic integration (None without it), and the error of ΔS
    G: float | None = None
    S: float | None = None
    S_error: float | None = None
    #: atoms in the supercell
    atoms: int = 0


def _mixing_sites(cs) -> np.ndarray:
    return np.array([len(s) > 1 for s in cs.chemical_symbols])


def supercell(ce, y: float, options: MonteCarloOptions):
    """(random decoration of the n×n×n supercell, composition it holds)."""
    cs = ce.cluster_space
    prim = cs.primitive_structure
    n = 1
    while n**3 * len(prim) < options.min_atoms:
        n += 1
    atoms = prim.repeat(n)
    on_mix = np.tile(_mixing_sites(cs), n**3)
    idx = np.flatnonzero(on_mix)
    k = int(round(y * len(idx)))
    symbols = np.array([s[0] for s in cs.chemical_symbols] * n**3, dtype=object)
    chosen = np.random.default_rng([options.seed, k]).permutation(idx)
    a, b = (p.capitalize() for p in ce.pair)
    symbols[chosen[:k]] = a
    symbols[chosen[k:]] = b
    atoms.set_chemical_symbols(list(symbols))
    return atoms, k / len(idx)


def _nn(atoms) -> float:
    from ase.neighborlist import neighbor_list

    return float(neighbor_list("d", atoms, 2.0 * (atoms.get_volume() / len(atoms)) ** (1 / 3)).min())


def sample(ce, atoms, T: float, options: MonteCarloOptions, sro: bool = True, seed: int = 0) -> dict:
    """One canonical run at ``T``: mean, error of ΔH per atom (eV), and the mean α of each shell."""
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer.data_analysis import analyze_data
    from mchammer.ensembles import CanonicalEnsemble
    from mchammer.observers import ShortRangeOrderObserver

    atoms = atoms.copy()
    n = len(atoms)
    calc = ClusterExpansionCalculator(atoms, ce.expansion)
    mc = CanonicalEnsemble(atoms, calc, temperature=float(T), random_seed=seed, ensemble_data_write_interval=n)
    if sro:
        mc.attach_observer(
            ShortRangeOrderObserver(ce.cluster_space, atoms, radius=options.sro_radius * _nn(atoms), interval=n)
        )
    mc.run(options.sweeps * n)
    data = mc.data_container.data
    data = data.iloc[int(len(data) * options.equilibration) :]
    stats = analyze_data(np.asarray(data["potential"], float) / n)
    err = float(stats["error_estimate"])
    if not math.isfinite(err):  # too short a run for a correlation length
        err = float(stats["std"]) / math.sqrt(len(data))
    alpha = {c: float(np.nanmean(data[c])) for c in data.columns if c.startswith("sro_")}
    return {"H": float(stats["mean"]), "H_error": err, "alpha": alpha}


def short_range_order(ce, y: float, options: MonteCarloOptions | None = None) -> list[SROResult]:
    """Monte Carlo at each of ``options.temperatures``, at the composition nearest ``y`` the supercell holds."""
    options = options or MonteCarloOptions()
    atoms, y_used = supercell(ce, y, options)
    s_mix = float(np.mean(_mixing_sites(ce.cluster_space)))
    temps = sorted(float(t) for t in options.temperatures)
    runs = {T: sample(ce, atoms, T, options, seed=options.seed + i) for i, T in enumerate(temps)}
    out = [SROResult(y_used, T, r["H"], r["H_error"], r["alpha"], atoms=len(atoms)) for T, r in runs.items()]
    if not options.integrate or not 0 < y_used < 1:
        return out
    # thermodynamic integration in β from the random alloy (β = 0)
    beta_max = 1.0 / (K_B * temps[0])
    grid = {1.0 / (K_B * T): (r["H"], r["H_error"]) for T, r in runs.items()}
    grid[0.0] = (ce.random_alloy(y_used), 0.0)
    for k, b in enumerate(np.linspace(0.0, beta_max, options.n_beta + 1)[1:-1]):
        if not any(abs(b - g) < 1e-3 * beta_max for g in grid):
            r = sample(ce, atoms, 1.0 / (K_B * b), options, sro=False, seed=options.seed + 1000 + k)
            grid[float(b)] = (r["H"], r["H_error"])
    betas = np.array(sorted(grid))
    H = np.array([grid[b][0] for b in betas])
    E = np.array([grid[b][1] for b in betas])
    ideal = s_mix * (y_used * math.log(y_used) + (1 - y_used) * math.log(1 - y_used))
    for res in out:
        b = 1.0 / (K_B * res.T)
        j = int(np.argmin(abs(betas - b)))
        # trapezoid weights of each grid point in ∫₀^b, for the value and its error
        w = np.zeros(len(betas))
        w[1 : j + 1] += 0.5 * np.diff(betas[: j + 1])
        w[:j] += 0.5 * np.diff(betas[: j + 1])
        bG = ideal + float(w @ H)
        res.G = bG / b
        res.S = (res.H - res.G) / res.T
        # ΔS = k(β⟨H⟩ − βΔG): the run at T enters both terms
        w[j] -= b
        res.S_error = K_B * math.sqrt(float(np.sum((w * E) ** 2)))
    return out
