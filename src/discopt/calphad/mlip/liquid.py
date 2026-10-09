"""Liquids: mixing enthalpies by MD, mixing free energies by thermodynamic integration.

A liquid has no lattice, so neither SQS nor cluster expansions apply. Here
the calculator runs molecular dynamics at constant temperature and zero
pressure (ASE's Langevin BAOAB integrator with a hydrostatic barostat), from
a lattice melted at ``melt_factor`` × T:

- **ΔH_mix(x, T)** = ⟨H⟩(x) − x⟨H⟩(A) − (1 − x)⟨H⟩(B), with ⟨H⟩ the mean
  potential energy plus PV per atom (the kinetic energy, 3kT/2 per atom,
  cancels). Its error is from block averages.
- **ΔG_mix(x, T)** by alchemical thermodynamic integration. One labelled atom
  is turned from B into A along U(λ) = (1 − λ) U_B + λ U_A, and
  ΔF = ∫₀¹ ⟨U_A − U_B⟩_λ dλ (Gauss–Legendre in λ). With N_A of the N atoms
  A, ΔF is the excess part of μ_A − μ_B at x = (N_A + ½)/N: the ideal part,
  kT ln(x/(1 − x)), is what relabelling identical atoms adds, and the masses
  add a constant. A polynomial p(x) is fitted to ΔF on ``n_mu`` such
  compositions from x ≈ 0 to x ≈ 1, and Gibbs–Duhem gives

      G_xs(x) = ∫₀ˣ p − x ∫₀¹ p,     ΔG_mix = G_xs + kT (x ln x + (1 − x) ln(1 − x)),

  with no absolute free energy of either pure liquid needed (the constant
  cancels between the two terms). The error is the fit's, from the λ-runs'
  block errors.

A run whose atoms barely move (mean-square displacement below ``min_msd``
Å² during sampling) did not melt; it is reported and warned about.
Everything is classical: no quantum nuclear effects.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from discopt.calphad.mlip.phonons import K_B

__all__ = ["LiquidOptions", "LiquidResult", "liquid_mixing"]


@dataclass(frozen=True)
class LiquidOptions:
    """Settings for liquid MD.

    The cell is the cubic FCC cell repeated n×n×n (4n³ atoms, the smallest n
    with at least ``atoms``), at the Vegard volume, decorated at random. Each
    run melts at ``melt_factor`` × T for ``melt_steps``, equilibrates at T for
    ``equilibration_steps``, then samples every ``interval`` of ``steps``
    (``timestep`` in fs). ``pressure`` is in GPa; ``None`` keeps the starting
    volume (NVT). ``thermostat`` and ``barostat`` are the time constants (fs)
    of the Langevin thermostat and the cell dynamics. ΔG uses ``n_mu``
    compositions, ``n_lambda`` Gauss–Legendre points in λ each, and a
    polynomial of degree ``degree`` (at most ``n_mu`` − 1) in x.
    """

    temperatures: tuple[float, ...] = (1500.0,)
    atoms: int = 108
    timestep: float = 2.0
    melt_factor: float = 2.0
    melt_steps: int = 2000
    equilibration_steps: int = 1000
    steps: int = 4000
    interval: int = 10
    pressure: float | None = 0.0
    thermostat: float = 100.0
    barostat: float = 1000.0
    free_energy: bool = True
    n_mu: int = 5
    n_lambda: int = 5
    degree: int = 3
    blocks: int = 5
    min_msd: float = 1.0
    seed: int = 0


@dataclass
class LiquidResult:
    """Liquid mixing at one composition and temperature, per atom (eV)."""

    x: float
    T: float
    H: float
    H_error: float
    G: float | None = None
    G_error: float | None = None
    #: mean volume per atom (Å³) and mean-square displacement while sampling (Å²)
    volume: float = 0.0
    msd: float = 0.0


@dataclass
class _Run:
    E: np.ndarray
    volume: float
    msd: float
    atoms: object
    dudl: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _block_error(v: np.ndarray, blocks: int) -> float:
    v = np.asarray(v, float)
    if len(v) == 0 or np.ptp(v) <= 1e-12 * max(1.0, float(np.max(np.abs(v)))):
        return 0.0  # constant to round-off: no statistical error
    if len(v) < 2 * blocks:
        return float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    means = [b.mean() for b in np.array_split(v, blocks)]
    return float(np.std(means, ddof=1) / math.sqrt(blocks))


def _lattice(pair, n_a: int, volume: float, options: LiquidOptions, rng):
    from ase.build import bulk

    n = 1
    while 4 * n**3 < options.atoms:
        n += 1
    atoms = bulk("Cu", "fcc", a=(4 * volume) ** (1 / 3), cubic=True).repeat(n)
    sym = np.array([pair[1].capitalize()] * len(atoms), dtype=object)
    sym[rng.permutation(len(atoms))[:n_a]] = pair[0].capitalize()
    atoms.set_chemical_symbols(list(sym))
    return atoms


def _transmute(calc, index: int, pair, lam: float):
    """An ASE calculator for U(λ) = (1 − λ) U(atom ``index`` is B) + λ U(it is A); ``results["dudl"]`` is U_A − U_B."""
    from ase.calculators.calculator import Calculator, all_changes

    class Transmute(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            out = []
            for s in (pair[1], pair[0]):
                a = self.atoms.copy()
                sym = a.get_chemical_symbols()
                sym[index] = s.capitalize()
                a.set_chemical_symbols(sym)
                a.calc = calc
                out.append((a.get_potential_energy(), a.get_forces(), a.get_stress()))
            (e0, f0, s0), (e1, f1, s1) = out
            self.results = {
                "energy": (1 - lam) * e0 + lam * e1,
                "forces": (1 - lam) * f0 + lam * f1,
                "stress": (1 - lam) * s0 + lam * s1,
                "dudl": e1 - e0,
            }

    return Transmute()


def _md(atoms, calc, T: float, options: LiquidOptions, rng, melt: bool, sample: bool) -> _Run:
    from ase import units
    from ase.md.langevinbaoab import LangevinBAOAB
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = atoms.copy()
    atoms.calc = calc
    npt = {}
    if options.pressure is not None:
        npt = {
            "externalstress": -options.pressure * units.GPa,
            "hydrostatic": True,
            "P_tau": options.barostat * units.fs,
        }

    def dyn(temp):
        return LangevinBAOAB(
            atoms, options.timestep * units.fs, temperature_K=temp, T_tau=options.thermostat * units.fs, rng=rng, **npt
        )

    if melt:
        MaxwellBoltzmannDistribution(atoms, temperature_K=options.melt_factor * T, rng=rng)
        dyn(options.melt_factor * T).run(options.melt_steps)
    dyn(T).run(options.equilibration_steps)
    if not sample:
        return _Run(np.zeros(0), 0.0, 0.0, atoms)
    md = dyn(T)
    pv = (options.pressure or 0.0) * units.GPa  # H = E + PV, eV/Å³
    E, V, D = [], [], []
    start = atoms.get_positions(wrap=False).copy()
    start_cell = atoms.cell.array.copy()

    def record():
        E.append((atoms.get_potential_energy() + pv * atoms.get_volume()) / len(atoms))
        V.append(atoms.get_volume() / len(atoms))
        if "dudl" in getattr(calc, "results", {}):
            D.append(calc.results["dudl"])

    md.attach(record, interval=options.interval)
    md.run(options.steps)
    # displacement in scaled coordinates, so the barostat's rescaling is not counted as diffusion
    disp = (atoms.get_positions(wrap=False) @ np.linalg.inv(atoms.cell.array)) - start @ np.linalg.inv(start_cell)
    msd = float(np.mean(np.sum((disp @ atoms.cell.array) ** 2, axis=1)))
    return _Run(np.array(E), float(np.mean(V)), msd, atoms, np.array(D))


def _enthalpy(calc, pair, n_a, volume, T, options, rng) -> _Run:
    atoms = _lattice(pair, n_a, volume, options, rng)
    return _md(atoms, calc, T, options, rng, melt=True, sample=True)


def _transmutation(calc, pair, n_a, volume, T, options, rng) -> tuple[float, float, float, float]:
    """(x of the midpoint, ΔF for one B → A at N_A = n_a (eV), its error, smallest msd)."""
    atoms = _lattice(pair, n_a, volume, options, rng)
    b_sites = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == pair[1].capitalize()]
    index = b_sites[0]
    nodes, weights = np.polynomial.legendre.leggauss(options.n_lambda)
    lams, weights = (nodes + 1) / 2, weights / 2
    means, errors, msd = [], [], math.inf
    melt = True
    for lam in lams:
        calc_l = _transmute(calc, index, pair, float(lam))
        run = _md(atoms, calc_l, T, options, rng, melt=melt, sample=True)
        atoms, melt = run.atoms, False
        means.append(float(np.mean(run.dudl)))
        errors.append(_block_error(run.dudl, options.blocks))
        msd = min(msd, run.msd)
    dF = float(np.dot(weights, means))
    err = float(math.sqrt(np.sum((weights * np.array(errors)) ** 2)))
    return (n_a + 0.5) / len(atoms), dF, err, msd


def _excess(xm, dF, err, degree: int):
    """G_xs(x) and its error, from a weighted polynomial fit of Δμ_xs at the midpoints."""
    xm, dF, err = (np.asarray(v, float) for v in (xm, dF, err))
    deg = min(degree, len(xm) - 1)
    A = np.vander(xm, deg + 1, increasing=True)  # p(x) = Σ c_k x^k
    if np.all(err == 0):  # no fluctuations (a model whose dU/dλ is constant): exact, unweighted
        c = np.linalg.lstsq(A, dF, rcond=None)[0]
        cov = np.zeros((deg + 1, deg + 1))
    else:
        w = 1.0 / np.maximum(err, err[err > 0].min())
        Aw = A * w[:, None]
        cov = np.linalg.pinv(Aw.T @ Aw)
        c = cov @ Aw.T @ (dF * w)
    k = np.arange(deg + 1)

    def gvec(x):
        # ∫₀ˣ x^k = x^(k+1)/(k+1), minus x ∫₀¹ x^k = x/(k+1)
        return (x ** (k + 1) - x) / (k + 1)

    def G(x):
        g = gvec(x)
        return float(g @ c), float(math.sqrt(max(g @ cov @ g, 0.0)))

    return G


def liquid_mixing(model, pair, grid, options: LiquidOptions | None = None, volumes=None) -> list[LiquidResult]:
    """Liquid ΔH_mix (and ΔG_mix) at each composition of ``grid`` (x of ``pair[0]``) and temperature.

    ``volumes`` gives the starting volume per atom of each element (Å³); the
    default is :func:`~discopt.calphad.mlip.prototypes.atomic_volume`.
    """
    from discopt.calphad.mlip.prototypes import atomic_volume

    options = options or LiquidOptions()
    if not hasattr(model, "calc"):
        raise TypeError("liquid MD needs an ASE calculator: build the model with mlip.from_ase")
    calc = model.calc
    vol = volumes or {p: atomic_volume(p.capitalize()) for p in pair}
    n = 1
    while 4 * n**3 < options.atoms:
        n += 1
    N = 4 * n**3
    out = []
    for it, T in enumerate(options.temperatures):
        rng = np.random.default_rng([options.seed, it])
        pure = {}
        for p, n_a in ((pair[0], N), (pair[1], 0)):
            pure[p] = _enthalpy(calc, pair, n_a, vol[p], T, options, rng)
        G = None
        if options.free_energy:
            counts = sorted({int(round(t * (N - 1))) for t in np.linspace(0, 1, options.n_mu)})
            pts = []
            for n_a in counts:
                v = (n_a * vol[pair[0]] + (N - n_a) * vol[pair[1]]) / N
                pts.append(_transmutation(calc, pair, n_a, v, T, options, rng))
            for xm, _, _, msd in pts:
                _melted(msd, options, f"liquid x={xm:.3f} at {T:g} K (transmutation)")
            G = _excess([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], options.degree)
        for x in np.atleast_1d(np.asarray(grid, float)):
            n_a = int(round(float(x) * N))
            xa = n_a / N
            v = xa * vol[pair[0]] + (1 - xa) * vol[pair[1]]
            run = _enthalpy(calc, pair, n_a, v, T, options, rng)
            _melted(run.msd, options, f"liquid x={xa:.3f} at {T:g} K")
            ea, eb = pure[pair[0]], pure[pair[1]]
            H = float(np.mean(run.E) - xa * np.mean(ea.E) - (1 - xa) * np.mean(eb.E))
            err = math.sqrt(
                _block_error(run.E, options.blocks) ** 2
                + (xa * _block_error(ea.E, options.blocks)) ** 2
                + ((1 - xa) * _block_error(eb.E, options.blocks)) ** 2
            )
            res = LiquidResult(xa, float(T), H, err, volume=run.volume, msd=run.msd)
            if G is not None and 0 < xa < 1:
                gx, ge = G(xa)
                res.G = gx + K_B * T * (xa * math.log(xa) + (1 - xa) * math.log(1 - xa))
                res.G_error = ge
            out.append(res)
        for p, r in pure.items():
            _melted(r.msd, options, f"liquid {p} at {T:g} K")
    return out


def _melted(msd: float, options: LiquidOptions, label: str) -> bool:
    if msd < options.min_msd:
        warnings.warn(
            f"{label}: mean-square displacement {msd:.2f} Å² while sampling, below {options.min_msd:g}: "
            "it did not melt (raise the temperature or melt_factor) or the run is too short",
            stacklevel=4,
        )
        return False
    return True
