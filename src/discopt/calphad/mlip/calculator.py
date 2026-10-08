"""Energies from any ASE calculator, with a relaxation policy, a prototype check and a cache.

The plug is an ASE ``Calculator``: a machine-learned potential (UMA, MACE,
MTP, ...), a classical one (EAM, EMT, ...), or anything that implements
``get_potential_energy`` and ``get_forces`` (and ``get_stress`` for a cell
relaxation). :func:`from_ase` wraps one as an :class:`EnergyModel`. Everything
runs in the calling process, one structure at a time.

A relaxation can carry a structure away from the prototype it started in: BCC
collapsing to FCC along the Bain path, or a Laves phase distorting. That
energy then belongs to another structure, and fitting it to the original phase
would be wrong. Each :class:`Relaxation` therefore records three measures and
sets ``left_prototype``, with the reason, when one crosses a threshold of the
:class:`RelaxPolicy`:

- the cell strain that breaks the symmetry of the parent lattice (all atoms
  taken as one species, so an SQS is judged by its FCC or BCC lattice). A
  change of c/a in a hexagonal cell keeps that symmetry and does not count; a
  shear of a cubic cell (the Bain path) does;
- the largest atomic displacement after the homogeneous strain is removed;
- the space group before and after (needs spglib), unless nothing was relaxed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "EV_TO_J_PER_MOL",
    "EnergyModel",
    "ASEModel",
    "RelaxPolicy",
    "Relaxation",
    "RelaxationCache",
    "from_ase",
    "structure_hash",
]

#: 1 eV per atom in J per mole of atoms (the Faraday constant, CODATA 2018)
EV_TO_J_PER_MOL = 96485.33212

_MODES = ("symmetry", "full", "fixed")
#: tolerance (Å) for the parent lattice's symmetry: loose, so a slightly perturbed start still counts as cubic
_LATTICE_SYMPREC = 0.1


@dataclass(frozen=True)
class RelaxPolicy:
    """How structures are relaxed, and when a relaxed structure counts as having left its prototype.

    ``mode``:

    - ``"symmetry"`` (default): positions and cell relax, constrained to the
      starting space group (ASE ``FixSymmetry``);
    - ``"full"``: positions and cell relax without constraints;
    - ``"fixed"``: no relaxation; the energy of the structure as built.

    ``cell=False`` keeps the cell fixed in the first two modes. ``fmax`` (eV/Å)
    and ``steps`` go to the ASE optimizer ``optimizer``.

    ``max_strain`` bounds the Green strain of the cell that breaks the parent
    lattice's symmetry, and ``max_displacement`` the largest atomic
    displacement after the homogeneous strain is removed, as a fraction of the
    nearest-neighbour distance. Neither enters the cache key.
    """

    mode: str = "symmetry"
    fmax: float = 0.01
    steps: int = 500
    cell: bool = True
    optimizer: str = "BFGS"
    symprec: float = 1e-3
    max_strain: float = 0.05
    max_displacement: float = 0.2

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError(f"unknown relaxation mode {self.mode!r}; use one of {_MODES}")

    def key(self) -> dict:
        """The settings that change the computed energy."""
        return {k: getattr(self, k) for k in ("mode", "fmax", "steps", "cell", "optimizer", "symprec")}


@dataclass
class Relaxation:
    """The outcome of relaxing one structure. Energies are in eV."""

    source: str
    formula: str
    natoms: int
    energy: float
    converged: bool
    steps: int
    #: Green strain of the cell that breaks the parent lattice's symmetry (largest absolute eigenvalue)
    strain: float
    #: largest atomic displacement after removing the homogeneous strain, in Å
    displacement: float
    #: nearest-neighbour distance of the starting structure, in Å
    nn_distance: float
    spacegroup_initial: str
    spacegroup_final: str
    #: final structure: atomic numbers, Cartesian positions (Å), cell (Å)
    numbers: list = field(repr=False, default_factory=list)
    positions: list = field(repr=False, default_factory=list)
    cell: list = field(repr=False, default_factory=list)
    cached: bool = False
    left_prototype: bool = False
    reason: str = ""

    @property
    def energy_per_atom(self) -> float:
        return self.energy / self.natoms

    def atoms(self):
        """The final structure as ``ase.Atoms``."""
        from ase import Atoms

        return Atoms(numbers=self.numbers, positions=self.positions, cell=self.cell, pbc=True)

    def judge(self, policy: RelaxPolicy, check_symmetry: bool = True) -> Relaxation:
        """Set ``left_prototype`` and ``reason`` from the thresholds of ``policy``."""
        reasons = []
        if self.strain > policy.max_strain:
            reasons.append(f"symmetry-breaking cell strain {self.strain:.3f} > {policy.max_strain}")
        rel = self.displacement / self.nn_distance if self.nn_distance > 0 else 0.0
        if rel > policy.max_displacement:
            reasons.append(
                f"atomic displacement {self.displacement:.3f} Å = {rel:.2f} of the nearest-neighbour "
                f"distance > {policy.max_displacement}"
            )
        if check_symmetry and self.spacegroup_initial and self.spacegroup_final != self.spacegroup_initial:
            reasons.append(f"space group {self.spacegroup_initial} -> {self.spacegroup_final}")
        if not self.converged:
            reasons.append(f"not converged in {self.steps} steps")
        self.left_prototype = bool(reasons)
        self.reason = "; ".join(reasons)
        return self


@runtime_checkable
class EnergyModel(Protocol):
    """What the data builders need from a model.

    ``name`` is the data source recorded as provenance on every observation,
    e.g. ``"mace-mp-0b3-medium"`` or ``"EMT"``. ``energies`` returns relaxed
    energies in eV/atom. A model may also provide ``relax`` (records with the
    prototype check) and ``uncertainty`` (an ensemble or committee spread, in
    eV/atom).
    """

    name: str

    def energies(self, structures, relax: RelaxPolicy) -> np.ndarray: ...


def structure_hash(atoms) -> str:
    """A hash of the atomic numbers, positions, cell and periodicity, rounded to 1e-6 Å."""
    payload = {
        "numbers": [int(z) for z in atoms.get_atomic_numbers()],
        "positions": np.round(atoms.get_positions(), 6).tolist(),
        "cell": np.round(np.asarray(atoms.cell), 6).tolist(),
        "pbc": [bool(p) for p in atoms.pbc],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class RelaxationCache:
    """Relaxations keyed by (source label, structure hash, relaxation settings).

    In memory, and also on disk as one JSON file per entry when ``path`` is
    given. ``path=True`` uses ``<discopt-calphad cache>/mlip``. A model whose
    weights change must get a new source label, or the cache returns the old
    energies.
    """

    def __init__(self, path=None):
        if path is True:
            from discopt.calphad.databases import cache_dir

            path = cache_dir() / "mlip"
        self.path = Path(path) if path else None
        self._mem: dict[str, dict] = {}

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        if key in self._mem:
            return self._mem[key]
        if self.path is not None:
            f = self.path / f"{key}.json"
            if f.is_file():
                self._mem[key] = json.loads(f.read_text())
                return self._mem[key]
        return None

    def put(self, key: str, value: dict) -> None:
        self._mem[key] = value
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            tmp = self.path / f".{key}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(value))
            os.replace(tmp, self.path / f"{key}.json")

    def __len__(self) -> int:
        return len(self._mem)


def _spacegroup(atoms, symprec: float) -> str:
    try:
        import spglib
    except ImportError:  # the prototype check then rests on strain and displacement
        return ""
    cell = (np.asarray(atoms.cell), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    return spglib.get_spacegroup(cell, symprec=symprec) or ""


def _nn_distance(atoms) -> float:
    from ase.neighborlist import neighbor_list

    cutoff = 2.0 * (atoms.get_volume() / len(atoms)) ** (1 / 3)
    d = neighbor_list("d", atoms, cutoff)
    return float(d.min()) if d.size else 0.0


def _lattice_rotations(atoms, symprec: float) -> list[np.ndarray]:
    """Cartesian point-group rotations of the parent lattice: every atom taken as the same species."""
    try:
        import spglib
    except ImportError:
        return [np.eye(3)]
    C = np.asarray(atoms.cell)
    sym = spglib.get_symmetry((C, atoms.get_scaled_positions(), [1] * len(atoms)), symprec=symprec)
    if sym is None:
        return [np.eye(3)]
    Cinv_T = np.linalg.inv(C).T
    return [C.T @ R @ Cinv_T for R in sym["rotations"]]


def _distortion(initial, final, rotations=None) -> tuple[float, float]:
    """(symmetry-breaking Green strain, largest displacement in Å after removing the homogeneous strain)."""
    c0, c1 = np.asarray(initial.cell), np.asarray(final.cell)
    F = np.linalg.solve(c0, c1).T  # deformation gradient on column vectors: c1.T = F @ c0.T
    E = 0.5 * (F.T @ F - np.eye(3))
    if rotations:
        E = E - sum(R @ E @ R.T for R in rotations) / len(rotations)
    strain = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (E + E.T)))))
    ds = final.get_scaled_positions(wrap=False) - initial.get_scaled_positions(wrap=False)
    ds -= np.round(ds)
    ds -= ds.mean(axis=0)  # rigid translation
    return strain, float(np.max(np.linalg.norm(ds @ c1, axis=1)))


class ASEModel:
    """An :class:`EnergyModel` around an ASE calculator. Build it with :func:`from_ase`."""

    def __init__(self, calc, name: str, cache: RelaxationCache | None = None, uncertainty=None):
        if not name:
            raise ValueError("a model needs a name: it is the data source recorded on every observation")
        self.calc = calc
        self.name = name
        self.cache = cache if cache is not None else RelaxationCache()
        self._uncertainty = uncertainty

    def __repr__(self) -> str:
        return f"ASEModel({self.name!r})"

    def relax(self, structures, relax: RelaxPolicy | None = None) -> list[Relaxation]:
        """Relax each structure in turn (or read it from the cache) and check it against its prototype."""
        policy = relax or RelaxPolicy()
        return [self._relax_one(a, policy) for a in structures]

    def energies(self, structures, relax: RelaxPolicy | None = None) -> np.ndarray:
        """Relaxed energies, eV/atom."""
        return np.array([r.energy_per_atom for r in self.relax(structures, relax)])

    def uncertainty(self, structures) -> np.ndarray:
        """Ensemble or committee spread in eV/atom, if the model was given one. Raises otherwise."""
        if self._uncertainty is None:
            raise NotImplementedError(f"{self.name} has no uncertainty estimate")
        return np.asarray([self._uncertainty(a) for a in structures], float)

    @property
    def has_uncertainty(self) -> bool:
        return self._uncertainty is not None

    def _relax_one(self, atoms0, policy: RelaxPolicy) -> Relaxation:
        key = RelaxationCache.key(self.name, structure_hash(atoms0), policy.key())
        hit = self.cache.get(key)
        if hit is not None:
            rec = Relaxation(**{**hit, "cached": True})
            return rec.judge(policy, check_symmetry=policy.mode != "fixed")
        rec = self._compute(atoms0, policy)
        self.cache.put(key, {k: v for k, v in asdict(rec).items() if k not in ("cached", "left_prototype", "reason")})
        return rec.judge(policy, check_symmetry=policy.mode != "fixed")

    def _compute(self, atoms0, policy: RelaxPolicy) -> Relaxation:
        from ase import Atoms
        from ase import optimize as ase_optimize

        atoms = Atoms(
            numbers=atoms0.get_atomic_numbers(), positions=atoms0.get_positions(), cell=atoms0.cell, pbc=atoms0.pbc
        )
        atoms.calc = self.calc
        sg0 = _spacegroup(atoms, policy.symprec)
        rotations = _lattice_rotations(atoms, _LATTICE_SYMPREC)
        nn = _nn_distance(atoms)
        converged, steps = True, 0
        if policy.mode != "fixed":
            if policy.mode == "symmetry":
                from ase.constraints import FixSymmetry

                atoms.set_constraint(FixSymmetry(atoms, symprec=policy.symprec))
            target = atoms
            if policy.cell:
                from ase.filters import FrechetCellFilter

                target = FrechetCellFilter(atoms)
            opt = getattr(ase_optimize, policy.optimizer)(target, logfile=None)
            converged = bool(opt.run(fmax=policy.fmax, steps=policy.steps))
            steps = int(opt.get_number_of_steps())
        energy = float(atoms.get_potential_energy())
        atoms.set_constraint()
        strain, disp = _distortion(atoms0, atoms, rotations)
        return Relaxation(
            source=self.name,
            formula=atoms.get_chemical_formula(),
            natoms=len(atoms),
            energy=energy,
            converged=converged,
            steps=steps,
            strain=strain,
            displacement=disp,
            nn_distance=nn,
            spacegroup_initial=sg0,
            spacegroup_final=_spacegroup(atoms, policy.symprec) if sg0 else "",
            numbers=[int(z) for z in atoms.get_atomic_numbers()],
            positions=atoms.get_positions().tolist(),
            cell=np.asarray(atoms.cell).tolist(),
        )


def from_ase(calc, name: str, *, cache=None, uncertainty=None) -> ASEModel:
    """Wrap an ASE calculator as an :class:`EnergyModel`.

    ``name`` is the data source label, e.g. ``"uma-s-1p1/omat"`` or ``"EMT"``.
    ``cache`` is a :class:`RelaxationCache`, a directory, ``True`` (the user
    cache), or None (in memory only). ``uncertainty`` is an optional callable
    ``atoms -> eV/atom`` that gives an ensemble or committee spread.
    """
    if cache is not None and not isinstance(cache, RelaxationCache):
        cache = RelaxationCache(cache)
    return ASEModel(calc, name, cache, uncertainty)
