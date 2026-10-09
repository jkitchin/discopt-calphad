"""Harmonic and quasi-harmonic phonons: vibrational S, Cp and H(T) − H(0) from any ASE calculator.

Force constants come from finite displacements in a supercell (``ase.phonons``,
so no extra dependency). Phonon energies ħω are then sampled on a
Monkhorst–Pack mesh without the Γ point, and the harmonic thermodynamics are
exact mesh averages, per atom:

    F = ⟨ħω/2 + kT ln(1 − e^(−x))⟩       U = ⟨ħω/2 + ħω/(e^x − 1)⟩
    S = k⟨x/(e^x − 1) − ln(1 − e^(−x))⟩   Cv = k⟨x² e^x/(e^x − 1)²⟩,   x = ħω/kT

There are 3 modes per atom. U includes the zero-point energy, so the enthalpy
at T relative to the static lattice is U(T), and H(T) − H(0) = U(T) − U(0).

**Quasi-harmonic.** Phonons at several volumes (isotropic strains, positions
relaxed at fixed cell) give F(V, T) = E₀(V) + F_vib(V, T). At each T, a
cubic in V is minimized for G(T) (pressure 0). S = −dG/dT and Cp = −T d²G/dT²
follow by central differences, and H = G + TS. Thermal expansion enters this
way; the harmonic result is at the relaxed volume only.

**Imaginary modes.** A negative ħω² means the structure is mechanically
unstable at that volume (BCC Cu, for example). Then no vibrational entropy
exists. :class:`Vibrations` reports the fraction of such modes, and the data
builder refuses the structure rather than fitting to it.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field

import numpy as np

from discopt.calphad.mlip.calculator import RelaxationCache, RelaxPolicy, structure_hash

__all__ = ["PhononOptions", "Vibrations", "harmonic", "quasi_harmonic", "thermo_from_modes"]

#: Boltzmann constant in eV/K
K_B = 8.617333262e-5


@dataclass(frozen=True)
class PhononOptions:
    """Settings for phonon calculations.

    ``min_length`` (Å) sets the supercell: each cell vector is repeated until
    it is at least that long. ``delta`` (Å) is the finite displacement.
    ``kpts`` is the Monkhorst–Pack mesh for the thermodynamics. With
    ``quasi_harmonic=True`` the volumes are the isotropic linear ``strains``.
    ``temperatures`` (K) are where the data builder reports entropies (and,
    with ``heat_capacity=True``, heat capacities). ``imaginary_tolerance``
    is the largest fraction of imaginary modes accepted (numerical noise near Γ).
    """

    min_length: float = 8.0
    delta: float = 0.01
    kpts: tuple[int, int, int] = (12, 12, 12)
    quasi_harmonic: bool = False
    strains: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02)
    temperatures: tuple[float, ...] = (298.15, 600.0, 900.0)
    heat_capacity: bool = False
    imaginary_tolerance: float = 1e-3

    def key(self) -> dict:
        return {"min_length": self.min_length, "delta": self.delta, "kpts": list(self.kpts)}


def thermo_from_modes(energies: np.ndarray, T: float) -> dict:
    """Harmonic U, F, S, Cv per mode-set average (eV, eV/K) from phonon energies ħω (eV), 3 per atom.

    Imaginary (negative) and zero modes are left out.
    """
    w = np.asarray(energies, float).ravel()
    n_per_atom = 3.0
    total = w.size
    w = w[w > 1e-6]
    if T <= 0:
        zpe = 0.5 * np.sum(w) / total * n_per_atom
        return {"U": zpe, "F": zpe, "S": 0.0, "Cv": 0.0}
    x = w / (K_B * T)
    with np.errstate(over="ignore"):
        em1 = np.expm1(x)
        log_term = np.log1p(-np.exp(-x))
        S = K_B * np.sum(x / em1 - log_term)
        U = np.sum(0.5 * w + w / em1)
        F = np.sum(0.5 * w + K_B * T * log_term)
        Cv = K_B * np.sum(np.where(x < 700, x * x * np.exp(np.minimum(x, 700)) / em1**2, 0.0))
    scale = n_per_atom / total
    return {"U": U * scale, "F": F * scale, "S": S * scale, "Cv": Cv * scale}


@dataclass
class Vibrations:
    """Vibrational thermodynamics of one structure, per atom (eV, eV/K)."""

    #: (volume per atom Å³, static energy per atom eV, phonon energies eV) for each volume
    volumes: list = field(default_factory=list)
    quasi_harmonic: bool = False
    #: fraction of imaginary modes, at the worst volume
    imaginary_fraction: float = 0.0

    def _G(self, T: float) -> tuple[float, float]:
        """(G per atom including the static energy, equilibrium volume per atom) at T."""
        if not self.quasi_harmonic or len(self.volumes) == 1:
            v, e0, w = self.volumes[len(self.volumes) // 2] if self.quasi_harmonic else self.volumes[0]
            return e0 + thermo_from_modes(w, T)["F"], v
        V = np.array([v for v, _, _ in self.volumes])
        F = np.array([e0 + thermo_from_modes(w, T)["F"] for _, e0, w in self.volumes])
        c = np.polyfit(V, F, 3)
        roots = np.roots(np.polyder(c))
        cand = [r.real for r in roots if abs(r.imag) < 1e-12 and V.min() * 0.95 <= r.real <= V.max() * 1.05]
        cand = [r for r in cand if np.polyval(np.polyder(c, 2), r) > 0] or [V[np.argmin(F)]]
        v = min(cand, key=lambda r: np.polyval(c, r))
        return float(np.polyval(c, v)), float(v)

    def static(self) -> float:
        """The static (0 K, no zero-point) energy per atom at the reference volume."""
        return self.volumes[len(self.volumes) // 2][1] if self.quasi_harmonic else self.volumes[0][1]

    def H(self, T: float) -> float:
        """Enthalpy per atom relative to the static energy: zero-point and thermal vibration (and expansion)."""
        if not self.quasi_harmonic:
            return thermo_from_modes(self.volumes[0][2], T)["U"]
        return self._G(T)[0] + T * self.S(T) - self.static()

    def S(self, T: float) -> float:
        if not self.quasi_harmonic:
            return thermo_from_modes(self.volumes[0][2], T)["S"]
        h = 1.0
        return -(self._G(T + h)[0] - self._G(T - h)[0]) / (2 * h)

    def Cp(self, T: float) -> float:
        """Cp per atom (eV/K); harmonic: Cv."""
        if not self.quasi_harmonic:
            return thermo_from_modes(self.volumes[0][2], T)["Cv"]
        h = 2.0
        return -T * (self._G(T + h)[0] - 2 * self._G(T)[0] + self._G(T - h)[0]) / (h * h)

    def volume(self, T: float) -> float:
        return self._G(T)[1]


def _repeat(atoms, min_length: float) -> tuple[int, int, int]:
    return tuple(max(1, math.ceil(min_length / float(np.linalg.norm(v)))) for v in atoms.cell)


def _modes(calc, atoms, options: PhononOptions) -> np.ndarray:
    """Phonon energies (eV) on the Monkhorst–Pack mesh, Γ excluded."""
    from ase.dft.kpoints import monkhorst_pack
    from ase.phonons import Phonons

    with tempfile.TemporaryDirectory() as tmp:
        ph = Phonons(
            atoms, calc, supercell=_repeat(atoms, options.min_length), delta=options.delta, name=f"{tmp}/phonon"
        )
        ph.run()
        ph.read(acoustic=True)
        k = monkhorst_pack(options.kpts)
        k = k[np.linalg.norm(k, axis=1) > 1e-9]
        return np.asarray(ph.band_structure(k, verbose=False), float)


def _cached_modes(model, atoms, options: PhononOptions) -> np.ndarray:
    cache = getattr(model, "cache", None)
    key = RelaxationCache.key("phonons", model.name, structure_hash(atoms), options.key())
    hit = cache.get(key) if cache is not None else None
    if hit is not None:
        return np.array(hit["modes"], float)
    w = _modes(model.calc, atoms, options)
    if cache is not None:
        cache.put(key, {"modes": w.tolist()})
    return w


def _imaginary(w: np.ndarray) -> float:
    return float(np.mean(np.asarray(w) < -1e-4))  # eV; noise near Γ is smaller


def harmonic(model, atoms, options: PhononOptions | None = None) -> Vibrations:
    """Harmonic phonons of a relaxed structure (``atoms``), with ``model``'s ASE calculator."""
    options = options or PhononOptions()
    if not hasattr(model, "calc"):
        raise TypeError("phonons need an ASE calculator: build the model with mlip.from_ase")
    a = atoms.copy()
    a.calc = model.calc
    e0 = float(a.get_potential_energy()) / len(a)
    w = _cached_modes(model, atoms, options)
    return Vibrations([(atoms.get_volume() / len(atoms), e0, w)], False, _imaginary(w))


def quasi_harmonic(model, atoms, options: PhononOptions | None = None, relax: RelaxPolicy | None = None) -> Vibrations:
    """Phonons at each isotropic strain of ``options.strains``, positions relaxed at fixed cell."""
    options = options or PhononOptions(quasi_harmonic=True)
    if not hasattr(model, "calc"):
        raise TypeError("phonons need an ASE calculator: build the model with mlip.from_ase")
    policy = relax or RelaxPolicy()
    policy = RelaxPolicy(
        mode=policy.mode,
        fmax=policy.fmax,
        steps=policy.steps,
        cell=False,
        optimizer=policy.optimizer,
        symprec=policy.symprec,
    )
    vols, worst = [], 0.0
    for eps in sorted(options.strains):
        strained = atoms.copy()
        strained.set_cell(atoms.cell * (1 + eps), scale_atoms=True)
        rel = model.relax([strained], policy)[0]
        relaxed = rel.atoms()
        w = _cached_modes(model, relaxed, options)
        worst = max(worst, _imaginary(w))
        vols.append((relaxed.get_volume() / len(relaxed), rel.energy_per_atom, w))
    return Vibrations(vols, True, worst)
