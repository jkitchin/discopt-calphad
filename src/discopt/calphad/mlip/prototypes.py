"""Crystal prototypes, and the map from a TDB phase's sublattices to their Wyckoff sites.

A :class:`Prototype` is a space group, named Wyckoff sites with one
representative position each, and the cell shape. ASE's ``crystal`` generates
the equivalent positions, so a prototype is a few lines of data.

A :class:`PhaseStructure` says which Wyckoff sites make up each sublattice of a
CALPHAD phase. A sublattice that holds only vacancies (the interstitial
``(VA)`` sublattice of FCC_A1, BCC_A2, ...) maps to ``None`` and is not built.
Decorating every sublattice with one constituent gives a CEF endmember. A VA
constituent leaves its sites empty.

Phase names are not standardized across databases (``LAVES_C15`` and
``C15_LAVES``; ``ALCU_THETA`` for C16). :data:`PHASES` holds the common
generic names, and every function that takes a phase also takes a
:class:`PhaseStructure` from the user: a registered prototype with a
sublattice map, or any ``ase.Atoms`` with the atoms of each sublattice
(:meth:`PhaseStructure.from_atoms`), for a structure that is not registered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path

import numpy as np

__all__ = ["Prototype", "PhaseStructure", "PROTOTYPES", "PHASES", "phase_structure", "atomic_volume"]


@dataclass(frozen=True)
class Prototype:
    """A crystal structure type: space group, Wyckoff sites and cell shape.

    ``sites`` maps a site label to one representative fractional position.
    ``cell`` is ``(b/a, c/a, alpha, beta, gamma)``. Positions are in ASE's
    default setting of the space group (origin choice 1).
    """

    name: str
    spacegroup: int
    sites: dict[str, tuple[float, float, float]]
    cell: tuple[float, float, float, float, float, float] = (1.0, 1.0, 90.0, 90.0, 90.0)
    description: str = ""
    #: a user structure (see :meth:`PhaseStructure.from_atoms`): ``ase.Atoms`` whose array
    #: ``"wyckoff"`` names each atom's site; used instead of the space group when set
    template: object = field(default=None, compare=False, repr=False)

    def build(self, decoration: dict[str, str], volume_per_atom: float | None = None):
        """``ase.Atoms`` with each site label occupied by an element symbol (``"VA"``: empty).

        Every site must be decorated. The cell is scaled to ``volume_per_atom``
        (Å³) when it is given. The array ``"wyckoff"`` holds each atom's site label.
        """
        from ase.spacegroup import crystal

        missing = set(self.sites) - set(decoration)
        if missing:
            raise ValueError(f"{self.name}: no decoration for sites {sorted(missing)}")
        if self.template is not None:
            atoms = self.template.copy()
            site = np.array(atoms.get_array("wyckoff"), dtype=object)
        else:
            labels = list(self.sites)
            ba, ca, al, be, ga = self.cell
            # Build with a placeholder element so empty (VA) sites keep their positions until removed.
            atoms = crystal(
                ["X"] * len(labels),
                basis=[self.sites[k] for k in labels],
                spacegroup=self.spacegroup,
                cellpar=[1.0, ba, ca, al, be, ga],
                primitive_cell=False,
            )
            kinds = atoms.get_array("spacegroup_kinds")
            site = np.array([labels[k] for k in kinds], dtype=object)
            del atoms.arrays["spacegroup_kinds"]
        symbols = [decoration[s] for s in site]
        keep = np.array([s.upper() != "VA" for s in symbols])
        atoms.set_chemical_symbols([s if k else "H" for s, k in zip(symbols, keep)])
        atoms.set_array("wyckoff", site.astype(str))
        atoms = atoms[keep]
        if len(atoms) == 0:
            raise ValueError(f"{self.name}: every site is vacant")
        if volume_per_atom is not None:
            scale = (volume_per_atom * len(atoms) / atoms.get_volume()) ** (1 / 3)
            atoms.set_cell(atoms.cell * scale, scale_atoms=True)
        return atoms

    def multiplicities(self) -> dict[str, int]:
        """Number of atoms on each site in the conventional cell."""
        atoms = self.build({k: "Cu" for k in self.sites})
        labels, counts = np.unique(atoms.get_array("wyckoff"), return_counts=True)
        return {str(k): int(n) for k, n in zip(labels, counts)}


def _hex(c_over_a):
    return (1.0, c_over_a, 90.0, 90.0, 120.0)


#: prototypes by Strukturbericht designation
PROTOTYPES: dict[str, Prototype] = {
    p.name: p
    for p in [
        Prototype("A1", 225, {"4a": (0, 0, 0)}, description="fcc (Cu)"),
        Prototype("A2", 229, {"2a": (0, 0, 0)}, description="bcc (W)"),
        Prototype("A3", 194, {"2c": (1 / 3, 2 / 3, 1 / 4)}, _hex(math.sqrt(8 / 3)), "hcp (Mg)"),
        Prototype("A4", 227, {"8a": (0, 0, 0)}, description="diamond (C)"),
        Prototype("B2", 221, {"1a": (0, 0, 0), "1b": (0.5, 0.5, 0.5)}, description="CsCl"),
        Prototype("L1_2", 221, {"1a": (0, 0, 0), "3c": (0, 0.5, 0.5)}, description="Cu3Au"),
        Prototype(
            "D0_19",
            194,
            {"2c": (1 / 3, 2 / 3, 1 / 4), "6h": (5 / 6, 2 / 3, 1 / 4)},
            _hex(math.sqrt(8 / 3) / 2),
            "Ni3Sn",
        ),
        Prototype("C15", 227, {"8a": (0, 0, 0), "16d": (5 / 8, 5 / 8, 5 / 8)}, description="MgCu2 Laves"),
        Prototype(
            "C14",
            194,
            {"2a": (0, 0, 0), "4f": (1 / 3, 2 / 3, 0.0629), "6h": (0.8305, 0.6610, 1 / 4)},
            _hex(1.64),
            "MgZn2 Laves",
        ),
        Prototype(
            "C36",
            194,
            {
                "4e": (0, 0, 0.09375),
                "4f1": (1 / 3, 2 / 3, 0.84375),
                "4f2": (1 / 3, 2 / 3, 0.125),
                "6g": (0.5, 0, 0),
                "6h": (1 / 6, 1 / 3, 1 / 4),
            },
            _hex(3.28),
            "MgNi2 Laves",
        ),
        Prototype(
            "C16",
            140,
            {"4a": (0, 0, 1 / 4), "8h": (0.158, 0.658, 0)},
            (1.0, 0.803, 90.0, 90.0, 90.0),
            "Al2Cu (theta)",
        ),
        Prototype(
            "D8_b",
            136,
            {
                "2a": (0, 0, 0),
                "4f": (0.3981, 0.3981, 0),
                "8i1": (0.4632, 0.1316, 0),
                "8i2": (0.7376, 0.0653, 0),
                "8j": (0.1823, 0.1823, 0.2524),
            },
            (1.0, 0.518, 90.0, 90.0, 90.0),
            "sigma (CrFe)",
        ),
        # From the AFLOW Encyclopedia of Crystallographic Prototypes
        # (https://aflow.org/prototype-encyclopedia/<label>), with their published
        # coordinates; each prototype is cited by its AFLOW label.
        Prototype(  # A3B_oP16_62_cd_c-001
            "D0_20",
            62,
            {"4c1": (0.511, 0.25, 0.415), "4c2": (0.869, 0.25, 0.555), "8d": (0.674, 0.053, 0.856)},
            (1.114, 0.7278, 90.0, 90.0, 90.0),
            "Al3Ni: Al 4c1 + 8d, Ni 4c2",
        ),
        Prototype(  # AB3_oP16_62_c_cd-001
            "D0_11",
            62,
            {"4c1": (0.39, 0.25, 0.05), "4c2": (0.036, 0.25, 0.852), "8d": (0.186, 0.063, 0.328)},
            (1.326, 0.8886, 90.0, 90.0, 90.0),
            "Fe3C cementite: C 4c1, Fe 4c2 + 8d",
        ),
        Prototype(  # A3B2_hP5_164_ad_d-001
            "D5_13",
            164,
            {"1a": (0, 0, 0), "2d1": (1 / 3, 2 / 3, 0.648), "2d2": (1 / 3, 2 / 3, 0.149)},
            _hex(1.214),
            "Al3Ni2: Al 1a + 2d1, Ni 2d2 (the CALPHAD vacancy site is 1b, not built)",
        ),
        Prototype(  # A3B_tI8_139_ad_b-001
            "D0_22",
            139,
            {"2a": (0, 0, 0), "2b": (0, 0, 0.5), "4d": (0, 0.5, 0.25)},
            (1.0, 2.227, 90.0, 90.0, 90.0),
            "Al3Ti: Al 2a + 4d, Ti 2b",
        ),
        Prototype(  # A3B_tI16_139_cde_e-001
            "D0_23",
            139,
            {"4c": (0, 0.5, 0), "4d": (0, 0.5, 0.25), "4e1": (0, 0, 0.375), "4e2": (0, 0, 0.1189)},
            (1.0, 4.322, 90.0, 90.0, 90.0),
            "Al3Zr: Al 4c + 4d + 4e1, Zr 4e2",
        ),
        Prototype(  # AB3_cF16_225_a_bc-001
            "D0_3",
            225,
            {"4a": (0, 0, 0), "4b": (0.5, 0.5, 0.5), "8c": (0.25, 0.25, 0.25)},
            description="BiF3 (Fe3Al): Bi 4a, F 4b + 8c",
        ),
        Prototype(  # AB_tP2_123_a_d-001
            "L1_0",
            123,
            {"1a": (0, 0, 0), "1d": (0.5, 0.5, 0.5)},
            (1.0, 1.31, 90.0, 90.0, 90.0),
            "CuAu: Au 1a, Cu 1d (primitive cell)",
        ),
        Prototype(  # AB_oP4_51_e_f-001
            "B19",
            51,
            {"2e": (0.25, 0, 0.8125), "2f": (0.25, 0.5, 0.3125)},
            (0.662, 1.021, 90.0, 90.0, 90.0),
            "AuCd: Au 2e, Cd 2f",
        ),
        Prototype(  # AB2_tI6_139_a_e-001
            "C11_b",
            139,
            {"2a": (0, 0, 0), "4e": (0, 0, 0.3353)},
            (1.0, 2.448, 90.0, 90.0, 90.0),
            "MoSi2: Mo 2a, Si 4e",
        ),
        Prototype(  # AB_hP4_194_c_a-001
            "B8_1",
            194,
            {"2a": (0, 0, 0), "2c": (1 / 3, 2 / 3, 0.25)},
            _hex(1.394),
            "NiAs: Ni 2a, As 2c",
        ),
        Prototype(  # A3B5_oC16_65_ah_bej-001
            "Ga3Pt5",
            65,
            {"2a": (0, 0, 0), "2b": (0.5, 0, 0), "4e": (0.25, 0.25, 0), "4h": (0.25, 0, 0.5), "4j": (0, 0.225, 0.5)},
            (0.9264, 0.4916, 90.0, 90.0, 90.0),
            "Ga3Pt5 (Al3Ni5): Ga 2a + 4h, Pt 2b + 4e + 4j",
        ),
    ]
}


@dataclass(frozen=True)
class PhaseStructure:
    """A prototype, and for each sublattice of the phase its Wyckoff sites (``None``: vacancies only)."""

    prototype: Prototype
    sublattices: tuple[tuple[str, ...] | None, ...]
    note: str = field(default="", compare=False)

    def __post_init__(self):
        used = [s for sub in self.sublattices if sub for s in sub]
        unknown = set(used) - set(self.prototype.sites)
        if unknown:
            raise ValueError(f"{self.prototype.name}: unknown sites {sorted(unknown)}")
        if len(set(used)) != len(used) or set(used) != set(self.prototype.sites):
            raise ValueError(f"{self.prototype.name}: every site must belong to exactly one sublattice")

    @classmethod
    def from_atoms(cls, atoms, sublattices, name: str = "custom", note: str = "") -> PhaseStructure:
        """A structure that is not registered: ``ase.Atoms`` and, per sublattice of the phase, its atoms.

        ``sublattices`` has one entry per sublattice of the database phase, in
        its order: the indices of the atoms on that sublattice, or None for a
        sublattice that holds only vacancies. Every atom must be on exactly one
        sublattice. The chemical symbols of ``atoms`` do not matter; each
        endmember decorates the sublattices anew. Give positions and cell as
        relaxed or reasonable as you have them: the cell is rescaled to a
        Vegard volume before relaxation, but not reshaped.
        """
        labels = np.full(len(atoms), "", dtype=object)
        subl = []
        for s, idx in enumerate(sublattices):
            if idx is None:
                subl.append(None)
                continue
            idx = [int(i) for i in idx]
            if any(labels[i] for i in idx):
                raise ValueError(f"an atom of sublattice {s} is already on another sublattice")
            labels[idx] = f"s{s}"
            subl.append((f"s{s}",))
        if any(lab == "" for lab in labels):
            raise ValueError("every atom must belong to a sublattice")
        template = atoms.copy()
        template.set_array("wyckoff", labels.astype(str))
        sites = {lab[0]: (0.0, 0.0, 0.0) for lab in subl if lab}
        return cls(Prototype(name, 0, sites, template=template, description="user structure"), tuple(subl), note)

    @classmethod
    def from_cif(cls, path, sublattices, name: str | None = None, note: str = "") -> PhaseStructure:
        """A structure from a CIF file (ICSD, Materials Project, the AFLOW prototype library, ...).

        ``sublattices`` has one entry per sublattice of the database phase, in
        its order: a list of selectors, or None for a vacancy-only sublattice. A
        selector is a CIF site label (``_atom_site_label``, for example ``"Al2"``)
        or an element symbol (``"Al"``, every site of that element). Every atom
        of the cell must be selected exactly once. Partially occupied sites are
        refused: decide the occupancy first.
        """
        from ase.io import read

        atoms = read(str(path), store_tags=True)
        labels = [str(v) for v in atoms.info.get("_atom_site_label", [])]
        occupancy = atoms.info.get("_atom_site_occupancy")
        if occupancy is not None and any(abs(float(o) - 1.0) > 1e-6 for o in occupancy):
            raise ValueError(f"{path}: partially occupied sites; choose an ordered structure first")
        if "spacegroup_kinds" not in atoms.arrays or not labels:
            raise ValueError(f"{path}: the CIF has no site labels to map onto sublattices")
        kinds = atoms.get_array("spacegroup_kinds")
        site_of = np.array([labels[k] for k in kinds], dtype=object)
        symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
        indices = []
        for selectors in sublattices:
            if selectors is None:
                indices.append(None)
                continue
            selectors = [selectors] if isinstance(selectors, str) else list(selectors)
            mask = np.zeros(len(atoms), bool)
            for sel in selectors:
                hit = (site_of == sel) | (symbols == sel)
                if not hit.any():
                    raise ValueError(f"{path}: no site or element {sel!r}; sites are {labels}")
                mask |= hit
            indices.append(list(np.flatnonzero(mask)))
        return cls.from_atoms(atoms, indices, name=name or Path(str(path)).stem, note=note)

    def site_counts(self) -> list[int]:
        """Atoms per sublattice in the conventional cell (0 for a vacancy-only sublattice)."""
        m = self.prototype.multiplicities()
        return [sum(m[s] for s in sub) if sub else 0 for sub in self.sublattices]

    def check_ratios(self, site_ratios, constituents=None, name: str = "") -> None:
        """Raise unless the prototype's site counts are proportional to the phase's site ratios.

        Sublattices mapped to ``None`` must hold only VA (when ``constituents`` is given).
        """
        if len(site_ratios) != len(self.sublattices):
            raise ValueError(
                f"{name}: the phase has {len(site_ratios)} sublattices, the structure map has {len(self.sublattices)}"
            )
        counts = self.site_counts()
        built = [(a, n) for a, n, sub in zip(site_ratios, counts, self.sublattices) if sub]
        if constituents is not None:
            for sub, cons in zip(self.sublattices, constituents):
                if sub is None and tuple(cons) != ("VA",):
                    raise ValueError(f"{name}: a sublattice with {cons} is mapped to no sites")
        ref_a, ref_n = built[0]
        for a, n in built[1:]:
            if not math.isclose(a / ref_a, n / ref_n, rel_tol=1e-3):
                raise ValueError(
                    f"{name}: site ratios {tuple(site_ratios)} do not match {self.prototype.name} "
                    f"site counts {tuple(counts)}"
                )

    def formula_units(self) -> float:
        """Formula units per conventional cell, for the phase's site ratios (set by :meth:`check_ratios`)."""
        counts = [n for n, sub in zip(self.site_counts(), self.sublattices) if sub]
        return float(reduce(math.gcd, counts))

    def build(self, endmember, volume_per_atom: float | None = None):
        """The structure of an endmember: one constituent per sublattice, e.g. ``("AL", "CU", "VA")``."""
        if len(endmember) != len(self.sublattices):
            raise ValueError(f"{self.prototype.name}: {len(self.sublattices)} sublattices, endmember {endmember}")
        decoration = {}
        for sub, c in zip(self.sublattices, endmember):
            if sub is None:
                if c.upper() != "VA":
                    raise ValueError(f"{self.prototype.name}: {c} on a vacancy-only sublattice")
                continue
            for s in sub:
                decoration[s] = c.capitalize() if c.upper() != "VA" else "VA"
        return self.prototype.build(decoration, volume_per_atom)


def _ps(proto, *sublattices, note=""):
    return PhaseStructure(PROTOTYPES[proto], tuple(tuple(s) if s else None for s in sublattices), note)


#: generic TDB phase names. A database may use other names: pass a PhaseStructure for those.
PHASES: dict[str, PhaseStructure] = {
    "FCC_A1": _ps("A1", ["4a"], None),
    "BCC_A2": _ps("A2", ["2a"], None),
    "HCP_A3": _ps("A3", ["2c"], None),
    "DIAMOND_A4": _ps("A4", ["8a"]),
    "B2_BCC": _ps("B2", ["1a"], ["1b"], None),
    "L12_FCC": _ps("L1_2", ["3c"], ["1a"], None),
    "D019_HCP": _ps("D0_19", ["6h"], ["2c"], None),
    "LAVES_C15": _ps("C15", ["16d"], ["8a"]),
    "C15_LAVES": _ps("C15", ["16d"], ["8a"]),
    "LAVES_C14": _ps("C14", ["2a", "6h"], ["4f"]),
    "C14_LAVES": _ps("C14", ["2a", "6h"], ["4f"]),
    "LAVES_C36": _ps("C36", ["4f2", "6g", "6h"], ["4e", "4f1"]),
    "C36_LAVES": _ps("C36", ["4f2", "6g", "6h"], ["4e", "4f1"]),
    "SIGMA": _ps("D8_b", ["2a", "8i2"], ["4f"], ["8i1", "8j"], note="the 10:4:16 sigma model"),
}


def phase_structure(name_or_structure, *sublattices) -> PhaseStructure:
    """A :class:`PhaseStructure` from a registered phase name, or from a prototype name and sublattice sites.

    ``phase_structure("FCC_A1")`` looks up :data:`PHASES`.
    ``phase_structure("C16", ["8h"], ["4a"])`` maps a phase's two sublattices
    onto the C16 sites (``None`` for a vacancy-only sublattice).
    """
    if isinstance(name_or_structure, PhaseStructure):
        return name_or_structure
    key = str(name_or_structure).upper() if not sublattices else str(name_or_structure)
    if not sublattices:
        if key not in PHASES:
            raise KeyError(
                f"no structure registered for phase {key}; pass one, e.g. "
                f"mlip.phase_structure('C16', ['8h'], ['4a']). Registered: {sorted(PHASES)}"
            )
        return PHASES[key]
    if key not in PROTOTYPES:
        raise KeyError(f"unknown prototype {key}; known: {sorted(PROTOTYPES)}")
    return _ps(key, *sublattices)


def atomic_volume(symbol: str) -> float:
    """Volume per atom (Å³) of an element in its ASE reference structure, for initial cells (Vegard's law)."""
    from ase.data import atomic_numbers, covalent_radii, reference_states

    z = atomic_numbers[symbol.capitalize()]
    ref = reference_states[z] or {}
    a = ref.get("a")
    per = {"fcc": 4, "bcc": 2, "sc": 1, "diamond": 8}
    if ref.get("symmetry") in per and a:
        return a**3 / per[ref["symmetry"]]
    if ref.get("symmetry") == "hcp" and a:
        return math.sqrt(3) / 2 * a * a * a * ref.get("c/a", math.sqrt(8 / 3)) / 2
    r = covalent_radii[z]
    return 4 / 3 * math.pi * r**3 / 0.74
