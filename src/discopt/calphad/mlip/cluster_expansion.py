"""Cluster expansions, fitted by icet, for mixing enthalpies at any composition.

This module only adapts icet, with no cluster-expansion code of its own:

1. icet enumerates the derivative superstructures of the parent lattice (every
   decoration of supercells up to ``max_size`` primitive cells);
2. the energy model relaxes them, and each gives a mixing energy per atom
   relative to the relaxed endmembers;
3. icet's trainstation fits the effective cluster interactions, with k-fold
   cross validation;
4. the expansion predicts the mixing enthalpy of the ideal random alloy at each
   grid composition, from icet's random-alloy cluster vector (an infinite
   supercell with random occupation, so no sampling noise).

The configurational part of σ is the cross-validation RMSE, the expected error
of a prediction for a structure the fit has not seen. Unlike SQS, any
composition can be predicted: the grid need not fit a supercell.

Structures that leave the parent lattice when relaxed are left out of the
training set (they are reported in :attr:`MLIPResult.flagged`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discopt.calphad.mlip.sqs import _active, _cluster_space, _icet, _template

__all__ = ["ClusterExpansionOptions", "ClusterExpansionFit", "training_structures", "fit", "random_alloy"]


@dataclass(frozen=True)
class ClusterExpansionOptions:
    """Settings for a cluster expansion.

    ``cutoffs`` are the pair, triplet, ... cutoffs as multiples of the
    nearest-neighbour distance (as for SQS). The training set is every
    derivative superstructure of up to ``max_size`` primitive cells, or a
    random subset of ``max_structures`` of them (the two endmembers always
    included). ``fit_method`` is any trainstation method (``"least-squares"``,
    ``"ardr"``, ``"lasso"``, ...), validated with ``n_splits``-fold cross
    validation. ``seed`` fixes the subset and the folds.
    """

    cutoffs: tuple[float, ...] = (2.1, 1.5)
    max_size: int = 4
    max_structures: int | None = 40
    fit_method: str = "least-squares"
    n_splits: int = 5
    seed: int = 0


@dataclass
class ClusterExpansionFit:
    """A fitted cluster expansion of the mixing energy (eV per atom) of one phase."""

    #: the icet ClusterExpansion
    expansion: object
    #: the icet ClusterSpace, and the symbol of its mixing sublattice
    cluster_space: object
    sublattice: str
    pair: tuple[str, str]
    #: training structures (ideal lattice), their site fractions of pair[0], and mixing energies (eV/atom)
    structures: list = field(default_factory=list)
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    energies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: cross-validation RMSE and training RMSE, eV per atom
    rmse_validation: float = 0.0
    rmse_train: float = 0.0

    @property
    def n_parameters(self) -> int:
        return len(self.expansion.parameters)

    def predict(self, atoms) -> float:
        """Mixing energy (eV per atom) of a decoration of the parent lattice."""
        return float(self.expansion.predict(atoms))

    def random_alloy(self, y: float) -> float:
        """Mixing energy (eV per atom) of the ideal random alloy at site fraction ``y`` of ``pair[0]``."""
        return random_alloy(self, y)

    def summary(self) -> str:
        return (
            f"cluster expansion: {self.n_parameters} parameters, {len(self.structures)} training structures, "
            f"CV RMSE {self.rmse_validation * 1000:.2f} meV/atom, train RMSE {self.rmse_train * 1000:.2f} meV/atom"
        )


def _site_fraction(atoms, pair, n_mixing: int) -> float:
    return atoms.get_chemical_symbols().count(pair[0].capitalize()) / n_mixing


def training_structures(ps, mixing: int, pair, fixed, volume_per_atom: float, options: ClusterExpansionOptions):
    """(cluster space, mixing-sublattice symbol, enumerated structures on the ideal lattice).

    The structures are scaled to ``volume_per_atom``; the caller rescales each
    to its own (Vegard) volume before relaxing.
    """
    from icet.tools import enumerate_structures

    template, allowed = _template(ps, mixing, pair, fixed, volume_per_atom)
    cs, _ = _cluster_space(template, allowed, options.cutoffs)
    symbol, _ = _active(cs, pair)
    prim = cs.primitive_structure.copy()
    prim_allowed = [list(s) for s in cs.chemical_symbols]
    prim.set_chemical_symbols([s[0] for s in prim_allowed])
    found = list(enumerate_structures(prim, range(1, options.max_size + 1), prim_allowed))
    n_prim_mix = sum(len(s) > 1 for s in prim_allowed)
    ends = [s for s in found if _site_fraction(s, pair, n_prim_mix * len(s) // len(prim)) in (0.0, 1.0)]
    # one of each endmember, the smallest
    ends = [
        min((s for s in ends if _site_fraction(s, pair, n_prim_mix * len(s) // len(prim)) == f), key=len)
        for f in (1.0, 0.0)
    ]
    rest = [s for s in found if _site_fraction(s, pair, n_prim_mix * len(s) // len(prim)) not in (0.0, 1.0)]
    if options.max_structures is not None and len(rest) + 2 > options.max_structures:
        rng = np.random.default_rng(options.seed)
        keep = sorted(rng.choice(len(rest), options.max_structures - 2, replace=False))
        rest = [rest[k] for k in keep]
    return cs, symbol, ends + rest


def fit(cs, symbol: str, pair, structures, energies, options: ClusterExpansionOptions) -> ClusterExpansionFit:
    """Fit a cluster expansion to mixing energies (eV/atom) of ``structures`` on the ideal lattice."""
    icet, _ = _icet()
    from trainstation import CrossValidationEstimator

    sc = icet.StructureContainer(cs)
    for k, (atoms, e) in enumerate(zip(structures, energies)):
        sc.add_structure(atoms, user_tag=str(k), properties={"mixing_energy": float(e)})
    A, b = sc.get_fit_data(key="mixing_energy")
    if len(b) <= A.shape[1]:
        raise ValueError(
            f"{len(b)} training structures for {A.shape[1]} cluster-expansion parameters; "
            "raise ClusterExpansionOptions.max_size or max_structures, or shorten the cutoffs"
        )
    cve = CrossValidationEstimator(
        (A, b), fit_method=options.fit_method, n_splits=min(options.n_splits, len(b)), seed=options.seed
    )
    cve.validate()
    cve.train()
    ce = icet.ClusterExpansion(cs, cve.parameters, metadata=cve.summary)
    n_mix = [sum(len(s) > 1 for s in cs.chemical_symbols) * len(a) // len(cs.primitive_structure) for a in structures]
    return ClusterExpansionFit(
        expansion=ce,
        cluster_space=cs,
        sublattice=symbol,
        pair=(pair[0], pair[1]),
        structures=list(structures),
        y=np.array([_site_fraction(a, pair, n) for a, n in zip(structures, n_mix)]),
        energies=np.asarray(energies, float),
        rmse_validation=float(cve.rmse_validation),
        rmse_train=float(cve.rmse_train),
    )


def random_alloy(ce: ClusterExpansionFit, y: float) -> float:
    """Mixing energy (eV/atom) of the random alloy at site fraction ``y``, from icet's random-alloy cluster vector."""
    from icet.tools.structure_generation import _get_sqs_cluster_vector

    a, b = (p.capitalize() for p in ce.pair)
    cv = _get_sqs_cluster_vector(ce.cluster_space, {ce.sublattice: {a: float(y), b: 1.0 - float(y)}})
    return float(np.dot(cv, ce.expansion.parameters))
