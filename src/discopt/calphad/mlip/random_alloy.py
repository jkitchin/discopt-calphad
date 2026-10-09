"""Random decorations of a lattice, for mixing enthalpies averaged over configurations.

An SQS (:mod:`~discopt.calphad.mlip.sqs`) is one structure optimized to look
random. Here instead, ``count`` decorations are drawn at random, each with the
composition exact, and the property is averaged over them. The draws are
independent, so their spread has clean statistics: the standard error of the
mean is s/√n, and more draws shrink it. No SQS tool is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

__all__ = ["RandomOptions", "supercell", "generate"]


@dataclass(frozen=True)
class RandomOptions:
    """Settings for random decorations.

    ``count`` decorations per composition, in a supercell of the conventional
    cell with between ``min_atoms`` and ``max_atoms`` atoms that holds the
    composition exactly (the most nearly cubic such repeat is used). ``seed``
    makes the draws reproducible.
    """

    count: int = 8
    min_atoms: int = 16
    max_atoms: int = 64
    seed: int = 0


def supercell(n_cell: int, n_mixing: int, y: float, options: RandomOptions) -> tuple[int, int, int]:
    """Repeats (a, b, c) of a cell with ``n_cell`` atoms and ``n_mixing`` mixing sites that hold ``y`` exactly."""
    best = None
    for rep in product(range(1, 7), repeat=3):
        cells = rep[0] * rep[1] * rep[2]
        n = n_cell * cells
        k = y * n_mixing * cells
        if not options.min_atoms <= n <= options.max_atoms or abs(k - round(k)) > 1e-9:
            continue
        key = (max(rep) / min(rep), n)  # as cubic as possible, then as small as possible
        if best is None or key < best[0]:
            best = (key, rep)
    if best is None:
        raise ValueError(
            f"no supercell of {options.min_atoms}-{options.max_atoms} atoms holds y = {y:g} exactly; "
            "change the composition grid or RandomOptions"
        )
    return best[1]


def generate(
    ps, mixing: int, pair, fixed, y: float, volume_per_atom: float, options: RandomOptions, offset: int = 0
) -> list:
    """``options.count`` random decorations with site fraction ``y`` of ``pair[0]`` on sublattice ``mixing``.

    ``fixed`` gives the constituent of every other sublattice. ``offset`` shifts
    the random stream, so different compositions get independent draws.
    """
    endmember = [pair[0] if s == mixing else fixed[s] for s in range(len(ps.sublattices))]
    cell = ps.build(endmember, volume_per_atom)
    sites = set(ps.sublattices[mixing])
    on_mix = np.array([w in sites for w in cell.get_array("wyckoff")])
    rep = supercell(len(cell), int(on_mix.sum()), y, options)
    big = cell.repeat(rep)
    mix_idx = np.flatnonzero(np.tile(on_mix, rep[0] * rep[1] * rep[2]))
    n_first = int(round(y * len(mix_idx)))
    rng = np.random.default_rng([options.seed, offset])
    out = []
    for _ in range(options.count):
        atoms = big.copy()
        symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
        chosen = rng.permutation(mix_idx)
        symbols[chosen[:n_first]] = pair[0].capitalize()
        symbols[chosen[n_first:]] = pair[1].capitalize()
        atoms.set_chemical_symbols(list(symbols))
        out.append(atoms)
    return out
