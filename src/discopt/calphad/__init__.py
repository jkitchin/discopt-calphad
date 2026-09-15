"""discopt.calphad: certified CALPHAD phase equilibrium.

POUNCE solves the restricted Gibbs minimization locally. discopt's global solver
prices each phase. Composition sets are added until no phase destabilizes, and
the result is certified to a stated tolerance. Plotting is in
:mod:`discopt.calphad.plot`, which needs matplotlib.
"""

from discopt.calphad.equilibrium import (
    DOMAIN_FLOOR,
    Certificate,
    CertificateLevel,
    CompositionSet,
    EquilibriumResult,
    PricingResult,
    RestrictedResult,
    certify,
    equilibrium,
    is_duplicate,
    price_phase,
    price_phases,
    solve_restricted,
    tangent_distance,
)
from discopt.calphad.models import RegularSolution
from discopt.calphad.ops import EXPR, INTERVAL, NUM, Interval

__version__ = "0.0.1"

__all__ = [
    "Certificate",
    "CertificateLevel",
    "CompositionSet",
    "DOMAIN_FLOOR",
    "EXPR",
    "EquilibriumResult",
    "INTERVAL",
    "Interval",
    "NUM",
    "PricingResult",
    "RegularSolution",
    "RestrictedResult",
    "certify",
    "equilibrium",
    "is_duplicate",
    "price_phase",
    "price_phases",
    "solve_restricted",
    "tangent_distance",
]
