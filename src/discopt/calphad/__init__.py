"""discopt.calphad: certified CALPHAD phase equilibrium.

POUNCE solves the restricted Gibbs minimization locally. discopt's global solver
prices each phase. Composition sets are added until no phase destabilizes, and
the result is certified to a stated tolerance. TDB databases are read with
:class:`ThermoDatabase`, which keeps parameters symbolic (SymPy). Plotting is in
:mod:`discopt.calphad.plot`, which needs matplotlib.
"""

from discopt.calphad.database import (
    DatabaseValue,
    PhaseNotApplicable,
    ThermoDatabase,
    UnsupportedModelError,
)
from discopt.calphad.equilibrium import (
    CERTIFY_MODES,
    DOMAIN_FLOOR,
    R_GAS,
    Certificate,
    CertificateLevel,
    CompositionSet,
    EquilibriumResult,
    PricingResult,
    RestrictedResult,
    certify,
    equilibrium,
    initial_set,
    is_duplicate,
    price_phase,
    price_phase_local,
    price_phases,
    solve_restricted,
    tangent_distance,
)
from discopt.calphad.models import BinarySolution, RegularSolution, StoichiometricPhase, TPoly
from discopt.calphad.ops import EXPR, INTERVAL, NUM, Interval
from discopt.calphad.symbolic import PiecewiseT, TDBSyntaxError

__version__ = "0.0.1"

__all__ = [
    "BinarySolution",
    "CERTIFY_MODES",
    "Certificate",
    "CertificateLevel",
    "CompositionSet",
    "DOMAIN_FLOOR",
    "DatabaseValue",
    "EXPR",
    "EquilibriumResult",
    "INTERVAL",
    "Interval",
    "NUM",
    "PhaseNotApplicable",
    "PiecewiseT",
    "PricingResult",
    "R_GAS",
    "RegularSolution",
    "RestrictedResult",
    "StoichiometricPhase",
    "TDBSyntaxError",
    "TPoly",
    "ThermoDatabase",
    "UnsupportedModelError",
    "certify",
    "equilibrium",
    "initial_set",
    "is_duplicate",
    "price_phase",
    "price_phase_local",
    "price_phases",
    "solve_restricted",
    "tangent_distance",
]
