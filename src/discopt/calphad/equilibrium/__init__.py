"""Equilibrium by restricted solves, global pricing, and certification."""

from discopt.calphad.equilibrium.certificate import Certificate, CertificateLevel, certify
from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, R_GAS, tangent_distance
from discopt.calphad.equilibrium.loop import (
    CERTIFY_MODES,
    EquilibriumResult,
    IterationRecord,
    equilibrium,
    initial_set,
    is_duplicate,
)
from discopt.calphad.equilibrium.pricing import (
    PricingResult,
    boundary_lower_bound,
    price_phase,
    price_phase_local,
    price_phases,
)
from discopt.calphad.equilibrium.restricted import CompositionSet, RestrictedResult, solve_restricted

__all__ = [
    "CERTIFY_MODES",
    "Certificate",
    "CertificateLevel",
    "CompositionSet",
    "DOMAIN_FLOOR",
    "EquilibriumResult",
    "IterationRecord",
    "PricingResult",
    "R_GAS",
    "RestrictedResult",
    "boundary_lower_bound",
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
