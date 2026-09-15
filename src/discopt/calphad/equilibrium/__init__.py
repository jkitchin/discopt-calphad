"""Equilibrium by restricted solves, global pricing, and certification."""

from discopt.calphad.equilibrium.certificate import Certificate, CertificateLevel, certify
from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, tangent_distance
from discopt.calphad.equilibrium.loop import EquilibriumResult, IterationRecord, equilibrium, is_duplicate
from discopt.calphad.equilibrium.pricing import PricingResult, boundary_lower_bound, price_phase, price_phases
from discopt.calphad.equilibrium.restricted import CompositionSet, RestrictedResult, solve_restricted

__all__ = [
    "Certificate",
    "CertificateLevel",
    "CompositionSet",
    "DOMAIN_FLOOR",
    "EquilibriumResult",
    "IterationRecord",
    "PricingResult",
    "RestrictedResult",
    "boundary_lower_bound",
    "certify",
    "equilibrium",
    "is_duplicate",
    "price_phase",
    "price_phases",
    "solve_restricted",
    "tangent_distance",
]
