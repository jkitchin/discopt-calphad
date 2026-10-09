"""CALPHAD data from any ASE calculator (issue #5).

A machine-learned potential (UMA, MACE, MTP, ...), a classical one (EAM, EMT),
or any other ASE calculator produces *data*, not phase models::

    structures --calculator--> energies --> Dataset --fit()--> ThermoDatabase --> certified equilibria

A calculator is not used as a phase model inside ``equilibrium()``: a
certificate needs every phase's Gibbs energy as an expression with interval
bounds, and a neural network has no such form. The certificate therefore
covers the fitted model. The calculator's accuracy enters only through the σ
of each data point.

Example::

    from ase.calculators.emt import EMT
    from discopt.calphad import databases, mlip

    model = mlip.from_ase(EMT(), name="EMT")
    data = mlip.dataset(model, databases.load("alcuy_zhang"), ("AL", "CU"),
                        compounds={"ALCU_THETA": ["AL:CU"]},
                        structures={"ALCU_THETA": ("C16", ["8h"], ["4a"])},
                        mixing={"FCC_A1": [0.25, 0.5, 0.75]},
                        model_sigma=2.0)  # kJ/mol; SQS values add their pooled spread

Nothing here imports a model package, and ``ase`` itself is imported only
when a structure is built (``pip install discopt-calphad[mlip]``; add
``[sqs]`` for icet).
"""

from discopt.calphad.mlip.active import (
    ActiveLearningResult,
    active_learning,
    candidates,
    compute_arguments,
    suggest,
)
from discopt.calphad.mlip.calculator import (
    EV_TO_J_PER_MOL,
    ASEModel,
    EnergyModel,
    Relaxation,
    RelaxationCache,
    RelaxPolicy,
    from_ase,
    structure_hash,
)
from discopt.calphad.mlip.cluster_expansion import ClusterExpansionFit, ClusterExpansionOptions
from discopt.calphad.mlip.committee import Committee, committee
from discopt.calphad.mlip.data import APPROXIMATION, MLIPResult, compute, dataset, reference_energies
from discopt.calphad.mlip.liquid import LiquidOptions, LiquidResult
from discopt.calphad.mlip.monte_carlo import MonteCarloOptions, SROResult
from discopt.calphad.mlip.phonons import PhononOptions, Vibrations
from discopt.calphad.mlip.prototypes import (
    PHASES,
    PROTOTYPES,
    PhaseStructure,
    Prototype,
    atomic_volume,
    phase_structure,
)
from discopt.calphad.mlip.random_alloy import RandomOptions
from discopt.calphad.mlip.sqs import SQSOptions

__all__ = [
    "APPROXIMATION",
    "ASEModel",
    "ActiveLearningResult",
    "active_learning",
    "candidates",
    "compute_arguments",
    "suggest",
    "ClusterExpansionFit",
    "ClusterExpansionOptions",
    "Committee",
    "EV_TO_J_PER_MOL",
    "EnergyModel",
    "LiquidOptions",
    "LiquidResult",
    "MLIPResult",
    "MonteCarloOptions",
    "PHASES",
    "PROTOTYPES",
    "PhaseStructure",
    "PhononOptions",
    "Prototype",
    "RelaxPolicy",
    "Relaxation",
    "RandomOptions",
    "RelaxationCache",
    "SQSOptions",
    "SROResult",
    "Vibrations",
    "atomic_volume",
    "committee",
    "compute",
    "dataset",
    "from_ase",
    "phase_structure",
    "reference_energies",
    "structure_hash",
]
