"""Local NLP adapter: POUNCE through discopt's evaluator.

The restricted equilibrium problem needs a *local* solve with multipliers.
``Model.solve`` runs global branch-and-bound on the bilinear restricted problem
and returns no duals from that route. So we compile the model with discopt's
evaluator and call its local NLP backend directly, as discopt-speciation does.

Both imports are private discopt API. They are kept in this one module so that
a future public route has only one place to change (discopt#1247 asks for
primal-dual warm starts and a KKT residual on ``Model.solve``).
"""

from __future__ import annotations

import numpy as np
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.solvers.nlp_backend import get_nlp_solver

__all__ = ["local_nlp_solve", "model_evaluator"]


def model_evaluator(model):
    """discopt's compiled evaluator (objective, gradient, Hessian, constraints)."""
    return make_evaluator(model)


def local_nlp_solve(model, x0, options: dict | None = None):
    """Solve ``model`` locally from ``x0``.

    ``x0`` lists the variables in declaration order. The returned ``NLPResult``
    has ``status``, ``x``, ``objective``, ``multipliers`` (one per constraint, in
    declaration order, with the convention L = f + λ·(g − rhs)) and ``iterations``.
    """
    ev = make_evaluator(model)
    lb, ub = ev.variable_bounds
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != lb.shape:
        raise ValueError(f"x0 has {x0.size} entries, model has {lb.size} variables")
    return get_nlp_solver("auto")(ev, np.clip(x0, lb, ub), options=options or {})
