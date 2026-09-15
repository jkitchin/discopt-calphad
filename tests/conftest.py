import pytest

from discopt.calphad import RegularSolution, equilibrium

EPS = 1e-5


@pytest.fixture(scope="session")
def gap_phase():
    return RegularSolution("LIQ", omega=3.0)


@pytest.fixture(scope="session")
def gap_result(gap_phase):
    return equilibrium(gap_phase, x=0.5, eps=EPS)


@pytest.fixture(scope="session")
def convex_phase():
    return RegularSolution("LIQ", omega=1.5)


@pytest.fixture(scope="session")
def convex_result(convex_phase):
    return equilibrium(convex_phase, x=0.5, eps=EPS)
