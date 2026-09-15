"""Plot helpers that do not need matplotlib."""

import numpy as np

from discopt.calphad.plot import lower_convex_hull


def test_hull_ignores_higher_points_at_shared_x():
    # two "phases" sampled on the same grid: the upper one must not add a vertical edge at x = 1
    x = np.linspace(0.0, 1.0, 11)
    lower, upper = (x - 0.5) ** 2, (x - 0.5) ** 2 + 1.0
    hx, hg = lower_convex_hull(np.concatenate([x, x]), np.concatenate([upper, lower]))
    np.testing.assert_allclose(hx, x)
    np.testing.assert_allclose(hg, lower)
