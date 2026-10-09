"""Local pricing and the certify= modes: none, final, every; certifying a result afterwards."""

import numpy as np
import pytest
from conftest import EPS
from test_cef import _recip_phase

from discopt.calphad import (
    CERTIFY_MODES,
    BinarySolution,
    CertificateLevel,
    TPoly,
    equilibrium,
    price_phase_local,
    price_phases,
    tangent_distance,
)
from discopt.calphad.equilibrium import pricing
from discopt.calphad.ops import NUM

_price_phase_local = pricing.price_phase_local


def test_local_pricing_matches_dense_grid(gap_phase):
    mu = np.array([gap_phase.gibbs([0.5], NUM)] * 2)  # tangent at x = 0.5
    ys = np.linspace(0, 1, 400_001)
    grid = float(np.min(tangent_distance(gap_phase, [ys], mu, NUM)))
    p = price_phase_local(gap_phase, mu)
    assert p.lb is None and p.bound_source == "local" and not p.certified(EPS)
    assert p.destabilizing(EPS)
    assert p.ub == pytest.approx(grid, abs=1e-8)


@pytest.mark.parametrize("mu", [(-2.0, -1.5), (-6.0, -1.0)])
def test_local_pricing_two_variable_cef_phase(mu):
    ph = _recip_phase()
    mu = np.array(mu)
    u = [v.ravel() for v in np.meshgrid(np.linspace(0, 1, 801), np.linspace(0, 1, 801))]
    grid = float(np.min(tangent_distance(ph, u, mu, NUM, 700.0)))
    p = price_phase_local(ph, mu, T=700.0)
    assert p.ub <= grid + 1e-9
    assert p.ub == pytest.approx(grid, abs=1e-4)


def test_price_phases_rejects_unknown_method(gap_phase):
    with pytest.raises(ValueError, match="method"):
        price_phases([gap_phase], [0.0, 0.0], method="grid")
    with pytest.raises(ValueError, match="certify"):
        equilibrium(gap_phase, x=0.5, certify="always")


def test_final_mode_prices_globally_only_at_convergence(gap_result, gap_phase):
    # the default: local pricing adds the second set, global pricing runs once, at the end
    first, last = gap_result.history
    assert all(p.bound_source == "local" for p in first.pricing) and first.local_pricing is None
    assert last.local_pricing is not None
    assert all(p.bound_source != "local" for p in last.pricing)
    assert gap_result.certificate.level == CertificateLevel.CERTIFIED_TO_TOLERANCE
    xa, xb = gap_phase.binodal()
    assert sorted(cs.x[0] for cs in gap_result.composition_sets) == pytest.approx([xa, xb], abs=1e-6)


def test_every_mode_matches_final(gap_result, gap_phase):
    res = equilibrium(gap_phase, x=0.5, eps=EPS, certify="every")
    assert all(p.bound_source != "local" for rec in res.history for p in rec.pricing)
    assert all(rec.local_pricing is None for rec in res.history)
    assert res.certificate.level == CertificateLevel.CERTIFIED_TO_TOLERANCE
    np.testing.assert_allclose(res.mu, gap_result.mu, atol=1e-7)
    assert res.G == pytest.approx(gap_result.G, abs=1e-8)


def test_none_mode_then_certify_later(gap_result, gap_phase):
    res = equilibrium(gap_phase, x=0.5, eps=EPS, certify="none")
    assert res.certificate.level == CertificateLevel.STABILITY_SEARCHED
    assert any("res.certify()" in m for m in res.certificate.messages)
    assert "STABILITY_SEARCHED" in res.summary()
    np.testing.assert_allclose(res.mu, gap_result.mu, atol=1e-7)

    cert = res.certify()
    assert res.certificate is cert and cert.level == CertificateLevel.CERTIFIED_TO_TOLERANCE
    assert all(lb is not None and lb >= -EPS for lb in cert.lower_bounds.values())
    assert res.history[-1].local_pricing is not None


def _blind_local(phase, mu, **kwargs):
    """Local pricing that never finds a destabilizing point (a well it cannot see)."""
    p = _price_phase_local(phase, mu, **kwargs)
    p.ub = max(p.ub, 0.0)
    return p


def test_final_global_step_catches_what_local_pricing_missed(monkeypatch, gap_phase):
    monkeypatch.setattr(pricing, "price_phase_local", _blind_local)
    res = equilibrium(gap_phase, x=0.5, eps=EPS)
    # local pricing saw nothing; the global step found the gap and the loop went on
    assert res.history[0].local_pricing is not None and len(res.history[0].added) == 1
    assert len(res.composition_sets) == 2
    assert res.certificate.level == CertificateLevel.CERTIFIED_TO_TOLERANCE


def test_certify_later_exposes_a_missed_phase(monkeypatch, gap_phase):
    monkeypatch.setattr(pricing, "price_phase_local", _blind_local)
    res = equilibrium(gap_phase, x=0.5, eps=EPS, certify="none")
    assert len(res.composition_sets) == 1
    assert res.certificate.level == CertificateLevel.STABILITY_SEARCHED
    cert = res.certify()
    assert cert.level == CertificateLevel.LOCAL_EQUILIBRIUM
    assert any("not the global equilibrium" in m for m in cert.messages)
    # continuing from the stored state reaches the certified two-phase answer
    res2 = equilibrium(res.phases, b=res.b, T=res.T, eps=EPS, initial_sets=res.composition_sets)
    assert len(res2.composition_sets) == 2
    assert res2.certificate.level == CertificateLevel.CERTIFIED_TO_TOLERANCE


def test_none_mode_multiphase_lens_matches_every():
    liq = BinarySolution("LIQUID", g=(TPoly(10000.0, -10.0), TPoly(15000.0, -10.0)))
    fcc = BinarySolution("FCC_A1")
    fast = equilibrium([liq, fcc], x=0.5, T=1250.0, eps=EPS, certify="none")
    full = equilibrium([liq, fcc], x=0.5, T=1250.0, eps=EPS, certify="every")
    assert fast.phase_names == full.phase_names == ["FCC_A1", "LIQUID"]
    np.testing.assert_allclose(fast.mu, full.mu, atol=1e-7)
    assert fast.certificate.level == CertificateLevel.STABILITY_SEARCHED
    assert full.certificate.level >= CertificateLevel.CERTIFIED_TO_TOLERANCE


def test_modes_are_exported():
    assert CERTIFY_MODES == ("none", "final", "every")
