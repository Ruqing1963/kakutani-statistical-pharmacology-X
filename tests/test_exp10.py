"""Unit tests for the Paper X model library. Run with `python -m pytest tests` or `python tests/test_exp10.py`."""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
import exp10_lib as M  # noqa: E402


def test_hellinger_matches_the_package_estimator():
    """The per-mode h of this library must be the per-residue h of kakutani_pharma.compute_dki."""
    from kakutani_pharma import compute_dki

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.9, size=40)
    q = np.clip(p + rng.uniform(-0.05, 0.05, size=40), 1e-6, 1 - 1e-6)
    ours = M.hellinger_bernoulli(p, q)
    theirs = compute_dki(p, q).per_residue
    # the package uses the cancelling form, so agreement is to its precision, not to ours
    assert np.allclose(ours, theirs, rtol=1e-7, atol=1e-15), np.max(np.abs(ours - theirs))


def test_stable_form_beats_the_cancelling_one_at_small_eps():
    """The identity used here is exact; the textbook form loses digits. Checked against the cubic
    expansion of Proposition 2.1, which is independent of both."""
    p0, eps = 0.15, 1e-3
    a1, b1 = M.quadratic_coefficient(p0), M.cubic_coefficient(p0)
    series = a1 * eps ** 2 - b1 * eps ** 3
    stable = float(M.hellinger_bernoulli(np.array([p0]), np.array([p0 + eps]))[0])
    naive = 2.0 * (1.0 - math.sqrt(p0 * (p0 + eps)) - math.sqrt((1 - p0) * (1 - p0 - eps)))
    assert abs(stable - series) / series < abs(naive - series) / series


def test_hellinger_is_symmetric_and_vanishes_on_the_diagonal():
    p = np.array([0.15, 0.3, 0.5, 0.8])
    assert np.allclose(M.hellinger_bernoulli(p, p), 0.0, atol=1e-15)
    q = np.array([0.2, 0.25, 0.6, 0.7])
    assert np.allclose(M.hellinger_bernoulli(p, q), M.hellinger_bernoulli(q, p), atol=1e-15)
    assert np.all(M.hellinger_bernoulli(p, q) > 0)


def test_proposition_21_expansion():
    """h(eps) = A_1 eps^2 - B_1 eps^3 + O(eps^4), with the closed forms of the two coefficients."""
    for p0 in (0.10, 0.15, 0.30, 0.45):
        a1, b1 = M.quadratic_coefficient(p0), M.cubic_coefficient(p0)
        res = []
        for eps in (1e-3, 1e-4):
            h = float(M.hellinger_bernoulli(np.array([p0]), np.array([p0 + eps]))[0])
            pred = a1 * eps ** 2 - b1 * eps ** 3
            rel = abs(h - pred) / h
            # what is left after the cubic is O(eps^4), so the relative residual is O(eps^2)
            assert rel < 1e2 * eps ** 2, (p0, eps, rel)
            res.append(rel)
        # falls ~100x per decade (quartic residual); the floor is set by cancellation in h
        assert res[0] / res[1] > 30, (p0, res)


def test_cubic_term_vanishes_at_one_half_which_is_paper_one():
    """At p0 = 1/2 the cubic coefficient is zero by the symmetry p -> 1-p, the first correction is
    the quartic of Paper I, Lemma 2.3, and the leading coefficient is Paper I's A = c^2."""
    assert abs(M.cubic_coefficient(0.5)) < 1e-15
    for c in (0.05, 0.1, 0.35):
        assert abs(M.leading_coefficient(0.5, c) - c * c) < 1e-15
    # Paper I: h(d) = d^2 + (5/4) d^4 + (21/8) d^6 + O(d^8) at p = 1/2
    for d in (1e-2, 1e-3):
        h = float(M.hellinger_bernoulli(np.array([0.5]), np.array([0.5 + d]))[0])
        quartic = d ** 2 + 1.25 * d ** 4
        sextic = quartic + (21.0 / 8.0) * d ** 6
        assert abs(h - quartic) / h < 1e1 * d ** 4, (d, abs(h - quartic) / h)
        # the next term is O(d^8); at d = 1e-3 the subtraction is already at the rounding floor
        assert abs(h - sextic) / h < max(1e-13, 1e2 * d ** 6), (d, abs(h - sextic) / h)


def test_cubic_coefficient_is_odd_about_one_half():
    for p0 in (0.1, 0.2, 0.35):
        assert abs(M.cubic_coefficient(p0) + M.cubic_coefficient(1 - p0)) < 1e-12


def test_tail_sum_trichotomy():
    """K_tail(N) diverges below alpha_c, grows logarithmically at it, and converges above."""
    ns = (10 ** 3, 10 ** 4, 10 ** 5)
    k_sub = [M.tail_terms(0.30, n).sum() for n in ns]
    k_crit = [M.tail_terms(0.50, n).sum() for n in ns]
    k_sup = [M.tail_terms(0.85, n).sum() for n in ns]
    # power growth below
    assert k_sub[2] / k_sub[1] > 2.0
    # logarithmic at criticality: equal increments per decade of N
    d1, d2 = k_crit[1] - k_crit[0], k_crit[2] - k_crit[1]
    assert abs(d1 / d2 - 1.0) < 0.05, (d1, d2)
    # convergent above
    assert (k_sup[2] - k_sup[1]) / k_sup[2] < 1e-2


def test_affinity_vanishes_exactly_when_the_sum_diverges():
    """The Kakutani dichotomy, in the only form a finite computation can show it."""
    n = 10 ** 5
    for a in (0.25, 0.30, 0.40):
        pi = math.exp(float(np.sum(np.log1p(-M.tail_terms(a, n) / 2))))
        assert pi < 1e-2, (a, pi)
    for a in (0.60, 0.72, 0.85, 1.00):
        pi = math.exp(float(np.sum(np.log1p(-M.tail_terms(a, n) / 2))))
        assert pi > 0.5, (a, pi)


def test_doubling_estimator_recovers_alpha():
    for a in (0.30, 0.40, 0.50, 0.60, 0.72, 0.85):
        a_hat = M.alpha_from_slope(M.slope_doubling(a, 1000, 100_000))
        assert abs(a_hat - a) < 0.01, (a, a_hat)


def test_doubling_estimator_sharpens_as_c_falls():
    """The bias is the cubic term of Proposition 2.1, so it is O(c) and must fall with c."""
    errs = [max(abs(M.alpha_from_slope(M.slope_doubling(a, 1000, 100_000, c=c)) - a)
                for a in (0.30, 0.50, 0.72))
            for c in (0.35, 0.10, 0.03)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-3


def test_slope_theory_and_phase_exponent():
    for a in (0.2, 0.5, 0.8):
        assert abs(M.slope_theory(a) - (1 - 2 * a)) < 1e-15
        assert abs(M.alpha_from_slope(M.slope_theory(a)) - a) < 1e-15
    assert M.gamma_phase(0.3) == 0.4
    assert M.gamma_phase(0.5) == 0.0
    assert M.gamma_phase(0.9) == 0.0


def test_hot_spots_do_not_depend_on_alpha():
    """The design law acts on the tail only; this is what makes the effect attributable to it."""
    for pep in M.PEPTIDES:
        base = float(M.hot_spot_terms(pep).sum())
        for a in (0.25, 0.5, 0.9):
            assert abs(M.kakutani_index(pep, a).K_hot - base) < 1e-15, (pep.name, a)


def test_index_splits_additively():
    for pep in M.PEPTIDES:
        for a in (0.3, 0.72):
            r = M.kakutani_index(pep, a)
            assert abs(r.K_N - (r.K_hot + r.K_tail)) < 1e-12
            assert abs(r.log_Pi - (r.log_Pi_hot + r.log_Pi_tail)) < 1e-12
            assert abs(r.Pi_N - math.exp(r.log_Pi)) < 1e-15


def test_abeta42_nucleates_faster_than_abeta40():
    """The two differ only in the C-terminal hot spot, and the tail exponent is shared."""
    a42 = M.kinetics(M.ABETA42, M.PATHOGENIC.alpha)
    a40 = M.kinetics(M.ABETA40, M.PATHOGENIC.alpha)
    assert a42["K_hot"] > a40["K_hot"]
    assert abs(a42["K_tail"] - a40["K_tail"]) < 1e-12
    assert a42["t_half_h"] < a40["t_half_h"]


def test_stabiliser_crosses_the_critical_exponent_and_slows_aggregation():
    assert M.PATHOGENIC.alpha < M.ALPHA_C < M.STABILISED.alpha
    path = M.kinetics(M.TAU_MTBR, M.PATHOGENIC.alpha)
    stab = M.kinetics(M.TAU_MTBR, M.STABILISED.alpha)
    assert path["phase"] == "singular" and stab["phase"] == "equivalent"
    assert path["K_N"] > stab["K_N"]
    assert stab["t_half_h"] / path["t_half_h"] > 1e7
    assert abs(path["t_half_h"] - M.T_HALF_TARGET) / M.T_HALF_TARGET < 0.05


def test_barrier_is_monotone_and_floored():
    ks = [0.0, 1.0, 5.0, 10.0, 100.0]
    bs = [M.barrier_kt(k) for k in ks]
    assert all(a >= b for a, b in zip(bs, bs[1:]))
    assert bs[0] == M.G0_KT
    assert bs[-1] == 0.0                       # floored, never negative
    assert M.half_time_hours(1e6) == M.T0_HOURS


def test_tht_curve_is_a_logistic_through_one_half():
    t_half = 12.0
    assert abs(float(M.tht_curve(np.array([t_half]), t_half)[0]) - 0.5) < 1e-12
    t = np.linspace(0, 100, 51)
    f = M.tht_curve(t, t_half)
    assert np.all(np.diff(f) >= 0)                  # saturates at 1.0 in floating point
    assert np.all(np.diff(f[t <= 2 * t_half]) > 0)  # strictly increasing while it is resolvable
    assert f[0] < 1e-2 and f[-1] > 0.99


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL  {name}: {exc!r}")
    print("ALL TESTS PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
