"""Unit tests for kakutani_pharma.calibration. Run with `python -m pytest tests` or `python tests/test_calibration.py`."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kakutani_pharma as kp  # noqa: E402


def ar1(rng, phi, n):
    e = rng.normal(size=n) * np.sqrt(1 - phi ** 2)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_isserlis_square_of_ar1():
    rng = np.random.default_rng(11)
    phi = 0.95
    x = ar1(rng, phi, 400_000)
    q = x ** 2 - np.mean(x ** 2)
    for s in (5, 20, 40):
        rho_q = np.corrcoef(q[:-s], q[s:])[0, 1]
        assert abs(rho_q - phi ** (2 * s)) < 0.02
    tau_x = kp.integrated_autocorrelation_time(x)
    tau_q = kp.integrated_autocorrelation_time(q)
    assert abs(tau_x / kp.tau_int_ar1(phi) - 1) < 0.1
    assert abs(tau_q / kp.tau_int_ar1(phi ** 2) - 1) < 0.1
    assert abs(tau_q / tau_x - (1 + phi ** 2) / (1 + phi) ** 2) < 0.03


def test_quadratic_theory_bound():
    rng = np.random.default_rng(12)
    phi = np.sort(rng.uniform(0.5, 0.99, 8))[::-1]
    lam = rng.uniform(0.2, 2.0, 8)
    Mt = rng.normal(size=(8, 8)); Mt = Mt + Mt.T
    t_q = kp.tau_int_quadratic(phi, lam, Mt)
    assert t_q <= kp.tau_int_ar1(phi[0]) + 1e-12
    only = np.zeros((8, 8)); only[0, 0] = 1.0
    assert abs(kp.tau_int_quadratic(phi, lam, only) - kp.tau_int_ar1(phi[0] ** 2)) < 1e-12


def test_influence_predicts_leave_one_out():
    rng = np.random.default_rng(13)
    p, n = 6, 400
    L = rng.normal(size=(p, p)) / np.sqrt(p) + np.eye(p)
    F_A = rng.normal(size=(n, p)) @ L.T
    F_B = rng.normal(size=(n, p)) @ (1.3 * L).T + 0.3
    base = kp.cki_from_frames(F_A, F_B, shrinkage=None).total
    q = kp.cki_influence(F_A, F_B, shrinkage=None)["total"]
    loo = np.array([kp.cki_from_frames(np.delete(F_A, t, axis=0), F_B, shrinkage=None).total - base for t in range(60)])
    r = np.corrcoef(loo, q[:60])[0, 1]
    assert r < -0.99          # removing a frame changes the index by about -q_t / n


def test_intervals():
    reps = np.arange(1, 101, dtype=float)
    ci = kp.intervals(40.0, reps, level=0.9)
    lo, hi = ci["percentile"]
    assert abs(ci["bias"] - 10.5) < 1e-12
    assert np.allclose(ci["shifted"], (lo - 10.5, hi - 10.5))
    assert np.allclose(ci["basic"], (80 - hi, 80 - lo))


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
