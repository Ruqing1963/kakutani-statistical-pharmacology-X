"""
calibration.py
==============
Autocorrelation of the second-order index and bootstrap confidence intervals.

The CKI of two samples is a smooth functional of the sample means and covariances. Its first-order (Frechet)
influence function for a frame x_t of ensemble A is
    q_t = x_t^T M x_t - tr(M S_A) + l^T x_t,
    M = (1/4) (Sbar^-1 - S_A^-1) - (1/16) v v^T,   l = -(1/4) v,   v = Sbar^-1 (mu_B - mu_A),
with x_t centred. The quadratic part carries the covariance terms, the linear part the mean term.

For a stationary Gaussian process with independent modes of lag-one autocorrelations phi_k (variances lam_k),
Isserlis' theorem gives Cov(q_t, q_{t+s}) = 2 sum_{k,l} Mt_kl^2 lam_k lam_l (phi_k phi_l)^s for the quadratic
part (Mt = M in the mode basis). Its integrated autocorrelation time is therefore the weighted average of
(1 + phi_k phi_l) / (1 - phi_k phi_l), never larger than that of the slowest linear mode, and asymptotically
one half of it in continuous time: tau_kl = tau_k tau_l / (tau_k + tau_l) <= max(tau) / 2.
"""

from __future__ import annotations

import numpy as np

from .estimators import mean_cov


def cki_influence(F_A: np.ndarray, F_B: np.ndarray, shrinkage: str | None = "ledoit-wolf") -> dict[str, np.ndarray]:
    """Per-frame influence series of CKI(A, B) with respect to the frames of A: quadratic, linear and total."""
    mA, CA, _ = mean_cov(F_A, shrinkage)
    mB, CB, _ = mean_cov(F_B, shrinkage)
    Sb_inv = np.linalg.inv(0.5 * (CA + CB))
    CA_inv = np.linalg.inv(CA)
    v = Sb_inv @ (mB - mA)
    M = 0.25 * (Sb_inv - CA_inv) - v[:, None] * v[None, :] / 16
    X = F_A - mA
    quad = np.einsum("ti,ij,tj->t", X, M, X)
    quad -= quad.mean()
    lin = X @ (-0.25 * v)
    return {"quadratic": quad, "linear": lin, "total": quad + lin, "M": M, "l": -0.25 * v}


def tau_int_ar1(phi: np.ndarray) -> np.ndarray:
    """Integrated autocorrelation time 1 + 2 sum_{s>=1} phi^s = (1 + phi) / (1 - phi) of a sequence with
    autocorrelation phi^s."""
    phi = np.asarray(phi, float)
    return (1 + phi) / (1 - phi)


def tau_int_quadratic(phi: np.ndarray, lam: np.ndarray, Mt: np.ndarray) -> float:
    """Isserlis prediction for the integrated autocorrelation time of x^T M x when x has independent Gaussian
    modes with variances lam_k and lag-one autocorrelations phi_k; Mt is M in the mode basis."""
    w = Mt ** 2 * np.outer(lam, lam)
    pp = np.outer(phi, phi)
    return float(np.sum(w * (1 + pp) / (1 - pp)) / np.sum(w))


def intervals(estimate: float, reps: np.ndarray, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile, bias-shifted and basic (centred) bootstrap intervals from replicate values."""
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    a = (1 - level) / 2
    lo, hi = np.quantile(reps, [a, 1 - a])
    bias = float(reps.mean() - estimate)
    return {"percentile": (float(lo), float(hi)), "shifted": (float(lo - bias), float(hi - bias)),
            "basic": (float(2 * estimate - hi), float(2 * estimate - lo)), "bias": bias}
