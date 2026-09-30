#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp10_lib.py
============
Statistical Pharmacology via Kakutani Dichotomy, Paper X -- model library.

Intrinsically disordered aggregation-prone segments of Alzheimer's disease, treated as product
measures on {coil, beta-sheet} over contact modes, and the kinetic-ensemble-stabiliser design law
that returns a pathogenic ensemble to the convergent side of the Kakutani dichotomy.

Model
    Each contact mode i carries a Bernoulli law: beta-sheet with probability p_i, coil otherwise.
    The healthy soluble monomer has p_i = p_0 throughout. The pathogenic pre-nucleation ensemble
    has p_i = p_0 + eps_i, and the excess splits into two parts:

        hot spots   a fixed set of residues (PHF6*, PHF6 for tau; CHC, C-terminus for Abeta)
                    with a large, alpha-independent excess eps_hot;
        tail        N long-range contact modes with eps_i = c i^(-alpha).

    The local squared Hellinger distance and the Kakutani affinity are

        h_i = 2 (1 - sqrt(p_i q_i) - sqrt((1-p_i)(1-q_i))),
        K_N = sum_i h_i,        Pi_N = prod_i (1 - h_i / 2).

    Theorem 3.1 of Paper I gives the trichotomy in alpha. Paper I worked at p_0 = 1/2, where the
    leading coefficient is A = c^2 and, by the symmetry p -> 1 - p, the first correction is quartic.
    Proposition 2.1 below extends the expansion to a general coil baseline, which the application
    needs because a disordered coil is not equally likely to be beta. The extension is not cosmetic:
    away from 1/2 there is a cubic term, so the corrections to the asymptotic constants of Theorem
    3.1 are linear in c rather than quadratic.

Kinetics
    Above alpha_c the pathogenic configuration is absolutely continuous with respect to the healthy
    ensemble: it is a fluctuation, and a nucleus formed from it dissolves back. Below alpha_c the two
    are mutually singular, so a nucleus is a distinct phase and does not dissolve. We take the
    nucleation barrier to fall linearly in the separation,

        dG(alpha) / kT = G_0 - lambda K_N(alpha),      t_half = t_0 exp(dG / kT),

    and the ThT trace to be the logistic F(t) = [1 + exp(-k (t - t_half))]^(-1).

The exponent alpha is the only quantity a stabiliser changes in this model. That is deliberate: it
makes the whole effect attributable to the tail, which is what the design law claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# Ensemble parameters
# ----------------------------------------------------------------------------
P0 = 0.15                # beta-sheet propensity of a residue in the healthy disordered coil
C_TAIL = 0.35            # amplitude of the long-range contact excess, eps_i = C_TAIL i^(-alpha)
N_MODES = 1000           # long-range contact modes surrounding the hot spots
ALPHA_C = 0.50           # critical exponent of Paper I, Theorem 3.1

# ----------------------------------------------------------------------------
# Nucleation kinetics (phenomenological; see the module docstring)
# ----------------------------------------------------------------------------
G0_KT = 30.0             # nucleation barrier of a fully equivalent pair, in k_B T
LAMBDA_KT = 2.513        # barrier reduction per unit of Kakutani separation, in k_B T
T0_HOURS = 2.70e-3       # molecular attempt time, hours (about 10 s, a cooperative multi-chain event)
K_THT = 0.45             # ThT logistic growth rate, 1/hour
T_HALF_TARGET = 10.0     # calibration anchor: hyperphosphorylated tau at N = 1000, hours


@dataclass(frozen=True)
class HotSpot:
    """An aggregation-nucleating segment: a short sequence whose beta propensity is raised by a
    fixed, alpha-independent amount when the protective long-range contacts are lost."""
    name: str
    sequence: str
    residues: str          # numbering in the canonical isoform, for the record
    eps: float             # excess beta propensity over the coil baseline

    @property
    def n(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class Peptide:
    """An intrinsically disordered aggregation-prone segment."""
    name: str
    label: str
    hot_spots: tuple
    note: str = ""

    @property
    def n_hot(self) -> int:
        return sum(h.n for h in self.hot_spots)


# Tau microtubule-binding region: the two hexapeptides that nucleate paired helical filaments.
# PHF6 is the stronger of the two, which is why it is the one that aggregates on its own.
TAU_MTBR = Peptide(
    "Tau-MTBR", "tau microtubule-binding region",
    (HotSpot("PHF6*", "VQIINK", "275-280", 0.26),
     HotSpot("PHF6", "VQIVYK", "306-311", 0.30)),
    note="R2 and R3 hexapeptides; hyperphosphorylation releases them from the paperclip fold")

# Abeta42 monomer: the central hydrophobic cluster and the C-terminal segment that Abeta40 lacks.
ABETA42 = Peptide(
    "Abeta42", "amyloid-beta 1-42 monomer",
    (HotSpot("CHC", "KLVFFA", "16-21", 0.26),
     HotSpot("C-term", "GVVIA", "38-42", 0.30)),
    note="the C-terminal hot spot is the two-residue difference from Abeta40")

# Abeta40 differs by the absence of Ile41 and Ala42, which truncates the C-terminal hot spot.
ABETA40 = Peptide(
    "Abeta40", "amyloid-beta 1-40 monomer",
    (HotSpot("CHC", "KLVFFA", "16-21", 0.26),
     HotSpot("C-term", "GVV", "38-40", 0.18)),
    note="same central cluster, a shorter and weaker C-terminal hot spot")

PEPTIDES = (TAU_MTBR, ABETA42, ABETA40)


@dataclass(frozen=True)
class State:
    """A physiological or pharmacological condition, identified by its tail exponent."""
    name: str
    alpha: float
    note: str = ""


HEALTHY = State("healthy", 0.85, "soluble monomer, long-range contacts intact")
PATHOGENIC = State("hyperphosphorylated", 0.30, "pathological phosphorylation, paperclip fold lost")
SUBTHRESHOLD = State("weak modulator", 0.42, "raises the exponent but not across alpha_c")
STABILISED = State("KMS-Tau-204", 0.72, "kinetic ensemble stabiliser, fuzzy multi-site binding")
CRITICAL = State("critical", ALPHA_C, "exactly at the Kakutani critical exponent")
STATES = (HEALTHY, PATHOGENIC, SUBTHRESHOLD, STABILISED)


# ----------------------------------------------------------------------------
# Local Hellinger distance
# ----------------------------------------------------------------------------
def hellinger_bernoulli(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Exact squared Hellinger distance between two Bernoulli laws, evaluated stably.

    The textbook form h = 2 (1 - sqrt(pq) - sqrt((1-p)(1-q))) subtracts two numbers of order one
    to produce a result of order eps^2, so for eps = q - p near 1e-3 it loses about ten digits.
    `kakutani_pharma.compute_dki` uses that form, which is harmless when the two ensembles differ
    appreciably but not when the tail modes are the object of study. The identity

        h = (sqrt(p) - sqrt(q))^2 + (sqrt(1-p) - sqrt(1-q))^2
          = (p - q)^2 [ (sqrt(p) + sqrt(q))^-2 + (sqrt(1-p) + sqrt(1-q))^-2 ]

    is algebraically the same and has no cancellation: the difference of square roots is written as
    (p - q) over their sum. The eps -> 0 limit of the bracket is 1 / (4 p (1-p)), which is A_1 of
    Proposition 2.1.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    d2 = (p - q) ** 2
    return d2 * (1.0 / (np.sqrt(p) + np.sqrt(q)) ** 2
                 + 1.0 / (np.sqrt(1.0 - p) + np.sqrt(1.0 - q)) ** 2)


def quadratic_coefficient(p0: float = P0) -> float:
    """A_1 = 1 / (4 p0 (1 - p0)), the coefficient of eps^2 in Proposition 2.1."""
    return 1.0 / (4.0 * p0 * (1.0 - p0))


def cubic_coefficient(p0: float = P0) -> float:
    """B_1 = (1/8) (1/p0^2 - 1/(1-p0)^2), the coefficient of -eps^3 in Proposition 2.1.

    This term is absent from Paper I, which worked at p0 = 1/2. Expanding both square roots of
    h = 2 (1 - sqrt(p(p+eps)) - sqrt((1-p)(1-p-eps))) gives

        h(eps) = eps^2 / (4 p (1-p)) - (eps^3 / 8) (1/p^2 - 1/(1-p)^2) + O(eps^4),

    and the cubic coefficient is odd under p -> 1 - p, so it vanishes exactly at p = 1/2 and the
    first correction there is the quartic (5/4) eps^4 of Paper I, Lemma 2.3. A disordered coil is
    not equally likely to be beta, so the application cannot use the symmetric case, and the
    leading correction to the asymptotic constants is linear in c rather than quadratic.
    """
    return 0.125 * (1.0 / p0 ** 2 - 1.0 / (1.0 - p0) ** 2)


def leading_coefficient(p0: float = P0, c: float = C_TAIL) -> float:
    """A of Proposition 2.1: the tail behaves as A i^(-2 alpha) with A = c^2 / (4 p0 (1 - p0)).

    At p0 = 1/2 this is A = c^2, which is the constant of Paper I, Theorem 3.1.
    """
    return c * c * quadratic_coefficient(p0)


# ----------------------------------------------------------------------------
# The two parts of the index
# ----------------------------------------------------------------------------
def hot_spot_terms(pep: Peptide, p0: float = P0) -> np.ndarray:
    """Per-residue h over the hot spots. Independent of alpha and of any stabiliser: these are the
    local segments, and the design law of this paper does not act on them."""
    eps = np.concatenate([np.full(h.n, h.eps) for h in pep.hot_spots])
    return hellinger_bernoulli(np.full(eps.size, p0), p0 + eps)


def tail_terms(alpha: float, n: int = N_MODES, p0: float = P0, c: float = C_TAIL) -> np.ndarray:
    """Per-mode h over the n long-range contact modes, eps_i = c i^(-alpha)."""
    i = np.arange(1, n + 1, dtype=float)
    eps = c * i ** (-alpha)
    return hellinger_bernoulli(np.full(n, p0), p0 + eps)


@dataclass
class IndexResult:
    """K_N and Pi_N, split into the alpha-independent hot spots and the alpha-dependent tail."""
    peptide: str
    alpha: float
    n: int
    K_hot: float
    K_tail: float
    K_N: float
    log_Pi_hot: float
    log_Pi_tail: float
    log_Pi: float
    Pi_N: float
    h_hot: np.ndarray = field(repr=False, default=None)
    h_tail: np.ndarray = field(repr=False, default=None)


def kakutani_index(pep: Peptide, alpha: float, n: int = N_MODES, p0: float = P0,
                   c: float = C_TAIL) -> IndexResult:
    """K_N = sum h_i and Pi_N = prod (1 - h_i/2) over hot spots and tail together.

    Pi_N is accumulated in the log, because the product underflows long before the sum does: that
    asymmetry is the practical face of the dichotomy.
    """
    hh, ht = hot_spot_terms(pep, p0), tail_terms(alpha, n, p0, c)
    lp_hot = float(np.sum(np.log1p(-hh / 2.0)))
    lp_tail = float(np.sum(np.log1p(-ht / 2.0)))
    return IndexResult(pep.name, alpha, n, float(hh.sum()), float(ht.sum()),
                       float(hh.sum() + ht.sum()), lp_hot, lp_tail, lp_hot + lp_tail,
                       math.exp(lp_hot + lp_tail), hh, ht)


# ----------------------------------------------------------------------------
# Phase diagnostics (Paper I)
# ----------------------------------------------------------------------------
def slope_theory(alpha: float) -> float:
    """Asymptotic slope of ln D_tail against ln N.

    D_tail(N) = K_tail(2N) - K_tail(N) = A sum_{N<i<=2N} i^(-2a) ~ A N^(1-2a) (2^(1-2a) - 1)/(1-2a),
    so the slope is 1 - 2a for every alpha, negative above alpha_c and zero exactly at it, where
    D_tail -> A ln 2 (Paper I, Proposition 3.2).
    """
    return 1.0 - 2.0 * alpha


def gamma_phase(alpha: float) -> float:
    """Growth exponent of K_tail(N) itself: max(0, 1-2a). Zero on the equivalent side, because the
    sum converges there however fast the increments decay."""
    return max(0.0, 1.0 - 2.0 * alpha)


def doubling_tail(alpha: float, n: int, p0: float = P0, c: float = C_TAIL) -> float:
    """D_tail(N) = K_tail(2N) - K_tail(N), the increment estimator of Paper I. It cancels the
    finite-window bias that a direct regression of K on N carries near alpha_c, where the pole of
    zeta at 2 alpha = 1 inflates the exponent."""
    i = np.arange(n + 1, 2 * n + 1, dtype=float)
    eps = c * i ** (-alpha)
    return float(np.sum(hellinger_bernoulli(np.full(i.size, p0), p0 + eps)))


def slope_doubling(alpha: float, n_lo: int, n_hi: int, p0: float = P0, c: float = C_TAIL) -> float:
    """Measured slope of ln D_tail against ln N over the window, to compare with `slope_theory`."""
    ns = np.unique(np.round(np.logspace(math.log10(n_lo), math.log10(n_hi), 12)).astype(int))
    d = np.array([doubling_tail(alpha, int(n), p0, c) for n in ns])
    return float(np.polyfit(np.log(ns.astype(float)), np.log(d), 1)[0])


def alpha_from_slope(slope: float) -> float:
    """Invert the estimator: a = (1 - slope) / 2. This is how alpha_eff of a stabilised ensemble
    would be read off an experiment that can measure contact-order statistics."""
    return 0.5 * (1.0 - slope)


# ----------------------------------------------------------------------------
# The convergent side, in closed form (Proposition 2.2)
# ----------------------------------------------------------------------------
def k_tail_infinite(alpha: float, p0: float = P0, c: float = C_TAIL, n: int = 2_000_000) -> float:
    """K_tail(alpha, infinity), for alpha > 1/2.

    The sum to n is exact; the remainder is completed analytically using the two leading terms of
    Proposition 2.1, whose tails are zeta(s) - H_n(s). What is neglected is the O(eps^4) part of
    the remainder beyond n, which for alpha > 1/2 and n = 2e6 is below 1e-15.
    """
    if alpha <= 0.5:
        return math.inf
    from scipy.special import zeta as _zeta

    i = np.arange(1, n + 1, dtype=float)
    eps = c * i ** (-alpha)
    h = hellinger_bernoulli(np.full(n, p0), p0 + eps)
    a1, b1 = quadratic_coefficient(p0), cubic_coefficient(p0)
    rem2 = float(_zeta(2 * alpha)) - float(np.sum(i ** (-2 * alpha)))
    rem3 = float(_zeta(3 * alpha)) - float(np.sum(i ** (-3 * alpha)))
    return float(np.sum(h)) + a1 * c ** 2 * rem2 - b1 * c ** 3 * rem3


def k_tail_two_term(alpha: float, p0: float = P0, c: float = C_TAIL) -> float:
    """Proposition 2.2: K_tail(alpha, infinity) = A zeta(2a) - c^3 B_1 zeta(3a) + O(c^4).

    Summing the expansion of Proposition 2.1 term by term over eps_i = c i^(-alpha) gives
    A_1 c^2 sum i^(-2a) - B_1 c^3 sum i^(-3a) + O(c^4 sum i^(-4a)), and each sum is a zeta value.
    The relative error is therefore O(c^2), so the formula is useful below about c = 0.1 and not at
    the physical c = 0.35, where the cubic term over-corrects.
    """
    from scipy.special import zeta as _zeta

    a1, b1 = quadratic_coefficient(p0), cubic_coefficient(p0)
    return a1 * c ** 2 * float(_zeta(2 * alpha)) - b1 * c ** 3 * float(_zeta(3 * alpha))


# ----------------------------------------------------------------------------
# Nucleation kinetics and the ThT trace
# ----------------------------------------------------------------------------
def barrier_kt(k_n: float, g0: float = G0_KT, lam: float = LAMBDA_KT) -> float:
    """Nucleation free-energy barrier in k_B T. Floored at zero: a barrier cannot be negative, and
    the floor is reached only deep in the singular phase, where the model has nothing left to say."""
    return max(0.0, g0 - lam * k_n)


def half_time_hours(k_n: float, g0: float = G0_KT, lam: float = LAMBDA_KT,
                    t0: float = T0_HOURS) -> float:
    """t_half = t_0 exp(dG / kT)."""
    return t0 * math.exp(barrier_kt(k_n, g0, lam))


def k_n_for_half_time(t_half: float, g0: float = G0_KT, lam: float = LAMBDA_KT,
                      t0: float = T0_HOURS) -> float:
    """Invert the barrier law: the K_N at which the half-time equals t_half."""
    return (g0 - math.log(t_half / t0)) / lam


def safe_exponent(pep: Peptide, t_ref: float = 100.0, p0: float = P0, c: float = C_TAIL) -> float:
    """The smallest exponent alpha* such that t_half > t_ref at *every* N, however large.

    Above alpha_c the tail sum converges, so K_N is bounded by K_hot + K_tail(alpha, infinity) for
    all N. Solving K_hot + K_tail(alpha, infinity) = K_N(t_ref) gives the exponent a stabiliser
    must actually reach: anything above it is safe at any system size, anything below it only looks
    safe until the assay is made larger.
    """
    k_star = k_n_for_half_time(t_ref) - float(hot_spot_terms(pep, p0).sum())
    f = lambda a: k_tail_infinite(a, p0, c, n=400_000) - k_star
    return float(brentq(f, 0.5005, 0.95, xtol=1e-7))


def tht_curve(t: np.ndarray, t_half: float, k: float = K_THT) -> np.ndarray:
    """F(t) = [1 + exp(-k (t - t_half))]^(-1), the normalised ThT fluorescence trace.

    Evaluated with the logistic function of scipy rather than the literal formula: the protected
    states have t_half of order 1e8 hours, so the exponent reaches 1e8 and the literal form
    overflows. The logistic saturates instead, which is the right answer.
    """
    from scipy.special import expit

    return expit(k * (np.asarray(t, float) - t_half))


def kinetics(pep: Peptide, alpha: float, n: int = N_MODES, **kw) -> dict:
    """Index and kinetics of one peptide in one state.

    `barrier_floor` marks the states where G_0 - lambda K_N has gone negative and the barrier has
    been clipped at zero. There the model reports t_0 and nothing more: it is deep in the singular
    phase, where a linear barrier law has no content left.
    """
    r = kakutani_index(pep, alpha, n, **kw)
    raw = G0_KT - LAMBDA_KT * r.K_N
    return {"peptide": pep.name, "alpha": alpha, "n": n,
            "K_hot": r.K_hot, "K_tail": r.K_tail, "K_N": r.K_N,
            "log_Pi": r.log_Pi, "Pi_N": r.Pi_N,
            "slope_theory": slope_theory(alpha), "gamma_phase": gamma_phase(alpha),
            "phase": "singular" if alpha <= ALPHA_C else "equivalent",
            "dG_kT": max(0.0, raw), "barrier_floor": int(raw < 0.0),
            "t_half_h": half_time_hours(r.K_N)}
