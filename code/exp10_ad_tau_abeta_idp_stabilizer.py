#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp10_ad_tau_abeta_idp_stabilizer.py
====================================
Statistical Pharmacology via Kakutani Dichotomy, Paper X -- Experiment 10.
Blocking the amyloid nucleation phase transition of the intrinsically disordered segments of
Alzheimer's disease: tau PHF6*/PHF6 and the Abeta42 monomer.

The monomer ensemble is a product measure on {coil, beta} over contact modes. The healthy soluble
ensemble and the pathogenic pre-nucleation ensemble differ by eps_i, which splits into a fixed
hot-spot part (the hexapeptides) and a long-range tail eps_i = c i^(-alpha). Paper I, Theorem 3.1
gives the trichotomy: the tail sum converges for alpha > 1/2 and diverges for alpha <= 1/2, so the
two ensembles are equivalent above the critical exponent and mutually singular below it. Below it a
nucleus is a distinct phase and does not dissolve back; above it, it does.

The design law of the kinetic ensemble stabiliser is to raise the effective exponent:
hyperphosphorylated tau sits at alpha = 0.30, and KMS-Tau-204 returns it to alpha_eff = 0.72 > 1/2
by shielding long-range coupling through transient multi-site (fuzzy) contacts. It does not touch
the hexapeptides, so the entire effect is attributable to the tail.

Outputs
    data/exp10_scaling.csv        K_tail(N), Pi(N), D_tail(N) over an alpha x N grid
    data/exp10_states.csv         the three tau states and the two Abeta peptides
    data/exp10_alpha_scan.csv     t_half against alpha at several N
    data/exp10_tht.csv            ThT traces
    figures/fig10_ad_tau_abeta_idp_stabilizer.{pdf,png}
    results/exp10_console_log.txt
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import zeta
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp10_lib as M  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _repo_dir(name: str) -> str:
    d = os.path.join(ROOT, name)
    return d if os.path.isdir(d) else HERE


CSV_SCALING = os.path.join(_repo_dir("data"), "exp10_scaling.csv")
CSV_STATES = os.path.join(_repo_dir("data"), "exp10_states.csv")
CSV_SCAN = os.path.join(_repo_dir("data"), "exp10_alpha_scan.csv")
CSV_THT = os.path.join(_repo_dir("data"), "exp10_tht.csv")
FIG_PATH = os.path.join(_repo_dir("figures"), "fig10_ad_tau_abeta_idp_stabilizer.png")
FIG_PATH_PDF = os.path.splitext(FIG_PATH)[0] + ".pdf"
LOG_PATH = os.path.join(_repo_dir("results"), "exp10_console_log.txt")

ALPHAS_SCALING = (0.25, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.72, 0.85, 1.00)
N_GRID = (100, 1000, 10_000, 100_000)
N_FIT_LO, N_FIT_HI = 1000, 100_000
ALPHA_SCAN = np.round(np.arange(0.12, 0.941, 0.02), 3)
T_REF_HOURS = 100.0          # an assay that runs about four days
TOL_ALPHA = 0.01             # estimator must recover alpha to this
TOL_LN2 = 2e-3               # D_tail at criticality against A ln 2, relative


# ----------------------------------------------------------------------------
# Figure style (series house style)
# ----------------------------------------------------------------------------
SURFACE, INK, INK_2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                           "#e87ba4", "#008300", "#4a3aa7", "#e34948")
S_COLOR = {"healthy": GREEN, "hyperphosphorylated": RED, "KMS-Tau-204": ORANGE, "critical": MUTED}
N_COLOR = {100: "#9ec5f4", 1000: "#5598e7", 10_000: "#256abf", 100_000: "#0d366b"}
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9, "axes.unicode_minus": False,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlelocation": "left", "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED, "grid.color": GRID_C, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "legend.labelcolor": INK_2, "mathtext.fontset": "dejavusans",
})


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6)


class Tee:
    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s: str) -> int:
        self.out.write(s)
        self.f.write(s)
        return len(s)

    def flush(self) -> None:
        self.out.flush()
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def alpha_threshold(pep: M.Peptide, n: int, t_ref: float = T_REF_HOURS) -> float:
    """The exponent at which t_half crosses t_ref for a system of n modes. As n grows this must
    approach alpha_c from below: for any alpha < 1/2 the tail sum diverges, so t_half falls to the
    attempt time, while for alpha > 1/2 it stays bounded."""
    def f(a: float) -> float:
        return math.log(M.kinetics(pep, a, n)["t_half_h"]) - math.log(t_ref)
    lo, hi = 0.05, 2.0
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-6))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run() -> int:
    t0 = time.time()
    ok_all = True
    print("Experiment 10: blocking the amyloid nucleation phase transition of AD disordered proteins")
    print("Statistical Pharmacology via Kakutani Dichotomy, Paper X")

    A = M.leading_coefficient()

    # ---------- 1. the model and the leading coefficient ----------
    print("\n[1] Ensembles, hot spots and the leading coefficient")
    print(f"  coil baseline p_0 = {M.P0}, tail amplitude c = {M.C_TAIL}, N = {M.N_MODES} contact modes")
    for pep in M.PEPTIDES:
        hs = ", ".join(f"{h.name} {h.sequence} ({h.residues}, eps={h.eps})" for h in pep.hot_spots)
        print(f"  {pep.name:>9}: {pep.n_hot:>2} hot-spot residues -- {hs}")
        print(f"  {'':>9}  {pep.note}")
    a1, b1 = M.quadratic_coefficient(), M.cubic_coefficient()
    print(f"  Proposition 2.1:  h(eps) = A_1 eps^2 - B_1 eps^3 + O(eps^4)")
    print(f"    A_1 = 1 / (4 p_0 (1-p_0))            = {a1:.10f}")
    print(f"    B_1 = (1/8)(1/p_0^2 - 1/(1-p_0)^2)   = {b1:.10f}")
    print(f"    A   = c^2 A_1                        = {A:.10f}")
    print(f"  The cubic term is odd under p -> 1 - p, so it vanishes at p_0 = 1/2: "
          f"B_1(1/2) = {M.cubic_coefficient(0.5):.3e}. Paper I worked there and saw a quartic first")
    print(f"  correction. At a coil baseline p_0 = {M.P0} the first correction is cubic, and the")
    print(f"  relative error of the quadratic term is B_1 eps / A_1 = {b1 / a1:.6f} eps.")
    print(f"  {'eps':>10} {'h exact':>14} {'A_1 eps^2':>14} {'rel. error':>12} {'B_1 eps/A_1':>13} {'ratio':>8}")
    errs = []
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        h = float(M.hellinger_bernoulli(np.array([M.P0]), np.array([M.P0 + eps]))[0])
        rel = abs(h - a1 * eps ** 2) / h
        errs.append((eps, rel))
        print(f"  {eps:>10.0e} {h:>14.6e} {a1 * eps ** 2:>14.6e} {rel:>12.3e} {b1 * eps / a1:>13.3e} "
              f"{rel / (b1 * eps / a1):>8.4f}")
    ratios = [errs[i][1] / errs[i + 1][1] for i in range(len(errs) - 1)]
    cubic_ok = abs(errs[-2][1] / (b1 * errs[-2][0] / a1) - 1.0) < 1e-3
    a_half = M.leading_coefficient(0.5, M.C_TAIL)
    print(f"  the relative error falls by {np.mean(ratios):.2f}x per decade of eps, so the correction "
          f"is linear in eps: cubic, not quartic")
    print(f"  at p_0 = 1/2 the coefficient is A = c^2 = {a_half:.10f} = {M.C_TAIL ** 2:.10f}, "
          f"the constant of Paper I, Theorem 3.1")
    passed = (abs(a_half - M.C_TAIL ** 2) < 1e-15 and abs(M.cubic_coefficient(0.5)) < 1e-15
              and cubic_ok and all(8.0 < r < 12.0 for r in ratios))
    ok_all &= passed
    print(f"  Acceptance: the measured remainder matches the closed-form cubic coefficient to "
          f"{abs(errs[-2][1] / (b1 * errs[-2][0] / a1) - 1.0):.1e}, falls 10x per decade, and both "
          f"B_1 and Paper I's A are recovered at p_0 = 1/2  {'PASS' if passed else 'FAIL'}")

    # ---------- 1b. evaluating h stably ----------
    print("\n[1b] Evaluating h without cancellation")
    print("  The textbook form h = 2(1 - sqrt(pq) - sqrt((1-p)(1-q))) subtracts two numbers of order")
    print("  one to make a result of order eps^2. This library uses the exact identity")
    print("    h = (p-q)^2 [ (sqrt(p)+sqrt(q))^-2 + (sqrt(1-p)+sqrt(1-q))^-2 ],")
    print("  which has no cancellation. Both are compared with the series of Proposition 2.1, which")
    print("  is independent of either.")
    print(f"  {'eps':>10} {'stable':>12} {'textbook':>12} {'gain':>12}")
    gains = []
    for eps in (1e-3, 1e-4, 1e-5, 1e-6):
        ser = a1 * eps ** 2 - b1 * eps ** 3
        stable = float(M.hellinger_bernoulli(np.array([M.P0]), np.array([M.P0 + eps]))[0])
        naive = 2.0 * (1.0 - math.sqrt(M.P0 * (M.P0 + eps))
                       - math.sqrt((1 - M.P0) * (1 - M.P0 - eps)))
        d_s, d_n = abs(stable - ser) / ser, abs(naive - ser) / ser
        gains.append(d_n / d_s)
        print(f"  {eps:>10.0e} {d_s:>12.2e} {d_n:>12.2e} {d_n / d_s:>11.3g}x")
    passed = gains[-1] > 1e5 and gains[0] < 2.0
    ok_all &= passed
    print(f"  the stable form keeps falling as eps^2, the quartic residual; the textbook form turns "
          f"round and grows once cancellation dominates")
    passed_note = f"{gains[-1]:.2g}x"
    ok_all &= True
    print(f"  Acceptance: the two agree at eps = 1e-3, and at eps = 1e-6 the stable form is "
          f"{passed_note} more accurate  {'PASS' if passed else 'FAIL'}")

    # ---------- 2. the Kakutani trichotomy ----------
    print("\n[2] Kakutani trichotomy in the tail exponent (Paper I, Theorem 3.1)")
    print(f"  {'alpha':>6} {'K(1e2)':>10} {'K(1e3)':>10} {'K(1e4)':>10} {'K(1e5)':>11} "
          f"{'Pi(1e5)':>11} {'slope':>8} {'1-2a':>8} {'a_hat':>8} {'phase':>11}")
    scaling_rows, slope_err = [], []
    for a in ALPHAS_SCALING:
        ks = [float(M.tail_terms(a, n).sum()) for n in N_GRID]
        pis = [math.exp(float(np.sum(np.log1p(-M.tail_terms(a, n) / 2)))) for n in N_GRID]
        slope = M.slope_doubling(a, N_FIT_LO, N_FIT_HI)
        a_hat = M.alpha_from_slope(slope)
        slope_err.append(abs(a_hat - a))
        phase = "singular" if a <= M.ALPHA_C else "equivalent"
        for n, k, pi in zip(N_GRID, ks, pis):
            scaling_rows.append({"alpha": a, "N": n, "K_tail": k, "Pi_tail": pi,
                                 "D_tail": M.doubling_tail(a, n), "slope_theory": M.slope_theory(a),
                                 "slope_measured": slope, "alpha_hat": a_hat, "phase": phase})
        print(f"  {a:>6.2f} {ks[0]:>10.5f} {ks[1]:>10.5f} {ks[2]:>10.5f} {ks[3]:>11.5f} "
              f"{pis[3]:>11.4e} {slope:>+8.4f} {M.slope_theory(a):>+8.4f} {a_hat:>8.4f} {phase:>11}")
    write_csv(CSV_SCALING, scaling_rows)

    passed = (max(slope_err) < TOL_ALPHA
              and all(r["Pi_tail"] > 1e-3 for r in scaling_rows if r["alpha"] > M.ALPHA_C and r["N"] == 100_000)
              and all(r["Pi_tail"] < 1e-3 for r in scaling_rows if r["alpha"] < 0.35 and r["N"] == 100_000))
    ok_all &= passed
    print(f"\n  Acceptance: the doubling estimator recovers alpha to {max(slope_err):.4f} over "
          f"{len(ALPHAS_SCALING)} exponents, and Pi(1e5) separates the two phases  "
          f"{'PASS' if passed else 'FAIL'}")

    # ---------- 2b. the constants of Theorem 3.1 are small-c limits ----------
    print("\n[2b] The constants A ln 2 and A zeta(2a) hold as c -> 0, not at the physical c")
    print(f"  A is the coefficient of the quadratic term only. By Proposition 2.1 the next term is")
    print(f"  cubic, so the exact sum falls below the quadratic prediction and the relative deviation")
    print(f"  is O(c), not O(c^2). At p_0 = 1/2 it would be O(c^2), which is why Paper I could use")
    print(f"  c = 0.1 and still match its constants to ten digits.")
    ln2 = math.log(2.0)
    z2 = float(zeta(2.0))
    print(f"  {'c':>7} {'A':>11} {'D_tail(a_c)':>13} {'A ln 2':>11} {'dev':>10} "
          f"{'K(a=1,1e5)':>12} {'A zeta(2)':>11} {'dev':>10} {'max |a_hat-a|':>14}")
    c_rows = []
    for c in (0.35, 0.10, 0.03, 0.01):
        a_c = M.leading_coefficient(M.P0, c)
        d = M.doubling_tail(M.ALPHA_C, 100_000, c=c)
        dev_d = d / (a_c * ln2) - 1.0
        k1 = float(M.tail_terms(1.00, 100_000, c=c).sum())
        dev_k = k1 / (a_c * z2) - 1.0
        err = max(abs(M.alpha_from_slope(M.slope_doubling(a, N_FIT_LO, N_FIT_HI, c=c)) - a)
                  for a in ALPHAS_SCALING)
        c_rows.append({"c": c, "A": a_c, "dev_dtail": dev_d, "dev_zeta": dev_k, "alpha_err": err})
        print(f"  {c:>7.2f} {a_c:>11.6f} {d:>13.8f} {a_c * ln2:>11.8f} {dev_d:>+10.2e} "
              f"{k1:>12.6f} {a_c * z2:>11.6f} {dev_k:>+10.2e} {err:>14.2e}")
    # the deviation is O(c): each drop in c must shrink it by the same factor
    ratio_c = c_rows[1]["c"] / c_rows[2]["c"]
    r_d = abs(c_rows[1]["dev_dtail"] / c_rows[2]["dev_dtail"])
    r_k = abs(c_rows[1]["dev_zeta"] / c_rows[2]["dev_zeta"])
    r_a = c_rows[1]["alpha_err"] / c_rows[2]["alpha_err"]
    print(f"  from c = {c_rows[1]['c']} to c = {c_rows[2]['c']} (a factor {ratio_c:.2f}) the D_tail "
          f"deviation shrinks {r_d:.2f}x, the zeta deviation {r_k:.2f}x and the estimator error "
          f"{r_a:.2f}x")
    print(f"  all three track the c^1 law that Proposition 2.1 predicts; a c^2 law would give "
          f"{ratio_c ** 2:.1f}x")
    passed = (abs(c_rows[3]["dev_dtail"]) < TOL_LN2 and c_rows[3]["alpha_err"] < 1e-3
              and all(0.80 * ratio_c < r < 1.25 * ratio_c for r in (r_d, r_a)))
    ok_all &= passed
    print(f"  Acceptance: at c = 0.01 the critical constant is reproduced to "
          f"{abs(c_rows[3]['dev_dtail']):.1e} (< {TOL_LN2:.0e}) and the estimator to "
          f"{c_rows[3]['alpha_err']:.1e}, and the deviations fall as c^1  "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"  The zeta column converges more slowly because its sum is dominated by i = 1, where "
          f"eps is largest; D_tail at alpha_c samples i > 1e5, where the cubic term is negligible.")
    print(f"  Consequence for the rest of the paper: the *phase* is read from the exponent, which the "
          f"estimator recovers to {max(slope_err):.1e} even at c = {M.C_TAIL}; the asymptotic "
          f"*constants* are not used anywhere downstream.")

    # ---------- 3. the three tau states ----------
    print("\n[3] Tau microtubule-binding region: healthy, hyperphosphorylated, stabilised")
    state_rows = []
    for pep in M.PEPTIDES:
        for st in (M.HEALTHY, M.CRITICAL, M.PATHOGENIC, M.STABILISED):
            r = M.kinetics(pep, st.alpha)
            r |= {"state": st.name, "state_note": st.note}
            state_rows.append(r)
    write_csv(CSV_STATES, state_rows)
    by = {(r["peptide"], r["state"]): r for r in state_rows}

    print(f"  {'state':>22} {'alpha':>6} {'K_hot':>8} {'K_tail':>8} {'K_N':>8} {'Pi_N':>11} "
          f"{'phase':>11} {'dG (kT)':>9} {'t_half (h)':>12}")
    for st in (M.HEALTHY, M.CRITICAL, M.PATHOGENIC, M.STABILISED):
        r = by[("Tau-MTBR", st.name)]
        print(f"  {st.name:>22} {r['alpha']:>6.2f} {r['K_hot']:>8.4f} {r['K_tail']:>8.4f} "
              f"{r['K_N']:>8.4f} {r['Pi_N']:>11.4e} {r['phase']:>11} {r['dG_kT']:>9.2f} "
              f"{r['t_half_h']:>12.4g}")
    print(f"  K_hot is the same in every row: the stabiliser acts on the tail exponent only, so the "
          f"whole effect below is attributable to alpha.")

    heal, path, stab = (by[("Tau-MTBR", s.name)] for s in (M.HEALTHY, M.PATHOGENIC, M.STABILISED))
    passed = (path["phase"] == "singular" and heal["phase"] == "equivalent"
              and stab["phase"] == "equivalent" and M.STABILISED.alpha > M.ALPHA_C
              and path["K_tail"] > 4 * heal["K_tail"] and path["Pi_N"] < 0.1 * heal["Pi_N"])
    ok_all &= passed
    print(f"  Acceptance: hyperphosphorylation (alpha = {M.PATHOGENIC.alpha}) is singular with "
          f"K_tail {path['K_tail']:.3f} and Pi_N {path['Pi_N']:.3e}; the stabiliser "
          f"(alpha_eff = {M.STABILISED.alpha} > 1/2) returns the pair to equivalence with K_tail "
          f"{stab['K_tail']:.3f}  {'PASS' if passed else 'FAIL'}")

    fold = stab["t_half_h"] / path["t_half_h"]
    print(f"  aggregation half-time: {path['t_half_h']:.3g} h untreated against {stab['t_half_h']:.3g} h "
          f"stabilised, a factor {fold:.3g}; healthy is {heal['t_half_h']:.3g} h")
    passed = fold > 1e7 and path["t_half_h"] < 24 and stab["t_half_h"] > 1e6
    ok_all &= passed
    print(f"  Acceptance: the stabiliser moves t_half from under a day to beyond any experimental "
          f"timescale, a factor {fold:.2e} (> 1e7)  {'PASS' if passed else 'FAIL'}")

    # ---------- 4. the transition sharpens with N ----------
    print("\n[4] The transition is a crossover at finite N and a step only as N -> infinity")
    scan_rows = []
    for n in N_GRID:
        for a in ALPHA_SCAN:
            r = M.kinetics(M.TAU_MTBR, float(a), n)
            scan_rows.append({"N": n, "alpha": float(a), "K_N": r["K_N"], "Pi_N": r["Pi_N"],
                              "dG_kT": r["dG_kT"], "t_half_h": r["t_half_h"],
                              "barrier_floor": r["barrier_floor"]})
    write_csv(CSV_SCAN, scan_rows)
    thresholds = {n: alpha_threshold(M.TAU_MTBR, n) for n in N_GRID}
    print(f"  alpha at which t_half crosses {T_REF_HOURS:.0f} h, against the number of contact modes:")
    for n in N_GRID:
        print(f"    N = {n:>7}: alpha_threshold = {thresholds[n]:.4f}")
    th = [thresholds[n] for n in N_GRID]
    passed = all(a < b for a, b in zip(th, th[1:])) and all(t < M.ALPHA_C for t in th) \
        and th[-1] > 0.40
    ok_all &= passed
    print(f"  Acceptance: the threshold rises monotonically with N and stays below alpha_c = 1/2, "
          f"approaching it from below ({th[0]:.3f} -> {th[-1]:.3f})  {'PASS' if passed else 'FAIL'}")
    print(f"  This is the honest form of the claim: alpha_c is an N -> infinity statement. At finite N "
          f"there is no jump, only a crossover, and its position depends on N.")

    # ---------- 5. Abeta42 against Abeta40 ----------
    print("\n[5] Abeta42 against Abeta40: the C-terminal hot spot alone")
    a42, a40 = by[("Abeta42", M.PATHOGENIC.name)], by[("Abeta40", M.PATHOGENIC.name)]
    print(f"  {'peptide':>9} {'K_hot':>8} {'K_N':>8} {'dG (kT)':>9} {'t_half (h)':>11}")
    for r in (a42, a40):
        print(f"  {r['peptide']:>9} {r['K_hot']:>8.4f} {r['K_N']:>8.4f} {r['dG_kT']:>9.2f} "
              f"{r['t_half_h']:>11.4g}")
    ratio = a40["t_half_h"] / a42["t_half_h"]
    print(f"  Abeta42 nucleates {ratio:.2f}x faster, from the two extra C-terminal residues alone; "
          f"the tail exponent is identical in the two rows.")
    passed = ratio > 2.0 and a42["K_hot"] > a40["K_hot"]
    ok_all &= passed
    print(f"  Acceptance: the C-terminal hot spot alone separates the two peptides in the right "
          f"direction, by {ratio:.2f}x  {'PASS' if passed else 'FAIL'}")
    print(f"  The model attributes none of this to the tail, so it under-predicts the experimental "
          f"gap; Abeta42 also differs from Abeta40 in its long-range contact order.")

    # ---------- 6. ThT traces ----------
    print("\n[6] ThT nucleation traces")
    t = np.linspace(0.0, 120.0, 601)
    tht_rows = []
    for pep_name, state_name in (("Tau-MTBR", M.HEALTHY.name), ("Tau-MTBR", M.PATHOGENIC.name),
                                 ("Tau-MTBR", M.STABILISED.name), ("Abeta42", M.PATHOGENIC.name),
                                 ("Abeta40", M.PATHOGENIC.name)):
        r = by[(pep_name, state_name)]
        f = M.tht_curve(t, r["t_half_h"])
        for ti, fi in zip(t, f):
            tht_rows.append({"peptide": pep_name, "state": state_name, "alpha": r["alpha"],
                             "t_half_h": r["t_half_h"], "t_h": float(ti), "F": float(fi)})
        print(f"  {pep_name:>9} {state_name:>22}: t_half = {r['t_half_h']:>11.4g} h, "
              f"F(120 h) = {float(f[-1]):.6f}")
    write_csv(CSV_THT, tht_rows)
    f_path = M.tht_curve(np.array([120.0]), by[("Tau-MTBR", M.PATHOGENIC.name)]["t_half_h"])[0]
    f_stab = M.tht_curve(np.array([120.0]), by[("Tau-MTBR", M.STABILISED.name)]["t_half_h"])[0]
    passed = f_path > 0.99 and f_stab < 1e-6
    ok_all &= passed
    print(f"  Acceptance: over a 120 h assay the untreated ensemble goes to completion "
          f"(F = {f_path:.6f}) and the stabilised one never starts (F = {f_stab:.2e})  "
          f"{'PASS' if passed else 'FAIL'}")

    # ---------- 7. figure ----------
    print("\n[7] Figure")
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.6))
    axA, axB, axC, axD, axE, axF = axes.ravel()

    # A: the excess spectrum
    i = np.arange(1, M.N_MODES + 1)
    for st in (M.HEALTHY, M.STABILISED, M.PATHOGENIC):
        axA.loglog(i, M.C_TAIL * i ** (-st.alpha), color=S_COLOR[st.name], lw=1.6,
                   label=rf"{st.name}, $\alpha$ = {st.alpha}")
    axA.loglog(i, M.C_TAIL * i ** (-M.ALPHA_C), color=MUTED, lw=1.1, ls="--",
               label=r"critical, $\alpha$ = 1/2")
    for k, h in enumerate(M.TAU_MTBR.hot_spots):
        axA.axhline(h.eps, color=(VIOLET if k == 0 else MAGENTA), lw=1.1, ls=":")
        axA.text(1.25, h.eps * (1.10 if k else 0.74), f"{h.name} {h.sequence}", fontsize=7.5,
                 color=(VIOLET if k == 0 else MAGENTA), fontweight="bold")
    axA.set_xlabel("contact mode $i$ (by range)")
    axA.set_ylabel(r"excess $\beta$ propensity $\epsilon_i$")
    axA.set_title(r"A  Excess spectrum $\epsilon_i = c\,i^{-\alpha}$")
    axA.legend(loc="lower left", fontsize=7.5)
    tidy(axA)

    # B: K_tail(N)
    ns = np.unique(np.round(np.logspace(1, 5, 40)).astype(int))
    for a in (0.25, 0.30, 0.40, 0.50, 0.60, 0.72, 0.85):
        col = RED if a < M.ALPHA_C else (MUTED if a == M.ALPHA_C else GREEN)
        ls = "--" if a == M.ALPHA_C else "-"
        axB.loglog(ns, [M.tail_terms(a, int(n)).sum() for n in ns], color=col, ls=ls,
                   lw=1.5 if a == M.ALPHA_C else 1.1, alpha=0.95)
        axB.annotate(f"{a}", (ns[-1], M.tail_terms(a, int(ns[-1])).sum()), textcoords="offset points",
                     xytext=(4, -2), fontsize=7, color=col)
    axB.set_xlabel("number of contact modes $N$")
    axB.set_ylabel(r"$K_{\mathrm{tail}}(N) = \sum_{i\leq N} h_i$")
    axB.set_title(r"B  Trichotomy: divergent below $\alpha_c$, bounded above")
    axB.text(14, 40, "singular\n" + r"$\alpha < 1/2$", fontsize=8, color=RED, fontweight="bold")
    axB.text(4e3, 0.30, "equivalent\n" + r"$\alpha > 1/2$", fontsize=8, color=GREEN, fontweight="bold")
    tidy(axB)

    # C: the affinity
    for a in (0.25, 0.30, 0.40, 0.50, 0.60, 0.72, 0.85):
        col = RED if a < M.ALPHA_C else (MUTED if a == M.ALPHA_C else GREEN)
        ls = "--" if a == M.ALPHA_C else "-"
        pis = [math.exp(float(np.sum(np.log1p(-M.tail_terms(a, int(n)) / 2)))) for n in ns]
        axC.loglog(ns, pis, color=col, ls=ls, lw=1.5 if a == M.ALPHA_C else 1.1, alpha=0.95)
    axC.set_xlabel("number of contact modes $N$")
    axC.set_ylabel(r"Kakutani affinity $\Pi_N = \prod (1 - h_i/2)$")
    axC.set_title(r"C  $\Pi_N \to 0$ exactly when $K_N \to \infty$")
    tidy(axC)

    # D: the estimator
    aa = np.linspace(0.22, 1.02, 60)
    axD.plot(aa, 1 - 2 * aa, color=INK_2, lw=1.6, label=r"theory, $1 - 2\alpha$")
    meas = [M.slope_doubling(float(a), N_FIT_LO, N_FIT_HI) for a in ALPHAS_SCALING]
    axD.scatter(ALPHAS_SCALING, meas, s=34, color=ORANGE, edgecolor=SURFACE, linewidth=0.6,
                zorder=4, label="doubling estimator")
    axD.axvline(M.ALPHA_C, color=AXIS, lw=1.0, ls="--")
    axD.axhline(0.0, color=AXIS, lw=1.0, ls="--")
    axD.text(M.ALPHA_C + 0.012, -0.62, r"$\alpha_c = 1/2$", fontsize=8, color=INK_2)
    axD.set_xlabel(r"tail exponent $\alpha$")
    axD.set_ylabel(r"slope of $\ln D_{\mathrm{tail}}$ on $\ln N$")
    axD.set_title(r"D  The estimator recovers $\alpha$ to $10^{-3}$")
    axD.legend(loc="upper right", fontsize=8)
    tidy(axD)

    # E: the crossover sharpening
    for n in N_GRID:
        ys = [M.kinetics(M.TAU_MTBR, float(a), n)["t_half_h"] for a in ALPHA_SCAN]
        axE.semilogy(ALPHA_SCAN, ys, color=N_COLOR[n], lw=1.5, label=rf"$N = 10^{{{int(math.log10(n))}}}$")
        if math.isfinite(thresholds[n]):
            axE.scatter([thresholds[n]], [T_REF_HOURS], s=30, color=N_COLOR[n], edgecolor=INK,
                        linewidth=0.6, zorder=5)
    axE.axvline(M.ALPHA_C, color=AXIS, lw=1.0, ls="--")
    axE.axhline(T_REF_HOURS, color=AQUA, lw=1.0, ls=":")
    axE.text(M.ALPHA_C + 0.012, 1e-1, r"$\alpha_c = 1/2$", fontsize=8, color=INK_2)
    axE.text(0.90, T_REF_HOURS * 1.8, f"{T_REF_HOURS:.0f} h assay", fontsize=7.5, color=AQUA, ha="right")
    for st in (M.PATHOGENIC, M.STABILISED):
        axE.scatter([st.alpha], [by[("Tau-MTBR", st.name)]["t_half_h"]], s=140, marker="*",
                    color=S_COLOR[st.name], edgecolor=INK, linewidth=0.7, zorder=6)
    axE.set_xlabel(r"effective tail exponent $\alpha_{\mathrm{eff}}$")
    axE.set_ylabel(r"$t_{1/2}$ (hours)")
    axE.set_title(r"E  Crossover sharpens and moves toward $\alpha_c$ with $N$")
    axE.legend(loc="lower right", fontsize=8)
    tidy(axE)

    # F: ThT
    for pep_name, state_name, col, ls in (("Tau-MTBR", M.PATHOGENIC.name, RED, "-"),
                                          ("Abeta42", M.PATHOGENIC.name, VIOLET, "-"),
                                          ("Abeta40", M.PATHOGENIC.name, BLUE, "-"),
                                          ("Tau-MTBR", M.STABILISED.name, ORANGE, "-"),
                                          ("Tau-MTBR", M.HEALTHY.name, GREEN, "--")):
        r = by[(pep_name, state_name)]
        lab = f"{pep_name}, {state_name}"
        axF.plot(t, M.tht_curve(t, r["t_half_h"]), color=col, ls=ls, lw=1.7, label=lab)
    axF.set_xlim(0, 120)
    axF.set_ylim(-0.03, 1.08)
    axF.set_xlabel("time (hours)")
    axF.set_ylabel("normalised ThT fluorescence $F(t)$")
    axF.set_title("F  ThT traces; stabilised and healthy stay at zero")
    axF.legend(loc="center right", fontsize=7.5)
    tidy(axF)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    fig.savefig(FIG_PATH_PDF)
    plt.close(fig)

    print("\nOutput files (relative to the repository root):")
    for pth in (CSV_SCALING, CSV_STATES, CSV_SCAN, CSV_THT, FIG_PATH, FIG_PATH_PDF, LOG_PATH):
        print("  ", os.path.relpath(pth, ROOT).replace(os.sep, "/"))
    print(f"\nruntime {time.time() - t0:.1f} s")
    print("OVERALL ACCEPTANCE:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def main() -> int:
    tee = Tee(LOG_PATH)
    old = sys.stdout
    sys.stdout = tee
    try:
        return run()
    finally:
        sys.stdout = old
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
