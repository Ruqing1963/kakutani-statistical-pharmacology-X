#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paper10_compute_tables.py
=========================
Analytic companion of Paper X. Reads the output of exp10_ad_tau_abeta_idp_stabilizer.py
(data/exp10_*.csv), recomputes every quantity quoted in the paper and writes the table bodies.

    Table 1   the three peptides, their hot spots and K_hot
    Table 2   Proposition 2.1: the cubic expansion away from p_0 = 1/2
    Table 3   the no-cancellation identity against the textbook form
    Table 4   the trichotomy scan in alpha
    Table 5   Proposition 2.2: the two-term zeta formula and the O(c) law
    Table 6   the four tau states
    Table 7   the sub-threshold false-positive trap
    Table 8   the finite-mode crossover threshold and the safe exponent
    Table 9   Abeta42 against Abeta40, two-factor decomposition
    checks    everything the paper asserts, as a pass/fail list
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import zeta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp10_lib as M  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA, RESULTS = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
T_REF = 100.0
N_GRID = (100, 1000, 10_000, 100_000)


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def m(x, d=3, sign=False):
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "--"
    if round(x, d) == 0:
        x = abs(x)
    return f"${x:+.{d}f}$" if sign else f"${x:.{d}f}$"


def e(x, d=2):
    if x == 0:
        return "$0$"
    k = int(math.floor(math.log10(abs(x))))
    return f"${x / 10 ** k:.{d}f} \\times 10^{{{k}}}$"


def write_rows(name: str, rows: list[str]) -> None:
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def alpha_threshold(n: int, t_ref: float = T_REF) -> float:
    f = lambda a: math.log(M.kinetics(M.TAU_MTBR, a, n)["t_half_h"]) - math.log(t_ref)
    return float(brentq(f, 0.05, 2.0, xtol=1e-9))


def main() -> int:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    summary: dict = {}
    out("=" * 96)
    out("Paper X companion: tables and checks")
    out("=" * 96)

    states = read_csv("exp10_states.csv")
    scaling = read_csv("exp10_scaling.csv")
    trap = read_csv("exp10_false_positive_trap.csv")
    by = {(r["peptide"], r["state"]): r for r in states}
    a1, b1 = M.quadratic_coefficient(), M.cubic_coefficient()
    A = M.leading_coefficient()

    # ---------------- Table 1: peptides ----------------
    out("\n[Table 1] The three disordered segments and their hot spots")
    t1 = []
    for pep in M.PEPTIDES:
        kh = float(M.hot_spot_terms(pep).sum())
        hs = "; ".join(f"{h.name} \\texttt{{{h.sequence}}} ({h.residues}), "
                       f"$\\epsilon={h.eps}$" for h in pep.hot_spots)
        out(f"  {pep.name:>9}: n_hot = {pep.n_hot:>2}, K_hot = {kh:.4f} -- "
            + "; ".join(f"{h.name} {h.sequence} ({h.residues}) eps={h.eps}" for h in pep.hot_spots))
        t1.append(f"{pep.name} & ${pep.n_hot}$ & {hs} & {m(kh, 4)} \\\\")
    write_rows("paper10_table1_rows.tex", t1)

    # ---------------- Table 2: Proposition 2.1 ----------------
    out("\n[Table 2] Proposition 2.1: the cubic term away from p_0 = 1/2")
    out(f"  A_1({M.P0}) = {a1:.10f},  B_1({M.P0}) = {b1:.10f},  B_1/A_1 = {b1 / a1:.6f}")
    out(f"  B_1(1/2) = {M.cubic_coefficient(0.5):.3e} (exactly zero; Paper I's symmetric case)")
    out(f"  {'eps':>9} {'h exact':>15} {'A_1 eps^2':>15} {'rel. error':>12} {'B_1 eps/A_1':>13} {'ratio':>9}")
    t2, ratios = [], []
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        h = float(M.hellinger_bernoulli(np.array([M.P0]), np.array([M.P0 + eps]))[0])
        rel = abs(h - a1 * eps ** 2) / h
        pred = b1 * eps / a1
        ratios.append(rel / pred)
        out(f"  {eps:>9.0e} {h:>15.7e} {a1 * eps ** 2:>15.7e} {rel:>12.4e} {pred:>13.4e} "
            f"{rel / pred:>9.5f}")
        t2.append(f"{e(eps, 0)} & {e(h, 4)} & {e(a1 * eps ** 2, 4)} & {e(rel, 3)} & {e(pred, 3)} & "
                  f"{m(rel / pred, 5)} \\\\")
    write_rows("paper10_table2_rows.tex", t2)
    best = min(abs(r - 1.0) for r in ratios[1:-1])
    out(f"  the measured remainder matches B_1 eps / A_1 to {best:.1e} in the resolvable window")
    summary["prop21"] = {"A_1": a1, "B_1": b1, "ratio_BA": b1 / a1, "best_match": best}

    # ---------------- Table 3: the identity ----------------
    out("\n[Table 3] The no-cancellation identity against the textbook form")
    t3, gains = [], []
    for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        ser = a1 * eps ** 2 - b1 * eps ** 3
        stable = float(M.hellinger_bernoulli(np.array([M.P0]), np.array([M.P0 + eps]))[0])
        naive = 2.0 * (1.0 - math.sqrt(M.P0 * (M.P0 + eps))
                       - math.sqrt((1 - M.P0) * (1 - M.P0 - eps)))
        d_s, d_n = abs(stable - ser) / ser, abs(naive - ser) / ser
        gains.append(d_n / max(d_s, 1e-300))
        out(f"  eps = {eps:.0e}: stable {d_s:.3e}, textbook {d_n:.3e}, gain {d_n / d_s:.4g}x")
        t3.append(f"{e(eps, 0)} & {e(d_s, 2)} & {e(d_n, 2)} & {e(d_n / d_s, 2)} \\\\")
    write_rows("paper10_table3_rows.tex", t3)
    summary["identity_gain_max"] = max(gains)

    # ---------------- Table 4: the trichotomy ----------------
    out("\n[Table 4] The trichotomy in alpha")
    alphas = sorted({r["alpha"] for r in scaling})
    t4, errs = [], []
    out(f"  {'alpha':>6} {'K(1e2)':>9} {'K(1e3)':>9} {'K(1e4)':>10} {'K(1e5)':>11} {'Pi(1e5)':>11} "
        f"{'slope':>8} {'a_hat':>8} {'phase':>11}")
    for a in alphas:
        rs = {int(r["N"]): r for r in scaling if r["alpha"] == a}
        r5 = rs[100_000]
        errs.append(abs(r5["alpha_hat"] - a))
        out(f"  {a:>6.2f} {rs[100]['K_tail']:>9.4f} {rs[1000]['K_tail']:>9.4f} "
            f"{rs[10_000]['K_tail']:>10.4f} {r5['K_tail']:>11.4f} {r5['Pi_tail']:>11.3e} "
            f"{r5['slope_measured']:>+8.4f} {r5['alpha_hat']:>8.4f} {r5['phase']:>11}")
        t4.append(f"{m(a, 2)} & {m(rs[100]['K_tail'], 3)} & {m(rs[1000]['K_tail'], 3)} & "
                  f"{m(rs[10_000]['K_tail'], 3)} & {m(r5['K_tail'], 3)} & {e(r5['Pi_tail'], 2)} & "
                  f"{m(r5['slope_measured'], 4, True)} & {m(r5['alpha_hat'], 4)} & {r5['phase']} \\\\")
    write_rows("paper10_table4_rows.tex", t4)
    out(f"  worst |alpha_hat - alpha| over {len(alphas)} exponents: {max(errs):.4f}")
    summary["estimator_worst"] = max(errs)

    # ---------------- Table 5: Proposition 2.2 ----------------
    out("\n[Table 5] Proposition 2.2: K_tail(a, inf) = A zeta(2a) - c^3 B_1 zeta(3a) + O(c^4)")
    out(f"  {'alpha':>6} {'c':>6} {'K_tail(inf)':>13} {'one term':>12} {'two term':>12} "
        f"{'1-term err':>11} {'2-term err':>11}")
    t5, two_errs = [], {}
    for a in (0.60, 0.72, 0.85):
        for c in (0.35, 0.10, 0.03):
            ex = M.k_tail_infinite(a, c=c, n=400_000)
            one = a1 * c ** 2 * float(zeta(2 * a))
            two = M.k_tail_two_term(a, c=c)
            e1, e2 = abs(ex - one) / ex, abs(ex - two) / ex
            two_errs[(a, c)] = e2
            out(f"  {a:>6.2f} {c:>6.2f} {ex:>13.7f} {one:>12.7f} {two:>12.7f} {e1:>11.2e} "
                f"{e2:>11.2e}")
            t5.append(f"{m(a, 2)} & {m(c, 2)} & {m(ex, 6)} & {m(one, 6)} & {m(two, 6)} & "
                      f"{e(e1, 2)} & {e(e2, 2)} \\\\")
    write_rows("paper10_table5_rows.tex", t5)
    r_c = two_errs[(0.72, 0.10)] / two_errs[(0.72, 0.03)]
    out(f"  the two-term error falls {r_c:.1f}x from c = 0.10 to c = 0.03 (a factor 3.33), "
        f"so it is O(c^2) in relative terms")
    out(f"  at c = {M.C_TAIL} the cubic over-corrects and the two-term formula is worse than the "
        f"one-term: the series in c is not usable at the physical amplitude")
    summary["prop22"] = {"two_term_errors": {f"{k[0]}_{k[1]}": v for k, v in two_errs.items()},
                         "ratio_c": r_c}

    # ---------------- Table 6: the tau states ----------------
    out("\n[Table 6] The four tau states at N = 1000")
    t6 = []
    order = (M.HEALTHY, M.STABILISED, M.CRITICAL, M.SUBTHRESHOLD, M.PATHOGENIC)
    for st in order:
        r = by.get(("Tau-MTBR", st.name))
        if r is None:
            r = M.kinetics(M.TAU_MTBR, st.alpha) | {"state": st.name}
        out(f"  {st.name:>22} a={r['alpha']:.2f} K_hot={r['K_hot']:.4f} K_tail={r['K_tail']:.4f} "
            f"K_N={r['K_N']:.4f} Pi={r['Pi_N']:.4e} {r['phase']:>10} dG={r['dG_kT']:.2f} "
            f"t_half={r['t_half_h']:.4g} h")
        t6.append(f"{st.name} & {m(r['alpha'], 2)} & {m(r['K_hot'], 4)} & {m(r['K_tail'], 4)} & "
                  f"{m(r['K_N'], 4)} & {e(r['Pi_N'], 2)} & {r['phase']} & {m(r['dG_kT'], 2)} & "
                  f"{e(r['t_half_h'], 2)} \\\\")
    write_rows("paper10_table6_rows.tex", t6)
    khots = {round(r["K_hot"], 12) for r in states if r["peptide"] == "Tau-MTBR"}
    out(f"  K_hot takes {len(khots)} distinct value(s) across all tau rows: the stabiliser acts on "
        f"the tail alone")

    # ---------------- Table 7: the trap ----------------
    out("\n[Table 7] The sub-threshold false-positive trap")
    t7 = []
    for r in trap:
        out(f"  {r['state']:>22} a={r['alpha']:.2f}  N=1e3: K={r['K_N_1e3']:.4f} "
            f"t={r['t_half_1e3']:.4g} h   N=1e5: K={r['K_N_1e5']:.4f} t={r['t_half_1e5']:.4g} h   "
            f"collapse {r['collapse']:.4g}x")
        t7.append(f"{r['state']} & {m(r['alpha'], 2)} & {m(r['K_N_1e3'], 4)} & "
                  f"{e(r['t_half_1e3'], 2)} & {m(r['K_N_1e5'], 4)} & {e(r['t_half_1e5'], 2)} & "
                  f"{e(r['collapse'], 2)} \\\\")
    write_rows("paper10_table7_rows.tex", t7)
    sub = [r for r in trap if r["alpha"] == M.SUBTHRESHOLD.alpha][0]
    path = [r for r in trap if r["alpha"] == M.PATHOGENIC.alpha][0]
    stab = [r for r in trap if r["alpha"] == M.STABILISED.alpha][0]
    out(f"  the weak modulator at N = 1e5 ({sub['t_half_1e5']:.4g} h) is as fast as the untreated "
        f"ensemble at N = 1e3 ({path['t_half_1e3']:.4g} h): ratio "
        f"{sub['t_half_1e5'] / path['t_half_1e3']:.3f}")

    # ---------------- Table 8: the crossover and the safe exponent ----------------
    out("\n[Table 8] Finite-mode crossover threshold and the exponent a stabiliser must reach")
    t8 = []
    ths = []
    for n in N_GRID + (10 ** 6, 10 ** 7):
        th = alpha_threshold(n)
        ths.append(th)
        out(f"  N = 1e{int(math.log10(n))}: alpha_th = {th:.4f}")
        t8.append(f"$10^{{{int(math.log10(n))}}}$ & {m(th, 4)} \\\\")
    write_rows("paper10_table8_rows.tex", t8)
    k_star_n = M.k_n_for_half_time(T_REF)
    k_star_tail = k_star_n - float(M.hot_spot_terms(M.TAU_MTBR).sum())
    a_star = M.safe_exponent(M.TAU_MTBR, T_REF)
    out(f"  K_N* = {k_star_n:.6f}, K_hot = {float(M.hot_spot_terms(M.TAU_MTBR).sum()):.6f}, "
        f"K_tail* = {k_star_tail:.6f}")
    out(f"  alpha* = {a_star:.5f}: above it, t_half > {T_REF:.0f} h at every N; alpha_th(N) "
        f"increases to alpha_c = 1/2 but alpha* > alpha_c")
    summary["crossover"] = {"alpha_th": dict(zip([str(n) for n in N_GRID + (10 ** 6, 10 ** 7)], ths)),
                            "K_tail_star": k_star_tail, "alpha_star": a_star}

    # ---------------- Table 9: Abeta42 against Abeta40 ----------------
    out("\n[Table 9] Abeta42 against Abeta40")
    kh42 = float(M.hot_spot_terms(M.ABETA42).sum())
    kh40 = float(M.hot_spot_terms(M.ABETA40).sum())
    t42 = M.kinetics(M.ABETA42, M.PATHOGENIC.alpha)["t_half_h"]
    out(f"  (i) hot spot only, same alpha: dK_hot = {kh42 - kh40:.4f}, "
        f"t42 = {t42:.4g} h, t40 = {M.kinetics(M.ABETA40, M.PATHOGENIC.alpha)['t_half_h']:.4g} h, "
        f"ratio {M.kinetics(M.ABETA40, M.PATHOGENIC.alpha)['t_half_h'] / t42:.2f}x")
    t9, ab = [], []
    for target in (2.93, 10.0, 20.0, 50.0):
        g = lambda a40: M.kinetics(M.ABETA40, a40)["t_half_h"] / t42 - target
        a40 = float(brentq(g, 0.2999, 0.45, xtol=1e-9))
        t40 = M.kinetics(M.ABETA40, a40)["t_half_h"]
        ab.append({"target": target, "alpha40": a40, "d_alpha": a40 - M.PATHOGENIC.alpha})
        out(f"  (ii) ratio {target:>5.2f}x needs alpha(Abeta40) = {a40:.4f} "
            f"(delta = {a40 - M.PATHOGENIC.alpha:+.4f}), t40 = {t40:.4g} h")
        t9.append(f"{m(target, 2)} & {m(a40, 4)} & {m(a40 - M.PATHOGENIC.alpha, 4, True)} & "
                  f"{e(t40, 2)} \\\\")
    write_rows("paper10_table9_rows.tex", t9)
    summary["abeta"] = {"dK_hot": kh42 - kh40, "inverse": ab}

    # ---------------- checks ----------------
    out("\n[Checks]")
    checks = {
        "B_1 vanishes at p_0 = 1/2": abs(M.cubic_coefficient(0.5)) < 1e-15,
        "B_1 is odd about 1/2": abs(M.cubic_coefficient(0.1) + M.cubic_coefficient(0.9)) < 1e-12,
        "the cubic coefficient matches the measured remainder to 1e-3": best < 1e-3,
        "A reduces to Paper I's c^2 at p_0 = 1/2": abs(M.leading_coefficient(0.5, 0.35) - 0.35 ** 2) < 1e-15,
        "the identity beats the textbook form by over 1e5 at small eps": max(gains) > 1e5,
        "the doubling estimator recovers alpha to 0.01": max(errs) < 0.01,
        "the two-term formula error is O(c^2)": 6.0 < r_c < 16.0,
        "the two-term formula is worse than one term at c = 0.35": (
            two_errs[(0.72, 0.35)] > abs(M.k_tail_infinite(0.72, c=0.35, n=400_000)
                                         - a1 * 0.35 ** 2 * float(zeta(1.44)))
            / M.k_tail_infinite(0.72, c=0.35, n=400_000)),
        "K_hot is the same in every tau row": len(khots) == 1,
        "the weak modulator collapses by over 1e4 between N = 1e3 and 1e5": sub["collapse"] > 1e4,
        "the stabiliser changes by under 10 % over the same range": abs(stab["collapse"] - 1) < 0.10,
        "at N = 1e5 the weak modulator is within 10 % of the untreated rate at N = 1e3": (
            abs(sub["t_half_1e5"] / path["t_half_1e3"] - 1.0) < 0.10),
        "alpha_th(N) increases monotonically": all(x < y for x, y in zip(ths, ths[1:])),
        "alpha_th(N) stays below alpha_c": all(t < M.ALPHA_C for t in ths),
        "alpha* exceeds alpha_c": a_star > M.ALPHA_C,
        "closing the Abeta gap needs delta alpha < 0.05": all(r["d_alpha"] < 0.05 for r in ab),
    }
    for k, v in checks.items():
        out(f"  {'PASS' if v else 'FAIL':>4}  {k}")
    ok = all(checks.values())
    summary["checks"] = checks
    out(f"\nALL CHECKS {'PASS' if ok else 'FAILED'}")

    with open(os.path.join(RESULTS, "paper10_tables.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(RESULTS, "paper10_tables_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwritten: results/paper10_table{1..9}_rows.tex, paper10_tables.json, "
          "paper10_tables_summary.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
