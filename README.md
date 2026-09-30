# Statistical Pharmacology via Kakutani Dichotomy X

**Blocking the Amyloid Nucleation Phase Transition of Intrinsically Disordered Proteins: Tau PHF6\*/PHF6, the Aβ42 Monomer, and the Kinetic Ensemble Stabiliser Design Law**

<!-- TODO: replace with the reserved Zenodo DOI -->
[![DOI](https://img.shields.io/badge/DOI-reserved%2C%20pending-lightgrey.svg)](https://doi.org/)

Zhengyi Chen<sup>1</sup>, Ruqing Chen<sup>2</sup>

<sup>1</sup> Guangxi Key Laboratory of Drug Discovery and Optimization, School of Pharmacy, Guilin Medical University, Guilin 541199, P. R. China — chenzhengyi@glmc.edu.cn
<sup>2</sup> GUT Geoservice Inc., Montreal, Québec, Canada — ruqing@hotmail.com

Paper X of the series applies the α<sub>c</sub> = 1/2 criticality theorem of **Paper I** (DOI [10.5281/zenodo.23018827](https://doi.org/10.5281/zenodo.23018827)) to the intrinsically disordered aggregation-prone segments of Alzheimer's disease. Where Papers VII–IX worked with folded proteins and covariance, this one works with an IDP, where there is no fixed pocket to dock into and the only structure is a *distribution over contacts*.

## Status

Experiment 10 is complete and passes all 9 acceptance criteria (`results/exp10_console_log.txt`, runtime about 4 s). The unit tests pass: 17 model tests plus the 12 inherited package tests.

The manuscript has not been written yet; the DOI badge above is a placeholder.

## The idea

The monomer ensemble is a product measure on {coil, β} over contact modes. The healthy soluble ensemble and the pathogenic pre-nucleation ensemble differ by

$$\epsilon_i = p_i^{(\beta)} - p_i^{(\mathrm{coil})},$$

which splits in two:

| part | what it is | depends on α? |
|---|---|---|
| **hot spots** | PHF6\* `VQIINK` and PHF6 `VQIVYK` (tau); the central hydrophobic cluster `KLVFFA` and the C-terminus (Aβ) | no |
| **tail** | *N* = 1000 long-range contact modes, ε<sub>i</sub> = c·i<sup>−α</sup> | yes |

By Paper I, Theorem 3.1 the tail sum converges for α > 1/2 and diverges for α ≤ 1/2. So the two ensembles are **equivalent** above the critical exponent and **mutually singular** below it. Below it a nucleus is a distinct phase and does not dissolve back; above it, it does. Hyperphosphorylation lowers α; the design law of a kinetic ensemble stabiliser is to raise it back.

**The stabiliser acts on α alone.** `K_hot` is identical in every row of every table, so the entire effect is attributable to the tail exponent. That is the claim being tested, and making the hot spots inert is what makes the test honest.

## Main results

### Proposition 2.1 — the expansion away from p = 1/2

Paper I worked at p₀ = 1/2, where the symmetry p → 1−p kills the odd terms and the first correction is quartic. A disordered coil is *not* equally likely to be β, so the application cannot use the symmetric case. Expanding both square roots:

$$h(\epsilon) = \underbrace{\frac{\epsilon^2}{4p_0(1-p_0)}}_{A_1} - \underbrace{\frac{\epsilon^3}{8}\left(\frac{1}{p_0^2}-\frac{1}{(1-p_0)^2}\right)}_{B_1} + O(\epsilon^4).$$

At p₀ = 0.15 this gives A₁ = 1.9608 and B₁ = 5.3825, so the relative error of the quadratic term is 2.745·ε — **measured 2.745×10⁻⁴ at ε = 10⁻⁴**, agreeing with the closed form to 1.7×10⁻⁴. The cubic coefficient is odd about 1/2 and vanishes there exactly, recovering Paper I.

**This matters downstream**: it makes the corrections to Paper I's asymptotic constants O(c) rather than O(c²), which is why Paper I could use c = 0.1 and still match A ln 2 to ten digits while this paper, at the physical c = 0.35, cannot.

### Evaluating h without cancellation

The textbook form `h = 2(1 − √(pq) − √((1−p)(1−q)))` subtracts two numbers of order one to produce a result of order ε². This library uses the exact identity

$$h = (p-q)^2\left[\left(\sqrt{p}+\sqrt{q}\right)^{-2} + \left(\sqrt{1-p}+\sqrt{1-q}\right)^{-2}\right],$$

which has none. Against the series of Proposition 2.1:

| ε | stable form | textbook form | gain |
|---|---|---|---|
| 10⁻³ | 1.18×10⁻⁵ | 1.18×10⁻⁵ | 1× |
| 10⁻⁴ | 1.19×10⁻⁷ | 1.22×10⁻⁷ | 1.03× |
| 10⁻⁵ | 1.19×10⁻⁹ | 5.77×10⁻⁷ | 485× |
| 10⁻⁶ | 1.39×10⁻¹¹ | 4.95×10⁻⁵ | **3.6×10⁶×** |

The stable form keeps falling as ε² (the quartic residual); the textbook form turns round and *grows* once cancellation dominates. `kakutani_pharma.compute_dki` uses the textbook form, which is harmless when two ensembles differ appreciably but not when the tail modes are the object of study.

### The trichotomy, and what the estimator recovers

| α | K(10²) | K(10³) | K(10⁴) | K(10⁵) | Π(10⁵) | slope | 1−2α | α̂ | phase |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 3.275 | 11.97 | 41.44 | 138.8 | 6.7×10⁻³¹ | +0.5188 | +0.5 | 0.2406 | singular |
| 0.30 | 2.489 | 7.464 | 20.80 | 55.54 | 8.5×10⁻¹³ | +0.4152 | +0.4 | 0.2924 | singular |
| 0.50 | 0.952 | 1.476 | 2.019 | 2.569 | 2.8×10⁻¹ | +0.0047 | 0.0 | 0.4977 | singular |
| 0.72 | 0.455 | 0.500 | 0.517 | 0.523 | 7.7×10⁻¹ | −0.4389 | −0.44 | 0.7195 | equivalent |
| 0.85 | 0.342 | 0.352 | 0.355 | 0.355 | 8.3×10⁻¹ | −0.6995 | −0.70 | 0.8498 | equivalent |

The doubling estimator of Paper I recovers α to **0.0094** at the physical c = 0.35 and to **3.4×10⁻⁴** at c = 0.01; the residual bias *is* the cubic term, and it falls as c¹ exactly as Proposition 2.1 predicts.

### The three tau states

| state | α | K_hot | K_tail | K_N | Π_N | phase | ΔG‡ | t½ |
|---|---|---|---|---|---|---|---|---|
| healthy | 0.85 | 1.2035 | 0.3525 | 1.556 | 4.51×10⁻¹ | equivalent | 26.09 kT | 5.8×10⁸ h |
| critical | 0.50 | 1.2035 | 1.4756 | 2.679 | 2.56×10⁻¹ | singular | 23.27 kT | 3.4×10⁷ h |
| **hyperphosphorylated** | **0.30** | 1.2035 | 7.4637 | 8.667 | 1.27×10⁻² | **singular** | 8.22 kT | **10.0 h** |
| **KMS-Tau-204** | **0.72** | 1.2035 | 0.5000 | 1.704 | 4.18×10⁻¹ | **equivalent** | 25.72 kT | **4.0×10⁸ h** |

The stabiliser moves t½ by a factor **4.0×10⁷**, from under a day to beyond any experimental timescale, by moving α from 0.30 to 0.72 and nothing else.

### The transition is a crossover at finite N

This is the result we want to state carefully, because it is where the model is most easily oversold. α<sub>c</sub> = 1/2 is an **N → ∞** statement. At finite N there is no jump — only a crossover, whose position depends on N:

| N | α at which t½ crosses 100 h |
|---|---|
| 10² | 0.131 |
| 10³ | 0.314 |
| 10⁴ | 0.390 |
| 10⁵ | 0.429 |

The threshold rises monotonically and approaches 1/2 **from below**. A figure showing a sharp step at α = 1/2 for a 1000-mode system would be a fabrication; panel E shows the family of crossovers instead.

### Aβ42 against Aβ40

| peptide | K_hot | K_N | ΔG‡ | t½ |
|---|---|---|---|---|
| Aβ42 | 1.0906 | 8.554 | 8.50 kT | 13.3 h |
| Aβ40 | 0.6633 | 8.127 | 9.58 kT | 39.0 h |

Aβ42 nucleates **2.93× faster**, from the two extra C-terminal residues alone — the tail exponent is identical in the two rows. The model therefore *under-predicts* the experimental gap, because Aβ42 also differs from Aβ40 in its long-range contact order, which this comparison holds fixed.

## Limits

- **(a)** Contact modes are an abstraction. There is no structure, no trajectory and no force field here: the ensemble is a product measure, and the index is computed from it exactly.
- **(b)** The hot-spot excesses (ε = 0.26–0.30) and the tail amplitude (c = 0.35) are chosen, not measured. What is *not* chosen is that the stabiliser acts only on α.
- **(c)** The barrier law ΔG‡ = G₀ − λK_N is phenomenological, with λ and G₀ calibrated so that hyperphosphorylated tau has t½ = 10 h at N = 1000. Every other half-time in the paper follows from that one anchor.
- **(d)** `KMS-Tau-204` is a point in the model's parameter space, not a compound. The model says what α_eff a stabiliser must reach; it does not say what molecule reaches it.
- **(e)** Product measures assume the contact modes are independent. Real IDP contacts are not, and correlated modes would change the critical exponent. This is the most consequential simplification in the paper.

## Repository layout

```
.
├── kakutani_pharma/     # package inherited from Papers VII-IX
├── code/
│   ├── exp10_lib.py                          # ensembles, hot spots, index, estimator, kinetics
│   └── exp10_ad_tau_abeta_idp_stabilizer.py  # the experiment
├── tests/
│   ├── test_exp10.py                         # 17 model tests
│   ├── test_kakutani_pharma.py               # 8 inherited
│   └── test_calibration.py                   # 4 inherited
├── data/                # exp10_scaling.csv, exp10_states.csv,
│                        # exp10_alpha_scan.csv, exp10_tht.csv
├── figures/fig10_ad_tau_abeta_idp_stabilizer.{pdf,png}
├── results/exp10_console_log.txt
├── CITATION.cff, LICENSE, requirements.txt, pyproject.toml, .gitignore, .gitattributes
└── README.md
```

## Reproducing

```bash
pip install -r requirements.txt
python tests/test_exp10.py && python tests/test_kakutani_pharma.py && python tests/test_calibration.py
python code/exp10_ad_tau_abeta_idp_stabilizer.py   # about 4 seconds; prints OVERALL ACCEPTANCE: PASS
```

## Funding

National Natural Science Foundation of China (No. 22464010), Guangxi Natural Science Foundation of China (No. 2025GXNSFAA069294), and the 2025 Bagui Youth Top Talent Project.

## License

Code under the MIT License; text, figures and data under CC BY 4.0 (see `LICENSE`).
