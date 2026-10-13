# Expected significance: from 5.2σ to 1.7σ

Review of the expected sensitivity of the scouting VVV search, status 13 October 2026. All numbers are
expected (Asimov) results for 255.8 fb⁻¹ from MC; data are blinded.

On 24 September the signal-region tool reported an expected significance of 5.24σ for the five analysis
categories. After the review, the five-channel combine fit gives 1.67σ with the luminosity, trigger and
cross-section uncertainties, and 1.72σ once the VV cross-section uncertainty is reduced to 10% (see
"Updates after the review"). The theory and pileup uncertainties of four categories are still to be added;
from boosted3 they are expected to lower the result by about 15%.

| Stage | 5-channel Z | Expected limit |
|---|---|---|
| 24 Sept, simple Asimov | 5.24 | – |
| combine, background MC statistics, k-fold SRs, QCD from MC | 2.64 | r < 0.73 |
| + per-SR ABCD for QCD | 2.17 | r < 0.82 |
| + luminosity, trigger, cross sections (VV 15%) | 1.67 | r < 1.31 |
| + theory and pileup (projected from boosted3) | ≈ 1.4–1.5 | – |

## The short version

The 5.24σ was not an early version of today's number. It was a different calculation, which assumed:

- each signal region was scored on the same Monte Carlo events that were used to choose it;
- the background was known exactly, although some regions held only a few dozen MC events;
- QCD could be taken directly from MC;
- there were no systematic uncertainties.

The review replaced each assumption with the real treatment, and each replacement cost sensitivity. Two
improvements recovered part of it: k-fold scoring, which lets every MC event take part, and a signal-region
search that knows the ABCD uncertainty.

## Per category

| Category | 24 Sept, simple Asimov | Combine, QCD from MC, no syst. | Per-SR ABCD, no syst. | Per-SR ABCD + lumi, trigger, xsec |
|---|---|---|---|---|
| 1-lepton | 3.60 | 1.84 | 1.53 | 1.32 |
| unresolved | 2.66 | 1.52 | 1.22 | 0.82 |
| resolved | 1.64 | 0.86 | 0.72 | 0.54 |
| boosted3 | 1.63 | 0.90 | 0.60 | 0.56 |
| 2-lepton | 1.44 | 0.93 | 0.59 | 0.51 |
| **5-channel** | **5.24** (quadrature) | **2.64** (quadrature 2.85) | **2.17** (quadrature 2.25) | **1.67** (quadrature 1.81) |

The current numbers use the k-fold models (`kfold_finedecor*`) and the signal regions of the ABCD-aware scan
(`sr_granularity/*_abcdscan`): boosted3 and 1-lepton 4 SRs, resolved 3, 2-lepton 4, unresolved 2 (its 3- and
4-SR points were still running). "Quadrature" is √ΣZ² of the single-channel fits; the full fit falls below it
because some nuisance parameters are shared between the channels. The unresolved per-SR ABCD values are the
converged fits of `combine_bdfix` (see step 7).

## 1. The starting point

On 24 September (`plotting/sr_sensitivity.py`) the significance of each signal region was computed with the
Asimov formula for a counting experiment with an exactly known background, and the 20 regions were added in
quadrature:

$$Z_i = \sqrt{2\left[(S+B)\ln\left(1+\frac{S}{B}\right) - S\right]}, \qquad Z_\mathrm{tot} = \sqrt{\sum_i Z_i^2}$$

| Leading SR per category | S | B | Z_i |
|---|---|---|---|
| 1-lepton SR1 (144 background MC events) | 16.5 | 23.5 | 3.09 |
| unresolved SR1 | 175 | 5466 | 2.36 |
| resolved SR1 | 74 | 2600 | 1.45 |
| boosted3 SR1 | 8.6 | 41.7 | 1.29 |
| 2-lepton SR1 | 4.4 | 12.5 | 1.18 |

```
1-lepton SR1:  (S+B) ln(1+S/B) = 40.0 × ln(1.702) = 21.28
               Z = √(2 × (21.28 − 16.5)) = √9.55 = 3.09

categories:    1-lepton 3.60, unresolved 2.66, resolved 1.64, boosted3 1.63, 2-lepton 1.44
total:         √(12.96 + 7.08 + 2.69 + 2.66 + 2.07) = √27.4 = 5.24
```

The formula itself is fine. The problems lie in what went into it: the SRs were chosen and scored on the
same events, B was treated as exact, QCD came from MC, and no systematic uncertainties were included.

## 2. Event weights and the model (boosted3 × 0.85)

The code review on 1 October found three problems that changed the yields themselves:

- **Negative generator weights were ignored.** The NLO VV samples have 17–25% negative-weight events.
  Counting events instead of summing signed weights scales a yield by (1 − 2f_incl)/(1 − 2f_sel), where f is
  the negative-weight fraction before and after the selection: for VV −9% (boosted3) to +15% (unresolved).
- **The model was heavily overtrained.** The VVV-vs-VV AUC was 0.93 on training and 0.69 on test events.
  Shallower trees (depth 4) halved the gap at the same test AUC.
- **Low-HT QCD was double counted.** The selected events of the HT < 200 GeV QCD samples are hard QCD
  collisions in the pileup overlay, already counted by the high-HT samples. They carried about 30% of the QCD
  weight and are now excluded.

Measured on boosted3 with the same simple method, the first two changes took it from 1.63 to 1.38. This is a
real physics change, not a statistical effect. The low-HT exclusion came later, with the ABCD work (step 6).

## 3. Optimisation bias (boosted3 × 0.49)

The SR search scores 10⁴–10⁵ candidate boxes and keeps the best. Every score is a noisy estimate, and the
largest of many noisy estimates is on average larger than the true value it estimates. Our MC made this
worse because QCD events were heavy: in the split-sample setup, a boosted3 QCD MC event stood for about 28
real events (median weight). If a box truly holds μ events-worth of QCD and each MC event has weight w, the
MC shows no QCD event at all with probability

$$P_0 = e^{-\mu/w}, \qquad \mu = w:\ P_0 = 0.37, \qquad \mu = w/2:\ P_0 = 0.61$$

Among thousands of candidates, the search prefers boxes where QCD happens to be missing from the MC. We
measured the bias by choosing the SRs on one split of the MC and scoring them on an independent one:

| Case | Z on the split used to choose the SRs | Z on an independent split |
|---|---|---|
| toy model | 2.34 | 0.59 |
| boosted3, same model, simple Asimov | 1.38 | 0.67 |
| boosted3 SR-granularity study, combine | 0.9–1.0 | 0.29–0.30 |

## 4. Background MC statistics (boosted3 × 0.45)

With the statistical uncertainty of the background MC, σ_B = √Σw², the Asimov significance becomes

$$Z = \sqrt{2\left[(S+B)\ln\frac{(S+B)(B+\sigma^2)}{B^2+(S+B)\sigma^2} - \frac{B^2}{\sigma^2}\ln\left(1+\frac{\sigma^2 S}{B(B+\sigma^2)}\right)\right]}$$

It reduces to the formula above for σ → 0. For the old leading region (S = 16.5, B = 23.5):

| σ_B / B | 0 | 8% | 20% | 40% |
|---|---|---|---|---|
| Z, 1-lepton SR1 of 24 Sept | 3.09 | 2.84 | 2.12 | 1.32 |

- The boosted3 regions chosen on one split held 8–24 background MC events, so σ_B was 35–45% of B.
- If each of the 20 regions of 24 Sept had σ_B = 10% of B, the total would fall from 5.23 to 3.20; with 30%
  it would be 1.92. The reviewers' estimate from the actual Σw² was about 3.0.
- In combine, with the MC statistics as nuisance parameters, boosted3 went from 0.67 to 0.30.

Two combine bugs on the way, both fixed: the placeholders for empty background templates (`gmN 0`) started at
one count instead of zero, which inflated the significance (Z = 0.90 with an inconsistent limit r < 19.6;
after the fix Z = 0.29, r < 5.9); and `-M Significance` returned exactly 0 for a valid card because of
combine's default range r ≥ 0.

## 5. Recovery: k-fold scoring (boosted3 × 2.8)

Steps 3 and 4 share one cause: too few independent MC events. The k-fold training (K = 5) scores every MC
event with a model that never saw it. The SR search then uses 8.9 million events instead of 0.72 million,
and each event's weight drops by about 5. Since the relative statistical uncertainty of a yield scales as
1/√N:

$$\frac{\sigma_B}{B} \propto \frac{1}{\sqrt{N_\mathrm{MC}}}: \qquad N_\mathrm{MC} \to 5\,N_\mathrm{MC}\ \Rightarrow\ \frac{\sigma_B}{B} \times \frac{1}{\sqrt{5}} = \times\,0.45$$

boosted3 with QCD from MC went from 0.30 to 0.83–0.87, depending on the number of SRs, and to 0.90 with the
finer decorrelation and the ABCD-aware SRs.

Open caveat: with k-fold, the SRs are chosen on the same events that give the final yields. The remaining
optimisation bias is much smaller than before but not yet measured; a split-half check would quantify it.

## 6. QCD from data control regions (5 channels × 0.82)

Each SR's QCD is predicted from its own low-mass control region C_i and the transfer factor TF = B/D of the
ABCD planes, fitted together with the SRs:

$$A_i = C_i \times \frac{B}{D}, \qquad \sigma^2(A_i) \approx \mathrm{TF}^2\left(C_i + \sum_\text{non-QCD} w^2\right) + \left(f_\mathrm{nc} A_i\right)^2, \quad f_\mathrm{nc} = 0.20$$

```
boosted3 SR1 (3-SR set): control region 97 events, of which 46 QCD
   QCD in C1 = 97 − 51 (non-QCD MC) = 46 ± √97 = 46 ± 9.8     → 21%
   ⊕ 20% non-closure                                           → 29% on the QCD prediction
```

| boosted3 best point | Z |
|---|---|
| per-SR ABCD | 0.60 |
| control-region fit frozen | 0.81 |
| without the 20% non-closure | 0.69 |
| without both | 0.96 |
| statistical uncertainties only | 1.23 |

Each control-region parameter is constrained only by its own small control region, so in the
background-only fit the QCD in the SRs can rise to absorb signal. In 2-lepton it absorbs 13 of 21 signal
events: Z falls from 0.96 to 0.59, while the limit barely moves (r < 2.09 → 2.18).

Gains in this step: excluding the low-HT QCD and moving the control window to 20–50 GeV made the method close
in MC (boosted3 0.33 → 0.50), and a signal-region search that includes the ABCD uncertainty raised it to
0.60. Over all five channels: 2.64 → 2.17.

## 7. Systematic uncertainties (5 channels × 0.77 so far)

A normalisation uncertainty δ_p on each background process p adds to the variance of the background:

$$Z \approx \frac{S}{\sqrt{B + \delta B^2}}, \qquad \delta B = \sqrt{\sum_p \left(\delta_p B_p\right)^2}$$

```
today's best 1-lepton SR:  S = 54, B = 442 (VV 158, top 150, V+jets 117)

δB = √( (0.15·158)² + (0.045·150)² + (0.10·117)² + (0.034·442)² ) = 31 events
        VV xsec       top xsec        V+jets xsec   lumi ⊕ trigger

Z:  2.52  →  1.39    (this one SR on its own; normalisations not constrained)
```

- The large SRs with little signal partly constrain these normalisations. 1-lepton with all four SRs keeps
  1.84 → 1.43 with QCD from MC, and 1.53 → 1.32 with ABCD.
- The most harmful uncertainties are correlated across all categories (VV cross section, luminosity,
  trigger). They do not average down, so the 5-channel fit gives 1.67 where the quadrature sum gives 1.81.
- Full set measured for boosted3 only: adding theory and pileup on top of luminosity, trigger and cross
  sections took it from 0.56 to 0.47. The main drivers are the parton-shower FSR variation (±8–18%, signal
  included) for the limit and the VV normalisation (cross section plus QCD scale) for the significance.

A fit problem found and fixed on the way: the ABCD regions B and D hold up to about 10⁹ events-worth in
unresolved. With the luminosity, trigger and cross-section uncertainties also acting on the non-QCD MC
there, the fits did not converge: the free QCD normalisations had to follow those nuisances to a precision of
about 3·10⁻⁵. These bins cannot constrain those nuisances anyway, and leaving them out changes the transfer
factor by at most 0.5%. Every fit tested since gives the same result with every minimiser strategy.

## The boosted3 trail

| Stage | boosted3 Z |
|---|---|
| 24 Sept, simple Asimov | 1.63 |
| new model, fixed weights | 1.38 |
| independent test split | 0.67 |
| + background MC statistics in combine | 0.30 |
| k-fold: all MC events | 0.85 |
| finer decorrelation, ABCD-aware SRs, low-HT QCD excluded | 0.90 |
| per-SR ABCD QCD | 0.60 |
| + systematics (theory, pileup, trigger, lumi, xsec; no jet systematics) | 0.47 |

## Updates after the review (13 October 2026)

**VV normalisation.** The ten VV samples are now normalised to NNLO QCD + NLO EW from MATRIX 2.1.0 (WW ×1.14,
WZ ×1.044, ZZ ×1.155; derivation in `combine/xsec_unc.json`), and the VV cross-section uncertainty is 10%
instead of 15%. On the 5-channel cards (old VV yields), 10% instead of 15% gives Z 1.668 → 1.725 and
r < 1.31 → 1.20; 5% would give 1.846. The reruns with the new VV yields were still running.

**What normalisation uncertainties cost.** On the 5-channel cards with VV at 10% (Z 1.725, r < 1.20),
freezing one normalisation at a time: VV 1.951 (+13%); top, V+jets, or luminosity plus trigger each at most
+0.7% (luminosity plus trigger r < 1.12); top, V+jets, luminosity and trigger together 1.799; all background
normalisations 2.277 (+32%). The large-background SRs already constrain top and V+jets (post-fit widths 0.71
and 0.42 of the prior, VV 0.79).

**Control regions within the current trees.** Regions orthogonal to the SRs and to the ABCD regions (the
mass window outside the SR boxes, leading-jet soft-drop mass above 150 or below 20 GeV) scanned in the k-fold
score caches and added to the 5-channel fit as one-bin cards (`build/cr_study/`):

| Control regions added (VV at 10%) | Z | Limit |
|---|---|---|
| none | 1.725 | r < 1.20 |
| 1-lepton top (p(top) ≥ 0.9, 94–97% pure) | 1.766 | r < 1.15 |
| 2-lepton VV (mass < 20 GeV, p(VV) ≥ 0.3, 51% pure) | 1.763 | r < 1.15 |
| all eight leptonic regions (top, VV, V+jets) | 1.808 | r < 1.07 |
| hadronic categories | no gain (QCD dominates every region) | |

VV cannot be isolated within the current trees: the VV-rich phase space is inside the SRs, and what remains
is at most 50–70% pure. Dedicated selections outside the current trees are the next step.

## Not included yet

- Theory and pileup uncertainties for resolved, unresolved, 1-lepton and 2-lepton (jobs running).
- Jet energy scale and resolution and jet mass scale and resolution (JES, JER, JMS, JMR): they need the
  reconversion with the jet-variation trees.
- The remaining optimisation bias of the k-fold SRs.
- The luminosity: 255.8 fb⁻¹ is still to be confirmed against the analysed 2024 G/H/I data; the significance
  scales roughly as √L.
