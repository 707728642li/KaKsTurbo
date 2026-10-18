# How KaKsTurbo computes Ka/Ks

[Workflow figure](assets/workflow.png) · [Numerical interpretation](NUMERICAL_VALIDATION.md) · [Input/output](INPUT_OUTPUT.md)

KaKsTurbo is a numerical implementation of pairwise codon-model fitting and model averaging. It uses established substitution-model families and the MA framework; these are not claimed as newly invented statistical models. The implementation combines structure-aware matrix work, a shared bounded optimizer, CPU/GPU execution and explicit diagnostics.

## 1. Preserve the requested comparison

Input pairs are directed, codon-aligned sequences under standard genetic code 1. Pair identity and requested order are preserved. Codon positions containing a gap, ambiguity or stop in either sequence are filtered jointly. The retained codon-pair counts and fixed model frequencies define the likelihood over all **61 sense codons**. Data-dependent frequency estimates are prepared before optimization.

## 2. Use matrix structure without removing codon states

The generator has 61 diagonal coordinates and both directions of 263 single-nucleotide edges: **587 terminal generator-Jacobian coordinates**. The implementation exploits that support in the derivative contraction and retains the full mean-rate derivative. It does not replace the 61-state process with a reduced biological state space.

Observed-cell reductions are used only for execution graphs with a zero squaring budget. Other budgets use the checked full calculation. The CPU backend reuses persistent workers/workspaces; the GPU backend batches likelihood/derivative evaluations, keeps reusable data resident and caches execution graphs. These are implementation and scheduling optimizations, distinct from model selection or estimation policy.

## 3. Optimize within fixed bounds

The shared optimizer minimizes negative log likelihood per retained codon in log-parameter coordinates. BFGS curvature information is combined with a Schur-corrected step on free coordinates when other coordinates are blocked at active bounds. Armijo line search, curvature safeguards and projected-gradient fallback control the step.

The standard raw bounds are time `[1e-6, 3]`, independent exchangeabilities `[0.01, 30]`, and omega `[0.001, 50]`. The production policy retains a 500-iteration budget and up to 64 finite-coordinate correction rounds. CPU and GPU use the same parameter bounds and stopping policy. Production finite-coordinate acceptance is separate from the saved-endpoint gradient diagnostic.

## Complete model averaging

Seven exchangeability families, each paired with equal or data-dependent frequencies, give fourteen candidates:

| Equal frequencies | Data-dependent frequencies |
|:---|:---|
| JC | F81 |
| K2P | HKY |
| TNEF | TN |
| K3P | K3PUF |
| TIMEF | TIM |
| TVMEF | TVM |
| SYM | GTR |

AICc weights use all fourteen candidates. Time, omega and frequencies are averaged with their model weights; independent exchangeabilities use the established family-specific conditional masks. These parameters reconstruct **one averaged GTR process**, from which Ka and Ks are recalculated. The final estimate is not obtained by simply averaging the fourteen displayed Ka/Ks ratios or selecting only the best model.

If required model points or AICc denominators are unavailable, the result retains the relevant NA/diagnostic state. Difficult models are not silently dropped to manufacture a complete average.

## 4. Make the numerical state inspectable

Results retain model weights, saved endpoints, parameter-bound and iteration-cap flags, and fresh projected-gradient diagnostics. The strict threshold is `1e-8` for the infinity norm of the projected normalized log-gradient. Passing the finite-coordinate gate does not imply passing this gradient threshold.

Independent validation re-evaluates saved objectives/gradients, extends optimization from fixed alternative starts and saved endpoints, and reconstructs the full average. A complete strict reference needs successful optimization, a gradient at or below the threshold, and derivative agreement **for every model**. This follow-up is a research validation procedure, not an automatic refit performed by the standard CLI. See [numerical validation](NUMERICAL_VALIDATION.md).

## Robust and reproduce modes

- **`robust`** is the default CPU/GPU implementation described above.
- **`reproduce`** uses the historical CPU estimator for compatibility work. It has a distinct numerical policy and is not a promise of equivalence to every KaKs_Calculator version.

The optional 22-column export changes presentation only. It does not switch the estimator, infer significance, or fill unavailable statistics with invented values. The implementation uses numerical optimization; it does not train or apply a machine-learning model.

## Statistical foundations and baseline

The candidate-model and model-averaging lineage is described in the original [KaKs_Calculator model-selection/model-averaging paper (Zhang et al., 2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5054075/). The measured comparison program is documented in [KaKs_Calculator 3.0 (Zhang, 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9801026/). These references credit the established framework and comparison tool; they do not validate KaKsTurbo's implementation or imply endorsement.
