import BootstrapCalibration.FiniteSample
import BootstrapCalibration.QuadraticRank
import BootstrapCalibration.OLS
import BootstrapCalibration.OrderStatisticMeasurable
import BootstrapCalibration.CalibrationScale

noncomputable section
open MeasureTheory Matrix
open scoped BigOperators

namespace BootstrapCalibration

def pairsRowLaw (n : ℕ) (hn : 0 < n) : FiniteLaw (Draws (Fin n) n) :=
  (uniformRows n hn).iid n

def pairsCountMatrix {n : ℕ} (sample : Draws (Fin n) n) : Matrix (Fin n) (Fin n) ℝ :=
  diagonal (fun i => (countHits (fun j => j = i) n sample : ℝ))

def pairsAccept {n : ℕ} (Q : Matrix (Fin n) (Fin 3) ℝ) (sample : Draws (Fin n) n) : Prop :=
  IsUnit (OLS.gram Q (pairsCountMatrix sample))

def acceptedPairsLaw (n : ℕ) (hn : 0 < n) (Q : Matrix (Fin n) (Fin 3) ℝ)
    (hr : 0 < (pairsRowLaw n hn).mass (pairsAccept Q)) : FiniteLaw (Draws (Fin n) n) :=
  (pairsRowLaw n hn).accepted (pairsAccept Q) hr

/-- Exact whitened pairs-refit perturbation as a matrix acting on the one
observation-noise vector. There is no additional bootstrap observation noise. -/
def pairsCoefficientMap {n : ℕ} (Q : Matrix (Fin n) (Fin 3) ℝ)
    (sample : Draws (Fin n) n) : Matrix (Fin 3) (Fin n) ℝ :=
  OLS.coefficientMap Q (pairsCountMatrix sample)

/-- Full-rank pairs resampling satisfies finite-sample Gaussian calibration
with an explicit design-only error. The observation vector is integrated over,
and the bootstrap law is the actual normalized accepted multinomial law. -/
theorem pairs_finite_sample_coverage (n : ℕ) (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin 3) ℝ) (hQ : Qᵀ * Q = 1)
    (hr : 0 < (pairsRowLaw n hn).mass (pairsAccept Q))
    (L : QuadraticSpace ≃L[ℝ] QuadraticSpace)
    (B k : ℕ) (cutoff : (Fin n → ℝ) → Draws (Draws (Fin n) n) B → ℝ)
    (hcutoffMeas : Measurable
      (fun z : (Fin n → ℝ) × Draws (Draws (Fin n) n) B => cutoff z.1 z.2))
    (hcutoffRank : ∀ z x, k ≤ countHits
      (fun a => quadraticPiSupremum L (pairsCoefficientMap Q a *ᵥ z) ≤ cutoff z x) B x)
    (p tau eta : ℝ) (hp1 : p ≤ 1) (htau : 0 < tau) (htaup : tau < p)
    (heta : 0 ≤ eta) (hrank : binomialTail p B k ≤ eta) :
    p - tau - eta -
      covarianceCalibrationError (acceptedPairsLaw n hn Q hr) (pairsCoefficientMap Q) / tau^2 ≤
      ((standardGaussianMeasure (Fin n)).prod
        ((acceptedPairsLaw n hn Q hr).iid B).toPMF.toMeasure).real
        {z | quadraticPiSupremum L (Qᵀ *ᵥ z.1) ≤ cutoff z.1 z.2} := by
  exact finite_sample_gaussian_calibration (acceptedPairsLaw n hn Q hr) Qᵀ
    (by simpa using hQ) (pairsCoefficientMap Q) L B k cutoff hcutoffMeas hcutoffRank
    p tau eta hp1 htau htaup heta hrank

/-- For distinct quadratic actions, the conditioning denominator is proved
positive from actual row sampling; it is not a bootstrap-validity assumption. -/
theorem pairsAccept_positive (n : ℕ) (hn : 0 < n) (hn3 : 3 ≤ n)
    (u : Fin n → ℝ) (hu : Function.Injective u)
    (C : Matrix (Fin 3) (Fin 3) ℝ) (hC : IsUnit C) :
    0 < (pairsRowLaw n hn).mass (pairsAccept (QuadraticRank.design u * C)) :=
  QuadraticRank.pairs_gram_acceptance_positive n hn hn3 u hu C hC


/-- The implemented sorted critical value, after undoing the one shared
studentizing scale. Certified upper endpoints may depend on the observed data. -/
def scaledSortedCutoff {n : ℕ} (scale : (Fin n → ℝ) → ℝ)
    (upper : (Fin n → ℝ) → Draws (Fin n) n → ℝ) (B : ℕ) (j : Fin B)
    (z : Fin n → ℝ) (x : Draws (Draws (Fin n) n) B) : ℝ :=
  scale z * sortedCritical (upper z) B x j

/-- Finite-sample coverage for the actual sorted, conservative calibration
rule. The remaining hypotheses on score endpoints are their measurability and
certified upper-bound property, not any sampling-distribution approximation. -/
theorem calibrated_pairs_coverage (n : ℕ) (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin 3) ℝ) (hQ : Qᵀ * Q = 1)
    (hr : 0 < (pairsRowLaw n hn).mass (pairsAccept Q))
    (L : QuadraticSpace ≃L[ℝ] QuadraticSpace)
    (scale : (Fin n → ℝ) → ℝ) (hscale : Measurable scale) (hscale0 : ∀ z, 0 ≤ scale z)
    (upper : (Fin n → ℝ) → Draws (Fin n) n → ℝ)
    (hupperMeas : ∀ a, Measurable (fun z => upper z a))
    (hupper : ∀ z a, quadraticPiSupremum L (pairsCoefficientMap Q a *ᵥ z) ≤ scale z * upper z a)
    (B : ℕ) (j : Fin B) (p tau eta : ℝ)
    (hp1 : p ≤ 1) (htau : 0 < tau) (htaup : tau < p) (heta : 0 ≤ eta)
    (hrank : binomialTail p B (j.val + 1) ≤ eta) :
    p - tau - eta -
      covarianceCalibrationError (acceptedPairsLaw n hn Q hr) (pairsCoefficientMap Q) / tau^2 ≤
      ((standardGaussianMeasure (Fin n)).prod
        ((acceptedPairsLaw n hn Q hr).iid B).toPMF.toMeasure).real
        {z | quadraticPiSupremum L (Qᵀ *ᵥ z.1) ≤ scaledSortedCutoff scale upper B j z.1 z.2} := by
  apply pairs_finite_sample_coverage n hn Q hQ hr L B (j.val + 1)
    (scaledSortedCutoff scale upper B j)
  · exact (hscale.comp measurable_fst).mul (measurable_sortedCritical_joint upper hupperMeas B j)
  · intro z x
    exact scaled_conservative_rank B (j.val + 1) x _ (upper z) (scale z)
      (sortedCritical (upper z) B x j) (hscale0 z) (hupper z)
      (sortedCritical_rank (upper z) B x j)
  all_goals assumption

end BootstrapCalibration
