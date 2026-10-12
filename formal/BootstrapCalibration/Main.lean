import BootstrapCalibration.OLSBootstrapError
import BootstrapCalibration.Whitening

noncomputable section
open MeasureTheory Matrix

namespace BootstrapCalibration

/-- The observed action design is fixed when stating conditional coverage. -/
structure QuadraticCalibrationDesign where
  n : ℕ
  sampleSize : 5 ≤ n
  action : Fin n → ℝ
  distinct : Function.Injective action

namespace QuadraticCalibrationDesign

variable (D : QuadraticCalibrationDesign)

theorem n_pos : 0 < D.n := by have := D.sampleSize; omega
theorem n_three : 3 ≤ D.n := by have := D.sampleSize; omega

def matrix : Matrix (Fin D.n) (Fin 3) ℝ := QuadraticRank.design D.action

theorem gram_isUnit : IsUnit (D.matrixᵀ * D.matrix) :=
  Whitening.quadratic_original_gram_isUnit D.n D.n_three D.action D.distinct

def normalized : Matrix (Fin D.n) (Fin 3) ℝ := Whitening.normalizedDesign D.matrix

theorem orthonormal : D.normalizedᵀ * D.normalized = 1 :=
  Whitening.normalizedDesign_orthonormal D.matrix D.gram_isUnit

def directions : QuadraticSpace ≃L[ℝ] QuadraticSpace :=
  Whitening.directionEquiv D.matrix D.gram_isUnit

theorem acceptance_positive : 0 < (pairsRowLaw D.n D.n_pos).mass (pairsAccept D.normalized) :=
  pairsAccept_positive D.n D.n_pos D.n_three D.action D.distinct _
    (Whitening.whitener_isUnit D.matrix D.gram_isUnit)

def law : FiniteLaw (Draws (Fin D.n) D.n) :=
  acceptedPairsLaw D.n D.n_pos D.normalized D.acceptance_positive

def acceptance : ℝ :=
  1 - ((D.n : ℝ) + (D.n.choose 2 : ℝ) * ((2 : ℝ)^D.n - 2)) / (D.n : ℝ)^D.n

theorem acceptance_eq : (pairsRowLaw D.n D.n_pos).mass (pairsAccept D.normalized) = D.acceptance := by
  have hsing := QuadraticRank.pairs_gram_singular_probability D.n D.n_pos D.action D.distinct
    (Whitening.whitener D.matrix) (Whitening.whitener_isUnit D.matrix D.gram_isUnit)
  have hcompl := (pairsRowLaw D.n D.n_pos).mass_compl (pairsAccept D.normalized)
  change (pairsRowLaw D.n D.n_pos).mass (fun x => ¬ pairsAccept D.normalized x) = _ at hsing
  rw [hcompl] at hsing
  unfold acceptance
  linarith

def error (a : ℝ) : ℝ := OLSBootstrapError.epsilon D.normalized a D.acceptance

/-- The candidate's finite-n, finite-B coverage bound for the original fixed
quadratic design. All statistical approximation terms are derived from this
design and the actual accepted pairs law. Score certification and measurability
are the only algorithm-facing assumptions. -/
theorem finite_sample_bound
    (scale : (Fin D.n → ℝ) → ℝ) (hscale : Measurable scale) (hscale0 : ∀ z, 0 ≤ scale z)
    (upper : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (hupperMeas : ∀ w, Measurable (fun z => upper z w))
    (hupper : ∀ z w, quadraticPiSupremum D.directions
      (pairsCoefficientMap D.normalized w *ᵥ z) ≤ scale z * upper z w)
    (B : ℕ) (j : Fin B) (p tau eta a : ℝ)
    (hp1 : p ≤ 1) (htau : 0 < tau) (htaup : tau < p) (heta : 0 ≤ eta)
    (ha0 : 0 < a) (ha1 : a < 1) (hrank : binomialTail p B (j.val + 1) ≤ eta) :
    p - tau - eta - D.error a / tau^2 ≤
      ((standardGaussianMeasure (Fin D.n)).prod (D.law.iid B).toPMF.toMeasure).real
        {z | quadraticPiSupremum D.directions (D.normalizedᵀ *ᵥ z.1) ≤
          scaledSortedCutoff scale upper B j z.1 z.2} := by
  have hcov := OLSBootstrapError.covarianceCalibrationError_le D.n_pos D.normalized
    D.orthonormal D.acceptance_positive a ha0 ha1
  rw [D.acceptance_eq] at hcov
  have hcoverage := calibrated_pairs_coverage D.n D.n_pos D.normalized D.orthonormal
    D.acceptance_positive D.directions scale hscale hscale0 upper hupperMeas hupper
    B j p tau eta hp1 htau htaup heta hrank
  have hdiv := div_le_div_of_nonneg_right hcov (sq_nonneg tau)
  exact (sub_le_sub_left hdiv (p - tau - eta)).trans hcoverage

end QuadraticCalibrationDesign
end BootstrapCalibration
