import BootstrapCalibration.Whitening
import BootstrapCalibration.QuadraticStatistic
import BootstrapCalibration.OLS
import BootstrapCalibration.ResidualScale

/-! Original quadratic-regression coordinates for the normalized calibration
theorem. All statements here are exact deterministic identities. -/

noncomputable section
open scoped Matrix BigOperators
open Matrix
namespace BootstrapCalibration.RegressionInterpretation
open Whitening

variable {ι : Type} [Fintype ι]

/-- Ordinary least-squares coefficients in the original polynomial basis. -/
def olsCoefficient (P : Matrix ι (Fin 3) ℝ) (Y : ι → ℝ) : Fin 3 → ℝ :=
  ((Pᵀ * P)⁻¹ * Pᵀ) *ᵥ Y

/-- Polynomial evaluation in the original coefficient coordinates. -/
def prediction (β : Fin 3 → ℝ) (u : ℝ) : ℝ := β 0 + u * β 1 + u^2 * β 2

lemma prediction_eq_dot (β : Fin 3 → ℝ) (u : ℝ) :
    prediction β u = dotProduct (quadraticFeatures u) β := by
  simp [prediction, quadraticFeatures, dotProduct, Fin.sum_univ_succ, add_assoc]

lemma direction_inner (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P))
    (v : QuadraticSpace) (x : Fin 3 → ℝ) :
    inner (𝕜 := ℝ) (directionEquiv P hP v)
        ((EuclideanSpace.equiv (Fin 3) ℝ).symm x) =
      dotProduct v (whitener P *ᵥ x) := by
  rw [directionEquiv_apply, EuclideanSpace.inner_eq_star_dotProduct]
  change dotProduct x (star ((whitener P)ᵀ *ᵥ (v : Fin 3 → ℝ))) = _
  simp only [star_trivial]
  rw [dotProduct_comm, Matrix.mulVec_transpose, ← Matrix.dotProduct_mulVec]

lemma olsMatrix_eq_whitened (P : Matrix ι (Fin 3) ℝ) :
    (Pᵀ * P)⁻¹ * Pᵀ = whitener P * (normalizedDesign P)ᵀ := by
  rw [normalizedDesign, Matrix.transpose_mul, ← Matrix.mul_assoc, whitener_covariance]

/-- The original coefficient error is exactly the inverse-root transform of
the normalized Gaussian projection, with signal cancellation proved. -/
theorem ols_error (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P))
    (β : Fin 3 → ℝ) (z : ι → ℝ) (σ : ℝ) :
    olsCoefficient P (P *ᵥ β + σ • z) - β =
      σ • (whitener P *ᵥ ((normalizedDesign P)ᵀ *ᵥ z)) := by
  have hdet := (Matrix.isUnit_iff_isUnit_det _).mp hP
  have hleft : ((Pᵀ * P)⁻¹ * Pᵀ) * P = 1 := by
    rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hdet]
  unfold olsCoefficient
  rw [Matrix.mulVec_add, Matrix.mulVec_smul, Matrix.mulVec_mulVec, hleft,
    Matrix.one_mulVec, add_sub_cancel_left, olsMatrix_eq_whitened,
    ← Matrix.mulVec_mulVec]

/-- The model equation needed by the frozen-predictor envelope, derived from
the original least-squares formula and the regression observation equation. -/
theorem prediction_error (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P))
    (β : Fin 3 → ℝ) (z : ι → ℝ) (σ u : ℝ) :
    prediction (olsCoefficient P (P *ᵥ β + σ • z)) u - prediction β u =
      σ * inner (𝕜 := ℝ) (directionEquiv P hP (quadraticFeatures u))
        ((EuclideanSpace.equiv (Fin 3) ℝ).symm ((normalizedDesign P)ᵀ *ᵥ z)) := by
  rw [prediction_eq_dot, prediction_eq_dot, ← dotProduct_sub, ols_error P hP,
    dotProduct_smul, smul_eq_mul, direction_inner]

/-- Squared evaluation-direction norm is exactly the inverse-Gram quadratic
form used in the original standard-error denominator. -/
theorem direction_norm_sq (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P))
    (v : QuadraticSpace) :
    ‖directionEquiv P hP v‖^2 = dotProduct v ((Pᵀ * P)⁻¹ *ᵥ v) := by
  rw [← real_inner_self_eq_norm_sq]
  have hv : directionEquiv P hP v =
      (EuclideanSpace.equiv (Fin 3) ℝ).symm ((whitener P)ᵀ *ᵥ v) := by
    rw [directionEquiv_apply]
    rfl
  rw [hv, ← hv]
  conv_lhs => arg 2; rw [hv]
  rw [direction_inner, Matrix.mulVec_mulVec, whitener_covariance]

theorem direction_norm_eq_sqrt (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P))
    (v : QuadraticSpace) :
    ‖directionEquiv P hP v‖ = Real.sqrt (dotProduct v ((Pᵀ * P)⁻¹ *ᵥ v)) := by
  rw [← direction_norm_sq P hP v, Real.sqrt_sq (norm_nonneg _)]

/-- Standard-error shape, before a common estimated noise scale is applied. -/
def standardErrorShape (P : Matrix ι (Fin 3) ℝ) (u : ℝ) : ℝ :=
  Real.sqrt (dotProduct (quadraticFeatures u) ((Pᵀ * P)⁻¹ *ᵥ quadraticFeatures u))

theorem standardErrorShape_eq_norm (P : Matrix ι (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (u : ℝ) :
    standardErrorShape P u = ‖directionEquiv P hP (quadraticFeatures u)‖ :=
  (direction_norm_eq_sqrt P hP (quadraticFeatures u)).symm

/-- The normalized coefficient event certifies the actual original OLS fit
and any frozen predictor on the entire real action line. -/
theorem original_regression_envelope (P : Matrix ι (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (β : Fin 3 → ℝ) (z : ι → ℝ)
    (σ q : ℝ) (hσ : 0 ≤ σ)
    (hq : quadraticPiSupremum (directionEquiv P hP) ((normalizedDesign P)ᵀ *ᵥ z) ≤ q)
    (fixed : ℝ → ℝ) :
    ∀ u, let w := σ * q * standardErrorShape P u
      let auxiliary := prediction (olsCoefficient P (P *ᵥ β + σ • z)) u
      |fixed u - prediction β u| ≤ radius (fixed u) auxiliary w ∧
      (0 ≤ radius (fixed u) auxiliary w - |fixed u - prediction β u| ∧
        radius (fixed u) auxiliary w - |fixed u - prediction β u| ≤ 2 * w) := by
  intro u
  rw [standardErrorShape_eq_norm P hP u]
  exact quadraticSupremum_envelope (directionEquiv P hP)
    ((EuclideanSpace.equiv (Fin 3) ℝ).symm ((normalizedDesign P)ᵀ *ᵥ z))
    σ q hσ hq fixed (prediction (olsCoefficient P (P *ᵥ β + σ • z))) (prediction β)
    (prediction_error P hP β z σ) u

/-- Weighted pairs-refit coefficients in the original polynomial basis. -/
def weightedOlsCoefficient (P : Matrix ι (Fin 3) ℝ) (W : Matrix ι ι ℝ)
    (Y : ι → ℝ) : Fin 3 → ℝ :=
  ((Pᵀ * W * P)⁻¹ * Pᵀ * W) *ᵥ Y

lemma normalizedDesign_mul_inv (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P)) :
    normalizedDesign P * (whitener P)⁻¹ = P := by
  rw [normalizedDesign, Matrix.mul_assoc,
    Matrix.mul_nonsing_inv _ ((Matrix.isUnit_iff_isUnit_det _).mp (whitener_isUnit P hP)),
    Matrix.mul_one]

lemma weighted_olsMatrix_eq_whitened (P : Matrix ι (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (W : Matrix ι ι ℝ) :
    (Pᵀ * W * P)⁻¹ * Pᵀ * W =
      whitener P * (OLS.gram (normalizedDesign P) W)⁻¹ * (normalizedDesign P)ᵀ * W := by
  let C := whitener P
  have hC := (Matrix.isUnit_iff_isUnit_det _).mp (whitener_isUnit P hP)
  have hCt : IsUnit Cᵀ.det := by simpa only [Matrix.det_transpose] using hC
  have hg : OLS.gram (normalizedDesign P) W = Cᵀ * (Pᵀ * W * P) * C := by
    simp only [OLS.gram, normalizedDesign, Matrix.transpose_mul, C, Matrix.mul_assoc]
  rw [hg, Matrix.mul_inv_rev, Matrix.mul_inv_rev, normalizedDesign, Matrix.transpose_mul]
  change _ = C * (C⁻¹ * ((Pᵀ * W * P)⁻¹ * Cᵀ⁻¹)) * (Cᵀ * Pᵀ) * W
  calc
    _ = (C * C⁻¹) * (Pᵀ * W * P)⁻¹ * (Cᵀ⁻¹ * Cᵀ) * Pᵀ * W := by
      rw [Matrix.mul_nonsing_inv C hC, Matrix.nonsing_inv_mul Cᵀ hCt]
      simp only [Matrix.one_mul, Matrix.mul_one]
    _ = _ := by simp only [Matrix.mul_assoc]

/-- Vector-coordinate version of the exact normalized refit identity. -/
lemma normalized_refit_error [DecidableEq ι] (Q : Matrix ι (Fin 3) ℝ)
    (hQ : Qᵀ * Q = 1) (W : Matrix ι ι ℝ) (hW : IsUnit (OLS.gram Q W).det)
    (β : Fin 3 → ℝ) (z : ι → ℝ) (σ : ℝ) :
    ((OLS.gram Q W)⁻¹ * Qᵀ * W) *ᵥ (Q *ᵥ β + σ • z) -
        Qᵀ *ᵥ (Q *ᵥ β + σ • z) = σ • (OLS.coefficientMap Q W *ᵥ z) := by
  have h := OLS.weighted_refit_signal_noise Q W
    (fun i (_ : Unit) => β i) (fun i (_ : Unit) => z i) σ hQ hW
  exact congrArg (fun M : Matrix (Fin 3) Unit ℝ => fun i => M i ()) h

/-- Actual original-basis pairs refits have the same normalized coefficient
map as the bootstrap theorem. Full rank is required only for the accepted
weighted Gram matrix. -/
theorem weighted_ols_error [DecidableEq ι] (P : Matrix ι (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (W : Matrix ι ι ℝ)
    (hW : IsUnit (OLS.gram (normalizedDesign P) W).det)
    (β : Fin 3 → ℝ) (z : ι → ℝ) (σ : ℝ) :
    weightedOlsCoefficient P W (P *ᵥ β + σ • z) -
        olsCoefficient P (P *ᵥ β + σ • z) =
      σ • (whitener P *ᵥ (OLS.coefficientMap (normalizedDesign P) W *ᵥ z)) := by
  have hY : P *ᵥ β + σ • z =
      normalizedDesign P *ᵥ ((whitener P)⁻¹ *ᵥ β) + σ • z := by
    rw [Matrix.mulVec_mulVec, normalizedDesign_mul_inv P hP]
  unfold weightedOlsCoefficient olsCoefficient
  rw [weighted_olsMatrix_eq_whitened P hP, olsMatrix_eq_whitened]
  have hm : whitener P * (OLS.gram (normalizedDesign P) W)⁻¹ * (normalizedDesign P)ᵀ * W =
      whitener P * ((OLS.gram (normalizedDesign P) W)⁻¹ * (normalizedDesign P)ᵀ * W) := by
    simp only [Matrix.mul_assoc]
  rw [hm,
    ← Matrix.mulVec_mulVec (P *ᵥ β + σ • z) (whitener P)
      ((OLS.gram (normalizedDesign P) W)⁻¹ * (normalizedDesign P)ᵀ * W),
    ← Matrix.mulVec_mulVec (P *ᵥ β + σ • z) (whitener P) (normalizedDesign P)ᵀ,
    ← Matrix.mulVec_sub, hY,
    normalized_refit_error _ (normalizedDesign_orthonormal P hP) W hW,
    Matrix.mulVec_smul]

theorem weighted_prediction_error [DecidableEq ι] (P : Matrix ι (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (W : Matrix ι ι ℝ)
    (hW : IsUnit (OLS.gram (normalizedDesign P) W).det)
    (β : Fin 3 → ℝ) (z : ι → ℝ) (σ u : ℝ) :
    prediction (weightedOlsCoefficient P W (P *ᵥ β + σ • z)) u -
        prediction (olsCoefficient P (P *ᵥ β + σ • z)) u =
      σ * inner (𝕜 := ℝ) (directionEquiv P hP (quadraticFeatures u))
        ((EuclideanSpace.equiv (Fin 3) ℝ).symm
          (OLS.coefficientMap (normalizedDesign P) W *ᵥ z)) := by
  rw [prediction_eq_dot, prediction_eq_dot, ← dotProduct_sub,
    weighted_ols_error P hP W hW, dotProduct_smul, smul_eq_mul, direction_inner]

/-- The original fitted residual is exactly the normalized residual projection. -/
theorem original_residual [DecidableEq ι] (P : Matrix ι (Fin 3) ℝ) (Y : ι → ℝ) :
    Y - P *ᵥ olsCoefficient P Y = OLS.residual (normalizedDesign P) *ᵥ Y := by
  unfold olsCoefficient
  rw [Matrix.mulVec_mulVec, olsMatrix_eq_whitened, ← Matrix.mul_assoc]
  change Y - (normalizedDesign P * (normalizedDesign P)ᵀ) *ᵥ Y = _
  simp only [OLS.residual, Matrix.sub_mulVec, Matrix.one_mulVec]

theorem original_residual_signal_noise [DecidableEq ι] (P : Matrix ι (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (β : Fin 3 → ℝ) (z : ι → ℝ) (σ : ℝ) :
    (P *ᵥ β + σ • z) - P *ᵥ olsCoefficient P (P *ᵥ β + σ • z) =
      σ • (OLS.residual (normalizedDesign P) *ᵥ z) := by
  rw [original_residual, Matrix.mulVec_add, Matrix.mulVec_smul, Matrix.mulVec_mulVec]
  have hR : OLS.residual (normalizedDesign P) * P = 0 := by
    conv_lhs => arg 2; rw [← normalizedDesign_mul_inv P hP]
    rw [← Matrix.mul_assoc,
      OLS.residual_annihilates_design _ (normalizedDesign_orthonormal P hP), Matrix.zero_mul]
  rw [hR, Matrix.zero_mulVec, zero_add]

/-- The fitted standard deviation is a function only of the observed data,
using the original OLS residual and n-3 residual degrees of freedom. -/
def fittedResidualStd (n : ℕ) (P : Matrix (Fin n) (Fin 3) ℝ) (Y : Fin n → ℝ) : ℝ :=
  ‖(EuclideanSpace.equiv (Fin n) ℝ).symm (Y - P *ᵥ olsCoefficient P Y)‖ /
    Real.sqrt ((n : ℝ) - 3)

theorem fittedResidualStd_eq (n : ℕ) (P : Matrix (Fin n) (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (β : Fin 3 → ℝ) (z : Fin n → ℝ)
    (σ : ℝ) (hσ : 0 ≤ σ) :
    fittedResidualStd n P (P *ᵥ β + σ • z) =
      σ * ResidualScale.fittedScaleRatio n (normalizedDesign P) z := by
  unfold fittedResidualStd ResidualScale.fittedScaleRatio
  rw [original_residual_signal_noise P hP, map_smul, norm_smul,
    Real.norm_eq_abs, abs_of_nonneg hσ, mul_div_assoc]

theorem standardErrorShape_pos (P : Matrix ι (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (u : ℝ) : 0 < standardErrorShape P u := by
  rw [standardErrorShape_eq_norm P hP]
  apply norm_pos_iff.mpr
  intro hz
  exact quadraticFeatures_ne_zero u ((directionEquiv P hP).injective (by simpa using hz))

/-- The actual studentized supremum over the entire real action line. Its
denominator uses the original data fit and is fixed across pairs resamples. -/
def weightedScore (n : ℕ) (P : Matrix (Fin n) (Fin 3) ℝ)
    (W : Matrix (Fin n) (Fin n) ℝ) (Y : Fin n → ℝ) : ℝ :=
  ⨆ u : ℝ, |prediction (weightedOlsCoefficient P W Y) u -
    prediction (olsCoefficient P Y) u| / (fittedResidualStd n P Y * standardErrorShape P u)

theorem weighted_studentized_supremum [DecidableEq ι] (P : Matrix ι (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (W : Matrix ι ι ℝ)
    (hW : IsUnit (OLS.gram (normalizedDesign P) W).det)
    (β : Fin 3 → ℝ) (z : ι → ℝ) (σ fitted : ℝ) (hσ : 0 ≤ σ) (hfitted : 0 < fitted) :
    (⨆ u : ℝ, |prediction (weightedOlsCoefficient P W (P *ᵥ β + σ • z)) u -
      prediction (olsCoefficient P (P *ᵥ β + σ • z)) u| / (fitted * standardErrorShape P u)) =
      (σ / fitted) * quadraticPiSupremum (directionEquiv P hP)
        (OLS.coefficientMap (normalizedDesign P) W *ᵥ z) := by
  change _ = (σ / fitted) * quadraticSupremum (directionEquiv P hP)
    ((EuclideanSpace.equiv (Fin 3) ℝ).symm (OLS.coefficientMap (normalizedDesign P) W *ᵥ z))
  rw [quadraticSupremum_apply, Real.mul_iSup_of_nonneg (div_nonneg hσ hfitted.le)]
  congr 1
  funext u
  rw [weighted_prediction_error P hP W hW, abs_mul, abs_of_nonneg hσ,
    standardErrorShape_eq_norm P hP]
  ring

/-- Under the model, the observable pairs score equals the normalized
coefficient statistic divided by the one shared fitted-scale ratio. -/
theorem weightedScore_eq (n : ℕ) (P : Matrix (Fin n) (Fin 3) ℝ)
    (hP : IsUnit (Pᵀ * P)) (W : Matrix (Fin n) (Fin n) ℝ)
    (hW : IsUnit (OLS.gram (normalizedDesign P) W).det)
    (β : Fin 3 → ℝ) (z : Fin n → ℝ) (σ : ℝ) (hσ : 0 < σ)
    (hscale : 0 < ResidualScale.fittedScaleRatio n (normalizedDesign P) z) :
    weightedScore n P W (P *ᵥ β + σ • z) =
      quadraticPiSupremum (directionEquiv P hP) (OLS.coefficientMap (normalizedDesign P) W *ᵥ z) /
        ResidualScale.fittedScaleRatio n (normalizedDesign P) z := by
  unfold weightedScore
  rw [weighted_studentized_supremum P hP W hW β z σ _ hσ.le (by
    rw [fittedResidualStd_eq n P hP β z σ hσ.le]
    exact mul_pos hσ hscale), fittedResidualStd_eq n P hP β z σ hσ.le]
  field_simp
  ring

end BootstrapCalibration.RegressionInterpretation
