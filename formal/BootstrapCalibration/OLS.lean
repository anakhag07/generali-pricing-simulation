import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Tactic

/-!
# Exact matrix algebra for quadratic pairs-bootstrap OLS

These results apply to every finite real design after whitening. They prove
the residual/refit and covariance-numerator identities; no bootstrap validity
or Gaussian distribution identity is assumed or asserted here.
-/

namespace BootstrapCalibration.OLS

open Matrix

variable {n p k : Type*} [Fintype n] [Fintype p]
  [DecidableEq n] [DecidableEq p]

def residual (Q : Matrix n p ℝ) : Matrix n n ℝ := 1 - Q * Q.transpose

def gram (Q : Matrix n p ℝ) (W : Matrix n n ℝ) : Matrix p p ℝ :=
  Q.transpose * W * Q

def perturbation (Q : Matrix n p ℝ) (W : Matrix n n ℝ) : Matrix p p ℝ :=
  gram Q W - 1

omit [Fintype n] [DecidableEq p] in
theorem transpose_residual (Q : Matrix n p ℝ) :
    (residual Q).transpose = residual Q := by
  simp [residual, Matrix.transpose_mul]

theorem orthogonal_residual (Q : Matrix n p ℝ)
    (hQ : Q.transpose * Q = 1) :
    Q.transpose * residual Q = 0 := by
  simp only [residual, Matrix.mul_sub, Matrix.mul_one]
  rw [← Matrix.mul_assoc, hQ, Matrix.one_mul, sub_self]

theorem residual_idempotent (Q : Matrix n p ℝ)
    (hQ : Q.transpose * Q = 1) :
    residual Q * residual Q = residual Q := by
  have h : Q.transpose * residual Q = 0 := orthogonal_residual Q hQ
  conv_lhs => lhs; unfold residual
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc, h,
    Matrix.mul_zero, sub_zero]

theorem residual_covariance (Q : Matrix n p ℝ)
    (hQ : Q.transpose * Q = 1) :
    residual Q * (residual Q).transpose = residual Q := by
  rw [transpose_residual, residual_idempotent Q hQ]

theorem residual_annihilates_design (Q : Matrix n p ℝ)
    (hQ : Q.transpose * Q = 1) : residual Q * Q = 0 := by
  simp [residual, Matrix.sub_mul, Matrix.mul_assoc, hQ]

omit [DecidableEq n] in
/-- The whitened OLS coefficient consists of signal plus projected noise. -/
theorem whitened_fit_signal_noise (Q : Matrix n p ℝ)
    (β : Matrix p k ℝ) (Z : Matrix n k ℝ) (σ : ℝ)
    (hQ : Q.transpose * Q = 1) :
    Q.transpose * (Q * β + σ • Z) = β + σ • (Q.transpose * Z) := by
  rw [Matrix.mul_add, ← Matrix.mul_assoc, hQ, Matrix.one_mul, Matrix.mul_smul]

theorem residual_signal_noise (Q : Matrix n p ℝ)
    (β : Matrix p k ℝ) (Z : Matrix n k ℝ) (σ : ℝ)
    (hQ : Q.transpose * Q = 1) :
    residual Q * (Q * β + σ • Z) = σ • (residual Q * Z) := by
  rw [Matrix.mul_add, ← Matrix.mul_assoc, residual_annihilates_design Q hQ,
    Matrix.zero_mul, zero_add, Matrix.mul_smul]

omit [Fintype p] in
theorem centered_gram (Q : Matrix n p ℝ) (W : Matrix n n ℝ)
    (hQ : Q.transpose * Q = 1) :
    Q.transpose * (W - 1) * Q = perturbation Q W := by
  simp [perturbation, gram, Matrix.mul_sub, Matrix.sub_mul, hQ]

theorem center_weights_on_residual (Q : Matrix n p ℝ) (W : Matrix n n ℝ)
    (hQ : Q.transpose * Q = 1) :
    Q.transpose * W * residual Q =
      Q.transpose * (W - 1) * residual Q := by
  simp [Matrix.mul_sub, Matrix.sub_mul, orthogonal_residual Q hQ]

/-- Weighted OLS minus the original whitened OLS fit is a weighted residual
refit. The assumption is nonsingularity of the resampled Gram matrix. -/
theorem weighted_refit_difference (Q : Matrix n p ℝ) (W : Matrix n n ℝ)
    (Y : Matrix n k ℝ) (hW : IsUnit (gram Q W).det) :
    (gram Q W)⁻¹ * Q.transpose * W * Y - Q.transpose * Y =
      (gram Q W)⁻¹ * Q.transpose * W * residual Q * Y := by
  have hinv : (gram Q W)⁻¹ * (Q.transpose * W * Q) = 1 :=
    Matrix.nonsing_inv_mul (gram Q W) hW
  simp only [residual, Matrix.mul_sub, Matrix.mul_one, Matrix.sub_mul]
  have hcancel : (gram Q W)⁻¹ * Q.transpose * W * (Q * Q.transpose) * Y =
      Q.transpose * Y := by
    calc
      _ = ((gram Q W)⁻¹ * (Q.transpose * W * Q)) * (Q.transpose * Y) := by
        simp only [Matrix.mul_assoc]
      _ = _ := by rw [hinv, Matrix.one_mul]
  rw [hcancel]

/-- The exact centered pairs-bootstrap refit identity, derived from the
weighted normal equations and the orthonormal-column condition. -/
theorem centered_weighted_refit_difference (Q : Matrix n p ℝ)
    (W : Matrix n n ℝ) (Y : Matrix n k ℝ)
    (hQ : Q.transpose * Q = 1) (hW : IsUnit (gram Q W).det) :
    (gram Q W)⁻¹ * Q.transpose * W * Y - Q.transpose * Y =
      (gram Q W)⁻¹ * Q.transpose * (W - 1) * residual Q * Y := by
  rw [weighted_refit_difference Q W Y hW]
  have h := center_weights_on_residual Q W hQ
  simpa only [Matrix.mul_assoc] using
    congrArg (fun M => (gram Q W)⁻¹ * M * Y) h

omit [DecidableEq p] in
theorem residual_sandwich (Q : Matrix n p ℝ)
    (S T : Matrix n n ℝ) :
    Q.transpose * S * residual Q * T * Q =
      Q.transpose * S * T * Q -
        (Q.transpose * S * Q) * (Q.transpose * T * Q) := by
  simp only [residual, Matrix.mul_sub, Matrix.mul_one, Matrix.sub_mul,
    Matrix.mul_assoc]

/-- Cross-covariance numerator for two bootstrap weight draws sharing the
same Gaussian residual vector. This is an algebraic identity only. -/
theorem cross_covariance_numerator (Q : Matrix n p ℝ)
    (W U : Matrix n n ℝ) (hQ : Q.transpose * Q = 1) :
    Q.transpose * (W - 1) * residual Q * (U - 1) * Q =
      Q.transpose * (W - 1) * (U - 1) * Q -
        perturbation Q W * perturbation Q U := by
  rw [residual_sandwich]
  rw [centered_gram Q W hQ, centered_gram Q U hQ]

theorem diagonal_covariance_numerator (Q : Matrix n p ℝ)
    (W : Matrix n n ℝ) (hQ : Q.transpose * Q = 1) :
    Q.transpose * (W - 1) * residual Q * (W - 1) * Q =
      Q.transpose * (W - 1) * (W - 1) * Q -
        perturbation Q W * perturbation Q W :=
  cross_covariance_numerator Q W W hQ

noncomputable def coefficientMap (Q : Matrix n p ℝ) (W : Matrix n n ℝ) : Matrix p n ℝ :=
  (gram Q W)⁻¹ * Q.transpose * (W - 1) * residual Q

omit [Fintype p] [DecidableEq n] [DecidableEq p] in
theorem transpose_gram (Q : Matrix n p ℝ) (W : Matrix n n ℝ)
    (hW : W.transpose = W) :
    (gram Q W).transpose = gram Q W := by
  simp [gram, Matrix.transpose_mul, hW, Matrix.mul_assoc]

omit [DecidableEq n] in
theorem transpose_gram_inverse (Q : Matrix n p ℝ) (W : Matrix n n ℝ)
    (hW : W.transpose = W) :
    ((gram Q W)⁻¹).transpose = (gram Q W)⁻¹ := by
  rw [Matrix.transpose_nonsing_inv, transpose_gram Q W hW]

/-- A pairs refit difference removes the unknown signal and scales linearly
with the observation noise. This proves the exact pivot's algebraic part. -/
theorem weighted_refit_signal_noise (Q : Matrix n p ℝ) (W : Matrix n n ℝ)
    (β : Matrix p k ℝ) (Z : Matrix n k ℝ) (σ : ℝ)
    (hQ : Q.transpose * Q = 1) (hW : IsUnit (gram Q W).det) :
    (gram Q W)⁻¹ * Q.transpose * W * (Q * β + σ • Z) -
        Q.transpose * (Q * β + σ • Z) =
      σ • (coefficientMap Q W * Z) := by
  rw [centered_weighted_refit_difference Q W (Q * β + σ • Z) hQ hW]
  rw [Matrix.mul_assoc, residual_signal_noise Q β Z σ hQ, Matrix.mul_smul]
  simp only [coefficientMap, Matrix.mul_assoc]

/-- The covariance product of the two exact coefficient maps has the
candidate proof's cross-block form. Symmetry applies to diagonal counts. -/
theorem coefficientMap_cross_covariance (Q : Matrix n p ℝ)
    (W U : Matrix n n ℝ) (hQ : Q.transpose * Q = 1)
    (hU : U.transpose = U) :
    coefficientMap Q W * (coefficientMap Q U).transpose =
      (gram Q W)⁻¹ *
        (Q.transpose * (W - 1) * (U - 1) * Q -
          perturbation Q W * perturbation Q U) * (gram Q U)⁻¹ := by
  calc
    _ = (gram Q W)⁻¹ * Q.transpose * (W - 1) *
        (residual Q * residual Q) * (U - 1) * Q * (gram Q U)⁻¹ := by
      simp only [coefficientMap, Matrix.transpose_mul, Matrix.transpose_sub,
        Matrix.transpose_one, Matrix.transpose_transpose, transpose_residual,
        transpose_gram_inverse Q U hU, hU, Matrix.mul_assoc]
    _ = (gram Q W)⁻¹ *
        (Q.transpose * (W - 1) * residual Q * (U - 1) * Q) * (gram Q U)⁻¹ := by
      rw [residual_idempotent Q hQ]
      simp only [Matrix.mul_assoc]
    _ = _ := by rw [cross_covariance_numerator Q W U hQ]

theorem coefficientMap_diagonal_covariance (Q : Matrix n p ℝ)
    (W : Matrix n n ℝ) (hQ : Q.transpose * Q = 1)
    (hW : W.transpose = W) :
    coefficientMap Q W * (coefficientMap Q W).transpose =
      (gram Q W)⁻¹ *
        (Q.transpose * (W - 1) * (W - 1) * Q -
          perturbation Q W * perturbation Q W) * (gram Q W)⁻¹ :=
  coefficientMap_cross_covariance Q W W hQ hW

/-- Exact covariance error numerator for any nonsingular finite Gram matrix.
The identity is derived algebraically, before any norm bound is applied. -/
theorem covariance_error_numerator (A B : Matrix p p ℝ)
    (hA : IsUnit A.det) :
    A⁻¹ * (B - (A - 1) * (A - 1)) * A⁻¹ - 1 =
      A⁻¹ * (B - 1 - 2 • (A - 1) - 2 • ((A - 1) * (A - 1))) * A⁻¹ := by
  have hcancel : A⁻¹ * (A * A) * A⁻¹ = 1 := by
    calc
      _ = (A⁻¹ * A) * (A * A⁻¹) := by simp only [Matrix.mul_assoc]
      _ = _ := by rw [Matrix.nonsing_inv_mul A hA, Matrix.mul_nonsing_inv A hA,
        Matrix.one_mul]
  have hnum : B - (A - 1) * (A - 1) - A * A =
      B - 1 - 2 • (A - 1) - 2 • ((A - 1) * (A - 1)) := by
    noncomm_ring
  rw [← hnum]
  conv_rhs => rw [Matrix.mul_sub, Matrix.sub_mul, hcancel]

end BootstrapCalibration.OLS
