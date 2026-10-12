import BootstrapCalibration.QuadraticRank
import Mathlib

/-! Constructive matrix whitening of a full-rank quadratic design through the
inverse positive-semidefinite square root of its original Gram matrix. -/

noncomputable section
open scoped Matrix BigOperators
namespace BootstrapCalibration.Whitening

variable {ι : Type} [Fintype ι]

lemma gram_posSemidef (P : Matrix ι (Fin 3) ℝ) : (Pᵀ * P).PosSemidef := by
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    Matrix.posSemidef_conjTranspose_mul_self P

/-- A fixed symmetric whitening matrix computed from the original design. -/
def whitener (P : Matrix ι (Fin 3) ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  (gram_posSemidef P).sqrt⁻¹

/-- The square root has full rank whenever the original Gram matrix does. -/
lemma gram_sqrt_isUnit (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P)) :
    IsUnit (gram_posSemidef P).sqrt := by
  have hdet : IsUnit (Pᵀ * P).det := (Matrix.isUnit_iff_isUnit_det _).mp hP
  have hsq := congrArg Matrix.det (gram_posSemidef P).sqrt_mul_self
  rw [Matrix.det_mul] at hsq
  have hs : IsUnit ((gram_posSemidef P).sqrt.det * (gram_posSemidef P).sqrt.det) := by
    rw [hsq]
    exact hdet
  exact (Matrix.isUnit_iff_isUnit_det _).mpr ((IsUnit.mul_iff.mp hs).1)

lemma whitener_isUnit (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P)) :
    IsUnit (whitener P) := Matrix.isUnit_nonsing_inv_iff.mpr (gram_sqrt_isUnit P hP)

lemma gram_sqrt_transpose (P : Matrix ι (Fin 3) ℝ) :
    (gram_posSemidef P).sqrtᵀ = (gram_posSemidef P).sqrt := by
  simpa only [Matrix.IsHermitian, Matrix.conjTranspose_eq_transpose_of_trivial] using
    (gram_posSemidef P).posSemidef_sqrt.isHermitian

lemma whitener_transpose (P : Matrix ι (Fin 3) ℝ) : (whitener P)ᵀ = whitener P := by
  rw [whitener, Matrix.transpose_nonsing_inv, gram_sqrt_transpose]

/-- The covariance represented by the whitening coordinates is the inverse Gram matrix. -/
theorem whitener_covariance (P : Matrix ι (Fin 3) ℝ) :
    whitener P * (whitener P)ᵀ = (Pᵀ * P)⁻¹ := by
  rw [whitener_transpose, whitener, ← Matrix.mul_inv_rev, Matrix.PosSemidef.sqrt_mul_self]

/-- Whitening makes the original quadratic Gram matrix exactly the identity. -/
theorem whitener_gram (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P)) :
    (whitener P)ᵀ * (Pᵀ * P) * whitener P = 1 := by
  have hB := gram_sqrt_isUnit P hP
  have hdetB := (Matrix.isUnit_iff_isUnit_det _).mp hB
  rw [whitener_transpose, ← (gram_posSemidef P).sqrt_mul_self]
  unfold whitener
  calc
    (gram_posSemidef P).sqrt⁻¹ * ((gram_posSemidef P).sqrt * (gram_posSemidef P).sqrt) *
        (gram_posSemidef P).sqrt⁻¹ =
      ((gram_posSemidef P).sqrt⁻¹ * (gram_posSemidef P).sqrt) *
        ((gram_posSemidef P).sqrt * (gram_posSemidef P).sqrt⁻¹) := by simp only [Matrix.mul_assoc]
    _ = 1 := by rw [Matrix.nonsing_inv_mul _ hdetB, Matrix.mul_nonsing_inv _ hdetB, Matrix.one_mul]

def normalizedDesign (P : Matrix ι (Fin 3) ℝ) : Matrix ι (Fin 3) ℝ := P * whitener P

/-- The normalized design has orthonormal columns, derived from the original design. -/
theorem normalizedDesign_orthonormal (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P)) :
    (normalizedDesign P)ᵀ * normalizedDesign P = 1 := by
  unfold normalizedDesign
  rw [Matrix.transpose_mul]
  have h := whitener_gram P hP
  simpa only [Matrix.mul_assoc] using h

/-- Distinct logged actions supply original-design full rank before whitening. -/
theorem quadratic_original_gram_isUnit (n : ℕ) (hn : 3 ≤ n)
    (u : Fin n → ℝ) (hu : Function.Injective u) :
    IsUnit ((QuadraticRank.design u)ᵀ * QuadraticRank.design u) := by
  have h := QuadraticRank.weightedGram_isUnit_of_three u hu (fun _ => 1)
    (by intro i; norm_num) (by simpa [QuadraticRank.positiveSupport] using hn)
  simpa only [QuadraticRank.weightedGram, Matrix.diagonal_one, Matrix.mul_one] using h

/-- Existence with an explicit inverse-root witness for the original quadratic model. -/
theorem quadratic_whitening (n : ℕ) (hn : 3 ≤ n)
    (u : Fin n → ℝ) (hu : Function.Injective u) :
    ∃ C : Matrix (Fin 3) (Fin 3) ℝ, IsUnit C ∧ Cᵀ = C ∧
      (QuadraticRank.design u * C)ᵀ * (QuadraticRank.design u * C) = 1 ∧
      C * Cᵀ = ((QuadraticRank.design u)ᵀ * QuadraticRank.design u)⁻¹ := by
  let P := QuadraticRank.design u
  have hP := quadratic_original_gram_isUnit n hn u hu
  exact ⟨whitener P, whitener_isUnit P hP, whitener_transpose P,
    normalizedDesign_orthonormal P hP, whitener_covariance P⟩

/-- An invertible real matrix gives the continuous coordinate equivalence used
by the normalized continuous-domain quadratic statistic. -/
def matrixContinuousEquiv (C : Matrix (Fin 3) (Fin 3) ℝ) (hC : IsUnit C) :
    EuclideanSpace ℝ (Fin 3) ≃L[ℝ] EuclideanSpace ℝ (Fin 3) :=
  ContinuousLinearEquiv.ofUnit
    (Units.map (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin 3)).toMonoidHom hC.unit)

lemma matrixContinuousEquiv_apply (C : Matrix (Fin 3) (Fin 3) ℝ) (hC : IsUnit C)
    (v : EuclideanSpace ℝ (Fin 3)) :
    matrixContinuousEquiv C hC v = Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin 3) C v := by
  change Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin 3) (hC.unit : Matrix (Fin 3) (Fin 3) ℝ) v = _
  rw [hC.unit_spec]

/-- The invertible map L=Cᵀ entering the normalized quadratic evaluation directions. -/
def directionEquiv (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P)) :
    EuclideanSpace ℝ (Fin 3) ≃L[ℝ] EuclideanSpace ℝ (Fin 3) :=
  matrixContinuousEquiv (whitener P)ᵀ
    ((Matrix.isUnit_transpose _).mpr (whitener_isUnit P hP))

lemma directionEquiv_apply (P : Matrix ι (Fin 3) ℝ) (hP : IsUnit (Pᵀ * P))
    (v : EuclideanSpace ℝ (Fin 3)) :
    directionEquiv P hP v = Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin 3) (whitener P)ᵀ v := by
  exact matrixContinuousEquiv_apply _ _ v

end BootstrapCalibration.Whitening
