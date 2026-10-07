import Mathlib

/-!
Exact Frobenius covariance identities obtained from orthogonal diagonalization.
These statements use explicit entry sums, independent of a matrix norm instance.
-/

noncomputable section
open scoped BigOperators Matrix
namespace BootstrapCalibration.SpectralComparison

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- The entrywise squared Frobenius norm equals the Gram-matrix trace. -/
theorem frobeniusSq_eq_trace (A : Matrix ι κ ℝ) :
    (∑ i, ∑ j, (A i j)^2) = Matrix.trace (Aᵀ * A) := by
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply, pow_two]
  exact Finset.sum_comm

/-- Orthogonal multiplication on the left preserves the squared Frobenius norm. -/
theorem frobeniusSq_orthogonal_left [DecidableEq ι]
    (U : Matrix ι ι ℝ) (A : Matrix ι κ ℝ) (hU : Uᵀ * U = 1) :
    (∑ i, ∑ j, ((U * A) i j)^2) = ∑ i, ∑ j, (A i j)^2 := by
  rw [frobeniusSq_eq_trace, frobeniusSq_eq_trace, Matrix.transpose_mul]
  congr 1
  calc
    Aᵀ * Uᵀ * (U * A) = Aᵀ * (Uᵀ * U) * A := by simp only [Matrix.mul_assoc]
    _ = Aᵀ * A := by rw [hU, Matrix.mul_one]

/-- Orthogonal multiplication on the right preserves the squared Frobenius norm. -/
theorem frobeniusSq_orthogonal_right [DecidableEq κ]
    (A : Matrix ι κ ℝ) (U : Matrix κ κ ℝ) (hU : Uᵀ * U = 1) :
    (∑ i, ∑ j, ((A * Uᵀ) i j)^2) = ∑ i, ∑ j, (A i j)^2 := by
  rw [frobeniusSq_eq_trace, frobeniusSq_eq_trace, Matrix.transpose_mul, Matrix.transpose_transpose]
  calc
    Matrix.trace (U * Aᵀ * (A * Uᵀ)) = Matrix.trace (U * (Aᵀ * A) * Uᵀ) := by
      simp only [Matrix.mul_assoc]
    _ = Matrix.trace (Uᵀ * U * (Aᵀ * A)) := Matrix.trace_mul_cycle _ _ _
    _ = Matrix.trace (Aᵀ * A) := by rw [hU, Matrix.one_mul]

/-- The diagonal case of the squared Frobenius norm. -/
theorem frobeniusSq_diagonal [DecidableEq ι] (d : ι → ℝ) :
    (∑ i, ∑ j, ((Matrix.diagonal d) i j)^2) = ∑ i, (d i)^2 := by
  simp [Matrix.diagonal_apply]

/-- Entrywise covariance error equals squared eigenvalue error for an explicit
orthogonal spectral representation. -/
theorem covariance_error_of_diagonalization [DecidableEq ι]
    (S U : Matrix ι ι ℝ) (eig : ι → ℝ)
    (hU : Uᵀ * U = 1) (hU' : U * Uᵀ = 1)
    (hspec : S = U * Matrix.diagonal eig * Uᵀ) :
    (∑ i, ∑ j, ((S - 1) i j)^2) = ∑ i, (eig i - 1)^2 := by
  have hdiff : S - 1 = U * Matrix.diagonal (fun i => eig i - 1) * Uᵀ := by
    rw [← Matrix.diagonal_sub, Matrix.diagonal_one, Matrix.mul_sub, Matrix.sub_mul,
      Matrix.mul_one, hU', ← hspec]
  rw [hdiff, frobeniusSq_orthogonal_right _ U hU,
    frobeniusSq_orthogonal_left U _ hU, frobeniusSq_diagonal]

/-- The preceding identity for mathlib's eigenvalues of any real symmetric matrix. -/
theorem covariance_error_eq_eigenvalues [DecidableEq ι]
    (S : Matrix ι ι ℝ) (hS : S.IsHermitian) :
    (∑ i, ∑ j, ((S - 1) i j)^2) = ∑ i, (hS.eigenvalues i - 1)^2 := by
  let U : Matrix ι ι ℝ := hS.eigenvectorUnitary
  have hU : Uᵀ * U = 1 := by
    simpa only [U, Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial]
      using unitary.coe_star_mul_self hS.eigenvectorUnitary
  have hU' : U * Uᵀ = 1 := by
    simpa only [U, unitary.coe_star, Matrix.star_eq_conjTranspose,
      Matrix.conjTranspose_eq_transpose_of_trivial]
      using unitary.coe_mul_star_self hS.eigenvectorUnitary
  apply covariance_error_of_diagonalization S U hS.eigenvalues hU hU'
  simpa only [U, RCLike.ofReal_real_eq_id, Function.id_comp, Matrix.star_eq_conjTranspose,
    Matrix.conjTranspose_eq_transpose_of_trivial] using hS.spectral_theorem

/-- A real Gram covariance has nonnegative eigenvalues. -/
theorem gram_eigenvalues_nonneg [DecidableEq ι] (A : Matrix ι κ ℝ) (i : ι) :
    0 ≤ (Matrix.isHermitian_mul_conjTranspose_self A).eigenvalues i :=
  Matrix.eigenvalues_self_mul_conjTranspose_nonneg A i

/-- The spectral squared-error identity instantiated for a Gram covariance. -/
theorem gram_covariance_error_eq_eigenvalues [DecidableEq ι] (A : Matrix ι κ ℝ) :
    (∑ i, ∑ j, (((A * Aᵀ - 1) : Matrix ι ι ℝ) i j)^2) =
      ∑ i, ((Matrix.isHermitian_mul_conjTranspose_self A).eigenvalues i - 1)^2 := by
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    covariance_error_eq_eigenvalues (A * Aᴴ) (Matrix.isHermitian_mul_conjTranspose_self A)

end BootstrapCalibration.SpectralComparison
