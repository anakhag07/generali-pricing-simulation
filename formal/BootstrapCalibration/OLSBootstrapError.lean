import BootstrapCalibration.PairsCalibration
import BootstrapCalibration.BootstrapError

/-! Closed finite-N covariance error for actual accepted multinomial pairs
resampling of an orthonormal quadratic design. -/

noncomputable section
open scoped BigOperators Classical Matrix.L2OpNorm
open Matrix
namespace BootstrapCalibration.OLSBootstrapError
open DesignMoments StableCovariance BootstrapError

variable {n : ℕ}

def alpha (Q : Matrix (Fin n) (Fin 3) ℝ) : ℝ := leverageSqSum Q - 3 / (n : ℝ)
def gamma (Q : Matrix (Fin n) (Fin 3) ℝ) : ℝ :=
  (3 - 10/(n : ℝ) + 8/(n : ℝ)^2) * leverageSqSum Q +
    (-1/(n : ℝ) + 6/(n : ℝ)^2 - 6/(n : ℝ)^3) * 3
def zeta (Q : Matrix (Fin n) (Fin 3) ℝ) : ℝ :=
  (1 - 2/(n : ℝ)) * leverageSqSum Q + 3/(n : ℝ)^2
def diagonalConstant (Q : Matrix (Fin n) (Fin 3) ℝ) (a : ℝ) : ℝ :=
  Real.sqrt (gamma Q) + Real.sqrt 3 / (n : ℝ) +
    2 * (1 + a) * Real.sqrt (alpha Q)
def crossConstant (Q : Matrix (Fin n) (Fin 3) ℝ) : ℝ :=
  Real.sqrt (zeta Q) + alpha Q
def epsilon (Q : Matrix (Fin n) (Fin 3) ℝ) (a r : ℝ) : ℝ :=
  min 1 (2 * alpha Q / (a^2 * r^2) +
    Real.sqrt (2 * (diagonalConstant Q a)^2 + 2 * (crossConstant Q)^2) /
      ((1 - a)^2 * r))

def pairNumerator (Q : Matrix (Fin n) (Fin 3) ℝ)
    (x y : Draws (Fin n) n) : Matrix (Fin 3) (Fin 3) ℝ :=
  Qᵀ * (pairsCountMatrix x - 1) * (pairsCountMatrix y - 1) * Q

lemma countMatrix_centered (x : Draws (Fin n) n) :
    pairsCountMatrix x - 1 = diagonal (fun i => centeredRowCount i x) := by
  rw [pairsCountMatrix, ← diagonal_one, diagonal_sub]
  rfl

lemma numerator_diagonal (Q : Matrix (Fin n) (Fin 3) ℝ)
    (x y : Draws (Fin n) n) :
    pairNumerator Q x y =
      Qᵀ * diagonal (fun i => centeredRowCount i x * centeredRowCount i y) * Q := by
  simp only [pairNumerator, countMatrix_centered]
  rw [Matrix.mul_assoc Qᵀ, diagonal_mul_diagonal]

lemma gram_centered (Q : Matrix (Fin n) (Fin 3) ℝ) (hQ : Qᵀ * Q = 1)
    (x : Draws (Fin n) n) :
    OLS.gram Q (pairsCountMatrix x) - 1 =
      Qᵀ * diagonal (fun i => centeredRowCount i x) * Q := by
  rw [← countMatrix_centered]
  exact (OLS.centered_gram Q (pairsCountMatrix x) hQ).symm

lemma alpha_moment (hn : 0 < n) (Q : Matrix (Fin n) (Fin 3) ℝ) (hQ : Qᵀ * Q = 1) :
    (pairsRowLaw n hn).expectation
      (fun x => frobeniusSq (OLS.gram Q (pairsCountMatrix x) - 1)) = alpha Q := by
  simp_rw [gram_centered Q hQ]
  exact multinomial_matrix_alpha n 3 hn Q hQ

lemma gamma_moment (hn : 0 < n) (Q : Matrix (Fin n) (Fin 3) ℝ) (hQ : Qᵀ * Q = 1) :
    (pairsRowLaw n hn).expectation
      (fun x => frobeniusSq (pairNumerator Q x x - (1 - 1/(n : ℝ)) • 1)) = gamma Q := by
  simp_rw [numerator_diagonal, ← sq]
  exact multinomial_matrix_gamma n 3 hn Q hQ

lemma zeta_moment (hn : 0 < n) (Q : Matrix (Fin n) (Fin 3) ℝ) (hQ : Qᵀ * Q = 1) :
    ((pairsRowLaw n hn).product (pairsRowLaw n hn)).expectation
      (fun x => frobeniusSq (pairNumerator Q x.1 x.2)) = zeta Q := by
  simp_rw [numerator_diagonal]
  exact multinomial_matrix_zeta n 3 hn Q hQ

lemma product_expectation_fst {Ω : Type} [Fintype Ω] (P : FiniteLaw Ω) (f : Ω → ℝ) :
    (P.product P).expectation (fun x => f x.1) = P.expectation f := by
  simpa only [mul_one, FiniteLaw.expectation_const] using
    P.product_expectation_mul P f (fun _ => 1)

lemma product_expectation_snd {Ω : Type} [Fintype Ω] (P : FiniteLaw Ω) (f : Ω → ℝ) :
    (P.product P).expectation (fun x => f x.2) = P.expectation f := by
  simpa only [one_mul, FiniteLaw.expectation_const] using
    P.product_expectation_mul P (fun _ => 1) f

lemma frobeniusSq_transpose {ι κ : Type*} [Fintype ι] [Fintype κ]
    (A : Matrix ι κ ℝ) : frobeniusSq Aᵀ = frobeniusSq A := by
  unfold frobeniusSq
  exact Finset.sum_comm

lemma frobeniusSq_fromBlocks {ι κ : Type*} [Fintype ι] [Fintype κ]
    (A : Matrix ι ι ℝ) (B : Matrix ι κ ℝ) (C : Matrix κ ι ℝ) (D : Matrix κ κ ℝ) :
    frobeniusSq (fromBlocks A B C D) =
      frobeniusSq A + frobeniusSq B + frobeniusSq C + frobeniusSq D := by
  simp only [frobeniusSq, Fintype.sum_sum_type, fromBlocks_apply₁₁, fromBlocks_apply₁₂,
    fromBlocks_apply₂₁, fromBlocks_apply₂₂, Finset.sum_add_distrib]
  ring

/-- Exact Frobenius decomposition of the covariance of two shared-noise
coefficient maps; the off-diagonal blocks are transposes. -/
lemma pair_covariance_frobenius {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq κ]
    (A B : Matrix κ ι ℝ) :
    frobeniusSq (fromRows A B * (fromRows A B)ᵀ - 1) =
      frobeniusSq (A * Aᵀ - 1) + frobeniusSq (B * Bᵀ - 1) +
        2 * frobeniusSq (A * Bᵀ) := by
  have hb : fromRows A B * (fromRows A B)ᵀ - 1 =
      fromBlocks (A * Aᵀ - 1) (A * Bᵀ) (B * Aᵀ) (B * Bᵀ - 1) := by
    rw [transpose_fromRows, fromRows_mul_fromCols]
    ext (i | i) (j | j) <;> simp [Matrix.one_apply]
  rw [hb, frobeniusSq_fromBlocks]
  have ht : frobeniusSq (B * Aᵀ) = frobeniusSq (A * Bᵀ) := by
    rw [← frobeniusSq_transpose (A * Bᵀ), Matrix.transpose_mul, Matrix.transpose_transpose]
  rw [ht]
  ring

lemma pairs_coefficient_covariance (Q : Matrix (Fin n) (Fin 3) ℝ)
    (hQ : Qᵀ * Q = 1) (x y : Draws (Fin n) n) :
    pairsCoefficientMap Q x * (pairsCoefficientMap Q y)ᵀ =
      (OLS.gram Q (pairsCountMatrix x))⁻¹ *
        (pairNumerator Q x y -
          (OLS.gram Q (pairsCountMatrix x) - 1) * (OLS.gram Q (pairsCountMatrix y) - 1)) *
      (OLS.gram Q (pairsCountMatrix y))⁻¹ := by
  exact OLS.coefficientMap_cross_covariance Q (pairsCountMatrix x) (pairsCountMatrix y) hQ
    (Matrix.diagonal_transpose _)

/-- Closed-form finite-N calibration error for the actual accepted pairs
law. Every moment and stability estimate is discharged from the design and
the multinomial law; the only assumptions are orthonormality, positive
acceptance probability and a stability parameter strictly between zero and one. -/
theorem covarianceCalibrationError_le (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin 3) ℝ) (hQ : Qᵀ * Q = 1)
    (hr : 0 < (pairsRowLaw n hn).mass (pairsAccept Q))
    (a : ℝ) (ha0 : 0 < a) (ha : a < 1) :
    covarianceCalibrationError (acceptedPairsLaw n hn Q hr) (pairsCoefficientMap Q) ≤
      epsilon Q a ((pairsRowLaw n hn).mass (pairsAccept Q)) := by
  let P := pairsRowLaw n hn
  let G : Draws (Fin n) n → Matrix (Fin 3) (Fin 3) ℝ :=
    fun x => OLS.gram Q (pairsCountMatrix x)
  let S : Draws (Fin n) n × Draws (Fin n) n → Prop :=
    fun x => ‖G x.1 - 1‖ ≤ a ∧ ‖G x.2 - 1‖ ≤ a
  let A : Draws (Fin n) n × Draws (Fin n) n → Prop :=
    fun x => pairsAccept Q x.1 ∧ pairsAccept Q x.2
  let r := P.mass (pairsAccept Q)
  let f : Draws (Fin n) n × Draws (Fin n) n → ℝ := fun x =>
    Real.sqrt (frobeniusSq (fromRows (pairsCoefficientMap Q x.1) (pairsCoefficientMap Q x.2) *
      (fromRows (pairsCoefficientMap Q x.1) (pairsCoefficientMap Q x.2))ᵀ - 1))
  have hmass : (P.product P).mass A = r^2 := by
    rw [FiniteLaw.product_mass]
    exact (pow_two r).symm
  have hSA : ∀ x, S x → A x := by
    intro x hx
    exact ⟨(Matrix.isUnit_iff_isUnit_det _).mpr (stable_isUnit_det _ a ha hx.1),
      (Matrix.isUnit_iff_isUnit_det _).mpr (stable_isUnit_det _ a ha hx.2)⟩
  have halpha : P.expectation (fun x => frobeniusSq (G x - 1)) = alpha Q :=
    alpha_moment hn Q hQ
  have hα₁ : (P.product P).expectation (fun x => frobeniusSq (G x.1 - 1)) = alpha Q :=
    (product_expectation_fst P _).trans halpha
  have hα₂ : (P.product P).expectation (fun x => frobeniusSq (G x.2 - 1)) = alpha Q :=
    (product_expectation_snd P _).trans halpha
  have hγ₁ : (P.product P).expectation
      (fun x => frobeniusSq (pairNumerator Q x.1 x.1 - (1 - 1/(n : ℝ)) • 1)) = gamma Q :=
    (product_expectation_fst P _).trans (gamma_moment hn Q hQ)
  have hγ₂ : (P.product P).expectation
      (fun x => frobeniusSq (pairNumerator Q x.2 x.2 - (1 - 1/(n : ℝ)) • 1)) = gamma Q :=
    (product_expectation_snd P _).trans (gamma_moment hn Q hQ)
  have hE : (P.product P).expectation
      (fun x => frobeniusSq ((G x.1 - 1) * (G x.2 - 1))) ≤ (alpha Q)^2 := by
    have h := independent_product_frobenius_moment P (fun x => G x - 1)
    rwa [halpha] at h
  have hbad : (P.product P).mass (fun x => ¬ S x) ≤ 2 * alpha Q / a^2 :=
    unstable_pair_mass_bound (P.product P) (fun x => G x.1) (fun x => G x.2)
      a (alpha Q) ha0 hα₁ hα₂
  have hf (x : Draws (Fin n) n × Draws (Fin n) n) : (f x)^2 =
      frobeniusSq ((G x.1)⁻¹ *
        (pairNumerator Q x.1 x.1 - (G x.1 - 1) * (G x.1 - 1)) * (G x.1)⁻¹ - 1) +
      frobeniusSq ((G x.2)⁻¹ *
        (pairNumerator Q x.2 x.2 - (G x.2 - 1) * (G x.2 - 1)) * (G x.2)⁻¹ - 1) +
      2 * frobeniusSq ((G x.1)⁻¹ *
        (pairNumerator Q x.1 x.2 - (G x.1 - 1) * (G x.2 - 1)) * (G x.2)⁻¹) := by
    dsimp [f]
    rw [Real.sq_sqrt (by
      unfold frobeniusSq
      exact Finset.sum_nonneg (fun i hi => Finset.sum_nonneg (fun j hj => sq_nonneg _))),
      pair_covariance_frobenius]
    simp only [pairs_coefficient_covariance Q hQ]
    rfl
  have hv : 1 - 1/(n : ℝ) ≤ 1 := sub_le_self _ (by positivity)
  have hd : Real.sqrt (gamma Q) + (1 - (1 - 1/(n : ℝ))) * Real.sqrt (3 : ℕ) +
      2 * (1 + a) * Real.sqrt (alpha Q) = diagonalConstant Q a := by
    unfold diagonalConstant
    norm_num
    ring
  have hmoment : (P.product P).expectation
      (fun x => @ite ℝ (S x) (Classical.propDecidable (S x)) ((f x)^2) 0) ≤
      (2 * (diagonalConstant Q a)^2 + 2 * (crossConstant Q)^2) / (1 - a)^4 := by
    simp_rw [hf]
    have h := stable_pair_blocks_moment_bound (P.product P)
      (fun x => G x.1) (fun x => G x.2)
      (fun x => pairNumerator Q x.1 x.1) (fun x => pairNumerator Q x.2 x.2)
      (fun x => pairNumerator Q x.1 x.2) S a (1 - 1/(n : ℝ))
      (alpha Q) (gamma Q) (zeta Q) ha0.le ha hv
      (fun x hx => hx.1) (fun x hx => hx.2) hα₁ hα₂ hγ₁ hγ₂ (zeta_moment hn Q hQ) hE
    rw [hd] at h
    convert h using 1
    apply FiniteLaw.expectation_congr
    intro x
    by_cases hx : S x <;> simp [hx]
  have hbound := accepted_covariance_error_bound (P.product P) A S r a (alpha Q)
    (diagonalConstant Q a) (crossConstant Q) hr ha hmass hSA f hbad hmoment
  change _ ≤ epsilon Q a r
  change _ ≤ epsilon Q a r at hbound
  calc
    covarianceCalibrationError (acceptedPairsLaw n hn Q hr) (pairsCoefficientMap Q) =
        ((P.accepted (pairsAccept Q) hr).product (P.accepted (pairsAccept Q) hr)).expectation
          (fun x => min 1 (f x)) := by
      symm
      exact FiniteLaw.product_expectation _ _ _
    _ = ((P.product P).accepted A (by rw [hmass]; exact sq_pos_of_pos hr)).expectation
        (fun x => min 1 (f x)) := by rw [FiniteLaw.product_accepted]
    _ ≤ epsilon Q a r := hbound

end BootstrapCalibration.OLSBootstrapError
