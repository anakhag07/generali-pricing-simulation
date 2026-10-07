import BootstrapCalibration.MultinomialMoments
import BootstrapCalibration.OLS
import BootstrapCalibration.AcceptedLaw

/-! Exact design-dependent second moments under the actual finite iid
multinomial sampling law. Squared Frobenius norms are explicit finite sums. -/

noncomputable section
open scoped BigOperators Classical

namespace BootstrapCalibration

namespace FiniteLaw

variable {Ω : Type} [Fintype Ω] (P : FiniteLaw Ω)

lemma expectation_sum {ι : Type*} [Fintype ι] (f : ι → Ω → ℝ) :
    P.expectation (fun x => ∑ i, f i x) = ∑ i, P.expectation (f i) := by
  simp only [expectation, Finset.mul_sum]
  rw [Finset.sum_comm]

/-- Factorization is proved from the actual product-law weights. -/
lemma product_expectation_mul {Ξ : Type} [Fintype Ξ] (Q : FiniteLaw Ξ)
    (f : Ω → ℝ) (g : Ξ → ℝ) :
    (P.product Q).expectation (fun x => f x.1 * g x.2) =
      P.expectation f * Q.expectation g := by
  rw [P.product_expectation]
  unfold expectation
  rw [Finset.sum_mul_sum]
  apply Finset.sum_congr rfl
  intro x hx
  apply Finset.sum_congr rfl
  intro y hy
  ring

/-- The elementary covariance expansion for a finite weighted sum. -/
lemma expectation_weighted_sum_sq {ι : Type*} [Fintype ι]
    (ξ : ι → Ω → ℝ) (v : ι → ℝ) (a b : ℝ)
    (hcov : ∀ i j, P.expectation (fun x => ξ i x * ξ j x) =
      a * (if i = j then 1 else 0) + b) :
    P.expectation (fun x => (∑ i, ξ i x * v i)^2) =
      a * (∑ i, (v i)^2) + b * (∑ i, v i)^2 := by
  have hex (x : Ω) : (∑ i, ξ i x * v i)^2 =
      ∑ i, ∑ j, (ξ i x * ξ j x) * (v i * v j) := by
    rw [pow_two, Finset.sum_mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    apply Finset.sum_congr rfl
    intro j hj
    ring
  rw [P.expectation_congr hex]
  simp_rw [expectation_sum, expectation_mul_const, hcov]
  simp only [add_mul, Finset.sum_add_distrib]
  have hdiag : (∑ i, ∑ j, a * (if i = j then 1 else 0) * (v i * v j)) =
      a * (∑ i, (v i)^2) := by
    simp [mul_ite, ite_mul, ← Finset.mul_sum, eq_comm, pow_two]
  rw [hdiag]
  congr 1
  simp only [pow_two, Finset.mul_sum, Finset.sum_mul, mul_assoc]
  rw [Finset.sum_comm]

end FiniteLaw

namespace DesignMoments

variable {n p : ℕ}

/-- The squared Frobenius norm, without an ambient matrix norm convention. -/
def frobeniusSq {m q : Type*} [Fintype m] [Fintype q] (A : Matrix m q ℝ) : ℝ :=
  ∑ i, ∑ j, (A i j)^2

def leverage (Q : Matrix (Fin n) (Fin p) ℝ) (i : Fin n) : ℝ :=
  ∑ a, (Q i a)^2

def leverageSqSum (Q : Matrix (Fin n) (Fin p) ℝ) : ℝ :=
  ∑ i, (leverage Q i)^2

/-- Entrywise form of Q-transpose times a diagonal matrix times Q. -/
def weightedGram (Q : Matrix (Fin n) (Fin p) ℝ) (ξ : Fin n → ℝ) :
    Matrix (Fin p) (Fin p) ℝ :=
  fun a b => ∑ i, ξ i * (Q i a * Q i b)

lemma weightedGram_eq (Q : Matrix (Fin n) (Fin p) ℝ) (ξ : Fin n → ℝ) :
    weightedGram Q ξ = Q.transpose * Matrix.diagonal ξ * Q := by
  ext a b
  simp [weightedGram, Matrix.mul_apply, Matrix.diagonal, Matrix.transpose_apply]
  apply Finset.sum_congr rfl
  intro i hi
  ring

lemma gram_entry (Q : Matrix (Fin n) (Fin p) ℝ) (a b : Fin p) :
    (Q.transpose * Q) a b = ∑ i, Q i a * Q i b := by
  rfl

lemma weightedGram_sub_constant (Q : Matrix (Fin n) (Fin p) ℝ)
    (ξ : Fin n → ℝ) (c : ℝ) :
    weightedGram Q (fun i => ξ i - c) =
      weightedGram Q ξ - c • (Q.transpose * Q) := by
  ext a b
  simp only [weightedGram, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul,
    gram_entry, sub_mul, Finset.sum_sub_distrib]
  rw [← Finset.mul_sum]

lemma fourth_entry_sum (Q : Matrix (Fin n) (Fin p) ℝ) :
    (∑ a, ∑ b, ∑ i, (Q i a * Q i b)^2) = leverageSqSum Q := by
  calc
    _ = ∑ a, ∑ i, ∑ b, (Q i a * Q i b)^2 := by
      apply Finset.sum_congr rfl
      intro a ha
      rw [Finset.sum_comm]
    _ = ∑ i, ∑ a, ∑ b, (Q i a * Q i b)^2 := by
      rw [Finset.sum_comm]
    _ = leverageSqSum Q := by
      unfold leverageSqSum leverage
      apply Finset.sum_congr rfl
      intro i hi
      simp only [mul_pow, pow_two, Finset.sum_mul_sum]
      apply Finset.sum_congr rfl
      intro a ha
      apply Finset.sum_congr rfl
      intro b hb
      ring

/-- Covariance coefficients a and b give an exact design moment after the
entry covariances have been derived from a finite sampling law. -/
lemma expectation_frobenius_weightedGram {Ω : Type} [Fintype Ω]
    (P : FiniteLaw Ω) (Q : Matrix (Fin n) (Fin p) ℝ)
    (ξ : Fin n → Ω → ℝ) (a b : ℝ)
    (hcov : ∀ i j, P.expectation (fun x => ξ i x * ξ j x) =
      a * (if i = j then 1 else 0) + b) :
    P.expectation (fun x => frobeniusSq (weightedGram Q (fun i => ξ i x))) =
      a * leverageSqSum Q + b * frobeniusSq (Q.transpose * Q) := by
  unfold frobeniusSq weightedGram
  simp_rw [FiniteLaw.expectation_sum,
    P.expectation_weighted_sum_sq ξ _ a b (by
      intro i j
      by_cases hij : i = j <;> simpa [hij] using hcov i j)]
  simp only [Finset.sum_add_distrib]
  simp_rw [← Finset.mul_sum]
  rw [fourth_entry_sum]
  rfl

@[simp] lemma frobeniusSq_one : frobeniusSq (1 : Matrix (Fin p) (Fin p) ℝ) = p := by
  simp [frobeniusSq, Matrix.one_apply]

/-- The alpha identity derives from actual size-n iid uniform row draws. -/
theorem multinomial_design_alpha (n p : ℕ) (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin p) ℝ) (hQ : Q.transpose * Q = 1) :
    ((uniformRows n hn).iid n).expectation
      (fun x => frobeniusSq (weightedGram Q (fun i => centeredRowCount i x))) =
      leverageSqSum Q - (p : ℝ) / n := by
  have hcov := multinomial_covariance n hn
  have hm := expectation_frobenius_weightedGram ((uniformRows n hn).iid n) Q
    centeredRowCount 1 (-1/(n : ℝ)) (by
      intro i j
      rw [hcov i j]
      ring)
  rw [hQ, frobeniusSq_one] at hm
  refine hm.trans ?_
  ring

/-- The gamma identity uses the proved fourth and cross moments of the
actual multinomial counts, rather than an assumed covariance structure. -/
theorem multinomial_design_gamma (n p : ℕ) (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin p) ℝ) (hQ : Q.transpose * Q = 1) :
    ((uniformRows n hn).iid n).expectation
      (fun x => frobeniusSq (weightedGram Q
        (fun i => (centeredRowCount i x)^2 - (1 - 1/(n : ℝ))))) =
      (3 - 10/(n : ℝ) + 8/(n : ℝ)^2) * leverageSqSum Q +
        (-1/(n : ℝ) + 6/(n : ℝ)^2 - 6/(n : ℝ)^3) * p := by
  have hcov : ∀ i j : Fin n,
      ((uniformRows n hn).iid n).expectation
        (fun x => ((centeredRowCount i x)^2 - (1 - 1/(n : ℝ))) *
          ((centeredRowCount j x)^2 - (1 - 1/(n : ℝ)))) =
      (3 - 10/(n : ℝ) + 8/(n : ℝ)^2) * (if i = j then 1 else 0) +
        (-1/(n : ℝ) + 6/(n : ℝ)^2 - 6/(n : ℝ)^3) := by
    intro i j
    by_cases hij : i = j
    · subst j
      calc
        _ = 3 - 11/(n : ℝ) + 14/(n : ℝ)^2 - 6/(n : ℝ)^3 := by
          simpa only [← sq] using multinomial_square_variance n hn i
        _ = _ := by simp only [↓reduceIte]; ring
    · simpa only [hij, ↓reduceIte, mul_zero, zero_add] using
        multinomial_square_covariance n hn i j hij
  have hm := expectation_frobenius_weightedGram ((uniformRows n hn).iid n) Q
    (fun i x => (centeredRowCount i x)^2 - (1 - 1/(n : ℝ)))
    (3 - 10/(n : ℝ) + 8/(n : ℝ)^2)
    (-1/(n : ℝ) + 6/(n : ℝ)^2 - 6/(n : ℝ)^3) hcov
  simpa only [hQ, frobeniusSq_one] using hm

/-- Two independent actual row resamples yield the zeta cross-weight moment. -/
theorem multinomial_design_zeta (n p : ℕ) (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin p) ℝ) (hQ : Q.transpose * Q = 1) :
    (((uniformRows n hn).iid n).product ((uniformRows n hn).iid n)).expectation
      (fun x => frobeniusSq (weightedGram Q
        (fun i => centeredRowCount i x.1 * centeredRowCount i x.2))) =
      (1 - 2/(n : ℝ)) * leverageSqSum Q + (p : ℝ)/(n : ℝ)^2 := by
  let P := (uniformRows n hn).iid n
  have hcov (i j : Fin n) :
      (P.product P).expectation
        (fun x => (centeredRowCount i x.1 * centeredRowCount i x.2) *
          (centeredRowCount j x.1 * centeredRowCount j x.2)) =
      (1 - 2/(n : ℝ)) * (if i = j then 1 else 0) + 1/(n : ℝ)^2 := by
    calc
      _ = (P.product P).expectation
          (fun x => (centeredRowCount i x.1 * centeredRowCount j x.1) *
            (centeredRowCount i x.2 * centeredRowCount j x.2)) := by
        apply FiniteLaw.expectation_congr
        intro x
        ring
      _ = (P.expectation (fun x => centeredRowCount i x * centeredRowCount j x))^2 := by
        simpa only [sq] using P.product_expectation_mul P
          (fun x => centeredRowCount i x * centeredRowCount j x)
          (fun x => centeredRowCount i x * centeredRowCount j x)
      _ = _ := by
        dsimp [P]
        rw [multinomial_covariance n hn i j]
        by_cases hij : i = j <;> simp only [hij, ↓reduceIte] <;> ring
  have hm := expectation_frobenius_weightedGram (P.product P) Q
    (fun i x => centeredRowCount i x.1 * centeredRowCount i x.2)
    (1 - 2/(n : ℝ)) (1/(n : ℝ)^2) hcov
  rw [hQ, frobeniusSq_one] at hm
  refine hm.trans ?_
  ring

/-- Alpha in the explicit matrix notation used in the statistical proof. -/
theorem multinomial_matrix_alpha (n p : ℕ) (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin p) ℝ) (hQ : Q.transpose * Q = 1) :
    ((uniformRows n hn).iid n).expectation
      (fun x => frobeniusSq
        (Q.transpose * Matrix.diagonal (fun i => centeredRowCount i x) * Q)) =
      leverageSqSum Q - (p : ℝ) / n := by
  simpa only [weightedGram_eq] using multinomial_design_alpha n p hn Q hQ

/-- Gamma in matrix form, including subtraction of the true mean matrix. -/
theorem multinomial_matrix_gamma (n p : ℕ) (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin p) ℝ) (hQ : Q.transpose * Q = 1) :
    ((uniformRows n hn).iid n).expectation
      (fun x => frobeniusSq
        (Q.transpose * Matrix.diagonal (fun i => (centeredRowCount i x)^2) * Q -
          (1 - 1/(n : ℝ)) • (1 : Matrix (Fin p) (Fin p) ℝ))) =
      (3 - 10/(n : ℝ) + 8/(n : ℝ)^2) * leverageSqSum Q +
        (-1/(n : ℝ) + 6/(n : ℝ)^2 - 6/(n : ℝ)^3) * p := by
  have hg (x : Draws (Fin n) n) :
      weightedGram Q (fun i => (centeredRowCount i x)^2 - (1 - 1/(n : ℝ))) =
        Q.transpose * Matrix.diagonal (fun i => (centeredRowCount i x)^2) * Q -
          (1 - 1/(n : ℝ)) • (1 : Matrix (Fin p) (Fin p) ℝ) := by
    rw [weightedGram_sub_constant Q (fun i => (centeredRowCount i x)^2)
      (1 - 1/(n : ℝ)), hQ, weightedGram_eq]
  simpa only [hg] using multinomial_design_gamma n p hn Q hQ

/-- Zeta is calculated under the independent product of two actual iid
resampling laws, with no independent-covariance assumption. -/
theorem multinomial_matrix_zeta (n p : ℕ) (hn : 0 < n)
    (Q : Matrix (Fin n) (Fin p) ℝ) (hQ : Q.transpose * Q = 1) :
    (((uniformRows n hn).iid n).product ((uniformRows n hn).iid n)).expectation
      (fun x => frobeniusSq
        (Q.transpose * Matrix.diagonal
          (fun i => centeredRowCount i x.1 * centeredRowCount i x.2) * Q)) =
      (1 - 2/(n : ℝ)) * leverageSqSum Q + (p : ℝ)/(n : ℝ)^2 := by
  simpa only [weightedGram_eq] using multinomial_design_zeta n p hn Q hQ

end DesignMoments
end BootstrapCalibration
