import BootstrapCalibration.DesignMoments
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.SpecificLimits.Normed

/-! Operator stability and mixed Euclidean-operator/Frobenius bounds.
The `Matrix.L2OpNorm` scope is essential: all plain matrix norms in this file
are Euclidean operator norms, never the default entrywise supremum norm. -/

noncomputable section
open scoped BigOperators Classical Matrix.L2OpNorm

namespace BootstrapCalibration.StableCovariance

open DesignMoments Matrix

variable {p : ℕ}

def frob (A : Matrix (Fin p) (Fin p) ℝ) : ℝ := Real.sqrt (frobeniusSq A)

lemma frobeniusSq_nonneg (A : Matrix (Fin p) (Fin p) ℝ) : 0 ≤ frobeniusSq A := by
  exact Finset.sum_nonneg (fun i hi => Finset.sum_nonneg (fun j hj => sq_nonneg _))

lemma frob_nonneg (A : Matrix (Fin p) (Fin p) ℝ) : 0 ≤ frob A := Real.sqrt_nonneg _

lemma frob_sq (A : Matrix (Fin p) (Fin p) ℝ) : (frob A)^2 = frobeniusSq A :=
  Real.sq_sqrt (frobeniusSq_nonneg A)

/-- A Euclidean operator perturbation smaller than one is nonsingular. -/
theorem stable_isUnit_det (A : Matrix (Fin p) (Fin p) ℝ)
    (a : ℝ) (ha : a < 1) (hA : ‖A - 1‖ ≤ a) : IsUnit A.det := by
  have hs : ‖(1 : Matrix (Fin p) (Fin p) ℝ) - A‖ < 1 := by
    rw [norm_sub_rev]
    exact hA.trans_lt ha
  have hu := isUnit_one_sub_of_norm_lt_one hs
  simpa only [sub_sub_cancel] using (Matrix.isUnit_iff_isUnit_det _).mp hu

/-- The inverse operator bound is proved from the inverse identity and norm
inequalities; it is not an assumed stability estimate. -/
theorem stable_inverse_norm (A : Matrix (Fin p) (Fin p) ℝ)
    (a : ℝ) (ha : a < 1) (hA : ‖A - 1‖ ≤ a) :
    ‖A⁻¹‖ ≤ 1 / (1 - a) := by
  have hu := stable_isUnit_det A a ha hA
  have hident : A⁻¹ = 1 - (A - 1) * A⁻¹ := by
    rw [Matrix.sub_mul, Matrix.mul_nonsing_inv A hu, Matrix.one_mul]
    abel
  have hone : ‖(1 : Matrix (Fin p) (Fin p) ℝ)‖ ≤ 1 := by
    rw [Matrix.cstar_norm_def, map_one]
    exact ContinuousLinearMap.norm_id_le
  have hb : ‖A⁻¹‖ ≤ 1 + a * ‖A⁻¹‖ := by
    calc
      ‖A⁻¹‖ = ‖1 - (A - 1) * A⁻¹‖ := congrArg norm hident
      _ ≤ ‖(1 : Matrix (Fin p) (Fin p) ℝ)‖ + ‖(A - 1) * A⁻¹‖ := norm_sub_le _ _
      _ ≤ 1 + a * ‖A⁻¹‖ := add_le_add hone
        ((norm_mul_le _ _).trans (mul_le_mul_of_nonneg_right hA (norm_nonneg _)))
  apply (le_div_iff₀ (sub_pos.mpr ha)).mpr
  nlinarith

lemma euclidean_norm_sq (v : Fin p → ℝ) :
    ‖(EuclideanSpace.equiv (Fin p) ℝ).symm v‖^2 = ∑ i, (v i)^2 := by
  simp only [PiLp.norm_sq_eq_of_L2, Real.norm_eq_abs, sq_abs]
  rfl

/-- Left multiplication contracts Frobenius norm by at most the Euclidean
operator norm, proved by applying the operator bound to each column. -/
theorem frobeniusSq_mul_le (A B : Matrix (Fin p) (Fin p) ℝ) :
    frobeniusSq (A * B) ≤ ‖A‖^2 * frobeniusSq B := by
  have hcol (j : Fin p) :
      (∑ i, ((A * B) i j)^2) ≤ ‖A‖^2 * (∑ i, (B i j)^2) := by
    let v : EuclideanSpace ℝ (Fin p) :=
      (EuclideanSpace.equiv (Fin p) ℝ).symm (fun i => B i j)
    have h := Matrix.l2_opNorm_mulVec A v
    have hs := sq_le_sq₀ (norm_nonneg _) (mul_nonneg (norm_nonneg _) (norm_nonneg _)) |>.mpr h
    rw [mul_pow] at hs
    have hv : ‖v‖^2 = ∑ i, (B i j)^2 := euclidean_norm_sq _
    rw [hv, euclidean_norm_sq] at hs
    exact hs
  unfold frobeniusSq
  rw [Finset.sum_comm, Finset.sum_comm (f := fun i j => (B i j)^2)]
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum (fun j hj => hcol j)

theorem frob_mul_le (A B : Matrix (Fin p) (Fin p) ℝ) :
    frob (A * B) ≤ ‖A‖ * frob B := by
  apply (sq_le_sq₀ (frob_nonneg _) (mul_nonneg (norm_nonneg _) (frob_nonneg _))).mp
  rw [mul_pow, frob_sq, frob_sq]
  exact frobeniusSq_mul_le A B

lemma frob_transpose (A : Matrix (Fin p) (Fin p) ℝ) : frob A.transpose = frob A := by
  unfold frob frobeniusSq
  congr 1
  simp only [Matrix.transpose_apply]
  rw [Finset.sum_comm]

lemma operator_norm_transpose (A : Matrix (Fin p) (Fin p) ℝ) : ‖A.transpose‖ = ‖A‖ := by
  simpa only [Matrix.conjTranspose, star_trivial] using Matrix.l2_opNorm_conjTranspose A

theorem frob_mul_le_right (A B : Matrix (Fin p) (Fin p) ℝ) :
    frob (A * B) ≤ frob A * ‖B‖ := by
  calc
    frob (A * B) = frob (B.transpose * A.transpose) := by
      rw [← Matrix.transpose_mul, frob_transpose]
    _ ≤ ‖B.transpose‖ * frob A.transpose := frob_mul_le _ _
    _ = frob A * ‖B‖ := by rw [frob_transpose, operator_norm_transpose, mul_comm]

theorem frob_sandwich_le (A B C : Matrix (Fin p) (Fin p) ℝ) :
    frob (A * B * C) ≤ ‖A‖ * frob B * ‖C‖ := by
  exact (frob_mul_le_right (A * B) C).trans
    (mul_le_mul_of_nonneg_right (frob_mul_le A B) (norm_nonneg C))

def flat (A : Matrix (Fin p) (Fin p) ℝ) : EuclideanSpace ℝ (Fin p × Fin p) :=
  (EuclideanSpace.equiv (Fin p × Fin p) ℝ).symm (fun ij => A ij.1 ij.2)

lemma frob_eq_flat_norm (A : Matrix (Fin p) (Fin p) ℝ) : frob A = ‖flat A‖ := by
  apply (sq_eq_sq₀ (frob_nonneg _) (norm_nonneg _)).mp
  rw [frob_sq, PiLp.norm_sq_eq_of_L2, Fintype.sum_prod_type]
  simp only [flat, Real.norm_eq_abs, sq_abs]
  rfl

lemma frob_add_le (A B : Matrix (Fin p) (Fin p) ℝ) :
    frob (A + B) ≤ frob A + frob B := by
  simp only [frob_eq_flat_norm]
  have hflat : flat (A + B) = flat A + flat B := rfl
  rw [hflat]
  exact norm_add_le _ _

lemma frob_neg (A : Matrix (Fin p) (Fin p) ℝ) : frob (-A) = frob A := by
  simp [frob, frobeniusSq]

lemma frob_sub_le (A B : Matrix (Fin p) (Fin p) ℝ) :
    frob (A - B) ≤ frob A + frob B := by
  simpa only [sub_eq_add_neg, frob_neg] using frob_add_le A (-B)

lemma frob_smul (c : ℝ) (A : Matrix (Fin p) (Fin p) ℝ) :
    frob (c • A) = |c| * frob A := by
  simp only [frob_eq_flat_norm]
  have hflat : flat (c • A) = c • flat A := rfl
  rw [hflat, norm_smul, Real.norm_eq_abs]

lemma frob_nsmul_two (A : Matrix (Fin p) (Fin p) ℝ) : frob (2 • A) = 2 * frob A := by
  simpa only [two_smul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] using
    frob_smul (2 : ℝ) A

lemma frob_one : frob (1 : Matrix (Fin p) (Fin p) ℝ) = Real.sqrt p := by
  simp only [frob, frobeniusSq_one]

theorem operator_norm_le_frob (A : Matrix (Fin p) (Fin p) ℝ) : ‖A‖ ≤ frob A := by
  rw [Matrix.cstar_norm_def]
  apply ContinuousLinearMap.opNorm_le_bound _ (frob_nonneg A)
  intro v
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (frob_nonneg _) (norm_nonneg _))).mp
  rw [mul_pow, frob_sq]
  have hrow (i : Fin p) :
      ((A *ᵥ v) i)^2 ≤ (∑ j, (A i j)^2) * (∑ j, (v j)^2) := by
    exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun j => A i j) (fun j => v j)
  have hs := Finset.sum_le_sum (fun i (hi : i ∈ Finset.univ) => hrow i)
  simp only [← Finset.sum_mul] at hs
  simpa only [PiLp.norm_sq_eq_of_L2, Real.norm_eq_abs, sq_abs, frobeniusSq] using hs

theorem frob_mul_le_frob (A B : Matrix (Fin p) (Fin p) ℝ) :
    frob (A * B) ≤ frob A * frob B :=
  (frob_mul_le A B).trans
    (mul_le_mul_of_nonneg_right (operator_norm_le_frob A) (frob_nonneg B))

lemma frob_centered_error (B : Matrix (Fin p) (Fin p) ℝ)
    (v : ℝ) (hv : v ≤ 1) :
    frob (B - 1) ≤ frob (B - v • 1) + (1 - v) * Real.sqrt p := by
  have he : B - 1 = (B - v • 1) + (v - 1) • (1 : Matrix (Fin p) (Fin p) ℝ) := by
    ext i j
    simp only [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  rw [he]
  convert frob_add_le (B - v • 1) ((v - 1) • (1 : Matrix (Fin p) (Fin p) ℝ)) using 1
  rw [frob_smul, frob_one, abs_of_nonpos (sub_nonpos.mpr hv)]
  ring

theorem stable_square_frob (E : Matrix (Fin p) (Fin p) ℝ)
    (a : ℝ) (hE : ‖E‖ ≤ a) : frob (E * E) ≤ a * frob E :=
  (frob_mul_le E E).trans (mul_le_mul_of_nonneg_right hE (frob_nonneg E))

/-- A stable inverse bounds the Frobenius norm of a sandwich uniformly. -/
theorem stable_frob_sandwich (A B C : Matrix (Fin p) (Fin p) ℝ)
    (a : ℝ) (ha : a < 1) (hA : ‖A - 1‖ ≤ a) (hC : ‖C - 1‖ ≤ a) :
    frob (A⁻¹ * B * C⁻¹) ≤ frob B / (1 - a)^2 := by
  have hiA := stable_inverse_norm A a ha hA
  have hiC := stable_inverse_norm C a ha hC
  have hp : 0 ≤ 1 / (1 - a) := le_of_lt (one_div_pos.mpr (sub_pos.mpr ha))
  calc
    frob (A⁻¹ * B * C⁻¹) ≤ ‖A⁻¹‖ * frob B * ‖C⁻¹‖ := frob_sandwich_le _ _ _
    _ ≤ (1 / (1 - a)) * frob B * (1 / (1 - a)) :=
      mul_le_mul (mul_le_mul_of_nonneg_right hiA (frob_nonneg B)) hiC
        (norm_nonneg _) (mul_nonneg hp (frob_nonneg B))
    _ = _ := by
      field_simp
      ring

/-- Deterministic diagonal-block error, before the residual covariance
matrix is centered by its known multinomial expectation. -/
theorem stable_diagonal_error (A B : Matrix (Fin p) (Fin p) ℝ)
    (a : ℝ) (ha : a < 1) (hA : ‖A - 1‖ ≤ a) :
    frob (A⁻¹ * (B - (A - 1) * (A - 1)) * A⁻¹ - 1) ≤
      (frob (B - 1) + 2 * (1 + a) * frob (A - 1)) / (1 - a)^2 := by
  rw [OLS.covariance_error_numerator A B (stable_isUnit_det A a ha hA)]
  refine (stable_frob_sandwich A _ A a ha hA hA).trans ?_
  apply div_le_div_of_nonneg_right _ (sq_nonneg _)
  have h1 := frob_sub_le (B - 1 - 2 • (A - 1)) (2 • ((A - 1) * (A - 1)))
  have h2 := frob_sub_le (B - 1) (2 • (A - 1))
  rw [frob_nsmul_two] at h1 h2
  have h3 := stable_square_frob (A - 1) a hA
  nlinarith

/-- Deterministic cross-block error for two stable independent-draw Gram
matrices. Independence is only used later when taking its expectation. -/
theorem stable_cross_error (A C D : Matrix (Fin p) (Fin p) ℝ)
    (a : ℝ) (ha : a < 1) (hA : ‖A - 1‖ ≤ a) (hC : ‖C - 1‖ ≤ a) :
    frob (A⁻¹ * (D - (A - 1) * (C - 1)) * C⁻¹) ≤
      (frob D + frob ((A - 1) * (C - 1))) / (1 - a)^2 := by
  exact (stable_frob_sandwich A _ C a ha hA hC).trans
    (div_le_div_of_nonneg_right (frob_sub_le _ _) (sq_nonneg _))

theorem stable_centered_diagonal_error (A B : Matrix (Fin p) (Fin p) ℝ)
    (a v : ℝ) (ha : a < 1) (hA : ‖A - 1‖ ≤ a) (hv : v ≤ 1) :
    frob (A⁻¹ * (B - (A - 1) * (A - 1)) * A⁻¹ - 1) ≤
      (frob (B - v • 1) + (1 - v) * Real.sqrt p +
        2 * (1 + a) * frob (A - 1)) / (1 - a)^2 := by
  refine (stable_diagonal_error A B a ha hA).trans ?_
  apply div_le_div_of_nonneg_right _ (sq_nonneg _)
  exact add_le_add_right (frob_centered_error B v hv) _

section FiniteL2

variable {Ω : Type} [Fintype Ω]

def weightedVector (P : FiniteLaw Ω) (f : Ω → ℝ) : EuclideanSpace ℝ Ω :=
  (EuclideanSpace.equiv Ω ℝ).symm (fun x => Real.sqrt (P.weight x) * f x)

/-- Root mean square under a finite law, represented as a genuine Euclidean
norm so that Minkowski's inequality is inherited from mathlib. -/
def rms (P : FiniteLaw Ω) (f : Ω → ℝ) : ℝ := ‖weightedVector P f‖

lemma rms_nonneg (P : FiniteLaw Ω) (f : Ω → ℝ) : 0 ≤ rms P f := norm_nonneg _

lemma rms_sq (P : FiniteLaw Ω) (f : Ω → ℝ) :
    (rms P f)^2 = P.expectation (fun x => (f x)^2) := by
  rw [rms, PiLp.norm_sq_eq_of_L2]
  change (∑ x, ‖Real.sqrt (P.weight x) * f x‖^2) = _
  simp only [Real.norm_eq_abs, sq_abs, mul_pow, Real.sq_sqrt (P.nonneg _),
    FiniteLaw.expectation]

lemma rms_add_le (P : FiniteLaw Ω) (f g : Ω → ℝ) :
    rms P (fun x => f x + g x) ≤ rms P f + rms P g := by
  have he : weightedVector P (fun x => f x + g x) =
      weightedVector P f + weightedVector P g := by
    ext x
    change Real.sqrt (P.weight x) * (f x + g x) =
      Real.sqrt (P.weight x) * f x + Real.sqrt (P.weight x) * g x
    ring
  unfold rms
  rw [he]
  exact norm_add_le _ _

lemma rms_const (P : FiniteLaw Ω) (c : ℝ) : rms P (fun _ => c) = |c| := by
  apply (sq_eq_sq₀ (rms_nonneg _ _) (abs_nonneg _)).mp
  rw [rms_sq, FiniteLaw.expectation_const, sq_abs]

lemma rms_mul_const (P : FiniteLaw Ω) (c : ℝ) (f : Ω → ℝ) :
    rms P (fun x => c * f x) = |c| * rms P f := by
  have he : weightedVector P (fun x => c * f x) = c • weightedVector P f := by
    ext x
    change Real.sqrt (P.weight x) * (c * f x) = c * (Real.sqrt (P.weight x) * f x)
    ring
  unfold rms
  rw [he, norm_smul, Real.norm_eq_abs]

lemma rms_mono (P : FiniteLaw Ω) (f g : Ω → ℝ)
    (hf : ∀ x, 0 ≤ f x) (hfg : ∀ x, f x ≤ g x) : rms P f ≤ rms P g := by
  apply (sq_le_sq₀ (rms_nonneg _ _) (rms_nonneg _ _)).mp
  rw [rms_sq, rms_sq]
  apply P.expectation_mono
  intro x
  exact (sq_le_sq₀ (hf x) ((hf x).trans (hfg x))).mpr (hfg x)

lemma rms_eq_sqrt_moment (P : FiniteLaw Ω) (f : Ω → ℝ)
    (a : ℝ) (ha : P.expectation (fun x => (f x)^2) = a) : rms P f = Real.sqrt a := by
  rw [← ha, ← rms_sq, Real.sqrt_sq_eq_abs, abs_of_nonneg (rms_nonneg _ _)]

/-- Explicit finite-law Minkowski bound for a nonnegative majorant. -/
theorem rms_majorant_three (P : FiniteLaw Ω) (f g h : Ω → ℝ)
    (c k d : ℝ) (hc : 0 ≤ c) (hk : 0 ≤ k) (hd : 0 < d)
    (hf : ∀ x, 0 ≤ f x)
    (hmajor : ∀ x, f x ≤ (g x + c + k * h x) / d) :
    rms P f ≤ (rms P g + c + k * rms P h) / d := by
  have htri : rms P (fun x => g x + c + k * h x) ≤
      rms P g + c + k * rms P h := by
    have h1 := rms_add_le P (fun x => g x + c) (fun x => k * h x)
    have h2 := rms_add_le P g (fun _ => c)
    rw [rms_mul_const, abs_of_nonneg hk] at h1
    rw [rms_const, abs_of_nonneg hc] at h2
    linarith
  calc
    rms P f ≤ rms P (fun x => (g x + c + k * h x) / d) := rms_mono P _ _ hf hmajor
    _ = d⁻¹ * rms P (fun x => g x + c + k * h x) := by
      simp_rw [div_eq_mul_inv, mul_comm _ d⁻¹]
      rw [rms_mul_const, abs_of_nonneg (inv_nonneg.mpr hd.le)]
    _ ≤ d⁻¹ * (rms P g + c + k * rms P h) :=
      mul_le_mul_of_nonneg_left htri (inv_nonneg.mpr hd.le)
    _ = _ := by ring

/-- Squared diagonal error on a stability event, with exactly the candidate
proof's d constant. Moment premises can be discharged by DesignMoments. -/
theorem stable_diagonal_moment_bound (P : FiniteLaw Ω)
    (A B : Ω → Matrix (Fin p) (Fin p) ℝ) (S : Ω → Prop)
    (a v α γ : ℝ) (ha0 : 0 ≤ a) (ha : a < 1) (hv : v ≤ 1)
    (hA : ∀ x, S x → ‖A x - 1‖ ≤ a)
    (hα : P.expectation (fun x => frobeniusSq (A x - 1)) = α)
    (hγ : P.expectation (fun x => frobeniusSq (B x - v • 1)) = γ) :
    P.expectation (fun x => if S x then
      frobeniusSq ((A x)⁻¹ * (B x - (A x - 1) * (A x - 1)) * (A x)⁻¹ - 1)
      else 0) ≤
      (Real.sqrt γ + (1 - v) * Real.sqrt p + 2 * (1 + a) * Real.sqrt α)^2 /
        (1 - a)^4 := by
  let f (x : Ω) := if S x then
    frob ((A x)⁻¹ * (B x - (A x - 1) * (A x - 1)) * (A x)⁻¹ - 1) else 0
  have hc : 0 ≤ (1 - v) * Real.sqrt p := mul_nonneg (sub_nonneg.mpr hv) (Real.sqrt_nonneg _)
  have hk : 0 ≤ 2 * (1 + a) := by positivity
  have hd : 0 < (1 - a)^2 := sq_pos_of_pos (sub_pos.mpr ha)
  have hf (x : Ω) : 0 ≤ f x := by
    dsimp [f]
    split_ifs <;> [exact frob_nonneg _; exact le_rfl]
  have hm (x : Ω) : f x ≤
      (frob (B x - v • 1) + (1 - v) * Real.sqrt p + 2 * (1 + a) * frob (A x - 1)) /
        (1 - a)^2 := by
    dsimp [f]
    split_ifs with hx
    · exact stable_centered_diagonal_error (A x) (B x) a v ha (hA x hx) hv
    · exact div_nonneg (add_nonneg (add_nonneg (frob_nonneg _) hc)
        (mul_nonneg hk (frob_nonneg _))) hd.le
  have hr := rms_majorant_three P f (fun x => frob (B x - v • 1))
    (fun x => frob (A x - 1)) ((1 - v) * Real.sqrt p)
    (2 * (1 + a)) ((1 - a)^2) hc hk hd hf hm
  rw [rms_eq_sqrt_moment P (fun x => frob (B x - v • 1)) γ
      (by simpa only [frob_sq] using hγ),
    rms_eq_sqrt_moment P (fun x => frob (A x - 1)) α
      (by simpa only [frob_sq] using hα)] at hr
  have hr0 : 0 ≤ (Real.sqrt γ + (1 - v) * Real.sqrt p +
      2 * (1 + a) * Real.sqrt α) / (1 - a)^2 := by positivity
  have hs := (sq_le_sq₀ (rms_nonneg P f) hr0).mpr hr
  rw [rms_sq] at hs
  simpa only [f, ite_pow, frob_sq, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
    div_pow, ← pow_mul] using hs

/-- The product-matrix second moment follows from actual independent
finite-law weights and Frobenius submultiplicativity. -/
theorem independent_product_frobenius_moment (P : FiniteLaw Ω)
    (E : Ω → Matrix (Fin p) (Fin p) ℝ) :
    (P.product P).expectation (fun x => frobeniusSq (E x.1 * E x.2)) ≤
      (P.expectation (fun x => frobeniusSq (E x)))^2 := by
  calc
    _ ≤ (P.product P).expectation (fun x => frobeniusSq (E x.1) * frobeniusSq (E x.2)) := by
      apply FiniteLaw.expectation_mono
      intro x
      have hs := (sq_le_sq₀ (frob_nonneg _) (mul_nonneg (frob_nonneg _) (frob_nonneg _))).mpr
        (frob_mul_le_frob (E x.1) (E x.2))
      simpa only [mul_pow, frob_sq] using hs
    _ = _ := by
      simpa only [sq] using P.product_expectation_mul P
        (fun x => frobeniusSq (E x)) (fun x => frobeniusSq (E x))

/-- Squared cross error on the stable event, with e = sqrt(zeta)+alpha.
The product-moment bound is supplied by independent_product_frobenius_moment. -/
theorem stable_cross_moment_bound (P : FiniteLaw Ω)
    (A C D : Ω → Matrix (Fin p) (Fin p) ℝ) (S : Ω → Prop)
    (a α ζ : ℝ) (ha : a < 1) (hα0 : 0 ≤ α)
    (hA : ∀ x, S x → ‖A x - 1‖ ≤ a)
    (hC : ∀ x, S x → ‖C x - 1‖ ≤ a)
    (hD : P.expectation (fun x => frobeniusSq (D x)) = ζ)
    (hE : P.expectation (fun x => frobeniusSq ((A x - 1) * (C x - 1))) ≤ α^2) :
    P.expectation (fun x => if S x then
      frobeniusSq ((A x)⁻¹ * (D x - (A x - 1) * (C x - 1)) * (C x)⁻¹)
      else 0) ≤ (Real.sqrt ζ + α)^2 / (1 - a)^4 := by
  let f (x : Ω) := if S x then
    frob ((A x)⁻¹ * (D x - (A x - 1) * (C x - 1)) * (C x)⁻¹) else 0
  have hd : 0 < (1 - a)^2 := sq_pos_of_pos (sub_pos.mpr ha)
  have hf (x : Ω) : 0 ≤ f x := by
    dsimp [f]
    split_ifs <;> [exact frob_nonneg _; exact le_rfl]
  have hm (x : Ω) : f x ≤
      (frob (D x) + 0 + 1 * frob ((A x - 1) * (C x - 1))) / (1 - a)^2 := by
    simp only [add_zero, one_mul]
    dsimp [f]
    split_ifs with hx
    · exact stable_cross_error (A x) (C x) (D x) a ha (hA x hx) (hC x hx)
    · exact div_nonneg (add_nonneg (frob_nonneg _) (frob_nonneg _)) hd.le
  have hr := rms_majorant_three P f (fun x => frob (D x))
    (fun x => frob ((A x - 1) * (C x - 1))) 0 1 ((1 - a)^2)
    le_rfl zero_le_one hd hf hm
  rw [add_zero, one_mul, rms_eq_sqrt_moment P (fun x => frob (D x)) ζ
    (by simpa only [frob_sq] using hD)] at hr
  have hre : rms P (fun x => frob ((A x - 1) * (C x - 1))) ≤ α := by
    apply (sq_le_sq₀ (rms_nonneg _ _) hα0).mp
    rw [rms_sq]
    simpa only [frob_sq] using hE
  have hr' : rms P f ≤ (Real.sqrt ζ + α) / (1 - a)^2 :=
    hr.trans (div_le_div_of_nonneg_right (add_le_add_left hre _) hd.le)
  have hr0 : 0 ≤ (Real.sqrt ζ + α) / (1 - a)^2 :=
    div_nonneg (add_nonneg (Real.sqrt_nonneg _) hα0) hd.le
  have hs := (sq_le_sq₀ (rms_nonneg P f) hr0).mpr hr'
  rw [rms_sq] at hs
  simpa only [f, ite_pow, frob_sq, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
    div_pow, ← pow_mul] using hs

/-- The probability of either unstable Gram matrix is bounded from the
actual Frobenius moments, using the proved operator/Frobenius comparison. -/
theorem unstable_pair_mass_bound (P : FiniteLaw Ω)
    (A C : Ω → Matrix (Fin p) (Fin p) ℝ) (a α : ℝ) (ha : 0 < a)
    (hA : P.expectation (fun x => frobeniusSq (A x - 1)) = α)
    (hC : P.expectation (fun x => frobeniusSq (C x - 1)) = α) :
    P.mass (fun x => ¬ (‖A x - 1‖ ≤ a ∧ ‖C x - 1‖ ≤ a)) ≤ 2 * α / a^2 := by
  have hsingle (M : Matrix (Fin p) (Fin p) ℝ) (hm : ¬ ‖M‖ ≤ a) : a^2 ≤ frobeniusSq M := by
    have hlt : a < frob M := (lt_of_not_ge hm).trans_le (operator_norm_le_frob M)
    have hs := (sq_le_sq₀ ha.le (frob_nonneg M)).mpr hlt.le
    simpa only [frob_sq] using hs
  have hpoint (x : Ω) :
      a^2 * (if ¬ (‖A x - 1‖ ≤ a ∧ ‖C x - 1‖ ≤ a) then 1 else 0) ≤
        frobeniusSq (A x - 1) + frobeniusSq (C x - 1) := by
    by_cases hx : ‖A x - 1‖ ≤ a ∧ ‖C x - 1‖ ≤ a
    · simpa [hx] using
        (add_nonneg (frobeniusSq_nonneg (A x - 1)) (frobeniusSq_nonneg (C x - 1)))
    · simp only [hx, not_false_eq_true, ↓reduceIte, mul_one]
      by_cases hAx : ‖A x - 1‖ ≤ a
      · have hCx : ¬ ‖C x - 1‖ ≤ a := fun hCx => hx ⟨hAx, hCx⟩
        exact (hsingle _ hCx).trans (le_add_of_nonneg_left (frobeniusSq_nonneg _))
      · exact (hsingle _ hAx).trans (le_add_of_nonneg_right (frobeniusSq_nonneg _))
  have hm := P.expectation_mono hpoint
  rw [FiniteLaw.expectation_const_mul, FiniteLaw.expectation_add, hA, hC] at hm
  have hind : P.expectation (fun x =>
      if ¬ (‖A x - 1‖ ≤ a ∧ ‖C x - 1‖ ≤ a) then (1 : ℝ) else 0) =
      P.mass (fun x => ¬ (‖A x - 1‖ ≤ a ∧ ‖C x - 1‖ ≤ a)) := by
    unfold FiniteLaw.expectation FiniteLaw.mass
    apply Finset.sum_congr rfl
    intro x hx
    by_cases h : ‖A x - 1‖ ≤ a ∧ ‖C x - 1‖ ≤ a <;> simp [h]
  rw [hind] at hm
  apply (le_div_iff₀ (sq_pos_of_pos ha)).mpr
  nlinarith

/-- The complete sum-of-covariance-blocks stable second-moment bound.
For a symmetric block covariance this sum is its squared Frobenius error. -/
theorem stable_pair_blocks_moment_bound (P : FiniteLaw Ω)
    (A C B₁ B₂ D : Ω → Matrix (Fin p) (Fin p) ℝ) (S : Ω → Prop)
    (a v α γ ζ : ℝ) (ha0 : 0 ≤ a) (ha : a < 1) (hv : v ≤ 1)
    (hA : ∀ x, S x → ‖A x - 1‖ ≤ a)
    (hC : ∀ x, S x → ‖C x - 1‖ ≤ a)
    (hα₁ : P.expectation (fun x => frobeniusSq (A x - 1)) = α)
    (hα₂ : P.expectation (fun x => frobeniusSq (C x - 1)) = α)
    (hγ₁ : P.expectation (fun x => frobeniusSq (B₁ x - v • 1)) = γ)
    (hγ₂ : P.expectation (fun x => frobeniusSq (B₂ x - v • 1)) = γ)
    (hD : P.expectation (fun x => frobeniusSq (D x)) = ζ)
    (hE : P.expectation (fun x => frobeniusSq ((A x - 1) * (C x - 1))) ≤ α^2) :
    P.expectation (fun x => if S x then
      frobeniusSq ((A x)⁻¹ * (B₁ x - (A x - 1) * (A x - 1)) * (A x)⁻¹ - 1) +
      frobeniusSq ((C x)⁻¹ * (B₂ x - (C x - 1) * (C x - 1)) * (C x)⁻¹ - 1) +
      2 * frobeniusSq ((A x)⁻¹ * (D x - (A x - 1) * (C x - 1)) * (C x)⁻¹)
      else 0) ≤
      (2 * (Real.sqrt γ + (1 - v) * Real.sqrt p + 2 * (1 + a) * Real.sqrt α)^2 +
        2 * (Real.sqrt ζ + α)^2) / (1 - a)^4 := by
  have hα0 : 0 ≤ α := by
    rw [← hα₁]
    exact P.expectation_nonneg (fun x => frobeniusSq_nonneg _)
  have h1 := stable_diagonal_moment_bound P A B₁ S a v α γ ha0 ha hv hA hα₁ hγ₁
  have h2 := stable_diagonal_moment_bound P C B₂ S a v α γ ha0 ha hv hC hα₂ hγ₂
  have h3 := stable_cross_moment_bound P A C D S a α ζ ha hα0 hA hC hD hE
  have hex (x : Ω) (u v w : ℝ) :
      (if S x then u + v + 2 * w else 0) =
        (if S x then u else 0) + (if S x then v else 0) +
          2 * (if S x then w else 0) := by
    split_ifs <;> ring
  simp_rw [hex]
  rw [FiniteLaw.expectation_add, FiniteLaw.expectation_add, FiniteLaw.expectation_const_mul]
  calc
    _ ≤ (Real.sqrt γ + (1 - v) * Real.sqrt p + 2 * (1 + a) * Real.sqrt α)^2 / (1 - a)^4 +
        (Real.sqrt γ + (1 - v) * Real.sqrt p + 2 * (1 + a) * Real.sqrt α)^2 / (1 - a)^4 +
        2 * ((Real.sqrt ζ + α)^2 / (1 - a)^4) := by linarith
    _ = _ := by ring

end FiniteL2

end BootstrapCalibration.StableCovariance
