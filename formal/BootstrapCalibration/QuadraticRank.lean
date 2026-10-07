import BootstrapCalibration.RankProbability

/-! Rank of weighted quadratic designs, using explicit polynomial separation
at three distinct actions and explicit kernel vectors for smaller supports. -/

noncomputable section
open scoped BigOperators Matrix Classical
namespace BootstrapCalibration.QuadraticRank

variable {ι : Type} [Fintype ι] [DecidableEq ι]

def design (u : ι → ℝ) : Matrix ι (Fin 3) ℝ := fun i => ![1, u i, (u i)^2]

def weightedGram (P : Matrix ι (Fin 3) ℝ) (w : ι → ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Pᵀ * Matrix.diagonal w * P

def positiveSupport (w : ι → ℝ) : Finset ι := Finset.univ.filter (fun i => 0 < w i)

omit [Fintype ι] [DecidableEq ι] in
lemma design_mulVec (u : ι → ℝ) (x : Fin 3 → ℝ) (i : ι) :
    (design u *ᵥ x) i = x 0 + u i * x 1 + (u i)^2 * x 2 := by
  simp [design, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  ring

/-- Three distinct roots force a quadratic coefficient vector to vanish. -/
lemma quadratic_three_roots (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (x : Fin 3 → ℝ)
    (ha : x 0 + a*x 1 + a^2*x 2 = 0)
    (hb : x 0 + b*x 1 + b^2*x 2 = 0)
    (hc : x 0 + c*x 1 + c^2*x 2 = 0) : x = 0 := by
  have hab' : (b-a)*(x 1+(a+b)*x 2)=0 := by nlinarith
  have hac' : (c-a)*(x 1+(a+c)*x 2)=0 := by nlinarith
  have hab0 : b-a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  have hac0 : c-a ≠ 0 := sub_ne_zero.mpr (Ne.symm hac)
  have h1 := (mul_eq_zero.mp hab').resolve_left hab0
  have h2 := (mul_eq_zero.mp hac').resolve_left hac0
  have hx2 : x 2 = 0 := by
    have h : (b-c)*x 2 = 0 := by nlinarith
    exact (mul_eq_zero.mp h).resolve_left (sub_ne_zero.mpr hbc)
  have hx1 : x 1 = 0 := by rw [hx2] at h1; linarith
  have hx0 : x 0 = 0 := by rw [hx1, hx2] at ha; linarith
  ext i
  fin_cases i
  · exact hx0
  · exact hx1
  · exact hx2

lemma gram_energy (P : Matrix ι (Fin 3) ℝ) (w : ι → ℝ) (x : Fin 3 → ℝ) :
    dotProduct x (weightedGram P w *ᵥ x) = ∑ i, w i * ((P *ᵥ x) i)^2 := by
  unfold weightedGram
  rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
    Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]
  simp only [dotProduct, Matrix.mulVec_diagonal]
  apply Finset.sum_congr rfl
  intro i hi
  ring

lemma gram_kernel_implies_vanish (P : Matrix ι (Fin 3) ℝ) (w : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (x : Fin 3 → ℝ) (hx : weightedGram P w *ᵥ x = 0) :
    ∀ i, 0 < w i → (P *ᵥ x) i = 0 := by
  have he : (∑ i, w i * ((P *ᵥ x) i)^2) = 0 := by
    rw [← gram_energy, hx]
    simp
  have hnonneg : ∀ i ∈ Finset.univ, 0 ≤ w i * ((P *ᵥ x) i)^2 := by
    intro i hi
    exact mul_nonneg (hw i) (sq_nonneg _)
  have hall := (Finset.sum_eq_zero_iff_of_nonneg hnonneg).mp he
  intro i hi
  have hz := hall i (Finset.mem_univ i)
  exact sq_eq_zero_iff.mp ((mul_eq_zero.mp hz).resolve_left (ne_of_gt hi))

lemma gram_kernel_of_vanish (P : Matrix ι (Fin 3) ℝ) (w : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (x : Fin 3 → ℝ)
    (hx : ∀ i, 0 < w i → (P *ᵥ x) i = 0) : weightedGram P w *ᵥ x = 0 := by
  have hd : Matrix.diagonal w *ᵥ (P *ᵥ x) = 0 := by
    ext i
    rw [Matrix.mulVec_diagonal]
    by_cases hi : w i = 0
    · simp [hi]
    · rw [hx i (lt_of_le_of_ne (hw i) (Ne.symm hi))]
      simp
  simp only [weightedGram, ← Matrix.mulVec_mulVec, hd, Matrix.mulVec_zero]

/-- With nonnegative weights and distinct actions, three positive rows suffice
for an invertible weighted quadratic Gram matrix. -/
theorem weightedGram_isUnit_of_three (u : ι → ℝ) (hu : Function.Injective u)
    (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i) (hs : 3 ≤ (positiveSupport w).card) :
    IsUnit (weightedGram (design u) w) := by
  obtain ⟨i,j,k,hi,hj,hk,hij,hik,hjk⟩ := Finset.two_lt_card_iff.mp hs
  have hip : 0 < w i := (Finset.mem_filter.mp hi).2
  have hjp : 0 < w j := (Finset.mem_filter.mp hj).2
  have hkp : 0 < w k := (Finset.mem_filter.mp hk).2
  have hker (x : Fin 3 → ℝ) (hx : weightedGram (design u) w *ᵥ x = 0) : x = 0 := by
    have hv := gram_kernel_implies_vanish (design u) w hw x hx
    apply quadratic_three_roots (u i) (u j) (u k)
      (fun h => hij (hu h)) (fun h => hik (hu h)) (fun h => hjk (hu h)) x
    · simpa only [design_mulVec] using hv i hip
    · simpa only [design_mulVec] using hv j hjp
    · simpa only [design_mulVec] using hv k hkp
  apply Matrix.mulVec_injective_iff_isUnit.mp
  intro x y hxy
  have hz : weightedGram (design u) w *ᵥ (x-y) = 0 := by
    rw [Matrix.mulVec_sub, hxy, sub_self]
  exact sub_eq_zero.mp (hker (x-y) hz)

lemma not_isUnit_of_kernel (M : Matrix (Fin 3) (Fin 3) ℝ) (x : Fin 3 → ℝ)
    (hx : x ≠ 0) (hz : M *ᵥ x = 0) : ¬ IsUnit M := by
  intro hM
  apply hx
  apply Matrix.mulVec_injective_iff_isUnit.mpr hM
  simpa only [Matrix.mulVec_zero] using hz

/-- Fewer than three positive rows leave a nonzero polynomial in the Gram kernel. -/
theorem weightedGram_not_isUnit_of_small_support (u : ι → ℝ) (w : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hs : (positiveSupport w).card ≤ 2) :
    ¬ IsUnit (weightedGram (design u) w) := by
  by_cases h0 : (positiveSupport w).card = 0
  · have hempty : positiveSupport w = ∅ := Finset.card_eq_zero.mp h0
    let x : Fin 3 → ℝ := ![1,0,0]
    have hx : x ≠ 0 := by intro h; have hh := congrFun h 0; change (1 : ℝ) = 0 at hh; norm_num at hh
    apply not_isUnit_of_kernel _ x hx
    apply gram_kernel_of_vanish _ w hw
    intro i hi
    have hm : i ∈ positiveSupport w := by simp [positiveSupport, hi]
    simp only [hempty, Finset.not_mem_empty] at hm
  · by_cases h1 : (positiveSupport w).card = 1
    · obtain ⟨a, ha⟩ := Finset.card_eq_one.mp h1
      let x : Fin 3 → ℝ := ![-u a,1,0]
      have hx : x ≠ 0 := by intro h; have hh := congrFun h 1; change (1 : ℝ) = 0 at hh; norm_num at hh
      apply not_isUnit_of_kernel _ x hx
      apply gram_kernel_of_vanish _ w hw
      intro i hi
      have hm : i ∈ positiveSupport w := by simp [positiveSupport, hi]
      have hia : i = a := by simpa only [ha, Finset.mem_singleton] using hm
      rw [hia, design_mulVec]
      simp [x]
    · have h2 : (positiveSupport w).card = 2 := by omega
      obtain ⟨a,b,hab,hS⟩ := Finset.card_eq_two.mp h2
      let x : Fin 3 → ℝ := ![u a*u b,-(u a+u b),1]
      have hx : x ≠ 0 := by intro h; have hh := congrFun h 2; change (1 : ℝ) = 0 at hh; norm_num at hh
      apply not_isUnit_of_kernel _ x hx
      apply gram_kernel_of_vanish _ w hw
      intro i hi
      have hm : i ∈ positiveSupport w := by simp [positiveSupport, hi]
      have hia : i = a ∨ i = b := by simpa only [hS, Finset.mem_insert, Finset.mem_singleton] using hm
      rcases hia with h | h
      · rw [h, design_mulVec]
        simp [x]
        ring
      · rw [h, design_mulVec]
        simp [x]
        ring

/-- Exact rank criterion for a quadratic design at distinct actions. -/
theorem weightedGram_isUnit_iff (u : ι → ℝ) (hu : Function.Injective u)
    (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i) :
    IsUnit (weightedGram (design u) w) ↔ 3 ≤ (positiveSupport w).card := by
  constructor
  · intro h
    by_contra hs
    exact weightedGram_not_isUnit_of_small_support u w hw (by omega) h
  · exact weightedGram_isUnit_of_three u hu w hw

lemma weightedGram_right_mul (P : Matrix ι (Fin 3) ℝ) (w : ι → ℝ)
    (C : Matrix (Fin 3) (Fin 3) ℝ) :
    weightedGram (P*C) w = Cᵀ * weightedGram P w * C := by
  simp only [weightedGram, Matrix.transpose_mul, Matrix.mul_assoc]

/-- Any invertible change of the three polynomial coordinates preserves the rank criterion. -/
theorem normalized_weightedGram_isUnit_iff (u : ι → ℝ) (hu : Function.Injective u)
    (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i)
    (C : Matrix (Fin 3) (Fin 3) ℝ) (hC : IsUnit C) :
    IsUnit (weightedGram (design u * C) w) ↔ 3 ≤ (positiveSupport w).card := by
  rw [weightedGram_right_mul]
  have hdetC : IsUnit C.det := (Matrix.isUnit_iff_isUnit_det C).mp hC
  rw [Matrix.isUnit_iff_isUnit_det, Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose,
    IsUnit.mul_iff, IsUnit.mul_iff]
  simp only [hdetC, true_and, and_true]
  rw [← Matrix.isUnit_iff_isUnit_det]
  exact weightedGram_isUnit_iff u hu w hw

omit [Fintype ι] in
lemma count_positive_iff_mem_support (m : ℕ) (sample : Draws ι m) (i : ι) :
    0 < countHits (fun a => a = i) m sample ↔ i ∈ drawSupport m sample := by
  induction m with
  | zero => simp [countHits, drawSupport]
  | succ m ih =>
    simp only [countHits, drawSupport, Finset.mem_insert]
    by_cases h : sample.1 = i
    · simp [h]
    · have h' : i ≠ sample.1 := Ne.symm h
      simpa only [h, h', ↓reduceIte, zero_add, false_or] using ih sample.2

/-- The strictly positive multinomial count coordinates are exactly the sampled rows. -/
lemma positiveSupport_counts (m : ℕ) (sample : Draws ι m) :
    positiveSupport (fun i => (countHits (fun a => a = i) m sample : ℝ)) = drawSupport m sample := by
  ext i
  simp only [positiveSupport, Finset.mem_filter, Finset.mem_univ, true_and, Nat.cast_pos]
  exact count_positive_iff_mem_support m sample i

/-- Exact acceptance event for normalized quadratic pairs refits. -/
theorem pairs_gram_isUnit_iff (u : ι → ℝ) (hu : Function.Injective u)
    (C : Matrix (Fin 3) (Fin 3) ℝ) (hC : IsUnit C)
    (m : ℕ) (sample : Draws ι m) :
    IsUnit (weightedGram (design u * C)
      (fun i => (countHits (fun a => a = i) m sample : ℝ))) ↔
      3 ≤ (drawSupport m sample).card := by
  rw [normalized_weightedGram_isUnit_iff u hu _ (by intro i; positivity) C hC,
    positiveSupport_counts]

/-- Exact singular-resample probability for distinct observed quadratic actions. -/
theorem pairs_gram_singular_probability (n : ℕ) (hn : 0 < n)
    (u : Fin n → ℝ) (hu : Function.Injective u)
    (C : Matrix (Fin 3) (Fin 3) ℝ) (hC : IsUnit C) :
    ((uniformRows n hn).iid n).mass (fun sample =>
      ¬ IsUnit (weightedGram (design u * C)
        (fun i => (countHits (fun a => a = i) n sample : ℝ)))) =
      ((n : ℝ) + (n.choose 2 : ℝ) * ((2 : ℝ)^n-2)) / (n : ℝ)^n := by
  have he : (fun sample : Draws (Fin n) n =>
      ¬ IsUnit (weightedGram (design u * C)
        (fun i => (countHits (fun a => a = i) n sample : ℝ)))) =
      (fun sample => (drawSupport n sample).card ≤ 2) := by
    funext sample
    apply propext
    rw [pairs_gram_isUnit_iff u hu C hC]
    omega
  rw [he]
  exact RankProbability.uniform_rows_support_le_two n hn

/-- The actual full-rank acceptance event has positive probability for n≥3. -/
theorem pairs_gram_acceptance_positive (n : ℕ) (hn : 0 < n) (hn3 : 3 ≤ n)
    (u : Fin n → ℝ) (hu : Function.Injective u)
    (C : Matrix (Fin 3) (Fin 3) ℝ) (hC : IsUnit C) :
    0 < ((uniformRows n hn).iid n).mass (fun sample =>
      IsUnit (weightedGram (design u * C)
        (fun i => (countHits (fun a => a = i) n sample : ℝ)))) := by
  have he : (fun sample : Draws (Fin n) n =>
      IsUnit (weightedGram (design u * C)
        (fun i => (countHits (fun a => a = i) n sample : ℝ)))) =
      (fun sample => 3 ≤ (drawSupport n sample).card) := by
    funext sample
    apply propext
    exact pairs_gram_isUnit_iff u hu C hC n sample
  rw [he]
  exact RankProbability.support_ge_three_positive n hn hn3

end BootstrapCalibration.QuadraticRank
