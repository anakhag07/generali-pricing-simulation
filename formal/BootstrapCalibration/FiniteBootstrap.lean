import Mathlib

noncomputable section
open scoped BigOperators Classical
namespace BootstrapCalibration

/-- A finite probability distribution, represented by its normalized atom weights. -/
structure FiniteLaw (α : Type) [Fintype α] where
  weight : α → ℝ
  nonneg : ∀ a, 0 ≤ weight a
  total : ∑ a, weight a = 1

namespace FiniteLaw
variable {α : Type} [Fintype α] (P : FiniteLaw α)

def mass (E : α → Prop) : ℝ := ∑ a, if E a then P.weight a else 0

lemma mass_nonneg (E : α → Prop) : 0 ≤ P.mass E := by
  classical
  apply Finset.sum_nonneg
  intro a ha
  split_ifs
  · exact P.nonneg a
  · exact le_rfl

lemma mass_mono {E F : α → Prop} (h : ∀ a, E a → F a) : P.mass E ≤ P.mass F := by
  classical
  apply Finset.sum_le_sum
  intro a ha
  by_cases hE : E a
  · simp [hE, h a hE]
  · simp only [hE, ↓reduceIte]
    split_ifs
    · exact P.nonneg a
    · exact le_rfl

@[simp] lemma mass_true : P.mass (fun _ => True) = 1 := by
  classical
  simpa [mass] using P.total

@[simp] lemma mass_false : P.mass (fun _ => False) = 0 := by
  classical
  simp [mass]

lemma mass_le_one (E : α → Prop) : P.mass E ≤ 1 := by
  simpa using P.mass_mono (E := E) (F := fun _ => True) (by simp)

lemma mass_compl (E : α → Prop) : P.mass (fun a => ¬ E a) = 1 - P.mass E := by
  classical
  have h : P.mass (fun a => ¬ E a) + P.mass E = 1 := by
    rw [mass, mass, ← Finset.sum_add_distrib, ← P.total]
    apply Finset.sum_congr rfl
    intro a ha
    by_cases h : E a <;> simp [h]
  linarith

/-- The same normalized weights as a mathlib probability mass function. -/
def toPMF : PMF α := PMF.ofFintype (fun a => ENNReal.ofReal (P.weight a)) (by
  rw [← ENNReal.ofReal_sum_of_nonneg (fun a _ => P.nonneg a), P.total]
  simp)

/-- Finite sums used below agree with mathlib's probability assigned to an event. -/
lemma toPMF_event (E : α → Prop) :
    P.toPMF.toOuterMeasure {a | E a} = ENNReal.ofReal (P.mass E) := by
  rw [PMF.toOuterMeasure_apply, tsum_fintype]
  unfold mass
  rw [ENNReal.ofReal_sum_of_nonneg (by intro a ha; split_ifs; exact P.nonneg a; exact le_rfl)]
  apply Finset.sum_congr rfl
  intro a ha
  by_cases h : E a <;> simp [Set.indicator_apply, h, toPMF]

def cdf (T : α → ℝ) (t : ℝ) : ℝ := P.mass (fun a => T a ≤ t)

lemma cdf_mono (T : α → ℝ) : Monotone (P.cdf T) := by
  intro x y hxy
  exact P.mass_mono (fun a ha => ha.trans hxy)

/-- The right-continuous CDF evaluated at an atom is super-uniform, including ties. -/
lemma cdf_bad_mass_le (T : α → ℝ) {p : ℝ} (hp : 0 ≤ p) :
    P.mass (fun a => P.cdf T (T a) < p) ≤ p := by
  classical
  let S := Finset.univ.filter (fun a => P.cdf T (T a) < p)
  by_cases hS : S.Nonempty
  · obtain ⟨a, ha, hmax⟩ := S.exists_max_image T hS
    have ha' : P.cdf T (T a) < p := (Finset.mem_filter.mp ha).2
    have hsub : P.mass (fun b => P.cdf T (T b) < p) ≤ P.cdf T (T a) := by
      apply P.mass_mono
      intro b hb
      exact hmax b (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hb⟩)
    exact hsub.trans ha'.le
  · have hnone : ∀ a, ¬ P.cdf T (T a) < p := by
      intro a ha
      exact hS ⟨a, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ha⟩⟩
    simpa [mass, hnone] using hp
end FiniteLaw

/-- The exact Bernoulli upper tail, evaluated by the standard binomial recursion. -/
def binomialTail (p : ℝ) : ℕ → ℕ → ℝ
  | 0, k => if k = 0 then 1 else 0
  | _+1, 0 => 1
  | n+1, k+1 => p * binomialTail p n k + (1-p) * binomialTail p n (k+1)

@[simp] lemma binomialTail_zero (p : ℝ) (n : ℕ) : binomialTail p n 0 = 1 := by
  cases n <;> simp [binomialTail]

lemma binomialTail_nonneg {p : ℝ} (hp : 0 ≤ p) (hp1 : p ≤ 1) (n k : ℕ) :
    0 ≤ binomialTail p n k := by
  induction n generalizing k with
  | zero => simp [binomialTail]; split_ifs <;> positivity
  | succ n ih =>
    cases k with
    | zero => simp
    | succ k =>
      simp only [binomialTail]
      exact add_nonneg (mul_nonneg hp (ih k)) (mul_nonneg (by linarith) (ih (k+1)))

lemma binomialTail_le_one {p : ℝ} (hp : 0 ≤ p) (hp1 : p ≤ 1) (n k : ℕ) :
    binomialTail p n k ≤ 1 := by
  induction n generalizing k with
  | zero => simp [binomialTail]; split_ifs <;> norm_num
  | succ n ih =>
    cases k with
    | zero => simp
    | succ k =>
      calc
        binomialTail p (n+1) (k+1) = p * binomialTail p n k + (1-p) * binomialTail p n (k+1) := rfl
        _ ≤ p * 1 + (1-p) * 1 := add_le_add (mul_le_mul_of_nonneg_left (ih k) hp)
          (mul_le_mul_of_nonneg_left (ih (k+1)) (by linarith))
        _ = 1 := by ring

lemma binomialTail_antitone_rank {p : ℝ} (hp : 0 ≤ p) (hp1 : p ≤ 1) (n k : ℕ) :
    binomialTail p n (k+1) ≤ binomialTail p n k := by
  induction n generalizing k with
  | zero => cases k <;> simp [binomialTail]
  | succ n ih =>
    cases k with
    | zero => simpa using binomialTail_le_one hp hp1 (n+1) 1
    | succ k =>
      exact add_le_add (mul_le_mul_of_nonneg_left (ih k) hp)
        (mul_le_mul_of_nonneg_left (ih (k+1)) (by linarith))

lemma binomialTail_mono {p q : ℝ} (hp : 0 ≤ p) (hpq : p ≤ q) (hq : q ≤ 1)
    (n k : ℕ) : binomialTail p n k ≤ binomialTail q n k := by
  induction n generalizing k with
  | zero => rfl
  | succ n ih =>
    cases k with
    | zero => simp
    | succ k =>
      calc
        binomialTail p (n+1) (k+1) ≤ p * binomialTail q n k + (1-p) * binomialTail q n (k+1) :=
          add_le_add (mul_le_mul_of_nonneg_left (ih k) hp)
            (mul_le_mul_of_nonneg_left (ih (k+1)) (by linarith))
        _ ≤ binomialTail q (n+1) (k+1) := by
          have hm := binomialTail_antitone_rank (hp.trans hpq) hq n k
          dsimp only [binomialTail]
          nlinarith

/-- Ordered iid draws, using nested products to expose the product distribution. -/
def Draws (α : Type) : ℕ → Type _
  | 0 => Unit
  | n+1 => α × Draws α n

instance drawsFintype (α : Type) [Fintype α] : (n : ℕ) → Fintype (Draws α n)
  | 0 => inferInstanceAs (Fintype Unit)
  | n+1 => @instFintypeProd α (Draws α n) _ (drawsFintype α n)

namespace FiniteLaw
variable {α : Type} [Fintype α] (P : FiniteLaw α)

def iid : (n : ℕ) → FiniteLaw (Draws α n)
  | 0 => ⟨fun _ => 1, by simp, by simp [Draws]⟩
  | n+1 => {
      weight := fun x => P.weight x.1 * (iid n).weight x.2
      nonneg := fun x => mul_nonneg (P.nonneg x.1) ((iid n).nonneg x.2)
      total := by
        change (∑ x : α × Draws α n, P.weight x.1 * (iid n).weight x.2) = 1
        rw [Fintype.sum_prod_type]
        simp_rw [← Finset.mul_sum, (iid n).total, mul_one]
        exact P.total }

end FiniteLaw

def countHits {α : Type} (E : α → Prop) : (n : ℕ) → Draws α n → ℕ
  | 0, _ => 0
  | n+1, x => (if E x.1 then 1 else 0) + countHits E n x.2

namespace FiniteLaw
variable {α : Type} [Fintype α] (P : FiniteLaw α)

/-- The count of hits in explicit iid samples has the exact binomial upper tail. -/
lemma iid_count_tail (E : α → Prop) (n k : ℕ) :
    (P.iid n).mass (fun x => k ≤ countHits E n x) = binomialTail (P.mass E) n k := by
  induction n generalizing k with
  | zero =>
    cases k <;> simp [iid, mass, countHits, binomialTail, Draws]
  | succ n ih =>
    cases k with
    | zero => simp
    | succ k =>
      have hs : (P.iid (n+1)).mass (fun x => k+1 ≤ countHits E (n+1) x) =
          P.mass E * (P.iid n).mass (fun x => k ≤ countHits E n x) +
          P.mass (fun a => ¬ E a) * (P.iid n).mass (fun x => k+1 ≤ countHits E n x) := by
        simp only [mass, iid, countHits]
        change (∑ x : α × Draws α n, if k+1 ≤ (if E x.1 then 1 else 0) + countHits E n x.2
          then P.weight x.1 * (P.iid n).weight x.2 else 0) = _
        rw [Fintype.sum_prod_type]
        rw [Finset.sum_mul, Finset.sum_mul, ← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro a ha
        by_cases h : E a
        · simp only [h, ↓reduceIte, Nat.add_comm 1, Nat.add_le_add_iff_right, not_true_eq_false,
            zero_mul, add_zero]
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro x hx
          split_ifs <;> simp
        · simp only [h, ↓reduceIte, zero_add, not_false_eq_true, zero_mul, zero_add]
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro x hx
          split_ifs <;> simp
      rw [hs, ih, ih, mass_compl]
      rfl

end FiniteLaw
lemma countHits_mono {α : Type} {E F : α → Prop} (h : ∀ a, E a → F a)
    (n : ℕ) (x : Draws α n) : countHits E n x ≤ countHits F n x := by
  induction n with
  | zero => simp [countHits]
  | succ n ih =>
    have ht := ih x.2
    dsimp only [countHits]
    by_cases hE : E x.1
    · simp only [hE, h x.1 hE, ↓reduceIte]
      omega
    · simp only [hE, ↓reduceIte, zero_add]
      split_ifs <;> omega

namespace FiniteLaw
variable {α : Type} [Fintype α] (P : FiniteLaw α)

/-- Exact finite-B quantile guarantee for any finite bootstrap law, allowing ties.
The rank hypothesis also covers certified upper bounds on each bootstrap score. -/
theorem quantile_failure_le (T : α → ℝ) {p : ℝ} (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (B k : ℕ) (Q : Draws α B → ℝ)
    (hQ : ∀ x, k ≤ countHits (fun a => T a ≤ Q x) B x) :
    (P.iid B).mass (fun x => P.cdf T (Q x) < p) ≤ binomialTail p B k := by
  let bad := fun a => P.cdf T (T a) < p
  calc
    (P.iid B).mass (fun x => P.cdf T (Q x) < p) ≤
        (P.iid B).mass (fun x => k ≤ countHits bad B x) := by
      apply mass_mono
      intro x hx
      apply (hQ x).trans
      apply countHits_mono
      intro a ha
      exact lt_of_le_of_lt (P.cdf_mono T ha) hx
    _ = binomialTail (P.mass bad) B k := P.iid_count_tail bad B k
    _ ≤ binomialTail p B k := binomialTail_mono (P.mass_nonneg bad)
      (P.cdf_bad_mass_le T hp) hp1 B k

/-- A rank with binomial failure at most eta certifies bootstrap mass at least p. -/
theorem quantile_mass_confidence (T : α → ℝ) {p eta : ℝ} (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (B k : ℕ) (Q : Draws α B → ℝ)
    (hQ : ∀ x, k ≤ countHits (fun a => T a ≤ Q x) B x)
    (hrank : binomialTail p B k ≤ eta) :
    1-eta ≤ (P.iid B).mass (fun x => p ≤ P.cdf T (Q x)) := by
  have hfail := (P.quantile_failure_le T hp hp1 B k Q hQ).trans hrank
  have hcompl := (P.iid B).mass_compl (fun x => P.cdf T (Q x) < p)
  simp only [not_lt] at hcompl
  linarith

end FiniteLaw
end BootstrapCalibration
