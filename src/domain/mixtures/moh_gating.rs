use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, Zip, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{
    common::rng::get_rng,
    domain::{
        mixtures::{
            moh::{HeadSelectionConfig, HeadSelectionStrategy},
            routing::{RoutingConfig, SelectionAlgorithm, apply_selection_algorithm},
            threshold::ThresholdPredictor,
        },
        richards::{RichardsCurve, RichardsGate},
    },
    infrastructure::optimizer::adam::Adam,
};

#[derive(Debug, Clone, Default)]
pub struct MoHStreamingWorkspace {
    pub xw: Array1<f32>,
    pub g: Array1<f32>,
    pub m: Array1<f32>,
}

fn enforce_min_max_heads_inplace(
    g_mat: &ArrayView2<f32>,
    m_mat: &mut ArrayViewMut2<f32>,
    min_heads: usize,
    max_heads: usize,
    always_on_heads: &[usize],
    renormalize_to_k: Option<usize>,
) {
    let n = g_mat.nrows();
    let h_total = g_mat.ncols();
    if n == 0 || h_total == 0 {
        return;
    }
    if m_mat.dim() != g_mat.dim() {
        return;
    }

    // Sanitize always-on head indices once.
    let mut always: Vec<usize> = Vec::new();
    for &h in always_on_heads {
        if h < h_total && !always.contains(&h) {
            always.push(h);
        }
    }

    let mut min_h = min_heads.min(h_total);
    if always.len() > min_h {
        min_h = always.len();
    }

    let mut max_h = max_heads.min(h_total);
    max_h = max_h.max(min_h.max(1));

    // If misconfigured (always_on > max), truncate always-on to max.
    if always.len() > max_h {
        always.truncate(max_h);
        min_h = min_h.min(max_h);
    }

    // Parallel iteration over tokens (rows)
    Zip::from(g_mat.axis_iter(Axis(0)))
        .and(m_mat.axis_iter_mut(Axis(0)))
        .par_for_each(|g_row, mut m_row| {
            // Collect all scores with indices
            // Capacity optimization: h_total is typically small
            let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(h_total);
            for h in 0..h_total {
                let v = g_row[h];
                let score = if v.is_finite() { v } else { f32::NEG_INFINITY };
                candidates.push((score, h));
            }

            // Sort descending by score
            candidates.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Determine which heads to keep
            // 1. Must include all 'always' heads
            // 2. Fill remaining spots with top scoring heads up to max_h
            // 3. Ensure at least min_h heads are active (if possible)

            // Start by selecting top max_h candidates

            // First pass: add top candidates until we hit max_h, but prioritize always-on later?
            // Actually, simply taking top max_h is the base strategy.
            // But we MUST have always-on heads.

            // Let's identify the set of heads to be active.
            // Strategy:
            // - Start with always-on heads.
            // - Add top-scoring heads that are not in always-on, until we reach max_h.
            // - But wait, if an always-on head has very low score, does it displace a high-scoring head? Yes, "always on".

            let mut count = 0;
            // Mark which heads are selected
            let mut keep_mask = vec![false; h_total];

            // 1. Force always-on
            for &h in &always {
                keep_mask[h] = true;
                count += 1;
            }

            // 2. Fill up to max_h with best scoring heads
            if count < max_h {
                for &(_score, h) in &candidates {
                    if !keep_mask[h] {
                        keep_mask[h] = true;
                        count += 1;
                        if count >= max_h {
                            break;
                        }
                    }
                }
            }

            // 3. If we still have fewer than min_h (e.g. because max_h < min_h which shouldn't happen due to checks,
            //    OR because we didn't find enough candidates? No, we iterated all h_total).
            //    Wait, logic above: max_h >= min_h. And we iterate all heads. So count should be max_h (or h_total).
            //    The only case count < min_h is if h_total < min_h, but we clamped min_h.
            //    So we are good on min_h implicitly if we fill up to max_h.
            //    BUT, what if max_h > min_h, but we only want to keep heads with high scores?
            //    The original code had logic: "keep only top max_h heads by g_mat".
            //    But it also had: "also ensure at least min_h are on".
            //    The previous logic was: select top max_h. Then ensure always-on. Then ensure min_h. Then clamp to max_h.

            //    Let's stick to the previous logic's intent but cleaner:
            //    Goal: Active set S.
            //    Constraint 1: always \subset S
            //    Constraint 2: |S| <= max_h
            //    Constraint 3: |S| >= min_h
            //    Preference: Maximize sum(scores in S)

            //    Algorithm:
            //    a. Start with S = always.
            //    b. If |S| > max_h, remove lowest scoring from S until |S| == max_h (But always-on are forced?
            //       Original code: "Strictly enforce the max-heads cap even after forcing always-on heads... drop the lowest-score non-always heads")
            //       My pre-check ensures always.len() <= max_h. So this won't happen.

            //    c. Add highest scoring non-S heads until |S| == min_h.
            //    d. If we allow more heads (up to max_h), should we add them?
            //       Original code: "For each token: keep only top max_h heads... ensure at least min_h".
            //       It implies we WANT top max_h, but we might be forced to drop some if always-on takes precedence?
            //       Actually, original code:
            //       1. Pick top max_h.
            //       2. Add always-on (might exceed max_h).
            //       3. Force min_h (from best).
            //       4. If active > max_h, drop lowest non-always.

            //    So effectively: Take union of (Top max_h) and (Always).
            //    Then if size > max_h, remove worst non-always.
            //    Also ensure size >= min_h (which is covered if we start with Top max_h and max_h >= min_h).

            //    Revised Clean Algorithm:
            //    1. Take all candidates.
            //    2. Separate into "Always" and "Others".
            //    3. "Always" are automatically in.
            //    4. "Others" are sorted by score.
            //    5. We have budget: remaining_slots = max_h - always.len().
            //    6. Take top `remaining_slots` from "Others".
            //    7. This gives us a set of size `max_h`.
            //    8. But wait, what if the original logic *filtered* based on mask `m_mat` inputs?
            //       Original: "Zero out everything not in best".
            //       But `m_mat` contained values from predictor/SoftTopP.
            //       The function `enforce_min_max_heads_inplace` is a filter on `m_mat`.
            //       It uses `g_mat` for scoring, but it modifies `m_mat`.
            //       Crucially: "If m_mat[[i, h]] > 0.0 ... active += 1".
            //       The original code respects the *existing* active heads in `m_mat` if they are within limits?
            //       No, line 123: "Zero out everything not in best". `best` comes from `g_mat`.
            //       So it overrides `m_mat`'s selection with `g_mat`'s top-k?
            //       Wait. If `use_learned_predictor` (line 458), `m_mat` is set to `t` (predictor output).
            //       Then `enforce...` is called.
            //       Inside `enforce...`:
            //         Line 67: "Pick top max_h heads by g_mat".
            //         Line 123: "Zero out everything not in best".
            //       This means the predictor's output is MASKED by the top-k of the *gate* scores `g_mat`?
            //       This seems to defeat the purpose of the predictor if `g_mat` (Richards gate) decides everything.
            //       BUT `g_mat` is derived from `gate.update_scaling_from_max_abs`.
            //       If `m_mat` (predictor) says "activate head 5", but head 5 is not in top-k of `g_mat`, it gets zeroed?
            //       YES. This is a "Hard Top-K" enforcement on top of whatever the predictor says.
            //       UNLESS `max_heads` is very large (== num_heads), in which case it does nothing.

            //       So, `m_mat` preserves its *values* (weights), but entries are zeroed if they are not in the allowed set.
            //       The allowed set is determined by `g_mat` scores + always_on.

            //       So my "Revised Clean Algorithm" is correct for determining the *mask*.
            //       Then we apply this mask to `m_mat`.

            // Reset mask
            keep_mask.fill(false);

            // 1. Mark always-on
            for &h in &always {
                keep_mask[h] = true;
            }

            // 2. Select others from candidates (which are sorted by score)
            let mut slots_left = max_h.saturating_sub(always.len());
            for &(_score, h) in &candidates {
                if slots_left == 0 {
                    break;
                }
                if !keep_mask[h] {
                    keep_mask[h] = true;
                    slots_left -= 1;
                }
            }

            // 3. Apply mask to m_row
            for h in 0..h_total {
                if !keep_mask[h] {
                    m_row[h] = 0.0;
                } else {
                    // If it was already 0.0, should we force it to 1.0?
                    // Original line 138: "Force always-on heads to be active... m_mat[[i, ah]] = 1.0;"
                    // Original line 149: "m_mat[[i, h]] = 1.0;" (for min_h enforcement)
                    // So YES, if selected, ensure it's at least 1.0?
                    // Wait, if predictor output was 0.5, and it's selected, should it stay 0.5 or become 1.0?
                    // Line 138 forces always-on to 1.0.
                    // Line 149 forces min_h additions to 1.0.
                    // But what about the top-k that were *already* in m_mat?
                    // Original line 132: "m_mat[[i, h]] = 0.0" (if not kept).
                    // It does NOT say "m_mat[[i, h]] = 1.0" for the kept ones generally.
                    // ONLY for always-on and forced min_h.

                    // So:
                    // - If always-on: set to 1.0 (override predictor).
                    // - If kept because of top-k: keep predictor value?
                    //   The original code didn't change m_mat values for the `best` list, only zeroed others.
                    //   EXCEPT for the explicit "Force always-on" loop and "Ensure min_h" loop.

                    // So my logic:
                    // If always-on: m_row[h] = 1.0.
                    // Else if kept: leave as is?
                    // BUT wait, what if `m_row[h]` was 0.0 (predictor said no), but it IS in top-k of `g_mat`?
                    // Original code would keep it in `best`, so it wouldn't be zeroed.
                    // But if `m_mat` was 0.0, it stays 0.0.
                    // UNLESS min_h logic forces it to 1.0.

                    if always.contains(&h) {
                        m_row[h] = 1.0;
                    }
                    // For others, we only zero if NOT in mask.
                    // But we might need to force to 1.0 if we fall below min_h?
                    // My selection logic above selects exactly `max_h` heads (if available).
                    // Since `max_h >= min_h`, we satisfy min_h count.
                    // But do we need to set their values to 1.0?
                    // Original code:
                    // "Ensure at least min_h heads are on... m_mat[[i, h]] = 1.0".
                    // This loop runs for `need` count.
                    // `need` starts at `min_h - always.len()`.
                    // It iterates `best` (sorted).
                    // So the top `min_h` heads (including always-on) get forced to 1.0?
                    // YES.

                    // So:
                    // 1. Top `min_h` heads (by g_mat score) -> Force to 1.0 (if not always-on, which is also 1.0).
                    // 2. Heads between `min_h` and `max_h` (by g_mat score) -> Keep original `m_mat` value (don't zero, don't force).
                    // 3. Heads below `max_h` -> Zero out.

                    // Let's refine the loop logic.
                }
            }

            // Re-apply logic strictly:
            // 1. Identify Top `max_h` heads from candidates.
            //    Note: always-on heads might NOT be in Top `max_h` of scores.
            //    But we MUST keep always-on.
            //    Original logic:
            //      a. `best` = Top `max_h` (pure score).
            //      b. Add `always` to `best` if not present (growing `best` beyond `max_h`).
            //      c. Zero out non-`best`.
            //      d. Force `always` to 1.0.
            //      e. Force top `min_h` from `best` to 1.0.
            //      f. If active (`m_mat > 0`) > `max_h`: Drop lowest score non-always.

            //    This is complex "active" definition.
            //    "Active" means `m_mat > 0`.
            //    If predictor output `m_mat` has zeros for everything, then step (f) sees active=0 (or just always/min_h).

            //    Let's replicate the effect exactly:
            //    Set S = Top `max_h` (by score).
            //    S = S U always.
            //    For h not in S: m_row[h] = 0.0.
            //    For h in always: m_row[h] = 1.0.
            //
            //    For h in Top `min_h` (by score) AND in S: m_row[h] = 1.0. (Wait, Top min_h is subset of Top max_h, so in S).
            //    So: Top `min_h` -> 1.0.

            //    Finally, check active count.
            //    Active = { h | m_row[h] > 0.0 }.
            //    If |Active| > max_h:
            //       Remove h from Active with lowest score (non-always) until |Active| == max_h.
            //       Set m_row[h] = 0.0.

            //    Let's implement this per row.

            // 1. Identify Top max_h and Top min_h
            // candidates is already sorted descending.
            let top_max_indices: Vec<usize> = candidates.iter().take(max_h).map(|x| x.1).collect();
            let top_min_indices: Vec<usize> = candidates.iter().take(min_h).map(|x| x.1).collect();

            // 2. Zero out if not in top_max AND not always
            //    (Original step 123)
            for h in 0..h_total {
                let is_top_max = top_max_indices.contains(&h);
                let is_always = always.contains(&h);
                if !is_top_max && !is_always {
                    m_row[h] = 0.0;
                }
            }

            // 3. Force always to 1.0
            for &h in &always {
                m_row[h] = 1.0;
            }

            // 4. Force top min_h to 1.0
            for &h in &top_min_indices {
                // Original: if always.contains(&h) continue; m_mat=1.0.
                // Since always is already 1.0, we can just set it.
                m_row[h] = 1.0;
            }

            // 5. Enforce max_h cap on Active heads
            let mut active_heads: Vec<(f32, usize)> = Vec::new();
            for h in 0..h_total {
                if m_row[h] > 0.0 {
                    // We need the score again.
                    // We can lookup in candidates or just store it.
                    // Linear scan of candidates is fine for small h.
                    // Or better, build a lookup or just iterate candidates?
                    // Since we iterate h, let's just find score.
                    let score = candidates
                        .iter()
                        .find(|&&x| x.1 == h)
                        .map(|x| x.0)
                        .unwrap_or(f32::NEG_INFINITY);
                    active_heads.push((score, h));
                }
            }

            if active_heads.len() > max_h {
                // Sort ascending by score to drop lowest
                active_heads
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let mut to_drop = active_heads.len() - max_h;
                for (_s, h) in active_heads {
                    if to_drop == 0 {
                        break;
                    }
                    if !always.contains(&h) {
                        m_row[h] = 0.0;
                        to_drop -= 1;
                    }
                }
            }

            // Renormalize if needed
            if let Some(k_target) = renormalize_to_k {
                let k_val = (k_target.max(1).min(h_total)) as f32;
                let mut sum = 0.0f32;
                for h in 0..h_total {
                    let v = m_row[h];
                    if v.is_finite() {
                        sum += v.max(0.0);
                    }
                }
                let eps = 1e-6f32;
                if sum > eps && sum.is_finite() {
                    let s = k_val / sum;
                    for h in 0..h_total {
                        let v = m_row[h];
                        let v = if v.is_finite() { v.max(0.0) } else { 0.0 };
                        m_row[h] = v * s;
                    }
                }
            }
        });
}

/// Shared Mixture-of-Heads (MoH) gating module.
///
/// This owns the gating parameters and metrics used to produce per-token per-head
/// activation weights. It is intended to be reusable across attention and SSM mixers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoHGating {
    /// Per-head gating projection: X·W_g
    pub w_g: Array2<f32>, // (embed_dim, num_heads)
    pub alpha_g: Array2<f32>, // (1, num_heads)
    pub beta_g: Array2<f32>,  // (1, num_heads)

    pub opt_w_g: Adam,
    pub opt_alpha_g: Adam,
    pub opt_beta_g: Adam,

    /// Learnable Richards gate used to map z -> g in (0,1)
    pub gate: RichardsGate,

    /// Learnable Richards curve for low-rank query gating
    pub low_rank_query_gate: RichardsCurve,

    /// Head selection configuration and metrics
    pub head_selection_config: HeadSelectionConfig,

    /// Optional learned threshold predictor (AutoDeco-inspired)
    pub threshold_predictor: Option<ThresholdPredictor>,

    pub opt_w_tau: Option<Adam>,
    pub opt_b_tau: Option<Adam>,
    pub opt_w2_tau: Option<Adam>,
    pub opt_b2_tau: Option<Adam>,
    pub opt_cond_w_tau: Option<Adam>,

    /// Cached SoftTopP mask (tokens x heads) from last forward pass.
    #[serde(skip_serializing, skip_deserializing)]
    pub cached_soft_top_p_mask: Option<Array2<f32>>,

    /// Cached max_abs_z per head from last forward pass (for verification parity)
    #[serde(skip_serializing, skip_deserializing)]
    pub last_max_abs_z: Option<Vec<f64>>,

    /// Optional overrides for max_abs_z (for verification parity)
    #[serde(skip_serializing, skip_deserializing)]
    pub verification_overrides: Option<Vec<f64>>,

    /// Training progress (0.0 to 1.0) for adaptive hyperparameters
    #[serde(skip_serializing, skip_deserializing)]
    pub training_progress: f64,
}

impl MoHGating {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let mut rng = get_rng();
        let std_g = (2.0f32 / embed_dim.max(1) as f32).sqrt();
        let normal_g = Normal::new(0.0, std_g as f64).unwrap();

        let w_g = Array2::<f32>::from_shape_fn((embed_dim, num_heads), |_| {
            normal_g.sample(&mut rng) as f32
        });
        let alpha_g = Array2::<f32>::ones((1, num_heads));
        let beta_g = Array2::<f32>::zeros((1, num_heads));

        let mut opt_w_g = Adam::new((embed_dim, num_heads));
        let mut opt_alpha_g = Adam::new((1, num_heads));
        let mut opt_beta_g = Adam::new((1, num_heads));
        opt_w_g.set_amsgrad(true);
        opt_alpha_g.set_amsgrad(true);
        opt_beta_g.set_amsgrad(true);

        Self {
            w_g,
            alpha_g,
            beta_g,
            opt_w_g,
            opt_alpha_g,
            opt_beta_g,
            gate: RichardsGate::new(),
            low_rank_query_gate: RichardsCurve::sigmoid(true),
            head_selection_config: HeadSelectionConfig::default(),
            threshold_predictor: None,
            opt_w_tau: None,
            opt_b_tau: None,
            opt_w2_tau: None,
            opt_b2_tau: None,
            opt_cond_w_tau: None,
            cached_soft_top_p_mask: None,
            last_max_abs_z: None,
            verification_overrides: None,
            training_progress: 0.0,
        }
    }

    /// Configure the gating strategy (and initialize predictor/optimizers if required).
    pub fn set_head_selection_config(&mut self, strategy: &HeadSelectionStrategy) {
        let num_heads = self.w_g.ncols();
        let embed_dim = self.w_g.nrows();
        self.head_selection_config = HeadSelectionConfig::from_strategy(strategy, num_heads);

        if self.head_selection_config.gating.use_learned_predictor
            && self.threshold_predictor.is_none()
        {
            let predictor_hidden_dim = 128.min(embed_dim / 2).max(32);
            self.threshold_predictor = Some(ThresholdPredictor::new_with_cond(
                embed_dim,
                predictor_hidden_dim,
                num_heads,
                embed_dim,
            ));

            self.opt_w_tau = Some(Adam::new((embed_dim, predictor_hidden_dim)));
            self.opt_b_tau = Some(Adam::new((predictor_hidden_dim, 1)));
            self.opt_w2_tau = Some(Adam::new((predictor_hidden_dim, num_heads)));
            self.opt_b2_tau = Some(Adam::new((num_heads, 1)));
            self.opt_cond_w_tau = Some(Adam::new((embed_dim, predictor_hidden_dim)));
        }
    }

    /// Set heads that should always remain active.
    ///
    /// This is applied on top of the configured selection strategy.
    pub fn set_always_on_heads(&mut self, heads: Vec<usize>) {
        self.head_selection_config.always_on_heads = heads;
    }

    /// Set verification overrides for max_abs_z (for parity testing)
    pub fn set_verification_overrides(&mut self, overrides: Option<Vec<f64>>) {
        self.verification_overrides = overrides;
    }

    /// Compute a token-level activity scalar from per-head weights.
    ///
    /// The scalar is the mean positive head weight in [0, 1].
    #[inline]
    pub fn token_activity_scalar_from_iter<I>(weights: I) -> f32
    where
        I: IntoIterator<Item = f32>,
    {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for w in weights {
            sum += w.max(0.0);
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            (sum / count as f32).clamp(0.0, 1.0)
        }
    }

    /// Build shared streaming MoH metrics from current per-head weights.
    ///
    /// Returns:
    /// 1. active-head count (used by avg-active-heads tracking)
    /// 2. per-head activity vector
    /// 3. per-token activity vector (single-token streaming => len 1)
    #[inline]
    pub fn summarize_streaming_weights(weights: &Array1<f32>) -> (f32, Vec<f32>, Vec<f32>) {
        let active_heads = weights.iter().filter(|&&w| w > 0.0).count() as f32;
        let head_vec = weights.to_vec();
        let token_scalar = Self::token_activity_scalar_from_iter(weights.iter().copied());
        (active_heads, head_vec, vec![token_scalar])
    }

    /// Returns the zero-copy gate input prefix expected by `w_g` for a single token.
    #[inline]
    pub fn gate_input_view<'a>(&self, input: &'a ArrayView1<f32>) -> ArrayView1<'a, f32> {
        let gd = self.w_g.nrows().min(input.len());
        input.slice(s![0..gd])
    }

    /// Returns the zero-copy gate input prefix expected by `w_g` for a token batch.
    #[inline]
    pub fn gate_input_view2<'a>(&self, input: &'a ArrayView2<f32>) -> ArrayView2<'a, f32> {
        let gd = self.w_g.nrows().min(input.ncols());
        input.slice(s![.., 0..gd])
    }

    /// Compute per-token per-head weights (tokens x heads) and update MoH metrics.
    ///
    /// Returns weights in [0,1] (not necessarily summing to 1).
    pub fn forward_weights(
        &mut self,
        input: &Array2<f32>,
        token_threshold_scale: Option<&Array2<f32>>,
        token_latent_features: Option<&Array2<f32>>,
    ) -> Array2<f32> {
        self.forward_weights_view(
            &input.view(),
            token_threshold_scale.map(|x| x.view()),
            token_latent_features.map(|x| x.view()),
        )
    }

    pub fn forward_weights_view(
        &mut self,
        input: &ArrayView2<f32>,
        token_threshold_scale: Option<ArrayView2<f32>>,
        token_latent_features: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let n = input.nrows();
        let num_heads = self.w_g.ncols();
        if n == 0 || num_heads == 0 {
            return Array2::<f32>::zeros((n, num_heads));
        }

        self.cached_soft_top_p_mask = None;

        // Compute X·W_g once: shape (n, num_heads)
        let xw = input.dot(&self.w_g);

        // Compute raw gate values g (tokens x heads) using Richards gate.
        let mut g_mat = Array2::<f32>::zeros((n, num_heads));

        // Helper arrays for outputs
        let mut head_sq_sums = Array1::<f32>::zeros(num_heads);
        let mut head_max_abs_z = Array1::<f64>::zeros(num_heads);
        let indices = Array1::from_iter(0..num_heads);

        // Parallel execution over heads
        Zip::from(g_mat.axis_iter_mut(Axis(1)))
            .and(xw.axis_iter(Axis(1)))
            .and(&indices)
            .and(&mut head_sq_sums)
            .and(&mut head_max_abs_z)
            .par_for_each(|mut g_col, xw_col, &h, sq_sum_out, max_z_out| {
                let a_h = self.alpha_g[[0, h]];
                let b_h = self.beta_g[[0, h]];
                let mut sq_sum = 0.0f32;
                let mut max_z = 0.0f64;

                // 1. Compute stats
                for i in 0..n {
                    let v = xw_col[i];
                    sq_sum += v * v;
                    let z = a_h * v + b_h;
                    max_z = max_z.max((z as f64).abs());
                }

                // 2. Override if needed
                if let Some(overrides) = &self.verification_overrides {
                    if h < overrides.len() {
                        max_z = overrides[h];
                    }
                }

                *sq_sum_out = sq_sum;
                *max_z_out = max_z;

                // 3. Compute Gate
                // Direct application without dynamic scaling for streaming parity
                for i in 0..n {
                    let z = a_h * xw_col[i] + b_h;
                    g_col[i] = self.gate.curve.forward_scalar_f32(z);
                }
            });

        let g_sq_sum: f32 = head_sq_sums.sum();
        self.last_max_abs_z = Some(head_max_abs_z.to_vec());

        // Compute head selection mask m (tokens x heads).
        let mut m_mat = Array2::<f32>::ones((n, num_heads));
        if self.head_selection_config.gating.use_learned_predictor {
            if let Some(predictor) = &mut self.threshold_predictor {
                let mut cond_input = input.to_owned();
                if let Some(scale) = token_threshold_scale {
                    let d = cond_input.ncols();
                    for i in 0..n {
                        let s0 = scale[[i, 0]];
                        for j in 0..d {
                            cond_input[[i, j]] *= s0;
                        }
                    }
                }
                let mut t =
                    predictor.predict_with_condition(&cond_input.view(), token_latent_features);

                let m = self
                    .head_selection_config
                    .threshold_modulation
                    .value(self.training_progress);
                t.mapv_inplace(|v| {
                    let v = if v.is_finite() { v } else { 0.0 };
                    (v * m).max(0.0)
                });

                // Normalize each row to sum=k (like the attention implementation).
                // Epsilon guard prevents huge amplification when the predictor collapses.
                let k = self.head_selection_config.gating.num_active.max(1) as f32;
                let eps = 1e-6f32;
                let uniform = k / num_heads.max(1) as f32;
                for i in 0..n {
                    let mut sum = 0.0f32;
                    for h in 0..num_heads {
                        sum += t[[i, h]];
                    }
                    if sum > eps && sum.is_finite() {
                        let s = k / sum;
                        for h in 0..num_heads {
                            t[[i, h]] *= s;
                        }
                    } else {
                        for h in 0..num_heads {
                            t[[i, h]] = uniform;
                        }
                    }
                }

                m_mat.assign(&t);
            }

            // Enforce min/max heads consistently (and keep sum=k semantics for predictor output).
            enforce_min_max_heads_inplace(
                &g_mat.view(),
                &mut m_mat.view_mut(),
                self.head_selection_config.min_heads,
                self.head_selection_config.max_heads,
                &self.head_selection_config.always_on_heads,
                Some(self.head_selection_config.gating.num_active),
            );

            // Update tau metrics based on mask.
            self.head_selection_config.metrics_tau_count += n;
            for v in m_mat.iter() {
                let vv = if v.is_finite() { *v } else { 0.0 };
                if vv < self.head_selection_config.metrics_tau_min {
                    self.head_selection_config.metrics_tau_min = vv;
                }
                if vv > self.head_selection_config.metrics_tau_max {
                    self.head_selection_config.metrics_tau_max = vv;
                }
                self.head_selection_config.metrics_tau_sum += vv;
            }
        } else if self.head_selection_config.gating.use_soft_top_p {
            // Use shared routing SoftTopP on g_mat.
            let cfg = RoutingConfig {
                algorithm: SelectionAlgorithm::SoftTopP {
                    top_p: self.head_selection_config.gating.top_p,
                },
                use_learned_predictor: false,
                num_active: self.head_selection_config.gating.num_active.max(1),
                temperature: 1.0,
                soft_top_p_alpha: self.head_selection_config.gating.soft_top_p_alpha,
            };
            let mut weights = apply_selection_algorithm(&g_mat.view(), &cfg);

            // Scale and clamp to mimic "active heads" semantics.
            let activation_scale = self.head_selection_config.max_heads.max(1) as f32;
            weights.mapv_inplace(|v| (v * activation_scale).clamp(0.0, 1.0));

            let m = self
                .head_selection_config
                .threshold_modulation
                .value(self.training_progress);
            weights.mapv_inplace(|v| (v * m).clamp(0.0, 1.0));

            if let Some(scale) = token_threshold_scale {
                for i in 0..n {
                    let s0 = scale[[i, 0]];
                    for h in 0..num_heads {
                        weights[[i, h]] = (weights[[i, h]] * s0).clamp(0.0, 1.0);
                    }
                }
            }

            self.cached_soft_top_p_mask = Some(weights.clone());
            m_mat.assign(&weights);

            // Enforce min/max heads (SoftTopP doesn't require sum=k semantics).
            enforce_min_max_heads_inplace(
                &g_mat.view(),
                &mut m_mat.view_mut(),
                self.head_selection_config.min_heads,
                self.head_selection_config.max_heads,
                &self.head_selection_config.always_on_heads,
                None,
            );

            // Update tau metrics based on mask.
            self.head_selection_config.metrics_tau_count += n;
            for v in m_mat.iter() {
                let vv = if v.is_finite() { *v } else { 0.0 };
                if vv < self.head_selection_config.metrics_tau_min {
                    self.head_selection_config.metrics_tau_min = vv;
                }
                if vv > self.head_selection_config.metrics_tau_max {
                    self.head_selection_config.metrics_tau_max = vv;
                }
                self.head_selection_config.metrics_tau_sum += vv;
            }
        }

        // Fixed strategy (and any other non-predictor, non-SoftTopP path): enforce min/max.
        if !self.head_selection_config.gating.use_learned_predictor
            && !self.head_selection_config.gating.use_soft_top_p
        {
            enforce_min_max_heads_inplace(
                &g_mat.view(),
                &mut m_mat.view_mut(),
                self.head_selection_config.min_heads,
                self.head_selection_config.max_heads,
                &self.head_selection_config.always_on_heads,
                None,
            );
        }

        // Effective weights.
        let mut eff = &g_mat * &m_mat;
        eff.mapv_inplace(|v| if v.is_finite() { v.max(0.0) } else { 0.0 });

        // Update gating metrics.
        self.head_selection_config.metrics_g_sq_sum += g_sq_sum;
        self.head_selection_config.metrics_g_count += n * num_heads;
        self.head_selection_config.update_metrics(&eff.view());

        eff
    }

    /// Single-step forward for streaming.
    ///
    /// Wraps `forward_weights_view` with 1-element batch logic.
    pub fn forward_weights_step(
        &mut self,
        input: &ndarray::ArrayView1<f32>,
        token_threshold_scale: Option<f32>,
        token_latent_features: Option<&ndarray::ArrayView1<f32>>,
    ) -> ndarray::Array1<f32> {
        let input_2d = input.view().insert_axis(ndarray::Axis(0));
        let scale_2d = token_threshold_scale.map(|s| ndarray::Array2::from_elem((1, 1), s));
        let features_2d = token_latent_features.map(|f| f.view().insert_axis(ndarray::Axis(0)));

        let out_2d =
            self.forward_weights_view(&input_2d, scale_2d.as_ref().map(|x| x.view()), features_2d);

        out_2d.row(0).to_owned()
    }

    pub fn moh_num_active(&self) -> usize {
        self.head_selection_config.gating.num_active
    }

    pub fn compute_moh_aux_losses(&self, target_avg_components: f32) -> (f32, f32, f32) {
        let lb = self.head_selection_config.compute_load_balance_loss();
        let cx = self
            .head_selection_config
            .compute_complexity_loss(target_avg_components);
        let sp = self.head_selection_config.compute_sparsity_loss();
        (lb, cx, sp)
    }

    pub fn compute_moh_aux_weighted_total(&self, target_avg_components: f32) -> f32 {
        let (lb, cx, sp) = self.compute_moh_aux_losses(target_avg_components);
        let g = &self.head_selection_config.gating;
        let imp = g.compute_importance_loss();
        let sw = g.compute_switch_balance_loss();
        (lb * g.load_balance_weight)
            + (cx * g.complexity_loss_weight)
            + (sp * g.sparsity_weight)
            + (imp * g.importance_loss_weight)
            + (sw * g.switch_balance_weight)
    }

    pub fn peek_tau_metrics(&self) -> Option<(f32, f32)> {
        if self.head_selection_config.metrics_tau_count > 0 {
            Some((
                self.head_selection_config.metrics_tau_min,
                self.head_selection_config.metrics_tau_max,
            ))
        } else {
            None
        }
    }

    pub fn take_tau_metrics(&mut self) -> Option<(f32, f32)> {
        if self.head_selection_config.metrics_tau_count > 0 {
            let min = self.head_selection_config.metrics_tau_min;
            let max = self.head_selection_config.metrics_tau_max;
            self.head_selection_config.metrics_tau_min = f32::INFINITY;
            self.head_selection_config.metrics_tau_max = f32::NEG_INFINITY;
            self.head_selection_config.metrics_tau_sum = 0.0;
            self.head_selection_config.metrics_tau_count = 0;
            Some((min, max))
        } else {
            None
        }
    }

    pub fn take_pred_norm(&mut self) -> Option<f32> {
        if self.head_selection_config.metrics_g_count > 0 {
            let rms = (self.head_selection_config.metrics_g_sq_sum
                / self.head_selection_config.metrics_g_count as f32)
                .sqrt();
            self.head_selection_config.metrics_g_sq_sum = 0.0;
            self.head_selection_config.metrics_g_count = 0;
            Some(rms)
        } else {
            None
        }
    }

    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        let num_heads = self.w_g.ncols();
        let mut res = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let tokens = self
                .head_selection_config
                .gating
                .metrics
                .token_count_per_component[h];
            let avg = if tokens > 0 {
                self.head_selection_config
                    .gating
                    .metrics
                    .active_sum_per_component[h]
                    / tokens as f32
            } else {
                0.0
            };
            res.push((avg, tokens));
            self.head_selection_config
                .gating
                .metrics
                .active_sum_per_component[h] = 0.0;
            self.head_selection_config
                .gating
                .metrics
                .token_count_per_component[h] = 0;
        }
        res
    }

    /// Compute gradients for MoH gating parameters given upstream gradients w.r.t. effective
    /// weights.
    ///
    /// Returns (grad_input, grad_params) where grad_params matches the ordering:
    /// w_g, alpha_g, beta_g, gate_poly, (optional predictor grads: w1,b1,w2,b2,cond_w,activation)
    pub fn compute_gradients_from_eff(
        &mut self,
        input: &Array2<f32>,
        eff_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.compute_gradients_from_eff_view(&input.view(), eff_grads)
    }

    pub fn compute_gradients_from_eff_view(
        &mut self,
        input: &ArrayView2<f32>,
        eff_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let n = input.nrows();
        let embed_dim = self.w_g.nrows();
        let num_heads = self.w_g.ncols();
        let mut grad_input = Array2::<f32>::zeros(input.raw_dim());

        let mut grad_w_g = Array2::<f32>::zeros((embed_dim, num_heads));
        let mut grad_alpha_g = Array2::<f32>::zeros((1, num_heads));
        let mut grad_beta_g = Array2::<f32>::zeros((1, num_heads));

        let n_gate_w = self.gate.parameters();
        let mut grad_gate_poly_vec = vec![0.0_f64; n_gate_w];

        // Compute X·W_g once.
        let xw = input.dot(&self.w_g);

        // Recompute raw gate values g_mat (needed for learned-predictor gradients) and
        // compute m_mat consistently with forward.
        let mut g_mat = Array2::<f32>::zeros((n, num_heads));
        for h in 0..num_heads {
            let a_h = self.alpha_g[[0, h]];
            let b_h = self.beta_g[[0, h]];

            // Ensure RichardsGate scaling matches the forward path.
            // Direct application without dynamic scaling for streaming parity
            for i in 0..n {
                let z = a_h * xw[[i, h]] + b_h;
                g_mat[[i, h]] = self.gate.curve.forward_scalar_f32(z);
            }
        }

        // Mask matrix m_mat for backward.
        let mut m_mat = Array2::<f32>::ones((n, num_heads));

        // For learned predictor: recompute predictor output and apply the same per-row
        // normalization. For SoftTopP: recompute the SoftTopP weights from g_mat (more
        // reliable than relying on cache).
        let mut pred_output: Option<Array2<f32>> = None;
        let mut pred_pre_norm: Option<Array2<f32>> = None;
        if self.head_selection_config.gating.use_learned_predictor {
            if let Some(pred) = &mut self.threshold_predictor {
                let mut p = pred.predict_with_condition(&input.view(), None);
                let mod_f = self
                    .head_selection_config
                    .threshold_modulation
                    .value(self.training_progress);
                p.mapv_inplace(|v| {
                    let v = if v.is_finite() { v } else { 0.0 };
                    (v * mod_f).max(0.0)
                });

                // Save pre-normalized output for correct normalization backward.
                pred_pre_norm = Some(p.clone());

                // Normalize each row to sum=k.
                let k = self.head_selection_config.gating.num_active.max(1) as f32;
                let eps = 1e-6f32;
                let uniform = k / num_heads.max(1) as f32;
                for i in 0..n {
                    let mut sum = 0.0f32;
                    for h in 0..num_heads {
                        sum += p[[i, h]];
                    }
                    if sum > eps && sum.is_finite() {
                        let s = k / sum;
                        for h in 0..num_heads {
                            p[[i, h]] *= s;
                        }
                    } else {
                        for h in 0..num_heads {
                            p[[i, h]] = uniform;
                        }
                    }
                }

                pred_output = Some(p.clone());
                m_mat.assign(&p);
            }

            enforce_min_max_heads_inplace(
                &g_mat.view(),
                &mut m_mat.view_mut(),
                self.head_selection_config.min_heads,
                self.head_selection_config.max_heads,
                &self.head_selection_config.always_on_heads,
                Some(self.head_selection_config.gating.num_active),
            );
        } else if self.head_selection_config.gating.use_soft_top_p {
            let cfg = RoutingConfig {
                algorithm: SelectionAlgorithm::SoftTopP {
                    top_p: self.head_selection_config.gating.top_p,
                },
                use_learned_predictor: false,
                num_active: self.head_selection_config.gating.num_active.max(1),
                temperature: 1.0,
                soft_top_p_alpha: self.head_selection_config.gating.soft_top_p_alpha,
            };
            let mut weights = apply_selection_algorithm(&g_mat.view(), &cfg);
            let activation_scale = self.head_selection_config.max_heads.max(1) as f32;
            weights.mapv_inplace(|v| (v * activation_scale).clamp(0.0, 1.0));
            let m = self
                .head_selection_config
                .threshold_modulation
                .value(self.training_progress);
            weights.mapv_inplace(|v| (v * m).clamp(0.0, 1.0));
            m_mat.assign(&weights);

            enforce_min_max_heads_inplace(
                &g_mat.view(),
                &mut m_mat.view_mut(),
                self.head_selection_config.min_heads,
                self.head_selection_config.max_heads,
                &self.head_selection_config.always_on_heads,
                None,
            );
        }

        if !self.head_selection_config.gating.use_learned_predictor
            && !self.head_selection_config.gating.use_soft_top_p
        {
            enforce_min_max_heads_inplace(
                &g_mat.view(),
                &mut m_mat.view_mut(),
                self.head_selection_config.min_heads,
                self.head_selection_config.max_heads,
                &self.head_selection_config.always_on_heads,
                None,
            );
        }

        for h in 0..num_heads {
            let w_g_col = self.w_g.slice(s![.., h..h + 1]);
            let a_h = self.alpha_g[[0, h]];
            let b_h = self.beta_g[[0, h]];

            for i in 0..n {
                let xw_ih = xw[[i, h]];
                let z = a_h * xw_ih + b_h;
                let m = m_mat[[i, h]];

                let d_eff = eff_grads[[i, h]];
                let d_eff = if d_eff.is_finite() { d_eff } else { 0.0 };
                let d_g = d_eff * m;

                let dphi_dz = self.gate.backward_scalar_f32(z);
                let grad_z = d_g * dphi_dz;

                // Richards curve parameter grads (uses upstream d_g).
                let gws = self.gate.grad_weights_scalar_f32(z, d_g);
                for (wi, gw) in gws.iter().enumerate() {
                    grad_gate_poly_vec[wi] += *gw;
                }

                // W_g slice grad
                {
                    let mut gw_slice = grad_w_g.slice_mut(s![.., h..h + 1]);
                    for d in 0..embed_dim {
                        gw_slice[[d, 0]] += a_h * input[[i, d]] * grad_z;
                    }
                }
                grad_alpha_g[[0, h]] += grad_z * xw_ih;
                grad_beta_g[[0, h]] += grad_z;

                // Input grad contribution (g-path)
                for d in 0..embed_dim {
                    grad_input[[i, d]] += a_h * w_g_col[[d, 0]] * grad_z;
                }
            }
        }

        // Predictor grads (and predictor->input gradients)
        let mut extra: Vec<Array2<f32>> = Vec::new();
        if self.head_selection_config.gating.use_learned_predictor {
            if let (Some(pred), Some(_)) = (&self.threshold_predictor, pred_output.as_ref()) {
                // dL/dm from eff = g*m
                let mut d_m = Array2::<f32>::zeros((n, num_heads));
                for i in 0..n {
                    for h in 0..num_heads {
                        let d_eff = eff_grads[[i, h]];
                        let d_eff = if d_eff.is_finite() { d_eff } else { 0.0 };
                        let g = g_mat[[i, h]];
                        let g = if g.is_finite() { g } else { 0.0 };
                        d_m[[i, h]] = d_eff * g;
                    }
                }

                // Backprop through row-normalization: m = k * u / sum(u), where u is the
                // pre-normalized predictor output. Use the saved pre-normalized
                // values from this function's predictor forward.
                let u = pred_pre_norm
                    .clone()
                    .unwrap_or_else(|| Array2::<f32>::zeros((n, num_heads)));

                let k = self.head_selection_config.gating.num_active.max(1) as f32;
                let mut d_u = Array2::<f32>::zeros((n, num_heads));
                for i in 0..n {
                    let mut sum_u = 0.0f32;
                    for h in 0..num_heads {
                        sum_u += u[[i, h]].max(0.0);
                    }
                    // Match the forward epsilon guard: if the normalization is effectively
                    // uniform/degenerate, treat it as a stop-gradient path.
                    let eps = 1e-6f32;
                    if sum_u <= eps || !sum_u.is_finite() {
                        continue;
                    }
                    let c = k / sum_u;
                    let mut dot = 0.0f32;
                    for h in 0..num_heads {
                        dot += d_m[[i, h]] * u[[i, h]].max(0.0);
                    }
                    let common = -(k * dot) / (sum_u * sum_u);
                    for h in 0..num_heads {
                        if u[[i, h]] > 0.0 {
                            d_u[[i, h]] = c * d_m[[i, h]] + common;
                        }
                    }
                }

                // u = modulation * predictor_output (modulation is a scalar).
                // Therefore dL/d(predictor_output) = modulation * dL/du.
                let mod_f = self
                    .head_selection_config
                    .threshold_modulation
                    .value(self.training_progress);
                let mut d_p = d_u;
                d_p.mapv_inplace(|v| v * mod_f);

                // Important: use the predictor instance with cached activations.
                let (dx_pred, gw1, gb1_1d, gw2, gb2_1d, gcond, gact) = {
                    let pred_mut = self
                        .threshold_predictor
                        .as_ref()
                        .expect("predictor must exist");
                    pred_mut.compute_gradients_with_input(&d_p)
                };

                // Predictor->input gradient
                grad_input += &dx_pred;

                let gb1 = gb1_1d
                    .clone()
                    .to_shape((gb1_1d.len(), 1))
                    .unwrap()
                    .to_owned();
                let gb2 = gb2_1d
                    .clone()
                    .to_shape((gb2_1d.len(), 1))
                    .unwrap()
                    .to_owned();
                extra.push(gw1);
                extra.push(gb1);
                extra.push(gw2);
                extra.push(gb2);
                if let Some(gcond) = gcond {
                    extra.push(gcond);
                } else {
                    extra.push(Array2::<f32>::zeros((embed_dim, pred.weights1.ncols())));
                }
                // Pack activation params into a 2D array like PolyAttention does.
                let act_arr = Array2::<f32>::from_shape_vec(
                    (gact.len(), 1),
                    gact.iter().map(|&x| x as f32).collect(),
                )
                .unwrap();
                extra.push(act_arr);
            } else if let Some(pred) = &self.threshold_predictor {
                // Keep shape compatibility even if forward cache is missing.
                let hidden_dim = pred.weights1.ncols();
                let act_len = pred.activation.scalar_weights_len();
                extra.push(Array2::<f32>::zeros((embed_dim, hidden_dim))); // w1
                extra.push(Array2::<f32>::zeros((hidden_dim, 1))); // b1
                extra.push(Array2::<f32>::zeros((hidden_dim, num_heads))); // w2
                extra.push(Array2::<f32>::zeros((num_heads, 1))); // b2
                extra.push(Array2::<f32>::zeros((embed_dim, hidden_dim))); // cond_w
                extra.push(Array2::<f32>::zeros((act_len, 1))); // activation
            } else {
                // No predictor available; fall back to minimal shapes.
                extra.push(Array2::<f32>::zeros((embed_dim, 1)));
                extra.push(Array2::<f32>::zeros((1, 1)));
                extra.push(Array2::<f32>::zeros((1, num_heads)));
                extra.push(Array2::<f32>::zeros((num_heads, 1)));
                extra.push(Array2::<f32>::zeros((embed_dim, 1)));
                extra.push(Array2::<f32>::zeros((1, 1)));
            }
        }

        let grad_gate_poly = Array2::<f32>::from_shape_vec(
            (grad_gate_poly_vec.len(), 1),
            grad_gate_poly_vec.into_iter().map(|x| x as f32).collect(),
        )
        .unwrap();

        let mut grads = vec![grad_w_g, grad_alpha_g, grad_beta_g, grad_gate_poly];
        grads.extend(extra);

        (grad_input, grads)
    }

    pub fn forward_weights_into(
        &self,
        input: &ArrayView1<f32>,
        workspace: &mut MoHStreamingWorkspace,
    ) {
        let num_heads = self.w_g.ncols();

        // Ensure workspace buffers are correctly sized for zero-allocation
        // streaming callers that may reuse default-initialized workspaces.
        if workspace.xw.len() != num_heads {
            workspace.xw = Array1::zeros(num_heads);
        }
        if workspace.g.len() != num_heads {
            workspace.g = Array1::zeros(num_heads);
        }
        if workspace.m.len() != num_heads {
            workspace.m = Array1::zeros(num_heads);
        }

        // 1. Projection: xw = input * w_g
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_g.t(), input, 0.0, &mut workspace.xw);

        // 2. Gate activation (Richards)
        let alpha = self.alpha_g.row(0);
        let beta = self.beta_g.row(0);

        for h in 0..num_heads {
            let z = alpha[h] * workspace.xw[h] + beta[h];

            // Direct application without dynamic scaling for streaming parity
            workspace.g[h] = self.gate.curve.forward_scalar_f32(z);
        }

        // 3. Selection
        // Use 2D view for compatibility with existing routing logic
        let g_view = workspace.g.view();
        let g_view_2d = g_view.to_shape((1, num_heads)).unwrap();

        if self.head_selection_config.gating.use_soft_top_p {
            let cfg = RoutingConfig {
                algorithm: SelectionAlgorithm::SoftTopP {
                    top_p: self.head_selection_config.gating.top_p,
                },
                use_learned_predictor: false,
                num_active: self.head_selection_config.gating.num_active.max(1),
                temperature: 1.0,
                soft_top_p_alpha: self.head_selection_config.gating.soft_top_p_alpha,
            };
            let mut weights_2d = apply_selection_algorithm(&g_view_2d.view(), &cfg);

            let activation_scale = self.head_selection_config.max_heads.max(1) as f32;
            weights_2d.mapv_inplace(|v| (v * activation_scale).clamp(0.0, 1.0));

            let m_val = self
                .head_selection_config
                .threshold_modulation
                .value(self.training_progress);
            weights_2d.mapv_inplace(|v| (v * m_val).clamp(0.0, 1.0));

            workspace.m.assign(&weights_2d.row(0));
        } else if self.head_selection_config.gating.use_learned_predictor {
            // Streaming predictor fallback: keep allocation-free Top-K approximation.
            let cfg = RoutingConfig {
                algorithm: SelectionAlgorithm::TopK {
                    k: self.head_selection_config.gating.num_active.max(1),
                },
                use_learned_predictor: false,
                num_active: self.head_selection_config.gating.num_active.max(1),
                temperature: 1.0,
                soft_top_p_alpha: self.head_selection_config.gating.soft_top_p_alpha,
            };
            let weights_2d = apply_selection_algorithm(&g_view_2d.view(), &cfg);
            workspace.m.assign(&weights_2d.row(0));
        } else {
            // Fixed/non-predictor path matches `forward_weights_view` semantics:
            // start with all ones then enforce min/max against gate scores.
            workspace.m.fill(1.0);
        }

        // 4. Enforce Min/Max Heads
        let m_view = workspace.m.view_mut();
        let mut m_view_2d = m_view.to_shape((1, num_heads)).unwrap();
        enforce_min_max_heads_inplace(
            &g_view_2d.view(),
            &mut m_view_2d.view_mut(),
            self.head_selection_config.min_heads,
            self.head_selection_config.max_heads,
            &self.head_selection_config.always_on_heads,
            None,
        );

        // 5. Convert selection mask to effective weights (match batch path: eff = g * m).
        for h in 0..num_heads {
            let v = workspace.g[h] * workspace.m[h];
            workspace.m[h] = if v.is_finite() { v.max(0.0) } else { 0.0 };
        }
    }

    pub fn apply_gradients(
        &mut self,
        grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::common::errors::Result<()> {
        // grads ordering described in compute_gradients_from_eff.
        if grads.len() < 4 {
            return Err(crate::common::errors::ModelError::GradientError {
                message: format!(
                    "MoHGating expected at least 4 grad arrays, got {}",
                    grads.len()
                ),
            });
        }
        let mut idx = 0usize;
        self.opt_w_g.step(&mut self.w_g, &grads[idx], lr);
        self.opt_alpha_g
            .step(&mut self.alpha_g, &grads[idx + 1], lr);
        self.opt_beta_g.step(&mut self.beta_g, &grads[idx + 2], lr);
        idx += 3;
        let grad_gate_poly = &grads[idx];
        let _ = self
            .gate
            .apply_gradients(std::slice::from_ref(grad_gate_poly), lr);
        idx += 1;

        if self.head_selection_config.gating.use_learned_predictor
            && let (Some(pred), Some(opt_w1), Some(opt_b1), Some(opt_w2), Some(opt_b2)) = (
                &mut self.threshold_predictor,
                &mut self.opt_w_tau,
                &mut self.opt_b_tau,
                &mut self.opt_w2_tau,
                &mut self.opt_b2_tau,
            )
        {
            if grads.len() < idx + 6 {
                return Err(crate::common::errors::ModelError::GradientError {
                    message: format!("MoHGating expected predictor grads, got {}", grads.len()),
                });
            }
            opt_w1.step(&mut pred.weights1, &grads[idx], lr);
            let mut bias1_reshaped = pred
                .bias1
                .clone()
                .to_shape((pred.bias1.len(), 1))
                .unwrap()
                .to_owned();
            opt_b1.step(&mut bias1_reshaped, &grads[idx + 1], lr);
            pred.bias1
                .assign(&bias1_reshaped.view().to_shape(pred.bias1.len()).unwrap());
            opt_w2.step(&mut pred.weights2, &grads[idx + 2], lr);
            let mut bias2_reshaped = pred
                .bias2
                .clone()
                .to_shape((pred.bias2.len(), 1))
                .unwrap()
                .to_owned();
            opt_b2.step(&mut bias2_reshaped, &grads[idx + 3], lr);
            pred.bias2
                .assign(&bias2_reshaped.view().to_shape(pred.bias2.len()).unwrap());
            if let Some(opt_cond) = &mut self.opt_cond_w_tau {
                opt_cond.step(&mut pred.cond_w, &grads[idx + 4], lr);
            }
            let grad_activation_vec: Vec<f64> = grads[idx + 5].iter().map(|&x| x as f64).collect();
            pred.activation.step(&grad_activation_vec, lr as f64);
        }

        Ok(())
    }

    pub fn grad_arrays_len(&self) -> usize {
        let mut n = 4; // w_g, alpha_g, beta_g, gate_poly
        if self.head_selection_config.gating.use_learned_predictor {
            n += 6;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn fixed_strategy_enforces_exact_num_active_heads() {
        let embed_dim = 16;
        let num_heads = 8;
        let mut g = MoHGating::new(embed_dim, num_heads);
        g.set_head_selection_config(&HeadSelectionStrategy::Fixed { num_active: 3 });

        // Deterministic-ish input.
        let n = 5;
        let mut x = Array2::<f32>::zeros((n, embed_dim));
        for i in 0..n {
            for j in 0..embed_dim {
                x[[i, j]] = ((i * embed_dim + j) as f32 * 0.0017).sin();
            }
        }

        let eff = g.forward_weights(&x, None, None);
        assert_eq!(eff.dim(), (n, num_heads));

        for i in 0..n {
            let mut active = 0usize;
            for h in 0..num_heads {
                if eff[[i, h]] > 0.0 {
                    active += 1;
                }
            }
            assert_eq!(active, 3);
        }
    }

    #[test]
    fn always_on_heads_are_always_active_under_fixed() {
        let embed_dim = 16;
        let num_heads = 8;
        let mut g = MoHGating::new(embed_dim, num_heads);
        g.set_head_selection_config(&HeadSelectionStrategy::Fixed { num_active: 3 });
        g.set_always_on_heads(vec![0, 1]);

        let n = 6;
        let mut x = Array2::<f32>::zeros((n, embed_dim));
        for i in 0..n {
            for j in 0..embed_dim {
                x[[i, j]] = (((i + 1) * (j + 3)) as f32 * 0.0009).cos();
            }
        }

        let eff = g.forward_weights(&x, None, None);
        for i in 0..n {
            assert!(eff[[i, 0]] > 0.0);
            assert!(eff[[i, 1]] > 0.0);

            let mut active = 0usize;
            for h in 0..num_heads {
                if eff[[i, h]] > 0.0 {
                    active += 1;
                }
            }
            assert_eq!(active, 3);
        }
    }
}
