use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::{
    adam::Adam,
    errors::Result,
    // feed_forward::FeedForward, // Removed
    llm::Layer,
    model_config::AdaptiveDepthConfig,
    dynamic_tanh_norm::DynamicTanhNorm,
    poly_attention::PolyAttention,
    swiglu::SwiGLU,
};

/// Tiny Recursive Model (TRM)
///
/// A parameter-efficient architecture that applies a single transformer block
/// recursively multiple times, similar to RNNs but with attention mechanism.
///
/// # Key Features
///
/// - **Weight Sharing**: Single transformer block reused across recursive depth
/// - **Memory Efficient**: O(1) parameters regardless of depth
/// - **Adaptive Residual Scaling**: Learned per-step scales prevent gradient issues
/// - **Per-Step Gradient Tracking**: Monitor gradient flow through recursive depth

///
/// # Architecture
///
/// ```text
/// x_0 = input
/// for step in 0..recursive_depth:
///     # Attention sublayer
///     attn_out = attention(norm1(x_step))
///     x_step = x_step + attention_scales[step] * attn_out
///     
///     # FFN sublayer
///     ffn_out = ffn(norm2(x_step))
///     x_{step+1} = x_step + ffn_scales[step] * ffn_out
/// output = x_{recursive_depth}
/// ```
///
/// # Gradient Stability
///
/// 1. **Adaptive Residual Scaling**: Each step has learned scale parameters
///    - Initialized based on position to prevent initial explosion/vanishing
///    - Updated via Adam optimizer during training
///
/// 2. **Per-Step Gradient Tracking**: Track gradient norms at each recursive step
///    - Identify vanishing/exploding patterns
///    - Log gradient flow for analysis
///
/// 3. **Gradient Accumulation**: Properly accumulate gradients across recursive steps
///    - Same block used multiple times → gradients sum up
///    - Careful bookkeeping in backward pass
///
/// 4. **Bidirectional LARS**: Layer-wise adaptive learning rates (handled externally)
///
/// # Reference
///
/// Inspired by:
/// - Universal Transformers (Dehghani et al., 2018)
/// - Adaptive Computation Time (Graves, 2016)
/// - Deep Equilibrium Models (Bai et al., 2019)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TinyRecursiveModel {
    /// Polynomial attention layer (reused recursively)
    attention: PolyAttention,

    /// Normalization after attention
    norm1: DynamicTanhNorm,

    /// Feedforward layer (reused recursively)
    #[serde(skip)]
    feed_forward_swiglu: Option<SwiGLU>,
    // #[serde(skip)] feed_forward_standard removed

    /// Normalization after feedforward
    norm2: DynamicTanhNorm,

    /// Recursive depth (number of times to apply the block)
    recursive_depth: usize,

    /// Adaptive residual scaling for attention sublayer per step (learned)
    attention_step_scales: Vec<f32>,

    /// Adaptive residual scaling for FFN sublayer per step (learned)
    ffn_step_scales: Vec<f32>,

    /// Optimizer for attention step scales
    attention_scale_optimizer: Adam,

    /// Optimizer for FFN step scales
    ffn_scale_optimizer: Adam,

    /// Cached states for backward pass (one per recursive step)
    #[serde(skip)]
    cached_states: Vec<Array2<f32>>,

    /// Cached attention outputs for backward pass
    #[serde(skip)]
    cached_attention_outputs: Vec<Array2<f32>>,

    /// Cached FFN outputs for backward pass
    #[serde(skip)]
    cached_ffn_outputs: Vec<Array2<f32>>,

    /// Gradient norms per recursive step (for tracking)
    #[serde(skip)]
    step_gradient_norms: Vec<f32>,

    /// Current epoch (for progress tracking)
    current_epoch: usize,

    /// Maximum epochs (for progress tracking)
    max_epochs: usize,

    /// Embedding dimension
    embedding_dim: usize,

    // use_swiglu removed; SwiGLU is always used

    // ===== Adaptive Recursive Depth Fields =====
    adaptive_depth_enabled: bool,
    max_recursive_depth: usize,
    halt_threshold: f32,
    ponder_loss_weight: f32,
    w_attn: Array2<f32>,
    w_halt: Array2<f32>,
    b_halt: f32,
    gumbel_temperature: f32,
    attn_optimizer: Adam,
    halt_optimizer: Adam,
    #[serde(skip)]
    actual_depths: Vec<usize>,
    #[serde(skip)]
    avg_depth: f32,
    #[serde(skip)]
    steps_taken: usize,
    #[serde(skip)]
    cached_halt_probs: Vec<Array1<f32>>,
    #[serde(skip)]
    cached_attn_weights: Vec<Array1<f32>>,
    #[serde(skip)]
    cached_contexts: Vec<Array1<f32>>,
}

impl TinyRecursiveModel {
    /// Create a new TinyRecursiveModel
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Embedding dimension
    /// * `hidden_dim` - Hidden dimension for feedforward layer
    /// * `num_heads` - Number of attention heads
    /// * `degree_p` - Polynomial degree `p` for PolyAttention (odd integer)
    /// * `recursive_depth` - Number of times to apply the block recursively
    /// * `adaptive_depth_config` - Optional adaptive depth configuration (enables ACT mechanism)
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        recursive_depth: usize,
        degree_p: usize,
        adaptive_depth_config: Option<AdaptiveDepthConfig>,
    ) -> Self {
        // Create polynomial attention layer (PolyAttention)
        let mut attention = PolyAttention::new(
            embedding_dim,
            num_heads,
            degree_p,
        );

        // Create normalization layers
        let norm1 = DynamicTanhNorm::new(embedding_dim);
        let norm2 = DynamicTanhNorm::new(embedding_dim);

        // Create feedforward layer (SwiGLU only)
        let feed_forward_swiglu = Some(SwiGLU::new(embedding_dim, hidden_dim));

        // Initialize adaptive step scales using ReZero-inspired approach
        let initial_scale = 0.01; // Small but non-zero for initial learning signal

        let attention_step_scales: Vec<f32> = vec![initial_scale; recursive_depth];
        let ffn_step_scales: Vec<f32> = vec![initial_scale; recursive_depth];

        // Create optimizers for step scales (small LR for stability)
        let attention_scale_optimizer = Adam::new((recursive_depth, 1));
        let ffn_scale_optimizer = Adam::new((recursive_depth, 1));

        // Initialize adaptive depth components
        let (adaptive_enabled, max_depth, halt_thresh, ponder_weight, w_attn, w_halt, b_halt, gumbel_temp) =
            if let Some(config) = adaptive_depth_config {
                let adaptive_enabled = true;
                let max_depth = config.max_depth;
                let halt_thresh = config.halt_threshold;
                let ponder_weight = config.ponder_weight;
                let w_attn = Array2::zeros((embedding_dim, 1));
                let w_halt = Array2::zeros((embedding_dim * 2, 1));
                let b_halt = 0.0;
                let gumbel_temp = 1.0;
                (adaptive_enabled, max_depth, halt_thresh, ponder_weight, w_attn, w_halt, b_halt, gumbel_temp)
            } else {
                let adaptive_enabled = false;
                let max_depth = recursive_depth;
                let halt_thresh = 1.0;
                let ponder_weight = 0.0;
                let w_attn = Array2::zeros((embedding_dim, 1));
                let w_halt = Array2::zeros((embedding_dim * 2, 1));
                let b_halt = 0.0;
                let gumbel_temp = 1.0;
                (adaptive_enabled, max_depth, halt_thresh, ponder_weight, w_attn, w_halt, b_halt, gumbel_temp)
            };

        Self {
            attention,
            norm1,
            feed_forward_swiglu,
            // feed_forward_standard: None, // Removed
            norm2,
            recursive_depth,
            attention_step_scales,
            ffn_step_scales,
            attention_scale_optimizer,
            ffn_scale_optimizer,
            cached_states: Vec::new(),
            cached_attention_outputs: Vec::new(),
            cached_ffn_outputs: Vec::new(),
            step_gradient_norms: vec![0.0; recursive_depth],
            current_epoch: 0,
            max_epochs: 100,
            embedding_dim,
            // use_swiglu,
            adaptive_depth_enabled: adaptive_enabled,
            max_recursive_depth: max_depth,
            halt_threshold: halt_thresh,
            ponder_loss_weight: ponder_weight,
            w_attn,
            w_halt,
            b_halt,
            gumbel_temperature: gumbel_temp,
            attn_optimizer: Adam::new((embedding_dim, 1)),
            halt_optimizer: Adam::new((embedding_dim * 2, 1)),
            actual_depths: Vec::new(),
            avg_depth: 0.0,
            steps_taken: 0,
            cached_halt_probs: Vec::new(),
            cached_attn_weights: Vec::new(),
            cached_contexts: Vec::new(),
        }
    }

    /// Set epoch information for progress tracking
    pub fn set_epoch_info(&mut self, current_epoch: usize, max_epochs: usize) {
        self.current_epoch = current_epoch;
        self.max_epochs = max_epochs;
    }

    /// Gumbel-Softmax: differentiable approximation to categorical sampling
    ///
    /// Formula: softmax((log(π_i) + g_i) / τ)
    /// where g_i ~ Gumbel(0, 1) and τ is temperature
    ///
    /// Properties:
    /// - τ → 0: approaches one-hot (argmax)
    /// - τ → ∞: approaches uniform distribution
    /// - Differentiable w.r.t. logits (maintains gradient flow)
    fn gumbel_softmax(logits: &Array1<f32>, temperature: f32, training: bool) -> Array1<f32> {
        if training {
            // Sample Gumbel noise: g ~ -log(-log(U)) where U ~ Uniform(0, 1)
            let mut rng = rand::rng();
            use rand::Rng;
            let gumbel_noise: Array1<f32> = Array1::from_shape_fn(logits.len(), |_| {
                let u: f32 = rng.random_range(1e-10..1.0); // Avoid log(0)
                -(-u.ln()).ln()
            });

            // Add Gumbel noise and scale by temperature
            let perturbed = (logits + &gumbel_noise) / temperature;

            // Apply softmax
            let max_val = perturbed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals = perturbed.mapv(|x| (x - max_val).exp());
            let sum_exp = exp_vals.sum();
            exp_vals / sum_exp
        } else {
            // Inference: use standard softmax (no noise)
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals = logits.mapv(|x| (x - max_val).exp());
            let sum_exp = exp_vals.sum();
            exp_vals / sum_exp
        }
    }

    /// Get gradient statistics for logging
    pub fn get_gradient_stats(&self) -> String {
        let avg_norm = self.step_gradient_norms.iter().sum::<f32>() / self.recursive_depth as f32;
        let min_norm = self.step_gradient_norms.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_norm = self.step_gradient_norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        format!(
            "TRM Steps: avg={:.2}, min={:.2}, max={:.2}",
            avg_norm, min_norm, max_norm
        )
    }

    /// Get step scales for logging
    pub fn get_step_scales(&self) -> String {
        let attn_scales: Vec<String> = self.attention_step_scales.iter()
            .enumerate()
            .map(|(i, s)| format!("S{}:{:.2}", i, s))
            .collect();
        let ffn_scales: Vec<String> = self.ffn_step_scales.iter()
            .enumerate()
            .map(|(i, s)| format!("S{}:{:.2}", i, s))
            .collect();

        format!(
            "Attn[{}] FFN[{}]",
            attn_scales.join(","),
            ffn_scales.join(",")
        )
    }

    /// Get MoH statistics from internal attention layer
    ///
    /// Returns (avg_routed_heads, mean_threshold, conf_avg, conf_min, fallback_pct, complexity_avg, complexity_min, complexity_max, pred_norm)
    /// Returns (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if MoH is not enabled.
    pub fn get_moh_stats(&self) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        // PolyAttention does not expose MoH stats; return neutral defaults
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Get temperature statistics from internal attention layer
    ///
    /// Returns (avg, min, max) or None if temperature predictor is not enabled.
    pub fn get_temperature_stats(&self) -> Option<(f32, f32, f32)> {
        // PolyAttention does not expose a temperature predictor; no stats available
        None
    }

    /// Get load balance loss from internal attention layer
    pub fn get_load_balance_loss(&self) -> f32 {
        0.0
    }

    /// Get dynamic loss from internal attention layer
    pub fn get_dynamic_loss(&self) -> f32 {
        0.0
    }

    /// Get dynamic loss weight from internal attention layer
    pub fn get_dynamic_loss_weight(&self, _training_progress: f32) -> f32 {
        0.0
    }

    /// Get ponder loss (penalizes excessive depth in adaptive mode)
    ///
    /// Ponder loss = (avg_depth / max_depth) * ponder_weight
    /// Encourages the model to use fewer recursive steps when possible.
    ///
    /// Returns 0.0 if adaptive depth is disabled or no forward pass has been run.
    pub fn get_ponder_loss(&self) -> f32 {
        if self.adaptive_depth_enabled && !self.actual_depths.is_empty() {
            (self.avg_depth / self.max_recursive_depth as f32) * self.ponder_loss_weight
        } else {
            0.0
        }
    }

    /// Get depth statistics for logging (adaptive depth only)
    ///
    /// Returns (avg_depth, min_depth, max_depth) or None if adaptive depth is disabled.
    ///
    /// # Example
    /// ```rust
    /// use llm::model_config::AdaptiveDepthConfig;
    /// use llm::trm::TinyRecursiveModel;
    /// let trm = TinyRecursiveModel::new(
    ///     64,   // embedding_dim
    ///     128,  // hidden_dim
    ///     4,    // num_heads
    ///     3,    // recursive_depth
    ///     3,    // degree_p for PolyAttention (odd)
    ///     Some(AdaptiveDepthConfig::default()),
    /// );
    /// if let Some((avg, min, max)) = trm.get_depth_stats() {
    ///     println!("Depth: avg={:.1} [{}-{}]", avg, min, max);
    /// }
    /// ```
    pub fn get_depth_stats(&self) -> Option<(f32, usize, usize)> {
        if self.adaptive_depth_enabled && !self.actual_depths.is_empty() {
            let min_depth = *self.actual_depths.iter().min().unwrap();
            let max_depth = *self.actual_depths.iter().max().unwrap();
            Some((self.avg_depth, min_depth, max_depth))
        } else {
            None
        }
    }
}

impl Layer for TinyRecursiveModel {
    fn layer_type(&self) -> &str {
        "TinyRecursiveModel"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Clear caches
        self.cached_states.clear();
        self.cached_attention_outputs.clear();
        self.cached_ffn_outputs.clear();
        self.cached_halt_probs.clear();
        self.cached_attn_weights.clear();
        self.cached_contexts.clear();

        // Initialize state
        let mut x = input.clone();
        self.cached_states.push(x.clone());

        // Determine actual recursive depth and initialize adaptive depth tracking
        let (actual_depth, batch_size) = if self.adaptive_depth_enabled {
            let batch_size = input.shape()[0];
            (self.max_recursive_depth, batch_size)
        } else {
            (self.recursive_depth, 0)
        };

        // Track halting for adaptive depth
        let mut cumulative_probs: Array1<f32> = if self.adaptive_depth_enabled {
            Array1::zeros(batch_size)
        } else {
            Array1::zeros(1) // Dummy array for non-adaptive mode
        };
        let mut active_mask = if self.adaptive_depth_enabled {
            vec![true; batch_size]
        } else {
            vec![] // Empty vec for non-adaptive mode
        };

        // Apply transformer block recursively
        let mut steps_taken = 0;
        for step in 0..actual_depth {
            // Check if all sequences have halted (adaptive depth only)
            if self.adaptive_depth_enabled && !active_mask.iter().any(|&a| a) {
                break; // All sequences halted, stop early
            }

            steps_taken += 1;

            // Attention sublayer with Pre-LN
            let normed = self.norm1.forward(&x);
            let attn_out = self.attention.forward(&normed);
            self.cached_attention_outputs.push(attn_out.clone());

            // Residual connection with adaptive scaling
            let attn_scale = self.attention_step_scales[step.min(self.recursive_depth - 1)];
            x = &x + &(&attn_out * attn_scale);
            self.cached_states.push(x.clone());

            // FFN sublayer with Pre-LN
            let normed = self.norm2.forward(&x);
            let ffn_out = {
                self.feed_forward_swiglu.as_mut().unwrap().forward(&normed)
            };
            self.cached_ffn_outputs.push(ffn_out.clone());

            // Residual connection with adaptive scaling
            let ffn_scale = self.ffn_step_scales[step.min(self.recursive_depth - 1)];
            x = &x + &(&ffn_out * ffn_scale);
            self.cached_states.push(x.clone());

            // Compute halting probability with Gumbel-Softmax attention (adaptive depth only)
            if self.adaptive_depth_enabled {
                // Step 1: Compute attention logits over sequence positions
                // attn_logits = x · W_attn  →  (seq_len, 1)
                let attn_logits_2d = x.dot(&self.w_attn);
                let attn_logits = attn_logits_2d.into_shape_with_order(batch_size).unwrap(); // (seq_len,)

                // Step 2: Apply Gumbel-Softmax to get differentiable attention weights
                // Training: adds Gumbel noise for exploration
                // Inference: standard softmax (deterministic)
                let training = self.current_epoch < self.max_epochs; // Simple training flag
                let attn_weights = Self::gumbel_softmax(&attn_logits, self.gumbel_temperature, training);
                self.cached_attn_weights.push(attn_weights.clone());

                // Step 3: Aggregate sequence context using attention weights
                // context = Σ(attn_weights[i] * x[i])  →  (embedding_dim,)
                let mut context: Array1<f32> = Array1::zeros(self.embedding_dim);
                for i in 0..batch_size {
                    let token_features = x.row(i);
                    let weighted_features = token_features.mapv(|v| v * attn_weights[i]);
                    context = &context + &weighted_features;
                }
                self.cached_contexts.push(context.clone());

                // Step 4: Concatenate per-token features with global context
                // For each token: [token_features, context]  →  (seq_len, embedding_dim * 2)
                let mut combined = Array2::zeros((batch_size, self.embedding_dim * 2));
                for i in 0..batch_size {
                    // First half: token-specific features
                    combined.slice_mut(s![i, ..self.embedding_dim])
                        .assign(&x.row(i));
                    // Second half: global context (same for all tokens)
                    combined.slice_mut(s![i, self.embedding_dim..])
                        .assign(&context);
                }

                // Step 5: Compute halting logits from concatenated features
                // halt_logits = combined · W_halt + b_halt  →  (seq_len, 1)
                let halt_logits_2d = combined.dot(&self.w_halt);
                let halt_logits = halt_logits_2d.column(0).to_owned(); // (seq_len,)
                let halt_logits = halt_logits + self.b_halt; // Add bias

                // Step 6: Apply sigmoid to get halting probabilities
                let p_halt = halt_logits.mapv(|x| 1.0 / (1.0 + (-x).exp()));

                // Step 7: Update cumulative probabilities and check for halting
                for i in 0..batch_size {
                    if active_mask[i] {
                        cumulative_probs[i] += p_halt[i];

                        // Check if token should halt
                        if cumulative_probs[i] >= self.halt_threshold {
                            active_mask[i] = false;
                        }
                    }
                }

                self.cached_halt_probs.push(p_halt);
            }
        }

        // Store steps taken for backward pass
        self.steps_taken = steps_taken;

        // Track actual depth used (adaptive depth only)
        if self.adaptive_depth_enabled {
            let depths: Vec<usize> = (0..batch_size)
                .map(|i| {
                    // Find first step where cumulative_p >= threshold
                    let mut cum_p = 0.0;
                    for (step_idx, p_halt) in self.cached_halt_probs.iter().enumerate() {
                        cum_p += p_halt[i];
                        if cum_p >= self.halt_threshold {
                            return step_idx + 1; // Halted at this step (1-indexed)
                        }
                    }
                    steps_taken // Reached max depth without halting
                })
                .collect();

            self.actual_depths = depths;
            self.avg_depth = self.actual_depths.iter().sum::<usize>() as f32 / batch_size as f32;
        }

        x
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Backward pass through recursive steps in reverse order
        let mut grad_x = output_grads.clone();

        // Accumulate gradients for shared components
        let mut attention_param_grads_acc: Option<Vec<Array2<f32>>> = None;
        let mut norm1_param_grads_acc: Option<Vec<Array2<f32>>> = None;
        let mut ffn_param_grads_acc: Option<Vec<Array2<f32>>> = None;
        let mut norm2_param_grads_acc: Option<Vec<Array2<f32>>> = None;

        // Gradients for step scales (use recursive_depth size, not steps_taken)
        let mut attention_scale_grads = vec![0.0; self.recursive_depth];
        let mut ffn_scale_grads = vec![0.0; self.recursive_depth];

        // Gradient normalization strategy (ReZero-inspired):
        // 1. Parameter gradients: normalized by 1/depth (shared weights accumulate)
        // 2. Residual gradients: NO additional scaling
        //    - ReZero initialization (scales ≈ 0) naturally prevents explosion
        //    - Forward: x = x + scale * sublayer_out (scale ≈ 0 initially)
        //    - Backward: grad_x flows through both paths, but scaled path is small
        //    - As scales grow during training, gradient flow increases naturally
        let depth_scale = 1.0 / self.steps_taken as f32;

        // Backpropagate through recursive steps in reverse (use actual steps taken)
        for step in (0..self.steps_taken).rev() {
            let state_idx = step * 2 + 1; // Index in cached_states after attention

            // FFN sublayer backward
            let x_before_ffn = &self.cached_states[state_idx];
            let ffn_out = &self.cached_ffn_outputs[step];
            let scale_idx = step.min(self.recursive_depth - 1);
            let ffn_scale = self.ffn_step_scales[scale_idx];

            // Gradient w.r.t. FFN scale: sum(grad_x * ffn_out)
            // Only accumulate gradient if step is within scale vector bounds
            if step < self.recursive_depth {
                ffn_scale_grads[step] = (&grad_x * ffn_out).sum();
            }

            // Gradient w.r.t. FFN output
            let grad_ffn_out = &grad_x * ffn_scale;

            // Gradient through norm2
            let (grad_normed, norm2_param_grads) = self.norm2.compute_gradients(x_before_ffn, &grad_ffn_out);

            // Accumulate norm2 gradients (normalize by recursive depth)
            if let Some(ref mut acc) = norm2_param_grads_acc {
                for (acc_grad, grad) in acc.iter_mut().zip(norm2_param_grads.iter()) {
                    *acc_grad = &*acc_grad + &(grad * depth_scale);
                }
            } else {
                norm2_param_grads_acc = Some(norm2_param_grads.iter().map(|g| g * depth_scale).collect());
            }

            // Gradient through FFN
            // FFN's compute_gradients expects unnormalized input and gradient from norm
            let (grad_ffn_input, ffn_param_grads) = {
                self.feed_forward_swiglu.as_ref().unwrap().compute_gradients(x_before_ffn, &grad_normed)
            };

            // Accumulate FFN gradients (normalize by recursive depth)
            if let Some(ref mut acc) = ffn_param_grads_acc {
                for (acc_grad, grad) in acc.iter_mut().zip(ffn_param_grads.iter()) {
                    *acc_grad = &*acc_grad + &(grad * depth_scale);
                }
            } else {
                ffn_param_grads_acc = Some(ffn_param_grads.iter().map(|g| g * depth_scale).collect());
            }

            // Add residual gradient (no additional scaling with ReZero)
            // The forward pass used ffn_scale (≈0 initially), so backward naturally scales
            grad_x = &grad_x + &grad_ffn_input;

            // Attention sublayer backward
            let state_idx = step * 2; // Index in cached_states before attention
            let x_before_attn = &self.cached_states[state_idx];
            let attn_out = &self.cached_attention_outputs[step];
            let attn_scale = self.attention_step_scales[scale_idx];

            // Gradient w.r.t. attention scale
            // Only accumulate gradient if step is within scale vector bounds
            if step < self.recursive_depth {
                attention_scale_grads[step] = (&grad_x * attn_out).sum();
            }

            // Gradient w.r.t. attention output
            let grad_attn_out = &grad_x * attn_scale;

            // Gradient through norm1
            let (grad_normed, norm1_param_grads) = self.norm1.compute_gradients(x_before_attn, &grad_attn_out);

            // Accumulate norm1 gradients (normalize by recursive depth)
            if let Some(ref mut acc) = norm1_param_grads_acc {
                for (acc_grad, grad) in acc.iter_mut().zip(norm1_param_grads.iter()) {
                    *acc_grad = &*acc_grad + &(grad * depth_scale);
                }
            } else {
                norm1_param_grads_acc = Some(norm1_param_grads.iter().map(|g| g * depth_scale).collect());
            }

            // Gradient through attention
            let (grad_attn_input, attn_param_grads) = self.attention.compute_gradients(x_before_attn, &grad_normed);

            // Accumulate attention gradients (normalize by recursive depth to prevent explosion)
            if let Some(ref mut acc) = attention_param_grads_acc {
                for (acc_grad, grad) in acc.iter_mut().zip(attn_param_grads.iter()) {
                    *acc_grad = &*acc_grad + &(grad * depth_scale);
                }
            } else {
                attention_param_grads_acc = Some(attn_param_grads.iter().map(|g| g * depth_scale).collect());
            }

            // Add residual gradient (no additional scaling with ReZero)
            // The forward pass used attn_scale (≈0 initially), so backward naturally scales
            grad_x = &grad_x + &grad_attn_input;
        }

        // Combine all parameter gradients
        let mut param_grads = Vec::new();

        // Attention gradients
        if let Some(grads) = attention_param_grads_acc {
            param_grads.extend(grads);
        }

        // Norm1 gradients
        if let Some(grads) = norm1_param_grads_acc {
            param_grads.extend(grads);
        }

        // FFN gradients
        if let Some(grads) = ffn_param_grads_acc {
            param_grads.extend(grads);
        }

        // Norm2 gradients
        if let Some(grads) = norm2_param_grads_acc {
            param_grads.extend(grads);
        }

        // Step scale gradients (as 1D arrays reshaped to 2D)
        param_grads.push(Array2::from_shape_vec((self.recursive_depth, 1), attention_scale_grads).unwrap());
        param_grads.push(Array2::from_shape_vec((self.recursive_depth, 1), ffn_scale_grads).unwrap());

        (grad_x, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        // Calculate number of parameters for each component
        // For PolyAttention: per-head (q,k,v), plus W_out, scalars (a,b,scale), gating (w_g, alpha_g, beta_g)
        let attention_params = self.attention.num_heads * 3 + 1 + 3 + 3;

        let norm_params = 3; // DynamicTanhNorm has 3 parameters (alpha, gamma, beta)

        // For SwiGLU: W_gate, W_up, W_down (3 matrices)
        let ffn_params = 3; // SwiGLU-only: W_gate, W_up, W_down

        // Split parameter gradients
        let mut idx = 0;

        // Apply attention gradients
        self.attention.apply_gradients(&param_grads[idx..idx + attention_params], lr)?;
        idx += attention_params;

        // Apply norm1 gradients
        self.norm1.apply_gradients(&param_grads[idx..idx + norm_params], lr)?;
        idx += norm_params;

        // Apply FFN gradients
        self.feed_forward_swiglu.as_mut().unwrap().apply_gradients(&param_grads[idx..idx + ffn_params], lr)?;
        idx += ffn_params;

        // Apply norm2 gradients
        self.norm2.apply_gradients(&param_grads[idx..idx + norm_params], lr)?;
        idx += norm_params;

        // Apply step scale gradients
        let attention_scale_grads_2d = &param_grads[idx];
        let ffn_scale_grads_2d = &param_grads[idx + 1];

        // Convert Vec<f32> to Array2 for Adam optimizer
        let mut attention_scales_2d = Array2::from_shape_vec(
            (self.recursive_depth, 1),
            self.attention_step_scales.clone()
        ).unwrap();
        let mut ffn_scales_2d = Array2::from_shape_vec(
            (self.recursive_depth, 1),
            self.ffn_step_scales.clone()
        ).unwrap();

        // Update step scales using Adam optimizer with small LR
        // Use 10x smaller LR than main parameters for stability
        let scale_lr = 0.0001;
        self.attention_scale_optimizer.step(&mut attention_scales_2d, attention_scale_grads_2d, scale_lr);
        self.ffn_scale_optimizer.step(&mut ffn_scales_2d, ffn_scale_grads_2d, scale_lr);

        // Convert back to Vec<f32> and clamp to conservative range [0.01, 0.5]
        // Tighter range prevents gradient explosion in recursive networks
        // Upper bound 0.5 ensures residual connections stay weak enough for stability
        self.attention_step_scales = attention_scales_2d.column(0).iter()
            .map(|&s| s.clamp(0.01, 0.5))
            .collect();
        self.ffn_step_scales = ffn_scales_2d.column(0).iter()
            .map(|&s| s.clamp(0.01, 0.5))
            .collect();

        // Update halting predictor with Gumbel-Softmax attention (adaptive depth only)
        if self.adaptive_depth_enabled && !self.cached_halt_probs.is_empty() {
            // Compute ponder loss gradient
            // Ponder loss: L_ponder = (avg_depth / max_depth) * ponder_weight
            // Gradient: dL/d(avg_depth) = ponder_weight / max_depth
            let ponder_grad_scale = self.ponder_loss_weight / self.max_recursive_depth as f32;

            // Get batch size (seq_len) from first cached state
            let batch_size = self.cached_states[0].shape()[0];

            // Initialize gradients
            let mut w_attn_grad = Array2::zeros(self.w_attn.dim());
            let mut w_halt_grad = Array2::zeros(self.w_halt.dim());
            let mut b_halt_grad = 0.0;

            // Backpropagate through each recursive step
            for step_idx in 0..self.cached_halt_probs.len() {
                let halt_prob = &self.cached_halt_probs[step_idx];
                let attn_weights = &self.cached_attn_weights[step_idx];
                let context = &self.cached_contexts[step_idx];
                let x = &self.cached_states[step_idx * 2 + 1]; // State after attention sublayer

                // Gradient of sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
                let sigmoid_grad = halt_prob.mapv(|p| p * (1.0 - p));

                // Gradient from ponder loss (encourage fewer steps)
                let ponder_contrib = sigmoid_grad.mapv(|g| g * ponder_grad_scale / batch_size as f32);

                // Backprop through halting predictor: halt_logits = combined · W_halt + b_halt
                // where combined = [x, context] (concatenated)
                for i in 0..batch_size {
                    let grad_scalar = ponder_contrib[i];

                    // Gradient for W_halt (first half: token features)
                    let token_features = x.row(i);
                    for j in 0..self.embedding_dim {
                        w_halt_grad[[j, 0]] += token_features[j] * grad_scalar;
                    }

                    // Gradient for W_halt (second half: context)
                    for j in 0..self.embedding_dim {
                        w_halt_grad[[self.embedding_dim + j, 0]] += context[j] * grad_scalar;
                    }

                    // Gradient for b_halt
                    b_halt_grad += grad_scalar;
                }

                // Backprop through context aggregation: context = Σ(attn_weights[i] * x[i])
                // Gradient w.r.t. attention weights
                let mut grad_attn_weights: Array1<f32> = Array1::zeros(batch_size);
                for i in 0..batch_size {
                    // Sum over all tokens that use this context
                    for j in 0..batch_size {
                        let grad_scalar: f32 = ponder_contrib[j];
                        // Gradient flows through W_halt (second half)
                        for k in 0..self.embedding_dim {
                            let x_val: f32 = x[[i, k]];
                            let w_val: f32 = w_halt_grad[[self.embedding_dim + k, 0]];
                            grad_attn_weights[i] += x_val * w_val * grad_scalar;
                        }
                    }
                }

                // Backprop through Gumbel-Softmax: attn_weights = softmax((x · W_attn) / temp)
                // Gradient of softmax: dL/dy_i = y_i * (dL/dy_i - Σ_j y_j * dL/dy_j)
                let sum_weighted_grad: f32 = (0..batch_size)
                    .map(|i| attn_weights[i] * grad_attn_weights[i])
                    .sum();

                let grad_attn_logits: Array1<f32> = Array1::from_shape_fn(batch_size, |i| {
                    let attn_w: f32 = attn_weights[i];
                    let grad_w: f32 = grad_attn_weights[i];
                    attn_w * (grad_w - sum_weighted_grad) / self.gumbel_temperature
                });

                // Backprop through attention logits: attn_logits = x · W_attn
                for i in 0..batch_size {
                    let token_features = x.row(i);
                    let grad_scalar: f32 = grad_attn_logits[i];
                    for j in 0..self.embedding_dim {
                        let feat_val: f32 = token_features[j];
                        w_attn_grad[[j, 0]] += feat_val * grad_scalar;
                    }
                }
            }

            // Update parameters using Adam optimizer
            self.attn_optimizer.step(&mut self.w_attn, &w_attn_grad, lr);
            self.halt_optimizer.step(&mut self.w_halt, &w_halt_grad, lr);
            self.b_halt -= lr * b_halt_grad;

            // Anneal Gumbel-Softmax temperature: τ_t = max(0.5, exp(-0.01 * epoch))
            // Starts at 1.0, decays to 0.5 over ~70 epochs
            let target_temp = (0.5_f32).max((-0.01 * self.current_epoch as f32).exp());
            self.gumbel_temperature = target_temp;
        }

        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        let attention_params = self.attention.parameters();
        let norm_params = self.norm1.parameters() + self.norm2.parameters();
        let ffn_params = self.feed_forward_swiglu.as_ref().unwrap().parameters();
        let scale_params = self.recursive_depth * 2; // attention_step_scales + ffn_step_scales

        attention_params + norm_params + ffn_params + scale_params
    }
}

