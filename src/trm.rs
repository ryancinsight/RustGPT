use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::{
    adam::Adam,
    errors::Result,
    feed_forward::FeedForward,
    llm::Layer,
    model_config::{AdaptiveDepthConfig, HeadSelectionStrategy},
    rms_norm::RMSNorm,
    self_attention::SelfAttention,
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
/// - **No Gradient Clipping**: Stability achieved through adaptive mechanisms
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
    /// Self-attention layer (reused recursively)
    attention: SelfAttention,

    /// Normalization after attention
    norm1: RMSNorm,

    /// Feedforward layer (reused recursively)
    #[serde(skip)]
    feed_forward_swiglu: Option<SwiGLU>,
    #[serde(skip)]
    feed_forward_standard: Option<FeedForward>,

    /// Normalization after feedforward
    norm2: RMSNorm,

    /// Recursive depth (number of times to apply the block)
    recursive_depth: usize,

    /// Adaptive residual scaling for attention sublayer per step (learned)
    /// Initialized: scale[step] = 0.5 + 0.5 * (step / recursive_depth)
    /// Early steps: smaller scales (0.5) to prevent explosion
    /// Late steps: larger scales (1.0) to prevent vanishing
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

    /// Whether to use SwiGLU (true) or standard FeedForward (false)
    use_swiglu: bool,

    // ===== Adaptive Recursive Depth Fields =====
    /// Whether adaptive depth is enabled
    adaptive_depth_enabled: bool,

    /// Maximum recursive depth (when adaptive depth is enabled)
    max_recursive_depth: usize,

    /// Cumulative halting probability threshold (e.g., 0.95)
    halt_threshold: f32,

    /// Weight for ponder loss (penalizes excessive depth)
    ponder_loss_weight: f32,

    /// Halting predictor weights: (embedding_dim, 1)
    /// Predicts per-sequence halting probability at each step
    w_halt: Array2<f32>,

    /// Halting predictor bias
    b_halt: f32,

    /// Optimizer for halting predictor
    halt_optimizer: Adam,

    /// Actual depths used per sequence in last forward pass
    #[serde(skip)]
    actual_depths: Vec<usize>,

    /// Average depth used in last forward pass
    #[serde(skip)]
    avg_depth: f32,

    /// Actual number of steps taken in last forward pass (for backward pass)
    #[serde(skip)]
    steps_taken: usize,

    /// Cached halting probabilities for backward pass
    #[serde(skip)]
    cached_halt_probs: Vec<Array1<f32>>,

    /// Cached pooled states for backward pass (for halting predictor gradients)
    #[serde(skip)]
    cached_pooled_states: Vec<Array2<f32>>,
}

impl TinyRecursiveModel {
    /// Create a new TinyRecursiveModel
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Embedding dimension
    /// * `hidden_dim` - Hidden dimension for feedforward layer
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `recursive_depth` - Number of times to apply the block recursively
    /// * `use_swiglu` - Whether to use SwiGLU (true) or standard FeedForward (false)
    /// * `max_seq_len` - Maximum sequence length
    /// * `head_selection` - Head selection strategy (AllHeads, MoH, or FullyAdaptiveMoH)
    /// * `adaptive_depth_config` - Optional adaptive depth configuration (enables ACT mechanism)
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        recursive_depth: usize,
        use_swiglu: bool,
        max_seq_len: usize,
        head_selection: HeadSelectionStrategy,
        adaptive_depth_config: Option<AdaptiveDepthConfig>,
    ) -> Self {
        // Create attention layer with GQA support
        let kv_heads = num_kv_heads.unwrap_or(num_heads);
        let mut attention = SelfAttention::new_with_gqa(
            embedding_dim,
            num_heads,
            kv_heads,
            false, // use_rope = false (positional encoding handled by embeddings)
            max_seq_len,
            None, // window_size = None (full attention)
        );

        // Set head selection strategy (MoH, AllHeads, or FullyAdaptiveMoH)
        // Use layer_idx=0 since TRM is a single block
        attention.set_head_selection(head_selection, 0);

        // Create normalization layers
        let norm1 = RMSNorm::new(embedding_dim);
        let norm2 = RMSNorm::new(embedding_dim);

        // Create feedforward layer
        let (feed_forward_swiglu, feed_forward_standard) = if use_swiglu {
            (Some(SwiGLU::new(embedding_dim, hidden_dim)), None)
        } else {
            (None, Some(FeedForward::new(embedding_dim, hidden_dim)))
        };

        // Initialize adaptive step scales using ReZero-inspired approach
        // ReZero (Bachlechner et al., 2020): Initialize residual scales to 0
        // This allows the network to start as identity and gradually learn transformations
        // Benefits:
        // 1. No gradient explosion at initialization (identity mapping)
        // 2. Scales grow during training as needed
        // 3. Natural gradient flow through skip connections
        //
        // We use small positive values instead of pure 0 to allow some initial gradient flow
        let initial_scale = 0.01; // Small but non-zero for initial learning signal

        let attention_step_scales: Vec<f32> = vec![initial_scale; recursive_depth];
        let ffn_step_scales: Vec<f32> = vec![initial_scale; recursive_depth];

        // Create optimizers for step scales (small LR for stability)
        let attention_scale_optimizer = Adam::new((recursive_depth, 1));
        let ffn_scale_optimizer = Adam::new((recursive_depth, 1));

        // Initialize adaptive depth components
        let (adaptive_enabled, max_depth, halt_thresh, ponder_weight, w_halt, b_halt) =
            if let Some(config) = adaptive_depth_config {
                // Adaptive depth enabled
                let mut rng = rand::thread_rng();
                use rand::Rng;

                // Initialize halting predictor weights with small random values
                let w = Array2::from_shape_fn((embedding_dim, 1), |_|
                    rng.gen_range(-0.01..0.01));

                // Initialize bias to large negative value to start at max_depth
                // sigmoid(-5) ≈ 0.007, so model will use max_depth initially
                // As training progresses, model learns to increase bias for simple inputs
                let b = -5.0;

                (
                    true,
                    config.max_depth,
                    config.halt_threshold,
                    config.ponder_weight,
                    w,
                    b,
                )
            } else {
                // Fixed depth mode (current behavior)
                (
                    false,
                    recursive_depth,
                    0.95,
                    0.0,
                    Array2::zeros((embedding_dim, 1)),
                    0.0,
                )
            };

        Self {
            attention,
            norm1,
            feed_forward_swiglu,
            feed_forward_standard,
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
            use_swiglu,
            // Adaptive recursive depth (configured or disabled)
            adaptive_depth_enabled: adaptive_enabled,
            max_recursive_depth: max_depth,
            halt_threshold: halt_thresh,
            ponder_loss_weight: ponder_weight,
            w_halt,
            b_halt,
            halt_optimizer: Adam::new((embedding_dim, 1)),
            actual_depths: Vec::new(),
            avg_depth: 0.0,
            steps_taken: 0,
            cached_halt_probs: Vec::new(),
            cached_pooled_states: Vec::new(),
        }
    }

    /// Set epoch information for progress tracking
    pub fn set_epoch_info(&mut self, current_epoch: usize, max_epochs: usize) {
        self.current_epoch = current_epoch;
        self.max_epochs = max_epochs;
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
    /// Returns (avg_routed_heads, mean_threshold, conf_avg, conf_min, fallback_pct, complexity_avg, pred_norm)
    /// Returns (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if MoH is not enabled.
    pub fn get_moh_stats(&self) -> (f32, f32, f32, f32, f32, f32, f32) {
        self.attention.get_moh_stats()
    }

    /// Get temperature statistics from internal attention layer
    ///
    /// Returns (avg, min, max) or None if temperature predictor is not enabled.
    pub fn get_temperature_stats(&self) -> Option<(f32, f32, f32)> {
        self.attention.get_temperature_stats()
    }

    /// Get load balance loss from internal attention layer
    pub fn get_load_balance_loss(&self) -> f32 {
        self.attention.get_load_balance_loss()
    }

    /// Get dynamic loss from internal attention layer
    pub fn get_dynamic_loss(&self) -> f32 {
        self.attention.get_dynamic_loss()
    }

    /// Get dynamic loss weight from internal attention layer
    pub fn get_dynamic_loss_weight(&self, training_progress: f32) -> f32 {
        self.attention.get_dynamic_loss_weight(training_progress)
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
    /// ```
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
        self.cached_pooled_states.clear();

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
            let ffn_out = if self.use_swiglu {
                self.feed_forward_swiglu.as_mut().unwrap().forward(&normed)
            } else {
                self.feed_forward_standard.as_mut().unwrap().forward(&normed)
            };
            self.cached_ffn_outputs.push(ffn_out.clone());

            // Residual connection with adaptive scaling
            let ffn_scale = self.ffn_step_scales[step.min(self.recursive_depth - 1)];
            x = &x + &(&ffn_out * ffn_scale);
            self.cached_states.push(x.clone());

            // Compute halting probability (adaptive depth only)
            if self.adaptive_depth_enabled {
                // x is already (batch_size, embedding_dim) - no pooling needed
                // Just use x directly as the pooled state
                self.cached_pooled_states.push(x.clone());

                // Compute complexity score based on hidden state norm
                // High norm = high activation = complex input = need more steps
                // Low norm = low activation = simple input = can halt early
                let complexity_scores: Vec<f32> = (0..batch_size)
                    .map(|i| {
                        let row = x.row(i);
                        // L2 norm of hidden state
                        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
                        // Normalize by embedding_dim to get per-dimension activation
                        norm / (x.shape()[1] as f32).sqrt()
                    })
                    .collect();

                // Compute halting logits: W_halt · x + b_halt
                let halt_logits_2d = x.dot(&self.w_halt); // (batch_size, 1)
                let halt_logits = halt_logits_2d.into_shape(batch_size).unwrap(); // (batch_size,)
                let halt_logits = halt_logits + self.b_halt; // Add bias

                // Apply sigmoid: p_halt = sigmoid(logits)
                let mut p_halt = halt_logits.mapv(|x| 1.0 / (1.0 + (-x).exp()));

                // Complexity-aware halting: reduce halting probability for complex inputs
                // If complexity > threshold (e.g., 0.5), reduce p_halt to encourage more steps
                for i in 0..batch_size {
                    let complexity = complexity_scores[i];
                    if complexity > 0.5 {
                        // High complexity: reduce halting probability
                        // Scale down by (1 - complexity), so high complexity → low p_halt
                        p_halt[i] *= (1.0 - complexity).max(0.1); // Keep at least 10% to avoid getting stuck
                    }
                }

                // Update cumulative probabilities (only for active sequences)
                for i in 0..batch_size {
                    if active_mask[i] {
                        cumulative_probs[i] += p_halt[i];

                        // Check if sequence should halt
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
            let (grad_ffn_input, ffn_param_grads) = if self.use_swiglu {
                self.feed_forward_swiglu.as_ref().unwrap().compute_gradients(x_before_ffn, &grad_normed)
            } else {
                self.feed_forward_standard.as_ref().unwrap().compute_gradients(x_before_ffn, &grad_normed)
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
        // For SelfAttention: 3 gradients per head (W_q, W_k, W_v)
        let attention_params = self.attention.num_heads * 3;

        let norm_params = 1; // RMSNorm has 1 parameter (gamma)

        // For SwiGLU: W_gate, W_up, W_down (3 matrices)
        // For FeedForward: W1, W2, bias1, bias2 (4 parameters)
        let ffn_params = if self.use_swiglu {
            3 // W_gate, W_up, W_down
        } else {
            4 // W1, W2, bias1, bias2
        };

        // Split parameter gradients
        let mut idx = 0;

        // Apply attention gradients
        self.attention.apply_gradients(&param_grads[idx..idx + attention_params], lr)?;
        idx += attention_params;

        // Apply norm1 gradients
        self.norm1.apply_gradients(&param_grads[idx..idx + norm_params], lr)?;
        idx += norm_params;

        // Apply FFN gradients
        if self.use_swiglu {
            self.feed_forward_swiglu.as_mut().unwrap().apply_gradients(&param_grads[idx..idx + ffn_params], lr)?;
        } else {
            self.feed_forward_standard.as_mut().unwrap().apply_gradients(&param_grads[idx..idx + ffn_params], lr)?;
        }
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

        // Update halting predictor (adaptive depth only)
        if self.adaptive_depth_enabled && !self.cached_halt_probs.is_empty() {
            // Compute ponder loss gradient
            // Ponder loss: L_ponder = (avg_depth / max_depth) * ponder_weight
            // Gradient: dL/d(avg_depth) = ponder_weight / max_depth
            let ponder_grad_scale = self.ponder_loss_weight / self.max_recursive_depth as f32;

            // Backpropagate through halting predictor
            let mut w_halt_grad = Array2::zeros(self.w_halt.dim());
            let mut b_halt_grad = 0.0;

            let batch_size = self.cached_pooled_states[0].shape()[0];

            for (step_idx, (halt_prob, pooled_state)) in self.cached_halt_probs.iter()
                .zip(self.cached_pooled_states.iter())
                .enumerate()
            {
                // Gradient of sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
                let sigmoid_grad = halt_prob.mapv(|p| p * (1.0 - p));

                // Gradient from ponder loss (encourage fewer steps)
                // Each sequence contributes to average depth
                let ponder_contrib = sigmoid_grad.mapv(|g| g * ponder_grad_scale / batch_size as f32);

                // Accumulate gradients for w_halt and b_halt
                for i in 0..batch_size {
                    let state_vec = pooled_state.row(i);
                    let grad_scalar = ponder_contrib[i];

                    // Gradient for w_halt: outer product of state and scalar gradient
                    for j in 0..state_vec.len() {
                        w_halt_grad[[j, 0]] += state_vec[j] * grad_scalar;
                    }

                    // Gradient for b_halt: sum of scalar gradients
                    b_halt_grad += grad_scalar;
                }
            }

            // Update halting predictor parameters using Adam optimizer
            // Use conservative learning rate (0.01 scale factor)
            let halt_lr = lr * 0.01;
            self.halt_optimizer.step(&mut self.w_halt, &w_halt_grad, halt_lr);
            self.b_halt -= halt_lr * b_halt_grad;
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
        let ffn_params = if self.use_swiglu {
            self.feed_forward_swiglu.as_ref().unwrap().parameters()
        } else {
            self.feed_forward_standard.as_ref().unwrap().parameters()
        };
        let scale_params = self.recursive_depth * 2; // attention_step_scales + ffn_step_scales

        attention_params + norm_params + ffn_params + scale_params
    }
}

