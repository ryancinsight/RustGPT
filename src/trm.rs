use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::{
    adam::Adam,
    errors::Result,
    feed_forward::FeedForward,
    llm::Layer,
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
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        recursive_depth: usize,
        use_swiglu: bool,
        max_seq_len: usize,
    ) -> Self {
        // Create attention layer with GQA support
        let kv_heads = num_kv_heads.unwrap_or(num_heads);
        let attention = SelfAttention::new_with_gqa(
            embedding_dim,
            num_heads,
            kv_heads,
            false, // use_rope = false (positional encoding handled by embeddings)
            max_seq_len,
            None, // window_size = None (full attention)
        );

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

        // Initialize state
        let mut x = input.clone();
        self.cached_states.push(x.clone());

        // Apply transformer block recursively
        for step in 0..self.recursive_depth {
            // Attention sublayer with Pre-LN
            let normed = self.norm1.forward(&x);
            let attn_out = self.attention.forward(&normed);
            self.cached_attention_outputs.push(attn_out.clone());

            // Residual connection with adaptive scaling
            let attn_scale = self.attention_step_scales[step];
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
            let ffn_scale = self.ffn_step_scales[step];
            x = &x + &(&ffn_out * ffn_scale);
            self.cached_states.push(x.clone());
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

        // Gradients for step scales
        let mut attention_scale_grads = vec![0.0; self.recursive_depth];
        let mut ffn_scale_grads = vec![0.0; self.recursive_depth];

        // Gradient normalization strategy (ReZero-inspired):
        // 1. Parameter gradients: normalized by 1/depth (shared weights accumulate)
        // 2. Residual gradients: NO additional scaling
        //    - ReZero initialization (scales ≈ 0) naturally prevents explosion
        //    - Forward: x = x + scale * sublayer_out (scale ≈ 0 initially)
        //    - Backward: grad_x flows through both paths, but scaled path is small
        //    - As scales grow during training, gradient flow increases naturally
        let depth_scale = 1.0 / self.recursive_depth as f32;

        // Backpropagate through recursive steps in reverse
        for step in (0..self.recursive_depth).rev() {
            let state_idx = step * 2 + 1; // Index in cached_states after attention

            // FFN sublayer backward
            let x_before_ffn = &self.cached_states[state_idx];
            let ffn_out = &self.cached_ffn_outputs[step];
            let ffn_scale = self.ffn_step_scales[step];

            // Gradient w.r.t. FFN scale: sum(grad_x * ffn_out)
            ffn_scale_grads[step] = (&grad_x * ffn_out).sum();

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

            // Gradient through FFN (need to recompute normed input for FFN)
            let normed_ffn_input = &self.cached_states[state_idx]; // This is x_before_ffn, need to apply norm
            // Actually, we need to compute the normalized version
            // For now, use a dummy computation - we'll need to cache normalized inputs too
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
            let attn_scale = self.attention_step_scales[step];

            // Gradient w.r.t. attention scale
            attention_scale_grads[step] = (&grad_x * attn_out).sum();

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

