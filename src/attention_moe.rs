//! Mixture-of-Experts for Attention with Mixture-of-Heads
//!
//! Implements hierarchical adaptive routing: MoE → Experts → MoH → Heads
//! Each attention expert uses MoH for dynamic head selection.
//!
//! # Architecture
//!
//! ```text
//! Input → MoE Router → Select Top-K Attention Experts
//!                   ↓
//!         Expert 1: Attention with MoH (shared + routed heads)
//!         Expert 2: Attention with MoH (shared + routed heads)
//!         ...
//!                   ↓
//!         Weighted Sum → Output
//! ```
//!
//! # Key Features
//!
//! - **Hierarchical Routing**: Two-level adaptive selection (experts → heads)
//! - **Expert Specialization**: Each expert can specialize its attention patterns
//! - **Shared Adaptive Mechanisms**: Warm-up, annealing, gradient smoothing
//! - **Interpretable Patterns**: Clear hierarchy of routing decisions
//!
//! # References
//!
//! - Jin, P., et al. (2024). MoH: Multi-Head Attention as Mixture-of-Head Attention.
//! - Fedus, W., et al. (2022). Switch Transformers.

use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;
use crate::model_config::HeadSelectionStrategy;
use crate::self_attention::SelfAttention;

/// Attention expert with MoH routing
///
/// Each expert is a full attention mechanism with its own MoH system.
/// Experts can specialize their head usage patterns.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AttentionExpert {
    /// Self-attention layer with MoH
    attention: SelfAttention,
    
    /// Expert index (for logging)
    expert_idx: usize,
}

impl AttentionExpert {
    /// Create a new attention expert with MoH
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Input/output dimension
    /// * `num_shared_heads` - Number of always-active shared heads
    /// * `num_routed_heads` - Number of adaptively-routed heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `use_learned_threshold` - Enable learned threshold predictor
    /// * `expert_idx` - Expert index (for logging)
    pub fn new(
        embedding_dim: usize,
        num_shared_heads: usize,
        num_routed_heads: usize,
        num_kv_heads: usize,
        use_learned_threshold: bool,
        expert_idx: usize,
    ) -> Self {
        // Create attention layer
        let total_heads = num_shared_heads + num_routed_heads;
        let mut attention = SelfAttention::new_with_gqa(
            embedding_dim,
            total_heads,
            num_kv_heads,
            false, // use_rope
            512,   // max_seq_len (placeholder)
            None,  // window_size
        );

        // Configure MoH
        let head_selection = HeadSelectionStrategy::MixtureOfHeads {
            num_shared_heads,
            num_active_routed_heads: num_routed_heads,
            load_balance_weight: 0.01,
            threshold_p_base: 0.5,
            dynamic_loss_weight_base: 1.0,
            use_learned_threshold,
            target_avg_routed_heads: (num_routed_heads as f32) / 2.0, // Target 50% of routed heads
            confidence_threshold: 0.6,
            use_confidence_fallback: false,
        };

        attention.set_head_selection(head_selection, expert_idx);

        AttentionExpert {
            attention,
            expert_idx,
        }
    }
    
    /// Forward pass through expert attention
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.attention.forward(input)
    }
    
    /// Backward pass through expert attention
    pub fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        self.attention.backward(grads, lr)
    }
    
    /// Get MoH statistics from this expert
    pub fn get_moh_stats(&self) -> (f32, f32, f32, f32, f32, f32, f32) {
        self.attention.get_moh_stats()
    }
    
    /// Get predictor weight norm from this expert
    pub fn get_predictor_weight_norm(&self) -> f32 {
        self.attention.get_predictor_weight_norm()
    }
    
    /// Set epoch information for warm-up and annealing
    pub fn set_epoch_info(&mut self, current_epoch: usize, max_epochs: usize) {
        self.attention.set_epoch_info(current_epoch, max_epochs);
    }
    
    /// Get expert index
    pub fn expert_idx(&self) -> usize {
        self.expert_idx
    }
}

/// Router for attention expert selection
///
/// Routes tokens to top-k attention experts using learned gating.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AttentionRouter {
    /// Gating weights: (embedding_dim, num_experts)
    w_gate: Array2<f32>,
    
    /// Number of experts
    num_experts: usize,
    
    /// Number of experts to activate per token (k in top-k)
    num_active_experts: usize,
    
    /// Load balance loss weight
    load_balance_weight: f32,
    
    /// Router z-loss weight
    router_z_loss_weight: f32,
    
    /// Optimizer for gating weights
    optimizer: Adam,
    
    /// Cached routing decisions
    cached_input: Option<Array2<f32>>,
    cached_logits: Option<Array2<f32>>,
    cached_expert_indices: Option<Vec<Vec<usize>>>,
    cached_expert_weights: Option<Vec<Vec<f32>>>,
}

impl AttentionRouter {
    /// Create a new attention router
    pub fn new(
        embedding_dim: usize,
        num_experts: usize,
        num_active_experts: usize,
        load_balance_weight: f32,
        router_z_loss_weight: f32,
    ) -> Self {
        let mut rng = rand::rng();
        
        // Xavier initialization for gating weights
        let std = (2.0 / (embedding_dim + num_experts) as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let w_gate = Array2::from_shape_fn((embedding_dim, num_experts), |_| {
            normal.sample(&mut rng)
        });
        
        AttentionRouter {
            w_gate,
            num_experts,
            num_active_experts,
            load_balance_weight,
            router_z_loss_weight,
            optimizer: Adam::new((embedding_dim, num_experts)),
            cached_input: None,
            cached_logits: None,
            cached_expert_indices: None,
            cached_expert_weights: None,
        }
    }
    
    /// Route tokens to attention experts using top-k selection
    ///
    /// Returns (expert_indices, expert_weights, auxiliary_loss)
    pub fn route(
        &mut self,
        input: &Array2<f32>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, f32) {
        let (seq_len, _) = input.dim();
        
        // 1. Compute routing logits: input @ w_gate
        let logits = input.dot(&self.w_gate); // (seq_len, num_experts)
        
        // 2. Apply softmax to get routing probabilities
        let probs = self.softmax(&logits);
        
        // 3. Select top-k experts per token
        let mut expert_indices = Vec::with_capacity(seq_len);
        let mut expert_weights = Vec::with_capacity(seq_len);
        
        for token_idx in 0..seq_len {
            let token_probs = probs.row(token_idx);
            
            // Get top-k expert indices
            let mut indexed_probs: Vec<(usize, f32)> = token_probs
                .iter()
                .enumerate()
                .map(|(idx, &prob)| (idx, prob))
                .collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let top_k_indices: Vec<usize> = indexed_probs
                .iter()
                .take(self.num_active_experts)
                .map(|(idx, _)| *idx)
                .collect();
            
            let top_k_weights: Vec<f32> = indexed_probs
                .iter()
                .take(self.num_active_experts)
                .map(|(_, prob)| *prob)
                .collect();
            
            // Normalize weights to sum to 1
            let weight_sum: f32 = top_k_weights.iter().sum();
            let normalized_weights: Vec<f32> = top_k_weights
                .iter()
                .map(|w| w / weight_sum)
                .collect();
            
            expert_indices.push(top_k_indices);
            expert_weights.push(normalized_weights);
        }
        
        // 4. Compute auxiliary losses
        let load_balance_loss = self.compute_load_balance_loss(&probs);
        let router_z_loss = self.compute_router_z_loss(&logits);
        let aux_loss = load_balance_loss * self.load_balance_weight
            + router_z_loss * self.router_z_loss_weight;
        
        // Cache for backward pass
        self.cached_input = Some(input.clone());
        self.cached_logits = Some(logits);
        self.cached_expert_indices = Some(expert_indices.clone());
        self.cached_expert_weights = Some(expert_weights.clone());
        
        (expert_indices, expert_weights, aux_loss)
    }
    
    /// Softmax activation
    fn softmax(&self, logits: &Array2<f32>) -> Array2<f32> {
        let (seq_len, num_experts) = logits.dim();
        let mut probs = Array2::zeros((seq_len, num_experts));
        
        for i in 0..seq_len {
            let row = logits.row(i);
            let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum();
            
            for j in 0..num_experts {
                probs[[i, j]] = (logits[[i, j]] - max_logit).exp() / exp_sum;
            }
        }
        
        probs
    }
    
    /// Compute load balance loss
    fn compute_load_balance_loss(&self, probs: &Array2<f32>) -> f32 {
        let (seq_len, num_experts) = probs.dim();
        
        // Average probability per expert across all tokens
        let mut expert_probs = Array1::zeros(num_experts);
        for j in 0..num_experts {
            expert_probs[j] = probs.column(j).mean().unwrap();
        }
        
        // Target: uniform distribution (1/num_experts)
        let target = 1.0 / (num_experts as f32);
        
        // L2 loss: sum((p_j - target)^2)
        expert_probs.iter().map(|&p| (p - target).powi(2)).sum()
    }
    
    /// Compute router z-loss (prevents logits from growing too large)
    fn compute_router_z_loss(&self, logits: &Array2<f32>) -> f32 {
        // Z-loss: log(sum(exp(logits)))^2
        let (seq_len, _) = logits.dim();
        let mut total_loss = 0.0;
        
        for i in 0..seq_len {
            let row = logits.row(i);
            let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = max_logit + row.iter().map(|&x| (x - max_logit).exp()).sum::<f32>().ln();
            total_loss += log_sum_exp.powi(2);
        }
        
        total_loss / (seq_len as f32)
    }
    
    /// Backward pass for router
    pub fn backward(&mut self, input: &Array2<f32>, lr: f32) {
        // Compute routing logits
        let logits = input.dot(&self.w_gate);
        
        // Compute gradients from auxiliary losses
        let probs = self.softmax(&logits);
        
        // Gradient approximation: push probabilities toward uniform distribution
        let (seq_len, num_experts) = probs.dim();
        let target_prob = 1.0 / (num_experts as f32);
        
        let mut grad_probs = Array2::zeros((seq_len, num_experts));
        for i in 0..seq_len {
            for j in 0..num_experts {
                grad_probs[[i, j]] = 2.0 * (probs[[i, j]] - target_prob);
            }
        }
        
        // Gradient w.r.t. w_gate
        let grad_w_gate = input.t().dot(&grad_probs);
        
        // Update parameters
        self.optimizer.step(&mut self.w_gate, &grad_w_gate, lr);
    }
}

/// Mixture-of-Experts layer for attention with hierarchical MoH
///
/// Combines a router with multiple attention experts, each using MoH.
/// Creates two-level adaptive routing: MoE → Experts → MoH → Heads
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AttentionMoELayer {
    /// Router for expert selection
    router: AttentionRouter,

    /// Attention experts (each with MoH)
    experts: Vec<AttentionExpert>,

    /// Embedding dimension
    embedding_dim: usize,

    /// Number of experts
    num_experts: usize,

    /// Number of active experts per token
    num_active_experts: usize,

    /// Cached auxiliary loss from last forward pass
    cached_aux_loss: f32,

    /// Cached input for backward pass
    cached_input: Option<Array2<f32>>,

    /// Cached expert outputs
    cached_expert_outputs: Option<Vec<Vec<Array2<f32>>>>,

    /// Cached routing decisions
    cached_routing_indices: Option<Vec<Vec<usize>>>,
    cached_routing_weights: Option<Vec<Vec<f32>>>,

    // === Per-Expert Tracking for Adaptive Learning ===
    /// Per-expert gradient norms (for Bidirectional LARS)
    #[serde(skip)]
    expert_grad_norms: Vec<f32>,

    /// Per-expert routing frequency (tokens routed to each expert)
    #[serde(skip)]
    expert_routing_counts: Vec<usize>,

    /// Per-expert total tokens processed (for load balancing)
    #[serde(skip)]
    expert_total_tokens: Vec<usize>,

    /// Per-expert adaptive learning rate scales
    #[serde(skip)]
    expert_lr_scales: Vec<f32>,
}

impl AttentionMoELayer {
    /// Create a new attention MoE layer
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Input/output dimension
    /// * `num_experts` - Total number of attention experts
    /// * `num_active_experts` - Number of experts to activate per token (k in top-k)
    /// * `num_shared_heads` - Number of shared heads per expert
    /// * `num_routed_heads` - Number of routed heads per expert
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `use_learned_threshold` - Enable learned threshold predictor for expert MoH
    /// * `load_balance_weight` - Weight for load balance loss
    /// * `router_z_loss_weight` - Weight for router z-loss
    pub fn new(
        embedding_dim: usize,
        num_experts: usize,
        num_active_experts: usize,
        num_shared_heads: usize,
        num_routed_heads: usize,
        num_kv_heads: usize,
        use_learned_threshold: bool,
        load_balance_weight: f32,
        router_z_loss_weight: f32,
    ) -> Self {
        // Create router
        let router = AttentionRouter::new(
            embedding_dim,
            num_experts,
            num_active_experts,
            load_balance_weight,
            router_z_loss_weight,
        );

        // Create attention experts (each with MoH)
        let experts: Vec<AttentionExpert> = (0..num_experts)
            .map(|idx| {
                AttentionExpert::new(
                    embedding_dim,
                    num_shared_heads,
                    num_routed_heads,
                    num_kv_heads,
                    use_learned_threshold,
                    idx,
                )
            })
            .collect();

        AttentionMoELayer {
            router,
            experts,
            embedding_dim,
            num_experts,
            num_active_experts,
            cached_aux_loss: 0.0,
            cached_input: None,
            cached_expert_outputs: None,
            cached_routing_indices: None,
            cached_routing_weights: None,
            // Initialize per-expert tracking
            expert_grad_norms: vec![0.0; num_experts],
            expert_routing_counts: vec![0; num_experts],
            expert_total_tokens: vec![0; num_experts],
            expert_lr_scales: vec![1.0; num_experts],
        }
    }

    /// Get the cached auxiliary loss from the last forward pass
    pub fn get_auxiliary_loss(&self) -> f32 {
        self.cached_aux_loss
    }

    /// Set epoch information for all experts (for warm-up and annealing)
    pub fn set_epoch_info(&mut self, current_epoch: usize, max_epochs: usize) {
        for expert in &mut self.experts {
            expert.set_epoch_info(current_epoch, max_epochs);
        }
    }

    /// Set number of active experts (for progressive sparsification)
    pub fn set_num_active_experts(&mut self, num_active: usize) {
        self.num_active_experts = num_active.min(self.num_experts);
        self.router.num_active_experts = self.num_active_experts;
    }

    /// Compute per-expert adaptive learning rates using Bidirectional LARS
    ///
    /// Similar to layer-wise LARS, but applied to experts:
    /// - High-gradient experts: Reduce LR to prevent over-updating
    /// - Low-gradient experts: Increase LR to prevent under-updating
    /// - Rarely-routed experts: Increase LR to accelerate learning
    ///
    /// This compensates for:
    /// 1. Gradient dilution from sparse expert activation
    /// 2. Uneven routing distribution across experts
    fn compute_expert_adaptive_lrs(&mut self, base_lr: f32) {
        // Target gradient norm (similar to layer-wise LARS)
        const TARGET_GRAD_NORM: f32 = 2.0;
        const POWER: f32 = 0.5; // Gentle adaptation (sqrt scaling)
        const EPSILON: f32 = 1e-6;

        // Compute average routing frequency
        let total_routed: usize = self.expert_routing_counts.iter().sum();
        let avg_routing_freq = if total_routed > 0 {
            total_routed as f32 / self.num_experts as f32
        } else {
            1.0
        };

        for expert_idx in 0..self.num_experts {
            let grad_norm = self.expert_grad_norms[expert_idx];
            let routing_count = self.expert_routing_counts[expert_idx];

            // Skip if no gradients
            if grad_norm < EPSILON {
                self.expert_lr_scales[expert_idx] = 1.0;
                continue;
            }

            // 1. Gradient-based scaling (Bidirectional LARS)
            let grad_scale = (TARGET_GRAD_NORM / (grad_norm + EPSILON)).powf(POWER);

            // 2. Routing-frequency-based scaling
            // Rarely-routed experts need higher LR to catch up
            let routing_freq = routing_count as f32;
            let routing_scale = if routing_freq < EPSILON {
                2.0 // Boost unrouted experts significantly
            } else {
                (avg_routing_freq / (routing_freq + EPSILON)).powf(0.3) // Gentle boost
            };

            // 3. Combine scales
            let combined_scale = grad_scale * routing_scale;

            // 4. Clamp to reasonable range
            self.expert_lr_scales[expert_idx] = combined_scale.clamp(0.5, 3.0);
        }
    }

    /// Reset per-batch tracking counters
    fn reset_batch_tracking(&mut self) {
        self.expert_routing_counts.fill(0);
    }

    /// Log per-expert statistics for monitoring
    pub fn log_expert_stats(&self, layer_idx: usize, epoch: usize) {
        // Aggregate MoH stats across experts
        let mut total_routed_heads = 0.0;
        let mut total_threshold = 0.0;
        let mut count = 0;

        for expert in &self.experts {
            let (avg_routed, mean_thresh, _, _, _, _, _) = expert.get_moh_stats();
            total_routed_heads += avg_routed;
            total_threshold += mean_thresh;
            count += 1;
        }

        let avg_routed_heads = if count > 0 { total_routed_heads / count as f32 } else { 0.0 };
        let avg_threshold = if count > 0 { total_threshold / count as f32 } else { 0.0 };

        // Log routing distribution
        let total_tokens: usize = self.expert_total_tokens.iter().sum();
        let routing_dist: Vec<String> = self.expert_total_tokens
            .iter()
            .enumerate()
            .map(|(idx, &count)| {
                let pct = if total_tokens > 0 {
                    (count as f32 / total_tokens as f32) * 100.0
                } else {
                    0.0
                };
                format!("E{}:{:.1}%", idx, pct)
            })
            .collect();

        // Log per-expert gradient norms and LR scales
        let grad_norms: Vec<String> = self.expert_grad_norms
            .iter()
            .enumerate()
            .map(|(idx, &norm)| format!("E{}:{:.2}", idx, norm))
            .collect();

        let lr_scales: Vec<String> = self.expert_lr_scales
            .iter()
            .enumerate()
            .map(|(idx, &scale)| format!("E{}:{:.2}x", idx, scale))
            .collect();

        tracing::info!(
            layer_idx = layer_idx,
            epoch = epoch,
            avg_routed_heads = format!("{:.2}h", avg_routed_heads),
            avg_threshold = format!("{:.2}p", avg_threshold),
            routing_dist = routing_dist.join(" | "),
            grad_norms = grad_norms.join(" | "),
            lr_scales = lr_scales.join(" | "),
            "AttentionMoE Stats"
        );
    }

    /// Get MoH statistics from all experts
    ///
    /// Returns Vec of (expert_idx, avg_routed_heads, threshold_p, conf_avg, conf_min, fallback_pct, complexity_avg, pred_norm)
    pub fn get_expert_moh_stats(&self) -> Vec<(usize, f32, f32, f32, f32, f32, f32, f32)> {
        self.experts
            .iter()
            .map(|expert| {
                let (avg_heads, threshold_p, conf_avg, conf_min, fallback_pct, complexity_avg, pred_norm) =
                    expert.get_moh_stats();
                (expert.expert_idx(), avg_heads, threshold_p, conf_avg, conf_min, fallback_pct, complexity_avg, pred_norm)
            })
            .collect()
    }

    /// Forward pass through attention MoE
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _) = input.dim();

        // 1. Route tokens to experts
        let (expert_indices, expert_weights, aux_loss) = self.router.route(input);
        self.cached_aux_loss = aux_loss;

        // Cache input and routing decisions for backward pass
        self.cached_input = Some(input.clone());
        self.cached_routing_indices = Some(expert_indices.clone());
        self.cached_routing_weights = Some(expert_weights.clone());

        // 2. Process each token through its selected experts
        let mut output = Array2::zeros((seq_len, self.embedding_dim));
        let mut all_expert_outputs = Vec::with_capacity(seq_len);

        for (token_idx, (indices, weights)) in expert_indices.iter().zip(expert_weights.iter()).enumerate() {
            let token_input = input.row(token_idx).to_owned().insert_axis(ndarray::Axis(0));
            let mut token_output = Array1::zeros(self.embedding_dim);
            let mut token_expert_outputs = Vec::new();

            for (expert_idx, weight) in indices.iter().zip(weights.iter()) {
                let expert = &mut self.experts[*expert_idx];
                let expert_output = expert.forward(&token_input);

                // Cache expert output for backward pass
                token_expert_outputs.push(expert_output.clone());

                // Add weighted expert contribution
                token_output = token_output + expert_output.row(0).to_owned() * *weight;
            }

            all_expert_outputs.push(token_expert_outputs);

            // Set output for this token
            for (i, &val) in token_output.iter().enumerate() {
                output[[token_idx, i]] = val;
            }
        }

        // Cache expert outputs for backward pass
        self.cached_expert_outputs = Some(all_expert_outputs);

        // 3. Add residual connection at MoE layer level
        output + input
    }

    /// Backward pass through attention MoE
    pub fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Get cached values from forward pass
        let cached_input = match &self.cached_input {
            Some(input) => input.clone(),
            None => return grads.clone(),
        };

        let routing_indices = match &self.cached_routing_indices {
            Some(indices) => indices.clone(),
            None => return grads.clone(),
        };

        let routing_weights = match &self.cached_routing_weights {
            Some(weights) => weights.clone(),
            None => return grads.clone(),
        };

        let (seq_len, embedding_dim) = grads.dim();

        // Initialize input gradients with residual connection gradient
        let mut input_grads = grads.clone();

        // Initialize expert gradient accumulator (separate from residual)
        let mut expert_grads = Array2::<f32>::zeros((seq_len, embedding_dim));

        // Reset per-batch tracking
        self.reset_batch_tracking();

        // Track per-expert gradients for adaptive LR computation
        let mut expert_grad_accumulators: Vec<f32> = vec![0.0; self.num_experts];

        // Backpropagate through each token's selected experts
        for (token_idx, (indices, weights)) in routing_indices.iter().zip(routing_weights.iter()).enumerate() {
            let token_grad = grads.row(token_idx).to_owned().insert_axis(ndarray::Axis(0));

            // Backpropagate to each active expert
            for (expert_idx, weight) in indices.iter().zip(weights.iter()) {
                // Track routing for this expert
                self.expert_routing_counts[*expert_idx] += 1;

                // Scale gradient by routing weight
                let weighted_grad = &token_grad * *weight;

                // Accumulate gradient norm for this expert
                let grad_norm: f32 = weighted_grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
                expert_grad_accumulators[*expert_idx] += grad_norm;

                // Apply adaptive learning rate (Bidirectional LARS for experts)
                let adaptive_scale = self.expert_lr_scales[*expert_idx];
                let expert_lr = lr * adaptive_scale;

                // Backpropagate through expert (expert updates its own parameters)
                let expert = &mut self.experts[*expert_idx];
                let expert_input_grad = expert.backward(&weighted_grad, expert_lr);

                // Accumulate expert's contribution to expert gradient
                for (i, &grad) in expert_input_grad.row(0).iter().enumerate() {
                    expert_grads[[token_idx, i]] += grad;
                }
            }
        }

        // Update per-expert gradient norms for adaptive LR
        for expert_idx in 0..self.num_experts {
            self.expert_grad_norms[expert_idx] = expert_grad_accumulators[expert_idx];
            self.expert_total_tokens[expert_idx] += self.expert_routing_counts[expert_idx];
        }

        // Compute adaptive learning rates for next iteration
        self.compute_expert_adaptive_lrs(lr);

        // Add expert gradients to input gradients
        input_grads = input_grads + expert_grads;

        // Update router parameters
        self.router.backward(&cached_input, lr);

        input_grads
    }
}


