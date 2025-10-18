//! Mixture of Experts (MoE) Layer
//!
//! Implements sparse MoE architecture with top-k routing for efficient scaling.
//! Based on Switch Transformers (Fedus et al., 2021) and Mixtral (Mistral AI, 2024).
//!
//! # Architecture
//!
//! ```text
//! Input → Router (Top-K) → Expert Networks → Weighted Sum → Output
//! ```
//!
//! # Key Features
//!
//! - **Sparse Activation**: Only k out of N experts active per token
//! - **Load Balancing**: Auxiliary loss prevents routing collapse
//! - **Router Z-Loss**: Stabilizes routing logits
//! - **Parameter Efficient**: Maintains total param count via smaller experts
//!
//! # References
//!
//! - Fedus et al., "Switch Transformers", 2021
//! - Jiang et al., "Mixtral of Experts", 2024
//! - Zhou et al., "Expert Choice Routing", 2022
//! - Zoph et al., "ST-MoE", 2022

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;

/// Simple expert network (SwiGLU without residual connection)
///
/// This is used within MoE layers where residual connections are handled
/// at the MoE layer level, not within individual experts.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Expert {
    w1: Array2<f32>,
    w2: Array2<f32>,
    w3: Array2<f32>,
    optimizer_w1: Adam,
    optimizer_w2: Adam,
    optimizer_w3: Adam,
    cached_input: Option<Array2<f32>>,
    cached_x1: Option<Array2<f32>>,
    cached_x2: Option<Array2<f32>>,
    cached_swish: Option<Array2<f32>>,
    cached_gated: Option<Array2<f32>>,
}

impl Expert {
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();
        // Use smaller initialization for experts since multiple experts contribute
        // Scale by 1/(2*d) to account for top-2 routing and prevent gradient explosion
        let std = (0.5 / input_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let w1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| normal.sample(&mut rng));
        let w2 = Array2::from_shape_fn((input_dim, hidden_dim), |_| normal.sample(&mut rng));
        let w3 = Array2::from_shape_fn((hidden_dim, input_dim), |_| normal.sample(&mut rng));

        Expert {
            w1: w1.clone(),
            w2: w2.clone(),
            w3: w3.clone(),
            optimizer_w1: Adam::new(w1.dim()),
            optimizer_w2: Adam::new(w2.dim()),
            optimizer_w3: Adam::new(w3.dim()),
            cached_input: None,
            cached_x1: None,
            cached_x2: None,
            cached_swish: None,
            cached_gated: None,
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Cache input for backward pass
        self.cached_input = Some(input.clone());

        // Compute xW₁ and xW₂
        let x1 = input.dot(&self.w1);
        let x2 = input.dot(&self.w2);

        // Apply Swish to x1: swish(x) = x * sigmoid(x)
        let swish = x1.mapv(|x| x / (1.0 + (-x).exp()));

        // Gate: swish(xW₁) ⊙ xW₂
        let gated = &swish * &x2;

        // Project back: (swish(xW₁) ⊙ xW₂)W₃
        let output = gated.dot(&self.w3);

        // Cache intermediate values for backward pass
        self.cached_x1 = Some(x1);
        self.cached_x2 = Some(x2);
        self.cached_swish = Some(swish);
        self.cached_gated = Some(gated);

        // NO residual connection - MoE layer handles this
        output
    }

    fn backward(&mut self, output_grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let x1 = self.cached_x1.as_ref().unwrap();
        let x2 = self.cached_x2.as_ref().unwrap();
        let swish = self.cached_swish.as_ref().unwrap();
        let gated = self.cached_gated.as_ref().unwrap();

        // Gradient w.r.t. gated
        let grad_gated = output_grads.dot(&self.w3.t());

        // Gradient w.r.t. W₃
        let grad_w3 = gated.t().dot(output_grads);

        // Gradient w.r.t. swish and x2
        let grad_swish = &grad_gated * x2;
        let grad_x2 = &grad_gated * swish;

        // Gradient w.r.t. x1 (through swish activation)
        let sigmoid = x1.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let grad_x1 = &grad_swish * &sigmoid * &(1.0 + x1 * &(1.0 - &sigmoid));

        // Gradient w.r.t. W₁ and W₂
        let grad_w1 = input.t().dot(&grad_x1);
        let grad_w2 = input.t().dot(&grad_x2);

        // Gradient w.r.t. input
        let grad_input = grad_x1.dot(&self.w1.t()) + grad_x2.dot(&self.w2.t());

        // Update parameters
        self.optimizer_w1.step(&mut self.w1, &grad_w1, lr);
        self.optimizer_w2.step(&mut self.w2, &grad_w2, lr);
        self.optimizer_w3.step(&mut self.w3, &grad_w3, lr);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.w1.len() + self.w2.len() + self.w3.len()
    }
}

/// Router network for expert selection
///
/// Learns to route tokens to appropriate experts using a gating mechanism.
/// Implements top-k routing with load balancing and router z-loss.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Router {
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
    
    /// Cached values for backward pass
    cached_input: Option<Array2<f32>>,
    cached_logits: Option<Array2<f32>>,
    cached_expert_indices: Option<Vec<Vec<usize>>>,
    cached_expert_weights: Option<Vec<Vec<f32>>>,
}

impl Router {
    /// Create a new router
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Input dimension
    /// * `num_experts` - Total number of experts
    /// * `num_active_experts` - Number of experts to activate per token (k in top-k)
    /// * `load_balance_weight` - Weight for load balance loss (typically 0.01)
    /// * `router_z_loss_weight` - Weight for router z-loss (typically 0.001)
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
        
        Router {
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
    
    /// Route tokens to experts using top-k selection
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (seq_len, embedding_dim)
    ///
    /// # Returns
    ///
    /// Tuple of (expert_indices, expert_weights, auxiliary_loss) where:
    /// - expert_indices: Vec of length seq_len, each containing k expert indices
    /// - expert_weights: Vec of length seq_len, each containing k normalized weights
    /// - auxiliary_loss: Combined load balance + router z-loss
    pub fn route(
        &mut self,
        input: &Array2<f32>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, f32) {
        let (seq_len, _) = input.dim();
        
        // 1. Compute routing logits: input @ w_gate
        let logits = input.dot(&self.w_gate); // (seq_len, num_experts)
        
        // 2. Top-k selection per token
        let mut expert_indices = Vec::with_capacity(seq_len);
        let mut expert_weights = Vec::with_capacity(seq_len);
        
        for token_idx in 0..seq_len {
            let token_logits = logits.row(token_idx);
            let (indices, weights) = self.top_k_routing(&token_logits);
            expert_indices.push(indices);
            expert_weights.push(weights);
        }
        
        // 3. Compute auxiliary losses
        let load_balance_loss = self.compute_load_balance_loss(&logits, &expert_indices);
        let router_z_loss = self.compute_router_z_loss(&logits);
        let auxiliary_loss = load_balance_loss + router_z_loss;
        
        // Cache for backward pass
        self.cached_input = Some(input.clone());
        self.cached_logits = Some(logits);
        self.cached_expert_indices = Some(expert_indices.clone());
        self.cached_expert_weights = Some(expert_weights.clone());
        
        (expert_indices, expert_weights, auxiliary_loss)
    }
    
    /// Top-k routing for a single token
    ///
    /// Selects top-k experts and computes normalized weights via softmax.
    fn top_k_routing(&self, logits: &ndarray::ArrayView1<f32>) -> (Vec<usize>, Vec<f32>) {
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        // Sort by logit value (descending)
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k
        let top_k: Vec<(usize, f32)> = indexed_logits
            .into_iter()
            .take(self.num_active_experts)
            .collect();
        
        // Extract indices and logits
        let indices: Vec<usize> = top_k.iter().map(|(i, _)| *i).collect();
        let top_logits: Vec<f32> = top_k.iter().map(|(_, v)| *v).collect();
        
        // Softmax over top-k logits
        let max_logit = top_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = top_logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let weights: Vec<f32> = exp_logits.iter().map(|&v| v / sum_exp).collect();
        
        (indices, weights)
    }
    
    /// Compute load balance loss
    ///
    /// Encourages uniform expert utilization across tokens.
    /// Formula: L_balance = α × Σ(i=1 to N) f_i × P_i
    /// where f_i = fraction of tokens routed to expert i
    ///       P_i = average routing probability for expert i
    fn compute_load_balance_loss(
        &self,
        logits: &Array2<f32>,
        expert_indices: &[Vec<usize>],
    ) -> f32 {
        let (seq_len, _) = logits.dim();
        
        // Compute routing probabilities via softmax
        let probs = self.softmax(logits);
        
        // Compute P_i: average routing probability for each expert
        let avg_probs: Vec<f32> = (0..self.num_experts)
            .map(|expert_idx| {
                probs.column(expert_idx).mean().unwrap_or(0.0)
            })
            .collect();
        
        // Compute f_i: fraction of tokens routed to each expert
        let mut expert_counts = vec![0.0; self.num_experts];
        for indices in expert_indices {
            for &expert_idx in indices {
                expert_counts[expert_idx] += 1.0;
            }
        }
        let fractions: Vec<f32> = expert_counts
            .iter()
            .map(|&count| count / (seq_len as f32))
            .collect();
        
        // Compute loss: Σ(f_i × P_i)
        let loss: f32 = fractions
            .iter()
            .zip(avg_probs.iter())
            .map(|(&f, &p)| f * p)
            .sum();
        
        self.load_balance_weight * loss * (self.num_experts as f32)
    }
    
    /// Compute router z-loss
    ///
    /// Prevents routing logits from growing too large.
    /// Formula: L_z = β × Σ(tokens) log²(Σ(experts) exp(logit_i))
    fn compute_router_z_loss(&self, logits: &Array2<f32>) -> f32 {
        let (seq_len, _) = logits.dim();
        
        let mut total_loss = 0.0;
        for token_idx in 0..seq_len {
            let token_logits = logits.row(token_idx);
            let max_logit = token_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = (token_logits.iter().map(|&v| (v - max_logit).exp()).sum::<f32>()).ln() + max_logit;
            total_loss += log_sum_exp * log_sum_exp;
        }
        
        self.router_z_loss_weight * total_loss / (seq_len as f32)
    }
    
    /// Softmax over experts dimension
    fn softmax(&self, logits: &Array2<f32>) -> Array2<f32> {
        let (seq_len, num_experts) = logits.dim();
        let mut probs = Array2::zeros((seq_len, num_experts));
        
        for token_idx in 0..seq_len {
            let token_logits = logits.row(token_idx);
            let max_logit = token_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = token_logits.iter().map(|&v| (v - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            
            for (expert_idx, &exp_val) in exp_logits.iter().enumerate() {
                probs[[token_idx, expert_idx]] = exp_val / sum_exp;
            }
        }
        
        probs
    }
    
    /// Get number of parameters
    pub fn parameters(&self) -> usize {
        self.w_gate.len()
    }
    
    /// Apply gradients to router weights
    pub fn apply_gradients(&mut self, grad_w_gate: &Array2<f32>, lr: f32) {
        self.optimizer.step(&mut self.w_gate, grad_w_gate, lr);
    }

    /// Backward pass for router
    ///
    /// Updates router parameters based on cached routing decisions and auxiliary losses.
    /// Uses a simplified gradient estimation approach.
    pub fn backward(&mut self, input: &Array2<f32>, lr: f32) {
        // Compute routing logits
        let logits = input.dot(&self.w_gate);

        // Compute gradients from auxiliary losses
        // For simplicity, we use the gradient of the softmax output
        // In practice, this would include policy gradient or REINFORCE
        let probs = self.softmax(&logits);

        // Gradient approximation: push probabilities toward uniform distribution
        // This encourages load balancing
        let (seq_len, num_experts) = probs.dim();
        let target_prob = 1.0 / (num_experts as f32);

        let mut grad_probs = Array2::zeros((seq_len, num_experts));
        for i in 0..seq_len {
            for j in 0..num_experts {
                grad_probs[[i, j]] = (probs[[i, j]] - target_prob) * self.load_balance_weight;
            }
        }

        // Backpropagate through softmax (simplified)
        let grad_logits = &grad_probs * 0.1; // Scale down to prevent instability

        // Compute weight gradients
        let grad_w_gate = input.t().dot(&grad_logits);

        // Apply gradients
        self.apply_gradients(&grad_w_gate, lr);
    }
}

/// Mixture of Experts Layer
///
/// Combines a router with multiple expert networks for sparse computation.
/// Each token is routed to top-k experts, and their outputs are combined.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MoELayer {
    /// Router for expert selection
    router: Router,

    /// Expert networks (each is a SwiGLU-style network without residual)
    experts: Vec<Expert>,

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

    /// Cached expert outputs (before weighting) for backward pass
    cached_expert_outputs: Option<Vec<Vec<Array2<f32>>>>,

    /// Cached routing decisions for backward pass
    cached_routing_indices: Option<Vec<Vec<usize>>>,
    cached_routing_weights: Option<Vec<Vec<f32>>>,
}

impl MoELayer {
    /// Create a new MoE layer
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Input/output dimension
    /// * `expert_hidden_dim` - Hidden dimension for each expert
    /// * `num_experts` - Total number of experts
    /// * `num_active_experts` - Number of experts to activate per token (k in top-k)
    /// * `load_balance_weight` - Weight for load balance loss (typically 0.01)
    /// * `router_z_loss_weight` - Weight for router z-loss (typically 0.001)
    ///
    /// # Returns
    ///
    /// A new MoE layer with initialized router and experts
    pub fn new(
        embedding_dim: usize,
        expert_hidden_dim: usize,
        num_experts: usize,
        num_active_experts: usize,
        load_balance_weight: f32,
        router_z_loss_weight: f32,
    ) -> Self {
        // Create router
        let router = Router::new(
            embedding_dim,
            num_experts,
            num_active_experts,
            load_balance_weight,
            router_z_loss_weight,
        );

        // Create experts (each is a SwiGLU-style network without residual)
        let experts: Vec<Expert> = (0..num_experts)
            .map(|_| Expert::new(embedding_dim, expert_hidden_dim))
            .collect();

        MoELayer {
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
        }
    }

    /// Get the cached auxiliary loss from the last forward pass
    pub fn get_auxiliary_loss(&self) -> f32 {
        self.cached_aux_loss
    }

    /// Get expert utilization statistics
    ///
    /// Returns a vector of length num_experts with the fraction of tokens
    /// routed to each expert in the last forward pass.
    pub fn get_expert_utilization(&self) -> Vec<f32> {
        // This would require caching routing decisions
        // For now, return uniform distribution as placeholder
        vec![1.0 / self.num_experts as f32; self.num_experts]
    }
}

impl Layer for MoELayer {
    fn layer_type(&self) -> &str {
        "MoE"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _) = input.dim();

        // 1. Route tokens to experts
        let (expert_indices, expert_weights, aux_loss) = self.router.route(input);
        self.cached_aux_loss = aux_loss;

        // Cache input and routing decisions for backward pass
        self.cached_input = Some(input.clone());
        self.cached_routing_indices = Some(expert_indices.clone());
        self.cached_routing_weights = Some(expert_weights.clone());

        // 2. Process each token through its selected experts
        // Experts do NOT include residual connections, so we can simply weight and sum
        let mut output = Array2::zeros((seq_len, self.embedding_dim));
        let mut all_expert_outputs = Vec::with_capacity(seq_len);

        for token_idx in 0..seq_len {
            let token_input = input.row(token_idx).to_owned().insert_axis(Axis(0));
            let indices = &expert_indices[token_idx];
            let weights = &expert_weights[token_idx];

            // Store expert outputs for this token
            let mut token_expert_outputs = Vec::with_capacity(self.num_active_experts);

            // Compute weighted sum of expert outputs
            let mut token_output = Array1::zeros(self.embedding_dim);

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

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
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

        // Backpropagate through each token's expert routing
        for token_idx in 0..seq_len {
            let token_grad = grads.row(token_idx).to_owned().insert_axis(Axis(0));
            let _token_input = cached_input.row(token_idx).to_owned().insert_axis(Axis(0));
            let indices = &routing_indices[token_idx];
            let weights = &routing_weights[token_idx];

            // Backpropagate to each active expert
            for (expert_idx, weight) in indices.iter().zip(weights.iter()) {
                // Scale gradient by routing weight for both parameter updates and input gradient
                let weighted_grad = &token_grad * *weight;

                // Use reduced learning rate for experts to prevent gradient explosion
                // Experts receive gradients from multiple tokens, so we scale down
                let expert_lr = lr * 0.5;

                // Backpropagate through expert (expert updates its own parameters)
                let expert = &mut self.experts[*expert_idx];
                let expert_input_grad = expert.backward(&weighted_grad, expert_lr);

                // Accumulate expert's contribution to expert gradient
                // The gradient is already weighted, so just add it
                for (i, &grad) in expert_input_grad.row(0).iter().enumerate() {
                    expert_grads[[token_idx, i]] += grad;
                }
            }
        }

        // Add expert gradients to input gradients
        // Forward: output = input + weighted_expert_outputs
        // Backward: grad_input = grad_output (residual) + grad_expert_outputs
        // Note: expert_grads already accounts for routing weights, so no additional scaling needed
        input_grads = input_grads + expert_grads;

        // Update router parameters
        // Router gradient is computed from routing decisions and expert performance
        // For now, we use a simplified approach: update router based on auxiliary loss
        self.router.backward(&cached_input, lr);

        input_grads
    }

    fn parameters(&self) -> usize {
        let router_params = self.router.parameters();
        let expert_params: usize = self.experts.iter().map(|e| e.parameters()).sum();
        router_params + expert_params
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Get cached values from forward pass
        let cached_input = match &self.cached_input {
            Some(input) => input,
            None => {
                // Fallback if no cached input
                return (output_grads.clone(), Vec::new());
            }
        };

        let routing_indices = match &self.cached_routing_indices {
            Some(indices) => indices,
            None => return (output_grads.clone(), Vec::new()),
        };

        let routing_weights = match &self.cached_routing_weights {
            Some(weights) => weights,
            None => return (output_grads.clone(), Vec::new()),
        };

        let (seq_len, _embedding_dim) = output_grads.dim();

        // Initialize input gradients (includes residual connection gradient)
        let mut input_grads = output_grads.clone();

        // Backpropagate through each token's expert routing
        for token_idx in 0..seq_len {
            let token_grad = output_grads.row(token_idx).to_owned().insert_axis(Axis(0));
            let _token_input = cached_input.row(token_idx).to_owned().insert_axis(Axis(0));
            let indices = &routing_indices[token_idx];
            let weights = &routing_weights[token_idx];

            // Backpropagate to each active expert
            for (expert_idx, weight) in indices.iter().zip(weights.iter()) {
                // Scale gradient by routing weight
                let weighted_grad = &token_grad * *weight;

                // Note: compute_gradients is not used for MoE
                // Experts update themselves in backward() method
                // For compute_gradients, we just pass through the gradient
            }
        }

        // No parameter gradients returned (experts and router update themselves)
        (input_grads, Vec::new())
    }

    fn apply_gradients(
        &mut self,
        _param_grads: &[Array2<f32>],
        _lr: f32,
    ) -> crate::errors::Result<()> {
        // Gradients are applied within backward() for now
        Ok(())
    }
}

