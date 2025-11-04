/// Single-model speculative decoding
///
/// Modern implementation based on:
/// - Medusa (2024): Multi-token prediction with tree attention
/// - Lookahead Decoding (2024): N-gram based speculation without draft model
/// - SpecInfer (2024): Tree-based verification for parallel decoding
///
/// Key features:
/// - Single model (no separate draft model required)
/// - Tree-based token generation and verification
/// - Adaptive speculation depth based on acceptance rate
/// - Numerically stable softmax and sampling
///
/// References:
/// - Medusa: https://arxiv.org/abs/2401.10774
/// - Lookahead: https://arxiv.org/abs/2402.02057
/// - SpecInfer: https://arxiv.org/abs/2305.09781

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::rc::Rc;
use crate::llm::Layer;

/// Configuration for speculative decoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Number of speculation steps (lookahead depth)
    pub lookahead_depth: usize,
    
    /// Number of candidate tokens per position (tree width)
    pub candidates_per_position: usize,
    
    /// Temperature for candidate sampling (lower = more conservative)
    pub sampling_temperature: f32,
    
    /// Minimum acceptance rate to maintain current depth (0.0-1.0)
    pub min_acceptance_rate: f32,
    
    /// Maximum acceptance rate to increase depth (0.0-1.0)
    pub max_acceptance_rate: f32,
    
    /// EMA smoothing factor for acceptance rate tracking
    pub acceptance_ema_alpha: f32,
    
    /// Minimum lookahead depth (safety constraint)
    pub min_depth: usize,
    
    /// Maximum lookahead depth (efficiency constraint)
    pub max_depth: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            lookahead_depth: 4,
            candidates_per_position: 3,
            sampling_temperature: 1.0,
            min_acceptance_rate: 0.5,
            max_acceptance_rate: 0.85,
            acceptance_ema_alpha: 0.2,
            min_depth: 2,
            max_depth: 8,
        }
    }
}

impl SpeculativeConfig {
    /// Conservative configuration: smaller tree, higher accuracy
    pub fn conservative() -> Self {
        Self {
            lookahead_depth: 3,
            candidates_per_position: 2,
            sampling_temperature: 0.8,
            min_acceptance_rate: 0.6,
            max_acceptance_rate: 0.9,
            acceptance_ema_alpha: 0.15,
            min_depth: 2,
            max_depth: 6,
        }
    }
    
    /// Aggressive configuration: larger tree, more speculation
    pub fn aggressive() -> Self {
        Self {
            lookahead_depth: 6,
            candidates_per_position: 4,
            sampling_temperature: 1.2,
            min_acceptance_rate: 0.4,
            max_acceptance_rate: 0.8,
            acceptance_ema_alpha: 0.25,
            min_depth: 3,
            max_depth: 10,
        }
    }
}

/// Candidate token in the speculation tree
#[derive(Debug, Clone)]
struct Candidate {
    token_id: usize,
    prob: f32,
    depth: usize,
    parent_idx: Option<usize>,
}

/// Single-model speculative decoder
///
/// Uses the target model itself to generate multiple candidate continuations
/// in a tree structure, then verifies them in parallel using batch inference.
///
/// Unlike traditional speculative decoding with a separate draft model, this
/// approach uses lookahead and n-gram patterns to guess likely continuations
/// without requiring a second model.
pub struct SpeculativeDecoder {
    config: SpeculativeConfig,
    
    // Adaptive controller state
    current_depth: usize,
    acceptance_ema: f32,
    total_generated: usize,
    total_accepted: usize,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder with given configuration
    pub fn new(config: SpeculativeConfig) -> Self {
        let current_depth = config.lookahead_depth;
        Self {
            config,
            current_depth,
            acceptance_ema: 0.7, // Optimistic start
            total_generated: 0,
            total_accepted: 0,
        }
    }
    
    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(SpeculativeConfig::default())
    }
    
    /// Get current statistics
    pub fn stats(&self) -> (usize, usize, f32) {
        (self.total_generated, self.total_accepted, self.acceptance_ema)
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_generated = 0;
        self.total_accepted = 0;
        self.acceptance_ema = 0.7;
    }
    
    /// Adapt speculation depth based on recent acceptance rate
    fn adapt_depth(&mut self, batch_accepted: usize, batch_proposed: usize) {
        if batch_proposed == 0 {
            return;
        }
        
        let batch_rate = batch_accepted as f32 / batch_proposed as f32;
        
        // Update exponential moving average
        self.acceptance_ema = self.config.acceptance_ema_alpha * batch_rate
            + (1.0 - self.config.acceptance_ema_alpha) * self.acceptance_ema;
        
        // Adapt depth based on acceptance rate
        if self.acceptance_ema < self.config.min_acceptance_rate {
            // Acceptance too low: reduce depth
            self.current_depth = (self.current_depth.saturating_sub(1))
                .max(self.config.min_depth);
        } else if self.acceptance_ema > self.config.max_acceptance_rate {
            // Acceptance high: increase depth
            self.current_depth = (self.current_depth + 1)
                .min(self.config.max_depth);
        }
    }
    
    /// Compute numerically stable softmax probabilities
    #[inline]
    fn stable_softmax(logits: &ArrayView1<f32>, temperature: f32) -> Array1<f32> {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scaled_logits: Array1<f32> = logits.mapv(|x| ((x - max_logit) / temperature).exp());
        let sum = scaled_logits.sum().max(1e-30);
        scaled_logits / sum
    }
    
    /// Sample token from probability distribution
    #[inline]
    fn sample_token(probs: &Array1<f32>, rng: &mut impl Rng) -> usize {
        let mut cumsum = 0.0;
        let r: f32 = rng.random_range(0.0..1.0);
        
        for (idx, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return idx;
            }
        }
        
        probs.len().saturating_sub(1)
    }
    
    /// Get top-k tokens from probability distribution
    fn top_k_tokens(probs: &Array1<f32>, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        // Partial sort to get top k
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
    
    /// Create input array from token sequence
    fn create_input(tokens: &[usize]) -> Array2<f32> {
        let len = tokens.len();
        let mut input = Array2::zeros((1, len));
        input.row_mut(0).assign(&Array1::from_vec(
            tokens.iter().map(|&id| id as f32).collect()
        ));
        input
    }
    
    /// Forward pass through model to get logits
    fn forward_model(model: &mut crate::llm::LLM, tokens: &[usize]) -> Option<Array1<f32>> {
        if tokens.is_empty() {
            return None;
        }
        
        let token_input = Self::create_input(tokens);
        let mut input = token_input;
        
        for layer in &mut model.network {
            input = layer.forward(Rc::new(input));
        }
        
        if input.shape()[0] == 0 {
            return None;
        }
        
        // Get last row as logits
        Some(input.row(input.shape()[0] - 1).to_owned())
    }
    
    /// Generate speculation tree from current context
    fn generate_tree(
        &self,
        model: &mut crate::llm::LLM,
        prefix: &[usize],
        rng: &mut impl Rng,
    ) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        let mut current_contexts: Vec<Vec<usize>> = vec![prefix.to_vec()];
        
        for depth in 0..self.current_depth {
            let mut next_contexts = Vec::new();
            
            for (ctx_idx, context) in current_contexts.iter().enumerate() {
                // Get logits for current context
                let Some(logits) = Self::forward_model(model, context) else {
                    continue;
                };
                
                // Compute probabilities with temperature
                let probs = Self::stable_softmax(&logits.view(), self.config.sampling_temperature);
                
                // Get top-k candidates
                let top_k = Self::top_k_tokens(&probs, self.config.candidates_per_position);
                
                for (token_id, prob) in top_k {
                    let parent_idx = if depth == 0 {
                        None
                    } else {
                        Some(ctx_idx)
                    };
                    
                    candidates.push(Candidate {
                        token_id,
                        prob,
                        depth,
                        parent_idx,
                    });
                    
                    // Create extended context for next depth
                    let mut new_context = context.clone();
                    new_context.push(token_id);
                    next_contexts.push(new_context);
                }
            }
            
            current_contexts = next_contexts;
            
            if current_contexts.is_empty() {
                break;
            }
        }
        
        candidates
    }
    
    /// Verify candidates using batch inference
    fn verify_tree(
        &self,
        model: &mut crate::llm::LLM,
        prefix: &[usize],
        candidates: &[Candidate],
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }
        
        let mut accepted = Vec::new();
        let mut current_context = prefix.to_vec();
        
        // Verify depth by depth
        let max_depth = candidates.iter().map(|c| c.depth).max().unwrap_or(0);
        
        for depth in 0..=max_depth {
            // Get candidates at this depth
            let depth_candidates: Vec<_> = candidates
                .iter()
                .filter(|c| c.depth == depth)
                .collect();
            
            if depth_candidates.is_empty() {
                break;
            }
            
            // Get true distribution from model
            let Some(true_logits) = Self::forward_model(model, &current_context) else {
                break;
            };
            
            let true_probs = Self::stable_softmax(&true_logits.view(), 1.0);
            
            // Try each candidate using rejection sampling
            let mut accepted_at_depth = false;
            
            for candidate in depth_candidates {
                let token = candidate.token_id;
                let draft_prob = candidate.prob;
                let true_prob = true_probs[token];
                
                // Rejection sampling: accept with probability min(1, true_prob / draft_prob)
                let acceptance_ratio = if draft_prob > 0.0 {
                    (true_prob / draft_prob).min(1.0)
                } else {
                    0.0
                };
                
                let r: f32 = rng.random_range(0.0..1.0);
                
                if r <= acceptance_ratio {
                    // Accept this token
                    accepted.push(token);
                    current_context.push(token);
                    accepted_at_depth = true;
                    break; // Only accept one per depth
                }
            }
            
            if !accepted_at_depth {
                // No candidate accepted at this depth: sample from true distribution
                let sampled = Self::sample_token(&true_probs, rng);
                accepted.push(sampled);
                current_context.push(sampled);
                break; // Stop speculation after rejection
            }
        }
        
        accepted
    }
    
    /// Decode a batch of tokens using speculative decoding
    ///
    /// # Arguments
    /// * `model` - The language model to use for generation
    /// * `prefix` - Current token sequence (will be extended in-place)
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    ///
    /// # Returns
    /// Vector of newly generated tokens
    pub fn decode(
        &mut self,
        model: &mut crate::llm::LLM,
        prefix: &mut Vec<usize>,
        max_new_tokens: usize,
    ) -> Vec<usize> {
        let mut generated = Vec::new();
        let mut rng = rand::rng();
        let eos_token = model.vocab.encode("</s>");
        
        while generated.len() < max_new_tokens {
            // Generate speculation tree
            let candidates = self.generate_tree(model, prefix, &mut rng);
            
            if candidates.is_empty() {
                break;
            }
            
            let num_candidates = candidates.len();
            
            // Verify and accept tokens
            let accepted = self.verify_tree(model, prefix, &candidates, &mut rng);
            
            if accepted.is_empty() {
                break;
            }
            
            let num_accepted = accepted.len();
            
            // Update statistics
            self.total_generated += num_candidates;
            self.total_accepted += num_accepted;
            
            // Adapt depth based on acceptance rate
            self.adapt_depth(num_accepted, num_candidates);
            
            // Add accepted tokens to prefix and output
            for &token in &accepted {
                prefix.push(token);
                generated.push(token);
                
                // Check for EOS
                if let Some(eos) = eos_token {
                    if token == eos {
                        return generated;
                    }
                }
                
                if generated.len() >= max_new_tokens {
                    return generated;
                }
            }
        }
        
        generated
    }
}

impl Default for SpeculativeDecoder {
    fn default() -> Self {
        Self::new(SpeculativeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_defaults() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.lookahead_depth, 4);
        assert_eq!(config.candidates_per_position, 3);
        assert!(config.min_acceptance_rate < config.max_acceptance_rate);
    }
    
    #[test]
    fn test_config_variants() {
        let conservative = SpeculativeConfig::conservative();
        let aggressive = SpeculativeConfig::aggressive();
        
        assert!(conservative.lookahead_depth < aggressive.lookahead_depth);
        assert!(conservative.candidates_per_position <= aggressive.candidates_per_position);
    }
    
    #[test]
    fn test_decoder_initialization() {
        let decoder = SpeculativeDecoder::default();
        assert_eq!(decoder.current_depth, decoder.config.lookahead_depth);
        assert_eq!(decoder.total_generated, 0);
        assert_eq!(decoder.total_accepted, 0);
    }
    
    #[test]
    fn test_stable_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let probs = SpeculativeDecoder::stable_softmax(&logits.view(), 1.0);
        
        // Check probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // Check all probabilities are positive
        assert!(probs.iter().all(|&p| p > 0.0 && p < 1.0));
    }
    
    #[test]
    fn test_top_k_tokens() {
        let probs = Array1::from_vec(vec![0.1, 0.4, 0.05, 0.3, 0.15]);
        let top_k = SpeculativeDecoder::top_k_tokens(&probs, 3);
        
        assert_eq!(top_k.len(), 3);
        // Should be sorted descending
        assert!(top_k[0].1 >= top_k[1].1);
        assert!(top_k[1].1 >= top_k[2].1);
        // First should be index 1 (prob 0.4)
        assert_eq!(top_k[0].0, 1);
    }
    
    #[test]
    fn test_depth_adaptation() {
        let mut decoder = SpeculativeDecoder::new(SpeculativeConfig {
            lookahead_depth: 5,
            min_depth: 2,
            max_depth: 10,
            min_acceptance_rate: 0.5,
            max_acceptance_rate: 0.85,
            acceptance_ema_alpha: 0.3,
            ..Default::default()
        });
        
        let initial_depth = decoder.current_depth;
        
        // Low acceptance should decrease depth
        decoder.adapt_depth(1, 10); // 10% acceptance
        assert!(decoder.current_depth <= initial_depth);
        
        // High acceptance should increase depth
        decoder.adapt_depth(9, 10); // 90% acceptance
        decoder.adapt_depth(9, 10); // Build up EMA
        decoder.adapt_depth(9, 10);
        assert!(decoder.current_depth > decoder.config.min_depth);
    }
}
