/// Greedy decoding (argmax token selection)
///
/// Simple but effective decoding strategy that always selects the most likely token
/// at each step. Fast and deterministic, suitable for many applications.
///
/// Advantages:
/// - Fast: O(1) per token (just argmax)
/// - Deterministic: same input always produces same output
/// - Memory efficient: no beam tracking
///
/// Disadvantages:
/// - No exploration: can miss better sequences
/// - Prone to repetition: especially with poorly calibrated models
/// - No backtracking: greedy choices are permanent

use ndarray::{Array1, Array2, ArrayView1};
use std::rc::Rc;
use crate::llm::Layer;

/// Greedy decoder with optional temperature and top-k/top-p filtering
pub struct GreedyDecoder {
    /// Temperature for softmax (1.0 = no change, <1.0 = sharper, >1.0 = smoother)
    pub temperature: f32,
    
    /// Top-k filtering: only consider top k tokens (None = no filtering)
    pub top_k: Option<usize>,
    
    /// Top-p (nucleus) filtering: only consider tokens with cumulative prob <= p
    pub top_p: Option<f32>,
    
    /// Repetition penalty: penalize recently generated tokens (1.0 = no penalty)
    pub repetition_penalty: f32,
    
    /// Window size for repetition penalty (how far back to look)
    pub repetition_window: usize,
}

impl Default for GreedyDecoder {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            repetition_window: 64,
        }
    }
}

impl GreedyDecoder {
    /// Create a new greedy decoder with default settings
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set temperature for sampling
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(0.01); // Prevent division by zero
        self
    }
    
    /// Enable top-k filtering
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }
    
    /// Enable top-p (nucleus) filtering
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p.clamp(0.0, 1.0));
        self
    }
    
    /// Set repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32, window: usize) -> Self {
        self.repetition_penalty = penalty.max(1.0);
        self.repetition_window = window;
        self
    }
    
    /// Compute numerically stable softmax
    #[inline]
    fn stable_softmax(logits: &ArrayView1<f32>, temperature: f32) -> Array1<f32> {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scaled_logits: Array1<f32> = logits.mapv(|x| ((x - max_logit) / temperature).exp());
        let sum = scaled_logits.sum().max(1e-30);
        scaled_logits / sum
    }
    
    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(
        &self,
        logits: &mut Array1<f32>,
        recent_tokens: &[usize],
    ) {
        if self.repetition_penalty <= 1.0 {
            return;
        }
        
        let window_start = recent_tokens.len().saturating_sub(self.repetition_window);
        let window = &recent_tokens[window_start..];
        
        for &token_id in window {
            if token_id < logits.len() {
                if logits[token_id] > 0.0 {
                    logits[token_id] /= self.repetition_penalty;
                } else {
                    logits[token_id] *= self.repetition_penalty;
                }
            }
        }
    }
    
    /// Apply top-k filtering to logits
    fn apply_top_k(&self, logits: &mut Array1<f32>, k: usize) {
        if k == 0 || k >= logits.len() {
            return;
        }
        
        // Find k-th largest value
        let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let threshold = logits[sorted_indices[k - 1]];
        
        // Zero out values below threshold
        for val in logits.iter_mut() {
            if *val < threshold {
                *val = f32::NEG_INFINITY;
            }
        }
    }
    
    /// Apply top-p (nucleus) filtering to logits
    fn apply_top_p(&self, logits: &mut Array1<f32>, p: f32) {
        if p >= 1.0 {
            return;
        }
        
        // Sort by probability (descending)
        let mut indexed: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Compute softmax for cumulative probability
        let max_logit = indexed[0].1;
        let mut cumsum = 0.0;
        let mut threshold_idx = indexed.len();
        
        for (i, &(_, logit)) in indexed.iter().enumerate() {
            let prob = (logit - max_logit).exp();
            cumsum += prob;
            
            if cumsum >= p * cumsum {
                threshold_idx = i + 1;
                break;
            }
        }
        
        // Zero out values outside top-p
        for &(idx, _) in &indexed[threshold_idx..] {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
    
    /// Find argmax token from probabilities
    #[inline]
    fn argmax(probs: &Array1<f32>) -> usize {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    /// Create input array from tokens
    fn create_input(tokens: &[usize]) -> Array2<f32> {
        let len = tokens.len();
        let mut input = Array2::zeros((1, len));
        input.row_mut(0).assign(&Array1::from_vec(
            tokens.iter().map(|&id| id as f32).collect()
        ));
        input
    }
    
    /// Forward pass through model to get next token logits
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
    
    /// Decode tokens using greedy strategy
    ///
    /// # Arguments
    /// * `model` - The language model to use for generation
    /// * `prefix` - Current token sequence (will be extended in-place)
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    ///
    /// # Returns
    /// Vector of newly generated tokens
    pub fn decode(
        &self,
        model: &mut crate::llm::LLM,
        prefix: &mut Vec<usize>,
        max_new_tokens: usize,
    ) -> Vec<usize> {
        let mut generated = Vec::new();
        let eos_token = model.vocab.encode("</s>");
        
        for _ in 0..max_new_tokens {
            // Get logits from model
            let Some(mut logits) = Self::forward_model(model, prefix) else {
                break;
            };
            
            // Apply repetition penalty
            self.apply_repetition_penalty(&mut logits, prefix);
            
            // Apply top-k filtering
            if let Some(k) = self.top_k {
                self.apply_top_k(&mut logits, k);
            }
            
            // Apply top-p filtering
            if let Some(p) = self.top_p {
                self.apply_top_p(&mut logits, p);
            }
            
            // Convert to probabilities with temperature
            let probs = Self::stable_softmax(&logits.view(), self.temperature);
            
            // Greedy selection (argmax)
            let next_token = Self::argmax(&probs);
            
            // Add to sequences
            prefix.push(next_token);
            generated.push(next_token);
            
            // Check for EOS
            if let Some(eos) = eos_token {
                if next_token == eos {
                    break;
                }
            }
        }
        
        generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decoder_default() {
        let decoder = GreedyDecoder::new();
        assert_eq!(decoder.temperature, 1.0);
        assert!(decoder.top_k.is_none());
        assert!(decoder.top_p.is_none());
        assert_eq!(decoder.repetition_penalty, 1.0);
    }
    
    #[test]
    fn test_decoder_builder() {
        let decoder = GreedyDecoder::new()
            .with_temperature(0.8)
            .with_top_k(50)
            .with_top_p(0.9)
            .with_repetition_penalty(1.2, 32);
        
        assert_eq!(decoder.temperature, 0.8);
        assert_eq!(decoder.top_k, Some(50));
        assert_eq!(decoder.top_p, Some(0.9));
        assert_eq!(decoder.repetition_penalty, 1.2);
        assert_eq!(decoder.repetition_window, 32);
    }
    
    #[test]
    fn test_stable_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let probs = GreedyDecoder::stable_softmax(&logits.view(), 1.0);
        
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs.iter().all(|&p| p > 0.0 && p < 1.0));
    }
    
    #[test]
    fn test_argmax() {
        let probs = Array1::from_vec(vec![0.1, 0.4, 0.3, 0.2]);
        let max_idx = GreedyDecoder::argmax(&probs);
        assert_eq!(max_idx, 1);
    }
    
    #[test]
    fn test_repetition_penalty() {
        let decoder = GreedyDecoder::new().with_repetition_penalty(1.5, 5);
        
        let mut logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let recent = vec![1, 3, 3]; // Tokens 1 and 3 repeated
        
        decoder.apply_repetition_penalty(&mut logits, &recent);
        
        // Token 1 and 3 should be penalized
        assert!(logits[1] < 2.0);
        assert!(logits[3] < 4.0);
        // Other tokens unchanged
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[4], 5.0);
    }
    
    #[test]
    fn test_top_k_filtering() {
        let decoder = GreedyDecoder::new().with_top_k(2);
        
        let mut logits = Array1::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0]);
        decoder.apply_top_k(&mut logits, 2);
        
        // Only top 2 (5.0 and 4.0) should remain
        assert!(logits[1].is_finite()); // 5.0
        assert!(logits[4].is_finite()); // 4.0
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative()); // filtered
        assert!(logits[2].is_infinite() && logits[2].is_sign_negative()); // filtered
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative()); // filtered
    }
    
    #[test]
    fn test_temperature_zero_protection() {
        let decoder = GreedyDecoder::new().with_temperature(0.0);
        // Should be clamped to minimum
        assert!(decoder.temperature >= 0.01);
    }
}
