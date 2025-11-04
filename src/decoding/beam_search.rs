/// Beam search decoding
///
/// Maintains multiple hypotheses (beams) during generation, exploring different
/// paths in parallel. More expensive than greedy but can find higher quality sequences.
///
/// Advantages:
/// - Explores multiple paths: can find better sequences than greedy
/// - Configurable trade-off: beam_width controls quality vs speed
/// - Length normalization: prevents bias toward shorter sequences
///
/// Disadvantages:
/// - Slower: O(beam_width) per step
/// - More memory: must track beam_width sequences
/// - Still no true backtracking: only explores within beam

use ndarray::{Array1, Array2, ArrayView1};
use std::rc::Rc;
use crate::llm::Layer;

/// A single hypothesis in the beam
#[derive(Debug, Clone)]
struct Hypothesis {
    /// Token sequence
    tokens: Vec<usize>,
    
    /// Log probability of sequence
    log_prob: f32,
    
    /// Whether sequence is complete (hit EOS)
    is_complete: bool,
}

impl Hypothesis {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            log_prob: 0.0,
            is_complete: false,
        }
    }
    
    /// Get normalized score (handles length normalization)
    fn score(&self, length_penalty: f32) -> f32 {
        let len = self.tokens.len() as f32;
        let penalty = if len > 0.0 {
            len.powf(length_penalty)
        } else {
            1.0
        };
        self.log_prob / penalty
    }
}

/// Beam search decoder
pub struct BeamSearchDecoder {
    /// Number of beams to maintain
    pub beam_width: usize,
    
    /// Length penalty (>1.0 encourages longer sequences, <1.0 encourages shorter)
    pub length_penalty: f32,
    
    /// Temperature for softmax
    pub temperature: f32,
    
    /// Early stopping: stop when top beam hits EOS
    pub early_stopping: bool,
    
    /// Diversity penalty: encourage different beams to diverge
    pub diversity_penalty: f32,
    
    /// Number of top sequences to return
    pub num_return_sequences: usize,
}

impl Default for BeamSearchDecoder {
    fn default() -> Self {
        Self {
            beam_width: 5,
            length_penalty: 1.0,
            temperature: 1.0,
            early_stopping: true,
            diversity_penalty: 0.0,
            num_return_sequences: 1,
        }
    }
}

impl BeamSearchDecoder {
    /// Create a new beam search decoder
    pub fn new(beam_width: usize) -> Self {
        Self {
            beam_width,
            ..Default::default()
        }
    }
    
    /// Set length penalty
    pub fn with_length_penalty(mut self, penalty: f32) -> Self {
        self.length_penalty = penalty;
        self
    }
    
    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(0.01);
        self
    }
    
    /// Enable/disable early stopping
    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }
    
    /// Set diversity penalty
    pub fn with_diversity_penalty(mut self, penalty: f32) -> Self {
        self.diversity_penalty = penalty;
        self
    }
    
    /// Set number of sequences to return
    pub fn with_num_return_sequences(mut self, num: usize) -> Self {
        self.num_return_sequences = num.max(1).min(self.beam_width);
        self
    }
    
    /// Compute numerically stable log softmax
    #[inline]
    fn log_softmax(logits: &ArrayView1<f32>, temperature: f32) -> Array1<f32> {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scaled: Array1<f32> = logits.mapv(|x| (x - max_logit) / temperature);
        let log_sum_exp = scaled.mapv(|x| x.exp()).sum().ln();
        scaled.mapv(|x| x - log_sum_exp)
    }
    
    /// Create input array from tokens
    fn create_input(tokens: &[usize]) -> Array2<f32> {
        if tokens.is_empty() {
            return Array2::zeros((1, 1));
        }
        
        let len = tokens.len();
        let mut input = Array2::zeros((1, len));
        input.row_mut(0).assign(&Array1::from_vec(
            tokens.iter().map(|&id| id as f32).collect()
        ));
        input
    }
    
    /// Forward pass through model
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
        
        Some(input.row(input.shape()[0] - 1).to_owned())
    }
    
    /// Get top-k tokens and their log probabilities
    fn top_k_log_probs(log_probs: &Array1<f32>, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = log_probs
            .iter()
            .enumerate()
            .map(|(i, &lp)| (i, lp))
            .collect();
        
        // Sort by log probability (descending)
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        indexed.truncate(k);
        indexed
    }
    
    /// Apply diversity penalty to discourage beams from choosing same tokens
    fn apply_diversity_penalty(
        &self,
        log_probs: &mut Array1<f32>,
        already_chosen: &[usize],
    ) {
        if self.diversity_penalty <= 0.0 {
            return;
        }
        
        for &token_id in already_chosen {
            if token_id < log_probs.len() {
                log_probs[token_id] -= self.diversity_penalty;
            }
        }
    }
    
    /// Decode using beam search
    ///
    /// # Arguments
    /// * `model` - The language model to use for generation
    /// * `prefix` - Initial token sequence
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    ///
    /// # Returns
    /// Vector of best sequences (up to num_return_sequences)
    pub fn decode(
        &self,
        model: &mut crate::llm::LLM,
        prefix: &[usize],
        max_new_tokens: usize,
    ) -> Vec<Vec<usize>> {
        let eos_token = model.vocab.encode("</s>");
        
        // Initialize beam with prefix
        let mut beams: Vec<Hypothesis> = vec![Hypothesis {
            tokens: prefix.to_vec(),
            log_prob: 0.0,
            is_complete: false,
        }];
        
        let mut completed: Vec<Hypothesis> = Vec::new();
        
        for _step in 0..max_new_tokens {
            let mut candidates: Vec<Hypothesis> = Vec::new();
            let mut already_chosen: Vec<usize> = Vec::new();
            
            // Expand each beam
            for beam in &beams {
                if beam.is_complete {
                    // Keep completed beams as-is
                    candidates.push(beam.clone());
                    continue;
                }
                
                // Get logits from model
                let Some(logits) = Self::forward_model(model, &beam.tokens) else {
                    continue;
                };
                
                // Compute log probabilities
                let mut log_probs = Self::log_softmax(&logits.view(), self.temperature);
                
                // Apply diversity penalty
                self.apply_diversity_penalty(&mut log_probs, &already_chosen);
                
                // Get top-k candidates for this beam
                let top_k = Self::top_k_log_probs(&log_probs, self.beam_width);
                
                for (token_id, log_prob) in top_k {
                    let mut new_hypothesis = beam.clone();
                    new_hypothesis.tokens.push(token_id);
                    new_hypothesis.log_prob += log_prob;
                    
                    // Check if complete
                    if let Some(eos) = eos_token {
                        if token_id == eos {
                            new_hypothesis.is_complete = true;
                        }
                    }
                    
                    candidates.push(new_hypothesis);
                    already_chosen.push(token_id);
                }
            }
            
            if candidates.is_empty() {
                break;
            }
            
            // Sort by score and keep top beam_width
            candidates.sort_by(|a, b| {
                b.score(self.length_penalty)
                    .partial_cmp(&a.score(self.length_penalty))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // Separate completed from ongoing
            let (complete, incomplete): (Vec<_>, Vec<_>) = 
                candidates.into_iter().partition(|h| h.is_complete);
            
            completed.extend(complete);
            beams = incomplete;
            
            // Keep only top beam_width beams
            beams.truncate(self.beam_width);
            
            // Early stopping: if best beam is complete
            if self.early_stopping && !completed.is_empty() {
                let best_complete_score = completed[0].score(self.length_penalty);
                let best_incomplete_score = beams.first()
                    .map(|h| h.score(self.length_penalty))
                    .unwrap_or(f32::NEG_INFINITY);
                
                if best_complete_score >= best_incomplete_score {
                    break;
                }
            }
            
            // Stop if all beams are complete
            if beams.is_empty() {
                break;
            }
        }
        
        // Collect all final hypotheses
        completed.extend(beams);
        
        // Sort by final score
        completed.sort_by(|a, b| {
            b.score(self.length_penalty)
                .partial_cmp(&a.score(self.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Return top num_return_sequences, removing prefix
        let prefix_len = prefix.len();
        completed
            .into_iter()
            .take(self.num_return_sequences)
            .map(|h| {
                if h.tokens.len() > prefix_len {
                    h.tokens[prefix_len..].to_vec()
                } else {
                    Vec::new()
                }
            })
            .collect()
    }
    
    /// Decode and return only the best sequence (convenience method)
    pub fn decode_one(
        &self,
        model: &mut crate::llm::LLM,
        prefix: &mut Vec<usize>,
        max_new_tokens: usize,
    ) -> Vec<usize> {
        let results = self.decode(model, prefix, max_new_tokens);
        
        if let Some(best) = results.into_iter().next() {
            // Update prefix with best sequence
            prefix.extend_from_slice(&best);
            best
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decoder_default() {
        let decoder = BeamSearchDecoder::default();
        assert_eq!(decoder.beam_width, 5);
        assert_eq!(decoder.length_penalty, 1.0);
        assert_eq!(decoder.temperature, 1.0);
        assert_eq!(decoder.early_stopping, true);
        assert_eq!(decoder.num_return_sequences, 1);
    }
    
    #[test]
    fn test_decoder_builder() {
        let decoder = BeamSearchDecoder::new(10)
            .with_length_penalty(1.2)
            .with_temperature(0.9)
            .with_early_stopping(false)
            .with_diversity_penalty(0.5)
            .with_num_return_sequences(3);
        
        assert_eq!(decoder.beam_width, 10);
        assert_eq!(decoder.length_penalty, 1.2);
        assert_eq!(decoder.temperature, 0.9);
        assert_eq!(decoder.early_stopping, false);
        assert_eq!(decoder.diversity_penalty, 0.5);
        assert_eq!(decoder.num_return_sequences, 3);
    }
    
    #[test]
    fn test_hypothesis_score() {
        let mut hyp = Hypothesis::new();
        hyp.tokens = vec![1, 2, 3];
        hyp.log_prob = -3.0;
        
        // No penalty (length_penalty = 1.0)
        let score1 = hyp.score(1.0);
        assert_eq!(score1, -3.0 / 3.0);
        
        // Favor longer (length_penalty = 1.2)
        let score2 = hyp.score(1.2);
        assert!(score2 > score1);
        
        // Favor shorter (length_penalty = 0.8)
        let score3 = hyp.score(0.8);
        assert!(score3 < score1);
    }
    
    #[test]
    fn test_log_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let log_probs = BeamSearchDecoder::log_softmax(&logits.view(), 1.0);
        
        // Sum of exp(log_probs) should be 1.0
        let sum: f32 = log_probs.mapv(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // All log probs should be negative
        assert!(log_probs.iter().all(|&x| x <= 0.0));
    }
    
    #[test]
    fn test_top_k_log_probs() {
        let log_probs = Array1::from_vec(vec![-1.0, -0.5, -2.0, -0.2, -1.5]);
        let top_k = BeamSearchDecoder::top_k_log_probs(&log_probs, 3);
        
        assert_eq!(top_k.len(), 3);
        // Should be sorted descending
        assert!(top_k[0].1 >= top_k[1].1);
        assert!(top_k[1].1 >= top_k[2].1);
        // First should be index 3 (log_prob -0.2)
        assert_eq!(top_k[0].0, 3);
    }
    
    #[test]
    fn test_diversity_penalty() {
        let decoder = BeamSearchDecoder::new(5).with_diversity_penalty(0.5);
        
        let mut log_probs = Array1::from_vec(vec![-1.0, -0.5, -0.8, -1.2]);
        let already_chosen = vec![1, 2];
        
        decoder.apply_diversity_penalty(&mut log_probs, &already_chosen);
        
        // Tokens 1 and 2 should be penalized
        assert!(log_probs[1] < -0.5);
        assert!(log_probs[2] < -0.8);
        // Others unchanged
        assert_eq!(log_probs[0], -1.0);
        assert_eq!(log_probs[3], -1.2);
    }
    
    #[test]
    fn test_num_return_sequences_clamped() {
        // Should be clamped to beam_width
        let decoder = BeamSearchDecoder::new(5).with_num_return_sequences(10);
        assert_eq!(decoder.num_return_sequences, 5);
        
        // Should be at least 1
        let decoder2 = BeamSearchDecoder::new(5).with_num_return_sequences(0);
        assert_eq!(decoder2.num_return_sequences, 1);
    }
    
    #[test]
    fn test_hypothesis_initialization() {
        let hyp = Hypothesis::new();
        assert!(hyp.tokens.is_empty());
        assert_eq!(hyp.log_prob, 0.0);
        assert!(!hyp.is_complete);
    }
}
