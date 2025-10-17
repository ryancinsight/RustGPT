use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Configuration for beam search generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamSearchConfig {
    /// Initial beam width (number of hypotheses to maintain)
    pub beam_width: usize,

    /// Enable adaptive beam width based on prediction confidence
    pub use_adaptive_beam: bool,

    /// Minimum beam width for adaptive beam search
    pub min_beam_width: usize,

    /// Maximum beam width for adaptive beam search
    pub max_beam_width: usize,

    /// Softmax entropy threshold for beam width adaptation
    /// Higher entropy (uncertain) → increase beam width
    /// Lower entropy (confident) → decrease beam width
    pub adaptation_threshold: f32,

    /// Maximum generation length
    pub max_length: usize,

    /// Sampling temperature (1.0 = no change, <1.0 = more confident, >1.0 = more diverse)
    pub temperature: f32,

    /// Length penalty alpha for scoring (score / length^alpha)
    /// Typical values: 0.6-1.0, 0.0 = no penalty
    pub length_penalty_alpha: f32,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            use_adaptive_beam: false,
            min_beam_width: 1,
            max_beam_width: 8,
            adaptation_threshold: 0.5,
            max_length: 100,
            temperature: 1.0,
            length_penalty_alpha: 1.0,
        }
    }
}

impl BeamSearchConfig {
    /// Create a new beam search configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set beam width
    pub fn with_beam_width(mut self, beam_width: usize) -> Self {
        self.beam_width = beam_width;
        self
    }

    /// Enable adaptive beam width
    pub fn with_adaptive_beam(mut self, use_adaptive: bool) -> Self {
        self.use_adaptive_beam = use_adaptive;
        self
    }

    /// Set min/max beam width for adaptive beam search
    pub fn with_beam_range(mut self, min: usize, max: usize) -> Self {
        self.min_beam_width = min;
        self.max_beam_width = max;
        self
    }

    /// Set adaptation threshold
    pub fn with_adaptation_threshold(mut self, threshold: f32) -> Self {
        self.adaptation_threshold = threshold;
        self
    }

    /// Set maximum generation length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set length penalty alpha
    pub fn with_length_penalty_alpha(mut self, alpha: f32) -> Self {
        self.length_penalty_alpha = alpha;
        self
    }
}

/// A single beam hypothesis during beam search
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token sequence for this hypothesis
    pub tokens: Vec<usize>,

    /// Cumulative log probability score
    pub score: f32,

    /// Whether this hypothesis is complete (hit end token or max length)
    pub is_complete: bool,
}

impl BeamHypothesis {
    /// Create a new beam hypothesis
    pub fn new(tokens: Vec<usize>, score: f32) -> Self {
        Self {
            tokens,
            score,
            is_complete: false,
        }
    }

    /// Get the normalized score (score / length^alpha)
    /// This prevents bias towards shorter sequences
    pub fn normalized_score(&self, alpha: f32) -> f32 {
        self.score / (self.tokens.len() as f32).powf(alpha)
    }
}

/// State for beam search generation
pub struct BeamSearchState {
    /// Current beam hypotheses
    pub beams: Vec<BeamHypothesis>,

    /// Current beam width (may change with adaptive beam)
    pub current_beam_width: usize,

    /// Completed hypotheses
    pub completed: Vec<BeamHypothesis>,
}

impl BeamSearchState {
    /// Create a new beam search state
    pub fn new(initial_tokens: Vec<usize>, beam_width: usize) -> Self {
        let initial_beam = BeamHypothesis::new(initial_tokens, 0.0);
        Self {
            beams: vec![initial_beam],
            current_beam_width: beam_width,
            completed: Vec::new(),
        }
    }

    /// Expand beams with new predictions
    pub fn expand(
        &mut self,
        predictions: &[Array1<f32>],
        config: &BeamSearchConfig,
        _vocab_size: usize,
    ) {
        let mut candidates = Vec::new();

        // For each current beam
        for (beam_idx, beam) in self.beams.iter().enumerate() {
            if beam.is_complete {
                continue;
            }

            let logits = &predictions[beam_idx];

            // Apply temperature to logits
            let logits = if (config.temperature - 1.0).abs() > 1e-6 {
                logits / config.temperature
            } else {
                logits.clone()
            };

            // Compute softmax: numerically stable
            let max_logit = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits = logits.mapv(|x| (x - max_logit).exp());
            let sum_exp = exp_logits.sum();
            let probs = exp_logits / sum_exp;

            // Get top-k tokens for this beam
            let mut token_scores: Vec<(usize, f32)> = probs
                .iter()
                .enumerate()
                .map(|(token_id, &prob)| (token_id, prob))
                .collect();

            // Sort by probability (descending)
            token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top beam_width candidates
            for (token_id, prob) in token_scores.iter().take(self.current_beam_width) {
                if *prob > 0.0 {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(*token_id);

                    // Score is cumulative log probability
                    let new_score = beam.score + prob.ln();

                    candidates.push(BeamHypothesis::new(new_tokens, new_score));
                }
            }
        }

        // Sort candidates by score (descending)
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Keep top beam_width candidates
        self.beams = candidates
            .into_iter()
            .take(self.current_beam_width)
            .collect();
    }

    /// Compute softmax entropy for adaptive beam width
    pub fn compute_entropy(&self, predictions: &[Array1<f32>]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }

        let mut total_entropy = 0.0;

        for logits in predictions {
            // Compute softmax
            let max_logit = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits = logits.mapv(|x| (x - max_logit).exp());
            let sum_exp = exp_logits.sum();
            let probs = exp_logits / sum_exp;

            // Compute entropy: H = -Σ p(x) * log(p(x))
            let entropy: f32 = probs
                .iter()
                .filter(|&&p| p > 1e-10)
                .map(|&p| -p * p.ln())
                .sum();

            total_entropy += entropy;
        }

        // Average entropy across all beams
        total_entropy / predictions.len() as f32
    }

    /// Adapt beam width based on prediction entropy
    pub fn adapt_beam_width(&mut self, entropy: f32, config: &BeamSearchConfig) {
        if !config.use_adaptive_beam {
            return;
        }

        // Normalize entropy to [0, 1] range
        // Typical entropy for uniform distribution over vocab_size is ln(vocab_size)
        // For vocab_size ~= 1000, max entropy ~= 6.9
        let normalized_entropy = (entropy / 7.0).min(1.0);

        // Adjust beam width based on entropy
        if normalized_entropy > config.adaptation_threshold {
            // High entropy (uncertain) → increase beam width
            self.current_beam_width = (self.current_beam_width + 1).min(config.max_beam_width);
        } else {
            // Low entropy (confident) → decrease beam width
            self.current_beam_width =
                (self.current_beam_width.saturating_sub(1)).max(config.min_beam_width);
        }
    }

    /// Mark beams as complete if they hit end token or max length
    pub fn mark_complete(&mut self, end_token: usize, max_length: usize) {
        for beam in &mut self.beams {
            if beam.tokens.last() == Some(&end_token) || beam.tokens.len() >= max_length {
                beam.is_complete = true;
            }
        }

        // Move completed beams to completed list efficiently
        let mut i = 0;
        while i < self.beams.len() {
            if self.beams[i].is_complete {
                let completed = self.beams.swap_remove(i);
                self.completed.push(completed);
            } else {
                i += 1;
            }
        }
    }

    /// Check if all beams are complete
    pub fn is_done(&self) -> bool {
        self.beams.is_empty()
    }

    /// Get the best hypothesis (highest normalized score)
    pub fn get_best(&self, config: &BeamSearchConfig) -> Option<&BeamHypothesis> {
        // Check both active and completed beams
        let all_beams = self.beams.iter().chain(self.completed.iter());

        all_beams.max_by(|a, b| {
            a.normalized_score(config.length_penalty_alpha)
                .partial_cmp(&b.normalized_score(config.length_penalty_alpha))
                .unwrap()
        })
    }
}
