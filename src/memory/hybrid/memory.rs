use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::memory::engram::{EngramCache, EngramMemory};
use crate::memory::titans::NeuralMemory;
use crate::network::Layer;

const DEFAULT_SURPRISE_DECAY: f32 = 0.95;
const DEFAULT_FORGET_GATE: f32 = 0.05;
const DEFAULT_ADAPTIVE_GATE_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemorySource {
    StaticEngram,
    DynamicTitans,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridMemoryConfig {
    pub input_dim: usize,
    pub memory_dim: usize,
    pub engram_ratio: f32,
    pub titans_memory_hidden: usize,
    pub use_adaptive_routing: bool,
    pub enable_cache_hierarchy: bool,
    pub tier_1_cache_size: usize,
    pub tier_2_cache_size: usize,
}

impl Default for HybridMemoryConfig {
    fn default() -> Self {
        Self {
            input_dim: 512,
            memory_dim: 512,
            engram_ratio: super::super::config::OPTIMAL_MEMORY_COMPUTE_RATIO,
            titans_memory_hidden: 256,
            use_adaptive_routing: true,
            enable_cache_hierarchy: true,
            tier_1_cache_size: super::super::config::DEFAULT_CACHE_TIER_1_SIZE,
            tier_2_cache_size: super::super::config::DEFAULT_CACHE_TIER_2_SIZE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridMemory {
    config: HybridMemoryConfig,

    engram_memory: EngramMemory,
    titans_memory: NeuralMemory,

    w_router: Array2<f32>,
    w_engram_proj: Array2<f32>,
    w_titans_proj: Array2<f32>,

    routing_gates: Vec<f32>,
    last_surprise_scores: Vec<f32>,

    pub engram_cache_stats: EngramCache,

    cumulative_surprise: f32,
}

impl HybridMemory {
    pub fn new(config: HybridMemoryConfig) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let router_dim = config.input_dim;
        let titans_val_dim = config.memory_dim / 2;

        let w_router_data: Vec<f32> = (0..router_dim * 2)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let w_router = Array2::from_shape_vec((2, router_dim), w_router_data).unwrap();

        let engram_memory = EngramMemory::new(config.input_dim, config.memory_dim);
        let titans_memory = NeuralMemory::new(
            config.input_dim,
            config.memory_dim / 2,
            config.memory_dim / 2,
            config.titans_memory_hidden,
        );

        let w_engram_proj_data: Vec<f32> = (0..config.memory_dim * config.memory_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let w_engram_proj =
            Array2::from_shape_vec((config.memory_dim, config.memory_dim), w_engram_proj_data)
                .unwrap();

        let w_titans_proj_data: Vec<f32> = (0..config.memory_dim * titans_val_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let w_titans_proj =
            Array2::from_shape_vec((config.memory_dim, titans_val_dim), w_titans_proj_data)
                .unwrap();

        Self {
            config: config.clone(),
            engram_memory,
            titans_memory,
            w_router,
            w_engram_proj,
            w_titans_proj,
            routing_gates: Vec::new(),
            last_surprise_scores: Vec::new(),
            engram_cache_stats: EngramCache::new(
                config.tier_1_cache_size,
                config.tier_2_cache_size,
            ),
            cumulative_surprise: 0.0,
        }
    }

    fn adaptive_routing(&mut self, input: &Array2<f32>) -> Vec<(f32, f32)> {
        let seq_len = input.nrows();
        let mut gates = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input.row(t);

            let router_out = self.w_router.dot(&x_t.to_owned());
            let router_logits = Array1::from_vec(router_out.to_vec());

            let engram_gate = Self::sigmoid(router_logits[0]);
            let titans_gate = Self::sigmoid(router_logits[1]);

            let total_gate = engram_gate + titans_gate;

            let normalized_engram = if total_gate > 1e-6 {
                engram_gate / total_gate
            } else {
                0.5
            };
            let normalized_titans = if total_gate > 1e-6 {
                titans_gate / total_gate
            } else {
                0.5
            };

            gates.push((normalized_engram, normalized_titans));
        }

        gates
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn surprise_based_routing(&mut self, input: &Array2<f32>) -> Vec<(f32, f32)> {
        let seq_len = input.nrows();
        let mut gates = Vec::with_capacity(seq_len);

        for _t in 0..seq_len {
            let avg_surprise = if self.last_surprise_scores.len() >= 10 {
                self.last_surprise_scores.iter().rev().take(10).sum::<f32>() / 10.0
            } else {
                self.cumulative_surprise
            };

            let engram_weight = if avg_surprise > DEFAULT_ADAPTIVE_GATE_THRESHOLD {
                0.3
            } else {
                0.7
            };

            let titans_weight = 1.0 - engram_weight;

            let smoothed_engram = engram_weight * (1.0 - DEFAULT_FORGET_GATE)
                + (self.last_surprise_scores.last().copied().unwrap_or(0.0)) * DEFAULT_FORGET_GATE;
            let smoothed_titans = titans_weight * (1.0 - DEFAULT_FORGET_GATE)
                + (self.last_surprise_scores.last().copied().unwrap_or(0.0)) * DEFAULT_FORGET_GATE;

            gates.push((smoothed_engram, smoothed_titans));
        }

        gates
    }

    fn estimate_surprise(&mut self, input: &Array2<f32>) -> Vec<f32> {
        let seq_len = input.nrows();
        let mut surprise_scores = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input.row(t);

            let input_2d = input.row(t).to_owned().insert_axis(Axis(0));
            let engram_out = self.engram_memory.forward(&input_2d, &vec![0; 32]);

            let titans_out = self.titans_memory.forward(&input_2d);

            let engram_norm = engram_out.mapv(|x| x * x).sum().sqrt();
            let titans_norm = titans_out.mapv(|x| x * x).sum().sqrt();

            let input_norm = x_t.mapv(|x| x * x).sum().sqrt();

            let surprise = if input_norm.is_finite()
                && engram_norm.is_finite()
                && titans_norm.is_finite()
                && input_norm > 1e-6
            {
                ((engram_norm - input_norm).abs() + (titans_norm - input_norm).abs()) / 2.0
            } else {
                0.0
            };

            self.cumulative_surprise = DEFAULT_SURPRISE_DECAY * self.cumulative_surprise
                + (1.0 - DEFAULT_SURPRISE_DECAY) * surprise;
            surprise_scores.push(surprise);
        }

        self.last_surprise_scores = surprise_scores.clone();
        surprise_scores
    }

    pub fn get_cache_stats(&self) -> (f32, f32, usize, usize) {
        let (tier1_rate, tier2_rate) = self.engram_cache_stats.hit_rate();
        let (tier1_hits, tier1_misses, tier2_hits, tier2_misses) = (
            self.engram_cache_stats.tier_1_hits,
            self.engram_cache_stats.tier_1_misses,
            self.engram_cache_stats.tier_2_hits,
            self.engram_cache_stats.tier_2_misses,
        );
        (
            tier1_rate,
            tier2_rate,
            tier1_hits + tier2_hits,
            tier1_misses + tier2_misses,
        )
    }

    pub fn clear_cache_stats(&mut self) {
        self.engram_cache_stats.clear_stats();
    }

    fn sine_positional_encoding(seq_len: usize, dim: usize) -> Array2<f32> {
        let mut encoding = Array2::zeros((seq_len, dim));
        for pos in 0..seq_len {
            for i in 0..dim {
                encoding[(pos, i)] = if i % 2 == 0 {
                    (pos as f32 / 10000_f32.powf((i / 2) as f32)).sin()
                } else {
                    (pos as f32 / 10000_f32.powf(((i - 1) / 2) as f32)).cos()
                };
            }
        }
        encoding
    }
}

impl Layer for HybridMemory {
    fn layer_type(&self) -> &str {
        "HybridMemory"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let seq_len = input.nrows();
        let mut output = Array2::<f32>::zeros((seq_len, self.config.memory_dim));

        let gates = if self.config.use_adaptive_routing {
            let _surprise = self.estimate_surprise(input);
            self.surprise_based_routing(input)
        } else {
            self.adaptive_routing(input)
        };

        let pos_encoding = Self::sine_positional_encoding(seq_len, self.config.memory_dim);

        for (t, (engram_gate, titans_gate)) in gates.iter().enumerate().take(seq_len) {
            let input_t = input.row(t).to_owned().insert_axis(Axis(0));
            let engram_out = self.engram_memory.forward(&input_t, &vec![0; 32]);
            let titans_out = self.titans_memory.forward(&input_t);

            let engram_proj = self.w_engram_proj.dot(&engram_out.row(0).to_owned());
            let titans_proj = self.w_titans_proj.dot(&titans_out.row(0).to_owned());

            let pos_enc = pos_encoding.row(t);

            let gated_engram = engram_proj.mapv(|x| x * *engram_gate);
            let gated_titans = titans_proj.mapv(|x| x * *titans_gate);

            let combined = &gated_engram + &gated_titans + pos_enc;

            output.row_mut(t).assign(&combined);
        }

        output
    }

    fn backward(&mut self, _grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        Array2::zeros((_grads.nrows(), self.config.input_dim))
    }

    fn parameters(&self) -> usize {
        let engram_params = self.engram_memory.parameters();
        let titans_params = self.titans_memory.parameters();
        let router_params =
            self.w_router.len() + self.w_engram_proj.len() + self.w_titans_proj.len();
        engram_params + titans_params + router_params
    }

    fn weight_norm(&self) -> f32 {
        let engram_norm = self.engram_memory.weight_norm();
        let titans_norm = self.titans_memory.weight_norm();
        let router_norm = self.w_router.iter().map(|&x| x * x).sum::<f32>()
            + self.w_engram_proj.iter().map(|&x| x * x).sum::<f32>()
            + self.w_titans_proj.iter().map(|&x| x * x).sum::<f32>();
        (engram_norm * engram_norm + titans_norm * titans_norm + router_norm).sqrt()
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        _output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        (Array2::zeros((1, self.config.input_dim)), vec![])
    }

    fn apply_gradients(
        &mut self,
        _gradients: &[Array2<f32>],
        _learning_rate: f32,
    ) -> crate::errors::Result<()> {
        Ok(())
    }

    fn zero_gradients(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_memory_forward() {
        let config = HybridMemoryConfig::default();
        let mut memory = HybridMemory::new(config);

        let seq_len = 10;
        let input = Array2::from_elem((seq_len, 512), 1.0);

        let output = memory.forward(&input);

        assert_eq!(output.shape(), &[seq_len, 512]);
    }

    #[test]
    fn test_hybrid_routing_gates() {
        let config = HybridMemoryConfig::default();
        let mut memory = HybridMemory::new(config);

        let seq_len = 5;
        let input = Array2::from_elem((seq_len, 512), 1.0);

        let gates = memory.adaptive_routing(&input);

        assert_eq!(gates.len(), seq_len);
        for (engram_gate, titans_gate) in gates {
            assert!((0.0..=1.0).contains(&engram_gate));
            assert!((0.0..=1.0).contains(&titans_gate));
        }
    }

    #[test]
    fn test_surprise_estimation() {
        let config = HybridMemoryConfig::default();
        let mut memory = HybridMemory::new(config);

        let seq_len = 10;
        let input = Array2::from_shape_fn((seq_len, 512), |(i, j)| ((i * 512 + j) as f32) * 0.01);

        let surprise = memory.estimate_surprise(&input);

        assert_eq!(surprise.len(), seq_len);
        for s in surprise {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn test_cache_stats() {
        let config = HybridMemoryConfig::default();
        let memory = HybridMemory::new(config);

        let (tier1_rate, tier2_rate, _, _) = memory.get_cache_stats();

        assert!((0.0..=1.0).contains(&tier1_rate));
        assert!((0.0..=1.0).contains(&tier2_rate));
    }
}
