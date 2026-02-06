use ndarray::{Array2, Zip};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::domain::memory::engram::{EngramCache, EngramMemory};
use crate::domain::memory::titans::NeuralMemory;
use crate::domain::network::Layer;

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
    pub surprise_decay: f32,
    pub forget_gate: f32,
    pub adaptive_gate_threshold: f32,
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
            surprise_decay: 0.95,
            forget_gate: 0.05,
            adaptive_gate_threshold: 0.5,
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
    engram_ratio_raw: f32,
    surprise_decay_raw: f32,
    forget_gate_raw: f32,
    adaptive_gate_threshold_raw: f32,

    routing_gates: Vec<(f32, f32)>,
    last_surprise_scores: Vec<f32>,

    cumulative_surprise: f32,
    #[serde(skip)]
    cached_pos_encoding: Option<(usize, Array2<f32>)>,
    #[serde(skip)]
    dummy_token_ids: Vec<usize>,
    #[serde(skip)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip)]
    cached_engram_out: Option<Array2<f32>>,
    #[serde(skip)]
    cached_titans_out: Option<Array2<f32>>,
    #[serde(skip)]
    cached_gates: Option<Vec<(f32, f32)>>,
    #[serde(skip)]
    cached_prev_gates: Option<(f32, f32)>,
    #[serde(skip)]
    cached_prev_cumulative_surprise: Option<f32>,
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

        let mut engram_memory = EngramMemory::new(config.input_dim, config.memory_dim);
        engram_memory.cache = if config.enable_cache_hierarchy {
            EngramCache::new(config.tier_1_cache_size, config.tier_2_cache_size)
        } else {
            EngramCache::new(0, 0)
        };
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

        let engram_ratio_raw = Self::logit(config.engram_ratio);
        let surprise_decay_raw = Self::logit(config.surprise_decay);
        let forget_gate_raw = Self::logit(config.forget_gate);
        let adaptive_gate_threshold_raw = Self::softplus_inv(config.adaptive_gate_threshold);

        Self {
            config: config.clone(),
            engram_memory,
            titans_memory,
            w_router,
            w_engram_proj,
            w_titans_proj,
            engram_ratio_raw,
            surprise_decay_raw,
            forget_gate_raw,
            adaptive_gate_threshold_raw,
            routing_gates: Vec::new(),
            last_surprise_scores: Vec::new(),
            cumulative_surprise: 0.0,
            cached_pos_encoding: None,
            dummy_token_ids: vec![0; 1],
            cached_input: None,
            cached_engram_out: None,
            cached_titans_out: None,
            cached_gates: None,
            cached_prev_gates: None,
            cached_prev_cumulative_surprise: None,
        }
    }

    fn ensure_dummy_token_ids(&mut self, seq_len: usize) {
        if self.dummy_token_ids.len() != seq_len {
            self.dummy_token_ids.resize(seq_len, 0);
        }
        for (idx, token) in self.dummy_token_ids.iter_mut().enumerate() {
            *token = idx;
        }
    }

    pub fn adaptive_routing(&mut self, input: &Array2<f32>) -> Vec<(f32, f32)> {
        let seq_len = input.nrows();
        let mut gates = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input.row(t);

            let router_out = self.w_router.dot(&x_t);
            let engram_gate = Self::sigmoid(router_out[0]);
            let titans_gate = Self::sigmoid(router_out[1]);

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

            let (normalized_engram, normalized_titans) =
                self.apply_engram_ratio(normalized_engram, normalized_titans);
            gates.push((normalized_engram, normalized_titans));
        }

        gates
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    fn logit(x: f32) -> f32 {
        let x = x.clamp(1e-6, 1.0 - 1e-6);
        (x / (1.0 - x)).ln()
    }

    fn softplus_inv(x: f32) -> f32 {
        let x = x.max(1e-6);
        (x.exp() - 1.0).ln()
    }

    fn engram_ratio(&self) -> f32 {
        Self::sigmoid(self.engram_ratio_raw)
    }

    fn surprise_decay(&self) -> f32 {
        Self::sigmoid(self.surprise_decay_raw)
    }

    fn forget_gate(&self) -> f32 {
        Self::sigmoid(self.forget_gate_raw)
    }

    fn adaptive_gate_threshold(&self) -> f32 {
        Self::softplus(self.adaptive_gate_threshold_raw)
    }

    fn apply_engram_ratio(&self, engram_gate: f32, titans_gate: f32) -> (f32, f32) {
        let ratio = self.engram_ratio();
        let scaled_engram = engram_gate * ratio;
        let scaled_titans = titans_gate * (1.0 - ratio);
        let denom = scaled_engram + scaled_titans + 1e-6;
        (scaled_engram / denom, scaled_titans / denom)
    }

    pub fn estimate_surprise(&mut self, input: &Array2<f32>) -> Vec<f32> {
        let seq_len = input.nrows();
        let mut surprise_scores = Vec::with_capacity(seq_len);

        self.ensure_dummy_token_ids(seq_len);
        let engram_out = self
            .engram_memory
            .forward(input, &self.dummy_token_ids);
        let titans_out = self.titans_memory.forward(input);

        for t in 0..seq_len {
            let x_t = input.row(t);
            let engram_norm = engram_out.row(t).mapv(|x| x * x).sum().sqrt();
            let titans_norm = titans_out.row(t).mapv(|x| x * x).sum().sqrt();

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

            surprise_scores.push(surprise);
        }

        self.last_surprise_scores = surprise_scores.clone();
        surprise_scores
    }

    fn pos_encoding(&mut self, seq_len: usize) -> &Array2<f32> {
        let rebuild = self
            .cached_pos_encoding
            .as_ref()
            .map(|(len, _)| *len != seq_len)
            .unwrap_or(true);
        if rebuild {
            let encoding = Self::sine_positional_encoding(seq_len, self.config.memory_dim);
            self.cached_pos_encoding = Some((seq_len, encoding));
        }
        &self.cached_pos_encoding.as_ref().unwrap().1
    }


    pub fn get_cache_stats(&self) -> (f32, f32, usize, usize) {
        let (tier1_rate, tier2_rate) = self.engram_memory.cache.hit_rate();
        let (tier1_hits, tier1_misses, tier2_hits, tier2_misses) = (
            self.engram_memory.cache.tier_1_hits,
            self.engram_memory.cache.tier_1_misses,
            self.engram_memory.cache.tier_2_hits,
            self.engram_memory.cache.tier_2_misses,
        );
        (
            tier1_rate,
            tier2_rate,
            tier1_hits + tier2_hits,
            tier1_misses + tier2_misses,
        )
    }

    pub fn clear_cache_stats(&mut self) {
        self.engram_memory.cache.clear_stats();
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

        self.ensure_dummy_token_ids(seq_len);
        let engram_out = self
            .engram_memory
            .forward(input, &self.dummy_token_ids);
        let titans_out = self.titans_memory.forward(input);
        let pos_encoding = self.pos_encoding(seq_len).to_owned();
        let mut surprise_scores = Vec::with_capacity(seq_len);
        let mut gates = Vec::with_capacity(seq_len);
        let prev_gates = self.routing_gates.last().copied().unwrap_or((0.5, 0.5));
        let mut prev_engram = prev_gates.0;
        let mut prev_titans = prev_gates.1;
        let mut cumulative_surprise = self.cumulative_surprise;
        let surprise_decay = self.surprise_decay();
        let forget_gate = self.forget_gate();
        let adaptive_gate_threshold = self.adaptive_gate_threshold();
        let mut output = Array2::<f32>::zeros((seq_len, self.config.memory_dim));

        for t in 0..seq_len {
            let (engram_gate, titans_gate) = if self.config.use_adaptive_routing {
                let input_norm = input.row(t).mapv(|x| x * x).sum().sqrt();
                let engram_norm = engram_out.row(t).mapv(|x| x * x).sum().sqrt();
                let titans_norm = titans_out.row(t).mapv(|x| x * x).sum().sqrt();

                let surprise = if input_norm.is_finite()
                    && engram_norm.is_finite()
                    && titans_norm.is_finite()
                    && input_norm > 1e-6
                {
                    ((engram_norm - input_norm).abs() + (titans_norm - input_norm).abs()) / 2.0
                } else {
                    0.0
                };
                surprise_scores.push(surprise);
                cumulative_surprise =
                    surprise_decay * cumulative_surprise + (1.0 - surprise_decay) * surprise;
                let avg_surprise = cumulative_surprise;

                let engram_weight = Self::sigmoid(adaptive_gate_threshold - avg_surprise);
                let titans_weight = 1.0 - engram_weight;

                let smoothed_engram =
                    engram_weight * (1.0 - forget_gate) + prev_engram * forget_gate;
                let smoothed_titans =
                    titans_weight * (1.0 - forget_gate) + prev_titans * forget_gate;

                let (smoothed_engram, smoothed_titans) =
                    self.apply_engram_ratio(smoothed_engram, smoothed_titans);
                prev_engram = smoothed_engram;
                prev_titans = smoothed_titans;
                (smoothed_engram, smoothed_titans)
            } else {
                let router_out = self.w_router.dot(&input.row(t));
                let engram_gate = Self::sigmoid(router_out[0]);
                let titans_gate = Self::sigmoid(router_out[1]);
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
                self.apply_engram_ratio(normalized_engram, normalized_titans)
            };
            gates.push((engram_gate, titans_gate));

            let mut engram_proj = self.w_engram_proj.dot(&engram_out.row(t));
            let mut titans_proj = self.w_titans_proj.dot(&titans_out.row(t));

            let pos_enc = pos_encoding.row(t);

            engram_proj.mapv_inplace(|x| x * engram_gate);
            titans_proj.mapv_inplace(|x| x * titans_gate);

            engram_proj += &titans_proj;
            engram_proj += &pos_enc;

            output.row_mut(t).assign(&engram_proj);
        }

        if self.config.use_adaptive_routing {
            self.last_surprise_scores = surprise_scores;
        }
        self.cached_prev_gates = Some(prev_gates);
        self.cached_prev_cumulative_surprise = Some(self.cumulative_surprise);
        self.routing_gates = gates;
        self.cumulative_surprise = cumulative_surprise;
        self.cached_input = Some(input.to_owned());
        self.cached_engram_out = Some(engram_out);
        self.cached_titans_out = Some(titans_out);
        self.cached_gates = Some(self.routing_gates.clone());
        output
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (input_grads, param_grads) = self.compute_gradients(input, grads);
        let _ = self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        let engram_params = self.engram_memory.parameters();
        let titans_params = self.titans_memory.parameters();
        let router_params =
            self.w_router.len() + self.w_engram_proj.len() + self.w_titans_proj.len();
        engram_params + titans_params + router_params + 4
    }

    fn weight_norm(&self) -> f32 {
        let engram_norm = self.engram_memory.weight_norm();
        let titans_norm = self.titans_memory.weight_norm();
        let router_norm = self.w_router.iter().map(|&x| x * x).sum::<f32>()
            + self.w_engram_proj.iter().map(|&x| x * x).sum::<f32>()
            + self.w_titans_proj.iter().map(|&x| x * x).sum::<f32>();
        let ratio = self.engram_ratio();
        let surprise_decay = self.surprise_decay();
        let forget_gate = self.forget_gate();
        let threshold = self.adaptive_gate_threshold();
        let scalar_norm = ratio * ratio
            + surprise_decay * surprise_decay
            + forget_gate * forget_gate
            + threshold * threshold;
        (engram_norm * engram_norm + titans_norm * titans_norm + router_norm + scalar_norm).sqrt()
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let engram_out = self
            .cached_engram_out
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let titans_out = self
            .cached_titans_out
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let gates = self
            .cached_gates
            .as_ref()
            .expect("forward must be called before compute_gradients");

        let mut d_w_router = Array2::<f32>::zeros(self.w_router.raw_dim());
        let mut d_w_engram_proj = Array2::<f32>::zeros(self.w_engram_proj.raw_dim());
        let mut d_w_titans_proj = Array2::<f32>::zeros(self.w_titans_proj.raw_dim());

        let mut engram_out_grads = Array2::<f32>::zeros(engram_out.raw_dim());
        let mut titans_out_grads = Array2::<f32>::zeros(titans_out.raw_dim());

        let mut router_input_grads = Array2::<f32>::zeros(input.raw_dim());
        let mut input_surprise_grads = Array2::<f32>::zeros(input.raw_dim());

        let engram_ratio = self.engram_ratio();
        let ratio_deriv = engram_ratio * (1.0 - engram_ratio);
        let surprise_decay = self.surprise_decay();
        let surprise_decay_deriv = surprise_decay * (1.0 - surprise_decay);
        let forget_gate = self.forget_gate();
        let forget_gate_deriv = forget_gate * (1.0 - forget_gate);
        let adaptive_gate_threshold = self.adaptive_gate_threshold();
        let adaptive_threshold_deriv = Self::sigmoid(self.adaptive_gate_threshold_raw);

        let mut d_ratio = 0.0f32;
        let mut d_surprise_decay = 0.0f32;
        let mut d_forget_gate = 0.0f32;
        let mut d_threshold = 0.0f32;

        let mut input_norms = Vec::new();
        let mut engram_norms = Vec::new();
        let mut titans_norms = Vec::new();
        let mut surprises = Vec::new();
        let mut cumulatives = Vec::new();
        let mut engram_weights = Vec::new();
        let mut smoothed_engram = Vec::new();
        let mut smoothed_titans = Vec::new();

        let prev_gates = self.cached_prev_gates.unwrap_or((0.5, 0.5));
        let prev_cumulative = self.cached_prev_cumulative_surprise.unwrap_or(self.cumulative_surprise);

        if self.config.use_adaptive_routing {
            let mut prev_engram = prev_gates.0;
            let mut prev_titans = prev_gates.1;
            let mut cumulative = prev_cumulative;

            for t in 0..input.nrows() {
                let x_t = input.row(t);
                let engram_norm = engram_out.row(t).mapv(|x| x * x).sum().sqrt();
                let titans_norm = titans_out.row(t).mapv(|x| x * x).sum().sqrt();
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

                cumulative = surprise_decay * cumulative + (1.0 - surprise_decay) * surprise;
                let engram_weight = Self::sigmoid(adaptive_gate_threshold - cumulative);
                let titans_weight = 1.0 - engram_weight;
                let g_engram = engram_weight * (1.0 - forget_gate) + prev_engram * forget_gate;
                let g_titans = titans_weight * (1.0 - forget_gate) + prev_titans * forget_gate;

                input_norms.push(input_norm);
                engram_norms.push(engram_norm);
                titans_norms.push(titans_norm);
                surprises.push(surprise);
                cumulatives.push(cumulative);
                engram_weights.push(engram_weight);
                smoothed_engram.push(g_engram);
                smoothed_titans.push(g_titans);

                prev_engram = g_engram;
                prev_titans = g_titans;
            }
        }

        for (t, &(engram_gate, titans_gate)) in gates.iter().enumerate() {
            let dy_t = output_grads.row(t);

            let engram_proj = self.w_engram_proj.dot(&engram_out.row(t));
            let titans_proj = self.w_titans_proj.dot(&titans_out.row(t));

            let d_engram_proj = dy_t.to_owned() * engram_gate;
            let d_titans_proj = dy_t.to_owned() * titans_gate;

            d_w_engram_proj += &d_engram_proj
                .clone()
                .insert_axis(ndarray::Axis(1))
                .dot(&engram_out.row(t).insert_axis(ndarray::Axis(0)));
            d_w_titans_proj += &d_titans_proj
                .clone()
                .insert_axis(ndarray::Axis(1))
                .dot(&titans_out.row(t).insert_axis(ndarray::Axis(0)));

            engram_out_grads
                .row_mut(t)
                .assign(&self.w_engram_proj.t().dot(&d_engram_proj));
            titans_out_grads
                .row_mut(t)
                .assign(&self.w_titans_proj.t().dot(&d_titans_proj));

            let d_engram_gate = dy_t.dot(&engram_proj);
            let d_titans_gate = dy_t.dot(&titans_proj);

            if !self.config.use_adaptive_routing {
                let eps = 1e-6;

                let x_t = input.row(t);
                let router_out = self.w_router.dot(&x_t);
                let engram_gate_raw = Self::sigmoid(router_out[0]);
                let titans_gate_raw = Self::sigmoid(router_out[1]);
                let total_gate = engram_gate_raw + titans_gate_raw + eps;
                let inv_total = 1.0 / total_gate;
                let inv_total_sq = inv_total * inv_total;

                let normalized_engram = engram_gate_raw * inv_total;
                let normalized_titans = titans_gate_raw * inv_total;

                let scaled_engram = normalized_engram * engram_ratio;
                let scaled_titans = normalized_titans * (1.0 - engram_ratio);
                let scaled_total = scaled_engram + scaled_titans + eps;
                let inv_scaled_total = 1.0 / scaled_total;
                let inv_scaled_total_sq = inv_scaled_total * inv_scaled_total;

                let d_scaled_engram = d_engram_gate * (scaled_titans + eps) * inv_scaled_total_sq
                    + d_titans_gate * (-scaled_titans) * inv_scaled_total_sq;
                let d_scaled_titans = d_engram_gate * (-scaled_engram) * inv_scaled_total_sq
                    + d_titans_gate * (scaled_engram + eps) * inv_scaled_total_sq;

                d_ratio += d_scaled_engram * normalized_engram
                    - d_scaled_titans * normalized_titans;

                let d_norm_engram = d_scaled_engram * engram_ratio;
                let d_norm_titans = d_scaled_titans * (1.0 - engram_ratio);

                let d_engram_raw = d_norm_engram * (titans_gate_raw + eps) * inv_total_sq
                    + d_norm_titans * (-titans_gate_raw) * inv_total_sq;
                let d_titans_raw = d_norm_engram * (-engram_gate_raw) * inv_total_sq
                    + d_norm_titans * (engram_gate_raw + eps) * inv_total_sq;

                let d_router0 = d_engram_raw * engram_gate_raw * (1.0 - engram_gate_raw);
                let d_router1 = d_titans_raw * titans_gate_raw * (1.0 - titans_gate_raw);

                let d_router = ndarray::Array1::from_vec(vec![d_router0, d_router1]);

                d_w_router += &d_router
                    .clone()
                    .insert_axis(ndarray::Axis(1))
                    .dot(&x_t.insert_axis(ndarray::Axis(0)));
                router_input_grads
                    .row_mut(t)
                    .assign(&self.w_router.t().dot(&d_router));
            }
        }

        if self.config.use_adaptive_routing {
            let mut d_prev_engram = 0.0f32;
            let mut d_prev_titans = 0.0f32;
            let mut d_c_next = 0.0f32;
            let eps = 1e-6;

            for t in (0..input.nrows()).rev() {
                let dy_t = output_grads.row(t);
                let engram_proj = self.w_engram_proj.dot(&engram_out.row(t));
                let titans_proj = self.w_titans_proj.dot(&titans_out.row(t));

                let d_out_engram = dy_t.dot(&engram_proj);
                let d_out_titans = dy_t.dot(&titans_proj);

                let g_engram = smoothed_engram[t];
                let g_titans = smoothed_titans[t];

                let scaled_engram = g_engram * engram_ratio;
                let scaled_titans = g_titans * (1.0 - engram_ratio);
                let denom = scaled_engram + scaled_titans + eps;
                let inv_denom = 1.0 / denom;
                let inv_denom_sq = inv_denom * inv_denom;

                let d_scaled_engram = d_out_engram * (scaled_titans + eps) * inv_denom_sq
                    + d_out_titans * (-scaled_titans) * inv_denom_sq;
                let d_scaled_titans = d_out_engram * (-scaled_engram) * inv_denom_sq
                    + d_out_titans * (scaled_engram + eps) * inv_denom_sq;

                d_ratio += d_scaled_engram * g_engram - d_scaled_titans * g_titans;

                let d_g_engram = d_scaled_engram * engram_ratio + d_prev_engram;
                let d_g_titans = d_scaled_titans * (1.0 - engram_ratio) + d_prev_titans;

                let w_t = engram_weights[t];
                let g_prev_engram = if t == 0 { prev_gates.0 } else { smoothed_engram[t - 1] };
                let g_prev_titans = if t == 0 { prev_gates.1 } else { smoothed_titans[t - 1] };

                let d_w_t = (1.0 - forget_gate) * (d_g_engram - d_g_titans);
                d_forget_gate += d_g_engram * (g_prev_engram - w_t)
                    + d_g_titans * (g_prev_titans - (1.0 - w_t));

                d_prev_engram = d_g_engram * forget_gate;
                d_prev_titans = d_g_titans * forget_gate;

                let d_z = d_w_t * w_t * (1.0 - w_t);
                d_threshold += d_z;

                let d_c_t = d_c_next - d_z;
                let c_prev = if t == 0 { prev_cumulative } else { cumulatives[t - 1] };
                d_surprise_decay += d_c_t * (c_prev - surprises[t]);
                let d_surprise = d_c_t * (1.0 - surprise_decay);
                d_c_next = d_c_t * surprise_decay;

                let input_norm = input_norms[t];
                let engram_norm = engram_norms[t];
                let titans_norm = titans_norms[t];
                let sign_engram = if engram_norm - input_norm >= 0.0 { 1.0 } else { -1.0 };
                let sign_titans = if titans_norm - input_norm >= 0.0 { 1.0 } else { -1.0 };

                let d_engram_norm = 0.5 * d_surprise * sign_engram;
                let d_titans_norm = 0.5 * d_surprise * sign_titans;
                let d_input_norm = -0.5 * d_surprise * (sign_engram + sign_titans);

                let inv_engram_norm = 1.0 / (engram_norm + eps);
                let inv_titans_norm = 1.0 / (titans_norm + eps);
                let inv_input_norm = 1.0 / (input_norm + eps);

                {
                    let engram_row = engram_out.row(t);
                    let mut engram_grad_row = engram_out_grads.row_mut(t);
                    Zip::from(&mut engram_grad_row)
                        .and(&engram_row)
                        .for_each(|g, &e| *g += d_engram_norm * e * inv_engram_norm);
                }

                {
                    let titans_row = titans_out.row(t);
                    let mut titans_grad_row = titans_out_grads.row_mut(t);
                    Zip::from(&mut titans_grad_row)
                        .and(&titans_row)
                        .for_each(|g, &e| *g += d_titans_norm * e * inv_titans_norm);
                }

                {
                    let input_row = input.row(t);
                    let mut input_grad_row = input_surprise_grads.row_mut(t);
                    Zip::from(&mut input_grad_row)
                        .and(&input_row)
                        .for_each(|g, &e| *g += d_input_norm * e * inv_input_norm);
                }
            }
        }

        let (engram_input_grads, engram_param_grads) = self.engram_memory.compute_gradients(
            input,
            &self.dummy_token_ids,
            &engram_out_grads,
        );
        let (titans_input_grads, titans_param_grads) =
            self.titans_memory.compute_gradients(input, &titans_out_grads);

        let mut input_grads = engram_input_grads + titans_input_grads;
        input_grads += &router_input_grads;
        input_grads += &input_surprise_grads;

        let mut param_grads = vec![
            d_w_router,
            d_w_engram_proj,
            d_w_titans_proj,
            Array2::from_elem((1, 1), d_ratio * ratio_deriv),
            Array2::from_elem((1, 1), d_surprise_decay * surprise_decay_deriv),
            Array2::from_elem((1, 1), d_forget_gate * forget_gate_deriv),
            Array2::from_elem((1, 1), d_threshold * adaptive_threshold_deriv),
        ];
        param_grads.extend(engram_param_grads);
        param_grads.extend(titans_param_grads);

        (input_grads, param_grads)
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::common::errors::Result<()> {
        let engram_grad_count = self.engram_memory.gradient_count();
        let titans_grad_count = self.titans_memory.gradient_count();
        let expected = 3 + 4 + engram_grad_count + titans_grad_count;
        if gradients.len() != expected {
            return Err(crate::common::errors::ModelError::GradientError {
                message: format!(
                    "HybridMemory gradient count mismatch: expected {}, got {}",
                    expected,
                    gradients.len()
                ),
            });
        }

        let mut idx = 0;
        self.w_router.scaled_add(-learning_rate, &gradients[idx]);
        idx += 1;
        self.w_engram_proj
            .scaled_add(-learning_rate, &gradients[idx]);
        idx += 1;
        self.w_titans_proj
            .scaled_add(-learning_rate, &gradients[idx]);
        idx += 1;

        self.engram_ratio_raw -= learning_rate * gradients[idx][[0, 0]];
        idx += 1;
        self.surprise_decay_raw -= learning_rate * gradients[idx][[0, 0]];
        idx += 1;
        self.forget_gate_raw -= learning_rate * gradients[idx][[0, 0]];
        idx += 1;
        self.adaptive_gate_threshold_raw -= learning_rate * gradients[idx][[0, 0]];
        idx += 1;

        let engram_grads = &gradients[idx..idx + engram_grad_count];
        self.engram_memory
            .apply_gradients(engram_grads, learning_rate)?;
        idx += engram_grad_count;

        let titans_grads = &gradients[idx..idx + titans_grad_count];
        self.titans_memory
            .apply_gradients(titans_grads, learning_rate)?;

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
