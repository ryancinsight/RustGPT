#![allow(dead_code)]
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use ndarray::{Array2, Zip};
use serde::{Deserialize, Serialize};

use crate::{
    common::errors::Result,
    domain::{
        attention::poly_attention::PolyAttention,
        layers::{
            diffusion::{DiffusionBlock, DiffusionBlockConfig, DiffusionCachedIntermediates},
            transformer::{
                TransformerBlock, TransformerBlockConfig,
                block::CachedIntermediates as TransformerCachedIntermediates,
            },
        },
        mixtures::MixtureOfDepthsConfig,
        models::config::ModelConfig,
        network::Layer,
    },
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HaltingConfig {
    /// Enable ACT-style halting / mixture-of-depth behavior.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// If true, the output is the ACT-weighted average across refinement steps.
    /// If false, the output is the final step state (still uses halting for early stop).
    #[serde(default = "default_true")]
    pub act_weighted_output: bool,

    /// Halting epsilon: treat tokens as halted once cumulative weight >= 1 - epsilon.
    #[serde(default = "default_halting_epsilon")]
    pub epsilon: f32,

    /// Convergence threshold used to derive a halting probability per token.
    /// Smaller rel-change => higher stop probability.
    #[serde(default = "default_halting_threshold")]
    pub threshold: f32,

    /// Slope for the sigmoid used to map (threshold - rel) to a halting probability.
    #[serde(default = "default_halting_slope")]
    pub slope: f32,
}

fn default_true() -> bool {
    true
}

fn default_halting_epsilon() -> f32 {
    0.01
}

fn default_halting_threshold() -> f32 {
    5e-4
}

fn default_halting_slope() -> f32 {
    12.0
}

impl Default for HaltingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            act_weighted_output: true,
            epsilon: default_halting_epsilon(),
            threshold: default_halting_threshold(),
            slope: default_halting_slope(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BlockTypeConfig {
    Transformer(TransformerBlockConfig),
    Diffusion(DiffusionBlockConfig),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LRMConfig {
    pub block_config: BlockTypeConfig,
    pub embed_dim: usize,
    pub num_recursions: usize,
    pub max_supervision_steps: usize,
    pub max_inference_steps: usize,
    pub latent_update_alpha: f32,
    pub min_alpha: f32,
    pub adapt_scale: f32,

    #[serde(default)]
    pub halting: HaltingConfig,

    #[serde(default)]
    pub mixture_of_depths: MixtureOfDepthsConfig,
}

impl Default for LRMConfig {
    fn default() -> Self {
        Self {
            block_config: BlockTypeConfig::Transformer(TransformerBlockConfig {
                embed_dim: 64,
                hidden_dim: 256,
                num_heads: 8,
                poly_degree: 3,
                max_pos: 1024,
                window_size: Some(16),
                use_moe: false,
                moe_config: None,
                head_selection: crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 8 },
                moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
                temporal_mixing: crate::domain::models::config::TemporalMixingType::Attention,
                use_adaptive_window: false,
                min_window_size: 512,
                max_window_size: 4096,
                window_adaptation_strategy:
                    crate::domain::models::config::WindowAdaptationStrategy::SequenceLengthBased,
                entropy_ema_alpha: 0.2,
                use_advanced_adaptive_residuals: true,
                titan_memory: crate::domain::models::config::TitanMemoryConfig::default(),
                eprop_adaptor: None,
            }),
            embed_dim: 64,
            num_recursions: 1,
            max_supervision_steps: 1,
            max_inference_steps: 1,
            latent_update_alpha: 0.05,
            min_alpha: 0.02,
            adapt_scale: 20.0,
            halting: HaltingConfig::default(),
            mixture_of_depths: MixtureOfDepthsConfig::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RecursiveBlockVariant {
    Transformer(Box<TransformerBlock>),
    Diffusion(Box<DiffusionBlock>),
}

impl RecursiveBlockVariant {
    fn forward_step(&mut self, input: &Array2<f32>, step: usize) -> Array2<f32> {
        match self {
            Self::Transformer(b) => b.forward(input),
            Self::Diffusion(b) => b.forward_with_timestep(input, step),
        }
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            Self::Transformer(b) => b.compute_gradients(input, output_grads),
            Self::Diffusion(b) => b.compute_gradients(input, output_grads),
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        match self {
            Self::Transformer(b) => b.apply_gradients(param_grads, lr),
            Self::Diffusion(b) => b.apply_gradients(param_grads, lr),
        }
    }

    fn parameters(&self) -> usize {
        match self {
            Self::Transformer(b) => b.parameter_count(),
            Self::Diffusion(b) => b.parameters(),
        }
    }

    fn weight_norm(&self) -> f32 {
        match self {
            Self::Transformer(b) => b.weight_norm(),
            Self::Diffusion(b) => b.weight_norm(),
        }
    }

    fn get_cache(&self) -> Option<CoreCache> {
        match self {
            Self::Transformer(b) => b.get_cache().map(CoreCache::Transformer),
            Self::Diffusion(b) => b.get_cache().map(CoreCache::Diffusion),
        }
    }

    fn set_cache(&self, cache: Option<CoreCache>) {
        match (self, cache) {
            (Self::Transformer(b), Some(CoreCache::Transformer(c))) => b.set_cache(Some(c)),
            (Self::Transformer(b), None) => b.set_cache(None),
            (Self::Diffusion(b), Some(CoreCache::Diffusion(c))) => b.set_cache(Some(c)),
            (Self::Diffusion(b), None) => b.set_cache(None),
            _ => tracing::warn!("Mismatched cache type in RecursiveBlockVariant::set_cache"),
        }
    }

    fn set_incoming_similarity_context(&mut self, context: Option<&Array2<f32>>) {
        match self {
            Self::Transformer(b) => b.set_incoming_similarity_context(context),
            Self::Diffusion(b) => b.set_incoming_similarity_context(context),
        }
    }

    fn activation_similarity_matrix(&self) -> &Array2<f32> {
        match self {
            Self::Transformer(b) => b.activation_similarity_matrix(),
            Self::Diffusion(b) => b.activation_similarity_matrix(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum CoreCache {
    Transformer(TransformerCachedIntermediates),
    Diffusion(DiffusionCachedIntermediates),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LRM {
    pub block: RwLock<RecursiveBlockVariant>,
    config: LRMConfig,
    #[serde(skip_serializing, skip_deserializing)]
    is_training: bool,
    #[serde(skip_serializing, skip_deserializing)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    latent_init: Option<LatentInit>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_supervision_outputs: Vec<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_step_states: Vec<SupervisionStepCache>,
    pub recursion_metrics: Vec<(f32, f32, f32)>,
    #[serde(skip_serializing, skip_deserializing)]
    param_partitions: RwLock<Option<ParamPartitions>>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_mean_input: Option<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    incoming_similarity_context: Option<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    activation_similarity_matrix: Array2<f32>,
}

#[derive(Clone, Debug, Default)]
struct ParamPartitions {
    block: usize,
    latent_w: usize,
    latent_b: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LatentInit {
    w: Array2<f32>,
    b: Array2<f32>,
}

impl LatentInit {
    fn new(embed_dim: usize) -> Self {
        let mut w = Array2::<f32>::zeros((embed_dim, embed_dim));
        for i in 0..embed_dim {
            w[[i, i]] = 0.01;
        }
        let b = Array2::<f32>::zeros((1, embed_dim));
        Self { w, b }
    }

    fn project(&self, mean_input: &Array2<f32>) -> Array2<f32> {
        let mut out = mean_input.dot(&self.w);
        out += &self.b;
        out
    }
}

#[derive(Clone, Debug)]
struct SupervisionStepCache {
    answer_cache: CoreCache,
    initial_z: Array2<f32>,
    y: Array2<f32>,

    /// ACT-style output weight for this refinement step (shape: (seq_len, 1)).
    /// Present when dynamic halting is enabled.
    halt_weight: Option<Array2<f32>>,
}

impl SupervisionStepCache {
    fn new(
        answer_cache: CoreCache,
        initial_z: Array2<f32>,
        y: Array2<f32>,
        halt_weight: Option<Array2<f32>>,
    ) -> Self {
        Self {
            answer_cache,
            initial_z,
            y,
            halt_weight,
        }
    }
}

pub struct PolyAttentionReadGuard<'a> {
    guard: RwLockReadGuard<'a, RecursiveBlockVariant>,
}

impl<'a> std::ops::Deref for PolyAttentionReadGuard<'a> {
    type Target = PolyAttention;

    fn deref(&self) -> &Self::Target {
        match &*self.guard {
            RecursiveBlockVariant::Transformer(b) => match &b.temporal_mixing().temporal_mixing {
                crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) => &**attn,
                _ => panic!("LRM attention() called but TransformerBlock is not using attention"),
            },
            RecursiveBlockVariant::Diffusion(b) => match &b.temporal_mixing.temporal_mixing {
                crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) => &**attn,
                _ => panic!("LRM attention() called but DiffusionBlock is not using attention"),
            },
        }
    }
}

pub struct PolyAttentionWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, RecursiveBlockVariant>,
}

impl<'a> std::ops::Deref for PolyAttentionWriteGuard<'a> {
    type Target = PolyAttention;

    fn deref(&self) -> &Self::Target {
        match &*self.guard {
            RecursiveBlockVariant::Transformer(b) => match &b.temporal_mixing().temporal_mixing {
                crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) => &**attn,
                _ => {
                    panic!("LRM attention_mut() called but TransformerBlock is not using attention")
                }
            },
            RecursiveBlockVariant::Diffusion(b) => match &b.temporal_mixing.temporal_mixing {
                crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) => &**attn,
                _ => panic!("LRM attention_mut() called but DiffusionBlock is not using attention"),
            },
        }
    }
}

impl<'a> std::ops::DerefMut for PolyAttentionWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut *self.guard {
            RecursiveBlockVariant::Transformer(b) => match &mut b.temporal_mixing_mut().temporal_mixing {
                crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) => &mut **attn,
                _ => {
                    panic!("LRM attention_mut() called but TransformerBlock is not using attention")
                }
            },
            RecursiveBlockVariant::Diffusion(b) => match &mut b.temporal_mixing.temporal_mixing {
                crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) => &mut **attn,
                _ => panic!("LRM attention_mut() called but DiffusionBlock is not using attention"),
            },
        }
    }
}

impl LRM {
    pub fn new(config: LRMConfig) -> Self {
        let block = match &config.block_config {
            BlockTypeConfig::Transformer(c) => {
                RecursiveBlockVariant::Transformer(Box::new(TransformerBlock::new(c.clone())))
            }
            BlockTypeConfig::Diffusion(c) => {
                RecursiveBlockVariant::Diffusion(Box::new(DiffusionBlock::new(c.clone())))
            }
        };

        Self {
            block: RwLock::new(block),
            config: config.clone(),
            is_training: false,
            cached_input: None,
            latent_init: Some(LatentInit::new(config.embed_dim)),
            cached_supervision_outputs: Vec::new(),
            cached_step_states: Vec::new(),
            recursion_metrics: Vec::new(),
            param_partitions: RwLock::new(None),
            cached_mean_input: None,
            incoming_similarity_context: None,
            activation_similarity_matrix: Array2::zeros((config.embed_dim, config.embed_dim)),
        }
    }

    pub fn from_model_config(config: &ModelConfig) -> Self {
        let block_config = if config.trm_use_diffusion {
            BlockTypeConfig::Diffusion(DiffusionBlockConfig {
                embed_dim: config.embedding_dim,
                hidden_dim: config.hidden_dim,
                num_heads: config.get_num_heads(),
                num_timesteps: 1000,
                noise_schedule: config.diffusion_noise_schedule.clone(),
                prediction_target: config.diffusion_prediction_target.clone(),
                timestep_strategy: config.diffusion_timestep_strategy,
                causal_attention: false,
                window_size: config.window_size,
                use_adaptive_window: config.use_adaptive_window,
                discrete_masked: false,
                poly_degree: config.get_poly_degree_p(),
                max_pos: config.max_seq_len,
                use_moe: config.moe_router.is_some(),
                moe_config: config
                    .moe_router
                    .as_ref()
                    .map(crate::domain::mixtures::moe::ExpertRouterConfig::from_router),
                head_selection: config.head_selection.clone(),
                moh_threshold_modulation: config.moh_threshold_modulation.clone(),
                titan_memory: config.titan_memory.clone(),
                time_embed_dim: config.embedding_dim,
                mask_token_id: None,
                temporal_mixing: config.temporal_mixing,
                use_advanced_adaptive_residuals: true,
                edm_sigma_data: crate::domain::layers::diffusion::EDM_SIGMA_DATA_DEFAULT,
                sampler: Default::default(),
                guidance: None,
                loss_weighting: Default::default(),
                use_p2_weighting: false,
                use_snr_weighting: false,
                adaptive_guidance: false,
                min_guidance_scale: 1.0,
                max_guidance_scale: 10.0,
                ddim_steps_policy: Default::default(),
            })
        } else {
            BlockTypeConfig::Transformer(TransformerBlockConfig {
                embed_dim: config.embedding_dim,
                hidden_dim: config.hidden_dim,
                num_heads: config.get_num_heads(),
                poly_degree: config.get_poly_degree_p(),
                max_pos: config.max_seq_len,
                window_size: config.window_size,
                use_moe: config.moe_router.is_some(),
                moe_config: config
                    .moe_router
                    .as_ref()
                    .map(crate::domain::mixtures::moe::ExpertRouterConfig::from_router),
                head_selection: config.head_selection.clone(),
                moh_threshold_modulation: config.moh_threshold_modulation.clone(),
                temporal_mixing: config.temporal_mixing,
                use_adaptive_window: config.use_adaptive_window,
                min_window_size: config.min_window_size,
                max_window_size: config.max_window_size,
                window_adaptation_strategy: config.window_adaptation_strategy,
                entropy_ema_alpha: config.entropy_ema_alpha,
                use_advanced_adaptive_residuals: true,
                titan_memory: config.titan_memory.clone(),
                eprop_adaptor: None,
            })
        };

        let c = LRMConfig {
            block_config,
            embed_dim: config.embedding_dim,
            num_recursions: config.trm_num_recursions.unwrap_or(2),
            max_supervision_steps: config.trm_max_supervision_steps.unwrap_or(16),
            max_inference_steps: config.trm_max_inference_steps.unwrap_or(2),
            latent_update_alpha: config.trm_latent_update_alpha.unwrap_or(0.05),
            min_alpha: 0.01,
            adapt_scale: 10.0,
            halting: HaltingConfig::default(),
            mixture_of_depths: MixtureOfDepthsConfig::default(),
        };
        Self::new(c)
    }

    pub fn max_seq_len(&self) -> Option<usize> {
        match &self.config.block_config {
            BlockTypeConfig::Transformer(cfg) => Some(cfg.max_pos),
            BlockTypeConfig::Diffusion(cfg) => Some(cfg.max_pos),
        }
    }

    pub fn attention(&self) -> PolyAttentionReadGuard<'_> {
        PolyAttentionReadGuard {
            guard: self.block.read().unwrap(),
        }
    }

    pub fn attention_mut(&self) -> PolyAttentionWriteGuard<'_> {
        PolyAttentionWriteGuard {
            guard: self.block.write().unwrap(),
        }
    }

    pub fn set_training_mode(&mut self, training: bool) {
        self.is_training = training;
    }

    pub fn set_latent_update_alpha(&mut self, alpha: f32) {
        self.config.latent_update_alpha = alpha;
    }

    pub fn get_supervision_outputs(&self) -> &[Array2<f32>] {
        &self.cached_supervision_outputs
    }

    pub fn set_recursions(&mut self, n: usize) {
        self.config.num_recursions = n;
    }

    pub fn set_supervision_steps(&mut self, n: usize) {
        self.config.max_supervision_steps = n;
    }

    pub fn set_inference_steps(&mut self, n: usize) {
        self.config.max_inference_steps = n;
    }

    pub fn activation_similarity_matrix(&self) -> &Array2<f32> {
        &self.activation_similarity_matrix
    }

    pub fn set_incoming_similarity_context(&mut self, context: Option<&Array2<f32>>) {
        if let Some(ctx) = context {
            if ctx.nrows() != self.config.embed_dim || ctx.ncols() != self.config.embed_dim {
                self.incoming_similarity_context = None;
                return;
            }

            if let Some(existing) = self.incoming_similarity_context.as_mut() {
                if existing.dim() == ctx.dim() {
                    existing.assign(ctx);
                } else {
                    *existing = ctx.clone();
                }
            } else {
                self.incoming_similarity_context = Some(ctx.clone());
            }
        } else {
            self.incoming_similarity_context = None;
        }
    }

    fn get_max_steps(&self) -> usize {
        if self.is_training {
            self.config.max_supervision_steps
        } else {
            self.config.max_inference_steps
        }
    }

    fn sanitize(t: &mut Array2<f32>) {
        for v in t.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }

    pub fn forward_recursive(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if self.config.num_recursions == 0 {
            let mut block_guard = self.block.write().unwrap();
            block_guard.set_incoming_similarity_context(self.incoming_similarity_context.as_ref());
            let out = block_guard.forward_step(input, 0);
            let ctx = block_guard.activation_similarity_matrix().clone();
            if self.activation_similarity_matrix.dim() == ctx.dim() {
                self.activation_similarity_matrix.assign(&ctx);
            } else {
                self.activation_similarity_matrix = ctx;
            }
            return Ok(out);
        }
        let mut y = input.clone();
        Self::sanitize(&mut y);
        // compute mean input across batch
        let embed_dim = self.config.embed_dim;
        let bsz = input.nrows();
        let mut mean = Array2::<f32>::zeros((1, embed_dim));
        for c in 0..embed_dim {
            let mut acc = 0.0f32;
            for r in 0..bsz {
                acc += input[[r, c]];
            }
            mean[[0, c]] = acc / (bsz as f32);
        }
        let mut z = if let Some(ref li) = self.latent_init {
            let z0 = li.project(&mean);
            let mut tiled = Array2::<f32>::zeros((bsz, embed_dim));
            for r in 0..bsz {
                tiled.row_mut(r).assign(&z0.row(0));
            }
            tiled
        } else {
            let li = LatentInit::new(embed_dim);
            let z0 = li.project(&mean);
            self.latent_init = Some(li);
            let mut tiled = Array2::<f32>::zeros((bsz, embed_dim));
            for r in 0..bsz {
                tiled.row_mut(r).assign(&z0.row(0));
            }
            tiled
        };
        // Store mean after using it to avoid a clone.
        self.cached_mean_input = Some(mean);
        Self::sanitize(&mut z);

        let mut max_steps = self.get_max_steps();
        // Mixture-of-Depths: sample a shallower cap during training.
        if self.is_training {
            max_steps = self
                .config
                .mixture_of_depths
                .sample_depth_cap(max_steps)
                .max(1);
        }
        self.cached_supervision_outputs.clear();
        self.cached_step_states.clear();

        // Reuse buffers to reduce per-step allocations.
        // `ans_in` is also used as a scratch buffer for recursion input (combined = y + z).
        let mut ans_in = Array2::<f32>::zeros((bsz, embed_dim));

        // Hold a single write guard across the entire iterative solve.
        // This is the “permission token” approach: acquire permission once, then
        // operate on data many times.
        let mut block_guard = self.block.write().unwrap();
        let mut similarity_ctx = self.incoming_similarity_context.clone();

        // ACT-style halting state (per token).
        let halting_enabled = self.config.halting.enabled;
        let mut halting_sum = if halting_enabled {
            Array2::<f32>::zeros((bsz, 1))
        } else {
            Array2::<f32>::zeros((0, 0))
        };
        let mut y_accum = if halting_enabled && self.config.halting.act_weighted_output {
            Array2::<f32>::zeros((bsz, embed_dim))
        } else {
            Array2::<f32>::zeros((0, 0))
        };

        // Optimization: during inference, when ACT halting is enabled, avoid computing
        // updates for tokens that have already halted.
        let sparse_inference = halting_enabled && !self.is_training;
        let halt_eps = self.config.halting.epsilon.clamp(1e-6, 0.5);

        for t in 0..max_steps {
            let initial_z = if self.is_training {
                Some(z.clone())
            } else {
                None
            };

            if sparse_inference {
                let prev_y = y.clone();
                // Determine which tokens are still active.
                let mut active_rows: Vec<usize> = Vec::new();
                for r in 0..bsz {
                    if halting_sum[[r, 0]] < 1.0 - halt_eps {
                        active_rows.push(r);
                    }
                }

                // If nothing is active, the model has fully halted.
                if active_rows.is_empty() {
                    break;
                }

                // Gather active rows for compute.
                let active_n = active_rows.len();
                let mut prev_y_active = Array2::<f32>::zeros((active_n, embed_dim));
                let mut z_active = Array2::<f32>::zeros((active_n, embed_dim));
                for (i, &r) in active_rows.iter().enumerate() {
                    prev_y_active.row_mut(i).assign(&prev_y.row(r));
                    z_active.row_mut(i).assign(&z.row(r));
                }

                // Run recursions on active rows only.
                let mut scratch_active = Array2::<f32>::zeros((active_n, embed_dim));
                let _ = self.run_recursions_with_guard(
                    &mut block_guard,
                    &prev_y_active,
                    &mut z_active,
                    &mut scratch_active,
                    &mut similarity_ctx,
                    false,
                );

                // Final answer step on active rows only.
                scratch_active.assign(&prev_y_active);
                scratch_active += &z_active;
                Self::sanitize(&mut scratch_active);
                block_guard.set_incoming_similarity_context(similarity_ctx.as_ref());
                let new_y_active = block_guard.forward_step(&scratch_active, 0);
                let ctx = block_guard.activation_similarity_matrix().clone();
                if let Some(existing) = similarity_ctx.as_mut() {
                    if existing.dim() == ctx.dim() {
                        existing.assign(&ctx);
                    } else {
                        *existing = ctx;
                    }
                } else {
                    similarity_ctx = Some(ctx);
                }

                // ACT halting weights for active rows only.
                let mut w = Array2::<f32>::zeros((bsz, 1));
                if halting_enabled {
                    let thr = self.config.halting.threshold.max(0.0);
                    let slope = self.config.halting.slope.max(0.0);
                    let last_step = t + 1 == max_steps;
                    let sigmoid = crate::domain::richards::RichardsCurve::sigmoid(false);

                    for (i, &r) in active_rows.iter().enumerate() {
                        let remaining = (1.0 - halting_sum[[r, 0]]).max(0.0);
                        if remaining <= 0.0 {
                            w[[r, 0]] = 0.0;
                            continue;
                        }

                        if last_step {
                            w[[r, 0]] = remaining;
                            continue;
                        }

                        // rel(token) = sum|dy| / (sum|y| + eps)
                        let mut diff_r = 0.0f32;
                        let mut ny_r = 0.0f32;
                        for c in 0..embed_dim {
                            let a = new_y_active[[i, c]];
                            let b = prev_y_active[[i, c]];
                            diff_r += (a - b).abs();
                            ny_r += a.abs();
                        }
                        let rel_r = diff_r / (ny_r + 1e-6);

                        let p = sigmoid.forward_scalar_f32((thr - rel_r) * slope);
                        let will_finish = halting_sum[[r, 0]] + p >= 1.0 - halt_eps;
                        w[[r, 0]] = if will_finish {
                            remaining
                        } else {
                            p.min(remaining)
                        };
                    }

                    if self.config.halting.act_weighted_output {
                        for (i, &r) in active_rows.iter().enumerate() {
                            let wr = w[[r, 0]];
                            if wr == 0.0 {
                                continue;
                            }
                            for c in 0..embed_dim {
                                y_accum[[r, c]] += wr * new_y_active[[i, c]];
                            }
                        }
                    }

                    // Update halting sums for active rows.
                    for &r in active_rows.iter() {
                        halting_sum[[r, 0]] = (halting_sum[[r, 0]] + w[[r, 0]]).min(1.0);
                    }
                }

                // Scatter active results back into full tensors.
                let mut new_y_full = prev_y;
                for (i, &r) in active_rows.iter().enumerate() {
                    new_y_full.row_mut(r).assign(&new_y_active.row(i));
                    z.row_mut(r).assign(&z_active.row(i));
                }
                y = new_y_full;
                Self::sanitize(&mut y);

                // Early stop once all tokens have halted.
                let mut all_halted = true;
                for r in 0..bsz {
                    if halting_sum[[r, 0]] < 1.0 - halt_eps {
                        all_halted = false;
                        break;
                    }
                }
                if all_halted {
                    break;
                }

                // Sparse inference path fully handled this step.
                continue;
            }

            let prev_y_owned = if self.is_training {
                Some(y.clone())
            } else {
                None
            };
            let prev_y_ref = prev_y_owned.as_ref().unwrap_or(&y);

            // Run recursions (don't capture caches during forward pass to save memory).
            let _ = self.run_recursions_with_guard(
                &mut block_guard,
                prev_y_ref,
                &mut z,
                &mut ans_in,
                &mut similarity_ctx,
                false,
            );

            ans_in.assign(prev_y_ref);
            ans_in += &z;
            Self::sanitize(&mut ans_in);

            // Final answer step.
            block_guard.set_incoming_similarity_context(similarity_ctx.as_ref());
            let new_y = block_guard.forward_step(&ans_in, 0);
            let ctx = block_guard.activation_similarity_matrix().clone();
            if let Some(existing) = similarity_ctx.as_mut() {
                if existing.dim() == ctx.dim() {
                    existing.assign(&ctx);
                } else {
                    *existing = ctx;
                }
            } else {
                similarity_ctx = Some(ctx);
            }
            let answer_cache = block_guard.get_cache();

            // Optional ACT-style halting weights derived from per-token convergence.
            // We intentionally keep this parameter-free and deterministic.
            let mut step_weight: Option<Array2<f32>> = None;
            if halting_enabled {
                let eps = self.config.halting.epsilon.clamp(1e-6, 0.5);
                let thr = self.config.halting.threshold.max(0.0);
                let slope = self.config.halting.slope.max(0.0);

                let mut w = Array2::<f32>::zeros((bsz, 1));
                let last_step = t + 1 == max_steps;

                // Compute per-token rel change and map to a halting probability.
                // rel(token) = sum|dy| / (sum|y| + eps)
                for r in 0..bsz {
                    let mut diff_r = 0.0f32;
                    let mut ny_r = 0.0f32;
                    for c in 0..embed_dim {
                        let a = new_y[[r, c]];
                        let b = prev_y_ref[[r, c]];
                        diff_r += (a - b).abs();
                        ny_r += a.abs();
                    }
                    let rel_r = diff_r / (ny_r + 1e-6);

                    let remaining = (1.0 - halting_sum[[r, 0]]).max(0.0);
                    if remaining <= 0.0 {
                        w[[r, 0]] = 0.0;
                        continue;
                    }

                    // On the last step, force remainder so weights sum to 1.
                    if last_step {
                        w[[r, 0]] = remaining;
                        continue;
                    }

                    // Higher stop probability when rel_r is below threshold.
                    let sigmoid = crate::domain::richards::RichardsCurve::sigmoid(false);
                    let p = sigmoid.forward_scalar_f32((thr - rel_r) * slope);
                    let will_finish = halting_sum[[r, 0]] + p >= 1.0 - eps;
                    w[[r, 0]] = if will_finish {
                        remaining
                    } else {
                        p.min(remaining)
                    };
                }

                // Apply weights to the ACT accumulator.
                if self.config.halting.act_weighted_output {
                    for r in 0..bsz {
                        let wr = w[[r, 0]];
                        if wr == 0.0 {
                            continue;
                        }
                        for c in 0..embed_dim {
                            y_accum[[r, c]] += wr * new_y[[r, c]];
                        }
                    }
                }

                // Update halting sums.
                Zip::from(halting_sum.rows_mut())
                    .and(w.rows())
                    .for_each(|mut hs, wr| {
                        hs[0] = (hs[0] + wr[0]).min(1.0);
                    });

                step_weight = Some(w);
            }

            // Compute a scalar convergence metric (used as a backstop when halting is disabled).
            let mut diff = 0.0f32;
            let mut ny = 0.0f32;
            for (a, b) in new_y.iter().zip(prev_y_ref.iter()) {
                diff += (*a - *b).abs();
                ny += a.abs();
            }
            let rel = if ny > 0.0 { diff / ny } else { diff };

            if self.is_training {
                // Store initial_z and prev_y instead of full recursion caches (Gradient
                // Checkpointing)
                if let (Some(cache), Some(initial_z)) = (answer_cache, initial_z) {
                    let prev_y = prev_y_owned.unwrap_or_else(|| y.clone());
                    self.cached_step_states.push(SupervisionStepCache::new(
                        cache,
                        initial_z,
                        prev_y,
                        step_weight,
                    ));
                } else {
                    // Keep semantics consistent: still advance y even if cache is missing.
                    // prev_y is dropped here.
                }
                self.cached_supervision_outputs.push(new_y.clone());
            }

            y = new_y;
            Self::sanitize(&mut y);
            if halting_enabled {
                // Early stop once all tokens have halted.
                let mut all_halted = true;
                let eps = self.config.halting.epsilon.clamp(1e-6, 0.5);
                for r in 0..bsz {
                    if halting_sum[[r, 0]] < 1.0 - eps {
                        all_halted = false;
                        break;
                    }
                }
                if all_halted {
                    break;
                }
            } else if rel < 1e-4 {
                break;
            }
        }

        let ctx = block_guard.activation_similarity_matrix().clone();
        if self.activation_similarity_matrix.dim() == ctx.dim() {
            self.activation_similarity_matrix.assign(&ctx);
        } else {
            self.activation_similarity_matrix = ctx;
        }

        if halting_enabled && self.config.halting.act_weighted_output {
            Ok(y_accum)
        } else {
            Ok(y)
        }
    }

    fn latent_init_gradients(&self, z_grads: &Array2<f32>) -> Option<(Array2<f32>, Array2<f32>)> {
        self.latent_init.as_ref()?;
        let mean = self.cached_mean_input.as_ref()?;
        if z_grads.ncols() != mean.ncols() {
            return None;
        }
        // reduce z_grads across batch to (1, embed_dim)
        let mut g = Array2::<f32>::zeros((1, mean.ncols()));
        for c in 0..mean.ncols() {
            let mut acc = 0.0f32;
            for r in 0..z_grads.nrows() {
                acc += z_grads[[r, c]];
            }
            g[[0, c]] = acc / (z_grads.nrows() as f32);
        }
        let mut grad_w = Array2::<f32>::zeros((mean.ncols(), mean.ncols()));
        for i in 0..mean.ncols() {
            for j in 0..mean.ncols() {
                grad_w[[i, j]] = mean[[0, i]] * g[[0, j]];
            }
        }
        let grad_b = g;
        Some((grad_w, grad_b))
    }

    fn compute_gradients_from_cache(
        &self,
        step_cache: &SupervisionStepCache,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Hold a single write guard for the entire backward pass.
        // This avoids repeated lock acquisition and removes the need for phased
        // read/write permission switching during checkpoint replay.
        let mut block_guard = self.block.write().unwrap();

        // 1) Backward through recursions
        // Gradient Checkpointing: Re-run forward pass to generate caches
        let mut z_replay = step_cache.initial_z.clone();
        let mut similarity_ctx = self.incoming_similarity_context.clone();

        // Replay the forward recursion to regenerate caches (checkpointing) AND
        // capture the exact per-step adaptive alpha used in the z-update.
        let mut scratch_combined = Array2::<f32>::zeros(step_cache.y.raw_dim());
        let rec_trace = self.run_recursions_trace_with_guard(
            &mut block_guard,
            &step_cache.y,
            &mut z_replay,
            &mut scratch_combined,
            &mut similarity_ctx,
        );

        // 2) Backward through final answer step.
        block_guard.set_incoming_similarity_context(similarity_ctx.as_ref());
        block_guard.set_cache(Some(step_cache.answer_cache.clone()));

        let input_to_block = match &step_cache.answer_cache {
            CoreCache::Transformer(c) => c.input_original.as_ref(),
            CoreCache::Diffusion(c) => c.input_used.as_ref(),
        };

        let (d_ans_in, mut all) = block_guard.compute_gradients(input_to_block, output_grads);

        // d_ans_in flows back to y and z.
        // d_y = d_ans_in, d_z = d_ans_in
        let mut d_z = d_ans_in.clone();
        let mut d_y = d_ans_in;

        // Reuse a temp buffer for d_block_out to avoid per-recursion allocations.
        let mut d_block_out = Array2::<f32>::zeros(d_z.raw_dim());

        // 3) Backprop through replayed recursion caches.
        for (rec, alpha) in rec_trace.iter().rev() {
            block_guard.set_cache(Some(rec.clone()));
            let rec_input = match rec {
                CoreCache::Transformer(c) => c.input_original.as_ref(),
                CoreCache::Diffusion(c) => c.input_used.as_ref(),
            };

            // Gradient of z update (treat alpha as a detached step-size):
            // z_new = (1-a)z + a*block_out
            // d_block_out = d_z * a
            let a = *alpha;
            d_block_out.assign(&d_z);
            d_block_out.mapv_inplace(|x| x * a);

            let (d_combined, rec_grads) = block_guard.compute_gradients(rec_input, &d_block_out);

            // Accumulate block grads into the answer-step grads.
            if all.len() == rec_grads.len() {
                for (bg, rg) in all.iter_mut().zip(rec_grads.iter()) {
                    bg.zip_mut_with(rg, |a, &b| *a += b);
                }
            } else {
                // Should not happen if block structure is constant
                tracing::warn!("Gradient length mismatch in LRM recursion");
            }

            // d_combined = d_y + d_z (since input was y+z)
            // z update: z = (1-a)z + a*block_out
            // d_z_prev = d_z * (1-a) + d_combined
            d_z.mapv_inplace(|x| x * (1.0 - a));
            d_z += &d_combined;
            d_y += &d_combined;

            // Gradient clipping to prevent explosion during BPTT
            // This is crucial for LRM stability during instruction tuning
            let clip_val = 1.0f32;
            d_z.mapv_inplace(|x| x.clamp(-clip_val, clip_val));
            d_y.mapv_inplace(|x| x.clamp(-clip_val, clip_val));
        }

        // Normalize accumulated gradients by the number of contributions (1 final + N recursions)
        // This prevents gradient magnitude from scaling linearly with recursion depth
        let num_contributions = 1.0 + rec_trace.len() as f32;
        if num_contributions > 1.0 {
            for g in all.iter_mut() {
                g.mapv_inplace(|x| x / num_contributions);
            }
        }

        let partitions = if let Some((gw, gb)) = self.latent_init_gradients(&d_z) {
            all.push(gw);
            all.push(gb);
            ParamPartitions {
                block: all.len() - 2,
                latent_w: 1,
                latent_b: 1,
            }
        } else {
            ParamPartitions {
                block: all.len(),
                latent_w: 0,
                latent_b: 0,
            }
        };
        if let Ok(mut guard) = self.param_partitions.write() {
            *guard = Some(partitions);
        }

        (d_y, all)
    }

    fn compute_gradients_lrm(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        if self.config.num_recursions == 0 {
            return self
                .block
                .read()
                .unwrap()
                .compute_gradients(_input, output_grads);
        }

        if self.is_training
            && self.config.halting.enabled
            && self.config.halting.act_weighted_output
        {
            // Full BPTT across outer refinement steps.
            // Output is a weighted sum of step outputs, and later steps depend on earlier y.
            if self.cached_step_states.is_empty() {
                return (output_grads.clone(), Vec::new());
            }

            let mut d_next = Array2::<f32>::zeros(output_grads.raw_dim());
            let mut accumulated_param_grads: Option<Vec<Array2<f32>>> = None;

            for step in self.cached_step_states.iter().rev() {
                let fallback_w;
                let w = match step.halt_weight.as_ref() {
                    Some(w) => w,
                    None => {
                        fallback_w = Array2::<f32>::ones((output_grads.nrows(), 1));
                        &fallback_w
                    }
                };

                // local_grad = output_grads * w (row-wise broadcast)
                let mut local_grad = output_grads.clone();
                for r in 0..local_grad.nrows() {
                    let wr = w[[r, 0]];
                    for c in 0..local_grad.ncols() {
                        local_grad[[r, c]] *= wr;
                    }
                }
                local_grad += &d_next;

                let (d_y, step_param_grads) = self.compute_gradients_from_cache(step, &local_grad);
                d_next = d_y;

                match &mut accumulated_param_grads {
                    None => {
                        accumulated_param_grads = Some(step_param_grads);
                    }
                    Some(acc) => {
                        if acc.len() == step_param_grads.len() {
                            for (a, b) in acc.iter_mut().zip(step_param_grads.iter()) {
                                a.zip_mut_with(b, |x, &y| *x += y);
                            }
                        } else {
                            tracing::warn!(
                                "LRM param gradient length mismatch across refinement steps"
                            );
                        }
                    }
                }
            }

            (d_next, accumulated_param_grads.unwrap_or_default())
        } else {
            // Legacy / faster path: only backprop through the last refinement step.
            if let Some(last_step) = self.cached_step_states.last() {
                self.compute_gradients_from_cache(last_step, output_grads)
            } else {
                (output_grads.clone(), Vec::new())
            }
        }
    }

    pub fn compute_gradients_at_step(
        &self,
        step_idx: usize,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        if self.config.num_recursions == 0 {
            return (output_grads.clone(), Vec::new());
        }

        if step_idx < self.cached_step_states.len() {
            self.compute_gradients_from_cache(&self.cached_step_states[step_idx], output_grads)
        } else {
            tracing::warn!(
                "compute_gradients_at_step called with invalid index {}",
                step_idx
            );
            (output_grads.clone(), Vec::new())
        }
    }

    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if self.config.num_recursions == 0 {
            let res = self.block.write().unwrap().apply_gradients(param_grads, lr);
            // Release forward caches early to reduce peak memory.
            self.cached_input = None;
            self.cached_mean_input = None;
            self.cached_supervision_outputs.clear();
            self.cached_step_states.clear();
            return res;
        }
        if param_grads.is_empty() {
            return Ok(());
        }

        let parts = self
            .param_partitions
            .read()
            .unwrap()
            .clone()
            .unwrap_or_default();

        let mut _idx = 0;
        let mut next_slice = |count: usize| {
            let end = _idx + count;
            let slice = &param_grads[_idx..end];
            _idx = end;
            slice
        };

        let block_grads = next_slice(parts.block);
        self.block
            .write()
            .unwrap()
            .apply_gradients(block_grads, lr)?;

        if let Some(li) = &mut self.latent_init {
            if parts.latent_w > 0 {
                let gw = &param_grads[_idx];
                _idx += 1;
                Zip::from(&mut li.w).and(gw).for_each(|w, &g| *w -= lr * g);
            }
            if parts.latent_b > 0 {
                let gb = &param_grads[_idx];
                _idx += 1;
                Zip::from(&mut li.b).and(gb).for_each(|b, &g| *b -= lr * g);
            }
        }

        // Release caches after gradient application to reduce memory pressure.
        self.cached_input = None;
        self.cached_mean_input = None;
        self.cached_supervision_outputs.clear();
        self.cached_step_states.clear();

        Ok(())
    }

    fn run_recursions(
        &self,
        y: &Array2<f32>,
        z: &mut Array2<f32>,
        capture_caches: bool,
    ) -> Vec<CoreCache> {
        let mut block_guard = self.block.write().unwrap();
        let mut scratch_combined = Array2::<f32>::zeros(y.raw_dim());
        let mut similarity_ctx = self.incoming_similarity_context.clone();
        self.run_recursions_with_guard(
            &mut block_guard,
            y,
            z,
            &mut scratch_combined,
            &mut similarity_ctx,
            capture_caches,
        )
    }

    fn run_recursions_with_guard(
        &self,
        block_guard: &mut RecursiveBlockVariant,
        y: &Array2<f32>,
        z: &mut Array2<f32>,
        scratch_combined: &mut Array2<f32>,
        similarity_ctx: &mut Option<Array2<f32>>,
        capture_caches: bool,
    ) -> Vec<CoreCache> {
        let mut caches = Vec::new();
        if scratch_combined.raw_dim() != y.raw_dim() {
            *scratch_combined = Array2::<f32>::zeros(y.raw_dim());
        }

        for r_step in 0..self.config.num_recursions {
            scratch_combined.assign(y);
            *scratch_combined += &*z;
            Self::sanitize(scratch_combined);
            block_guard.set_incoming_similarity_context(similarity_ctx.as_ref());
            let block_out = block_guard.forward_step(scratch_combined, r_step);
            let ctx = block_guard.activation_similarity_matrix().clone();
            if let Some(existing) = similarity_ctx.as_mut() {
                if existing.dim() == ctx.dim() {
                    existing.assign(&ctx);
                } else {
                    *existing = ctx.clone();
                }
            } else {
                *similarity_ctx = Some(ctx.clone());
            }

            if capture_caches && let Some(cache) = block_guard.get_cache() {
                caches.push(cache);
            }

            let mut new_z = block_out;
            Self::sanitize(&mut new_z);

            let a_base = self.config.latent_update_alpha;
            let mut diff = 0.0f32;
            let mut nz = 0.0f32;
            for (a, b) in new_z.iter().zip(z.iter()) {
                diff += (*a - *b).abs();
                nz += b.abs();
            }
            let rel = if nz > 0.0 { diff / nz } else { diff };
            let a = (a_base / (1.0 + rel * self.config.adapt_scale))
                .max(self.config.min_alpha)
                .min(a_base);
            let r = 1.0 - a;
            if (r - 1.0).abs() > f32::EPSILON {
                z.mapv_inplace(|v| v * r);
            }
            z.scaled_add(a, &new_z);
            Self::sanitize(z);
        }

        caches
    }

    fn run_recursions_trace_with_guard(
        &self,
        block_guard: &mut RecursiveBlockVariant,
        y: &Array2<f32>,
        z: &mut Array2<f32>,
        scratch_combined: &mut Array2<f32>,
        similarity_ctx: &mut Option<Array2<f32>>,
    ) -> Vec<(CoreCache, f32)> {
        let mut trace = Vec::new();
        if scratch_combined.raw_dim() != y.raw_dim() {
            *scratch_combined = Array2::<f32>::zeros(y.raw_dim());
        }

        for r_step in 0..self.config.num_recursions {
            scratch_combined.assign(y);
            *scratch_combined += &*z;
            Self::sanitize(scratch_combined);
            block_guard.set_incoming_similarity_context(similarity_ctx.as_ref());
            let block_out = block_guard.forward_step(scratch_combined, r_step);
            let ctx = block_guard.activation_similarity_matrix().clone();
            if let Some(existing) = similarity_ctx.as_mut() {
                if existing.dim() == ctx.dim() {
                    existing.assign(&ctx);
                } else {
                    *existing = ctx.clone();
                }
            } else {
                *similarity_ctx = Some(ctx.clone());
            }

            let mut new_z = block_out;
            Self::sanitize(&mut new_z);

            let a_base = self.config.latent_update_alpha;
            let mut diff = 0.0f32;
            let mut nz = 0.0f32;
            for (a, b) in new_z.iter().zip(z.iter()) {
                diff += (*a - *b).abs();
                nz += b.abs();
            }
            let rel = if nz > 0.0 { diff / nz } else { diff };
            let a = (a_base / (1.0 + rel * self.config.adapt_scale))
                .max(self.config.min_alpha)
                .min(a_base);

            if let Some(cache) = block_guard.get_cache() {
                trace.push((cache, a));
            }

            let r = 1.0 - a;
            if (r - 1.0).abs() > f32::EPSILON {
                z.mapv_inplace(|v| v * r);
            }
            z.scaled_add(a, &new_z);
            Self::sanitize(z);
        }

        trace
    }
}

impl Layer for LRM {
    fn layer_type(&self) -> &str {
        "LRM"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Only cache input when we truly need it (num_recursions == 0 path).
        self.cached_input = if self.config.num_recursions == 0 {
            Some(input.clone())
        } else {
            None
        };
        match self.forward_recursive(input) {
            Ok(r) => r,
            Err(_) => input.clone(),
        }
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let _ = input;
        self.compute_gradients_lrm(input, output_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        self.apply_gradients(param_grads, lr)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        if self.config.num_recursions == 0 {
            if let Some(input) = &self.cached_input {
                let (ig, pg) = self.compute_gradients_lrm(input, grads);
                let _ = self.apply_gradients(&pg, lr);
                return ig;
            }
            return grads.clone();
        }
        // Recursion mode uses cached_step_states rather than cached_input.
        if let Some(last_step) = self.cached_step_states.last() {
            let (ig, pg) = self.compute_gradients_from_cache(last_step, grads);
            let _ = self.apply_gradients(&pg, lr);
            ig
        } else {
            grads.clone()
        }
    }

    fn parameters(&self) -> usize {
        let base = self.block.read().unwrap().parameters();
        let latent = self
            .latent_init
            .as_ref()
            .map(|l| l.w.len() + l.b.len())
            .unwrap_or(0);
        base + latent
    }

    fn weight_norm(&self) -> f32 {
        let base_sq = self.block.read().unwrap().weight_norm().powi(2);
        let latent_sq = if let Some(li) = &self.latent_init {
            li.w.iter().map(|x| x * x).sum::<f32>() + li.b.iter().map(|x| x * x).sum::<f32>()
        } else {
            0.0
        };
        (base_sq + latent_sq).sqrt()
    }

    fn zero_gradients(&mut self) {
        // LRM doesn't maintain internal gradient state beyond the block
        // The underlying TransformerBlock handles its own gradient state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_lrm_forward_shapes() {
        let mut lrm = LRM::new(LRMConfig::default());
        let input = Array2::<f32>::zeros((4, 64));
        let out = lrm.forward(&input);
        assert_eq!(out.shape(), input.shape());
    }
    #[test]
    fn test_lrm_gradients_and_apply() {
        let mut lrm = LRM::new(LRMConfig::default());
        let input = Array2::<f32>::zeros((2, 64));
        let out = lrm.forward(&input);
        let grads = Array2::<f32>::ones(out.raw_dim());
        let (in_grad, param_grads) = lrm.compute_gradients(&input, &grads);
        assert_eq!(in_grad.shape(), input.shape());
        if !param_grads.is_empty() {
            let _ = lrm.apply_gradients(&param_grads, 1e-3);
        }
    }

    #[test]
    fn test_lrm_training_act_halting_bptt_runs() {
        let cfg = LRMConfig {
            max_supervision_steps: 4,
            max_inference_steps: 2,
            halting: HaltingConfig {
                enabled: true,
                act_weighted_output: true,
                ..Default::default()
            },
            mixture_of_depths: MixtureOfDepthsConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut lrm = LRM::new(cfg);
        lrm.set_training_mode(true);

        let input = Array2::<f32>::zeros((4, 64));
        let out = lrm.forward(&input);
        assert_eq!(out.shape(), input.shape());

        let grads = Array2::<f32>::ones(out.raw_dim());
        let (in_grad, param_grads) = lrm.compute_gradients(&input, &grads);
        assert_eq!(in_grad.shape(), input.shape());
        assert!(!param_grads.is_empty());
        lrm.apply_gradients(&param_grads, 1e-3).unwrap();
    }
}
