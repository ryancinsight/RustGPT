#![allow(dead_code)]
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    errors::Result,
    llm::Layer,
    model_config::ModelConfig,
};

use super::{
    transformer_block::{FeedForwardVariant, TransformerBlock},
};
use std::sync::RwLock;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LRMConfig {
    pub embed_dim: usize,
    pub num_recursions: usize,
    pub max_supervision_steps: usize,
    pub max_inference_steps: usize,
    pub latent_update_alpha: f32,
    pub min_alpha: f32,
    pub adapt_scale: f32,
}

impl Default for LRMConfig {
    fn default() -> Self {
        Self {
            embed_dim: 64,
            num_recursions: 1,
            max_supervision_steps: 1,
            max_inference_steps: 1,
            latent_update_alpha: 0.05,
            min_alpha: 0.02,
            adapt_scale: 20.0,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LRM {
    pub transformer: TransformerBlock,
    config: LRMConfig,
    #[serde(skip_serializing, skip_deserializing)]
    is_training: bool,
    #[serde(skip_serializing, skip_deserializing)]
    cached_input: Option<Array2<f32>>, 
    #[serde(skip_serializing, skip_deserializing)]
    latent_init: Option<LatentInit>, 
    #[serde(skip_serializing, skip_deserializing)]
    cached_core_state: Option<CoreCache>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_supervision_outputs: Vec<Array2<f32>>, 
    #[serde(skip_serializing, skip_deserializing)]
    cached_step_states: Vec<SupervisionStepCache>, 
    pub recursion_metrics: Vec<(f32, f32, f32)>,
    #[serde(skip_serializing, skip_deserializing)]
    param_partitions: RwLock<Option<ParamPartitions>>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_mean_input: Option<Array2<f32>>,
}

#[derive(Clone, Debug, Default)]
struct ParamPartitions {
    attention: usize,
    feedforward: usize,
    pre_ffn_norm: usize,
    pre_attn_norm: usize,
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
        for i in 0..embed_dim { w[[i, i]] = 0.01; }
        let b = Array2::<f32>::zeros((1, embed_dim));
        Self { w, b }
    }
    fn project(&self, mean_input: &Array2<f32>) -> Array2<f32> {
        let mut out = mean_input.dot(&self.w);
        out = &out + &self.b;
        out
    }
}

#[derive(Clone, Debug)]
struct BlockCache {
    input: Array2<f32>,
    norm1_out: Option<Array2<f32>>,
    norm2_out: Option<Array2<f32>>,
}

impl BlockCache {
    pub(crate) fn new_input(input: Array2<f32>) -> Self { Self { input, norm1_out: None, norm2_out: None } }
    fn new_answer(input: Array2<f32>, norm1_out: Array2<f32>, norm2_out: Array2<f32>) -> Self {
        Self { input, norm1_out: Some(norm1_out), norm2_out: Some(norm2_out) }
    }
}

#[derive(Clone, Debug)]
enum CoreCache {
    Transformer(BlockCache),
}

#[derive(Clone, Debug)]
struct PartitionedGrads {
    attn: Vec<Array2<f32>>,
    ffn: Vec<Array2<f32>>,
    pre_ffn: Vec<Array2<f32>>,
    pre_attn: Vec<Array2<f32>>,
}

#[derive(Clone, Debug)]
struct SupervisionStepCache {
    answer_cache: CoreCache,
    recursion_caches: Vec<CoreCache>,
}

impl SupervisionStepCache {
    fn new(answer_cache: CoreCache, recursion_caches: Vec<CoreCache>) -> Self {
        Self { answer_cache, recursion_caches }
    }
}

impl LRM {
    pub fn new(config: LRMConfig) -> Self {
        let tb = crate::transformer::TransformerBlock::new(
            crate::transformer::TransformerBlockConfig {
                embed_dim: config.embed_dim,
                hidden_dim: config.embed_dim * 4,
                num_heads: 8,
                poly_degree: 3,
                max_pos: 1024,
                window_size: Some(16),
                use_moe: false,
                moe_config: None,
                head_selection: crate::mixtures::HeadSelectionStrategy::Fixed { num_active: 8 },
                use_adaptive_window: false,
                min_window_size: 512,
                max_window_size: 4096,
                window_adaptation_strategy: crate::model_config::WindowAdaptationStrategy::SequenceLengthBased,
                entropy_ema_alpha: 0.2,
            },
        );

        Self {
            transformer: tb,
            config: config.clone(),
            is_training: false,
            cached_input: None,
            latent_init: Some(LatentInit::new(config.embed_dim)),
            cached_core_state: None,
            cached_supervision_outputs: Vec::new(),
            cached_step_states: Vec::new(),
            recursion_metrics: Vec::new(),
            param_partitions: RwLock::new(None),
            cached_mean_input: None,
        }
    }

    pub fn from_model_config(config: &ModelConfig) -> Self {
        let c = LRMConfig {
            embed_dim: config.embedding_dim,
            num_recursions: config.trm_num_recursions.unwrap_or(2),
            max_supervision_steps: config.trm_max_supervision_steps.unwrap_or(16),
            max_inference_steps: config.trm_max_inference_steps.unwrap_or(2),
            latent_update_alpha: config.trm_latent_update_alpha.unwrap_or(0.05),
            min_alpha: 0.01,
            adapt_scale: 10.0,
        };
        let mut lrm = Self::new(c);
        lrm.transformer = crate::transformer::TransformerBlock::from_model_config(config, 0);
        lrm
    }

    pub fn set_training_mode(&mut self, training: bool) { self.is_training = training; }
    pub fn set_latent_update_alpha(&mut self, alpha: f32) { self.config.latent_update_alpha = alpha; }
    pub fn get_supervision_outputs(&self) -> &[Array2<f32>] { &self.cached_supervision_outputs }
    pub fn set_recursions(&mut self, n: usize) { self.config.num_recursions = n; }
    pub fn set_supervision_steps(&mut self, n: usize) { self.config.max_supervision_steps = n; }
    pub fn set_inference_steps(&mut self, n: usize) { self.config.max_inference_steps = n; }
    fn get_max_steps(&self) -> usize { if self.is_training { self.config.max_supervision_steps } else { self.config.max_inference_steps } }

    fn sanitize(t: &mut Array2<f32>) { for v in t.iter_mut() { if !v.is_finite() { *v = 0.0; } } }

    pub fn forward_recursive(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if self.config.num_recursions == 0 {
            let out = self.transformer.forward(input);
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
            for r in 0..bsz { acc += input[[r, c]]; }
            mean[[0, c]] = acc / (bsz as f32);
        }
        self.cached_mean_input = Some(mean.clone());
        let mut z = if let Some(ref li) = self.latent_init {
            let z0 = li.project(&mean);
            let mut tiled = Array2::<f32>::zeros((bsz, embed_dim));
            for r in 0..bsz { tiled.row_mut(r).assign(&z0.row(0)); }
            tiled
        } else {
            let li = LatentInit::new(embed_dim);
            let z0 = li.project(&mean);
            self.latent_init = Some(li);
            let mut tiled = Array2::<f32>::zeros((bsz, embed_dim));
            for r in 0..bsz { tiled.row_mut(r).assign(&z0.row(0)); }
            tiled
        };
        Self::sanitize(&mut z);

        let max_steps = self.get_max_steps();
        self.cached_supervision_outputs.clear();
        self.cached_step_states.clear();
        self.cached_core_state = None;

        for _t in 0..max_steps {
            let prev_y = y.clone();
            let mut recursion_caches = Vec::new();

            for _ in 0..self.config.num_recursions {
                let mut combined = &y + &z;
                Self::sanitize(&mut combined);

                let norm1 = self.transformer.pre_attention_norm.forward(&combined);
                let attn = self.transformer.attention.forward(&norm1);
                // metrics collection disabled for performance during recursion
                let residual1 = &combined + &attn;
                let norm2 = self.transformer.pre_ffn_norm.forward(&residual1);
                let ffn = match &mut self.transformer.feedforward {
                    FeedForwardVariant::RichardsGlu(l) => l.forward(&norm2),
                    FeedForwardVariant::MixtureOfExperts(l) => l.forward(&norm2),
                };
                if self.is_training { recursion_caches.push(CoreCache::Transformer(BlockCache::new_answer(combined.clone(), norm1.clone(), norm2.clone()))); }
                let mut new_z = &residual1 + &ffn;
                Self::sanitize(&mut new_z);
                let a_base = self.config.latent_update_alpha;
                let rel = {
                    let diff = (&new_z - &z).mapv(|x| x.abs()).sum();
                    let nz = z.mapv(|x| x.abs()).sum();
                    if nz > 0.0 { diff / nz } else { diff }
                };
                let a = (a_base / (1.0 + rel * self.config.adapt_scale)).max(self.config.min_alpha).min(a_base);
                let r = 1.0 - a;
                if (r - 1.0).abs() > f32::EPSILON { z.mapv_inplace(|v| v * r); }
                z.scaled_add(a, &new_z);
                Self::sanitize(&mut z);
            }

            let mut ans_in = &y + &z;
            Self::sanitize(&mut ans_in);
            let norm1 = self.transformer.pre_attention_norm.forward(&ans_in);
            let attn = self.transformer.attention.forward(&norm1);
            // metrics collection disabled for performance during recursion
            let residual1 = &ans_in + &attn;
            let norm2 = self.transformer.pre_ffn_norm.forward(&residual1);
            let ffn = match &mut self.transformer.feedforward {
                FeedForwardVariant::RichardsGlu(l) => l.forward(&norm2),
                FeedForwardVariant::MixtureOfExperts(l) => l.forward(&norm2),
            };
            let mut new_y = &residual1 + &ffn;
            Self::sanitize(&mut new_y);
            self.cached_core_state = Some(CoreCache::Transformer(BlockCache::new_answer(ans_in.clone(), norm1.clone(), norm2.clone())));
            y = new_y;
            Self::sanitize(&mut y);
            if self.is_training {
                self.cached_supervision_outputs.push(y.clone());
                if let Some(ac) = self.cached_core_state.clone() { self.cached_step_states.push(SupervisionStepCache::new(ac, recursion_caches)); }
            }

            let diff = (&y - &prev_y).mapv(|x| x.abs()).sum();
            let ny = y.mapv(|x| x.abs()).sum();
            let rel = if ny > 0.0 { diff / ny } else { diff };
            if rel < 1e-4 { break; }
        }

        Ok(y)
    }

    #[allow(dead_code)]
    fn backward_block(&mut self, st: &BlockCache, up: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        let _ = self.transformer.forward(&st.input);
        self.transformer.compute_gradients(&st.input, up)
    }

    fn backward_block_cached(&self, st: &BlockCache, up: &Array2<f32>) -> (Array2<f32>, PartitionedGrads) {
        let n1 = st.norm1_out.as_ref().unwrap();
        let n2 = st.norm2_out.as_ref().unwrap();
        let ffn_grads = up.clone();
        let residual1_grads = up.clone();
        let (ffn_input_grad, ffn_param_grads) = match &self.transformer.feedforward {
            FeedForwardVariant::RichardsGlu(l) => l.compute_gradients(n2, &ffn_grads),
            FeedForwardVariant::MixtureOfExperts(l) => l.compute_gradients(n2, &ffn_grads),
        };
        let residual1_total = &residual1_grads + &ffn_input_grad;
        let attn_out_grads = residual1_total.clone();
        let (attn_input_grad, attn_param_grads) = self.transformer.attention.compute_gradients(n1, &attn_out_grads);
        let (norm1_input_grad, pre_attn_param_grads) = self.transformer.pre_attention_norm.compute_gradients(&st.input, &attn_input_grad);
        let final_in = &residual1_total + &norm1_input_grad;
        let pre_ffn_param_grads = self.transformer.pre_ffn_norm.compute_gradients(n2, &ffn_input_grad).1;
        let parts = PartitionedGrads { attn: attn_param_grads, ffn: ffn_param_grads, pre_ffn: pre_ffn_param_grads, pre_attn: pre_attn_param_grads };
        (final_in, parts)
    }

    fn latent_init_gradients(&self, z_grads: &Array2<f32>) -> Option<(Array2<f32>, Array2<f32>)> {
        if self.latent_init.is_none() { return None; }
        let mean = self.cached_mean_input.as_ref()?;
        if z_grads.ncols() != mean.ncols() { return None; }
        // reduce z_grads across batch to (1, embed_dim)
        let mut g = Array2::<f32>::zeros((1, mean.ncols()));
        for c in 0..mean.ncols() {
            let mut acc = 0.0f32;
            for r in 0..z_grads.nrows() { acc += z_grads[[r, c]]; }
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

    fn compute_gradients_lrm(&self, _input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        if self.config.num_recursions == 0 {
            return self.transformer.compute_gradients(_input, output_grads);
        }
        let mut all = Vec::new();
        if let Some(core) = &self.cached_core_state {
            // Derive partitions by computing component gradients separately
            let (final_in, mut pg) = match core {
                CoreCache::Transformer(s) => {
                    let mut grads_attn: Vec<Array2<f32>> = Vec::new();
                    let mut grads_ffn: Vec<Array2<f32>> = Vec::new();
                    let mut grads_pre_ffn: Vec<Array2<f32>> = Vec::new();
                    let mut grads_pre_attn: Vec<Array2<f32>> = Vec::new();
                    let ffn_grads = output_grads.clone();
                    let residual1_grads = output_grads.clone();
                    let (ffn_input_grad, ffn_param_grads) = match &self.transformer.feedforward {
                        FeedForwardVariant::RichardsGlu(l) => l.compute_gradients(s.norm2_out.as_ref().unwrap(), &ffn_grads),
                        FeedForwardVariant::MixtureOfExperts(l) => l.compute_gradients(s.norm2_out.as_ref().unwrap(), &ffn_grads),
                    };
                    grads_ffn.extend(ffn_param_grads);
                    let residual1_total = &residual1_grads + &ffn_input_grad;
                    let attn_out_grads = residual1_total.clone();
                    let (attn_input_grad, attn_param_grads) = self.transformer.attention.compute_gradients(s.norm1_out.as_ref().unwrap(), &attn_out_grads);
                    grads_attn.extend(attn_param_grads);
                    let (norm1_input_grad, pre_attn_param_grads) = self.transformer.pre_attention_norm.compute_gradients(&s.input, &attn_input_grad);
                    grads_pre_attn.extend(pre_attn_param_grads);
                    let final_in = &residual1_total + &norm1_input_grad;
                    let pre_ffn_param_grads = self.transformer.pre_ffn_norm.compute_gradients(s.norm2_out.as_ref().unwrap(), &ffn_input_grad).1;
                    grads_pre_ffn.extend(pre_ffn_param_grads);
                    // Aggregate recursion caches gradients by partition
                    if !self.cached_step_states.is_empty() {
                        let idx = self.cached_step_states.len() - 1;
                        let recs = &self.cached_step_states[idx].recursion_caches;
                        for rec in recs.iter().rev() {
                            let (rec_in, rec_parts) = match rec {
                                CoreCache::Transformer(rs) => self.backward_block_cached(rs, &residual1_total),
                            };
                            for i in 0..grads_attn.len() {
                                if i < rec_parts.attn.len() {
                                    let a = &grads_attn[i];
                                    let b = &rec_parts.attn[i];
                                    if a.raw_dim() == b.raw_dim() { grads_attn[i] = a + b; } else { tracing::warn!(target: "lrm", part = "attn", idx = i, "shape mismatch in recursion aggregation"); }
                                }
                            }
                            for i in 0..grads_ffn.len() {
                                if i < rec_parts.ffn.len() {
                                    let a = &grads_ffn[i];
                                    let b = &rec_parts.ffn[i];
                                    if a.raw_dim() == b.raw_dim() { grads_ffn[i] = a + b; } else { tracing::warn!(target: "lrm", part = "ffn", idx = i, "shape mismatch in recursion aggregation"); }
                                }
                            }
                            for i in 0..grads_pre_ffn.len() {
                                if i < rec_parts.pre_ffn.len() {
                                    let a = &grads_pre_ffn[i];
                                    let b = &rec_parts.pre_ffn[i];
                                    if a.raw_dim() == b.raw_dim() { grads_pre_ffn[i] = a + b; } else { tracing::warn!(target: "lrm", part = "pre_ffn", idx = i, "shape mismatch in recursion aggregation"); }
                                }
                            }
                            for i in 0..grads_pre_attn.len() {
                                if i < rec_parts.pre_attn.len() {
                                    let a = &grads_pre_attn[i];
                                    let b = &rec_parts.pre_attn[i];
                                    if a.raw_dim() == b.raw_dim() { grads_pre_attn[i] = a + b; } else { tracing::warn!(target: "lrm", part = "pre_attn", idx = i, "shape mismatch in recursion aggregation"); }
                                }
                            }
                            let _ = rec_in;
                        }
                    }
                    // Save partitions for apply
                    if let Ok(mut guard) = self.param_partitions.write() {
                        if guard.is_none() {
                            let parts = ParamPartitions {
                                attention: grads_attn.len(),
                                feedforward: grads_ffn.len(),
                                pre_ffn_norm: grads_pre_ffn.len(),
                                pre_attn_norm: grads_pre_attn.len(),
                                latent_w: 0,
                                latent_b: 0,
                            };
                            *guard = Some(parts);
                        }
                    }
                    let mut combined: Vec<Array2<f32>> = Vec::with_capacity(
                        grads_attn.len() + grads_ffn.len() + grads_pre_ffn.len() + grads_pre_attn.len(),
                    );
                    combined.extend(grads_attn);
                    combined.extend(grads_ffn);
                    combined.extend(grads_pre_ffn);
                    combined.extend(grads_pre_attn);
                    (final_in, combined)
                }
            };
            all.append(&mut pg);
            if let Some((gw, gb)) = self.latent_init_gradients(&final_in) {
                all.push(gw);
                all.push(gb);
                if let Ok(mut guard) = self.param_partitions.write() {
                    if let Some(parts) = guard.as_mut() {
                        parts.latent_w = 1;
                        parts.latent_b = 1;
                    } else {
                        *guard = Some(ParamPartitions { attention: 0, feedforward: 0, pre_ffn_norm: 0, pre_attn_norm: 0, latent_w: 1, latent_b: 1 });
                    }
                }
            }
            (final_in.clone(), all)
        } else {
            (output_grads.clone(), Vec::new())
        }
    }

    pub fn compute_gradients_at_step(
        &self,
        step_idx: usize,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        if step_idx >= self.cached_step_states.len() {
            return (output_grads.clone(), Vec::new());
        }
        let step = &self.cached_step_states[step_idx];
        match &step.answer_cache {
            CoreCache::Transformer(s) => {
                let mut grads_attn: Vec<Array2<f32>> = Vec::new();
                let mut grads_ffn: Vec<Array2<f32>> = Vec::new();
                let mut grads_pre_ffn: Vec<Array2<f32>> = Vec::new();
                let mut grads_pre_attn: Vec<Array2<f32>> = Vec::new();
                let ffn_grads = output_grads.clone();
                let residual1_grads = output_grads.clone();
                let (ffn_input_grad, ffn_param_grads) = match &self.transformer.feedforward {
                    FeedForwardVariant::RichardsGlu(l) => l.compute_gradients(s.norm2_out.as_ref().unwrap(), &ffn_grads),
                    FeedForwardVariant::MixtureOfExperts(l) => l.compute_gradients(s.norm2_out.as_ref().unwrap(), &ffn_grads),
                };
                grads_ffn.extend(ffn_param_grads);
                let residual1_total = &residual1_grads + &ffn_input_grad;
                let attn_out_grads = residual1_total.clone();
                let (attn_input_grad, attn_param_grads) = self.transformer.attention.compute_gradients(s.norm1_out.as_ref().unwrap(), &attn_out_grads);
                grads_attn.extend(attn_param_grads);
                let (norm1_input_grad, pre_attn_param_grads) = self.transformer.pre_attention_norm.compute_gradients(&s.input, &attn_input_grad);
                grads_pre_attn.extend(pre_attn_param_grads);
                let final_in = &residual1_total + &norm1_input_grad;
                let pre_ffn_param_grads = self.transformer.pre_ffn_norm.compute_gradients(s.norm2_out.as_ref().unwrap(), &ffn_input_grad).1;
                grads_pre_ffn.extend(pre_ffn_param_grads);
                let parts = ParamPartitions {
                    attention: grads_attn.len(),
                    feedforward: grads_ffn.len(),
                    pre_ffn_norm: grads_pre_ffn.len(),
                    pre_attn_norm: grads_pre_attn.len(),
                    latent_w: 0,
                    latent_b: 0,
                };
                if let Ok(mut guard) = self.param_partitions.write() { *guard = Some(parts); }
                let mut combined: Vec<Array2<f32>> = Vec::with_capacity(
                    grads_attn.len() + grads_ffn.len() + grads_pre_ffn.len() + grads_pre_attn.len(),
                );
                combined.extend(grads_attn);
                combined.extend(grads_ffn);
                combined.extend(grads_pre_ffn);
                combined.extend(grads_pre_attn);
                (final_in, combined)
            }
        }
    }

    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if self.config.num_recursions == 0 {
            return self.transformer.apply_gradients(param_grads, lr);
        }
        if param_grads.is_empty() { return Ok(()); }
        let mut parts = self
            .param_partitions
            .read()
            .ok()
            .and_then(|guard| guard.clone())
            .unwrap_or_else(|| ParamPartitions::default());
        if parts.attention == 0 && parts.feedforward == 0 && parts.pre_ffn_norm == 0 && parts.pre_attn_norm == 0 {
            parts.attention = self.transformer.attention.parameters();
            parts.feedforward = match &self.transformer.feedforward { FeedForwardVariant::RichardsGlu(l) => l.parameters(), FeedForwardVariant::MixtureOfExperts(l) => l.parameters() };
            parts.pre_ffn_norm = self.transformer.pre_ffn_norm.parameters();
            parts.pre_attn_norm = self.transformer.pre_attention_norm.parameters();
            if let Ok(mut guard) = self.param_partitions.write() { *guard = Some(parts.clone()); }
        }
        let mut idx = 0usize;
        let mut take = |n: usize| { let s = idx; idx += n.min(param_grads.len().saturating_sub(idx)); s..idx };
        let r_attn = take(parts.attention);
        let r_ffn = take(parts.feedforward);
        let r_pre_ffn = take(parts.pre_ffn_norm);
        let r_pre_attn = take(parts.pre_attn_norm);
        const EPS: f32 = 1e-6;
        const MIN_SCALE: f32 = 0.8;
        const MAX_SCALE: f32 = 1.2;
        if r_attn.len() == parts.attention {
            let mut gnorm: f32 = 0.0;
            for g in &param_grads[r_attn.clone()] { gnorm += g.iter().map(|&x| x * x).sum::<f32>(); }
            let gnorm = gnorm.sqrt();
            let wnorm = self.transformer.attention.weight_norm();
            let scale = (wnorm / (gnorm + EPS)).clamp(MIN_SCALE, MAX_SCALE);
            let lr_attn = lr * scale;
            self.transformer.attention.apply_gradients(&param_grads[r_attn.clone()], lr_attn)?;
        }
        if r_ffn.len() == parts.feedforward {
            let mut gnorm: f32 = 0.0;
            for g in &param_grads[r_ffn.clone()] { gnorm += g.iter().map(|&x| x * x).sum::<f32>(); }
            let gnorm = gnorm.sqrt();
            let wnorm = match &self.transformer.feedforward {
                FeedForwardVariant::RichardsGlu(l) => l.weight_norm(),
                FeedForwardVariant::MixtureOfExperts(l) => l.weight_norm(),
            };
            let scale = (wnorm / (gnorm + EPS)).clamp(MIN_SCALE, MAX_SCALE);
            let lr_ffn = lr * scale;
            match &mut self.transformer.feedforward {
                FeedForwardVariant::RichardsGlu(l) => l.apply_gradients(&param_grads[r_ffn.clone()], lr_ffn)?,
                FeedForwardVariant::MixtureOfExperts(l) => l.apply_gradients(&param_grads[r_ffn.clone()], lr_ffn)?,
            }
        }
        if r_pre_ffn.len() == parts.pre_ffn_norm {
            let mut gnorm: f32 = 0.0;
            for g in &param_grads[r_pre_ffn.clone()] { gnorm += g.iter().map(|&x| x * x).sum::<f32>(); }
            let gnorm = gnorm.sqrt();
            let wnorm = self.transformer.pre_ffn_norm.weight_norm();
            let scale = (wnorm / (gnorm + EPS)).clamp(MIN_SCALE, MAX_SCALE);
            let lr_pre_ffn = lr * scale;
            self.transformer.pre_ffn_norm.apply_gradients(&param_grads[r_pre_ffn.clone()], lr_pre_ffn)?;
        }
        if r_pre_attn.len() == parts.pre_attn_norm {
            let mut gnorm: f32 = 0.0;
            for g in &param_grads[r_pre_attn.clone()] { gnorm += g.iter().map(|&x| x * x).sum::<f32>(); }
            let gnorm = gnorm.sqrt();
            let wnorm = self.transformer.pre_attention_norm.weight_norm();
            let scale = (wnorm / (gnorm + EPS)).clamp(MIN_SCALE, MAX_SCALE);
            let lr_pre_attn = lr * scale;
            self.transformer.pre_attention_norm.apply_gradients(&param_grads[r_pre_attn.clone()], lr_pre_attn)?;
        }

        if let Some(li) = &mut self.latent_init {
            let r_latent_w = take(parts.latent_w);
            let r_latent_b = take(parts.latent_b);
            if r_latent_w.len() == parts.latent_w {
                let gw = &param_grads[r_latent_w.start];
                if gw.raw_dim() == li.w.raw_dim() {
                    let wnorm: f32 = li.w.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let gnorm: f32 = gw.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let scale = (wnorm / (gnorm + EPS)).clamp(MIN_SCALE, MAX_SCALE);
                    let lr_w = lr * scale;
                    li.w = (&li.w - &(gw * lr_w)).to_owned();
                }
            }
            if r_latent_b.len() == parts.latent_b {
                let gb = &param_grads[r_latent_b.start];
                if gb.raw_dim() == li.b.raw_dim() {
                    let bnorm: f32 = li.b.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let gnorm: f32 = gb.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let scale = (bnorm / (gnorm + EPS)).clamp(MIN_SCALE, MAX_SCALE);
                    let lr_b = lr * scale;
                    li.b = (&li.b - &(gb * lr_b)).to_owned();
                }
            }
        }
        Ok(())
    }

}

impl Layer for LRM {
    fn layer_type(&self) -> &str { "LRM" }
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        match self.forward_recursive(input) { Ok(r) => r, Err(_) => input.clone() }
    }
    fn compute_gradients(&self, input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        let _ = input;
        self.compute_gradients_lrm(input, output_grads)
    }
    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> { self.apply_gradients(param_grads, lr) }
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        if let Some(input) = &self.cached_input {
            let (ig, pg) = self.compute_gradients_lrm(input, grads);
            let _ = self.apply_gradients(&pg, lr);
            ig
        } else { grads.clone() }
    }
    fn parameters(&self) -> usize {
        let base = self.transformer.parameter_count();
        let latent = self.latent_init.as_ref().map(|l| l.w.len() + l.b.len()).unwrap_or(0);
        base + latent
    }
    fn weight_norm(&self) -> f32 { self.transformer.weight_norm() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_lrm_forward_shapes() {
        let mut lrm = LRM::new(LRMConfig { embed_dim: 16, num_recursions: 1, max_supervision_steps: 2, max_inference_steps: 1, latent_update_alpha: 0.05, min_alpha: 0.02, adapt_scale: 20.0 });
        let input = Array2::<f32>::zeros((4, 16));
        let out = lrm.forward(&input);
        assert_eq!(out.shape(), input.shape());
    }
    #[test]
    fn test_lrm_gradients_and_apply() {
        let mut lrm = LRM::new(LRMConfig { embed_dim: 8, num_recursions: 1, max_supervision_steps: 1, max_inference_steps: 1, latent_update_alpha: 0.05, min_alpha: 0.02, adapt_scale: 20.0 });
        let input = Array2::<f32>::zeros((2, 8));
        let out = lrm.forward(&input);
        let grads = Array2::<f32>::ones(out.raw_dim());
        let (in_grad, param_grads) = lrm.compute_gradients(&input, &grads);
        assert_eq!(in_grad.shape(), input.shape());
        if !param_grads.is_empty() {
            let _ = lrm.apply_gradients(&param_grads, 1e-3);
        }
    }
}