#![allow(dead_code)]
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    attention::poly_attention::PolyAttention,
    errors::Result,
    network::Layer,
    model_config::ModelConfig,
    transformer::{
        diffusion_block::{DiffusionBlock, DiffusionBlockConfig, DiffusionCachedIntermediates},
        transformer_block::{TransformerBlock, TransformerBlockConfig, CachedIntermediates as TransformerCachedIntermediates},
    },
};

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
                head_selection: crate::mixtures::HeadSelectionStrategy::Fixed { num_active: 8 },
                use_adaptive_window: false,
                min_window_size: 512,
                max_window_size: 4096,
                window_adaptation_strategy: crate::model_config::WindowAdaptationStrategy::SequenceLengthBased,
                entropy_ema_alpha: 0.2,
            }),
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
pub enum RecursiveBlockVariant {
    Transformer(TransformerBlock),
    Diffusion(DiffusionBlock),
}

impl RecursiveBlockVariant {
    fn forward_step(&mut self, input: &Array2<f32>, step: usize) -> Array2<f32> {
        match self {
            Self::Transformer(b) => b.forward(input),
            Self::Diffusion(b) => b.forward_with_timestep(input, step),
        }
    }

    fn compute_gradients(&self, input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
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
struct SupervisionStepCache {
    answer_cache: CoreCache,
    initial_z: Array2<f32>,
    y: Array2<f32>,
}

impl SupervisionStepCache {
    fn new(answer_cache: CoreCache, initial_z: Array2<f32>, y: Array2<f32>) -> Self {
        Self { answer_cache, initial_z, y }
    }
}

pub struct PolyAttentionReadGuard<'a> {
    guard: RwLockReadGuard<'a, RecursiveBlockVariant>,
}

impl<'a> std::ops::Deref for PolyAttentionReadGuard<'a> {
    type Target = PolyAttention;
    fn deref(&self) -> &Self::Target {
        match &*self.guard {
            RecursiveBlockVariant::Transformer(b) => &b.attention,
            RecursiveBlockVariant::Diffusion(b) => &b.attention,
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
            RecursiveBlockVariant::Transformer(b) => &b.attention,
            RecursiveBlockVariant::Diffusion(b) => &b.attention,
        }
    }
}

impl<'a> std::ops::DerefMut for PolyAttentionWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut *self.guard {
            RecursiveBlockVariant::Transformer(b) => &mut b.attention,
            RecursiveBlockVariant::Diffusion(b) => &mut b.attention,
        }
    }
}

impl LRM {
    pub fn new(config: LRMConfig) -> Self {
        let block = match &config.block_config {
            BlockTypeConfig::Transformer(c) => RecursiveBlockVariant::Transformer(TransformerBlock::new(c.clone())),
            BlockTypeConfig::Diffusion(c) => RecursiveBlockVariant::Diffusion(DiffusionBlock::new(c.clone())),
        };

        Self {
            block: RwLock::new(block),
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
        // Default to Transformer for now if not specified, or infer from config
        // Assuming Transformer base for standard LRM usage unless specified otherwise
        let block_config = BlockTypeConfig::Transformer(TransformerBlockConfig {
            embed_dim: config.embedding_dim,
            hidden_dim: config.hidden_dim,
            num_heads: config.get_num_heads(),
            poly_degree: config.get_poly_degree_p(),
            max_pos: config.max_seq_len,
            window_size: config.window_size,
            use_moe: config.moe_router.is_some(),
            moe_config: None,
            head_selection: config.head_selection.clone(),
            use_adaptive_window: config.use_adaptive_window,
            min_window_size: config.min_window_size,
            max_window_size: config.max_window_size,
            window_adaptation_strategy: config.window_adaptation_strategy,
            entropy_ema_alpha: config.entropy_ema_alpha,
        });

        let c = LRMConfig {
            block_config,
            embed_dim: config.embedding_dim,
            num_recursions: config.trm_num_recursions.unwrap_or(2),
            max_supervision_steps: config.trm_max_supervision_steps.unwrap_or(16),
            max_inference_steps: config.trm_max_inference_steps.unwrap_or(2),
            latent_update_alpha: config.trm_latent_update_alpha.unwrap_or(0.05),
            min_alpha: 0.01,
            adapt_scale: 10.0,
        };
        Self::new(c)
    }

    pub fn attention(&self) -> PolyAttentionReadGuard {
        PolyAttentionReadGuard {
            guard: self.block.read().unwrap(),
        }
    }

    pub fn attention_mut(&self) -> PolyAttentionWriteGuard {
        PolyAttentionWriteGuard {
            guard: self.block.write().unwrap(),
        }
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
            let out = self.block.write().unwrap().forward_step(input, 0);
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
            let initial_z = z.clone();
            
            // Run recursions (don't capture caches during forward pass to save memory)
            let _ = self.run_recursions(&y, &mut z, false);

            let mut ans_in = &y + &z;
            Self::sanitize(&mut ans_in);
            
            let new_y = self.block.write().unwrap().forward_step(&ans_in, 0); // Final answer step
            
            if self.is_training {
                if let Some(cache) = self.block.read().unwrap().get_cache() {
                    self.cached_core_state = Some(cache.clone());
                    // Store initial_z and y instead of full recursion caches (Gradient Checkpointing)
                    self.cached_step_states.push(SupervisionStepCache::new(cache, initial_z, y.clone()));
                }
                self.cached_supervision_outputs.push(new_y.clone());
            }
            
            y = new_y;
            Self::sanitize(&mut y);

            let diff = (&y - &prev_y).mapv(|x| x.abs()).sum();
            let ny = y.mapv(|x| x.abs()).sum();
            let rel = if ny > 0.0 { diff / ny } else { diff };
            if rel < 1e-4 { break; }
        }

        Ok(y)
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

    fn compute_gradients_from_cache(
        &self,
        step_cache: &SupervisionStepCache,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let mut all = Vec::new();
        
        // 1. Backward through final answer step
        self.block.read().unwrap().set_cache(Some(step_cache.answer_cache.clone()));
        
        let input_to_block = match &step_cache.answer_cache {
            CoreCache::Transformer(c) => &c.0,
            CoreCache::Diffusion(c) => &c.input,
        };
        
        let (d_ans_in, block_grads) = self.block.read().unwrap().compute_gradients(input_to_block, output_grads);
        all.extend(block_grads);
        
        // 2. Backward through recursions
        // Gradient Checkpointing: Re-run forward pass to generate caches
        let mut z_replay = step_cache.initial_z.clone();
        
        // No unsafe needed anymore!
        let recs = self.run_recursions(&step_cache.y, &mut z_replay, true);
        
        // d_ans_in flows back to y and z.
        // d_y = d_ans_in, d_z = d_ans_in
        let mut d_z = d_ans_in.clone();
        let mut d_y = d_ans_in.clone();
        
        for rec in recs.iter().rev() {
            self.block.read().unwrap().set_cache(Some(rec.clone()));
            let rec_input = match rec {
                CoreCache::Transformer(c) => &c.0,
                CoreCache::Diffusion(c) => &c.input,
            };
            
            // Gradient of z update: z_new = (1-a)z + a*block_out
            // d_block_out = d_z * a
            let a = self.config.latent_update_alpha; 
            let d_block_out = d_z.mapv(|x| x * a);
            
            let (d_combined, rec_grads) = self.block.read().unwrap().compute_gradients(rec_input, &d_block_out);
            
            // Accumulate block grads
            // Note: We need to be careful about the order of gradients in 'all'.
            // If we just extend, we get [answer_grads, rec_N_grads, rec_N-1_grads...]
            // But apply_gradients expects a summed gradient for the shared block.
            // So we should sum them up.
            
            if all.len() == rec_grads.len() {
                for (bg, rg) in all.iter_mut().zip(rec_grads.iter()) {
                    *bg = bg as &Array2<f32> + rg;
                }
            } else {
                // Should not happen if block structure is constant
                tracing::warn!("Gradient length mismatch in LRM recursion");
            }
            
            // d_combined = d_y + d_z
            // z update: z = (1-a)z + a*block_out
            // d_z_prev = d_z * (1-a) + d_combined_z
            // d_combined = d_y + d_z (since input was y+z)
            d_z = &d_z * (1.0 - a) + &d_combined;
            d_y = &d_y + &d_combined;

            // Gradient clipping to prevent explosion during BPTT
            // This is crucial for LRM stability during instruction tuning
            let clip_val = 1.0f32;
            d_z.mapv_inplace(|x| x.clamp(-clip_val, clip_val));
            d_y.mapv_inplace(|x| x.clamp(-clip_val, clip_val));
        }

        // Normalize accumulated gradients by the number of contributions (1 final + N recursions)
        // This prevents gradient magnitude from scaling linearly with recursion depth
        let num_contributions = 1.0 + recs.len() as f32;
        if num_contributions > 1.0 {
            for g in all.iter_mut() {
                g.mapv_inplace(|x| x / num_contributions);
            }
        }
        
        if let Some((gw, gb)) = self.latent_init_gradients(&d_z) {
            all.push(gw);
            all.push(gb);
            if let Ok(mut guard) = self.param_partitions.write() {
                *guard = Some(ParamPartitions { 
                    block: all.len() - 2,
                    latent_w: 1, 
                    latent_b: 1 
                });
            }
        } else {
             if let Ok(mut guard) = self.param_partitions.write() {
                *guard = Some(ParamPartitions { 
                    block: all.len(),
                    latent_w: 0, 
                    latent_b: 0 
                });
            }
        }
        
        (d_y, all)
    }

    fn compute_gradients_lrm(&self, _input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        if self.config.num_recursions == 0 {
            return self.block.read().unwrap().compute_gradients(_input, output_grads);
        }
        
        // Use the last step state for the main backward pass
        if let Some(last_step) = self.cached_step_states.last() {
            self.compute_gradients_from_cache(last_step, output_grads)
        } else {
            (output_grads.clone(), Vec::new())
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
            tracing::warn!("compute_gradients_at_step called with invalid index {}", step_idx);
            (output_grads.clone(), Vec::new())
        }
    }

    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if self.config.num_recursions == 0 {
            return self.block.write().unwrap().apply_gradients(param_grads, lr);
        }
        if param_grads.is_empty() { return Ok(()); }
        
        let parts = self.param_partitions.read().unwrap().clone().unwrap_or_default();
        
        let mut idx = 0;
        let mut next_slice = |count: usize| {
            let end = idx + count;
            let slice = &param_grads[idx..end];
            idx = end;
            slice
        };
        
        let block_grads = next_slice(parts.block);
        self.block.write().unwrap().apply_gradients(block_grads, lr)?;
        
        if let Some(li) = &mut self.latent_init {
            if parts.latent_w > 0 {
                let gw = &param_grads[idx];
                idx += 1;
                li.w = &li.w - &(gw * lr);
            }
            if parts.latent_b > 0 {
                let gb = &param_grads[idx];
                idx += 1;
                li.b = &li.b - &(gb * lr);
            }
        }
        
        Ok(())
    }

    fn run_recursions(&self, y: &Array2<f32>, z: &mut Array2<f32>, capture_caches: bool) -> Vec<CoreCache> {
        let mut caches = Vec::new();
        for r_step in 0..self.config.num_recursions {
            let mut combined = y + &*z;
            Self::sanitize(&mut combined);
            let block_out = self.block.write().unwrap().forward_step(&combined, r_step);
            
            if capture_caches {
                if let Some(cache) = self.block.read().unwrap().get_cache() {
                    caches.push(cache);
                }
            }
            
            let mut new_z = block_out;
            Self::sanitize(&mut new_z);
            
            let a_base = self.config.latent_update_alpha;
            let rel = {
                let diff = (&new_z - &*z).mapv(|x| x.abs()).sum();
                let nz = z.mapv(|x| x.abs()).sum();
                if nz > 0.0 { diff / nz } else { diff }
            };
            let a = (a_base / (1.0 + rel * self.config.adapt_scale)).max(self.config.min_alpha).min(a_base);
            let r = 1.0 - a;
            if (r - 1.0).abs() > f32::EPSILON { z.mapv_inplace(|v| v * r); }
            z.scaled_add(a, &new_z);
            Self::sanitize(z);
        }
        caches
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
        let base = self.block.read().unwrap().parameters();
        let latent = self.latent_init.as_ref().map(|l| l.w.len() + l.b.len()).unwrap_or(0);
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
}