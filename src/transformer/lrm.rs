#![allow(dead_code)]
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

use crate::{
    attention::poly_attention::PolyAttention,
    errors::Result,
    llm::Layer,
    model_config::ModelConfig,
    transformer::{
        common::FeedForwardVariant,
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
    pub block: RecursiveBlockVariant,
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
    recursion_caches: Vec<CoreCache>,
}

impl SupervisionStepCache {
    fn new(answer_cache: CoreCache, recursion_caches: Vec<CoreCache>) -> Self {
        Self { answer_cache, recursion_caches }
    }
}

impl LRM {
    pub fn new(config: LRMConfig) -> Self {
        let block = match &config.block_config {
            BlockTypeConfig::Transformer(c) => RecursiveBlockVariant::Transformer(TransformerBlock::new(c.clone())),
            BlockTypeConfig::Diffusion(c) => RecursiveBlockVariant::Diffusion(DiffusionBlock::new(c.clone())),
        };

        Self {
            block,
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

    pub fn attention(&self) -> &PolyAttention {
        match &self.block {
            RecursiveBlockVariant::Transformer(b) => &b.attention,
            RecursiveBlockVariant::Diffusion(b) => &b.attention,
        }
    }

    pub fn attention_mut(&mut self) -> &mut PolyAttention {
        match &mut self.block {
            RecursiveBlockVariant::Transformer(b) => &mut b.attention,
            RecursiveBlockVariant::Diffusion(b) => &mut b.attention,
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
            let out = self.block.forward_step(input, 0);
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

            for r_step in 0..self.config.num_recursions {
                let mut combined = &y + &z;
                Self::sanitize(&mut combined);

                // Forward pass through the block
                // For diffusion, we use r_step as timestep, or maybe _t?
                // Using r_step implies "depth" in the recursion.
                let block_out = self.block.forward_step(&combined, r_step);
                
                if self.is_training {
                    if let Some(cache) = self.block.get_cache() {
                        recursion_caches.push(cache);
                    }
                }

                // Update latent z
                // new_z = block_out (residual + ffn)
                // In original LRM, new_z = residual1 + ffn
                // block_out IS residual1 + ffn (output of block)
                let mut new_z = block_out; 
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
            
            let new_y = self.block.forward_step(&ans_in, 0); // Final answer step
            
            if self.is_training {
                if let Some(cache) = self.block.get_cache() {
                    self.cached_core_state = Some(cache.clone());
                    self.cached_step_states.push(SupervisionStepCache::new(cache, recursion_caches));
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

    fn compute_gradients_lrm(&self, _input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        if self.config.num_recursions == 0 {
            return self.block.compute_gradients(_input, output_grads);
        }
        let mut all = Vec::new();
        if let Some(core) = &self.cached_core_state {
            // 1. Backward through final answer step
            self.block.set_cache(Some(core.clone()));
            // We need the input to the block to compute gradients properly?
            // compute_gradients usually takes (input, output_grads).
            // But we don't have the exact input easily available unless we stored it in cache.
            // TransformerBlock cache stores input! DiffusionBlock cache stores input!
            // So we can extract input from cache.
            let input_to_block = match core {
                CoreCache::Transformer(c) => &c.0,
                CoreCache::Diffusion(c) => &c.input,
            };
            
            let (mut d_ans_in, mut block_grads) = self.block.compute_gradients(input_to_block, output_grads);
            
            // 2. Backward through recursions
            if !self.cached_step_states.is_empty() {
                let idx = self.cached_step_states.len() - 1;
                let recs = &self.cached_step_states[idx].recursion_caches;
                
                // d_ans_in flows back to y and z.
                // d_y = d_ans_in, d_z = d_ans_in
                let mut d_z = d_ans_in.clone();
                // d_y accumulates? No, y is updated step by step.
                // But here we are at the end of the loop.
                
                // Actually, the gradient flow in LRM is complex (BPTT).
                // For simplicity in this "Tiny Recursive Model" implementation, 
                // we will just accumulate gradients from all steps into the block weights.
                // And backprop through the last few steps if needed.
                // The original implementation did BPTT through the recursion loop.
                
                for rec in recs.iter().rev() {
                    self.block.set_cache(Some(rec.clone()));
                    let rec_input = match rec {
                        CoreCache::Transformer(c) => &c.0,
                        CoreCache::Diffusion(c) => &c.input,
                    };
                    
                    // Gradient of z update: z_new = (1-a)z + a*block_out
                    // d_block_out = d_z * a
                    let a = self.config.latent_update_alpha; // Simplified, ignoring adaptive alpha gradient for now
                    let d_block_out = d_z.mapv(|x| x * a);
                    
                    let (d_combined, rec_grads) = self.block.compute_gradients(rec_input, &d_block_out);
                    
                    // Accumulate block grads
                    if block_grads.len() == rec_grads.len() {
                        for (bg, rg) in block_grads.iter_mut().zip(rec_grads.iter()) {
                            *bg = bg as &Array2<f32> + rg;
                        }
                    }
                    
                    // d_combined = d_y + d_z
                    d_z = &d_z * (1.0 - a) + &d_combined;
                }
                
                d_ans_in = d_z; // Approximation for input gradient
            }
            
            all.extend(block_grads);
            
            if let Some((gw, gb)) = self.latent_init_gradients(&d_ans_in) {
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
            
            (d_ans_in, all)
        } else {
            (output_grads.clone(), Vec::new())
        }
    }

    pub fn compute_gradients_at_step(
        &self,
        step_idx: usize,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Simplified implementation for now
        self.compute_gradients_lrm(&Array2::zeros((0,0)), output_grads)
    }

    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if self.config.num_recursions == 0 {
            return self.block.apply_gradients(param_grads, lr);
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
        self.block.apply_gradients(block_grads, lr)?;
        
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
        let base = self.block.parameters();
        let latent = self.latent_init.as_ref().map(|l| l.w.len() + l.b.len()).unwrap_or(0);
        base + latent
    }
    fn weight_norm(&self) -> f32 { self.block.weight_norm() }
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