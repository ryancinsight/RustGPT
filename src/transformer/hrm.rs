#![allow(dead_code)]
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

use crate::{
    errors::Result,
    network::Layer,
    model_config::ModelConfig,
    transformer::transformer_block::{TransformerBlock, TransformerBlockConfig},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HRMConfig {
    pub bottom_config: TransformerBlockConfig,
    pub top_config: TransformerBlockConfig,
    pub stride: usize,
    pub embed_dim: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HRM {
    pub bottom_block: TransformerBlock,
    pub top_block: TransformerBlock,
    
    // Linear projection for downsampling: (embed_dim) -> (embed_dim)
    // We average pool first, then project.
    pub downsample_w: Array2<f32>,
    pub downsample_b: Array2<f32>,
    
    // Linear projection for upsampling: (embed_dim) -> (embed_dim)
    // We project, then repeat.
    pub upsample_w: Array2<f32>,
    pub upsample_b: Array2<f32>,
    
    config: HRMConfig,
    
    #[serde(skip_serializing, skip_deserializing)]
    cached_intermediates: Option<HRMCache>,
    
    #[serde(skip_serializing, skip_deserializing)]
    param_partitions: RwLock<Option<HRMPartitions>>,
}

#[derive(Clone, Debug)]
struct HRMCache {
    input: Array2<f32>,
    bottom_out: Array2<f32>,
    pooled: Array2<f32>,
    coarse_input: Array2<f32>,
    coarse_out: Array2<f32>,
    fine_projected: Array2<f32>,
}

#[derive(Clone, Debug, Default)]
struct HRMPartitions {
    bottom: usize,
    top: usize,
    downsample: usize,
    upsample: usize,
}

impl HRM {
    pub fn new(config: HRMConfig) -> Self {
        let bottom_block = TransformerBlock::new(config.bottom_config.clone());
        let top_block = TransformerBlock::new(config.top_config.clone());
        
        let dim = config.embed_dim;
        // Initialize identity-like projections
        let mut downsample_w = Array2::eye(dim);
        let downsample_b = Array2::<f32>::zeros((1, dim));
        let mut upsample_w = Array2::eye(dim);
        let upsample_b = Array2::<f32>::zeros((1, dim));
        
        // Add small noise to break symmetry if needed, but identity is a good start for residual-like behavior
        
        Self {
            bottom_block,
            top_block,
            downsample_w,
            downsample_b,
            upsample_w,
            upsample_b,
            config,
            cached_intermediates: None,
            param_partitions: RwLock::new(None),
        }
    }

    pub fn from_model_config(config: &ModelConfig) -> Self {
        let stride = 2; // Default stride
        let mut bottom_cfg = TransformerBlockConfig {
            embed_dim: config.embedding_dim,
            hidden_dim: config.hidden_dim,
            num_heads: config.get_num_heads(),
            poly_degree: config.get_poly_degree_p(),
            max_pos: config.max_seq_len,
            window_size: config.window_size,
            use_moe: config.moe_router.is_some(),
            moe_config: None, // Simplified for now
            head_selection: config.head_selection.clone(),
            use_adaptive_window: config.use_adaptive_window,
            min_window_size: config.min_window_size,
            max_window_size: config.max_window_size,
            window_adaptation_strategy: config.window_adaptation_strategy,
            entropy_ema_alpha: config.entropy_ema_alpha,
        };
        
        // Top block might have larger effective window or different capacity
        let mut top_cfg = bottom_cfg.clone();
        top_cfg.max_pos = config.max_seq_len / stride;
        if let Some(w) = top_cfg.window_size {
            top_cfg.window_size = Some(w / stride);
        }

        let hrm_config = HRMConfig {
            bottom_config: bottom_cfg,
            top_config: top_cfg,
            stride,
            embed_dim: config.embedding_dim,
        };
        
        Self::new(hrm_config)
    }

    fn downsample(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (seq_len, dim) = input.dim();
        let stride = self.config.stride;
        let out_len = (seq_len + stride - 1) / stride;
        
        let mut pooled = Array2::<f32>::zeros((out_len, dim));
        
        // Average pooling
        for i in 0..out_len {
            let start = i * stride;
            let end = (start + stride).min(seq_len);
            let count = (end - start) as f32;
            
            for j in 0..dim {
                let mut sum = 0.0;
                for k in start..end {
                    sum += input[[k, j]];
                }
                pooled[[i, j]] = sum / count;
            }
        }
        
        // Linear projection
        let projected = pooled.dot(&self.downsample_w) + &self.downsample_b;
        (projected, pooled)
    }

    fn upsample(&self, input: &Array2<f32>, target_len: usize) -> (Array2<f32>, Array2<f32>) {
        // Linear projection first
        let projected = input.dot(&self.upsample_w) + &self.upsample_b;
        
        let (in_len, dim) = projected.dim();
        let stride = self.config.stride;
        
        let mut output = Array2::<f32>::zeros((target_len, dim));
        
        // Nearest neighbor upsampling (repeat)
        for i in 0..target_len {
            let src_idx = i / stride;
            if src_idx < in_len {
                for j in 0..dim {
                    output[[i, j]] = projected[[src_idx, j]];
                }
            }
        }
        
        (output, projected)
    }

    fn downsample_backward(&self, grad_output: &Array2<f32>, pooled: &Array2<f32>, orig_len: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // grad_output: (out_len, dim)
        // pooled: (out_len, dim)
        
        // dL/dW = pooled^T * grad_output
        let grad_w = pooled.t().dot(grad_output);
        // dL/db = sum(grad_output, axis=0)
        let mut grad_b = Array2::<f32>::zeros((1, self.config.embed_dim));
        for i in 0..grad_output.nrows() {
            for j in 0..grad_output.ncols() {
                grad_b[[0, j]] += grad_output[[i, j]];
            }
        }
        
        // dL/dPooled = grad_output * W^T
        let grad_pooled = grad_output.dot(&self.downsample_w.t());
        
        // Backprop through average pooling
        let mut grad_input = Array2::<f32>::zeros((orig_len, self.config.embed_dim));
        let stride = self.config.stride;
        let (out_len, dim) = grad_pooled.dim();
        
        for i in 0..out_len {
            let start = i * stride;
            let end = (start + stride).min(orig_len);
            let count = (end - start) as f32;
            let scale = 1.0 / count;
            
            for k in start..end {
                for j in 0..dim {
                    grad_input[[k, j]] += grad_pooled[[i, j]] * scale;
                }
            }
        }
        
        (grad_input, grad_w, grad_b)
    }

    fn upsample_backward(&self, grad_output: &Array2<f32>, projected: &Array2<f32>, coarse_len: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // grad_output: (target_len, dim)
        // projected: (coarse_len, dim) (input to upsample repeat)
        
        // Backprop through repeat
        let mut grad_projected = Array2::<f32>::zeros((coarse_len, self.config.embed_dim));
        let stride = self.config.stride;
        let (target_len, dim) = grad_output.dim();
        
        for i in 0..target_len {
            let src_idx = i / stride;
            if src_idx < coarse_len {
                for j in 0..dim {
                    grad_projected[[src_idx, j]] += grad_output[[i, j]];
                }
            }
        }
        
        // Backprop through linear
        // dL/dW = input_to_proj^T * grad_projected
        // But wait, we don't have input_to_proj here (it's coarse_out).
        // We need to return grad w.r.t weights and input.
        // Let's assume the caller provides the input to the forward pass (coarse_out).
        // Actually, `projected` is the OUTPUT of the linear layer.
        // We need the INPUT to the linear layer to compute dW.
        // Let's adjust the signature or cache usage.
        
        // We will compute dW/db in the main backward function where we have access to coarse_out.
        // Here we just return grad_projected (dL/dLinearOutput).
        
        // Wait, I put linear BEFORE repeat in forward.
        // forward: input -> linear -> projected -> repeat -> output
        // backward: grad_output -> un-repeat -> grad_projected -> linear_backward -> grad_input
        
        // So this function should return grad_projected.
        // And then we do linear backward outside.
        
        (grad_projected, Array2::<f32>::zeros((0,0)), Array2::<f32>::zeros((0,0))) // Placeholders
    }
}

impl Layer for HRM {
    fn layer_type(&self) -> &str { "HRM" }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 1. Bottom block
        let bottom_out = self.bottom_block.forward(input);
        
        // 2. Downsample
        let (coarse_input, pooled) = self.downsample(&bottom_out);
        
        // 3. Top block
        let coarse_out = self.top_block.forward(&coarse_input);
        
        // 4. Upsample
        let (fine_projected, fine_linear_out) = self.upsample(&coarse_out, input.nrows());
        
        // 5. Combine (Residual)
        let output = &bottom_out + &fine_projected;
        
        self.cached_intermediates = Some(HRMCache {
            input: input.clone(),
            bottom_out,
            pooled,
            coarse_input,
            coarse_out,
            fine_projected: fine_linear_out, // Store the output of linear layer (before repeat)
        });
        
        output
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::<f32>::zeros((0,0)), grads);
        let _ = self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn compute_gradients(&self, _input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        if let Some(cache) = &self.cached_intermediates {
            let mut all_grads = Vec::new();
            
            // Output = bottom_out + fine_projected
            // Gradients split
            let d_bottom_out_1 = output_grads.clone();
            let d_fine_projected = output_grads.clone();
            
            // 4. Upsample Backward
            // forward: coarse_out -> linear -> fine_linear_out -> repeat -> fine_projected
            // backward: d_fine_projected -> un-repeat -> d_fine_linear_out -> linear_backward -> d_coarse_out
            
            let (d_fine_linear_out, _, _) = self.upsample_backward(&d_fine_projected, &cache.fine_projected, cache.coarse_out.nrows());
            
            // Linear backward
            // dL/dW_up = coarse_out^T * d_fine_linear_out
            let d_upsample_w = cache.coarse_out.t().dot(&d_fine_linear_out);
            // dL/db_up = sum(d_fine_linear_out)
            let mut d_upsample_b = Array2::<f32>::zeros((1, self.config.embed_dim));
            for i in 0..d_fine_linear_out.nrows() {
                for j in 0..d_fine_linear_out.ncols() {
                    d_upsample_b[[0, j]] += d_fine_linear_out[[i, j]];
                }
            }
            // dL/d_coarse_out = d_fine_linear_out * W_up^T
            let d_coarse_out = d_fine_linear_out.dot(&self.upsample_w.t());
            
            // 3. Top Block Backward
            let (d_coarse_input, top_grads) = self.top_block.compute_gradients(&cache.coarse_input, &d_coarse_out);
            
            // 2. Downsample Backward
            // forward: bottom_out -> pool -> pooled -> linear -> coarse_input
            // backward: d_coarse_input -> linear_backward -> d_pooled -> un-pool -> d_bottom_out_2
            
            // Linear backward
            // dL/dW_down = pooled^T * d_coarse_input
            let d_downsample_w = cache.pooled.t().dot(&d_coarse_input);
            // dL/db_down = sum(d_coarse_input)
            let mut d_downsample_b = Array2::<f32>::zeros((1, self.config.embed_dim));
            for i in 0..d_coarse_input.nrows() {
                for j in 0..d_coarse_input.ncols() {
                    d_downsample_b[[0, j]] += d_coarse_input[[i, j]];
                }
            }
            // dL/d_pooled = d_coarse_input * W_down^T
            let d_pooled = d_coarse_input.dot(&self.downsample_w.t());
            
            // Un-pool
            // Reimplementing un-pool logic here properly since helper was weird
             let mut d_bottom_out_2_real = Array2::<f32>::zeros((cache.bottom_out.nrows(), self.config.embed_dim));
             let stride = self.config.stride;
             let (out_len, dim) = d_pooled.dim();
             for i in 0..out_len {
                let start = i * stride;
                let end = (start + stride).min(cache.bottom_out.nrows());
                let count = (end - start) as f32;
                let scale = 1.0 / count;
                for k in start..end {
                    for j in 0..dim {
                        d_bottom_out_2_real[[k, j]] += d_pooled[[i, j]] * scale;
                    }
                }
            }
            
            // Combine bottom gradients
            let d_bottom_out_total = d_bottom_out_1 + d_bottom_out_2_real;
            
            // 1. Bottom Block Backward
            let (d_input, bottom_grads) = self.bottom_block.compute_gradients(&cache.input, &d_bottom_out_total);
            
            // Collect all gradients
            // Order: bottom, top, downsample, upsample
            all_grads.extend(bottom_grads);
            all_grads.extend(top_grads);
            all_grads.push(d_downsample_w);
            all_grads.push(d_downsample_b);
            all_grads.push(d_upsample_w);
            all_grads.push(d_upsample_b);
            
            // Update partitions
            if let Ok(mut guard) = self.param_partitions.write() {
                *guard = Some(HRMPartitions {
                    bottom: self.bottom_block.parameters(), // Approximation, actually need grad counts
                    top: self.top_block.parameters(),
                    downsample: 2,
                    upsample: 2,
                });
                // Correct counts based on actual returned vectors
                // We need to know how many grads bottom/top return.
                // compute_gradients returns a Vec.
                // Let's recalculate.
            }
             // We need exact counts for apply_gradients
             let bottom_count = self.bottom_block.compute_gradients(&cache.input, &d_bottom_out_total).1.len();
             let top_count = self.top_block.compute_gradients(&cache.coarse_input, &d_coarse_out).1.len();
             
             if let Ok(mut guard) = self.param_partitions.write() {
                *guard = Some(HRMPartitions {
                    bottom: bottom_count,
                    top: top_count,
                    downsample: 2,
                    upsample: 2,
                });
             }

            (d_input, all_grads)
        } else {
            (output_grads.clone(), Vec::new())
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        let partitions = self.param_partitions.read().unwrap().clone().unwrap_or_default();
        
        let mut idx = 0;
        let mut next_slice = |count: usize| {
            let end = idx + count;
            let slice = &param_grads[idx..end];
            idx = end;
            slice
        };
        
        let bottom_grads = next_slice(partitions.bottom);
        let top_grads = next_slice(partitions.top);
        let downsample_grads = next_slice(partitions.downsample);
        let upsample_grads = next_slice(partitions.upsample);
        
        self.bottom_block.apply_gradients(bottom_grads, lr)?;
        self.top_block.apply_gradients(top_grads, lr)?;
        
        // Apply downsample grads
        if downsample_grads.len() == 2 {
            let dw = &downsample_grads[0];
            let db = &downsample_grads[1];
            self.downsample_w = &self.downsample_w - &(dw * lr);
            self.downsample_b = &self.downsample_b - &(db * lr);
        }
        
        // Apply upsample grads
        if upsample_grads.len() == 2 {
            let dw = &upsample_grads[0];
            let db = &upsample_grads[1];
            self.upsample_w = &self.upsample_w - &(dw * lr);
            self.upsample_b = &self.upsample_b - &(db * lr);
        }
        
        Ok(())
    }

    fn parameters(&self) -> usize {
        self.bottom_block.parameters() + self.top_block.parameters() + 
        self.downsample_w.len() + self.downsample_b.len() + 
        self.upsample_w.len() + self.upsample_b.len()
    }

    fn weight_norm(&self) -> f32 {
        self.bottom_block.weight_norm() + self.top_block.weight_norm() +
        (self.downsample_w.iter().map(|x| x*x).sum::<f32>() +
         self.upsample_w.iter().map(|x| x*x).sum::<f32>()).sqrt()
    }

    fn zero_gradients(&mut self) {
        // HRM doesn't maintain internal gradient state beyond cached intermediates
        // Reset cached intermediates to free memory
        self.cached_intermediates = None;
        if let Ok(mut guard) = self.param_partitions.write() {
            *guard = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixtures::HeadSelectionStrategy;

    #[test]
    fn test_hrm_shapes() {
        let dim = 16;
        let stride = 2;
        let bottom_cfg = TransformerBlockConfig {
            embed_dim: dim,
            hidden_dim: dim * 2,
            num_heads: 4,
            poly_degree: 3,
            max_pos: 32,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
            use_adaptive_window: false,
            min_window_size: 4,
            max_window_size: 32,
            window_adaptation_strategy: crate::model_config::WindowAdaptationStrategy::Fixed,
            entropy_ema_alpha: 0.1,
        };
        let top_cfg = bottom_cfg.clone();
        
        let config = HRMConfig {
            bottom_config: bottom_cfg,
            top_config: top_cfg,
            stride,
            embed_dim: dim,
        };
        
        let mut hrm = HRM::new(config);
        let input = Array2::zeros((8, dim));
        let output = hrm.forward(&input);
        
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_hrm_gradients() {
        let dim = 8;
        let stride = 2;
        let bottom_cfg = TransformerBlockConfig {
            embed_dim: dim,
            hidden_dim: dim * 2,
            num_heads: 2,
            poly_degree: 3,
            max_pos: 16,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 2 },
            use_adaptive_window: false,
            min_window_size: 4,
            max_window_size: 16,
            window_adaptation_strategy: crate::model_config::WindowAdaptationStrategy::Fixed,
            entropy_ema_alpha: 0.1,
        };
        let top_cfg = bottom_cfg.clone();
        
        let config = HRMConfig {
            bottom_config: bottom_cfg,
            top_config: top_cfg,
            stride,
            embed_dim: dim,
        };
        
        let mut hrm = HRM::new(config);
        let input = Array2::zeros((4, dim));
        let _ = hrm.forward(&input);
        let grads = Array2::ones((4, dim));
        
        let (in_grads, param_grads) = hrm.compute_gradients(&input, &grads);
        assert_eq!(in_grads.shape(), input.shape());
        assert!(!param_grads.is_empty());
        
        // Test apply
        hrm.apply_gradients(&param_grads, 0.01).unwrap();
    }
}
