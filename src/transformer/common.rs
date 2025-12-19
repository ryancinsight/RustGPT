use serde::{Deserialize, Serialize};
use ndarray::{Array2, parallel::prelude::*};
use crate::{
    richards::{RichardsGlu, RichardsNorm},
    mixtures::{
        HeadSelectionStrategy,
        moe::{MixtureOfExperts, ExpertRouterConfig},
    },
    network::Layer,
    attention::poly_attention::PolyAttention,
    model_config::TemporalMixingType,
    ssm::rg_lru::{MoHRgLru, RgLru},
};

/// Temporal-mixing layer variants shared between TransformerBlock and DiffusionBlock.
///
/// Important: this enum is *tagged* (not `untagged`) to avoid ambiguous decoding when
/// multiple variants share field names (e.g., attention vs RG-LRU MoH).
///
/// Legacy attention-only checkpoints are still supported via TransformerBlock's custom
/// deserializer, which maps the old `attention: PolyAttention` field into this enum.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", content = "data")]
pub enum TemporalMixingLayer {
    Attention(PolyAttention),
    RgLruMoH(MoHRgLru),
    RgLru(RgLru),
}

impl TemporalMixingLayer {
    #[inline]
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.forward(input),
            TemporalMixingLayer::RgLruMoH(layer) => layer.forward(input),
            TemporalMixingLayer::RgLru(layer) => layer.forward(input),
        }
    }

    #[inline]
    pub fn forward_with_causal(&mut self, input: &Array2<f32>, causal: bool) -> Array2<f32> {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.forward_impl(input, causal),
            TemporalMixingLayer::RgLruMoH(layer) => layer.forward(input),
            TemporalMixingLayer::RgLru(layer) => layer.forward(input),
        }
    }

    #[inline]
    pub fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::RgLruMoH(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::RgLru(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    #[inline]
    pub fn apply_gradients(&mut self, grads: &[Array2<f32>], lr: f32) -> crate::errors::Result<()> {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::RgLruMoH(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::RgLru(layer) => layer.apply_gradients(grads, lr),
        }
    }

    #[inline]
    pub fn parameters(&self) -> usize {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.parameters(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.parameters(),
            TemporalMixingLayer::RgLru(layer) => layer.parameters(),
        }
    }

    #[inline]
    pub fn weight_norm(&self) -> f32 {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.weight_norm(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.weight_norm(),
            TemporalMixingLayer::RgLru(layer) => layer.weight_norm(),
        }
    }
}

/// Feedforward network variants used in transformer blocks
#[derive(Serialize, Deserialize, Debug)]
pub enum FeedForwardVariant {
    /// Standard RichardsGlu feedforward
    RichardsGlu(Box<RichardsGlu>),

    /// Mixture-of-Experts feedforward
    MixtureOfExperts(Box<MixtureOfExperts>),
}

impl FeedForwardVariant {
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.forward(input),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.forward(input),
        }
    }

    pub fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.backward(grads, lr),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.backward(grads, lr),
        }
    }

    pub fn compute_gradients(&self, input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.compute_gradients(input, output_grads),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> crate::errors::Result<()> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.apply_gradients(param_grads, lr),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    pub fn parameters(&self) -> usize {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.parameters(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.parameters(),
        }
    }

    pub fn weight_norm(&self) -> f32 {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.weight_norm(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.weight_norm(),
        }
    }
}

/// Configuration shared between TransformerBlock and DiffusionBlock
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommonLayerConfig {
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub poly_degree: usize,
    pub max_pos: usize,
    pub window_size: Option<usize>,
    pub use_moe: bool,
    pub moe_config: Option<ExpertRouterConfig>,
    pub head_selection: HeadSelectionStrategy,
    #[serde(default)]
    pub temporal_mixing: TemporalMixingType,
}

/// Common layers shared between TransformerBlock and DiffusionBlock
#[derive(Serialize, Deserialize, Debug)]
pub struct CommonLayers {
    pub pre_attention_norm: RichardsNorm,
    pub temporal_mixing: TemporalMixingLayer,
    pub pre_ffn_norm: RichardsNorm,
    pub feedforward: FeedForwardVariant,
}

impl CommonLayers {
    pub fn new(config: &CommonLayerConfig) -> Self {
        let pre_attention_norm = RichardsNorm::new(config.embed_dim);

        let temporal_mixing = match config.temporal_mixing {
            TemporalMixingType::Attention => {
                let mut attention = PolyAttention::new(
                    config.embed_dim,
                    config.num_heads,
                    config.poly_degree,
                    config.max_pos,
                    config.window_size,
                );
                attention.set_head_selection_config(&config.head_selection);
                TemporalMixingLayer::Attention(attention)
            }
            TemporalMixingType::RgLru => {
                TemporalMixingLayer::RgLruMoH(MoHRgLru::new(
                    config.embed_dim,
                    config.num_heads,
                    &config.head_selection,
                ))
            }
        };

        let pre_ffn_norm = RichardsNorm::new(config.embed_dim);

        let feedforward = if config.use_moe {
            if let Some(moe_config) = &config.moe_config {
                // Keep parameter count roughly constant vs dense FFN by shrinking expert_hidden_dim
                // when MoE is enabled. This is important for tiny-model regimes (e.g. ~36k params)
                // where MoE should not inflate total parameters by num_experts.
                let router_hidden_dim = (config.embed_dim / 4).max(32);
                let baseline_ffn_params = RichardsGlu::new(config.embed_dim, config.hidden_dim).parameters();

                let mut adj = moe_config.clone();
                let suggested = (config.hidden_dim / adj.num_experts.max(1)).max(4);
                if adj.expert_hidden_dim > suggested {
                    adj.expert_hidden_dim = suggested;
                }

                // If we're still above the baseline (router overhead, head-conditioning),
                // decrement a bit until we fit.
                for _ in 0..32 {
                    let moe_params = MixtureOfExperts::new(config.embed_dim, router_hidden_dim, adj.clone()).parameters();
                    if moe_params <= baseline_ffn_params {
                        break;
                    }
                    if adj.expert_hidden_dim <= 4 {
                        break;
                    }
                    adj.expert_hidden_dim = adj.expert_hidden_dim.saturating_sub(1).max(4);
                }

                let moe_layer = MixtureOfExperts::new(config.embed_dim, router_hidden_dim, adj);
                FeedForwardVariant::MixtureOfExperts(Box::new(moe_layer))
            } else {
                let richards_glu = RichardsGlu::new(config.embed_dim, config.hidden_dim);
                FeedForwardVariant::RichardsGlu(Box::new(richards_glu))
            }
        } else {
            let richards_glu = RichardsGlu::new(config.embed_dim, config.hidden_dim);
            FeedForwardVariant::RichardsGlu(Box::new(richards_glu))
        };

        Self {
            pre_attention_norm,
            temporal_mixing,
            pre_ffn_norm,
            feedforward,
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.pre_attention_norm.parameters()
            + self.temporal_mixing.parameters()
            + self.pre_ffn_norm.parameters()
            + self.feedforward.parameters()
    }

    pub fn weight_norm(&self) -> f32 {
        (self.pre_attention_norm.weight_norm().powi(2)
            + self.temporal_mixing.weight_norm().powi(2)
            + self.pre_ffn_norm.weight_norm().powi(2)
            + self.feedforward.weight_norm().powi(2))
        .sqrt()
    }
}

/// Helper to sanitize and globally clip gradients
pub fn sanitize_and_clip_gradients(param_grads: &[Array2<f32>], clip_threshold: f32) -> Vec<Array2<f32>> {
    let pairs: Vec<(Array2<f32>, f32)> = param_grads
        .par_iter()
        .map(|g| {
            let mut gg = g.clone();
            gg.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
            let s = gg.iter().map(|&x| x * x).sum::<f32>();
            (gg, s)
        })
        .collect();
        
    let mut sanitized: Vec<Array2<f32>> = pairs.iter().map(|(gg, _)| gg.clone()).collect();
    let norm_sq: f32 = pairs.iter().map(|(_, s)| *s).sum();
    let nrm = norm_sq.sqrt();
    
    if nrm.is_finite() && nrm > clip_threshold && nrm > 0.0 {
        let scale = clip_threshold / nrm;
        for gg in &mut sanitized {
            gg.mapv_inplace(|x| x * scale);
        }
    }
    sanitized
}

/// Helper to apply gradients with LARS-style adaptive scaling
pub fn apply_adaptive_gradients<F>(
    grads: &[Array2<f32>],
    weight_norm: f32,
    lr: f32,
    mut apply_fn: F,
) -> crate::errors::Result<()>
where
    F: FnMut(&[Array2<f32>], f32) -> crate::errors::Result<()>,
{
    if grads.is_empty() {
        return Ok(());
    }
    
    let gnorm: f32 = grads
        .iter()
        .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
        .sum::<f32>()
        .sqrt();
        
    let wnorm = weight_norm.max(1e-6);
    let scale = (wnorm / (gnorm.max(1e-6))).clamp(0.01, 5.0);
    
    let scaled: Vec<Array2<f32>> = grads
        .par_iter()
        .map(|g| {
            let mut gg = g.clone();
            gg.mapv_inplace(|x| x * scale);
            gg
        })
        .collect();
        
    apply_fn(&scaled, lr)
}
