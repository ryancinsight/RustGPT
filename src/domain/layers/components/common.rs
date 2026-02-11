use ndarray::{Array2, parallel::prelude::*};
use serde::{Deserialize, Serialize};

use crate::domain::{
    attention::{
        poly_attention::PolyAttention,
        position::config::{CoPEConfig, CoPEVariant},
    },
    layers::ssm::{
        Mamba, Mamba2, MoHMamba, MoHMamba2,
        rg_lru::{MoHRgLru, RgLru},
    },
    memory::titans::{NeuralMemory, TitansMAC},
    mixtures::{
        HeadSelectionStrategy,
        moe::{ExpertRouterConfig, MixtureOfExperts},
    },
    models::config::{TemporalMixingType, TitanMemoryConfig},
    network::Layer,
    richards::{RichardsGlu, RichardsNorm},
};

/// Temporal-mixing layer variants shared between TransformerBlock and DiffusionBlock.
///
/// Important: this enum is *tagged* (not `untagged`) to avoid ambiguous decoding when
/// multiple variants share field names (e.g., attention vs RG-LRU MoH).
///
/// Legacy attention-only checkpoints are still supported via TransformerBlock's custom
/// deserializer, which maps the old `attention: PolyAttention` field into this enum.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "data")]
pub enum TemporalMixingLayer {
    Attention(Box<PolyAttention>),
    RgLruMoH(Box<MoHRgLru>),
    RgLru(Box<RgLru>),
    MambaMoH(Box<MoHMamba>),
    Mamba(Box<Mamba>),
    Mamba2MoH(Box<MoHMamba2>),
    Mamba2(Box<Mamba2>),
    Titans(Box<TitansMAC>),
}

#[derive(Default, Debug, Clone)]
pub(crate) struct TitanMemoryWorkspace {
    acc: Vec<f32>,
}

impl TitanMemoryWorkspace {
    pub(crate) fn reset(&mut self) {
        self.acc.fill(0.0);
    }
}

impl TitanMemoryConfig {
    #[cfg(test)]
    pub(crate) fn apply_into_out(&self, out: &mut Array2<f32>, input: &Array2<f32>) {
        let mut ws = TitanMemoryWorkspace::default();
        self.apply_into_out_with_workspace(out, input, &mut ws);
    }

    pub(crate) fn apply_into_out_with_workspace(
        &self,
        out: &mut Array2<f32>,
        input: &Array2<f32>,
        workspace: &mut TitanMemoryWorkspace,
    ) {
        if !self.enabled {
            return;
        }
        let n = input.nrows();
        let d = input.ncols();
        assert_eq!(out.nrows(), n);
        assert_eq!(out.ncols(), d);
        assert!(self.scale.is_finite());
        assert!(self.eta.is_finite());
        assert!(self.decay.is_finite());
        assert!(self.eta >= 0.0);
        assert!(self.decay >= 0.0 && self.decay <= 1.0);

        let retain = 1.0 - self.decay;
        if workspace.acc.len() != d {
            workspace.acc.resize(d, 0.0);
            workspace.acc.fill(0.0); // Reset new elements or if resized
        }
        // Note: We do NOT reset workspace.acc here for streaming persistence.
        // Batch callers must call workspace.reset() explicitly before processing a sequence.
        
        for i in 0..n {
            for j in 0..d {
                let next = retain * workspace.acc[j] + self.eta * input[[i, j]];
                workspace.acc[j] = next;
                out[[i, j]] += self.scale * next;
            }
        }
    }

    pub(crate) fn apply_step_into(
        &self,
        input: &ndarray::ArrayView1<f32>,
        out: &mut ndarray::Array1<f32>,
        workspace: &mut TitanMemoryWorkspace,
    ) {
        if !self.enabled {
            return;
        }
        let d = input.len();
        assert_eq!(out.len(), d);
        assert!(self.scale.is_finite());
        assert!(self.eta.is_finite());
        assert!(self.decay.is_finite());
        assert!(self.eta >= 0.0);
        assert!(self.decay >= 0.0 && self.decay <= 1.0);

        let retain = 1.0 - self.decay;
        if workspace.acc.len() != d {
            workspace.acc.resize(d, 0.0);
            workspace.acc.fill(0.0);
        }

        for j in 0..d {
            let next = retain * workspace.acc[j] + self.eta * input[j];
            workspace.acc[j] = next;
            out[j] += self.scale * next;
        }
    }

    #[cfg(test)]
    pub(crate) fn input_grads_from_output_grads(&self, output_grads: &Array2<f32>) -> Array2<f32> {
        if !self.enabled {
            return Array2::zeros(output_grads.raw_dim());
        }
        let n = output_grads.nrows();
        let d = output_grads.ncols();
        assert!(self.scale.is_finite());
        assert!(self.eta.is_finite());
        assert!(self.decay.is_finite());
        assert!(self.eta >= 0.0);
        assert!(self.decay >= 0.0 && self.decay <= 1.0);

        let retain = 1.0 - self.decay;
        let mut input_grads = Array2::<f32>::zeros(output_grads.raw_dim());
        for j in 0..d {
            let mut b = 0.0f32;
            for i in (0..n).rev() {
                let g = output_grads[[i, j]];
                assert!(g.is_finite());
                b = retain * b + g;
                input_grads[[i, j]] = self.scale * self.eta * b;
            }
        }
        input_grads
    }

    pub(crate) fn add_input_grads_from_output_grads_into(
        &self,
        output_grads: &Array2<f32>,
        input_grads: &mut Array2<f32>,
    ) {
        if !self.enabled {
            return;
        }
        let n = output_grads.nrows();
        let d = output_grads.ncols();
        assert_eq!(input_grads.nrows(), n);
        assert_eq!(input_grads.ncols(), d);
        assert!(self.scale.is_finite());
        assert!(self.eta.is_finite());
        assert!(self.decay.is_finite());
        assert!(self.eta >= 0.0);
        assert!(self.decay >= 0.0 && self.decay <= 1.0);

        let retain = 1.0 - self.decay;
        let coeff = self.scale * self.eta;
        for j in 0..d {
            let mut b = 0.0f32;
            for i in (0..n).rev() {
                let g = output_grads[[i, j]];
                assert!(g.is_finite());
                b = retain * b + g;
                input_grads[[i, j]] += coeff * b;
            }
        }
    }
}

impl TemporalMixingLayer {
    #[inline]
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.forward(input),
            TemporalMixingLayer::RgLruMoH(layer) => layer.forward(input),
            TemporalMixingLayer::RgLru(layer) => layer.forward(input),
            TemporalMixingLayer::MambaMoH(layer) => layer.forward(input),
            TemporalMixingLayer::Mamba(layer) => layer.forward(input),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.forward(input),
            TemporalMixingLayer::Mamba2(layer) => layer.forward(input),
            TemporalMixingLayer::Titans(layer) => layer.forward(input),
        }
    }

    /// Streaming forward step for token-by-token inference.
    ///
    /// Currently only implemented for PolyAttention.
    pub fn forward_step(&mut self, input: &ndarray::Array1<f32>) -> ndarray::Array1<f32> {
        let mut output = ndarray::Array1::zeros(input.raw_dim());
        self.forward_step_into(&input.view(), &mut output);
        output
    }

    pub fn forward_step_into(&mut self, input: &ndarray::ArrayView1<f32>, output: &mut ndarray::Array1<f32>) {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.forward_step_into(input, output),
            TemporalMixingLayer::RgLru(layer) => layer.forward_step_into(input, output),
            TemporalMixingLayer::RgLruMoH(layer) => layer.forward_step_into(input, output),
            TemporalMixingLayer::Mamba(layer) => layer.forward_step_into(input, output),
            TemporalMixingLayer::MambaMoH(layer) => layer.forward_step_into(input, output),
            TemporalMixingLayer::Mamba2(layer) => layer.forward_step_into(input, output),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.forward_step_into(input, output),
            TemporalMixingLayer::Titans(layer) => layer.forward_step_into(input, output),
        }
    }

    pub fn set_training_progress(&mut self, progress: f64) {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.set_training_progress(progress),
            TemporalMixingLayer::RgLruMoH(layer) => layer.set_training_progress(progress),
            TemporalMixingLayer::RgLru(layer) => layer.set_training_progress(progress),
            TemporalMixingLayer::MambaMoH(layer) => layer.set_training_progress(progress),
            TemporalMixingLayer::Mamba(layer) => layer.set_training_progress(progress),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.set_training_progress(progress),
            TemporalMixingLayer::Mamba2(layer) => layer.set_training_progress(progress),
            TemporalMixingLayer::Titans(layer) => layer.set_training_progress(progress),
        }
    }

    #[inline]
    pub fn forward_with_causal(&mut self, input: &Array2<f32>, causal: bool) -> Array2<f32> {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.forward_impl(input, causal),
            TemporalMixingLayer::RgLruMoH(layer) => layer.forward(input),
            TemporalMixingLayer::RgLru(layer) => layer.forward(input),
            TemporalMixingLayer::MambaMoH(layer) => {
                let _ = causal;
                layer.forward(input)
            }
            TemporalMixingLayer::Mamba(layer) => {
                let _ = causal;
                layer.forward(input)
            }
            TemporalMixingLayer::Mamba2MoH(layer) => {
                let _ = causal;
                layer.forward(input)
            }
            TemporalMixingLayer::Mamba2(layer) => {
                let _ = causal;
                layer.forward(input)
            }
            TemporalMixingLayer::Titans(layer) => {
                let _ = causal; // TitansMAC implies causal
                layer.forward(input)
            }
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
            TemporalMixingLayer::MambaMoH(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Mamba(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Mamba2(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Titans(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    #[inline]
    pub fn apply_gradients(&mut self, grads: &[Array2<f32>], lr: f32) -> crate::common::errors::Result<()> {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::RgLruMoH(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::RgLru(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::MambaMoH(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::Mamba(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::Mamba2(layer) => layer.apply_gradients(grads, lr),
            TemporalMixingLayer::Titans(layer) => layer.apply_gradients(grads, lr),
        }
    }

    #[inline]
    pub fn parameters(&self) -> usize {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.parameters(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.parameters(),
            TemporalMixingLayer::RgLru(layer) => layer.parameters(),
            TemporalMixingLayer::MambaMoH(layer) => layer.parameters(),
            TemporalMixingLayer::Mamba(layer) => layer.parameters(),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.parameters(),
            TemporalMixingLayer::Mamba2(layer) => layer.parameters(),
            TemporalMixingLayer::Titans(layer) => layer.parameters(),
        }
    }

    #[inline]
    pub fn weight_norm(&self) -> f32 {
        match self {
            TemporalMixingLayer::Attention(layer) => layer.weight_norm(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.weight_norm(),
            TemporalMixingLayer::RgLru(layer) => layer.weight_norm(),
            TemporalMixingLayer::MambaMoH(layer) => layer.weight_norm(),
            TemporalMixingLayer::Mamba(layer) => layer.weight_norm(),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.weight_norm(),
            TemporalMixingLayer::Mamba2(layer) => layer.weight_norm(),
            TemporalMixingLayer::Titans(layer) => layer.weight_norm(),
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::Rng;

    use super::*;
    use crate::common::rng::{get_rng_with_subseed, set_seed};

    #[test]
    fn titan_memory_linear_adjoint_matches_backward() {
        let cfg = TitanMemoryConfig {
            enabled: true,
            scale: 0.3,
            eta: 0.7,
            decay: 0.2,
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let x = Array2::from_shape_fn((7, 5), |(i, j)| (i as f32 * 0.01) - (j as f32 * 0.02));
        let g = Array2::from_shape_fn((7, 5), |(i, j)| (i as f32 * 0.03) + (j as f32 * 0.01));

        let mut y = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out(&mut y, &x);
        let gx = cfg.input_grads_from_output_grads(&g);

        let lhs: f64 = y
            .iter()
            .zip(g.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let rhs: f64 = x
            .iter()
            .zip(gx.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        assert!((lhs - rhs).abs() < 1e-5);
    }

    #[test]
    fn titan_memory_linear_adjoint_random_seeded() {
        set_seed(0xC0FFEE);
        let mut rng_cfg = get_rng_with_subseed(1);
        let cfg = TitanMemoryConfig {
            enabled: true,
            scale: rng_cfg.random_range(-1.0..1.0),
            eta: rng_cfg.random_range(0.0..1.0),
            decay: rng_cfg.random_range(0.0..1.0),
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let mut rng_x = get_rng_with_subseed(2);
        let x = Array2::from_shape_fn((19, 13), |_| rng_x.random_range(-1.0..1.0));
        let mut rng_g = get_rng_with_subseed(3);
        let g = Array2::from_shape_fn((19, 13), |_| rng_g.random_range(-1.0..1.0));

        let mut y = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out(&mut y, &x);
        let gx = cfg.input_grads_from_output_grads(&g);

        let lhs: f64 = y
            .iter()
            .zip(g.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let rhs: f64 = x
            .iter()
            .zip(gx.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        let tol = 1e-4 * (1.0 + lhs.abs() + rhs.abs());
        assert!((lhs - rhs).abs() <= tol);
    }

    #[test]
    fn titan_memory_disabled_is_noop() {
        let cfg = TitanMemoryConfig {
            enabled: false,
            scale: 0.3,
            eta: 0.7,
            decay: 0.2,
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let x = Array2::from_shape_fn((3, 4), |(i, j)| (i as f32) + (j as f32));
        let mut y = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out(&mut y, &x);
        assert!(y.iter().all(|&v| v == 0.0));

        let g = Array2::from_shape_fn((3, 4), |(i, j)| (i as f32) - (j as f32));
        let gx = cfg.input_grads_from_output_grads(&g);
        assert!(gx.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn common_layers_mamba_uses_moh_when_moe_enabled() {
        let config = CommonLayerConfig {
            embed_dim: 16,
            hidden_dim: 32,
            num_heads: 4,
            poly_degree: 2,
            max_pos: 32,
            window_size: None,
            use_moe: true,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Mamba,
        };

        let layers = CommonLayers::new(&config);
        assert!(matches!(
            layers.temporal_mixing,
            TemporalMixingLayer::MambaMoH(_)
        ));
    }

    #[test]
    fn common_layers_mamba2_uses_moh_when_moe_enabled() {
        let config = CommonLayerConfig {
            embed_dim: 16,
            hidden_dim: 32,
            num_heads: 4,
            poly_degree: 2,
            max_pos: 32,
            window_size: None,
            use_moe: true,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Mamba2,
        };

        let layers = CommonLayers::new(&config);
        assert!(matches!(
            layers.temporal_mixing,
            TemporalMixingLayer::Mamba2MoH(_)
        ));
    }

    #[test]
    fn titan_memory_add_input_grads_matches_allocating_version() {
        let cfg = TitanMemoryConfig {
            enabled: true,
            scale: 0.11,
            eta: 0.9,
            decay: 0.05,
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let g = Array2::from_shape_fn((9, 4), |(i, j)| (i as f32 * 0.02) - (j as f32 * 0.03));
        let ref_gx = cfg.input_grads_from_output_grads(&g);

        let mut gx = Array2::<f32>::zeros(g.raw_dim());
        cfg.add_input_grads_from_output_grads_into(&g, &mut gx);

        assert_eq!(gx.dim(), ref_gx.dim());
        for (&a, &b) in gx.iter().zip(ref_gx.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn titan_memory_workspace_is_equivalent_to_fresh_workspace() {
        let cfg = TitanMemoryConfig {
            enabled: true,
            scale: 0.17,
            eta: 0.23,
            decay: 0.11,
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let x = Array2::from_shape_fn((11, 7), |(i, j)| (i as f32 * 0.007) - (j as f32 * 0.013));
        let mut y_fresh = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out(&mut y_fresh, &x);

        // Initialize workspace with correct size (matching input dimension) and reset it
        let mut ws = TitanMemoryWorkspace { acc: vec![0.0; x.ncols()] };
        ws.reset(); // Ensure clean state
        let mut y_ws1 = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out_with_workspace(&mut y_ws1, &x, &mut ws);

        // For second call, need fresh workspace (batch callers must reset explicitly)
        let mut ws2 = TitanMemoryWorkspace { acc: vec![0.0; x.ncols()] };
        ws2.reset();
        let mut y_ws2 = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out_with_workspace(&mut y_ws2, &x, &mut ws2);

        assert_eq!(y_fresh.dim(), y_ws1.dim());
        assert_eq!(y_fresh.dim(), y_ws2.dim());
        for ((&a, &b), &c) in y_fresh.iter().zip(y_ws1.iter()).zip(y_ws2.iter()) {
            assert!((a - b).abs() < 1e-6);
            assert!((a - c).abs() < 1e-6);
        }
    }

    proptest! {
        #[test]
        fn titan_memory_adjoint_property_holds(
            n in 1usize..33,
            d in 1usize..33,
            scale in -1.0f32..1.0f32,
            eta in 0.0f32..1.0f32,
            decay in 0.0f32..1.0f32,
            x_flat in prop::collection::vec(-1.0f32..1.0f32, 1..(33*33)),
            g_flat in prop::collection::vec(-1.0f32..1.0f32, 1..(33*33)),
        ) {
            let len = n * d;
            prop_assume!(x_flat.len() >= len);
            prop_assume!(g_flat.len() >= len);

            let cfg = TitanMemoryConfig {
                enabled: true,
                scale,
                eta,
                decay,
                segment_len: 128,
                persistent_len: 32,
                hidden_dim: 64,
                ..TitanMemoryConfig::default()
            };
            let x = Array2::from_shape_vec((n, d), x_flat[..len].to_vec()).unwrap();
            let g = Array2::from_shape_vec((n, d), g_flat[..len].to_vec()).unwrap();

            let mut y = Array2::<f32>::zeros(x.raw_dim());
            cfg.apply_into_out(&mut y, &x);
            let gx = cfg.input_grads_from_output_grads(&g);

            let lhs: f64 = y.iter().zip(g.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
            let rhs: f64 = x.iter().zip(gx.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();

            let tol = 1e-4 * (1.0 + lhs.abs() + rhs.abs());
            prop_assert!((lhs - rhs).abs() <= tol);
        }
    }
}

/// Feedforward network variants used in transformer blocks
#[derive(Serialize, Deserialize, Debug, Clone)]
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

    /// Streaming forward step for token-by-token inference.
    pub fn forward_step(&mut self, input: &ndarray::Array1<f32>) -> ndarray::Array1<f32> {
        let mut output = ndarray::Array1::zeros(input.raw_dim());
        self.forward_step_into(&input.view(), &mut output);
        output
    }

    pub fn forward_step_into(&mut self, input: &ndarray::ArrayView1<f32>, output: &mut ndarray::Array1<f32>) {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.forward_step_into(input, output),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.forward_step_into(input, output),
        }
    }

    pub fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.backward(grads, lr),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.backward(grads, lr),
        }
    }

    pub fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.compute_gradients(input, output_grads),
            FeedForwardVariant::MixtureOfExperts(layer) => {
                layer.compute_gradients(input, output_grads)
            }
        }
    }

    pub fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::common::errors::Result<()> {
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
    pub moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar,
    #[serde(default)]
    pub titan_memory: TitanMemoryConfig,
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

        let temporal_mixing =
            match config.temporal_mixing {
                TemporalMixingType::Attention => {
                    let cope_config = CoPEConfig {
                        variant: CoPEVariant::Standard,
                        max_pos: config.max_pos,
                        window_size: config.window_size,
                    };
                    let mut attention = PolyAttention::new(
                        config.embed_dim,
                        config.num_heads,
                        config.poly_degree,
                        cope_config,
                    );
                    attention.set_titan_memory_config(config.titan_memory.clone());
                    attention.set_head_selection_config(&config.head_selection);
                    attention.moh.head_selection_config.threshold_modulation = config.moh_threshold_modulation.clone();
                    TemporalMixingLayer::Attention(Box::new(attention))
                }
                TemporalMixingType::RgLru => {
                    if config.use_moe {
                        TemporalMixingLayer::RgLruMoH(Box::new({
                            let mut layer = MoHRgLru::new(config.embed_dim, config.num_heads, &config.head_selection);
                            layer.moh.head_selection_config.threshold_modulation = config.moh_threshold_modulation.clone();
                            layer
                        }))
                    } else {
                        TemporalMixingLayer::RgLru(Box::new(RgLru::new(
                            config.embed_dim,
                        )))
                    }
                }
                TemporalMixingType::Mamba => {
                    if config.use_moe {
                        TemporalMixingLayer::MambaMoH(Box::new({
                            let mut layer = MoHMamba::new(config.embed_dim, config.num_heads, &config.head_selection);
                            layer.moh.head_selection_config.threshold_modulation = config.moh_threshold_modulation.clone();
                            layer
                        }))
                    } else {
                        TemporalMixingLayer::Mamba(Box::new(Mamba::new(
                            config.embed_dim,
                        )))
                    }
                }
                TemporalMixingType::Mamba2 => {
                    if config.use_moe {
                        TemporalMixingLayer::Mamba2MoH(Box::new({
                            let mut layer = MoHMamba2::new(config.embed_dim, config.num_heads, &config.head_selection);
                            layer.moh.head_selection_config.threshold_modulation = config.moh_threshold_modulation.clone();
                            layer
                        }))
                    } else {
                        TemporalMixingLayer::Mamba2(Box::new(Mamba2::new(
                            config.embed_dim,
                        )))
                    }
                }
                TemporalMixingType::Titans => {
                    let cope_config = CoPEConfig {
                        variant: CoPEVariant::Standard,
                        max_pos: config.max_pos,
                        window_size: config.window_size,
                    };
                    let mut attention = PolyAttention::new(
                        config.embed_dim,
                        config.num_heads,
                        config.poly_degree,
                        cope_config,
                    );
                    attention.set_titan_memory_config(config.titan_memory.clone());
                    attention.set_head_selection_config(&config.head_selection);

                    let memory = NeuralMemory::new(
                        config.embed_dim,
                        config.embed_dim,
                        config.embed_dim,
                        config.titan_memory.hidden_dim,
                    );

                    let mac = TitansMAC::new(
                        attention,
                        memory,
                        config.titan_memory.persistent_len,
                        config.titan_memory.segment_len,
                    );

                    TemporalMixingLayer::Titans(Box::new(mac))
                }
            };

        let pre_ffn_norm = RichardsNorm::new(config.embed_dim);

        let feedforward = if config.use_moe {
            if let Some(moe_config) = &config.moe_config {
                // Keep parameter count roughly constant vs dense FFN by shrinking expert_hidden_dim
                // when MoE is enabled. This is important for tiny-model regimes (e.g. ~36k params)
                // where MoE should not inflate total parameters by num_experts.
                let router_hidden_dim = (config.embed_dim / 4).max(32);
                let baseline_ffn_params =
                    RichardsGlu::new(config.embed_dim, config.hidden_dim).parameters();

                let mut adj = moe_config.clone();
                let suggested = (config.hidden_dim / adj.num_experts.max(1)).max(4);
                if adj.expert_hidden_dim > suggested {
                    adj.expert_hidden_dim = suggested;
                }

                // If we're still above the baseline (router overhead, head-conditioning),
                // decrement a bit until we fit.
                for _ in 0..32 {
                    let moe_params =
                        MixtureOfExperts::new(config.embed_dim, router_hidden_dim, adj.clone())
                            .parameters();
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
pub fn sanitize_and_clip_gradients(
    param_grads: &[Array2<f32>],
    clip_threshold: f32,
) -> Vec<Array2<f32>> {
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
) -> crate::common::errors::Result<()>
where
    F: FnMut(&[Array2<f32>], f32) -> crate::common::errors::Result<()>,
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
