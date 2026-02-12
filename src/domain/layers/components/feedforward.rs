//! Shared Feedforward Component
//!
//! This component provides a unified feedforward interface that can be used
//! by multiple architectures (Transformer, Diffusion, SSM).

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

use crate::{
    common::errors::Result,
    domain::layers::components::common::FeedForwardVariant,
    domain::network::Layer,
};

/// Shared feedforward component
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedFeedforward {
    /// The underlying feedforward variant
    pub feedforward: FeedForwardVariant,
}

impl SharedFeedforward {
    /// Create a new shared feedforward component
    pub fn new(feedforward: FeedForwardVariant) -> Self {
        Self { feedforward }
    }

    /// Forward pass through the feedforward network
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.feedforward.forward(input)
    }

    /// Backward pass through the feedforward network
    pub fn backward(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.feedforward.compute_gradients(input, output_grads)
    }

    #[inline]
    pub fn variant(&self) -> &FeedForwardVariant {
        &self.feedforward
    }

    #[inline]
    pub fn variant_mut(&mut self) -> &mut FeedForwardVariant {
        &mut self.feedforward
    }

    #[inline]
    pub fn as_moe(
        &self,
    ) -> Option<&crate::domain::mixtures::moe::MixtureOfExperts> {
        self.feedforward.as_moe()
    }

    #[inline]
    pub fn as_moe_mut(
        &mut self,
    ) -> Option<&mut crate::domain::mixtures::moe::MixtureOfExperts> {
        self.feedforward.as_moe_mut()
    }

    /// Apply gradients to the feedforward network
    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        self.feedforward.apply_gradients(param_grads, lr)
    }

    /// Get the number of parameters
    pub fn parameters(&self) -> usize {
        self.feedforward.parameters()
    }

    /// Get the weight norm
    pub fn weight_norm(&self) -> f32 {
        self.feedforward.weight_norm()
    }

    /// Zero out gradients
    pub fn zero_gradients(&mut self) {
        self.feedforward.zero_gradients()
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &str {
        match &self.feedforward {
            FeedForwardVariant::RichardsGlu(_) => "RichardsGlu",
            FeedForwardVariant::MixtureOfExperts(_) => "MixtureOfExperts",
        }
    }

    pub fn forward_with_token_head_activity(
        &mut self,
        input: &Array2<f32>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity_vec: Option<&[f32]>,
    ) -> Array2<f32> {
        self.feedforward.forward_with_token_head_activity(
            input,
            head_activity_ratio,
            head_activity_vec,
            token_head_activity_vec,
        )
    }

    /// Forward pass with FiLM conditioning and optional token head activity
    pub fn forward_with_film(
        &mut self,
        input: &Array2<f32>,
        gamma: Option<&Array1<f32>>,
        beta: Option<&Array1<f32>>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity_vec: Option<&[f32]>,
    ) -> Array2<f32> {
        if let (Some(g), Some(b)) = (gamma, beta) {
            let mut modified = input.clone();
            for mut row in modified.outer_iter_mut() {
                row.zip_mut_with(g, |x, &g_val| *x *= 1.0 + g_val);
                row.zip_mut_with(b, |x, &b_val| *x += b_val);
            }
            self.forward_with_token_head_activity(
                &modified,
                head_activity_ratio,
                head_activity_vec,
                token_head_activity_vec,
            )
        } else {
            self.forward_with_token_head_activity(
                input,
                head_activity_ratio,
                head_activity_vec,
                token_head_activity_vec,
            )
        }
    }

    pub fn forward_step_into(
        &mut self,
        input: &ArrayView1<f32>,
        output: &mut Array1<f32>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity: Option<f32>,
    ) {
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => {
                layer.forward_step_into(input, output);
            }
            FeedForwardVariant::MixtureOfExperts(layer) => {
                layer.forward_step_with_head_features_into(
                    input,
                    output,
                    head_activity_ratio,
                    head_activity_vec,
                    token_head_activity,
                );
            }
        }
    }

    pub fn set_training_progress(&mut self, progress: f64) {
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.set_training_progress(progress),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.set_training_progress(progress),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SharedFeedforward;
    use crate::{
        domain::layers::components::common::FeedForwardVariant,
        domain::mixtures::{
            gating::GatingConfig,
            moe::{ExpertRouterConfig, LearnedKAdapter, MixtureOfExperts},
        },
    };

    #[test]
    fn test_shared_feedforward_forwards_token_head_activity_to_moe() {
        let mut config = ExpertRouterConfig {
            num_experts: 4,
            expert_hidden_dim: 16,
            diversity_weight: 0.005,
            gating: GatingConfig {
                num_active: 3,
                load_balance_weight: 0.01,
                sparsity_weight: 0.001,
                ..Default::default()
            },
            ..Default::default()
        };
        config.use_head_conditioning = true;
        config.use_learned_k_adaptation = true;

        let mut moe = MixtureOfExperts::new(32, 8, config);
        moe.k_adapter = Some(LearnedKAdapter {
            w: ndarray::Array2::from_shape_vec((2, 1), vec![0.0, 20.0]).unwrap(),
            b: ndarray::Array2::from_shape_vec((1, 1), vec![-10.0]).unwrap(),
        });

        let mut processor =
            SharedFeedforward::new(FeedForwardVariant::MixtureOfExperts(Box::new(moe)));

        let input = ndarray::Array2::<f32>::from_shape_vec((2, 32), vec![0.1; 64]).unwrap();
        let token_h = vec![0.0f32, 1.0f32];

        let _out = processor.forward_with_token_head_activity(
            &input,
            Some(0.0),
            None,
            Some(token_h.as_slice()),
        );

        let FeedForwardVariant::MixtureOfExperts(moe) = &processor.feedforward else {
            panic!("expected MoE feedforward");
        };

        let router_in = moe.test_cached_router_input().unwrap();
        assert!((router_in[[0, 32]] - 0.0).abs() < 1e-6);
        assert!((router_in[[1, 32]] - 1.0).abs() < 1e-6);

        let alpha = moe.test_cached_k_alpha().unwrap();
        assert!(alpha[0] < 0.01);
        assert!(alpha[1] > 0.99);
    }
}
