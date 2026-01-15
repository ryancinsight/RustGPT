use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{layers::components::common::FeedForwardVariant, network::Layer};

#[derive(Serialize, Deserialize, Debug)]
pub struct FeedforwardProcessor {
    feedforward: FeedForwardVariant,
}

impl FeedforwardProcessor {
    pub fn new(feedforward: FeedForwardVariant) -> Self {
        Self { feedforward }
    }

    pub fn forward(
        &mut self,
        input: &Array2<f32>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
    ) -> Array2<f32> {
        self.forward_with_token_head_activity(input, head_activity_ratio, head_activity_vec, None)
    }

    pub fn forward_with_token_head_activity(
        &mut self,
        input: &Array2<f32>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity_vec: Option<&[f32]>,
    ) -> Array2<f32> {
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.forward(input),
            FeedForwardVariant::MixtureOfExperts(layer) => layer
                .forward_with_head_features_and_token_activity(
                    input,
                    head_activity_ratio,
                    head_activity_vec,
                    token_head_activity_vec,
                ),
        }
    }

    pub fn backward(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match &self.feedforward {
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
    ) -> crate::errors::Result<()> {
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.apply_gradients(param_grads, lr),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    pub fn parameters(&self) -> usize {
        self.feedforward.parameters()
    }

    pub fn weight_norm(&self) -> f32 {
        self.feedforward.weight_norm()
    }

    pub fn zero_gradients(&mut self) {
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.zero_gradients(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.zero_gradients(),
        }
    }

    pub fn layer_type(&self) -> &str {
        match &self.feedforward {
            FeedForwardVariant::RichardsGlu(_) => "RichardsGlu",
            FeedForwardVariant::MixtureOfExperts(_) => "MixtureOfExperts",
        }
    }

    pub fn get_head_activity_metrics(&self) -> (Option<f32>, Option<&[f32]>) {
        match &self.feedforward {
            FeedForwardVariant::MixtureOfExperts(_layer) => (None, None),
            _ => (None, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FeedforwardProcessor;
    use crate::{
        layers::components::common::FeedForwardVariant,
        mixtures::{
            gating::GatingConfig,
            moe::{ExpertRouterConfig, LearnedKAdapter, MixtureOfExperts},
        },
    };

    #[test]
    fn test_feedforward_processor_forwards_token_head_activity_to_moe() {
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
            FeedforwardProcessor::new(FeedForwardVariant::MixtureOfExperts(Box::new(moe)));

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
