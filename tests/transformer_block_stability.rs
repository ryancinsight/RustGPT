use core::iter::Iterator;

use llm::{
    Layer,
    layers::transformer::{TransformerBlock, TransformerBlockConfig},
    mixtures::HeadSelectionStrategy,
};
use ndarray::Array2;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]
    #[test]
    fn gradients_are_finite_and_bounded(seq_len in 8usize..64, embed_dim in 32usize..256) {
        let nh = (1..=8usize.min(embed_dim)).rev().find(|&h| embed_dim % h == 0).unwrap_or(1);
        let config = TransformerBlockConfig {
            embed_dim,
            hidden_dim: embed_dim * 2,
            num_heads: nh,
            poly_degree: 3,
            max_pos: seq_len.saturating_sub(1),
            window_size: Some(seq_len),
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 8 },
            moh_threshold_modulation: llm::richards::adaptive::AdaptiveScalar::default(),
            temporal_mixing: llm::model_config::TemporalMixingType::Attention,
            use_adaptive_window: false,
            min_window_size: seq_len,
            max_window_size: seq_len,
            window_adaptation_strategy: llm::model_config::WindowAdaptationStrategy::Fixed,
            entropy_ema_alpha: 0.2,
            use_advanced_adaptive_residuals: false,
            titan_memory: llm::model_config::TitanMemoryConfig::default(),
            eprop_adaptor: None,
        };
        let mut block = TransformerBlock::new(config);
        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let _out = block.forward(&input);
        let grads = Array2::<f32>::ones((seq_len, embed_dim));
        let (in_grad, param_grads) = block.compute_gradients(&input, &grads);
        for &x in in_grad.iter() { prop_assert!(x.is_finite()); }
        let gnorm: f32 = in_grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let onorm: f32 = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
        prop_assert!(gnorm <= onorm * 200.0);
        for g in param_grads.iter() {
            for &x in g.iter() { prop_assert!(x.is_finite()); }
        }
    }
}
