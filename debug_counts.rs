use llm::{
    Layer,
    mixtures::HeadSelectionStrategy,
    layers::{
        diffusion::{DiffusionBlock, DiffusionBlockConfig},
        transformer::{TransformerBlock, TransformerBlockConfig},
    },
};
use ndarray::Array2;

fn main() {
    let tcfg = TransformerBlockConfig {
        embed_dim: 64,
        hidden_dim: 128,
        num_heads: 8,
        poly_degree: 3,
        max_pos: 79,
        window_size: None,
        use_moe: false,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed { num_active: 8 },
        temporal_mixing: llm::model_config::TemporalMixingType::Attention,
        use_adaptive_window: false,
        min_window_size: 512,
        max_window_size: 4096,
        window_adaptation_strategy: llm::model_config::WindowAdaptationStrategy::SequenceLengthBased,
        entropy_ema_alpha: 0.2,
        use_advanced_adaptive_residuals: true,
    };
    let mut tblock = TransformerBlock::new(tcfg.clone());

    let dcfg: DiffusionBlockConfig = tcfg.into();
    let mut dblock = DiffusionBlock::new(dcfg);
    dblock.set_timestep(10);

    let input = Array2::zeros((16, 64));
    let _ = tblock.forward(&input);
    let _ = dblock.forward(&input);

    let grads = Array2::ones((16, 64));
    let (_t_in_grad, t_param_grads) = tblock.compute_gradients(&input, &grads);
    let (_d_in_grad, d_param_grads) = dblock.compute_gradients(&input, &grads);
    println!("t_param_grads_len={}", t_param_grads.len());
    println!("d_param_grads_len={}", d_param_grads.len());
}
