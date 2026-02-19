use llm::domain::layers::diffusion::{
    DiffusionBlock, DiffusionBlockConfig, DiffusionPredictionTarget, DiffusionSampler,
    EDM_SIGMA_DATA_DEFAULT, LossWeighting, NoiseSchedule,
};
use llm::domain::mixtures::moh::HeadSelectionStrategy;
use llm::domain::models::config::{
    DiffusionTimestepStrategy, TemporalMixingType, TitanMemoryConfig,
};
use llm::domain::network::Layer; // Import Layer trait
use llm::domain::richards::adaptive::AdaptiveScalar;
use ndarray::Array2;

#[test]
fn test_diffusion_block_forward_backward() {
    // Setup configuration
    let config = DiffusionBlockConfig {
        embed_dim: 32,
        hidden_dim: 64,
        num_heads: 4,
        num_timesteps: 100,
        noise_schedule: NoiseSchedule::Linear {
            beta_min: 1e-4,
            beta_max: 0.02,
        },
        prediction_target: DiffusionPredictionTarget::Epsilon,
        timestep_strategy: DiffusionTimestepStrategy::Uniform,
        causal_attention: false,
        window_size: None,
        use_adaptive_window: false,
        discrete_masked: false,
        poly_degree: 1,
        max_pos: 128,
        use_moe: false,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
        moh_threshold_modulation: AdaptiveScalar::default(),
        titan_memory: TitanMemoryConfig::default(),
        time_embed_dim: 32,
        mask_token_id: None,
        temporal_mixing: TemporalMixingType::Attention,
        use_advanced_adaptive_residuals: false,
        edm_sigma_data: EDM_SIGMA_DATA_DEFAULT,
        sampler: DiffusionSampler::DDPM,
        guidance: None,
        loss_weighting: LossWeighting::default(),
        use_p2_weighting: false,
        use_snr_weighting: false,
        adaptive_guidance: false,
        min_guidance_scale: 1.0,
        max_guidance_scale: 10.0,
        ddim_steps_policy: Default::default(),
    };

    let mut block = DiffusionBlock::new(config);

    // Create dummy inputs
    let seq_len = 10;
    let embed_dim = 32;

    // Create random input (deterministic for test)
    let input = Array2::from_elem((seq_len, embed_dim), 0.5f32);
    let timestep = 50;

    // Forward pass (removed extra boolean argument)
    let output = block.forward_with_timestep(&input, timestep);

    // Verify output shape and finiteness
    assert_eq!(output.shape(), &[seq_len, embed_dim]);
    assert!(output.iter().all(|x: &f32| x.is_finite()));

    // Backward pass simulation
    // Note: compute_gradients is from Layer trait
    let grad_output = Array2::from_elem((seq_len, embed_dim), 0.1f32);
    let (grad_input, param_grads) = block.compute_gradients(&grad_output, &input);

    // Verify gradient shapes and finiteness
    assert_eq!(grad_input.shape(), &[seq_len, embed_dim]);
    assert!(grad_input.iter().all(|x: &f32| x.is_finite()));

    // Param grads should be present (Attn, FFN, Norms, FiLM, TimeCond)
    assert!(
        !param_grads.is_empty(),
        "Parameter gradients should be generated"
    );

    // Verify param grads are finite
    for grad in &param_grads {
        assert!(grad.iter().all(|x: &f32| x.is_finite()));
    }
}
