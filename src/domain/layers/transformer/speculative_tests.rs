#[cfg(test)]
mod tests {
    use crate::{
        domain::layers::{
            diffusion::{
                DiffusionBlock, DiffusionBlockConfig, DiffusionPredictionTarget,
                EDM_SIGMA_DATA_DEFAULT, NoiseSchedule,
            },
            transformer::speculative::SpeculativeSamplingConfig,
        },
        domain::mixtures::HeadSelectionStrategy,
        domain::models::config::{
            DiffusionTimestepStrategy, TemporalMixingType, TitanMemoryConfig,
        },
    };
    fn create_dummy_block() -> DiffusionBlock {
        let config = DiffusionBlockConfig {
            embed_dim: 16,
            hidden_dim: 32,
            num_heads: 2,
            num_timesteps: 10,
            noise_schedule: NoiseSchedule::Linear {
                beta_min: 0.0001,
                beta_max: 0.02,
            },
            prediction_target: DiffusionPredictionTarget::Epsilon,
            edm_sigma_data: EDM_SIGMA_DATA_DEFAULT,
            timestep_strategy: DiffusionTimestepStrategy::Uniform,
            causal_attention: false,
            window_size: None,
            use_adaptive_window: false,
            discrete_masked: false,
            poly_degree: 1,
            max_pos: 10,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: TitanMemoryConfig::default(),
            time_embed_dim: 16,
            mask_token_id: None,
            temporal_mixing: TemporalMixingType::Attention,
            use_advanced_adaptive_residuals: false, // Disable for testing
            sampler: Default::default(),
            guidance: None,
            loss_weighting: Default::default(),
            use_p2_weighting: false,
            use_snr_weighting: false,
            adaptive_guidance: false,
            min_guidance_scale: 1.0,
            max_guidance_scale: 10.0,
            ddim_steps_policy: Default::default(),
        };
        DiffusionBlock::new(config)
    }

    #[test]
    fn test_speculative_sampling_runs() {
        let mut target_model = create_dummy_block();
        let mut draft_model = create_dummy_block();

        // Use new constructor instead of struct literal
        let config = SpeculativeSamplingConfig::new(2, 0.1, 1);

        let shape = (1, 16);
        let sample = target_model.speculative_sample(&mut draft_model, shape, Some(5), &config);

        assert_eq!(sample.shape(), &[1, 16]);
    }
}
