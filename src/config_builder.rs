use crate::{
    EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN,
    cli::Args,
    model_config::{ArchitectureType, AttentionType, ModelConfig, WindowAdaptationStrategy},
};

/// Build a complete model configuration from CLI arguments
pub fn build_model_config(args: &Args) -> ModelConfig {
    // Choose architecture based on CLI flags
    let architecture = if args.trm {
        ArchitectureType::TRM
    } else if args.diffusion {
        ArchitectureType::Diffusion
    } else {
        ArchitectureType::Autoregressive
    };

    let use_dynamic_tanh_norm = true;
    let num_kv_heads: Option<usize> = Some(4); // GQA with 4 KV heads
    let window_size: Option<usize> = Some(4096); // Mistral-style sliding window
    let use_adaptive_window: bool = true;
    let min_window_size: usize = 512;
    let max_window_size: usize = 4096;
    let window_adaptation_strategy = WindowAdaptationStrategy::AttentionEntropy;

    // Create base configuration
    let mut config =
        ModelConfig::transformer(EMBEDDING_DIM, HIDDEN_DIM, 1, MAX_SEQ_LEN, None, Some(8));

    // Apply architecture-specific settings
    config.architecture = architecture;
    config.diffusion_prediction_target = args.diffusion_prediction_target.into();
    config.diffusion_min_snr_gamma = args.diffusion_min_snr_gamma.max(1e-6);
    config.diffusion_noise_schedule = args.diffusion_noise_schedule.into();
    config.diffusion_timestep_strategy = args.diffusion_timestep_strategy.into();

    // Apply TRM-specific settings
    if args.trm {
        config.trm_use_diffusion = args.diffusion;
        config.trm_num_recursions = args.trm_recursions;
        config.trm_max_supervision_steps = args.trm_supervision_steps;
        config.trm_max_inference_steps = args.trm_inference_steps;
        config.trm_latent_moh_enabled = args.trm_latent_moh;
        config.trm_latent_moh_top_p_min = Some(args.trm_latent_moh_top_p_min);
        config.trm_latent_moh_top_p_max = Some(args.trm_latent_moh_top_p_max);
    }

    // Apply modern LLM enhancements
    config.use_dynamic_tanh_norm = use_dynamic_tanh_norm;
    config.num_kv_heads = num_kv_heads;
    config.window_size = window_size;
    config.use_adaptive_window = use_adaptive_window;
    config.min_window_size = min_window_size;
    config.max_window_size = max_window_size;
    config.window_adaptation_strategy = window_adaptation_strategy;

    // Set attention mechanism to PolyAttention
    config.attention = AttentionType::PolyAttention { degree_p: 3 };

    // Select temporal mixing mechanism (attention vs SSM-style RG-LRU)
    config.temporal_mixing = args.temporal_mixing.into();

    // Enable MoE if requested
    if args.moe {
        config.moe_router = Some(crate::mixtures::moe::ExpertRouter::LearnedMoE {
            num_experts: 4,
            num_active_experts: 2,
            expert_hidden_dim: HIDDEN_DIM / 2,
            load_balance_weight: 0.01,
            sparsity_weight: 0.001,
            diversity_weight: 0.005,
            routing_mode: crate::mixtures::moe::ExpertRoutingMode::TokenChoiceTopK,
            capacity_factor: 0.0,
            min_expert_capacity: 0,
            renormalize_after_capacity: true,
            z_loss_weight: 0.0,
            use_head_conditioning: true,
            use_learned_k_adaptation: true,
            shared_experts: vec![],
            shared_expert_scale: 0.0,
        });
    }

    config
}
