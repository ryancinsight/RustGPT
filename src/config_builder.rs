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
    config.spiking_neuron_model = args.spiking.map(Into::into);

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

    // Residual decorrelation auxiliary objective (redundancy reduction)
    config.residual_decorrelation_weight = args.residual_decorrelation_weight.max(0.0);
    config.residual_decorrelation_adaptive = args.residual_decorrelation_adaptive;

    // Residual hard-negative repulsion objective
    config.residual_hardneg_weight = args.residual_hardneg_weight.max(0.0);
    config.residual_hardneg_adaptive = args.residual_hardneg_adaptive;
    config.residual_hardneg_k = args.residual_hardneg_k.max(1);
    config.residual_hardneg_margin = args.residual_hardneg_margin;
    config.residual_hardneg_temperature = args.residual_hardneg_temperature.max(1e-6);
    config.residual_hardneg_bank_size = args.residual_hardneg_bank_size;

    // Adaptive MoH threshold modulation
    config.moh_threshold_modulation = if args.moh_threshold_modulation_adaptive {
        let mut curve = crate::richards::RichardsCurve::default();
        curve.m = Some(args.moh_threshold_modulation_curve_m as f64);
        curve.k = Some(args.moh_threshold_modulation_curve_k as f64);
        crate::richards::adaptive::AdaptiveScalar::Richards {
            curve,
            output_scale: args.moh_threshold_modulation,
        }
    } else {
        crate::richards::adaptive::AdaptiveScalar::Fixed(args.moh_threshold_modulation)
    };

    let num_heads = config.get_num_heads().max(1);
    if args.hard_heads {
        config.head_selection = crate::mixtures::moh::HeadSelectionStrategy::Fixed {
            num_active: num_heads,
        };
    } else if args.eprop && args.moe {
        let num_active = num_heads.div_ceil(2).max(1);
        config.head_selection = crate::mixtures::moh::HeadSelectionStrategy::Learned {
            num_active,
            load_balance_weight: 0.01,
            complexity_loss_weight: 0.005,
            sparsity_weight: 0.001,
            importance_loss_weight: 0.0,
            switch_balance_weight: 0.0,
            training_mode: crate::mixtures::gating::GatingTrainingMode::Coupled,
        };
    }

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

    // Enable E-Prop if requested
    if args.eprop {
        config.eprop_enabled = true;
        // If spiking model is specified, use it for eprop config
        if let Some(spiking_cli) = args.spiking {
            config.eprop_neuron_config = Some(spiking_cli.into());
        } else {
            // Default to LIF if not specified
            config.eprop_neuron_config = Some(crate::eprop::config::NeuronConfig::lif());
        }
    }

    config
}
