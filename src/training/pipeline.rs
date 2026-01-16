use tracing::warn;

use crate::{Vocab, cli::Args, dataset_loader::Dataset, llm::LLM};

pub fn configure_speculative_sampling_from_args(
    args: &Args,
    config: &crate::model_config::ModelConfig,
    llm: &mut LLM,
) {
    if !args.speculative {
        return;
    }

    let gamma = args.speculative_gamma.max(1);
    let tau = args.speculative_tau.max(1e-6);
    let draft_layers = args
        .speculative_draft_layers
        .unwrap_or_else(|| config.num_layers.max(1))
        .max(1);

    let (mode, auto_detect_msg) = if let Some(ref mode_str) = args.speculative_mode {
        match mode_str.to_lowercase().as_str() {
            "transformer" | "trans" | "t" => (
                crate::layers::transformer::speculative::SpeculativeMode::Transformer,
                None,
            ),
            "diffusion" | "diff" | "d" => (
                crate::layers::transformer::speculative::SpeculativeMode::Diffusion,
                None,
            ),
            _ => {
                warn!(
                    "Unknown speculative mode '{}', auto-detecting from model type",
                    mode_str
                );
                (
                    if args.diffusion {
                        crate::layers::transformer::speculative::SpeculativeMode::Diffusion
                    } else {
                        crate::layers::transformer::speculative::SpeculativeMode::Transformer
                    },
                    None,
                )
            }
        }
    } else if args.diffusion {
        (
            crate::layers::transformer::speculative::SpeculativeMode::Diffusion,
            Some("Auto-detected speculative mode: Diffusion (based on --diffusion flag)"),
        )
    } else {
        (
            crate::layers::transformer::speculative::SpeculativeMode::Transformer,
            Some("Auto-detected speculative mode: Transformer (default model type)"),
        )
    };

    if let Some(existing) = llm.speculative_config() {
        let same_mode = llm.speculative_mode() == mode;
        let same_gamma = existing.gamma == gamma;
        let same_tau = (existing.tau - tau).abs() <= 1e-6;
        let same_draft_layers = existing.draft_layers == draft_layers;
        if same_mode && same_gamma && same_tau && same_draft_layers {
            return;
        }
    }

    if let Some(msg) = auto_detect_msg {
        println!("{msg}");
    }

    println!(
        "Enabling speculative sampling (mode={:?}, gamma={}, tau={}, draft_layers={})",
        mode, gamma, tau, draft_layers
    );
    llm.enable_speculative_sampling(gamma, tau, draft_layers, mode);
}

/// Orchestrate the complete training pipeline
pub fn run_training_pipeline(
    args: &Args,
    dataset: &Dataset,
    _vocab: &Vocab,
    config: &crate::model_config::ModelConfig,
    mut llm: LLM,
) -> crate::Result<LLM> {
    // Training-only auxiliary objectives.
    llm.set_residual_decorrelation_training(
        config.residual_decorrelation_weight,
        config.residual_decorrelation_adaptive,
    );

    llm.set_residual_hardneg_training(
        config.residual_hardneg_weight,
        config.residual_hardneg_adaptive,
        config.residual_hardneg_k,
        config.residual_hardneg_margin,
        config.residual_hardneg_temperature,
        config.residual_hardneg_bank_size,
    );

    // Configure speculative sampling if enabled
    if args.speculative {
        configure_speculative_sampling_from_args(args, config, &mut llm);
    }

    // Determine training mode and run appropriate training
    let use_eprop = args.eprop;
    if use_eprop {
        println!("\n✓ ES-D-RTRL E-PROP TRAINING MODE ENABLED");
        println!("Using online eligibility-based learning with O(N) trace approximation.");
        println!("ES-D-RTRL characteristics:");
        println!("  • Diagonal Jacobian approximation (D-RTRL)");
        println!("  • Rank-one exponential smoothing");
        println!("  • Forward-mode gradient computation");
        println!("  • Enhanced numerical stability controls");
        println!("  • O(N) complexity vs O(N²) standard e-prop\n");
    }

    // Run training based on architecture
    if args.trm {
        run_trm_training(args, dataset, &mut llm)?;
        llm.set_trm_inference_mode();
    } else if args.diffusion {
        run_diffusion_training(args, dataset, &mut llm)?;
    } else {
        run_standard_training(args, dataset, &mut llm)?;
    }

    Ok(llm)
}

/// Run TRM (Tiny Recursive Model) training
fn run_trm_training(args: &Args, dataset: &Dataset, llm: &mut LLM) -> crate::Result<()> {
    let pre_texts: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();
    let chat_texts: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.set_trm_training_mode();

    if let Some(n) = args.trm_recursions {
        llm.set_trm_recursions(n);
    }
    llm.set_trm_steps(args.trm_supervision_steps, args.trm_inference_steps);

    println!(
        "\n=== PRE-TRAINING LRM (CE) ===\nPre-training on {} examples for {} epochs",
        pre_texts.len(),
        args.pretrain_epochs
    );
    llm.train_with_warmup(pre_texts, args.pretrain_epochs, 0.0005, 4, 15)?;

    println!(
        "\n=== INSTRUCTION TUNING LRM (CE) ===\nInstruction tuning on {} examples for {} epochs",
        chat_texts.len(),
        args.instruction_epochs
    );
    llm.train_with_warmup(chat_texts, args.instruction_epochs, 0.0005, 4, 15)?;

    Ok(())
}

/// Run diffusion model training
fn run_diffusion_training(args: &Args, dataset: &Dataset, llm: &mut LLM) -> crate::Result<()> {
    let pre_texts: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.train_diffusion_ce(
        pre_texts,
        args.pretrain_epochs,
        0.0005,
        4,
        args.diffusion_ce_weight,
        args.validation_ratio,
        args.diffusion_min_snr_gamma,
        args.save_every.map(|n| n.get()),
        Some(args.checkpoint_dir.clone()),
        Some("pretrain".to_string()),
    )?;

    let chat_texts: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.train_diffusion_ce(
        chat_texts,
        args.instruction_epochs,
        0.0005,
        4,
        args.diffusion_ce_weight,
        args.validation_ratio,
        args.diffusion_min_snr_gamma,
        args.save_every.map(|n| n.get()),
        Some(args.checkpoint_dir.clone()),
        Some("instruction".to_string()),
    )?;

    Ok(())
}

/// Run standard transformer training
fn run_standard_training(args: &Args, dataset: &Dataset, llm: &mut LLM) -> crate::Result<()> {
    if args.continue_from.is_none() {
        println!("\n=== PRE-TRAINING MODEL ===");
        let pre_count = dataset.pretraining_data.len();
        println!(
            "Pre-training on {} examples for {} epochs with learning rate {}",
            pre_count, args.pretrain_epochs, 0.0005
        );
        let pre_texts: Vec<&str> = dataset
            .pretraining_data
            .iter()
            .map(|s| s.as_str())
            .collect();
        llm.train_with_warmup(pre_texts, args.pretrain_epochs, 0.0005, 4, 15)?;
    } else {
        println!("\n=== SKIPPING PRE-TRAINING ===");
        println!("Model already trained, proceeding directly to instruction tuning");
    }

    println!("\n=== INSTRUCTION TUNING ===");
    let instruction_lr = 0.0005;
    let instruction_epochs = args.instruction_epochs;
    let chat_count = dataset.chat_training_data.len();
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        chat_count, instruction_epochs, instruction_lr
    );
    let chat_texts: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();
    llm.train_with_warmup(chat_texts, instruction_epochs, instruction_lr, 4, 15)?;

    Ok(())
}
