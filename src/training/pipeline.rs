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
    if args.eprop {
        return Err(crate::errors::ModelError::Training {
            message: "--eprop is incompatible with the standard LLM training pipeline".to_string(),
        });
    }

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

pub fn run_eprop_training_pipeline(
    args: &Args,
    dataset: &Dataset,
    vocab: &Vocab,
    config: &crate::model_config::ModelConfig,
) -> crate::Result<()> {
    use ndarray::Array2;

    if args.trm || args.diffusion {
        return Err(crate::errors::ModelError::InvalidInput {
            message: "--eprop cannot be combined with --trm or --diffusion".to_string(),
        });
    }

    let vocab_size = vocab.size();
    let embedding_dim = config.embedding_dim;
    let num_neurons = config.hidden_dim;

    println!("\n✓ E-PROP TRAINING MODE ENABLED");
    println!("Using online eligibility-based learning (no backward pass).");
    println!(
        "E-prop model: input_dim={}, num_neurons={}, output_dim={}",
        embedding_dim, num_neurons, vocab_size
    );

    let mut eprop_config = crate::eprop::EPropConfig {
        num_neurons,
        input_dim: embedding_dim,
        output_dim: vocab_size,
        learning_rate: 5e-4,
        ..Default::default()
    };
    eprop_config.num_cycles = 1;

    let mut trainer = crate::eprop::EPropTrainer::new(eprop_config).map_err(|e| {
        crate::errors::ModelError::Training {
            message: format!("Failed to initialize EPropTrainer: {e}"),
        }
    })?;

    let token_embeddings = init_eprop_token_embeddings(vocab_size, embedding_dim);

    if args.continue_from.is_some() {
        warn!("--continue-from is ignored in --eprop mode (no e-prop checkpointing wired)");
    }

    if args.pretrain_epochs > 0 && !dataset.pretraining_data.is_empty() {
        println!(
            "\n=== E-PROP PRE-TRAINING ===\nTraining on {} examples for {} epochs",
            dataset.pretraining_data.len(),
            args.pretrain_epochs
        );
        train_eprop_language_model(
            &mut trainer,
            &token_embeddings,
            vocab,
            &dataset.pretraining_data,
            args.pretrain_epochs,
        )?;
    }

    if args.instruction_epochs > 0 && !dataset.chat_training_data.is_empty() {
        println!(
            "\n=== E-PROP INSTRUCTION TUNING ===\nTraining on {} examples for {} epochs",
            dataset.chat_training_data.len(),
            args.instruction_epochs
        );
        train_eprop_language_model(
            &mut trainer,
            &token_embeddings,
            vocab,
            &dataset.chat_training_data,
            args.instruction_epochs,
        )?;
    }

    if let Some(avg) = trainer.stats().avg_loss(100) {
        println!("\nE-prop training complete. Recent avg loss: {:.6}", avg);
    } else {
        println!("\nE-prop training complete.");
    }

    fn init_eprop_token_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        use rand_distr::{Distribution, Normal};

        let mut rng = crate::rng::get_rng();
        let normal = Normal::new(0.0, 0.02).expect("normal distribution parameters are valid");
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| normal.sample(&mut rng))
    }

    fn train_eprop_language_model(
        trainer: &mut crate::eprop::EPropTrainer,
        token_embeddings: &Array2<f32>,
        vocab: &Vocab,
        texts: &[String],
        epochs: usize,
    ) -> crate::Result<()> {
        let mut token_buf = Vec::new();

        for epoch in 0..epochs {
            let mut total_loss = 0.0f32;
            let mut steps = 0usize;

            trainer.reset_traces();

            for text in texts {
                vocab.tokenize_into(text, &mut token_buf);
                if token_buf.len() < 2 {
                    continue;
                }

                trainer.reset_state();

                for window in token_buf.windows(2) {
                    let input_id = window[0];
                    let target_id = window[1];

                    let input = token_embeddings.row(input_id).to_owned();

                    let loss = trainer
                        .train_step_classification(&input, target_id)
                        .map_err(|e| crate::errors::ModelError::Training {
                            message: format!("E-prop training step failed: {e}"),
                        })?;

                    total_loss += loss;
                    steps += 1;
                }
            }

            let avg_loss = if steps > 0 {
                total_loss / steps as f32
            } else {
                f32::NAN
            };

            println!(
                "Epoch {}/{}: avg_loss={:.6} (steps={})",
                epoch + 1,
                epochs,
                avg_loss,
                steps
            );
        }

        Ok(())
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn eprop_flag_rejects_standard_llm_pipeline() {
        let args = Args::parse_from(["llm", "--eprop"]);
        let dataset = Dataset {
            pretraining_data: vec!["hello world".to_string()],
            chat_training_data: vec!["hello".to_string()],
        };
        let vocab = Vocab::default();
        let config = crate::model_config::ModelConfig::transformer(8, 16, 1, 16, None, Some(1));
        let network = crate::model_builder::build_network(&config, &vocab);
        let llm = LLM::new(vocab.clone(), network);

        let res = run_training_pipeline(&args, &dataset, &vocab, &config, llm);
        assert!(matches!(
            res,
            Err(crate::errors::ModelError::Training { .. })
        ));
    }
}
