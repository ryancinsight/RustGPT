use std::num::NonZeroUsize;
use sysinfo::System;
use tracing::warn;

use crate::{
    application::encoding::Vocab,
    common::errors::Result,
    domain::{models::llm::LLM, richards::RichardsCurve},
    infrastructure::persistence::dataset::Dataset,
    presentation::cli::args::Args,
};

use crate::domain::richards::AdaptiveScalar;

const DEFAULT_BATCH_SIZE: usize = 4;
const DEFAULT_GRAD_ACCUM_STEPS: usize = 1;

#[derive(Clone, Copy, Debug)]
enum BatchingStage {
    Pretrain,
    Instruction,
    Diffusion,
}

#[derive(Clone, Copy, Debug)]
struct StageBatchingConfig {
    batch_size: usize,
    grad_accum_steps: usize,
}

impl StageBatchingConfig {
    #[inline]
    fn effective_batch(self) -> usize {
        self.batch_size.saturating_mul(self.grad_accum_steps)
    }
}

#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b.saturating_sub(1)) / b.max(1)
}

#[inline]
fn normalize_memory_units(bytes_or_kib: u64) -> u64 {
    // Handle older sysinfo unit behavior (KiB) without depending on crate internals.
    if bytes_or_kib < 1_000_000_000 {
        bytes_or_kib.saturating_mul(1024)
    } else {
        bytes_or_kib
    }
}

fn detect_available_memory_bytes() -> Option<u64> {
    let mut system = System::new();
    system.refresh_memory();
    let available = normalize_memory_units(system.available_memory());
    if available > 0 { Some(available) } else { None }
}

fn resolve_memory_budget_gib(args: &Args) -> f32 {
    let detected_gib = detect_available_memory_bytes()
        .map(|bytes| (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) as f32)
        .unwrap_or(8.0);

    // Keep training comfortably below RAM ceilings to avoid paging.
    let auto_budget = (detected_gib * 0.5).max(1.0);
    if let Some(user_budget) = args.memory_budget_gb {
        auto_budget.min(user_budget.max(1.0))
    } else {
        auto_budget
    }
}

fn pretraining_text_count(dataset: &Dataset) -> usize {
    let mut total = dataset.pretraining_data.len();
    for img in &dataset.image_training_data {
        total = total.saturating_add(1 + img.conversations.len());
    }
    for vid in &dataset.video_training_data {
        total = total.saturating_add(1 + vid.conversations.len());
    }
    for aud in &dataset.speech_training_data {
        total = total.saturating_add(1 + aud.conversations.len());
    }
    total
}

fn estimated_training_state_gib(model_params: usize) -> f32 {
    // Rough Adam footprint: params + grads + m + v (all f32) => 16 bytes per parameter.
    let bytes = (model_params as f64) * 16.0;
    (bytes / (1024.0 * 1024.0 * 1024.0)) as f32
}

fn model_size_batch_scale(model_params: usize, stage: BatchingStage) -> f32 {
    let params_m = (model_params as f64 / 1_000_000.0) as f32;
    let base: f32 = match params_m {
        x if x <= 5.0 => 1.0,
        x if x <= 20.0 => 0.85,
        x if x <= 50.0 => 0.70,
        x if x <= 100.0 => 0.55,
        x if x <= 250.0 => 0.40,
        _ => 0.30,
    };
    let stage_scale: f32 = match stage {
        BatchingStage::Pretrain => 0.9,
        BatchingStage::Instruction => 1.0,
        // Diffusion usually needs extra headroom.
        BatchingStage::Diffusion => 0.8,
    };
    (base * stage_scale).clamp(0.1_f32, 1.0_f32)
}

fn tune_stage_batching(
    stage: BatchingStage,
    num_examples: usize,
    memory_budget_gib: f32,
    model_params: usize,
) -> StageBatchingConfig {
    let data_target = match num_examples {
        0..=50_000 => 16,
        50_001..=250_000 => 32,
        250_001..=1_000_000 => 64,
        1_000_001..=5_000_000 => 128,
        _ => 256,
    };

    let estimated_model_state_gib = estimated_training_state_gib(model_params);
    let usable_memory_gib = (memory_budget_gib - estimated_model_state_gib).max(1.0);
    let model_scale = model_size_batch_scale(model_params, stage);

    let memory_effective_cap = match usable_memory_gib {
        x if x < 4.0 => 8,
        x if x < 8.0 => 16,
        x if x < 16.0 => 32,
        x if x < 32.0 => 64,
        _ => 128,
    };

    let memory_micro_cap = match usable_memory_gib {
        x if x < 4.0 => 1,
        x if x < 8.0 => 2,
        x if x < 16.0 => 4,
        x if x < 32.0 => 8,
        x if x < 64.0 => 16,
        _ => 32,
    };

    let stage_scale = match stage {
        BatchingStage::Pretrain => 1.0,
        BatchingStage::Instruction => 1.0,
        // Diffusion path generally carries heavier per-sample compute/memory.
        BatchingStage::Diffusion => 0.5,
    };

    let mut effective_target = ((data_target as f32) * stage_scale * model_scale).round() as usize;
    effective_target = effective_target.max(1).min(memory_effective_cap.max(1));
    if num_examples > 0 {
        effective_target = effective_target.min(num_examples.max(1));
    }

    let mut micro_batch = ((memory_micro_cap as f32) * stage_scale * model_scale).round() as usize;
    micro_batch = micro_batch.max(1).min(effective_target.max(1));
    if num_examples > 0 {
        micro_batch = micro_batch.min(num_examples.max(1));
    }

    let grad_accum_steps = ceil_div(effective_target.max(1), micro_batch.max(1)).clamp(1, 128);
    StageBatchingConfig {
        batch_size: micro_batch.max(1),
        grad_accum_steps,
    }
}

fn resolve_stage_batching(
    stage: BatchingStage,
    requested_batch_size: Option<usize>,
    requested_grad_accum_steps: Option<usize>,
    num_examples: usize,
    args: &Args,
    memory_budget_gib: f32,
    model_params: usize,
) -> StageBatchingConfig {
    if args.auto_tune_batching {
        let tuned = tune_stage_batching(stage, num_examples, memory_budget_gib, model_params);
        StageBatchingConfig {
            batch_size: requested_batch_size.unwrap_or(tuned.batch_size).max(1),
            grad_accum_steps: requested_grad_accum_steps
                .unwrap_or(tuned.grad_accum_steps)
                .max(1),
        }
    } else {
        StageBatchingConfig {
            batch_size: requested_batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1),
            grad_accum_steps: requested_grad_accum_steps
                .unwrap_or(DEFAULT_GRAD_ACCUM_STEPS)
                .max(1),
        }
    }
}

pub fn configure_speculative_sampling_from_args(
    args: &Args,
    config: &crate::domain::models::config::ModelConfig,
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
                crate::domain::layers::transformer::speculative::SpeculativeMode::Transformer,
                None,
            ),
            "diffusion" | "diff" | "d" => (
                crate::domain::layers::transformer::speculative::SpeculativeMode::Diffusion,
                None,
            ),
            _ => {
                warn!(
                    "Unknown speculative mode '{}', auto-detecting from model type",
                    mode_str
                );
                (
                    if args.diffusion {
                        crate::domain::layers::transformer::speculative::SpeculativeMode::Diffusion
                    } else {
                        crate::domain::layers::transformer::speculative::SpeculativeMode::Transformer
                    },
                    None,
                )
            }
        }
    } else if args.diffusion {
        (
            crate::domain::layers::transformer::speculative::SpeculativeMode::Diffusion,
            Some("Auto-detected speculative mode: Diffusion (based on --diffusion flag)"),
        )
    } else {
        (
            crate::domain::layers::transformer::speculative::SpeculativeMode::Transformer,
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
    config: &crate::domain::models::config::ModelConfig,
    mut llm: LLM,
) -> Result<LLM> {
    let model_params = llm.total_parameters();

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
    if args.diffusion {
        run_diffusion_training(args, dataset, &mut llm, model_params)?;
    } else {
        run_standard_training(args, dataset, &mut llm, model_params)?;
    }

    Ok(llm)
}

/// Run diffusion model training
fn run_diffusion_training(
    args: &Args,
    dataset: &Dataset,
    llm: &mut LLM,
    model_params: usize,
) -> Result<()> {
    // Construct adaptive scalars for diffusion hyperparameters
    let ce_weight = if args.diffusion_ce_weight_adaptive {
        let mut curve = RichardsCurve::new_default();
        // Sigmoid ramp: centered at m (halfway through training), steepness k
        // This allows the weight to ramp up from ~0 to output_scale over the course of training
        curve.m = Some(args.diffusion_ce_weight_curve_m as f64);
        curve.k = Some(args.diffusion_ce_weight_curve_k as f64);
        AdaptiveScalar::Richards {
            curve: Box::new(curve),
            output_scale: args.diffusion_ce_weight,
        }
    } else {
        AdaptiveScalar::Fixed(args.diffusion_ce_weight)
    };

    let min_snr_gamma = if args.diffusion_min_snr_gamma_adaptive {
        let mut curve = RichardsCurve::new_default();
        // Sigmoid ramp: centered at m (halfway through training), steepness k
        curve.m = Some(args.diffusion_min_snr_gamma_curve_m as f64);
        curve.k = Some(args.diffusion_min_snr_gamma_curve_k as f64);
        AdaptiveScalar::Richards {
            curve: Box::new(curve),
            output_scale: args.diffusion_min_snr_gamma,
        }
    } else {
        AdaptiveScalar::Fixed(args.diffusion_min_snr_gamma)
    };

    let pre_texts: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();
    let memory_budget_gib = resolve_memory_budget_gib(args);
    let pretrain_batching = resolve_stage_batching(
        BatchingStage::Diffusion,
        args.diffusion_batch_size,
        args.diffusion_gradient_accumulation_steps,
        pre_texts.len(),
        args,
        memory_budget_gib,
        model_params,
    );
    tracing::info!(
        stage = "diffusion-pretrain",
        auto_tune = args.auto_tune_batching,
        memory_budget_gib,
        model_params_m = (model_params as f64) / 1_000_000.0,
        est_training_state_gib = estimated_training_state_gib(model_params),
        dataset_examples = pre_texts.len(),
        batch_size = pretrain_batching.batch_size,
        grad_accum_steps = pretrain_batching.grad_accum_steps,
        effective_batch = pretrain_batching.effective_batch(),
        "Resolved training batching"
    );

    llm.train_diffusion_ce_with_accumulation(
        pre_texts,
        args.pretrain_epochs,
        0.0005,
        pretrain_batching.batch_size,
        pretrain_batching.grad_accum_steps,
        ce_weight.clone(),
        args.validation_ratio,
        min_snr_gamma.clone(),
        args.save_every.map(|n: NonZeroUsize| n.get()),
        Some(args.checkpoint_dir.clone()),
        Some("pretrain".to_string()),
    )?;

    let chat_texts: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();
    let instruction_batching = resolve_stage_batching(
        BatchingStage::Diffusion,
        args.diffusion_batch_size,
        args.diffusion_gradient_accumulation_steps,
        chat_texts.len(),
        args,
        memory_budget_gib,
        model_params,
    );
    tracing::info!(
        stage = "diffusion-instruction",
        auto_tune = args.auto_tune_batching,
        memory_budget_gib,
        model_params_m = (model_params as f64) / 1_000_000.0,
        est_training_state_gib = estimated_training_state_gib(model_params),
        dataset_examples = chat_texts.len(),
        batch_size = instruction_batching.batch_size,
        grad_accum_steps = instruction_batching.grad_accum_steps,
        effective_batch = instruction_batching.effective_batch(),
        "Resolved training batching"
    );

    llm.train_diffusion_ce_with_accumulation(
        chat_texts,
        args.instruction_epochs,
        0.0005,
        instruction_batching.batch_size,
        instruction_batching.grad_accum_steps,
        ce_weight,
        args.validation_ratio,
        min_snr_gamma,
        args.save_every.map(|n: NonZeroUsize| n.get()),
        Some(args.checkpoint_dir.clone()),
        Some("instruction".to_string()),
    )?;

    Ok(())
}

/// Run standard transformer training
fn run_standard_training(
    args: &Args,
    dataset: &Dataset,
    llm: &mut LLM,
    model_params: usize,
) -> Result<()> {
    // Log multimodal data usage
    let multimodal_stats = format!(
        "Training data: {} pretraining, {} chat, {} images, {} speech, {} video",
        dataset.pretraining_data.len(),
        dataset.chat_training_data.len(),
        dataset.image_training_data.len(),
        dataset.speech_training_data.len(),
        dataset.video_training_data.len()
    );
    println!("{}", multimodal_stats);
    tracing::info!("{}", multimodal_stats);

    let memory_budget_gib = resolve_memory_budget_gib(args);
    let pretrain_example_count = pretraining_text_count(dataset);
    let pretrain_batching = resolve_stage_batching(
        BatchingStage::Pretrain,
        args.pretrain_batch_size,
        args.pretrain_gradient_accumulation_steps,
        pretrain_example_count,
        args,
        memory_budget_gib,
        model_params,
    );
    let instruction_batching = resolve_stage_batching(
        BatchingStage::Instruction,
        args.instruction_batch_size,
        args.instruction_gradient_accumulation_steps,
        dataset.chat_training_data.len(),
        args,
        memory_budget_gib,
        model_params,
    );
    tracing::info!(
        stage = "pretrain",
        auto_tune = args.auto_tune_batching,
        memory_budget_gib,
        model_params_m = (model_params as f64) / 1_000_000.0,
        est_training_state_gib = estimated_training_state_gib(model_params),
        dataset_examples = pretrain_example_count,
        batch_size = pretrain_batching.batch_size,
        grad_accum_steps = pretrain_batching.grad_accum_steps,
        effective_batch = pretrain_batching.effective_batch(),
        "Resolved training batching"
    );
    tracing::info!(
        stage = "instruction",
        auto_tune = args.auto_tune_batching,
        memory_budget_gib,
        model_params_m = (model_params as f64) / 1_000_000.0,
        est_training_state_gib = estimated_training_state_gib(model_params),
        dataset_examples = dataset.chat_training_data.len(),
        batch_size = instruction_batching.batch_size,
        grad_accum_steps = instruction_batching.grad_accum_steps,
        effective_batch = instruction_batching.effective_batch(),
        "Resolved training batching"
    );

    if args.continue_from.is_none() {
        println!("\n=== PRE-TRAINING MODEL ===");
        if dataset.has_multimodal_data() {
            let all_text_data = dataset.get_all_text_data();
            println!(
                "Pre-training on {} text examples (including {} multimodal captions/transcripts) for {} epochs with learning rate {}, batch size {}, grad accumulation {}, effective batch {}",
                all_text_data.len(),
                dataset.image_training_data.len()
                    + dataset.speech_training_data.len()
                    + dataset.video_training_data.len(),
                args.pretrain_epochs,
                0.0005,
                pretrain_batching.batch_size,
                pretrain_batching.grad_accum_steps,
                pretrain_batching.effective_batch()
            );
            let pre_texts: Vec<&str> = all_text_data.iter().map(|s| s.as_str()).collect();
            llm.train_with_warmup_with_accumulation(
                pre_texts,
                args.pretrain_epochs,
                0.0005,
                pretrain_batching.batch_size,
                15,
                pretrain_batching.grad_accum_steps,
            )?;
        } else {
            println!(
                "Pre-training on {} text examples for {} epochs with learning rate {}, batch size {}, grad accumulation {}, effective batch {}",
                dataset.pretraining_data.len(),
                args.pretrain_epochs,
                0.0005,
                pretrain_batching.batch_size,
                pretrain_batching.grad_accum_steps,
                pretrain_batching.effective_batch()
            );
            let pre_texts: Vec<&str> = dataset
                .pretraining_data
                .iter()
                .map(|s| s.as_str())
                .collect();
            llm.train_with_warmup_with_accumulation(
                pre_texts,
                args.pretrain_epochs,
                0.0005,
                pretrain_batching.batch_size,
                15,
                pretrain_batching.grad_accum_steps,
            )?;
        }
    } else {
        println!("\n=== SKIPPING PRE-TRAINING ===");
        println!("Model already trained, proceeding directly to instruction tuning");
    }

    println!("\n=== INSTRUCTION TUNING ===");
    let instruction_lr = 0.0005;
    let instruction_epochs = args.instruction_epochs;
    let chat_count = dataset.chat_training_data.len();
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}, batch size {}, grad accumulation {}, effective batch {}",
        chat_count,
        instruction_epochs,
        instruction_lr,
        instruction_batching.batch_size,
        instruction_batching.grad_accum_steps,
        instruction_batching.effective_batch()
    );
    let chat_texts: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();
    llm.train_with_warmup_with_accumulation(
        chat_texts,
        instruction_epochs,
        instruction_lr,
        instruction_batching.batch_size,
        15,
        instruction_batching.grad_accum_steps,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    // Legacy pipeline tests removed during training-path simplification.
}
