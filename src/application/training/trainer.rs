use crate::{
    common::errors::Result,
    domain::{models::llm::LLM, richards::AdaptiveScalar},
};

/// Training functionality for language models
pub struct Trainer;

pub struct DiffusionCeTrainConfig {
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub ce_weight: AdaptiveScalar,
    pub validation_ratio: f32,
    pub min_snr_gamma: AdaptiveScalar,
    pub checkpoint_every: Option<usize>,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_stage: Option<String>,
}

impl Trainer {
    /// Basic training method
    pub fn train(llm: &mut LLM, data: Vec<&str>, epochs: usize, lr: f32) -> Result<()> {
        Self::train_with_batch_size(llm, data, epochs, lr, 1)
    }

    /// Train with configurable batch size for improved performance
    pub fn train_with_batch_size(
        llm: &mut LLM,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
    ) -> Result<()> {
        Self::train_with_warmup(llm, data, epochs, lr, batch_size, 15)
    }

    /// Train with learning rate warmup for stability
    ///
    /// Warmup prevents gradient explosion in early training by gradually increasing
    /// the learning rate from 0 to the target value over warmup_epochs.
    ///
    /// Reference: "Attention is All You Need" (Vaswani et al., 2017)
    pub fn train_with_warmup(
        llm: &mut LLM,
        data: Vec<&str>,
        epochs: usize,
        target_lr: f32,
        batch_size: usize,
        warmup_epochs: usize,
    ) -> Result<()> {
        llm.train_with_warmup(data, epochs, target_lr, batch_size, warmup_epochs)
    }

    /// Train with learning rate warmup and gradient accumulation.
    pub fn train_with_warmup_with_accumulation(
        llm: &mut LLM,
        data: Vec<&str>,
        epochs: usize,
        target_lr: f32,
        batch_size: usize,
        warmup_epochs: usize,
        gradient_accumulation_steps: usize,
    ) -> Result<()> {
        llm.train_with_warmup_with_accumulation(
            data,
            epochs,
            target_lr,
            batch_size,
            warmup_epochs,
            gradient_accumulation_steps,
        )
    }

    /// Train diffusion model with cross-entropy loss
    pub fn train_diffusion_ce(
        llm: &mut LLM,
        data: Vec<&str>,
        config: DiffusionCeTrainConfig,
    ) -> Result<()> {
        llm.train_diffusion_ce(
            data,
            config.epochs,
            config.lr,
            config.batch_size,
            config.ce_weight,
            config.validation_ratio,
            config.min_snr_gamma,
            config.checkpoint_every,
            config.checkpoint_dir,
            config.checkpoint_stage,
        )
    }

    /// Train diffusion model with cross-entropy loss and gradient accumulation
    pub fn train_diffusion_ce_with_accumulation(
        llm: &mut LLM,
        data: Vec<&str>,
        config: DiffusionCeTrainConfig,
    ) -> Result<()> {
        llm.train_diffusion_ce_with_accumulation(
            data,
            config.epochs,
            config.lr,
            config.batch_size,
            config.gradient_accumulation_steps,
            config.ce_weight,
            config.validation_ratio,
            config.min_snr_gamma,
            config.checkpoint_every,
            config.checkpoint_dir,
            config.checkpoint_stage,
        )
    }
}
