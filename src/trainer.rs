use crate::{errors::Result, llm::LLM};

/// Training functionality for language models
pub struct Trainer;

pub struct DiffusionCeTrainConfig {
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub ce_weight: f32,
    pub validation_ratio: f32,
    pub min_snr_gamma: f32,
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

    /// Train TRM model for autoencoding
    pub fn train_trm_autoencoding(
        llm: &mut LLM,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
    ) -> Result<()> {
        llm.train_trm_autoencoding(data, epochs, lr, batch_size)
    }

    /// Complete TRM training (autoencoding + generation)
    pub fn train_trm_complete(
        llm: &mut LLM,
        data: Vec<&str>,
        chat_data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        warmup_epochs: usize,
    ) -> Result<()> {
        llm.train_trm_complete(data, chat_data, epochs, batch_size, lr, warmup_epochs)
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
        )
    }
}
