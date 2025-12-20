use crate::llm::LLM;

/// Inference functionality for language models (prediction, sampling, tokenization)
pub struct InferenceEngine;

impl InferenceEngine {
    /// Generate text prediction from input
    pub fn predict(llm: &mut LLM, text: &str) -> String {
        llm.predict(text)
    }

    /// Sample from diffusion model
    pub fn sample_diffusion(llm: &mut LLM, max_length: usize, steps: Option<usize>) -> String {
        llm.sample_diffusion(max_length, steps)
    }

    /// Sample from diffusion model with prompt
    pub fn sample_diffusion_with_prompt(
        llm: &mut LLM,
        prompt: &str,
        max_length: usize,
        steps: Option<usize>,
    ) -> String {
        llm.sample_diffusion_with_prompt(prompt, max_length, steps)
    }

    /// Tokenize text into token IDs
    pub fn tokenize(llm: &LLM, text: &str) -> Vec<usize> {
        llm.tokenize(text)
    }
}
