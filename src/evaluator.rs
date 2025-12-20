use crate::{Vocab, errors::Result, llm::LLM, metrics::text::corpus_bleu_1_2};

/// Evaluation and metrics functionality for language models
pub struct Evaluator;

impl Evaluator {
    /// Evaluate perplexity for diffusion models
    pub fn evaluate_perplexity_diffusion(llm: &mut LLM, data: Vec<&str>) -> Result<f32> {
        llm.evaluate_perplexity_diffusion(data)
    }

    /// Evaluate BLEU scores for generated text
    pub fn evaluate_bleu(llm: &LLM, inputs: Vec<&str>, outputs: Vec<&str>) -> Result<(f32, f32)> {
        llm.evaluate_bleu(inputs, outputs)
    }

    /// Get total parameter count
    pub fn total_parameters(llm: &LLM) -> usize {
        llm.total_parameters()
    }

    /// Get total weight norm (L2 norm of all parameters)
    pub fn total_weight_norm(llm: &LLM) -> f32 {
        llm.total_weight_norm()
    }

    /// Get network description
    pub fn network_description(llm: &LLM) -> String {
        llm.network_description()
    }

    /// Compute BLEU score between two texts
    pub fn compute_bleu(reference: &str, candidate: &str, vocab: &Vocab) -> (f32, f32) {
        let ref_tokens = vec![vocab.tokenize(reference)];
        let cand_tokens = vec![vocab.tokenize(candidate)];
        corpus_bleu_1_2(&ref_tokens, &cand_tokens)
    }
}
