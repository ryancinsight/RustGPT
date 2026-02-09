use std::{collections::HashMap, fs};

use ndarray::{Array2, Axis, s};
use rand::Rng;
use rand_distr::Distribution;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument, warn};

use crate::{
    application::{decoding::GreedyDecoder, encoding::Vocab, training::ContinualLearningManager},
    common::{
        errors::{ModelError, Result},
        rng::get_rng,
    },
    domain::{
        layers::transformer::speculative::{SpeculativeMode, SpeculativeSamplingConfig},
        metrics::text::corpus_bleu_1_2,
        models::config::DiffusionTimestepStrategy,
        network::{Layer, LayerEnum},
        richards::AdaptiveScalar,
    },
};

impl LayerEnum {
    // Removed downcast helpers for SelfAttention/TRM to simplify to PolyAttention-only
}

#[derive(Debug, Clone)]
struct ToolCall {
    name: String,
    args: String,
}

#[derive(Debug)]
struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

#[derive(Debug, Clone)]
struct ToolDefinition {
    handler: fn(&str) -> Result<String>,
}

impl ToolRegistry {
    fn with_defaults() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };
        registry.register("calculator", tool_calculator);
        registry.register("echo", tool_echo);
        registry
    }

    fn register(&mut self, name: &str, handler: fn(&str) -> Result<String>) {
        self.tools.insert(name.to_string(), ToolDefinition { handler });
    }

    fn call(&self, name: &str, args: &str) -> Result<String> {
        let Some(tool) = self.tools.get(name) else {
            return Err(ModelError::InvalidInput {
                message: format!("Unknown tool: {}", name),
            });
        };
        (tool.handler)(args)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CalcToken {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Neg,
}

fn tool_echo(args: &str) -> Result<String> {
    Ok(args.to_string())
}

fn tool_calculator(args: &str) -> Result<String> {
    let value = eval_expression(args)?;
    Ok(value.to_string())
}

fn eval_expression(input: &str) -> Result<f64> {
    let tokens = tokenize_expr(input)?;
    let rpn = to_rpn(&tokens)?;
    eval_rpn(&rpn)
}

fn tokenize_expr(input: &str) -> Result<Vec<CalcToken>> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    let mut prev_op = true;
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch.is_ascii_digit() || ch == '.' {
            let mut buf = String::new();
            let mut has_dot = false;
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    buf.push(c);
                    chars.next();
                } else if c == '.' && !has_dot {
                    has_dot = true;
                    buf.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Some(&c) = chars.peek() {
                if c == 'e' || c == 'E' {
                    buf.push(c);
                    chars.next();
                    if let Some(&sign) = chars.peek() {
                        if sign == '+' || sign == '-' {
                            buf.push(sign);
                            chars.next();
                        }
                    }
                    let mut has_exp_digits = false;
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() {
                            buf.push(d);
                            chars.next();
                            has_exp_digits = true;
                        } else {
                            break;
                        }
                    }
                    if !has_exp_digits {
                        return Err(ModelError::InvalidInput {
                            message: "Invalid exponent".to_string(),
                        });
                    }
                }
            }
            let value = buf.parse::<f64>().map_err(|e| ModelError::InvalidInput {
                message: format!("Invalid number: {}", e),
            })?;
            tokens.push(CalcToken::Number(value));
            prev_op = false;
            continue;
        }
        match ch {
            '+' => {
                chars.next();
                if prev_op {
                    continue;
                }
                tokens.push(CalcToken::Plus);
                prev_op = true;
            }
            '-' => {
                chars.next();
                if prev_op {
                    tokens.push(CalcToken::Neg);
                } else {
                    tokens.push(CalcToken::Minus);
                    prev_op = true;
                }
            }
            '*' => {
                chars.next();
                tokens.push(CalcToken::Star);
                prev_op = true;
            }
            '/' => {
                chars.next();
                tokens.push(CalcToken::Slash);
                prev_op = true;
            }
            '(' => {
                chars.next();
                tokens.push(CalcToken::LParen);
                prev_op = true;
            }
            ')' => {
                chars.next();
                tokens.push(CalcToken::RParen);
                prev_op = false;
            }
            _ => {
                return Err(ModelError::InvalidInput {
                    message: format!("Invalid character: {}", ch),
                });
            }
        }
    }
    Ok(tokens)
}

fn precedence(tok: CalcToken) -> usize {
    match tok {
        CalcToken::Neg => 3,
        CalcToken::Star | CalcToken::Slash => 2,
        CalcToken::Plus | CalcToken::Minus => 1,
        _ => 0,
    }
}

fn is_right_assoc(tok: CalcToken) -> bool {
    matches!(tok, CalcToken::Neg)
}

fn to_rpn(tokens: &[CalcToken]) -> Result<Vec<CalcToken>> {
    let mut output = Vec::new();
    let mut ops: Vec<CalcToken> = Vec::new();
    for &tok in tokens {
        match tok {
            CalcToken::Number(_) => output.push(tok),
            CalcToken::Plus
            | CalcToken::Minus
            | CalcToken::Star
            | CalcToken::Slash
            | CalcToken::Neg => {
                let tok_prec = precedence(tok);
                while let Some(&top) = ops.last() {
                    if matches!(top, CalcToken::LParen) {
                        break;
                    }
                    let top_prec = precedence(top);
                    let should_pop = if is_right_assoc(tok) {
                        tok_prec < top_prec
                    } else {
                        tok_prec <= top_prec
                    };
                    if should_pop {
                        output.push(ops.pop().unwrap());
                    } else {
                        break;
                    }
                }
                ops.push(tok);
            }
            CalcToken::LParen => ops.push(tok),
            CalcToken::RParen => {
                let mut found = false;
                while let Some(top) = ops.pop() {
                    if matches!(top, CalcToken::LParen) {
                        found = true;
                        break;
                    }
                    output.push(top);
                }
                if !found {
                    return Err(ModelError::InvalidInput {
                        message: "Mismatched parentheses".to_string(),
                    });
                }
            }
        }
    }
    while let Some(top) = ops.pop() {
        if matches!(top, CalcToken::LParen | CalcToken::RParen) {
            return Err(ModelError::InvalidInput {
                message: "Mismatched parentheses".to_string(),
            });
        }
        output.push(top);
    }
    Ok(output)
}

fn eval_rpn(tokens: &[CalcToken]) -> Result<f64> {
    let mut stack: Vec<f64> = Vec::new();
    for &tok in tokens {
        match tok {
            CalcToken::Number(v) => stack.push(v),
            CalcToken::Neg => {
                let Some(v) = stack.pop() else {
                    return Err(ModelError::InvalidInput {
                        message: "Invalid expression".to_string(),
                    });
                };
                stack.push(-v);
            }
            CalcToken::Plus | CalcToken::Minus | CalcToken::Star | CalcToken::Slash => {
                let Some(b) = stack.pop() else {
                    return Err(ModelError::InvalidInput {
                        message: "Invalid expression".to_string(),
                    });
                };
                let Some(a) = stack.pop() else {
                    return Err(ModelError::InvalidInput {
                        message: "Invalid expression".to_string(),
                    });
                };
                let v = match tok {
                    CalcToken::Plus => a + b,
                    CalcToken::Minus => a - b,
                    CalcToken::Star => a * b,
                    CalcToken::Slash => {
                        if b == 0.0 {
                            return Err(ModelError::InvalidInput {
                                message: "Division by zero".to_string(),
                            });
                        }
                        a / b
                    }
                    _ => unreachable!(),
                };
                stack.push(v);
            }
            _ => {
                return Err(ModelError::InvalidInput {
                    message: "Invalid expression".to_string(),
                });
            }
        }
    }
    if stack.len() != 1 {
        return Err(ModelError::InvalidInput {
            message: "Invalid expression".to_string(),
        });
    }
    Ok(stack[0])
}

fn parse_tool_call(vocab: &Vocab, tokens: &[usize]) -> Result<ToolCall> {
    let mut parts: Vec<&str> = Vec::new();
    for &id in tokens {
        if let Some(tok) = vocab.decode(id) {
            parts.push(tok);
        }
    }
    let name = parse_tool_name(&parts)?;
    let args = parse_tool_args(&parts)?;
    Ok(ToolCall { name, args })
}

fn parse_tool_name(tokens: &[&str]) -> Result<String> {
    for (idx, tok) in tokens.iter().enumerate() {
        if tok.eq_ignore_ascii_case("name") {
            if let Some(eq_idx) = tokens[idx + 1..].iter().position(|t| *t == "=") {
                let val_idx = idx + 1 + eq_idx + 1;
                if let Some(name) = tokens.get(val_idx) {
                    return Ok(name.to_string());
                }
            }
        }
    }
    Err(ModelError::InvalidInput {
        message: "Missing tool name".to_string(),
    })
}

fn parse_tool_args(tokens: &[&str]) -> Result<String> {
    for (idx, tok) in tokens.iter().enumerate() {
        if tok.eq_ignore_ascii_case("args") {
            if let Some(eq_idx) = tokens[idx + 1..].iter().position(|t| *t == "=") {
                let val_idx = idx + 1 + eq_idx + 1;
                if val_idx >= tokens.len() {
                    return Err(ModelError::InvalidInput {
                        message: "Missing tool args".to_string(),
                    });
                }
                let args = tokens[val_idx..].join(" ");
                return Ok(args);
            }
        }
    }
    Err(ModelError::InvalidInput {
        message: "Missing tool args".to_string(),
    })
}

fn response_span_from_tokens(vocab: &Vocab, tokens: &[usize]) -> Option<(usize, usize)> {
    if tokens.is_empty() {
        return None;
    }
    let mut seen_user_tag = false;
    for (idx, &tid) in tokens.iter().enumerate() {
        let Some(text) = vocab.decode(tid) else {
            continue;
        };
        if text.eq_ignore_ascii_case("user") {
            seen_user_tag = true;
            continue;
        }
        if !seen_user_tag {
            continue;
        }
        if text.eq_ignore_ascii_case("assistant") {
            let colon_after = tokens
                .get(idx + 1)
                .and_then(|&next_id| vocab.decode(next_id))
                .map(|tok| tok == ":")
                .unwrap_or(false);
            if !colon_after {
                continue;
            }
            let start = idx + 2; // skip "Assistant" and following ':'
            if start >= tokens.len() {
                return None;
            }
            let mut end = tokens.len();
            if tokens
                .last()
                .and_then(|&id| vocab.decode(id))
                .is_some_and(|last_tok| last_tok == "</s>" && end > start)
            {
                end -= 1;
            }
            if start >= end {
                return None;
            }
            return Some((start, end));
        }
    }
    None
}

#[derive(Serialize, Deserialize, Debug)]
pub enum DecoderType {
    Greedy(GreedyDecoder),
}

impl DecoderType {
    pub fn layer_type(&self) -> &str {
        match self {
            DecoderType::Greedy(_) => "GreedyDecoder",
        }
    }

    pub fn parameters(&self) -> usize {
        match self {
            DecoderType::Greedy(_) => 0, // Greedy has no parameters
        }
    }
}

#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<LayerEnum>,
    decoder: DecoderType,
    // EMA of median per-layer gradient norm to stabilize adaptive LR balance
    median_grad_ema: Option<f32>,
    #[serde(default)]
    speculative_config: Option<SpeculativeSamplingConfig>,
    #[serde(default)]
    speculative_mode: SpeculativeMode,

    // Scratch buffers (not serialized) for allocation-free tokenization on repeated inference
    // calls.
    #[serde(skip, default)]
    tokenize_scratch: Vec<usize>,

    /// Optional runtime override for diffusion sampling steps (e.g. from CLI).
    ///
    /// Not serialized: checkpoints should carry model defaults via diffusion block config.
    #[serde(skip, default)]
    diffusion_steps_override: Option<usize>,

    /// Training-only hyperparameters (not serialized).
    #[serde(skip, default)]
    training_hparams: TrainingHyperParams,

    /// Non-serialized memory bank for hard-negative residual repulsion.
    #[serde(skip, default)]
    residual_neg_bank: ResidualNegBank,

    /// Non-serialized scratch buffers for training to avoid re-allocations.
    #[serde(skip, default)]
    training_scratch: TrainingScratch,
    #[serde(skip, default)]
    tool_registry: ToolRegistry,

    /// Continual learning manager for online learning from user feedback
    #[serde(skip, default)]
    continual_learning: Option<ContinualLearningManager>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrainingHyperParams {
    residual_decorrelation_weight: f32,
    residual_decorrelation_adaptive: bool,

    residual_hardneg_weight: f32,
    residual_hardneg_adaptive: bool,
    residual_hardneg_k: usize,
    residual_hardneg_margin: f32,
    residual_hardneg_temperature: f32,
    residual_hardneg_bank_size: usize,
}

#[derive(Debug, Default)]
struct ResidualNegBank {
    items: Vec<Vec<f32>>,
    next: usize,
}

impl ResidualNegBank {
    fn push(&mut self, v: Vec<f32>, max: usize) {
        if max == 0 {
            return;
        }
        if self.items.len() < max {
            self.items.push(v);
            return;
        }
        if self.items.is_empty() {
            self.items.push(v);
            self.next = 0;
            return;
        }
        let idx = self.next % max;
        self.items[idx] = v;
        self.next = (self.next + 1) % max;
    }

    fn as_slice(&self) -> &[Vec<f32>] {
        self.items.as_slice()
    }
}

#[derive(Debug, Default)]
struct TrainingScratch {
    accumulated_param_grads: Vec<Vec<Array2<f32>>>,
    layer_grad_norms: Vec<f32>,
    layer_inputs: Vec<Array2<f32>>,

    // For train_diffusion_ce
    grads_per_layer: Vec<Option<Vec<Array2<f32>>>>,
}

impl TrainingScratch {
    /// Reset scratch buffers for a new training batch.
    fn reset(&mut self, network_len: usize) {
        // Ensure outer vectors have correct length, but reuse inner allocations.
        if self.accumulated_param_grads.len() != network_len {
            self.accumulated_param_grads = (0..network_len).map(|_| Vec::new()).collect();
        } else {
            for grads in &mut self.accumulated_param_grads {
                grads.clear();
            }
        }

        if self.layer_grad_norms.len() != network_len {
            self.layer_grad_norms = vec![0.0; network_len];
        } else {
            for norm in &mut self.layer_grad_norms {
                *norm = 0.0;
            }
        }

        if self.grads_per_layer.len() != network_len {
            self.grads_per_layer = vec![None; network_len];
        } else {
            for slot in &mut self.grads_per_layer {
                *slot = None;
            }
        }

        self.layer_inputs.clear();
    }
}

impl std::fmt::Debug for LLM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLM")
            .field("vocab", &self.vocab)
            .field("network", &self.network)
            .finish()
    }
}

impl Default for LLM {
    fn default() -> Self {
        use crate::domain::models::{builder::build_network, config::ModelConfig};

        let config = ModelConfig::default();
        let vocab = Vocab::default();
        let network = build_network(&config, &vocab);

        let decoder = DecoderType::Greedy(GreedyDecoder::new());

        Self {
            vocab,
            network,
            decoder,
            median_grad_ema: None,
            speculative_config: None,
            speculative_mode: SpeculativeMode::Diffusion, /* Default to diffusion mode for
                                                           * backward compatibility */
            tokenize_scratch: Vec::new(),
            diffusion_steps_override: None,
            training_hparams: TrainingHyperParams::default(),
            residual_neg_bank: ResidualNegBank::default(),
            training_scratch: TrainingScratch::default(),
            tool_registry: ToolRegistry::with_defaults(),
            continual_learning: None,
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab, network: Vec<LayerEnum>) -> Self {
        let decoder = DecoderType::Greedy(GreedyDecoder::new());

        Self {
            vocab,
            network,
            decoder,
            median_grad_ema: None,
            speculative_config: None,
            speculative_mode: SpeculativeMode::Diffusion, /* Default to diffusion mode for
                                                           * backward compatibility */
            tokenize_scratch: Vec::new(),
            diffusion_steps_override: None,
            training_hparams: TrainingHyperParams::default(),
            residual_neg_bank: ResidualNegBank::default(),
            training_scratch: TrainingScratch::default(),
            tool_registry: ToolRegistry::with_defaults(),
            continual_learning: None,
        }
    }

    pub fn set_residual_decorrelation_training(&mut self, weight: f32, adaptive: bool) {
        self.training_hparams.residual_decorrelation_weight = weight.max(0.0);
        self.training_hparams.residual_decorrelation_adaptive = adaptive;
    }

    pub fn set_residual_hardneg_training(
        &mut self,
        weight: f32,
        adaptive: bool,
        k: usize,
        margin: f32,
        temperature: f32,
        bank_size: usize,
    ) {
        self.training_hparams.residual_hardneg_weight = weight.max(0.0);
        self.training_hparams.residual_hardneg_adaptive = adaptive;
        self.training_hparams.residual_hardneg_k = k.max(1);
        self.training_hparams.residual_hardneg_margin = margin;
        self.training_hparams.residual_hardneg_temperature = temperature.max(1e-6);
        self.training_hparams.residual_hardneg_bank_size = bank_size;
    }

    /// Create LLM with GreedyDecoder
    pub fn with_greedy_decoder(vocab: Vocab, network: Vec<LayerEnum>) -> Self {
        let decoder = DecoderType::Greedy(GreedyDecoder::new());
        Self {
            vocab,
            network,
            decoder,
            median_grad_ema: None,
            speculative_config: None,
            speculative_mode: SpeculativeMode::Diffusion, /* Default to diffusion mode for
                                                           * backward compatibility */
            tokenize_scratch: Vec::new(),
            diffusion_steps_override: None,
            training_hparams: TrainingHyperParams::default(),
            residual_neg_bank: ResidualNegBank::default(),
            training_scratch: TrainingScratch::default(),
            tool_registry: ToolRegistry::with_defaults(),
            continual_learning: None,
        }
    }

    /// Switch to GreedyDecoder
    pub fn enable_greedy(&mut self) {
        let decoder = DecoderType::Greedy(GreedyDecoder::new());
        self.decoder = decoder;
    }

    pub fn set_diffusion_steps_override(&mut self, steps: Option<usize>) {
        self.diffusion_steps_override = steps;
    }

    pub fn enable_speculative_sampling(
        &mut self,
        gamma: usize,
        tau: f32,
        draft_layers: usize,
        mode: SpeculativeMode,
    ) {
        if gamma == 0 || draft_layers == 0 {
            warn!(
                "Speculative sampling requested with invalid gamma={} or draft_layers={}",
                gamma, draft_layers
            );
            self.speculative_config = None;
            return;
        }
        // Use the new constructor which handles clamping
        let cfg = SpeculativeSamplingConfig::new(gamma, tau, draft_layers);
        self.speculative_config = Some(cfg);
        self.speculative_mode = mode;
        info!(
            "Enabled speculative sampling: mode={}, {}",
            mode,
            cfg.description()
        );
    }

    /// Disable speculative sampling, revert to greedy decoding
    pub fn disable_speculative_sampling(&mut self) {
        self.speculative_config = None;
        info!("Disabled speculative sampling, using greedy decoding");
    }

    /// Check if speculative sampling is enabled
    pub fn is_speculative_enabled(&self) -> bool {
        self.speculative_config.is_some()
    }

    /// Get the current speculative sampling configuration (if enabled)
    pub fn speculative_config(&self) -> Option<&SpeculativeSamplingConfig> {
        self.speculative_config.as_ref()
    }

    /// Get the current speculative mode
    pub fn speculative_mode(&self) -> SpeculativeMode {
        self.speculative_mode
    }

    /// Generate next token using speculative sampling for transformers
    ///
    /// This implements speculative decoding where a lightweight draft model (early layers)
    /// proposes candidate tokens, and the full model verifies them.
    ///
    /// Algorithm:
    /// 1. Draft phase: Generate γ candidate tokens using only draft_layers of the model
    /// 2. Verify phase: Score candidates with full model
    /// 3. Accept/reject: Use probability ratio threshold τ for rejection sampling
    ///
    /// Reference: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.,
    /// 2022)
    pub fn generate_speculative_transformer(
        &mut self,
        current_tokens: &[usize],
        gamma: usize,
        tau: f32,
        draft_layers: usize,
    ) -> usize {
        use ndarray::Array2;

        // Ensure we have tokens to work with
        if current_tokens.is_empty() {
            return self.vocab.encode("</s>").unwrap_or(0);
        }

        let vocab_size = self.vocab.size();

        // Convert tokens to embeddings (convert to f32)
        let token_ids_f32 = Array2::from_shape_vec(
            (1, current_tokens.len()),
            current_tokens.iter().map(|&x| x as f32).collect(),
        )
        .expect("Failed to create token array");

        // Forward pass through embeddings
        let mut draft_hidden = self.network[0].forward(&token_ids_f32); // TokenEmbeddings

        // Forward through draft layers (early layers of main model)
        // Use fewer layers for faster draft generation
        let draft_end_idx = draft_layers.min(self.network.len().saturating_sub(2));

        for i in 1..=draft_end_idx {
            draft_hidden = self.network[i].forward(&draft_hidden);
        }

        // Get draft logits (using output projection)
        let draft_logits = if let Some(LayerEnum::OutputProjection(op)) = self.network.last_mut() {
            op.forward(&draft_hidden)
        } else {
            draft_hidden.clone()
        };

        // Get probabilities for last position from draft model
        let last_row = draft_logits.row(draft_logits.shape()[0] - 1);
        let draft_probs = crate::domain::soft::Softmax::new().forward_immutable_row(&last_row);

        // Get top-γ candidates from draft model
        let candidates = self.get_top_k_tokens_from_probs(&draft_probs, gamma);

        if candidates.is_empty() {
            // Fallback to greedy from draft
            return draft_probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        // Get full model probabilities for verification
        // Run through all layers (full model)
        let full_logits = self.get_sequence_logit_row(current_tokens);
        let target_probs = crate::domain::soft::Softmax::new().forward_immutable_row(&full_logits.view());

        // Speculative decoding acceptance with rejection sampling
        // Accept token i with probability min(1, p_target(i) / p_draft(i))
        let mut rng = get_rng();

        for &candidate_token in &candidates {
            if candidate_token >= vocab_size {
                continue; // Skip invalid tokens
            }

            let q_draft = draft_probs[candidate_token].max(1e-10);
            let q_target = target_probs[candidate_token].max(1e-10);

            // Rejection sampling: accept with probability min(1, q_target/q_draft)
            let acceptance_prob = (q_target / q_draft).min(1.0);

            // For tau threshold mode: accept if ratio exceeds tau
            // For probabilistic mode: accept with probability = acceptance_prob
            if acceptance_prob >= tau {
                // Additional probabilistic rejection for better distribution matching
                let r: f32 = rng.random();
                if r < acceptance_prob {
                    return candidate_token;
                }
            }
        }

        // No candidates accepted - sample from adjusted distribution
        // p_adjusted = max(0, p_target - p_draft) normalized
        // This ensures we sample from the "residual" of the target distribution
        let mut sum = 0.0f32;
        for i in 0..vocab_size {
            let p_adj = (target_probs[i] - draft_probs[i]).max(0.0);
            sum += p_adj;
        }

        if sum > 1e-10 {
            // Sample from adjusted distribution
            let r: f32 = rng.random::<f32>() * sum;
            let mut cumsum = 0.0f32;
            for i in 0..vocab_size {
                cumsum += (target_probs[i] - draft_probs[i]).max(0.0);
                if cumsum >= r {
                    return i;
                }
            }
        }

        // Ultimate fallback: greedy from target
        target_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(candidates[0])
    }

    /// Get logit for the last position of a sequence
    fn get_sequence_logit_row(&mut self, tokens: &[usize]) -> ndarray::Array1<f32> {
        use ndarray::{Array1, Array2};

        if tokens.is_empty() {
            return Array1::zeros(self.vocab.size());
        }

        let mut token_ids = Array2::<f32>::zeros((1, tokens.len()));
        for (i, &token) in tokens.iter().enumerate() {
            token_ids[[0, i]] = token as f32;
        }

        // Forward through embeddings
        let mut hidden = self.network[0].forward(&token_ids);

        // Similarity context threaded across successive TransformerBlock layers.
        let mut similarity_ctx: Option<Array2<f32>> = None;

        // Forward through all layers except output projection
        let network_len = self.network.len();
        for i in 1..network_len {
            match &mut self.network[i] {
                LayerEnum::OutputProjection(_) => break,
                LayerEnum::TransformerBlock(block) => {
                    block.set_incoming_similarity_context(similarity_ctx.as_ref());
                    hidden = block.forward(&hidden);
                    if let Some(existing) = similarity_ctx.as_mut() {
                        existing.assign(block.activation_similarity_matrix());
                    } else {
                        similarity_ctx = Some(block.activation_similarity_matrix().clone());
                    }
                }
                LayerEnum::DiffusionBlock(block) => {
                    block.set_incoming_similarity_context(similarity_ctx.as_ref());
                    hidden = block.forward(&hidden);
                    if let Some(existing) = similarity_ctx.as_mut() {
                        existing.assign(block.activation_similarity_matrix());
                    } else {
                        similarity_ctx = Some(block.activation_similarity_matrix().clone());
                    }
                }
                LayerEnum::LRM(block) => {
                    block.set_incoming_similarity_context(similarity_ctx.as_ref());
                    hidden = block.forward(&hidden);
                    if let Some(existing) = similarity_ctx.as_mut() {
                        existing.assign(block.activation_similarity_matrix());
                    } else {
                        similarity_ctx = Some(block.activation_similarity_matrix().clone());
                    }
                }
                layer => {
                    similarity_ctx = None;
                    hidden = layer.forward(&hidden);
                }
            }
        }

        // Apply output projection if it exists
        let logits = if let Some(LayerEnum::OutputProjection(op)) = self.network.last_mut() {
            op.forward(&hidden)
        } else {
            hidden
        };

        // Return logits for the last position
        logits.row(logits.shape()[0] - 1).to_owned()
    }

    /// Get top-k token IDs from a probability row.
    ///
    /// Uses a fixed-size min-heap so this is $O(V \log k)$ rather than sorting the whole vocab.
    fn get_top_k_tokens_from_probs(&self, probs: &ndarray::Array1<f32>, k: usize) -> Vec<usize> {
        use std::{
            cmp::{Ordering, Reverse},
            collections::BinaryHeap,
        };

        #[derive(Copy, Clone, Debug)]
        struct Score(f32);
        impl PartialEq for Score {
            fn eq(&self, other: &Self) -> bool {
                self.0.to_bits() == other.0.to_bits()
            }
        }
        impl Eq for Score {}
        impl PartialOrd for Score {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Score {
            fn cmp(&self, other: &Self) -> Ordering {
                match (self.0.is_nan(), other.0.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    (false, false) => self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal),
                }
            }
        }

        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<(Reverse<Score>, usize)> = BinaryHeap::with_capacity(k + 1);

        for (i, &p) in probs.iter().enumerate() {
            let score = Score(p);
            if heap.len() < k {
                heap.push((Reverse(score), i));
                continue;
            }
            let Some((Reverse(min_score), _)) = heap.peek() else {
                continue;
            };
            if score > *min_score {
                heap.pop();
                heap.push((Reverse(score), i));
            }
        }

        let mut out: Vec<(Score, usize)> = heap.into_iter().map(|(Reverse(s), i)| (s, i)).collect();
        out.sort_by(|a, b| b.0.cmp(&a.0));
        out.into_iter().map(|(_, i)| i).collect()
    }
}

impl LLM {
    fn forward_diffusion_stack(
        &mut self,
        block_indices: &[usize],
        input: &Array2<f32>,
        t_idx: usize,
    ) -> Array2<f32> {
        let mut hidden = input.clone();
        let mut similarity_ctx: Option<Array2<f32>> = None;
        for &idx in block_indices {
            if let LayerEnum::DiffusionBlock(block) = &mut self.network[idx] {
                block.set_timestep(t_idx);
                block.set_incoming_similarity_context(similarity_ctx.as_ref());
                hidden = block.forward_with_timestep(&hidden, t_idx);
                if let Some(existing) = similarity_ctx.as_mut() {
                    existing.assign(block.activation_similarity_matrix());
                } else {
                    similarity_ctx = Some(block.activation_similarity_matrix().clone());
                }
            }
        }
        hidden
    }

    fn apply_ddim_step(
        &self,
        scheduler_block_idx: usize,
        current: &Array2<f32>,
        t_idx: usize,
        predicted_noise: &Array2<f32>,
    ) -> Array2<f32> {
        if let LayerEnum::DiffusionBlock(block) = &self.network[scheduler_block_idx] {
            block
                .noise_scheduler
                .ddim_step(current, t_idx, predicted_noise, 0.0, None)
        } else {
            current.clone()
        }
    }

    pub fn network_description(&self) -> String {
        let network_layers = self.network.iter().map(|layer| layer.layer_type()).fold(
            String::new(),
            |mut acc, layer_type| {
                if !acc.is_empty() {
                    acc.push_str(", ");
                }
                acc.push_str(layer_type);
                acc
            },
        );

        // Include decoder type in the description
        // Show speculative decoder when enabled, otherwise show base decoder type
        let decoder_desc = match (&self.speculative_config, self.speculative_mode) {
            (Some(cfg), SpeculativeMode::Transformer) => {
                format!(
                    "SpeculativeDecoder(γ={}, τ={:.4}, layers={})",
                    cfg.gamma, cfg.tau, cfg.draft_layers
                )
            }
            (Some(cfg), SpeculativeMode::Diffusion) => {
                format!(
                    "SpeculativeDiffusion(γ={}, τ={:.4}, layers={})",
                    cfg.gamma, cfg.tau, cfg.draft_layers
                )
            }
            (None, _) => self.decoder.layer_type().to_string(),
        };

        format!("{}, {}", network_layers, decoder_desc)
    }

    /// Get a detailed decoder description including speculative mode info
    pub fn decoder_description(&self) -> String {
        match (&self.speculative_config, self.speculative_mode) {
            (Some(cfg), mode) => {
                format!(
                    "Speculative {} (γ={}, τ={:.4}, draft_layers={}, temp={:.2}, top_p={:.2})",
                    mode, cfg.gamma, cfg.tau, cfg.draft_layers, cfg.temperature, cfg.top_p
                )
            }
            (None, _) => "Greedy (deterministic argmax)".to_string(),
        }
    }

    pub fn total_parameters(&self) -> usize {
        // Sum the parameters across all layers in the network
        let network_params = self
            .network
            .iter()
            .map(|layer| layer.parameters())
            .sum::<usize>();

        // Add decoder parameters
        network_params + self.decoder.parameters()
    }

    /// Set TRM layers to inference mode for faster prediction
    pub fn set_trm_inference_mode(&mut self) {
        for layer in &mut self.network {
            if let LayerEnum::LRM(lrm) = layer {
                lrm.set_training_mode(false);
            }
        }
    }

    /// Set TRM layers to training mode for full supervision steps
    pub fn set_trm_training_mode(&mut self) {
        for layer in &mut self.network {
            match layer {
                LayerEnum::LRM(lrm) => {
                    lrm.set_training_mode(true);
                }
                LayerEnum::TransformerBlock(block) => {
                    block.set_training_mode(true);
                }
                _ => {}
            }
        }
    }

    pub fn set_trm_recursions(&mut self, n: usize) {
        for layer in &mut self.network {
            if let LayerEnum::LRM(lrm) = layer {
                lrm.set_recursions(n);
            }
        }
    }

    pub fn set_trm_steps(&mut self, supervision: Option<usize>, inference: Option<usize>) {
        for layer in &mut self.network {
            if let LayerEnum::LRM(lrm) = layer {
                if let Some(s) = supervision {
                    lrm.set_supervision_steps(s);
                }
                if let Some(i) = inference {
                    lrm.set_inference_steps(i);
                }
            }
        }
    }

    #[inline]
    pub fn predict(&mut self, text: &str) -> String {
        let output_tokens = self.forward(text);

        // Handle empty output
        if output_tokens.is_empty() {
            return String::new();
        }

        // Convert token_ids to a string (pre-alloc + robust unknown fallback)
        self.vocab.decode_tokens_to_string(&output_tokens)
    }

    #[inline]
    pub fn predict_with_limit(&mut self, text: &str, max_new_tokens: usize) -> String {
        let output_tokens = self.forward_with_limit(text, max_new_tokens);
        if output_tokens.is_empty() {
            return String::new();
        }
        self.vocab.decode_tokens_to_string(&output_tokens)
    }

    pub fn max_sequence_len(&self) -> usize {
        let mut max_len = 0usize;
        for layer in &self.network {
            let candidate = match layer {
                LayerEnum::TransformerBlock(block) => Some(block.max_seq_len()),
                LayerEnum::DiffusionBlock(block) => Some(block.max_seq_len()),
                LayerEnum::LRM(lrm) => lrm.max_seq_len(),
                _ => None,
            };
            if let Some(len) = candidate {
                if len > max_len {
                    max_len = len;
                }
            }
        }
        max_len
    }

    #[inline]
    fn forward(&mut self, text: &str) -> Vec<usize> {
        self.forward_with_limit(text, usize::MAX)
    }

    #[inline]
    fn forward_with_limit(&mut self, text: &str, max_new_tokens: usize) -> Vec<usize> {
        // Tokenize the input text (reuse a scratch Vec to avoid repeated allocations).
        // We `take` the buffer out of `self` so we don't hold a mutable borrow of `self` across
        // calls that also require `&mut self`.
        let mut tokenized = std::mem::take(&mut self.tokenize_scratch);
        self.vocab.tokenize_into(text, &mut tokenized);
        let mut output_tokens: Vec<usize> = Vec::new();

        // Safety check: ensure we have at least one token
        if tokenized.is_empty() {
            self.tokenize_scratch = tokenized;
            return output_tokens;
        }

        let input_len = tokenized.len();
        let max_seq_len = self.max_sequence_len().max(input_len.max(1));

        // Pre-allocate to avoid repeated growth reallocations during generation.
        output_tokens.reserve(max_seq_len.saturating_sub(input_len));

        // Hoist EOS lookup out of the loop.
        let eos_token = self.vocab.encode("</s>");
        let tool_call_start = self.vocab.encode("<tool_call>");
        let tool_call_end = self.vocab.encode("</tool_call>");
        let tool_result_start = self.vocab.encode("<tool_result>");
        let tool_result_end = self.vocab.encode("</tool_result>");

        // Prevent overflow if input_len >= max_seq_len
        if input_len >= max_seq_len {
            self.tokenize_scratch = tokenized;
            return output_tokens;
        }

        let available_steps = max_seq_len.saturating_sub(input_len);
        let generation_steps = available_steps.min(max_new_tokens);
        for _ in 0..generation_steps {
            // let tokenized_clone = tokenized.clone();

            // Check if we're approaching the maximum sequence length
            if output_tokens.len() >= max_seq_len.saturating_sub(1) {
                break;
            }

            let mut token_input = Array2::zeros((1, tokenized.len()));
            for (i, &token_id) in tokenized.iter().enumerate() {
                token_input[[0, i]] = token_id as f32;
            }
            let mut input = token_input;

            // Forward pass through all layers except output projection to get hidden states
            // Similarity context threaded across successive TransformerBlock layers.
            let mut similarity_ctx: Option<Array2<f32>> = None;

            for layer in self.network.iter_mut() {
                input = match layer {
                    LayerEnum::TransformerBlock(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(&input);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    LayerEnum::DiffusionBlock(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(&input);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    LayerEnum::LRM(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(&input);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    _ => {
                        similarity_ctx = None;
                        layer.forward(&input)
                    }
                };
            }

            let logits = input;

            // Safety check: ensure we have at least one token
            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit_row = logits.row(logits.shape()[0] - 1);

            let next_token = if let (Some(cfg), SpeculativeMode::Transformer) =
                (self.speculative_config, self.speculative_mode)
            {
                // Use speculative sampling for transformers
                self.generate_speculative_transformer(
                    tokenized.as_slice(),
                    cfg.gamma,
                    cfg.tau,
                    cfg.draft_layers,
                )
            } else {
                // Use regular decoding
                match &mut self.decoder {
                    DecoderType::Greedy(decoder) => {
                        // Simple greedy decoding: argmax directly from logits (no softmax needed)
                        decoder.decode_row(last_logit_row)
                    }
                }
            };

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if let (Some(start_id), Some(end_id)) = (tool_call_start, tool_call_end) {
                if next_token == end_id {
                    let end_pos = output_tokens.len().saturating_sub(1);
                    if end_pos > 0 {
                        if let Some(start_pos) =
                            output_tokens[..end_pos].iter().rposition(|&id| id == start_id)
                        {
                            if start_pos < end_pos {
                                let call_tokens = &output_tokens[start_pos + 1..end_pos];
                                let tool_result = match parse_tool_call(&self.vocab, call_tokens) {
                                    Ok(call) => match self.tool_registry.call(&call.name, &call.args)
                                    {
                                        Ok(res) => res,
                                        Err(err) => format!("ToolError: {}", err),
                                    },
                                    Err(err) => format!("ToolError: {}", err),
                                };
                                let tool_output = if let (Some(rs), Some(re)) =
                                    (tool_result_start, tool_result_end)
                                {
                                    let _ = (rs, re);
                                    format!(
                                        "<tool_result> {} </tool_result>",
                                        tool_result
                                    )
                                } else {
                                    tool_result
                                };
                                let tool_tokens = self.vocab.tokenize(&tool_output);
                                output_tokens.extend(tool_tokens.iter().copied());
                                tokenized.extend(tool_tokens);
                            }
                        }
                    }
                }
            }

            if eos_token.is_some_and(|eos| next_token == eos) {
                break;
            }
        }

        self.tokenize_scratch = tokenized;
        output_tokens
    }

    #[instrument(skip(self, data))]
    pub fn train(&mut self, data: Vec<&str>, epochs: usize, lr: f32) -> Result<()> {
        self.train_with_batch_size(data, epochs, lr, 1)
    }

    /// Train with configurable batch size for improved performance
    pub fn train_with_batch_size(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
    ) -> Result<()> {
        self.train_with_warmup(data, epochs, lr, batch_size, 15) // 15 warmup epochs for better stability
    }

    /// Train with learning rate warmup for stability
    ///
    /// Warmup prevents gradient explosion in early training by gradually increasing
    /// the learning rate from 0 to the target value over warmup_epochs.
    ///
    /// Reference: "Attention is All You Need" (Vaswani et al., 2017)
    pub fn train_with_warmup(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        target_lr: f32,
        batch_size: usize,
        warmup_epochs: usize,
    ) -> Result<()> {
        // Set TRM layers to training mode (full supervision steps)
        self.set_trm_training_mode();

        // Store previous richards_glu richards weights for delta tracking
        let mut prev_richards_glu_weights: Vec<Vec<f64>> = Vec::new();

        let mut scratch = std::mem::take(&mut self.training_scratch);
        let res: Result<()> = (|| {
            for epoch in 0..epochs {
                let t_epoch_start = std::time::Instant::now();
                let mut total_loss = 0.0;
                let mut total_base_loss = 0.0;
                let mut total_grad_norm = 0.0;
                let mut batch_count = 0;
                let mut total_examples = 0usize;
                let mut per_layer_param_grad_norm_sq: Vec<f32> = vec![0.0; self.network.len()];

                // Learning rate warmup + cosine annealing
                // Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov &
                // Hutter, 2016)
                let effective_lr = if epoch < warmup_epochs {
                    // Linear warmup: gradually increase LR from 0 to target
                    target_lr * ((epoch + 1) as f32 / warmup_epochs as f32)
                } else {
                    // Cosine annealing after warmup to escape loss plateaus
                    // Formula: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
                    let t = (epoch - warmup_epochs) as f32;
                    let t_max = (epochs - warmup_epochs) as f32;
                    let lr_min = target_lr * 0.10; // Minimum LR is 10% of base LR (gentler decay)
                    let lr_max = target_lr;

                    lr_min
                        + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos())
                };

                // Compute training progress for adaptive MoH
                let training_progress = if epoch < warmup_epochs {
                    0.0
                } else {
                    (epoch - warmup_epochs) as f64 / (epochs - warmup_epochs) as f64
                };
                for layer in &mut self.network {
                    layer.set_training_progress(training_progress);
                }
                // Process data in batches
                for batch_strs in data.chunks(batch_size.max(1)) {
                    let batch_tokenized: Vec<Vec<usize>> = batch_strs
                        .par_iter()
                        .map(|input| self.tokenize(input))
                        .collect();

                    let (batch_loss, batch_base_loss, grad_norm, layer_param_grad_norm_sq) =
                        self.train_batch_profiled(&batch_tokenized, effective_lr, &mut scratch)?;
                    total_loss += batch_loss;
                    total_base_loss += batch_base_loss;
                    total_grad_norm += grad_norm;
                    batch_count += 1;
                    total_examples += batch_tokenized.len();
                    for (i, s) in layer_param_grad_norm_sq.into_iter().enumerate() {
                        if i < per_layer_param_grad_norm_sq.len() {
                            per_layer_param_grad_norm_sq[i] += s;
                        }
                    }
                }

                let avg_loss = total_loss / batch_count as f32;
                let avg_base_loss = total_base_loss / batch_count as f32;
                let avg_grad_norm = total_grad_norm / batch_count as f32;
                let per_layer_rms: Vec<f32> = per_layer_param_grad_norm_sq
                    .iter()
                    .map(|&s| (s / (batch_count as f32).max(1.0)).sqrt())
                    .collect();

                // Normalize by parameter count so layers with fewer parameters (e.g., RichardsNorm)
                // are not misinterpreted as "dead" purely due to scale differences.
                let layer_param_counts: Vec<usize> = self
                    .network
                    .iter()
                    .map(|layer| layer.parameters().max(1))
                    .collect();
                let per_layer_rms_per_param: Vec<f32> = per_layer_rms
                    .iter()
                    .enumerate()
                    .map(|(i, &raw)| {
                        let param_count = layer_param_counts.get(i).copied().unwrap_or(1) as f32;
                        if param_count > 0.0 {
                            raw / param_count.sqrt()
                        } else {
                            raw
                        }
                    })
                    .collect();

                tracing::info!(
                    epoch = epoch,
                    per_layer_rms = ?per_layer_rms,
                    per_layer_rms_per_param = ?per_layer_rms_per_param,
                    layer_param_counts = ?layer_param_counts,
                    "Transformer epoch layer param grad RMS"
                );
                let names: Vec<&str> = self.network.iter().map(|l| l.layer_type()).collect();
                tracing::debug!(epoch = epoch, per_layer = ?names, per_layer_rms = ?per_layer_rms, "Layer RMS breakdown");

                // NFR-5.2: Training divergence detection
                if avg_loss.is_nan() || avg_loss.is_infinite() {
                    return Err(ModelError::Training {
                        message: format!(
                            "Training diverged at epoch {}: loss is {} (NaN or Inf detected)",
                            epoch, avg_loss
                        ),
                    });
                }

                if avg_loss > 1e6 {
                    return Err(ModelError::Training {
                        message: format!(
                            "Training diverged at epoch {}: loss exceeded threshold (loss = {:.2e} > 1e6)",
                            epoch, avg_loss
                        ),
                    });
                }

                // Aggregate MoH instrumentation from PolyAttention layers at epoch end
                let mut tau_min_epoch = f32::INFINITY;
                let mut tau_max_epoch = f32::NEG_INFINITY;
                let mut tau_available = false;
                let mut pred_norm_sum = 0.0f32;
                let mut pred_norm_count = 0usize;
                let mut avg_heads_per_token_sum = 0.0f32;
                let mut heads_layers_count = 0usize;
                let mut total_heads_sum = 0usize;
                let mut avg_experts_sum = 0.0f32;
                let mut significant_experts_sum = 0.0f32;
                let mut routing_entropy_sum = 0.0f32;
                let mut experts_load_cv_sum = 0.0f32;
                let mut experts_load_cv_count = 0usize;
                let mut experts_layers_count = 0usize;
                let mut total_experts_sum = 0usize;

                for layer in &mut self.network {
                    if let LayerEnum::PolyAttention(pa) = layer {
                        if let Some((min_tau, max_tau)) = pa.take_tau_metrics() {
                            tau_available = true;
                            if min_tau < tau_min_epoch {
                                tau_min_epoch = min_tau;
                            }
                            if max_tau > tau_max_epoch {
                                tau_max_epoch = max_tau;
                            }
                        }
                        if let Some(rms_g) = pa.take_pred_norm() {
                            pred_norm_sum += rms_g;
                            pred_norm_count += 1;
                        }
                        let per_head = pa.get_head_metrics_and_reset();
                        if !per_head.is_empty() {
                            let layer_avg_active_heads =
                                per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                            avg_heads_per_token_sum += layer_avg_active_heads;
                            heads_layers_count += 1;
                            total_heads_sum += per_head.len();
                        }
                    }
                    if let LayerEnum::TransformerBlock(block) = layer {
                        // Pull through MoH instrumentation from the temporal-mixing layer.
                        match block.temporal_mixing_mut() {
                            crate::domain::layers::components::common::TemporalMixingLayer::Attention(
                                attn,
                            ) => {
                                if let Some((min_tau, max_tau)) = attn.take_tau_metrics() {
                                    tau_available = true;
                                    if min_tau < tau_min_epoch {
                                        tau_min_epoch = min_tau;
                                    }
                                    if max_tau > tau_max_epoch {
                                        tau_max_epoch = max_tau;
                                    }
                                }
                                if let Some(rms_g) = attn.take_pred_norm() {
                                    pred_norm_sum += rms_g;
                                    pred_norm_count += 1;
                                }
                                let per_head = attn.get_head_metrics_and_reset();
                                if !per_head.is_empty() {
                                    let layer_avg_active_heads =
                                        per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                                    avg_heads_per_token_sum += layer_avg_active_heads;
                                    heads_layers_count += 1;
                                    total_heads_sum += per_head.len();
                                }
                            }
                            crate::domain::layers::components::common::TemporalMixingLayer::RgLruMoH(
                                rglru,
                            ) => {
                                if let Some((min_tau, max_tau)) = rglru.take_tau_metrics() {
                                    tau_available = true;
                                    if min_tau < tau_min_epoch {
                                        tau_min_epoch = min_tau;
                                    }
                                    if max_tau > tau_max_epoch {
                                        tau_max_epoch = max_tau;
                                    }
                                }
                                if let Some(rms_g) = rglru.take_pred_norm() {
                                    pred_norm_sum += rms_g;
                                    pred_norm_count += 1;
                                }
                                let per_head = rglru.get_head_metrics_and_reset();
                                if !per_head.is_empty() {
                                    let layer_avg_active_heads =
                                        per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                                    avg_heads_per_token_sum += layer_avg_active_heads;
                                    heads_layers_count += 1;
                                    total_heads_sum += per_head.len();
                                }
                            }
                            crate::domain::layers::components::common::TemporalMixingLayer::MambaMoH(m) => {
                                if let Some((min_tau, max_tau)) = m.take_tau_metrics() {
                                    tau_available = true;
                                    if min_tau < tau_min_epoch {
                                        tau_min_epoch = min_tau;
                                    }
                                    if max_tau > tau_max_epoch {
                                        tau_max_epoch = max_tau;
                                    }
                                }
                                if let Some(rms_g) = m.take_pred_norm() {
                                    pred_norm_sum += rms_g;
                                    pred_norm_count += 1;
                                }
                                let per_head = m.get_head_metrics_and_reset();
                                if !per_head.is_empty() {
                                    let layer_avg_active_heads =
                                        per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                                    avg_heads_per_token_sum += layer_avg_active_heads;
                                    heads_layers_count += 1;
                                    total_heads_sum += per_head.len();
                                }
                            }
                            crate::domain::layers::components::common::TemporalMixingLayer::Mamba2MoH(
                                m,
                            ) => {
                                if let Some((min_tau, max_tau)) = m.take_tau_metrics() {
                                    tau_available = true;
                                    if min_tau < tau_min_epoch {
                                        tau_min_epoch = min_tau;
                                    }
                                    if max_tau > tau_max_epoch {
                                        tau_max_epoch = max_tau;
                                    }
                                }
                                if let Some(rms_g) = m.take_pred_norm() {
                                    pred_norm_sum += rms_g;
                                    pred_norm_count += 1;
                                }
                                let per_head = m.get_head_metrics_and_reset();
                                if !per_head.is_empty() {
                                    let layer_avg_active_heads =
                                        per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                                    avg_heads_per_token_sum += layer_avg_active_heads;
                                    heads_layers_count += 1;
                                    total_heads_sum += per_head.len();
                                }
                            }
                            _ => {}
                        }

                        // Pull through MoE metrics when MoE is used inside the block.
                        if let crate::domain::layers::components::common::FeedForwardVariant::MixtureOfExperts(
                            moe,
                        ) = block.feedforward()
                        {
                        let layer_avg_active_experts = moe.config.get_avg_active_experts();
                        let layer_significant_experts = moe.config.get_avg_significant_experts();
                        let layer_routing_entropy = moe.config.get_routing_entropy();
                        let (_v, _sd, cv) = moe.config.gating.metrics.get_load_distribution_stats();
                        avg_experts_sum += layer_avg_active_experts;
                        significant_experts_sum += layer_significant_experts;
                        routing_entropy_sum += layer_routing_entropy;
                        experts_load_cv_sum += if cv.is_finite() { cv } else { 0.0 };
                        experts_load_cv_count += 1;
                        experts_layers_count += 1;
                        total_experts_sum += moe.config.num_experts;
                    }
                    }
                    if let LayerEnum::DiffusionBlock(block) = layer {
                        // Pull through MoE metrics when MoE is used inside the diffusion block.
                        if let crate::domain::layers::components::common::FeedForwardVariant::MixtureOfExperts(
                        moe,
                    ) = &block.feedforward
                    {
                        let layer_avg_active_experts = moe.config.get_avg_active_experts();
                        let layer_significant_experts = moe.config.get_avg_significant_experts();
                        let layer_routing_entropy = moe.config.get_routing_entropy();
                        let (_v, _sd, cv) = moe.config.gating.metrics.get_load_distribution_stats();
                        avg_experts_sum += layer_avg_active_experts;
                        significant_experts_sum += layer_significant_experts;
                        routing_entropy_sum += layer_routing_entropy;
                        experts_load_cv_sum += if cv.is_finite() { cv } else { 0.0 };
                        experts_load_cv_count += 1;
                        experts_layers_count += 1;
                        total_experts_sum += moe.config.num_experts;
                    }
                    }
                    if let LayerEnum::LRM(lrm) = layer {
                        if let Some((min_tau, max_tau)) = lrm.attention_mut().take_tau_metrics() {
                            tau_available = true;
                            if min_tau < tau_min_epoch {
                                tau_min_epoch = min_tau;
                            }
                            if max_tau > tau_max_epoch {
                                tau_max_epoch = max_tau;
                            }
                        }
                        if let Some(rms_g) = lrm.attention_mut().take_pred_norm() {
                            pred_norm_sum += rms_g;
                            pred_norm_count += 1;
                        }
                        let per_head = lrm.attention_mut().get_head_metrics_and_reset();
                        if !per_head.is_empty() {
                            let layer_avg_active_heads =
                                per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                            avg_heads_per_token_sum += layer_avg_active_heads;
                            heads_layers_count += 1;
                            total_heads_sum += per_head.len();
                        }

                        // Pull through MoE metrics when MoE is used inside the recursive core
                        // block. LRM wraps either a TransformerBlock or
                        // DiffusionBlock.
                        let guard = lrm.block.read().unwrap();
                        match &*guard {
                        crate::domain::layers::recurrence::lrm::RecursiveBlockVariant::Transformer(b) => {
                            if let crate::domain::layers::components::common::FeedForwardVariant::MixtureOfExperts(moe) =
                                b.feedforward()
                            {
                                let layer_avg_active_experts = moe.config.get_avg_active_experts();
                                let layer_significant_experts = moe.config.get_avg_significant_experts();
                                let layer_routing_entropy = moe.config.get_routing_entropy();
                                let (_v, _sd, cv) = moe.config.gating.metrics.get_load_distribution_stats();
                                avg_experts_sum += layer_avg_active_experts;
                                significant_experts_sum += layer_significant_experts;
                                routing_entropy_sum += layer_routing_entropy;
                                experts_load_cv_sum += if cv.is_finite() { cv } else { 0.0 };
                                experts_load_cv_count += 1;
                                experts_layers_count += 1;
                                total_experts_sum += moe.config.num_experts;
                            }
                        }
                        crate::domain::layers::recurrence::lrm::RecursiveBlockVariant::Diffusion(b) => {
                            if let crate::domain::layers::components::common::FeedForwardVariant::MixtureOfExperts(moe) =
                                &b.feedforward
                            {
                                let layer_avg_active_experts = moe.config.get_avg_active_experts();
                                let layer_significant_experts = moe.config.get_avg_significant_experts();
                                let layer_routing_entropy = moe.config.get_routing_entropy();
                                let (_v, _sd, cv) = moe.config.gating.metrics.get_load_distribution_stats();
                                avg_experts_sum += layer_avg_active_experts;
                                significant_experts_sum += layer_significant_experts;
                                routing_entropy_sum += layer_routing_entropy;
                                experts_load_cv_sum += if cv.is_finite() { cv } else { 0.0 };
                                experts_load_cv_count += 1;
                                experts_layers_count += 1;
                                total_experts_sum += moe.config.num_experts;
                            }
                        }
                    }
                    }
                    if let LayerEnum::MixtureOfExperts(moe) = layer {
                        let layer_avg_active_experts = moe.config.get_avg_active_experts();
                        let layer_significant_experts = moe.config.get_avg_significant_experts();
                        let layer_routing_entropy = moe.config.get_routing_entropy();
                        let (_v, _sd, cv) = moe.config.gating.metrics.get_load_distribution_stats();
                        avg_experts_sum += layer_avg_active_experts;
                        significant_experts_sum += layer_significant_experts;
                        routing_entropy_sum += layer_routing_entropy;
                        experts_load_cv_sum += if cv.is_finite() { cv } else { 0.0 };
                        experts_load_cv_count += 1;
                        experts_layers_count += 1;
                        total_experts_sum += moe.config.num_experts;
                    }
                }

                let tau_min_log = if tau_available {
                    Some(tau_min_epoch)
                } else {
                    None
                };
                let tau_max_log = if tau_available {
                    Some(tau_max_epoch)
                } else {
                    None
                };
                let tau_range_log = if tau_available {
                    Some(tau_max_epoch - tau_min_epoch)
                } else {
                    None
                };
                let pred_norm_rms = if pred_norm_count > 0 {
                    pred_norm_sum / pred_norm_count as f32
                } else {
                    0.0
                };
                let pred_norm_rms_log = if pred_norm_count > 0 {
                    Some(pred_norm_rms)
                } else {
                    None
                };
                let avg_active_heads = if heads_layers_count > 0 {
                    avg_heads_per_token_sum / heads_layers_count as f32
                } else {
                    0.0
                };
                let avg_active_heads_log = if heads_layers_count > 0 {
                    Some(avg_active_heads)
                } else {
                    None
                };
                let avg_active_experts = if experts_layers_count > 0 {
                    avg_experts_sum / experts_layers_count as f32
                } else {
                    0.0
                };
                let avg_significant_experts = if experts_layers_count > 0 {
                    significant_experts_sum / experts_layers_count as f32
                } else {
                    0.0
                };
                let avg_routing_entropy = if experts_layers_count > 0 {
                    routing_entropy_sum / experts_layers_count as f32
                } else {
                    0.0
                };
                let experts_load_cv = if experts_load_cv_count > 0 {
                    experts_load_cv_sum / experts_load_cv_count as f32
                } else {
                    0.0
                };

                // Presentable (active/total) counts and a coupled ratio.
                let total_heads = if heads_layers_count > 0 {
                    ((total_heads_sum as f32) / (heads_layers_count as f32))
                        .round()
                        .max(0.0) as usize
                } else {
                    0
                };
                let total_experts = if experts_layers_count > 0 {
                    ((total_experts_sum as f32) / (experts_layers_count as f32))
                        .round()
                        .max(0.0) as usize
                } else {
                    0
                };

                let avg_active_heads_s = if avg_active_heads.is_finite() {
                    avg_active_heads.max(0.0)
                } else {
                    0.0
                };
                let avg_significant_experts_s = if avg_significant_experts.is_finite() {
                    avg_significant_experts.max(0.0)
                } else {
                    0.0
                };

                let active_heads = if total_heads > 0 {
                    avg_active_heads_s.round().clamp(0.0, total_heads as f32) as usize
                } else {
                    0
                };
                // For display, treat "active experts" as those with significant weight (> 0.1).
                let active_experts = if total_experts > 0 {
                    avg_significant_experts_s
                        .round()
                        .clamp(0.0, total_experts as f32) as usize
                } else {
                    0
                };
                let heads_per_expert = if active_experts > 0 {
                    active_heads as f32 / active_experts as f32
                } else {
                    0.0
                };

                // Balanced discrete distribution implied by (active_heads, active_experts).
                // If active_heads is not divisible by active_experts, the best possible split is:
                // - remainder experts get ceil(active_heads/active_experts)
                // - the rest get floor(active_heads/active_experts)
                let (heads_per_expert_min, heads_per_expert_max, heads_per_expert_remainder) =
                    if active_experts > 0 {
                        let min_h = active_heads / active_experts;
                        let rem = active_heads % active_experts;
                        let max_h = min_h + if rem > 0 { 1 } else { 0 };
                        (min_h, max_h, rem)
                    } else {
                        (0, 0, 0)
                    };

                tracing::info!(
                    epoch = epoch,
                    tau_available = tau_available,
                    tau_min = ?tau_min_log,
                    tau_max = ?tau_max_log,
                    tau_range = ?tau_range_log,
                    pred_norm_rms = ?pred_norm_rms_log,
                    avg_active_heads = ?avg_active_heads_log,
                    active_heads = active_heads,
                    total_heads = total_heads,
                    avg_active_experts = avg_active_experts,
                    avg_significant_experts = avg_significant_experts,
                    active_experts = active_experts,
                    total_experts = total_experts,
                    heads_per_expert = heads_per_expert,
                    heads_per_expert_min = heads_per_expert_min,
                    heads_per_expert_max = heads_per_expert_max,
                    heads_per_expert_remainder = heads_per_expert_remainder,
                    avg_routing_entropy = avg_routing_entropy,
                    experts_load_cv = experts_load_cv,
                    "Attention/MoH/MoE metrics: heads {}/{}; experts {}/{}; heads/expert {:.2}",
                    active_heads,
                    total_heads,
                    active_experts,
                    total_experts,
                    heads_per_expert
                );

                // Collect current richards_glu richards weights for delta tracking
                let mut current_richards_glu_weights: Vec<Vec<f64>> = Vec::new();
                let mut richards_training_status: Vec<bool> = Vec::new();
                for layer in &self.network {
                    if let LayerEnum::RichardsGlu(richards_glu) = layer {
                        current_richards_glu_weights.push(richards_glu.gate.weights());
                        richards_training_status.push(richards_glu.gate.has_trained_parameters());
                    }
                }

                // Debug: Check if Richards parameters are being trained
                let trained_layers = richards_training_status
                    .iter()
                    .filter(|&&trained| trained)
                    .count();
                if !current_richards_glu_weights.is_empty() {
                    tracing::debug!(
                        "RichardsGlu training status: {}/{} layers have trained parameters",
                        trained_layers,
                        current_richards_glu_weights.len()
                    );
                }

                // Compute delta changes in richards_glu richards coefficients
                let mut richards_glu_delta_sum = 0.0;
                let mut richards_glu_param_count = 0;
                let mut total_weight_changes = 0;
                let mut significant_changes = 0;

                if !prev_richards_glu_weights.is_empty()
                    && current_richards_glu_weights.len() == prev_richards_glu_weights.len()
                {
                    for (layer_idx, (prev_layer, curr_layer)) in prev_richards_glu_weights
                        .iter()
                        .zip(current_richards_glu_weights.iter())
                        .enumerate()
                    {
                        if prev_layer.len() == curr_layer.len() {
                            for (param_idx, (prev_w, curr_w)) in
                                prev_layer.iter().zip(curr_layer.iter()).enumerate()
                            {
                                let delta = (curr_w - prev_w).abs();
                                richards_glu_delta_sum += delta;
                                richards_glu_param_count += 1;
                                total_weight_changes += 1;

                                // Count significant changes (> 1e-6 relative change)
                                if delta > 1e-6 {
                                    significant_changes += 1;
                                }

                                // Debug: Log unusual parameter values
                                if delta > 1.0 {
                                    tracing::debug!(
                                        "Large Richards parameter change in layer {} param {}: {:.6} -> {:.6} (delta: {:.6})",
                                        layer_idx,
                                        param_idx,
                                        prev_w,
                                        curr_w,
                                        delta
                                    );
                                }
                            }
                        } else {
                            tracing::warn!(
                                "RichardsGlu layer {} weight length mismatch: prev={}, curr={}",
                                layer_idx,
                                prev_layer.len(),
                                curr_layer.len()
                            );
                        }
                    }
                } else if prev_richards_glu_weights.is_empty() {
                    tracing::debug!("No previous RichardsGlu weights available (first epoch)");
                } else {
                    tracing::warn!(
                        "RichardsGlu layer count mismatch: prev={}, curr={}",
                        prev_richards_glu_weights.len(),
                        current_richards_glu_weights.len()
                    );
                }

                // Debug: Log parameter change statistics
                if richards_glu_param_count > 0 {
                    let avg_delta = richards_glu_delta_sum / richards_glu_param_count as f64;
                    let significant_ratio =
                        significant_changes as f64 / total_weight_changes as f64;

                    tracing::debug!(
                        "RichardsGlu delta stats: {} params, avg_delta={:.2e}, significant_changes={}/{} ({:.1}%)",
                        richards_glu_param_count,
                        avg_delta,
                        significant_changes,
                        total_weight_changes,
                        significant_ratio * 100.0
                    );
                }
                let avg_richards_glu_delta = if richards_glu_param_count > 0 {
                    richards_glu_delta_sum / richards_glu_param_count as f64
                } else {
                    0.0
                };

                // Update previous weights
                prev_richards_glu_weights = current_richards_glu_weights;

                // NFR-7.3: Training metrics
                let warmup_status = if epoch < warmup_epochs {
                    format!(" (warmup {}/{})", epoch + 1, warmup_epochs)
                } else {
                    String::new()
                };

                let epoch_ms = t_epoch_start.elapsed().as_secs_f64() as f32 * 1000.0;
                let tokens_per_sec = if total_examples > 0 {
                    (total_examples as f32) / (t_epoch_start.elapsed().as_secs_f32().max(1e-6))
                } else {
                    0.0
                };
                let tau_opt = if tau_available {
                    Some((tau_min_epoch, tau_max_epoch))
                } else {
                    None
                };
                let metrics = crate::domain::attention::poly_attention::DegreeAdaptationMetrics {
                    epoch_index: epoch,
                    loss_delta: 0.0,
                    grad_norm: avg_grad_norm,
                    epoch_ms,
                    tokens_per_sec,
                    tau_range: tau_opt,
                    pred_norm_rms: if pred_norm_rms.is_finite() {
                        Some(pred_norm_rms)
                    } else {
                        None
                    },
                };
                for layer in &mut self.network {
                    if let LayerEnum::TransformerBlock(tb) = layer
                        && let crate::domain::layers::components::common::TemporalMixingLayer::Attention(
                            attn,
                        ) = tb.temporal_mixing_mut()
                    {
                        attn.adapt_degree(&metrics);
                    }
                    if let LayerEnum::DiffusionBlock(db) = layer
                        && let crate::domain::layers::components::common::TemporalMixingLayer::Attention(
                            attn,
                        ) = &mut db.temporal_mixing
                    {
                        attn.adapt_degree(&metrics);
                    }
                    if let LayerEnum::PolyAttention(pa) = layer {
                        pa.adapt_degree(&metrics);
                    }
                }

                info!(
                    epoch = epoch,
                    loss = avg_loss,
                    base_loss = avg_base_loss,
                    grad_norm = avg_grad_norm,
                    learning_rate = effective_lr,
                    tau_min = tau_min_log,
                    tau_max = tau_max_log,
                    tau_range = tau_range_log,
                    pred_norm_rms = pred_norm_rms,
                    avg_active_heads = avg_active_heads,
                    avg_active_experts = avg_active_experts,
                    avg_significant_experts = avg_significant_experts,
                    avg_routing_entropy = avg_routing_entropy,
                    richards_glu_richards_delta = avg_richards_glu_delta,
                    "Training epoch completed{}",
                    warmup_status
                );
            }

            Ok(())
        })();
        self.training_scratch = scratch;
        res
    }

    #[instrument(skip(self, data))]
    pub fn train_with_warmup_eprop(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        target_lr: f32,
        batch_size: usize,
        warmup_epochs: usize,
    ) -> Result<()> {
        self.set_trm_training_mode();

        for epoch in 0..epochs {
            let t_epoch_start = std::time::Instant::now();
            let mut total_loss = 0.0f32;
            let mut total_base_loss = 0.0f32;
            let mut total_grad_norm = 0.0f32;
            let mut batch_count = 0usize;
            let mut per_layer_param_grad_norm_sq: Vec<f32> = vec![0.0; self.network.len()];

            let effective_lr = if epoch < warmup_epochs {
                target_lr * ((epoch + 1) as f32 / warmup_epochs.max(1) as f32)
            } else {
                let t = (epoch - warmup_epochs) as f32;
                let t_max = (epochs.saturating_sub(warmup_epochs)).max(1) as f32;
                let lr_min = target_lr * 0.10;
                let lr_max = target_lr;
                lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos())
            };

            for batch_strs in data.chunks(batch_size.max(1)) {
                let batch_tokenized: Vec<Vec<usize>> = batch_strs
                    .par_iter()
                    .map(|input| self.tokenize(input))
                    .collect();

                let (batch_loss, batch_base_loss, grad_norm, layer_param_grad_norm_sq) =
                    self.train_batch_eprop_profiled(&batch_tokenized, effective_lr)?;
                total_loss += batch_loss;
                total_base_loss += batch_base_loss;
                total_grad_norm += grad_norm;
                batch_count += 1;
                for (i, s) in layer_param_grad_norm_sq.into_iter().enumerate() {
                    if i < per_layer_param_grad_norm_sq.len() {
                        per_layer_param_grad_norm_sq[i] += s;
                    }
                }
            }

            let avg_loss = total_loss / (batch_count.max(1) as f32);
            let avg_base_loss = total_base_loss / (batch_count.max(1) as f32);
            let avg_grad_norm = total_grad_norm / (batch_count.max(1) as f32);
            let per_layer_rms: Vec<f32> = per_layer_param_grad_norm_sq
                .iter()
                .map(|&s| (s / (batch_count.max(1) as f32)).sqrt())
                .collect();

            let epoch_ms = t_epoch_start.elapsed().as_millis();
            info!(
                epoch = epoch,
                loss = avg_loss,
                base_loss = avg_base_loss,
                grad_norm = avg_grad_norm,
                learning_rate = effective_lr,
                per_layer_rms = ?per_layer_rms,
                epoch_ms = epoch_ms,
                "E-prop-style training epoch completed"
            );
        }

        Ok(())
    }

    /// Train TRM layers using autoencoding (pretraining phase)
    /// During autoencoding, the model learns to reconstruct its input through recursive processing
    /// This is the first phase of TRM training before chat-tuning
    #[instrument(skip(self, data))]
    pub fn train_trm_autoencoding(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
    ) -> Result<()> {
        // Set TRM layers to training mode (full supervision steps)
        self.set_trm_training_mode();

        info!(
            "Starting TRM autoencoding pretraining: {} epochs, {} sequences",
            epochs,
            data.len()
        );

        let mut scratch = std::mem::take(&mut self.training_scratch);
        let res: Result<()> = (|| {
            for epoch in 0..epochs {
                let mut total_loss = 0.0;
                let mut total_base_loss = 0.0;
                let mut total_grad_norm = 0.0;
                let mut batch_count = 0;
                // Process data in batches
                for batch_strs in data.chunks(batch_size.max(1)) {
                    let batch_tokenized: Vec<Vec<usize>> = batch_strs
                        .par_iter()
                        .map(|input| self.tokenize(input))
                        .collect();

                    let (batch_loss, batch_base_loss, grad_norm) =
                        self.train_batch_trm_autoencoding(&batch_tokenized, lr, &mut scratch)?;
                    total_loss += batch_loss;
                    total_base_loss += batch_base_loss;
                    total_grad_norm += grad_norm;
                    batch_count += 1;
                }

                let avg_loss = total_loss / data.len() as f32;
                let avg_base_loss = total_base_loss / data.len() as f32;
                let avg_grad_norm = total_grad_norm / batch_count as f32;

                // NFR-5.2: Training divergence detection
                if avg_loss.is_nan() || avg_loss.is_infinite() {
                    return Err(ModelError::Training {
                        message: format!(
                            "TRM autoencoding diverged at epoch {}: loss is {} (NaN or Inf detected)",
                            epoch, avg_loss
                        ),
                    });
                }

                let mut tau_min_epoch = f32::INFINITY;
                let mut tau_max_epoch = f32::NEG_INFINITY;
                let mut tau_available = false;
                let mut pred_norm_sum = 0.0f32;
                let mut pred_norm_count = 0usize;
                let mut avg_heads_per_token_sum = 0.0f32;
                let mut heads_layers_count = 0usize;
                for layer in &mut self.network {
                    if let LayerEnum::LRM(lrm) = layer {
                        if let Some((min_tau, max_tau)) = lrm.attention_mut().take_tau_metrics() {
                            tau_available = true;
                            if min_tau < tau_min_epoch {
                                tau_min_epoch = min_tau;
                            }
                            if max_tau > tau_max_epoch {
                                tau_max_epoch = max_tau;
                            }
                        }
                        if let Some(rms_g) = lrm.attention_mut().take_pred_norm() {
                            pred_norm_sum += rms_g;
                            pred_norm_count += 1;
                        }
                        let per_head = lrm.attention_mut().get_head_metrics_and_reset();
                        if !per_head.is_empty() {
                            let layer_avg_active_heads =
                                per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                            avg_heads_per_token_sum += layer_avg_active_heads;
                            heads_layers_count += 1;
                        }
                    }
                }
                let tau_min_log = if tau_available {
                    tau_min_epoch
                } else {
                    f32::NAN
                };
                let tau_max_log = if tau_available {
                    tau_max_epoch
                } else {
                    f32::NAN
                };
                let tau_range_log = if tau_available {
                    tau_max_epoch - tau_min_epoch
                } else {
                    f32::NAN
                };
                let pred_norm_rms = if pred_norm_count > 0 {
                    pred_norm_sum / pred_norm_count as f32
                } else {
                    f32::NAN
                };
                let avg_active_heads = if heads_layers_count > 0 {
                    avg_heads_per_token_sum / heads_layers_count as f32
                } else {
                    f32::NAN
                };

                let mut lb_loss = f32::NAN;
                let mut cx_loss = f32::NAN;
                let mut sp_loss = f32::NAN;
                let mut rec_avg_heads = f32::NAN;
                let mut rec_tau_min = f32::NAN;
                let mut rec_tau_max = f32::NAN;
                for layer in &self.network {
                    if let LayerEnum::LRM(lrm) = layer {
                        lb_loss = lrm
                            .attention()
                            .moh
                            .head_selection_config
                            .compute_load_balance_loss();
                        cx_loss = lrm
                            .attention()
                            .moh
                            .head_selection_config
                            .compute_complexity_loss(lrm.attention().moh_num_active() as f32);
                        sp_loss = lrm
                            .attention()
                            .moh
                            .head_selection_config
                            .compute_sparsity_loss();
                        if !lrm.recursion_metrics.is_empty() {
                            let mut hsum = 0.0f32;
                            let mut c = 0usize;
                            let mut tmin = f32::INFINITY;
                            let mut tmax = f32::NEG_INFINITY;
                            for (h, mn, mx) in lrm.recursion_metrics.iter().cloned() {
                                hsum += h;
                                c += 1;
                                if mn < tmin {
                                    tmin = mn;
                                }
                                if mx > tmax {
                                    tmax = mx;
                                }
                            }
                            rec_avg_heads = if c > 0 { hsum / c as f32 } else { f32::NAN };
                            rec_tau_min = if c > 0 { tmin } else { f32::NAN };
                            rec_tau_max = if c > 0 { tmax } else { f32::NAN };
                        }
                        break;
                    }
                }

                info!(
                    epoch = epoch,
                    loss = avg_loss,
                    base_loss = avg_base_loss,
                    grad_norm = avg_grad_norm,
                    tau_min = tau_min_log,
                    tau_max = tau_max_log,
                    tau_range = tau_range_log,
                    pred_norm_rms = pred_norm_rms,
                    avg_active_heads = avg_active_heads,
                    rec_avg_heads = rec_avg_heads,
                    rec_tau_min = rec_tau_min,
                    rec_tau_max = rec_tau_max,
                    moh_lb = lb_loss,
                    moh_cx = cx_loss,
                    moh_sp = sp_loss,
                    "LRM autoencoding epoch completed"
                );

                for layer in &mut self.network {
                    if let LayerEnum::LRM(lrm) = layer {
                        let heads = lrm.attention().num_heads() as f32;
                        let h_ratio = if avg_active_heads.is_finite() && heads > 0.0 {
                            (avg_active_heads / heads).clamp(0.1, 1.0)
                        } else {
                            0.5
                        };
                        lrm.set_latent_update_alpha(0.03 + 0.05 * (1.0 - h_ratio));
                        let ent = lrm
                            .attention()
                            .moh
                            .head_selection_config
                            .gating
                            .get_gating_entropy();
                        let g = &mut lrm.attention_mut().moh.head_selection_config.gating;
                        if ent < 0.2 {
                            g.load_balance_weight = (g.load_balance_weight + 0.01).min(0.2);
                        }
                        if avg_active_heads.is_finite() {
                            if avg_active_heads > heads * 0.5 {
                                g.sparsity_weight = (g.sparsity_weight + 0.01).min(0.2);
                            } else {
                                g.sparsity_weight = (g.sparsity_weight * 0.95).max(0.0);
                            }
                        }
                        g.complexity_loss_weight = (g.complexity_loss_weight * 0.9) + 0.01;
                    }
                }
            }

            Ok(())
        })();
        self.training_scratch = scratch;
        res
    }

    /// Complete TRM training pipeline: autoencoding pretraining + chat-tuning
    /// Phase 1: Autoencoding - TRM learns to reconstruct input through recursion
    /// Phase 2: Chat-tuning - Standard next-token prediction on conversational data
    #[instrument(skip(self, pretraining_data, chat_data))]
    pub fn train_trm_complete(
        &mut self,
        pretraining_data: Vec<&str>,
        chat_data: Vec<&str>,
        autoencoding_epochs: usize,
        chat_epochs: usize,
        lr: f32,
        batch_size: usize,
    ) -> Result<()> {
        info!(
            "Starting TRM complete training: {} autoencoding epochs + {} chat-tuning epochs",
            autoencoding_epochs, chat_epochs
        );

        // Phase 1: Autoencoding pretraining
        if autoencoding_epochs > 0 {
            info!("Phase 1: TRM Autoencoding Pretraining");
            self.train_trm_autoencoding(pretraining_data, autoencoding_epochs, lr, batch_size)?;
        }

        // Phase 2: Chat-tuning (standard next-token prediction)
        if chat_epochs > 0 {
            info!("Phase 2: Chat-Tuning (next-token prediction)");
            self.train_with_warmup(chat_data, chat_epochs, lr, batch_size, 15)?;
        }

        info!("TRM training completed successfully");
        Ok(())
    }

    /// Train on a single batch using TRM autoencoding
    /// For autoencoding, the TRM layer learns to reconstruct its embedded input
    fn train_batch_trm_autoencoding(
        &mut self,
        batch: &[Vec<usize>],
        lr: f32,
        scratch: &mut TrainingScratch,
    ) -> Result<(f32, f32, f32)> {
        let mut batch_loss = 0.0;
        let mut batch_base_loss = 0.0;

        // Reset scratch buffers for the new batch, reusing allocations.
        scratch.reset(self.network.len());

        let mut embeddings_idx: Option<usize> = None;
        let mut trm_idx: Option<usize> = None;
        let mut norm_idx: Option<usize> = None;
        let mut out_proj_idx: Option<usize> = None;
        for (i, layer) in self.network.iter().enumerate() {
            match layer {
                LayerEnum::TokenEmbeddings(_) => {
                    if embeddings_idx.is_none() {
                        embeddings_idx = Some(i)
                    }
                }
                LayerEnum::LRM(_) => {
                    if trm_idx.is_none() {
                        trm_idx = Some(i)
                    }
                }
                LayerEnum::DynamicTanhNorm(_) => norm_idx = Some(i),
                LayerEnum::OutputProjection(_) => out_proj_idx = Some(i),
                _ => {}
            }
        }

        for sequence in batch {
            if sequence.len() < 2 {
                continue;
            }
            let input_ids = &sequence[..sequence.len() - 1];
            let target_ids = &sequence[1..];
            let mut ids_arr = Array2::<f32>::zeros((1, input_ids.len()));
            for (i, &token_id) in input_ids.iter().enumerate() {
                ids_arr[[0, i]] = token_id as f32;
            }

            let emb_idx = embeddings_idx.unwrap();
            let mut hidden = match &mut self.network[emb_idx] {
                LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                _ => ids_arr.clone(),
            };

            let t_idx = trm_idx.unwrap();
            let trm_input_saved = hidden.clone();
            hidden = match &mut self.network[t_idx] {
                LayerEnum::LRM(l) => l.forward(&hidden),
                _ => hidden,
            };

            if let Some(nidx) = norm_idx {
                hidden = match &mut self.network[nidx] {
                    LayerEnum::DynamicTanhNorm(n) => n.forward(&hidden),
                    _ => hidden,
                };
            }

            let logits = if let Some(opidx) = out_proj_idx {
                match &mut self.network[opidx] {
                    LayerEnum::OutputProjection(op) => op.forward(&hidden),
                    _ => hidden.clone(),
                }
            } else {
                hidden.clone()
            };

            let probs = crate::domain::soft::Softmax::new().forward_immutable(&logits.view());
            let sce_cfg = crate::domain::loss::SymmetricCEConfig::default();
            let sce = crate::domain::loss::symmetric_cross_entropy(
                &probs,
                target_ids,
                sce_cfg.alpha,
                sce_cfg.beta,
                sce_cfg.epsilon,
            );
            let loss_norm = sce / (target_ids.len().max(1) as f32);
            batch_loss += loss_norm;
            batch_base_loss += loss_norm;

            // Auxiliary: residual decorrelation (redundancy reduction) on the pre-logit hidden
            // state.
            let mut decor_grad_opt: Option<Array2<f32>> = None;
            let base_w = self.training_hparams.residual_decorrelation_weight;
            if base_w > 0.0 {
                let difficulty = if self.training_hparams.residual_decorrelation_adaptive {
                    (loss_norm / (loss_norm + 1.0)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let w = base_w * (1.0 + difficulty);
                let decor_loss = crate::domain::loss::residual_decorrelation_loss(&hidden.view());
                batch_loss += w * decor_loss;
                let decor_grad = crate::domain::loss::residual_decorrelation_gradients(&hidden.view());
                decor_grad_opt = Some(decor_grad.mapv(|x| x * w));
            }

            // Auxiliary: hard-negative residual repulsion (pooled hidden vs memory bank).
            let mut hardneg_grad_opt: Option<Array2<f32>> = None;
            let base_hn_w = self.training_hparams.residual_hardneg_weight;
            if base_hn_w > 0.0 {
                let difficulty = if self.training_hparams.residual_hardneg_adaptive {
                    (loss_norm / (loss_norm + 1.0)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let w = base_hn_w * (1.0 + difficulty);

                // Mean-pool across tokens.
                let rows = hidden.nrows().max(1);
                let cols = hidden.ncols();
                let mut anchor = vec![0.0f32; cols];
                for i in 0..rows {
                    for j in 0..cols {
                        let v = hidden[[i, j]];
                        anchor[j] += if v.is_finite() { v } else { 0.0 };
                    }
                }
                let inv = 1.0f32 / (rows as f32);
                for a in &mut anchor {
                    *a *= inv;
                }

                let (hn_loss, grad_anchor) = crate::domain::loss::hard_negative_repulsion_loss_and_grad(
                    &anchor,
                    self.residual_neg_bank.as_slice(),
                    self.training_hparams.residual_hardneg_k,
                    self.training_hparams.residual_hardneg_margin,
                    self.training_hparams.residual_hardneg_temperature,
                );
                batch_loss += w * hn_loss;

                // Distribute pooled gradient equally back to each token row.
                let mut g = Array2::<f32>::zeros(hidden.raw_dim());
                for i in 0..rows {
                    for j in 0..cols {
                        g[[i, j]] = (grad_anchor[j] * w) * inv;
                    }
                }
                hardneg_grad_opt = Some(g);

                // Update memory bank with current anchor (detached).
                self.residual_neg_bank
                    .push(anchor, self.training_hparams.residual_hardneg_bank_size);
            }

            let target_avg = match &self.network[t_idx] {
                LayerEnum::LRM(l) => l.attention().moh_num_active() as f32,
                _ => 0.0,
            };
            let moh_penalty = match &self.network[t_idx] {
                LayerEnum::LRM(l) => l.attention().compute_moh_aux_weighted_total(target_avg),
                _ => 0.0,
            };

            let moe_penalty = match &self.network[t_idx] {
                LayerEnum::LRM(lrm) => {
                    let guard = lrm.block.read().unwrap();
                    match &*guard {
                        crate::domain::layers::recurrence::lrm::RecursiveBlockVariant::Transformer(b) => {
                            if let crate::domain::layers::components::common::FeedForwardVariant::MixtureOfExperts(moe) = b.feedforward() {
                                moe.last_aux_loss()
                            } else {
                                0.0
                            }
                        }
                        crate::domain::layers::recurrence::lrm::RecursiveBlockVariant::Diffusion(b) => {
                            if let crate::domain::layers::components::common::FeedForwardVariant::MixtureOfExperts(moe) = &b.feedforward {
                                moe.last_aux_loss()
                            } else {
                                0.0
                            }
                        }
                    }
                }
                _ => 0.0,
            };

            if moh_penalty > 10.0 {
                info!("High MoH penalty in batch: {}", moh_penalty);
            }

            if moe_penalty > 10.0 {
                info!("High MoE penalty in batch: {}", moe_penalty);
            }

            batch_loss += moh_penalty;
            batch_loss += moe_penalty;

            let grads_logits = crate::domain::loss::symmetric_cross_entropy_gradients(
                &probs,
                target_ids,
                sce_cfg.alpha,
                sce_cfg.beta,
                sce_cfg.epsilon,
            );

            let (mut grad_hidden, op_param_grads) = if let Some(opidx) = out_proj_idx {
                match &mut self.network[opidx] {
                    LayerEnum::OutputProjection(op) => op.compute_gradients(&hidden, &grads_logits),
                    _ => (grads_logits.clone(), Vec::new()),
                }
            } else {
                (grads_logits.clone(), Vec::new())
            };

            if let Some(decor_grad) = decor_grad_opt {
                grad_hidden = grad_hidden + decor_grad;
            }

            if let Some(hn_grad) = hardneg_grad_opt {
                grad_hidden = grad_hidden + hn_grad;
            }
            if let Some(opidx) = out_proj_idx {
                Self::accumulate_layer_gradients(
                    &mut scratch.accumulated_param_grads[opidx],
                    op_param_grads,
                    "OutputProjection",
                );
            }

            if let Some(nidx) = norm_idx {
                grad_hidden = match &mut self.network[nidx] {
                    LayerEnum::DynamicTanhNorm(n) => n.backward(&grad_hidden, lr),
                    _ => grad_hidden,
                };
            }

            let (trm_in_grad, trm_param_grads) = match &self.network[t_idx] {
                LayerEnum::LRM(layer) => layer.compute_gradients(&trm_input_saved, &grad_hidden),
                _ => (grad_hidden.clone(), Vec::new()),
            };
            let _ = trm_in_grad;
            Self::accumulate_layer_gradients(
                &mut scratch.accumulated_param_grads[t_idx],
                trm_param_grads,
                "LRM",
            );
            scratch.layer_grad_norms[t_idx] +=
                grad_hidden.iter().map(|&x| x * x).sum::<f32>().sqrt();
        }

        let batch_scale = 1.0 / batch.len().max(1) as f32;
        for (layer_idx, param_grads) in scratch.accumulated_param_grads.iter_mut().enumerate() {
            if param_grads.is_empty() {
                continue;
            }
            for grad in param_grads.iter_mut() {
                grad.mapv_inplace(|x| x * batch_scale);
            }
            let grads_slice = param_grads.as_slice();
            self.network[layer_idx].apply_gradients(grads_slice, lr)?;
        }

        let total_grad_norm = scratch
            .layer_grad_norms
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        Ok((batch_loss, batch_base_loss, total_grad_norm))
    }

    fn accumulate_layer_gradients(
        accumulator: &mut Vec<Array2<f32>>,
        new_grads: Vec<Array2<f32>>,
        layer_name: &str,
    ) {
        if new_grads.is_empty() {
            return;
        }
        if accumulator.is_empty() {
            *accumulator = new_grads;
            return;
        }
        if accumulator.len() != new_grads.len() {
            warn!(
                layer = layer_name,
                existing = accumulator.len(),
                incoming = new_grads.len(),
                "TRM autoencoding gradient accumulation length mismatch; replacing accumulator"
            );
            *accumulator = new_grads;
            return;
        }
        for (acc, grad) in accumulator.iter_mut().zip(new_grads.into_iter()) {
            *acc += &grad;
        }
    }

    fn train_batch_eprop_profiled(
        &mut self,
        batch: &[Vec<usize>],
        lr: f32,
    ) -> Result<(f32, f32, f32, Vec<f32>)> {
        // E-Prop is enabled when TransformerBlock layers are present with eligibility traces
        // initialized
        let _eprop_enabled = self
            .network
            .iter()
            .any(|layer| matches!(layer, LayerEnum::TransformerBlock(_)));

        // Re-use the profiled training logic which is now capable of handling E-Prop gradients
        // via the updated TransformerBlock::backward / compute_gradients implementation.
        // We duplicate the logic here to allow for future E-Prop specific divergence
        // (e.g. different learning rules, eligibility trace logging, etc.) without coupling.

        let check_finite = std::env::var_os("RUSTGPT_CHECK_FINITE").is_some();
        let mut batch_loss = 0.0;
        let mut batch_base_loss = 0.0;
        let mut accumulated_param_grads: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut layer_grad_norms: Vec<f32> = Vec::new(); // Track per-layer gradient norms

        // Initialize accumulated gradients for each layer
        for _ in &self.network {
            accumulated_param_grads.push(Vec::new());
            layer_grad_norms.push(0.0);
        }

        // OutputProjection index (used to attach residual decorrelation to the pre-logit hidden
        // state).
        let mut out_proj_idx: Option<usize> = None;
        for (i, layer) in self.network.iter().enumerate() {
            if matches!(layer, LayerEnum::OutputProjection(_)) {
                out_proj_idx = Some(i);
            }
        }

        let mut layer_inputs: Vec<Array2<f32>> = Vec::with_capacity(self.network.len());

        // Process each sequence in the batch
        for training_row in batch {
            if training_row.len() < 2 {
                continue;
            }

            // 1. Slice input and targets
            let input_ids = &training_row[..training_row.len() - 1]; // Exclude the last token
            let target_ids = &training_row[1..]; // This is a vector. Each element is the index in the vocab.

            // Forward pass with signal propagation variance tracking
            let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
            for (i, &token_id) in input_ids.iter().enumerate() {
                input[[0, i]] = token_id as f32;
            }

            // Track forward pass variance for signal propagation analysis
            let mut layer_variances: Vec<f32> = Vec::new();
            layer_inputs.clear();

            let mut similarity_ctx: Option<Array2<f32>> = None;

            for layer in &mut self.network {
                layer_inputs.push(input);
                let input_ref = layer_inputs.last().unwrap();
                input = match layer {
                    LayerEnum::TransformerBlock(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(input_ref);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    LayerEnum::DiffusionBlock(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(input_ref);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    LayerEnum::LRM(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(input_ref);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    _ => {
                        similarity_ctx = None;
                        layer.forward(input_ref)
                    }
                };

                // Compute variance of layer output in single pass
                let (sum, sum_sq) = input
                    .iter()
                    .fold((0.0, 0.0), |(s, sq), &x| (s + x, sq + x * x));
                let n = input.len() as f32;
                let mean = sum / n;
                let variance = (sum_sq / n) - mean * mean;
                layer_variances.push(variance);
            }

            let logits = input;
            let probs = crate::domain::soft::Softmax::new().forward_immutable(&logits.view());

            // Symmetric cross-entropy loss and gradients
            let sce_cfg = crate::domain::loss::SymmetricCEConfig::default();
            let sce = crate::domain::loss::symmetric_cross_entropy(
                &probs,
                target_ids,
                sce_cfg.alpha,
                sce_cfg.beta,
                sce_cfg.epsilon,
            );
            let sce_norm = sce / (target_ids.len().max(1) as f32);
            batch_loss += sce_norm;
            batch_base_loss += sce_norm;

            // Auxiliary residual decorrelation (redundancy reduction)
            let decor_grad_opt: Option<(usize, Array2<f32>)> = if let Some(op_idx) = out_proj_idx {
                let base_w = self.training_hparams.residual_decorrelation_weight;
                if base_w > 0.0 {
                    let difficulty = if self.training_hparams.residual_decorrelation_adaptive {
                        (sce_norm / (sce_norm + 1.0)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let w = base_w * (1.0 + difficulty);
                    let hidden_prelogit = &layer_inputs[op_idx];
                    let dl = crate::domain::loss::residual_decorrelation_loss(&hidden_prelogit.view());
                    batch_loss += w * dl;
                    let dg = crate::domain::loss::residual_decorrelation_gradients(&hidden_prelogit.view());
                    Some((op_idx, dg.mapv(|x| x * w)))
                } else {
                    None
                }
            } else {
                None
            };

            // Auxiliary hard-negative repulsion
            let hardneg_grad_opt: Option<(usize, Array2<f32>)> = if let Some(op_idx) = out_proj_idx
            {
                let base_w = self.training_hparams.residual_hardneg_weight;
                if base_w > 0.0 {
                    let difficulty = if self.training_hparams.residual_hardneg_adaptive {
                        (sce_norm / (sce_norm + 1.0)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let w = base_w * (1.0 + difficulty);
                    let hidden_prelogit = &layer_inputs[op_idx];
                    let rows = hidden_prelogit.nrows();
                    let cols = hidden_prelogit.ncols();

                    let mut anchor_all = vec![0.0f32; cols];
                    let mut anchor_even = vec![0.0f32; cols];
                    let mut anchor_odd = vec![0.0f32; cols];
                    let mut count_even = 0usize;
                    let mut count_odd = 0usize;

                    for i in 0..rows {
                        let is_even = i % 2 == 0;
                        if is_even {
                            count_even += 1;
                        } else {
                            count_odd += 1;
                        }
                        for j in 0..cols {
                            let v = hidden_prelogit[[i, j]];
                            let v = if v.is_finite() { v } else { 0.0 };
                            anchor_all[j] += v;
                            if is_even {
                                anchor_even[j] += v;
                            } else {
                                anchor_odd[j] += v;
                            }
                        }
                    }

                    let inv_all = 1.0f32 / (rows.max(1) as f32);
                    for a in &mut anchor_all {
                        *a *= inv_all;
                    }
                    let inv_even = if count_even > 0 {
                        1.0f32 / (count_even as f32)
                    } else {
                        0.0
                    };
                    for a in &mut anchor_even {
                        *a *= inv_even;
                    }
                    let inv_odd = if count_odd > 0 {
                        1.0f32 / (count_odd as f32)
                    } else {
                        0.0
                    };
                    for a in &mut anchor_odd {
                        *a *= inv_odd;
                    }

                    let (hn_loss, grad_anchor) = crate::domain::loss::hard_negative_repulsion_loss_and_grad(
                        &anchor_all,
                        self.residual_neg_bank.as_slice(),
                        self.training_hparams.residual_hardneg_k,
                        self.training_hparams.residual_hardneg_margin,
                        self.training_hparams.residual_hardneg_temperature,
                    );
                    let (nce_loss, grad_even, grad_odd) = if count_even > 0 && count_odd > 0 {
                        crate::domain::loss::info_nce_loss_and_grads(
                            &anchor_even,
                            &anchor_odd,
                            self.residual_neg_bank.as_slice(),
                            self.training_hparams.residual_hardneg_k,
                            self.training_hparams.residual_hardneg_temperature,
                        )
                    } else {
                        (0.0, vec![0.0f32; cols], vec![0.0f32; cols])
                    };
                    batch_loss += w * (hn_loss + nce_loss);

                    let mut g = Array2::<f32>::zeros(hidden_prelogit.raw_dim());
                    for i in 0..rows {
                        let is_even = i % 2 == 0;
                        for j in 0..cols {
                            let mut v = grad_anchor[j] * inv_all;
                            if count_even > 0 && count_odd > 0 {
                                if is_even {
                                    v += grad_even[j] * inv_even;
                                } else {
                                    v += grad_odd[j] * inv_odd;
                                }
                            }
                            g[[i, j]] = v * w;
                        }
                    }

                    // Update memory bank.
                    self.residual_neg_bank
                        .push(anchor_all, self.training_hparams.residual_hardneg_bank_size);

                    Some((op_idx, g))
                } else {
                    None
                }
            } else {
                None
            };

            // Compute gradients w.r.t. logits
            let mut grads_output = crate::domain::loss::symmetric_cross_entropy_gradients(
                &probs,
                target_ids,
                sce_cfg.alpha,
                sce_cfg.beta,
                sce_cfg.epsilon,
            );

            // Handle LRM supervision if present
            let mut lrm_index: Option<usize> = None;
            for (i, layer) in self.network.iter().enumerate() {
                if let LayerEnum::LRM(_) = layer {
                    lrm_index = Some(i);
                    break;
                }
            }
            if let Some(t_idx) = lrm_index {
                let aux_steps: &[Array2<f32>] = match &self.network[t_idx] {
                    LayerEnum::LRM(lrm) => lrm.get_supervision_outputs(),
                    _ => &[],
                };
                let mut aux_loss_sum = 0.0f32;
                if !aux_steps.is_empty() {
                    let mut rn_idx: Option<usize> = None;
                    let mut op_idx: Option<usize> = None;
                    for i in (t_idx + 1)..self.network.len() {
                        if matches!(self.network[i], LayerEnum::DynamicTanhNorm(_)) {
                            rn_idx = Some(i);
                            break;
                        }
                    }
                    if let Some(rn_i) = rn_idx {
                        for i in (rn_i + 1)..self.network.len() {
                            if matches!(self.network[i], LayerEnum::OutputProjection(_)) {
                                op_idx = Some(i);
                                break;
                            }
                        }
                    }
                    let (rn_idx, op_idx) = match (rn_idx, op_idx) {
                        (Some(rn), Some(op)) => (rn, op),
                        _ => {
                            batch_loss += aux_loss_sum;
                            continue;
                        }
                    };
                    let mut rn_clone = match &self.network[rn_idx] {
                        LayerEnum::DynamicTanhNorm(n) => n.clone(),
                        _ => {
                            batch_loss += aux_loss_sum;
                            continue;
                        }
                    };
                    let mut op_clone = match &self.network[op_idx] {
                        LayerEnum::OutputProjection(op) => op.clone(),
                        _ => {
                            batch_loss += aux_loss_sum;
                            continue;
                        }
                    };
                    let steps_total = aux_steps.len();
                    let aux_base: f32 = 1.0;
                    let decay_rate: f32 = 0.6;
                    for (si, y_t) in aux_steps.iter().enumerate() {
                        let norm_y = rn_clone.forward(y_t);
                        let logits_t = op_clone.forward(&norm_y);
                        let probs_t =
                            crate::domain::soft::Softmax::new().forward_immutable(&logits_t.view());
                        let sce_t = crate::domain::loss::symmetric_cross_entropy(
                            &probs_t,
                            target_ids,
                            sce_cfg.alpha,
                            sce_cfg.beta,
                            sce_cfg.epsilon,
                        );
                        let sce_t_norm = sce_t / (target_ids.len().max(1) as f32);
                        let pos_from_end = (steps_total.saturating_sub(1)).saturating_sub(si);
                        let step_weight = aux_base * decay_rate.powf(pos_from_end as f32);
                        if step_weight < 1e-5 {
                            continue;
                        }
                        aux_loss_sum += sce_t_norm * step_weight;
                        let mut grad_logits_t = crate::domain::loss::symmetric_cross_entropy_gradients(
                            &probs_t,
                            target_ids,
                            sce_cfg.alpha,
                            sce_cfg.beta,
                            sce_cfg.epsilon,
                        );
                        grad_logits_t.mapv_inplace(|v| v * step_weight);
                        let (grad_norm_in, _) =
                            op_clone.compute_gradients(&norm_y, &grad_logits_t);
                        let (grad_y_in, _) = rn_clone.compute_gradients(y_t, &grad_norm_in);

                        let lrm_param_grads_step = match &self.network[t_idx] {
                            LayerEnum::LRM(layer) => {
                                let (_in_grad_unused, param_grads) =
                                    layer.compute_gradients_at_step(si, &grad_y_in);
                                param_grads
                            }
                            _ => Vec::new(),
                        };
                        if !lrm_param_grads_step.is_empty() {
                            if accumulated_param_grads[t_idx].is_empty() {
                                accumulated_param_grads[t_idx] = lrm_param_grads_step;
                            } else {
                                for (acc_grad, new_grad) in accumulated_param_grads[t_idx]
                                    .iter_mut()
                                    .zip(lrm_param_grads_step)
                                {
                                    *acc_grad += &new_grad;
                                }
                            }
                        }
                    }
                }
                batch_loss += aux_loss_sum;
            }

            // Backward pass: compute parameter gradients for each layer
            // TransformerBlock::compute_gradients() will return E-Prop gradients if enabled.
            for (rev_idx, layer) in self.network.iter().rev().enumerate() {
                let layer_idx = self.network.len() - 1 - rev_idx;
                let (input_grads, param_grads) =
                    layer.compute_gradients(&layer_inputs[layer_idx], &grads_output);

                if check_finite {
                    if let Some((bad_i, bad_v)) =
                        input_grads.iter().enumerate().find(|(_, v)| !v.is_finite())
                    {
                        return Err(crate::common::errors::ModelError::Training {
                            message: format!(
                                "Non-finite input_grads at layer {} index {}: {}",
                                layer_idx, bad_i, bad_v
                            ),
                        });
                    }
                    for (g_idx, g) in param_grads.iter().enumerate() {
                        if let Some((bad_i, bad_v)) =
                            g.iter().enumerate().find(|(_, v)| !v.is_finite())
                        {
                            return Err(crate::common::errors::ModelError::Training {
                                message: format!(
                                    "Non-finite param_grads[{}] at layer {} index {}: {}",
                                    g_idx, layer_idx, bad_i, bad_v
                                ),
                            });
                        }
                    }
                }

                let layer_grad_norm: f32 = input_grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
                layer_grad_norms[layer_idx] += layer_grad_norm;
                grads_output = input_grads;

                if let Some((op_idx, ref decor_grad)) = decor_grad_opt
                    && layer_idx == op_idx
                {
                    grads_output = &grads_output + decor_grad;
                }

                if let Some((op_idx, ref hn_grad)) = hardneg_grad_opt
                    && layer_idx == op_idx
                {
                    grads_output = &grads_output + hn_grad;
                }

                if accumulated_param_grads[layer_idx].is_empty() {
                    accumulated_param_grads[layer_idx] = param_grads;
                } else {
                    for (acc_grad, new_grad) in accumulated_param_grads[layer_idx]
                        .iter_mut()
                        .zip(param_grads)
                    {
                        *acc_grad += &new_grad;
                    }
                }
            }
        }

        // Average layer-wise gradient norms
        for norm in &mut layer_grad_norms {
            *norm /= batch.len() as f32;
        }

        let max_layer_grad = layer_grad_norms.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_layer_grad > 10.0 {
            tracing::warn!(
                "Layer-wise gradient norms: {:?}",
                layer_grad_norms
                    .iter()
                    .enumerate()
                    .map(|(i, &norm)| format!(
                        "L{}({}): {:.2}",
                        i,
                        self.network[i].layer_type(),
                        norm
                    ))
                    .collect::<Vec<_>>()
            );
        }

        // Prepare averaged gradients and detect anomalies
        let mut averaged_grads_per_layer: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut total_grad_norm_sq = 0.0f32;
        let mut layer_param_grad_norm_sq: Vec<f32> = vec![0.0; self.network.len()];

        for (layer_idx, param_grads) in accumulated_param_grads.into_iter().enumerate() {
            if !param_grads.is_empty() {
                let averaged_grads: Vec<Array2<f32>> = param_grads
                    .into_iter()
                    .map(|grad| grad / batch.len() as f32)
                    .collect();

                let max_reasonable_grad_per_param = 5.0;
                let max_total_grad_norm =
                    (averaged_grads.iter().map(|g| g.len()).sum::<usize>() as f32).sqrt()
                        * max_reasonable_grad_per_param;
                let mut total_layer_grad_norm_sq = 0.0;
                for grad in &averaged_grads {
                    total_layer_grad_norm_sq += grad.iter().map(|&x| x * x).sum::<f32>();
                }
                let total_layer_grad_norm = total_layer_grad_norm_sq.sqrt();
                let scale = if total_layer_grad_norm > max_total_grad_norm {
                    max_total_grad_norm / total_layer_grad_norm
                } else {
                    1.0
                };

                let mut clipped_grads: Vec<Array2<f32>> = if scale < 1.0 {
                    averaged_grads
                        .into_iter()
                        .map(|grad| grad.mapv(|x| x * scale))
                        .collect()
                } else {
                    averaged_grads
                };

                const MAX_GRAD_ABS: f32 = 5000.0;
                let mut max_abs: f32 = 0.0;
                for g in &clipped_grads {
                    for &v in g.iter() {
                        if v.abs() > max_abs {
                            max_abs = v.abs();
                        }
                    }
                }
                if max_abs > MAX_GRAD_ABS {
                    let s = MAX_GRAD_ABS / max_abs;
                    for g in &mut clipped_grads {
                        g.mapv_inplace(|v| v * s);
                    }
                }

                if check_finite {
                    for (g_idx, g) in clipped_grads.iter().enumerate() {
                        if let Some((bad_i, bad_v)) =
                            g.iter().enumerate().find(|(_, v)| !v.is_finite())
                        {
                            return Err(crate::common::errors::ModelError::Training {
                                message: format!(
                                    "Non-finite clipped_grads[{}] at layer {} index {}: {}",
                                    g_idx, layer_idx, bad_i, bad_v
                                ),
                            });
                        }
                    }
                } else {
                    for grad in &mut clipped_grads {
                        grad.iter_mut().for_each(|v| {
                            if !v.is_finite() {
                                *v = 0.0
                            }
                        });
                    }
                }

                if let Err(e) = Self::detect_gradient_anomalies(&clipped_grads) {
                    tracing::error!("Gradient anomaly detected in layer {}", layer_idx);
                    return Err(e);
                }

                let mut s_layer = 0.0f32;
                for grad in &clipped_grads {
                    let s = grad.iter().map(|&x| x * x).sum::<f32>();
                    total_grad_norm_sq += s;
                    s_layer += s;
                }
                layer_param_grad_norm_sq[layer_idx] += s_layer;
                averaged_grads_per_layer.push(clipped_grads);
            } else {
                averaged_grads_per_layer.push(Vec::new());
            }
        }

        let grad_norm = total_grad_norm_sq.sqrt();
        let per_layer_grad_norms: Vec<f32> = self
            .network
            .iter()
            .zip(&averaged_grads_per_layer)
            .map(|(_layer, grads)| {
                if grads.is_empty() {
                    0.0
                } else {
                    let mut s = 0.0f32;
                    for g in grads {
                        s += g.iter().map(|&x| x * x).sum::<f32>();
                    }
                    s.sqrt()
                }
            })
            .collect();

        let mut nonzero: Vec<f32> = per_layer_grad_norms
            .iter()
            .cloned()
            .filter(|&v| v > 0.0)
            .collect();
        let median_grad_norm = if nonzero.is_empty() {
            grad_norm.max(1e-6)
        } else {
            nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = nonzero.len() / 2;
            if nonzero.len() % 2 == 0 {
                (nonzero[mid - 1] + nonzero[mid]) * 0.5
            } else {
                nonzero[mid]
            }
        };

        const EMA_BETA: f32 = 0.9;
        let _median_smoothed = if let Some(prev) = self.median_grad_ema {
            let sm = EMA_BETA * prev + (1.0 - EMA_BETA) * median_grad_norm;
            self.median_grad_ema = Some(sm);
            sm
        } else {
            self.median_grad_ema = Some(median_grad_norm);
            median_grad_norm
        };

        // Compute adaptive learning rates
        let adaptive_lrs: Vec<f32> = self
            .network
            .iter()
            .zip(&averaged_grads_per_layer)
            .enumerate()
            .map(|(layer_idx, (layer, grads))| {
                if grads.is_empty() {
                    lr
                } else {
                    Self::compute_layer_adaptive_lr_static(
                        layer,
                        grads,
                        lr,
                        layer_idx,
                        median_grad_norm,
                    )
                }
            })
            .collect();

        // Apply gradients (this will now route E-Prop gradients via ParamPartitions in
        // apply_gradients)
        for ((layer, grads), adaptive_lr) in self
            .network
            .iter_mut()
            .zip(averaged_grads_per_layer)
            .zip(adaptive_lrs)
        {
            if !grads.is_empty() {
                layer.apply_gradients(&grads, adaptive_lr)?;
            }
        }

        Ok((
            batch_loss,
            batch_base_loss,
            grad_norm,
            layer_param_grad_norm_sq,
        ))
    }

    /// Train on a single batch of sequences
    /// Returns (batch_loss, batch_base_loss, gradient_norm, layer_grad_norms)
    fn train_batch_profiled(
        &mut self,
        batch: &[Vec<usize>],
        lr: f32,
        scratch: &mut TrainingScratch,
    ) -> Result<(f32, f32, f32, Vec<f32>)> {
        let check_finite = std::env::var_os("RUSTGPT_CHECK_FINITE").is_some();
        let mut batch_loss = 0.0;
        let mut batch_base_loss = 0.0;

        // Reset scratch buffers for the new batch, reusing allocations.
        scratch.reset(self.network.len());

        // OutputProjection index (used to attach residual decorrelation to the pre-logit hidden
        // state).
        let mut out_proj_idx: Option<usize> = None;
        for (i, layer) in self.network.iter().enumerate() {
            if matches!(layer, LayerEnum::OutputProjection(_)) {
                out_proj_idx = Some(i);
            }
        }

        // Process each sequence in the batch
        for training_row in batch {
            if training_row.len() < 2 {
                continue;
            }

            // 1. Slice input and targets
            let input_ids = &training_row[..training_row.len() - 1]; // Exclude the last token
            let target_ids = &training_row[1..]; // This is a vector. Each element is the index in the vocab.

            // Forward pass with signal propagation variance tracking
            let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
            for (i, &token_id) in input_ids.iter().enumerate() {
                input[[0, i]] = token_id as f32;
            }

            // Track forward pass variance for signal propagation analysis
            // Reference: "Deep Information Propagation" (Schoenholz et al., 2017)
            // Ideal: Var(x_l) ≈ Var(x_0) for all layers (isometry condition)
            let mut layer_variances: Vec<f32> = Vec::new();
            scratch.layer_inputs.clear();

            let mut similarity_ctx: Option<Array2<f32>> = None;

            for layer in &mut self.network {
                scratch.layer_inputs.push(input);
                let input_ref = scratch.layer_inputs.last().unwrap();
                input = match layer {
                    LayerEnum::TransformerBlock(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(input_ref);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    LayerEnum::DiffusionBlock(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(input_ref);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    LayerEnum::LRM(block) => {
                        block.set_incoming_similarity_context(similarity_ctx.as_ref());
                        let out = block.forward(input_ref);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(block.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(block.activation_similarity_matrix().clone());
                        }
                        out
                    }
                    _ => {
                        similarity_ctx = None;
                        layer.forward(input_ref)
                    }
                };

                // Compute variance of layer output in single pass
                let (sum, sum_sq) = input
                    .iter()
                    .fold((0.0, 0.0), |(s, sq), &x| (s + x, sq + x * x));
                let n = input.len() as f32;
                let mean = sum / n;
                let variance = (sum_sq / n) - mean * mean;
                layer_variances.push(variance);
            }

            let logits = input;
            let probs = crate::domain::soft::Softmax::new().forward_immutable(&logits.view());

            // Symmetric cross-entropy loss and gradients
            let sce_cfg = crate::domain::loss::SymmetricCEConfig::default();
            let sce = crate::domain::loss::symmetric_cross_entropy(
                &probs,
                target_ids,
                sce_cfg.alpha,
                sce_cfg.beta,
                sce_cfg.epsilon,
            );
            let sce_norm = sce / (target_ids.len().max(1) as f32);
            batch_loss += sce_norm;
            batch_base_loss += sce_norm;

            // Auxiliary residual decorrelation (redundancy reduction) on the pre-logit hidden
            // state.
            let decor_grad_opt: Option<(usize, Array2<f32>)> = if let Some(op_idx) = out_proj_idx {
                let base_w = self.training_hparams.residual_decorrelation_weight;
                if base_w > 0.0 {
                    let difficulty = if self.training_hparams.residual_decorrelation_adaptive {
                        (sce_norm / (sce_norm + 1.0)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let w = base_w * (1.0 + difficulty);
                    let hidden_prelogit = &scratch.layer_inputs[op_idx];
                    let dl = crate::domain::loss::residual_decorrelation_loss(&hidden_prelogit.view());
                    batch_loss += w * dl;
                    let dg = crate::domain::loss::residual_decorrelation_gradients(&hidden_prelogit.view());
                    Some((op_idx, dg.mapv(|x| x * w)))
                } else {
                    None
                }
            } else {
                None
            };

            // Auxiliary hard-negative repulsion on pooled pre-logit hidden state.
            let hardneg_grad_opt: Option<(usize, Array2<f32>)> = if let Some(op_idx) = out_proj_idx
            {
                let base_w = self.training_hparams.residual_hardneg_weight;
                if base_w > 0.0 {
                    let difficulty = if self.training_hparams.residual_hardneg_adaptive {
                        (sce_norm / (sce_norm + 1.0)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let w = base_w * (1.0 + difficulty);
                    let hidden_prelogit = &scratch.layer_inputs[op_idx];
                    let rows = hidden_prelogit.nrows().max(1);
                    let cols = hidden_prelogit.ncols();

                    // Mean-pool.
                    let mut anchor = vec![0.0f32; cols];
                    for i in 0..rows {
                        for j in 0..cols {
                            let v = hidden_prelogit[[i, j]];
                            anchor[j] += if v.is_finite() { v } else { 0.0 };
                        }
                    }
                    let inv = 1.0f32 / (rows as f32);
                    for a in &mut anchor {
                        *a *= inv;
                    }

                    let (hn_loss, grad_anchor) = crate::domain::loss::hard_negative_repulsion_loss_and_grad(
                        &anchor,
                        self.residual_neg_bank.as_slice(),
                        self.training_hparams.residual_hardneg_k,
                        self.training_hparams.residual_hardneg_margin,
                        self.training_hparams.residual_hardneg_temperature,
                    );
                    batch_loss += w * hn_loss;

                    // Spread pooled grad across tokens.
                    let mut g = Array2::<f32>::zeros(hidden_prelogit.raw_dim());
                    for i in 0..rows {
                        for j in 0..cols {
                            g[[i, j]] = (grad_anchor[j] * w) * inv;
                        }
                    }

                    // Update memory bank.
                    self.residual_neg_bank
                        .push(anchor, self.training_hparams.residual_hardneg_bank_size);

                    Some((op_idx, g))
                } else {
                    None
                }
            } else {
                None
            };

            // Compute gradients w.r.t. logits
            let mut grads_output = crate::domain::loss::symmetric_cross_entropy_gradients(
                &probs,
                target_ids,
                sce_cfg.alpha,
                sce_cfg.beta,
                sce_cfg.epsilon,
            );

            let mut lrm_index: Option<usize> = None;
            for (i, layer) in self.network.iter().enumerate() {
                if let LayerEnum::LRM(_) = layer {
                    lrm_index = Some(i);
                    break;
                }
            }
            if let Some(t_idx) = lrm_index {
                let aux_steps: &[Array2<f32>] = match &self.network[t_idx] {
                    LayerEnum::LRM(lrm) => lrm.get_supervision_outputs(),
                    _ => &[],
                };
                let mut aux_loss_sum = 0.0f32;
                if !aux_steps.is_empty() {
                    // IMPORTANT: Do NOT call forward() on real layers here.
                    // OutputProjection/DynamicTanhNorm rely on internal cached_input for gradients;
                    // calling forward() for aux supervision would overwrite caches and corrupt the
                    // main backward pass.

                    // Find the normalization layer after the LRM and the output projection layer.
                    let mut rn_idx: Option<usize> = None;
                    let mut op_idx: Option<usize> = None;
                    for i in (t_idx + 1)..self.network.len() {
                        if matches!(self.network[i], LayerEnum::DynamicTanhNorm(_)) {
                            rn_idx = Some(i);
                            break;
                        }
                    }
                    if let Some(rn_i) = rn_idx {
                        for i in (rn_i + 1)..self.network.len() {
                            if matches!(self.network[i], LayerEnum::OutputProjection(_)) {
                                op_idx = Some(i);
                                break;
                            }
                        }
                    }

                    let (rn_idx, op_idx) = match (rn_idx, op_idx) {
                        (Some(rn), Some(op)) => (rn, op),
                        _ => {
                            tracing::warn!(
                                "TRM supervision skipped: could not find Norm/OutputProjection after LRM"
                            );
                            // Still add the aux loss (0.0) and proceed with main backward.
                            batch_loss += aux_loss_sum;
                            continue;
                        }
                    };

                    // Clone layers to keep aux supervision cache-isolated.
                    let mut rn_clone = match &self.network[rn_idx] {
                        LayerEnum::DynamicTanhNorm(n) => n.clone(),
                        _ => {
                            batch_loss += aux_loss_sum;
                            continue;
                        }
                    };
                    let mut op_clone = match &self.network[op_idx] {
                        LayerEnum::OutputProjection(op) => op.clone(),
                        _ => {
                            batch_loss += aux_loss_sum;
                            continue;
                        }
                    };

                    let steps_total = aux_steps.len();
                    let aux_base: f32 = 1.0;
                    let decay_rate: f32 = 0.6; // decay towards earlier steps
                    for (si, y_t) in aux_steps.iter().enumerate() {
                        let norm_y = rn_clone.forward(y_t);
                        let logits_t = op_clone.forward(&norm_y);
                        let probs_t =
                            crate::domain::soft::Softmax::new().forward_immutable(&logits_t.view());
                        let sce_t = crate::domain::loss::symmetric_cross_entropy(
                            &probs_t,
                            target_ids,
                            sce_cfg.alpha,
                            sce_cfg.beta,
                            sce_cfg.epsilon,
                        );
                        let sce_t_norm = sce_t / (target_ids.len().max(1) as f32);
                        let pos_from_end = (steps_total.saturating_sub(1)).saturating_sub(si);
                        let step_weight = aux_base * decay_rate.powf(pos_from_end as f32);

                        if step_weight < 1e-5 {
                            continue;
                        }

                        aux_loss_sum += sce_t_norm * step_weight;
                        let mut grad_logits_t = crate::domain::loss::symmetric_cross_entropy_gradients(
                            &probs_t,
                            target_ids,
                            sce_cfg.alpha,
                            sce_cfg.beta,
                            sce_cfg.epsilon,
                        );
                        grad_logits_t.mapv_inplace(|v| v * step_weight);
                        let (grad_norm_in, _) =
                            op_clone.compute_gradients(&norm_y, &grad_logits_t);
                        let (grad_y_in, _) = rn_clone.compute_gradients(y_t, &grad_norm_in);

                        let lrm_param_grads_step = match &self.network[t_idx] {
                            LayerEnum::LRM(layer) => {
                                let (_in_grad_unused, param_grads) =
                                    layer.compute_gradients_at_step(si, &grad_y_in);
                                param_grads
                            }
                            _ => Vec::new(),
                        };
                        if !lrm_param_grads_step.is_empty() {
                            if scratch.accumulated_param_grads[t_idx].is_empty() {
                                scratch.accumulated_param_grads[t_idx] = lrm_param_grads_step;
                            } else {
                                for (acc_grad, new_grad) in scratch.accumulated_param_grads[t_idx]
                                    .iter_mut()
                                    .zip(lrm_param_grads_step)
                                {
                                    *acc_grad += &new_grad;
                                }
                            }
                        }
                    }
                }

                if aux_loss_sum > 10.0 {
                    tracing::info!("TRM Supervision Loss: {}", aux_loss_sum);
                }

                let target_avg = match &self.network[t_idx] {
                    LayerEnum::LRM(l) => l.attention().moh_num_active() as f32,
                    _ => 0.0,
                };
                let moh_penalty = match &self.network[t_idx] {
                    LayerEnum::LRM(l) => l.attention().compute_moh_aux_weighted_total(target_avg),
                    _ => 0.0,
                };
                let moe_penalty = match &self.network[t_idx] {
                    LayerEnum::LRM(lrm) => {
                        let guard = lrm.block.read().unwrap();
                        match &*guard {
                            crate::domain::layers::recurrence::lrm::RecursiveBlockVariant::Transformer(b) => {
                                if let crate::domain::layers::components::common::FeedForwardVariant::MixtureOfExperts(moe) = b.feedforward() {
                                    moe.last_aux_loss()
                                } else {
                                    0.0
                                }
                            }
                            crate::domain::layers::recurrence::lrm::RecursiveBlockVariant::Diffusion(b) => {
                                if let crate::domain::layers::components::common::FeedForwardVariant::MixtureOfExperts(moe) = &b.feedforward {
                                    moe.last_aux_loss()
                                } else {
                                    0.0
                                }
                            }
                        }
                    }
                    _ => 0.0,
                };
                if moh_penalty > 0.01 {
                    tracing::info!("MoH Penalty (not in loss): {}", moh_penalty);
                }
                if moe_penalty > 0.01 {
                    tracing::info!("MoE Penalty (not in loss): {}", moe_penalty);
                }

                batch_loss += aux_loss_sum;
            }

            // Backward pass: compute parameter gradients for each layer
            // Note: AttentionMoE layers use backward() directly and are handled separately
            for (rev_idx, layer) in self.network.iter().rev().enumerate() {
                let layer_idx = self.network.len() - 1 - rev_idx;

                let (input_grads, param_grads) =
                    layer.compute_gradients(&scratch.layer_inputs[layer_idx], &grads_output);

                if check_finite {
                    if let Some((bad_i, bad_v)) =
                        input_grads.iter().enumerate().find(|(_, v)| !v.is_finite())
                    {
                        return Err(crate::common::errors::ModelError::Training {
                            message: format!(
                                "Non-finite input_grads at layer {} ({}) index {}: {}",
                                layer_idx,
                                layer.layer_type(),
                                bad_i,
                                bad_v
                            ),
                        });
                    }

                    for (g_idx, g) in param_grads.iter().enumerate() {
                        if let Some((bad_i, bad_v)) =
                            g.iter().enumerate().find(|(_, v)| !v.is_finite())
                        {
                            return Err(crate::common::errors::ModelError::Training {
                                message: format!(
                                    "Non-finite param_grads[{}] at layer {} ({}) index {}: {}",
                                    g_idx,
                                    layer_idx,
                                    layer.layer_type(),
                                    bad_i,
                                    bad_v
                                ),
                            });
                        }
                    }
                }

                let layer_grad_norm: f32 = input_grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
                scratch.layer_grad_norms[layer_idx] += layer_grad_norm;

                grads_output = input_grads;

                if let Some((op_idx, ref decor_grad)) = decor_grad_opt
                    && layer_idx == op_idx
                {
                    // grads_output is now dL/d(hidden_prelogit).
                    grads_output = grads_output + decor_grad.clone();
                }

                if let Some((op_idx, ref hn_grad)) = hardneg_grad_opt
                    && layer_idx == op_idx
                {
                    grads_output = grads_output + hn_grad.clone();
                }

                if scratch.accumulated_param_grads[layer_idx].is_empty() {
                    scratch.accumulated_param_grads[layer_idx] = param_grads;
                } else {
                    for (acc_grad, new_grad) in scratch.accumulated_param_grads[layer_idx]
                        .iter_mut()
                        .zip(param_grads)
                    {
                        *acc_grad += &new_grad;
                    }
                }
            }
        }

        // Average layer-wise gradient norms
        for norm in &mut scratch.layer_grad_norms {
            *norm /= batch.len() as f32;
        }

        // Log layer-wise gradient norms for debugging (only if any exceed threshold)
        let max_layer_grad = scratch
            .layer_grad_norms
            .iter()
            .fold(0.0f32, |a, &b| a.max(b));
        if max_layer_grad > 10.0 {
            tracing::warn!(
                "Layer-wise gradient norms: {:?}",
                scratch
                    .layer_grad_norms
                    .iter()
                    .enumerate()
                    .map(|(i, &norm)| format!(
                        "L{}({}): {:.2}",
                        i,
                        self.network[i].layer_type(),
                        norm
                    ))
                    .collect::<Vec<_>>()
            );
        }

        // PolyAttention-only: no auxiliary routing losses

        // Prepare averaged gradients and detect anomalies
        let mut averaged_grads_per_layer: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut total_grad_norm_sq = 0.0f32;
        let mut layer_param_grad_norm_sq: Vec<f32> = vec![0.0; self.network.len()];

        for (layer_idx, param_grads) in scratch.accumulated_param_grads.iter_mut().enumerate() {
            if !param_grads.is_empty() {
                let averaged_grads: Vec<Array2<f32>> = param_grads
                    .iter()
                    .map(|grad| grad / batch.len() as f32)
                    .collect();

                // Apply mathematically justified gradient clipping based on attention mechanism
                // properties For attention mechanisms, gradients should be bounded
                // by softmax properties and attention score ranges Maximum gradient
                // norm = sqrt(n_params) * max_reasonable_gradient_per_param
                // where max_reasonable_gradient_per_param ≈ 10.0 (based on clamped attention scores
                // [-10, 10])
                let max_reasonable_grad_per_param = 5.0;
                let max_total_grad_norm =
                    (averaged_grads.iter().map(|g| g.len()).sum::<usize>() as f32).sqrt()
                        * max_reasonable_grad_per_param;
                let mut total_layer_grad_norm_sq = 0.0;

                // First pass: compute total gradient norm for this layer
                for grad in &averaged_grads {
                    total_layer_grad_norm_sq += grad.iter().map(|&x| x * x).sum::<f32>();
                }
                let total_layer_grad_norm = total_layer_grad_norm_sq.sqrt();

                // Second pass: clip if needed using mathematically justified threshold
                let scale = if total_layer_grad_norm > max_total_grad_norm {
                    max_total_grad_norm / total_layer_grad_norm
                } else {
                    1.0
                };

                let mut clipped_grads: Vec<Array2<f32>> = if scale < 1.0 {
                    averaged_grads
                        .into_iter()
                        .map(|grad| grad.mapv(|x| x * scale))
                        .collect()
                } else {
                    averaged_grads
                };

                // Max-magnitude safety scaling
                const MAX_GRAD_ABS: f32 = 5000.0;
                let mut max_abs: f32 = 0.0;
                for g in &clipped_grads {
                    for &v in g.iter() {
                        if v.abs() > max_abs {
                            max_abs = v.abs();
                        }
                    }
                }
                if max_abs > MAX_GRAD_ABS {
                    let s = MAX_GRAD_ABS / max_abs;
                    for g in &mut clipped_grads {
                        g.mapv_inplace(|v| v * s);
                    }
                    tracing::warn!(
                        layer_idx = layer_idx,
                        max_abs,
                        scale = s,
                        "Applied max-abs gradient scaling"
                    );
                }

                if check_finite {
                    for (g_idx, g) in clipped_grads.iter().enumerate() {
                        if let Some((bad_i, bad_v)) =
                            g.iter().enumerate().find(|(_, v)| !v.is_finite())
                        {
                            return Err(crate::common::errors::ModelError::Training {
                                message: format!(
                                    "Non-finite clipped_grads[{}] at layer {} ({}) index {}: {}",
                                    g_idx,
                                    layer_idx,
                                    self.network[layer_idx].layer_type(),
                                    bad_i,
                                    bad_v
                                ),
                            });
                        }
                    }
                } else {
                    // Sanitize non-finite gradients proactively
                    for grad in &mut clipped_grads {
                        grad.iter_mut().for_each(|v| {
                            if !v.is_finite() {
                                *v = 0.0
                            }
                        });
                    }
                }

                // Detect gradient anomalies (poisoning/training instability)
                if let Err(e) = Self::detect_gradient_anomalies(&clipped_grads) {
                    tracing::error!(
                        layer_idx = layer_idx,
                        layer_type = self.network[layer_idx].layer_type(),
                        "Gradient anomaly detected in layer"
                    );
                    return Err(e);
                }

                // Compute L2 norm of gradients for this layer (after clipping)
                let mut s_layer = 0.0f32;
                for grad in &clipped_grads {
                    let s = grad.iter().map(|&x| x * x).sum::<f32>();
                    total_grad_norm_sq += s;
                    s_layer += s;
                }
                layer_param_grad_norm_sq[layer_idx] += s_layer;

                averaged_grads_per_layer.push(clipped_grads);
            } else {
                averaged_grads_per_layer.push(Vec::new());
            }
        }

        // Compute global gradient norm (L2 norm across all parameters)
        let grad_norm = total_grad_norm_sq.sqrt();

        // Compute per-layer gradient norms (post-clipping)
        let per_layer_grad_norms: Vec<f32> = self
            .network
            .iter()
            .zip(&averaged_grads_per_layer)
            .map(|(_layer, grads)| {
                if grads.is_empty() {
                    0.0
                } else {
                    let mut s = 0.0f32;
                    for g in grads {
                        s += g.iter().map(|&x| x * x).sum::<f32>();
                    }
                    s.sqrt()
                }
            })
            .collect();

        // Median of non-zero per-layer gradient norms as bidirectional target
        let mut nonzero: Vec<f32> = per_layer_grad_norms
            .iter()
            .cloned()
            .filter(|&v| v > 0.0)
            .collect();
        let median_grad_norm = if nonzero.is_empty() {
            grad_norm.max(1e-6)
        } else {
            nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = nonzero.len() / 2;
            if nonzero.len() % 2 == 0 {
                (nonzero[mid - 1] + nonzero[mid]) * 0.5
            } else {
                nonzero[mid]
            }
        };

        // EMA-smooth the median to reduce step-to-step volatility
        const EMA_BETA: f32 = 0.9; // 90% memory, gentle smoothing
        let _median_smoothed = if let Some(prev) = self.median_grad_ema {
            let sm = EMA_BETA * prev + (1.0 - EMA_BETA) * median_grad_norm;
            self.median_grad_ema = Some(sm);
            sm
        } else {
            self.median_grad_ema = Some(median_grad_norm);
            median_grad_norm
        };

        // Apply accumulated and averaged gradients with layer-wise adaptive learning rates
        // Reference: "LARS: Layer-wise Adaptive Rate Scaling" (You et al., 2017)
        // Formula: lr_layer = lr_base * trust_coef * ||W|| / (||∇W|| + weight_decay * ||W|| + ε)
        // This balances gradient flow across layers of different depths

        // Compute adaptive learning rates for all layers first (to avoid borrow checker issues)
        let adaptive_lrs: Vec<f32> = self
            .network
            .iter()
            .zip(&averaged_grads_per_layer)
            .enumerate()
            .map(|(layer_idx, (layer, grads))| {
                if grads.is_empty() {
                    lr
                } else {
                    Self::compute_layer_adaptive_lr_static(
                        layer,
                        grads,
                        lr,
                        layer_idx,
                        median_grad_norm,
                    )
                }
            })
            .collect();

        // Apply gradients with computed adaptive learning rates
        for ((layer, grads), adaptive_lr) in self
            .network
            .iter_mut()
            .zip(averaged_grads_per_layer)
            .zip(adaptive_lrs)
        {
            if !grads.is_empty() {
                layer.apply_gradients(&grads, adaptive_lr)?;
            }
        }

        // PolyAttention-only: no learned threshold predictors to update

        Ok((
            batch_loss,
            batch_base_loss,
            grad_norm,
            layer_param_grad_norm_sq,
        ))
    }

    /// Compute layer-wise adaptive learning rate using bidirectional LARS
    /// Reference: "LARS: Layer-wise Adaptive Rate Scaling" (You et al., 2017)
    ///
    /// Bidirectional approach: Balance gradient flow across all layers
    /// - High-gradient layers (L0-L2): Reduce LR to prevent over-updating
    /// - Low-gradient layers (L6-L14): Increase LR to prevent under-updating
    /// - Target: All layers converge at similar rates
    ///
    /// Formula (trust-ratio + bidirectional balance):
    /// lr_layer = lr_base * clamp( (||W|| / (||∇W|| + ε)) * (median_grad_norm / (||∇W|| +
    /// ε))^power, [min,max] )
    /// - Trust-ratio term encourages proportionate updates relative to parameter scale
    /// - Bidirectional balance aligns layer grad norms towards the batch median
    fn compute_layer_adaptive_lr_static(
        layer: &LayerEnum,
        grads: &[Array2<f32>],
        base_lr: f32,
        layer_idx: usize,
        median_grad_norm: f32,
    ) -> f32 {
        // Skip for layers without gradients
        if grads.is_empty() {
            return base_lr;
        }

        // Compute gradient norm ||∇W||
        let grad_norm: f32 = grads
            .iter()
            .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt();

        // Avoid division by zero
        const EPSILON: f32 = 1e-6;
        if grad_norm < EPSILON {
            return base_lr;
        }

        // Trust-ratio term: ||W|| / ||∇W||
        let w_norm = layer.weight_norm();
        if w_norm < EPSILON {
            return base_lr;
        }
        let trust_ratio = w_norm / (grad_norm + EPSILON);

        // Bidirectional balance relative to batch median
        const POWER_BALANCE: f32 = 0.5; // Gentle correction
        let balance_scale = (median_grad_norm / (grad_norm + EPSILON)).powf(POWER_BALANCE);

        // Combined scale with conservative clamping
        // Tighter bounds reduce jitter and large swings
        // Expanded range to allow LARS to effectively throttle exploding gradients (e.g. in TRM)
        const MIN_SCALE: f32 = 0.01;
        const MAX_SCALE: f32 = 5.0;
        let scale = (trust_ratio * balance_scale).clamp(MIN_SCALE, MAX_SCALE);
        let adaptive_lr = base_lr * scale;

        // Log adaptive LR for debugging (use RUST_LOG=debug to see)
        if layer_idx <= 2 || layer_idx >= 12 {
            tracing::debug!(
                layer_idx = layer_idx,
                layer_type = layer.layer_type(),
                grad_norm = grad_norm,
                base_lr = base_lr,
                adaptive_lr = adaptive_lr,
                scale = scale,
                "Bidirectional LARS"
            );
        }

        adaptive_lr
    }

    /// Detect gradient anomalies that may indicate training instability or poisoning
    fn detect_gradient_anomalies(grads: &[Array2<f32>]) -> Result<()> {
        for (i, grad) in grads.iter().enumerate() {
            let max_grad = grad.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            if max_grad > crate::GRADIENT_ANOMALY_THRESHOLD {
                tracing::warn!(
                    "Gradient anomaly detected in layer {}: max gradient magnitude {}",
                    i,
                    max_grad
                );
                return Err(ModelError::GradientError {
                    message: format!(
                        "Gradient anomaly detected in layer {}: max gradient magnitude {}",
                        i, max_grad
                    ),
                });
            }

            let nan_count = grad.iter().filter(|&x| x.is_nan()).count();
            let inf_count = grad.iter().filter(|&x| x.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                tracing::error!(
                    "Non-finite gradients detected in layer {}: {} NaN, {} Inf values",
                    i,
                    nan_count,
                    inf_count
                );
                // Log some sample values for debugging
                let first_10: Vec<f32> = grad.iter().take(10).cloned().collect();
                tracing::error!("First 10 gradient values: {:?}", first_10);
                return Err(ModelError::GradientError {
                    message: format!("Non-finite gradients detected in layer {}", i),
                });
            }
        }
        Ok(())
    }

    #[inline]
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        self.vocab.tokenize(text)
    }

    /// In-place tokenization to reuse a caller-provided buffer.
    #[inline]
    pub fn tokenize_into(&self, text: &str, out: &mut Vec<usize>) {
        self.vocab.tokenize_into(text, out)
    }

    /// Save model to JSON format (human-readable, larger file size)
    pub fn save_json(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;
        fs::write(path, json).map_err(ModelError::from)?;
        Ok(())
    }

    /// Load model from JSON format
    pub fn load_json(path: &str) -> Result<Self> {
        let data = fs::read_to_string(path).map_err(ModelError::from)?;
        let llm: LLM = serde_json::from_str(&data).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;
        Ok(llm)
    }

    /// Save model to binary format (compact, faster, smaller file size)
    pub fn save_binary(&self, path: &str) -> Result<()> {
        let config = bincode::config::standard();
        let encoded =
            bincode::serde::encode_to_vec(self, config).map_err(|e| ModelError::Serialization {
                source: Box::new(e),
            })?;
        fs::write(path, encoded).map_err(ModelError::from)?;
        Ok(())
    }

    /// Load model from binary format
    pub fn load_binary(path: &str) -> Result<Self> {
        let data = fs::read(path).map_err(ModelError::from)?;
        let config = bincode::config::standard();
        let (llm, _): (LLM, usize) =
            bincode::serde::decode_from_slice(&data, config).map_err(|e| {
                ModelError::Serialization {
                    source: Box::new(e),
                }
            })?;
        Ok(llm)
    }

    /// Save model (auto-detects format from extension: .json or .bin)
    pub fn save(&self, path: &str) -> Result<()> {
        if path.ends_with(".json") {
            self.save_json(path)
        } else {
            self.save_binary(path)
        }
    }

    pub fn total_weight_norm(&self) -> f32 {
        self.network.iter().map(|layer| layer.weight_norm()).sum()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_diffusion_ce(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        ce_weight: AdaptiveScalar,
        validation_ratio: f32,
        min_snr_gamma: AdaptiveScalar,
        checkpoint_every: Option<usize>,
        checkpoint_dir: Option<String>,
        checkpoint_stage: Option<String>,
    ) -> Result<()> {
        let mut diffusion_blocks_idx: Vec<usize> = Vec::new();
        let mut embeddings_idx: Option<usize> = None;
        let mut norm_idx: Option<usize> = None;
        let mut out_proj_idx: Option<usize> = None;
        for (i, layer) in self.network.iter().enumerate() {
            match layer {
                LayerEnum::TokenEmbeddings(_) => {
                    if embeddings_idx.is_none() {
                        embeddings_idx = Some(i)
                    }
                }
                LayerEnum::DiffusionBlock(_) => diffusion_blocks_idx.push(i),
                LayerEnum::DynamicTanhNorm(_) => norm_idx = Some(i),
                LayerEnum::OutputProjection(_) => out_proj_idx = Some(i),
                _ => {}
            }
        }
        if embeddings_idx.is_none() || diffusion_blocks_idx.is_empty() || out_proj_idx.is_none() {
            return Err(ModelError::Training {
                message: String::from(
                    "Missing required layers for diffusion CE (embeddings/diffusion/output)",
                ),
            });
        }
        let first_block = diffusion_blocks_idx[0];
        let num_timesteps = if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
            b.noise_scheduler.num_timesteps()
        } else {
            1000
        };

        // "Learn" an effective DDIM step count over training by tracking validation loss trends.
        // This is stored in the diffusion block config so it is checkpointed, while still
        // remaining overridable at runtime via CLI.
        let mut ddim_steps_min: usize = 16;
        let mut ddim_steps_max: usize = 256;
        let mut learned_ddim_steps: usize =
            if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                match b.config.ddim_steps_policy {
                    crate::domain::layers::diffusion::DdimStepsPolicy::Fixed(k) => k.max(1),
                    crate::domain::layers::diffusion::DdimStepsPolicy::Auto {
                        min_steps,
                        max_steps,
                    } => {
                        ddim_steps_min = min_steps.max(1);
                        ddim_steps_max = max_steps.max(ddim_steps_min);
                        // Start from ~T/10 like common practice; then adapt during training.
                        ((num_timesteps.max(1) as f32 / 10.0).round() as usize).max(1)
                    }
                }
            } else {
                ((num_timesteps.max(1) as f32 / 10.0).round() as usize).max(1)
            };
        learned_ddim_steps = learned_ddim_steps
            .min(num_timesteps.max(1))
            .clamp(ddim_steps_min, ddim_steps_max);
        let mut prev_val_loss: Option<f32> = None;
        let mut steps_plateau_epochs: usize = 0;

        let timestep_strategy = if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
            b.timestep_strategy()
        } else {
            DiffusionTimestepStrategy::Uniform
        };
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let mut rng = get_rng();

        let mut denoise_ema_per_t = vec![1.0f32; num_timesteps.max(1)];
        let mut denoise_cnt_per_t = vec![0u32; num_timesteps.max(1)];
        let denoise_ema_decay: f32 = 0.99;
        let denoise_importance_power: f32 = 0.5;
        let min_samples_before_adapt: u32 = 64;

        // Online normalization for per-example MSE weights.
        // Keeps loss/gradient scale stable even when the (adaptive) weighting becomes skewed.
        let mut mse_weight_ema: f32 = 1.0;
        let mse_weight_ema_decay: f32 = 0.995;
        let mse_weight_min: f32 = 0.1;
        let mse_weight_max: f32 = 10.0;
        let richards_sigmoid = crate::domain::richards::RichardsCurve::sigmoid(false);
        let lambda_ce_schedule = |t: usize| -> f32 {
            let total = num_timesteps.max(1) as f32;
            let center = 0.5 * total;
            let sigma = (0.15 * total).max(1.0);
            let capped_t = t.min(num_timesteps.saturating_sub(1)) as f32;
            let x = (center - capped_t) / sigma;
            let s = richards_sigmoid.forward_scalar_f32(x);
            s.clamp(0.5, 1.0)
        };
        let log_dir = std::path::Path::new("training_logs");
        let _ = std::fs::create_dir_all(log_dir);
        let ts = format!("{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
        let mut log_file =
            std::fs::File::create(log_dir.join(format!("diffusion-{}.csv", ts))).ok();
        if let Some(f) = &mut log_file {
            use std::io::Write;
            let _ = writeln!(
                f,
                "epoch,loss,sce,mse,lambda_ce,lr,grad_norm,val_loss,val_sce,val_mse"
            );
        }
        let mut lr_scale = 1.0f32;
        let mut best_loss = f32::INFINITY;
        let mut plateau_epochs = 0usize;
        let plateau_patience = 5usize;
        let plateau_reduce = 0.5f32;
        let min_lr_scale = 0.1f32;
        let effective_batch_size = batch_size.max(1);

        // Warmup epochs default to 15% of total for stability
        let warmup_epochs = ((epochs as f32) * 0.15).ceil() as usize;

        // Split data into training and validation sets
        let val_start = (data.len() as f32 * (1.0 - validation_ratio)).floor() as usize;
        let train_data = &data[..val_start];
        let val_data = &data[val_start..];

        for epoch in 0..epochs {
            let t_epoch_start = std::time::Instant::now();
            // Learning rate warmup + cosine annealing (SGDR)
            let base_lr = if epoch < warmup_epochs {
                lr * ((epoch + 1) as f32 / warmup_epochs as f32)
            } else {
                let t = (epoch - warmup_epochs) as f32;
                let t_max = (epochs - warmup_epochs).max(1) as f32;
                let lr_min = lr * 0.10;
                let lr_max = lr;
                lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos())
            };
            let effective_lr = base_lr * lr_scale;

            let training_progress = if epochs > warmup_epochs {
                (epoch.saturating_sub(warmup_epochs) as f64)
                    / ((epochs.saturating_sub(warmup_epochs)).max(1) as f64)
            } else {
                0.0
            };
            for layer in &mut self.network {
                layer.set_training_progress(training_progress);
            }
            let current_gamma = min_snr_gamma.value(training_progress);
            let current_ce_weight = ce_weight.value(training_progress);

            // Epoch-level adaptive sampling CDF over the curriculum-active timesteps.
            let max_t_epoch =
                ((num_timesteps as f32) * ((epoch + 1) as f32 / epochs as f32)).round() as usize;
            let active_steps_epoch = max_t_epoch.max(1);
            let sampling_cdf: Vec<f32> = {
                // Normalize difficulty by mean to avoid collapsing onto a narrow band of
                // timesteps as the EMA evolves.
                let mut diff_sum = 0.0f32;
                let mut diff_count = 0u32;
                for t in 0..active_steps_epoch {
                    if denoise_cnt_per_t.get(t).copied().unwrap_or(0) >= min_samples_before_adapt {
                        diff_sum += denoise_ema_per_t.get(t).copied().unwrap_or(1.0).max(1e-12);
                        diff_count = diff_count.saturating_add(1);
                    }
                }
                let diff_mean = if diff_count > 0 {
                    (diff_sum / diff_count as f32).max(1e-12)
                } else {
                    1.0
                };

                let mut weights = Vec::with_capacity(active_steps_epoch);
                // Base distribution (schedule/target-aware) + online adaptive reweighting (learned from
                // data via per-timestep EMA difficulty).
                let base_weights_full: Vec<f32> =
                    if let LayerEnum::DiffusionBlock(b0) = &self.network[first_block] {
                        match timestep_strategy {
                            DiffusionTimestepStrategy::MinSnr => (0..num_timesteps)
                                .map(|t| b0.min_snr_weight(t, current_gamma).max(1e-12))
                                .collect(),
                            DiffusionTimestepStrategy::EdmLogNormal => {
                                // EDM log-normal sampling over σ, discretized over timesteps.
                                let p_mean: f32 = -1.2;
                                let p_std: f32 = 1.2;
                                let norm_const: f32 =
                                    1.0 / (p_std * (2.0 * std::f32::consts::PI).sqrt());
                                (0..num_timesteps)
                                    .map(|t| {
                                        if t == 0 {
                                            return 0.0;
                                        }
                                        let alpha_bar = b0
                                            .noise_scheduler
                                            .sqrt_alpha_cumprod(t)
                                            .powi(2)
                                            .clamp(1e-12, 1.0 - 1e-12);
                                        let sigma = crate::domain::layers::diffusion::edm::sigma_from_alpha_bar(
                                            alpha_bar,
                                        )
                                        .max(1e-6);
                                        let log_sigma = sigma.ln();
                                        let z = (log_sigma - p_mean) / p_std;
                                        (norm_const * (-0.5 * z * z).exp()).max(1e-12)
                                    })
                                    .collect()
                            }
                            DiffusionTimestepStrategy::Uniform => vec![1.0f32; num_timesteps],
                        }
                    } else {
                        vec![1.0f32; num_timesteps]
                    };

                for t in 0..active_steps_epoch {
                    let base = base_weights_full.get(t).copied().unwrap_or(1.0).max(1e-12);
                    let adapt_ready =
                        denoise_cnt_per_t.get(t).copied().unwrap_or(0) >= min_samples_before_adapt;
                    let diff = if adapt_ready {
                        let d = denoise_ema_per_t.get(t).copied().unwrap_or(1.0).max(1e-12);
                        let ratio = (d / diff_mean).clamp(0.25, 4.0);
                        ratio.powf(denoise_importance_power)
                    } else {
                        1.0
                    };
                    weights.push((base * diff).max(1e-12));
                }
                let sum: f32 = weights.iter().sum();
                if sum > 0.0 && sum.is_finite() {
                    let mut acc = 0.0f32;
                    weights
                        .into_iter()
                        .map(|w| {
                            acc += w / sum;
                            acc.min(1.0)
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            };
            let mut total_loss = 0.0f32;
            let mut total_mse = 0.0f32;
            let mut mse_examples = 0usize;
            let mut total_ce = 0.0f32;
            let mut total_lambda_ce = 0.0f32;
            let mut count = 0usize;
            let mut total_grad_norm_sq = 0.0f32;

            for batch_strs in train_data.chunks(effective_batch_size) {
                let batch_tokenized: Vec<Vec<usize>> = batch_strs
                    .par_iter()
                    .map(|input| self.tokenize(input))
                    .collect();

                let batch_response_spans: Vec<Option<(usize, usize)>> = batch_tokenized
                    .iter()
                    .map(|seq| response_span_from_tokens(&self.vocab, seq))
                    .collect();

                self.training_scratch.reset(self.network.len());
                let mut examples_in_batch = 0usize;
                for (i, training_row) in batch_tokenized.iter().enumerate() {
                    if training_row.len() < 2 {
                        continue;
                    }
                    examples_in_batch += 1;

                    let response_span = batch_response_spans[i];

                    let input_ids = &training_row[..training_row.len() - 1];
                    let target_ids = &training_row[1..];

                    let mut ids_arr = Array2::<f32>::zeros((1, input_ids.len()));
                    for (i, &token_id) in input_ids.iter().enumerate() {
                        ids_arr[[0, i]] = token_id as f32;
                    }

                    // x0 via embeddings
                    let emb_idx = embeddings_idx.unwrap();
                    let x0 = match &mut self.network[emb_idx] {
                        LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                        _ => {
                            return Err(ModelError::Training {
                                message: String::from("Embeddings layer missing"),
                            });
                        }
                    };

                    // Decide discrete masked vs continuous path per first diffusion block
                    let is_discrete =
                        if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                            b.is_discrete_masked()
                        } else {
                            false
                        };
                    let mask_id_opt =
                        if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                            b.mask_token_id()
                        } else {
                            None
                        };
                    let mut noise = Array2::<f32>::zeros(x0.raw_dim());
                    if let Some(slice) = noise.as_slice_mut() {
                        if crate::common::rng::is_seeded() {
                            // Deterministic mode: avoid parallel RNG call-order sensitivity.
                            for v in slice.iter_mut() {
                                *v = normal.sample(&mut rng) as f32;
                            }
                        } else {
                            slice.par_iter_mut().for_each(|v| {
                                *v = normal.sample(&mut get_rng()) as f32;
                            });
                        }
                    } else {
                        for v in noise.iter_mut() {
                            *v = normal.sample(&mut rng) as f32;
                        }
                    }
                    // Adaptive timestep sampling (curriculum by epoch + complexity)
                    let complexity = {
                        let unique = training_row
                            .iter()
                            .copied()
                            .collect::<std::collections::BTreeSet<usize>>()
                            .len() as f32;
                        (unique / training_row.len().max(1) as f32).clamp(0.0, 1.0)
                    };
                    let active_steps = active_steps_epoch;
                    let candidate = if sampling_cdf.is_empty() {
                        rng.random_range(0..active_steps)
                    } else {
                        let r: f32 = rng.random();
                        let mut lo = 0usize;
                        let mut hi = sampling_cdf.len();
                        while lo < hi {
                            let mid = (lo + hi) / 2;
                            if sampling_cdf[mid] < r {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        let idx = if lo >= sampling_cdf.len() {
                            sampling_cdf.len() - 1
                        } else {
                            lo
                        };
                        idx.min(active_steps.saturating_sub(1))
                    };
                    let t = (((1.0 - complexity) * candidate as f32).round() as usize)
                        .min(active_steps - 1);
                    let (x_t, sqrt_a, sqrt_one_minus_a, discrete_used) = if is_discrete {
                        let mask_token_id = mask_id_opt
                            .or_else(|| self.vocab.encode("<mask>"))
                            .unwrap_or(self.vocab.encode_or_unknown("<unk>").unwrap_or(0));
                        let ids_masked =
                            if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                                if let Some(ds) = &b.discrete_scheduler {
                                    if let Some((span_start, span_end)) = response_span {
                                        ds.mask_sequence_span_at_t(
                                            &ids_arr,
                                            mask_token_id,
                                            t,
                                            span_start,
                                            span_end,
                                        )
                                    } else {
                                        ds.mask_sequence_at_t(&ids_arr, mask_token_id, t)
                                    }
                                } else {
                                    ids_arr.clone()
                                }
                            } else {
                                ids_arr.clone()
                            };
                        let x_t_local = match &mut self.network[embeddings_idx.unwrap()] {
                            LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_masked),
                            _ => x0.clone(),
                        };
                        (x_t_local, 1.0, 0.0, true)
                    } else if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                        let x_t_local = b.noise_scheduler.q_sample(&x0, t, &noise);
                        let sa = b.noise_scheduler.sqrt_alpha_cumprod(t);
                        let soa = b.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                        (x_t_local, sa, soa, false)
                    } else {
                        continue;
                    };

                    // Predict via full diffusion stack (epsilon or v parameterization)
                    let mut pred = x_t.clone();
                    for &idx in &diffusion_blocks_idx {
                        if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                            b.set_causal_attention(true);
                            b.set_timestep(t);
                            pred = b.forward_with_timestep(&pred, t);
                        }
                    }

                    // Recover x0_hat for continuous path according to parameterization
                    let x0_hat = if discrete_used {
                        pred.clone()
                    } else if let LayerEnum::DiffusionBlock(b0) = &self.network[first_block] {
                        match b0.prediction_target() {
                            crate::domain::layers::diffusion::DiffusionPredictionTarget::Epsilon => {
                                let sa = sqrt_a.max(1e-6);
                                let pred_scaled = &pred * sqrt_one_minus_a;
                                (&x_t - &pred_scaled) / sa
                            }
                            crate::domain::layers::diffusion::DiffusionPredictionTarget::VPrediction => {
                                (&x_t * sqrt_a) - (&pred * sqrt_one_minus_a)
                            }
                            crate::domain::layers::diffusion::DiffusionPredictionTarget::Sample => {
                                pred.clone()
                            }
                            crate::domain::layers::diffusion::DiffusionPredictionTarget::EdmX0 => {
                                pred.clone()
                            }
                        }
                    } else {
                        pred.clone()
                    };

                    // Forward through final norm (if present) and output projection
                    let mut hidden = x0_hat.clone();
                    if let Some(nidx) = norm_idx
                        && let LayerEnum::DynamicTanhNorm(norm) = &mut self.network[nidx]
                    {
                        hidden = norm.forward(&hidden);
                    }

                    let logits = if let Some(opidx) = out_proj_idx {
                        if let LayerEnum::OutputProjection(op) = &mut self.network[opidx] {
                            op.forward(&hidden)
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };
                    let probs = crate::domain::soft::Softmax::new().forward_immutable(&logits.view());
                    let target_len = target_ids.len();
                    let probs_slice = probs.slice(s![0..target_len, ..]);
                    let lambda_ce = if discrete_used {
                        1.0f32
                    } else {
                        lambda_ce_schedule(t)
                    };
                    let lambda_eps = if discrete_used {
                        0.0f32
                    } else {
                        1.0f32 - lambda_ce
                    };
                    total_lambda_ce += lambda_ce;
                    let sce = crate::domain::loss::symmetric_cross_entropy(
                        &probs_slice.to_owned(),
                        target_ids,
                        current_ce_weight * lambda_ce,
                        current_ce_weight * lambda_ce,
                        1e-4,
                    );

                    // Auxiliary: residual decorrelation on pre-logit hidden.
                    let mut decor_term: f32 = 0.0;
                    let mut decor_grad_opt: Option<Array2<f32>> = None;
                    let base_w = self.training_hparams.residual_decorrelation_weight;
                    if base_w > 0.0 {
                        let difficulty = if self.training_hparams.residual_decorrelation_adaptive {
                            (sce / (sce + 1.0)).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let w = base_w * (1.0 + difficulty);
                        let dl = crate::domain::loss::residual_decorrelation_loss(&hidden.view());
                        decor_term = w * dl;
                        let dg = crate::domain::loss::residual_decorrelation_gradients(&hidden.view());
                        decor_grad_opt = Some(dg.mapv(|x| x * w));
                    }

                    // Auxiliary: hard-negative repulsion on pooled pre-logit hidden.
                    let mut hardneg_term: f32 = 0.0;
                    let mut hardneg_grad_opt: Option<Array2<f32>> = None;
                    let base_hn_w = self.training_hparams.residual_hardneg_weight;
                    if base_hn_w > 0.0 {
                        let difficulty = if self.training_hparams.residual_hardneg_adaptive {
                            (sce / (sce + 1.0)).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let w = base_hn_w * (1.0 + difficulty);

                        let rows = hidden.nrows().max(1);
                        let cols = hidden.ncols();
                        let mut anchor = vec![0.0f32; cols];
                        for i in 0..rows {
                            for j in 0..cols {
                                let v = hidden[[i, j]];
                                anchor[j] += if v.is_finite() { v } else { 0.0 };
                            }
                        }
                        let inv = 1.0f32 / (rows as f32);
                        for a in &mut anchor {
                            *a *= inv;
                        }

                        let (hn_loss, grad_anchor) =
                            crate::domain::loss::hard_negative_repulsion_loss_and_grad(
                                &anchor,
                                self.residual_neg_bank.as_slice(),
                                self.training_hparams.residual_hardneg_k,
                                self.training_hparams.residual_hardneg_margin,
                                self.training_hparams.residual_hardneg_temperature,
                            );
                        hardneg_term = w * hn_loss;

                        let mut g = Array2::<f32>::zeros(hidden.raw_dim());
                        for i in 0..rows {
                            for j in 0..cols {
                                g[[i, j]] = (grad_anchor[j] * w) * inv;
                            }
                        }
                        hardneg_grad_opt = Some(g);

                        self.residual_neg_bank
                            .push(anchor, self.training_hparams.residual_hardneg_bank_size);
                    }

                    let (denoise_target, w_mse_raw) = if discrete_used {
                        (None, 1.0f32)
                    } else if let LayerEnum::DiffusionBlock(b0) = &self.network[first_block] {
                        let mut w = b0.min_snr_weight(t, current_gamma);
                        if b0.prediction_target()
                            == crate::domain::layers::diffusion::DiffusionPredictionTarget::EdmX0
                        {
                            w *= b0.edm_loss_weight(t);
                        }
                        (Some(b0.training_target(&x0, &noise, t)), w)
                    } else {
                        (None, 1.0f32)
                    };

                    if !discrete_used {
                        mse_weight_ema = mse_weight_ema_decay * mse_weight_ema
                            + (1.0 - mse_weight_ema_decay) * w_mse_raw.max(1e-12);
                    }
                    let w_mse = if discrete_used {
                        1.0f32
                    } else {
                        (w_mse_raw / mse_weight_ema.max(1e-6)).clamp(mse_weight_min, mse_weight_max)
                    };

                    // CE grads expanded to full logits shape
                    let mut grads_logits = Array2::<f32>::zeros(logits.raw_dim());
                    let sce_grads_slice = crate::domain::loss::symmetric_cross_entropy_gradients(
                        &probs_slice.to_owned(),
                        target_ids,
                        current_ce_weight * lambda_ce,
                        current_ce_weight * lambda_ce,
                        1e-4,
                    );
                    grads_logits
                        .slice_mut(ndarray::s![0..target_len, ..])
                        .assign(&sce_grads_slice);

                    // Backward through output projection
                    let (mut grad_hidden, op_param_grads) = if let Some(opidx) = out_proj_idx {
                        if let LayerEnum::OutputProjection(op) = &mut self.network[opidx] {
                            op.compute_gradients(&hidden, &grads_logits)
                        } else {
                            (grads_logits.clone(), Vec::new())
                        }
                    } else {
                        (grads_logits.clone(), Vec::new())
                    };

                    if let Some(dg) = decor_grad_opt {
                        grad_hidden = grad_hidden + dg;
                    }

                    if let Some(dg) = hardneg_grad_opt {
                        grad_hidden = grad_hidden + dg;
                    }
                    if let Some(opidx) = out_proj_idx
                        && !op_param_grads.is_empty()
                    {
                        if let Some(slot) = &mut self.training_scratch.grads_per_layer[opidx] {
                            for (i, g) in op_param_grads.iter().enumerate() {
                                if i < slot.len() {
                                    slot[i] = &slot[i] + g;
                                } else {
                                    slot.push(g.clone());
                                }
                            }
                        } else {
                            self.training_scratch.grads_per_layer[opidx] =
                                Some(op_param_grads.clone());
                        }
                    }

                    // Backward through norm to x0_hat
                    if let Some(nidx) = norm_idx
                        && let LayerEnum::DynamicTanhNorm(norm) = &mut self.network[nidx]
                    {
                        grad_hidden = norm.backward(&grad_hidden, lr);
                    }

                    // Build gradient for diffusion stack from mixed objectives
                    let mut grad_pred = if discrete_used {
                        // Discrete masked: CE only path, treat as grad on predicted embeddings
                        grad_hidden.clone()
                    } else if let LayerEnum::DiffusionBlock(b0) = &self.network[first_block] {
                        let grad_ce = match b0.prediction_target() {
                            crate::domain::layers::diffusion::DiffusionPredictionTarget::Epsilon => {
                                let sa = sqrt_a.max(1e-6);
                                let coeff = -sqrt_one_minus_a / sa;
                                grad_hidden.mapv(|x| x * coeff)
                            }
                            crate::domain::layers::diffusion::DiffusionPredictionTarget::VPrediction => {
                                let coeff = -sqrt_one_minus_a;
                                grad_hidden.mapv(|x| x * coeff)
                            }
                            crate::domain::layers::diffusion::DiffusionPredictionTarget::Sample => {
                                grad_hidden.clone()
                            }
                            crate::domain::layers::diffusion::DiffusionPredictionTarget::EdmX0 => {
                                grad_hidden.clone()
                            }
                        };
                        let mut grad_total = grad_ce.mapv(|x| x * lambda_ce);
                        if let Some(target) = denoise_target.as_ref() {
                            let mut grad_mse = &pred - target;
                            let denom = (pred.nrows() * pred.ncols()) as f32;
                            if denom > 0.0 {
                                grad_mse.mapv_inplace(|x| (2.0 / denom) * x);
                            } else {
                                grad_mse.fill(0.0);
                            }
                            grad_total = grad_total + grad_mse.mapv(|x| x * (lambda_eps * w_mse));
                        }
                        grad_total
                    } else {
                        grad_hidden.clone()
                    };

                    // Gradient clipping by global norm
                    let grad_norm_pred: f32 = grad_pred.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let clip_norm: f32 = 2.0;
                    if grad_norm_pred > clip_norm && grad_norm_pred.is_finite() {
                        let scale = clip_norm / grad_norm_pred;
                        grad_pred.mapv_inplace(|g| g * scale);
                    }

                    // Backprop through diffusion stack (reverse order)
                    for &idx in diffusion_blocks_idx.iter().rev() {
                        let (in_grad, param_grads) = match &self.network[idx] {
                            LayerEnum::DiffusionBlock(b) => b.compute_gradients(&x_t, &grad_pred),
                            _ => (grad_pred.clone(), Vec::<Array2<f32>>::new()),
                        };
                        if !param_grads.is_empty() {
                            if let Some(slot) = &mut self.training_scratch.grads_per_layer[idx] {
                                for (i, g) in param_grads.iter().enumerate() {
                                    if i < slot.len() {
                                        slot[i] = &slot[i] + g;
                                    } else {
                                        slot.push(g.clone());
                                    }
                                }
                            } else {
                                self.training_scratch.grads_per_layer[idx] =
                                    Some(param_grads.clone());
                            }
                        }
                        grad_pred = in_grad;
                    }

                    // Map gradients from x_t back to x_0 and update embeddings
                    let grad_x0 = if discrete_used {
                        // Discrete masked: x_t derived from embeddings(ids_masked) directly
                        grad_pred.clone()
                    } else {
                        // Continuous: x_t = sqrt(a) * x0 + sqrt(1-a) * noise → dL/dx0 = sqrt(a) *
                        // dL/dx_t
                        let sa = sqrt_a.max(1e-6);
                        grad_pred.mapv(|g| g * sa)
                    };

                    if let Some(eidx) = embeddings_idx
                        && let LayerEnum::TokenEmbeddings(layer) = &mut self.network[eidx]
                    {
                        let (emb_in_grad, emb_param_grads) =
                            layer.compute_gradients(&ids_arr, &grad_x0);
                        let _ = emb_in_grad;
                        if !emb_param_grads.is_empty() {
                            if let Some(slot) = &mut self.training_scratch.grads_per_layer[eidx] {
                                for (i, g) in emb_param_grads.iter().enumerate() {
                                    if i < slot.len() {
                                        slot[i] = &slot[i] + g;
                                    } else {
                                        slot.push(g.clone());
                                    }
                                }
                            } else {
                                self.training_scratch.grads_per_layer[eidx] =
                                    Some(emb_param_grads.clone());
                            }
                        }
                    }

                    // Losses and grad norm
                    // Track epsilon MSE separately for monitoring when using continuous noise
                    let mse = if let Some(target) = denoise_target.as_ref() {
                        crate::domain::loss::epsilon_mse(&pred, target)
                    } else {
                        0.0
                    };
                    if !discrete_used {
                        total_mse += mse;
                        mse_examples += 1;
                    }
                    let loss = if discrete_used {
                        sce
                    } else {
                        lambda_ce * sce + (lambda_eps * w_mse) * mse
                    } + decor_term
                        + hardneg_term;
                    total_loss += loss;
                    total_ce += sce;
                    count += 1;
                    total_grad_norm_sq += grad_pred.iter().map(|&x| x * x).sum::<f32>();

                    // Update adaptive timestep sampler statistics (learned difficulty).
                    if !discrete_used && t < denoise_ema_per_t.len() {
                        let prev = denoise_ema_per_t[t];
                        denoise_ema_per_t[t] =
                            denoise_ema_decay * prev + (1.0 - denoise_ema_decay) * mse.max(0.0);
                        denoise_cnt_per_t[t] = denoise_cnt_per_t[t].saturating_add(1);
                    }
                }
                // Apply averaged grads per layer after batch
                let mut grads_per_layer =
                    std::mem::take(&mut self.training_scratch.grads_per_layer);
                for (idx, maybe_grads) in grads_per_layer.iter_mut().enumerate() {
                    if let Some(mut grads) = maybe_grads.take() {
                        if examples_in_batch > 0 {
                            for g in &mut grads {
                                *g = g.mapv(|x| x / examples_in_batch as f32);
                            }
                        }
                        let clip_layer = 1000.0f32;
                        for g in &mut grads {
                            let nrm: f32 = g.iter().map(|&x| x * x).sum::<f32>().sqrt();
                            if nrm.is_finite() && nrm > clip_layer {
                                let scale = clip_layer / nrm;
                                g.mapv_inplace(|x| x * scale);
                            }
                        }
                        // Detect anomalies before applying
                        Self::detect_gradient_anomalies(&grads)?;
                        match &mut self.network[idx] {
                            LayerEnum::DiffusionBlock(b) => {
                                b.apply_gradients(&grads, effective_lr)?
                            }
                            LayerEnum::OutputProjection(op) => {
                                op.apply_gradients(&grads, effective_lr)?
                            }
                            LayerEnum::TokenEmbeddings(layer) => {
                                layer.apply_gradients(&grads, effective_lr)?
                            }
                            _ => {}
                        }
                    }
                }
                self.training_scratch.grads_per_layer = grads_per_layer;
            }

            let avg_loss = if count > 0 {
                total_loss / count as f32
            } else {
                0.0
            };
            let avg_sce = if count > 0 {
                total_ce / count as f32
            } else {
                0.0
            };
            let avg_mse = if mse_examples > 0 {
                total_mse / mse_examples as f32
            } else {
                0.0
            };
            let avg_lambda_ce = if count > 0 {
                total_lambda_ce / count as f32
            } else {
                0.0
            };
            let grad_norm = total_grad_norm_sq.sqrt();
            let epoch_ms = t_epoch_start.elapsed().as_secs_f64() as f32 * 1000.0;
            let tokens_per_sec = if count > 0 {
                (count as f32) / (t_epoch_start.elapsed().as_secs_f32().max(1e-6))
            } else {
                0.0
            };
            let mut tau_range: Option<(f32, f32)> = None;
            let mut pred_norm_rms: Option<f32> = None;
            for layer in &mut self.network {
                if let LayerEnum::TransformerBlock(tb) = layer
                    && let crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) =
                        tb.temporal_mixing_mut()
                {
                    tau_range = attn.take_tau_metrics();
                    pred_norm_rms = attn.take_pred_norm();
                }
                if let LayerEnum::DiffusionBlock(db) = layer
                    && let crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) =
                        &mut db.temporal_mixing
                {
                    tau_range = attn.take_tau_metrics();
                    pred_norm_rms = attn.take_pred_norm();
                }
                if let LayerEnum::LRM(lrm) = layer {
                    tau_range = lrm.attention_mut().take_tau_metrics();
                    pred_norm_rms = lrm.attention_mut().take_pred_norm();
                }
            }
            let metrics = crate::domain::attention::poly_attention::DegreeAdaptationMetrics {
                epoch_index: epoch,
                loss_delta: 0.0,
                grad_norm,
                epoch_ms,
                tokens_per_sec,
                tau_range,
                pred_norm_rms,
            };
            for layer in &mut self.network {
                if let LayerEnum::TransformerBlock(tb) = layer
                    && let crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) =
                        tb.temporal_mixing_mut()
                {
                    attn.adapt_degree(&metrics);
                }
                if let LayerEnum::DiffusionBlock(db) = layer
                    && let crate::domain::layers::components::common::TemporalMixingLayer::Attention(attn) =
                        &mut db.temporal_mixing
                {
                    attn.adapt_degree(&metrics);
                }
                if let LayerEnum::LRM(lrm) = layer {
                    lrm.attention_mut().adapt_degree(&metrics);
                }
            }
            // Validation split (last 10% examples)
            let mut val_loss_total = 0.0f32;
            let mut val_sce_total = 0.0f32;
            let mut val_mse_total = 0.0f32;
            let mut val_count = 0usize;

            for batch_strs in val_data.chunks(effective_batch_size) {
                let batch_tokenized: Vec<Vec<usize>> = batch_strs
                    .par_iter()
                    .map(|input| self.tokenize(input))
                    .collect();
                let batch_response_spans: Vec<Option<(usize, usize)>> = batch_tokenized
                    .iter()
                    .map(|seq| response_span_from_tokens(&self.vocab, seq))
                    .collect();

                for (i, training_row) in batch_tokenized.iter().enumerate() {
                    if training_row.len() < 2 {
                        continue;
                    }
                    let response_span = batch_response_spans[i];
                    let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];
                let mut ids_arr = Array2::<f32>::zeros((1, input_ids.len()));
                for (i, &tid) in input_ids.iter().enumerate() {
                    ids_arr[[0, i]] = tid as f32;
                }
                let emb_idx = embeddings_idx.unwrap();
                let x0 = match &mut self.network[emb_idx] {
                    LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                    _ => continue,
                };
                let first_block = diffusion_blocks_idx[0];
                let is_discrete = if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                    b.is_discrete_masked()
                } else {
                    false
                };
                let mask_id_opt = if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                    b.mask_token_id()
                } else {
                    None
                };
                let mut noise = Array2::<f32>::zeros(x0.raw_dim());
                if let Some(slice) = noise.as_slice_mut() {
                    slice.par_iter_mut().for_each(|v| {
                        *v = normal.sample(&mut get_rng()) as f32;
                    });
                } else {
                    for v in noise.iter_mut() {
                        *v = normal.sample(&mut rng) as f32;
                    }
                }
                let t = rng.random_range(0..num_timesteps.max(1));
                let (x_t, sqrt_a, sqrt_one_minus_a, discrete_used) = {
                    if is_discrete {
                        let mask_token_id = mask_id_opt
                            .or_else(|| self.vocab.encode("<mask>"))
                            .unwrap_or(self.vocab.encode_or_unknown("<unk>").unwrap_or(0));
                        let ids_masked =
                            if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                                if let Some(ds) = &b.discrete_scheduler {
                                    if let Some((span_start, span_end)) = response_span {
                                        ds.mask_sequence_span_at_t(
                                            &ids_arr,
                                            mask_token_id,
                                            t,
                                            span_start,
                                            span_end,
                                        )
                                    } else {
                                        ds.mask_sequence_at_t(&ids_arr, mask_token_id, t)
                                    }
                                } else {
                                    ids_arr.clone()
                                }
                            } else {
                                ids_arr.clone()
                            };
                        let x_t_local = match &mut self.network[embeddings_idx.unwrap()] {
                            LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_masked),
                            _ => x0.clone(),
                        };
                        (x_t_local, 1.0, 0.0, true)
                    } else if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                        let x_t_local = b.noise_scheduler.q_sample(&x0, t, &noise);
                        let sa = b.noise_scheduler.sqrt_alpha_cumprod(t);
                        let soa = b.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                        (x_t_local, sa, soa, false)
                    } else {
                        continue;
                    }
                };
                let mut pred = x_t.clone();
                for &idx in &diffusion_blocks_idx {
                    if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                        b.set_causal_attention(true);
                        b.set_timestep(t);
                        pred = b.forward_with_timestep(&pred, t);
                    }
                }
                let x0_hat = if discrete_used {
                    pred.clone()
                } else if let LayerEnum::DiffusionBlock(b0) = &self.network[first_block] {
                    match b0.prediction_target() {
                        crate::domain::layers::diffusion::DiffusionPredictionTarget::Epsilon => {
                            let sa = sqrt_a.max(1e-6);
                            let pred_scaled = &pred * sqrt_one_minus_a;
                            (&x_t - &pred_scaled) / sa
                        }
                        crate::domain::layers::diffusion::DiffusionPredictionTarget::VPrediction => {
                            (&x_t * sqrt_a) - (&pred * sqrt_one_minus_a)
                        }
                        crate::domain::layers::diffusion::DiffusionPredictionTarget::Sample => pred.clone(),
                        crate::domain::layers::diffusion::DiffusionPredictionTarget::EdmX0 => pred.clone(),
                    }
                } else {
                    pred.clone()
                };
                let mut hidden = x0_hat.clone();
                if let Some(nidx) = norm_idx
                    && let LayerEnum::DynamicTanhNorm(norm) = &mut self.network[nidx]
                {
                    hidden = norm.forward(&hidden);
                }
                let logits = if let Some(opidx) = out_proj_idx {
                    if let LayerEnum::OutputProjection(op) = &mut self.network[opidx] {
                        op.forward(&hidden)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };
                let probs = crate::domain::soft::Softmax::new().forward_immutable(&logits.view());
                let target_len = target_ids.len();
                let probs_slice = probs.slice(s![0..target_len, ..]);
                let (denoise_target, w_mse) = if discrete_used {
                    (None, 1.0f32)
                } else if let LayerEnum::DiffusionBlock(b0) = &self.network[first_block] {
                    let mut w = b0.min_snr_weight(t, current_gamma);
                    if b0.prediction_target()
                        == crate::domain::layers::diffusion::DiffusionPredictionTarget::EdmX0
                    {
                        w *= b0.edm_loss_weight(t);
                    }
                    (Some(b0.training_target(&x0, &noise, t)), w)
                } else {
                    (None, 1.0f32)
                };
                let ce = crate::domain::loss::symmetric_cross_entropy(
                    &probs_slice.to_owned(),
                    target_ids,
                    current_ce_weight,
                    current_ce_weight,
                    1e-4,
                );
                let mse = if let Some(target) = denoise_target.as_ref() {
                    crate::domain::loss::epsilon_mse(&pred, target) * w_mse
                } else {
                    0.0
                };
                let lambda_ce = if discrete_used {
                    1.0f32
                } else {
                    lambda_ce_schedule(t)
                };
                val_loss_total += lambda_ce * ce + (1.0 - lambda_ce) * mse;
                val_sce_total += ce;
                val_mse_total += mse;
                val_count += 1;
            }
            }
            let val_loss = if val_count > 0 {
                val_loss_total / val_count as f32
            } else {
                0.0
            };
            let val_sce = if val_count > 0 {
                val_sce_total / val_count as f32
            } else {
                0.0
            };
            let val_mse = if val_count > 0 {
                val_mse_total / val_count as f32
            } else {
                0.0
            };
            info!(
                epoch = epoch,
                loss = avg_loss,
                sce = avg_sce,
                mse = avg_mse,
                lambda_ce = avg_lambda_ce,
                lr = effective_lr,
                grad_norm = grad_norm,
                val_loss = val_loss,
                val_sce = val_sce,
                val_mse = val_mse,
                "Diffusion mixed (CE+MSE) epoch"
            );

            // Update learned DDIM steps after validation is computed.
            if val_loss.is_finite() {
                if let Some(prev) = prev_val_loss
                    && prev.is_finite()
                {
                    let rel_improvement = (prev - val_loss) / prev.max(1e-6);

                    if rel_improvement > 0.01 {
                        steps_plateau_epochs = 0;
                        learned_ddim_steps = ((learned_ddim_steps as f32) * 0.90).round() as usize;
                    } else if rel_improvement < -0.005 {
                        steps_plateau_epochs = steps_plateau_epochs.saturating_add(1);
                        learned_ddim_steps = ((learned_ddim_steps as f32) * 1.15).round() as usize;
                    } else {
                        steps_plateau_epochs = steps_plateau_epochs.saturating_add(1);
                        if steps_plateau_epochs >= 2 {
                            learned_ddim_steps =
                                ((learned_ddim_steps as f32) * 1.05).round() as usize;
                            steps_plateau_epochs = 0;
                        }
                    }
                }
                prev_val_loss = Some(val_loss);

                learned_ddim_steps = learned_ddim_steps
                    .max(1)
                    .min(num_timesteps.max(1))
                    .clamp(ddim_steps_min, ddim_steps_max);

                for &idx in &diffusion_blocks_idx {
                    if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                        b.config.ddim_steps_policy =
                            crate::domain::layers::diffusion::DdimStepsPolicy::Fixed(learned_ddim_steps);
                    }
                }
                info!(
                    epoch = epoch,
                    ddim_steps = learned_ddim_steps,
                    "Updated learned DDIM steps policy"
                );
            }
            if let Some(f) = &mut log_file {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{},{},{},{},{},{},{},{},{},{}",
                    epoch,
                    avg_loss,
                    avg_sce,
                    avg_mse,
                    avg_lambda_ce,
                    effective_lr,
                    grad_norm,
                    val_loss,
                    val_sce,
                    val_mse
                );
            }
            if avg_loss + 1e-5 < best_loss {
                best_loss = avg_loss;
                plateau_epochs = 0;
            } else {
                plateau_epochs += 1;
            }
            if plateau_epochs >= plateau_patience {
                if lr_scale > min_lr_scale {
                    lr_scale = (lr_scale * plateau_reduce).max(min_lr_scale);
                    warn!(
                        epoch = epoch,
                        lr_scale = lr_scale,
                        "Reduce-on-plateau triggered: scaling LR"
                    );
                }
                plateau_epochs = 0;
            }

            if let Some(every) = checkpoint_every
                && every > 0
                && (epoch + 1) % every == 0
            {
                let dir = checkpoint_dir.as_deref().unwrap_or("models");
                std::fs::create_dir_all(dir).map_err(ModelError::from)?;

                let stage = checkpoint_stage.as_deref().unwrap_or("diffusion");
                let checkpoint_path = diffusion_checkpoint_path(
                    std::path::Path::new(dir),
                    &ts,
                    stage,
                    epoch + 1,
                    epochs,
                );
                let checkpoint_path_str = checkpoint_path.to_string_lossy().to_string();
                let description = format!(
                    "Diffusion checkpoint stage={} epoch={}/{}",
                    stage,
                    epoch + 1,
                    epochs
                );
                self.save_versioned(&checkpoint_path_str, Some(description))?;
                info!(
                    epoch = epoch,
                    path = checkpoint_path_str,
                    "Saved diffusion checkpoint"
                );
            }
        }

        Ok(())
    }

    /// Sample from reverse diffusion process for generative decoding
    ///
    /// Starts from pure noise and progressively denoises to generate sequences.
    pub fn sample_diffusion(&mut self, max_length: usize, steps: Option<usize>) -> String {
        self.sample_diffusion_with_prompt("", max_length, steps)
    }

    pub fn sample_diffusion_with_prompt(
        &mut self,
        prompt: &str,
        max_length: usize,
        steps: Option<usize>,
    ) -> String {
        let mut rng = get_rng();

        // Tokenize the prompt if provided
        let prompt_tokens = if !prompt.is_empty() {
            self.tokenize(prompt)
        } else {
            Vec::new()
        };

        // Get embedding dimension from the first layer (TokenEmbeddings)
        let embedding_dim =
            if let Some(LayerEnum::TokenEmbeddings(embeddings)) = self.network.first() {
                embeddings.token_embeddings.ncols()
            } else {
                return "Error: Cannot determine embedding dimension".to_string();
            };

        // Snapshot token embeddings (for prompt conditioning) before borrowing network mutably
        let token_embs_cloned = match self.network.first() {
            Some(LayerEnum::TokenEmbeddings(embeddings)) => {
                Some(embeddings.token_embeddings.clone())
            }
            _ => None,
        };

        // Get diffusion block indices
        let mut diffusion_blocks_idx: Vec<usize> = Vec::new();
        for (i, layer) in self.network.iter().enumerate() {
            if let LayerEnum::DiffusionBlock(_) = layer {
                diffusion_blocks_idx.push(i);
            }
        }

        if diffusion_blocks_idx.is_empty() {
            return "Error: No diffusion blocks found".to_string();
        }

        let (total_timesteps, steps_policy) = match &self.network[diffusion_blocks_idx[0]] {
            LayerEnum::DiffusionBlock(b0) => (
                b0.noise_scheduler.num_timesteps(),
                b0.config.ddim_steps_policy.clone(),
            ),
            _ => return "Error: No diffusion blocks found".to_string(),
        };

        let requested_steps = steps.or(self.diffusion_steps_override);
        let steps = match requested_steps {
            Some(k) => k.max(1).min(total_timesteps.max(1)),
            None => steps_policy.resolve(total_timesteps, max_length, prompt_tokens.len()),
        };

        // Calculate available length for generation (accounting for prompt)
        let _available_length = max_length.saturating_sub(prompt_tokens.len());

        // Start with pure noise: x_T ~ N(0, I), but condition first positions on prompt embeddings
        let mut current_sample = Array2::<f32>::zeros((max_length, embedding_dim));
        for i in 0..max_length {
            for j in 0..embedding_dim {
                current_sample[[i, j]] = rng.random::<f32>() * 2.0 - 1.0;
            }
        }
        if !prompt_tokens.is_empty() {
            // Replace the first K rows with prompt token embeddings
            let k = prompt_tokens.len().min(max_length);
            if let Some(token_embs) = token_embs_cloned {
                for (i, &token_id) in prompt_tokens.iter().take(k).enumerate() {
                    let tid = token_id.min(token_embs.nrows().saturating_sub(1));
                    current_sample.row_mut(i).assign(&token_embs.row(tid));
                }
            }
        }

        // Reverse diffusion process: x_{t-1} = 1/√ᾱ_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t *
        // z
        let is_discrete = diffusion_blocks_idx.iter().any(|&idx| {
            if let LayerEnum::DiffusionBlock(b) = &self.network[idx] {
                b.is_discrete_masked()
            } else {
                false
            }
        });
        if is_discrete {
            let mask_token_id =
                if let LayerEnum::DiffusionBlock(b0) = &self.network[diffusion_blocks_idx[0]] {
                    b0.mask_token_id()
                } else {
                    None
                }
                .or_else(|| self.vocab.encode("<mask>"))
                .unwrap_or(self.vocab.encode_or_unknown("<unk>").unwrap_or(0));
            let mut ids_arr = Array2::<f32>::zeros((1, max_length));
            for i in 0..max_length {
                ids_arr[[0, i]] = mask_token_id as f32;
            }
            for (i, &token_id) in prompt_tokens.iter().take(max_length).enumerate() {
                ids_arr[[0, i]] = token_id as f32;
            }

            for t in (1..=steps).rev() {
                let step_idx = t - 1;
                let t_idx = crate::domain::layers::diffusion::map_step_to_timestep(
                    step_idx,
                    steps,
                    total_timesteps,
                );
                for &idx in &diffusion_blocks_idx {
                    if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                        b.set_timestep(t_idx);
                    }
                }
                let x_t = match &mut self.network[0] {
                    LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                    _ => current_sample.clone(),
                };
                let mut hidden = x_t.clone();
                let mut similarity_ctx: Option<Array2<f32>> = None;
                for &idx in &diffusion_blocks_idx {
                    if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                        b.set_incoming_similarity_context(similarity_ctx.as_ref());
                        hidden = b.forward_with_timestep(&hidden, t_idx);
                        if let Some(existing) = similarity_ctx.as_mut() {
                            existing.assign(b.activation_similarity_matrix());
                        } else {
                            similarity_ctx = Some(b.activation_similarity_matrix().clone());
                        }
                    }
                }
                for layer in &mut self.network {
                    if let LayerEnum::DynamicTanhNorm(norm) = layer {
                        hidden = norm.forward(&hidden);
                    }
                }
                let mut logits: Option<Array2<f32>> = None;
                for layer in &mut self.network {
                    if let LayerEnum::OutputProjection(op) = layer {
                        logits = Some(op.forward(&hidden));
                        break;
                    }
                }
                let logits = match logits {
                    Some(l) => l,
                    None => break,
                };
                let softmax = crate::domain::soft::Softmax::new();
                let probs = softmax.forward_immutable(&logits.view());
                if let LayerEnum::DiffusionBlock(b0) = &self.network[diffusion_blocks_idx[0]]
                    && let Some(ds) = &b0.discrete_scheduler
                {
                    ids_arr = ds.reverse_unmask_step(&ids_arr, &probs, mask_token_id, t_idx, 0.9);
                }
                let mut cur_unmasked = 0usize;
                for i in 0..max_length {
                    if ids_arr[[0, i]] != mask_token_id as f32 {
                        cur_unmasked += 1;
                    }
                }
                if cur_unmasked >= max_length {
                    break;
                }
            }
            current_sample = match &mut self.network[0] {
                LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                _ => current_sample,
            };
        } else {
            for &idx in &diffusion_blocks_idx {
                if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                    b.set_use_ema_for_sampling(true);
                }
            }
            let scheduler_idx = diffusion_blocks_idx[0];
            let mut used_speculative = false;
            if let Some(cfg) = self.speculative_config {
                let draft_len = cfg.draft_layers.min(diffusion_blocks_idx.len());
                if draft_len > 0 {
                    let draft_indices = diffusion_blocks_idx[..draft_len].to_vec();
                    let mut t = steps;
                    used_speculative = true;
                    while t > 0 {
                        let step_idx = t - 1;
                        let t_idx = crate::domain::layers::diffusion::map_step_to_timestep(
                            step_idx,
                            steps,
                            total_timesteps,
                        );
                        let main_pred = self.forward_diffusion_stack(
                            &diffusion_blocks_idx,
                            &current_sample,
                            t_idx,
                        );
                        let draft_pred =
                            self.forward_diffusion_stack(&draft_indices, &current_sample, t_idx);
                        let mse = main_pred
                            .iter()
                            .zip(draft_pred.iter())
                            .map(|(a, b)| {
                                let diff = a - b;
                                diff * diff
                            })
                            .sum::<f32>()
                            / main_pred.len().max(1) as f32;

                        if mse > cfg.tau {
                            current_sample = self.apply_ddim_step(
                                scheduler_idx,
                                &current_sample,
                                t_idx,
                                &main_pred,
                            );
                            t -= 1;
                            continue;
                        }

                        current_sample = self.apply_ddim_step(
                            scheduler_idx,
                            &current_sample,
                            t_idx,
                            &draft_pred,
                        );
                        t -= 1;

                        let mut accepted = 1usize;
                        while accepted < cfg.gamma && t > 0 {
                            let next_step_idx = t - 1;
                            let next_t_idx = crate::domain::layers::diffusion::map_step_to_timestep(
                                next_step_idx,
                                steps,
                                total_timesteps,
                            );
                            let draft_pred = self.forward_diffusion_stack(
                                &draft_indices,
                                &current_sample,
                                next_t_idx,
                            );
                            current_sample = self.apply_ddim_step(
                                scheduler_idx,
                                &current_sample,
                                next_t_idx,
                                &draft_pred,
                            );
                            accepted += 1;
                            t -= 1;
                        }
                    }
                }
            }
            if !used_speculative {
                for t in (1..=steps).rev() {
                    let step_idx = t - 1;
                    let t_idx = crate::domain::layers::diffusion::map_step_to_timestep(
                        step_idx,
                        steps,
                        total_timesteps,
                    );
                    let predicted_noise =
                        self.forward_diffusion_stack(&diffusion_blocks_idx, &current_sample, t_idx);
                    current_sample = self.apply_ddim_step(
                        scheduler_idx,
                        &current_sample,
                        t_idx,
                        &predicted_noise,
                    );
                }
            }
            for &idx in &diffusion_blocks_idx {
                if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                    b.set_use_ema_for_sampling(false);
                }
            }
        }

        // Decode using OutputProjection on the denoised embeddings
        // Pass through final DynamicTanhNorm if present
        let mut hidden = current_sample.clone();
        for layer in &mut self.network {
            if let LayerEnum::DynamicTanhNorm(norm) = layer {
                hidden = norm.forward(&hidden);
            }
        }

        // Find OutputProjection layer and compute logits
        let mut logits: Option<Array2<f32>> = None;
        for layer in &mut self.network {
            if let LayerEnum::OutputProjection(op) = layer {
                logits = Some(op.forward(&hidden));
                break;
            }
        }
        let logits = match logits {
            Some(l) => l,
            None => return "Error: No OutputProjection found".to_string(),
        };

        let mut tokens = prompt_tokens.clone();
        let temperature: f32 = 1.0;
        let top_p: f32 = 0.9;
        let softmax = crate::domain::soft::Softmax::new();
        for i in prompt_tokens.len()..max_length {
            let mut row_scaled = logits.row(i).to_owned();
            if temperature > 0.0 {
                row_scaled.mapv_inplace(|x| x / temperature);
            }
            let row2d = row_scaled.insert_axis(Axis(0));
            let probs_row2d = softmax.forward_immutable(&row2d.view());
            let probs_row = probs_row2d.row(0).to_owned();
            // Nucleus (top-p) sampling
            let mut indexed: Vec<(usize, f32)> = probs_row
                .iter()
                .enumerate()
                .map(|(tid, &p)| (tid, p.max(0.0)))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            let mut cutoff = 0usize;
            for (k, &(_, p)) in indexed.iter().enumerate() {
                cum += p;
                cutoff = k;
                if cum >= top_p {
                    break;
                }
            }
            let nucleus = &indexed[..=cutoff];
            let sum_p: f32 = nucleus.iter().map(|&(_, p)| p).sum();
            let r: f32 = rng.random::<f32>();
            let mut acc = 0.0f32;
            let mut chosen = nucleus[0].0;
            for &(tid, p) in nucleus {
                acc += p / (sum_p.max(1e-8));
                if r <= acc {
                    chosen = tid;
                    break;
                }
            }
            tokens.push(chosen);
            if chosen == 0 {
                break;
            }
        }

        let decoded_text = tokens
            .iter()
            .filter_map(|&token_id| self.vocab.decode(token_id))
            .collect::<Vec<&str>>()
            .join(" ");

        format!("Generated text: {}", decoded_text)
    }

    pub fn evaluate_perplexity_diffusion(&mut self, data: Vec<&str>) -> Result<f32> {
        let tokenized = data
            .par_iter()
            .map(|s| self.tokenize(s))
            .collect::<Vec<Vec<usize>>>();
        let mut total_ce = 0.0f32;
        let mut count = 0usize;
        // Use t=0 path to approximate language modeling
        // Build layer indices once
        let mut diffusion_blocks_idx: Vec<usize> = Vec::new();
        let mut embeddings_idx: Option<usize> = None;
        let mut norm_idx: Option<usize> = None;
        let mut out_proj_idx: Option<usize> = None;
        for (i, layer) in self.network.iter().enumerate() {
            match layer {
                LayerEnum::TokenEmbeddings(_) => {
                    if embeddings_idx.is_none() {
                        embeddings_idx = Some(i)
                    }
                }
                LayerEnum::DiffusionBlock(_) => diffusion_blocks_idx.push(i),
                LayerEnum::DynamicTanhNorm(_) => norm_idx = Some(i),
                LayerEnum::OutputProjection(_) => out_proj_idx = Some(i),
                _ => {}
            }
        }
        if embeddings_idx.is_none() || diffusion_blocks_idx.is_empty() || out_proj_idx.is_none() {
            return Err(ModelError::Training {
                message: String::from("Missing layers for diffusion perplexity eval"),
            });
        }
        for seq in tokenized.iter() {
            if seq.len() < 2 {
                continue;
            }
            let input_ids = &seq[..seq.len() - 1];
            let target_ids = &seq[1..];
            let mut ids_arr = ndarray::Array2::<f32>::zeros((1, input_ids.len()));
            for (i, &tid) in input_ids.iter().enumerate() {
                ids_arr[[0, i]] = tid as f32;
            }
            let x0 = match &mut self.network[embeddings_idx.unwrap()] {
                LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                _ => continue,
            };
            let mut hidden = x0.clone();
            let mut similarity_ctx: Option<Array2<f32>> = None;
            for &idx in &diffusion_blocks_idx {
                if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                    b.set_timestep(0);
                    b.set_incoming_similarity_context(similarity_ctx.as_ref());
                    hidden = b.forward_with_timestep(&hidden, 0);
                    if let Some(existing) = similarity_ctx.as_mut() {
                        existing.assign(b.activation_similarity_matrix());
                    } else {
                        similarity_ctx = Some(b.activation_similarity_matrix().clone());
                    }
                }
            }
            if let Some(nidx) = norm_idx
                && let LayerEnum::DynamicTanhNorm(norm) = &mut self.network[nidx]
            {
                hidden = norm.forward(&hidden);
            }
            let logits = if let Some(opidx) = out_proj_idx {
                if let LayerEnum::OutputProjection(op) = &mut self.network[opidx] {
                    op.forward(&hidden)
                } else {
                    continue;
                }
            } else {
                continue;
            };
            let probs = crate::domain::soft::Softmax::new().forward_immutable(&logits.view());
            let target_len = target_ids.len();
            let probs_slice = probs.slice(s![0..target_len, ..]);
            let ce = crate::domain::loss::symmetric_cross_entropy(
                &probs_slice.to_owned(),
                target_ids,
                1.0,
                1.0,
                1e-4,
            );
            total_ce += ce;
            count += 1;
        }
        if count == 0 {
            return Ok(f32::INFINITY);
        }
        let avg_ce = total_ce / (count as f32);
        let ppl = (avg_ce).exp();
        Ok(ppl)
    }

    pub fn evaluate_bleu(&self, inputs: Vec<&str>, outputs: Vec<&str>) -> Result<(f32, f32)> {
        let refs = inputs
            .iter()
            .map(|s| self.vocab.tokenize(s))
            .collect::<Vec<Vec<usize>>>();
        let cands = outputs
            .iter()
            .map(|s| self.vocab.tokenize(s))
            .collect::<Vec<Vec<usize>>>();
        let (b1, b2) = corpus_bleu_1_2(&refs, &cands);
        Ok((b1, b2))
    }

    /// Load model (auto-detects format from extension: .json or .bin)
    pub fn load(path: &str) -> Result<Self> {
        if path.ends_with(".json") {
            Self::load_json(path)
        } else {
            Self::load_binary(path)
        }
    }
}

fn diffusion_checkpoint_path(
    checkpoint_dir: &std::path::Path,
    run_tag: &str,
    stage: &str,
    epoch_1_based: usize,
    total_epochs: usize,
) -> std::path::PathBuf {
    let safe_stage: String = stage
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    checkpoint_dir.join(format!(
        "rustgpt-{}-{}-epoch{:04}-of{:04}.bin",
        safe_stage, run_tag, epoch_1_based, total_epochs
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_description_includes_decoder() {
        let llm = LLM::default();
        let description = llm.network_description();

        // Should include network layers and decoder type
        assert!(description.contains("OutputProjection"));
        assert!(description.contains("GreedyDecoder"));
        println!("Network description: {}", description);
    }

    #[test]
    fn test_greedy_decoder_creation() {
        let vocab = Vocab::default();
        let network = Vec::new(); // Empty network for testing
        let llm = LLM::with_greedy_decoder(vocab, network);

        match llm.decoder {
            DecoderType::Greedy(_) => {}
        }

        assert_eq!(llm.decoder.layer_type(), "GreedyDecoder");
    }

    #[test]
    fn test_decoder_switching() {
        let mut llm = LLM::default();

        // Should start with GreedyDecoder
        assert_eq!(llm.decoder.layer_type(), "GreedyDecoder");

        // Switch to Greedy (should remain Greedy)
        llm.enable_greedy();
        assert_eq!(llm.decoder.layer_type(), "GreedyDecoder");
    }

    #[test]
    fn test_transformer_speculative_sampling_configuration() {
        let vocab = Vocab::default();
        let network = Vec::new();
        let mut llm = LLM::new(vocab, network);

        // Check initial state
        assert_eq!(llm.speculative_mode, SpeculativeMode::Diffusion);
        assert!(llm.speculative_config.is_none());

        // Enable transformer speculative sampling
        llm.enable_speculative_sampling(4, 0.1, 2, SpeculativeMode::Transformer);

        // Verify configuration
        assert_eq!(llm.speculative_mode, SpeculativeMode::Transformer);
        assert!(llm.speculative_config.is_some());

        let config = llm.speculative_config.as_ref().unwrap();
        assert_eq!(config.gamma, 4);
        assert_eq!(config.tau, 0.1);
        assert_eq!(config.draft_layers, 2);
    }

    #[test]
    fn test_response_span_detection() {
        let vocab = Vocab::new(vec![
            "User",
            "Assistant",
            ":",
            "Hello",
            "World",
            "</s>",
            "<unk>",
            "<mask>",
        ]);
        let tokens = vec![
            vocab.encode("User").unwrap(),
            vocab.encode(":").unwrap(),
            vocab.encode("Hello").unwrap(),
            vocab.encode("Assistant").unwrap(),
            vocab.encode(":").unwrap(),
            vocab.encode("World").unwrap(),
            vocab.encode("</s>").unwrap(),
        ];
        let span = response_span_from_tokens(&vocab, &tokens).expect("span");
        assert_eq!(span, (5, 6));
    }

    #[test]
    fn test_accumulate_layer_gradients_adds_sequences() {
        let mut accumulator = vec![Array2::<f32>::zeros((2, 2))];
        let grads_first = vec![Array2::<f32>::from_elem((2, 2), 1.0)];
        let grads_second = vec![Array2::<f32>::from_elem((2, 2), 2.0)];

        LLM::accumulate_layer_gradients(&mut accumulator, grads_first, "TestLayer");
        LLM::accumulate_layer_gradients(&mut accumulator, grads_second, "TestLayer");

        assert_eq!(accumulator[0], Array2::<f32>::from_elem((2, 2), 3.0));
    }

    #[test]
    fn test_accumulate_layer_gradients_replaces_on_mismatch() {
        let mut accumulator = vec![Array2::<f32>::zeros((2, 2))];
        let mismatched = vec![
            Array2::<f32>::from_elem((2, 2), 1.0),
            Array2::<f32>::from_elem((2, 2), 1.0),
        ];

        LLM::accumulate_layer_gradients(&mut accumulator, mismatched, "TestLayer");

        assert_eq!(accumulator.len(), 2);
        assert!(
            accumulator
                .iter()
                .all(|grad| grad.iter().all(|&v| (v - 1.0).abs() < 1e-6))
        );
    }

    #[test]
    fn test_diffusion_checkpoint_path_format() {
        let p = diffusion_checkpoint_path(
            std::path::Path::new("models"),
            "20260101-000000",
            "pre train",
            3,
            10,
        );
        let fname = p.file_name().unwrap().to_string_lossy();
        assert!(fname.contains("rustgpt-pre_train-20260101-000000-epoch0003-of0010.bin"));
    }
}
#[test]
fn test_ce_loss_normalized() {
    let probs = ndarray::Array2::<f32>::from_elem((4, 8), 1.0 / 8.0);
    let targets = vec![1usize, 2usize, 3usize, 4usize];
    let sce = crate::domain::loss::symmetric_cross_entropy(&probs, &targets, 1.0, 1.0, 1e-4);
    let norm = sce / targets.len() as f32;
    assert!(norm.is_finite());
    assert!(norm > 0.0);
}
