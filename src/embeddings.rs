use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{
    Vocab, adam::Adam, model_config::{ModelConfig, TitanMemoryConfig}, network::Layer, rng::get_rng,
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TokenEmbeddings {
    pub token_embeddings: Array2<f32>,
    #[serde(skip, default)]
    pub cached_token_ids: Option<Vec<usize>>,
    #[serde(skip, default)]
    pub cached_input_dim: Option<(usize, usize)>,
    #[serde(default)]
    pub titan_memory: TitanMemoryConfig,
    pub token_optimizer: Adam,
}

impl Default for TokenEmbeddings {
    fn default() -> Self {
        let embedding_dim = ModelConfig::default().embedding_dim;
        Self::new(Vocab::default(), embedding_dim)
    }
}

impl TokenEmbeddings {
    pub fn new(vocab: Vocab, embedding_dim: usize) -> Self {
        Self::new_with_titan_memory(vocab, TitanMemoryConfig::default(), embedding_dim)
    }

    pub fn new_with_titan_memory(
        vocab: Vocab,
        titan_memory: TitanMemoryConfig,
        embedding_dim: usize,
    ) -> Self {
        let vocab_size = vocab.size();
        Self {
            token_embeddings: Self::init_embeddings(vocab_size, embedding_dim),
            cached_token_ids: None,
            cached_input_dim: None,
            titan_memory,
            token_optimizer: Adam::new((vocab_size, embedding_dim)),
        }
    }

    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = get_rng();
        // Proper embedding initialization: std = 1 / sqrt(embedding_dim)
        // Reference: "Attention is All You Need" (Vaswani et al., 2017)
        // This prevents gradient explosion in early layers
        let std = 1.0 / (embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| normal.sample(&mut rng))
    }

    #[inline]
    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            let safe_token_id = token_id.min(embeddings.nrows().saturating_sub(1));
            token_embeds
                .row_mut(i)
                .assign(&embeddings.row(safe_token_id));
        }
        token_embeds
    }

    #[inline]
    pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
        Self::get_token_embeddings(&self.token_embeddings, token_ids)
    }

    #[inline]
    fn token_ids_from_input(input: &Array2<f32>, vocab_size: usize) -> Vec<usize> {
        if vocab_size == 0 {
            return vec![0; input.len()];
        }
        let max_id = vocab_size.saturating_sub(1);
        input
            .iter()
            .map(|&x| {
                if !x.is_finite() || x < 0.0 {
                    0usize
                } else {
                    let raw = if x >= (usize::MAX as f32) {
                        usize::MAX
                    } else {
                        x as usize
                    };
                    raw.min(max_id)
                }
            })
            .collect()
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[inline]
    fn splitmix64_next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn unit_vector_from_seed(seed: u64, out: &mut [f32]) {
        let d = out.len();
        if d == 0 {
            return;
        }
        let mut state = seed;
        let mut sumsq = 0.0f32;
        for v in out.iter_mut() {
            let u = Self::splitmix64_next(&mut state);
            let x = ((u >> 40) as f32) * (1.0 / 16777216.0);
            let x = x.mul_add(2.0, -1.0);
            *v = x;
            sumsq += x * x;
        }
        let ms = sumsq / (d as f32);
        let inv = 1.0 / (ms + 1e-8).sqrt();
        for v in out.iter_mut() {
            *v *= inv;
        }
    }

    #[inline]
    fn ngram_hash(tokens: &[usize], position: usize, ngram_order: usize, head: u64) -> u64 {
        let n = ngram_order.max(1);
        let start = position.saturating_add(1).saturating_sub(n);
        let mut h = 0x6A09E667F3BCC909u64 ^ head.wrapping_mul(0x9E3779B97F4A7C15);
        for &tok in tokens.iter().take(position + 1).skip(start) {
            let x = (tok as u64).wrapping_add(0x9E3779B97F4A7C15);
            h ^= x;
            h = h.wrapping_mul(0xBF58476D1CE4E5B9);
            h = h.rotate_left(31);
        }
        h
    }

    #[inline]
    fn engram_key_for_position(
        tokens: &[usize],
        position: usize,
        ngram_order: usize,
        num_heads: usize,
        key_out: &mut [f32],
        head_buf: &mut [f32],
    ) {
        key_out.fill(0.0);
        if num_heads == 0 || key_out.is_empty() {
            return;
        }
        for h in 0..num_heads {
            let hash = Self::ngram_hash(tokens, position, ngram_order, h as u64);
            Self::unit_vector_from_seed(hash, head_buf);
            for j in 0..key_out.len() {
                key_out[j] += head_buf[j];
            }
        }
        let inv = 1.0 / (num_heads as f32);
        for v in key_out.iter_mut() {
            *v *= inv;
        }
    }

    fn apply_engram_into(&self, token_ids: &[usize], out: &mut Array2<f32>) {
        if !self.titan_memory.enabled || !self.titan_memory.engram_enabled {
            return;
        }
        let n = out.nrows();
        let d = out.ncols();
        if n == 0 || d == 0 {
            return;
        }
        let ngram_order = self.titan_memory.engram_ngram_order.max(1);
        let num_heads = self.titan_memory.engram_num_heads;
        let scale = self.titan_memory.engram_scale;
        if !scale.is_finite() || scale == 0.0 || num_heads == 0 {
            return;
        }

        let sqrt_d = (d as f32).sqrt();
        let eps = 1e-8f32;
        let mut key = vec![0.0f32; d];
        let mut head_buf = vec![0.0f32; d];

        let seq_len = token_ids.len().min(n);
        for t in 0..seq_len {
            Self::engram_key_for_position(
                token_ids,
                t,
                ngram_order,
                num_heads,
                &mut key,
                &mut head_buf,
            );

            let mut dot_xk = 0.0f32;
            let mut sumsq_x = 0.0f32;
            let mut sumsq_k = 0.0f32;
            for j in 0..d {
                let x = out[[t, j]];
                let k = key[j];
                dot_xk += x * k;
                sumsq_x += x * x;
                sumsq_k += k * k;
            }

            let r_x = (sumsq_x / (d as f32) + eps).sqrt();
            let r_k = (sumsq_k / (d as f32) + eps).sqrt();
            let denom = (r_x * r_k * sqrt_d).max(eps);
            let s = dot_xk / denom;
            let gate = Self::sigmoid(s);

            for j in 0..d {
                out[[t, j]] += scale * gate * key[j];
            }
        }
    }
}

impl Layer for TokenEmbeddings {
    fn layer_type(&self) -> &str {
        "TokenEmbeddings"
    }

    #[inline]
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input shape is [1, sequence_length]
        self.cached_input_dim = Some(input.dim());
        self.cached_token_ids = Some(Self::token_ids_from_input(
            input,
            self.token_embeddings.nrows(),
        ));
        let token_ids = self.cached_token_ids.as_deref().unwrap_or(&[]);
        let mut out = self.embed_tokens(token_ids); // shape is [sequence_length, embedding_dim]
        self.apply_engram_into(token_ids, &mut out);
        out
    }

    #[inline]
    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let token_ids = if input.is_empty() {
            self.cached_token_ids.as_ref().cloned().unwrap_or_default()
        } else {
            Self::token_ids_from_input(input, self.token_embeddings.nrows())
        };
        let grads = output_grads.view(); // (sequence_length, embedding_dim)

        // Initialize gradients for token embeddings
        let mut token_grads = Array2::zeros(self.token_embeddings.dim());

        if grads.nrows() != token_ids.len() {
            tracing::warn!(
                layer = "TokenEmbeddings",
                token_ids = token_ids.len(),
                grad_rows = grads.nrows(),
                "Sequence length mismatch between token ids and output gradients; clamping"
            );
        }

        let seq_len = token_ids.len().min(grads.nrows());
        let engram_enabled = self.titan_memory.enabled && self.titan_memory.engram_enabled;
        let ngram_order = self.titan_memory.engram_ngram_order.max(1);
        let num_heads = self.titan_memory.engram_num_heads;
        let scale = self.titan_memory.engram_scale;
        let d = self.token_embeddings.ncols();
        let sqrt_d = (d as f32).sqrt();
        let eps = 1e-8f32;
        let mut key = vec![0.0f32; d];
        let mut head_buf = vec![0.0f32; d];

        for (i, &token_id) in token_ids.iter().enumerate().take(seq_len) {
            let safe_token_id = token_id.min(self.token_embeddings.nrows().saturating_sub(1));
            let x_row = self.token_embeddings.row(safe_token_id);
            let g_row = grads.row(i);

            if engram_enabled && num_heads != 0 && scale.is_finite() && scale != 0.0 && d != 0 {
                Self::engram_key_for_position(
                    &token_ids,
                    i,
                    ngram_order,
                    num_heads,
                    &mut key,
                    &mut head_buf,
                );

                let mut dot_xk = 0.0f32;
                let mut sumsq_x = 0.0f32;
                let mut sumsq_k = 0.0f32;
                let mut dot_gk = 0.0f32;
                for j in 0..d {
                    let x = x_row[j];
                    let k = key[j];
                    let g = g_row[j];
                    dot_xk += x * k;
                    sumsq_x += x * x;
                    sumsq_k += k * k;
                    dot_gk += g * k;
                }

                let r_x = (sumsq_x / (d as f32) + eps).sqrt();
                let r_k = (sumsq_k / (d as f32) + eps).sqrt();
                let denom = (r_x * r_k * sqrt_d).max(eps);
                let s = dot_xk / denom;
                let gate = Self::sigmoid(s);
                let gate_prime = gate * (1.0 - gate);
                let c = 1.0 / ((r_k * sqrt_d).max(eps));
                let inv_r_x = 1.0 / r_x.max(eps);
                let inv_r_x3 = inv_r_x * inv_r_x * inv_r_x;
                let coeff = scale * dot_gk * gate_prime * c;
                let corr = dot_xk * (d as f32).recip() * inv_r_x3;

                for j in 0..d {
                    let x = x_row[j];
                    let k = key[j];
                    let ds_dx = k * inv_r_x - corr * x;
                    token_grads[[safe_token_id, j]] += g_row[j] + coeff * ds_dx;
                }
            } else {
                for j in 0..d {
                    token_grads[[safe_token_id, j]] += g_row[j];
                }
            }
        }

        // Gradients do not propagate into discrete token ids; return zeros with input shape.
        let input_shape = if !input.is_empty() {
            input.dim()
        } else {
            self.cached_input_dim.unwrap_or((1, token_ids.len()))
        };
        let input_grads = Array2::<f32>::zeros(input_shape);
        (input_grads, vec![token_grads])
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        if param_grads.len() != 1 {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "TokenEmbeddings expected 1 parameter gradient, got {}",
                    param_grads.len()
                ),
            });
        }
        let mut grad = param_grads[0].clone();
        grad.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
        let gnorm: f32 = grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let wnorm = self.weight_norm().max(1e-6);
        let clip = 5.0f32;
        let mut scale = (wnorm / gnorm.max(1e-6)).clamp(0.5, 2.0);
        if gnorm.is_finite() && gnorm > clip && gnorm > 0.0 {
            scale *= clip / gnorm;
        }
        grad.mapv_inplace(|x| x * scale);
        self.token_optimizer
            .step(&mut self.token_embeddings, &grad, lr);
        Ok(())
    }

    fn zero_gradients(&mut self) {
        // TokenEmbeddings doesn't maintain internal gradients state
        // Gradients are computed on-demand in compute_gradients
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Unwrap is safe here: backward is only called from training loop which validates inputs
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.token_embeddings.len()
    }

    fn weight_norm(&self) -> f32 {
        let sumsq = self.token_embeddings.iter().map(|&w| w * w).sum::<f32>();
        sumsq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::rng::set_seed;

    fn make_token_id_input(ids: &[usize]) -> Array2<f32> {
        let mut input = Array2::<f32>::zeros((1, ids.len()));
        for (i, &id) in ids.iter().enumerate() {
            input[[0, i]] = id as f32;
        }
        input
    }

    #[test]
    fn test_engram_disabled_matches_plain_embeddings() {
        set_seed(123);
        let vocab = Vocab::default();
        let cfg = TitanMemoryConfig {
            engram_enabled: false,
            ..Default::default()
        };
        let embedding_dim = ModelConfig::default().embedding_dim;
        let mut emb = TokenEmbeddings::new_with_titan_memory(vocab, cfg, embedding_dim);

        let ids = vec![0usize, 1, 2, 3, 4, 5];
        let input = make_token_id_input(&ids);
        let out = emb.forward(&input);
        let plain = emb.embed_tokens(&ids);

        assert_eq!(out.dim(), plain.dim());
        assert!(out.iter().all(|v| v.is_finite()));
        for (a, b) in out.iter().zip(plain.iter()) {
            assert!((*a - *b).abs() <= 1e-6);
        }
    }

    #[test]
    fn test_engram_embedding_gradient_matches_finite_difference() {
        set_seed(7);
        let vocab = Vocab::default();
        let cfg = TitanMemoryConfig {
            engram_enabled: true,
            engram_scale: 0.2,
            engram_ngram_order: 3,
            engram_num_heads: 3,
            ..Default::default()
        };
        let embedding_dim = ModelConfig::default().embedding_dim;
        let mut emb = TokenEmbeddings::new_with_titan_memory(vocab, cfg, embedding_dim);

        let ids = vec![1usize, 2, 3, 1];
        let input = make_token_id_input(&ids);
        let out = emb.forward(&input);

        let mut upstream = Array2::<f32>::zeros(out.dim());
        for (i, v) in upstream.iter_mut().enumerate() {
            *v = ((i as f32) * 0.01).sin();
        }

        let (_in_grads, grads) = emb.compute_gradients(&input, &upstream);
        let table_grads = &grads[0];

        let token_id = ids[0].min(emb.token_embeddings.nrows().saturating_sub(1));
        let dim = 0usize.min(emb.token_embeddings.ncols().saturating_sub(1));
        let analytic = table_grads[[token_id, dim]];

        let eps = 1e-3f32;

        let mut emb_p = emb.clone();
        emb_p.token_embeddings[[token_id, dim]] += eps;
        let out_p = emb_p.forward(&input);
        let loss_p: f32 = out_p.iter().zip(upstream.iter()).map(|(a, b)| a * b).sum();

        let mut emb_m = emb.clone();
        emb_m.token_embeddings[[token_id, dim]] -= eps;
        let out_m = emb_m.forward(&input);
        let loss_m: f32 = out_m.iter().zip(upstream.iter()).map(|(a, b)| a * b).sum();

        let numeric = (loss_p - loss_m) / (2.0 * eps);
        let denom = analytic.abs().max(numeric.abs()).max(1e-4);
        let rel = (analytic - numeric).abs() / denom;
        assert!(rel < 2e-2);
    }
}
