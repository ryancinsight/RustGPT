use ndarray::{Array2, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;
use crate::{EMBEDDING_DIM, MAX_SEQ_LEN, Vocab};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Embeddings {
    pub token_embeddings: Array2<f32>,
    pub positional_embeddings: Array2<f32>,
    pub cached_input: Option<Array2<f32>>,
    pub token_optimizer: Adam,
    pub positional_optimizer: Adam,
    pub use_positional: bool,
}

impl Default for Embeddings {
    fn default() -> Self {
        Self {
            token_embeddings: Self::init_embeddings(Vocab::default_words().len(), EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((Vocab::default_words().len(), EMBEDDING_DIM)),
            positional_optimizer: Adam::new((MAX_SEQ_LEN, EMBEDDING_DIM)),
            use_positional: true,
        }
    }
}

impl Embeddings {
    pub fn new(vocab: Vocab) -> Self {
        Self::new_with_positional(vocab, true)
    }

    pub fn new_with_positional(vocab: Vocab, use_positional: bool) -> Self {
        Self {
            token_embeddings: Self::init_embeddings(vocab.size(), EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((vocab.size(), EMBEDDING_DIM)),
            positional_optimizer: Adam::new((MAX_SEQ_LEN, EMBEDDING_DIM)),
            use_positional,
        }
    }

    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        // Proper embedding initialization: std = 1 / sqrt(embedding_dim)
        // Reference: "Attention is All You Need" (Vaswani et al., 2017)
        // This prevents gradient explosion in early layers
        let std = 1.0 / (embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| normal.sample(&mut rng))
    }

    fn init_positional_embeddings(max_seq_len: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        // Positional embeddings use same initialization as token embeddings
        // Reference: "Attention is All You Need" (Vaswani et al., 2017)
        let std = 1.0 / (embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        Array2::from_shape_fn((max_seq_len, embedding_dim), |_| normal.sample(&mut rng))
    }

    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            // Defensive check: clamp token_id to valid range to prevent panic
            // Invalid tokens are mapped to token 0 (typically <UNK> or <PAD>)
            let safe_token_id = if token_id >= embeddings.nrows() {
                tracing::warn!(
                    token_id = token_id,
                    vocab_size = embeddings.nrows(),
                    "Token ID out of bounds, clamping to 0"
                );
                0
            } else {
                token_id
            };
            token_embeds
                .row_mut(i)
                .assign(&embeddings.row(safe_token_id));
        }
        token_embeds
    }

    fn get_positional_embeddings(
        positional_encodings: &Array2<f32>,
        seq_len: usize,
    ) -> Array2<f32> {
        // Defensive check: clamp seq_len to max_seq_len to prevent panic
        let safe_seq_len = if seq_len > positional_encodings.nrows() {
            tracing::warn!(
                seq_len = seq_len,
                max_seq_len = positional_encodings.nrows(),
                "Sequence length exceeds maximum, clamping"
            );
            positional_encodings.nrows()
        } else {
            seq_len
        };
        positional_encodings
            .slice(s![0..safe_seq_len, ..])
            .to_owned()
    }

    pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
        let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);
        if self.use_positional {
            let position_embeds =
                Self::get_positional_embeddings(&self.positional_embeddings, token_ids.len());
            token_embeds + position_embeds // Element-wise sum
        } else {
            token_embeds
        }
    }
}

impl Layer for Embeddings {
    fn layer_type(&self) -> &str {
        "Embeddings"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input shape is [1, sequence_length]
        self.cached_input = Some(input.clone());
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        self.embed_tokens(&token_ids) // shape is [sequence_length, embedding_dim]
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self.cached_input.as_ref().unwrap();
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let grads = output_grads.view(); // (sequence_length, embedding_dim)

        // Initialize gradients for embeddings
        let mut token_grads = Array2::zeros(self.token_embeddings.dim());
        let mut positional_grads = Array2::zeros(self.positional_embeddings.dim());

        for (i, &token_id) in token_ids.iter().enumerate() {
            // Defensive check: clamp token_id to valid range
            let safe_token_id = if token_id >= self.token_embeddings.nrows() {
                tracing::warn!(
                    token_id = token_id,
                    vocab_size = self.token_embeddings.nrows(),
                    "Token ID out of bounds in gradient computation, clamping to 0"
                );
                0
            } else {
                token_id
            };
            let grad_row = grads.row(i);

            // Accumulate token embedding gradients efficiently (no temp variable)
            {
                let mut token_row = token_grads.row_mut(safe_token_id);
                token_row += &grad_row;
            }

            if self.use_positional {
                // Accumulate positional embedding gradients efficiently (no temp variable)
                let mut pos_row = positional_grads.row_mut(i);
                pos_row += &grad_row;
            }
        }

        if self.use_positional {
            (output_grads.clone(), vec![token_grads, positional_grads])
        } else {
            (output_grads.clone(), vec![token_grads])
        }
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        if self.use_positional {
            if param_grads.len() != 2 {
                return Err(crate::errors::ModelError::GradientError {
                    message: format!(
                        "Embeddings expected 2 parameter gradients (token + positional), got {}",
                        param_grads.len()
                    ),
                });
            }

            self.token_optimizer
                .step(&mut self.token_embeddings, &param_grads[0], lr);
            self.positional_optimizer
                .step(&mut self.positional_embeddings, &param_grads[1], lr);
        } else {
            if param_grads.len() != 1 {
                return Err(crate::errors::ModelError::GradientError {
                    message: format!(
                        "Embeddings (no-positional) expected 1 parameter gradient (token), got {}",
                        param_grads.len()
                    ),
                });
            }
            self.token_optimizer
                .step(&mut self.token_embeddings, &param_grads[0], lr);
        }
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Unwrap is safe here: backward is only called from training loop which validates inputs
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        if self.use_positional {
            self.token_embeddings.len() + self.positional_embeddings.len()
        } else {
            self.token_embeddings.len()
        }
    }
}
