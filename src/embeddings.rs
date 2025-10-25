use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{EMBEDDING_DIM, Vocab, adam::Adam, llm::Layer};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TokenEmbeddings {
    pub token_embeddings: Array2<f32>,
    pub cached_input: Option<Array2<f32>>,
    pub token_optimizer: Adam,
}

impl Default for TokenEmbeddings {
    fn default() -> Self {
        Self::new(Vocab::default())
    }
}

impl TokenEmbeddings {
    pub fn new(vocab: Vocab) -> Self {
        Self {
            token_embeddings: Self::init_embeddings(vocab.size(), EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((vocab.size(), EMBEDDING_DIM)),
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



    #[inline]
    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            let safe_token_id = token_id.min(embeddings.nrows().saturating_sub(1));
            token_embeds.row_mut(i).assign(&embeddings.row(safe_token_id));
        }
        token_embeds
    }



    #[inline]
    pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
        Self::get_token_embeddings(&self.token_embeddings, token_ids)
    }
}

impl Layer for TokenEmbeddings {
    fn layer_type(&self) -> &str {
        "TokenEmbeddings"
    }

    #[inline]
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input shape is [1, sequence_length]
        self.cached_input = Some(input.clone());
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        self.embed_tokens(&token_ids) // shape is [sequence_length, embedding_dim]
    }

    #[inline]
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self.cached_input.as_ref().unwrap();
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let grads = output_grads.view(); // (sequence_length, embedding_dim)

        // Initialize gradients for token embeddings
        let mut token_grads = Array2::zeros(self.token_embeddings.dim());

        for (i, &token_id) in token_ids.iter().enumerate() {
            // Clamp token_id to valid range
            let safe_token_id = token_id.min(self.token_embeddings.nrows().saturating_sub(1));
            let grad_row = grads.row(i);

            // Accumulate gradients
            token_grads.row_mut(safe_token_id).zip_mut_with(&grad_row, |a, &b| *a += b);
        }

        (output_grads.clone(), vec![token_grads])
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
        self.token_optimizer
            .step(&mut self.token_embeddings, &param_grads[0], lr);
        Ok(())
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
}
