use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramEmbedding {
    pub table: Array2<f32>,
    pub ngram_order: usize,
    pub num_heads: usize,
    pub embedding_dim: usize,
}

impl EngramEmbedding {
    pub fn new(
        num_heads: usize,
        ngram_order: usize,
        embedding_dim: usize,
        table_size: usize,
    ) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let data: Vec<f32> = (0..table_size * embedding_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let table = Array2::from_shape_vec((table_size, embedding_dim), data).unwrap();

        Self {
            table,
            ngram_order,
            num_heads,
            embedding_dim,
        }
    }

    pub fn lookup(&self, hash_idx: usize) -> Array1<f32> {
        self.table.row(hash_idx).to_owned()
    }
}
