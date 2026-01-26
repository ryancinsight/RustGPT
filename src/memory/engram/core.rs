use ndarray::{Array1, Array2, Zip};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use super::super::config::{
    DEFAULT_CACHE_TIER_1_SIZE, DEFAULT_CACHE_TIER_2_SIZE, DEFAULT_ENGRAM_NGRAM_ORDER,
    DEFAULT_ENGRAM_NUM_HEADS, DEFAULT_ENGRAM_TABLE_SIZE,
};
use super::cache::EngramCache;
use super::embedding::EngramEmbedding;

fn multiplicative_xor_hash(tokens: &[usize], table_size: usize, seed: u64) -> usize {
    let mut hash: u64 = seed;
    for &token in tokens {
        hash = hash.wrapping_mul(0x5DEECE66D).wrapping_add(token as u64);
    }
    ((hash >> 32) as usize) % table_size
}

fn compute_ngram_hashes(
    tokens: &[usize],
    position: usize,
    ngram_order: usize,
    num_heads: usize,
    table_size: usize,
) -> Vec<usize> {
    assert!(table_size > 0);
    let end = (position + 1).min(tokens.len());
    let n = ngram_order.max(1);
    let start = end.saturating_sub(n);
    let ngram = &tokens[start..end];

    let mut hashes = Vec::with_capacity(num_heads);
    for head in 0..num_heads {
        let hash = multiplicative_xor_hash(ngram, table_size, head as u64);
        hashes.push(hash);
    }
    hashes
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramMemory {
    pub embedding: EngramEmbedding,
    pub cache: EngramCache,
    pub w_gate_q: Array2<f32>,
    pub w_gate_k: Array2<f32>,
    pub w_gate_v: Array2<f32>,
    pub ngram_order: usize,
    pub num_heads: usize,
    pub memory_dim: usize,
    pub input_dim: usize,
    #[serde(skip)]
    scratch_sum: Array1<f32>,
}

impl EngramMemory {
    pub fn new(input_dim: usize, memory_dim: usize) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let w_gate_q_data: Vec<f32> = (0..memory_dim * input_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let w_gate_q = Array2::from_shape_vec((memory_dim, input_dim), w_gate_q_data).unwrap();

        let w_gate_k_data: Vec<f32> = (0..memory_dim * memory_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let w_gate_k = Array2::from_shape_vec((memory_dim, memory_dim), w_gate_k_data).unwrap();

        let w_gate_v_data: Vec<f32> = (0..memory_dim * memory_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let w_gate_v = Array2::from_shape_vec((memory_dim, memory_dim), w_gate_v_data).unwrap();

        Self {
            embedding: EngramEmbedding::new(
                DEFAULT_ENGRAM_NUM_HEADS,
                DEFAULT_ENGRAM_NGRAM_ORDER,
                memory_dim,
                DEFAULT_ENGRAM_TABLE_SIZE,
            ),
            cache: EngramCache::new(DEFAULT_CACHE_TIER_1_SIZE, DEFAULT_CACHE_TIER_2_SIZE),
            w_gate_q,
            w_gate_k,
            w_gate_v,
            ngram_order: DEFAULT_ENGRAM_NGRAM_ORDER,
            num_heads: DEFAULT_ENGRAM_NUM_HEADS,
            memory_dim,
            input_dim,
            scratch_sum: Array1::zeros(memory_dim),
        }
    }

    fn rms_norm(x: &Array1<f32>, eps: f32) -> Array1<f32> {
        let sq_norm = x.iter().map(|&v| v * v).sum::<f32>() + eps;
        let norm = sq_norm.sqrt();
        x.mapv(|v| v / norm)
    }

    pub fn forward(&mut self, input: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let seq_len = input.nrows();
        assert!(token_ids.len() >= seq_len);
        let mut output = Array2::<f32>::zeros((seq_len, self.memory_dim));

        if self.scratch_sum.len() != self.memory_dim {
            self.scratch_sum = Array1::zeros(self.memory_dim);
        }

        for t in 0..seq_len {
            let x_t = input.row(t);

            let hashes = compute_ngram_hashes(
                token_ids,
                t,
                self.ngram_order,
                self.num_heads,
                self.embedding.table.nrows(),
            );

            self.scratch_sum.fill(0.0);
            let mut count = 0usize;
            for &hash_idx in hashes.iter() {
                if let Some(cached) = self.cache.get(hash_idx) {
                    Zip::from(&mut self.scratch_sum)
                        .and(cached)
                        .for_each(|a, b| *a += *b);
                } else {
                    let embedding = self.embedding.lookup(hash_idx);
                    self.cache.insert(hash_idx, embedding.clone());
                    Zip::from(&mut self.scratch_sum)
                        .and(&embedding)
                        .for_each(|a, b| *a += *b);
                }
                count += 1;
            }

            if count > 0 {
                let denom = count as f32;
                self.scratch_sum.mapv_inplace(|v| v / denom);
            }

            let q_t = self.w_gate_q.dot(&x_t);
            let k_t = self.w_gate_k.dot(&self.scratch_sum);
            let v_t = self.w_gate_v.dot(&self.scratch_sum);

            let q_norm = Self::rms_norm(&q_t, 1e-8);
            let k_norm = Self::rms_norm(&k_t, 1e-8);

            let gate_alpha = sigmoid(q_norm.dot(&k_norm) / (self.memory_dim as f32).sqrt());

            let gated_memory = v_t.mapv(|v| v * gate_alpha);

            output.row_mut(t).assign(&gated_memory);
        }

        output
    }

    pub fn parameters(&self) -> usize {
        let embedding_params = self.embedding.table.len();
        let gate_params = self.w_gate_q.len() + self.w_gate_k.len() + self.w_gate_v.len();
        embedding_params + gate_params
    }

    pub fn cache_stats(&self) -> (f32, f32) {
        self.cache.hit_rate()
    }

    pub fn weight_norm(&self) -> f32 {
        let mut sum_sq = 0.0;
        sum_sq += self.embedding.table.mapv(|x| x * x).sum();
        sum_sq += self.w_gate_q.mapv(|x| x * x).sum();
        sum_sq += self.w_gate_k.mapv(|x| x * x).sum();
        sum_sq += self.w_gate_v.mapv(|x| x * x).sum();
        sum_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_engram_hash_collision_resilience() {
        let mut memory = EngramMemory::new(128, 128);

        let dummy_tokens = [1usize, 2, 3, 4, 5].repeat(100);
        let dummy_input = Array2::zeros((5, 128));

        let _output = memory.forward(&dummy_input, &dummy_tokens);

        assert_eq!(memory.embedding.num_heads, DEFAULT_ENGRAM_NUM_HEADS);
    }

    #[test]
    fn test_engram_cache_hit_rates() {
        let mut cache = EngramCache::new(100, 1000);

        let embedding = Array1::zeros(128);
        cache.insert_raw(42, embedding.clone());
        cache.insert_raw(43, embedding.clone());

        assert!(cache.get(42).is_some());
        assert!(cache.get(9999).is_none());
    }

    #[test]
    fn test_engram_dimensions() {
        let mut memory = EngramMemory::new(256, 256);

        let seq_len = 10;
        let input = Array2::zeros((seq_len, 256));
        let dummy_tokens = vec![0; 32];

        let output = memory.forward(&input, &dummy_tokens);

        assert_eq!(output.shape(), &[seq_len, 256]);
    }
}
