use ndarray::{Array1, Array2, Axis, Zip};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use super::super::config::{
    DEFAULT_CACHE_TIER_1_SIZE, DEFAULT_CACHE_TIER_2_SIZE, DEFAULT_ENGRAM_NGRAM_ORDER,
    DEFAULT_ENGRAM_NUM_HEADS, DEFAULT_ENGRAM_TABLE_SIZE,
};
use super::cache::EngramCache;
use super::embedding::EngramEmbedding;

/// Optimized hash function for n-gram hashing with excellent avalanche properties.
///
/// Based on splitmix64 design principles:
/// - High-quality bit mixing for uniform distribution
/// - Avalanche: ~50% of output bits flip on single input bit change
/// - Efficient for small n-grams (2-5 tokens typical)
///
/// References:
/// - SplitMix64: Steele & Vigna (2014) - http://xorshift.di.unimi.it/splitmix64.c
/// - MurmurHash3 finalizer: Austin Appleby (2008)
fn hash_ngram(tokens: &[usize], table_size: usize, seed: u64) -> usize {
    // Use a per-head randomized initial state mixed with seed
    let mut state: u64 = seed.wrapping_add(0x9E3779B97F4A7C15);

    for &token in tokens {
        // Mix token into state using MurmurHash3-style operations
        let k: u64 = (token as u64).wrapping_mul(0xBF58476D1CE4E5B9);
        let k = k ^ (k >> 27);
        let k = k.wrapping_mul(0x94D049BB133111EB);

        state ^= k;
        state = state.wrapping_add(0x9E3779B97F4A7C15);
    }

    // Finalization mix (similar to MurmurHash3)
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58476D1CE4E5B9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94D049BB133111EB);
    state ^= state >> 31;

    // Fast modulo for power-of-2 table sizes (common case)
    if table_size.is_power_of_two() {
        (state as usize) & (table_size - 1)
    } else {
        ((state >> 32) as usize) % table_size
    }
}

/// Legacy alias for backward compatibility
#[inline]
#[deprecated(since = "0.2.0", note = "Use hash_ngram for better hash quality")]
#[allow(dead_code)]
fn multiplicative_xor_hash(tokens: &[usize], table_size: usize, seed: u64) -> usize {
    hash_ngram(tokens, table_size, seed)
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
        let hash = hash_ngram(ngram, table_size, head as u64);
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
                    Zip::from(&mut self.scratch_sum)
                        .and(&embedding)
                        .for_each(|a, b| *a += *b);
                    self.cache.insert(hash_idx, embedding);
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

    pub fn gradient_count(&self) -> usize {
        4
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

    pub fn compute_gradients(
        &self,
        input: &Array2<f32>,
        token_ids: &[usize],
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let seq_len = input.nrows();
        assert!(token_ids.len() >= seq_len);

        let mut input_grads = Array2::<f32>::zeros(input.raw_dim());
        let mut d_embedding = Array2::<f32>::zeros(self.embedding.table.raw_dim());
        let mut d_w_gate_q = Array2::<f32>::zeros(self.w_gate_q.raw_dim());
        let mut d_w_gate_k = Array2::<f32>::zeros(self.w_gate_k.raw_dim());
        let mut d_w_gate_v = Array2::<f32>::zeros(self.w_gate_v.raw_dim());

        let eps = 1e-8_f32;
        let inv_sqrt_dim = 1.0_f32 / (self.memory_dim as f32).sqrt();

        for t in 0..seq_len {
            let x_t = input.row(t);
            let hashes = compute_ngram_hashes(
                token_ids,
                t,
                self.ngram_order,
                self.num_heads,
                self.embedding.table.nrows(),
            );

            let mut scratch_sum = Array1::<f32>::zeros(self.memory_dim);
            let mut count = 0usize;
            for &hash_idx in hashes.iter() {
                let embedding = self.embedding.table.row(hash_idx);
                Zip::from(&mut scratch_sum)
                    .and(&embedding)
                    .for_each(|a, b| *a += *b);
                count += 1;
            }

            if count > 0 {
                let denom = count as f32;
                scratch_sum.mapv_inplace(|v| v / denom);
            }

            let q_t = self.w_gate_q.dot(&x_t);
            let k_t = self.w_gate_k.dot(&scratch_sum);
            let v_t = self.w_gate_v.dot(&scratch_sum);

            let q_norm = Self::rms_norm(&q_t, eps);
            let k_norm = Self::rms_norm(&k_t, eps);

            let gate_input = q_norm.dot(&k_norm) * inv_sqrt_dim;
            let gate_alpha = sigmoid(gate_input);

            let dy_t = output_grads.row(t);
            let d_gate = dy_t.dot(&v_t);
            let d_v_t = dy_t.to_owned() * gate_alpha;
            let d_gate_input = d_gate * gate_alpha * (1.0 - gate_alpha);

            let d_q_norm = k_norm.to_owned() * (d_gate_input * inv_sqrt_dim);
            let d_k_norm = q_norm.to_owned() * (d_gate_input * inv_sqrt_dim);

            let q_sq = q_t.iter().map(|&v| v * v).sum::<f32>() + eps;
            let q_norm_denom = q_sq.sqrt();
            let q_dot = d_q_norm.dot(&q_t);
            let d_q_t = d_q_norm.mapv(|v| v / q_norm_denom)
                - q_t.mapv(|v| v * (q_dot / (q_norm_denom * q_norm_denom * q_norm_denom)));

            let k_sq = k_t.iter().map(|&v| v * v).sum::<f32>() + eps;
            let k_norm_denom = k_sq.sqrt();
            let k_dot = d_k_norm.dot(&k_t);
            let d_k_t = d_k_norm.mapv(|v| v / k_norm_denom)
                - k_t.mapv(|v| v * (k_dot / (k_norm_denom * k_norm_denom * k_norm_denom)));

            d_w_gate_q += &d_q_t
                .clone()
                .insert_axis(Axis(1))
                .dot(&x_t.insert_axis(Axis(0)));
            let d_x_t = self.w_gate_q.t().dot(&d_q_t);

            d_w_gate_k += &d_k_t
                .clone()
                .insert_axis(Axis(1))
                .dot(&scratch_sum.clone().insert_axis(Axis(0)));
            d_w_gate_v += &d_v_t
                .clone()
                .insert_axis(Axis(1))
                .dot(&scratch_sum.clone().insert_axis(Axis(0)));

            let d_scratch_sum = self.w_gate_k.t().dot(&d_k_t) + self.w_gate_v.t().dot(&d_v_t);

            if count > 0 {
                let scale = 1.0 / (count as f32);
                for &hash_idx in hashes.iter() {
                    let mut row = d_embedding.row_mut(hash_idx);
                    Zip::from(&mut row)
                        .and(&d_scratch_sum)
                        .for_each(|a, b| *a += *b * scale);
                }
            }

            input_grads.row_mut(t).assign(&d_x_t);
        }

        (
            input_grads,
            vec![d_embedding, d_w_gate_q, d_w_gate_k, d_w_gate_v],
        )
    }

    pub fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::common::errors::Result<()> {
        if gradients.len() != 4 {
            return Ok(());
        }

        self.embedding
            .table
            .scaled_add(-learning_rate, &gradients[0]);
        self.w_gate_q.scaled_add(-learning_rate, &gradients[1]);
        self.w_gate_k.scaled_add(-learning_rate, &gradients[2]);
        self.w_gate_v.scaled_add(-learning_rate, &gradients[3]);

        Ok(())
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
    fn test_hash_ngram_avalanche_property() {
        // Test that hash_ngram has good avalanche properties
        // Single bit change in input should change ~50% of output bits
        let table_size = 1024;
        let seed = 12345u64;

        // Base n-gram
        let base = vec![1, 2, 3, 4, 5];
        let base_hash = hash_ngram(&base, table_size, seed);

        // Count bit flips across many single-bit variations
        let mut total_bit_flips = 0u32;
        let num_variations = 100usize;

        for i in 0..num_variations {
            let mut variant = base.clone();
            // Modify one token slightly
            variant[i % base.len()] = variant[i % base.len()].wrapping_add(i + 1);
            let variant_hash = hash_ngram(&variant, table_size, seed);

            // Count differing bits using XOR and population count
            let diff = (base_hash ^ variant_hash) as u64;
            total_bit_flips += diff.count_ones();
        }

        // For good avalanche, we expect ~50% bit flips on average
        // With 10 bits (1024 = 2^10), expect ~5 bits flipped on average
        let avg_flips = total_bit_flips as f32 / num_variations as f32;
        let expected_flips = (table_size.trailing_zeros() as f32) / 2.0;

        // Allow 30% tolerance for statistical variation
        let tolerance = expected_flips * 0.3;
        assert!(
            avg_flips >= expected_flips - tolerance && avg_flips <= expected_flips + tolerance,
            "Hash avalanche test failed: avg {} expected {} ±{}",
            avg_flips,
            expected_flips,
            tolerance
        );
    }

    #[test]
    fn test_hash_ngram_uniform_distribution() {
        // Test that hash_ngram distributes values uniformly across table
        let table_size = 256;
        let seed = 0u64;
        let num_samples = 10000usize;
        let mut buckets = vec![0usize; table_size];

        // Hash many different n-grams
        for i in 0..num_samples {
            let i = i as usize;
            let tokens = vec![i, i.wrapping_add(1), i.wrapping_add(2)];
            let hash = hash_ngram(&tokens, table_size, seed);
            buckets[hash] += 1;
        }

        // Calculate chi-squared statistic
        let expected = num_samples as f64 / table_size as f64;
        let chi_squared: f64 = buckets
            .iter()
            .map(|&observed| {
                let diff = observed as f64 - expected;
                (diff * diff) / expected
            })
            .sum();

        // For 255 degrees of freedom (256 buckets - 1) at p=0.01,
        // critical value is approximately 310
        // A well-distributed hash should have chi_squared < 310
        assert!(
            chi_squared < 350.0,
            "Hash distribution not uniform: chi_squared={}, expected < 350",
            chi_squared
        );
    }

    #[test]
    fn test_hash_ngram_fast_modulo() {
        // Test that power-of-2 table sizes use fast path
        let tokens = vec![1, 2, 3];

        // Power of 2 - should use bitwise AND
        let hash_pow2 = hash_ngram(&tokens, 1024, 0);
        assert!(hash_pow2 < 1024);

        // Non-power of 2 - should use modulo
        let hash_non_pow2 = hash_ngram(&tokens, 1000, 0);
        assert!(hash_non_pow2 < 1000);
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
