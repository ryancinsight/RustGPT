use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use crate::common::rng::get_rng;

fn mix64(seed: u64, idx: u64) -> u64 {
    let mut z = seed ^ idx;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn mix64_with_t(seed: u64, t: u64, idx: u64) -> u64 {
    mix64(seed ^ t, idx)
}

/// Discrete masked diffusion scheduler with absorbing-state [MASK]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscreteMaskScheduler {
    /// Number of diffusion timesteps (sampling steps)
    pub num_timesteps: usize,
    /// Per-timestep mask ratios in [0,1]
    pub mask_ratios: Array1<f32>,
    /// RNG seed for reproducible masking
    pub seed: u64,
}

impl DiscreteMaskScheduler {
    pub fn new(num_timesteps: usize) -> Self {
        let mut ratios = Array1::<f32>::zeros(num_timesteps);
        for t in 0..num_timesteps {
            let frac = t as f32 / (num_timesteps.max(1) as f32);
            ratios[t] = frac.clamp(0.0, 1.0);
        }
        Self {
            num_timesteps,
            mask_ratios: ratios,
            seed: 42,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sample a global mask ratio t ~ U[0,1] and apply absorbing-state masking
    /// ids: (1, seq_len) float array of token ids; returns masked ids array
    pub fn mask_sequence(&self, ids: &Array2<f32>, mask_token_id: usize) -> (Array2<f32>, f32) {
        let seq_len = ids.ncols();
        let uniform = Uniform::new(0.0f32, 1.0f32).expect("uniform[0,1]");
        let mut rng = get_rng();
        let t_ratio = uniform.sample(&mut rng);
        let k = ((t_ratio * seq_len as f32).round() as usize).min(seq_len);
        let mut indices: Vec<usize> = (0..seq_len).collect();
        let random_salt = rng.random::<u64>();
        indices.sort_by_key(|&i| mix64(self.seed ^ random_salt, i as u64));
        let mut masked = ids.clone();
        for &pos in indices.iter().take(k) {
            masked[[0, pos]] = mask_token_id as f32;
        }
        (masked, t_ratio)
    }

    pub fn mask_sequence_at_t(
        &self,
        ids: &Array2<f32>,
        mask_token_id: usize,
        t: usize,
    ) -> Array2<f32> {
        let seq_len = ids.ncols();
        let ratio = if t < self.mask_ratios.len() {
            self.mask_ratios[t]
        } else {
            1.0
        };
        let k = ((ratio * seq_len as f32).round() as usize).min(seq_len);
        let mut indices: Vec<usize> = (0..seq_len).collect();
        let mut rng = get_rng();
        let random_salt = rng.random::<u64>();
        indices.sort_by_key(|&i| mix64_with_t(self.seed ^ random_salt, t as u64, i as u64));
        let mut masked = ids.clone();
        for &pos in indices.iter().take(k) {
            masked[[0, pos]] = mask_token_id as f32;
        }
        masked
    }

    pub fn target_unmasked_count_at_t(&self, seq_len: usize, t: usize) -> usize {
        let ratio = if t < self.mask_ratios.len() {
            self.mask_ratios[t]
        } else {
            1.0
        };
        let masked = ((ratio * seq_len as f32).round() as usize).min(seq_len);
        seq_len.saturating_sub(masked)
    }

    pub fn reverse_unmask_step(
        &self,
        ids: &Array2<f32>,
        probs: &Array2<f32>,
        mask_token_id: usize,
        t: usize,
        top_p: f32,
    ) -> Array2<f32> {
        let seq_len = ids.ncols();
        let target_unmasked = self.target_unmasked_count_at_t(seq_len, t);
        let mut current_unmasked = 0usize;
        for i in 0..seq_len {
            if ids[[0, i]] != mask_token_id as f32 {
                current_unmasked += 1;
            }
        }
        let need = target_unmasked.saturating_sub(current_unmasked);
        if need == 0 {
            return ids.clone();
        }
        let mut masked_positions: Vec<(usize, f32)> = (0..seq_len)
            .filter(|&i| ids[[0, i]] == mask_token_id as f32)
            .map(|i| {
                let mut m = 0.0f32;
                for &p in probs.row(i).iter() {
                    if p > m {
                        m = p;
                    }
                }
                (i, m)
            })
            .collect();
        masked_positions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = ids.clone();
        let mut rng = get_rng();
        for &(pos, _) in masked_positions.iter().take(need) {
            let row = probs.row(pos);
            let mut indexed: Vec<(usize, f32)> = row
                .iter()
                .enumerate()
                .map(|(tid, &p)| (tid, p.max(0.0)))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            let mut cutoff = 0usize;
            for (i, &(_, p)) in indexed.iter().enumerate() {
                cum += p;
                cutoff = i;
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
            out[[0, pos]] = chosen as f32;
        }
        out
    }

    /// Remask low-confidence positions (flexible remasking)
    /// confidence: (1, seq_len) in [0,1]; threshold in [0,1]
    pub fn remask(
        &self,
        ids: &Array2<f32>,
        confidence: &Array2<f32>,
        threshold: f32,
        mask_token_id: usize,
    ) -> Array2<f32> {
        let mut out = ids.clone();
        let seq_len = ids.ncols();
        for i in 0..seq_len {
            if confidence[[0, i]] < threshold {
                out[[0, i]] = mask_token_id as f32;
            }
        }
        out
    }

    pub fn mask_sequence_span_at_t(
        &self,
        ids: &Array2<f32>,
        mask_token_id: usize,
        t: usize,
        span_start: usize,
        span_end: usize,
    ) -> Array2<f32> {
        let seq_len = ids.ncols();
        if span_start >= span_end || span_start >= seq_len {
            return self.mask_sequence_at_t(ids, mask_token_id, t);
        }
        let span_end = span_end.min(seq_len);
        let available = span_end.saturating_sub(span_start);
        if available == 0 {
            return ids.clone();
        }
        let ratio = if t < self.mask_ratios.len() {
            self.mask_ratios[t]
        } else {
            1.0
        };
        let k = ((ratio * available as f32).round() as usize).min(available);
        if k == 0 {
            return ids.clone();
        }
        let mut indices: Vec<usize> = (span_start..span_end).collect();
        let mut rng = get_rng();
        let random_salt = rng.random::<u64>();
        indices.sort_by_key(|&i| mix64_with_t(self.seed ^ random_salt, t as u64, i as u64));
        let mut masked = ids.clone();
        for idx in indices.into_iter().take(k) {
            masked[[0, idx]] = mask_token_id as f32;
        }
        masked
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, s};

    use super::*;

    #[test]
    fn mask_sequence_span_only_affects_range() {
        let mut scheduler = DiscreteMaskScheduler::new(4);
        scheduler.mask_ratios = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let ids = array![[1., 2., 3., 4., 5.]];
        let masked = scheduler.mask_sequence_span_at_t(&ids, 99, 2, 1, 4);
        // Positions outside the span remain unchanged
        assert_eq!(masked[[0, 0]], 1.0);
        assert_eq!(masked[[0, 4]], 5.0);
        let span_slice = masked.slice(s![0, 1..4]);
        assert!(span_slice.iter().all(|&v| (v - 99.0).abs() < f32::EPSILON));
    }
}
