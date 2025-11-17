//! Shared metrics and utilities used across the LLM library.
//!
//! This module contains metrics structures and utility functions commonly used
//! in mixture models (MoE, MoH) and potentially other components.

pub mod perf;
pub mod text;
pub mod topk;

pub use perf::{
    EstimateInput, FlopsEstimate, estimate_diffusion_block, estimate_transformer_block,
    estimate_trm,
};
use serde::{Deserialize, Serialize};
pub use text::{bleu_1_2, corpus_bleu_1_2};
pub use topk::{compute_nim, compute_nim_from_normalized, select_top_k};

/// Per-head metrics used by MoH (and potentially other head-based mixtures).
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct PerHeadMetrics {
    pub active_sum_per_head: Vec<f32>,
    pub token_count_per_head: Vec<usize>,

    // tau statistics for learned threshold predictor
    pub tau_min: f32,
    pub tau_max: f32,
    pub tau_sum: f32,
    pub tau_count: usize,

    // predictor norm stats
    pub g_sq_sum: f32,
    pub g_count: usize,
}

impl PerHeadMetrics {
    pub fn new(num_heads: usize) -> Self {
        Self {
            active_sum_per_head: vec![0.0; num_heads],
            token_count_per_head: vec![0; num_heads],
            tau_min: f32::INFINITY,
            tau_max: f32::NEG_INFINITY,
            tau_sum: 0.0,
            tau_count: 0,
            g_sq_sum: 0.0,
            g_count: 0,
        }
    }

    /// Accumulate per-head active sums and token counts (batch-level flush)
    pub fn flush_active(&mut self, active_sums: &[f32], token_counts: &[usize]) {
        for (h, v) in active_sums.iter().enumerate() {
            self.active_sum_per_head[h] += *v;
            self.token_count_per_head[h] += token_counts[h];
        }
    }

    pub fn update_tau_stats(&mut self, local_min: f32, local_max: f32, count: usize) {
        if count > 0 {
            self.tau_min = self.tau_min.min(local_min);
            self.tau_max = self.tau_max.max(local_max);
            self.tau_count += count;
        }
    }

    pub fn update_pred_norm(&mut self, g_sq_sum_local: f32, g_count_local: usize) {
        self.g_sq_sum += g_sq_sum_local;
        self.g_count += g_count_local;
    }

    pub fn reset_head_metrics(&mut self) {
        for v in &mut self.active_sum_per_head {
            *v = 0.0;
        }
        for c in &mut self.token_count_per_head {
            *c = 0;
        }
        self.tau_min = f32::INFINITY;
        self.tau_max = f32::NEG_INFINITY;
        self.tau_sum = 0.0;
        self.tau_count = 0;
        self.g_sq_sum = 0.0;
        self.g_count = 0;
    }

    /// Return per-head average active and token counts, then reset those counters.
    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        let mut res = Vec::with_capacity(self.active_sum_per_head.len());
        for h in 0..self.active_sum_per_head.len() {
            let tokens = self.token_count_per_head[h];
            let avg = if tokens > 0 {
                self.active_sum_per_head[h] / tokens as f32
            } else {
                0.0
            };
            res.push((avg, tokens));
            self.active_sum_per_head[h] = 0.0;
            self.token_count_per_head[h] = 0;
        }
        res
    }

    pub fn take_tau_metrics(&mut self) -> Option<(f32, f32)> {
        if self.tau_count > 0 {
            let min = self.tau_min;
            let max = self.tau_max;
            self.tau_min = f32::INFINITY;
            self.tau_max = f32::NEG_INFINITY;
            self.tau_sum = 0.0;
            self.tau_count = 0;
            Some((min, max))
        } else {
            None
        }
    }

    pub fn take_pred_norm(&mut self) -> Option<f32> {
        if self.g_count > 0 {
            let rms = (self.g_sq_sum / self.g_count as f32).sqrt();
            self.g_sq_sum = 0.0;
            self.g_count = 0;
            Some(rms)
        } else {
            None
        }
    }
}

/// Simple NIM metrics container used by MoE
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct NimMetrics {
    pub nim_sum: f32,
    pub token_count: usize,
    pub actual_expert_count_sum: usize,
    pub actual_expert_token_count: usize,
}

impl NimMetrics {
    pub fn new() -> Self {
        Self {
            nim_sum: 0.0,
            token_count: 0,
            actual_expert_count_sum: 0,
            actual_expert_token_count: 0,
        }
    }

    pub fn add(&mut self, nim: f32) {
        self.nim_sum += nim;
        self.token_count += 1;
    }

    pub fn add_actual_count(&mut self, actual_count: usize) {
        self.actual_expert_count_sum += actual_count;
        self.actual_expert_token_count += 1;
    }

    pub fn get_and_reset(&mut self) -> Option<(f32, usize)> {
        if self.token_count > 0 {
            let avg = self.nim_sum / self.token_count as f32;
            let tokens = self.token_count;
            self.nim_sum = 0.0;
            self.token_count = 0;
            Some((avg, tokens))
        } else {
            None
        }
    }

    pub fn get_actual_and_reset(&mut self) -> Option<f32> {
        if self.actual_expert_token_count > 0 {
            let avg = self.actual_expert_count_sum as f32 / self.actual_expert_token_count as f32;
            self.actual_expert_count_sum = 0;
            self.actual_expert_token_count = 0;
            Some(avg)
        } else {
            None
        }
    }
}
