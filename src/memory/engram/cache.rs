use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::RandomState;

use ndarray::Array1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramCache {
    pub tier_1: HashMap<usize, Array1<f32>, RandomState>,
    pub tier_2: HashMap<usize, Array1<f32>, RandomState>,
    pub tier_1_size: usize,
    pub tier_2_size: usize,
    pub tier_1_hits: usize,
    pub tier_1_misses: usize,
    pub tier_2_hits: usize,
    pub tier_2_misses: usize,
}

impl EngramCache {
    pub fn new(tier_1_size: usize, tier_2_size: usize) -> Self {
        Self {
            tier_1: HashMap::with_capacity_and_hasher(tier_1_size, RandomState::new()),
            tier_2: HashMap::with_capacity_and_hasher(tier_2_size, RandomState::new()),
            tier_1_size,
            tier_2_size,
            tier_1_hits: 0,
            tier_1_misses: 0,
            tier_2_hits: 0,
            tier_2_misses: 0,
        }
    }

    pub fn get(&mut self, hash_idx: usize) -> Option<&Array1<f32>> {
        if self.tier_1_size == 0 && self.tier_2_size == 0 {
            return None;
        }
        if self.tier_1.contains_key(&hash_idx) {
            self.tier_1_hits += 1;
            return self.tier_1.get(&hash_idx);
        }

        self.tier_1_misses += 1;
        if let Some(embedding) = self.tier_2.get(&hash_idx).cloned() {
            self.tier_2_hits += 1;
            if self.tier_1_size > 0 {
                if self.tier_1.len() >= self.tier_1_size {
                    if let Some(key) = self.tier_1.keys().next().copied() {
                        self.tier_1.remove(&key);
                    }
                }
                self.tier_1.insert(hash_idx, embedding);
                return self.tier_1.get(&hash_idx);
            }
            return self.tier_2.get(&hash_idx);
        }

        self.tier_2_misses += 1;
        None
    }

    pub fn insert(&mut self, hash_idx: usize, embedding: Array1<f32>) {
        if self.tier_1_size == 0 && self.tier_2_size == 0 {
            return;
        }
        if self.tier_1_size > 0 && self.tier_1.len() < self.tier_1_size {
            self.tier_1.insert(hash_idx, embedding.clone());
        } else if self.tier_2_size > 0 && self.tier_2.len() < self.tier_2_size {
            self.tier_2.insert(hash_idx, embedding.clone());
        }
    }

    pub fn insert_raw(&mut self, hash_idx: usize, embedding: Array1<f32>) {
        self.tier_1.insert(hash_idx, embedding);
    }

    pub fn clear_stats(&mut self) {
        self.tier_1_hits = 0;
        self.tier_1_misses = 0;
        self.tier_2_hits = 0;
        self.tier_2_misses = 0;
    }

    pub fn hit_rate(&self) -> (f32, f32) {
        let tier_1_total = self.tier_1_hits + self.tier_1_misses;
        let tier_2_total = self.tier_2_hits + self.tier_2_misses;
        let tier_1_rate = if tier_1_total > 0 {
            self.tier_1_hits as f32 / tier_1_total as f32
        } else {
            0.0
        };
        let tier_2_rate = if tier_2_total > 0 {
            self.tier_2_hits as f32 / tier_2_total as f32
        } else {
            0.0
        };
        (tier_1_rate, tier_2_rate)
    }
}
