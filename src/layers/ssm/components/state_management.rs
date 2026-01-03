//! State Management Component for SSMs
//!
//! Provides efficient state management with automatic cache invalidation
//! and memory optimization for state space models.

use ndarray::Array2;
use std::collections::HashMap;

/// State cache with automatic invalidation
#[derive(Debug, Clone)]
pub struct StateCache {
    /// Cached states keyed by cache identifier
    states: HashMap<String, Array2<f32>>,
    /// Cache validity tracking
    valid: bool,
    /// Embedding dimension for validation
    embed_dim: usize,
    /// Sequence length for validation
    seq_len: Option<usize>,
}

impl StateCache {
    /// Create a new state cache
    pub fn new(embed_dim: usize) -> Self {
        Self {
            states: HashMap::new(),
            valid: false,
            embed_dim,
            seq_len: None,
        }
    }

    /// Invalidate cache when input dimensions change
    pub fn invalidate_if_needed(&mut self, input: &Array2<f32>) {
        let new_seq_len = input.nrows();
        let new_embed_dim = input.ncols();
        
        if new_embed_dim != self.embed_dim || Some(new_seq_len) != self.seq_len {
            self.invalidate();
            self.embed_dim = new_embed_dim;
            self.seq_len = Some(new_seq_len);
        }
    }

    /// Manually invalidate cache
    pub fn invalidate(&mut self) {
        self.states.clear();
        self.valid = false;
    }

    /// Cache a state array
    pub fn cache_state(&mut self, key: &str, state: Array2<f32>) {
        self.states.insert(key.to_string(), state);
        self.valid = true;
    }

    /// Retrieve a cached state
    pub fn get_state(&self, key: &str) -> Option<&Array2<f32>> {
        self.states.get(key)
    }

    /// Retrieve a cached state mutably
    pub fn get_state_mut(&mut self, key: &str) -> Option<&mut Array2<f32>> {
        self.states.get_mut(key)
    }

    /// Remove a specific cached state
    pub fn remove_state(&mut self, key: &str) {
        self.states.remove(key);
    }

    /// Check if cache is valid
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.states.values()
            .map(|arr| arr.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Clear memory by removing large cached states
    pub fn clear_large_states(&mut self, max_size_bytes: usize) {
        let mut total_size = self.memory_usage();
        if total_size <= max_size_bytes {
            return;
        }

        // Sort states by size (descending) and remove largest first
        let mut states_by_size: Vec<_> = self.states.iter()
            .map(|(k, v)| (k.clone(), v.len() * std::mem::size_of::<f32>()))
            .collect();
        
        states_by_size.sort_by(|a, b| b.1.cmp(&a.1));

        for (key, size) in states_by_size {
            if total_size <= max_size_bytes {
                break;
            }
            self.states.remove(&key);
            total_size -= size;
        }
        
        if self.states.is_empty() {
            self.valid = false;
        }
    }
}

impl Default for StateCache {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Smart state manager that handles cache invalidation automatically
#[derive(Debug, Clone)]
pub struct StateManager {
    cache: StateCache,
    max_memory_bytes: usize,
}

impl StateManager {
    /// Create a new state manager with memory limit
    pub fn new(embed_dim: usize, max_memory_bytes: usize) -> Self {
        Self {
            cache: StateCache::new(embed_dim),
            max_memory_bytes,
        }
    }

    /// Get the underlying cache
    pub fn cache(&mut self, input: &Array2<f32>) -> &mut StateCache {
        self.cache.invalidate_if_needed(input);
        self.cache.clear_large_states(self.max_memory_bytes);
        &mut self.cache
    }

    /// Invalidate cache manually
    pub fn invalidate(&mut self) {
        self.cache.invalidate();
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.cache.memory_usage()
    }
}