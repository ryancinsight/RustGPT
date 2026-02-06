use ndarray::{Array2, s};

/// Cache for a single attention head storing Key and Value matrices
#[derive(Debug, Clone)]
pub struct HeadCache {
    /// Key matrix cache (capacity, head_dim)
    pub k: Array2<f32>,
    /// Value matrix cache (capacity, head_dim)
    pub v: Array2<f32>,
    /// Current number of tokens stored in the cache
    pub len: usize,
}

impl HeadCache {
    /// Create a new cache with specified capacity
    pub fn new(capacity: usize, head_dim: usize) -> Self {
        Self {
            k: Array2::zeros((capacity, head_dim)),
            v: Array2::zeros((capacity, head_dim)),
            len: 0,
        }
    }

    /// Reset the cache (clear stored tokens)
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Append new Key and Value states to the cache
    pub fn append(&mut self, k: &Array2<f32>, v: &Array2<f32>) {
        let n = k.nrows();
        let head_dim = self.k.ncols();

        // Ensure input dimensions match
        assert_eq!(k.ncols(), head_dim, "Key dimension mismatch");
        assert_eq!(v.ncols(), head_dim, "Value dimension mismatch");
        assert_eq!(v.nrows(), n, "Key/Value row count mismatch");

        // Resize if necessary (though usually capacity is pre-set to max_seq_len)
        if self.len + n > self.k.nrows() {
            let new_cap = (self.len + n).max(self.k.nrows() * 2);

            let mut new_k = Array2::zeros((new_cap, head_dim));
            let mut new_v = Array2::zeros((new_cap, head_dim));

            if self.len > 0 {
                new_k.slice_mut(s![0..self.len, ..]).assign(&self.k.slice(s![0..self.len, ..]));
                new_v.slice_mut(s![0..self.len, ..]).assign(&self.v.slice(s![0..self.len, ..]));
            }

            self.k = new_k;
            self.v = new_v;
        }

        // Append new data
        self.k.slice_mut(s![self.len..self.len+n, ..]).assign(k);
        self.v.slice_mut(s![self.len..self.len+n, ..]).assign(v);
        self.len += n;
    }

    /// Get valid Key slice
    pub fn key_view(&self) -> ndarray::ArrayView2<'_, f32> {
        self.k.slice(s![0..self.len, ..])
    }

    /// Get valid Value slice
    pub fn value_view(&self) -> ndarray::ArrayView2<'_, f32> {
        self.v.slice(s![0..self.len, ..])
    }
}
