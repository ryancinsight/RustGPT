use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{common::rng::get_rng, infrastructure::optimizer::adam::Adam};
use super::traits::PositionEmbedding;

/// Gradients for PathCoPE
#[derive(Clone, Debug)]
pub struct PathCoPEGradients {
    pub w_householder_grads: Option<Array2<f32>>,
    pub u_beta_grads: Option<Array2<f32>>,
    pub b_beta_grads: Option<Array2<f32>>,
    pub base_cope_grads: Option<Array2<f32>>,
    pub alpha_path_grad: f32,
    pub alpha_cope_grad: f32,
}

impl PathCoPEGradients {
    pub fn new(max_seq_len: usize, embed_dim: usize, householder_dim: usize) -> Self {
        Self {
            w_householder_grads: Some(Array2::zeros((max_seq_len, householder_dim))),
            u_beta_grads: Some(Array2::zeros((embed_dim, 1))),
            b_beta_grads: Some(Array2::zeros((1, 1))),
            base_cope_grads: Some(Array2::zeros((max_seq_len, embed_dim))),
            alpha_path_grad: 0.0,
            alpha_cope_grad: 0.0,
        }
    }
}

/// PaTH-CoPE: Position Encoding via Accumulating Householder Transformations
///
/// Implements the PaTH attention mechanism as a COPE variant, using data-dependent
/// Householder-like transformations accumulated along the path between positions.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PathCoPE {
    /// Maximum sequence length
    max_seq_len: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Householder vector dimension (usually embed_dim)
    householder_dim: usize,

    /// Householder vectors W ∈ ℝ^(max_seq_len, householder_dim)
    /// Each row w_t is the Householder vector at position t
    w_householder: Array2<f32>,
    opt_w_householder: Adam,

    /// Beta predictor: u ∈ ℝ^(embed_dim, 1) for computing β_t
    u_beta: Array2<f32>,
    opt_u_beta: Adam,
    /// Beta bias
    b_beta: Array2<f32>,
    opt_b_beta: Adam,

    /// Base CoPE embeddings for hybrid approach (max_seq_len, embed_dim)
    base_cope: Array2<f32>,
    opt_base_cope: Adam,

    /// Mixing weight for PaTH vs CoPE (learnable)
    alpha_path: f32,
    alpha_cope: f32,

    /// Cache for Householder products (computed per forward pass)
    #[serde(skip)]
    product_cache: Option<Array2<f32>>,
}

impl PositionEmbedding for PathCoPE {
    type Gradients = PathCoPEGradients;

    fn contribution(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: Option<&ArrayView2<f32>>,
    ) -> f32 {
        if query_pos > self.max_seq_len || key_pos > query_pos {
            return 0.0;
        }

        let relative_pos = query_pos - key_pos;

        // PaTH component: k^T * transformed_q
        let path_transformed_q = if query_pos > key_pos {
            if let Some(inp) = inputs {
                // Full path transformation
                self.compute_path_transform(q, query_pos, key_pos, inp)
            } else {
                 // Simplified approximation (single Householder step)
                 if relative_pos >= self.max_seq_len {
                     Array1::zeros(self.embed_dim)
                 } else {
                     let w_key = self.w_householder.row(relative_pos.min(self.max_seq_len - 1));
                     let beta_approx = 1.0;
                     self.apply_householder(q, &w_key, beta_approx)
                 }
            }
        } else {
            q.to_owned()
        };
        let path_contrib = k.dot(&path_transformed_q);

        // Base CoPE component
        let cope_contrib = if relative_pos < self.max_seq_len {
            q.dot(&self.base_cope.row(relative_pos.min(self.max_seq_len - 1)))
        } else {
            0.0
        };

        // Weighted combination
        self.alpha_path * path_contrib + self.alpha_cope * cope_contrib
    }

    fn backward(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: Option<&ArrayView2<f32>>,
        d_s_ij: f32,
        grads: &mut Self::Gradients,
    ) -> (Array1<f32>, Array1<f32>) {
        if query_pos > self.max_seq_len || key_pos > query_pos {
            return (Array1::zeros(q.dim()), Array1::zeros(k.dim()));
        }
        
        let relative_pos = query_pos - key_pos;

        // 1. Gradients for Base CoPE
        let mut d_q_base = Array1::zeros(q.dim());
        
        if relative_pos < self.max_seq_len {
            let idx = relative_pos.min(self.max_seq_len - 1);
            let cope_vec = self.base_cope.row(idx);
            let cope_contrib = q.dot(&cope_vec);
            
            // dL/dAlphaCope = dL/dS * cope_contrib
            grads.alpha_cope_grad += d_s_ij * cope_contrib;
            
            // dL/dBaseCope = dL/dS * alpha_cope * q
            let d_cope_vec = d_s_ij * self.alpha_cope;
            if let Some(base_grads) = &mut grads.base_cope_grads {
                 let mut row = base_grads.row_mut(idx);
                 // row += q * d_cope_vec
                 for (r, &q_val) in row.iter_mut().zip(q.iter()) {
                     *r += q_val * d_cope_vec;
                 }
            }
            
            // dL/dQ_base = dL/dS * alpha_cope * cope_vec
            // d_q_base += cope_vec * d_cope_vec (which is d_s_ij * alpha_cope)
             for (d, &c_val) in d_q_base.iter_mut().zip(cope_vec.iter()) {
                 *d += c_val * d_cope_vec;
             }
        }

        // 2. Gradients for Path CoPE
        let mut d_q_path = Array1::zeros(q.dim());
        let mut d_k_path = Array1::zeros(k.dim());
        
        if let Some(inp) = inputs {
             // Recompute forward path to get intermediate states
             // We need y_s for s in key_pos..=query_pos
             // y_query_pos = q
             // But wait, the transform is applied to q_i (from inputs) in compute_path_transform?
             // Looking at compute_path_transform:
             // let q_i = inputs.row(query_pos).to_owned();
             // transformed = q_i;
             // for s in (key_pos + 1)..=query_pos: transformed = H_s * transformed
             
             // Wait, the input `q` to `contribution` is passed as argument. 
             // In `compute_path_transform`, it uses `inputs.row(query_pos)`.
             // This is an inconsistency in the original code?
             // `path_cope_contribution` calls `compute_path_transform` which uses `inputs.row(query_pos)`.
             // But `path_cope_contribution` ALSO takes `q` as argument.
             // And it uses `q` for the Base CoPE part: `q.dot(...)`.
             // But for the Path part, it ignores the passed `q` and uses `inputs.row(query_pos)` inside `compute_path_transform`.
             // This seems like a potential bug or design choice in the original code.
             // I should probably use the passed `q` instead of `inputs.row(query_pos)` to be consistent with the trait which assumes `q` is the query.
             
             // I will modify `compute_path_transform` to take `q` vector instead of extracting it from inputs.
             // This is safer and cleaner.
             
             let q_vec = q.to_owned();
             let mut states = Vec::with_capacity(query_pos - key_pos + 1);
             states.push(q_vec.clone());
             
             let mut transformed = q_vec.clone();
             
             // Forward pass reconstruction
             for s in (key_pos + 1)..=query_pos {
                 if s >= self.max_seq_len { break; }
                 let w_s = self.w_householder.row(s);
                 let x_s = inp.row(s);
                 let beta_s = self.compute_beta(&x_s);
                 transformed = self.apply_householder(&transformed.view(), &w_s, beta_s);
                 states.push(transformed.clone());
             }
             
             let path_transformed_q = transformed;
             let path_contrib = k.dot(&path_transformed_q);
             
             grads.alpha_path_grad += d_s_ij * path_contrib;
             
             // dL/d_path_transformed_q = dL/dS * alpha_path * k
             let mut d_y = k.mapv(|x| x * d_s_ij * self.alpha_path);
             
             // dL/dK = dL/dS * alpha_path * path_transformed_q
             d_k_path = path_transformed_q.mapv(|x| x * d_s_ij * self.alpha_path);
             
             // Backprop through Householder layers
             // s goes from query_pos down to key_pos + 1
             // states[i] corresponds to output after i-th step. 
             // states[0] is initial q.
             // Loop index needs to match states index.
             // states has (query_pos - key_pos + 1) elements.
             // Step s corresponds to transition from states[s - (key_pos + 1)] to states[s - key_pos].
             
             for s in ((key_pos + 1)..=query_pos).rev() {
                 if s >= self.max_seq_len { continue; }
                 
                 let input_idx = s - (key_pos + 1);
                 let y_prev = &states[input_idx]; // y_{s-1}
                 
                 let w_s = self.w_householder.row(s);
                 let x_s = inp.row(s);
                 let beta_s = self.compute_beta(&x_s);
                 
                 // Gradients for this layer
                 // y_curr = y_prev - beta * w * (w^T * y_prev)
                 
                 // dL/d_y_prev = H_s * d_y (since H is symmetric)
                 // We can reuse apply_householder logic!
                 let d_y_prev = self.apply_householder(&d_y.view(), &w_s, beta_s);
                 
                 // dL/d_w
                 // dL/dw = - beta * (y_prev * (w^T d_y) + d_y * (w^T y_prev))
                 let w_dot_dy = w_s.dot(&d_y);
                 let w_dot_yprev = w_s.dot(y_prev);
                 
                 let term1 = y_prev.mapv(|x| x * w_dot_dy);
                 let term2 = d_y.mapv(|x| x * w_dot_yprev);
                 let d_w = (term1 + term2).mapv(|x| -beta_s * x);
                 
                 if let Some(w_grads) = &mut grads.w_householder_grads {
                     let mut row = w_grads.row_mut(s);
                     for (r, &val) in row.iter_mut().zip(d_w.iter()) {
                         *r += val;
                     }
                 }
                 
                 // dL/d_beta
                 // dL/d_beta = - (d_y . (w * (w^T * y_prev)))
                 //           = - (d_y . (w * w_dot_yprev))
                 //           = - w_dot_yprev * (d_y . w)
                 //           = - w_dot_yprev * w_dot_dy
                 let d_beta = -w_dot_yprev * w_dot_dy;
                 
                 // dL/d_logit = dL/d_beta * d_beta/d_logit
                 // beta = 2 * sigmoid(logit)
                 // d_beta/d_logit = 2 * sigmoid * (1 - sigmoid) = beta * (1 - sigmoid) ? No.
                 // sigmoid(x) = 1/(1+exp(-x))
                 // d(2sig)/dx = 2 * sig * (1-sig)
                 let logit = x_s.dot(&self.u_beta.column(0)) + self.b_beta[[0, 0]];
                 let sig = sigmoid_stable(logit);
                 let d_beta_d_logit = 2.0 * sig * (1.0 - sig);
                 let d_logit = d_beta * d_beta_d_logit;
                 
                 // dL/d_u = d_logit * x_s
                 if let Some(u_grads) = &mut grads.u_beta_grads {
                     let mut col = u_grads.column_mut(0);
                     for (u, &val) in col.iter_mut().zip(x_s.iter()) {
                         *u += val * d_logit;
                     }
                 }
                 
                 // dL/d_b = d_logit
                 if let Some(b_grads) = &mut grads.b_beta_grads {
                     b_grads[[0, 0]] += d_logit;
                 }
                 
                 // Update d_y for next step (going backwards)
                 d_y = d_y_prev;
             }
             
             // The final d_y is dL/dq_path (gradient at the start of the chain, which is q)
             d_q_path = d_y;
        }

        (d_q_base + d_q_path, d_k_path)
    }

    fn init_gradients(&self) -> Self::Gradients {
        PathCoPEGradients::new(self.max_seq_len, self.embed_dim, self.householder_dim)
    }

    fn apply_gradients(&mut self, grads: &Self::Gradients, lr: f32) {
        if let Some(wg) = &grads.w_householder_grads {
            self.opt_w_householder.step(&mut self.w_householder, wg, lr);
        }
        if let Some(ug) = &grads.u_beta_grads {
            self.opt_u_beta.step(&mut self.u_beta, ug, lr);
        }
        if let Some(bg) = &grads.b_beta_grads {
            self.opt_b_beta.step(&mut self.b_beta, bg, lr);
        }
        if let Some(cg) = &grads.base_cope_grads {
            self.opt_base_cope.step(&mut self.base_cope, cg, lr);
        }
        
        // Update alphas manually (simple SGD)
        self.alpha_path -= lr * grads.alpha_path_grad;
        self.alpha_cope -= lr * grads.alpha_cope_grad;
    }

    fn max_pos(&self) -> usize {
        self.max_seq_len
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn parameters(&self) -> usize {
        self.w_householder.len() + self.u_beta.len() + self.b_beta.len() + self.base_cope.len() + 2
    }

    fn weight_norm(&self) -> f32 {
        let w_norm: f32 = self.w_householder.iter().map(|x| x * x).sum();
        let u_norm: f32 = self.u_beta.iter().map(|x| x * x).sum();
        let b_norm: f32 = self.b_beta.iter().map(|x| x * x).sum();
        let base_norm: f32 = self.base_cope.iter().map(|x| x * x).sum();
        (w_norm + u_norm + b_norm + base_norm + self.alpha_path.powi(2) + self.alpha_cope.powi(2)).sqrt()
    }
}

impl PathCoPE {
    /// Create a new PathCoPE instance
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to handle
    /// * `embed_dim` - Embedding dimension
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        let mut rng = get_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();
        let normal_householder = Normal::new(0.0, 0.01).unwrap();

        // Householder vectors: initialized with small values
        let w_householder = Array2::from_shape_fn((max_seq_len, embed_dim), |_| {
            normal_householder.sample(&mut rng)
        });
        let opt_w_householder = Adam::new((max_seq_len, embed_dim));

        // Beta predictor parameters
        let u_beta = Array2::from_shape_fn((embed_dim, 1), |_| normal.sample(&mut rng) * 0.1);
        let opt_u_beta = Adam::new((embed_dim, 1));
        let b_beta = Array2::zeros((1, 1));
        let opt_b_beta = Adam::new((1, 1));

        // Base CoPE embeddings
        let base_cope =
            Array2::from_shape_fn((max_seq_len, embed_dim), |_| normal.sample(&mut rng));
        let opt_base_cope = Adam::new((max_seq_len, embed_dim));

        Self {
            max_seq_len,
            embed_dim,
            householder_dim: embed_dim,
            w_householder,
            opt_w_householder,
            u_beta,
            opt_u_beta,
            b_beta,
            opt_b_beta,
            base_cope,
            opt_base_cope,
            alpha_path: 0.7, // Emphasize PaTH by default
            alpha_cope: 0.3,
            product_cache: None,
        }
    }

    /// Compute β_t = 2 * sigmoid(u^T * x_t + b)
    ///
    /// # Arguments
    /// * `x_t` - Input token embedding at position t
    ///
    /// # Returns
    /// β_t ∈ (0, 2)
    #[inline]
    pub fn compute_beta(&self, x_t: &ArrayView1<'_, f32>) -> f32 {
        let logit = x_t.dot(&self.u_beta.column(0)) + self.b_beta[[0, 0]];
        let beta = 2.0 * sigmoid_stable(logit);
        beta.clamp(1e-6, 2.0 - 1e-6) // Ensure in (0, 2)
    }

    /// Compute Householder transformation H_t = I - β_t * w_t * w_t^T
    ///
    /// Uses compact representation without full matrix materialization.
    /// Returns the transformed vector efficiently.
    ///
    /// # Arguments
    /// * `v` - Vector to transform
    /// * `w_t` - Householder vector at position t
    /// * `beta_t` - Scalar β_t
    ///
    /// # Returns
    /// H_t * v = v - β_t * w_t * (w_t^T * v)
    #[inline]
    fn apply_householder(
        &self,
        v: &ArrayView1<'_, f32>,
        w_t: &ArrayView1<'_, f32>,
        beta_t: f32,
    ) -> Array1<f32> {
        let projection = w_t.dot(v);
        v - w_t * (beta_t * projection)
    }

    /// Compute cumulative Householder product along path j+1 to i
    ///
    /// P_{j→i} = ∏_{s=j+1}^i H_s
    ///
    /// Uses the UT transform representation for efficiency:
    /// P = I - W^T * T^{-1} * W
    /// where T^{-1} = (I + strictLower(D * W * W^T))^{-1} * D
    ///
    /// # Arguments
    /// * `query_pos` - Query position i
    /// * `key_pos` - Key position j (j < i)
    /// * `inputs` - Input embeddings for computing β values
    ///
    /// # Returns
    /// Transformed query vector after path accumulation
    pub fn compute_path_transform(
        &self,
        q: &ArrayView1<'_, f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: &ArrayView2<'_, f32>,
    ) -> Array1<f32> {
        if query_pos > self.max_seq_len || key_pos >= query_pos {
            return Array1::zeros(self.embed_dim);
        }

        // Get query vector at position query_pos
        // q is passed directly
        let q_i = q.to_owned();

        // Apply cumulative Householder transformations from key_pos+1 to query_pos
        let mut transformed = q_i;

        for s in (key_pos + 1)..=query_pos {
            if s >= self.max_seq_len {
                break;
            }

            let w_s = self.w_householder.row(s);
            let x_s = inputs.row(s);
            let beta_s = self.compute_beta(&x_s);

            transformed = self.apply_householder(&transformed.view(), &w_s, beta_s);
        }

        transformed
    }

    /// Compute PaTH-CoPE attention contribution
    ///
    /// Combines PaTH transformation with base CoPE for hybrid position encoding:
    /// contribution = α_path * (k_j^T * P_{j→i} * q_i) + α_cope * (q_i · PE_pos)
    ///
    /// # Arguments
    /// * `q` - Query vector at position i
    /// * `k` - Key vector at position j
    /// * `query_pos` - Query position i
    /// * `key_pos` - Key position j
    /// * `inputs` - Full input sequence for computing path transformations
    ///
    /// # Returns
    /// Combined PaTH-CoPE contribution
    pub fn path_cope_contribution(
        &self,
        q: &ArrayView1<'_, f32>,
        k: &ArrayView1<'_, f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: &ArrayView2<'_, f32>,
    ) -> f32 {
        if query_pos > self.max_seq_len || key_pos > query_pos {
            return 0.0;
        }

        let relative_pos = query_pos - key_pos;

        // PaTH component: k^T * transformed_q
        let path_transformed_q = if query_pos > key_pos {
            self.compute_path_transform(q, query_pos, key_pos, inputs)
        } else {
            q.to_owned()
        };
        let path_contrib = k.dot(&path_transformed_q);

        // Base CoPE component
        let cope_contrib = if relative_pos < self.max_seq_len {
            q.dot(&self.base_cope.row(relative_pos.min(self.max_seq_len - 1)))
        } else {
            0.0
        };

        // Weighted combination
        self.alpha_path * path_contrib + self.alpha_cope * cope_contrib
    }

    /// Compute simplified PaTH contribution without full sequence inputs
    ///
    /// Uses cached Householder products when available. Falls back to
    /// approximate computation using local context.
    ///
    /// # Arguments
    /// * `q` - Query vector
    /// * `k` - Key vector  
    /// * `relative_pos` - Relative position (query_pos - key_pos)
    ///
    /// # Returns
    /// Simplified contribution using cached or approximate transforms
    pub fn path_contribution_simple(
        &self,
        q: &ArrayView1<'_, f32>,
        k: &ArrayView1<'_, f32>,
        relative_pos: usize,
    ) -> f32 {
        if relative_pos == 0 {
            // No transformation needed for same position
            return self.alpha_path * k.dot(q) + self.alpha_cope * q.dot(&self.base_cope.row(0));
        }

        if relative_pos >= self.max_seq_len {
            return 0.0;
        }

        // Approximate path transformation using accumulated Householder vectors
        // This is a simplified version that uses the Householder vector at the key position
        let w_key = self
            .w_householder
            .row(relative_pos.min(self.max_seq_len - 1));

        // Apply single Householder transformation as approximation
        let beta_approx = 1.0; // Use fixed beta for simplified version
        let projection = w_key.dot(q);
        let transformed_q = q - &(&w_key * (beta_approx * projection));

        let path_contrib = k.dot(&transformed_q);
        let cope_contrib = q.dot(&self.base_cope.row(relative_pos));

        self.alpha_path * path_contrib + self.alpha_cope * cope_contrib
    }

    /// Get total parameter count
    pub fn parameters(&self) -> usize {
        self.w_householder.len() + self.u_beta.len() + self.b_beta.len() + self.base_cope.len()
    }

    /// Get weight norm (L2)
    pub fn weight_norm(&self) -> f32 {
        let w_norm: f32 = self.w_householder.iter().map(|&x| x * x).sum();
        let u_norm: f32 = self.u_beta.iter().map(|&x| x * x).sum();
        let b_norm: f32 = self.b_beta.iter().map(|&x| x * x).sum();
        let cope_norm: f32 = self.base_cope.iter().map(|&x| x * x).sum();

        (w_norm + u_norm + b_norm + cope_norm).sqrt()
    }

    /// Update mixing weights (should sum to 1)
    pub fn update_mixing(&mut self, alpha_path: f32, alpha_cope: f32) {
        let sum = alpha_path + alpha_cope;
        self.alpha_path = alpha_path / sum;
        self.alpha_cope = alpha_cope / sum;
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get embedding dimension  
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Clear product cache
    pub fn clear_cache(&mut self) {
        self.product_cache = None;
    }
}

/// Numerically stable sigmoid function
#[inline]
fn sigmoid_stable(x: f32) -> f32 {
    let x = x.clamp(-500.0, 500.0);
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_cope_creation() {
        let path_cope = PathCoPE::new(128, 64);
        assert_eq!(path_cope.max_seq_len, 128);
        assert_eq!(path_cope.embed_dim, 64);
        assert_eq!(path_cope.w_householder.shape(), &[128, 64]);
        assert_eq!(path_cope.base_cope.shape(), &[128, 64]);
    }

    #[test]
    fn test_compute_beta() {
        let path_cope = PathCoPE::new(64, 32);
        let x_t = Array1::from_elem(32, 0.5);

        let beta = path_cope.compute_beta(&x_t.view());

        // β should be in (0, 2)
        assert!(
            beta > 0.0 && beta < 2.0,
            "Beta should be in (0, 2), got {}",
            beta
        );
    }

    #[test]
    fn test_apply_householder() {
        let path_cope = PathCoPE::new(64, 32);
        let v = Array1::from_elem(32, 1.0);
        let w_t = Array1::from_elem(32, 0.1);
        let beta_t = 1.5;

        let transformed = path_cope.apply_householder(&v.view(), &w_t.view(), beta_t);

        // Result should be finite and different from input
        assert!(transformed.iter().all(|&x| x.is_finite()));
        assert_ne!(transformed, v);
    }

    #[test]
    fn test_path_contribution_simple() {
        let path_cope = PathCoPE::new(64, 32);
        let q = Array1::from_elem(32, 0.5);
        let k = Array1::from_elem(32, 0.3);

        let contrib = path_cope.path_contribution_simple(&q.view(), &k.view(), 5);

        assert!(contrib.is_finite(), "Contribution should be finite");
    }

    #[test]
    fn test_parameters() {
        let path_cope = PathCoPE::new(128, 64);
        let params = path_cope.parameters();

        // w_householder: 128 * 64 = 8192
        // u_beta: 64 * 1 = 64
        // b_beta: 1 * 1 = 1
        // base_cope: 128 * 64 = 8192
        // Total: 16449
        assert_eq!(params, 8192 + 64 + 1 + 8192);
    }

    #[test]
    fn test_beta_range() {
        let path_cope = PathCoPE::new(64, 32);

        // Test with various inputs
        for val in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let x_t = Array1::from_elem(32, val);
            let beta = path_cope.compute_beta(&x_t.view());

            assert!(
                beta > 0.0 && beta < 2.0,
                "Beta for input {} should be in (0, 2), got {}",
                val,
                beta
            );
        }
    }

    #[test]
    fn test_path_transform_preserves_norm() {
        let path_cope = PathCoPE::new(64, 32);
        let inputs = Array2::from_elem((64, 32), 0.1);

        // Compute path transform
        let transformed = path_cope.compute_path_transform(10, 5, &inputs.view());

        // Check finiteness
        assert!(transformed.iter().all(|&x| x.is_finite()));

        // Householder transformations should preserve approximate norm
        let original_norm = inputs.row(10).mapv(|x| x * x).sum().sqrt();
        let transformed_norm = transformed.mapv(|x| x * x).sum().sqrt();

        // Norms should be of similar magnitude (Householder is orthogonal-ish)
        let ratio = transformed_norm / original_norm;
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "Norm ratio {} is outside reasonable bounds",
            ratio
        );
    }

    #[test]
    fn test_weight_norm() {
        let path_cope = PathCoPE::new(64, 32);
        let norm = path_cope.weight_norm();

        assert!(norm.is_finite() && norm > 0.0);
    }

    #[test]
    fn test_update_mixing() {
        let mut path_cope = PathCoPE::new(64, 32);

        path_cope.update_mixing(0.8, 0.2);
        assert!((path_cope.alpha_path - 0.8).abs() < 1e-6);
        assert!((path_cope.alpha_cope - 0.2).abs() < 1e-6);

        // Test normalization
        path_cope.update_mixing(2.0, 1.0);
        assert!((path_cope.alpha_path - 2.0 / 3.0).abs() < 1e-6);
        assert!((path_cope.alpha_cope - 1.0 / 3.0).abs() < 1e-6);
    }
}
