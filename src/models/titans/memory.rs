use ndarray::{Array1, Array2, Axis};
use crate::network::Layer;
use serde::{Deserialize, Serialize};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Weights for the inner MLP memory network.
/// Structure: Input (Key) -> Hidden -> Output (Value)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemoryWeights {
    pub w1: Array2<f32>, // Hidden x Key
    pub b1: Array1<f32>, // Hidden
    pub w2: Array2<f32>, // Value x Hidden
    pub b2: Array1<f32>, // Value
}

impl MemoryWeights {
    pub fn new(key_dim: usize, hidden_dim: usize, val_dim: usize, rng: &mut impl Rng) -> Self {
        let normal = Normal::new(0.0, 0.02).unwrap();

        let w1_vec: Vec<f32> = (0..hidden_dim * key_dim).map(|_| normal.sample(rng)).collect();
        let w2_vec: Vec<f32> = (0..val_dim * hidden_dim).map(|_| normal.sample(rng)).collect();

        Self {
            w1: Array2::from_shape_vec((hidden_dim, key_dim), w1_vec).unwrap(),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::from_shape_vec((val_dim, hidden_dim), w2_vec).unwrap(),
            b2: Array1::zeros(val_dim),
        }
    }

    pub fn zeros(key_dim: usize, hidden_dim: usize, val_dim: usize) -> Self {
        Self {
            w1: Array2::zeros((hidden_dim, key_dim)),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::zeros((val_dim, hidden_dim)),
            b2: Array1::zeros(val_dim),
        }
    }

    // Scale weights by a factor
    pub fn scale(&mut self, factor: f32) {
        self.w1.mapv_inplace(|x| x * factor);
        self.b1.mapv_inplace(|x| x * factor);
        self.w2.mapv_inplace(|x| x * factor);
        self.b2.mapv_inplace(|x| x * factor);
    }

    // Add other weights to self
    pub fn add(&mut self, other: &MemoryWeights) {
        self.w1 = &self.w1 + &other.w1;
        self.b1 = &self.b1 + &other.b1;
        self.w2 = &self.w2 + &other.w2;
        self.b2 = &self.b2 + &other.b2;
    }
}

/// Neural Long-Term Memory Module (LMM)
///
/// As described in "Titans: Learning to Memorize at Test Time" (Arxiv 2501.00663).
/// This module acts as a meta-learner that updates its own parameters at test time
/// based on the "surprise" (gradient) of the input data.
#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralMemory {
    // Configuration
    input_dim: usize,
    key_dim: usize,
    val_dim: usize,
    memory_hidden_dim: usize,

    // Meta-parameters (Learnable projections)
    w_q: Array2<f32>, // key_dim x input_dim
    w_k: Array2<f32>, // key_dim x input_dim
    w_v: Array2<f32>, // val_dim x input_dim

    // Adaptive hyperparameters projections (producing scalars)
    // We project input to 1 dimension
    w_alpha: Array1<f32>, // input_dim
    w_eta: Array1<f32>,   // input_dim
    w_theta: Array1<f32>, // input_dim

    // Initial Memory State (Meta-learned initialization)
    init_memory: MemoryWeights,

    // Current State (Evolving during forward pass)
    // In a real implementation, this should probably be transient or managed via a State struct passed to forward,
    // but Layer trait implies internal state for RNNs in this codebase.
    curr_memory: Option<MemoryWeights>,
    momentum: Option<MemoryWeights>,
}

impl NeuralMemory {
    pub fn new(input_dim: usize, key_dim: usize, val_dim: usize, memory_hidden_dim: usize) -> Self {
        let mut rng = rand::rng(); // Fixed deprecated warning
        let normal = Normal::new(0.0, 0.02).unwrap();

        let w_q_data: Vec<f32> = (0..key_dim*input_dim).map(|_| normal.sample(&mut rng)).collect();
        let w_q = Array2::from_shape_vec((key_dim, input_dim), w_q_data).unwrap();

        let w_k_data: Vec<f32> = (0..key_dim*input_dim).map(|_| normal.sample(&mut rng)).collect();
        let w_k = Array2::from_shape_vec((key_dim, input_dim), w_k_data).unwrap();

        let w_v_data: Vec<f32> = (0..val_dim*input_dim).map(|_| normal.sample(&mut rng)).collect();
        let w_v = Array2::from_shape_vec((val_dim, input_dim), w_v_data).unwrap();

        let w_alpha_data: Vec<f32> = (0..input_dim).map(|_| normal.sample(&mut rng)).collect();
        let w_alpha = Array1::from_shape_vec(input_dim, w_alpha_data).unwrap();

        let w_eta_data: Vec<f32> = (0..input_dim).map(|_| normal.sample(&mut rng)).collect();
        let w_eta = Array1::from_shape_vec(input_dim, w_eta_data).unwrap();

        let w_theta_data: Vec<f32> = (0..input_dim).map(|_| normal.sample(&mut rng)).collect();
        let w_theta = Array1::from_shape_vec(input_dim, w_theta_data).unwrap();

        Self {
            input_dim,
            key_dim,
            val_dim,
            memory_hidden_dim,

            w_q,
            w_k,
            w_v,

            w_alpha,
            w_eta,
            w_theta,

            init_memory: MemoryWeights::new(key_dim, memory_hidden_dim, val_dim, &mut rng),

            curr_memory: None,
            momentum: None,
        }
    }

    /// Reset memory to initial state
    pub fn reset_memory(&mut self) {
        self.curr_memory = Some(self.init_memory.clone());
        self.momentum = Some(MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim));
    }

    fn mlp_forward(weights: &MemoryWeights, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        // h = ReLU(W1 x + b1)
        let z = weights.w1.dot(input) + &weights.b1;
        let h = z.mapv(|x| x.max(0.0)); // ReLU
        // y = W2 h + b2
        let y = weights.w2.dot(&h) + &weights.b2;
        (y, h)
    }

    /// The core mechanism of Titans: updating memory based on surprise.
    pub fn update_memory_step(&mut self, k: &Array1<f32>, v: &Array1<f32>, alpha: f32, eta: f32, theta: f32) {
         if self.curr_memory.is_none() {
             self.reset_memory();
         }

         let memory = self.curr_memory.as_ref().unwrap();

         // 1. Forward pass through Memory MLP
         // We need intermediate values for gradient
         let z = memory.w1.dot(k) + &memory.b1;
         let h = z.mapv(|x| x.max(0.0)); // ReLU
         let v_pred = memory.w2.dot(&h) + &memory.b2;

         // 2. Compute Gradient of MSE Loss: L = 0.5 * ||v_pred - v||^2
         // dL/dv_pred = v_pred - v
         let grad_output = &v_pred - v;

         // Backprop through MLP to get gradients for W1, b1, W2, b2
         // dL/dW2 = grad_output * h^T
         // dL/db2 = grad_output
         let grad_w2 = grad_output.clone().insert_axis(Axis(1)).dot(&h.clone().insert_axis(Axis(0)));
         let grad_b2 = grad_output.clone();

         // dL/dh = W2^T * grad_output
         let grad_h = memory.w2.t().dot(&grad_output);

         // dL/dz = grad_h * step(z) (ReLU derivative)
         let grad_z = grad_h * z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

         // dL/dW1 = grad_z * k^T
         // dL/db1 = grad_z
         let grad_w1 = grad_z.clone().insert_axis(Axis(1)).dot(&k.clone().insert_axis(Axis(0)));
         let grad_b1 = grad_z;

         // 3. Update Momentum S_t
         // S_t = eta * S_{t-1} - theta * grad
         let momentum = self.momentum.as_mut().unwrap();

         // Apply decay
         momentum.scale(eta);

         // Subtract scaled gradient
         // We construct a temporary gradient object
         // Note: theta is learning rate

         momentum.w1 = &momentum.w1 - &(&grad_w1 * theta);
         momentum.b1 = &momentum.b1 - &(&grad_b1 * theta);
         momentum.w2 = &momentum.w2 - &(&grad_w2 * theta);
         momentum.b2 = &momentum.b2 - &(&grad_b2 * theta);

         // 4. Update Memory M_t
         // M_t = (1 - alpha) * M_{t-1} + S_t
         let memory_mut = self.curr_memory.as_mut().unwrap();
         memory_mut.scale(1.0 - alpha);
         memory_mut.add(momentum);
    }
}

impl Layer for NeuralMemory {
    fn layer_type(&self) -> &str {
        "NeuralMemory"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Input: (Batch, Seq, Feature) or (Seq, Feature)?
        // The codebase typically uses 2D arrays (Seq/Batch, Feature) or similar.
        // Assuming input is (Seq_Len, Input_Dim).

        // Lazy initialization to support autoregressive decoding where forward is called per token.
        // For batch processing of independent sequences, ensure reset_memory() is called before forward().
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        let seq_len = input.nrows();
        let mut output = Array2::<f32>::zeros((seq_len, self.val_dim)); // Using val_dim as output dim? Or input_dim?
        // Usually attention returns same dim as input, but here we project to Val dim.
        // If this is used in MAC/MAG, the output dimension matters.
        // We'll assume the layer output is `val_dim`.

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();

            // 1. Projections
            let q_t = self.w_q.dot(&x_t);
            let k_t = self.w_k.dot(&x_t);
            let v_t = self.w_v.dot(&x_t);

            // Sigmoid for gates to keep them in [0, 1]
            fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

            let alpha_t = sigmoid(self.w_alpha.dot(&x_t));
            let eta_t = sigmoid(self.w_eta.dot(&x_t));
            let theta_t = sigmoid(self.w_theta.dot(&x_t)); // Learning rate probably shouldn't be sigmoid constrained to 1.0 if large LR needed, but 0-1 is safe.

            // 2. Inference (Retrieval) using M_{t-1}
            // M_{t-1} is in self.curr_memory (initialized to init_memory)
            let (y_t, _) = Self::mlp_forward(self.curr_memory.as_ref().unwrap(), &q_t);

            // Store output
            output.row_mut(t).assign(&y_t);

            // 3. Update Memory M_{t-1} -> M_t
            self.update_memory_step(&k_t, &v_t, alpha_t, eta_t, theta_t);
        }

        output
    }

    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        // TODO: Implement backward pass through the meta-learning process.
        // This is complex because the weights M_t depend on the history.
        // Fully training the meta-parameters (projections) requires Truncated Backpropagation Through Time (TBPTT)
        // or Real-Time Recurrent Learning (RTRL) to account for how W_Q, W_K, W_V influence the sequence of M_t updates.
        //
        // Currently, this implementation returns zeros for input gradients, effectively making the projections fixed (random)
        // reservoirs unless a simplified gradient approximation is implemented.

        // Returning zeros for input gradients for now
        Array2::zeros((grads.nrows(), self.input_dim))
    }

    fn parameters(&self) -> usize {
        // Sum of all meta-parameters
        let w_q_params = self.w_q.len();
        let w_k_params = self.w_k.len();
        let w_v_params = self.w_v.len();
        let w_gates = self.w_alpha.len() + self.w_eta.len() + self.w_theta.len();

        let memory_params =
            self.init_memory.w1.len() +
            self.init_memory.b1.len() +
            self.init_memory.w2.len() +
            self.init_memory.b2.len();

        w_q_params + w_k_params + w_v_params + w_gates + memory_params
    }

    fn weight_norm(&self) -> f32 {
        let mut sum_sq = 0.0;
        sum_sq += self.w_q.mapv(|x| x*x).sum();
        sum_sq += self.w_k.mapv(|x| x*x).sum();
        sum_sq += self.w_v.mapv(|x| x*x).sum();
        sum_sq += self.w_alpha.mapv(|x| x*x).sum();
        sum_sq += self.w_eta.mapv(|x| x*x).sum();
        sum_sq += self.w_theta.mapv(|x| x*x).sum();

        let m = &self.init_memory;
        sum_sq += m.w1.mapv(|x| x*x).sum();
        sum_sq += m.b1.mapv(|x| x*x).sum();
        sum_sq += m.w2.mapv(|x| x*x).sum();
        sum_sq += m.b2.mapv(|x| x*x).sum();

        sum_sq.sqrt()
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        _output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // TODO: Implement gradient computation
        // Return zeros for now

        // We need to return gradients for all parameters.
        // Constructing zero gradients matching shapes.
        let mut param_grads = Vec::new();
        param_grads.push(Array2::zeros(self.w_q.raw_dim()));
        param_grads.push(Array2::zeros(self.w_k.raw_dim()));
        param_grads.push(Array2::zeros(self.w_v.raw_dim()));

        // 1D arrays need to be wrapped or handled? Layer trait expects Array2.
        // Usually we reshape 1D to (1, N) or (N, 1).
        // Let's assume we treat them as Row vectors (1, N) for the interface.
        param_grads.push(self.w_alpha.clone().insert_axis(Axis(0)).mapv(|_| 0.0));
        param_grads.push(self.w_eta.clone().insert_axis(Axis(0)).mapv(|_| 0.0));
        param_grads.push(self.w_theta.clone().insert_axis(Axis(0)).mapv(|_| 0.0));

        param_grads.push(Array2::zeros(self.init_memory.w1.raw_dim()));
        param_grads.push(self.init_memory.b1.clone().insert_axis(Axis(0)).mapv(|_| 0.0));
        param_grads.push(Array2::zeros(self.init_memory.w2.raw_dim()));
        param_grads.push(self.init_memory.b2.clone().insert_axis(Axis(0)).mapv(|_| 0.0));

        (Array2::zeros((_input.nrows(), self.input_dim)), param_grads)
    }

    fn apply_gradients(
        &mut self,
        _gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()> {
        // TODO: Update the meta-parameters
        // Assuming gradients come in same order as pushed above
        if _gradients.len() != 10 {
             // Basic check
             return Ok(());
        }

        let mut idx = 0;

        // w_q
        self.w_q.scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;
        // w_k
        self.w_k.scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;
        // w_v
        self.w_v.scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;

        // w_alpha (vec)
        self.w_alpha.scaled_add(-learning_rate, &_gradients[idx].row(0));
        idx += 1;
        // w_eta (vec)
        self.w_eta.scaled_add(-learning_rate, &_gradients[idx].row(0));
        idx += 1;
        // w_theta (vec)
        self.w_theta.scaled_add(-learning_rate, &_gradients[idx].row(0));
        idx += 1;

        // init_memory
        self.init_memory.w1.scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;
        self.init_memory.b1.scaled_add(-learning_rate, &_gradients[idx].row(0));
        idx += 1;
        self.init_memory.w2.scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;
        self.init_memory.b2.scaled_add(-learning_rate, &_gradients[idx].row(0));
        // idx += 1;

        Ok(())
    }

    fn zero_gradients(&mut self) {
        // TODO: Zero gradients
    }
}
