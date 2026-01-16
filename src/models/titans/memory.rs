use ndarray::{Array1, Array2, Axis, s};
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

/// Trace of forward pass for BPTT
struct ForwardTrace {
    qs: Vec<Array1<f32>>,
    ks: Vec<Array1<f32>>,
    vs: Vec<Array1<f32>>,
    alphas: Vec<f32>,
    etas: Vec<f32>,
    thetas: Vec<f32>,
    // Memories M_t. Index t corresponds to memory AFTER update at step t.
    // We also need M_{-1} which is init_memory.
    // So memories[t] is M_t.
    memories: Vec<MemoryWeights>,
    // Momentums S_t. Index t corresponds to S_t (updated at step t).
    momentums: Vec<MemoryWeights>,
}

/// Trace of MAC forward pass
struct MacForwardTrace {
    qs: Vec<Array1<f32>>,
    ks: Vec<Array1<f32>>,
    // vs is not stored explicitly as it's computed from update_inputs, but we need it for gradient
    // computation w.r.t update_inputs. We can recompute or store. Let's store.
    vs: Vec<Array1<f32>>,
    alphas: Vec<f32>,
    etas: Vec<f32>,
    thetas: Vec<f32>,
    // Memory state used for retrieval at step t.
    // In MAC, this is M_{chunk_start} for all t in chunk.
    retrieval_memories: Vec<MemoryWeights>,
    // Memory state AFTER update at step t.
    update_memories: Vec<MemoryWeights>,
    // Momentum at step t
    momentums: Vec<MemoryWeights>,
}

/// Neural Long-Term Memory Module (LMM)
///
/// As described in "Titans: Learning to Memorize at Test Time" (Arxiv 2501.00663).
/// This module acts as a meta-learner that updates its own parameters at test time
/// based on the "surprise" (gradient) of the input data.
#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralMemory {
    // Configuration
    pub input_dim: usize,
    pub key_dim: usize,
    pub val_dim: usize,
    pub memory_hidden_dim: usize,

    // Meta-parameters (Learnable projections)
    pub w_q: Array2<f32>, // key_dim x input_dim
    pub w_k: Array2<f32>, // key_dim x input_dim
    pub w_v: Array2<f32>, // val_dim x input_dim

    // Adaptive hyperparameters projections (producing scalars)
    // We project input to 1 dimension
    pub w_alpha: Array1<f32>, // input_dim
    pub w_eta: Array1<f32>,   // input_dim
    pub w_theta: Array1<f32>, // input_dim

    // Initial Memory State (Meta-learned initialization)
    pub init_memory: MemoryWeights,

    // Current State (Evolving during forward pass)
    // In a real implementation, this should probably be transient or managed via a State struct passed to forward,
    // but Layer trait implies internal state for RNNs in this codebase.
    curr_memory: Option<MemoryWeights>,
    momentum: Option<MemoryWeights>,
}

impl NeuralMemory {
    pub fn new(input_dim: usize, key_dim: usize, val_dim: usize, memory_hidden_dim: usize) -> Self {
        let mut rng = rand::rng();
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
         let z = memory.w1.dot(k) + &memory.b1;
         let h = z.mapv(|x| x.max(0.0)); // ReLU
         let v_pred = memory.w2.dot(&h) + &memory.b2;

         // 2. Compute Gradient of MSE Loss: L = 0.5 * ||v_pred - v||^2
         // dL/dv_pred = v_pred - v
         let grad_output = &v_pred - v;

         // Backprop through MLP to get gradients for W1, b1, W2, b2
         let grad_w2 = grad_output.clone().insert_axis(Axis(1)).dot(&h.clone().insert_axis(Axis(0)));
         let grad_b2 = grad_output.clone();

         let grad_h = memory.w2.t().dot(&grad_output);
         let grad_z = grad_h * z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

         let grad_w1 = grad_z.clone().insert_axis(Axis(1)).dot(&k.clone().insert_axis(Axis(0)));
         let grad_b1 = grad_z;

         // 3. Update Momentum S_t
         let momentum = self.momentum.as_mut().unwrap();

         // Apply decay
         momentum.scale(eta);

         // Subtract scaled gradient
         momentum.w1 = &momentum.w1 - &(&grad_w1 * theta);
         momentum.b1 = &momentum.b1 - &(&grad_b1 * theta);
         momentum.w2 = &momentum.w2 - &(&grad_w2 * theta);
         momentum.b2 = &momentum.b2 - &(&grad_b2 * theta);

         // 4. Update Memory M_t
         let memory_mut = self.curr_memory.as_mut().unwrap();
         memory_mut.scale(1.0 - alpha);
         memory_mut.add(momentum);
    }

    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    /// Retrieve memory content based on input query (without updating memory)
    /// Used in MAC architecture: Retrieve h_t using input context.
    pub fn retrieve(&self, input: &Array2<f32>) -> Array2<f32> {
        let memory = self.curr_memory.as_ref().unwrap_or(&self.init_memory);

        let seq_len = input.nrows();
        let mut output = Array2::<f32>::zeros((seq_len, self.val_dim));

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();
            let q_t = self.w_q.dot(&x_t);
            let (y_t, _) = Self::mlp_forward(memory, &q_t);
            output.row_mut(t).assign(&y_t);
        }
        output
    }

    /// Update memory state based on input (typically Attention Output in MAC)
    /// Used in MAC architecture: Update Memory using Attention output.
    pub fn update(&mut self, input: &Array2<f32>) {
         if self.curr_memory.is_none() {
             self.reset_memory();
         }

         let seq_len = input.nrows();

         for t in 0..seq_len {
             let x_t = input.row(t).to_owned();

             let k_t = self.w_k.dot(&x_t);
             let v_t = self.w_v.dot(&x_t);

             let alpha_t = Self::sigmoid(self.w_alpha.dot(&x_t));
             let eta_t = Self::sigmoid(self.w_eta.dot(&x_t));
             let theta_t = Self::sigmoid(self.w_theta.dot(&x_t));

             self.update_memory_step(&k_t, &v_t, alpha_t, eta_t, theta_t);
         }
    }

    // Internal forward pass that returns trace for BPTT
    fn forward_with_trace(&self, input: &Array2<f32>) -> (Array2<f32>, ForwardTrace) {
        let seq_len = input.nrows();
        let mut output = Array2::<f32>::zeros((seq_len, self.val_dim));

        // State reconstruction
        // We use local state variables instead of modifying self, to keep it clean
        let mut curr_memory = self.init_memory.clone();
        let mut momentum = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        // Trace containers
        let mut qs = Vec::with_capacity(seq_len);
        let mut ks = Vec::with_capacity(seq_len);
        let mut vs = Vec::with_capacity(seq_len);
        let mut alphas = Vec::with_capacity(seq_len);
        let mut etas = Vec::with_capacity(seq_len);
        let mut thetas = Vec::with_capacity(seq_len);
        let mut memories = Vec::with_capacity(seq_len);
        let mut momentums = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();

            // 1. Projections
            let q_t = self.w_q.dot(&x_t);
            let k_t = self.w_k.dot(&x_t);
            let v_t = self.w_v.dot(&x_t);

            let alpha_t = Self::sigmoid(self.w_alpha.dot(&x_t));
            let eta_t = Self::sigmoid(self.w_eta.dot(&x_t));
            let theta_t = Self::sigmoid(self.w_theta.dot(&x_t));

            // Store trace inputs
            qs.push(q_t.clone());
            ks.push(k_t.clone());
            vs.push(v_t.clone());
            alphas.push(alpha_t);
            etas.push(eta_t);
            thetas.push(theta_t);

            // 2. Inference (Retrieval) using M_{t-1}
            let (y_t, _) = Self::mlp_forward(&curr_memory, &q_t);
            output.row_mut(t).assign(&y_t);

            // 3. Update Step
            // Re-implement update step logic locally to capture state

            // Forward pass for gradient (surprise)
            // L = 0.5 * ||v_pred - v||^2
            let (v_pred, h) = Self::mlp_forward(&curr_memory, &k_t); // using M_{t-1}
            let grad_output = &v_pred - &v_t;

            // Gradients w.r.t weights
             // dL/dW2 = grad_output * h^T
            let grad_w2 = grad_output.clone().insert_axis(Axis(1)).dot(&h.clone().insert_axis(Axis(0)));
            let grad_b2 = grad_output.clone();

            let grad_h = curr_memory.w2.t().dot(&grad_output);
            let z = curr_memory.w1.dot(&k_t) + &curr_memory.b1; // Recompute z
            let grad_z = grad_h * z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            let grad_w1 = grad_z.clone().insert_axis(Axis(1)).dot(&k_t.clone().insert_axis(Axis(0)));
            let grad_b1 = grad_z;

            // S_t = eta * S_{t-1} - theta * grad
            momentum.scale(eta_t);
            momentum.w1 = &momentum.w1 - &(&grad_w1 * theta_t);
            momentum.b1 = &momentum.b1 - &(&grad_b1 * theta_t);
            momentum.w2 = &momentum.w2 - &(&grad_w2 * theta_t);
            momentum.b2 = &momentum.b2 - &(&grad_b2 * theta_t);

            momentums.push(momentum.clone()); // Store S_t

            // M_t = (1 - alpha) * M_{t-1} + S_t
            curr_memory.scale(1.0 - alpha_t);
            curr_memory.add(&momentum);

            memories.push(curr_memory.clone()); // Store M_t
        }

        (output, ForwardTrace {
            qs, ks, vs, alphas, etas, thetas, memories, momentums
        })
    }

    // MAC-specific forward trace reconstruction for decoupled retrieval/update
    fn forward_mac_with_trace(
        &self,
        queries: &Array2<f32>,
        update_inputs: &Array2<f32>,
        segment_len: usize,
    ) -> MacForwardTrace {
        let seq_len = queries.nrows();
        // State reconstruction
        let mut curr_memory = self.init_memory.clone();
        let mut momentum = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        // Trace containers
        let mut qs = Vec::with_capacity(seq_len);
        let mut ks = Vec::with_capacity(seq_len);
        let mut vs = Vec::with_capacity(seq_len);
        let mut alphas = Vec::with_capacity(seq_len);
        let mut etas = Vec::with_capacity(seq_len);
        let mut thetas = Vec::with_capacity(seq_len);
        let mut retrieval_memories = Vec::with_capacity(seq_len);
        let mut update_memories = Vec::with_capacity(seq_len);
        let mut momentums = Vec::with_capacity(seq_len);

        let mut retrieval_memory_snapshot = curr_memory.clone();

        for t in 0..seq_len {
            // Check if start of segment
            if t % segment_len == 0 {
                retrieval_memory_snapshot = curr_memory.clone();
            }

            // Retrieval inputs
            let q_in = queries.row(t).to_owned();
            let q_t = self.w_q.dot(&q_in);
            qs.push(q_t);
            retrieval_memories.push(retrieval_memory_snapshot.clone());

            // Update inputs
            let u_in = update_inputs.row(t).to_owned();
            let k_t = self.w_k.dot(&u_in);
            let v_t = self.w_v.dot(&u_in);
            let alpha_t = Self::sigmoid(self.w_alpha.dot(&u_in));
            let eta_t = Self::sigmoid(self.w_eta.dot(&u_in));
            let theta_t = Self::sigmoid(self.w_theta.dot(&u_in));

            ks.push(k_t.clone());
            vs.push(v_t.clone());
            alphas.push(alpha_t);
            etas.push(eta_t);
            thetas.push(theta_t);

            // Update Step
            // Forward pass for gradient (surprise)
            // L = 0.5 * ||v_pred - v||^2
            let (v_pred, h) = Self::mlp_forward(&curr_memory, &k_t); // using M_{t-1}
            let grad_output = &v_pred - &v_t;

            // Gradients w.r.t weights
            let grad_w2 = grad_output.clone().insert_axis(Axis(1)).dot(&h.clone().insert_axis(Axis(0)));
            let grad_b2 = grad_output.clone();

            let grad_h = curr_memory.w2.t().dot(&grad_output);
            let z = curr_memory.w1.dot(&k_t) + &curr_memory.b1;
            let grad_z = grad_h * z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            let grad_w1 = grad_z.clone().insert_axis(Axis(1)).dot(&k_t.clone().insert_axis(Axis(0)));
            let grad_b1 = grad_z.clone();

            // S_t = eta * S_{t-1} - theta * grad
            momentum.scale(eta_t);
            momentum.w1 = &momentum.w1 - &(&grad_w1 * theta_t);
            momentum.b1 = &momentum.b1 - &(&grad_b1 * theta_t);
            momentum.w2 = &momentum.w2 - &(&grad_w2 * theta_t);
            momentum.b2 = &momentum.b2 - &(&grad_b2 * theta_t);

            momentums.push(momentum.clone()); // Store S_t

            // M_t = (1 - alpha) * M_{t-1} + S_t
            curr_memory.scale(1.0 - alpha_t);
            curr_memory.add(&momentum);

            update_memories.push(curr_memory.clone()); // Store M_t
        }

        MacForwardTrace {
            qs, ks, vs, alphas, etas, thetas, retrieval_memories, update_memories, momentums
        }
    }
}

impl Layer for NeuralMemory {
    fn layer_type(&self) -> &str {
        "NeuralMemory"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        let seq_len = input.nrows();
        let mut output = Array2::<f32>::zeros((seq_len, self.val_dim));

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();

            let q_t = self.w_q.dot(&x_t);
            let k_t = self.w_k.dot(&x_t);
            let v_t = self.w_v.dot(&x_t);

            let alpha_t = Self::sigmoid(self.w_alpha.dot(&x_t));
            let eta_t = Self::sigmoid(self.w_eta.dot(&x_t));
            let theta_t = Self::sigmoid(self.w_theta.dot(&x_t));

            let (y_t, _) = Self::mlp_forward(self.curr_memory.as_ref().unwrap(), &q_t);
            output.row_mut(t).assign(&y_t);

            self.update_memory_step(&k_t, &v_t, alpha_t, eta_t, theta_t);
        }

        output
    }

    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        // Standard backward interface.
        Array2::zeros((grads.nrows(), self.input_dim))
    }

    fn parameters(&self) -> usize {
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
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Fallback to split implementation with identical inputs
        // This keeps logic unified
        let (d_q, d_u, grads) = self.compute_gradients_split(input, input, output_grads, 1);
        (d_q + d_u, grads)
    }

    fn apply_gradients(
        &mut self,
        _gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()> {
        if _gradients.len() != 10 {
             return Ok(());
        }

        let mut idx = 0;

        self.w_q.scaled_add(-learning_rate, &_gradients[idx]); idx += 1;
        self.w_k.scaled_add(-learning_rate, &_gradients[idx]); idx += 1;
        self.w_v.scaled_add(-learning_rate, &_gradients[idx]); idx += 1;

        self.w_alpha.scaled_add(-learning_rate, &_gradients[idx].row(0)); idx += 1;
        self.w_eta.scaled_add(-learning_rate, &_gradients[idx].row(0)); idx += 1;
        self.w_theta.scaled_add(-learning_rate, &_gradients[idx].row(0)); idx += 1;

        self.init_memory.w1.scaled_add(-learning_rate, &_gradients[idx]); idx += 1;
        self.init_memory.b1.scaled_add(-learning_rate, &_gradients[idx].row(0)); idx += 1;
        self.init_memory.w2.scaled_add(-learning_rate, &_gradients[idx]); idx += 1;
        self.init_memory.b2.scaled_add(-learning_rate, &_gradients[idx].row(0));

        Ok(())
    }

    fn zero_gradients(&mut self) {
        // No-op as gradients are not stored in the struct
    }
}

impl NeuralMemory {
    pub fn compute_gradients_split(
        &self,
        queries: &Array2<f32>,
        update_inputs: &Array2<f32>,
        d_retrieved: &Array2<f32>,
        segment_len: usize,
    ) -> (Array2<f32>, Array2<f32>, Vec<Array2<f32>>) {
        let trace = self.forward_mac_with_trace(queries, update_inputs, segment_len);
        let seq_len = queries.nrows();

        let mut d_wq = Array2::<f32>::zeros(self.w_q.raw_dim());
        let mut d_wk = Array2::<f32>::zeros(self.w_k.raw_dim());
        let mut d_wv = Array2::<f32>::zeros(self.w_v.raw_dim());
        let mut d_w_alpha = Array1::<f32>::zeros(self.w_alpha.raw_dim());
        let mut d_w_eta = Array1::<f32>::zeros(self.w_eta.raw_dim());
        let mut d_w_theta = Array1::<f32>::zeros(self.w_theta.raw_dim());

        let mut d_init_memory = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        // State Gradients
        // d_M_next: gradient w.r.t M_t (at end of step t)
        let mut d_M_next = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);
        let mut d_S_next = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        let mut d_queries = Array2::<f32>::zeros(queries.raw_dim());
        let mut d_update_inputs = Array2::<f32>::zeros(update_inputs.raw_dim());

        // Accumulator for dL/dM_{chunk_start}
        // This collects gradients from retrieval steps in the chunk
        let mut d_M_chunk_start = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        for t in (0..seq_len).rev() {
            let dy_t = d_retrieved.row(t); // dL/dh_t (retrieved)

            // Inputs
            let q_in = queries.row(t);
            let u_in = update_inputs.row(t);

            let q_t = &trace.qs[t];
            let k_t = &trace.ks[t];
            let v_t = &trace.vs[t];
            let alpha_t = trace.alphas[t];
            let eta_t = trace.etas[t];
            let theta_t = trace.thetas[t];

            let m_prev = if t == 0 { &self.init_memory } else { &trace.update_memories[t-1] };
            let m_retrieval = &trace.retrieval_memories[t];
            let s_prev = if t == 0 {
                MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim)
            } else {
                trace.momentums[t-1].clone()
            };

            // 1. Gradients from Retrieval (y_t = MLP(M_retrieval, q_t))
            // Accumulate dL/dM_retrieval into d_M_chunk_start
            // Also compute dL/dq_in

            let z_q = m_retrieval.w1.dot(q_t) + &m_retrieval.b1;
            let h_q = z_q.mapv(|x| x.max(0.0));

            let grad_h_q = m_retrieval.w2.t().dot(&dy_t);
            let grad_z_q = &grad_h_q * z_q.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let d_qt = m_retrieval.w1.t().dot(&grad_z_q);

            d_wq = d_wq + d_qt.clone().insert_axis(Axis(1)).dot(&q_in.insert_axis(Axis(0)));
            let d_qin = self.w_q.t().dot(&d_qt);
            d_queries.row_mut(t).assign(&d_qin);

            // Accumulate to d_M_chunk_start
            d_M_chunk_start.w2 = d_M_chunk_start.w2 + dy_t.clone().insert_axis(Axis(1)).dot(&h_q.insert_axis(Axis(0)));
            d_M_chunk_start.b2 = d_M_chunk_start.b2 + &dy_t;
            d_M_chunk_start.w1 = d_M_chunk_start.w1 + grad_z_q.clone().insert_axis(Axis(1)).dot(&q_t.clone().insert_axis(Axis(0)));
            d_M_chunk_start.b1 = d_M_chunk_start.b1 + &grad_z_q;

            // 2. Gradients from Update Rule
            // This is same as standard but m_prev is used
            // d_M_next holds dL/dM_t

            // If we are at end of chunk (t+1 is start of next chunk, or t is end of seq),
            // d_M_next should receive d_M_chunk_start from next chunk?
            // Actually, M_t becomes M_{t+1}_prev.
            // If t+1 is start of new chunk, M_{t+1}_prev is also M_{chunk_start} for that new chunk.
            // So d_M_next should accumulate d_M_chunk_start IF t is end of previous chunk.

            // Wait, d_M_next flows backwards.
            // When we move from t+1 to t:
            // If t+1 was start of chunk, then M_t was M_{chunk_start} for that chunk.
            // So d_M_next must include the accumulated d_M_chunk_start from that chunk.
            // And reset d_M_chunk_start for the current chunk.

            // Check if t+1 is start of chunk (or end of sequence handling)
            // t is current step. t+1 is next step.
            // If (t+1) % segment_len == 0, then t was the last step of a chunk.
            // So M_t was the M_{chunk_start} for the *next* chunk.
            // So we add d_M_chunk_start (accumulated for next chunk) to d_M_next.
            if (t + 1) % segment_len == 0 && t + 1 < seq_len {
                 // We just finished processing the next chunk (in reverse), so d_M_chunk_start contains its gradients.
                 // Add to d_M_next.
                 d_M_next.add(&d_M_chunk_start);
                 // Reset accumulator for current chunk
                 d_M_chunk_start = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);
            }
            // Special case: if we are at the very end of sequence, d_M_chunk_start might contain gradients
            // if we assume future usage? No, output gradients are zero for future.
            // But within this loop, we accumulate into d_M_chunk_start for the *current* chunk.
            // When we hit the *start* of the current chunk (t % segment_len == 0),
            // the d_M_chunk_start accumulated so far is for *this* chunk.
            // But we need to pass it to M_{t-1}.
            // Wait, M_{chunk_start} is M_{t-1} when t is start of chunk.
            // So at t (start of chunk), we process update.
            // dL/dM_t flows to M_{t-1}.
            // AND d_M_chunk_start flows to M_{t-1}.
            // So at t where t % segment_len == 0, we add d_M_chunk_start to the gradient flowing to M_{t-1}.

            // Update logic:
            let d_M_curr = d_M_next.clone();

            // d_alpha
            let mut val_alpha = 0.0;
            val_alpha += (d_M_curr.w1.clone() * &m_prev.w1).sum();
            val_alpha += (d_M_curr.b1.clone() * &m_prev.b1).sum();
            val_alpha += (d_M_curr.w2.clone() * &m_prev.w2).sum();
            val_alpha += (d_M_curr.b2.clone() * &m_prev.b2).sum();
            let d_alpha = -val_alpha;

            // d_St
            let mut d_St = d_M_curr.clone();
            let mut scaled_s_next = d_S_next.clone();
            scaled_s_next.scale(eta_t);
            d_St.add(&scaled_s_next);

            // dL/dM_{t-1} partial from update
            d_M_next.scale(1.0 - alpha_t);

            // If t is start of chunk, add retrieval gradients to M_{t-1}
            if t % segment_len == 0 {
                d_M_next.add(&d_M_chunk_start);
                // Reset for safety, though it will be reset at next boundary check (t-1)
                // Actually, d_M_chunk_start accumulates for the *current* chunk being processed in reverse.
                // At t=start, we dump it into M_{t-1}.
                // Then we should zero it out so it doesn't double count?
                // Yes. Because we are moving to t-1 which is in previous chunk.
                d_M_chunk_start = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);
            }

            // Gradients w.r.t update inputs (u_in)
            // Flows from d_St -> G_t -> k_t, v_t
            // And from d_alpha, d_eta, d_theta -> u_in

            let mut d_uin = Array1::<f32>::zeros(u_in.len());

            // 1. d_alpha path
            let d_z_alpha = d_alpha * alpha_t * (1.0 - alpha_t);
            d_w_alpha = d_w_alpha + (u_in.mapv(|x| x * d_z_alpha));
            d_uin = d_uin + (&self.w_alpha * d_z_alpha);

            // 2. d_eta path
            let mut val_eta = 0.0;
            val_eta += (d_St.w1.clone() * &s_prev.w1).sum();
            val_eta += (d_St.b1.clone() * &s_prev.b1).sum();
            val_eta += (d_St.w2.clone() * &s_prev.w2).sum();
            val_eta += (d_St.b2.clone() * &s_prev.b2).sum();
            let d_eta = val_eta;
            let d_z_eta = d_eta * eta_t * (1.0 - eta_t);
            d_w_eta = d_w_eta + (u_in.mapv(|x| x * d_z_eta));
            d_uin = d_uin + (&self.w_eta * d_z_eta);

            // 3. d_theta path
            let z_k = m_prev.w1.dot(k_t) + &m_prev.b1;
            let h_k = z_k.mapv(|x| x.max(0.0));
            let v_pred = m_prev.w2.dot(&h_k) + &m_prev.b2;
            let delta = &v_pred - &trace.vs[t];

            let g_w2 = delta.clone().insert_axis(Axis(1)).dot(&h_k.clone().insert_axis(Axis(0)));
            let g_b2 = delta.clone();
            let grad_h_k = m_prev.w2.t().dot(&delta);
            let grad_z_k = &grad_h_k * z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let g_w1 = grad_z_k.clone().insert_axis(Axis(1)).dot(&k_t.clone().insert_axis(Axis(0)));
            let g_b1 = grad_z_k.clone();

            let mut val_theta = 0.0;
            val_theta += (d_St.w1.clone() * &g_w1).sum();
            val_theta += (d_St.b1.clone() * &g_b1).sum();
            val_theta += (d_St.w2.clone() * &g_w2).sum();
            val_theta += (d_St.b2.clone() * &g_b2).sum();
            let d_theta = -val_theta;
            let d_z_theta = d_theta * theta_t * (1.0 - theta_t);
            d_w_theta = d_w_theta + (u_in.mapv(|x| x * d_z_theta));
            d_uin = d_uin + (&self.w_theta * d_z_theta);

            // 4. d_G_t path -> k_t, v_t
            // Same U calculation as before
            let u_w1 = d_St.w1.mapv(|x| -theta_t * x);
            let u_b1 = d_St.b1.mapv(|x| -theta_t * x);
            let u_w2 = d_St.w2.mapv(|x| -theta_t * x);
            let u_b2 = d_St.b2.mapv(|x| -theta_t * x);

            let sigma_prime = z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let u_w2_t_delta = u_w2.t().dot(&delta);
            let term1_inner = &sigma_prime * &u_w2_t_delta;
            let term1 = m_prev.w1.t().dot(&term1_inner);
            let w2_t_delta = m_prev.w2.t().dot(&delta);
            let epsilon = &w2_t_delta * &sigma_prime;
            let term2 = u_w1.t().dot(&epsilon);
            let d_kt = term1 + term2;

            d_wk = d_wk + d_kt.clone().insert_axis(Axis(1)).dot(&u_in.insert_axis(Axis(0)));
            d_uin = d_uin + self.w_k.t().dot(&d_kt);

            let u_w1_k_ub1 = u_w1.dot(k_t) + &u_b1;
            let term_v_2 = m_prev.w2.dot(&(&sigma_prime * &u_w1_k_ub1));
            let term_v_1 = u_w2.dot(&h_k) + &u_b2;
            let d_vt = -(term_v_1 + term_v_2);

            d_wv = d_wv + d_vt.clone().insert_axis(Axis(1)).dot(&u_in.insert_axis(Axis(0)));
            d_uin = d_uin + self.w_v.t().dot(&d_vt);

            d_update_inputs.row_mut(t).assign(&d_uin);

            d_S_next = d_St;
        }

        d_init_memory.add(&d_M_next);

        let mut param_grads = Vec::new();
        param_grads.push(d_wq);
        param_grads.push(d_wk);
        param_grads.push(d_wv);
        param_grads.push(d_w_alpha.insert_axis(Axis(0)));
        param_grads.push(d_w_eta.insert_axis(Axis(0)));
        param_grads.push(d_w_theta.insert_axis(Axis(0)));

        param_grads.push(d_init_memory.w1);
        param_grads.push(d_init_memory.b1.insert_axis(Axis(0)));
        param_grads.push(d_init_memory.w2);
        param_grads.push(d_init_memory.b2.insert_axis(Axis(0)));

        (d_queries, d_update_inputs, param_grads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_neural_memory_gradients_non_zero() {
        let input_dim = 4;
        let key_dim = 2;
        let val_dim = 2;
        let memory_hidden_dim = 8;
        let mut memory = NeuralMemory::new(input_dim, key_dim, val_dim, memory_hidden_dim);

        // Random input
        let seq_len = 5;
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let input_vec: Vec<f32> = (0..seq_len * input_dim).map(|_| normal.sample(&mut rng)).collect();
        let input = Array2::from_shape_vec((seq_len, input_dim), input_vec).unwrap();

        // Forward
        memory.forward(&input);

        // Dummy output grads
        let output_grads_vec: Vec<f32> = (0..seq_len * val_dim).map(|_| 1.0).collect();
        let output_grads = Array2::from_shape_vec((seq_len, val_dim), output_grads_vec).unwrap();

        let (_input_grads, param_grads) = memory.compute_gradients(&input, &output_grads);

        // Check if w_q gradient is non-zero
        let w_q_grad = &param_grads[0];
        assert!(w_q_grad.iter().any(|&x| x.abs() > 1e-6), "w_q gradients are all zero!");

        // Check w_k gradient
        let w_k_grad = &param_grads[1];
        assert!(w_k_grad.iter().any(|&x| x.abs() > 1e-6), "w_k gradients are all zero!");
    }
}
