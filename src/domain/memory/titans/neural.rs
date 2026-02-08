use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::domain::network::Layer;

pub type TitansMemory = NeuralMemory;

#[derive(Debug, Clone, Default)]
pub struct NeuralMemoryStreamingWorkspace {
    // For retrieve
    pub q: Array1<f32>,
    pub z_ret: Array1<f32>,
    pub h_ret: Array1<f32>,
    pub y_ret: Array1<f32>,

    // For update
    pub k: Array1<f32>,
    pub v: Array1<f32>,
    
    // For update_memory_step internal
    pub z_upd: Array1<f32>,
    pub h_upd: Array1<f32>,
    pub v_pred: Array1<f32>,
    pub grad_output: Array1<f32>,
    
    pub grad_w2: Array2<f32>,
    pub grad_b2: Array1<f32>,
    
    pub grad_h: Array1<f32>,
    pub grad_z: Array1<f32>,
    
    pub grad_w1: Array2<f32>,
    pub grad_b1: Array1<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWeights {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

impl MemoryWeights {
    pub fn new(key_dim: usize, hidden_dim: usize, val_dim: usize, rng: &mut impl Rng) -> Self {
        let normal = Normal::new(0.0, 0.02).unwrap();

        let w1_vec: Vec<f32> = (0..hidden_dim * key_dim)
            .map(|_| normal.sample(rng))
            .collect();
        let w2_vec: Vec<f32> = (0..val_dim * hidden_dim)
            .map(|_| normal.sample(rng))
            .collect();

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

    pub fn scale(&mut self, factor: f32) {
        self.w1.mapv_inplace(|x| x * factor);
        self.b1.mapv_inplace(|x| x * factor);
        self.w2.mapv_inplace(|x| x * factor);
        self.b2.mapv_inplace(|x| x * factor);
    }

    pub fn add(&mut self, other: &MemoryWeights) {
        self.w1 = &self.w1 + &other.w1;
        self.b1 = &self.b1 + &other.b1;
        self.w2 = &self.w2 + &other.w2;
        self.b2 = &self.b2 + &other.b2;
    }
}

struct ForwardTrace {
    qs: Vec<Array1<f32>>,
    ks: Vec<Array1<f32>>,
    vs: Vec<Array1<f32>>,
    alphas: Vec<f32>,
    etas: Vec<f32>,
    thetas: Vec<f32>,
    memories: Vec<MemoryWeights>,
    momentums: Vec<MemoryWeights>,
}

struct MacForwardTrace {
    qs: Vec<Array1<f32>>,
    ks: Vec<Array1<f32>>,
    vs: Vec<Array1<f32>>,
    alphas: Vec<f32>,
    etas: Vec<f32>,
    thetas: Vec<f32>,
    retrieval_memories: Vec<MemoryWeights>,
    update_memories: Vec<MemoryWeights>,
    momentums: Vec<MemoryWeights>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMemory {
    pub input_dim: usize,
    pub key_dim: usize,
    pub val_dim: usize,
    pub memory_hidden_dim: usize,

    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,

    pub w_alpha: Array1<f32>,
    pub w_eta: Array1<f32>,
    pub w_theta: Array1<f32>,

    pub init_memory: MemoryWeights,

    #[serde(skip)]
    curr_memory: Option<MemoryWeights>,

    #[serde(skip)]
    momentum: Option<MemoryWeights>,
}

impl NeuralMemory {
    pub fn new(input_dim: usize, key_dim: usize, val_dim: usize, memory_hidden_dim: usize) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let w_q_data: Vec<f32> = (0..key_dim * input_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let w_q = Array2::from_shape_vec((key_dim, input_dim), w_q_data).unwrap();

        let w_k_data: Vec<f32> = (0..key_dim * input_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let w_k = Array2::from_shape_vec((key_dim, input_dim), w_k_data).unwrap();

        let w_v_data: Vec<f32> = (0..val_dim * input_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
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

    pub fn reset_memory(&mut self) {
        self.curr_memory = Some(self.init_memory.clone());
        self.momentum = Some(MemoryWeights::zeros(
            self.key_dim,
            self.memory_hidden_dim,
            self.val_dim,
        ));
    }

    fn mlp_forward(weights: &MemoryWeights, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let z = weights.w1.dot(input) + &weights.b1;
        let h = z.mapv(|x| x.max(0.0));
        let y = weights.w2.dot(&h) + &weights.b2;
        (y, h)
    }

    pub fn update_memory_step(
        &mut self,
        k: &Array1<f32>,
        v: &Array1<f32>,
        alpha: f32,
        eta: f32,
        theta: f32,
    ) {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        let memory = self.curr_memory.as_ref().unwrap();

        let z = memory.w1.dot(k) + &memory.b1;
        let h = z.mapv(|x| x.max(0.0));
        let v_pred = memory.w2.dot(&h) + &memory.b2;

        let grad_output = &v_pred - v;

        let grad_w2 = grad_output
            .clone()
            .insert_axis(Axis(1))
            .dot(&h.clone().insert_axis(Axis(0)));
        let grad_b2 = grad_output.clone();

        let grad_h = memory.w2.t().dot(&grad_output);
        let grad_z = grad_h * z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        let grad_w1 = grad_z
            .clone()
            .insert_axis(Axis(1))
            .dot(&k.clone().insert_axis(Axis(0)));
        let grad_b1 = grad_z;

        let momentum = self.momentum.as_mut().unwrap();

        momentum.scale(eta);

        momentum.w1 = &momentum.w1 - &(&grad_w1 * theta);
        momentum.b1 = &momentum.b1 - &(&grad_b1 * theta);
        momentum.w2 = &momentum.w2 - &(&grad_w2 * theta);
        momentum.b2 = &momentum.b2 - &(&grad_b2 * theta);

        let memory_mut = self.curr_memory.as_mut().unwrap();
        memory_mut.scale(1.0 - alpha);
        memory_mut.add(momentum);

        if cfg!(debug_assertions) {
             println!("Batch Update: k[0]={:.6} alpha={:.6} theta={:.6} Grad_w1_sum={:.6} M_w1_sum={:.6} Mem_w1_sum={:.6}", 
                k[0], alpha, theta, grad_w1.sum(), self.momentum.as_ref().unwrap().w1.sum(), self.curr_memory.as_ref().unwrap().w1.sum());
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn retrieve(&self, input: &Array2<f32>) -> Array2<f32> {
        let memory = self.curr_memory.as_ref().unwrap_or(&self.init_memory);

        let seq_len = input.nrows();
        let mut output = Array2::<f32>::zeros((seq_len, self.val_dim));

        for t in 0..seq_len {
            let x_t = input.row(t);
            let q_t = self.w_q.dot(&x_t);
            let (y_t, _) = Self::mlp_forward(memory, &q_t);
            output.row_mut(t).assign(&y_t);
        }
        output
    }

    /// Retrieve memory for a single step (query input -> memory output)
    pub fn retrieve_step(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut out = Array1::zeros(self.val_dim);
        let mut ws = NeuralMemoryStreamingWorkspace {
            q: Array1::zeros(self.key_dim),
            z_ret: Array1::zeros(self.memory_hidden_dim),
            h_ret: Array1::zeros(self.memory_hidden_dim),
            y_ret: Array1::zeros(self.val_dim),
            ..Default::default()
        };
        self.retrieve_step_into(&input.view(), &mut out, &mut ws);
        out
    }

    /// Retrieve memory for a single step into output buffer (zero allocation)
    pub fn retrieve_step_into(
        &self, 
        input: &ndarray::ArrayView1<f32>, 
        output: &mut Array1<f32>,
        ws: &mut NeuralMemoryStreamingWorkspace
    ) {
        let memory = self.curr_memory.as_ref().unwrap_or(&self.init_memory);
        
        // q = W_q * input
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_q, input, 0.0, &mut ws.q);
        
        // MLP Forward with workspace
        // z = W1 * q + b1
        ndarray::linalg::general_mat_vec_mul(1.0, &memory.w1, &ws.q, 0.0, &mut ws.z_ret);
        ws.z_ret += &memory.b1;
        
        // h = ReLU(z)
        ws.h_ret.assign(&ws.z_ret);
        ws.h_ret.mapv_inplace(|x| x.max(0.0));
        
        // y = W2 * h + b2
        ndarray::linalg::general_mat_vec_mul(1.0, &memory.w2, &ws.h_ret, 0.0, &mut ws.y_ret);
        ws.y_ret += &memory.b2;
        
        output.assign(&ws.y_ret);
    }

    pub fn update(&mut self, input: &Array2<f32>) {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        let seq_len = input.nrows();

        for t in 0..seq_len {
            let x_t = input.row(t);

            let k_t = self.w_k.dot(&x_t);
            let v_t = self.w_v.dot(&x_t);

            let alpha_t = Self::sigmoid(self.w_alpha.dot(&x_t));
            let eta_t = Self::sigmoid(self.w_eta.dot(&x_t));
            let theta_t = Self::sigmoid(self.w_theta.dot(&x_t));

            self.update_memory_step(&k_t, &v_t, alpha_t, eta_t, theta_t);
        }
    }

    /// Update memory for a single step
    pub fn update_step(&mut self, input: &Array1<f32>) {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        let k = self.w_k.dot(input);
        let v = self.w_v.dot(input);

        let alpha = Self::sigmoid(self.w_alpha.dot(input));
        let eta = Self::sigmoid(self.w_eta.dot(input));
        let theta = Self::sigmoid(self.w_theta.dot(input));

        self.update_memory_step(&k, &v, alpha, eta, theta);
    }

    pub fn update_step_with_workspace(
        &mut self, 
        input: &ndarray::ArrayView1<f32>, 
        ws: &mut NeuralMemoryStreamingWorkspace
    ) {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        // k = W_k * input
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_k, input, 0.0, &mut ws.k);
        
        // v = W_v * input
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_v, input, 0.0, &mut ws.v);

        let alpha = Self::sigmoid(self.w_alpha.dot(input));
        let eta = Self::sigmoid(self.w_eta.dot(input));
        let theta = Self::sigmoid(self.w_theta.dot(input));

        self.update_memory_step_with_workspace(alpha, eta, theta, ws);
    }

    pub fn update_memory_step_with_workspace(
        &mut self,
        alpha: f32,
        eta: f32,
        theta: f32,
        ws: &mut NeuralMemoryStreamingWorkspace,
    ) {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        let memory = self.curr_memory.as_ref().unwrap();

        // z = W1 * k + b1
        // k is already in ws.k from update_step_with_workspace
        ndarray::linalg::general_mat_vec_mul(1.0, &memory.w1, &ws.k, 0.0, &mut ws.z_upd);
        ws.z_upd += &memory.b1;

        // h = ReLU(z)
        ws.h_upd.assign(&ws.z_upd);
        ws.h_upd.mapv_inplace(|x| x.max(0.0));

        // v_pred = W2 * h + b2
        ndarray::linalg::general_mat_vec_mul(1.0, &memory.w2, &ws.h_upd, 0.0, &mut ws.v_pred);
        ws.v_pred += &memory.b2;

        // grad_output = v_pred - v
        // v is in ws.v
        ws.grad_output.assign(&ws.v_pred);
        ws.grad_output -= &ws.v;

        // grad_w2 = grad_output * h^T (Outer Product)
        // (V, 1) * (1, H) -> (V, H)
        ndarray::linalg::general_mat_mul(
            1.0,
            &ws.grad_output.view().insert_axis(Axis(1)),
            &ws.h_upd.view().insert_axis(Axis(0)),
            0.0,
            &mut ws.grad_w2
        );

        // grad_b2 = grad_output
        ws.grad_b2.assign(&ws.grad_output);

        // grad_h = W2^T * grad_output
        ndarray::linalg::general_mat_vec_mul(1.0, &memory.w2.t(), &ws.grad_output, 0.0, &mut ws.grad_h);

        // grad_z = grad_h * step(z)
        ws.grad_z.assign(&ws.grad_h);
        ndarray::Zip::from(&mut ws.grad_z)
            .and(&ws.z_upd)
            .for_each(|gz, &z| {
                if z <= 0.0 { *gz = 0.0; }
            });

        // grad_w1 = grad_z * k^T (Outer Product)
        // (H, 1) * (1, K) -> (H, K)
        ndarray::linalg::general_mat_mul(
            1.0,
            &ws.grad_z.view().insert_axis(Axis(1)),
            &ws.k.view().insert_axis(Axis(0)),
            0.0,
            &mut ws.grad_w1
        );

        // grad_b1 = grad_z
        ws.grad_b1.assign(&ws.grad_z);

        // Update Momentum
        let momentum = self.momentum.as_mut().unwrap();
        momentum.scale(eta);

        momentum.w1.scaled_add(-theta, &ws.grad_w1);
        momentum.b1.scaled_add(-theta, &ws.grad_b1);
        momentum.w2.scaled_add(-theta, &ws.grad_w2);
        momentum.b2.scaled_add(-theta, &ws.grad_b2);

        // Update Memory
        let memory_mut = self.curr_memory.as_mut().unwrap();
        memory_mut.scale(1.0 - alpha);
        memory_mut.add(momentum);

        if cfg!(debug_assertions) {
             println!("Stream Update: k[0]={:.6} alpha={:.6} theta={:.6} Grad_w1_sum={:.6} M_w1_sum={:.6} Mem_w1_sum={:.6}", 
                ws.k[0], alpha, theta, ws.grad_w1.sum(), self.momentum.as_ref().unwrap().w1.sum(), self.curr_memory.as_ref().unwrap().w1.sum());
        }
    }

    fn forward_with_trace(&self, input: &Array2<f32>) -> (Array2<f32>, ForwardTrace) {
        let seq_len = input.nrows();
        let mut output = Array2::<f32>::zeros((seq_len, self.val_dim));

        let mut curr_memory = self.init_memory.clone();
        let mut momentum = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        let mut qs = Vec::with_capacity(seq_len);
        let mut ks = Vec::with_capacity(seq_len);
        let mut vs = Vec::with_capacity(seq_len);
        let mut alphas = Vec::with_capacity(seq_len);
        let mut etas = Vec::with_capacity(seq_len);
        let mut thetas = Vec::with_capacity(seq_len);
        let mut memories = Vec::with_capacity(seq_len);
        let mut momentums = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input.row(t);

            let q_t = self.w_q.dot(&x_t);
            let k_t = self.w_k.dot(&x_t);
            let v_t = self.w_v.dot(&x_t);

            let alpha_t = Self::sigmoid(self.w_alpha.dot(&x_t));
            let eta_t = Self::sigmoid(self.w_eta.dot(&x_t));
            let theta_t = Self::sigmoid(self.w_theta.dot(&x_t));

            qs.push(q_t.clone());
            ks.push(k_t.clone());
            vs.push(v_t.clone());
            alphas.push(alpha_t);
            etas.push(eta_t);
            thetas.push(theta_t);

            let (y_t, _) = Self::mlp_forward(&curr_memory, &q_t);
            output.row_mut(t).assign(&y_t);

            let (v_pred, h) = Self::mlp_forward(&curr_memory, &k_t);
            let grad_output = &v_pred - &v_t;

            let grad_w2 = grad_output
                .clone()
                .insert_axis(Axis(1))
                .dot(&h.clone().insert_axis(Axis(0)));
            let grad_b2 = grad_output.clone();

            let grad_h = curr_memory.w2.t().dot(&grad_output);
            let z = curr_memory.w1.dot(&k_t) + &curr_memory.b1;
            let grad_z = grad_h * z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            let grad_w1 = grad_z
                .clone()
                .insert_axis(Axis(1))
                .dot(&k_t.clone().insert_axis(Axis(0)));
            let grad_b1 = grad_z;

            momentum.scale(eta_t);
            momentum.w1 = &momentum.w1 - &(&grad_w1 * theta_t);
            momentum.b1 = &momentum.b1 - &(&grad_b1 * theta_t);
            momentum.w2 = &momentum.w2 - &(&grad_w2 * theta_t);
            momentum.b2 = &momentum.b2 - &(&grad_b2 * theta_t);

            momentums.push(momentum.clone());

            curr_memory.scale(1.0 - alpha_t);
            curr_memory.add(&momentum);

            memories.push(curr_memory.clone());
        }

        (
            output,
            ForwardTrace {
                qs,
                ks,
                vs,
                alphas,
                etas,
                thetas,
                memories,
                momentums,
            },
        )
    }

    /// Process a single time step using a workspace to minimize allocations.
    pub fn forward_step_with_workspace(
        &mut self,
        input: &Array1<f32>,
        ws: &mut NeuralMemoryStreamingWorkspace,
    ) -> Array1<f32> {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        // 1. Projections into workspace
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_q, input, 0.0, &mut ws.q);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_k, input, 0.0, &mut ws.k);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_v, input, 0.0, &mut ws.v);

        let alpha = Self::sigmoid(self.w_alpha.dot(input));
        let eta = Self::sigmoid(self.w_eta.dot(input));
        let theta = Self::sigmoid(self.w_theta.dot(input));

        // 2. Retrieve from current memory
        let memory = self.curr_memory.as_ref().expect("Memory not initialized");

        // z = W1 * q + b1
        ndarray::linalg::general_mat_vec_mul(1.0, &memory.w1, &ws.q, 0.0, &mut ws.z_ret);
        ws.z_ret += &memory.b1;

        // h = ReLU(z)
        ws.h_ret.assign(&ws.z_ret);
        ws.h_ret.mapv_inplace(|x| x.max(0.0));

        // y = W2 * h + b2
        ndarray::linalg::general_mat_vec_mul(1.0, &memory.w2, &ws.h_ret, 0.0, &mut ws.y_ret);
        ws.y_ret += &memory.b2;

        let output = ws.y_ret.clone();

        // 3. Update memory
        self.update_memory_step_with_workspace(alpha, eta, theta, ws);

        output
    }

    /// Process a single time step, updating memory and returning prediction.
    /// Useful for streaming/online inference where the sequence is processed token-by-token.
    ///
    /// # Arguments
    /// * `input` - Input vector for the current time step (size: input_dim)
    ///
    /// # Returns
    /// * Prediction vector for the current time step (size: val_dim)
    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        // 1. Projections
        let q = self.w_q.dot(input);
        let k = self.w_k.dot(input);
        let v = self.w_v.dot(input);

        let alpha = Self::sigmoid(self.w_alpha.dot(input));
        let eta = Self::sigmoid(self.w_eta.dot(input));
        let theta = Self::sigmoid(self.w_theta.dot(input));

        // 2. Retrieve from current memory
        // Scope the borrow of curr_memory to ensure it ends before update_memory_step
        let y = {
            let memory = self.curr_memory.as_ref().expect("Memory not initialized");
            let (y, _) = Self::mlp_forward(memory, &q);
            y
        };

        // 3. Update memory
        self.update_memory_step(&k, &v, alpha, eta, theta);

        y
    }

    fn forward_mac_with_trace(
        &self,
        queries: &Array2<f32>,
        update_inputs: &Array2<f32>,
        segment_len: usize,
    ) -> MacForwardTrace {
        let seq_len = queries.nrows();
        let mut curr_memory = self.init_memory.clone();
        let mut momentum = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

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
            if t % segment_len == 0 {
                retrieval_memory_snapshot = curr_memory.clone();
            }

            let q_in = queries.row(t);
            let q_t = self.w_q.dot(&q_in);
            qs.push(q_t);
            retrieval_memories.push(retrieval_memory_snapshot.clone());

            let u_in = update_inputs.row(t);
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

            let (v_pred, h) = Self::mlp_forward(&curr_memory, &k_t);
            let grad_output = &v_pred - &v_t;

            let grad_w2 = grad_output
                .clone()
                .insert_axis(Axis(1))
                .dot(&h.clone().insert_axis(Axis(0)));
            let grad_b2 = grad_output.clone();

            let grad_h = curr_memory.w2.t().dot(&grad_output);
            let z = curr_memory.w1.dot(&k_t) + &curr_memory.b1;
            let grad_z = grad_h * z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            let grad_w1 = grad_z
                .clone()
                .insert_axis(Axis(1))
                .dot(&k_t.clone().insert_axis(Axis(0)));
            let grad_b1 = grad_z.clone();

            momentum.scale(eta_t);
            momentum.w1 = &momentum.w1 - &(&grad_w1 * theta_t);
            momentum.b1 = &momentum.b1 - &(&grad_b1 * theta_t);
            momentum.w2 = &momentum.w2 - &(&grad_w2 * theta_t);
            momentum.b2 = &momentum.b2 - &(&grad_b2 * theta_t);

            momentums.push(momentum.clone());

            curr_memory.scale(1.0 - alpha_t);
            curr_memory.add(&momentum);

            update_memories.push(curr_memory.clone());
        }

        MacForwardTrace {
            qs,
            ks,
            vs,
            alphas,
            etas,
            thetas,
            retrieval_memories,
            update_memories,
            momentums,
        }
    }

    pub fn forward_optimized(&mut self, input: &Array2<f32>) -> Array2<f32> {
        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        let seq_len = input.nrows();
        
        // 1. Vectorized Projections
        // (T, In) x (In, K)^T -> (T, K)
        let q_all = input.dot(&self.w_q.t());
        let k_all = input.dot(&self.w_k.t());
        let v_all = input.dot(&self.w_v.t());
        
        let mut alpha_all = input.dot(&self.w_alpha); // (T)
        let mut eta_all = input.dot(&self.w_eta);     // (T)
        let mut theta_all = input.dot(&self.w_theta); // (T)
        
        // Apply sigmoid activation
        alpha_all.mapv_inplace(Self::sigmoid);
        eta_all.mapv_inplace(Self::sigmoid);
        theta_all.mapv_inplace(Self::sigmoid);

        let mut output = Array2::<f32>::zeros((seq_len, self.val_dim));
        
        // Workspace for inner loop to avoid allocations
        let mut ws = NeuralMemoryStreamingWorkspace {
            q: Array1::zeros(self.key_dim),
            z_ret: Array1::zeros(self.memory_hidden_dim),
            h_ret: Array1::zeros(self.memory_hidden_dim),
            y_ret: Array1::zeros(self.val_dim),
            
            k: Array1::zeros(self.key_dim),
            v: Array1::zeros(self.val_dim), // v dim is val_dim!
            
            z_upd: Array1::zeros(self.memory_hidden_dim),
            h_upd: Array1::zeros(self.memory_hidden_dim),
            v_pred: Array1::zeros(self.val_dim),
            grad_output: Array1::zeros(self.val_dim),
            
            grad_w2: Array2::zeros((self.val_dim, self.memory_hidden_dim)),
            grad_b2: Array1::zeros(self.val_dim),
            
            grad_h: Array1::zeros(self.memory_hidden_dim),
            grad_z: Array1::zeros(self.memory_hidden_dim),
            
            grad_w1: Array2::zeros((self.memory_hidden_dim, self.key_dim)),
            grad_b1: Array1::zeros(self.memory_hidden_dim),
        };

        // 2. Sequential Memory Update
        for t in 0..seq_len {
            // Retrieve inputs from pre-calculated arrays
            ws.q.assign(&q_all.row(t));
            
            // Retrieve: y = Memory(q)
            let memory = self.curr_memory.as_ref().unwrap();
            
            // MLP Forward (Retrieve)
            // z = W1 * q + b1
            ndarray::linalg::general_mat_vec_mul(1.0, &memory.w1, &ws.q, 0.0, &mut ws.z_ret);
            ws.z_ret += &memory.b1;
            
            // h = ReLU(z)
            ws.h_ret.assign(&ws.z_ret);
            ws.h_ret.mapv_inplace(|x| x.max(0.0));
            
            // y = W2 * h + b2
            ndarray::linalg::general_mat_vec_mul(1.0, &memory.w2, &ws.h_ret, 0.0, &mut ws.y_ret);
            ws.y_ret += &memory.b2;
            
            // Store output
            output.row_mut(t).assign(&ws.y_ret);
            
            // Update: Memory.update(k, v, alpha, eta, theta)
            ws.k.assign(&k_all.row(t));
            ws.v.assign(&v_all.row(t));
            
            let alpha = alpha_all[t];
            let eta = eta_all[t];
            let theta = theta_all[t];
            
            self.update_memory_step_with_workspace(alpha, eta, theta, &mut ws);
        }
        
        output
    }

    pub fn gradient_count(&self) -> usize {
        10
    }
}

impl Layer for NeuralMemory {
    fn layer_type(&self) -> &str {
        "NeuralMemory"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_optimized(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        Array2::zeros((grads.nrows(), self.input_dim))
    }

    fn parameters(&self) -> usize {
        let w_q_params = self.w_q.len();
        let w_k_params = self.w_k.len();
        let w_v_params = self.w_v.len();
        let w_gates = self.w_alpha.len() + self.w_eta.len() + self.w_theta.len();

        let memory_params = self.init_memory.w1.len()
            + self.init_memory.b1.len()
            + self.init_memory.w2.len()
            + self.init_memory.b2.len();

        w_q_params + w_k_params + w_v_params + w_gates + memory_params
    }

    fn weight_norm(&self) -> f32 {
        let mut sum_sq = 0.0;
        sum_sq += self.w_q.mapv(|x| x * x).sum();
        sum_sq += self.w_k.mapv(|x| x * x).sum();
        sum_sq += self.w_v.mapv(|x| x * x).sum();
        sum_sq += self.w_alpha.mapv(|x| x * x).sum();
        sum_sq += self.w_eta.mapv(|x| x * x).sum();
        sum_sq += self.w_theta.mapv(|x| x * x).sum();

        let m = &self.init_memory;
        sum_sq += m.w1.mapv(|x| x * x).sum();
        sum_sq += m.b1.mapv(|x| x * x).sum();
        sum_sq += m.w2.mapv(|x| x * x).sum();
        sum_sq += m.b2.mapv(|x| x * x).sum();

        sum_sq.sqrt()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let (_, trace) = self.forward_with_trace(input);
        let seq_len = input.nrows();

        let mut d_wq = Array2::<f32>::zeros(self.w_q.raw_dim());
        let mut d_wk = Array2::<f32>::zeros(self.w_k.raw_dim());
        let mut d_wv = Array2::<f32>::zeros(self.w_v.raw_dim());
        let mut d_w_alpha = Array1::<f32>::zeros(self.w_alpha.raw_dim());
        let mut d_w_eta = Array1::<f32>::zeros(self.w_eta.raw_dim());
        let mut d_w_theta = Array1::<f32>::zeros(self.w_theta.raw_dim());

        let mut d_init_memory =
            MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        let mut d_m_next = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);
        let mut d_s_next = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        let mut input_grads = Array2::<f32>::zeros(input.raw_dim());

        for t in (0..seq_len).rev() {
            let x_t = input.row(t);
            let dy_t = output_grads.row(t);

            let q_t = &trace.qs[t];
            let k_t = &trace.ks[t];
            let alpha_t = trace.alphas[t];
            let eta_t = trace.etas[t];
            let theta_t = trace.thetas[t];

            let m_prev = if t == 0 {
                &self.init_memory
            } else {
                &trace.memories[t - 1]
            };
            let s_prev = if t == 0 {
                MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim)
            } else {
                trace.momentums[t - 1].clone()
            };

            let d_m_curr = d_m_next.clone();

            let mut val_alpha = 0.0;
            val_alpha += (d_m_curr.w1.clone() * &m_prev.w1).sum();
            val_alpha += (d_m_curr.b1.clone() * &m_prev.b1).sum();
            val_alpha += (d_m_curr.w2.clone() * &m_prev.w2).sum();
            val_alpha += (d_m_curr.b2.clone() * &m_prev.b2).sum();
            let d_alpha = -val_alpha;

            let mut d_s_t = d_m_curr.clone();
            let mut scaled_s_next = d_s_next.clone();
            scaled_s_next.scale(eta_t);
            d_s_t.add(&scaled_s_next);

            d_m_next.scale(1.0 - alpha_t);

            let z_q = m_prev.w1.dot(q_t) + &m_prev.b1;
            let h_q = z_q.mapv(|x| x.max(0.0));

            let grad_h_q = m_prev.w2.t().dot(&dy_t);
            let grad_z_q = &grad_h_q * z_q.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let d_qt = m_prev.w1.t().dot(&grad_z_q);

            d_wq = d_wq
                + d_qt
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&x_t.insert_axis(Axis(0)));
            let mut d_xt = self.w_q.t().dot(&d_qt);

            d_m_next.w2 = d_m_next.w2 + dy_t.insert_axis(Axis(1)).dot(&h_q.insert_axis(Axis(0)));
            d_m_next.b2.zip_mut_with(&dy_t, |a, &b| *a += b);
            d_m_next.w1 = d_m_next.w1
                + grad_z_q
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&q_t.clone().insert_axis(Axis(0)));
            d_m_next.b1 += &grad_z_q;

            let d_z_alpha = d_alpha * alpha_t * (1.0 - alpha_t);
            d_w_alpha = d_w_alpha + (&x_t * d_z_alpha);
            d_xt = d_xt + (&self.w_alpha * d_z_alpha);

            let mut val_eta = 0.0;
            val_eta += (d_s_t.w1.clone() * &s_prev.w1).sum();
            val_eta += (d_s_t.b1.clone() * &s_prev.b1).sum();
            val_eta += (d_s_t.w2.clone() * &s_prev.w2).sum();
            val_eta += (d_s_t.b2.clone() * &s_prev.b2).sum();
            let d_eta = val_eta;

            let d_z_eta = d_eta * eta_t * (1.0 - eta_t);
            d_w_eta = d_w_eta + (&x_t * d_z_eta);
            d_xt = d_xt + (&self.w_eta * d_z_eta);

            let z_k = m_prev.w1.dot(k_t) + &m_prev.b1;
            let h_k = z_k.mapv(|x| x.max(0.0));
            let v_pred = m_prev.w2.dot(&h_k) + &m_prev.b2;
            let delta = &v_pred - &trace.vs[t];

            let g_w2 = delta
                .clone()
                .insert_axis(Axis(1))
                .dot(&h_k.clone().insert_axis(Axis(0)));
            let g_b2 = delta.clone();

            let grad_h_k = m_prev.w2.t().dot(&delta);
            let grad_z_k = &grad_h_k * z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let g_w1 = grad_z_k
                .clone()
                .insert_axis(Axis(1))
                .dot(&k_t.clone().insert_axis(Axis(0)));
            let g_b1 = grad_z_k.clone();

            let mut val_theta = 0.0;
            val_theta += (d_s_t.w1.clone() * &g_w1).sum();
            val_theta += (d_s_t.b1.clone() * &g_b1).sum();
            val_theta += (d_s_t.w2.clone() * &g_w2).sum();
            val_theta += (d_s_t.b2.clone() * &g_b2).sum();
            let d_theta = -val_theta;

            let d_z_theta = d_theta * theta_t * (1.0 - theta_t);
            d_w_theta = d_w_theta + (&x_t * d_z_theta);
            d_xt = d_xt + (&self.w_theta * d_z_theta);

            let u_w1 = d_s_t.w1.mapv(|x| -theta_t * x);
            let u_b1 = d_s_t.b1.mapv(|x| -theta_t * x);
            let u_w2 = d_s_t.w2.mapv(|x| -theta_t * x);
            let u_b2 = d_s_t.b2.mapv(|x| -theta_t * x);

            let sigma_prime = z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            let u_w2_t_delta = u_w2.t().dot(&delta);
            let term1_inner = &sigma_prime * &u_w2_t_delta;
            let term1 = m_prev.w1.t().dot(&term1_inner);

            let w2_t_delta = m_prev.w2.t().dot(&delta);
            let epsilon = &w2_t_delta * &sigma_prime;
            let term2 = u_w1.t().dot(&epsilon);

            let d_kt = term1 + term2;

            d_wk = d_wk
                + d_kt
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&x_t.insert_axis(Axis(0)));
            d_xt = d_xt + self.w_k.t().dot(&d_kt);

            let u_w1_k_ub1 = u_w1.dot(k_t) + &u_b1;
            let term_v_2 = m_prev.w2.dot(&(&sigma_prime * &u_w1_k_ub1));
            let term_v_1 = u_w2.dot(&h_k) + &u_b2;
            let d_vt = -(term_v_1 + term_v_2);

            d_wv = d_wv
                + d_vt
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&x_t.insert_axis(Axis(0)));
            d_xt = d_xt + self.w_v.t().dot(&d_vt);

            input_grads.row_mut(t).assign(&d_xt);

            d_s_next = d_s_t;
        }

        d_init_memory.add(&d_m_next);

        let param_grads = vec![
            d_wq,
            d_wk,
            d_wv,
            d_w_alpha.insert_axis(Axis(0)),
            d_w_eta.insert_axis(Axis(0)),
            d_w_theta.insert_axis(Axis(0)),
            d_init_memory.w1,
            d_init_memory.b1.insert_axis(Axis(0)),
            d_init_memory.w2,
            d_init_memory.b2.insert_axis(Axis(0)),
        ];

        (input_grads, param_grads)
    }

    fn apply_gradients(
        &mut self,
        _gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::common::errors::Result<()> {
        if _gradients.len() != 10 {
            return Ok(());
        }

        let mut idx = 0;

        self.w_q.scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;
        self.w_k.scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;
        self.w_v.scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;

        self.w_alpha
            .scaled_add(-learning_rate, &_gradients[idx].row(0));
        idx += 1;
        self.w_eta
            .scaled_add(-learning_rate, &_gradients[idx].row(0));
        idx += 1;
        self.w_theta
            .scaled_add(-learning_rate, &_gradients[idx].row(0));
        idx += 1;

        self.init_memory
            .w1
            .scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;
        self.init_memory
            .b1
            .scaled_add(-learning_rate, &_gradients[idx].row(0));
        idx += 1;
        self.init_memory
            .w2
            .scaled_add(-learning_rate, &_gradients[idx]);
        idx += 1;
        self.init_memory
            .b2
            .scaled_add(-learning_rate, &_gradients[idx].row(0));

        Ok(())
    }

    fn zero_gradients(&mut self) {}
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

        let mut d_init_memory =
            MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        let mut d_m_next = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);
        let mut d_s_next = MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        let mut d_queries = Array2::<f32>::zeros(queries.raw_dim());
        let mut d_update_inputs = Array2::<f32>::zeros(update_inputs.raw_dim());

        let mut d_m_chunk_start =
            MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);

        for t in (0..seq_len).rev() {
            let dy_t = d_retrieved.row(t);

            let q_in = queries.row(t);
            let u_in = update_inputs.row(t);

            let q_t = &trace.qs[t];
            let k_t = &trace.ks[t];
            let alpha_t = trace.alphas[t];
            let eta_t = trace.etas[t];
            let theta_t = trace.thetas[t];

            let m_prev = if t == 0 {
                &self.init_memory
            } else {
                &trace.update_memories[t - 1]
            };
            let m_retrieval = &trace.retrieval_memories[t];
            let s_prev = if t == 0 {
                MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim)
            } else {
                trace.momentums[t - 1].clone()
            };

            let z_q = m_retrieval.w1.dot(q_t) + &m_retrieval.b1;
            let h_q = z_q.mapv(|x| x.max(0.0));

            let grad_h_q = m_retrieval.w2.t().dot(&dy_t);
            let grad_z_q = &grad_h_q * z_q.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let d_qt = m_retrieval.w1.t().dot(&grad_z_q);

            d_wq = d_wq
                + d_qt
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&q_in.insert_axis(Axis(0)));
            let d_qin = self.w_q.t().dot(&d_qt);
            d_queries.row_mut(t).assign(&d_qin);

            d_m_chunk_start.w2 =
                d_m_chunk_start.w2 + dy_t.insert_axis(Axis(1)).dot(&h_q.insert_axis(Axis(0)));
            d_m_chunk_start.b2.zip_mut_with(&dy_t, |a, &b| *a += b);
            d_m_chunk_start.w1 = d_m_chunk_start.w1
                + grad_z_q
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&q_t.clone().insert_axis(Axis(0)));
            d_m_chunk_start.b1 += &grad_z_q;

            if (t + 1) % segment_len == 0 && t + 1 < seq_len {
                d_m_next.add(&d_m_chunk_start);
                d_m_chunk_start =
                    MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);
            }

            let d_m_curr = d_m_next.clone();

            let mut val_alpha = 0.0;
            val_alpha += (d_m_curr.w1.clone() * &m_prev.w1).sum();
            val_alpha += (d_m_curr.b1.clone() * &m_prev.b1).sum();
            val_alpha += (d_m_curr.w2.clone() * &m_prev.w2).sum();
            val_alpha += (d_m_curr.b2.clone() * &m_prev.b2).sum();
            let d_alpha = -val_alpha;

            let mut d_s_t = d_m_curr.clone();
            let mut scaled_s_next = d_s_next.clone();
            scaled_s_next.scale(eta_t);
            d_s_t.add(&scaled_s_next);

            d_m_next.scale(1.0 - alpha_t);

            if t % segment_len == 0 {
                d_m_next.add(&d_m_chunk_start);
                d_m_chunk_start =
                    MemoryWeights::zeros(self.key_dim, self.memory_hidden_dim, self.val_dim);
            }

            let mut d_uin = Array1::<f32>::zeros(u_in.len());

            let d_z_alpha = d_alpha * alpha_t * (1.0 - alpha_t);
            d_w_alpha = d_w_alpha + (u_in.mapv(|x| x * d_z_alpha));
            d_uin = d_uin + (&self.w_alpha * d_z_alpha);

            let mut val_eta = 0.0;
            val_eta += (d_s_t.w1.clone() * &s_prev.w1).sum();
            val_eta += (d_s_t.b1.clone() * &s_prev.b1).sum();
            val_eta += (d_s_t.w2.clone() * &s_prev.w2).sum();
            val_eta += (d_s_t.b2.clone() * &s_prev.b2).sum();
            let d_eta = val_eta;
            let d_z_eta = d_eta * eta_t * (1.0 - eta_t);
            d_w_eta = d_w_eta + (u_in.mapv(|x| x * d_z_eta));
            d_uin = d_uin + (&self.w_eta * d_z_eta);

            let z_k = m_prev.w1.dot(k_t) + &m_prev.b1;
            let h_k = z_k.mapv(|x| x.max(0.0));
            let v_pred = m_prev.w2.dot(&h_k) + &m_prev.b2;
            let delta = &v_pred - &trace.vs[t];

            let g_w2 = delta
                .clone()
                .insert_axis(Axis(1))
                .dot(&h_k.clone().insert_axis(Axis(0)));
            let g_b2 = delta.clone();
            let grad_h_k = m_prev.w2.t().dot(&delta);
            let grad_z_k = &grad_h_k * z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let g_w1 = grad_z_k
                .clone()
                .insert_axis(Axis(1))
                .dot(&k_t.clone().insert_axis(Axis(0)));
            let g_b1 = grad_z_k.clone();

            let mut val_theta = 0.0;
            val_theta += (d_s_t.w1.clone() * &g_w1).sum();
            val_theta += (d_s_t.b1.clone() * &g_b1).sum();
            val_theta += (d_s_t.w2.clone() * &g_w2).sum();
            val_theta += (d_s_t.b2.clone() * &g_b2).sum();
            let d_theta = -val_theta;
            let d_z_theta = d_theta * theta_t * (1.0 - theta_t);
            d_w_theta = d_w_theta + (u_in.mapv(|x| x * d_z_theta));
            d_uin = d_uin + (&self.w_theta * d_z_theta);

            let u_w1 = d_s_t.w1.mapv(|x| -theta_t * x);
            let u_b1 = d_s_t.b1.mapv(|x| -theta_t * x);
            let u_w2 = d_s_t.w2.mapv(|x| -theta_t * x);
            let u_b2 = d_s_t.b2.mapv(|x| -theta_t * x);

            let sigma_prime = z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let u_w2_t_delta = u_w2.t().dot(&delta);
            let term1_inner = &sigma_prime * &u_w2_t_delta;
            let term1 = m_prev.w1.t().dot(&term1_inner);
            let w2_t_delta = m_prev.w2.t().dot(&delta);
            let epsilon = &w2_t_delta * &sigma_prime;
            let term2 = u_w1.t().dot(&epsilon);
            let d_kt = term1 + term2;

            d_wk = d_wk
                + d_kt
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&u_in.insert_axis(Axis(0)));
            d_uin = d_uin + self.w_k.t().dot(&d_kt);

            let u_w1_k_ub1 = u_w1.dot(k_t) + &u_b1;
            let term_v_2 = m_prev.w2.dot(&(&sigma_prime * &u_w1_k_ub1));
            let term_v_1 = u_w2.dot(&h_k) + &u_b2;
            let d_vt = -(term_v_1 + term_v_2);

            d_wv = d_wv
                + d_vt
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&u_in.insert_axis(Axis(0)));
            d_uin = d_uin + self.w_v.t().dot(&d_vt);

            d_update_inputs.row_mut(t).assign(&d_uin);

            d_s_next = d_s_t;
        }

        d_init_memory.add(&d_m_next);

        let param_grads = vec![
            d_wq,
            d_wk,
            d_wv,
            d_w_alpha.insert_axis(Axis(0)),
            d_w_eta.insert_axis(Axis(0)),
            d_w_theta.insert_axis(Axis(0)),
            d_init_memory.w1,
            d_init_memory.b1.insert_axis(Axis(0)),
            d_init_memory.w2,
            d_init_memory.b2.insert_axis(Axis(0)),
        ];

        (d_queries, d_update_inputs, param_grads)
    }
}
