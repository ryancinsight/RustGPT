use ndarray::{Array1, Array2, Axis, s, Zip};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{
    attention::sliding_window_attention::SlidingWindowAttention,
    models::titans::memory::{MemoryWeights, NeuralMemory},
    network::Layer,
};

/// Memory As Gate (MAG) Architecture
///
/// "Sliding window attention (SWA) as a short-term memory and our neural memory module
/// as a long-term memory, combining by a gating."
#[derive(Serialize, Deserialize, Debug)]
pub struct TitansMAG {
    pub swa: SlidingWindowAttention,
    pub memory: NeuralMemory,

    // Gating parameters: Input is [y_swa; y_mem] (2 * dim) -> Output is gate values (dim)
    pub gate_w: Array2<f32>,
    pub gate_b: Array1<f32>,

    pub segment_len: usize,

    #[serde(skip)]
    cached_input: Option<Array2<f32>>,
}

// Helpers for NeuralMemory access
fn mlp_forward(weights: &MemoryWeights, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
    let z = weights.w1.dot(input) + &weights.b1;
    let h = z.mapv(|x| x.max(0.0));
    let y = weights.w2.dot(&h) + &weights.b2;
    (y, h)
}

impl TitansMAG {
    pub fn new(
        swa: SlidingWindowAttention,
        memory: NeuralMemory,
        segment_len: usize,
    ) -> Self {
        let input_dim = swa.embed_dim;
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let w_vec: Vec<f32> = (0..2 * input_dim * input_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let gate_w = Array2::from_shape_vec((2 * input_dim, input_dim), w_vec).unwrap();

        let b_vec: Vec<f32> = (0..input_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let gate_b = Array1::from_shape_vec(input_dim, b_vec).unwrap();

        Self {
            swa,
            memory,
            gate_w,
            gate_b,
            segment_len,
            cached_input: None,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_static(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Layer for TitansMAG {
    fn layer_type(&self) -> &str {
        "TitansMAG"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        let seq_len = input.nrows();
        let dim = input.ncols();

        // 1. SWA Forward (on full sequence)
        let swa_out = self.swa.forward(input);

        // 2. Memory & Gating Loop (in segments)
        let mut outputs = Array2::<f32>::zeros((seq_len, dim));
        let mut processed = 0;

        while processed < seq_len {
            let end = std::cmp::min(processed + self.segment_len, seq_len);
            let segment_len = end - processed;

            let input_seg = input.slice(s![processed..end, ..]).to_owned();
            let swa_seg = swa_out.slice(s![processed..end, ..]).to_owned();

            // Retrieve (using current memory state)
            let mem_seg = self.memory.retrieve(&input_seg);

            // Gating
            let mut o_seg = Array2::<f32>::zeros((segment_len, dim));
            for t in 0..segment_len {
                let y = swa_seg.row(t);
                let m = mem_seg.row(t);

                // Concat [y, m]
                let mut concat = Array1::<f32>::zeros(2 * dim);
                concat.slice_mut(s![0..dim]).assign(&y);
                concat.slice_mut(s![dim..2*dim]).assign(&m);

                let z = concat.dot(&self.gate_w) + &self.gate_b;
                let g = z.mapv(|x| Self::sigmoid(x));

                let o = &g * &y + (1.0 - &g) * &m;
                o_seg.row_mut(t).assign(&o);
            }

            // Update Memory with O
            self.memory.update(&o_seg);

            // Store Output
            outputs.slice_mut(s![processed..end, ..]).assign(&o_seg);

            processed = end;
        }

        outputs
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().expect("forward must be called before backward");
        let (input_grads, param_grads) = self.compute_gradients(input, grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.swa.parameters() + self.memory.parameters() + self.gate_w.len() + self.gate_b.len()
    }

    fn weight_norm(&self) -> f32 {
        let mut sum_sq = 0.0;
        sum_sq += self.swa.weight_norm().powi(2);
        sum_sq += self.memory.weight_norm().powi(2);
        sum_sq += self.gate_w.mapv(|x| x * x).sum();
        sum_sq += self.gate_b.mapv(|x| x * x).sum();
        sum_sq.sqrt()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let mut swa_clone = self.swa.clone(); // Expensive but safe
        let swa_out = swa_clone.forward(input);

        // 2. Re-run Memory/Gating forward to capture traces and O
        let seq_len = input.nrows();
        let dim = input.ncols();
        let mut _outputs = Array2::<f32>::zeros((seq_len, dim));

        let mut processed = 0;
        let mut memory_clone = self.memory.clone();
        memory_clone.reset_memory();

        struct StepData {
            y: Array1<f32>, // swa out
            m: Array1<f32>, // mem out
            g: Array1<f32>, // gate
            q_t: Array1<f32>,
            k_t: Array1<f32>,
            v_val: Array1<f32>, // v target for update
            alpha: f32,
            eta: f32,
            theta: f32,
            m_prev: MemoryWeights, // M_{t-1} or M_{start}
            s_prev: MemoryWeights, // S_{t-1}
        }

        let mut trace = Vec::with_capacity(seq_len);

        let mut curr_memory = self.memory.init_memory.clone();
        let mut momentum = MemoryWeights::zeros(self.memory.key_dim, self.memory.memory_hidden_dim, self.memory.val_dim);
        let mut retrieval_memory_snapshot;

        processed = 0;
        while processed < seq_len {
            let end = std::cmp::min(processed + self.segment_len, seq_len);
            let segment_len = end - processed;

            // Snapshot for retrieval
            retrieval_memory_snapshot = curr_memory.clone();

            let mut o_seg = Array2::<f32>::zeros((segment_len, dim));

            for t in 0..segment_len {
                let global_t = processed + t;
                let input_t = input.row(global_t).to_owned();
                let swa_t = swa_out.row(global_t).to_owned();

                // Retrieval
                let q_t = self.memory.w_q.dot(&input_t);
                let (y_mem, _) = mlp_forward(&retrieval_memory_snapshot, &q_t);

                // Gating
                let mut concat = Array1::<f32>::zeros(2 * dim);
                concat.slice_mut(s![0..dim]).assign(&swa_t);
                concat.slice_mut(s![dim..2*dim]).assign(&y_mem);
                let z = concat.dot(&self.gate_w) + &self.gate_b;
                let g = z.mapv(|x| Self::sigmoid_static(x));

                let o = &g * &swa_t + (1.0 - &g) * &y_mem;
                o_seg.row_mut(t).assign(&o);

                // Update inputs (O is used as input for update)
                let u_in = o;
                let k_t = self.memory.w_k.dot(&u_in);
                let v_t = self.memory.w_v.dot(&u_in);
                let alpha_t = Self::sigmoid_static(self.memory.w_alpha.dot(&u_in));
                let eta_t = Self::sigmoid_static(self.memory.w_eta.dot(&u_in));
                let theta_t = Self::sigmoid_static(self.memory.w_theta.dot(&u_in));

                // Store trace
                trace.push(StepData {
                    y: swa_t,
                    m: y_mem,
                    g: g,
                    q_t: q_t,
                    k_t: k_t.clone(),
                    v_val: v_t.clone(),
                    alpha: alpha_t,
                    eta: eta_t,
                    theta: theta_t,
                    m_prev: curr_memory.clone(), // This is M_{t-1} for update
                    s_prev: momentum.clone(),
                });

                // Perform Update state tracking locally (needed for next step's trace)
                let (v_pred, h) = mlp_forward(&curr_memory, &k_t);
                let grad_output = &v_pred - &v_t;

                let grad_w2 = grad_output.clone().insert_axis(Axis(1)).dot(&h.clone().insert_axis(Axis(0)));
                let grad_b2 = grad_output.clone();
                let grad_h = curr_memory.w2.t().dot(&grad_output);
                let z_k = curr_memory.w1.dot(&k_t) + &curr_memory.b1;
                let grad_z = grad_h * z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                let grad_w1 = grad_z.clone().insert_axis(Axis(1)).dot(&k_t.clone().insert_axis(Axis(0)));
                let grad_b1 = grad_z;

                momentum.scale(eta_t);
                momentum.w1 = &momentum.w1 - &(&grad_w1 * theta_t);
                momentum.b1 = &momentum.b1 - &(&grad_b1 * theta_t);
                momentum.w2 = &momentum.w2 - &(&grad_w2 * theta_t);
                momentum.b2 = &momentum.b2 - &(&grad_b2 * theta_t);

                curr_memory.scale(1.0 - alpha_t);
                curr_memory.add(&momentum);
            }

            // Update Memory with O
            // Use clone just to keep state aligned if we used memory_clone methods later
            // But we are manually tracking state in curr_memory/momentum variables.
            // memory_clone is unused inside loop, only reset before.
            // Actually we don't need memory_clone if we manually track state!
            // But let's leave it as is.
            memory_clone.update(&o_seg);

            // Store Output
            _outputs.slice_mut(s![processed..end, ..]).assign(&o_seg);

            processed = end;
        }

        // Backward Loop
        let mut input_grads = Array2::<f32>::zeros(input.raw_dim());

        // Accumulators
        let mut d_gate_w = Array2::<f32>::zeros(self.gate_w.raw_dim());
        let mut d_gate_b = Array1::<f32>::zeros(self.gate_b.raw_dim());
        let mut d_swa_out = Array2::<f32>::zeros(swa_out.raw_dim());

        let mut d_wq = Array2::<f32>::zeros(self.memory.w_q.raw_dim());
        let mut d_wk = Array2::<f32>::zeros(self.memory.w_k.raw_dim());
        let mut d_wv = Array2::<f32>::zeros(self.memory.w_v.raw_dim());
        let mut d_w_alpha = Array1::<f32>::zeros(self.memory.w_alpha.raw_dim());
        let mut d_w_eta = Array1::<f32>::zeros(self.memory.w_eta.raw_dim());
        let mut d_w_theta = Array1::<f32>::zeros(self.memory.w_theta.raw_dim());
        let mut d_init_memory = MemoryWeights::zeros(self.memory.key_dim, self.memory.memory_hidden_dim, self.memory.val_dim);

        // State for backward loop
        let mut d_m_next = MemoryWeights::zeros(self.memory.key_dim, self.memory.memory_hidden_dim, self.memory.val_dim);
        let mut d_s_next = MemoryWeights::zeros(self.memory.key_dim, self.memory.memory_hidden_dim, self.memory.val_dim);
        let mut d_m_chunk_start = MemoryWeights::zeros(self.memory.key_dim, self.memory.memory_hidden_dim, self.memory.val_dim);

        for t in (0..seq_len).rev() {
            let data = &trace[t];
            let swa_t = &data.y;
            let mem_t = &data.m;
            let g = &data.g;

            // 1. Calculate dL/dO_t
            let mut d_o_t = output_grads.row(t).to_owned();

            // Check logic for memory accumulation
            if (t + 1) % self.segment_len == 0 && t + 1 < seq_len {
                 d_m_next.add(&d_m_chunk_start);
                 d_m_chunk_start = MemoryWeights::zeros(self.memory.key_dim, self.memory.memory_hidden_dim, self.memory.val_dim);
            }

            let d_m_curr = d_m_next.clone();

            let m_prev = &data.m_prev;
            let s_prev = &data.s_prev;
            let alpha = data.alpha;
            let eta = data.eta;
            let theta = data.theta;
            let k_t = &data.k_t;
            let v_t = &data.v_val;

            // d_alpha
            let mut val_alpha = 0.0;
            val_alpha += (d_m_curr.w1.clone() * &m_prev.w1).sum();
            val_alpha += (d_m_curr.b1.clone() * &m_prev.b1).sum();
            val_alpha += (d_m_curr.w2.clone() * &m_prev.w2).sum();
            val_alpha += (d_m_curr.b2.clone() * &m_prev.b2).sum();
            let d_alpha = -val_alpha;

            let mut d_s_t = d_m_curr.clone();
            let mut scaled_s_next = d_s_next.clone();
            scaled_s_next.scale(eta);
            d_s_t.add(&scaled_s_next);

            d_m_next.scale(1.0 - alpha);

             if t % self.segment_len == 0 {
                d_m_next.add(&d_m_chunk_start);
                d_m_chunk_start = MemoryWeights::zeros(self.memory.key_dim, self.memory.memory_hidden_dim, self.memory.val_dim);
            }

            let mut d_uin = Array1::<f32>::zeros(dim);

            // Recompute O_t (u_in)
            let o_t = g * swa_t + (1.0 - g) * mem_t;
            let u_in = &o_t;

            // d_alpha path
            let d_z_alpha = d_alpha * alpha * (1.0 - alpha);
            d_w_alpha = d_w_alpha + (u_in * d_z_alpha);
            d_uin = d_uin + (&self.memory.w_alpha * d_z_alpha);

            // d_eta path
            let mut val_eta = 0.0;
            val_eta += (d_s_t.w1.clone() * &s_prev.w1).sum();
            val_eta += (d_s_t.b1.clone() * &s_prev.b1).sum();
            val_eta += (d_s_t.w2.clone() * &s_prev.w2).sum();
            val_eta += (d_s_t.b2.clone() * &s_prev.b2).sum();
            let d_eta = val_eta;
            let d_z_eta = d_eta * eta * (1.0 - eta);
            d_w_eta = d_w_eta + (u_in * d_z_eta);
            d_uin = d_uin + (&self.memory.w_eta * d_z_eta);

            // d_theta path
            let z_k = m_prev.w1.dot(k_t) + &m_prev.b1;
            let h_k = z_k.mapv(|x| x.max(0.0));
            let v_pred = m_prev.w2.dot(&h_k) + &m_prev.b2;
            let delta = &v_pred - v_t;

            let g_w2 = delta.clone().insert_axis(Axis(1)).dot(&h_k.clone().insert_axis(Axis(0)));
            let g_b2 = delta.clone();
            let grad_h_k = m_prev.w2.t().dot(&delta);
            let grad_z_k = &grad_h_k * z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let g_w1 = grad_z_k.clone().insert_axis(Axis(1)).dot(&k_t.clone().insert_axis(Axis(0)));
            let g_b1 = grad_z_k.clone();

            let mut val_theta = 0.0;
            val_theta += (d_s_t.w1.clone() * &g_w1).sum();
            val_theta += (d_s_t.b1.clone() * &g_b1).sum();
            val_theta += (d_s_t.w2.clone() * &g_w2).sum();
            val_theta += (d_s_t.b2.clone() * &g_b2).sum();
            let d_theta = -val_theta;
            let d_z_theta = d_theta * theta * (1.0 - theta);
            d_w_theta = d_w_theta + (u_in * d_z_theta);
            d_uin = d_uin + (&self.memory.w_theta * d_z_theta);

            // d_G_t path (to k, v)
             let u_w1 = d_s_t.w1.mapv(|x| -theta * x);
             let u_b1 = d_s_t.b1.mapv(|x| -theta * x);
             let u_w2 = d_s_t.w2.mapv(|x| -theta * x);
             let u_b2 = d_s_t.b2.mapv(|x| -theta * x);

             let sigma_prime = z_k.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
             let u_w2_t_delta = u_w2.t().dot(&delta);
             let term1_inner = &sigma_prime * &u_w2_t_delta;
             let term1 = m_prev.w1.t().dot(&term1_inner);
             let w2_t_delta = m_prev.w2.t().dot(&delta);
             let epsilon = &w2_t_delta * &sigma_prime;
             let term2 = u_w1.t().dot(&epsilon);
             let d_kt = term1 + term2;

             d_wk = d_wk + d_kt.clone().insert_axis(Axis(1)).dot(&u_in.clone().insert_axis(Axis(0)));
             d_uin = d_uin + self.memory.w_k.t().dot(&d_kt);

             let u_w1_k_ub1 = u_w1.dot(k_t) + &u_b1;
             let term_v_2 = m_prev.w2.dot(&(&sigma_prime * &u_w1_k_ub1));
             let term_v_1 = u_w2.dot(&h_k) + &u_b2;
             let d_vt = -(term_v_1 + term_v_2);

             d_wv = d_wv + d_vt.clone().insert_axis(Axis(1)).dot(&u_in.clone().insert_axis(Axis(0)));
             d_uin = d_uin + self.memory.w_v.t().dot(&d_vt);

             d_s_next = d_s_t;

             // Now add d_uin to d_o_t
             d_o_t += &d_uin;

             // 2. Backprop through Gate Combination
             let d_g = &d_o_t * (swa_t - mem_t);
             let d_y = &d_o_t * g;
             let d_m = &d_o_t * (1.0 - g);

             // Backprop through Gate Weights
             let d_z = d_g * g * (1.0 - g);

             d_gate_b = d_gate_b + &d_z;

             let mut concat = Array1::<f32>::zeros(2 * dim);
             concat.slice_mut(s![0..dim]).assign(swa_t);
             concat.slice_mut(s![dim..2*dim]).assign(mem_t);

             d_gate_w = d_gate_w + concat.insert_axis(Axis(1)).dot(&d_z.clone().insert_axis(Axis(0)));

             let d_concat = self.gate_w.dot(&d_z);
             let d_y_from_gate = d_concat.slice(s![0..dim]);
             let d_m_from_gate = d_concat.slice(s![dim..2*dim]);

             let d_y_total = d_y + d_y_from_gate;
             let d_m_total = d_m + d_m_from_gate;

             d_swa_out.row_mut(t).assign(&d_y_total);

             // Retrieval Gradients
             let chunk_start_idx = t - (t % self.segment_len);
             let m_snapshot = &trace[chunk_start_idx].m_prev;

             let q_t = &trace[t].q_t;
             let dy_t = d_m_total;

             let z_q = m_snapshot.w1.dot(q_t) + &m_snapshot.b1;
             let h_q = z_q.mapv(|x| x.max(0.0));

             let grad_h_q = m_snapshot.w2.t().dot(&dy_t);
             let grad_z_q = &grad_h_q * z_q.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
             let d_qt = m_snapshot.w1.t().dot(&grad_z_q);

             let input_t = input.row(t);
             d_wq = d_wq + d_qt.clone().insert_axis(Axis(1)).dot(&input_t.insert_axis(Axis(0)));
             let d_xt_from_q = self.memory.w_q.t().dot(&d_qt);

             input_grads.row_mut(t).add_assign(&d_xt_from_q);

             d_m_chunk_start.w2 = d_m_chunk_start.w2 + dy_t.clone().insert_axis(Axis(1)).dot(&h_q.insert_axis(Axis(0)));
             d_m_chunk_start.b2.zip_mut_with(&dy_t, |a, &b| *a += b);
             d_m_chunk_start.w1 = d_m_chunk_start.w1 + grad_z_q.clone().insert_axis(Axis(1)).dot(&q_t.clone().insert_axis(Axis(0)));
             d_m_chunk_start.b1 += &grad_z_q;
        }

        d_init_memory.add(&d_m_next);

        let (swa_input_grads, swa_param_grads) = swa_clone.compute_gradients(input, &d_swa_out);

        input_grads = input_grads + swa_input_grads;

        let mut all_grads = swa_param_grads;

        all_grads.push(d_wq);
        all_grads.push(d_wk);
        all_grads.push(d_wv);
        all_grads.push(d_w_alpha.insert_axis(Axis(0)));
        all_grads.push(d_w_eta.insert_axis(Axis(0)));
        all_grads.push(d_w_theta.insert_axis(Axis(0)));
        all_grads.push(d_init_memory.w1);
        all_grads.push(d_init_memory.b1.insert_axis(Axis(0)));
        all_grads.push(d_init_memory.w2);
        all_grads.push(d_init_memory.b2.insert_axis(Axis(0)));

        all_grads.push(d_gate_w);
        all_grads.push(d_gate_b.insert_axis(Axis(0)));

        (input_grads, all_grads)
    }

    fn apply_gradients(&mut self, gradients: &[Array2<f32>], lr: f32) -> crate::errors::Result<()> {
        let swa_params = 3;
        let memory_params = 10;
        let gate_params = 2;

        if gradients.len() != swa_params + memory_params + gate_params {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "TitansMAG gradient count mismatch: expected {}, got {}",
                    swa_params + memory_params + gate_params,
                    gradients.len()
                ),
            });
        }

        let swa_grads = &gradients[0..swa_params];
        self.swa.apply_gradients(swa_grads, lr)?;

        let memory_grads = &gradients[swa_params..swa_params + memory_params];
        self.memory.apply_gradients(memory_grads, lr)?;

        let gate_grads = &gradients[swa_params + memory_params..];
        self.gate_w.scaled_add(-lr, &gate_grads[0]);
        self.gate_b.scaled_add(-lr, &gate_grads[1].row(0));

        Ok(())
    }

    fn zero_gradients(&mut self) {
        self.swa.zero_gradients();
        self.memory.zero_gradients();
    }
}

use std::ops::AddAssign;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::sliding_window_attention::SlidingWindowAttention;
    use crate::models::titans::memory::NeuralMemory;
    use ndarray::Array2;

    #[test]
    fn test_titans_mag_forward() {
        let input_dim = 8;
        let window_size = 4;
        let memory_hidden_dim = 4;
        let segment_len = 2;

        let swa = SlidingWindowAttention::new(input_dim, window_size);
        let memory = NeuralMemory::new(input_dim, input_dim, input_dim, memory_hidden_dim);

        let mut mag = TitansMAG::new(swa, memory, segment_len);

        let seq_len = 6;
        let input = Array2::<f32>::ones((seq_len, input_dim));

        let output = mag.forward(&input);

        assert_eq!(output.dim(), (seq_len, input_dim));
    }

    #[test]
    fn test_titans_mag_gradients_shape() {
        let input_dim = 4;
        let window_size = 2;
        let memory_hidden_dim = 4;
        let segment_len = 2;

        let swa = SlidingWindowAttention::new(input_dim, window_size);
        let memory = NeuralMemory::new(input_dim, input_dim, input_dim, memory_hidden_dim);

        let mut mag = TitansMAG::new(swa, memory, segment_len);

        let seq_len = 4;
        let input = Array2::<f32>::ones((seq_len, input_dim));
        // Need to call forward first to cache input/state
        let _ = mag.forward(&input);

        let output_grads = Array2::<f32>::ones((seq_len, input_dim));

        let (input_grads, param_grads) = mag.compute_gradients(&input, &output_grads);

        assert_eq!(input_grads.dim(), (seq_len, input_dim));
        assert!(!param_grads.is_empty());

        // Check SWA grads + Memory grads + Gate grads
        // SWA: 3
        // Memory: 10
        // Gate: 2
        // Total: 15
        assert_eq!(param_grads.len(), 3 + 10 + 2);

        // Check for finiteness
        for (i, g) in param_grads.iter().enumerate() {
            assert!(g.iter().all(|x| x.is_finite()), "Gradient {} contains non-finite values", i);
        }
    }
}
