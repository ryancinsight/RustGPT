use ndarray::{Array1, Array2, Axis, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use super::neural::{MemoryWeights, NeuralMemory, NeuralMemoryStreamingWorkspace};
use crate::domain::{
    attention::poly_attention::{PolyAttention, PolyAttentionCache, PolyAttentionContextWorkspace},
    network::Layer,
};

#[derive(Debug, Clone)]
pub struct TitansMACStreamingWorkspace {
    pub context_input: Array2<f32>,
    pub attention_input: Array2<f32>,
    pub h_t: Array1<f32>,
    pub neural_memory_workspace: NeuralMemoryStreamingWorkspace,
    pub poly_context_workspace: PolyAttentionContextWorkspace,
    pub update_buffer: Array2<f32>,
    pub segment_input_buffer: Array2<f32>,
    pub buffer_idx: usize,
}

/// Memory As Context (MAC) Architecture
///
/// "We treat the memory as a context to the current information."
/// Segment-based approach where memory processes past segment and output is concatenated
/// with current segment input to attention.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TitansMAC {
    // Core branch (Attention)
    pub core: PolyAttention,

    // Long-term Memory branch (NeuralMemory)
    pub memory: NeuralMemory,

    // Persistent Memory parameters (Learnable)
    // Dimension: (persistent_len, input_dim)
    pub persistent_memory: Array2<f32>,

    pub segment_len: usize,
    pub persistent_len: usize,

    #[serde(skip)]
    cached_input: Option<Array2<f32>>,

    #[serde(skip)]
    pub cached_forward_data: Option<Vec<SegmentForwardData>>,

    #[serde(skip)]
    pub streaming_workspace: Option<TitansMACStreamingWorkspace>,
}

#[derive(Clone, Debug)]
pub struct SegmentForwardData {
    pub seg_out: Array2<f32>,
    pub poly_caches: Vec<PolyAttentionCache>,
}

impl TitansMAC {
    pub fn new(
        core: PolyAttention,
        memory: NeuralMemory,
        persistent_len: usize,
        segment_len: usize,
    ) -> Self {
        let input_dim = core.embed_dim;
        // Use fixed seed for deterministic behavior during verification
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.02).unwrap();

        let p_vec: Vec<f32> = (0..persistent_len * input_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let persistent_memory = Array2::from_shape_vec((persistent_len, input_dim), p_vec).unwrap();

        Self {
            core,
            memory,
            persistent_memory,
            segment_len,
            persistent_len,
            cached_input: None,
            cached_forward_data: None,
            streaming_workspace: None,
        }
    }

    // Helper to retrieve and concat
    fn process_segment(&mut self, segment: &Array2<f32>) -> (Array2<f32>, Vec<PolyAttentionCache>) {
        // 1. Retrieve h_t from Memory using input context (segment) as query.
        let h_t = self.memory.retrieve(segment);

        if cfg!(debug_assertions) {
            for i in 0..segment.nrows() {
                if i == segment.nrows() - 1 {
                    println!("Batch Step {} (Last):", i);
                    println!("  Retrieve Input (segment row): {:?}", segment.row(i));
                    println!("  h_t: {:?}", h_t.row(i));
                }
            }
        }

        // 2. Process token-by-token to ensure correct causality (context = P + h_t[i])
        let p_len = self.persistent_len;
        let s_len = segment.nrows();
        let d = segment.ncols();

        let mut seg_out_rows = Vec::with_capacity(s_len);
        let mut caches = Vec::with_capacity(s_len);

        for i in 0..s_len {
            let current_h_t = h_t.row(i);
            let current_seg = segment.row(i);

            // Input: [Persistent | h_t[i] | Segment[i]]
            let total_len = p_len + 1 + 1;
            let mut input_seq = Array2::<f32>::zeros((total_len, d));

            input_seq
                .slice_mut(s![0..p_len, ..])
                .assign(&self.persistent_memory);
            input_seq.row_mut(p_len).assign(&current_h_t);
            input_seq.row_mut(p_len + 1).assign(&current_seg);

            // Forward detached (causal=true ensures C attends to A, B, C)
            let (attn_out, cache) = self.core.forward_detached(&input_seq, true);

            if cfg!(debug_assertions) {
                println!("Batch Step {}:", i);
                println!("  Input: {:?}", current_seg);
                println!("  h_t: {:?}", current_h_t);
                // Check context slice
                // input_seq row p_len is h_t
            }

            // Extract output for segment[i] (last row)
            let out_i = attn_out.row(total_len - 1).to_owned();

            seg_out_rows.push(out_i);
            caches.push(cache);
        }

        // Stack output
        let mut seg_out = Array2::<f32>::zeros((s_len, d));
        for (i, row) in seg_out_rows.iter().enumerate() {
            seg_out.row_mut(i).assign(row);
        }

        // 3. Update Memory using Attention output
        self.memory.update(&seg_out);

        (seg_out, caches)
    }

    /// Process a single token step (Streaming/Rolling mode)
    ///
    /// This enables token-by-token inference where the memory state is maintained
    /// and updated incrementally.
    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(input.len());
        self.forward_step_into(&input.view(), &mut output);
        output
    }

    /// Process a single token step into output buffer (Zero Allocation)
    pub fn forward_step_into(
        &mut self,
        input: &ndarray::ArrayView1<f32>,
        output: &mut Array1<f32>,
    ) {
        let p_len = self.persistent_len;
        let d = input.len();
        let context_len = p_len + 1; // Persistent + Memory

        // Initialize workspace if needed
        if self.streaming_workspace.is_none() {
            self.streaming_workspace = Some(TitansMACStreamingWorkspace {
                context_input: Array2::zeros((context_len, d)),
                attention_input: Array2::zeros((context_len + 1, d)),
                h_t: Array1::zeros(self.memory.val_dim),
                neural_memory_workspace: NeuralMemoryStreamingWorkspace {
                    q: Array1::zeros(self.memory.key_dim),
                    z_ret: Array1::zeros(self.memory.memory_hidden_dim),
                    h_ret: Array1::zeros(self.memory.memory_hidden_dim),
                    y_ret: Array1::zeros(self.memory.val_dim),

                    k: Array1::zeros(self.memory.key_dim),
                    v: Array1::zeros(self.memory.val_dim),
                    z_upd: Array1::zeros(self.memory.memory_hidden_dim),
                    h_upd: Array1::zeros(self.memory.memory_hidden_dim),
                    v_pred: Array1::zeros(self.memory.val_dim),
                    grad_output: Array1::zeros(self.memory.val_dim),
                    grad_w2: Array2::zeros((self.memory.val_dim, self.memory.memory_hidden_dim)),
                    grad_b2: Array1::zeros(self.memory.val_dim),
                    grad_h: Array1::zeros(self.memory.memory_hidden_dim),
                    grad_z: Array1::zeros(self.memory.memory_hidden_dim),
                    grad_w1: Array2::zeros((self.memory.memory_hidden_dim, self.memory.key_dim)),
                    grad_b1: Array1::zeros(self.memory.memory_hidden_dim),
                },
                poly_context_workspace: PolyAttentionContextWorkspace::new(context_len, d),
                update_buffer: Array2::zeros((self.segment_len, d)),
                segment_input_buffer: Array2::zeros((self.segment_len, d)),
                buffer_idx: 0,
            });
        }

        let workspace = self.streaming_workspace.as_mut().unwrap();
        if workspace.context_input.nrows() != context_len || workspace.context_input.ncols() != d {
            workspace.context_input = Array2::zeros((context_len, d));
            workspace.attention_input = Array2::zeros((context_len + 1, d));
            workspace.poly_context_workspace = PolyAttentionContextWorkspace::new(context_len, d);
        }

        // 1. Retrieve h_t from Memory using input as query
        self.memory.retrieve_step_into(
            input,
            &mut workspace.h_t,
            &mut workspace.neural_memory_workspace,
        );

        if cfg!(debug_assertions) {
            println!("Stream Step {}:", workspace.buffer_idx);
            println!("  Input: {:?}", input);
            println!("  h_t: {:?}", workspace.h_t);
        }

        // 2. Construct Context [Persistent | h_t]
        // Copy persistent memory
        workspace
            .context_input
            .slice_mut(s![0..p_len, ..])
            .assign(&self.persistent_memory);

        // Copy h_t
        workspace
            .context_input
            .row_mut(p_len)
            .assign(&workspace.h_t);

        // 3. Match batch path exactly: run detached attention on [Persistent | h_t | token].
        workspace
            .attention_input
            .slice_mut(s![0..context_len, ..])
            .assign(&workspace.context_input);
        workspace.attention_input.row_mut(context_len).assign(input);
        let (attn_out, _) = self.core.forward_detached(&workspace.attention_input, true);
        output.assign(&attn_out.row(context_len));

        // 4. Buffer output for Segment-based Memory Update
        workspace
            .update_buffer
            .row_mut(workspace.buffer_idx)
            .assign(&output.view());
        workspace.buffer_idx += 1;

        // 5. Update Memory if segment is full
        if workspace.buffer_idx >= self.segment_len {
            for i in 0..self.segment_len {
                self.memory.update_step_with_workspace(
                    &workspace.update_buffer.row(i),
                    &mut workspace.neural_memory_workspace,
                );
            }
            workspace.buffer_idx = 0;
        }
    }
}

impl Layer for TitansMAC {
    fn layer_type(&self) -> &str {
        "TitansMAC"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        let seq_len = input.nrows();
        let input_dim = input.ncols();

        let mut outputs = Vec::new();
        let mut forward_data = Vec::new();
        let mut processed = 0;

        // Initialize memory state for tracking
        self.memory.reset_memory();

        while processed < seq_len {
            let end = std::cmp::min(processed + self.segment_len, seq_len);
            let segment = input.slice(s![processed..end, ..]).to_owned();

            let (seg_out, poly_caches) = self.process_segment(&segment);
            outputs.push(seg_out.clone());

            forward_data.push(SegmentForwardData {
                seg_out,
                poly_caches,
            });

            processed = end;
        }

        self.cached_forward_data = Some(forward_data);

        if outputs.is_empty() {
            return Array2::zeros((0, input_dim));
        }

        let total_rows: usize = outputs.iter().map(|a| a.nrows()).sum();
        let mut result = Array2::<f32>::zeros((total_rows, input_dim));

        let mut cursor = 0;
        for out in outputs {
            let rows = out.nrows();
            result.slice_mut(s![cursor..cursor + rows, ..]).assign(&out);
            cursor += rows;
        }

        result
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (input_grads, param_grads) = self.compute_gradients(input, grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.core.parameters() + self.memory.parameters() + self.persistent_memory.len()
    }

    fn weight_norm(&self) -> f32 {
        let mut sum_sq = 0.0;
        sum_sq += self.core.weight_norm().powi(2);
        sum_sq += self.memory.weight_norm().powi(2);
        sum_sq += self.persistent_memory.mapv(|x| x * x).sum();
        sum_sq.sqrt()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let seq_len = input.nrows();
        let input_dim = input.ncols();

        // 1. Re-run forward pass to capture state
        type MemoryTraceEntry = (
            Array1<f32>,
            Array1<f32>,
            Array1<f32>,
            f32,
            f32,
            f32,
            MemoryWeights,
            MemoryWeights,
        );
        struct SegmentData {
            segment: Array2<f32>,
            #[allow(dead_code)]
            h_t: Array2<f32>,
            seg_out: Array2<f32>,
            poly_caches: Vec<PolyAttentionCache>,
            memory_before: MemoryWeights,
            momentum_before: MemoryWeights,
            memory_trace: Vec<MemoryTraceEntry>, // Trace for update loop
        }

        let mut forward_data = Vec::new();
        let mut curr_memory = self.memory.init_memory.clone();
        let mut momentum = MemoryWeights::zeros(
            self.memory.key_dim,
            self.memory.memory_hidden_dim,
            self.memory.val_dim,
        );

        let mut processed = 0;
        while processed < seq_len {
            let end = std::cmp::min(processed + self.segment_len, seq_len);
            let segment = input.slice(s![processed..end, ..]).to_owned();
            let s_len = segment.nrows();

            // Retrieve (using curr_memory snapshot)
            let mut h_t = Array2::<f32>::zeros((s_len, self.memory.val_dim));
            // MLP Forward for retrieval
            for r in 0..s_len {
                let x = segment.row(r).to_owned();
                let q = self.memory.w_q.dot(&x);
                let z = curr_memory.w1.dot(&q) + &curr_memory.b1;
                let h = z.mapv(|x: f32| x.max(0.0));
                let y = curr_memory.w2.dot(&h) + &curr_memory.b2;
                h_t.row_mut(r).assign(&y);
            }

            // Context & Core
            let p_len = self.persistent_len;

            // Try to use cached forward outputs if available
            let (seg_out, poly_caches) = if let Some(cached_data) = &self.cached_forward_data
                && forward_data.len() < cached_data.len()
            {
                let d = &cached_data[forward_data.len()];
                (d.seg_out.clone(), d.poly_caches.clone())
            } else {
                // Fallback: re-compute token-wise
                let mut seg_out_rows = Vec::with_capacity(s_len);
                let mut caches = Vec::with_capacity(s_len);
                let d = input_dim;

                for i in 0..s_len {
                    let current_h_t = h_t.row(i);
                    let current_seg = segment.row(i);

                    let total_len = p_len + 1 + 1;
                    let mut input_seq = Array2::<f32>::zeros((total_len, d));

                    input_seq
                        .slice_mut(s![0..p_len, ..])
                        .assign(&self.persistent_memory);
                    input_seq.row_mut(p_len).assign(&current_h_t);
                    input_seq.row_mut(p_len + 1).assign(&current_seg);

                    let (attn_out, cache) = self.core.forward_detached(&input_seq, true);
                    let out_i = attn_out.row(total_len - 1).to_owned();
                    seg_out_rows.push(out_i);
                    caches.push(cache);
                }

                let mut seg_out = Array2::<f32>::zeros((s_len, d));
                for (i, row) in seg_out_rows.iter().enumerate() {
                    seg_out.row_mut(i).assign(row);
                }
                (seg_out, caches)
            };

            // Update Memory Logic
            let memory_before = curr_memory.clone();
            let momentum_before = momentum.clone();
            let mut memory_trace = Vec::new();

            for r in 0..seg_out.nrows() {
                let x = seg_out.row(r).to_owned();
                let k = self.memory.w_k.dot(&x);
                let v = self.memory.w_v.dot(&x);
                let alpha = 1.0 / (1.0 + (-self.memory.w_alpha.dot(&x)).exp());
                let eta = 1.0 / (1.0 + (-self.memory.w_eta.dot(&x)).exp());
                let theta = 1.0 / (1.0 + (-self.memory.w_theta.dot(&x)).exp());

                let z = curr_memory.w1.dot(&k) + &curr_memory.b1;
                let h = z.mapv(|val: f32| val.max(0.0));
                let v_pred = curr_memory.w2.dot(&h) + &curr_memory.b2;
                let grad_output = &v_pred - &v;

                let grad_w2 = grad_output
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&h.clone().insert_axis(Axis(0)));
                let grad_b2 = grad_output.clone();
                let grad_h = curr_memory.w2.t().dot(&grad_output);
                let grad_z = grad_h * z.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 });
                let grad_w1 = grad_z
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&k.clone().insert_axis(Axis(0)));
                let grad_b1 = grad_z;

                momentum.scale(eta);
                momentum.w1 = &momentum.w1 - &(&grad_w1 * theta);
                momentum.b1 = &momentum.b1 - &(&grad_b1 * theta);
                momentum.w2 = &momentum.w2 - &(&grad_w2 * theta);
                momentum.b2 = &momentum.b2 - &(&grad_b2 * theta);

                let mem_prev = curr_memory.clone(); // Store M_{t-1} for this step
                let mom_curr = momentum.clone(); // Store S_t

                curr_memory.scale(1.0 - alpha);
                curr_memory.add(&momentum);

                memory_trace.push((k, v, x, alpha, eta, theta, mem_prev, mom_curr));
            }

            forward_data.push(SegmentData {
                segment,
                h_t,
                seg_out,
                poly_caches,
                memory_before,
                momentum_before,
                memory_trace,
            });

            processed = end;
        }

        // Backward Pass
        let mut core_param_grads_accum: Vec<Array2<f32>> = Vec::new();
        // Initialize core param grads accumulators (copy shape from first dummy run or similar)
        // We'll just append and sum later or initialize zeros.
        // Better: get shape from `core.parameters()`?
        // We will just collect all list of lists and reduce them later.

        let mut persistent_grad = Array2::<f32>::zeros(self.persistent_memory.raw_dim());

        // Memory gradients accumulators
        let mut d_wq = Array2::<f32>::zeros(self.memory.w_q.raw_dim());
        let mut d_wk = Array2::<f32>::zeros(self.memory.w_k.raw_dim());
        let mut d_wv = Array2::<f32>::zeros(self.memory.w_v.raw_dim());
        let mut d_w_alpha = Array1::<f32>::zeros(self.memory.w_alpha.raw_dim());
        let mut d_w_eta = Array1::<f32>::zeros(self.memory.w_eta.raw_dim());
        let mut d_w_theta = Array1::<f32>::zeros(self.memory.w_theta.raw_dim());
        let mut d_init_memory = MemoryWeights::zeros(
            self.memory.key_dim,
            self.memory.memory_hidden_dim,
            self.memory.val_dim,
        );

        let mut d_m_next = MemoryWeights::zeros(
            self.memory.key_dim,
            self.memory.memory_hidden_dim,
            self.memory.val_dim,
        );
        let mut d_s_next = MemoryWeights::zeros(
            self.memory.key_dim,
            self.memory.memory_hidden_dim,
            self.memory.val_dim,
        );

        let mut input_grads = Array2::<f32>::zeros(input.raw_dim());

        let mut global_t_end = input.nrows();

        for (_seg_idx, data) in forward_data.iter().enumerate().rev() {
            let seg_len = data.segment.nrows();
            let global_t_start = global_t_end - seg_len;

            // 1. Memory Update Backward (Backprop through time within segment)
            // dL/dM_next flows in.
            // We compute dL/d_seg_out (from memory update) -> d_update_inputs
            // And update d_M_next (flowing to start of segment).

            let mut d_update_inputs = Array2::<f32>::zeros(data.seg_out.raw_dim());

            for t in (0..seg_len).rev() {
                let (k, v, u_in, alpha, eta, theta, m_prev, _s_curr) = &data.memory_trace[t];
                // Note: m_prev is M_{t-1} relative to this step. s_curr is S_t.

                // d_m_next is dL/dM_t
                let d_m_curr = d_m_next.clone();

                // d_alpha
                let mut val_alpha = 0.0;
                val_alpha += (d_m_curr.w1.clone() * &m_prev.w1).sum();
                val_alpha += (d_m_curr.b1.clone() * &m_prev.b1).sum();
                val_alpha += (d_m_curr.w2.clone() * &m_prev.w2).sum();
                val_alpha += (d_m_curr.b2.clone() * &m_prev.b2).sum();
                let d_alpha = -val_alpha;

                let mut d_s_t = d_m_curr.clone();
                let mut scaled_s_next = d_s_next.clone();
                scaled_s_next.scale(*eta);
                d_s_t.add(&scaled_s_next);

                d_m_next.scale(1.0 - alpha); // Now d_m_next is dL/dM_{t-1} from update

                let mut d_uin = Array1::<f32>::zeros(u_in.len());

                let d_z_alpha = d_alpha * alpha * (1.0 - alpha);
                d_w_alpha = d_w_alpha + (u_in * d_z_alpha);
                d_uin = d_uin + (&self.memory.w_alpha * d_z_alpha);

                let mut val_eta = 0.0;
                // S_{t-1} is needed. If t=0, it's momentum_before. Else trace[t-1].
                let s_prev = if t == 0 {
                    &data.momentum_before
                } else {
                    &data.memory_trace[t - 1].7
                };

                val_eta += (d_s_t.w1.clone() * &s_prev.w1).sum();
                val_eta += (d_s_t.b1.clone() * &s_prev.b1).sum();
                val_eta += (d_s_t.w2.clone() * &s_prev.w2).sum();
                val_eta += (d_s_t.b2.clone() * &s_prev.b2).sum();
                let d_eta = val_eta;
                let d_z_eta = d_eta * eta * (1.0 - eta);
                d_w_eta = d_w_eta + (u_in * d_z_eta);
                d_uin = d_uin + (&self.memory.w_eta * d_z_eta);

                // d_theta
                let z_k = m_prev.w1.dot(k) + &m_prev.b1;
                let h_k = z_k.mapv(|x| x.max(0.0));
                let v_pred = m_prev.w2.dot(&h_k) + &m_prev.b2;
                let delta = &v_pred - v;

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
                    .dot(&k.clone().insert_axis(Axis(0)));
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

                // d_G_t
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

                d_wk = d_wk
                    + d_kt
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&u_in.clone().insert_axis(Axis(0)));
                d_uin = d_uin + self.memory.w_k.t().dot(&d_kt);

                let u_w1_k_ub1 = u_w1.dot(k) + &u_b1;
                let term_v_2 = m_prev.w2.dot(&(&sigma_prime * &u_w1_k_ub1));
                let term_v_1 = u_w2.dot(&h_k) + &u_b2;
                let d_vt = -(term_v_1 + term_v_2);

                d_wv = d_wv
                    + d_vt
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&u_in.clone().insert_axis(Axis(0)));
                d_uin = d_uin + self.memory.w_v.t().dot(&d_vt);

                d_update_inputs.row_mut(t).assign(&d_uin);
                d_s_next = d_s_t;
            }

            // 2. Combine gradients for seg_out
            let d_seg_out_loss = output_grads.slice(s![global_t_start..global_t_end, ..]);
            let d_seg_out_total = &d_seg_out_loss + &d_update_inputs;

            // 3. Backprop Core (Token-wise)
            let p_len = self.persistent_len;
            let mut d_ht_seg = Array2::<f32>::zeros((seg_len, self.memory.val_dim));

            for t in 0..seg_len {
                let d_out_t = d_seg_out_total.row(t);

                // Construct d_context_out for this token's attention pass
                // Context: [Persistent | h_t[t] | segment[t]]
                let total_len = p_len + 1 + 1;
                let mut d_context_out = Array2::<f32>::zeros((total_len, input_dim));
                d_context_out.row_mut(total_len - 1).assign(&d_out_t);

                let cache = &data.poly_caches[t];
                let (d_context, core_pg) = self
                    .core
                    .compute_gradients_with_cache(cache, &d_context_out);

                // Accumulate core params
                if core_param_grads_accum.is_empty() {
                    core_param_grads_accum = core_pg;
                } else {
                    for (acc, new) in core_param_grads_accum.iter_mut().zip(core_pg.iter()) {
                        *acc += new;
                    }
                }

                // Distribute d_context
                // d_context rows: [0..p -> Persistent, p -> h_t[t], p+1 -> segment[t]]
                persistent_grad += &d_context.slice(s![0..p_len, ..]);
                d_ht_seg.row_mut(t).assign(&d_context.row(p_len));

                // Input gradients from segment part
                let d_s = d_context.row(p_len + 1);
                let mut current_grad = input_grads.row_mut(global_t_start + t);
                current_grad += &d_s;
            }

            // 4. Memory Retrieval Backward
            // Accumulate dL/dM_start from all retrieval steps in this segment
            let m_start = &data.memory_before;

            for t in 0..seg_len {
                let dy_t = d_ht_seg.row(t); // dL/dh_t
                let q_in = data.segment.row(t);

                let q_t = self.memory.w_q.dot(&q_in);

                let z_q = m_start.w1.dot(&q_t) + &m_start.b1;
                let h_q = z_q.mapv(|x| x.max(0.0));

                let grad_h_q = m_start.w2.t().dot(&dy_t);
                let grad_z_q = &grad_h_q * z_q.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                let d_qt = m_start.w1.t().dot(&grad_z_q);

                d_wq = d_wq
                    + d_qt
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&q_in.insert_axis(Axis(0)));
                let d_qin = self.memory.w_q.t().dot(&d_qt);

                // Add to input gradients (segment part)
                // Note: input_grads already has contribution from d_segment_seg (from core).
                // Now we add contribution from memory retrieval query.
                let mut current_grad = input_grads.row_mut(global_t_start + t);
                current_grad += &d_qin;

                // Accumulate to d_m_next (which flows to M_start, i.e. M_{k-1})
                d_m_next.w2 =
                    d_m_next.w2 + dy_t.insert_axis(Axis(1)).dot(&h_q.insert_axis(Axis(0)));
                d_m_next.b2.zip_mut_with(&dy_t, |a, &b| *a += b);
                d_m_next.w1 = d_m_next.w1
                    + grad_z_q
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&q_t.clone().insert_axis(Axis(0)));
                d_m_next.b1 += &grad_z_q;
            }

            global_t_end = global_t_start;
        }

        d_init_memory.add(&d_m_next);

        // Collect all params
        // Core params first (from accum)
        let mut all_grads = core_param_grads_accum;

        // Memory params
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

        // Persistent memory
        all_grads.push(persistent_grad);

        (input_grads, all_grads)
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        lr: f32,
    ) -> crate::common::errors::Result<()> {
        let core_params = self.core.parameters();
        let memory_params = 10;
        let persistent_params = 1;

        if gradients.len() != core_params + memory_params + persistent_params {
            return Err(crate::common::errors::ModelError::GradientError {
                message: format!(
                    "TitansMAC gradient count mismatch: expected {}, got {}",
                    core_params + memory_params + persistent_params,
                    gradients.len()
                ),
            });
        }

        let core_grads = &gradients[0..core_params];
        self.core.apply_gradients(core_grads, lr)?;

        let memory_grads = &gradients[core_params..core_params + memory_params];
        self.memory.apply_gradients(memory_grads, lr)?;

        let persistent_grad = &gradients[core_params + memory_params];
        self.persistent_memory.scaled_add(-lr, persistent_grad);
        Ok(())
    }

    fn zero_gradients(&mut self) {
        self.core.zero_gradients();
        self.memory.zero_gradients();
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::domain::attention::poly_attention::PolyAttention;
    use crate::domain::attention::position::config::{CoPEConfig, CoPEVariant};
    use crate::domain::memory::titans::NeuralMemory;

    #[test]
    fn test_titans_mac_forward() {
        let input_dim = 16;
        let num_heads = 4;
        let memory_hidden_dim = 8;
        let segment_len = 4;
        let persistent_len = 2;

        let cope_config = CoPEConfig {
            variant: CoPEVariant::Standard,
            max_pos: 64,
            window_size: None,
        };
        let poly = PolyAttention::new(input_dim, num_heads, 3, cope_config);
        let memory = NeuralMemory::new(input_dim, input_dim, input_dim, memory_hidden_dim);

        let mut mac = TitansMAC::new(poly, memory, persistent_len, segment_len);

        // Input: (8, 16)
        let seq_len = 8;
        let input = Array2::<f32>::zeros((seq_len, input_dim));

        let output = mac.forward(&input);

        assert_eq!(output.dim(), (seq_len, input_dim));
    }

    #[test]
    fn test_titans_mac_gradients_shape() {
        let input_dim = 8;
        let num_heads = 2;
        let memory_hidden_dim = 4;
        let segment_len = 2;
        let persistent_len = 2;

        let cope_config = CoPEConfig {
            variant: CoPEVariant::Standard,
            max_pos: 16,
            window_size: None,
        };
        let poly = PolyAttention::new(input_dim, num_heads, 1, cope_config);
        let memory = NeuralMemory::new(input_dim, input_dim, input_dim, memory_hidden_dim);

        let mac = TitansMAC::new(poly, memory, persistent_len, segment_len);

        let seq_len = 4;
        let input = Array2::<f32>::ones((seq_len, input_dim));
        let output_grads = Array2::<f32>::ones((seq_len, input_dim));

        let (input_grads, param_grads) = mac.compute_gradients(&input, &output_grads);

        assert_eq!(input_grads.dim(), (seq_len, input_dim));
        assert!(!param_grads.is_empty());
        assert!(param_grads.iter().all(|g| g.iter().all(|x| x.is_finite())));
    }
}
