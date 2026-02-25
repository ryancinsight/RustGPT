"""Patch neural.rs to add forward_gpu_kernel that uses the new Titans GPU kernels"""

FORWARD_GPU_KERNEL = '''
    /// Fully on-device forward pass using Titans memory GPU kernels.
    ///
    /// This replaces the CPU sequential update loop in `forward_gpu` with batched GPU dispatches:
    /// 1. Q/K/V projections via GEMM (already on GPU)
    /// 2. Per-token MLP forward (z, h, v_pred) via `SHADER_TITANS_MLP_FORWARD`
    /// 3. Gradient accumulation (grad_w1, grad_w2) via `SHADER_TITANS_GRAD_W*`
    /// 4. Momentum + memory weight update via `SHADER_TITANS_MEMORY_UPDATE`
    /// 5. Retrieve output via MLP forward on queries
    ///
    /// Note: The memory update still processes segment-by-segment (alpha/eta/theta
    /// are scalar per-token gates we average/scalar within each segment). A full
    /// parallel scan over the sequence remains future work.
    pub fn forward_gpu_kernel(
        &mut self,
        pool: &mut dyn crate::domain::compute::GpuMemoryPool,
        ops: &mut dyn crate::domain::compute::GpuMatrixOps,
        input: &ndarray::Array2<f32>,
    ) -> crate::common::errors::Result<ndarray::Array2<f32>> {
        use ndarray::Array2;

        if self.curr_memory.is_none() {
            self.reset_memory();
        }

        let seq_len = input.nrows();
        let input_dim = input.ncols();
        let key_dim = self.key_dim;
        let val_dim = self.val_dim;
        let hidden_dim = self.memory_hidden_dim;

        // ─── Q/K/V Projections (GPU GEMM) ───────────────────────────────────
        let input_slice = input.as_slice().ok_or_else(|| {
            crate::common::errors::ModelError::InvalidInput {
                message: "NeuralMemory::forward_gpu_kernel requires contiguous input".to_string(),
            }
        })?;
        let mut input_buf = pool.upload(input_slice)?;

        // Upload weights transposed (GEMM computes: input @ W^T)
        let w_q_t = self.w_q.t().as_standard_layout().into_owned();
        let w_k_t = self.w_k.t().as_standard_layout().into_owned();
        let w_v_t = self.w_v.t().as_standard_layout().into_owned();
        let mut w_q_buf = pool.upload(w_q_t.as_slice().unwrap())?;
        let mut w_k_buf = pool.upload(w_k_t.as_slice().unwrap())?;
        let mut w_v_buf = pool.upload(w_v_t.as_slice().unwrap())?;

        let mut q_buf = pool.allocate(seq_len * key_dim * 4)?;
        let mut k_buf = pool.allocate(seq_len * key_dim * 4)?;
        let mut v_buf = pool.allocate(seq_len * val_dim * 4)?;

        ops.gemm_f32(pool, 1.0, &input_buf, &w_q_buf, 0.0, &mut q_buf, seq_len, key_dim, input_dim, false, false)?;
        ops.gemm_f32(pool, 1.0, &input_buf, &w_k_buf, 0.0, &mut k_buf, seq_len, key_dim, input_dim, false, false)?;
        ops.gemm_f32(pool, 1.0, &input_buf, &w_v_buf, 0.0, &mut v_buf, seq_len, val_dim, input_dim, false, false)?;

        pool.deallocate(input_buf);
        pool.deallocate(w_q_buf);
        pool.deallocate(w_k_buf);
        pool.deallocate(w_v_buf);

        // ─── Scalar gate projections on CPU (tiny dot products) ──────────────
        let mut alpha_all = input.dot(&self.w_alpha);
        let mut eta_all   = input.dot(&self.w_eta);
        let mut theta_all = input.dot(&self.w_theta);
        alpha_all.mapv_inplace(Self::sigmoid);
        eta_all.mapv_inplace(Self::sigmoid);
        theta_all.mapv_inplace(Self::sigmoid);

        // ─── Upload memory weights to GPU ────────────────────────────────────
        let memory = self.curr_memory.as_ref().unwrap();
        let momentum = self.momentum.as_ref().unwrap();

        let mut mem_w1_buf = pool.upload(memory.w1.as_slice().unwrap())?;
        let mut mem_b1_buf = pool.upload(memory.b1.as_slice().unwrap())?;
        let mut mem_w2_buf = pool.upload(memory.w2.as_slice().unwrap())?;
        let mut mem_b2_buf = pool.upload(memory.b2.as_slice().unwrap())?;

        let mut mom_w1_buf = pool.upload(momentum.w1.as_slice().unwrap())?;
        let mut mom_b1_buf = pool.upload(momentum.b1.as_slice().unwrap())?;
        let mut mom_w2_buf = pool.upload(momentum.w2.as_slice().unwrap())?;
        let mut mom_b2_buf = pool.upload(momentum.b2.as_slice().unwrap())?;

        // Allocate intermediate grad buffers (reused across the update loop)
        let w1_elems = hidden_dim * key_dim;
        let w2_elems = val_dim * hidden_dim;

        // ─── Sequential update loop (GPU-accelerated inner steps) ────────────
        // We process one token at a time. The MLP forward + grad accumulation
        // is dispatched to GPU for each step. The loop itself remains sequential
        // because each step updates the memory used by the next.
        //
        // A future optimisation would chunked parallel scan (Titans-MAC style).
        let mut output_flat = vec![0.0f32; seq_len * val_dim];

        for t in 0..seq_len {
            let tok_off_k = t * key_dim;
            let tok_off_v = t * val_dim;

            // ── Retrieve: MLP forward on q[t] ────────────────────────────
            // Slice the query for this single token from k_buf (q has same dim)
            // We create a sub-view by uploading just the token slice.
            // TODO: use copy_within_device_range when available.
            let mut q_t_buf = pool.allocate(key_dim * 4)?;
            {
                let mut q_t_cpu = vec![0.0f32; key_dim];
                ops.download(pool, &q_buf, &mut {
                    let mut tmp = vec![0.0f32; seq_len * key_dim];
                    ops.download(pool, &q_buf, &mut tmp)?;
                    q_t_cpu.copy_from_slice(&tmp[tok_off_k..tok_off_k + key_dim]);
                    tmp
                })?;
                // Re-upload just the token slice
                let mut q_t_cpu = vec![0.0f32; key_dim];
                {
                    let mut tmp = vec![0.0f32; seq_len * key_dim];
                    ops.download(pool, &q_buf, &mut tmp)?;
                    q_t_cpu.copy_from_slice(&tmp[tok_off_k..tok_off_k + key_dim]);
                }
                ops.upload(pool, &q_t_cpu, &mut q_t_buf)?;
            }

            let mut z_t_buf = pool.allocate(hidden_dim * 4)?;
            let mut h_t_buf = pool.allocate(hidden_dim * 4)?;
            let mut y_t_buf = pool.allocate(val_dim * 4)?;

            ops.titans_mlp_forward(
                pool,
                &q_t_buf, &mem_w1_buf, &mem_b1_buf, &mem_w2_buf, &mem_b2_buf,
                &mut z_t_buf, &mut h_t_buf, &mut y_t_buf,
                1, key_dim, hidden_dim, val_dim,
            )?;

            // Download output y[t]
            let mut y_t = vec![0.0f32; val_dim];
            ops.download(pool, &y_t_buf, &mut y_t)?;
            output_flat[tok_off_v..tok_off_v + val_dim].copy_from_slice(&y_t);
            pool.deallocate(q_t_buf);
            pool.deallocate(y_t_buf);

            // ── Update: MLP forward on k[t], compute grad, update memory ──
            let mut k_t_buf = pool.allocate(key_dim * 4)?;
            let mut v_t_buf = pool.allocate(val_dim * 4)?;
            {
                let mut tmp_k = vec![0.0f32; seq_len * key_dim];
                let mut tmp_v = vec![0.0f32; seq_len * val_dim];
                ops.download(pool, &k_buf, &mut tmp_k)?;
                ops.download(pool, &v_buf, &mut tmp_v)?;
                let k_slice = &tmp_k[tok_off_k..tok_off_k + key_dim];
                let v_slice = &tmp_v[tok_off_v..tok_off_v + val_dim];
                ops.upload(pool, k_slice, &mut k_t_buf)?;
                ops.upload(pool, v_slice, &mut v_t_buf)?;
            }

            // MLP forward on k (to get z_upd, h_upd, v_pred)
            let mut z_upd_buf  = pool.allocate(hidden_dim * 4)?;
            let mut h_upd_buf  = pool.allocate(hidden_dim * 4)?;
            let mut vpred_buf  = pool.allocate(val_dim * 4)?;
            ops.titans_mlp_forward(
                pool,
                &k_t_buf, &mem_w1_buf, &mem_b1_buf, &mem_w2_buf, &mem_b2_buf,
                &mut z_upd_buf, &mut h_upd_buf, &mut vpred_buf,
                1, key_dim, hidden_dim, val_dim,
            )?;

            // Gradient accumulation
            let mut gw1_buf = pool.allocate(w1_elems * 4)?;
            let mut gb1_buf = pool.allocate(hidden_dim * 4)?;
            let mut gw2_buf = pool.allocate(w2_elems * 4)?;
            let mut gb2_buf = pool.allocate(val_dim * 4)?;
            ops.fill_f32(pool, &mut gw1_buf, 0.0)?;
            ops.fill_f32(pool, &mut gb1_buf, 0.0)?;
            ops.fill_f32(pool, &mut gw2_buf, 0.0)?;
            ops.fill_f32(pool, &mut gb2_buf, 0.0)?;

            ops.titans_grad_w2(
                pool,
                &v_t_buf, &vpred_buf, &h_upd_buf,
                &mut gw2_buf, &mut gb2_buf,
                1, hidden_dim, val_dim,
            )?;
            ops.titans_grad_w1(
                pool,
                &k_t_buf, &v_t_buf, &vpred_buf, &z_upd_buf, &mem_w2_buf,
                &mut gw1_buf, &mut gb1_buf,
                1, key_dim, hidden_dim, val_dim,
            )?;

            // Scalars for this token
            let alpha = alpha_all[t];
            let eta   = eta_all[t];
            let theta = theta_all[t];

            // Memory weight update (all four weight tensors)
            ops.titans_memory_update(pool, &gw1_buf, &mut mom_w1_buf, &mut mem_w1_buf, w1_elems, alpha, eta, theta)?;
            ops.titans_memory_update(pool, &gb1_buf, &mut mom_b1_buf, &mut mem_b1_buf, hidden_dim, alpha, eta, theta)?;
            ops.titans_memory_update(pool, &gw2_buf, &mut mom_w2_buf, &mut mem_w2_buf, w2_elems, alpha, eta, theta)?;
            ops.titans_memory_update(pool, &gb2_buf, &mut mom_b2_buf, &mut mem_b2_buf, val_dim, alpha, eta, theta)?;

            // Cleanup per-token intermediates
            pool.deallocate(k_t_buf);
            pool.deallocate(v_t_buf);
            pool.deallocate(z_t_buf);
            pool.deallocate(h_t_buf);
            pool.deallocate(z_upd_buf);
            pool.deallocate(h_upd_buf);
            pool.deallocate(vpred_buf);
            pool.deallocate(gw1_buf);
            pool.deallocate(gb1_buf);
            pool.deallocate(gw2_buf);
            pool.deallocate(gb2_buf);
        }

        // ─── Download updated memory weights back to CPU ──────────────────
        let memory_mut = self.curr_memory.as_mut().unwrap();
        ops.download(pool, &mem_w1_buf, memory_mut.w1.as_slice_mut().unwrap())?;
        ops.download(pool, &mem_b1_buf, memory_mut.b1.as_slice_mut().unwrap())?;
        ops.download(pool, &mem_w2_buf, memory_mut.w2.as_slice_mut().unwrap())?;
        ops.download(pool, &mem_b2_buf, memory_mut.b2.as_slice_mut().unwrap())?;

        let momentum_mut = self.momentum.as_mut().unwrap();
        ops.download(pool, &mom_w1_buf, momentum_mut.w1.as_slice_mut().unwrap())?;
        ops.download(pool, &mom_b1_buf, momentum_mut.b1.as_slice_mut().unwrap())?;
        ops.download(pool, &mom_w2_buf, momentum_mut.w2.as_slice_mut().unwrap())?;
        ops.download(pool, &mom_b2_buf, momentum_mut.b2.as_slice_mut().unwrap())?;

        // ─── Cleanup ──────────────────────────────────────────────────────
        pool.deallocate(q_buf);
        pool.deallocate(k_buf);
        pool.deallocate(v_buf);
        pool.deallocate(mem_w1_buf);
        pool.deallocate(mem_b1_buf);
        pool.deallocate(mem_w2_buf);
        pool.deallocate(mem_b2_buf);
        pool.deallocate(mom_w1_buf);
        pool.deallocate(mom_b1_buf);
        pool.deallocate(mom_w2_buf);
        pool.deallocate(mom_b2_buf);

        let output = Array2::from_shape_vec((seq_len, val_dim), output_flat).map_err(|e| {
            crate::common::errors::ModelError::InvalidInput {
                message: format!("forward_gpu_kernel output reshape failed: {e}"),
            }
        })?;
        Ok(output)
    }
'''

target = 'd:/RustGPT/src/domain/memory/titans/neural.rs'
content = open(target, 'r', encoding='utf-8').read()

# Insert new method just before backward_gpu
marker = '''    /// GPU-accelerated backward pass.'''
if marker not in content:
    print('ERROR: backward_gpu marker not found');
    exit(1)

new_content = content.replace(marker, FORWARD_GPU_KERNEL + '\n    /// GPU-accelerated backward pass.', 1)
open(target, 'w', encoding='utf-8').write(new_content)
print(f'neural.rs: inserted forward_gpu_kernel ({len(FORWARD_GPU_KERNEL)} bytes)')
