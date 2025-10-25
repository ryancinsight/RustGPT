use llm::poly_attention::PolyAttention;
use llm::Layer; // bring trait in scope for compute_gradients
use ndarray::{Array2, s};
use approx::assert_abs_diff_eq;

fn manual_gate_poly_grads(
    pa: &PolyAttention,
    input: &Array2<f32>,
    output_grads: &Array2<f32>,
) -> Vec<f64> {
    let n = input.nrows();
    let dh = pa.head_dim;
    let dk_scale = 1.0f32 / (dh as f32).sqrt();

    // Single head assumption in this test
    assert_eq!(pa.num_heads, 1);

    let head = &pa.heads[0];
    let q = input.dot(&head.w_q);
    let k = input.dot(&head.w_k);
    let v = input.dot(&head.w_v);

    let start = 0;
    let end = dh;
    let w_block = pa.w_out.slice(s![start..end, ..]); // (d_h, D)

    // Per-head gating forward: z = a_h * (X·W_g[:,h]) + b_h; g = phi(z)
    let w_g_col = pa.w_g.slice(s![.., 0..1]);
    let xw_col = input.dot(&w_g_col); // (N,1)
    let a_h = pa.alpha_g[[0, 0]];
    let b_h = pa.beta_g[[0, 0]];

    // z and scaling update for Richards curve
    let mut z_col = xw_col.clone();
    z_col.mapv_inplace(|v| a_h * v + b_h);
    let max_abs_z = z_col
        .iter()
        .fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
    let mut gate_poly = pa.gate_poly.clone();
    gate_poly.update_scaling_from_max_abs(max_abs_z);

    // Scalar poly params
    let a = pa.a[[0, 0]];
    let b = pa.b[[0, 0]];
    let scale = pa.scale[[0, 0]];
    let p_i32 = pa.p as i32;

    // Accumulator for gate_poly parameter grads
    let n_gate_w = gate_poly.weights().len();
    let mut grad_gate_poly_vec = vec![0.0_f64; n_gate_w];

    let q_s = q.as_slice_memory_order().expect("Q contiguous");
    let k_s = k.as_slice_memory_order().expect("K contiguous");
    let v_s = v.as_slice_memory_order().expect("V contiguous");
    let og_s = output_grads
        .as_slice_memory_order()
        .expect("output grads contiguous");

    for i in 0..n {
        // gyg_row = W_out_block · og_i
        let og_i = &og_s[i * pa.embed_dim..(i + 1) * pa.embed_dim];
        let mut gyg_row = vec![0.0f32; dh];
        for u in 0..dh {
            let mut acc = 0.0f32;
            for d in 0..pa.embed_dim {
                acc += og_i[d] * w_block[[u, d]];
            }
            gyg_row[u] = acc;
        }

        // y_pre_row via banded sum (use full prefix for this test)
        let mut y_pre_row = vec![0.0f32; dh];
        let j_start = 0;
        let qi = &q_s[i * dh..(i + 1) * dh];
        for j in j_start..=i {
            let kj = &k_s[j * dh..(j + 1) * dh];
            let mut s_ij = 0.0f32;
            for u in 0..dh {
                s_ij += qi[u] * kj[u];
            }
            s_ij *= dk_scale;
            let sp = s_ij.powi(p_i32);
            let phi_s = scale * (a * sp + b);
            let vj = &v_s[j * dh..(j + 1) * dh];
            for u in 0..dh {
                y_pre_row[u] += phi_s * vj[u];
            }
        }

        // dL/dg = dot(gyg_row, y_pre_row) (no learned threshold here)
        let mut grad_base = 0.0f32;
        for u in 0..dh {
            grad_base += gyg_row[u] * y_pre_row[u];
        }
        let dL_dg = grad_base as f64;

        // Accumulate gate_poly parameter grads for this token
        let zf = z_col[[i, 0]] as f64;
        let token_grads = gate_poly.grad_weights_scalar(zf, dL_dg);
        for wi in 0..n_gate_w {
            grad_gate_poly_vec[wi] += token_grads[wi];
        }
    }

    grad_gate_poly_vec
}

#[test]
fn gate_poly_grads_match_manual_sum_single_head() {
    // Small, deterministic setup
    let embed_dim = 4;
    let num_heads = 1;
    let p = 3; // odd degree
    let max_pos = 8;
    let window_size = None; // use default full band

    let mut pa = PolyAttention::new(embed_dim, num_heads, p, max_pos, window_size);

    // Disable learned threshold routing to match the manual derivation
    // (default is false, but assert to be explicit)
    assert!(!pa.use_learned_threshold);

    // Input (N, D)
    let n = 3;
    let input = Array2::<f32>::from_shape_vec(
        (n, embed_dim),
        vec![
            0.1, -0.2, 0.3, -0.4,
            0.5, 0.6, -0.7, 0.8,
            -0.9, 1.0, -1.1, 1.2,
        ],
    )
    .unwrap();

    // Forward to cache input
    let _out = pa.forward_impl(&input, true);

    // Create output grads (N, D)
    let output_grads = Array2::<f32>::from_shape_vec(
        (n, embed_dim),
        vec![
            0.02, -0.01, 0.03, 0.04,
            -0.05, 0.06, 0.07, -0.08,
            0.09, -0.10, 0.11, -0.12,
        ],
    )
    .unwrap();

    // Library gradients via Layer trait
    let (_input_grads, param_grads): (Array2<f32>, Vec<Array2<f32>>) =
        Layer::compute_gradients(&pa, &input, &output_grads);

    // Index for gate_poly grads: 3*H + 1 + 3 + 3
    let idx_gate_poly = 3 * num_heads + 1 + 3 + 3;
    assert!(idx_gate_poly < param_grads.len());
    let gate_poly_grads = &param_grads[idx_gate_poly];
    assert_eq!(gate_poly_grads.nrows(), 1);

    // Manual gradients
    let manual = manual_gate_poly_grads(&pa, &input, &output_grads);

    // Compare element-wise with tolerance
    let lib_vec: Vec<f64> = gate_poly_grads.iter().map(|&x| x as f64).collect();
    assert_eq!(lib_vec.len(), manual.len());
    for (lib_g, man_g) in lib_vec.iter().zip(manual.iter()) {
        assert_abs_diff_eq!(lib_g, man_g, epsilon = 1e-6);
    }
}