use ndarray::{Array1, Array2, Axis};
use llm::domain::network::Layer;
use llm::domain::layers::ssm::{
    Mamba2ScanInput, Mamba2ScanBackwardInput, SelectiveScanner, SelectiveScanConfig
};

#[test]
fn test_mamba2_fused_scan_correctness() {
    let t = 10;
    let num_heads = 2;
    let head_dim = 4;
    let d = num_heads * head_dim; // 8
    let n = 4; // State dim

    // Create inputs
    let u = Array2::<f32>::from_shape_fn((t, d), |(i, j)| (i as f32 + j as f32) * 0.1);
    let a = Array2::<f32>::from_shape_fn((t, num_heads), |(i, h)| 0.9 + (i as f32 * 0.01) + (h as f32 * 0.01));
    let b = Array2::<f32>::from_shape_fn((t, num_heads * n), |(i, k)| 0.5 + (i as f32 * 0.01) - (k as f32 * 0.01));
    let c = Array2::<f32>::from_shape_fn((t, num_heads * n), |(i, k)| 0.5 - (i as f32 * 0.01) + (k as f32 * 0.01));
    let d_skip = Array1::<f32>::from_elem(d, 0.1);

    // Run fused scan
    let scanner = SelectiveScanner::new();
    let input = Mamba2ScanInput {
        u: u.view(),
        a: a.view(),
        b: b.view(),
        c: c.view(),
        d_skip: d_skip.view(),
        head_dim,
    };

    let (state, z, _y) = scanner.fused_mamba2_scan(input);

    // Reference implementation
    let mut state_ref = Array2::<f32>::zeros((t, d * n));
    let mut z_ref = Array2::<f32>::zeros((t, d));
    let mut s = Array1::<f32>::zeros(d * n);

    for ti in 0..t {
        for h in 0..num_heads {
            let a_val = a[[ti, h]];
            // In fused kernel we use a passed in, assuming it's already discretized or whatever.
            // In Mamba2 code: s = a * s_prev + b * u.
            // Wait, in Mamba2 code there is `kk = (1-a)/a_scale`.
            // But `fused_mamba2_scan` takes `b` as input.
            // In `Mamba::forward_mamba2_impl`, `b_eff` is passed which is `b_t * kk`.
            // So `fused_mamba2_scan` implements `s = a * s + b_eff * u`.
            // So my reference here should just use `b`.

            for j_local in 0..head_dim {
                let j = h * head_dim + j_local;
                let u_val = u[[ti, j]];
                let d_val = d_skip[j];
                
                let mut z_val = d_val * u_val;
                
                for k in 0..n {
                    let idx = j * n + k;
                    let b_idx = h * n + k;
                    
                    let s_prev = s[idx];
                    let b_val = b[[ti, b_idx]];
                    let c_val = c[[ti, b_idx]];
                    
                    let s_new = a_val * s_prev + b_val * u_val;
                    s[idx] = s_new;
                    state_ref[[ti, idx]] = s_new;
                    
                    z_val += c_val * s_new;
                }
                z_ref[[ti, j]] = z_val;
            }
        }
    }

    // Compare
    let diff_state = (&state - &state_ref).mapv(|x: f32| x.abs()).sum();
    let diff_z = (&z - &z_ref).mapv(|x: f32| x.abs()).sum();

    println!("State diff: {}", diff_state);
    println!("Z diff: {}", diff_z);

    assert!(diff_state < 1e-4, "State mismatch");
    assert!(diff_z < 1e-4, "Z mismatch");
}

#[test]
fn test_mamba2_backward_shapes() {
    use llm::domain::layers::ssm::{Mamba, MambaConfig};
    
    let d_model = 16;
    let seq_len = 10;
    
    let config = MambaConfig::enhanced();
    let mut mamba = Mamba::new_with_config(d_model, 4, config);
    
    let input = Array2::<f32>::zeros((seq_len, d_model));
    let output = mamba.forward_mamba2(&input);
    
    assert_eq!(output.dim(), (seq_len, d_model));
    
    let grads = Array2::<f32>::ones((seq_len, d_model));
    // backward() calls compute_gradients() which checks cached_kind
    // forward_mamba2 sets cached_kind = Mamba2
    let input_grads = mamba.backward(&grads, 0.001);
    
    assert_eq!(input_grads.dim(), (seq_len, d_model));
    assert!(!input_grads.iter().any(|x: &f32| x.is_nan()), "NaN in grads");
}
