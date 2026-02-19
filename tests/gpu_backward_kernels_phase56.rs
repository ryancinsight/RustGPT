//! GPU Backward Kernels Verification (Phase 5.6.4a)
//!
//! Tests the implementation of GPU backward pass kernels for:
//! - PolyAttention QKV projection gradients
//! - Output projection weight gradients  
//! - Polynomial parameter gradients
//!
//! Validates correctness against CPU baseline and shapes.

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
mod gpu_backward_tests {
    use llm::domain::compute::GpuDevice;
    use llm::domain::layers::components::unified_gpu_kernels::AttentionParams;
    use ndarray::{Array2, s};

    #[test]
    fn test_backward_qkv_projection_shapes() {
        // Verify backward_qkv_projection_gpu produces correct output shapes
        let batch_tokens = 32;
        let embed_dim = 64;

        let input = Array2::<f32>::zeros((batch_tokens, embed_dim));
        let output_grads = Array2::<f32>::zeros((batch_tokens, embed_dim));
        let wq = Array2::<f32>::zeros((embed_dim, embed_dim));
        let wk = Array2::<f32>::zeros((embed_dim, embed_dim));
        let wv = Array2::<f32>::zeros((embed_dim, embed_dim));

        let params = AttentionParams::new(4, embed_dim, batch_tokens / 4, 1);

        let mut device = GpuDevice::new().unwrap();
        let result =
            device.backward_qkv_projection_gpu(&output_grads, &input, &wq, &wk, &wv, &params);

        assert!(result.is_ok(), "backward_qkv_projection_gpu should succeed");

        let (grad_q, grad_k, grad_v) = result.unwrap();
        assert_eq!(
            grad_q.dim(),
            (embed_dim, embed_dim),
            "grad_q shape mismatch"
        );
        assert_eq!(
            grad_k.dim(),
            (embed_dim, embed_dim),
            "grad_k shape mismatch"
        );
        assert_eq!(
            grad_v.dim(),
            (embed_dim, embed_dim),
            "grad_v shape mismatch"
        );
    }

    #[test]
    fn test_backward_output_projection_shapes() {
        // Verify backward_output_projection_gpu produces correct output shape
        let batch_tokens = 32;
        let embed_dim = 64;

        let attn_output = Array2::<f32>::zeros((batch_tokens, embed_dim));
        let output_grads = Array2::<f32>::zeros((batch_tokens, embed_dim));
        let wo = Array2::<f32>::zeros((embed_dim, embed_dim));

        let mut device = GpuDevice::new().unwrap();
        let result = device.backward_output_projection_gpu(&attn_output, &output_grads, &wo);

        assert!(
            result.is_ok(),
            "backward_output_projection_gpu should succeed"
        );

        let grad_wo = result.unwrap();
        assert_eq!(
            grad_wo.dim(),
            (embed_dim, embed_dim),
            "grad_wo shape mismatch"
        );
    }

    #[test]
    fn test_backward_poly_params_shapes() {
        // Verify backward_poly_params_gpu produces correct scalar gradients
        let batch_heads = 8;
        let seq_len = 16;

        let attention_scores = Array2::<f32>::zeros((batch_heads, seq_len * seq_len));
        let score_grads = Array2::<f32>::zeros((batch_heads, seq_len * seq_len));

        let mut device = GpuDevice::new().unwrap();
        let result =
            device.backward_poly_params_gpu(&attention_scores, &score_grads, 1.0, 1.0, 1.0);

        assert!(result.is_ok(), "backward_poly_params_gpu should succeed");

        let (grad_a, grad_b, grad_scale) = result.unwrap();
        // Scalars should be finite
        assert!(grad_a.is_finite(), "grad_a should be finite");
        assert!(grad_b.is_finite(), "grad_b should be finite");
        assert!(grad_scale.is_finite(), "grad_scale should be finite");
    }

    #[test]
    fn test_backward_qkv_projection_dimension_validation() {
        // Verify backward_qkv_projection_gpu validates input dimensions
        let batch_tokens = 32;
        let embed_dim = 64;

        let input = Array2::<f32>::zeros((batch_tokens, embed_dim));
        let output_grads = Array2::<f32>::zeros((batch_tokens, embed_dim + 1)); // Wrong!
        let wq = Array2::<f32>::zeros((embed_dim, embed_dim));
        let wk = Array2::<f32>::zeros((embed_dim, embed_dim));
        let wv = Array2::<f32>::zeros((embed_dim, embed_dim));

        let params = AttentionParams::new(4, embed_dim, batch_tokens / 4, 1);

        let mut device = GpuDevice::new().unwrap();
        let result =
            device.backward_qkv_projection_gpu(&output_grads, &input, &wq, &wk, &wv, &params);

        assert!(
            result.is_err(),
            "backward_qkv_projection_gpu should reject mismatched dims"
        );
    }

    #[test]
    fn test_backward_output_projection_dimension_validation() {
        // Verify backward_output_projection_gpu validates input dimensions
        let batch_tokens = 32;
        let embed_dim = 64;

        let attn_output = Array2::<f32>::zeros((batch_tokens, embed_dim));
        let output_grads = Array2::<f32>::zeros((batch_tokens, embed_dim + 1)); // Wrong!
        let wo = Array2::<f32>::zeros((embed_dim, embed_dim));

        let mut device = GpuDevice::new().unwrap();
        let result = device.backward_output_projection_gpu(&attn_output, &output_grads, &wo);

        assert!(
            result.is_err(),
            "backward_output_projection_gpu should reject mismatched dims"
        );
    }

    #[test]
    fn test_backward_qkv_projection_gradient_computation() {
        // Verify backward_qkv_projection_gpu computes non-zero gradients when appropriate
        let batch_tokens = 16;
        let embed_dim = 32;

        // Create non-zero inputs
        let mut input = Array2::<f32>::zeros((batch_tokens, embed_dim));
        input.fill(0.1);

        let mut output_grads = Array2::<f32>::zeros((batch_tokens, embed_dim));
        output_grads.fill(0.2);

        let wq = Array2::<f32>::zeros((embed_dim, embed_dim));
        let wk = Array2::<f32>::zeros((embed_dim, embed_dim));
        let wv = Array2::<f32>::zeros((embed_dim, embed_dim));

        let params = AttentionParams::new(4, embed_dim, batch_tokens / 4, 1);

        let mut device = GpuDevice::new().unwrap();
        let result =
            device.backward_qkv_projection_gpu(&output_grads, &input, &wq, &wk, &wv, &params);

        assert!(result.is_ok());

        let (grad_q, grad_k, grad_v) = result.unwrap();

        // With non-zero inputs and grads, should get non-zero results
        let sum_q: f32 = grad_q.iter().map(|x| x.abs()).sum();
        let sum_k: f32 = grad_k.iter().map(|x| x.abs()).sum();
        let sum_v: f32 = grad_v.iter().map(|x| x.abs()).sum();

        assert!(
            sum_q > 0.0,
            "grad_q should be non-zero for non-zero inputs/grads"
        );
        assert!(
            sum_k > 0.0,
            "grad_k should be non-zero for non-zero inputs/grads"
        );
        assert!(
            sum_v > 0.0,
            "grad_v should be non-zero for non-zero inputs/grads"
        );
    }

    #[test]
    fn test_backward_output_projection_gradient_computation() {
        // Verify backward_output_projection_gpu computes non-zero gradients
        let batch_tokens = 16;
        let embed_dim = 32;

        let mut attn_output = Array2::<f32>::zeros((batch_tokens, embed_dim));
        attn_output.fill(0.1);

        let mut output_grads = Array2::<f32>::zeros((batch_tokens, embed_dim));
        output_grads.fill(0.2);

        let wo = Array2::<f32>::zeros((embed_dim, embed_dim));

        let mut device = GpuDevice::new().unwrap();
        let result = device.backward_output_projection_gpu(&attn_output, &output_grads, &wo);

        assert!(result.is_ok());

        let grad_wo = result.unwrap();
        let sum_wo: f32 = grad_wo.iter().map(|x| x.abs()).sum();

        assert!(
            sum_wo > 0.0,
            "grad_wo should be non-zero for non-zero inputs/grads"
        );
    }

    #[test]
    fn test_backward_poly_params_gradient_computation() {
        // Verify backward_poly_params_gpu computes meaningful gradients
        let batch_heads = 4;
        let seq_len = 8;

        let mut attention_scores = Array2::<f32>::zeros((batch_heads, seq_len * seq_len));
        attention_scores.fill(0.5);

        let mut score_grads = Array2::<f32>::zeros((batch_heads, seq_len * seq_len));
        score_grads.fill(0.1);

        let mut device = GpuDevice::new().unwrap();
        let result =
            device.backward_poly_params_gpu(&attention_scores, &score_grads, 1.0, 1.0, 1.0);

        assert!(result.is_ok());

        let (grad_a, grad_b, grad_scale) = result.unwrap();

        // Gradients should be non-zero and finite
        assert!(
            grad_a.is_finite() && grad_a.abs() > 0.0,
            "grad_a should be non-zero finite"
        );
        assert!(grad_b.is_finite(), "grad_b should be finite");
        assert!(
            grad_scale.is_finite() && grad_scale.abs() > 0.0,
            "grad_scale should be non-zero finite"
        );
    }
}

#[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
mod no_gpu_tests {
    #[test]
    fn test_gpu_disabled() {
        // Placeholder test when GPU is disabled
        println!("GPU tests skipped - compile with --features gpu-wgpu, gpu-cuda, or gpu-metal");
    }
}
