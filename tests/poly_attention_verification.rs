use llm::domain::attention::poly_attention::PolyAttention;
use llm::domain::attention::position::config::{CoPEConfig, CoPEVariant};
use llm::domain::models::config::TitanMemoryConfig;
use llm::domain::network::Layer; // Import Layer trait
use ndarray::Array2;
use rand::Rng;

#[test]
fn test_long_context_streaming_consistency() {
    let embed_dim = 4;
    let num_heads = 1;
    let _head_dim = 4;
    let p = 1;
    let max_pos = 16;
    let window_size = 4; // Small window to force wrapping quickly

    let cope_config = CoPEConfig {
        variant: CoPEVariant::Standard,
        max_pos,
        window_size: Some(window_size),
    };
    let mut attention = PolyAttention::new(embed_dim, num_heads, p, cope_config);

    // Disable random gating for determinism
    attention
        .moh
        .head_selection_config
        .gating
        .use_learned_predictor = false;
    attention.moh.head_selection_config.gating.use_soft_top_p = false;
    // Set fixed gating weights to avoid random initialization issues if any
    attention.moh.gate.curve = llm::domain::richards::RichardsCurve::sigmoid(false);
    // Neutralize gating logic: alpha=0, beta=10 -> sigmoid(10) ~ 1.0
    attention.moh.alpha_g.fill(0.0);
    attention.moh.beta_g.fill(10.0); // Ensures gate ~ 1.0

    // Enable Titan Memory to verify its state persistence
    attention.set_titan_memory_config(TitanMemoryConfig {
        enabled: true,
        scale: 0.1,
        eta: 0.5,
        decay: 0.1,
        ..TitanMemoryConfig::default()
    });

    let seq_len = 20; // 5x window size
    let mut rng = rand::rng();
    let input_data: Vec<f32> = (0..seq_len * embed_dim)
        .map(|_| rng.random_range(-0.5..0.5))
        .collect();
    let input = Array2::from_shape_vec((seq_len, embed_dim), input_data).unwrap();

    // 1. Run Batch Forward
    let batch_output = attention.forward(&input); // causal=true implicit

    // 2. Run Streaming Forward
    let mut stream_outputs = Vec::new();
    for i in 0..seq_len {
        let input_step = input.row(i).to_owned();
        let out_step = attention.forward_step(&input_step);
        stream_outputs.push(out_step);
    }

    // 3. Compare
    let mut max_diff = 0.0;
    for i in 0..seq_len {
        let batch_row = batch_output.row(i);
        let stream_row = &stream_outputs[i];

        let diff = (&batch_row - stream_row).mapv(|x| x.abs()).sum();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-5 {
            println!(
                "Row {}: Batch {:?} vs Stream {:?}",
                i, batch_row, stream_row
            );
            println!("Diff: {}", diff);
        }
    }

    assert!(
        max_diff < 1e-5,
        "Streaming output diverged from batch output. Max diff: {}",
        max_diff
    );
}

#[test]
fn test_poly_attention_forward_backward() {
    let embed_dim = 64;
    let num_heads = 4;
    let p = 3;
    let max_pos = 128;
    let window_size = Some(32);
    let batch_size = 2;

    let cope_config = CoPEConfig {
        variant: CoPEVariant::Standard,
        max_pos,
        window_size,
    };
    let mut poly_attn = PolyAttention::new(embed_dim, num_heads, p, cope_config);

    // Create random input (using vec since ndarray-rand might not be available)
    let input_vec: Vec<f32> = (0..batch_size * embed_dim)
        .map(|x| (x as f32) / 1000.0)
        .collect();
    let input = Array2::from_shape_vec((batch_size, embed_dim), input_vec).unwrap();

    // Forward pass
    // Use the trait method or the public impl method?
    // forward_impl is pub, so we can use it.
    let output = poly_attn.forward_impl(&input, true);

    assert_eq!(output.dim(), (batch_size, embed_dim));

    // Create random output gradients
    let grads_vec: Vec<f32> = (0..batch_size * embed_dim)
        .map(|x| (x as f32) / 1000.0)
        .collect();
    let output_grads = Array2::from_shape_vec((batch_size, embed_dim), grads_vec).unwrap();

    // Backward pass
    // Must import Layer trait to use backward if the method on struct is private
    let input_grads = poly_attn.backward(&output_grads, 0.01);

    assert_eq!(input_grads.dim(), (batch_size, embed_dim));
}

#[test]
fn test_poly_attention_monolithic_shapes() {
    let embed_dim = 64;
    let num_heads = 4;
    let p = 3;
    let max_pos = 128;
    let window_size = Some(32);

    let cope_config = CoPEConfig {
        variant: CoPEVariant::Standard,
        max_pos,
        window_size,
    };
    let poly_attn = PolyAttention::new(embed_dim, num_heads, p, cope_config);

    // Verify monolithic weights shapes
    assert_eq!(poly_attn.w_q.dim(), (embed_dim, embed_dim));
    assert_eq!(poly_attn.w_k.dim(), (embed_dim, embed_dim));
    assert_eq!(poly_attn.w_v.dim(), (embed_dim, embed_dim));
}
