use llm::domain::attention::sliding_window_attention::{
    SlidingWindowAttention, SlidingWindowStreamingWorkspace,
};
use ndarray::Array1;
use rand::Rng;

#[test]
fn test_sliding_window_workspace_equivalence() {
    let embed_dim = 16;
    let window_size = 4;
    let mut attention = SlidingWindowAttention::new(embed_dim, window_size);
    let mut attention_ws = attention.clone(); // Clone for parallel test

    // Ensure weights are identical (clone does deep copy of arrays)

    let mut ws = SlidingWindowStreamingWorkspace::new(embed_dim, window_size);

    let mut rng = rand::rng();

    for i in 0..10 {
        let input_data: Vec<f32> = (0..embed_dim).map(|_| rng.random()).collect();
        let input = Array1::from_shape_vec(embed_dim, input_data).unwrap();

        let out_alloc = attention.forward_step(&input);
        let out_ws = attention_ws.forward_step_with_workspace(&input, &mut ws);

        // Check equivalence
        let diff = &out_alloc - &out_ws;
        let max_diff = diff.mapv(f32::abs).fold(0.0, |a, b| f32::max(a, *b));

        assert!(max_diff < 1e-5, "Step {}: max diff {}", i, max_diff);
    }
}
