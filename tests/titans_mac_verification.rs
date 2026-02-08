use ndarray::{Array1, Array2};
use rand::Rng;
use llm::domain::{
    attention::poly_attention::PolyAttention,
    memory::titans::{mac::TitansMAC, neural::NeuralMemory},
    network::Layer,
};

#[test]
fn test_titans_mac_streaming_consistency() {
    let embed_dim = 32;
    let num_heads = 4;
    let persistent_len = 8;
    let segment_len = 1; // Must be 1 to match streaming behavior exactly
    
    // Setup
    let poly = PolyAttention::new(embed_dim, num_heads, 3, 128, None);
    // val_dim must match embed_dim because memory is used as context token
    // new(input_dim, key_dim, val_dim, memory_hidden_dim)
    let memory = NeuralMemory::new(embed_dim, 16, embed_dim, 16);
    
    let mut mac = TitansMAC::new(poly, memory, persistent_len, segment_len);
    
    // Inputs
    let seq_len = 20;
    let mut rng = rand::rng();
    let input_data: Vec<f32> = (0..seq_len * embed_dim).map(|_| rng.random()).collect();
    let input = Array2::from_shape_vec((seq_len, embed_dim), input_data).unwrap();
    
    // 1. Batch Forward (segment_len = 1)
    println!("Running Batch Forward...");
    let batch_output = mac.forward(&input);
    
    // 2. Streaming Forward
    println!("Running Streaming Forward...");
    // Reset memory for fairness (though MAC resets inside forward(), streaming does not auto-reset?)
    // TitansMAC::forward() calls self.memory.reset_memory().
    // We need to manually reset memory for streaming if we want to match.
    // However, TitansMAC doesn't expose reset_memory() publicly?
    // self.memory is public.
    mac.memory.reset_memory();
    // Also clear streaming workspace if needed? No, workspace is stateless between steps mostly.
    
    let mut stream_output = Array2::zeros((seq_len, embed_dim));
    let mut step_out = Array1::zeros(embed_dim);
    
    for i in 0..seq_len {
        let input_step = input.row(i);
        mac.forward_step_into(&input_step, &mut step_out);
        stream_output.row_mut(i).assign(&step_out);
    }
    
    // 3. Compare
    let diff = &batch_output - &stream_output;
    let max_diff = diff.mapv(|x: f32| x.abs()).fold(0.0f32, |a, b| f32::max(a, *b));
    let mse = diff.mapv(|x: f32| x * x).sum() / diff.len() as f32;
    
    println!("Max Diff: {}", max_diff);
    println!("MSE: {}", mse);
    
    // Tolerance might need to be higher due to float accumulation differences
    // Batch path uses process_segment -> forward -> dot
    // Streaming path uses forward_step_with_context_into -> general_mat_vec_mul
    // Operations are mathematically equivalent but order differs slightly.
    assert!(max_diff < 1e-4, "Streaming output diverges from batch output");
}
