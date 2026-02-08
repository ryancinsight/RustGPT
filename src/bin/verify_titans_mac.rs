use llm::domain::attention::poly_attention::PolyAttention;
use llm::domain::memory::titans::mac::TitansMAC;
use llm::domain::memory::titans::NeuralMemory;
use llm::domain::network::Layer;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

fn main() {
    let input_dim = 16;
    let num_heads = 4;
    let memory_hidden_dim = 16;
    let segment_len = 4;
    let persistent_len = 4;
    let seq_len = 8;

    // Initialize components
    let poly = PolyAttention::new(input_dim, num_heads, 3, 64, None);
    let memory = NeuralMemory::new(input_dim, input_dim, input_dim, memory_hidden_dim);
    let mut mac = TitansMAC::new(poly, memory, persistent_len, segment_len);

    // Create random input
    let mut rng = rand::rng();
    let mut input_data = Vec::with_capacity(seq_len * input_dim);
    for _ in 0..seq_len * input_dim {
        input_data.push(rng.random::<f32>());
    }
    let input = Array2::from_shape_vec((seq_len, input_dim), input_data).unwrap();

    // Clone for streaming BEFORE running batch (to ensure identical initial state)
    let mut mac_stream = mac.clone();

    // 1. Batch Forward
    println!("Running Batch Forward...");
    let batch_output = mac.forward(&input);

    // 2. Streaming Forward
    println!("Running Streaming Forward...");
    // No need to reset memory, we have a fresh clone
    
    let mut stream_output = Array2::<f32>::zeros((seq_len, input_dim));
    for i in 0..seq_len {
        let token = input.row(i).to_owned();
        let out_token = mac_stream.forward_step(&token);
        stream_output.row_mut(i).assign(&out_token);
    }

    // 3. Compare
    let diff = &batch_output - &stream_output;
    let max_diff = diff.mapv(|x| x.abs()).iter().fold(0.0f32, |a, &b| a.max(b));
    let mean_diff = diff.mapv(|x| x.abs()).mean().unwrap_or(0.0);

    println!("Max Difference: {:.9}", max_diff);
    println!("Mean Difference: {:.9}", mean_diff);

    if max_diff < 1e-5 {
        println!("SUCCESS: Batch and Streaming outputs match!");
    } else {
        println!("FAILURE: Significant divergence detected.");
        
        // Print first few divergent rows
        // for i in 0..seq_len {
        //     let row_diff = (&batch_output.row(i) - &stream_output.row(i)).mapv(|x| x.abs()).sum();
        //     if row_diff > 1e-5 {
        //         println!("Row {} diff: {}", i, row_diff);
        //         println!("Batch: {:?}", batch_output.row(i));
        //         println!("Stream: {:?}", stream_output.row(i));
        //         if i >= 5 { break; } 
        //     }
        // }
    }
}
