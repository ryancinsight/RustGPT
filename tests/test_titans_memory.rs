use ndarray::Array2;
use llm::models::titans::memory::NeuralMemory;
use llm::network::Layer;
use rand::Rng;

#[test]
fn test_neural_memory_dimensions() {
    let input_dim = 16;
    let key_dim = 8;
    let val_dim = 8;
    let hidden_dim = 32;
    let seq_len = 10;

    let mut memory = NeuralMemory::new(input_dim, key_dim, val_dim, hidden_dim);

    // Create random input
    let mut rng = rand::rng();
    let data: Vec<f32> = (0..seq_len * input_dim).map(|_| rng.random()).collect();
    let input = Array2::from_shape_vec((seq_len, input_dim), data).unwrap();

    let output = memory.forward(&input);

    assert_eq!(output.shape(), &[seq_len, val_dim]);
}

#[test]
fn test_neural_memory_learning() {
    let input_dim = 4;
    let key_dim = 4;
    let val_dim = 4;
    let hidden_dim = 8;
    let seq_len = 5;

    let mut memory = NeuralMemory::new(input_dim, key_dim, val_dim, hidden_dim);

    let mut rng = rand::rng();
    let data: Vec<f32> = (0..seq_len * input_dim).map(|_| rng.random()).collect();
    let input = Array2::from_shape_vec((seq_len, input_dim), data).unwrap();

    let output = memory.forward(&input);

    // Check if the output varies across the sequence (implies state change / distinct inputs processed)
    let first = output.row(0);
    let last = output.row(seq_len - 1);

    // They should be different (random weights, random inputs)
    assert_ne!(first, last);
}

#[test]
fn test_neural_memory_persistence() {
    let input_dim = 4;
    let key_dim = 4;
    let val_dim = 4;
    let hidden_dim = 8;

    let mut memory = NeuralMemory::new(input_dim, key_dim, val_dim, hidden_dim);
    let mut rng = rand::rng();

    // Create two inputs
    let input_a_data: Vec<f32> = (0..input_dim).map(|_| rng.random()).collect();
    let input_a = Array2::from_shape_vec((1, input_dim), input_a_data).unwrap();

    let input_b_data: Vec<f32> = (0..input_dim).map(|_| rng.random()).collect();
    let input_b = Array2::from_shape_vec((1, input_dim), input_b_data.clone()).unwrap();

    // 1. Process A
    memory.forward(&input_a);

    // 2. Process B (should be influenced by A)
    let output_b_with_context = memory.forward(&input_b);

    // 3. Reset
    memory.reset_memory();

    // 4. Process B (fresh start)
    let output_b_fresh = memory.forward(&input_b);

    // They should be different because context A changed the memory weights in step 1.
    assert_ne!(output_b_with_context, output_b_fresh);
}
