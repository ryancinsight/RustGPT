use ndarray::Array1;
use llm::richards::{RichardsAttention, Variant};

fn main() {
    println!("Testing RichardsAttention implementation...");
    
    // Test 1: Sigmoid-based attention (similar to swish)
    let sigmoid_attention = RichardsAttention::sigmoid(false);
    
    // Test with some sample inputs
    let test_inputs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let x = Array1::from(test_inputs.clone());
    
    println!("\n=== Sigmoid-based RichardsAttention ===");
    println!("Input: {:?}", test_inputs);
    
    let output = sigmoid_attention.forward(&x);
    println!("Output (x * sigmoid(x)): {:?}", output.to_vec());
    
    // Test scalar version
    println!("\nScalar tests:");
    for &input in &test_inputs {
        let scalar_output = sigmoid_attention.forward_scalar(input);
        println!("  x={:.1}, x*sigmoid(x)={:.6}", input, scalar_output);
    }
    
    // Test 2: Tanh-based attention
    let tanh_attention = RichardsAttention::tanh(false);
    
    println!("\n=== Tanh-based RichardsAttention ===");
    let tanh_output = tanh_attention.forward(&x);
    println!("Output (x * tanh_variant(x)): {:?}", tanh_output.to_vec());
    
    // Test 3: Learnable sigmoid attention
    let mut learnable_attention = RichardsAttention::new_learnable(Variant::Sigmoid);
    
    println!("\n=== Learnable RichardsAttention ===");
    let learnable_output = learnable_attention.forward(&x);
    println!("Initial output: {:?}", learnable_output.to_vec());
    
    // Test gradient computation
    let derivative = learnable_attention.derivative(&x);
    println!("Derivative: {:?}", derivative.to_vec());
    
    // Test parameter access
    let weights = learnable_attention.weights();
    println!("Current weights: {:?}", weights);
    
    // Test 4: Compare with manual swish computation
    println!("\n=== Comparison with manual swish ===");
    for &input in &test_inputs {
        let sigmoid_val = 1.0 / (1.0 + (-input).exp());
        let manual_swish = input * sigmoid_val;
        let richards_swish = sigmoid_attention.forward_scalar(input);
        println!("  x={:.1}, manual_swish={:.6}, richards_swish={:.6}, diff={:.8}", 
                 input, manual_swish, richards_swish, (manual_swish - richards_swish).abs());
    }
    
    println!("\nRichardsAttention implementation test completed successfully!");
}