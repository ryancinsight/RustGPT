use ndarray::Array1;
use llm::richards::{RichardsCurve, Variant};

fn main() {
    println!("Testing a,b coefficient constraints for Richards curve variants...\n");

    // Test inputs
    let inputs = Array1::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

    // Test Sigmoid variant with learnable=true (should have a=1, b=0 fixed)
    println!("=== Sigmoid Variant (learnable=true) ===");
    let sigmoid_learnable = RichardsCurve::new_learnable(Variant::Sigmoid);
    let sigmoid_output = sigmoid_learnable.forward(&inputs);
    println!("Input: {:?}", inputs);
    println!("Output: {:?}", sigmoid_output);
    println!("Weights (6 params - nu,k,m,beta,scale,shift): {:?}", sigmoid_learnable.weights());
    println!("a coefficient (fixed): {:?}", sigmoid_learnable.a);
    println!("b coefficient (fixed): {:?}", sigmoid_learnable.b);
    
    // Check range is [0,1] for positive inputs
    let positive_inputs = Array1::from(vec![0.0, 1.0, 2.0, 3.0]);
    let positive_outputs = sigmoid_learnable.forward(&positive_inputs);
    println!("Positive inputs: {:?}", positive_inputs);
    println!("Positive outputs (should be in [0,1]): {:?}", positive_outputs);
    
    println!();

    // Test Gompertz variant with learnable=true (should have a=1, b=0 fixed)
    println!("=== Gompertz Variant (learnable=true) ===");
    let gompertz_learnable = RichardsCurve::new_learnable(Variant::Gompertz);
    let gompertz_output = gompertz_learnable.forward(&inputs);
    println!("Input: {:?}", inputs);
    println!("Output: {:?}", gompertz_output);
    println!("Weights (6 params - nu,k,m,beta,scale,shift): {:?}", gompertz_learnable.weights());
    println!("a coefficient (fixed): {:?}", gompertz_learnable.a);
    println!("b coefficient (fixed): {:?}", gompertz_learnable.b);
    
    let gompertz_positive_outputs = gompertz_learnable.forward(&positive_inputs);
    println!("Positive inputs: {:?}", positive_inputs);
    println!("Positive outputs (should be in [0,1]): {:?}", gompertz_positive_outputs);
    
    println!();

    // Test Tanh variant with learnable=true (should have a=1, b=0 fixed, but with 2σ(2x)-1 transform)
    println!("=== Tanh Variant (learnable=true) ===");
    let tanh_learnable = RichardsCurve::new_learnable(Variant::Tanh);
    let tanh_output = tanh_learnable.forward(&inputs);
    println!("Input: {:?}", inputs);
    println!("Output: {:?}", tanh_output);
    println!("Weights (6 params - nu,k,m,beta,scale,shift): {:?}", tanh_learnable.weights());
    println!("a coefficient (fixed): {:?}", tanh_learnable.a);
    println!("b coefficient (fixed): {:?}", tanh_learnable.b);
    
    let tanh_positive_outputs = tanh_learnable.forward(&positive_inputs);
    println!("Positive inputs: {:?}", positive_inputs);
    println!("Positive outputs (should be in [-1,1]): {:?}", tanh_positive_outputs);
    
    println!();

    // Test new_fully_learnable (should have all 8 parameters learnable including a,b)
    println!("=== Fully Learnable (all 8 params) ===");
    let fully_learnable = RichardsCurve::new_fully_learnable();
    let fully_output = fully_learnable.forward(&inputs);
    println!("Input: {:?}", inputs);
    println!("Output: {:?}", fully_output);
    println!("Weights (8 params - nu,k,m,beta,a,b,scale,shift): {:?}", fully_learnable.weights());
    println!("a coefficient (learnable): {:?}", fully_learnable.a);
    println!("b coefficient (learnable): {:?}", fully_learnable.b);
    
    println!();

    // Compare parameter counts
    println!("=== Parameter Count Comparison ===");
    println!("Sigmoid learnable: {} parameters", sigmoid_learnable.weights().len());
    println!("Gompertz learnable: {} parameters", gompertz_learnable.weights().len());
    println!("Tanh learnable: {} parameters", tanh_learnable.weights().len());
    println!("Fully learnable: {} parameters", fully_learnable.weights().len());
    
    println!("\nTest completed successfully!");
}