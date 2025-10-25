use ndarray::{Array1, Array2};
use llm::swiglu::SwiGLU;
use llm::llm::Layer;

fn main() {
    println!("Testing SwiGLU with RichardsActivation...");
    
    // Create a SwiGLU layer
    let mut swiglu = SwiGLU::new(4, 8);
    
    // Create test input (batch_size=2, embedding_dim=4)
    let input = Array2::from_shape_vec((2, 4), vec![
        1.0, 0.5, -0.5, 2.0,
        -1.0, 1.5, 0.0, -0.5,
    ]).unwrap();
    
    println!("Input shape: {:?}", input.shape());
    println!("Input:\n{:?}", input);
    
    // Test forward pass
    let output = swiglu.forward(&input);
    println!("\nOutput shape: {:?}", output.shape());
    println!("Output:\n{:?}", output);
    
    // Test parameter count
    let param_count = swiglu.parameters();
    println!("\nTotal parameters: {}", param_count);
    
    // Test gradient computation
    let output_grads = Array2::ones(output.raw_dim());
    let (input_grads, param_grads) = swiglu.compute_gradients(&input, &output_grads);
    
    println!("\nInput gradients shape: {:?}", input_grads.shape());
    println!("Number of parameter gradient blocks: {}", param_grads.len());
    
    for (i, grad) in param_grads.iter().enumerate() {
        println!("Parameter gradient {} shape: {:?}", i, grad.shape());
    }
    
    // Test gradient application
    let lr = 0.001;
    match swiglu.apply_gradients(&param_grads, lr) {
        Ok(()) => println!("\nGradient application successful!"),
        Err(e) => println!("\nGradient application failed: {:?}", e),
    }
    
    // Test another forward pass to ensure everything still works
    let output2 = swiglu.forward(&input);
    println!("\nSecond forward pass output shape: {:?}", output2.shape());
    
    // Check that outputs are different (parameters should have changed)
    let diff_norm = (&output - &output2).mapv(|x| x * x).sum().sqrt();
    println!("Difference norm between outputs: {:.6}", diff_norm);
    
    if diff_norm > 1e-6 {
        println!("✓ Parameters updated successfully (outputs differ)");
    } else {
        println!("⚠ Parameters may not have updated (outputs identical)");
    }
    
    println!("\nSwiGLU with RichardsActivation test completed!");
}