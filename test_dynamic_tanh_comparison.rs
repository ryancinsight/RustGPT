use ndarray::Array2;
use llm::dynamic_tanh_norm::DynamicTanhNorm;
use llm::richards::RichardsCurve;

fn main() {
    println!("Testing DynamicTanhNorm with RichardsCurve vs standard tanh");
    
    let dim = 4;
    let batch_size = 2;
    
    // Create test input
    let input = Array2::from_shape_vec((batch_size, dim), 
        vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 1.5]).unwrap();
    
    println!("Input shape: {:?}", input.shape());
    println!("Input values:\n{:?}", input);
    
    // Test with RichardsCurve
    let mut layer = DynamicTanhNorm::new(dim);
    let output = layer.normalize(&input);
    
    println!("\nOutput with RichardsCurve tanh:");
    println!("Shape: {:?}", output.shape());
    println!("Values:\n{:?}", output);
    
    // Test RichardsCurve tanh directly
    let richards = RichardsCurve::tanh(false);
    println!("\nDirect RichardsCurve tanh comparison:");
    for i in 0..batch_size {
        for j in 0..dim {
            let x = input[[i, j]];
            let tanh_val = x.tanh();
            let richards_val = richards.forward_scalar(x as f64) as f32;
            println!("x={:.1}, tanh={:.6}, richards={:.6}, diff={:.6}", 
                     x, tanh_val, richards_val, (tanh_val - richards_val).abs());
        }
    }
    
    println!("\nNote: RichardsCurve::tanh(false) implements 2*sigmoid(2*x) - 1");
    println!("This is mathematically equivalent to tanh(x) but computed differently.");
    println!("The small differences are due to numerical precision in the computation.");
    
    // Test that the layer produces consistent results
    let output2 = layer.normalize(&input);
    println!("\nConsistency check - same input should produce same output:");
    let mut all_match = true;
    for i in 0..batch_size {
        for j in 0..dim {
            let diff = (output[[i, j]] - output2[[i, j]]).abs();
            if diff > 1e-6 {
                all_match = false;
                println!("Mismatch at [{}, {}]: {:.6} vs {:.6}, diff={:.6}", 
                         i, j, output[[i, j]], output2[[i, j]], diff);
            }
        }
    }
    if all_match {
        println!("✓ All outputs match - layer is deterministic");
    }
}