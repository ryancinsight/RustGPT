use llm::richards::{RichardsCurve, Variant};

fn main() {
    println!("🚨 Demonstrating the Learning Bug in RichardsCurve");
    println!("{}", "=".repeat(60));
    
    // Create a None variant (should be fully learnable)
    let mut curve = RichardsCurve::new_learnable(Variant::None);
    
    println!("Initial state:");
    println!("  Weights count: {}", curve.weights().len());
    println!("  Weights: {:?}", curve.weights());
    
    // Simulate multiple training steps with consistent gradients
    let gradients = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]; // 8 gradients for 8 parameters
    let learning_rate = 0.1;
    
    for epoch in 0..5 {
        println!("\nEpoch {}:", epoch);
        println!("  Before step - Weights count: {}", curve.weights().len());
        
        if curve.weights().len() > 0 {
            println!("  Before step - First 3 weights: {:?}", 
                     &curve.weights()[..3.min(curve.weights().len())]);
        }
        
        // Apply gradients
        curve.step(&gradients[..curve.weights().len()], learning_rate);
        
        println!("  After step - Weights count: {}", curve.weights().len());
        if curve.weights().len() > 0 {
            println!("  After step - First 3 weights: {:?}", 
                     &curve.weights()[..3.min(curve.weights().len())]);
        } else {
            println!("  ❌ NO MORE LEARNABLE PARAMETERS!");
        }
    }
    
    println!("\n🔍 Final Analysis:");
    println!("Expected: Parameters should continue learning for all 5 epochs");
    println!("Actual: Parameters stop being learnable after epoch 0");
    println!("Bug: Once a parameter becomes Some(value), it's no longer considered learnable");
}