use llm::richards::{RichardsCurve, Variant};

fn main() {
    println!("🚀 Dramatic Parameter Learning Demonstration");
    println!("============================================\n");

    // Create curves with different variants
    let mut sigmoid_curve = RichardsCurve::new_learnable(Variant::Sigmoid);
    let mut none_curve = RichardsCurve::new_learnable(Variant::None);
    let mut fully_learnable_curve = RichardsCurve::new_fully_learnable();

    // Helper function to extract all parameters
    let get_all_params = |curve: &RichardsCurve| -> Vec<f64> {
        vec![
            curve.nu.unwrap_or(1.0),
            curve.k.unwrap_or(1.0),
            curve.m.unwrap_or(0.0),
            curve.beta.unwrap_or(1.0),
            curve.a.unwrap_or(1.0),
            curve.b.unwrap_or(0.0),
            curve.scale.unwrap_or(1.0),
            curve.shift.unwrap_or(0.0),
        ]
    };

    println!("1. Initial Parameter States:");
    println!("  Sigmoid (6 learnable): {:?}", get_all_params(&sigmoid_curve));
    println!("  None (8 learnable): {:?}", get_all_params(&none_curve));
    println!("  Fully learnable (8 learnable): {:?}", get_all_params(&fully_learnable_curve));
    println!();

    // Aggressive training parameters
    let learning_rate = 0.5;  // Much higher learning rate
    let epochs = 100;         // More epochs
    
    println!("2. Aggressive Training (LR={}, Epochs={}):", learning_rate, epochs);
    println!("   Training with synthetic data to maximize parameter changes...\n");

    // Training loop with diverse inputs and targets
    for epoch in 0..epochs {
        // Use multiple diverse training examples per epoch
        let training_examples = [
            (0.1, 0.9),   // Low input, high target
            (0.5, 0.2),   // Mid input, low target  
            (0.9, 0.8),   // High input, high target
            (-0.5, 0.1),  // Negative input, low target
            (1.5, 0.7),   // Large input, mid target
        ];

        for (x, target) in training_examples.iter() {
            // Train Sigmoid curve
            let output = sigmoid_curve.forward_scalar(*x);
            let grad_output = 2.0 * (output - target); // Amplified gradient
            let gradients = sigmoid_curve.grad_weights_scalar(*x, grad_output);
            sigmoid_curve.step(&gradients, learning_rate);

            // Train None curve
            let output = none_curve.forward_scalar(*x);
            let grad_output = 2.0 * (output - target); // Amplified gradient
            let gradients = none_curve.grad_weights_scalar(*x, grad_output);
            none_curve.step(&gradients, learning_rate);

            // Train Fully learnable curve
            let output = fully_learnable_curve.forward_scalar(*x);
            let grad_output = 2.0 * (output - target); // Amplified gradient
            let gradients = fully_learnable_curve.grad_weights_scalar(*x, grad_output);
            fully_learnable_curve.step(&gradients, learning_rate);
        }

        // Print progress every 20 epochs
        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("   Epoch {}:", epoch);
            println!("     Sigmoid: {:?}", get_all_params(&sigmoid_curve));
            println!("     None: {:?}", get_all_params(&none_curve));
            println!("     Fully learnable: {:?}", get_all_params(&fully_learnable_curve));
            println!();
        }
    }

    println!("3. Final Parameter Analysis:");
    let sigmoid_final = get_all_params(&sigmoid_curve);
    let none_final = get_all_params(&none_curve);
    let fully_final = get_all_params(&fully_learnable_curve);

    println!("  Sigmoid final: {:?}", sigmoid_final);
    println!("  None final: {:?}", none_final);
    println!("  Fully learnable final: {:?}", fully_final);
    println!();

    // Calculate parameter changes
    let sigmoid_initial = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let none_initial = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let fully_initial = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];

    println!("4. Parameter Change Magnitudes:");
    
    // Sigmoid changes (only first 6 parameters are learnable)
    let sigmoid_changes: Vec<f64> = sigmoid_initial[0..6].iter()
        .zip(sigmoid_final[0..6].iter())
        .map(|(init, final_val)| (final_val - init).abs())
        .collect();
    println!("  Sigmoid changes: {:?}", sigmoid_changes);
    println!("  Sigmoid max change: {:.6}", sigmoid_changes.iter().fold(0.0f64, |a, &b| a.max(b)));

    // None variant changes (all 8 parameters are learnable)
    let none_changes: Vec<f64> = none_initial.iter()
        .zip(none_final.iter())
        .map(|(init, final_val)| (final_val - init).abs())
        .collect();
    println!("  None changes: {:?}", none_changes);
    println!("  None max change: {:.6}", none_changes.iter().fold(0.0f64, |a, &b| a.max(b)));

    // Fully learnable changes (all 8 parameters are learnable)
    let fully_changes: Vec<f64> = fully_initial.iter()
        .zip(fully_final.iter())
        .map(|(init, final_val)| (final_val - init).abs())
        .collect();
    println!("  Fully learnable changes: {:?}", fully_changes);
    println!("  Fully learnable max change: {:.6}", fully_changes.iter().fold(0.0f64, |a, &b| a.max(b)));
    println!();

    // Verify a,b coefficient behavior
    println!("5. Richards Coefficients (a,b) Analysis:");
    println!("  Sigmoid a,b: {:.6}, {:.6} (should remain 1.0, 0.0)", 
             sigmoid_curve.a.unwrap_or(1.0), sigmoid_curve.b.unwrap_or(0.0));
    println!("  None a,b: {:.6}, {:.6} (should change dramatically)", 
             none_curve.a.unwrap_or(1.0), none_curve.b.unwrap_or(0.0));
    println!("  Fully learnable a,b: {:.6}, {:.6} (should change dramatically)", 
             fully_learnable_curve.a.unwrap_or(1.0), fully_learnable_curve.b.unwrap_or(0.0));

    // Verify equivalence between None and fully_learnable
    let none_vs_fully_diff: f64 = none_final.iter()
        .zip(fully_final.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    
    println!("\n6. Equivalence Check:");
    println!("  None vs Fully learnable difference: {:.10}", none_vs_fully_diff);
    if none_vs_fully_diff < 1e-6 {
        println!("  ✅ None variant and new_fully_learnable() are equivalent!");
    } else {
        println!("  ❌ None variant and new_fully_learnable() differ!");
    }

    println!("\n🎉 Dramatic learning demonstration complete!");
    println!("   The None variant allows ALL parameters to change significantly,");
    println!("   while Sigmoid keeps a,b coefficients fixed at 1.0, 0.0");
}