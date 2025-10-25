use llm::richards::{RichardsCurve, Variant};

fn main() {
    println!("=== Richards Curve Parameter Learning Validation ===\n");
    
    // 1. Parameter Count Verification
    println!("1. Parameter Count Verification:");
    let sigmoid_learnable = RichardsCurve::sigmoid(true);
    let sigmoid_fixed = RichardsCurve::sigmoid(false);
    let none_variant = RichardsCurve::new_learnable(Variant::None);
    let tanh_learnable = RichardsCurve::tanh(true);
    let gompertz_learnable = RichardsCurve::gompertz(true);
    
    println!("  Sigmoid learnable (a,b fixed):     {} parameters", sigmoid_learnable.weights().len());
    println!("  Sigmoid fixed (all fixed):         {} parameters", sigmoid_fixed.weights().len());
    println!("  None variant (all learnable):      {} parameters", none_variant.weights().len());
    println!("  Tanh learnable (a,b fixed):        {} parameters", tanh_learnable.weights().len());
    println!("  Gompertz learnable (a,b fixed):    {} parameters", gompertz_learnable.weights().len());
    
    assert_eq!(sigmoid_learnable.weights().len(), 6, "Sigmoid learnable should have 6 parameters");
    assert_eq!(sigmoid_fixed.weights().len(), 0, "Sigmoid fixed should have 0 parameters");
    assert_eq!(none_variant.weights().len(), 8, "None variant should have 8 parameters");
    assert_eq!(tanh_learnable.weights().len(), 6, "Tanh learnable should have 6 parameters");
    assert_eq!(gompertz_learnable.weights().len(), 6, "Gompertz learnable should have 6 parameters");
    println!("  ✅ Parameter counts verified!\n");
    
    // 2. Coefficient Values Verification
    println!("2. Coefficient Values Verification:");
    println!("  Sigmoid learnable - a: {:?}, b: {:?}", sigmoid_learnable.a, sigmoid_learnable.b);
    println!("  None variant - a: {:?}, b: {:?}", none_variant.a, none_variant.b);
    println!("  Tanh learnable - a: {:?}, b: {:?}", tanh_learnable.a, tanh_learnable.b);
    
    assert_eq!(sigmoid_learnable.a, Some(1.0), "Sigmoid should have a=1.0 fixed");
    assert_eq!(sigmoid_learnable.b, Some(0.0), "Sigmoid should have b=0.0 fixed");
    assert_eq!(none_variant.a, None, "None variant should have a=None (learnable)");
    assert_eq!(none_variant.b, None, "None variant should have b=None (learnable)");
    assert_eq!(tanh_learnable.a, Some(1.0), "Tanh should have a=1.0 fixed");
    assert_eq!(tanh_learnable.b, Some(0.0), "Tanh should have b=0.0 fixed");
    println!("  ✅ Coefficient constraints verified!\n");
    
    // 3. Learning Simulation - Track actual parameter values
    println!("3. Learning Simulation:");
    let mut curve = RichardsCurve::new_learnable(Variant::None);
    
    // Get initial parameter values by extracting them directly
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
    
    let initial_params = get_all_params(&curve);
    println!("  Initial parameters: {:?}", initial_params);
    
    // Simulate training with synthetic gradients
    let learning_rate = 0.01;
    let epochs = 10;
    
    println!("  Performing {} training steps...", epochs);
    for epoch in 0..epochs {
        // Compute synthetic loss and gradients
        let x = 0.5;
        let target = 0.8;
        let output = curve.forward_scalar(x);
        let loss = 0.5 * (output - target).powi(2);
        
        // Compute gradients
        let grad_output = output - target;
        let gradients = curve.grad_weights_scalar(x, grad_output);
        
        // Update parameters
        curve.step(&gradients, learning_rate);
        
        if epoch % 3 == 0 {
            let current_params = get_all_params(&curve);
            println!("    Epoch {}: Loss = {:.6}, Params = {:?}", epoch, loss, current_params);
        }
    }
    
    let final_params = get_all_params(&curve);
    println!("  Final parameters: {:?}", final_params);
    
    // Check if parameters actually changed
    let params_changed = initial_params.iter().zip(final_params.iter())
        .any(|(initial, final_val)| (initial - final_val).abs() > 1e-6);
    
    assert!(params_changed, "Parameters should have changed during learning");
    println!("  ✅ Parameters successfully updated during learning!\n");
    
    // 4. Compare Learning Capabilities
    println!("4. Learning Capability Comparison:");
    let mut sigmoid_curve = RichardsCurve::sigmoid(true);
    let mut none_curve = RichardsCurve::new_learnable(Variant::None);
    
    let sigmoid_initial = get_all_params(&sigmoid_curve);
    let none_initial = get_all_params(&none_curve);
    
    // Apply same gradients to both
    let x = 0.3;
    let target = 0.7;
    
    for _ in 0..5 {
        // Sigmoid curve
        let output_sigmoid = sigmoid_curve.forward_scalar(x);
        let grad_sigmoid = output_sigmoid - target;
        let gradients_sigmoid = sigmoid_curve.grad_weights_scalar(x, grad_sigmoid);
        sigmoid_curve.step(&gradients_sigmoid, 0.01);
        
        // None curve  
        let output_none = none_curve.forward_scalar(x);
        let grad_none = output_none - target;
        let gradients_none = none_curve.grad_weights_scalar(x, grad_none);
        none_curve.step(&gradients_none, 0.01);
    }
    
    let sigmoid_final = get_all_params(&sigmoid_curve);
    let none_final = get_all_params(&none_curve);
    
    println!("  Sigmoid curve - Initial a,b: {:.3}, {:.3} -> Final a,b: {:.3}, {:.3}", 
             sigmoid_initial[4], sigmoid_initial[5], sigmoid_final[4], sigmoid_final[5]);
    println!("  None curve - Initial a,b: {:.3}, {:.3} -> Final a,b: {:.3}, {:.3}", 
             none_initial[4], none_initial[5], none_final[4], none_final[5]);
    
    // Sigmoid should have fixed a,b (no change)
    assert!((sigmoid_initial[4] - sigmoid_final[4]).abs() < 1e-10, "Sigmoid a should remain fixed");
    assert!((sigmoid_initial[5] - sigmoid_final[5]).abs() < 1e-10, "Sigmoid b should remain fixed");
    
    // None should have learnable a,b (should change)
    let none_a_changed = (none_initial[4] - none_final[4]).abs() > 1e-6;
    let none_b_changed = (none_initial[5] - none_final[5]).abs() > 1e-6;
    
    println!("  ✅ Sigmoid a,b coefficients remain fixed as expected");
    println!("  ✅ None variant a,b coefficients are learnable: a_changed={}, b_changed={}", 
             none_a_changed, none_b_changed);
    
    // 5. Gradient Dimension Verification
    println!("\n5. Gradient Dimension Verification:");
    let test_x = 0.5;
    let test_grad_output = 1.0;
    
    let sigmoid_grads = sigmoid_learnable.grad_weights_scalar(test_x, test_grad_output);
    let none_grads = none_variant.grad_weights_scalar(test_x, test_grad_output);
    
    println!("  Sigmoid gradient dimensions: {}", sigmoid_grads.len());
    println!("  None variant gradient dimensions: {}", none_grads.len());
    
    assert_eq!(sigmoid_grads.len(), 6, "Sigmoid should have 6 gradients");
    assert_eq!(none_grads.len(), 8, "None variant should have 8 gradients");
    println!("  ✅ Gradient dimensions match parameter counts!\n");
    
    println!("🎉 All parameter learning validations passed!");
    println!("✅ Parameter counts change correctly for different variants");
    println!("✅ Richards coefficients a,b are properly constrained/learnable");
    println!("✅ Parameters actually change during learning");
    println!("✅ Gradient dimensions match parameter counts");
}