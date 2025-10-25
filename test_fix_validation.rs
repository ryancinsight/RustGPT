use std::path::Path;

mod richards;
use richards::RichardsCurve;

fn main() {
    println!("Testing RichardsCurve::tanh fix validation");
    
    // Test the fixed RichardsCurve::tanh(false)
    let richards = RichardsCurve::tanh(false);
    
    println!("\n=== Fixed RichardsCurve Parameters ===");
    println!("nu: {:?}", richards.nu);
    println!("k: {:?}", richards.k);
    println!("m: {:?}", richards.m);
    println!("beta: {:?}", richards.beta);
    println!("a: {:?}", richards.a);
    println!("b: {:?}", richards.b);
    println!("scale: {:?}", richards.scale);
    println!("shift: {:?}", richards.shift);
    
    // Test inputs ranging from small to large values
    let test_inputs = vec![
        0.1f64, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
        -0.1, -0.5, -1.0, -1.5, -2.0, -3.0, -5.0, -10.0
    ];
    
    println!("\n=== Forward Pass Comparison (Fixed) ===");
    let mut max_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    
    for &x in &test_inputs {
        let richards_output = richards.forward_scalar(x);
        let std_tanh_output = x.tanh();
        let diff = (richards_output - std_tanh_output).abs();
        let rel_diff = if std_tanh_output.abs() > 1e-10 {
            diff / std_tanh_output.abs()
        } else {
            diff
        };
        
        max_diff = max_diff.max(diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        
        println!("x={:6.1}: Richards={:10.6}, StdTanh={:10.6}, AbsDiff={:10.6}, RelDiff={:8.4}%", 
                 x, richards_output, std_tanh_output, diff, rel_diff * 100.0);
    }
    
    println!("\nMax absolute difference: {:.6}", max_diff);
    println!("Max relative difference: {:.4}%", max_rel_diff * 100.0);
    
    // Check if the fix is successful
    if max_diff < 1e-10 {
        println!("✅ SUCCESS: RichardsCurve::tanh now matches standard tanh with machine precision!");
    } else if max_diff < 1e-6 {
        println!("✅ GOOD: RichardsCurve::tanh matches standard tanh within acceptable tolerance");
    } else {
        println!("❌ ISSUE: RichardsCurve::tanh still has significant differences from standard tanh");
    }
    
    println!("\n=== Gradient Comparison (Fixed) ===");
    let mut max_grad_diff = 0.0f64;
    let mut max_grad_rel_diff = 0.0f64;
    
    for &x in &test_inputs {
        let richards_grad = richards.backward_scalar(x);
        
        // Standard tanh derivative: 1 - tanh²(x)
        let tanh_x = x.tanh();
        let std_tanh_grad = 1.0 - tanh_x * tanh_x;
        
        let grad_diff = (richards_grad - std_tanh_grad).abs();
        let grad_rel_diff = if std_tanh_grad.abs() > 1e-10 {
            grad_diff / std_tanh_grad.abs()
        } else {
            grad_diff
        };
        
        max_grad_diff = max_grad_diff.max(grad_diff);
        max_grad_rel_diff = max_grad_rel_diff.max(grad_rel_diff);
        
        println!("x={:6.1}: RichardsGrad={:10.6}, StdTanhGrad={:10.6}, AbsDiff={:10.6}, RelDiff={:8.4}%", 
                 x, richards_grad, std_tanh_grad, grad_diff, grad_rel_diff * 100.0);
    }
    
    println!("\nMax gradient absolute difference: {:.6}", max_grad_diff);
    println!("Max gradient relative difference: {:.4}%", max_grad_rel_diff * 100.0);
    
    // Check gradient fix
    if max_grad_diff < 1e-10 {
        println!("✅ SUCCESS: RichardsCurve gradients now match standard tanh gradients with machine precision!");
    } else if max_grad_diff < 1e-6 {
        println!("✅ GOOD: RichardsCurve gradients match standard tanh gradients within acceptable tolerance");
    } else {
        println!("❌ ISSUE: RichardsCurve gradients still have significant differences from standard tanh gradients");
    }
    
    println!("\n=== Gradient Norm Impact Analysis (Fixed) ===");
    
    // Simulate a batch of inputs and compute gradient norms
    let batch_size = 100;
    let mut richards_grad_norm_sq = 0.0f64;
    let mut std_tanh_grad_norm_sq = 0.0f64;
    
    for i in 0..batch_size {
        let x = (i as f64 - 50.0) * 0.1; // Range from -5.0 to 4.9
        
        // Compute gradients
        let richards_grad = richards.backward_scalar(x);
        
        let tanh_x = x.tanh();
        let std_tanh_grad = 1.0 - tanh_x * tanh_x;
        
        richards_grad_norm_sq += richards_grad * richards_grad;
        std_tanh_grad_norm_sq += std_tanh_grad * std_tanh_grad;
    }
    
    let richards_grad_norm = richards_grad_norm_sq.sqrt();
    let std_tanh_grad_norm = std_tanh_grad_norm_sq.sqrt();
    let grad_norm_ratio = richards_grad_norm / std_tanh_grad_norm;
    
    println!("Richards gradient norm: {:.6}", richards_grad_norm);
    println!("Standard tanh gradient norm: {:.6}", std_tanh_grad_norm);
    println!("Gradient norm ratio (Richards/Standard): {:.6}", grad_norm_ratio);
    
    if (grad_norm_ratio - 1.0).abs() < 0.01 {
        println!("✅ EXCELLENT: Gradient norms are nearly identical!");
    } else if grad_norm_ratio > 1.1 {
        println!("⚠️  WARNING: Richards curve produces {:.1}% higher gradient norms!", 
                 (grad_norm_ratio - 1.0) * 100.0);
    } else if grad_norm_ratio < 0.9 {
        println!("ℹ️  INFO: Richards curve produces {:.1}% lower gradient norms", 
                 (1.0 - grad_norm_ratio) * 100.0);
    } else {
        println!("✅ Gradient norms are similar between Richards and standard tanh");
    }
    
    println!("\n=== Summary ===");
    println!("Fix applied: Changed k parameter from 2.0 to 1.0 in RichardsCurve::tanh(false)");
    println!("This ensures that 2*sigmoid(2*x) - 1 = tanh(x) mathematically");
    
    if max_diff < 1e-10 && max_grad_diff < 1e-10 {
        println!("🎉 COMPLETE SUCCESS: Both forward pass and gradients now match standard tanh!");
    } else if max_diff < 1e-6 && max_grad_diff < 1e-6 {
        println!("✅ SUCCESS: Fix resolves the approximation issues within acceptable tolerance");
    } else {
        println!("❌ PARTIAL: Fix may need further refinement");
    }
}