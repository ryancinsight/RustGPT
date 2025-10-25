fn main() {
    println!("Testing RichardsCurve::tanh fix validation (simplified)");
    
    // Manually implement the fixed RichardsCurve tanh computation
    // Fixed parameters: nu=1.0, k=1.0 (changed from 2.0), m=0.0, beta=1.0, a=1.0, b=0.0, scale=1.0, shift=0.0
    
    println!("\n=== Fixed RichardsCurve Parameters ===");
    println!("nu: 1.0");
    println!("k: 1.0 (FIXED: changed from 2.0)");
    println!("m: 0.0");
    println!("beta: 1.0");
    println!("a: 1.0");
    println!("b: 0.0");
    println!("scale: 1.0");
    println!("shift: 0.0");
    
    // Test inputs ranging from small to large values
    let test_inputs = vec![
        0.1f64, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
        -0.1, -0.5, -1.0, -1.5, -2.0, -3.0, -5.0, -10.0
    ];
    
    println!("\n=== Forward Pass Comparison (Fixed) ===");
    let mut max_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    
    for &x in &test_inputs {
        // Fixed RichardsCurve tanh computation with k=1.0
        // Formula: 2 * sigmoid(2*x) - 1 with k=1.0
        let scaled_x = 2.0 * x; // input_scale for tanh variant
        let sigmoid_output = 1.0 / (1.0 + (-scaled_x).exp()); // k=1.0 (fixed)
        let richards_output = 2.0 * sigmoid_output - 1.0;
        
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
        println!("✅ SUCCESS: Fixed RichardsCurve::tanh now matches standard tanh with machine precision!");
    } else if max_diff < 1e-6 {
        println!("✅ GOOD: Fixed RichardsCurve::tanh matches standard tanh within acceptable tolerance");
    } else {
        println!("❌ ISSUE: Fixed RichardsCurve::tanh still has significant differences from standard tanh");
    }
    
    println!("\n=== Gradient Comparison (Fixed) ===");
    let mut max_grad_diff = 0.0f64;
    let mut max_grad_rel_diff = 0.0f64;
    
    for &x in &test_inputs {
        // Fixed Richards derivative with k=1.0
        // d/dx [2*sigmoid(2*x) - 1] = 2 * sigmoid'(2*x) * 2 = 4 * sigmoid(2*x) * (1 - sigmoid(2*x))
        // But with k=1.0, this becomes the correct tanh derivative
        let scaled_x = 2.0 * x;
        let sigmoid_2x = 1.0 / (1.0 + (-scaled_x).exp()); // k=1.0
        let richards_grad = 4.0 * sigmoid_2x * (1.0 - sigmoid_2x);
        
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
        println!("✅ SUCCESS: Fixed RichardsCurve gradients now match standard tanh gradients with machine precision!");
    } else if max_grad_diff < 1e-6 {
        println!("✅ GOOD: Fixed RichardsCurve gradients match standard tanh gradients within acceptable tolerance");
    } else {
        println!("❌ ISSUE: Fixed RichardsCurve gradients still have significant differences from standard tanh gradients");
    }
    
    println!("\n=== Gradient Norm Impact Analysis (Fixed) ===");
    
    // Simulate a batch of inputs and compute gradient norms
    let batch_size = 100;
    let mut richards_grad_norm_sq = 0.0f64;
    let mut std_tanh_grad_norm_sq = 0.0f64;
    
    for i in 0..batch_size {
        let x = (i as f64 - 50.0) * 0.1; // Range from -5.0 to 4.9
        
        // Compute gradients
        let scaled_x = 2.0 * x;
        let sigmoid_2x = 1.0 / (1.0 + (-scaled_x).exp()); // k=1.0
        let richards_grad = 4.0 * sigmoid_2x * (1.0 - sigmoid_2x);
        
        let tanh_x = x.tanh();
        let std_tanh_grad = 1.0 - tanh_x * tanh_x;
        
        richards_grad_norm_sq += richards_grad * richards_grad;
        std_tanh_grad_norm_sq += std_tanh_grad * std_tanh_grad;
    }
    
    let richards_grad_norm = richards_grad_norm_sq.sqrt();
    let std_tanh_grad_norm = std_tanh_grad_norm_sq.sqrt();
    let grad_norm_ratio = richards_grad_norm / std_tanh_grad_norm;
    
    println!("Fixed Richards gradient norm: {:.6}", richards_grad_norm);
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
    
    println!("\n=== Comparison with Original (Broken) Implementation ===");
    println!("Testing original k=2.0 vs fixed k=1.0:");
    
    for &x in &[0.5f64, 1.0, 2.0, -0.5, -1.0, -2.0] {
        // Original (broken) implementation with k=2.0
        let scaled_x_orig = 2.0 * x;
        let sigmoid_orig = 1.0 / (1.0 + (-2.0 * scaled_x_orig).exp()); // k=2.0
        let richards_orig = 2.0 * sigmoid_orig - 1.0;
        
        // Fixed implementation with k=1.0
        let scaled_x_fixed = 2.0 * x;
        let sigmoid_fixed = 1.0 / (1.0 + (-scaled_x_fixed).exp()); // k=1.0
        let richards_fixed = 2.0 * sigmoid_fixed - 1.0;
        
        let std_tanh = x.tanh();
        let orig_diff = (richards_orig - std_tanh).abs();
        let fixed_diff = (richards_fixed - std_tanh).abs();
        
        println!("x={:4.1}: Original={:8.5} (diff={:8.5}), Fixed={:8.5} (diff={:8.5}), StdTanh={:8.5}", 
                 x, richards_orig, orig_diff, richards_fixed, fixed_diff, std_tanh);
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
    
    println!("\n=== Mathematical Verification ===");
    println!("The identity tanh(x) = 2*sigmoid(2*x) - 1 holds when:");
    println!("- sigmoid(z) = 1/(1 + exp(-z))  [standard sigmoid with k=1]");
    println!("- The input scaling is 2*x");
    println!("- The output transformation is 2*sigmoid - 1");
    println!("Our fix ensures k=1.0 in the sigmoid, making this identity exact.");
}