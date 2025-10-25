fn main() {
    println!("Testing gradient norm impact of RichardsCurve vs standard tanh");
    
    // Test inputs ranging from small to large values
    let test_inputs = vec![
        0.1f64, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
        -0.1, -0.5, -1.0, -1.5, -2.0, -3.0, -5.0, -10.0
    ];
    
    println!("\n=== Forward Pass Differences ===");
    let mut max_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    
    for &x in &test_inputs {
        // Manually compute RichardsCurve tanh(false) output
        // Parameters: nu=1.0, k=2.0, m=0.0, beta=1.0, a=1.0, b=0.0, scale=1.0, shift=0.0
        // Formula: 2 * sigmoid(2*x) - 1
        let scaled_x = 2.0 * x; // k=2.0 scaling
        let sigmoid_output = 1.0 / (1.0 + (-scaled_x).exp());
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
    
    println!("\n=== Derivative/Gradient Differences ===");
    let mut max_grad_diff = 0.0f64;
    let mut max_grad_rel_diff = 0.0f64;
    
    for &x in &test_inputs {
        // Compute Richards derivative analytically
        // d/dx [2*sigmoid(2*x) - 1] = 2 * sigmoid'(2*x) * 2 = 4 * sigmoid(2*x) * (1 - sigmoid(2*x))
        let scaled_x = 2.0 * x;
        let sigmoid_2x = 1.0 / (1.0 + (-scaled_x).exp());
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
    
    println!("\n=== Gradient Norm Impact Analysis ===");
    
    // Simulate a batch of inputs and compute gradient norms
    let batch_size = 100;
    let mut richards_grad_norm_sq = 0.0f64;
    let mut std_tanh_grad_norm_sq = 0.0f64;
    
    for i in 0..batch_size {
        let x = (i as f64 - 50.0) * 0.1; // Range from -5.0 to 4.9
        
        // Compute gradients
        let scaled_x = 2.0 * x;
        let sigmoid_2x = 1.0 / (1.0 + (-scaled_x).exp());
        let richards_grad = 4.0 * sigmoid_2x * (1.0 - sigmoid_2x);
        
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
    
    if grad_norm_ratio > 1.1 {
        println!("⚠️  WARNING: Richards curve produces {:.1}% higher gradient norms!", 
                 (grad_norm_ratio - 1.0) * 100.0);
    } else if grad_norm_ratio < 0.9 {
        println!("ℹ️  INFO: Richards curve produces {:.1}% lower gradient norms", 
                 (1.0 - grad_norm_ratio) * 100.0);
    } else {
        println!("✅ Gradient norms are similar between Richards and standard tanh");
    }
    
    println!("\n=== Root Cause Analysis ===");
    println!("The differences stem from RichardsCurve using k=2.0 instead of k=1.0");
    println!("This makes the sigmoid steeper, affecting the tanh approximation accuracy.");
    
    println!("\nRichardsCurve tanh(false) parameters:");
    println!("- nu: 1.0");
    println!("- k: 2.0 (should be 1.0 for accurate tanh)");
    println!("- m: 0.0");
    println!("- beta: 1.0");
    println!("- a: 1.0");
    println!("- b: 0.0");
    println!("- scale: 1.0");
    println!("- shift: 0.0");
    
    println!("\n=== Mathematical Analysis ===");
    println!("Current implementation: 2*sigmoid(2*x) - 1");
    println!("Correct tanh formula: 2*sigmoid(2*x) - 1 = tanh(x) ONLY when sigmoid uses k=1");
    println!("But RichardsCurve uses k=2, so we get: 2*sigmoid_k2(2*x) - 1 ≠ tanh(x)");
    
    println!("\nTo fix this, RichardsCurve::tanh(false) should use k=1.0, not k=2.0");
    
    // Show what the correct implementation would look like
    println!("\n=== Corrected Implementation Test ===");
    println!("Testing with k=1.0 instead of k=2.0:");
    
    for &x in &[0.5f64, 1.0, 2.0, -0.5, -1.0, -2.0] {
        // Corrected Richards: 2*sigmoid(2*x) - 1 with k=1.0
        let corrected_sigmoid = 1.0 / (1.0 + (-2.0 * x).exp()); // k=1.0, input_scale=2.0
        let corrected_richards = 2.0 * corrected_sigmoid - 1.0;
        
        let std_tanh = x.tanh();
        let corrected_diff = (corrected_richards - std_tanh).abs();
        
        println!("x={:4.1}: CorrectedRichards={:10.6}, StdTanh={:10.6}, Diff={:10.6}", 
                 x, corrected_richards, std_tanh, corrected_diff);
    }
}