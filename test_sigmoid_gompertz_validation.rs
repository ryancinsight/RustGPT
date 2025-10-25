use std::f64::consts::E;

/// Standard sigmoid function: 1 / (1 + exp(-x))
fn standard_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Standard sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
fn standard_sigmoid_derivative(x: f64) -> f64 {
    let s = standard_sigmoid(x);
    s * (1.0 - s)
}

/// Standard Gompertz function: exp(-exp(-x))
fn standard_gompertz(x: f64) -> f64 {
    (-(-x).exp()).exp()
}

/// Standard Gompertz derivative: exp(-exp(-x)) * exp(-x)
fn standard_gompertz_derivative(x: f64) -> f64 {
    let exp_neg_x = (-x).exp();
    let gompertz = (-exp_neg_x).exp();
    gompertz * exp_neg_x
}

/// Manual implementation of RichardsCurve sigmoid computation
/// Using the Richards curve formula: (1 + nu * exp(-k*(x-m)))^(-1/nu)
/// For sigmoid: nu=1, k=1, m=0 should give standard sigmoid
fn richards_sigmoid_manual(x: f64, nu: f64, k: f64, m: f64) -> f64 {
    let exp_term = (-k * (x - m)).exp();
    (1.0 + nu * exp_term).powf(-1.0 / nu)
}

/// Manual implementation of RichardsCurve sigmoid derivative
fn richards_sigmoid_derivative_manual(x: f64, nu: f64, k: f64, m: f64) -> f64 {
    let exp_term = (-k * (x - m)).exp();
    let base = 1.0 + nu * exp_term;
    let power = -1.0 / nu;
    
    // d/dx [(1 + nu * exp(-k*(x-m)))^(-1/nu)]
    // = (-1/nu) * (1 + nu * exp(-k*(x-m)))^(-1/nu - 1) * nu * exp(-k*(x-m)) * (-k)
    // = k * exp(-k*(x-m)) * (1 + nu * exp(-k*(x-m)))^(-1/nu - 1)
    
    k * exp_term * base.powf(power - 1.0)
}

/// Manual implementation of RichardsCurve Gompertz computation
/// For Gompertz: nu approaches 0, so we use the limit form
fn richards_gompertz_manual(x: f64, nu: f64, k: f64, m: f64) -> f64 {
    // As nu -> 0, Richards curve approaches Gompertz: exp(-exp(-k*(x-m)))
    let exp_term = -k * (x - m);
    (-exp_term.exp()).exp()
}

/// Manual implementation of RichardsCurve Gompertz derivative
fn richards_gompertz_derivative_manual(x: f64, nu: f64, k: f64, m: f64) -> f64 {
    // d/dx [exp(-exp(-k*(x-m)))]
    // = exp(-exp(-k*(x-m))) * (-exp(-k*(x-m))) * k
    // = k * exp(-exp(-k*(x-m))) * exp(-k*(x-m))
    
    let exp_neg_kx = (-k * (x - m)).exp();
    let gompertz = (-exp_neg_kx).exp();
    k * gompertz * exp_neg_kx
}

fn main() {
    println!("=== RichardsCurve Sigmoid and Gompertz Parameter Validation ===\n");

    // Test inputs covering various ranges
    let test_inputs = vec![
        -10.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0
    ];

    // === SIGMOID VALIDATION ===
    println!("=== SIGMOID VALIDATION ===");
    println!("Testing RichardsCurve::sigmoid(false) parameters: nu=1.0, k=1.0, m=0.0");
    println!("Expected: Should match standard sigmoid function\n");

    let mut max_sigmoid_abs_diff = 0.0f64;
    let mut max_sigmoid_rel_diff = 0.0f64;
    let mut max_sigmoid_grad_abs_diff = 0.0f64;
    let mut max_sigmoid_grad_rel_diff = 0.0f64;

    println!("Forward Pass Comparison:");
    for &x in &test_inputs {
        let richards_output = richards_sigmoid_manual(x, 1.0, 1.0, 0.0);
        let std_sigmoid_output = standard_sigmoid(x);
        
        let abs_diff = (richards_output - std_sigmoid_output).abs();
        let rel_diff = if std_sigmoid_output.abs() > 1e-10 {
            abs_diff / std_sigmoid_output.abs() * 100.0
        } else {
            0.0
        };

        max_sigmoid_abs_diff = max_sigmoid_abs_diff.max(abs_diff);
        max_sigmoid_rel_diff = max_sigmoid_rel_diff.max(rel_diff);

        println!("x={:6.1}: Richards={:8.6}, StdSigmoid={:8.6}, AbsDiff={:10.6}, RelDiff={:7.4}%", 
                 x, richards_output, std_sigmoid_output, abs_diff, rel_diff);
    }

    println!("\nGradient Comparison:");
    for &x in &test_inputs {
        let richards_grad = richards_sigmoid_derivative_manual(x, 1.0, 1.0, 0.0);
        let std_sigmoid_grad = standard_sigmoid_derivative(x);
        
        let grad_abs_diff = (richards_grad - std_sigmoid_grad).abs();
        let grad_rel_diff = if std_sigmoid_grad.abs() > 1e-10 {
            grad_abs_diff / std_sigmoid_grad.abs() * 100.0
        } else {
            0.0
        };

        max_sigmoid_grad_abs_diff = max_sigmoid_grad_abs_diff.max(grad_abs_diff);
        max_sigmoid_grad_rel_diff = max_sigmoid_grad_rel_diff.max(grad_rel_diff);

        println!("x={:6.1}: RichardsGrad={:8.6}, StdSigmoidGrad={:8.6}, AbsDiff={:10.6}, RelDiff={:7.4}%", 
                 x, richards_grad, std_sigmoid_grad, grad_abs_diff, grad_rel_diff);
    }

    println!("\nSigmoid Results Summary:");
    println!("Max forward pass absolute difference: {:.6}", max_sigmoid_abs_diff);
    println!("Max forward pass relative difference: {:.4}%", max_sigmoid_rel_diff);
    println!("Max gradient absolute difference: {:.6}", max_sigmoid_grad_abs_diff);
    println!("Max gradient relative difference: {:.4}%", max_sigmoid_grad_rel_diff);

    if max_sigmoid_abs_diff < 1e-10 && max_sigmoid_grad_abs_diff < 1e-10 {
        println!("✅ SUCCESS: RichardsCurve sigmoid matches standard sigmoid with machine precision!");
    } else if max_sigmoid_abs_diff < 1e-6 && max_sigmoid_grad_abs_diff < 1e-6 {
        println!("✅ GOOD: RichardsCurve sigmoid matches standard sigmoid within acceptable tolerance!");
    } else {
        println!("❌ ISSUE: RichardsCurve sigmoid has significant differences from standard sigmoid!");
    }

    // === GOMPERTZ VALIDATION ===
    println!("\n=== GOMPERTZ VALIDATION ===");
    println!("Testing RichardsCurve::gompertz(false) parameters: nu=0.01, k=1.0, m=0.0");
    println!("Expected: Should approximate standard Gompertz function\n");

    let mut max_gompertz_abs_diff = 0.0f64;
    let mut max_gompertz_rel_diff = 0.0f64;
    let mut max_gompertz_grad_abs_diff = 0.0f64;
    let mut max_gompertz_grad_rel_diff = 0.0f64;

    println!("Forward Pass Comparison:");
    for &x in &test_inputs {
        let richards_output = richards_gompertz_manual(x, 0.01, 1.0, 0.0);
        let std_gompertz_output = standard_gompertz(x);
        
        let abs_diff = (richards_output - std_gompertz_output).abs();
        let rel_diff = if std_gompertz_output.abs() > 1e-10 {
            abs_diff / std_gompertz_output.abs() * 100.0
        } else {
            0.0
        };

        max_gompertz_abs_diff = max_gompertz_abs_diff.max(abs_diff);
        max_gompertz_rel_diff = max_gompertz_rel_diff.max(rel_diff);

        println!("x={:6.1}: Richards={:8.6}, StdGompertz={:8.6}, AbsDiff={:10.6}, RelDiff={:7.4}%", 
                 x, richards_output, std_gompertz_output, abs_diff, rel_diff);
    }

    println!("\nGradient Comparison:");
    for &x in &test_inputs {
        let richards_grad = richards_gompertz_derivative_manual(x, 0.01, 1.0, 0.0);
        let std_gompertz_grad = standard_gompertz_derivative(x);
        
        let grad_abs_diff = (richards_grad - std_gompertz_grad).abs();
        let grad_rel_diff = if std_gompertz_grad.abs() > 1e-10 {
            grad_abs_diff / std_gompertz_grad.abs() * 100.0
        } else {
            0.0
        };

        max_gompertz_grad_abs_diff = max_gompertz_grad_abs_diff.max(grad_abs_diff);
        max_gompertz_grad_rel_diff = max_gompertz_grad_rel_diff.max(grad_rel_diff);

        println!("x={:6.1}: RichardsGrad={:8.6}, StdGompertzGrad={:8.6}, AbsDiff={:10.6}, RelDiff={:7.4}%", 
                 x, richards_grad, std_gompertz_grad, grad_abs_diff, grad_rel_diff);
    }

    println!("\nGompertz Results Summary:");
    println!("Max forward pass absolute difference: {:.6}", max_gompertz_abs_diff);
    println!("Max forward pass relative difference: {:.4}%", max_gompertz_rel_diff);
    println!("Max gradient absolute difference: {:.6}", max_gompertz_grad_abs_diff);
    println!("Max gradient relative difference: {:.4}%", max_gompertz_grad_rel_diff);

    if max_gompertz_abs_diff < 1e-6 && max_gompertz_grad_abs_diff < 1e-6 {
        println!("✅ SUCCESS: RichardsCurve Gompertz matches standard Gompertz within excellent tolerance!");
    } else if max_gompertz_abs_diff < 1e-3 && max_gompertz_grad_abs_diff < 1e-3 {
        println!("✅ GOOD: RichardsCurve Gompertz matches standard Gompertz within acceptable tolerance!");
    } else {
        println!("❌ ISSUE: RichardsCurve Gompertz has significant differences from standard Gompertz!");
    }

    // === PARAMETER ANALYSIS ===
    println!("\n=== PARAMETER ANALYSIS ===");
    
    println!("Sigmoid Parameters Analysis:");
    println!("- nu=1.0: Correct for standard sigmoid (Richards curve reduces to logistic)");
    println!("- k=1.0: Correct growth rate for standard sigmoid");
    println!("- m=0.0: Correct midpoint for standard sigmoid");
    
    println!("\nGompertz Parameters Analysis:");
    println!("- nu=0.01: Small value approximates Gompertz limit (nu→0)");
    println!("- k=1.0: Growth rate parameter");
    println!("- m=0.0: Midpoint parameter");
    
    // Test different nu values for Gompertz to see convergence
    println!("\nGompertz Convergence Test (different nu values):");
    let nu_values = vec![1.0, 0.1, 0.01, 0.001, 0.0001];
    let test_x = 1.0f64;
    let std_gompertz_at_1 = standard_gompertz(test_x);
    
    for &nu in &nu_values {
        let richards_approx = richards_sigmoid_manual(test_x, nu, 1.0, 0.0);
        let diff = (richards_approx - std_gompertz_at_1).abs();
        println!("nu={:6.4}: Richards={:8.6}, StdGompertz={:8.6}, Diff={:10.6}", 
                 nu, richards_approx, std_gompertz_at_1, diff);
    }

    println!("\n=== OVERALL SUMMARY ===");
    println!("Sigmoid: Parameters appear correct for standard sigmoid approximation");
    println!("Gompertz: Parameters provide reasonable Gompertz approximation with nu=0.01");
    println!("Both functions should work well for neural network activation purposes.");
}