mod adam;
mod richards;

use richards::RichardsCurve;

fn main() {
    let richards = RichardsCurve::tanh(false);
    
    println!("RichardsCurve::tanh(false) parameters:");
    println!("nu: {:?}", richards.nu);
    println!("k: {:?}", richards.k);
    println!("m: {:?}", richards.m);
    println!("beta: {:?}", richards.beta);
    println!("a: {:?}", richards.a);
    println!("b: {:?}", richards.b);
    println!("scale: {:?}", richards.scale);
    println!("shift: {:?}", richards.shift);
    
    println!("\nTesting values:");
    for x in [-2.0f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
        let richards_result = richards.forward_scalar(x);
        let std_tanh = x.tanh();
        let diff = (richards_result - std_tanh).abs();
        
        println!("x={:5.1}: Richards={:8.5}, std_tanh={:8.5}, diff={:8.5}", 
                 x, richards_result, std_tanh, diff);
    }
    
    // Let's trace through the computation manually for x=1.0
    let x: f64 = 1.0;
    println!("\nManual computation trace for x={}:", x);
    
    // Step 1: Input scaling
    let scale = 1.0; // from scale parameter
    let shift = 0.0; // from shift parameter
    let cx = scale * x + shift;
    println!("cx = scale * x + shift = {} * {} + {} = {}", scale, x, shift, cx);
    
    // Step 2: Variant-specific input scaling (Tanh uses 2.0)
    let input_scale = 2.0;
    let input = input_scale * cx;
    println!("input = input_scale * cx = {} * {} = {}", input_scale, cx, input);
    
    // Step 3: Richards sigmoid computation
    let nu = 1.0;
    let k = 2.0;
    let m = 0.0;
    let exponent: f64 = -k * (input - m);
    println!("exponent = -k * (input - m) = -{} * ({} - {}) = {}", k, input, m, exponent);
    
    let richards_sigmoid = 1.0 / (1.0 + (exponent.exp()).powf(1.0 / nu));
    println!("richards_sigmoid = 1 / (1 + exp({})^(1/{})) = {}", exponent, nu, richards_sigmoid);
    
    // Step 4: Variant-specific output transformation (Tanh: 2*sigmoid - 1)
    let gate = (richards_sigmoid * 2.0) - 1.0;
    println!("gate = (richards_sigmoid * 2.0) - 1.0 = ({} * 2.0) - 1.0 = {}", richards_sigmoid, gate);
    
    // Step 5: Final affine transformation
    let a = 1.0;
    let b = 0.0;
    let final_result = a * gate + b;
    println!("final_result = a * gate + b = {} * {} + {} = {}", a, gate, b, final_result);
    
    println!("Standard tanh({}) = {}", x, x.tanh());
    println!("Difference: {}", (final_result - x.tanh()).abs());
    
    // Let's also check what 2*sigmoid(2*x) - 1 should give us
    let expected_tanh_approx = 2.0 * (1.0 / (1.0 + (-2.0 * x).exp())) - 1.0;
    println!("Expected 2*sigmoid(2*x) - 1 = {}", expected_tanh_approx);
    println!("Difference from expected: {}", (final_result - expected_tanh_approx).abs());
}