use ndarray::Array2;
use llm::{RichardsGate, Layer};

fn main() {
    println!("🧪 Testing RichardsGate Parameter Learning");
    println!("==========================================");

    // Create a RichardsGate
    let mut gate = RichardsGate::new();

    println!("Initial weights: {:?}", gate.weights());

    // Create some dummy input and gradients
    let input = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -0.5, 0.5, 2.0]).unwrap();
    let output_grads = Array2::from_shape_vec((2, 3), vec![0.1, -0.1, 0.05, 0.2, -0.15, 0.1]).unwrap();

    // Do a few training steps
    for epoch in 0..3 {
        println!("\nEpoch {}", epoch);

        // Forward pass
        let output = gate.forward(&input);
        println!("  Output: {:?}", output.row(0));

        // Compute gradients
        let (input_grads, param_grads) = gate.compute_gradients(&input, &output_grads).unwrap();
        println!("  Parameter gradients: nu={:.6}, k={:.6}, m={:.6}, temp={:.6}",
                 param_grads[0][[0, 0]], param_grads[1][[0, 0]], param_grads[2][[0, 0]], param_grads[3][[0, 0]]);

        // Apply gradients with learning rate
        gate.apply_gradients(&param_grads, 0.1).unwrap();

        println!("  Weights after update: {:?}", gate.weights());
    }

    println!("\n✅ RichardsGate learning test completed!");
    println!("If the weights changed between epochs, learning is working.");
}
