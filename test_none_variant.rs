use llm::richards::{RichardsCurve, Variant};

fn main() {
    println!("Testing None variant functionality...");

    // Test 1: new_learnable(Variant::None) should have all 8 parameters learnable
    let none_variant = RichardsCurve::new_learnable(Variant::None);
    let none_weights = none_variant.weights();
    println!("None variant parameter count: {}", none_weights.len());
    assert_eq!(
        none_weights.len(),
        8,
        "None variant should have 8 learnable parameters"
    );

    // Test 2: new_fully_learnable() should be equivalent to new_learnable(Variant::None)
    let fully_learnable = RichardsCurve::new_fully_learnable();
    let fully_learnable_weights = fully_learnable.weights();
    println!(
        "Fully learnable parameter count: {}",
        fully_learnable_weights.len()
    );
    assert_eq!(
        fully_learnable_weights.len(),
        8,
        "Fully learnable should have 8 parameters"
    );

    // Test 3: Compare outputs - they should be identical for same inputs
    let test_input = 0.5;
    let none_output = none_variant.forward_scalar(test_input);
    let fully_learnable_output = fully_learnable.forward_scalar(test_input);
    println!("None variant output: {}", none_output);
    println!("Fully learnable output: {}", fully_learnable_output);
    assert!(
        (none_output - fully_learnable_output).abs() < 1e-10,
        "Outputs should be identical"
    );

    // Test 4: Verify that None variant has no input/output transformations (like Sigmoid/Gompertz)
    let sigmoid_variant = RichardsCurve::new_learnable(Variant::Sigmoid);
    let sigmoid_output = sigmoid_variant.forward_scalar(test_input);
    println!("Sigmoid variant output: {}", sigmoid_output);

    // Test 5: Verify parameter structure - None should have a,b as None (learnable)
    println!(
        "None variant output_gain parameter: {:?}",
        none_variant.output_gain
    );
    println!(
        "None variant output_bias parameter: {:?}",
        none_variant.output_bias
    );
    assert!(
        none_variant.output_gain.is_none(),
        "None variant should have learnable output_gain parameter"
    );
    assert!(
        none_variant.output_bias.is_none(),
        "None variant should have learnable output_bias parameter"
    );

    // Test 6: Compare with constrained variants
    let sigmoid_constrained = RichardsCurve::new_learnable(Variant::Sigmoid);
    let sigmoid_weights = sigmoid_constrained.weights();
    println!("Sigmoid variant parameter count: {}", sigmoid_weights.len());
    assert_eq!(
        sigmoid_weights.len(),
        6,
        "Sigmoid variant should have 6 learnable parameters"
    );

    println!(
        "Sigmoid variant output_gain parameter: {:?}",
        sigmoid_constrained.output_gain
    );
    println!(
        "Sigmoid variant output_bias parameter: {:?}",
        sigmoid_constrained.output_bias
    );
    assert!(
        sigmoid_constrained.output_gain.is_some(),
        "Sigmoid variant should have fixed output_gain parameter"
    );
    assert!(
        sigmoid_constrained.output_bias.is_some(),
        "Sigmoid variant should have fixed output_bias parameter"
    );

    println!("✅ All None variant tests passed!");
}
