use llm::richards::{RichardsCurve, Variant};

fn main() {
    println!("Testing None variant functionality...");
    
    // Test 1: new_learnable(Variant::None) should have all 8 parameters learnable
    let none_variant = RichardsCurve::new_learnable(Variant::None);
    let none_weights = none_variant.weights();
    println!("None variant parameter count: {}", none_weights.len());
    assert_eq!(none_weights.len(), 8, "None variant should have 8 learnable parameters");
    
    // Test 2: new_fully_learnable() should be equivalent to new_learnable(Variant::None)
    let fully_learnable = RichardsCurve::new_fully_learnable();
    let fully_learnable_weights = fully_learnable.weights();
    println!("Fully learnable parameter count: {}", fully_learnable_weights.len());
    assert_eq!(fully_learnable_weights.len(), 8, "Fully learnable should have 8 parameters");
    
    // Test 3: Compare outputs - they should be identical for same inputs
    let test_input = 0.5;
    let none_output = none_variant.forward_scalar(test_input);
    let fully_learnable_output = fully_learnable.forward_scalar(test_input);
    println!("None variant output: {}", none_output);
    println!("Fully learnable output: {}", fully_learnable_output);
    assert!((none_output - fully_learnable_output).abs() < 1e-10, "Outputs should be identical");
    
    // Test 4: Verify that None variant has no input/output transformations (like Sigmoid/Gompertz)
    let sigmoid_variant = RichardsCurve::new_learnable(Variant::Sigmoid);
    let sigmoid_output = sigmoid_variant.forward_scalar(test_input);
    println!("Sigmoid variant output: {}", sigmoid_output);
    
    // Test 5: Verify parameter structure - None should have a,b as None (learnable)
    println!("None variant a parameter: {:?}", none_variant.a);
    println!("None variant b parameter: {:?}", none_variant.b);
    assert!(none_variant.a.is_none(), "None variant should have learnable a parameter");
    assert!(none_variant.b.is_none(), "None variant should have learnable b parameter");
    
    // Test 6: Compare with constrained variants
    let sigmoid_constrained = RichardsCurve::new_learnable(Variant::Sigmoid);
    let sigmoid_weights = sigmoid_constrained.weights();
    println!("Sigmoid variant parameter count: {}", sigmoid_weights.len());
    assert_eq!(sigmoid_weights.len(), 6, "Sigmoid variant should have 6 learnable parameters");
    
    println!("Sigmoid variant a parameter: {:?}", sigmoid_constrained.a);
    println!("Sigmoid variant b parameter: {:?}", sigmoid_constrained.b);
    assert!(sigmoid_constrained.a.is_some(), "Sigmoid variant should have fixed a parameter");
    assert!(sigmoid_constrained.b.is_some(), "Sigmoid variant should have fixed b parameter");
    
    println!("✅ All None variant tests passed!");
}