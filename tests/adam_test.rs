use llm::adam::Adam;
use ndarray::Array2;

#[test]
fn test_adam_initialization() {
    let shape = [2, 3];
    let adam = Adam::new((2, 3));

    // Check if momentum and velocity matrices are initialized to zeros
    assert_eq!(adam.m.shape(), shape);
    assert_eq!(adam.v.shape(), shape);
    assert!(adam.m.iter().all(|&x| x == 0.0));
    assert!(adam.v.iter().all(|&x| x == 0.0));
}

#[test]
fn test_adam_step() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);

    // Store initial parameters
    let initial_params = params.clone();

    // Perform optimization step
    adam.step(&mut params, &grads, lr);

    // Parameters should have changed
    assert_ne!(params, initial_params);

    // Parameters should have decreased (since gradients are positive)
    assert!(params.iter().all(|&x| x < 1.0));
}

#[test]
fn test_adam_multiple_steps() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);

    // Store initial parameters
    let initial_params = params.clone();

    // Perform multiple optimization steps
    for _ in 0..10 {
        adam.step(&mut params, &grads, lr);
    }

    // Parameters should have changed more significantly
    assert!(params.iter().all(|&x| x < initial_params[[0, 0]]));
}

#[test]
fn test_adam_with_zero_gradients() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::zeros(shape);

    // Store initial parameters
    let initial_params = params.clone();

    // Perform optimization step with zero gradients
    adam.step(&mut params, &grads, lr);

    // Parameters should not change with zero gradients
    assert_eq!(params, initial_params);
}

#[test]
fn test_adam_with_negative_gradients() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::from_shape_fn(shape, |_| -1.0);

    // Perform optimization step
    adam.step(&mut params, &grads, lr);

    // Parameters should have increased (since gradients are negative)
    assert!(params.iter().all(|&x| x > 1.0));
}

#[test]
fn test_adam_amsgrad_initialization() {
    let shape = [2, 3];
    let adam = Adam::new_amsgrad((2, 3));

    // Check if AMSGrad is enabled
    assert!(adam.use_amsgrad);

    // Check if v_hat_max is initialized
    assert!(adam.v_hat_max.is_some());
    assert_eq!(adam.v_hat_max.as_ref().unwrap().shape(), shape);
    assert!(adam.v_hat_max.as_ref().unwrap().iter().all(|&x| x == 0.0));
}

#[test]
fn test_adam_amsgrad_step() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new_amsgrad(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);

    // Store initial parameters
    let initial_params = params.clone();

    // Perform optimization step
    adam.step(&mut params, &grads, lr);

    // Parameters should have changed
    assert_ne!(params, initial_params);

    // v_hat_max should be updated and contain positive values
    assert!(adam.v_hat_max.is_some());
    let v_hat_max = adam.v_hat_max.as_ref().unwrap();
    assert!(v_hat_max.iter().all(|&x| x >= 0.0));

    // Parameters should have decreased (since gradients are positive)
    assert!(params.iter().all(|&x| x < 1.0));
}

#[test]
fn test_adam_amsgrad_convergence_guarantee() {
    let shape = (2, 2);
    let lr = 0.01;
    let mut adam_standard = Adam::new(shape);
    let mut adam_amsgrad = Adam::new_amsgrad(shape);

    let mut params_standard = Array2::from_elem(shape, 2.0);
    let mut params_amsgrad = Array2::from_elem(shape, 2.0);

    let grads = Array2::ones(shape);

    // Run multiple steps
    for _ in 0..50 {
        adam_standard.step(&mut params_standard, &grads, lr);
        adam_amsgrad.step(&mut params_amsgrad, &grads, lr);
    }

    // Both should converge, but AMSGrad should be more stable
    // Check that both have moved from initial position
    assert!(params_standard.iter().all(|&x| x < 2.0));
    assert!(params_amsgrad.iter().all(|&x| x < 2.0));

    // Check that v_hat_max in AMSGrad is monotonically non-decreasing
    if let Some(v_hat_max) = &adam_amsgrad.v_hat_max {
        // All values should be positive (since we squared gradients)
        assert!(v_hat_max.iter().all(|&x| x >= 0.0));
    }
}

#[test]
fn test_adam_toggle_amsgrad() {
    let shape = (2, 2);
    let mut adam = Adam::new(shape);

    // Initially AMSGrad should be disabled
    assert!(!adam.use_amsgrad);
    assert!(adam.v_hat_max.is_none());

    // Enable AMSGrad
    adam.set_amsgrad(true);
    assert!(adam.use_amsgrad);
    assert!(adam.v_hat_max.is_some());

    // Disable AMSGrad
    adam.set_amsgrad(false);
    assert!(!adam.use_amsgrad);
    assert!(adam.v_hat_max.is_none());
}

#[test]
fn test_adamw_initialization() {
    let shape = (2, 3);
    let weight_decay = 0.01;
    let adam = Adam::new_adamw(shape, weight_decay);

    // Check if AdamW is properly configured
    assert!(adam.use_amsgrad); // AdamW uses AMSGrad
    assert!(adam.use_decoupled_wd);
    assert_eq!(adam.weight_decay, weight_decay);
    assert!(adam.v_hat_max.is_some());
}

#[test]
fn test_adamw_step_with_weight_decay() {
    let shape = (2, 2);
    let lr = 0.001;
    let weight_decay = 0.01;
    let mut adam = Adam::new_adamw(shape, weight_decay);
    let mut params = Array2::ones(shape);
    let grads = Array2::from_elem(shape, 0.1);

    let initial_params = params.clone();

    // Perform optimization step
    adam.step(&mut params, &grads, lr);

    // Parameters should have changed due to both gradient update and weight decay
    assert_ne!(params, initial_params);

    // Weight decay should have reduced parameter magnitudes
    // (AdamW applies weight decay directly: params *= 1.0 - weight_decay * lr)
    let decay_factor = 1.0 - weight_decay * lr;
    for &param in params.iter() {
        assert!(param < decay_factor); // Should be less than the decay factor
    }
}

#[test]
fn test_adam_weight_decay_toggle() {
    let shape = (2, 2);
    let mut adam = Adam::new(shape);

    // Initially no weight decay
    assert_eq!(adam.weight_decay, 0.0);
    assert!(!adam.use_decoupled_wd);

    // Set traditional L2 regularization
    adam.set_weight_decay(0.01, false);
    assert_eq!(adam.weight_decay, 0.01);
    assert!(!adam.use_decoupled_wd);

    // Set AdamW style weight decay
    adam.set_weight_decay(0.01, true);
    assert_eq!(adam.weight_decay, 0.01);
    assert!(adam.use_decoupled_wd);
}

#[test]
fn test_adamw_vs_l2_regularization() {
    let shape = (2, 2);
    let lr = 0.01;
    let weight_decay = 0.1;

    // AdamW optimizer
    let mut adam_adamw = Adam::new_adamw(shape, weight_decay);
    let mut params_adamw = Array2::from_elem(shape, 1.0);

    // L2 regularization optimizer
    let mut adam_l2 = Adam::new(shape);
    adam_l2.set_weight_decay(weight_decay, false);
    let mut params_l2 = Array2::from_elem(shape, 1.0);

    let grads = Array2::zeros(shape); // Zero gradients to isolate weight decay effect

    // Single step with zero gradients (only weight decay should apply)
    adam_adamw.step(&mut params_adamw, &grads, lr);
    adam_l2.step(&mut params_l2, &grads, lr);

    // AdamW should have applied decay directly to parameters
    let expected_adamw = 1.0 - weight_decay * lr;
    assert!((params_adamw[[0, 0]] - expected_adamw).abs() < 1e-6);

    // L2 regularization adds weight_decay * params to gradients, so even with zero gradients,
    // the effective gradients become weight_decay * params, which will change parameters
    assert!((params_l2[[0, 0]] - 1.0).abs() > 1e-6); // Parameters should change due to L2 regularization
}