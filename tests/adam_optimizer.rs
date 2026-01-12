use approx::assert_abs_diff_eq;
use llm::Adam;
use ndarray::Array2;

#[test]
fn adam_first_step_matches_bias_corrected_sign_update() {
    // On step 1, bias-corrected Adam has m_hat=g and v_hat=g^2.
    // Update = lr * g / (sqrt(g^2)+eps) ~= lr * sign(g).
    let mut opt = Adam::new((1, 1));
    let mut params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();

    let grads_pos = Array2::from_shape_vec((1, 1), vec![2.0]).unwrap();
    opt.step(&mut params, &grads_pos, 0.01);
    assert_abs_diff_eq!(params[[0, 0]], -0.01, epsilon = 1e-6);

    // Reset and test negative gradient.
    let mut opt = Adam::new((1, 1));
    let mut params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
    let grads_neg = Array2::from_shape_vec((1, 1), vec![-3.0]).unwrap();
    opt.step(&mut params, &grads_neg, 0.01);
    assert_abs_diff_eq!(params[[0, 0]], 0.01, epsilon = 1e-6);
}

#[test]
fn adam_decoupled_weight_decay_scales_params_even_with_zero_grads() {
    let mut opt = Adam::new_adamw((1, 1), 0.1);
    let mut params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let grads = Array2::zeros((1, 1));

    // AdamW decoupled: params *= (1 - wd*lr)
    opt.step(&mut params, &grads, 0.01);
    assert_abs_diff_eq!(params[[0, 0]], 0.999, epsilon = 1e-6);
}

#[test]
fn adam_non_finite_grads_are_ignored() {
    let mut opt = Adam::new((1, 1));
    let mut params = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
    let grads = Array2::from_shape_vec((1, 1), vec![f32::NAN]).unwrap();

    opt.step(&mut params, &grads, 0.01);
    // With NaN grads treated as 0, params should be unchanged.
    assert_abs_diff_eq!(params[[0, 0]], 0.5, epsilon = 1e-6);
}
