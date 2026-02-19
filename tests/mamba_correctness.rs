use llm::domain::layers::ssm::{Mamba, MambaConfig};
use llm::domain::network::Layer;
use ndarray::Array2;

#[test]
fn test_mamba_forward_backward_shapes() {
    let d_model = 16;
    let seq_len = 10;

    // Create Mamba layer
    let mut config = MambaConfig::enhanced();
    // Force parallel scan to test our new kernel
    config.scan_config.method = llm::domain::layers::ssm::ScanMethod::Parallel;

    let mut mamba = Mamba::new_with_config(d_model, 4, config);

    // Create input
    let input = Array2::<f32>::zeros((seq_len, d_model)).mapv(|_| rand::random::<f32>());

    // Forward
    let output = mamba.forward(&input);
    assert_eq!(output.dim(), (seq_len, d_model));
    assert!(
        !output.iter().any(|x: &f32| x.is_nan()),
        "Output contains NaN"
    );

    // Backward
    let grads = Array2::<f32>::ones((seq_len, d_model));
    let input_grads = mamba.backward(&grads, 0.001);

    assert_eq!(input_grads.dim(), (seq_len, d_model));
    assert!(
        !input_grads.iter().any(|x: &f32| x.is_nan()),
        "Input grads contain NaN"
    );
}

#[test]
fn test_mamba_numerical_stability() {
    let d_model = 32;
    let seq_len = 128;

    let mut config = MambaConfig::enhanced();
    let mut mamba = Mamba::new_with_config(d_model, 4, config);

    // Create inputs with some magnitude
    let input =
        Array2::<f32>::zeros((seq_len, d_model)).mapv(|_| (rand::random::<f32>() - 0.5) * 2.0);

    let output = mamba.forward(&input);

    // Check if output exploded
    let max_val = output
        .iter()
        .fold(0.0f32, |a: f32, &b: &f32| a.max(b.abs()));
    println!("Max output value: {}", max_val);
    assert!(max_val < 1e6, "Output exploded");

    // Backward
    let grads = Array2::<f32>::zeros((seq_len, d_model)).mapv(|_| rand::random::<f32>() - 0.5);
    let input_grads = mamba.backward(&grads, 0.001);

    let max_grad = input_grads
        .iter()
        .fold(0.0f32, |a: f32, &b: &f32| a.max(b.abs()));
    println!("Max input grad: {}", max_grad);
    assert!(max_grad < 1e6, "Gradient exploded");
}
