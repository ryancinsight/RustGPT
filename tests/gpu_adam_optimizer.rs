//! Integration tests for GPU Adam Optimizer
//!
//! These tests verify that the GPU Adam optimizer produces numerically
//! correct results compared to the CPU Adam implementation.

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
mod tests {
    use ndarray::Array2;
    use llm::{
        domain::compute::GpuDevice,
        domain::compute_backend::ComputeBackend,
        infrastructure::optimizer::adam::Adam,
        infrastructure::optimizer::gpu_adam::{GpuAdam, GpuAdamConfig},
    };
    use std::sync::{Arc, Mutex};

    /// Helper to create a GPU device (skips test if unavailable)
    fn create_gpu_device() -> Option<Arc<Mutex<GpuDevice>>> {
        let device = GpuDevice::new(ComputeBackend::Vulkan).ok()?;
        Some(Arc::new(Mutex::new(device)))
    }

    /// Test basic GPU Adam creation
    #[test]
    fn test_gpu_adam_creation() {
        let Some(device) = create_gpu_device() else {
            eprintln!("Skipping test: GPU device not available");
            return;
        };

        let param_count = 100;
        let optimizer = GpuAdam::new(device, param_count);

        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert_eq!(opt.param_count(), param_count);
        assert_eq!(opt.timestep(), 0);
        assert!(!opt.is_amsgrad());
    }

    /// Test GPU Adam with AMSGrad
    #[test]
    fn test_gpu_adam_amsgrad() {
        let Some(device) = create_gpu_device() else {
            eprintln!("Skipping test: GPU device not available");
            return;
        };

        let param_count = 100;
        let optimizer = GpuAdam::new_amsgrad(device, param_count);

        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert!(opt.is_amsgrad());
        assert!(opt.v_max_buffer().is_some());
    }

    /// Test GPU AdamW creation
    #[test]
    fn test_gpu_adamw_creation() {
        let Some(device) = create_gpu_device() else {
            eprintln!("Skipping test: GPU device not available");
            return;
        };

        let param_count = 100;
        let weight_decay = 0.01;
        let optimizer = GpuAdam::new_adamw(device, param_count, weight_decay);

        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert!(opt.is_decoupled_wd());
        assert!((opt.weight_decay() - weight_decay).abs() < 1e-6);
    }

    /// Test GPU Adam reset
    #[test]
    fn test_gpu_adam_reset() {
        let Some(device) = create_gpu_device() else {
            eprintln!("Skipping test: GPU device not available");
            return;
        };

        let param_count = 100;
        let mut optimizer = GpuAdam::new(device, param_count).unwrap();

        // Initial timestep should be 0
        assert_eq!(optimizer.timestep(), 0);

        // Reset should work
        let result = optimizer.reset();
        assert!(result.is_ok());
        assert_eq!(optimizer.timestep(), 0);
    }

    /// Test GPU Adam vs CPU Adam numerical equivalence
    ///
    /// This test verifies that the GPU Adam produces the same results
    /// as the CPU Adam implementation within numerical tolerance.
    #[test]
    fn test_gpu_adam_numerical_equivalence() {
        let Some(device) = create_gpu_device() else {
            eprintln!("Skipping test: GPU device not available");
            return;
        };

        let param_count = 64;
        let lr = 0.001;

        // Create CPU Adam
        let mut cpu_adam = Adam::new((8, 8));

        // Create GPU Adam
        let mut gpu_adam = GpuAdam::new(device.clone(), param_count).unwrap();

        // Initialize parameters
        let mut cpu_params = Array2::from_shape_fn((8, 8), |(i, j)| {
            (i as f32 * 8.0 + j as f32) / 64.0 - 0.5
        });

        // Create gradients
        let grads = Array2::from_shape_fn((8, 8), |(i, j)| {
            0.1 * ((i as f32 + j as f32) / 16.0 - 0.5)
        });

        // CPU step
        cpu_adam.step(&mut cpu_params, &grads, lr);

        // GPU step
        {
            let mut gpu = device.lock().unwrap();

            // Upload parameters
            let mut params_buffer = gpu.allocate_f32(param_count).unwrap();
            let params_slice: Vec<f32> = (0..param_count)
                .map(|i| {
                    let row = i / 8;
                    let col = i % 8;
                    (row as f32 * 8.0 + col as f32) / 64.0 - 0.5
                })
                .collect();
            gpu.upload(&params_slice, &mut params_buffer).unwrap();

            // Upload gradients
            let mut grads_buffer = gpu.allocate_f32(param_count).unwrap();
            let grads_slice: Vec<f32> = grads.iter().cloned().collect();
            gpu.upload(&grads_slice, &mut grads_buffer).unwrap();

            // Perform GPU Adam step
            gpu_adam.step(&mut params_buffer, &grads_buffer, lr).unwrap();

            // Download updated parameters
            let mut result_params = vec![0.0f32; param_count];
            gpu.download(&params_buffer, &mut result_params).unwrap();

            // Compare with CPU results
            let cpu_params_flat: Vec<f32> = cpu_params.iter().cloned().collect();

            for i in 0..param_count {
                let diff = (result_params[i] - cpu_params_flat[i]).abs();
                // Allow for floating point differences between GPU and CPU
                assert!(
                    diff < 1e-4,
                    "Parameter {} differs: GPU={}, CPU={}, diff={}",
                    i,
                    result_params[i],
                    cpu_params_flat[i],
                    diff
                );
            }
        }
    }

    /// Test GPU Adam multiple steps
    #[test]
    fn test_gpu_adam_multiple_steps() {
        let Some(device) = create_gpu_device() else {
            eprintln!("Skipping test: GPU device not available");
            return;
        };

        let param_count = 32;
        let lr = 0.01;

        // Create GPU Adam
        let mut gpu_adam = GpuAdam::new(device.clone(), param_count).unwrap();

        {
            let mut gpu = device.lock().unwrap();

            // Initialize parameters
            let mut params_buffer = gpu.allocate_f32(param_count).unwrap();
            let initial_params: Vec<f32> = (0..param_count).map(|i| i as f32 * 0.1).collect();
            gpu.upload(&initial_params, &mut params_buffer).unwrap();

            // Perform multiple optimization steps
            for step in 0..5 {
                // Create gradients (simple gradient descent direction)
                let grads: Vec<f32> = (0..param_count).map(|i| (i as f32 + step as f32) * 0.01).collect();
                let mut grads_buffer = gpu.allocate_f32(param_count).unwrap();
                gpu.upload(&grads, &mut grads_buffer).unwrap();

                gpu_adam.step(&mut params_buffer, &grads_buffer, lr).unwrap();

                gpu.deallocate(grads_buffer);
            }

            // Download and verify parameters changed
            let mut final_params = vec![0.0f32; param_count];
            gpu.download(&params_buffer, &mut final_params).unwrap();

            // Parameters should have changed from initial values
            let mut changed = false;
            for i in 0..param_count {
                if (final_params[i] - initial_params[i]).abs() > 1e-6 {
                    changed = true;
                    break;
                }
            }
            assert!(changed, "Parameters should have changed after optimization steps");

            // Timestep should be 5
            assert_eq!(gpu_adam.timestep(), 5);
        }
    }

    /// Test GPU Adam with zero learning rate
    #[test]
    fn test_gpu_adam_zero_lr() {
        let Some(device) = create_gpu_device() else {
            eprintln!("Skipping test: GPU device not available");
            return;
        };

        let param_count = 16;
        let lr = 0.0;

        let mut gpu_adam = GpuAdam::new(device.clone(), param_count).unwrap();

        {
            let mut gpu = device.lock().unwrap();

            // Initialize parameters
            let mut params_buffer = gpu.allocate_f32(param_count).unwrap();
            let initial_params: Vec<f32> = (0..param_count).map(|i| i as f32 * 0.1).collect();
            gpu.upload(&initial_params, &mut params_buffer).unwrap();

            // Create gradients
            let grads: Vec<f32> = vec![0.1; param_count];
            let mut grads_buffer = gpu.allocate_f32(param_count).unwrap();
            gpu.upload(&grads, &mut grads_buffer).unwrap();

            // Step with zero LR should not change parameters
            gpu_adam.step(&mut params_buffer, &grads_buffer, lr).unwrap();

            // Download and verify parameters unchanged
            let mut final_params = vec![0.0f32; param_count];
            gpu.download(&params_buffer, &mut final_params).unwrap();

            for i in 0..param_count {
                assert!(
                    (final_params[i] - initial_params[i]).abs() < 1e-6,
                    "Parameters should not change with zero LR"
                );
            }

            // Timestep should still be incremented
            assert_eq!(gpu_adam.timestep(), 1);
        }
    }

    /// Test GPU Adam weight decay (AdamW)
    #[test]
    fn test_gpu_adam_weight_decay() {
        let Some(device) = create_gpu_device() else {
            eprintln!("Skipping test: GPU device not available");
            return;
        };

        let param_count = 16;
        let lr = 0.01;
        let weight_decay = 0.1;

        // Create AdamW optimizer
        let config = GpuAdamConfig::adamw(weight_decay);
        let mut gpu_adam = GpuAdam::with_config(device.clone(), param_count, config).unwrap();

        {
            let mut gpu = device.lock().unwrap();

            // Initialize parameters with positive values
            let mut params_buffer = gpu.allocate_f32(param_count).unwrap();
            let initial_params: Vec<f32> = vec![1.0; param_count];
            gpu.upload(&initial_params, &mut params_buffer).unwrap();

            // Zero gradients - only weight decay should apply
            let grads: Vec<f32> = vec![0.0; param_count];
            let mut grads_buffer = gpu.allocate_f32(param_count).unwrap();
            gpu.upload(&grads, &mut grads_buffer).unwrap();

            gpu_adam.step(&mut params_buffer, &grads_buffer, lr).unwrap();

            // Download parameters
            let mut final_params = vec![0.0f32; param_count];
            gpu.download(&params_buffer, &mut final_params).unwrap();

            // With AdamW, parameters should decrease due to weight decay
            // θ = θ * (1 - lr * weight_decay) = 1.0 * (1 - 0.01 * 0.1) ≈ 0.999
            for i in 0..param_count {
                assert!(
                    final_params[i] < initial_params[i],
                    "Weight decay should reduce parameters: {} vs {}",
                    final_params[i],
                    initial_params[i]
                );
            }
        }
    }
}
