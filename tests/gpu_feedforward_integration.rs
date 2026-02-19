#[cfg(feature = "wgpu")]
mod tests {
    use llm::domain::{
        compute::gpu_device::GpuDevice,
        layers::components::{common::FeedForwardVariant, feedforward::SharedFeedforward},
        richards::RichardsGlu,
    };
    use ndarray::Array2;
    use rand::Rng;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn test_shared_feedforward_gpu_integration() {
        // 1. Initialize RichardsGlu and wrap in SharedFeedforward
        let embedding_dim = 64;
        let hidden_dim = 256;
        let richards_glu = RichardsGlu::new(embedding_dim, hidden_dim);
        let mut layer =
            SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(richards_glu)));

        // 2. Create random input
        let batch_size = 8;
        let mut rng = rand::rng();
        let input_data: Vec<f32> = (0..batch_size * embedding_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let input = Array2::from_shape_vec((batch_size, embedding_dim), input_data).unwrap();

        // 3. Run CPU forward pass
        println!("Running CPU forward pass...");
        let output_cpu = layer.forward(&input);

        // 4. Initialize GPU Device
        println!("Initializing GPU device...");
        // WGPU maps to ComputeBackend::Vulkan in this codebase currently
        let device = match GpuDevice::new(llm::domain::compute_backend::ComputeBackend::Vulkan) {
            Ok(d) => Arc::new(Mutex::new(d)),
            Err(e) => {
                println!("Skipping GPU test: {}", e);
                return;
            }
        };

        // 5. Configure SharedFeedforward for GPU
        layer.set_compute_backend(llm::domain::compute_backend::ComputeBackend::Vulkan);
        layer.set_gpu_device(device.clone());

        // 6. Run GPU forward pass
        println!("Running GPU forward pass...");
        let output_gpu = layer.forward(&input);

        // 7. Compare results
        let epsilon = 1e-4;

        let cpu_slice = output_cpu.as_slice().unwrap();
        let gpu_slice = output_gpu.as_slice().unwrap();

        assert_eq!(cpu_slice.len(), gpu_slice.len(), "Output lengths mismatch");

        let mut max_diff = 0.0;
        for (i, (c, g)) in cpu_slice.iter().zip(gpu_slice.iter()).enumerate() {
            let diff = (*c - *g).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > epsilon {
                panic!(
                    "Mismatch at index {}: CPU={}, GPU={}, diff={} (max allowed {})",
                    i, c, g, diff, epsilon
                );
            }
        }

        println!("Max difference between CPU and GPU: {}", max_diff);
        println!("Integration test passed!");
    }
}
