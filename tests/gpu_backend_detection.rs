//! Tests for GPU backend detection and initialization
//!
//! Validates automatic backend detection with GPU-preferred policy.

use llm::domain::compute_backend::{
    ComputeBackendPreference, detect_available_gpu_backends, resolve_compute_backend,
};

#[test]
fn cpu_backend_always_available() {
    let resolved = resolve_compute_backend(ComputeBackendPreference::Cpu);
    assert!(resolved.is_ok());
    let backend = resolved.unwrap();
    assert_eq!(backend.as_str(), "cpu");
    assert!(!backend.is_gpu());
}

#[test]
fn auto_gpu_detects_available() {
    let backends = detect_available_gpu_backends();
    println!("Detected GPU backends: {:?}", backends);
    // No assertion - this depends on hardware
}

#[test]
fn parse_cuda_preference() {
    let resolved = resolve_compute_backend(ComputeBackendPreference::Cuda);
    // May pass or fail depending on CUDA availability
    if let Ok(backend) = resolved {
        assert!(backend.is_gpu());
        assert_eq!(backend.as_str(), "cuda");
    } else {
        println!("CUDA not available on this system");
    }
}

#[test]
fn parse_metal_preference() {
    let resolved = resolve_compute_backend(ComputeBackendPreference::Metal);
    // May pass or fail depending on platform and Metal availability
    if let Ok(backend) = resolved {
        assert!(backend.is_gpu());
        assert_eq!(backend.as_str(), "metal");
    } else {
        println!("Metal not available on this system");
    }
}

#[test]
#[cfg(feature = "gpu-cuda")]
fn cuda_memory_pool_creation() {
    use llm::domain::compute::CudaMemoryPool;

    let pool = CudaMemoryPool::new(0);
    assert!(pool.is_ok(), "Failed to create CUDA memory pool");

    let mut pool = pool.unwrap();
    let buf = pool.allocate(1024);
    assert!(buf.is_ok(), "Failed to allocate GPU buffer");

    let buf = buf.unwrap();
    assert_eq!(buf.size_bytes(), 1024);
    assert_eq!(buf.size_f32(), 256); // 1024 bytes / 4 bytes per f32

    let stats = pool.memory_stats();
    assert_eq!(stats.allocation_count, 1);
    assert_eq!(stats.used_bytes, 1024);
}

#[test]
#[cfg(feature = "gpu-cuda")]
fn cuda_multiple_allocations() {
    use llm::domain::compute::CudaMemoryPool;

    let mut pool = CudaMemoryPool::new(0).expect("Failed to create pool");

    let buf1 = pool.allocate(512).expect("Failed to allocate buf1");
    let buf2 = pool.allocate(1024).expect("Failed to allocate buf2");
    let buf3 = pool.allocate(2048).expect("Failed to allocate buf3");

    let stats = pool.memory_stats();
    assert_eq!(stats.allocation_count, 3);
    assert_eq!(stats.used_bytes, 512 + 1024 + 2048);

    pool.deallocate(buf2);
    let stats = pool.memory_stats();
    assert_eq!(stats.allocation_count, 2);
    assert_eq!(stats.used_bytes, 512 + 2048);

    pool.deallocate(buf1);
    pool.deallocate(buf3);
    let stats = pool.memory_stats();
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.used_bytes, 0);
}

#[test]
#[cfg(feature = "gpu-cuda")]
fn cuda_ops_creation() {
    use llm::domain::compute::CudaMatrixOps;

    let ops = CudaMatrixOps::new(0);
    // Just verify instantiation succeeds
    drop(ops);
}

#[test]
#[cfg(feature = "gpu-metal")]
fn metal_memory_pool_creation() {
    use llm::domain::compute::MetalMemoryPool;

    let pool = MetalMemoryPool::new();
    assert!(pool.is_ok(), "Failed to create Metal memory pool");

    let mut pool = pool.unwrap();
    let buf = pool.allocate(2048);
    assert!(buf.is_ok(), "Failed to allocate GPU buffer");

    let buf = buf.unwrap();
    assert_eq!(buf.size_bytes(), 2048);
    assert_eq!(buf.size_f32(), 512); // 2048 bytes / 4 bytes per f32
}

#[test]
#[cfg(feature = "gpu-metal")]
fn metal_ops_creation() {
    use llm::domain::compute::MetalMatrixOps;

    let ops = MetalMatrixOps::new();
    // Just verify instantiation succeeds
    drop(ops);
}
