#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::{Array2, ArrayBase, Data, Ix2};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::{
    common::errors::{ModelError, Result},
    domain::{compute::GpuDevice, compute_backend::ComputeBackend},
};

/// Resolve a shared GPU device for a component backend update.
///
/// Reuses a pre-attached device when its backend matches, otherwise creates a
/// new device for the requested backend.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn resolve_or_create_gpu_device(
    existing_device: Option<Arc<Mutex<GpuDevice>>>,
    compute_backend: ComputeBackend,
    component_name: &str,
) -> Result<(Arc<Mutex<GpuDevice>>, bool)> {
    if let Some(device) = existing_device {
        let backend_matches = device
            .lock()
            .map_err(|_| ModelError::Backend {
                message: format!(
                    "Failed to lock pre-attached {} GPU device for backend validation",
                    component_name
                ),
            })?
            .backend()
            == compute_backend;

        if backend_matches {
            return Ok((device, true));
        }
    }

    Ok((
        Arc::new(Mutex::new(GpuDevice::new(compute_backend)?)),
        false,
    ))
}

/// Execute a single GEMM using an attached GPU device.
///
/// This consolidates common upload/GEMM/download/deallocate plumbing used by
/// shared components and MoE/router paths.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn gpu_gemm_with_attached_device<S1, S2>(
    device_arc: &Arc<Mutex<GpuDevice>>,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    op_name: &str,
) -> Result<Array2<f32>>
where
    S1: Data<Elem = f32>,
    S2: Data<Elem = f32>,
{
    if m == 0 || n == 0 || k == 0 {
        return Ok(Array2::<f32>::zeros((m, n)));
    }

    let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
        message: format!("Failed to lock GPU device for {op_name}"),
    })?;
    let (pool, ops) = device.execution_context();

    let a_layout = a.as_standard_layout();
    let a_slice = a_layout
        .as_slice()
        .ok_or_else(|| ModelError::InvalidInput {
            message: format!("{op_name}: lhs matrix must be contiguous"),
        })?;

    let b_layout = b.as_standard_layout();
    let b_slice = b_layout
        .as_slice()
        .ok_or_else(|| ModelError::InvalidInput {
            message: format!("{op_name}: rhs matrix must be contiguous"),
        })?;

    let a_buf = pool.upload(a_slice)?;
    let b_buf = pool.upload(b_slice)?;
    let mut c_buf = pool.allocate(m * n * 4)?;

    ops.gemm_f32(
        pool, 1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k, trans_a, trans_b,
    )?;

    let mut out = Array2::<f32>::zeros((m, n));
    let out_slice = out.as_slice_mut().ok_or_else(|| ModelError::InvalidInput {
        message: format!("{op_name}: output matrix must be contiguous"),
    })?;
    pool.download(&c_buf, out_slice)?;

    pool.deallocate(a_buf);
    pool.deallocate(b_buf);
    pool.deallocate(c_buf);

    Ok(out)
}
