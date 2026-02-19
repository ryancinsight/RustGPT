//! Compute Tensor Abstraction
//!
//! Provides a unified tensor type that can hold data on either CPU or GPU.
//! This enables components to work transparently across compute backends.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::gpu_memory::GpuBuffer;

/// Tensor shape metadata for GPU operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TensorShape {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
}

impl TensorShape {
    /// Create a new tensor shape
    #[inline]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// Get total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Check if shape represents empty tensor
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Get size in bytes for f32 elements
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<f32>()
    }

    /// Convert to tuple
    #[inline]
    pub fn as_tuple(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

/// A tensor that can reside on CPU or GPU
#[derive(Debug, Clone)]
pub enum ComputeTensor {
    /// CPU-resident tensor (ndarray)
    Cpu(Array2<f32>),
    /// GPU-resident tensor (buffer handle) with optional shape metadata
    Gpu {
        buffer: GpuBuffer,
        shape: Option<TensorShape>,
    },
}

impl ComputeTensor {
    /// Create a new CPU tensor
    #[inline]
    pub fn cpu(data: Array2<f32>) -> Self {
        Self::Cpu(data)
    }

    /// Create a new GPU tensor from a buffer
    #[inline]
    pub fn gpu(buffer: GpuBuffer) -> Self {
        Self::Gpu {
            buffer,
            shape: None,
        }
    }

    /// Create a new GPU tensor from a buffer with explicit shape metadata.
    #[inline]
    pub fn gpu_with_shape(buffer: GpuBuffer, shape: TensorShape) -> Self {
        Self::Gpu {
            buffer,
            shape: Some(shape),
        }
    }

    /// Check if this tensor is on CPU
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    /// Check if this tensor is on GPU
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu { .. })
    }

    /// Get the CPU data if this is a CPU tensor
    #[inline]
    pub fn as_cpu(&self) -> Option<&Array2<f32>> {
        match self {
            Self::Cpu(arr) => Some(arr),
            Self::Gpu { .. } => None,
        }
    }

    /// Get mutable CPU data if this is a CPU tensor
    #[inline]
    pub fn as_cpu_mut(&mut self) -> Option<&mut Array2<f32>> {
        match self {
            Self::Cpu(arr) => Some(arr),
            Self::Gpu { .. } => None,
        }
    }

    /// Get the GPU buffer if this is a GPU tensor
    #[inline]
    pub fn as_gpu(&self) -> Option<&GpuBuffer> {
        match self {
            Self::Cpu(_) => None,
            Self::Gpu { buffer, .. } => Some(buffer),
        }
    }

    /// Get GPU shape metadata if available.
    #[inline]
    pub fn gpu_shape(&self) -> Option<TensorShape> {
        match self {
            Self::Cpu(arr) => Some(TensorShape::new(arr.nrows(), arr.ncols())),
            Self::Gpu { shape, .. } => *shape,
        }
    }

    /// Get the shape of the tensor (rows, cols)
    pub fn shape(&self) -> Option<(usize, usize)> {
        match self {
            Self::Cpu(arr) => Some((arr.nrows(), arr.ncols())),
            Self::Gpu { shape, .. } => shape.map(|s| s.as_tuple()),
        }
    }

    /// Get the total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(arr) => arr.len(),
            Self::Gpu { buffer, shape } => shape.map_or(buffer.size_f32(), |s| s.len()),
        }
    }

    /// Check if the tensor is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A 1D tensor that can reside on CPU or GPU
#[derive(Debug, Clone)]
pub enum ComputeTensor1D {
    /// CPU-resident 1D tensor
    Cpu(Array1<f32>),
    /// GPU-resident buffer with optional logical length metadata
    Gpu {
        buffer: GpuBuffer,
        len: Option<usize>,
    },
}

impl ComputeTensor1D {
    /// Create a new CPU tensor
    #[inline]
    pub fn cpu(data: Array1<f32>) -> Self {
        Self::Cpu(data)
    }

    /// Create a new GPU tensor from a buffer
    #[inline]
    pub fn gpu(buffer: GpuBuffer) -> Self {
        Self::Gpu { buffer, len: None }
    }

    /// Create a new GPU tensor from a buffer with explicit length metadata.
    #[inline]
    pub fn gpu_with_len(buffer: GpuBuffer, len: usize) -> Self {
        Self::Gpu {
            buffer,
            len: Some(len),
        }
    }

    /// Check if this tensor is on CPU
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    /// Check if this tensor is on GPU
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu { .. })
    }

    /// Get the CPU data if this is a CPU tensor
    #[inline]
    pub fn as_cpu(&self) -> Option<&Array1<f32>> {
        match self {
            Self::Cpu(arr) => Some(arr),
            Self::Gpu { .. } => None,
        }
    }

    /// Get mutable CPU data if this is a CPU tensor
    #[inline]
    pub fn as_cpu_mut(&mut self) -> Option<&mut Array1<f32>> {
        match self {
            Self::Cpu(arr) => Some(arr),
            Self::Gpu { .. } => None,
        }
    }

    /// Get the GPU buffer if this is a GPU tensor
    #[inline]
    pub fn as_gpu(&self) -> Option<&GpuBuffer> {
        match self {
            Self::Cpu(_) => None,
            Self::Gpu { buffer, .. } => Some(buffer),
        }
    }

    /// Get GPU length metadata if available.
    #[inline]
    pub fn gpu_len(&self) -> Option<usize> {
        match self {
            Self::Cpu(arr) => Some(arr.len()),
            Self::Gpu { len, .. } => *len,
        }
    }

    /// Get the total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(arr) => arr.len(),
            Self::Gpu { buffer, len } => len.unwrap_or_else(|| buffer.size_f32()),
        }
    }

    /// Check if the tensor is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::compute::gpu_memory::GpuBuffer;
    use ndarray::Array2;

    #[test]
    fn compute_tensor_cpu_operations() {
        let arr = Array2::zeros((10, 20));
        let tensor = ComputeTensor::cpu(arr.clone());

        assert!(tensor.is_cpu());
        assert!(!tensor.is_gpu());
        assert_eq!(tensor.shape(), Some((10, 20)));
        assert_eq!(tensor.len(), 200);
        assert!(tensor.as_cpu().is_some());
        assert!(tensor.as_gpu().is_none());
    }

    #[test]
    fn tensor_shape_operations() {
        let shape = TensorShape::new(10, 20);

        assert_eq!(shape.len(), 200);
        assert_eq!(shape.size_bytes(), 200 * 4);
        assert_eq!(shape.as_tuple(), (10, 20));
        assert!(!shape.is_empty());
    }

    #[test]
    fn tensor_shape_empty() {
        let shape = TensorShape::new(0, 10);
        assert!(shape.is_empty());

        let shape = TensorShape::new(10, 0);
        assert!(shape.is_empty());
    }

    #[test]
    fn compute_tensor_gpu_shape_metadata() {
        let buffer = GpuBuffer {
            id: 7,
            size_bytes: 80 * std::mem::size_of::<f32>(),
        };
        let tensor = ComputeTensor::gpu_with_shape(buffer, TensorShape::new(8, 10));

        assert!(tensor.is_gpu());
        assert_eq!(tensor.shape(), Some((8, 10)));
        assert_eq!(tensor.len(), 80);
    }
}
