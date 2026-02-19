//! Metal Backend Implementation (macOS)
//!
//! Provides Metal-specific memory management and matrix operations for Apple GPUs.

pub mod memory;
pub mod ops;

pub use memory::MetalMemoryPool;
pub use ops::MetalMatrixOps;
