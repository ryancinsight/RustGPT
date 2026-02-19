//! Attention Module
//!
//! This module provides attention mechanisms for transformer architectures:
//!
//! - **PolyAttention**: Polynomial attention with adaptive degree and MoH gating
//! - **RingAttention**: O(1) memory attention for unbounded context
//! - **PagedAttention**: Memory-efficient KV cache with OS-like paging (vLLM)
//! - **SlidingWindowAttention**: Fixed-size sliding window attention
//! - **Streaming Optimized**: High-performance streaming operations
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Attention Module                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │  config.rs        - Weight initialization and configuration  │
//! │  forward.rs       - Batch forward pass implementations      │
//! │  cache.rs         - KV cache with ring buffer pattern       │
//! │  memory.rs        - Memory-efficient attention utilities    │
//! │  params.rs        - Parameter information and tracking      │
//! │  poly_attention.rs - Main polynomial attention layer        │
//! │  position/        - Position encoding variants (CoPE)       │
//! │  ring_attention.rs - Unbounded context attention            │
//! │  paged_attention.rs - Memory-efficient KV cache (vLLM)      │
//! │  sliding_window.rs - Fixed window attention                 │
//! │  streaming_optimized.rs - Hot path optimizations            │
//! │  utils.rs         - Shared utilities                       │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Research Alignment
//!
//! - **Polynomial Attention**: Bolya et al. (2023) - Hyena Hierarchy
//! - **Ring Attention**: Liu et al. (2023) - arXiv:2309.01809
//! - **Flash Attention**: Dao et al. (2022) - Memory-efficient attention
//! - **vLLM PagedAttention**: Kwon et al. (2023) - KV cache management
//!

pub mod cache;
pub mod config;
pub mod forward;
pub mod head;
pub mod memory;
pub mod paged_attention;
pub mod params;
pub mod poly_attention;
pub mod poly_attention_gpu;
pub mod position;
pub mod ring_attention;
pub mod sliding_window_attention;
pub mod streaming_optimized;
pub mod utils;

// Re-export commonly used types for convenience
pub use cache::HeadCache;
pub use paged_attention::{PagedKVCache, PagedKVCacheConfig, PagedKVCacheStats};
pub use poly_attention::{PolyAttention, PolyAttentionStreamingWorkspace};
pub use ring_attention::{RingAttention, RingAttentionConfig};
pub use sliding_window_attention::{SlidingWindowAttention, SlidingWindowCache};
