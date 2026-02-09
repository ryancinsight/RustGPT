# ADR-019: Streaming and Rolling Buffer Optimization

## Status

Accepted

## Context

The codebase audit identified several opportunities for performance optimization:

1. **Duplicate Ring Buffer Implementations**: Multiple modules implemented their own ring buffer logic for sliding window and rolling operations
2. **Missing Zero-Allocation Hot Paths**: Several modules had `forward_step` methods that allocated on every call
3. **Duplicate Sigmoid Implementations**: 11+ implementations of `sigmoid` existed across the codebase (note: these are internal gate functions, not primary activations - the codebase uses learnable Richards activations for primary activation functions)
4. **Inconsistent Streaming Workspace Patterns**: Each module had its own workspace pattern without a common interface

## Decision

### 1. Unified Ring Buffer Abstraction (SSOT)

Created [`src/common/utils/ring_buffer.rs`](../../src/common/utils/ring_buffer.rs) with:

- **`RingBuffer1D<T>`**: Generic 1D ring buffer for rolling operations
  - O(1) push and access
  - `weighted_sum_into()` for efficient convolution operations
  - Iterator support for sequential access
  
- **`RingBuffer2D<T>`**: 2D ring buffer for matrix rolling operations
  - Row-wise circular buffering
  - Efficient sliding window access
  
- **`SlidingWindowKVCache`**: Specialized KV cache for attention mechanisms
  - Pre-allocated K/V storage
  - Causal masking support
  - O(1) token appending

### 2. Zero-Allocation Streaming Workspaces

Added `forward_step_into` implementations to:

- **[`RingAttention`](../../src/domain/attention/ring_attention.rs:519)**:
  - Pre-allocated Q/K/V projection buffers
  - Online softmax accumulator
  - Block-wise score computation without allocation

- **[`NeuralMemory`](../../src/domain/memory/titans/neural.rs:636)**:
  - Zero-copy MLP forward pass
  - In-place memory update computation
  - Reusable gradient buffers

### 3. Streaming Workspace Trait

Created [`src/common/utils/streaming.rs`](../../src/common/utils/streaming.rs) with:

- **`StreamingWorkspace`** trait for common workspace operations
- **`GenericStreamingWorkspace`** for reusable buffer management
- **`WorkspaceManager`** for coordinated multi-layer workspace allocation

## Consequences

### Positive

1. **Memory Efficiency**: Ring buffers provide O(1) memory complexity for unbounded context
2. **Zero-Allocation Hot Path**: `forward_step_into` methods eliminate per-token allocations
3. **Code Consolidation**: Single source of truth for ring buffers and streaming workspaces
4. **Maintainability**: Consistent patterns across modules
5. **Test Coverage**: 504 tests passing with new functionality tested

### Performance Characteristics

| Operation | Before | After |
|-----------|--------|-------|
| Token inference (per token) | Allocates buffers | Zero allocation |
| Rolling context | Manual index management | `RingBuffer1D` abstraction |
| KV cache append | Multiple implementations | `SlidingWindowKVCache` |

## Implementation Notes

### Borrow Checker Considerations

The `forward_step_into` implementation in `RingAttention` required careful structuring to avoid simultaneous mutable/immutable borrows. The solution:

1. Cache computation parameters before borrowing workspace
2. Inline score computation to avoid `self` borrow during workspace use
3. Use separate scopes for different borrow phases

### Migration Path

Existing code continues to work. The new abstractions are:

1. **Additive**: New methods added alongside existing ones
2. **Optional**: Modules can opt-in to zero-allocation paths
3. **Compatible**: Same mathematical results, different performance characteristics

## Related

- [ADR-015: Zero-Copy Refactoring](015-zero-copy-refactoring.md)
- [ADR-016: Self-Attention Refactoring](016-self-attention-refactoring.md)

## References

- Ring Attention: Liu et al. (2023) - arXiv:2309.01809
- Online Softmax: Milakov & Gimelshein (2018)
- Flash Attention: Dao et al. (2022)
