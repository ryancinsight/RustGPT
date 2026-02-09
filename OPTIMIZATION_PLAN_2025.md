# RustGPT Comprehensive Optimization Plan 2025

## Executive Summary

**Current State**: 527/528 tests passing, solid architectural foundation with memory pools, streaming workspaces, and pre-allocated buffers.

**Optimization Goals**:
1. **Performance**: Zero-allocation streaming, SIMD-friendly operations, cache-conscious access patterns
2. **Memory Efficiency**: Tiered caching, workspace reuse, memory pool integration
3. **Dynamism over Parameters**: Adaptive computation, dynamic context length, runtime capacity adjustment
4. **Code Cleanliness**: SRP/SOC enforcement, deep vertical hierarchy, single source of truth
5. **Correctness**: Property-based testing, formal invariants, numerical stability

## Phase 1: Streaming & Rolling Optimizations

### 1.1 PolyAttention Streaming Enhancements
**Status**: [IN PROGRESS]

Current Implementation:
- `forward_step_into` with TLS workspace pool
- Pre-allocated `PolyAttentionWorkspace`
- Circular buffer for KV cache

Optimizations:
- [ ] Inline hot path functions to reduce call overhead
- [ ] SIMD-vectorized polynomial activation evaluation
- [ ] Cache-line-aligned buffer layout
- [ ] Branchless scoring for common window sizes
- [ ] Prefetch hints for sequential access patterns

### 1.2 Memory Pool Integration
**Status**: [IN PROGRESS]

Current Implementation:
- `ThreadLocalPool` with power-of-2 sizing
- `BufferBucket` for categorized storage
- `MemoryPool` for thread-safe shared access

Optimizations:
- [ ] Integrate memory pool into all Layer trait implementations
- [ ] Tiered pool sizing based on access frequency
- [ ] NUMA-aware allocation for large systems
- [ ] Pool statistics and monitoring

### 1.3 Sliding Window Optimizations
**Status**: [IN PROGRESS]

Current Implementation:
- `SlidingWindowCache` with pre-sized buffers
- `valid_range()` for circular buffer access
- Titan memory state integration

Optimizations:
- [ ] Vectorized cache updates
- [ ] Block-based memory layout for better cache locality
- [ ] Adaptive window sizing based on entropy
- [ ] Streaming cache compression for long contexts

## Phase 2: Dynamism & Adaptive Computation

### 2.1 Adaptive Degree Selection
**Status**: [IN PROGRESS]

Current Implementation:
- `AdaptiveDegreeConfig` with loss/gradient metrics
- Dynamic degree adjustment based on convergence signals

Optimizations:
- [ ] Per-token degree selection
- [ ] Learned degree policy network
- [ ] Multi-scale polynomial evaluation
- [ ] Early exit for converged computations

### 2.2 Dynamic Context Length
**Status**: [PLANNED]

Research Alignment:
- StreamingLLM: Keep initial tokens + sliding window
- H2O: Heavy hitter tokens retention
- Scissorhands: Semantic compression

Implementation:
- [ ] Token importance scoring
- [ ] Hierarchical memory eviction
- [ ] Compress-and-retrieve for long contexts
- [ ] Dynamic allocation based on sequence complexity

### 2.3 Mixture-of-Heads Optimization
**Status**: [IN PROGRESS]

Current Implementation:
- Threshold predictor for head selection
- Richards curve gating
- Load balancing losses

Optimizations:
- [ ] Sparse head activation patterns
- [ ] Hardware-aware head grouping
- [ ] Gradient checkpointing for unused heads
- [ ] Dynamic expert capacity

## Phase 3: Architecture & Code Quality

### 3.1 SRP/SOC Enforcement
**Status**: [IN PROGRESS]

Current Issues:
- Forward methods mix concerns (layer ops + window adapt + cache mgmt)
- Complex type signatures in some modules
- Some modules >500 lines

Refactoring:
- [ ] Extract `WindowAdapter` trait for window management
- [ ] Extract `GradPartitioner` for gradient routing
- [ ] Separate `CacheManager` from computation logic
- [ ] Split large modules into focused submodules

### 3.2 Deep Vertical Hierarchy
**Status**: [ONGOING]

Current Structure:
```
src/domain/
  attention/
    position/
      cope.rs
      factorized_cope.rs
      gated_cope.rs
      ...
```

Enhancements:
- [ ] Organize by bounded context
- [ ] Feature-gated module compilation
- [ ] Clear dependency boundaries
- [ ] Module-level documentation

### 3.3 Single Source of Truth
**Status**: [ONGOING]

Current State:
- Constants defined in `lib.rs`
- Configuration in `models/config.rs`
- Some duplication in layer configs

Consolidation:
- [ ] Centralized configuration system
- [ ] Type-safe configuration builders
- [ ] Configuration validation at startup
- [ ] Runtime configuration updates

## Phase 4: Testing & Correctness

### 4.1 Property-Based Testing
**Status**: [IN PROGRESS]

Current Implementation:
- Proptest for Richards curves
- Some mathematical invariants tested

Expansions:
- [ ] Property tests for attention stability
- [ ] Round-trip invariants for serialization
- [ ] Gradient correctness verification
- [ ] Numerical stability bounds

### 4.2 Formal Invariants
**Status**: [IN PROGRESS]

Current Documentation:
- 4 theorems in PolyAttention
- Literature citations for stability bounds

Additions:
- [ ] Runtime invariant checking (debug builds)
- [ ] Compile-time invariant proofs where possible
- [ ] Fuzzing for edge cases
- [ ] Concolic testing for path coverage

## Phase 5: Research Integration

### 5.1 Latest Memory Research
**Status**: [PLANNED]

Research Papers:
- "Memory Transformers" (Titans follow-up)
- "Recurrent Memory" (RMT)
- "Compressive Transformers"

Implementation:
- [ ] Hierarchical memory compression
- [ ] Content-addressable retrieval
- [ ] Temporal memory decay mechanisms
- [ ] Memory attention mechanisms

### 5.2 Context Length Research
**Status**: [IN PROGRESS]

Current Implementation:
- Ring Attention (unbounded context)
- Sliding window attention
- CoPE positional encoding

Research Integration:
- [ ] LongRoPE for extended contexts
- [ ] Yarn-style length extrapolation
- [ ] Dynamic NTK-aware scaling
- [ ] Context parallel training

### 5.3 Efficient Attention Research
**Status**: [IN PROGRESS]

Current Implementation:
- Polynomial attention (sub-quadratic)
- Flash Attention-style workspace reuse

Research Integration:
- [ ] Linear attention approximations
- [ ] Local attention patterns
- [ ] Strided attention for long range
- [ ] Sparse attention patterns

## Implementation Order

### Sprint 1: Streaming Performance
1. Fix remaining test warnings
2. Inline hot path functions in PolyAttention
3. Add SIMD hints for polynomial evaluation
4. Integrate memory pool into transformer blocks
5. Run benchmarks and validate improvements

### Sprint 2: Dynamism Enhancements
1. Implement per-token degree selection
2. Add dynamic context length management
3. Optimize MoH head selection
4. Add adaptive computation budget
5. Test with varying sequence lengths

### Sprint 3: Architecture Cleanup
1. Extract WindowAdapter trait
2. Separate CacheManager
3. Split oversized modules
4. Add module-level documentation
5. Validate no regressions

### Sprint 4: Testing & Research
1. Add property-based tests
2. Implement runtime invariant checking
3. Integrate latest research findings
4. Fuzzing and edge case testing
5. Final validation and documentation

## Success Metrics

- **Performance**: <5μs per token streaming latency
- **Memory**: <2x model size peak memory
- **Tests**: >95% code coverage, 0 known bugs
- **Code Quality**: 0 clippy warnings, all modules <500 lines
- **Correctness**: All property tests pass, formal invariants validated

## Current Test Status

```
test result: 527 passed; 1 failed; 1 ignored

Failures:
  - infrastructure::persistence::mnist_loader::tests::test_mnist_header_parsing
    (gzip header issue, not critical to core)
```

## Next Actions

1. Fix MNIST test (low priority)
2. Begin Sprint 1 streaming optimizations
3. Add comprehensive benchmarks
4. Document optimization decisions
