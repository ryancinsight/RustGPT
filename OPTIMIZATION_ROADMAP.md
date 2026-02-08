# RustGPT Performance Optimization Roadmap

## Executive Summary

This roadmap outlines incremental, in-place optimizations for the RustGPT codebase with emphasis on:
- **Dynamism** over number of parameters
- **Memory efficiency** through pre-allocation and pooling
- **Streaming/rolling performance** for token-by-token inference
- **Correctness** maintained through comprehensive testing
- **Cleanliness** via SSOT, SRP, SOC principles

## Current State (Baseline)

- **Tests**: 425 passing, 1 ignored
- **Warnings**: 2 minor (deprecated function, unused mut)
- **Architecture Grade**: A- (well-structured, mathematically rigorous)
- **Performance Grade**: B+ (good parallelization, some allocation hotspots)

## Optimization Phases

### Phase 1: Streaming Performance (Week 1)
**Focus**: Zero-allocation hot paths, workspace optimization

#### 1.1 PolyAttention Streaming Optimization
- **Target**: `forward_step_into()` in `poly_attention.rs`
- **Issue**: PERF-001 - Redundant workspace resize checks
- **Solution**: 
  - Pre-size workspaces to max expected dimensions
  - Remove resize checks from hot path
  - Add prefetching for next positions
- **Expected Gain**: 5-10% latency improvement
- **Validation**: Streaming token generation benchmark

#### 1.2 Memory Pool Implementation
- **Target**: Gradient computation temporary arrays
- **Issue**: MEM-002 - Repeated `Array2::zeros()` allocations
- **Solution**:
  - Implement `ThreadLocalBufferPool` for common shapes
  - Use object pooling for (N, D) shaped arrays
  - Replace hot-path allocations with pool requests
- **Expected Gain**: 15-20% reduction in allocation count
- **Validation**: Memory pressure test during training

### Phase 2: Parallelization & Vectorization (Week 2)
**Focus**: SIMD-friendly operations, head processing parallelization

#### 2.1 Head Processing Vectorization
- **Target**: `forward_step_into()` loop over heads
- **Issue**: PERF-002 - Sequential head processing
- **Solution**:
  - Use `ndarray::Zip` for parallel head processing
  - Add explicit SIMD paths for polynomial evaluation
  - Chunk processing for better cache locality
- **Expected Gain**: 10-15% throughput improvement
- **Validation**: Training throughput benchmark

#### 2.2 Lock-Free Cache Patterns
- **Target**: `CachedIntermediates` with `Arc<RwLock<>>`
- **Issue**: MEM-001 - Contention in multi-threaded training
- **Solution**:
  - Evaluate lock-free atomic patterns
  - Thread-local caches for inference
  - Read-heavy optimizations
- **Expected Gain**: Reduced contention in multi-threaded scenarios
- **Validation**: Concurrent training benchmark

### Phase 3: Dynamism & Adaptation (Week 3)
**Focus**: Fine-grained adaptive mechanisms, dynamic sizing

#### 3.1 Adaptive Degree Fine-Grained Control
- **Target**: `adapt_degree()` in `poly_attention.rs`
- **Issue**: DYN-001 - 2-step increments only, no fine-grained control
- **Solution**:
  - Continuous adaptation with smoothing
  - Gradient-based hints for degree adjustment
  - Per-head degree adaptation (if beneficial)
- **Expected Gain**: Faster adaptation to data complexity changes
- **Validation**: Convergence rate comparison

#### 3.2 Dynamic Segment Sizing
- **Target**: `TitansMAC` segment length
- **Issue**: DYN-002 - Fixed segment length at construction
- **Solution**:
  - Content-based dynamic segment sizing
  - Adaptive window based on sequence characteristics
  - Memory-pressure-aware segment management
- **Expected Gain**: Better handling of variable-length sequences
- **Validation**: Variable-length sequence benchmark

### Phase 4: Advanced Memory Optimizations (Week 4)
**Focus**: Ring attention, paged attention patterns, quantization prep

#### 4.1 Ring Attention for Unbounded Context
- **Target**: Long context handling
- **Research**: Ring Attention (arXiv:2309.01809)
- **Solution**:
  - Block-wise computation pattern
  - Circular buffer for KV cache
  - Memory-efficient attention for >32K context
- **Expected Gain**: Unbounded context with O(1) memory
- **Validation**: Long-context benchmark (64K+ tokens)

#### 4.2 PagedAttention-Style KV Cache
- **Target**: KV cache management
- **Research**: vLLM PagedAttention
- **Solution**:
  - Block-table based KV cache
  - Memory sharing between sequences
  - Dynamic memory allocation
- **Expected Gain**: 2-4x memory efficiency for batch inference
- **Validation**: Batch inference memory usage

## Implementation Strategy

### For Each Optimization:

1. **R&D Phase** (20% of effort)
   - Define mathematical/architectural basis
   - Design verification plan
   - Create feature branch

2. **Implementation** (50% of effort)
   - Atomic implementation with tests
   - Document invariants and assumptions
   - Add telemetry/logging

3. **Verification** (30% of effort)
   - Run full test suite (425 tests)
   - Benchmark against baseline
   - Property-based testing for edge cases

### Code Quality Requirements:

- **No unwrap() in production code** - use proper error handling
- **No panic!() calls** - return Result types
- **Documentation**: rustdoc for all public APIs with mathematical invariants
- **Testing**: Unit tests + property tests + integration tests
- **Benchmarks**: Criterion.rs for performance-critical paths

## Success Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Streaming Latency | - | 20% improvement | Tokens/sec autoregressive |
| Training Throughput | - | 15% improvement | Samples/sec |
| Memory Allocations | - | 30% reduction | Peak allocations |
| Cache Efficiency | - | >90% hit rate | Engram + SlidingWindow |
| Test Pass Rate | 425/426 | 100% | `cargo test --lib` |
| Clippy Warnings | 2 | 0 | `cargo clippy` |

## Latest Research Integration

### Context Length
- **Ring Attention**: Implement for infinite context support
- **Linear Attention**: Polynomial attention already provides O(n·d) complexity
- **Titans**: MAC architecture aligns with arXiv:2501.00663

### Memory Efficiency
- **Flash Attention**: Block-wise computation pattern applicable
- **PagedAttention**: vLLM-style KV cache management
- **Quantization**: Prep for INT8/FP16 support (future sprint)

### Training Stability
- **MuP (Maximal Update Parametrization)**: Consider for LR transfer
- **Gradients with Momentum**: Adam with AMSgrad already implemented ✓

## Testing Requirements

Each optimization must:
1. Maintain backward compatibility (serde roundtrip)
2. Preserve mathematical correctness (gradient checks)
3. Improve or maintain benchmark scores
4. Include property-based tests for edge cases
5. Pass full test suite: `cargo test --lib`

## First Sprint: Week 1 Deliverables

### Day 1-2: PolyAttention Streaming Optimization
- [ ] Implement pre-sized streaming workspaces
- [ ] Remove resize checks from hot path
- [ ] Add streaming token generation benchmark
- [ ] Run tests: `cargo test --lib`

### Day 3-4: Memory Pool Foundation
- [ ] Design `ThreadLocalBufferPool` API
- [ ] Implement pool for (N, D) shaped arrays
- [ ] Integrate into gradient computation
- [ ] Run tests: `cargo test --lib`

### Day 5: Validation & Documentation
- [ ] Benchmark streaming latency
- [ ] Measure allocation reduction
- [ ] Update rustdoc with optimization notes
- [ ] Run full test suite

## Risk Mitigation

1. **Numerical Stability**: All optimizations must preserve gradient flow
2. **Backward Compatibility**: Maintain serde serialization compatibility
3. **Test Coverage**: Any new code must have >90% test coverage
4. **Performance Regression**: Benchmark before/after each change

## Next Steps

1. Begin Phase 1.1: PolyAttention streaming optimization
2. Create feature branch for incremental changes
3. Run baseline benchmarks for comparison
4. Implement first optimization with full test coverage
