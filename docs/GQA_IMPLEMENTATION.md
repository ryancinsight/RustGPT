# Group-Query Attention (GQA) Implementation - Phase 2

## Overview

Group-Query Attention (GQA) has been successfully implemented for the TransformerBlock architecture. This completes **Phase 2** of the Transformer Modernization initiative, adding the attention optimization used in LLaMA 2 70B and Mistral 7B.

## What is Group-Query Attention (GQA)?

GQA is an attention mechanism that reduces KV cache size while maintaining model quality by sharing key/value heads across multiple query heads.

### Attention Mechanism Comparison

**Multi-Head Attention (MHA)** - Standard Transformer:
```
num_heads query heads → num_heads key heads → num_heads value heads
Example: 8 Q heads → 8 K heads → 8 V heads
```

**Group-Query Attention (GQA)** - Modern LLMs:
```
num_heads query heads → num_kv_heads key heads → num_kv_heads value heads
Example: 8 Q heads → 4 K heads → 4 V heads (2 queries per KV head)
```

**Multi-Query Attention (MQA)** - Extreme case:
```
num_heads query heads → 1 key head → 1 value head
Example: 8 Q heads → 1 K head → 1 V head (all queries share 1 KV head)
```

### Mathematical Formulation

For GQA with `num_heads` query heads and `num_kv_heads` key-value heads:

1. **Group size**: `g = num_heads / num_kv_heads`
2. **Query head i uses KV head**: `⌈i / g⌉`
3. **Attention computation**:
   ```
   For query head i:
     kv_idx = i / g
     Attention(Q_i, K_kv_idx, V_kv_idx) = softmax(Q_i K_kv_idx^T / √d_k) V_kv_idx
   ```

## Implementation Details

### Files Modified

1. **`src/model_config.rs`**
   - Added `num_kv_heads: Option<usize>` field
   - Added `get_num_kv_heads()` helper method
   - Updated all constructor methods

2. **`src/self_attention.rs`**
   - Added `num_kv_heads` field to `SelfAttention` struct
   - Created `new_with_gqa()` constructor
   - Modified `forward()` to implement GQA grouping
   - Updated `parameters()` to account for reduced KV parameters
   - Maintained backward compatibility with MHA

3. **`src/model_builder.rs`**
   - Updated `build_transformer_layers()` to use `new_with_gqa()`
   - Passes `config.num_kv_heads` to attention layers

4. **`src/main.rs`**
   - Added GQA configuration section with documentation
   - Added `num_kv_heads` configuration variable
   - Applied configuration: `config.num_kv_heads = num_kv_heads`

5. **`tests/gqa_test.rs`** (NEW)
   - 16 comprehensive tests for GQA functionality
   - Tests for MHA, GQA, and MQA configurations
   - Parameter reduction verification
   - RoPE integration tests
   - Training stability tests

### Key Implementation Features

**Backward Compatibility**:
- `num_kv_heads = None` → defaults to `num_heads` (standard MHA)
- Existing code continues to work without changes

**GQA Forward Pass**:
1. Compute Q for all `num_heads` query heads
2. Compute K, V for only `num_kv_heads` key-value heads
3. Group query heads to share KV heads
4. Compute attention for each query head with its corresponding KV head

**Parameter Efficiency**:
- MHA: `num_heads × (d_k × d_k) × 3` parameters (Q, K, V)
- GQA: `num_heads × (d_k × d_k) + num_kv_heads × (d_k × d_k) × 2` parameters

## Configuration

### In `src/main.rs`

```rust
// ============================================================================
// GROUP-QUERY ATTENTION (GQA) CONFIGURATION
// ============================================================================

let num_kv_heads: Option<usize> = None; // None for MHA, Some(4) for GQA, Some(1) for MQA

// Apply configuration
config.num_kv_heads = num_kv_heads;
```

### Configuration Options

| Configuration | num_kv_heads | Description | KV Cache Reduction |
|--------------|--------------|-------------|-------------------|
| **MHA** | `None` or `Some(8)` | Standard Multi-Head Attention | 1x (baseline) |
| **GQA (4 heads)** | `Some(4)` | 2 queries per KV head | 2x reduction |
| **GQA (2 heads)** | `Some(2)` | 4 queries per KV head | 4x reduction |
| **MQA** | `Some(1)` | All queries share 1 KV head | 8x reduction |

## Parameter Reduction

### Example: 8 Query Heads, 128 Embedding Dimension

**Head dimension**: `d_k = 128 / 8 = 16`

**MHA (8 KV heads)**:
- Q parameters: `8 × 16 × 16 = 2,048`
- K parameters: `8 × 16 × 16 = 2,048`
- V parameters: `8 × 16 × 16 = 2,048`
- **Total**: 6,144 parameters

**GQA (4 KV heads)**:
- Q parameters: `8 × 16 × 16 = 2,048`
- K parameters: `4 × 16 × 16 = 1,024`
- V parameters: `4 × 16 × 16 = 1,024`
- **Total**: 4,096 parameters
- **Reduction**: 2,048 parameters (33% reduction)

**MQA (1 KV head)**:
- Q parameters: `8 × 16 × 16 = 2,048`
- K parameters: `1 × 16 × 16 = 256`
- V parameters: `1 × 16 × 16 = 256`
- **Total**: 2,560 parameters
- **Reduction**: 3,584 parameters (58% reduction)

## Expected Benefits

### 1. Reduced KV Cache Size

During autoregressive generation, the KV cache stores previously computed keys and values:
- **MHA**: Cache size = `num_heads × seq_len × d_k`
- **GQA**: Cache size = `num_kv_heads × seq_len × d_k`
- **Reduction**: `(num_heads - num_kv_heads) / num_heads`

Example with 8 query heads:
- **GQA (4 KV heads)**: 50% cache reduction
- **GQA (2 KV heads)**: 75% cache reduction
- **MQA (1 KV head)**: 87.5% cache reduction

### 2. Faster Inference

- **Smaller memory bandwidth**: Less data to load from memory
- **Faster decoding**: Reduced memory access bottleneck
- **Lower latency**: Especially beneficial for long sequences

### 3. Lower Memory Usage

- **Training**: Reduced memory footprint for KV parameters
- **Inference**: Smaller KV cache during generation
- **Deployment**: Enables larger models on limited hardware

### 4. Minimal Quality Degradation

Empirical results from LLaMA 2 and Mistral show:
- **GQA (4-8 KV heads)**: Minimal quality loss vs MHA
- **MQA (1 KV head)**: Some quality degradation but still effective

## Testing

### Test Results

```
✅ All 161 tests passing (16 new GQA tests)
✅ Zero clippy warnings
✅ Backward compatibility maintained
✅ RoPE integration verified
```

### GQA Tests (16 tests)

1. **`test_gqa_creation`** - Verify GQA initialization
2. **`test_mha_backward_compatibility`** - Ensure MHA still works
3. **`test_mqa_extreme_case`** - Test MQA with 1 KV head
4. **`test_gqa_invalid_grouping`** - Verify error handling
5. **`test_gqa_invalid_kv_heads`** - Verify validation
6. **`test_gqa_parameter_reduction`** - Verify parameter counts
7. **`test_gqa_forward_pass`** - Test forward computation
8. **`test_gqa_with_rope`** - Test RoPE integration
9. **`test_gqa_backward_pass`** - Test gradient computation
10. **`test_gqa_different_sequence_lengths`** - Test various lengths
11. **`test_gqa_vs_mha_output_similarity`** - Compare outputs
12. **`test_gqa_kv_cache_size_reduction`** - Verify cache reduction
13. **`test_gqa_training_stability`** - Test training stability
14. **`test_gqa_with_rope_integration`** - Test RoPE + GQA
15. **`test_gqa_grouping_correctness`** - Verify grouping logic
16. **`test_gqa_parameter_count_consistency`** - Verify parameter counts

## Industry Alignment

GQA is used in modern LLMs:

| Model | Architecture | num_heads | num_kv_heads | Reduction |
|-------|-------------|-----------|--------------|-----------|
| **LLaMA 2 7B** | GQA | 32 | 32 | 1x (MHA) |
| **LLaMA 2 70B** | GQA | 64 | 8 | 8x |
| **Mistral 7B** | GQA | 32 | 8 | 4x |
| **RustGPT** | GQA | 8 | 4 (configurable) | 2x |

## Usage Example

```rust
use llm::self_attention::SelfAttention;

// Standard MHA (8 query heads, 8 KV heads)
let mha = SelfAttention::new_with_gqa(128, 8, 8, false, 512);

// GQA with 4 KV heads (2x cache reduction)
let gqa = SelfAttention::new_with_gqa(128, 8, 4, false, 512);

// MQA with 1 KV head (8x cache reduction)
let mqa = SelfAttention::new_with_gqa(128, 8, 1, false, 512);

// GQA with RoPE
let gqa_rope = SelfAttention::new_with_gqa(128, 8, 4, true, 512);
```

## References

1. **GQA**: Ainslie et al. (2023), "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", arXiv:2305.13245
2. **MQA**: Shazeer (2019), "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150
3. **LLaMA 2**: Touvron et al. (2023), "Llama 2: Open Foundation and Fine-Tuned Chat Models", arXiv:2307.09288
4. **Mistral**: Jiang et al. (2023), "Mistral 7B", arXiv:2310.06825

## Summary

GQA implementation is **complete and tested**. The implementation:
- ✅ Supports MHA, GQA, and MQA configurations
- ✅ Reduces KV cache size (2x-8x reduction)
- ✅ Maintains backward compatibility
- ✅ Integrates with RoPE
- ✅ Passes all 161 tests with zero warnings
- ✅ Matches industry-standard implementations

**Phase 2 is now complete! You have the attention optimization used in LLaMA 2 70B and Mistral 7B.**

Execute `cargo run --release` with `num_kv_heads = Some(4)` to start training with GQA.

