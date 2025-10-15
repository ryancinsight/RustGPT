# Parameter Count Analysis: Transformer vs HyperMixer

## Configuration
- `embedding_dim` = 128
- `hidden_dim` = 256
- `num_layers` = 3
- `max_seq_len` = 80
- `vocab_size` = 533
- `hypernetwork_hidden_dim` = 32

## Transformer Architecture

### Per Layer (TransformerBlock):
1. **SelfAttention**:
   - `w_q`: (128, 128) = 16,384
   - `w_k`: (128, 128) = 16,384
   - `w_v`: (128, 128) = 16,384
   - **Total**: 49,152 parameters

2. **FeedForward**:
   - `w1`: (128, 256) = 32,768
   - `b1`: (1, 256) = 256
   - `w2`: (256, 128) = 32,768
   - `b2`: (1, 128) = 128
   - **Total**: 65,920 parameters

3. **LayerNorm** (×2):
   - `gamma`: (1, 128) = 128
   - `beta`: (1, 128) = 128
   - **Total per LayerNorm**: 256 parameters
   - **Total for 2 LayerNorms**: 512 parameters

**Per TransformerBlock**: 49,152 + 65,920 + 512 = **115,584 parameters**

### Shared Components:
1. **Embeddings**:
   - Token embeddings: (533, 128) = 68,224
   - Positional embeddings: (80, 128) = 10,240
   - **Total**: 78,464 parameters

2. **OutputProjection**:
   - `w_out`: (128, 533) = 68,224
   - `b_out`: (1, 533) = 533
   - **Total**: 68,757 parameters

### Total Transformer Parameters:
- Embeddings: 78,464
- 3 × TransformerBlock: 3 × 115,584 = 346,752
- OutputProjection: 68,757
- **TOTAL**: 78,464 + 346,752 + 68,757 = **493,973 parameters** ✓

---

## HyperMixer Architecture

### Per Layer (HyperMixerBlock):

1. **TokenMixingMLP**:
   - Contains a **Hypernetwork** that generates weights dynamically
   - Hypernetwork parameters:
     - `w1`: (128, 32) = 4,096
     - `b1`: (1, 32) = 32
     - `w2`: (32, output_size) where output_size = ?
     - `b2`: (1, output_size) = output_size
   
   **Calculating output_size**:
   - The hypernetwork generates weights for a token-mixing MLP
   - Token mixing MLP structure:
     - `w1`: (max_seq_len, token_mixing_hidden_dim) = (80, 64) = 5,120
     - `b1`: (1, 64) = 64
     - `w2`: (64, max_seq_len) = (64, 80) = 5,120
     - `b2`: (1, 80) = 80
   - **output_size** = 5,120 + 64 + 5,120 + 80 = **10,384**
   
   **Hypernetwork parameters**:
   - `w1`: (128, 32) = 4,096
   - `b1`: (1, 32) = 32
   - `w2`: (32, 10,384) = **332,288** ⚠️ **HUGE!**
   - `b2`: (1, 10,384) = 10,384
   - **Total**: 4,096 + 32 + 332,288 + 10,384 = **346,800 parameters**

2. **ChannelMixingMLP**:
   - `w1`: (128, 256) = 32,768
   - `b1`: (1, 256) = 256
   - `w2`: (256, 128) = 32,768
   - `b2`: (1, 128) = 128
   - **Total**: 65,920 parameters

3. **LayerNorm** (×2):
   - Same as Transformer: 512 parameters

**Per HyperMixerBlock**: 346,800 + 65,920 + 512 = **413,232 parameters**

### Shared Components:
- Same as Transformer: 78,464 + 68,757 = 147,221 parameters

### Total HyperMixer Parameters:
- Embeddings: 78,464
- 3 × HyperMixerBlock: 3 × 413,232 = 1,239,696
- OutputProjection: 68,757
- **TOTAL**: 78,464 + 1,239,696 + 68,757 = **1,386,917 parameters** ✓

---

## Analysis

### Why is HyperMixer 2.8× larger?

The culprit is the **Hypernetwork's w2 matrix**: (32, 10,384) = **332,288 parameters**

This single matrix is:
- 6.7× larger than the entire SelfAttention module (49,152 params)
- 5× larger than the FeedForward module (65,920 params)
- 2.9× larger than the entire TransformerBlock (115,584 params)

### Root Cause

The hypernetwork must generate weights for a token-mixing MLP that operates on sequences up to `max_seq_len = 80` tokens. The generated weight matrices scale with `max_seq_len`:

- Token mixing w1: `max_seq_len × hidden_dim` = 80 × 64 = 5,120
- Token mixing w2: `hidden_dim × max_seq_len` = 64 × 80 = 5,120
- Total generated weights: **10,384 parameters**

The hypernetwork's output layer must produce all these weights, resulting in:
- `w2`: (hypernetwork_hidden_dim, output_size) = (32, 10,384) = **332,288 parameters**

### Comparison

| Component | Transformer | HyperMixer | Ratio |
|-----------|-------------|------------|-------|
| Token Mixing | 49,152 (SelfAttention) | 346,800 (Hypernetwork) | 7.1× |
| Channel Mixing | 65,920 (FeedForward) | 65,920 (ChannelMixingMLP) | 1.0× |
| LayerNorms | 512 | 512 | 1.0× |
| **Per Block** | **115,584** | **413,232** | **3.6×** |
| **Total Model** | **493,973** | **1,386,917** | **2.8×** |

---

## Potential Optimizations

### 1. Reduce `max_seq_len`
- Current: 80 → Proposed: 40
- Would reduce output_size from 10,384 to 5,264
- Hypernetwork w2: (32, 5,264) = 168,448 (50% reduction)
- **New total**: ~1,050,000 parameters (24% reduction)

### 2. Reduce `hypernetwork_hidden_dim`
- Current: 32 → Proposed: 16
- Hypernetwork w2: (16, 10,384) = 166,144 (50% reduction)
- **New total**: ~1,220,000 parameters (12% reduction)

### 3. Reduce `token_mixing_hidden_dim`
- Current: 64 (embedding_dim/2) → Proposed: 32
- Would reduce output_size from 10,384 to 5,264
- Same effect as reducing max_seq_len
- **New total**: ~1,050,000 parameters (24% reduction)

### 4. Use Low-Rank Factorization for Hypernetwork w2
- Instead of: w2 (32, 10,384)
- Use: w2_a (32, rank) × w2_b (rank, 10,384)
- With rank=64: 32×64 + 64×10,384 = 2,048 + 664,576 = 666,624
- Still large, but 50% reduction from 332,288 per layer

### 5. Share Hypernetwork Across Layers
- Use the same hypernetwork for all 3 layers
- Would reduce from 3 × 346,800 to 346,800 + 2 × 65,920
- **New total**: ~680,000 parameters (51% reduction)
- **Trade-off**: Less expressive, each layer would generate same mixing patterns

---

## Conclusion

The HyperMixer is significantly larger than the Transformer primarily due to the hypernetwork's need to generate large weight matrices that scale with `max_seq_len`. The most effective optimizations would be:

1. **Reduce max_seq_len** (if acceptable for the task)
2. **Share hypernetwork across layers** (most dramatic reduction)
3. **Reduce token_mixing_hidden_dim** (maintains expressiveness)

The current implementation is **correct** but **parameter-heavy** by design. This is a fundamental trade-off of the HyperMixer architecture.

