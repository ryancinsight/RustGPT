# Mathematical Comparison: TRM vs HRM

## Overview

This document provides a side-by-side mathematical comparison of the Tiny Recursive Model (TRM) and Hierarchical Reasoning Model (HRM), followed by the derivation of a Learning Reasoning Model (LRM) that focuses on auditing recursive thought processes.

## TRM (Tiny Recursive Model)

### Mathematical Formulation

TRM uses a single shared transformer network $f$ that recursively improves answers through latent reasoning.

**Core Operations:**
- **Latent Recursion:** $z^{(k+1)} = f(x, y^{(current)}, z^{(k)})$ for $k = 0$ to $n$
- **Answer Update:** $y^{(new)} = f(y^{(current)}, z^{(n+1)})$

**Deep Supervision Algorithm:**
```
Initialize: x (question), y⁰ (initial answer), z⁰ (latent init)

For t = 1 to T (supervision steps):
    // Phase 1: Latent recursion (no gradients for T-1 steps)
    With torch.no_grad():
        For j = 1 to T-1:
            For i = 0 to n:
                z = f(x, y, z)
            y, z = update_answer(y, z)

    // Phase 2: Final recursion with gradients
    For i = 0 to n:
        z = f(x, y, z)
    y, z = update_answer(y, z)

    // Loss computation
    loss = CrossEntropy(ŷ, y_target) + BinaryCrossEntropy(q, correct)
    loss.backward()
    z = z.detach()  // Detach for next supervision step
```

### Architecture
- **Single Network:** One 2-layer transformer with shared weights
- **Parameters:** 7M total parameters
- **Recursion:** n=6 latent updates, T=3 supervision steps
- **Key Innovation:** Single network recursion with deep supervision

## HRM (Hierarchical Reasoning Model)

### Mathematical Formulation

HRM uses two networks operating at different frequencies with hierarchical latent features.

**Network Definitions:**
- $f_L$: High-frequency network, outputs $z_H$
- $f_H$: Low-frequency network, outputs $z_L$

**Core Operations:**
- $z_H^{(k+1)} = f_L(x, y, z_L^{(k)}, z_H^{(k)})$
- $z_L^{(k+1)} = f_H(x, y, z_L^{(k)}, z_H^{(k)})$

**Hierarchical Recursion:**
```
For each supervision step t:
    For k = 1 to K (recursion depth):
        z_H = f_L(x, y, z_L, z_H)
        z_L = f_H(x, y, z_L, z_H)
    y = update_answer(y, z_H, z_L)
```

### Architecture
- **Dual Networks:** Two 4-layer transformers ($f_L$, $f_H$)
- **Parameters:** ~27M total parameters
- **Recursion:** n=2, T=2 (hierarchical frequencies)
- **Key Innovation:** Biologically-inspired hierarchical processing

## Side-by-Side Mathematical Comparison

| Aspect | TRM | HRM |
|--------|-----|-----|
| **Networks** | Single $f$ | Dual $f_L$, $f_H$ |
| **Latent Space** | Single $z$ | Dual $z_L$, $z_H$ |
| **Recursion Pattern** | Sequential: $z → z → ... → y$ | Hierarchical: $z_L ↔ z_H$ |
| **Frequency** | Single frequency | High/Low frequency |
| **Weight Sharing** | Complete sharing | Separate networks |
| **Complexity** | O(n) recursion steps | O(n²) interactions |
| **Biological Inspiration** | Simple recursion | Hierarchical brain frequencies |

## Visual Architecture Diagrams

### TRM Architecture
```
Input: x (question), y⁰ (initial answer)
       ↓
   ┌─────────────────┐
   │   Latent Init   │ z⁰
   └─────────────────┘
           ↓
   ┌─────────────────┐     ┌─────────────────┐
   │  Recursion × n  │ --> │  Answer Update  │
   │     z → z        │     │     y → y      │
   └─────────────────┘     └─────────────────┘
           ↓                        ↓
       Updated z               Updated y
           ↓                        ↓
   ┌─────────────────┐     ┌─────────────────┐
   │ Deep Supervision │ --> │   Loss & Back  │
   │   (T steps)      │     │   Propagation   │
   └─────────────────┘     └─────────────────┘
```

### HRM Architecture
```
Input: x (question), y⁰ (initial answer)
       ↓
   ┌─────────────────┐     ┌─────────────────┐
   │    High Freq     │     │    Low Freq      │
   │      f_L         │     │      f_H         │
   │  z_H ← z_L,z_H   │     │  z_L ← z_L,z_H   │
   └─────────────────┘     └─────────────────┘
           ↕                        ↕
   ┌─────────────────┐     ┌─────────────────┐
   │   Hierarchical   │     │   Recursion     │
   │   Interactions   │     │   × depth       │
   └─────────────────┘     └─────────────────┘
           ↓                        ↓
   ┌─────────────────┐     ┌─────────────────┐
   │ Deep Supervision │ --> │   Loss & Back  │
   │   (T steps)      │     │   Propagation   │
   └─────────────────┘     └─────────────────┘
```

# Learning Reasoning Model (LRM): Auditing Recursive Thought

## ✅ Implementation Status: COMPLETED

The LRM has been implemented in `src/lrm.rs` with full Rust integration. Key components include:
- **Auditing Architecture**: Multi-head network with confidence scoring and error detection
- **Reasoning Traces**: Complete audit trails for transparency and validation
- **Adaptive Control**: Dynamic recursion depth based on problem complexity
- **Training Integration**: Specialized loss functions for auditing capabilities

## Motivation

While TRM and HRM focus on recursive reasoning, they lack mechanisms to audit and validate the quality of their recursive thought processes. LRM introduces **recursive auditing** - the ability to evaluate not just the final answer, but the quality and correctness of each reasoning step.

## Core Innovations

### 1. Recursive Confidence Scoring
Each recursive step produces not only an updated answer/latent, but also a confidence score that can be audited.

### 2. Thought Process Validation
LRM maintains a "reasoning trace" that can be validated against known correct reasoning patterns.

### 3. Adaptive Recursion Depth
Recursion depth adapts based on problem complexity and current confidence levels.

## Mathematical Formulation

### Confidence-Augmented Recursion

**Single Step with Auditing:**
```
(z^{k+1}, c_z^{k+1}) = f_audit(x, y, z^k, c_z^k)
(y^{new}, c_y^{new}) = g_audit(y, z^{k+1}, c_z^{k+1})
```

Where:
- $c_z^k ∈ [0,1]$: Confidence in latent reasoning at step k
- $c_y^{new} ∈ [0,1]$: Confidence in updated answer
- $f_audit, g_audit$: Auditing-enhanced update functions

### Reasoning Trace Validation

**Trace Definition:**
```
τ = [(z^0, c_z^0), (z^1, c_z^1), ..., (z^n, c_z^n), (y^final, c_y^final)]
```

**Validation Function:**
```
V(τ, τ_correct) = ∏_{k=0}^n similarity(z^k, z_correct^k) · confidence_weight(c_z^k)
```

### Adaptive Recursion Control

**Early Stopping Criterion:**
```
stop = c_y^{current} > θ_confidence ∨ k ≥ n_max
```

**Dynamic Depth Selection:**
```
n_adaptive = min(n_max, max(n_min, complexity_predictor(x, y^0)))
```

## LRM Architecture

### Multi-Head Auditing Network

```
Input: x, y, z, c_prev
       ↓
   ┌─────────────────┐
   │  Reasoning Head │ → z^{new}, reasoning_logits
   └─────────────────┘
           ↓
   ┌─────────────────┐
   │ Confidence Head │ → c_z^{new}
   └─────────────────┘
           ↓
   ┌─────────────────┐
   │  Auditing Head  │ → audit_score, error_flags
   └─────────────────┘
           ↓
   ┌─────────────────┐
   │ Answer Update   │ → y^{new}, c_y^{new}
   └─────────────────┘
```

### Loss Functions

**Primary Loss (Answer Correctness):**
```
L_answer = CrossEntropy(ŷ_final, y_target)
```

**Auditing Loss (Reasoning Quality):**
```
L_audit = MSE(audit_score, true_reasoning_quality) + BinaryCrossEntropy(error_flags, true_errors)
```

**Confidence Calibration Loss:**
```
L_confidence = ConfidenceCalibrationLoss(c_final, accuracy_indicator)
```

**Total Loss:**
```
L_total = L_answer + λ_audit · L_audit + λ_conf · L_confidence
```

## Implementation Strategy

### Phase 1: TRM Foundation
- Start with TRM architecture
- Add confidence heads to existing recursion

### Phase 2: Auditing Integration
- Add auditing network for reasoning validation
- Implement trace collection during forward pass

### Phase 3: Adaptive Control
- Add complexity predictors
- Implement dynamic recursion depth control
- Add early stopping based on confidence

## Expected Benefits

1. **Improved Reliability:** Auditing catches reasoning errors early
2. **Adaptive Efficiency:** Stops recursion when confident
3. **Explainability:** Reasoning traces provide insight into decision process
4. **Robustness:** Validates reasoning against known patterns
5. **Generalization:** Learns to recognize good vs bad reasoning patterns

## Research Directions

1. **Auditing Dataset Creation:** Develop datasets with reasoning traces
2. **Confidence Calibration:** Improve confidence score accuracy
3. **Reasoning Pattern Mining:** Discover common successful reasoning patterns
4. **Multi-Modal Auditing:** Extend to vision, code, mathematics
5. **Meta-Learning:** Learn how to audit different types of reasoning
