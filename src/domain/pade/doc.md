# Chebyshev–Padé approximation (exp)

This module provides a numerically-stable, fast approximation for `exp(x)` used throughout the project
(attention softmax/logsumexp, routing, SSM state updates, loss functions, etc.).

## Scope and structure

- The public API is intentionally small:
  - `pade::PadeExp::exp(f64) -> f64` (core scalar exp)
  - `pade::exp<T: ExpScalar>(T) -> T` (generic helper for `f32`/`f64` call sites)
  - `pade::PrecisionLevel` and `PadeExp::exp_with_precision`
- Implementation details live under a *deep, vertical* module hierarchy in `pade/exp/**`:
  - `approximants/*` – rational approximants ([3/3], [5/5], [7/7], …)
  - `range_reduction/*` – range reduction and binary scaling (`ldexp`)
  - `array/*` – ndarray helpers (`exp_array`, in-place, iter-based)
  - `simd/*` – SIMD dispatch scaffolding (currently safe fallbacks)
  - `analysis/*` – accuracy benchmarks, bounds, diagnostics

## Notes on correctness

- Special values are handled explicitly: NaN propagates, `+∞` returns `+∞`, `-∞` returns `0`.
- Overflow/underflow are bounded to match IEEE-754 behavior in practical ranges.
- For gradients, the project treats the stable approximation as a drop-in replacement for `std::exp`,
  so `exp_grad(x)` evaluates the same approximation again.

## Usage

```rust
use llm::domain::pade;

let y: f32 = pade::exp(1.0f32);
let z: f64 = pade::PadeExp::exp(1.0);
```
