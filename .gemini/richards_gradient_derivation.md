# Extended Richards Curve Gradient Derivations

## Forward Pass
```
σ = [1 + β * exp(-k(input-m))]^(-1/ν)
```

Where (for defaults):
- β = 1.0
- input = x (after all transformations with defaults)

## Gradients

### ∂σ/∂ν (nu gradient)
Using logarithmic differentiation:
```
ln(σ) = (-1/ν) * ln(base)
(1/σ) * ∂σ/∂ν = (1/ν²) * ln(base)
∂σ/∂ν = σ * ln(base) / ν²
```

### ∂σ/∂k (k gradient)
Chain rule through base and exponent:
```
∂σ/∂base = (-1/ν) * base^(-1/ν - 1) = (-1/ν) * σ / base
∂base/∂exponent = β * exp(exponent) = β * exp_term  
∂exponent/∂k = -(input - m)

∂σ/∂k = (∂σ/∂base) * (∂base/∂exponent) * (∂exponent/∂k)
      = [(-1/ν) * σ / base] * [β * exp_term] * [-(input - m)]
      = (1/ν) * σ * β * exp_term * (input - m) / base
```

For β=1: exp_term/base = 1 - σ^ν (approximately 1 - σ for small ν differences)

More accurately, for Richards curve:
```
∂σ/∂k = (1/ν) * σ * exp_term * (input - m) / base
```

### ∂σ/∂m (m gradient)
```
∂exponent/∂m = k

∂σ/∂m = (∂σ/∂base) * (∂base/∂exponent) * (∂exponent/∂m)
      = [(-1/ν) * σ / base] * [β * exp_term] * [k]
      = (-k/ν) * σ * β * exp_term / base
```

### ∂σ/∂β (beta gradient)
```
∂base/∂β = exp(exponent) = exp_term

∂σ/∂β = (∂σ/∂base) * (∂base/∂β)
      = [(-1/ν) * σ / base] * exp_term
      = (-1/ν) * σ * exp_term / base
```

### ∂σ/∂temp (temperature gradient)
Chain through input:
```
∂input/∂temp = -input_scale * scale * adaptive_normalized / temp²
             = -input_scale * scale * temp_scaled / temp

∂σ/∂input = (∂σ/∂base) * (∂base/∂exponent) * (∂exponent/∂input)
          = [(-1/ν) * σ / base] * [β * exp_term] * [-k]
          = (k/ν) * σ * β * exp_term / base

∂σ/∂temp = (∂σ/∂input) * (∂input/∂temp)
```

## Final Formulas (β=1 default)

```rust
// Nu gradient
d_sigma_d_nu = sigma * base.ln() / (nu * nu)

// K gradient  
d_sigma_d_k = (1.0 / nu) * sigma * exp_term * (input - m) / base

// M gradient
d_sigma_d_m = (-k / nu) * sigma * exp_term / base

// Beta gradient (for β learnable)
d_sigma_d_beta = (-1.0 / nu) * sigma * exp_term / base

// Temperature gradient
d_sigma_d_temp = (k / nu) * sigma * exp_term / base * (-temp_scaled / temp)
```
