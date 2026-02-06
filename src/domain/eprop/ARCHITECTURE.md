# E-prop Module Architecture

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                      src/eprop/mod.rs                   │
│  • EPropError (error types)                             │
│  • Result<T> type alias                                 │
│  • Re-exports all public APIs                           │
└───────────────┬─────────────────────────────────────────┘
                │
                ├──────────────────┬──────────────────┬───────────────┬──────────────┐
                │                  │                  │               │              │
                ▼                  ▼                  ▼               ▼              ▼
    ┌───────────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  ┌────────────┐
    │   config.rs       │  │  neuron.rs   │  │  traces.rs   │  │trainer.rs│  │ utils.rs  │
    │ ───────────────── │  │ ──────────── │  │ ──────────── │  │──────────│  │────────────│
    │ • NeuronModel     │  │ • NeuronState│  │ • Eligibility│  │ • EProp  │  │ • outer_   │
    │ • NeuronConfig    │  │ • NeuronDynam│  │   Traces     │  │   Trainer│  │   product  │
    │ • EPropConfig     │  │   ics        │  │ • TraceUpdat │  │ • Training│  │ • clip_grad│
    │                   │  │              │  │   er         │  │   Stats  │  │ • cosine_  │
    │ Depends on: NONE  │  │ Depends on:  │  │ Depends on:  │  │ Depends: │  │   similar  │
    │                   │  │ • config     │  │ • config     │  │ • ALL    │  │ • norms    │
    └───────────────────┘  │              │  │ • neuron     │  │          │  │ • losses   │
                           └──────────────┘  └──────────────┘  └──────────┘  └────────────┘
```

## Data Flow

### Training Step Pipeline

```
Input (Array1<f32>)
    │
    ▼
┌──────────────────────────────────────────┐
│  1. Forward Pass (trainer.rs)           │
│     • Compute input current              │
│     • Call NeuronDynamics.update()       │
│     • Call TraceUpdater.update()         │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────┐
    │  NeuronState   │ ─────┐
    │  • voltage     │      │
    │  • spikes      │      │ Used by both
    │  • surrogate   │      │
    └────────────────┘      │
                            │
             ┌──────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  2. Trace Update (traces.rs)             │
│     • Update ε^x (presynaptic)           │
│     • Update ε^f (postsynaptic)          │
│     • Update ε^a (adaptation, if ALIF)   │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────┐
    │ Eligibility    │
    │ Traces         │
    │  • eps_x       │
    │  • eps_f       │
    │  • eps_a?      │
    └────────┬───────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  3. Compute Output (trainer.rs)          │
│     output = W_out · spikes              │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  4. Compute Loss & Learning Signal       │
│     loss = MSE(output, target)           │
│     L_t = ∂loss/∂spikes                  │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  5. Apply Gradient Update (trainer.rs)   │
│     • grad = (L_t · ε^f) ⊗ ε^x           │
│     • W -= η · clip(grad)                │
└──────────────────────────────────────────┘
             │
             ▼
        Updated Weights
```

## Complexity Analysis

### Memory Footprint

```
Component            Standard e-prop    ES-D-RTRL       Savings
─────────────────────────────────────────────────────────────────
Eligibility Traces   O(N × N × I)      O(N + I)        N² / (N+I)
Neuron State         O(N)              O(N)            Same
Weights              O(N² + N×I)       O(N² + N×I)     Same
─────────────────────────────────────────────────────────────────
Total                O(N²I)            O(N² + NI)      ~N factor
```

For N=128, I=64: **~128× memory reduction for traces**

### Computational Cost per Timestep

```
Operation              Standard e-prop    ES-D-RTRL
─────────────────────────────────────────────────────
Neuron Dynamics        O(N² + N×I)       O(N² + N×I)
Trace Update           O(N²×I)           O(N + I)
Gradient Computation   O(N²×I)           O(N×I + N²)
─────────────────────────────────────────────────────
Total                  O(N²I)            O(N² + NI)
```

## Interface Contracts

### Configuration → Neuron
```rust
NeuronConfig {
    model: NeuronModel,     // LIF or ALIF
    alpha: f32,             // Membrane decay ∈ (0,1)
    v_threshold: f32,       // Spike threshold > 0
    rho: f32,              // Adaptation decay ∈ (0,1)
    beta: f32,             // Adaptation strength ≥ 0
    gamma_pd: f32,         // Surrogate param > 0
}
    ↓
NeuronDynamics::update(state, input_current)
    → Result<()>
```

### Neuron → Traces
```rust
NeuronState {
    voltage: Array1<f32>,         // Membrane potentials
    spikes: Array1<f32>,          // Binary spikes {0,1}
    filtered_spikes: Array1<f32>, // Low-pass filtered
    surrogate_deriv: Array1<f32>, // ∂z/∂v approximation
    adaptation?: Array1<f32>,     // ALIF only
}
    ↓
TraceUpdater::update(traces, state, input)
    → Result<()>
```

### Traces → Trainer
```rust
EligibilityTraces {
    eps_x: Array1<f32>,    // Presynaptic (input_dim)
    eps_f: Array1<f32>,    // Postsynaptic (num_neurons)
    eps_a?: Array1<f32>,   // Adaptation (num_neurons)
}
    ↓
TraceUpdater::compute_gradient_factors(traces, L_t)
    → Result<(modulated_eps_f, eps_x)>
    ↓
utils::outer_product(mod_f, eps_x)
    → Array2<f32>  [gradient]
```

## Error Handling Flow

```
User Call
    │
    ▼
EPropTrainer::train_step()
    │
    ├─► forward()
    │    ├─► NeuronDynamics::update()
    │    │    └─► TraceDimensionMismatch?
    │    └─► TraceUpdater::update()
    │         └─► TraceDimensionMismatch?
    │
    ├─► compute_output()
    │    └─► (infallible)
    │
    └─► apply_update()
         ├─► compute_gradient_factors()
         │    └─► TraceDimensionMismatch?
         └─► clip_gradient()
              └─► (infallible)
    │
    ▼
Result<f32, EPropError>
    │
    └─► User handles error or propagates
```

## Test Coverage Map

```
config.rs (10 tests)
├─ NeuronConfig validation
│  ├─ Valid defaults
│  ├─ Invalid alpha (≤0 or ≥1)
│  └─ Invalid parameters per model
└─ EPropConfig validation
   ├─ Valid defaults
   ├─ Zero dimensions
   └─ Invalid hyperparameters

neuron.rs (16 tests)
├─ State management
│  ├─ Creation with/without adaptation
│  └─ Reset functionality
├─ LIF dynamics
│  ├─ No spike (weak input)
│  ├─ Spike (strong input)
│  └─ Spike reset
└─ ALIF dynamics
   ├─ Adaptation accumulation
   └─ Threshold increase

traces.rs (14 tests)
├─ Initialization
├─ Reset functionality
├─ Presynaptic update
├─ Postsynaptic update
├─ Adaptation update (ALIF)
├─ Gradient factor computation
└─ Exponential decay verification

trainer.rs (15 tests)
├─ Trainer creation
├─ Forward pass
├─ Multi-cycle forward
├─ Single train step
├─ Multiple train steps
├─ State reset
├─ Gradient clipping
├─ Weight export/import
├─ Statistics tracking
└─ ALIF integration

utils.rs (21 tests)
├─ Outer product
├─ Gradient clipping
├─ Cosine similarity
├─ Vector norms
├─ Normalization
├─ Activations (ReLU, softmax)
└─ Loss functions (MSE, cross-entropy)
```

## Public API Surface

```rust
// Main entry point
pub struct EPropTrainer { ... }

impl EPropTrainer {
    pub fn new(config: EPropConfig) -> Result<Self>
    pub fn forward(&mut self, input: &Array1<f32>) -> Result<Array1<f32>>
    pub fn forward_cycles(&mut self, input: &Array1<f32>, cycles: Option<usize>) -> Result<Array1<f32>>
    pub fn train_step(&mut self, input: &Array1<f32>, target: &Array1<f32>) -> Result<f32>
    pub fn apply_update(&mut self, learning_signal: &Array1<f32>) -> Result<()>
    pub fn compute_output(&self) -> Array1<f32>
    pub fn reset_state(&mut self)
    pub fn stats(&self) -> &TrainingStats
    pub fn export_weights(&self) -> HashMap<String, Array2<f32>>
    pub fn import_weights(&mut self, weights: HashMap<String, Array2<f32>>) -> Result<()>
}

// Configuration
pub struct EPropConfig { ... }
pub struct NeuronConfig { ... }
pub enum NeuronModel { LIF, ALIF }

// Statistics
pub struct TrainingStats {
    pub num_updates: usize,
    pub avg_firing_rate: f32,
    pub grad_norms: Vec<f32>,
    pub losses: Vec<f32>,
    pub bptt_similarity: Option<f32>,
}

// Utilities
pub fn outer_product(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32>
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32
pub fn clip_gradient(grad: Array2<f32>, max_norm: f32) -> Array2<f32>
```

## Thread Safety

Currently **NOT thread-safe** (uses `&mut self`):
- `EPropTrainer` requires exclusive access
- No internal synchronization
- Designed for single-threaded training

For multi-threaded training:
- Create separate trainers per thread
- Use message passing for coordination
- Aggregate gradients externally

## Future Architectural Enhancements

1. **Sparse Weights**: Replace `Array2<f32>` with CSR format
2. **Multi-Layer**: Stack multiple `EPropTrainer` instances
3. **GPU Support**: Add CUDA/Vulkan backend via trait
4. **Async Training**: Separate trace update from gradient application
5. **Distributed**: Add parameter server for multi-node training
