# RustGPT Development Guidelines

## Build Commands

### Standard Commands
```bash
# Build library (no tests)
cargo build --lib

# Build with optimization and checks
cargo build --release

# Build with all features
cargo build --all-features

# Check code without building
cargo check --lib

# Check with all targets
cargo check --all-targets
```

### Test Commands
```bash
# Run all library tests
cargo test --lib

# Run specific test module
cargo test --lib attention
cargo test --lib models::llm
cargo test --lib training

# Run tests in release mode (for benchmarking)
cargo test --lib --release

# Run specific test function
cargo test --lib test_function_name

# Run tests matching a pattern
cargo test --lib test_attention

# Run tests in a file
cargo test --lib src/attention/poly_attention.rs

# Run next test that failed
cargo test --lib -- --next

# Run tests until failure
cargo test --lib -- --no-fail-fast

# Run tests with output
cargo test --lib -- --nocapture

# Run tests with immediate output (for debugging)
cargo test --lib test_name -- -- --exact -- --nocapture

# Show test output (stdout)
cargo test --lib -- -- --show-output

# Run doctests
cargo test --doc

# Ignore specific files/directories
cargo test --lib --ignore src/test_module
```

### Linting Commands
```bash
# Run clippy on all targets
cargo clippy --all-targets -- -W clippy::all

# Fix clippy warnings automatically
cargo clippy --fix --lib --allow-dirty --allow-staged

# Check for specific clippy lints
cargo clippy --lib -W clippy::needless_range_loop
cargo clippy --lib -W clippy::clone_on_ref_ptr

# Deny specific lints (fail if found)
cargo clippy --lib -D clippy::unwrap_used
```

### Formatting Commands
```bash
# Format code in-place
cargo fmt

# Check formatting without making changes
cargo fmt -- --check

# Format specific files
cargo fmt src/models/llm.rs src/attention/

# Format only modified files
cargo fmt -- --write-mode=overwrite
```

### Documentation Commands
```bash
# Generate and open documentation
cargo doc --open

# Generate documentation without private items
cargo doc --no-deps --document-private-items

# Check documentation
cargo doc --no-deps
```

### Clean Commands
```bash
# Clean build artifacts
cargo clean

# Clean debug builds only
cargo clean --debug

# Clean release builds only
cargo clean --release

# Clean specific package
cargo clean -p llm

# Clean dependency cache
cargo clean --doc

# Dry run (show what would be cleaned)
cargo clean --dry-run

# Remove old lockfile and re-resolve
rm Cargo.lock
cargo update
```

---

## Code Style Guidelines

### Imports and Module Organization
```rust
// Order: std external crates -> local crates -> local modules

// 1. Standard library imports
use std::fs;
use std::collections::HashMap;
use std::sync::Arc;

// 2. External dependencies
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

// 3. Local crate modules
use crate::model_config;
use crate::attention;
use crate::training;

// 4. Use self-referential imports only when needed
use self::ModelError;
```

### Naming Conventions
```rust
// Constants: UPPER_SNAKE_CASE
pub const MAX_SEQ_LEN: usize = 256;
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

// Types: PascalCase
pub struct Adam { ... }
pub enum ModelError { ... }

// Functions: snake_case
pub fn train_batch(...) -> Result<...> { ... }

// Modules: snake_case
pub mod attention;
pub mod models::llm;

// Fields: snake_case
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<LayerEnum>,
}

// Lifetime names: short lowercase
fn forward<'a>(&'a self, ...) { ... }
```

### Type Annotations
```rust
// Always annotate function parameters with types
pub fn compute_loss(&self, logits: &Array2<f32>, targets: &[usize]) -> f32

// Use explicit types for complex generics
let grad: Array2<f32> = compute_gradients::<f32>(...);

// Prefer &str over &String for function parameters
fn process(&self, input: &str) -> Result<...>

// Use Box<dyn Trait> for trait objects sparingly
// Prefer generics when performance matters
```

### Error Handling
```rust
// Use Result<T> for operations that can fail
pub fn train_batch(&mut self, ...) -> Result<()> {
    // ...
}

// Use custom Error type with descriptive variants
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Failed to load model from {path}")]
    Load { path: String },
    
    #[error("Training diverged at epoch {epoch}")]
    Training { epoch: usize },
    
    #[error("Invalid input: {reason}")]
    InvalidInput { reason: String },
}

// Use ? operator for error propagation
pub fn forward(&mut self, ...) -> Result<Array2<f32>> {
    let input = load_input()?;
    let output = self.network.forward(&input)?;
    Ok(output)
}

// Context for errors
let result = operation().map_err(|e| {
    ModelError::Generic {
        message: format!("Failed to process: {}", e),
    }
})?;

// Use unwrap_or for default values
let value = optional_value.unwrap_or(0.0);
let value = unsafe { value.unwrap_unchecked() }; // Only when absolutely certain
```

### Constants and Magic Numbers
```rust
// Define constants at module or crate level
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;
pub const DEFAULT_BATCH_SIZE: usize = 32;
pub const DEFAULT_EPOCHS: usize = 100;
pub const GRADIENT_CLIP_THRESHOLD: f32 = 5.0;
pub const EPSILON: f32 = 1e-6;

// Document non-obvious constants
pub const TRM_AUX_DECAY_RATE: f32 = 0.6; // Decay towards earlier steps
pub const LARS_EMA_BETA: f32 = 0.9; // 90% memory for gradient statistics

// Group related constants
pub struct TrainingHyperParams {
    pub lars_ema_beta: f32,
    pub lars_power_balance: f32,
    pub lars_min_scale: f32,
    pub lars_max_scale: f32,
}
```

### Documentation Standards
```rust
// Crate-level documentation
//! RustGPT: A Large Language Model implementation in Rust
//!
//! This crate provides:
//! - Transformer-based and recurrent architectures (LRM, Mamba)
//! - Multiple attention mechanisms (PolyAttention, MoH)
//! - Training pipelines with E-Prop and standard backpropagation
//! - Inference with speculative decoding

// Module-level documentation
//! Adam optimizer with AMSGrad and AdamW variants
//!
//! Provides efficient, numerically stable implementations of:
//! - Standard Adam optimizer
//! - AMSGrad variant with maximum gradient tracking
//! - AdamW with decoupled weight decay
//!
//! # Examples
//!
//! ```rust
//! use llm::adam::Adam;
//!
//! let mut optimizer = Adam::new((hidden_dim, vocab_size));
//! let mut model = Model::new(...);
//!
//! for epoch in 0..epochs {
//!     for batch in data.chunks(batch_size) {
//!         let (loss, ...) = model.train_batch(batch, 0.001)?;
//!         
//!         // Compute gradients
//!         let grads = model.compute_gradients(...)?;
//!         
//!         // Update parameters
//!         optimizer.update(params, &grads, 0.001)?;
//!     }
//! }
//! ```

// Struct-level documentation
/// Adam optimizer configuration
///
/// Uses hyperparameters β₁ and β₂ for gradient moment estimation,
/// with optional AMSGrad variant and decoupled weight decay.
///
/// # Fields
///
/// - `beta1`: Exponential decay rate for first moment estimate
/// - `beta2`: Exponential decay rate for second moment estimate  
/// - `epsilon`: Small constant for numerical stability
/// - `weight_decay`: L2 regularization coefficient (AdamW only)
/// - `use_amsgrad`: Enable AMSGrad variant for better convergence
///
/// # Examples
///
/// ```rust
/// let optimizer = Adam {
///     beta1: 0.9,
///     beta2: 0.999,
///     epsilon: 1e-8,
///     use_amsgrad: true,
///     weight_decay: 0.01,
/// };
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub timestep: u32,
    pub m: Array2<f32>,
    pub v: Array2<f32>,
    pub v_hat_max: Option<Array2<f32>>,
    pub use_amsgrad: bool,
    pub weight_decay: f32,
    pub use_decoupled_wd: bool,
}

// Function-level documentation
/// Update optimizer parameters with gradients
///
/// # Arguments
///
/// * `params` - Parameters to update (must match shape of gradients)
/// * `grads` - Computed gradients (same shape as params)
/// * `lr` - Learning rate for this update
///
/// # Returns
///
/// Returns `Result<()>` indicating success or failure
///
/// # Panics
///
/// This function will panic if:
/// - `params` and `grads` have mismatched shapes
/// - Learning rate is NaN or infinite
/// - Gradient norms exceed safety thresholds
///
/// # Safety
///
/// - Validates input shapes before update
/// - Clips gradients to prevent explosion
/// - Handles NaN/Inf in gradients gracefully
pub fn update(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) -> Result<()> {
    // Validate shapes
    if params.shape() != grads.shape() {
        return Err(ModelError::ShapeMismatch {
            expected: vec![params.shape()],
            actual: vec![grads.shape()],
            message: format!("Parameter and gradient shapes mismatch"),
        });
    }
    
    // Clip gradients
    let grad_norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if grad_norm > GRADIENT_CLIP_THRESHOLD {
        let scale = GRADIENT_CLIP_THRESHOLD / grad_norm;
        grads.mapv_inplace(|x| x * scale);
    }
    
    // Update parameters
    self.timestep += 1;
    // ... update logic
    
    Ok(())
}
```

### Memory Management
```rust
// Use references (&) instead of clones where possible
fn process_batch(&self, input: &Array2<f32>) -> Array2<f32> {
    let output = self.network.forward(input);  // No clone
    output
}

// Use views (ArrayView2, ArrayView1) for read-only access
fn compute_loss(logits: &ArrayView2<f32>, targets: &[usize]) -> f32 {
    let logsumexp = logits.iter().map(|&x| x.exp()).sum();
    // ... compute loss without cloning
}

// Reuse allocations via scratch buffers
struct TrainingScratch {
    accumulated_param_grads: Vec<Vec<Array2<f32>>>,
    layer_inputs: Vec<Array2<f32>>,
}

// Pre-allocate when size is known
let mut buffer = Vec::with_capacity(expected_size);

// Use in-place operations
array.mapv_inplace(|x| x * 2.0);

// Clear and reuse vectors instead of reallocating
scratch.layer_inputs.clear();
for item in batch {
    scratch.layer_inputs.push(item);
}
```

### Performance Guidelines
```rust
// Use Rayon for parallel iteration
use rayon::prelude::*;

let results: Vec<_> = data.par_iter()
    .map(|item| expensive_computation(item))
    .collect();

// Minimize allocations in hot paths
// - Prefer iterators over collect when intermediate results not needed
data.iter().filter(|x| x > threshold).count()

// - Use Cow for conditionally borrowed data
use std::borrow::Cow;

// Batch operations on arrays
let sum = array.sum_axis(Axis(0));

// Use SIMD-friendly operations where possible
// - Avoid branches in tight loops
// - Use contiguous memory layouts

// Cache expensive computations
let cached = expensive_operation();
if needed {
    process(cached);
}
```

### Safety Guidelines
```rust
// Use safe APIs when performance is not critical
let value = vec.get(i).copied().unwrap_or(0.0);

// Use unsafe only when absolutely necessary and well-documented
/// SAFETY: The caller must ensure index is within bounds
pub unsafe fn get_unchecked(&self, index: usize) -> &T {
    debug_assert!(index < self.len(), "Index out of bounds");
    unsafe { self.data.get_unchecked(index) }
}

// Add debug assertions for invariants
debug_assert!(!batch.is_empty(), "Batch cannot be empty");
debug_assert!(layer_idx < self.network.len(), "Invalid layer index");

// Use #[must_use] for results that must not be ignored
#[must_use]
pub fn compute_gradients(&self, ...) -> Array2<f32> {
    // ...
}

// Avoid unwrap() on external data
let config = config_file.read()
    .map_err(|e| Error::Io { source: e })?;

// Prefer checked arithmetic
let result = x.checked_mul(y).ok_or_else(|| {
    Error::Overflow { operation: "multiplication overflow" }
});
```

### Testing Guidelines
```rust
// Use descriptive test names
#[test]
fn test_adam_gradient_update_works_correctly() {
    // ...
}

// Use assertions to verify expected behavior
assert!(result.is_ok(), "Expected success");
assert_eq!(loss, expected_loss, "Loss computation incorrect");
assert!((output - expected).abs() < 1e-5, "Output out of tolerance");

// Use property-based testing
use proptest::proptest;

#[proptest]
fn prop_forward_is_deterministic(input: Vec<f32>) {
    // ... property test
}

// Test edge cases
#[test]
fn test_empty_input() { }
#[test]
fn test_single_element() { }
#[test]
fn test_max_capacity() { }

// Use fixtures for common test data
fn setup_test_model() -> LLM {
    // ...
}

// Organize tests into modules
// tests/
// ├── attention/
// ├── models/
// ├── training/
// └── integration/

// Use test utilities when available
#[cfg(test)]
mod test_utils {
    pub fn assert_approx_eq(a: f32, b: f32, tolerance: f32) {
        assert!((a - b).abs() < tolerance);
    }
}
```

### Borrow Checker Guidelines
```rust
// Avoid reborrowing in loops
for item in &items {  // ✗ Problem: reborrows &items
    process(item);
}

// Solution: Use indices or collect once
let items_vec: Vec<_> = items.iter().collect();
for item in &items_vec {  // ✓ OK: items_vec is separate
    process(item);
}

// Avoid multiple mutable borrows
let (a, b) = data.split_at_mut(mid);  // ✗ Problem: mutable borrows overlap

// Solution: Use scopes or clone one part
let prefix = data[..mid].to_vec();
let suffix = data[mid..].to_vec();
// process both parts

// Re-borrowing pattern to avoid
// Instead of:
let x = self.data.get(i);
process(x);
let y = self.data.get(i + 1);  // ❌ Reborrow!
process(y);

// Use separate borrows
{
    let x = self.data.get(i);
    process(x);
    drop(x);
}
{
    let y = self.data.get(i + 1);
    process(y);
}

// Using raw pointers for complex cases (when necessary)
// Only use this as last resort and document heavily
let self_ptr = self as *mut _;
let result = unsafe { &mut *self_ptr }.method(...);
```

### Clone Guidelines
```rust
// Clone only when necessary
// ✗ Bad: Clones large array unnecessarily
let data_clone = large_array.clone();

// ✓ Good: Use references
let data_ref = &large_array;

// ✓ Good: Use views
let slice = &array.slice(s![..10]);

// Clone small data
let small_clone = small_vec.clone();  // Acceptable

// Use clone for ownership transfer (into_iter()), not &clone()
let owned: Vec<_> = data.iter().cloned().collect();

// Avoid clone in tight loops
// Bad:
for _ in 0..1000 {
    let _ = expensive_computation(data.clone());
}

// Good:
let data_ref = &data;
for _ in 0..1000 {
    let _ = expensive_computation(data_ref);
}

// Consider using Cow for conditional cloning
use std::borrow::Cow;

fn process(data: &[f32]) -> f32 {
    let data = Cow::Borrowed(data);
    // ... may or may not clone based on conditions
}
```

### Async Guidelines
```rust
// This codebase primarily uses synchronous code with Rayon for parallelism

// If adding async in future:
use tokio::task::spawn;

// Use async functions sparingly
async fn load_data() -> Result<Vec<String>> {
    // ...
}

// Use ? for error propagation in async contexts
async fn process() -> Result<()> {
    let data = fetch_data().await?;
    process(data)?;
    Ok(())
}
```

### Struct and Enum Guidelines
```rust
// Derive common traits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub name: String,
    pub size: usize,
}

// Use #[non_exhaustive] for enums that may grow
#[non_exhaustive]
pub enum LayerType {
    Dense,
    Convolutional,
    Attention,
    // Future variants can be added without breaking code
}

// Prefer transparent wrappers over newtypes
pub struct LayerId(pub usize);

// Use field-less enums for options
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Relu,
    Gelu,
    Swish,
}

// Implement Default for complex types
impl Default for TrainingHyperParams {
    fn default() -> Self {
        Self {
            residual_decorrelation_weight: 0.0,
            residual_decorrelation_adaptive: false,
            // ... all fields with sensible defaults
        }
    }
}
```

### Attribute Usage
```rust
// Use #[inline] for small, hot functions
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Use #[cold] for error paths
#[cold]
fn handle_error(error: ModelError) -> ! {
    panic!("{}", error);
}

// Use #[must_use] for important returns
#[must_use]
pub fn compute_loss(&self, ...) -> f32 {
    // ...
}

// Use #[allow(dead_code)] sparingly with justification
#[allow(dead_code)]
fn legacy_function() {
    // TODO: Remove when migrating to new system
}

// Use #[cfg(test)] for test-only code
#[cfg(test)]
pub mod test_utils {
    pub fn create_test_model() -> LLM {
        // ...
    }
}
```

### Common Patterns
```rust
// Builder pattern for complex construction
impl LayerConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }
    
    pub fn build(self) -> Result<Layer> {
        // Validate and construct
        Ok(Layer::from_config(self)?)
    }
}

// Visit pattern for heterogeneous data structures
trait Visitor {
    fn visit_dense(&mut self, layer: &DenseLayer);
    fn visit_attention(&mut self, layer: &AttentionLayer);
}

impl From<DenseLayer> for LayerEnum {
    fn from(layer: DenseLayer) -> Self {
        LayerEnum::Dense(Box::new(layer))
    }
}

// Result chaining
pub fn validate_and_process(input: &str) -> Result<Output> {
    let parsed = parse(input)?;
    let validated = validate(parsed)?;
    let processed = process(validated)?;
    Ok(processed)
}

// Context pattern for passing state through computation
struct Context {
    config: Config,
    buffer: Vec<f32>,
}

fn run_with_context(config: Config) -> Result<Vec<f32>> {
    let ctx = Context::new(config);
    process_step_1(&mut ctx)?;
    process_step_2(&mut ctx)?;
    Ok(ctx.into_results())
}
```

---

## Running a Single Test

```bash
# Run a specific test
cargo test --lib attention::poly_attention::tests::test_forward

# Run tests matching a pattern
cargo test --lib test_attention

# Run tests in a file
cargo test --lib src/attention/poly_attention.rs

# Run next failing test
cargo test --lib -- --next

# Run tests until failure
cargo test --lib -- --no-fail-fast

# Run tests with filter (no capture)
cargo test --lib 2>&1 | grep test_name

# Run tests with immediate output (for debugging)
cargo test --lib test_name -- -- --nocapture

# Show test output with colors (force even in CI)
cargo test --lib test_name -- --exact -- --nocapture
```

---

## CI/CD Integration

```bash
# Run all checks
cargo clippy --all-targets
cargo fmt --check
cargo test --lib

# Build for release
cargo build --release

# Create documentation
cargo doc --no-deps
```

---

## Workspace-Specific Notes

RustGPT is a single-crate workspace with these characteristics:

1. **Large codebase** (~50,000+ lines) - Changes require careful consideration of impact
2. **ML training code** - Numerical stability and gradient handling are critical
3. **Heavy use of ndarray** - Memory management in hot paths is important
4. **Complex module interdependencies** - Changes to core types affect many modules

### Module Dependencies
```
src/
├── attention/         # Attention mechanisms (depends on nothing)
├── layers/           # Network layers (depends on attention, soft, richards)
├── models/            # Model implementations (depends on layers, embeddings, loss)
├── training/          # Training pipelines (depends on models, loss, data)
├── encoding/          # Tokenization (depends on nothing)
├── embeddings/        # Embeddings (depends on nothing)
├── loss/              # Loss functions (depends on nothing)
├── metrics/           # Evaluation metrics (depends on nothing)
├── eprop/             # E-Prop optimizer (depends on layers)
├── inference/          # Inference utilities (depends on models)
├── cli/               # CLI parsing (depends on config)
├── utils/             # Utilities (depends on rng)
```

### Key Types to Be Aware Of

```rust
// Main model type
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<LayerEnum>,  // Enum of all layer types
    pub training_scratch: TrainingScratch,  // Scratch buffers for training
    pub median_grad_ema: Option<f32>,  // EMA of gradient norms
}

// Layer types (can grow over time)
pub enum LayerEnum {
    TokenEmbeddings(TokenEmbeddings),
    RichardsGlu(Box<RichardsGlu>),
    MixtureOfExperts(Box<MixtureOfExperts>),
    DynamicTanhNorm(Box<RichardsNorm>),
    OutputProjection(OutputProjection),
    PolyAttention(Box<PolyAttention>),
    TransformerBlock(Box<TransformerBlock>),
    DiffusionBlock(Box<DiffusionBlock>),
    LRM(Box<LRM>),
    TitansMemory(Box<NeuralMemory>),
    LifLayer(Box<LifLayer>),
    AlifLayer(Box<AlifLayer>),
}

// Configuration types
pub struct TrainingHyperParams {
    pub residual_decorrelation_weight: f32,
    pub residual_decorrelation_adaptive: bool,
    pub residual_hardneg_weight: f32,
    pub residual_hardneg_adaptive: bool,
    pub residual_hardneg_k: usize,
    pub residual_hardneg_margin: f32,
    pub residual_hardneg_temperature: f32,
    pub residual_hardneg_bank_size: usize,
}
```

---

## Best Practices Summary

1. **Always run `cargo fmt` before committing**
2. **Always run `cargo clippy --all-targets` before committing**
3. **Always run `cargo test --lib` before committing**
4. **Use descriptive commit messages**
5. **Keep functions focused and small (<100 lines when possible)**
6. **Document public APIs**
7. **Handle errors gracefully with context**
8. **Prefer composition over inheritance**
9. **Use references (&) instead of clones when safe**
10. **Use `Result<T>` instead of panicking for recoverable errors**

---

## Notes for Agentic Coding

### Complexity Considerations
- This codebase is large and complex
- Changes to core types (LayerEnum, LLM) have widespread impact
- Testing changes is critical due to ML numerical behavior
- Performance changes should be benchmarked

### Common Pitfalls
- **Don't introduce clone() in hot training loops** - Use references or scratch buffers
- **Don't use unwrap() on external/untrusted data** - Use ? for Result propagation
- **Don't break borrow checker without understanding** - Use scope separation or Cow
- **Don't ignore clippy warnings** - They often indicate real issues
- **Don't change function signatures without considering callers** - Many functions use these

### Recommended Workflow
1. Make small, focused changes
2. Test the specific module/function affected
3. Run benchmarks for performance-critical paths
4. Review clippy output carefully
5. Ensure formatting passes
6. Update documentation if public API changes

### Testing Strategy
- Unit tests: Test individual components in isolation
- Integration tests: Test interactions between modules
- Property tests: Verify invariants across random inputs
- Regression tests: Ensure bug fixes don't introduce new issues
- Benchmark tests: Verify performance doesn't degrade
