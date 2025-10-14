# Architecture Decision Record: Model Persistence Implementation

## Status
Accepted

## Context

RustGPT requires the ability to save and load trained model parameters to enable:

- Training continuation across sessions
- Model deployment and sharing
- Experimentation with different training configurations
- Educational demonstration of model serialization

The implementation must support the complex model architecture consisting of:

- Vocabulary (HashMap-based encoding/decoding)
- Multiple layer types (Embeddings, SelfAttention, FeedForward, LayerNorm, OutputProjection)
- Large parameter arrays (ndarray::Array2<f32>)
- Optimizer state (Adam with momentum and velocity arrays)

## Decision

Implement model persistence using:

- **Serialization Format**: JSON via serde_json for human-readable, cross-platform compatibility
- **Layer Polymorphism**: LayerEnum enum to replace Box<dyn Layer> for serializable trait objects
- **API Design**: LLM::save() and LLM::load() methods with Result-based error handling
- **Validation**: Structural validation on load to ensure architecture consistency

## Rationale

### JSON Format Choice

- **Human-readable**: Enables inspection and debugging of model parameters
- **Cross-platform**: No endianness or architecture dependencies
- **Version control friendly**: Text-based diffs for parameter changes
- **Ecosystem compatibility**: Works with existing JSON tools and APIs
- **Trade-off**: Larger file size and slower I/O compared to binary formats

### LayerEnum for Polymorphism

- **Serializable**: Enum variants can derive Serialize/Deserialize
- **Type Safety**: Compile-time guarantees of layer types
- **Performance**: No dynamic dispatch overhead in hot paths
- **Maintainability**: Clear enumeration of supported layer types

### API Design

- **Simple**: save(path) and load(path) methods on LLM
- **Error Handling**: Result types for I/O and deserialization errors
- **Consistency**: Follows Rust standard library patterns (e.g., fs::read/write)

## Alternatives Considered

### Binary Serialization (bincode)

- **Pros**: Smaller files, faster I/O, better performance
- **Cons**: Not human-readable, potential compatibility issues
- **Decision**: Deferred for future optimization; JSON sufficient for educational use

### Protocol Buffers / FlatBuffers

- **Pros**: Efficient, strongly typed, cross-language
- **Cons**: Additional complexity, schema management
- **Decision**: Overkill for current scope; JSON adequate

### Separate Parameter Files

- **Pros**: Modular loading, partial model updates
- **Cons**: Complexity in managing multiple files
- **Decision**: Monolithic approach simpler for MVP

## Implementation Details

### LayerEnum Definition

```rust
#[derive(Serialize, Deserialize)]
pub enum LayerEnum {
    Embeddings(Embeddings),
    SelfAttention(Box<crate::self_attention::SelfAttention>),
    FeedForward(Box<crate::feed_forward::FeedForward>),
    LayerNorm(crate::layer_norm::LayerNorm),
    OutputProjection(OutputProjection),
}
```

Large layer variants (SelfAttention, FeedForward) are boxed to reduce enum size and satisfy clippy's large_enum_variant warning.

### Save/Load Methods

```rust
impl LLM {
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = fs::read_to_string(path)?;
        let llm: LLM = serde_json::from_str(&data)?;
        Ok(llm)
    }
}
```

### Required Changes

- Add Serialize/Deserialize derives to all layer structs
- Enable ndarray serde feature
- Replace Vec<Box<dyn Layer>> with Vec<LayerEnum>
- Update LLM::default() to use LayerEnum variants
- Flatten TransformerBlock into individual layers

## Consequences

### Positive

- ✅ Model persistence enables training continuation
- ✅ Human-readable format aids debugging
- ✅ Type-safe layer polymorphism
- ✅ Simple, intuitive API

### Negative

- ⚠️ JSON files larger than binary formats
- ⚠️ Serialization slower than optimized binary
- ⚠️ Breaking change to Layer trait (Box<dyn Layer> → LayerEnum)

### Risks

- **File Size**: Large models may produce very large JSON files
- **Performance**: Save/load operations may be slow for large models
- **Compatibility**: Future layer types require enum variant addition

## Mitigation

- **File Size**: Implement compression if needed (future)
- **Performance**: Optimize ndarray serialization (future binary format)
- **Compatibility**: Version field for forward compatibility (future)

## Testing

- Unit tests for save/load round-trip
- Property tests for serialization consistency
- Integration tests with trained models

## Future Considerations

- Binary format for production use
- Partial loading for fine-tuning
- Model versioning and migration
- Compression support

## References

- ndarray serde documentation
- serde_json API
- Burn framework persistence patterns
- Rust serialization benchmarks
