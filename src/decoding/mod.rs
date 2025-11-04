/// Decoding strategies for language model inference
///
/// This module provides various decoding algorithms for generating text from language models:
/// - Speculative decoding: Fast parallel decoding using single-model speculation
/// - Speculative beam search: Combines beam search quality with speculative speed (zero overhead)
/// - Greedy decoding: Simple argmax token selection
/// - Beam search: Multi-hypothesis search for higher quality generation

pub mod speculative;
pub mod speculative_beam;
pub mod greedy;
pub mod beam_search;

// Re-export main types
pub use speculative::SpeculativeDecoder;
pub use speculative_beam::SpeculativeBeamDecoder;
pub use greedy::GreedyDecoder;
pub use beam_search::BeamSearchDecoder;
