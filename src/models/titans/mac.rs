use serde::{Deserialize, Serialize};

/// Memory As Context (MAC) Architecture
///
/// "We treat the memory as a context to the current information."
/// Segment-based approach where memory processes past segment and output is concatenated
/// with current segment input to attention.
#[derive(Serialize, Deserialize, Debug)]
pub struct TitansMAC {
    // TODO: Contain the Core branch (Attention).
    // TODO: Contain the Long-term Memory branch (NeuralMemory).
    // TODO: Contain Persistent Memory parameters.
}

impl TitansMAC {
    // TODO: Implement forward logic:
    // 1. Chunk sequence into segments.
    // 2. For segment t:
    //    a. Retrieve h_t from Memory using input context as query.
    //    b. Concatenate [Persistent | h_t | Segment_t].
    //    c. Pass to Attention.
    //    d. Update Memory using Attention output.
}
