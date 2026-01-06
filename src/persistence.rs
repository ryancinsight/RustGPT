// Consolidated: persistence is implemented directly on `LLM`.
//
// The actual versioned container + integrity checks live in `model_persistence.rs`,
// and the public API surface is `LLM::{save, load, save_binary, load_binary, save_json, load_json,
// save_versioned, load_versioned}`.
