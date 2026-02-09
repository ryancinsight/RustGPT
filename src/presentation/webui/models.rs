//! Request/Response models for the Web UI API
//!
//! Defines the data structures used in API requests and responses
//! for type-safe JSON serialization/deserialization

use serde::{Deserialize, Serialize};

// =============================================================================
// Chat Completion Models
// =============================================================================

/// Chat completion request
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model name to use for completion
    pub model: Option<String>,

    /// List of messages in the conversation
    pub messages: Vec<ChatMessage>,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Repetition penalty
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Stop sequences
    #[serde(default)]
    pub stop: Vec<String>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// User identifier for tracking
    #[serde(default)]
    pub user: Option<String>,

    /// Conversation session ID for maintaining context
    #[serde(default)]
    pub conversation_id: Option<String>,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role: "system", "user", or "assistant"
    pub role: String,

    /// Message content
    pub content: String,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    /// Unique identifier for this completion
    pub id: String,

    /// Object type (always "chat.completion")
    pub object: String,

    /// Creation timestamp
    pub created: i64,

    /// Model used for completion
    pub model: String,

    /// List of completion choices
    pub choices: Vec<ChatCompletionChoice>,

    /// Token usage information
    pub usage: TokenUsage,

    /// Generation statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<GenerationStats>,
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    /// Index of this choice
    pub index: usize,

    /// The generated message
    pub message: ChatMessage,

    /// Finish reason: "stop", "length", or "error"
    pub finish_reason: String,
}

/// Streaming chat completion chunk
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    /// Unique identifier for this completion
    pub id: String,

    /// Object type (always "chat.completion.chunk")
    pub object: String,

    /// Creation timestamp
    pub created: i64,

    /// Model used for completion
    pub model: String,

    /// List of choices (usually one in streaming mode)
    pub choices: Vec<ChatCompletionChunkChoice>,
}

/// Streaming chat completion choice
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkChoice {
    /// Index of this choice
    pub index: usize,

    /// Delta content (incremental)
    pub delta: ChatMessageDelta,

    /// Finish reason if complete
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Chat message delta for streaming
#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatMessageDelta {
    /// Role (only present in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Content delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// =============================================================================
// Token Usage and Stats
// =============================================================================

/// Token usage information
#[derive(Debug, Clone, Serialize)]
pub struct TokenUsage {
    /// Tokens in the prompt
    pub prompt_tokens: usize,

    /// Tokens in the completion
    pub completion_tokens: usize,

    /// Total tokens
    pub total_tokens: usize,
}

/// Generation statistics
#[derive(Debug, Clone, Serialize)]
pub struct GenerationStats {
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: f64,

    /// Total generation time in milliseconds
    pub total_time_ms: f64,

    /// Tokens per second
    pub tokens_per_second: f64,
}

// =============================================================================
// Model Management Models
// =============================================================================

/// Model information response
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfoResponse {
    /// Model identifier
    pub id: String,

    /// Model name
    pub name: String,

    /// Model architecture type
    pub architecture: String,

    /// Model configuration summary
    pub config: ModelConfigSummary,

    /// Model file path
    pub path: String,

    /// Model size in bytes
    pub size_bytes: u64,

    /// Model size in human-readable format
    pub size_human: String,

    /// Creation timestamp
    pub created_at: Option<String>,

    /// Model is currently loaded
    pub is_loaded: bool,
}

/// Model configuration summary
#[derive(Debug, Clone, Serialize)]
pub struct ModelConfigSummary {
    /// Embedding dimension
    pub embedding_dim: usize,

    /// Hidden dimension
    pub hidden_dim: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,
}

/// List models response
#[derive(Debug, Clone, Serialize)]
pub struct ListModelsResponse {
    /// List of available models
    pub data: Vec<ModelInfoResponse>,

    /// Total number of models
    pub total: usize,
}

/// Load model request
#[derive(Debug, Clone, Deserialize)]
pub struct LoadModelRequest {
    /// Model name or path to load
    pub model: String,
}

/// Load model response
#[derive(Debug, Clone, Serialize)]
pub struct LoadModelResponse {
    /// Success status
    pub success: bool,

    /// Model information
    pub model: ModelInfoResponse,

    /// Load time in milliseconds
    pub load_time_ms: u64,
}

/// Unload model response
#[derive(Debug, Clone, Serialize)]
pub struct UnloadModelResponse {
    /// Success status
    pub success: bool,

    /// Message
    pub message: String,
}

// =============================================================================
// Text Completion Models
// =============================================================================

/// Text completion request (legacy style)
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model name to use
    pub model: Option<String>,

    /// Prompt text
    pub prompt: String,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: usize,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences
    #[serde(default)]
    pub stop: Vec<String>,
}

/// Text completion response
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    /// Unique identifier
    pub id: String,

    /// Object type
    pub object: String,

    /// Creation timestamp
    pub created: i64,

    /// Model used
    pub model: String,

    /// Completion choices
    pub choices: Vec<CompletionChoice>,

    /// Token usage
    pub usage: TokenUsage,
}

/// Text completion choice
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    /// Index
    pub index: usize,

    /// Generated text
    pub text: String,

    /// Logprobs (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,

    /// Finish reason
    pub finish_reason: String,
}

// =============================================================================
// Server Status Models
// =============================================================================

/// Server status response
#[derive(Debug, Clone, Serialize)]
pub struct ServerStatusResponse {
    /// Server status
    pub status: String,

    /// Server version
    pub version: String,

    /// Current model loaded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_model: Option<String>,

    /// Server uptime in seconds
    pub uptime_seconds: u64,

    /// Request statistics
    pub stats: ServerStats,
}

/// Server statistics
#[derive(Debug, Clone, Serialize)]
pub struct ServerStats {
    /// Total requests
    pub total_requests: u64,

    /// Successful requests
    pub successful_requests: u64,

    /// Failed requests
    pub failed_requests: u64,

    /// Requests per second
    pub requests_per_second: f64,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// Total tokens generated
    pub total_tokens_generated: u64,

    /// Success rate percentage
    pub success_rate: f64,
}

/// Health check response
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    /// Health status
    pub status: String,

    /// Current timestamp
    pub timestamp: i64,
}

// =============================================================================
// Conversation Models
// =============================================================================

/// Conversation response
#[derive(Debug, Clone, Serialize)]
pub struct ConversationResponse {
    /// Conversation ID
    pub id: String,

    /// Created timestamp
    pub created_at: String,

    /// Last activity timestamp
    pub last_activity: String,

    /// Number of messages
    pub message_count: usize,

    /// Model name
    pub model: String,
}

/// Conversation detail response
#[derive(Debug, Clone, Serialize)]
pub struct ConversationDetailResponse {
    /// Conversation ID
    pub id: String,

    /// Created timestamp
    pub created_at: String,

    /// Last activity timestamp
    pub last_activity: String,

    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,

    /// Model name
    pub model: String,
}

/// List conversations response
#[derive(Debug, Clone, Serialize)]
pub struct ListConversationsResponse {
    /// List of conversations
    pub data: Vec<ConversationResponse>,

    /// Total number of conversations
    pub total: usize,
}

// =============================================================================
// Error Models
// =============================================================================

/// API error response
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// Error information
    pub error: ErrorDetail,
}

/// Error detail
#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,

    /// Error type/code
    #[serde(rename = "type")]
    pub error_type: String,

    /// Error code (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

// =============================================================================
// Default Value Functions
// =============================================================================

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

fn default_top_k() -> usize {
    40
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_n() -> usize {
    1
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Format bytes to human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Generate a unique ID
pub fn generate_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let counter = COUNTER.fetch_add(1, Ordering::SeqCst);

    format!("{}-{}-{:06x}", prefix, timestamp, counter)
}

/// Get current timestamp
pub fn current_timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}
