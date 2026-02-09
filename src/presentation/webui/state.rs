//! Application state for the Web UI
//!
//! Manages shared state across all request handlers including:
//! - Loaded model reference
//! - Configuration
//! - Request counters and metrics
//! - Conversation history cache

use crate::domain::models::config::ModelConfig;
use crate::infrastructure::persistence::model_storage::ModelStorage;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::WebUiConfig;

/// Shared application state for the web UI server
#[derive(Clone)]
pub struct AppState {
    /// Inner state protected by locks for thread-safe access
    inner: Arc<RwLock<AppStateInner>>,
    /// Web UI configuration (read-only, cloneable)
    pub config: WebUiConfig,
    /// Model storage backend
    pub storage: Arc<dyn ModelStorage>,
}

/// Internal mutable state
#[derive(Debug)]
struct AppStateInner {
    /// Currently loaded model configuration
    current_model: Option<ModelInfo>,
    /// Request statistics
    stats: RequestStats,
    /// Server start time
    started_at: std::time::Instant,
    /// Active conversation sessions
    conversations: std::collections::HashMap<String, Conversation>,
}

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name/identifier
    pub name: String,
    /// Model configuration
    pub config: ModelConfig,
    /// Path to model file
    pub path: String,
    /// Loaded timestamp
    pub loaded_at: chrono::DateTime<chrono::Utc>,
    /// Model size in bytes
    pub size_bytes: u64,
}

/// Request statistics
#[derive(Debug, Clone, Default)]
pub struct RequestStats {
    /// Total number of requests processed
    pub total_requests: u64,
    /// Number of successful requests
    pub successful_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Total inference tokens generated
    pub total_tokens_generated: u64,
    /// Average inference latency in milliseconds
    pub avg_latency_ms: f64,
}

/// Conversation session
#[derive(Debug, Clone)]
pub struct Conversation {
    /// Session ID
    pub id: String,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Model used for this conversation
    pub model_name: String,
}

/// Chat message
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    /// Message role (user/assistant/system)
    pub role: String,
    /// Message content
    pub content: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl AppState {
    /// Create a new application state
    pub fn new(config: WebUiConfig, storage: Arc<dyn ModelStorage>) -> Self {
        let inner = AppStateInner {
            current_model: None,
            stats: RequestStats::default(),
            started_at: std::time::Instant::now(),
            conversations: std::collections::HashMap::new(),
        };

        Self {
            inner: Arc::new(RwLock::new(inner)),
            config,
            storage,
        }
    }

    /// Get the currently loaded model info
    pub async fn current_model(&self) -> Option<ModelInfo> {
        self.inner.read().await.current_model.clone()
    }

    /// Set the currently loaded model
    pub async fn set_model(&self, model: Option<ModelInfo>) {
        self.inner.write().await.current_model = model;
    }

    /// Get request statistics
    pub async fn stats(&self) -> RequestStats {
        self.inner.read().await.stats.clone()
    }

    /// Record a successful request
    pub async fn record_success(&self, tokens_generated: u64, latency_ms: f64) {
        let mut guard = self.inner.write().await;
        guard.stats.total_requests += 1;
        guard.stats.successful_requests += 1;
        guard.stats.total_tokens_generated += tokens_generated;

        // Update rolling average latency
        let n = guard.stats.successful_requests as f64;
        guard.stats.avg_latency_ms =
            (guard.stats.avg_latency_ms * (n - 1.0) + latency_ms) / n;
    }

    /// Record a failed request
    pub async fn record_failure(&self) {
        let mut guard = self.inner.write().await;
        guard.stats.total_requests += 1;
        guard.stats.failed_requests += 1;
    }

    /// Get server uptime
    pub async fn uptime(&self) -> std::time::Duration {
        self.inner.read().await.started_at.elapsed()
    }

    /// Get or create a conversation
    pub async fn get_or_create_conversation(
        &self,
        id: &str,
        model_name: &str,
    ) -> Conversation {
        let mut guard = self.inner.write().await;

        if let Some(conv) = guard.conversations.get(id) {
            let mut conv = conv.clone();
            conv.last_activity = chrono::Utc::now();
            guard.conversations.insert(id.to_string(), conv.clone());
            conv
        } else {
            let conv = Conversation {
                id: id.to_string(),
                created_at: chrono::Utc::now(),
                last_activity: chrono::Utc::now(),
                messages: Vec::new(),
                model_name: model_name.to_string(),
            };
            guard.conversations.insert(id.to_string(), conv.clone());
            conv
        }
    }

    /// Add a message to a conversation
    pub async fn add_message(&self, conversation_id: &str, message: Message) {
        let mut guard = self.inner.write().await;
        if let Some(conv) = guard.conversations.get_mut(conversation_id) {
            conv.messages.push(message);
            conv.last_activity = chrono::Utc::now();
        }
    }

    /// Get conversation by ID
    pub async fn get_conversation(&self, id: &str) -> Option<Conversation> {
        self.inner.read().await.conversations.get(id).cloned()
    }

    /// List all conversations
    pub async fn list_conversations(&self) -> Vec<Conversation> {
        self.inner
            .read()
            .await
            .conversations
            .values()
            .cloned()
            .collect()
    }

    /// Delete a conversation
    pub async fn delete_conversation(&self, id: &str) -> bool {
        let mut guard = self.inner.write().await;
        guard.conversations.remove(id).is_some()
    }

    /// Cleanup old conversations (older than specified duration)
    pub async fn cleanup_old_conversations(&self, max_age: std::time::Duration) {
        let cutoff = chrono::Utc::now() - chrono::Duration::from_std(max_age).unwrap_or_default();
        let mut guard = self.inner.write().await;
        guard
            .conversations
            .retain(|_, conv| conv.last_activity > cutoff);
    }
}

impl RequestStats {
    /// Calculate requests per second
    pub fn requests_per_second(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs > 0.0 {
            self.total_requests as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    /// Calculate success rate percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        } else {
            100.0
        }
    }
}
