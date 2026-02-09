//! HTTP request handlers for the Web UI API
//!
//! Implements the business logic for all API endpoints including:
//! - Chat completions (OpenAI-compatible)
//! - Text completions
//! - Model management
//! - Server status and health checks

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Sse},
    Extension,
};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;

use super::{
    models::*,
    state::{AppState, Message},
    WebUiError, WebUiResult,
};

/// Shared model inference engine extension
#[derive(Clone)]
pub struct InferenceEngine;

// =============================================================================
// Chat Completions
// =============================================================================

/// POST /v1/chat/completions
/// Create a chat completion (OpenAI-compatible)
pub async fn create_chat_completion(
    State(state): State<AppState>,
    Extension(_engine): Extension<Arc<InferenceEngine>>,
    Json(request): Json<ChatCompletionRequest>,
) -> WebUiResult<impl IntoResponse> {
    let start_time = Instant::now();

    // Get or create conversation
    let conversation_id = request
        .conversation_id
        .clone()
        .unwrap_or_else(|| generate_id("conv"));

    let current_model_opt = state.current_model().await;
    let model_name = match request.model.clone() {
        Some(name) => name,
        None => match current_model_opt {
            Some(m) => m.name,
            None => return Err(WebUiError::InvalidConfig("No model specified or loaded".to_string())),
        },
    };

    let _conversation = state
        .get_or_create_conversation(&conversation_id, &model_name)
        .await;

    // Add user messages to conversation history
    for msg in &request.messages {
        if msg.role == "user" {
            state
                .add_message(
                    &conversation_id,
                    Message {
                        role: msg.role.clone(),
                        content: msg.content.clone(),
                        timestamp: chrono::Utc::now(),
                    },
                )
                .await;
        }
    }

    if request.stream {
        // Streaming response
        let stream = create_chat_completion_stream(
            state,
            request,
            conversation_id,
            model_name,
            start_time,
        )
        .await?;

        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming response
        let response = generate_chat_completion(
            &state,
            &request,
            &conversation_id,
            &model_name,
            start_time,
        )
        .await?;

        // Add assistant response to conversation
        if let Some(choice) = response.choices.first() {
            state
                .add_message(
                    &conversation_id,
                    Message {
                        role: "assistant".to_string(),
                        content: choice.message.content.clone(),
                        timestamp: chrono::Utc::now(),
                    },
                )
                .await;
        }

        // Record success metrics
        let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        state
            .record_success(response.usage.completion_tokens as u64, latency_ms)
            .await;

        Ok((StatusCode::OK, Json(response)).into_response())
    }
}

/// Generate a chat completion response
async fn generate_chat_completion(
    _state: &AppState,
    request: &ChatCompletionRequest,
    _conversation_id: &str,
    model_name: &str,
    start_time: Instant,
) -> WebUiResult<ChatCompletionResponse> {
    // TODO: Integrate with actual inference engine
    // For now, return a mock response

    let prompt_tokens: usize = request
        .messages
        .iter()
        .map(|m| m.content.split_whitespace().count())
        .sum();

    // Simulate generation with a simple echo response
    let last_user_message = request
        .messages
        .iter()
        .rfind(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let generated_content = format!(
        "This is a mock response. You said: {}\n\n[Model: {} | Temp: {:.1} | Max tokens: {}]",
        last_user_message, model_name, request.temperature, request.max_tokens
    );

    let completion_tokens = generated_content.split_whitespace().count();
    let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok(ChatCompletionResponse {
        id: generate_id("chatcmpl"),
        object: "chat.completion".to_string(),
        created: current_timestamp(),
        model: model_name.to_string(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: generated_content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        stats: Some(GenerationStats {
            time_to_first_token_ms: total_time_ms * 0.1, // Mock TTFB
            total_time_ms,
            tokens_per_second: (completion_tokens as f64) / (total_time_ms / 1000.0),
        }),
    })
}

/// Create a streaming chat completion
async fn create_chat_completion_stream(
    _state: AppState,
    _request: ChatCompletionRequest,
    _conversation_id: String,
    _model_name: String,
    _start_time: Instant,
) -> WebUiResult<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>
{
    // TODO: Implement actual streaming with inference engine
    // For now, return a mock stream

    use axum::response::sse::Event;
    use futures::stream;

    let stream = stream::iter(vec![
        Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: generate_id("chatcmpl"),
                object: "chat.completion.chunk".to_string(),
                created: current_timestamp(),
                model: _model_name.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            })
            .unwrap(),
        )),
        Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: generate_id("chatcmpl"),
                object: "chat.completion.chunk".to_string(),
                created: current_timestamp(),
                model: _model_name.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatMessageDelta {
                        role: None,
                        content: Some("This ".to_string()),
                    },
                    finish_reason: None,
                }],
            })
            .unwrap(),
        )),
        Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: generate_id("chatcmpl"),
                object: "chat.completion.chunk".to_string(),
                created: current_timestamp(),
                model: _model_name.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatMessageDelta {
                        role: None,
                        content: Some("is ".to_string()),
                    },
                    finish_reason: None,
                }],
            })
            .unwrap(),
        )),
        Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: generate_id("chatcmpl"),
                object: "chat.completion.chunk".to_string(),
                created: current_timestamp(),
                model: _model_name,
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatMessageDelta {
                        role: None,
                        content: Some("a mock streaming response.".to_string()),
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            })
            .unwrap(),
        )),
    ]);

    Ok(stream)
}

// =============================================================================
// Text Completions
// =============================================================================

/// POST /v1/completions
/// Create a text completion (legacy OpenAI-compatible)
pub async fn create_completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> WebUiResult<impl IntoResponse> {
    let start_time = Instant::now();

    let current_model_opt = state.current_model().await;
    let model_name = match request.model.clone() {
        Some(name) => name,
        None => match current_model_opt {
            Some(m) => m.name,
            None => "default".to_string(),
        },
    };

    let prompt_tokens = request.prompt.split_whitespace().count();

    // Simulate generation
    let generated_text = format!(
        "This is a mock completion response.\n\nPrompt preview: {}...\n[Model: {} | Temp: {:.1}]",
        &request.prompt[..request.prompt.len().min(50)],
        model_name,
        request.temperature
    );

    let completion_tokens = generated_text.split_whitespace().count();
    let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // Record metrics
    state
        .record_success(completion_tokens as u64, latency_ms)
        .await;

    let response = CompletionResponse {
        id: generate_id("cmpl"),
        object: "text_completion".to_string(),
        created: current_timestamp(),
        model: model_name,
        choices: vec![CompletionChoice {
            index: 0,
            text: generated_text,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok((StatusCode::OK, Json(response)))
}

// =============================================================================
// Model Management
// =============================================================================

/// GET /v1/models
/// List available models
pub async fn list_models(State(state): State<AppState>) -> WebUiResult<impl IntoResponse> {
    let current_model = state.current_model().await;

    // TODO: Scan model directory for available models
    // For now, return current model if loaded
    let models = if let Some(current) = current_model {
        vec![ModelInfoResponse {
            id: current.name.clone(),
            name: current.name.clone(),
            architecture: format!("{:?}", current.config.architecture),
            config: ModelConfigSummary {
                embedding_dim: current.config.embedding_dim,
                hidden_dim: current.config.hidden_dim,
                num_layers: current.config.num_layers,
                num_heads: current.config.get_num_heads(),
                max_seq_len: current.config.max_seq_len,
            },
            path: current.path.clone(),
            size_bytes: current.size_bytes,
            size_human: format_bytes(current.size_bytes),
            created_at: Some(current.loaded_at.to_rfc3339()),
            is_loaded: true,
        }]
    } else {
        vec![]
    };

    Ok((
        StatusCode::OK,
        Json(ListModelsResponse {
            total: models.len(),
            data: models,
        }),
    ))
}

/// GET /v1/models/:model_id
/// Get model information
pub async fn get_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> WebUiResult<impl IntoResponse> {
    let current_model = state.current_model().await;

    if let Some(model) = current_model {
        if model.name == model_id {
            return Ok((
                StatusCode::OK,
                Json(ModelInfoResponse {
                    id: model.name.clone(),
                    name: model.name.clone(),
                    architecture: format!("{:?}", model.config.architecture),
                    config: ModelConfigSummary {
                        embedding_dim: model.config.embedding_dim,
                        hidden_dim: model.config.hidden_dim,
                        num_layers: model.config.num_layers,
                        num_heads: model.config.get_num_heads(),
                        max_seq_len: model.config.max_seq_len,
                    },
                    path: model.path.clone(),
                    size_bytes: model.size_bytes,
                    size_human: format_bytes(model.size_bytes),
                    created_at: Some(model.loaded_at.to_rfc3339()),
                    is_loaded: true,
                }),
            ));
        }
    }

    Err(WebUiError::ModelNotFound(model_id))
}

/// POST /v1/models/load
/// Load a model
pub async fn load_model(
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> WebUiResult<impl IntoResponse> {
    let start_time = Instant::now();

    // TODO: Actually load the model using the inference engine
    // For now, simulate loading

    let model_info = super::state::ModelInfo {
        name: request.model.clone(),
        config: crate::domain::models::config::ModelConfig::default(),
        path: format!("{}/{}.bin", state.config.model_dir, request.model),
        loaded_at: chrono::Utc::now(),
        size_bytes: 100_000_000, // Mock 100MB
    };

    state.set_model(Some(model_info.clone())).await;

    let load_time_ms = start_time.elapsed().as_millis() as u64;

    Ok((
        StatusCode::OK,
        Json(LoadModelResponse {
            success: true,
            model: ModelInfoResponse {
                id: model_info.name.clone(),
                name: model_info.name.clone(),
                architecture: format!("{:?}", model_info.config.architecture),
                config: ModelConfigSummary {
                    embedding_dim: model_info.config.embedding_dim,
                    hidden_dim: model_info.config.hidden_dim,
                    num_layers: model_info.config.num_layers,
                    num_heads: model_info.config.get_num_heads(),
                    max_seq_len: model_info.config.max_seq_len,
                },
                path: model_info.path.clone(),
                size_bytes: model_info.size_bytes,
                size_human: format_bytes(model_info.size_bytes),
                created_at: Some(model_info.loaded_at.to_rfc3339()),
                is_loaded: true,
            },
            load_time_ms,
        }),
    ))
}

/// POST /v1/models/unload
/// Unload the current model
pub async fn unload_model(State(state): State<AppState>) -> WebUiResult<impl IntoResponse> {
    state.set_model(None).await;

    Ok((
        StatusCode::OK,
        Json(UnloadModelResponse {
            success: true,
            message: "Model unloaded successfully".to_string(),
        }),
    ))
}

// =============================================================================
// Server Status
// =============================================================================

/// GET /v1/status
/// Get server status and statistics
pub async fn get_status(State(state): State<AppState>) -> WebUiResult<impl IntoResponse> {
    let uptime = state.uptime().await;
    let stats = state.stats().await;
    let current_model = state.current_model().await;

    let uptime_secs = uptime.as_secs();

    Ok((
        StatusCode::OK,
        Json(ServerStatusResponse {
            status: "healthy".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            current_model: current_model.map(|m| m.name),
            uptime_seconds: uptime_secs,
            stats: ServerStats {
                total_requests: stats.total_requests,
                successful_requests: stats.successful_requests,
                failed_requests: stats.failed_requests,
                requests_per_second: stats.requests_per_second(uptime_secs as f64),
                avg_latency_ms: stats.avg_latency_ms,
                total_tokens_generated: stats.total_tokens_generated,
                success_rate: stats.success_rate(),
            },
        }),
    ))
}

/// GET /health
/// Simple health check endpoint
pub async fn health_check() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "healthy".to_string(),
            timestamp: current_timestamp(),
        }),
    )
}

// =============================================================================
// Conversations
// =============================================================================

/// GET /v1/conversations
/// List all conversations
pub async fn list_conversations(State(state): State<AppState>) -> WebUiResult<impl IntoResponse> {
    let conversations = state.list_conversations().await;

    let data: Vec<ConversationResponse> = conversations
        .into_iter()
        .map(|c| ConversationResponse {
            id: c.id,
            created_at: c.created_at.to_rfc3339(),
            last_activity: c.last_activity.to_rfc3339(),
            message_count: c.messages.len(),
            model: c.model_name,
        })
        .collect();

    let total = data.len();

    Ok((StatusCode::OK, Json(ListConversationsResponse { data, total })))
}

/// GET /v1/conversations/:id
/// Get conversation details
pub async fn get_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> WebUiResult<impl IntoResponse> {
    let conversation = state
        .get_conversation(&id)
        .await
        .ok_or_else(|| WebUiError::InvalidConfig(format!("Conversation not found: {}", id)))?;

    Ok((
        StatusCode::OK,
        Json(ConversationDetailResponse {
            id: conversation.id,
            created_at: conversation.created_at.to_rfc3339(),
            last_activity: conversation.last_activity.to_rfc3339(),
            messages: conversation
                .messages
                .into_iter()
                .map(|m| ChatMessage {
                    role: m.role,
                    content: m.content,
                })
                .collect(),
            model: conversation.model_name,
        }),
    ))
}

/// DELETE /v1/conversations/:id
/// Delete a conversation
pub async fn delete_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let deleted = state.delete_conversation(&id).await;

    if deleted {
        (StatusCode::NO_CONTENT, ())
    } else {
        (StatusCode::NOT_FOUND, ())
    }
}

// =============================================================================
// Static Files
// =============================================================================

/// Serve static files (HTML, CSS, JS for the web UI)
pub async fn serve_static() -> impl IntoResponse {
    // TODO: Serve actual static files from configured directory
    (
        StatusCode::OK,
        [("Content-Type", "text/html")],
        include_str!("static/index.html"),
    )
}

// =============================================================================
// Error Handling
// =============================================================================

impl IntoResponse for WebUiError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_message) = match &self {
            WebUiError::ModelNotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            WebUiError::InvalidConfig(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            WebUiError::Inference(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            WebUiError::Server(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            WebUiError::Io(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
        };

        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message: error_message,
                error_type: format!("{:?}", self),
                code: Some(format!("{}", status.as_u16())),
            },
        });

        (status, body).into_response()
    }
}

// =============================================================================
// Query Parameters
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
}
