//! API route definitions for the Web UI
//!
//! Defines all HTTP routes and their handlers using axum's router

use axum::{
    middleware,
    response::IntoResponse,
    routing::{delete, get, post},
    Router,
};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};

use super::{
    config::WebUiConfig,
    handlers::*,
    state::AppState,
};

/// Create the main router with all API routes
pub fn create_router(state: AppState, config: &WebUiConfig) -> Router {
    let cors = if config.cors_enabled {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        CorsLayer::new()
    };

    Router::new()
        // Health check (no auth required)
        .route("/health", get(health_check))
        // Static files for web UI
        .route("/", get(serve_static))
        .route("/ui/*path", get(serve_static))
        // Add state for static routes
        .with_state(state.clone())
        // API v1 routes
        .merge(api_v1_routes(state))
        // Global middleware
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}

/// API v1 routes
fn api_v1_routes(state: AppState) -> Router {
    Router::new()
        // API documentation
        .route("/v1/docs", get(api_docs))
        // Chat completions (OpenAI-compatible)
        .route("/v1/chat/completions", post(create_chat_completion))
        // Text completions (legacy)
        .route("/v1/completions", post(create_completion))
        // Models
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model_id", get(get_model))
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/unload", post(unload_model))
        // Server status
        .route("/v1/status", get(get_status))
        // Conversations
        .route("/v1/conversations", get(list_conversations))
        .route("/v1/conversations/:id", get(get_conversation))
        .route("/v1/conversations/:id", delete(delete_conversation))
        // Add state
        .with_state(state)
}

/// Create router with authentication middleware
pub fn create_router_with_auth(
    state: AppState,
    config: &WebUiConfig,
    api_key: String,
) -> Router {
    let public_routes = Router::new()
        .route("/health", get(health_check))
        .route("/", get(serve_static))
        .route("/ui/*path", get(serve_static))
        .with_state(state.clone());

    let protected_routes = api_v1_routes(state.clone()).layer(middleware::from_fn_with_state(
        api_key.clone(),
        auth_middleware,
    ));

    let cors = if config.cors_enabled {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        CorsLayer::new()
    };

    public_routes
        .merge(protected_routes)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}

/// Authentication middleware
async fn auth_middleware(
    axum::extract::State(expected_key): axum::extract::State<String>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> impl axum::response::IntoResponse {
    use axum::http::{header, StatusCode};

    // Check for Authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    let is_valid = match auth_header {
        Some(header) => {
            // Support "Bearer <token>" format
            let token = header.strip_prefix("Bearer ").unwrap_or(header);
            token == expected_key
        }
        None => {
            // Also check query parameter for API key
            false
        }
    };

    if is_valid {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({
                "error": {
                    "message": "Invalid or missing API key",
                    "type": "authentication_error"
                }
            })),
        )
            .into_response()
    }
}

/// API documentation endpoint handler
pub async fn api_docs() -> impl axum::response::IntoResponse {
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/markdown")],
        include_str!("API_DOCS.md"),
    )
}
