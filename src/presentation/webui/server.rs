//! HTTP server implementation for the Web UI
//!
//! Sets up and runs the axum HTTP server with graceful shutdown support

use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn};

use crate::infrastructure::persistence::model_storage::ModelStorage;

use super::{
    WebUiError, WebUiResult,
    config::WebUiConfig,
    routes::{create_router, create_router_with_auth},
    state::AppState,
};

/// Run the web UI server
///
/// This function starts the HTTP server and blocks until a shutdown signal is received.
/// It supports graceful shutdown on SIGINT or SIGTERM.
///
/// # Arguments
///
/// * `config` - Web UI configuration
/// * `storage` - Model storage backend for loading/saving models
///
/// # Example
///
/// ```no_run
/// use llm::presentation::webui::{WebUiConfig, run_server};
/// use llm::infrastructure::persistence::model_storage::FileModelStorage;
///
/// async fn start_server() {
///     let config = WebUiConfig::with_host_port("0.0.0.0", 8080);
///     let storage = FileModelStorage::new("models");
///     
///     run_server(config, storage).await.unwrap();
/// }
/// ```
pub async fn run_server(
    config: WebUiConfig,
    storage: impl ModelStorage + 'static,
) -> WebUiResult<()> {
    let addr = config.socket_addr();
    let state = AppState::new(config.clone(), Arc::new(storage));

    // Create router with or without authentication
    let app = if config.auth_enabled {
        if let Some(api_key) = &config.api_key {
            info!("Web UI server starting with authentication enabled");
            create_router_with_auth(state, &config, api_key.clone())
        } else {
            warn!("Auth enabled but no API key provided, running without authentication");
            create_router(state, &config)
        }
    } else {
        info!("Web UI server starting without authentication");
        create_router(state, &config)
    };

    // Log configuration
    info!("Web UI Configuration:");
    info!("  Address: {}", addr);
    info!(
        "  CORS: {}",
        if config.cors_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    info!(
        "  WebSocket: {}",
        if config.websocket_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    info!("  Max body size: {} MB", config.max_body_size / 1024 / 1024);
    info!("  Model directory: {}", config.model_dir);

    // Bind to address
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| WebUiError::Server(format!("Failed to bind to {}: {}", addr, e)))?;

    info!("Web UI server listening on http://{}", addr);

    // Start server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| WebUiError::Server(format!("Server error: {}", e)))?;

    info!("Web UI server shut down gracefully");

    Ok(())
}

/// Run server with custom state (for testing or advanced use cases)
pub async fn run_server_with_state(config: WebUiConfig, state: AppState) -> WebUiResult<()> {
    let addr = config.socket_addr();

    let app = if config.auth_enabled {
        if let Some(api_key) = &config.api_key {
            create_router_with_auth(state, &config, api_key.clone())
        } else {
            create_router(state, &config)
        }
    } else {
        create_router(state, &config)
    };

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| WebUiError::Server(format!("Failed to bind to {}: {}", addr, e)))?;

    info!("Web UI server listening on http://{}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| WebUiError::Server(format!("Server error: {}", e)))?;

    Ok(())
}

/// Graceful shutdown signal handler
///
/// Waits for SIGINT (Ctrl+C) or SIGTERM signals to initiate shutdown
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, shutting down gracefully...");
        }
        _ = terminate => {
            info!("Received SIGTERM, shutting down gracefully...");
        }
    }
}

/// Server builder for more flexible configuration
pub struct ServerBuilder {
    config: WebUiConfig,
    state: Option<AppState>,
    middleware: Vec<Box<dyn Fn(axum::Router) -> axum::Router + Send + Sync>>,
}

impl ServerBuilder {
    /// Create a new server builder with default configuration
    pub fn new(config: WebUiConfig) -> Self {
        Self {
            config,
            state: None,
            middleware: Vec::new(),
        }
    }

    /// Set custom application state
    pub fn with_state(mut self, state: AppState) -> Self {
        self.state = Some(state);
        self
    }

    /// Add custom middleware
    pub fn with_middleware<F>(mut self, middleware: F) -> Self
    where
        F: Fn(axum::Router) -> axum::Router + Send + Sync + 'static,
    {
        self.middleware.push(Box::new(middleware));
        self
    }

    /// Build and run the server
    pub async fn run(self, storage: impl ModelStorage + 'static) -> WebUiResult<()> {
        let addr = self.config.socket_addr();
        let state = self
            .state
            .unwrap_or_else(|| AppState::new(self.config.clone(), Arc::new(storage)));

        let mut app = create_router(state, &self.config);

        // Apply custom middleware
        for mw in self.middleware {
            app = mw(app);
        }

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| WebUiError::Server(format!("Failed to bind to {}: {}", addr, e)))?;

        info!("Web UI server listening on http://{}", addr);

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| WebUiError::Server(format!("Server error: {}", e)))?;

        Ok(())
    }
}
