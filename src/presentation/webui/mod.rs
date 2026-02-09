//! Web UI module for LLM trained models
//!
//! Provides a web-based interface for:
//! - Interactive inference with trained models
//! - Model management (load, save, list)
//! - Configuration management
//! - Real-time monitoring and metrics
//! - Chat interface with conversation history

pub mod config;
pub mod handlers;
pub mod models;
pub mod routes;
pub mod server;
pub mod state;

pub use config::WebUiConfig;
pub use server::run_server;
pub use state::AppState;

use std::net::SocketAddr;
use thiserror::Error;

/// Errors that can occur in the web UI module
#[derive(Error, Debug)]
pub enum WebUiError {
    #[error("Server error: {0}")]
    Server(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type WebUiResult<T> = Result<T, WebUiError>;

/// Default port for the web UI server
pub const DEFAULT_PORT: u16 = 8080;

/// Default host for the web UI server
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// Create a default socket address for the web UI server
pub fn default_socket_addr() -> SocketAddr {
    format!("{}:{}", DEFAULT_HOST, DEFAULT_PORT)
        .parse()
        .expect("Invalid default socket address")
}
