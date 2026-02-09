//! Web UI configuration
//!
//! Defines configuration options for the web interface server

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Configuration for the Web UI server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebUiConfig {
    /// Server bind address
    #[serde(default = "default_host")]
    pub host: String,

    /// Server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// Enable CORS for cross-origin requests
    #[serde(default = "default_cors_enabled")]
    pub cors_enabled: bool,

    /// CORS allowed origins (when cors_enabled is true)
    #[serde(default = "default_cors_origins")]
    pub cors_origins: Vec<String>,

    /// Maximum request body size in bytes
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,

    /// Request timeout in seconds
    #[serde(default = "default_request_timeout")]
    pub request_timeout: u64,

    /// Enable request logging
    #[serde(default = "default_logging_enabled")]
    pub logging_enabled: bool,

    /// Static files directory for web assets
    #[serde(default = "default_static_dir")]
    pub static_dir: Option<String>,

    /// Enable WebSocket for real-time updates
    #[serde(default = "default_websocket_enabled")]
    pub websocket_enabled: bool,

    /// Maximum number of concurrent connections
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Model directory path
    #[serde(default = "default_model_dir")]
    pub model_dir: String,

    /// Data directory for conversations and other data
    #[serde(default = "default_data_dir")]
    pub data_dir: Option<String>,

    /// Enable authentication
    #[serde(default = "default_auth_enabled")]
    pub auth_enabled: bool,

    /// API key for authentication (when auth_enabled is true)
    #[serde(default)]
    pub api_key: Option<String>,
}

impl Default for WebUiConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            cors_enabled: default_cors_enabled(),
            cors_origins: default_cors_origins(),
            max_body_size: default_max_body_size(),
            request_timeout: default_request_timeout(),
            logging_enabled: default_logging_enabled(),
            static_dir: default_static_dir(),
            websocket_enabled: default_websocket_enabled(),
            max_connections: default_max_connections(),
            model_dir: default_model_dir(),
            data_dir: default_data_dir(),
            auth_enabled: default_auth_enabled(),
            api_key: None,
        }
    }
}

impl WebUiConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration with custom host and port
    pub fn with_host_port(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            ..Default::default()
        }
    }

    /// Get the socket address from configuration
    pub fn socket_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("Invalid socket address configuration")
    }

    /// Builder method: enable CORS
    pub fn with_cors(mut self, enabled: bool) -> Self {
        self.cors_enabled = enabled;
        self
    }

    /// Builder method: set CORS origins
    pub fn with_cors_origins(mut self, origins: Vec<String>) -> Self {
        self.cors_origins = origins;
        self
    }

    /// Builder method: set static files directory
    pub fn with_static_dir(mut self, dir: impl Into<String>) -> Self {
        self.static_dir = Some(dir.into());
        self
    }

    /// Builder method: enable WebSocket
    pub fn with_websocket(mut self, enabled: bool) -> Self {
        self.websocket_enabled = enabled;
        self
    }

    /// Builder method: set model directory
    pub fn with_model_dir(mut self, dir: impl Into<String>) -> Self {
        self.model_dir = dir.into();
        self
    }

    /// Builder method: set data directory
    pub fn with_data_dir(mut self, dir: impl Into<String>) -> Self {
        self.data_dir = Some(dir.into());
        self
    }

    /// Builder method: enable authentication
    pub fn with_auth(mut self, enabled: bool, api_key: Option<String>) -> Self {
        self.auth_enabled = enabled;
        self.api_key = api_key;
        self
    }
}

// Default value functions for serde
fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_cors_enabled() -> bool {
    true
}

fn default_cors_origins() -> Vec<String> {
    vec!["*".to_string()]
}

fn default_max_body_size() -> usize {
    10 * 1024 * 1024 // 10 MB
}

fn default_request_timeout() -> u64 {
    300 // 5 minutes
}

fn default_logging_enabled() -> bool {
    true
}

fn default_static_dir() -> Option<String> {
    Some("static".to_string())
}

fn default_websocket_enabled() -> bool {
    true
}

fn default_max_connections() -> usize {
    100
}

fn default_model_dir() -> String {
    "models".to_string()
}

fn default_data_dir() -> Option<String> {
    None
}

fn default_auth_enabled() -> bool {
    false
}
