use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Serialization error: {source}")]
    Serialization {
        #[from]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Training error: {message}")]
    Training { message: String },

    #[error("Inference error: {message}")]
    Inference { message: String },

    #[error("Tokenization error: {message}")]
    Tokenization { message: String },

    #[error("Dataset loading error: {source}")]
    DatasetLoad {
        #[from]
        source: std::io::Error,
    },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Gradient computation error: {message}")]
    GradientError { message: String },

    #[error("Shape mismatch: expected {expected:?}, actual {actual:?}. {message}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        message: String,
    },

    #[error("Generic error: {0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;
