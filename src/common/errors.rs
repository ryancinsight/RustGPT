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

    #[error("Backend error: {message}")]
    Backend { message: String },

    #[error("GPU device not found: {message}")]
    GpuDeviceNotFound { message: String },

    #[error("GPU initialization failed: {message}")]
    GpuInitializationError { message: String },

    #[error("GPU memory allocation failed: {message}")]
    GpuMemoryAllocation { message: String },

    #[error("GPU shader compilation failed: {message}")]
    GpuShaderCompilation { message: String },

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    #[error("Generic error: {0}")]
    Generic(String),

    #[error("Dimension mismatch: {message}")]
    DimensionMismatch { message: String },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatchDetailed { expected: String, got: String },

    #[error("Lock error: {message}")]
    Lock { message: String },

    #[error("Invalid state: {message}")]
    InvalidState { message: String },
}

impl From<ndarray::ShapeError> for ModelError {
    fn from(err: ndarray::ShapeError) -> Self {
        ModelError::ShapeMismatch {
            expected: vec![],
            actual: vec![],
            message: format!("{:?}", err),
        }
    }
}

pub type Result<T> = std::result::Result<T, ModelError>;
