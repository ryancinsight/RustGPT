pub mod gating;
pub mod metrics;
pub mod moe;
pub mod moh;
pub mod routing;
pub mod threshold;

// Re-export shared gating types
pub use gating::{GatingConfig, GatingStrategy, select_top_k_components};
// Re-export shared metrics
pub use metrics::MixtureMetrics;
// Re-export MoE types for convenience
pub use moe::{
    ExpertRouter, ExpertRouterConfig, ExpertRouterImpl, ExpertSelector, MixtureOfExperts,
    RichardsExpert,
};
// Re-export MoH types for convenience
pub use moh::{HeadRouter, HeadSelectionConfig, HeadSelectionStrategy};
// Re-export shared routing types
pub use routing::{
    Router, RoutingConfig, RoutingResult, SelectionAlgorithm, apply_selection_algorithm,
    compute_avg_active_components, compute_routing_entropy,
};
// Re-export shared threshold predictor
pub use threshold::ThresholdPredictor;
