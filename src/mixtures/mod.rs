pub mod gating;
pub mod metrics;
pub mod moh;
pub mod moe;
pub mod routing;
pub mod threshold;

// Re-export shared gating types
pub use gating::{GatingStrategy, GatingConfig, select_top_k_components};
// Re-export shared metrics
pub use metrics::MixtureMetrics;
// Re-export shared routing types
pub use routing::{Router, RoutingConfig, RoutingResult, SelectionAlgorithm, apply_selection_algorithm, compute_routing_entropy, compute_avg_active_components};
// Re-export shared threshold predictor
pub use threshold::ThresholdPredictor;
// Re-export MoH types for convenience
pub use moh::{HeadSelectionStrategy, HeadSelectionConfig, HeadRouter};
// Re-export MoE types for convenience
pub use moe::{ExpertRouter, ExpertRouterConfig, ExpertRouterImpl, ExpertSelector, RichardsExpert, MixtureOfExperts};