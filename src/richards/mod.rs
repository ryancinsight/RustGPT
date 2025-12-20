use serde::{Deserialize, Serialize};

pub mod richards_act;
pub mod richards_curve;
pub mod richards_gate;
pub mod richards_glu;
pub mod richards_norm;

// Keep the root `richards` namespace tight: re-export only the primary public types.
pub use self::{
    richards_act::{RichardsActivation, RichardsAttention},
    richards_curve::{RichardsCurve, WeightsIter},
    richards_gate::RichardsGate,
    richards_glu::RichardsGlu,
    richards_norm::RichardsNorm,
};

/// Variant types for Richards curve initialization and constraints
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum Variant {
    /// Standard sigmoid: σ(x), with output_gain=1, output_bias=0 fixed
    Sigmoid,
    /// Hyperbolic tangent approximation: 2σ(2x) - 1, with output_gain=1, output_bias=0 fixed
    Tanh,
    /// Gompertz curve: ν clamped low (e.g., 0.01), with output_gain=1, output_bias=0 fixed
    Gompertz,
    /// Adaptive normalization with running statistics tracking
    Adaptive,
    /// Polynomial input transformation before Richards activation
    Polynomial,
    /// No constraints, all parameters learnable including output_gain, output_bias
    None,
}
