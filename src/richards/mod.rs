pub mod act;
pub mod curve;
pub mod gate;
pub mod glu;
pub mod norm;
pub mod scalar;
pub mod types;

// Keep the root `richards` namespace tight: re-export only the primary public types.
pub use self::{
    act::{RichardsActivation, RichardsAttention},
    curve::{RichardsCurve, WeightsIter},
    gate::RichardsGate,
    glu::RichardsGlu,
    norm::RichardsNorm,
    scalar::RichardsScalar,
    types::Variant,
};

// Lightweight crate-wide helpers (generic + f32 wrappers).
pub use self::scalar::{
    dsigmoid, dsigmoid_f32, dsilu, dsilu_f32, dtanh, dtanh_f32, sigmoid, sigmoid_f32, silu,
    silu_f32, tanh, tanh_f32,
};
