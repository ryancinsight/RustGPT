pub mod richards_act;
pub mod richards_curve;
pub mod richards_norm;

pub use self::richards_act::*;
pub use self::richards_curve::{RichardsCurve, WeightsIter};
pub use self::richards_norm::*;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use crate::adam::Adam;
use rayon::prelude::*;



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
