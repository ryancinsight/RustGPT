#[path = "../richards_curve.rs"]
mod impl_;

pub use impl_::{RichardsCurve, WeightsIter};

/// Internal numerics used by sibling richards submodules.
///
/// Kept `pub(crate)` so they don't leak outside the crate.
pub(crate) mod numerics {
	pub(crate) use super::impl_::{exp_f32_richards, softplus_f32_richards};
}
