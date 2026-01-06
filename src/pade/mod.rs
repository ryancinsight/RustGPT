#![doc = include_str!("doc.md")]

pub mod api;
pub mod exp;

pub use api::{exp, ExpScalar};

#[allow(deprecated)]
pub use api::{exp_f32, exp_f64};
pub use exp::{PadeExp, PrecisionLevel};
