mod core;
mod pade_exp;
mod precision;
mod utils;

pub(super) mod analysis;
pub(super) mod approximants;
pub(super) mod array;
pub(super) mod range_reduction;
pub(super) mod simd;

pub use pade_exp::PadeExp;
pub use precision::PrecisionLevel;

#[cfg(test)]
mod tests;
