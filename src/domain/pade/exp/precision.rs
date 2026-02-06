/// Defines hierarchical accuracy requirements for different computational domains,
/// enabling optimal performance-precision tradeoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionLevel {
    /// Quantum precision: < 1e-18 relative error
    QUANTUM,

    /// Sub-atomic precision: < 1e-17 relative error
    SUBATOMIC,

    /// Atomic precision: < 1e-15 relative error
    ATOMIC,

    /// Molecular precision: < 1e-12 relative error
    MOLECULAR,

    /// Macroscopic precision: < 1e-10 relative error
    MACROSCOPIC,
}
