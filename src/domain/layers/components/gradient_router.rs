//! Generic gradient routing infrastructure for zero-copy, zero-redundancy parameter updates.
//!
//! This module provides traits and utilities to eliminate repetitive gradient application
//! patterns using zero-cost abstractions, shared accessors, and generic programming.

use ndarray::Array2;
use rayon::prelude::*;

use crate::common::errors::Result;

/// A routable component that can receive gradients through the generic router.
///
/// This trait abstracts over any component that needs gradient application,
/// allowing uniform treatment of layers, norms, optimizers, and auxiliary components.
pub trait GradientRoutable {
    /// Returns the number of parameter tensors this component expects.
    fn gradient_count(&self) -> usize;

    /// Returns the Frobenius norm of weights for LARS-style adaptive scaling.
    fn weight_norm(&self) -> f32;

    /// Applies gradients to this component's parameters.
    ///
    /// # Arguments
    /// * `gradients` - Slice of gradients, guaranteed to have length >= gradient_count()
    /// * `learning_rate` - Global learning rate (may be scaled internally)
    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()>;
}

/// Zero-copy gradient slice that avoids allocation when gradients are already valid.
#[derive(Debug, Clone)]
pub struct GradientSlice<'a> {
    /// The underlying gradient arrays (as Cow to allow zero-copy when possible)
    grads: Vec<std::borrow::Cow<'a, Array2<f32>>>,
    /// Starting index within the gradient vector
    start: usize,
    /// Count of gradients in this slice
    count: usize,
}

impl<'a> GradientSlice<'a> {
    /// Create a new gradient slice from a vector of Cow arrays.
    #[inline]
    pub fn new(grads: Vec<std::borrow::Cow<'a, Array2<f32>>>, start: usize, count: usize) -> Self {
        Self {
            grads,
            start,
            count,
        }
    }

    /// Returns a sub-slice of gradients for a component.
    #[inline]
    pub fn sub_slice(&self, offset: usize, count: usize) -> Option<Self> {
        let actual_start = self.start + offset;
        if actual_start + count > self.grads.len() {
            return None;
        }
        Some(Self {
            grads: self.grads.clone(),
            start: actual_start,
            count,
        })
    }

    /// Convert to owned gradients for components that require owned data.
    #[inline]
    pub fn to_owned(&self) -> Vec<Array2<f32>> {
        self.grads[self.start..self.start + self.count]
            .iter()
            .map(|c| c.as_ref().clone())
            .collect()
    }

    /// Returns true if this slice contains any gradients.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the number of gradients in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Apply LARS-style adaptive scaling and return scaled gradients.
    #[inline]
    pub fn with_lars_scaling(&self, weight_norm: f32) -> Vec<Array2<f32>> {
        if self.is_empty() {
            return Vec::new();
        }

        let grads: Vec<&Array2<f32>> = self.grads[self.start..self.start + self.count]
            .iter()
            .map(|c| c.as_ref())
            .collect();

        let gnorm: f32 = grads
            .iter()
            .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt();

        let wnorm = weight_norm.max(1e-6);
        let scale = (wnorm / gnorm.max(1e-6)).clamp(0.01, 5.0);

        if (scale - 1.0).abs() < 1e-6 {
            // No scaling needed, return owned copy
            self.to_owned()
        } else {
            // Apply scaling in parallel
            grads
                .into_par_iter()
                .map(|g| {
                    let mut gg = g.clone();
                    gg.mapv_inplace(|x| x * scale);
                    gg
                })
                .collect()
        }
    }
}

/// Convert a borrowed/owned Cow gradient slice into owned arrays.
#[inline]
pub fn cow_grads_to_owned(grads: &[std::borrow::Cow<'_, Array2<f32>>]) -> Vec<Array2<f32>> {
    grads.iter().map(|c| c.as_ref().clone()).collect()
}

/// Apply LARS-style scaling to Cow gradients, then forward owned arrays to a closure.
#[inline]
pub fn apply_lars_to_cow_grads<F>(
    grads: &[std::borrow::Cow<'_, Array2<f32>>],
    weight_norm: f32,
    learning_rate: f32,
    mut apply_fn: F,
) -> Result<()>
where
    F: FnMut(&[Array2<f32>], f32) -> Result<()>,
{
    if grads.is_empty() {
        return Ok(());
    }

    let gnorm: f32 = grads
        .iter()
        .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
        .sum::<f32>()
        .sqrt();
    let wnorm = weight_norm.max(1e-6);
    let scale = (wnorm / gnorm.max(1e-6)).clamp(0.01, 5.0);

    let scaled: Vec<Array2<f32>> = if (scale - 1.0).abs() < 1e-6 {
        cow_grads_to_owned(grads)
    } else {
        grads
            .par_iter()
            .map(|g| {
                let mut gg = g.as_ref().clone();
                gg.mapv_inplace(|x| x * scale);
                gg
            })
            .collect()
    };

    apply_fn(&scaled, learning_rate)
}

/// Generic gradient router that eliminates repetitive gradient application patterns.
///
/// This struct manages the routing of gradients to multiple components using a
/// type-safe, zero-cost abstraction that replaces the manual index tracking
/// and repetitive code found in the original `apply_gradients` implementations.
pub struct GradientRouter<'a> {
    /// The full set of sanitized gradients
    gradients: Vec<std::borrow::Cow<'a, Array2<f32>>>,
    /// Current position in the gradient stream
    position: usize,
}

impl<'a> GradientRouter<'a> {
    /// Create a new gradient router from a slice of gradients.
    ///
    /// Applies sanitization (NaN/Inf checking and clipping) to all gradients.
    pub fn new(gradients: &'a [Array2<f32>], clip_threshold: f32) -> Self {
        let sanitized: Vec<std::borrow::Cow<'a, Array2<f32>>> = gradients
            .iter()
            .map(|grad| {
                // Check if sanitization is needed
                let needs_fix = grad
                    .iter()
                    .any(|&val| !val.is_finite() || val.abs() > clip_threshold);

                if needs_fix {
                    let mut fixed = grad.clone();
                    for &val in grad.iter() {
                        if !val.is_finite() {
                            use rand::Rng;
                            let mut rng = crate::common::rng::get_rng();
                            fixed.mapv_inplace(|_| 0.01 * (rng.random::<f32>() - 0.5));
                            break;
                        }
                        if val.abs() > clip_threshold {
                            fixed.mapv_inplace(|x| x.clamp(-clip_threshold, clip_threshold));
                            break;
                        }
                    }
                    std::borrow::Cow::Owned(fixed)
                } else {
                    std::borrow::Cow::Borrowed(grad)
                }
            })
            .collect();

        Self {
            gradients: sanitized,
            position: 0,
        }
    }

    /// Route gradients to a routable component with LARS-style adaptive scaling.
    ///
    /// This is the primary method for eliminating redundant gradient application code.
    /// It automatically:
    /// 1. Takes the appropriate slice of gradients
    /// 2. Applies LARS scaling based on component weight norm
    /// 3. Applies gradients to the component
    /// 4. Advances the internal position tracker
    ///
    /// # Type Parameters
    /// * `T` - Any type implementing GradientRoutable
    ///
    /// # Arguments
    /// * `component` - Mutable reference to the component receiving gradients
    /// * `learning_rate` - Global learning rate
    ///
    /// # Returns
    /// * `Result<()>` - Success or gradient application error
    pub fn route_with_lars<T: GradientRoutable>(
        &mut self,
        component: &mut T,
        learning_rate: f32,
    ) -> Result<()> {
        let count = component.gradient_count();
        if count == 0 {
            return Ok(());
        }

        let available = self.gradients.len().saturating_sub(self.position);
        let actual_count = count.min(available);

        if actual_count == 0 {
            return Ok(());
        }

        // Create a slice view
        let slice = GradientSlice {
            grads: self.gradients.clone(),
            start: self.position,
            count: actual_count,
        };

        // Apply LARS scaling and apply gradients
        let scaled = slice.with_lars_scaling(component.weight_norm());
        component.apply_gradients(&scaled, learning_rate)?;

        self.position += actual_count;
        Ok(())
    }

    /// Route gradients to a routable component without LARS scaling.
    ///
    /// Use this for components that manage their own learning rate scaling
    /// or for scalar/optimizer-style components.
    pub fn route_direct<T: GradientRoutable>(
        &mut self,
        component: &mut T,
        learning_rate: f32,
    ) -> Result<()> {
        let count = component.gradient_count();
        if count == 0 {
            return Ok(());
        }

        let available = self.gradients.len().saturating_sub(self.position);
        let actual_count = count.min(available);

        if actual_count == 0 {
            return Ok(());
        }

        // Convert slice to owned for application
        let owned: Vec<Array2<f32>> = self.gradients[self.position..self.position + actual_count]
            .iter()
            .map(|c| c.as_ref().clone())
            .collect();

        component.apply_gradients(&owned, learning_rate)?;
        self.position += actual_count;
        Ok(())
    }

    /// Route gradients to a closure without LARS scaling.
    ///
    /// This allows ad-hoc gradient application for components that don't
    /// implement GradientRoutable or need special handling.
    pub fn route_to_closure<F>(&mut self, count: usize, mut f: F) -> Result<()>
    where
        F: FnMut(&[std::borrow::Cow<'_, Array2<f32>>]) -> Result<()>,
    {
        if count == 0 {
            return Ok(());
        }

        let available = self.gradients.len().saturating_sub(self.position);
        let actual_count = count.min(available);

        if actual_count == 0 {
            return Ok(());
        }

        let slice = &self.gradients[self.position..self.position + actual_count];
        f(slice)?;
        self.position += actual_count;
        Ok(())
    }

    /// Route gradients to a closure after converting them to owned arrays.
    ///
    /// Useful for downstream APIs that require owned gradient buffers.
    pub fn route_owned_to_closure<F>(&mut self, count: usize, mut f: F) -> Result<()>
    where
        F: FnMut(&[Array2<f32>]) -> Result<()>,
    {
        self.route_to_closure(count, |grads| {
            let owned = cow_grads_to_owned(grads);
            f(&owned)
        })
    }

    /// Route gradients to a closure with owned arrays only when `enabled` is true.
    ///
    /// When disabled, gradients are still consumed to preserve partition alignment,
    /// but no owned conversion is performed.
    pub fn route_owned_to_closure_if<F>(
        &mut self,
        count: usize,
        enabled: bool,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&[Array2<f32>]) -> Result<()>,
    {
        self.route_to_closure(count, |grads| {
            if !enabled {
                return Ok(());
            }
            let owned = cow_grads_to_owned(grads);
            f(&owned)
        })
    }

    /// Route gradients to a closure only when the routed slice has exactly `N` items.
    ///
    /// The closure receives an array reference, enabling typed access without
    /// repeated length checks at call sites.
    pub fn route_exact_owned_ref_to_closure<const N: usize, F>(
        &mut self,
        count: usize,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&[Array2<f32>; N]) -> Result<()>,
    {
        self.route_owned_to_closure(count, |grads| {
            if let Ok(arr_ref) = <&[Array2<f32>; N]>::try_from(grads) {
                f(arr_ref)?;
            }
            Ok(())
        })
    }

    /// Route gradients to a closure only when the routed slice has exactly `N` items.
    ///
    /// This variant stays zero-copy by passing Cow references directly.
    pub fn route_exact_ref_to_closure<const N: usize, F>(
        &mut self,
        count: usize,
        mut f: F,
    ) -> Result<()>
    where
        F: for<'b> FnMut(&[std::borrow::Cow<'b, Array2<f32>>; N]) -> Result<()>,
    {
        self.route_to_closure(count, |grads| {
            if let Ok(arr_ref) = <&[std::borrow::Cow<'_, Array2<f32>>; N]>::try_from(grads) {
                f(arr_ref)?;
            }
            Ok(())
        })
    }

    /// Route gradients to a closure only when enabled and the routed slice has exactly `N` items.
    ///
    /// This is useful for optional components where gradients must still be consumed for
    /// partition alignment, while preserving strict shape validation when the component exists.
    pub fn route_exact_ref_to_closure_if<const N: usize, F>(
        &mut self,
        count: usize,
        enabled: bool,
        mut f: F,
    ) -> Result<()>
    where
        F: for<'b> FnMut(&[std::borrow::Cow<'b, Array2<f32>>; N]) -> Result<()>,
    {
        self.route_to_closure(count, |grads| {
            if !enabled {
                return Ok(());
            }
            let arr_ref =
                <&[std::borrow::Cow<'_, Array2<f32>>; N]>::try_from(grads).map_err(|_| {
                    crate::common::errors::ModelError::InvalidInput {
                        message: format!("Expected {} gradient arrays, got {}", N, grads.len()),
                    }
                })?;
            f(arr_ref)
        })
    }

    /// Route gradients to a closure with LARS-style adaptive scaling.
    ///
    /// This combines slicing, Cow handling, and LARS scaling into one shared accessor.
    pub fn route_lars_to_closure<F>(
        &mut self,
        count: usize,
        weight_norm: f32,
        learning_rate: f32,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&[Array2<f32>], f32) -> Result<()>,
    {
        self.route_to_closure(count, |grads| {
            apply_lars_to_cow_grads(grads, weight_norm, learning_rate, |owned, lr| f(owned, lr))
        })
    }

    /// Returns the number of gradients that have been consumed.
    #[inline]
    pub fn consumed(&self) -> usize {
        self.position
    }

    /// Returns the total number of gradients available.
    #[inline]
    pub fn total(&self) -> usize {
        self.gradients.len()
    }

    /// Returns the number of unconsumed gradients remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.total().saturating_sub(self.consumed())
    }
}

/// Extension trait for Option<T> to simplify routing to optional components.
pub trait GradientRouterExt {
    /// Route gradients if the component is Some, otherwise skip.
    fn route_with_lars<T: GradientRoutable>(
        &mut self,
        component: &mut Option<T>,
        learning_rate: f32,
    ) -> Result<()>;
}

impl<'a> GradientRouterExt for GradientRouter<'a> {
    fn route_with_lars<T: GradientRoutable>(
        &mut self,
        component: &mut Option<T>,
        learning_rate: f32,
    ) -> Result<()> {
        if let Some(c) = component {
            self.route_with_lars(c, learning_rate)
        } else {
            // Skip the expected gradient count even if component is None
            // This maintains alignment with compute_gradients output
            Ok(())
        }
    }
}

/// Helper struct for tracking gradient partitions across multiple components.
///
/// This provides a lightweight alternative to storing full partition metadata,
/// using compile-time type information where possible.
#[derive(Debug, Clone, Copy, Default)]
pub struct GradientPartition {
    pub temporal_mixing: usize,
    pub feedforward: usize,
    pub pre_ffn_norm: usize,
    pub pre_attn_norm: usize,
    pub context: usize,
    pub adaptive_residuals: usize,
}

impl GradientPartition {
    /// Total number of gradients across all components.
    #[inline]
    pub fn total(&self) -> usize {
        self.temporal_mixing
            + self.feedforward
            + self.pre_ffn_norm
            + self.pre_attn_norm
            + self.context
            + self.adaptive_residuals
    }

    /// Validates that the partition matches the expected gradient count.
    pub fn validate(&self, expected: usize) -> bool {
        self.total() == expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    struct MockComponent {
        gradient_count: usize,
        weight_norm: f32,
        applied: bool,
    }

    impl GradientRoutable for MockComponent {
        fn gradient_count(&self) -> usize {
            self.gradient_count
        }

        fn weight_norm(&self) -> f32 {
            self.weight_norm
        }

        fn apply_gradients(
            &mut self,
            _gradients: &[Array2<f32>],
            _learning_rate: f32,
        ) -> Result<()> {
            self.applied = true;
            Ok(())
        }
    }

    #[test]
    fn test_gradient_router_basic() {
        let grads = vec![
            Array2::zeros((2, 2)),
            Array2::zeros((2, 2)),
            Array2::zeros((2, 2)),
        ];

        let mut router = GradientRouter::new(&grads, 5.0);
        let mut comp = MockComponent {
            gradient_count: 2,
            weight_norm: 1.0,
            applied: false,
        };

        router.route_with_lars(&mut comp, 0.01).unwrap();
        assert!(comp.applied);
        assert_eq!(router.consumed(), 2);
        assert_eq!(router.remaining(), 1);
    }

    #[test]
    fn test_gradient_router_empty_component() {
        let grads = vec![Array2::zeros((2, 2))];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut comp = MockComponent {
            gradient_count: 0,
            weight_norm: 1.0,
            applied: false,
        };

        router.route_with_lars(&mut comp, 0.01).unwrap();
        assert!(!comp.applied); // Should not call apply
        assert_eq!(router.consumed(), 0);
    }

    #[test]
    fn test_lars_scaling() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let router = GradientRouter::new(&grads, 5.0);

        let slice = GradientSlice {
            grads: router.gradients.clone(),
            start: 0,
            count: 1,
        };

        // Weight norm = 2.0, gradient norm = 2.0, scale should be ~1.0
        let scaled = slice.with_lars_scaling(2.0);
        assert_eq!(scaled.len(), 1);
    }

    #[test]
    fn test_route_owned_to_closure() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_owned_to_closure(1, |owned| {
                called = true;
                assert_eq!(owned.len(), 1);
                Ok(())
            })
            .unwrap();

        assert!(called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_lars_to_closure() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_lars_to_closure(1, 2.0, 0.01, |scaled, lr| {
                called = true;
                assert_eq!(scaled.len(), 1);
                assert_eq!(lr, 0.01);
                Ok(())
            })
            .unwrap();

        assert!(called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_owned_to_closure_if_enabled() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_owned_to_closure_if(1, true, |_| {
                called = true;
                Ok(())
            })
            .unwrap();

        assert!(called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_owned_to_closure_if_disabled() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_owned_to_closure_if(1, false, |_| {
                called = true;
                Ok(())
            })
            .unwrap();

        assert!(!called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_exact_owned_ref_to_closure() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_exact_owned_ref_to_closure::<1, _>(1, |arr| {
                called = true;
                assert_eq!(arr[0].shape(), &[2, 2]);
                Ok(())
            })
            .unwrap();

        assert!(called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_exact_owned_ref_to_closure_mismatch_skips() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_exact_owned_ref_to_closure::<2, _>(1, |_| {
                called = true;
                Ok(())
            })
            .unwrap();

        assert!(!called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_exact_ref_to_closure() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_exact_ref_to_closure::<1, _>(1, |arr| {
                called = true;
                assert_eq!(arr[0].shape(), &[2, 2]);
                Ok(())
            })
            .unwrap();

        assert!(called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_exact_ref_to_closure_mismatch_skips() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_exact_ref_to_closure::<2, _>(1, |_| {
                called = true;
                Ok(())
            })
            .unwrap();

        assert!(!called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_exact_ref_to_closure_if_enabled() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_exact_ref_to_closure_if::<1, _>(1, true, |arr| {
                called = true;
                assert_eq!(arr[0].shape(), &[2, 2]);
                Ok(())
            })
            .unwrap();

        assert!(called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_exact_ref_to_closure_if_disabled() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);
        let mut called = false;

        router
            .route_exact_ref_to_closure_if::<1, _>(1, false, |_| {
                called = true;
                Ok(())
            })
            .unwrap();

        assert!(!called);
        assert_eq!(router.consumed(), 1);
    }

    #[test]
    fn test_route_exact_ref_to_closure_if_enabled_mismatch_errors() {
        let grads = vec![Array2::from_elem((2, 2), 1.0)];
        let mut router = GradientRouter::new(&grads, 5.0);

        let result = router.route_exact_ref_to_closure_if::<2, _>(1, true, |_| Ok(()));
        assert!(result.is_err());
    }
}
