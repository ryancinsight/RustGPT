//! Generic gradient operations for zero-cost abstraction across CoPE variants.
//!
//! This module provides traits and utilities for common gradient operations
//! that are shared across different CoPE gradient types, eliminating
//! repetitive boilerplate through zero-cost generics.

use ndarray::{Array2, Zip};

/// Trait for gradient types that can accumulate gradients from another instance.
///
/// This provides a uniform interface for gradient accumulation across all
/// CoPE variants, enabling generic code to work with any gradient type.
///
/// # Mathematical Invariant
/// For gradients g1, g2: accumulate(g1, g2) produces g where
/// ∀i: g[i] = g1[i] + g2[i] (element-wise addition)
pub trait AccumulateGradients {
    /// Accumulate gradients from `other` into `self`.
    ///
    /// # Panics
    /// May panic if gradient structures are incompatible (e.g., mismatched dimensions).
    fn accumulate(&mut self, other: &Self);
}

/// Trait for gradient types that can be serialized to a flat vector.
///
/// This enables uniform serialization across gradient types for
/// checkpointing, gradient sharing, and distributed training.
pub trait GradientsToVec {
    /// Serialize gradients to a flat vector.
    ///
    /// The order of elements is implementation-defined but consistent
    /// for a given gradient type.
    fn to_vec(&self) -> Vec<f32>;
}

/// Zero-cost abstraction for accumulating Option<Array2<f32>> gradients.
///
/// This helper eliminates repetitive `if let (Some(a), Some(b))` patterns
/// while maintaining zero runtime overhead.
#[inline(always)]
pub fn accumulate_optional_arrays(a: &mut Option<Array2<f32>>, b: &Option<Array2<f32>>) {
    match (a.as_mut(), b.as_ref()) {
        (Some(a_arr), Some(b_arr)) => {
            Zip::from(a_arr).and(b_arr).par_for_each(|a_val, &b_val| {
                *a_val += b_val;
            });
        }
        (None, Some(b_arr)) => {
            *a = Some(b_arr.clone());
        }
        _ => {} // (Some, None) or (None, None): nothing to do
    }
}

/// Zero-cost abstraction for appending Option<Array2<f32>> contents to a vector.
///
/// Eliminates repetitive pattern matching when serializing optional gradients.
#[inline(always)]
pub fn append_optional_array_to_vec(v: &mut Vec<f32>, arr: &Option<Array2<f32>>) {
    if let Some(a) = arr {
        v.extend(a.iter());
    }
}

/// Generic implementation of gradient accumulation for types containing
/// optional array fields.
///
/// This macro generates efficient accumulation code for structs with
/// optional gradient arrays, using parallel operations where beneficial.
#[macro_export]
macro_rules! impl_accumulate_for_optional_fields {
    ($type:ty, $($field:ident),+ $(,)?) => {
        impl $crate::domain::attention::position::gradient_ops::AccumulateGradients for $type {
            #[inline]
            fn accumulate(&mut self, other: &Self) {
                $(
                    $crate::domain::attention::position::gradient_ops::accumulate_optional_arrays(
                        &mut self.$field,
                        &other.$field
                    );
                )+
            }
        }
    };
}

/// Generic implementation of to_vec for types containing optional array fields.
#[macro_export]
macro_rules! impl_to_vec_for_optional_fields {
    ($type:ty, $($field:ident),+ $(,)?) => {
        impl $crate::domain::attention::position::gradient_ops::GradientsToVec for $type {
            #[inline]
            fn to_vec(&self) -> Vec<f32> {
                let mut v = Vec::new();
                $(
                    $crate::domain::attention::position::gradient_ops::append_optional_array_to_vec(
                        &mut v, &self.$field
                    );
                )+
                v
            }
        }
    };
}

/// Composite trait for CoPE gradient types that support both operations.
pub trait CoPEGradients: AccumulateGradients + GradientsToVec {}

impl<T: AccumulateGradients + GradientsToVec> CoPEGradients for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[derive(Debug, Default)]
    struct TestGradients {
        field_a: Option<Array2<f32>>,
        field_b: Option<Array2<f32>>,
        scalar: f32,
    }

    impl_accumulate_for_optional_fields!(TestGradients, field_a, field_b);
    impl_to_vec_for_optional_fields!(TestGradients, field_a, field_b);

    #[test]
    fn test_accumulate_optional_arrays_both_some() {
        let mut a = Some(Array2::from_elem((2, 3), 1.0f32));
        let b = Some(Array2::from_elem((2, 3), 2.0f32));
        
        accumulate_optional_arrays(&mut a, &b);
        
        assert!(a.as_ref().unwrap().iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_accumulate_optional_arrays_none_to_some() {
        let mut a: Option<Array2<f32>> = None;
        let b = Some(Array2::from_elem((2, 3), 2.0f32));
        
        accumulate_optional_arrays(&mut a, &b);
        
        assert!(a.is_some());
        assert!(a.as_ref().unwrap().iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_accumulate_optional_arrays_some_to_none() {
        let mut a = Some(Array2::from_elem((2, 3), 1.0f32));
        let b: Option<Array2<f32>> = None;
        
        accumulate_optional_arrays(&mut a, &b);
        
        assert!(a.as_ref().unwrap().iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_accumulate_trait_macro() {
        let mut g1 = TestGradients {
            field_a: Some(Array2::from_elem((2, 2), 1.0)),
            field_b: None,
            scalar: 0.0,
        };
        let g2 = TestGradients {
            field_a: Some(Array2::from_elem((2, 2), 2.0)),
            field_b: Some(Array2::from_elem((3, 3), 3.0)),
            scalar: 0.0,
        };
        
        g1.accumulate(&g2);
        
        assert!(g1.field_a.as_ref().unwrap().iter().all(|&x| (x - 3.0).abs() < 1e-6));
        assert!(g1.field_b.as_ref().unwrap().iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_to_vec_trait_macro() {
        let g = TestGradients {
            field_a: Some(Array2::from_elem((2, 2), 1.0)),
            field_b: Some(Array2::from_elem((1, 2), 2.0)),
            scalar: 0.0,
        };
        
        let v = g.to_vec();
        
        // field_a: 4 elements of 1.0, field_b: 2 elements of 2.0
        assert_eq!(v.len(), 6);
        assert!(v[..4].iter().all(|&x| (x - 1.0).abs() < 1e-6));
        assert!(v[4..].iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_append_optional_array_to_vec() {
        let mut v = vec![0.0, 0.0];
        let arr = Some(Array2::from_elem((2, 2), 1.0f32));
        
        append_optional_array_to_vec(&mut v, &arr);
        
        assert_eq!(v.len(), 6);
        assert!(v[..2].iter().all(|&x| x == 0.0));
        assert!(v[2..].iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_append_optional_array_none() {
        let mut v = vec![1.0, 2.0];
        let arr: Option<Array2<f32>> = None;
        
        append_optional_array_to_vec(&mut v, &arr);
        
        assert_eq!(v, vec![1.0, 2.0]);
    }
}
