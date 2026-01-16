#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use crate::attention::poly_attention::PolyAttention;

    #[test]
    fn test_apply_gradients_panic() {
        let mut pa = PolyAttention::new(32, 4, 3, 64, Some(8));
        let n = 2;
        let d = 32;
        let input = Array2::<f32>::zeros((n, d));
        let output_grads = Array2::<f32>::ones((n, d));

        let (_gi, param_grads) = pa.compute_gradients_parallel(&input, &output_grads);

        // This should panic currently because of the unwrap() on a failed apply_gradients call
        pa.apply_gradients(&param_grads, 0.01).unwrap();
    }
}
