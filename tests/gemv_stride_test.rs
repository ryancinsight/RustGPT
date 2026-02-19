#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, s};

    #[test]
    fn test_gemv_negative_stride() {
        let n = 3;
        let d = 2;
        // Matrix: [[0,0], [1,1], [2,2]]
        let mat = Array2::from_shape_fn((n, d), |(i, _)| i as f32);

        // Reversed: [[2,2], [1,1], [0,0]]
        let mat_rev = mat.slice(s![..;-1, ..]);

        // Vector: [1, 0]
        let vec = Array1::from_vec(vec![1.0, 0.0]);

        let mut out = Array1::zeros(n);

        // out = mat_rev * vec
        // Expected:
        // Row 0: [2,2] * [1,0] = 2
        // Row 1: [1,1] * [1,0] = 1
        // Row 2: [0,0] * [1,0] = 0
        // Result: [2, 1, 0]

        ndarray::linalg::general_mat_vec_mul(1.0, &mat_rev, &vec, 0.0, &mut out);

        println!("Output: {:?}", out);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 0.0);
    }
}
