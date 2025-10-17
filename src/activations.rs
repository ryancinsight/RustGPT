use ndarray::Array2;

/// Swish activation function (also known as SiLU)
///
/// Swish(x) = x * sigmoid(x) = x * (1 / (1 + e^{-x}))
#[inline]
pub fn swish(x: &Array2<f32>) -> Array2<f32> {
    let sigmoid = x.mapv(|val| 1.0 / (1.0 + (-val).exp()));
    x * &sigmoid
}

/// Derivative of Swish with respect to x
///
/// d/dx Swish(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
#[inline]
pub fn swish_derivative(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|val| {
        let s = 1.0 / (1.0 + (-val).exp());
        s + val * s * (1.0 - s)
    })
}
