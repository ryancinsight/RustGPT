use serde::{Deserialize, Deserializer, Serialize};

use super::mamba::Mamba;
use crate::network::Layer;

/// A pragmatic "Mamba-2 style" temporal mixer.
///
/// Implemented as a thin wrapper around the full `Mamba` reference
/// implementation to avoid duplicating scan/gradient logic.
///
/// Differences vs `Mamba`:
/// - larger default convolution kernel
#[derive(Serialize, Debug, Clone)]
pub struct Mamba2 {
    #[serde(flatten)]
    pub inner: Mamba,
}

impl<'de> Deserialize<'de> for Mamba2 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = Mamba::deserialize(deserializer)?;
        Ok(Self { inner })
    }
}

impl Mamba2 {
    pub fn new(embed_dim: usize) -> Self {
        Self::new_with_kernel(embed_dim, 8)
    }

    pub fn new_with_kernel(embed_dim: usize, conv_kernel: usize) -> Self {
        Self {
            inner: Mamba::new_with_kernel(embed_dim, conv_kernel),
        }
    }
}

impl Layer for Mamba2 {
    fn layer_type(&self) -> &str {
        "Mamba2"
    }

    fn forward(&mut self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        self.inner.forward(input)
    }

    fn backward(&mut self, grads: &ndarray::Array2<f32>, lr: f32) -> ndarray::Array2<f32> {
        self.inner.backward(grads, lr)
    }

    fn parameters(&self) -> usize {
        self.inner.parameters()
    }

    fn weight_norm(&self) -> f32 {
        self.inner.weight_norm()
    }

    fn compute_gradients(
        &self,
        input: &ndarray::Array2<f32>,
        output_grads: &ndarray::Array2<f32>,
    ) -> (ndarray::Array2<f32>, Vec<ndarray::Array2<f32>>) {
        self.inner.compute_gradients(input, output_grads)
    }

    fn apply_gradients(
        &mut self,
        gradients: &[ndarray::Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()> {
        self.inner.apply_gradients(gradients, learning_rate)
    }

    fn zero_gradients(&mut self) {
        self.inner.zero_gradients();
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn mamba2_forward_backward_shapes() {
        let mut layer = Mamba2::new_with_kernel(16, 5);
        let x = Array2::<f32>::zeros((8, 16));
        let y = layer.forward(&x);
        assert_eq!(y.shape(), [8, 16]);

        let grads = Array2::<f32>::ones((8, 16));
        let dx = layer.backward(&grads, 1e-3);
        assert_eq!(dx.shape(), [8, 16]);
        assert!(dx.iter().all(|v| v.is_finite()));
    }
}
