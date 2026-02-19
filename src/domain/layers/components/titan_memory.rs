use ndarray::{Array1, Array2, Zip};
use serde::{Deserialize, Serialize};

use crate::domain::models::config::TitanMemoryConfig;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct TitanMemoryWorkspace {
    pub acc: Vec<f32>,
}

impl TitanMemoryWorkspace {
    pub fn reset(&mut self) {
        self.acc.fill(0.0);
    }
}

/// Extension trait or helper methods for TitanMemoryConfig logic.
/// Since we cannot define an impl block for a type defined in another crate (unless we own it),
/// and TitanMemoryConfig is in crate::domain::models::config.
/// But we are in the same crate! So we CAN define the impl block here.
/// Rust allows impl blocks in different modules of the same crate.

impl TitanMemoryConfig {
    #[cfg(test)]
    pub fn apply_into_out(&self, out: &mut Array2<f32>, input: &Array2<f32>) {
        let mut ws = TitanMemoryWorkspace::default();
        self.apply_into_out_with_workspace(out, input, &mut ws);
    }

    pub fn apply_into_out_with_workspace(
        &self,
        out: &mut Array2<f32>,
        input: &Array2<f32>,
        workspace: &mut TitanMemoryWorkspace,
    ) {
        if !self.enabled {
            return;
        }
        let n = input.nrows();
        let d = input.ncols();
        assert_eq!(out.nrows(), n);
        assert_eq!(out.ncols(), d);
        assert!(self.scale.is_finite());
        assert!(self.eta.is_finite());
        assert!(self.decay.is_finite());
        assert!(self.eta >= 0.0);
        assert!(self.decay >= 0.0 && self.decay <= 1.0);

        let retain = 1.0 - self.decay;
        if workspace.acc.len() != d {
            workspace.acc.resize(d, 0.0);
            workspace.acc.fill(0.0); // Reset new elements or if resized
        }
        // Note: We do NOT reset workspace.acc here for streaming persistence.
        // Batch callers must call workspace.reset() explicitly before processing a sequence.

        // Optimized loop with Zip and cache-friendly iteration
        for i in 0..n {
            let input_row = input.row(i);
            let mut out_row = out.row_mut(i);

            Zip::from(&mut workspace.acc)
                .and(&input_row)
                .and(&mut out_row)
                .for_each(|acc, &inp, out_val| {
                    let next = retain * *acc + self.eta * inp;
                    *acc = next;
                    *out_val += self.scale * next;
                });
        }
    }

    pub fn apply_step_into(
        &self,
        input: &ndarray::ArrayView1<f32>,
        out: &mut ndarray::Array1<f32>,
        workspace: &mut TitanMemoryWorkspace,
    ) {
        if !self.enabled {
            return;
        }
        let d = input.len();
        assert_eq!(out.len(), d);
        assert!(self.scale.is_finite());
        assert!(self.eta.is_finite());
        assert!(self.decay.is_finite());
        assert!(self.eta >= 0.0);
        assert!(self.decay >= 0.0 && self.decay <= 1.0);

        let retain = 1.0 - self.decay;
        if workspace.acc.len() != d {
            workspace.acc.resize(d, 0.0);
            workspace.acc.fill(0.0);
        }

        for j in 0..d {
            let next = retain * workspace.acc[j] + self.eta * input[j];
            workspace.acc[j] = next;
            out[j] += self.scale * next;
        }
    }

    #[cfg(test)]
    pub fn input_grads_from_output_grads(&self, output_grads: &Array2<f32>) -> Array2<f32> {
        if !self.enabled {
            return Array2::zeros(output_grads.raw_dim());
        }
        let n = output_grads.nrows();
        let d = output_grads.ncols();
        assert!(self.scale.is_finite());
        assert!(self.eta.is_finite());
        assert!(self.decay.is_finite());
        assert!(self.eta >= 0.0);
        assert!(self.decay >= 0.0 && self.decay <= 1.0);

        let retain = 1.0 - self.decay;
        let mut input_grads = Array2::<f32>::zeros(output_grads.raw_dim());

        let mut b = Array1::<f32>::zeros(d);
        for i in (0..n).rev() {
            let g_row = output_grads.row(i);
            let mut in_g_row = input_grads.row_mut(i);

            Zip::from(&mut b)
                .and(&g_row)
                .and(&mut in_g_row)
                .for_each(|b_val, &g, in_g| {
                    *b_val = retain * *b_val + g;
                    *in_g = self.scale * self.eta * *b_val;
                });
        }
        input_grads
    }

    pub fn add_input_grads_from_output_grads_into(
        &self,
        output_grads: &Array2<f32>,
        input_grads: &mut Array2<f32>,
    ) {
        if !self.enabled {
            return;
        }
        let n = output_grads.nrows();
        let d = output_grads.ncols();
        assert_eq!(input_grads.nrows(), n);
        assert_eq!(input_grads.ncols(), d);
        assert!(self.scale.is_finite());
        assert!(self.eta.is_finite());
        assert!(self.decay.is_finite());
        assert!(self.eta >= 0.0);
        assert!(self.decay >= 0.0 && self.decay <= 1.0);

        let retain = 1.0 - self.decay;
        let coeff = self.scale * self.eta;

        let mut b = Array1::<f32>::zeros(d);
        for i in (0..n).rev() {
            let g_row = output_grads.row(i);
            let mut in_g_row = input_grads.row_mut(i);

            Zip::from(&mut b)
                .and(&g_row)
                .and(&mut in_g_row)
                .for_each(|b_val, &g, in_g| {
                    *b_val = retain * *b_val + g;
                    *in_g += coeff * *b_val;
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rng::{get_rng_with_subseed, set_seed};
    use ndarray::Array2;
    use proptest::prelude::*;

    #[test]
    fn titan_memory_linear_adjoint_matches_backward() {
        let cfg = TitanMemoryConfig {
            enabled: true,
            scale: 0.3,
            eta: 0.7,
            decay: 0.2,
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let x = Array2::from_shape_fn((7, 5), |(i, j)| (i as f32 * 0.01) - (j as f32 * 0.02));
        let g = Array2::from_shape_fn((7, 5), |(i, j)| (i as f32 * 0.03) + (j as f32 * 0.01));

        let mut y = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out(&mut y, &x);
        let gx = cfg.input_grads_from_output_grads(&g);

        let lhs: f64 = y
            .iter()
            .zip(g.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let rhs: f64 = x
            .iter()
            .zip(gx.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        assert!((lhs - rhs).abs() < 1e-5);
    }

    #[test]
    fn titan_memory_linear_adjoint_random_seeded() {
        set_seed(0xC0FFEE);
        let mut rng_cfg = get_rng_with_subseed(1);
        let cfg = TitanMemoryConfig {
            enabled: true,
            scale: rng_cfg.random_range(-1.0..1.0),
            eta: rng_cfg.random_range(0.0..1.0),
            decay: rng_cfg.random_range(0.0..1.0),
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let mut rng_x = get_rng_with_subseed(2);
        let x = Array2::from_shape_fn((19, 13), |_| rng_x.random_range(-1.0..1.0));
        let mut rng_g = get_rng_with_subseed(3);
        let g = Array2::from_shape_fn((19, 13), |_| rng_g.random_range(-1.0..1.0));

        let mut y = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out(&mut y, &x);
        let gx = cfg.input_grads_from_output_grads(&g);

        let lhs: f64 = y
            .iter()
            .zip(g.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let rhs: f64 = x
            .iter()
            .zip(gx.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        let tol = 1e-4 * (1.0 + lhs.abs() + rhs.abs());
        assert!((lhs - rhs).abs() <= tol);
    }

    #[test]
    fn titan_memory_disabled_is_noop() {
        let cfg = TitanMemoryConfig {
            enabled: false,
            scale: 0.3,
            eta: 0.7,
            decay: 0.2,
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let x = Array2::from_shape_fn((3, 4), |(i, j)| (i as f32) + (j as f32));
        let mut y = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out(&mut y, &x);
        assert!(y.iter().all(|&v| v == 0.0));

        let g = Array2::from_shape_fn((3, 4), |(i, j)| (i as f32) - (j as f32));
        let gx = cfg.input_grads_from_output_grads(&g);
        assert!(gx.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn titan_memory_add_input_grads_matches_allocating_version() {
        let cfg = TitanMemoryConfig {
            enabled: true,
            scale: 0.11,
            eta: 0.9,
            decay: 0.05,
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let g = Array2::from_shape_fn((9, 4), |(i, j)| (i as f32 * 0.02) - (j as f32 * 0.03));
        let ref_gx = cfg.input_grads_from_output_grads(&g);

        let mut gx = Array2::<f32>::zeros(g.raw_dim());
        cfg.add_input_grads_from_output_grads_into(&g, &mut gx);

        assert_eq!(gx.dim(), ref_gx.dim());
        for (&a, &b) in gx.iter().zip(ref_gx.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn titan_memory_workspace_is_equivalent_to_fresh_workspace() {
        let cfg = TitanMemoryConfig {
            enabled: true,
            scale: 0.17,
            eta: 0.23,
            decay: 0.11,
            segment_len: 128,
            persistent_len: 32,
            hidden_dim: 64,
            ..TitanMemoryConfig::default()
        };

        let x = Array2::from_shape_fn((11, 7), |(i, j)| (i as f32 * 0.007) - (j as f32 * 0.013));
        let mut y_fresh = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out(&mut y_fresh, &x);

        // Initialize workspace with correct size (matching input dimension) and reset it
        let mut ws = TitanMemoryWorkspace {
            acc: vec![0.0; x.ncols()],
        };
        ws.reset(); // Ensure clean state
        let mut y_ws1 = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out_with_workspace(&mut y_ws1, &x, &mut ws);

        // For second call, need fresh workspace (batch callers must reset explicitly)
        let mut ws2 = TitanMemoryWorkspace {
            acc: vec![0.0; x.ncols()],
        };
        ws2.reset();
        let mut y_ws2 = Array2::<f32>::zeros(x.raw_dim());
        cfg.apply_into_out_with_workspace(&mut y_ws2, &x, &mut ws2);

        assert_eq!(y_fresh.dim(), y_ws1.dim());
        assert_eq!(y_fresh.dim(), y_ws2.dim());
        for ((&a, &b), &c) in y_fresh.iter().zip(y_ws1.iter()).zip(y_ws2.iter()) {
            assert!((a - b).abs() < 1e-6);
            assert!((a - c).abs() < 1e-6);
        }
    }

    proptest! {
        #[test]
        fn titan_memory_adjoint_property_holds(
            n in 1usize..33,
            d in 1usize..33,
            scale in -1.0f32..1.0f32,
            eta in 0.0f32..1.0f32,
            decay in 0.0f32..1.0f32,
            x_flat in prop::collection::vec(-1.0f32..1.0f32, 1..(33*33)),
            g_flat in prop::collection::vec(-1.0f32..1.0f32, 1..(33*33)),
        ) {
            let len = n * d;
            prop_assume!(x_flat.len() >= len);
            prop_assume!(g_flat.len() >= len);

            let cfg = TitanMemoryConfig {
                enabled: true,
                scale,
                eta,
                decay,
                segment_len: 128,
                persistent_len: 32,
                hidden_dim: 64,
                ..TitanMemoryConfig::default()
            };
            let x = Array2::from_shape_vec((n, d), x_flat[..len].to_vec()).unwrap();
            let g = Array2::from_shape_vec((n, d), g_flat[..len].to_vec()).unwrap();

            let mut y = Array2::<f32>::zeros(x.raw_dim());
            cfg.apply_into_out(&mut y, &x);
            let gx = cfg.input_grads_from_output_grads(&g);

            let lhs: f64 = y.iter().zip(g.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
            let rhs: f64 = x.iter().zip(gx.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();

            let tol = 1e-4 * (1.0 + lhs.abs() + rhs.abs());
            prop_assert!((lhs - rhs).abs() <= tol);
        }
    }
}
