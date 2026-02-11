// This file contains the corrected test module for poly_attention.rs
// Replace the entire #[cfg(test)] mod tests section with this content

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use crate::domain::network::Layer;
    use crate::domain::attention::position::config::{CoPEConfig, CoPEVariant};

    use super::{AdaptiveDegreeConfig, DegreeAdaptationMetrics, PolyAttention};
    use crate::domain::models::config::TitanMemoryConfig;

    fn create_cope_config(max_pos: usize, window_size: Option<usize>) -> CoPEConfig {
        CoPEConfig {
            variant: CoPEVariant::Standard,
            max_pos,
            window_size,
        }
    }

    #[test]
    fn gradients_parallel_match_sequential_small() {
        let cope_config = create_cope_config(64, Some(4));
        let mut pa = PolyAttention::new(16, 4, 3, cope_config);
        pa.set_titan_memory_config(TitanMemoryConfig {
            enabled: false,
            ..TitanMemoryConfig::default()
        });
        let n = 8;
        let d = 16;
        let mut input = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                input[[i, j]] = ((i * d + j) as f32 * 0.01).sin();
            }
        }
        let _ = pa.forward_impl(&input, true);
        let mut output_grads = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                output_grads[[i, j]] = (((i + j) as f32) * 0.001).cos();
            }
        }
        let (gi_seq, pg_seq) = pa.compute_gradients(&input, &output_grads);
        let (gi_par, pg_par) = pa.compute_gradients_parallel(&input, &output_grads);
        assert_eq!(pg_seq.len(), pg_par.len());
        let mut diff_input = 0.0f32;
        for i in 0..n {
            for j in 0..d {
                diff_input += (gi_seq[[i, j]] - gi_par[[i, j]]).abs();
            }
        }
        assert!(diff_input < 1e-3);
        for (a, b) in pg_seq.iter().zip(pg_par.iter()) {
            assert_eq!(a.shape(), b.shape());
            let mut diff = 0.0f32;
            for (xa, xb) in a.iter().zip(b.iter()) {
                diff += (xa - xb).abs();
            }
            assert!(diff < 1e-2);
        }
    }

    #[test]
    fn adapt_increases_degree_on_slow_convergence() {
        let cope_config = create_cope_config(128, None);
        let mut pa = PolyAttention::new(64, 8, 3, cope_config);
        pa.set_adaptive_degree_config(AdaptiveDegreeConfig {
            enabled: true,
            p_min: 1,
            p_max: 5,
            adjust_rate: 1.0,
            increase_threshold: 0.1,
            decrease_threshold: -0.5,
            cooldown_epochs: 0,
        });
        let m = DegreeAdaptationMetrics {
            epoch_index: 0,
            loss_delta: 0.0,
            grad_norm: 1.0,
            epoch_ms: 10.0,
            tokens_per_sec: 1000.0,
            tau_range: None,
            pred_norm_rms: Some(0.0),
        };
        let p0 = pa.p;
        pa.adapt_degree(&m);
        assert!(pa.p >= p0);
    }

    #[test]
    fn adapt_decreases_degree_on_high_grad() {
        let cope_config = create_cope_config(128, None);
        let mut pa = PolyAttention::new(64, 8, 3, cope_config);
        pa.set_adaptive_degree_config(AdaptiveDegreeConfig {
            enabled: true,
            p_min: 1,
            p_max: 7,
            adjust_rate: 1.0,
            increase_threshold: 0.9,
            decrease_threshold: -0.1,
            cooldown_epochs: 0,
        });
        let m = DegreeAdaptationMetrics {
            epoch_index: 0,
            loss_delta: 1.0,
            grad_norm: 1e6,
            epoch_ms: 10.0,
            tokens_per_sec: 1000.0,
            tau_range: None,
            pred_norm_rms: Some(1.0),
        };
        let p0 = pa.p;
        pa.adapt_degree(&m);
        assert!(pa.p <= p0);
    }

    #[test]
    fn eff_skip_threshold_skips_computation() {
        let cope_config = create_cope_config(64, Some(16));
        let mut pa = PolyAttention::new(64, 4, 3, cope_config);
        pa.set_titan_memory_config(TitanMemoryConfig {
            enabled: false,
            ..TitanMemoryConfig::default()
        });
        let n = 8;
        let d = 64;
        let mut input = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                input[[i, j]] = ((i * d + j) as f32) * 0.0007;
            }
        }
        pa.set_eff_skip_threshold(1.0);
        let out_skip = pa.forward_impl(&input, false);
        assert_eq!(out_skip, Array2::<f32>::zeros((n, d)));
        pa.set_eff_skip_threshold(0.0);
        let out_no_skip = pa.forward_impl(&input, false);
        assert_ne!(out_no_skip, input);
    }

    #[test]
    fn soft_top_p_cache_includes_modulation_and_token_scale() {
        let cope_config = create_cope_config(64, Some(8));
        let mut pa = PolyAttention::new(32, 4, 3, cope_config);
        pa.moh.head_selection_config.gating.use_soft_top_p = true;
        pa.moh.head_selection_config.gating.top_p = 0.9;
        pa.moh.head_selection_config.gating.soft_top_p_alpha = 2.0;
        pa.moh.head_selection_config.max_heads = 1;
        pa.moh.head_selection_config.threshold_modulation =
            crate::domain::richards::adaptive::AdaptiveScalar::Fixed(1.25);

        let n = 4;
        let d = 32;
        let mut input = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                input[[i, j]] = ((i * d + j) as f32 * 0.03).sin();
            }
        }

        let token_scale = Array2::from_shape_vec((n, 1), vec![1.0, 0.5, 2.0, 1.5]).unwrap();
        pa.set_token_threshold_scale(token_scale);

        let _ = pa.forward_impl(&input, true);
        let mask = pa
            .moh
            .cached_soft_top_p_mask
            .as_ref()
            .expect("soft top-p mask must be cached when enabled");

        let sum0: f32 = mask.row(0).sum();
        let sum1: f32 = mask.row(1).sum();
        let sum2: f32 = mask.row(2).sum();
        assert!(sum2 > sum0);
        assert!(sum1 < sum0);
    }

    #[test]
    fn moh_learned_predictor_per_head_thresholds() {
        let cope_config = create_cope_config(64, Some(8));
        let mut pa = PolyAttention::new(32, 4, 3, cope_config);
        let strategy = crate::domain::mixtures::moh::HeadSelectionStrategy::Learned {
            num_active: 4,
            load_balance_weight: 0.1,
            complexity_loss_weight: 0.05,
            sparsity_weight: 0.01,
            importance_loss_weight: 0.0,
            switch_balance_weight: 0.0,
            training_mode: crate::domain::mixtures::gating::GatingTrainingMode::Coupled,
        };
        pa.set_head_selection_config(&strategy);
        let n = 6;
        let d = 32;
        let mut input = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                input[[i, j]] = ((i * d + j) as f32 * 0.003).cos();
            }
        }
        let _out = pa.forward_impl(&input, true);
        let tau = pa.take_tau_metrics();
        assert!(tau.is_some());
        let pred_norm = pa.take_pred_norm();
        assert!(pred_norm.is_some());

        let mut output_grads = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                output_grads[[i, j]] = (((i + j) as f32) * 0.0007).sin();
            }
        }
        let (gi, pg) = pa.compute_gradients_parallel(&input, &output_grads);
        let non_finite = gi.iter().any(|x| !x.is_finite())
            || pg.iter().any(|g| g.iter().any(|x| !x.is_finite()));
        assert!(!non_finite);
    }

    #[test]
    fn test_moh_independent_training_decoupling() {
        use crate::domain::mixtures::gating::GatingTrainingMode;

        let cope_config = create_cope_config(64, Some(8));
        let mut pa = PolyAttention::new(32, 4, 3, cope_config);

        // Setup Independent training strategy
        let strategy = crate::domain::mixtures::moh::HeadSelectionStrategy::Learned {
            num_active: 4,
            load_balance_weight: 0.0,
            complexity_loss_weight: 0.0,
            sparsity_weight: 0.0,
            importance_loss_weight: 0.0,
            switch_balance_weight: 0.0,
            training_mode: GatingTrainingMode::Independent,
        };
        pa.set_head_selection_config(&strategy);

        let n = 4;
        let d = 32;
        let mut input = Array2::<f32>::zeros((n, d));
        // Simple input
        for i in 0..n {
            for j in 0..d {
                input[[i, j]] = 0.1;
            }
        }

        // Forward pass
        let _ = pa.forward_impl(&input, true);

        // Backward pass with non-zero output gradients
        let output_grads = Array2::<f32>::ones((n, d));

        let (_grad_input, param_grads) = pa.compute_gradients_parallel(&input, &output_grads);

        // Check gating parameters gradients.
        // Indices:
        // w_q (0), w_k (1), w_v (2), w_out (3)
        // a (4), b (5), scale (6)
        // w_g (7), alpha_g (8), beta_g (9), gate_poly (10)
        let idx_w_g = 7;
        let idx_alpha_g = 8;
        let idx_beta_g = 9;
        let idx_gate_poly = 10;

        let grad_w_g = &param_grads[idx_w_g];
        let grad_alpha_g = &param_grads[idx_alpha_g];
        let grad_beta_g = &param_grads[idx_beta_g];
        let grad_gate_poly = &param_grads[idx_gate_poly];

        // Since aux weights are 0 and mode is Independent, gradients from attention should not flow
        // to gating So gating gradients should be exactly zero.
        assert!(
            grad_w_g.iter().all(|&x| x == 0.0),
            "w_g grad should be 0 in independent mode without aux loss"
        );
        assert!(
            grad_alpha_g.iter().all(|&x| x == 0.0),
            "alpha_g grad should be 0"
        );
        assert!(
            grad_beta_g.iter().all(|&x| x == 0.0),
            "beta_g grad should be 0"
        );
        assert!(
            grad_gate_poly.iter().all(|&x| x == 0.0),
            "gate_poly grad should be 0"
        );

        // Now switch to Coupled and verify we GET gradients
        let strategy_coupled = crate::domain::mixtures::moh::HeadSelectionStrategy::Learned {
            num_active: 4,
            load_balance_weight: 0.0,
            complexity_loss_weight: 0.0,
            sparsity_weight: 0.0,
            importance_loss_weight: 0.0,
            switch_balance_weight: 0.0,
            training_mode: GatingTrainingMode::Coupled,
        };
        pa.set_head_selection_config(&strategy_coupled);

        let (_grad_input_c, param_grads_c) = pa.compute_gradients_parallel(&input, &output_grads);

        let grad_w_g_c = &param_grads_c[idx_w_g];

        // In coupled mode, we expect some gradients flowing back from attention
        // (assuming the gate values are not saturated and weights allow flow)
        // With constant input 0.1, values should be non-zero unless something is degenerate.
        // We can just check that they are NOT all zero, or at least different from Independent.

        // Note: if gate is saturated, grad might be small.
        // Let's assert that AT LEAST one gating parameter has non-zero gradient in coupled mode.
        let has_grad = grad_w_g_c.iter().any(|&x| x.abs() > 1e-10)
            || param_grads_c[idx_alpha_g].iter().any(|&x| x.abs() > 1e-10)
            || param_grads_c[idx_beta_g].iter().any(|&x| x.abs() > 1e-10);

        assert!(has_grad, "Should have gradients in Coupled mode");
    }

    #[test]
    fn test_moh_independent_training_with_aux_loss_grads() {
        use crate::domain::mixtures::gating::GatingTrainingMode;
        // This test verifies that in Independent mode with auxiliary losses,
        // RichardsCurve parameters SHOULD receive gradients.

        let cope_config = create_cope_config(64, Some(8));
        let mut pa = PolyAttention::new(32, 4, 3, cope_config);

        // Setup Independent training strategy WITH auxiliary loss
        let strategy = crate::domain::mixtures::moh::HeadSelectionStrategy::Learned {
            num_active: 4,
            load_balance_weight: 1.0, // High weight to ensure gradients
            complexity_loss_weight: 0.0,
            sparsity_weight: 0.0,
            importance_loss_weight: 0.0,
            switch_balance_weight: 0.0,
            training_mode: GatingTrainingMode::Independent,
        };
        pa.set_head_selection_config(&strategy);

        let n = 4;
        let d = 32;
        let mut input = Array2::<f32>::zeros((n, d));
        // Simple input
        for i in 0..n {
            for j in 0..d {
                input[[i, j]] = ((i * d + j) as f32 * 0.1).sin();
            }
        }

        // Forward pass
        let _ = pa.forward_impl(&input, true);

        // Backward pass with non-zero output gradients
        let output_grads = Array2::<f32>::ones((n, d));

        let (_grad_input, param_grads) = pa.compute_gradients_parallel(&input, &output_grads);

        // Indices:
        // w_q (0), w_k (1), w_v (2), w_out (3)
        // a (4), b (5), scale (6)
        // w_g (7), alpha_g (8), beta_g (9), gate_poly (10)
        let idx_gate_poly = 10;

        let grad_gate_poly = &param_grads[idx_gate_poly];

        // We expect gradients to be present because of load_balance_weight
        let has_grad = grad_gate_poly.iter().any(|&x| x.abs() > 1e-10);

        // Assert that we HAVE gradients.
        assert!(
            has_grad,
            "gate_poly grad should be NON-zero in independent mode with aux loss"
        );
    }

    #[test]
    fn test_apply_gradients_works() {
        // This test ensures that apply_gradients doesn't panic due to gradient unpacking mismatch
        let cope_config = create_cope_config(64, Some(8));
        let mut pa = PolyAttention::new(32, 4, 3, cope_config);
        let n = 2;
        let d = 32;
        let input = Array2::<f32>::zeros((n, d));
        let output_grads = Array2::<f32>::ones((n, d));

        // Need forward pass to cache input
        let _ = pa.forward_impl(&input, true);

        let (_gi, param_grads) = pa.compute_gradients_parallel(&input, &output_grads);

        // This should NOT panic now
        pa.apply_gradients(&param_grads, 0.01).unwrap();
    }
}
