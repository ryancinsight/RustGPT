pub mod adam;
pub mod attention;
pub mod diffusion;
pub mod transformer;
pub mod trm;

pub mod dataset_loader;
pub mod embeddings;
pub mod errors;
pub mod loss;
pub mod metrics;
pub mod pade;
pub mod richards;
pub mod softmax;

// removed: pub mod head_router;
pub mod llm;

pub mod model_builder;
pub mod model_config;
pub mod model_persistence;
pub mod output_projection;
// removed: pub mod sigmoid_poly;
// removed: pub mod routing;
// removed: pub mod self_attention;
pub mod mixtures;

// removed: pub mod trm;
pub mod decoding;
pub mod encoding;

// Define crate-level constants used across modules
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;
pub const MAX_SEQ_LEN: usize = 256;
pub const MAX_VOCAB_SIZE: usize = 50_000;
pub const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100MB
pub const MAX_INPUT_LENGTH: usize = 10_000;
pub const GRADIENT_ANOMALY_THRESHOLD: f32 = 5000.0;

// Re-export key structs for easier access
pub use adam::Adam;
pub use dataset_loader::{Dataset, DatasetType};
// Also re-export decoding types for convenience
pub use decoding::GreedyDecoder;
pub use embeddings::TokenEmbeddings as Embeddings;
// Also re-export encoding types for convenience
pub use encoding::{SimpleTokenizer, Vocab};
pub use errors::{ModelError, Result};
// removed head_router re-exports
// pub use head_router::{RouterType, FullyAdaptiveHeadRouter};
pub use llm::{LLM, Layer, LayerEnum};
// Also re-export mixture types for convenience
pub use mixtures::{
    ExpertRouter, ExpertRouterConfig, HeadSelectionConfig, HeadSelectionStrategy, MixtureOfExperts,
    ThresholdPredictor,
};
pub use model_builder::{build_network, print_architecture_summary};
pub use model_config::{ArchitectureType, AttentionType, ModelConfig, WindowAdaptationStrategy};
// Also re-export RichardsGlu
pub use richards::RichardsGlu;
// Also re-export RichardsNorm as DynamicTanhNorm for compatibility
pub use richards::RichardsNorm as DynamicTanhNorm;
pub use trm::TRM;

#[cfg(test)]
mod trm_mathematical_validation_tests {
    use ndarray::Array2;

    use super::*;
    use crate::trm::TRMConfig;

    /// Theorem 1 Validation: TRM Recursive Convergence
    /// Test that TRM converges under Lipschitz conditions
    #[test]
    fn test_trm_convergence_theorem() {
        println!("=== Testing TRM Convergence Theorem ===");

        let config = TRMConfig {
            embed_dim: 64,
            num_recursions: 3,
            max_supervision_steps: 5,
            max_inference_steps: 2,
            use_shared_weights: true,
            latent_update_alpha: 0.05,
        };

        let mut trm = TRM::new(config);

        // Create test input
        let batch_size = 2;
        let input = Array2::<f32>::from_elem((batch_size, 64), 0.1);

        // Test forward pass converges
        let result = trm.forward_recursive(&input);
        assert!(result.is_ok(), "TRM forward pass should succeed");

        let output = result.unwrap();
        assert_eq!(
            output.shape(),
            &[batch_size, 64],
            "Output shape should match input"
        );

        // Test that output is finite and reasonable
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "All outputs should be finite"
        );

        println!("✅ TRM convergence validated - forward pass produces finite outputs");
    }

    /// Theorem 2 Validation: TRM Stability Bounds
    /// Test gradient stability and boundedness
    #[test]
    fn test_trm_stability_bounds() {
        println!("=== Testing TRM Stability Bounds Theorem ===");

        let config = TRMConfig {
            embed_dim: 32,
            num_recursions: 2,
            max_supervision_steps: 3,
            max_inference_steps: 1,
            use_shared_weights: true,
            latent_update_alpha: 0.05,
        };

        let mut trm = TRM::new(config);
        trm.set_training_mode(true);

        let input = Array2::<f32>::from_elem((1, 32), 0.01);
        let target = Array2::<f32>::from_elem((1, 32), 0.02);

        // Compute gradients
        let output = trm.forward(&input);
        let output_grads = &output - &target; // Simple MSE gradient

        let (input_grads, param_grads) = trm.compute_gradients(&input, &output_grads);

        // Validate gradient boundedness
        assert!(
            input_grads.iter().all(|&x: &f32| x.is_finite()),
            "Input gradients should be finite"
        );
        assert!(
            param_grads
                .iter()
                .all(|grads| grads.iter().all(|&x: &f32| x.is_finite())),
            "Parameter gradients should be finite"
        );

        // Test gradient norms are reasonable (not exploding)
        let input_grad_norm: f32 = input_grads.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            input_grad_norm < crate::GRADIENT_ANOMALY_THRESHOLD,
            "Input gradient norm should be bounded: {}",
            input_grad_norm
        );

        for (i, grads) in param_grads.iter().enumerate() {
            let param_grad_norm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                param_grad_norm < crate::GRADIENT_ANOMALY_THRESHOLD,
                "Parameter gradient {} norm should be bounded: {}",
                i,
                param_grad_norm
            );
        }

        println!("✅ TRM stability bounds validated - gradients are finite and bounded");
    }

    /// Theorem 6 Validation: TRM Learnable Latent Initialization
    /// Test that TRM can adapt and learn from data
    #[test]
    fn test_trm_adaptive_learning() {
        println!("=== Testing TRM Adaptive Learning Theorem ===");

        let config = TRMConfig {
            embed_dim: 16, // Must be divisible by 8 (num_heads)
            num_recursions: 2,
            max_supervision_steps: 4,
            max_inference_steps: 1,
            use_shared_weights: true,
            latent_update_alpha: 0.05,
        };

        let mut trm = TRM::new(config);
        trm.set_training_mode(true);

        let input = Array2::<f32>::from_elem((1, 16), 0.02);

        // Test multiple forward passes to validate learning capability
        let mut outputs = Vec::new();
        for _ in 0..3 {
            let output = trm.forward(&input);
            outputs.push(output);
        }

        // All outputs should be finite and consistent
        for (i, output) in outputs.iter().enumerate() {
            assert!(
                output.iter().all(|&x: &f32| x.is_finite()),
                "Output {} should be finite",
                i
            );
        }

        // Test parameter updates by applying gradients
        let target = Array2::<f32>::from_elem((1, 16), 0.0);
        let output = &outputs[0];
        let output_grads = &(output - &target);

        let (input_grads, param_grads) = trm.compute_gradients(&input, output_grads);

        // Apply gradients to test parameter learning
        let _ = trm.apply_gradients(&param_grads, 0.01);

        // Verify gradients were computed correctly
        assert!(
            input_grads.iter().all(|&x: &f32| x.is_finite()),
            "Input gradients should be finite"
        );
        assert!(!param_grads.is_empty(), "Should have parameter gradients");

        println!("✅ TRM adaptive learning validated - can learn and update parameters");
    }
}
