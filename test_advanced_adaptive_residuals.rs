use ndarray::Array2;
use rand::thread_rng;
use rand_distr;

use crate::transformer::transformer_block::SimilarityMetric;

/// Test the advanced adaptive residuals functionality independently
#[test]
fn test_weight_similarity_computation() {
    // Create test weight matrices
    let embed_dim = 8;
    let seq_len = 4;

    let mut rng = thread_rng();

    // Create mock attention and FFN weights
    let attention_weights = Array2::from_shape_fn((seq_len, embed_dim), |_| {
        rand_distr::StandardNormal.sample(&mut rng)
    });

    let ffn_weights = Array2::from_shape_fn((seq_len, embed_dim), |_| {
        rand_distr::StandardNormal.sample(&mut rng)
    });

    // Test similarity metrics would be added here once implementation is ready
    // For now, just validate the basic structure

    assert_eq!(attention_weights.shape(), [seq_len, embed_dim]);
    assert_eq!(ffn_weights.shape(), [seq_len, embed_dim]);

    println!("✓ Weight similarity computation test placeholder passed");
}

#[test]
fn test_similarity_metrics() {
    use crate::transformer::transformer_block::AdvancedAdaptiveResiduals;

    // Test cosine similarity
    let a = Array2::from(vec![1.0, 2.0, 3.0]);
    let b = Array2::from(vec![1.0, 2.0, 3.0]);

    let similarity = AdvancedAdaptiveResiduals::cosine_similarity(&a, &b);
    assert!((similarity - 1.0).abs() < 1e-6); // Should be exactly 1.0 for identical vectors

    // Test different vectors
    let c = Array2::from(vec![1.0, 0.0, 0.0]);
    let d = Array2::from(vec![0.0, 1.0, 0.0]);

    let similarity2 = AdvancedAdaptiveResiduals::cosine_similarity(&c, &d);
    assert!(similarity2 < 0.1); // Should be near 0 for orthogonal vectors

    println!("✓ Cosine similarity tests passed");
}

#[test]
fn test_pearson_correlation() {
    use crate::transformer::transformer_block::AdvancedAdaptiveResiduals;

    // Test perfectly correlated vectors
    let a = Array2::from(vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array2::from(vec![2.0, 4.0, 6.0, 8.0]); // 2*a

    let correlation = AdvancedAdaptiveResiduals::pearson_correlation(&a, &b);
    assert!((correlation - 1.0).abs() < 1e-6);

    // Test anti-correlated vectors
    let c = Array2::from(vec![1.0, 2.0, 3.0, 4.0]);
    let d = Array2::from(vec![-1.0, -2.0, -3.0, -4.0]); // -1*c

    let correlation2 = AdvancedAdaptiveResiduals::pearson_correlation(&c, &d);
    assert!((correlation2 - (-1.0)).abs() < 1e-6);

    println!("✓ Pearson correlation tests passed");
}
