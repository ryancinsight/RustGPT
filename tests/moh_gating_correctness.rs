use llm::domain::mixtures::moh_gating::MoHGating;
use ndarray::Array2;
use rand::Rng;

#[test]
fn test_moh_gating_basic_correctness() {
    let embed_dim = 64;
    let num_heads = 8;
    let mut gating = MoHGating::new(embed_dim, num_heads);

    // Set up a basic strategy

    let batch_size = 10;
    let input =
        Array2::<f32>::from_shape_fn((batch_size, embed_dim), |_| rand::rng().random::<f32>());

    let weights = gating.forward_weights(&input, None, None);

    assert_eq!(weights.dim(), (batch_size, num_heads));

    // Check values are in [0, 1] (or at least non-negative)
    for v in weights.iter() {
        assert!(*v >= 0.0);
    }
}

#[test]
fn test_moh_gating_constraints() {
    let embed_dim = 64;
    let num_heads = 16;
    let mut gating = MoHGating::new(embed_dim, num_heads);

    // Configure strict constraints
    gating.head_selection_config.min_heads = 2;
    gating.head_selection_config.max_heads = 4;
    gating.head_selection_config.always_on_heads.push(0);

    let batch_size = 50;
    let input =
        Array2::<f32>::from_shape_fn((batch_size, embed_dim), |_| rand::rng().random::<f32>());

    let weights = gating.forward_weights(&input, None, None);

    // Verify constraints for each token
    for i in 0..batch_size {
        let row = weights.row(i);
        let active_count = row.iter().filter(|&&x| x > 0.0).count();

        assert!(
            active_count >= 2,
            "Token {} has {} active heads, expected >= 2",
            i,
            active_count
        );
        assert!(
            active_count <= 4,
            "Token {} has {} active heads, expected <= 4",
            i,
            active_count
        );
        assert!(
            row[0] > 0.0,
            "Always-on head 0 is not active for token {}",
            i
        );
    }
}

#[test]
fn test_moh_gating_large_batch_parallel() {
    let embed_dim = 128;
    let num_heads = 32;
    let mut gating = MoHGating::new(embed_dim, num_heads);

    // Large batch to trigger parallel execution effectively
    let batch_size = 4096;
    let input =
        Array2::<f32>::from_shape_fn((batch_size, embed_dim), |_| rand::rng().random::<f32>());

    let weights = gating.forward_weights(&input, None, None);

    assert_eq!(weights.dim(), (batch_size, num_heads));
}
