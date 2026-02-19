use llm::domain::attention::position::cope::CoPE;
use llm::domain::attention::position::factorized_cope::FactorizedCoPE;
use llm::domain::attention::position::gated_cope::GatedCoPE;
use llm::domain::attention::position::hierarchical_cope::HierarchicalCoPE;
use llm::domain::attention::position::optimized_cope::OptimizedCoPE;
use llm::domain::attention::position::path_cope::PathCoPE;
use llm::domain::attention::position::traits::PositionEmbedding;
use llm::domain::attention::position::window_aware_cope::WindowAwareCoPE;
use ndarray::{Array1, Array2};

#[test]
fn test_hierarchical_cope_trait_implementation() {
    let mut hierarchical = HierarchicalCoPE::new(4, 3, 32); // 3 chunks of size 4
    let q = Array1::from_elem(32, 0.5);
    let k = Array1::from_elem(32, 0.5);

    // Test contribution
    let contrib = hierarchical.contribution(&q.view(), &k.view(), 10, 2, None);
    assert!(contrib.is_finite());

    // Test backward
    let mut grads = hierarchical.init_gradients();
    let (dq, dk) = hierarchical.backward(&q.view(), &k.view(), 10, 2, None, 1.0, &mut grads);

    assert_eq!(dq.len(), 32);
    assert_eq!(dk.len(), 32);

    // Test apply_gradients
    hierarchical.apply_gradients(&grads, 0.01);
}

#[test]
fn test_path_cope_trait_implementation() {
    let mut path = PathCoPE::new(20, 32);
    let q = Array1::from_elem(32, 0.5);
    let k = Array1::from_elem(32, 0.5);

    // Create inputs for PathCoPE
    let inputs = Array2::from_elem((20, 32), 0.1);

    // Test contribution
    let contrib = path.contribution(&q.view(), &k.view(), 10, 5, Some(&inputs.view()));
    assert!(contrib.is_finite());

    // Test backward
    let mut grads = path.init_gradients();
    let (dq, dk) = path.backward(
        &q.view(),
        &k.view(),
        10,
        5,
        Some(&inputs.view()),
        1.0,
        &mut grads,
    );

    assert_eq!(dq.len(), 32);
    assert_eq!(dk.len(), 32);

    // Test apply_gradients
    path.apply_gradients(&grads, 0.01);
}

#[test]
fn test_optimized_cope_trait_implementation() {
    let mut optimized = OptimizedCoPE::new(20, 32, 8);
    let q = Array1::from_elem(32, 0.5);
    let k = Array1::from_elem(32, 0.5);

    // Test contribution
    let contrib = optimized.contribution(&q.view(), &k.view(), 5, 2, None);
    assert!(contrib.is_finite());

    // Test backward
    let mut grads = optimized.init_gradients();
    let (dq, dk) = optimized.backward(&q.view(), &k.view(), 5, 2, None, 1.0, &mut grads);

    assert_eq!(dq.len(), 32);
    assert_eq!(dk.len(), 32);

    // Test apply_gradients
    optimized.apply_gradients(&grads, 0.01);
}

#[test]
fn test_window_aware_cope_trait_implementation() {
    let inner = CoPE::new(20, 32);
    let mut window_aware = WindowAwareCoPE::new(inner, Some(5));
    let q = Array1::from_elem(32, 0.5);
    let k = Array1::from_elem(32, 0.5);

    // Test contribution (inside window)
    let contrib = window_aware.contribution(&q.view(), &k.view(), 4, 2, None); // pos = 2 < 5
    assert!(contrib.is_finite());

    // Test contribution (outside window)
    let contrib_out = window_aware.contribution(&q.view(), &k.view(), 10, 2, None); // pos = 8 >= 5
    assert_eq!(contrib_out, 0.0);

    // Test backward
    let mut grads = window_aware.init_gradients();
    let (dq, dk) = window_aware.backward(&q.view(), &k.view(), 4, 2, None, 1.0, &mut grads);

    assert_eq!(dq.len(), 32);
    assert_eq!(dk.len(), 32);

    // Test apply_gradients
    window_aware.apply_gradients(&grads, 0.01);
}

#[test]
fn test_gated_cope_trait_implementation() {
    let mut gated = GatedCoPE::new(20, 32);
    let q = Array1::from_elem(32, 0.5);
    let k = Array1::from_elem(32, 0.5);

    // Test contribution
    let contrib = gated.contribution(&q.view(), &k.view(), 5, 2, None);
    assert!(contrib.is_finite());

    // Test backward
    let mut grads = gated.init_gradients();
    let (dq, dk) = gated.backward(&q.view(), &k.view(), 5, 2, None, 1.0, &mut grads);

    assert_eq!(dq.len(), 32);
    assert_eq!(dk.len(), 32);

    // Test apply_gradients
    gated.apply_gradients(&grads, 0.01);
}

#[test]
fn test_factorized_cope_trait_implementation() {
    let mut factorized = FactorizedCoPE::new(20, 32, 8);
    let q = Array1::from_elem(32, 0.5);
    let k = Array1::from_elem(32, 0.5);

    // Test contribution
    let contrib = factorized.contribution(&q.view(), &k.view(), 5, 2, None);
    assert!(contrib.is_finite());

    // Test backward
    let mut grads = factorized.init_gradients();
    let (dq, dk) = factorized.backward(&q.view(), &k.view(), 5, 2, None, 1.0, &mut grads);

    assert_eq!(dq.len(), 32);
    assert_eq!(dk.len(), 32);

    // Test apply_gradients
    factorized.apply_gradients(&grads, 0.01);
}
