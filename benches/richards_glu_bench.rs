use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llm::richards::RichardsGlu;
use llm::network::Layer;
use ndarray::Array2;
use rand::prelude::*;

fn bench_richards_glu_compute_gradients(c: &mut Criterion) {
    // Setup
    let embedding_dim = 128;
    let hidden_dim = 512;
    let batch_size = 64;

    let mut layer = RichardsGlu::new(embedding_dim, hidden_dim);
    let mut rng = rand::rng();

    let input = Array2::from_shape_fn((batch_size, embedding_dim), |_| rng.random::<f32>());
    let output_grads = Array2::from_shape_fn((batch_size, embedding_dim), |_| rng.random::<f32>());

    // Perform one forward pass to populate cache
    layer.forward(&input);

    c.bench_function("RichardsGlu::compute_gradients", |b| {
        b.iter(|| {
            // We need to re-populate cache if compute_gradients consumes it?
            // Checking implementation: compute_gradients uses cache but doesn't consume it (cloned() currently).
            // But if I change implementation to not clone, it will borrow.
            // Wait, cached fields are Option<Array2>.
            // forward sets them.
            // compute_gradients reads them.

            layer.compute_gradients(black_box(&input), black_box(&output_grads))
        })
    });
}

criterion_group!(benches, bench_richards_glu_compute_gradients);
criterion_main!(benches);
