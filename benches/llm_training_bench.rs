use criterion::{criterion_group, criterion_main, Criterion};
use llm::LLM;

fn bench_train_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("llm_training");

    // Create a small model for benchmarking
    let mut llm = LLM::default();

    let data = vec![
        "Hello world this is a test sequence",
        "Another test sequence for training",
        "Performance optimization is fun",
        "Rust is a great language",
        "Deep learning models are cool",
        "Optimizing code is important",
    ];

    // We use a small epoch count (1) and batch size (2)
    // The focus is on the training loop performance.

    group.bench_function("train_1_epoch", |b| {
        b.iter(|| {
            // Clone data to avoid consumption if necessary,
            // though train_with_batch_size takes Vec<&str> which copies references.
            // But we can just pass data.clone()
            let _ = llm.train_with_batch_size(data.clone(), 1, 0.001, 2);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_train_batch);
criterion_main!(benches);
