use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llm::{Layer, trm::TRM};
use ndarray::Array2;

/// Benchmark TRM forward pass performance
fn bench_trm_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("TRM Forward Pass");

    // Test different embedding dimensions
    for embed_dim in [64, 128, 256, 512].iter() {
        // Create TRM with different sizes
        let config = llm::trm::TRMConfig {
            embed_dim: *embed_dim,
            num_recursions: 2,
            max_supervision_steps: 4,
            max_inference_steps: 2,
            use_shared_weights: true,
        };
        let mut trm = TRM::new(config);

        // Create test input
        let seq_len = 32;
        let input = Array2::from_elem((seq_len, *embed_dim), 0.1);

        group.bench_function(format!("embed_dim_{}_seq_{}", embed_dim, seq_len), |b| {
            b.iter(|| {
                let _result = trm.forward(black_box(&input));
            });
        });
    }

    group.finish();
}

/// Benchmark TRM training gradient computation
fn bench_trm_training_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("TRM Training Gradients");

    // Create TRM for training
    let config = llm::trm::TRMConfig {
        embed_dim: 128,
        num_recursions: 2,
        max_supervision_steps: 4,
        max_inference_steps: 2,
        use_shared_weights: true,
    };
    let mut trm = TRM::new(config);
    trm.set_training_mode(true);

    // Create test inputs
    let seq_len = 16;
    let question = Array2::from_elem((seq_len, 128), 0.1);
    let initial_answer = Array2::from_elem((seq_len, 128), 0.05);
    let target = Array2::from_elem((seq_len, 128), 0.2);

    group.bench_function("training_gradients", |b| {
        b.iter(|| {
            let (_loss, _grads) = trm
                .compute_training_gradients(black_box(&question), black_box(&target))
                .unwrap();
        });
    });

    group.finish();
}

/// Benchmark TRM inference mode (fewer steps)
fn bench_trm_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("TRM Inference");

    let config = llm::trm::TRMConfig {
        embed_dim: 256,
        num_recursions: 2,
        max_supervision_steps: 8,
        max_inference_steps: 1, // Only 1 step for inference
        use_shared_weights: true,
    };
    let mut trm = TRM::new(config);
    trm.set_training_mode(false); // Inference mode

    let seq_len = 64;
    let input = Array2::from_elem((seq_len, 256), 0.1);

    group.bench_function("inference_mode", |b| {
        b.iter(|| {
            let _result = trm.forward(black_box(&input));
        });
    });

    group.finish();
}

/// Benchmark TRM parameter count scaling
fn bench_trm_parameter_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("TRM Parameter Count");

    for embed_dim in [64, 128, 256, 512].iter() {
        let config = llm::trm::TRMConfig {
            embed_dim: *embed_dim,
            num_recursions: 2,
            max_supervision_steps: 4,
            max_inference_steps: 2,
            use_shared_weights: true,
        };
        let trm = TRM::new(config);

        group.bench_function(format!("param_count_embed_{}", embed_dim), |b| {
            b.iter(|| {
                let _count = black_box(trm.parameter_count());
            });
        });
    }

    group.finish();
}

/// Benchmark TRM stability with different recursion depths
fn bench_trm_recursion_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("TRM Recursion Depth");

    for recursions in [1, 2, 3, 4].iter() {
        let config = llm::trm::TRMConfig {
            embed_dim: 128,
            num_recursions: *recursions,
            max_supervision_steps: 2,
            max_inference_steps: 1,
            use_shared_weights: true,
        };
        let mut trm = TRM::new(config);

        let seq_len = 16;
        let input = Array2::from_elem((seq_len, 128), 0.1);

        group.bench_function(format!("recursions_{}", recursions), |b| {
            b.iter(|| {
                let _result = trm.forward(black_box(&input));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_trm_forward,
    bench_trm_training_gradients,
    bench_trm_inference,
    bench_trm_parameter_count,
    bench_trm_recursion_depth
);
criterion_main!(benches);
