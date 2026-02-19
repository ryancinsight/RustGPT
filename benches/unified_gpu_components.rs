//! Performance Benchmarks for Unified GPU Components
//!
//! Measures performance of shared components across Transformer, Diffusion, and SSM.
//! Compares GPU vs CPU execution paths where applicable.
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks
//! cargo bench --bench unified_gpu_components
//!
//! # Run specific benchmark
//! cargo bench --bench unified_gpu_components -- attention
//!
//! # Run with GPU features
//! cargo bench --bench unified_gpu_components --features gpu-wgpu
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use rand::Rng;

// ============================================================================
// Benchmark Helpers
// ============================================================================

fn random_matrix(rows: usize, cols: usize) -> Array2<f32> {
    use rand::distributions::{Distribution, Uniform};
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-1.0f32, 1.0f32);
    Array2::from_shape_fn((rows, cols), |_| dist.sample(&mut rng))
}

fn random_vector(len: usize) -> Array1<f32> {
    use rand::distributions::{Distribution, Uniform};
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-1.0f32, 1.0f32);
    Array1::from_shape_fn(len, |_| dist.sample(&mut rng))
}

// ============================================================================
// Buffer Pool Benchmarks
// ============================================================================

fn bench_buffer_pool(c: &mut Criterion) {
    use rustgpt::domain::layers::components::{SharedBufferManager, UnifiedBufferPool};

    let mut group = c.benchmark_group("buffer_pool");

    // Single allocation
    group.bench_function("allocate_1kb", |b| {
        let pool = UnifiedBufferPool::new();
        b.iter(|| {
            let _buf = pool.allocate(1024).unwrap();
        });
    });

    // Repeated allocation with reuse
    group.bench_function("allocate_reuse_1kb", |b| {
        let pool = UnifiedBufferPool::new();
        b.iter(|| {
            let buf = pool.allocate(1024).unwrap();
            drop(buf);
        });
    });

    // Large allocation
    group.bench_function("allocate_1mb", |b| {
        let pool = UnifiedBufferPool::new();
        b.iter(|| {
            let _buf = pool.allocate(1024 * 1024).unwrap();
        });
    });

    // Shared buffer manager
    group.bench_function("shared_manager_unified", |b| {
        let manager = SharedBufferManager::unified();
        b.iter(|| {
            let _buf = manager.unified_pool().allocate(4096).unwrap();
        });
    });

    group.finish();
}

// ============================================================================
// Attention Context Benchmarks
// ============================================================================

fn bench_attention_context(c: &mut Criterion) {
    use rustgpt::domain::layers::components::SharedAttentionContext;

    let mut group = c.benchmark_group("attention_context");

    // Test different sizes
    for size in [64, 128, 256, 512].iter() {
        let embed_dim = *size;
        let batch_size = 32;

        group.throughput(Throughput::Elements(batch_size as u64 * embed_dim as u64));

        // CPU forward pass
        let bench_id = BenchmarkId::new("cpu_forward", size);
        group.bench_with_input(bench_id, size, |b, _| {
            let mut context = SharedAttentionContext::new(embed_dim);
            let input = random_matrix(batch_size, embed_dim);
            let strength = 1.0;

            b.iter(|| context.forward(&input, strength).unwrap());
        });
    }

    group.finish();
}

// ============================================================================
// Matrix Operations Benchmarks
// ============================================================================

fn bench_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_ops");

    // GEMM operations
    for size in [64, 128, 256, 512].iter() {
        let n = *size;

        group.throughput(Throughput::Elements((n * n * n) as u64));

        let bench_id = BenchmarkId::new("gemm", size);
        group.bench_with_input(bench_id, size, |b, _| {
            let a = random_matrix(n, n);
            let c = random_matrix(n, n);

            b.iter(|| black_box(a.dot(&c)));
        });
    }

    // Matrix-vector multiplication
    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;

        group.throughput(Throughput::Elements((n * n) as u64));

        let bench_id = BenchmarkId::new("gemv", size);
        group.bench_with_input(bench_id, size, |b, _| {
            let a = random_matrix(n, n);
            let v = random_vector(n);

            b.iter(|| black_box(a.dot(&v)));
        });
    }

    group.finish();
}

// ============================================================================
// Softmax Benchmarks
// ============================================================================

fn bench_softmax(c: &mut Criterion) {
    use rustgpt::domain::soft::softmax::softmax;

    let mut group = c.benchmark_group("softmax");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;

        group.throughput(Throughput::Elements(n as u64));

        let bench_id = BenchmarkId::new("row", size);
        group.bench_with_input(bench_id, size, |b, _| {
            let mut input = random_matrix(1, n);

            b.iter(|| {
                softmax(input.row_mut(0));
                black_box(&input)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Activation Function Benchmarks
// ============================================================================

fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        let input = random_matrix(1, n);

        group.throughput(Throughput::Elements(n as u64));

        // ReLU
        let bench_id = BenchmarkId::new("relu", size);
        group.bench_with_input(bench_id, size, |b, _| {
            b.iter(|| black_box(input.mapv(|x| x.max(0.0))));
        });

        // GELU
        let bench_id = BenchmarkId::new("gelu", size);
        group.bench_with_input(bench_id, size, |b, _| {
            b.iter(|| black_box(input.mapv(|x| 0.5 * x * (1.0 + (x * 0.7978845608).tanh()))));
        });

        // SiLU
        let bench_id = BenchmarkId::new("silu", size);
        group.bench_with_input(bench_id, size, |b, _| {
            b.iter(|| black_box(input.mapv(|x| x / (1.0 + (-x).exp()))));
        });

        // Tanh
        let bench_id = BenchmarkId::new("tanh", size);
        group.bench_with_input(bench_id, size, |b, _| {
            b.iter(|| black_box(input.mapv(|x| x.tanh())));
        });
    }

    group.finish();
}

// ============================================================================
// Layer Normalization Benchmarks
// ============================================================================

fn bench_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");

    for (batch, dim) in [(32, 256), (64, 256), (32, 512), (64, 512)].iter() {
        let input = random_matrix(*batch, *dim);
        let gamma = random_matrix(1, *dim);
        let beta = random_matrix(1, *dim);
        let eps = 1e-5;

        group.throughput(Throughput::Elements((*batch * *dim) as u64));

        let bench_id = BenchmarkId::new("forward", format!("{}_{}", batch, dim));
        group.bench_with_input(bench_id, &(*batch, *dim), |b, _| {
            b.iter(|| {
                let mut output = input.clone();
                for i in 0..*batch {
                    let row = output.row_mut(i);
                    let mean: f32 = row.sum() / *dim as f32;
                    let var: f32 =
                        row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / *dim as f32;
                    let std = (var + eps).sqrt();

                    for j in 0..*dim {
                        row[j] = (row[j] - mean) / std * gamma[[0, j]] + beta[[0, j]];
                    }
                }
                black_box(output)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Multi-Head Attention Benchmarks (CPU)
// ============================================================================

fn bench_multihead_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("multihead_attention");

    for (seq_len, embed_dim, num_heads) in [(64, 256, 4), (128, 256, 4), (64, 512, 8)].iter() {
        let batch_size = 4;
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let input = random_matrix(batch_size * seq_len, *embed_dim);
        let wq = random_matrix(*embed_dim, *embed_dim);
        let wk = random_matrix(*embed_dim, *embed_dim);
        let wv = random_matrix(*embed_dim, *embed_dim);
        let wo = random_matrix(*embed_dim, *embed_dim);

        group.throughput(Throughput::Elements(
            (batch_size * seq_len * embed_dim) as u64,
        ));

        let bench_id = BenchmarkId::new(
            "cpu_forward",
            format!("{}_{}_{}", seq_len, embed_dim, num_heads),
        );
        group.bench_with_input(bench_id, &(*seq_len, *embed_dim, *num_heads), |b, _| {
            b.iter(|| {
                // QKV projections
                let q = input.dot(&wq);
                let k = input.dot(&wk);
                let v = input.dot(&wv);

                // Simplified attention (no reshaping for multi-head)
                let scores = q.dot(&k.t()).mapv(|x| x * scale);

                // Softmax per row
                let mut attn_weights = scores.clone();
                for i in 0..attn_weights.nrows() {
                    let row = attn_weights.row_mut(i);
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
                    for j in 0..row.len() {
                        row[j] = (row[j] - max).exp() / sum;
                    }
                }

                // Attention output
                let attn_out = attn_weights.dot(&v);

                // Output projection
                let output = attn_out.dot(&wo);

                black_box(output)
            });
        });
    }

    group.finish();
}

// ============================================================================
// SSM Scan Benchmarks
// ============================================================================

fn bench_ssm_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssm_scan");

    for (seq_len, embed_dim, state_dim) in [(64, 256, 64), (128, 256, 64), (256, 512, 128)].iter() {
        let batch_size = 4;

        let input = random_matrix(batch_size * seq_len, *embed_dim);

        group.throughput(Throughput::Elements(
            (batch_size * seq_len * embed_dim) as u64,
        ));

        let bench_id = BenchmarkId::new(
            "selective_scan",
            format!("{}_{}_{}", seq_len, embed_dim, state_dim),
        );
        group.bench_with_input(bench_id, &(*seq_len, *embed_dim, *state_dim), |b, _| {
            b.iter(|| {
                let mut state = Array2::zeros((batch_size, *state_dim));
                let mut output = Array2::zeros((batch_size * seq_len, *embed_dim));

                let a_decay = 0.9f32;
                let b_scale = 0.1f32;
                let c_scale = 1.0f32;
                let d_skip = 0.5f32;

                for t in 0..*seq_len {
                    for b in 0..batch_size {
                        let t_offset = b * seq_len + t;

                        for e in 0..*embed_dim {
                            let x_t = input[[t_offset, e]];

                            for s in 0..state_dim.min(*embed_dim) {
                                let prev_state = state[[b, s]];
                                state[[b, s]] = a_decay * prev_state + b_scale * x_t;
                            }

                            let mut y_t = d_skip * x_t;
                            for s in 0..state_dim.min(*embed_dim) {
                                y_t += c_scale * state[[b, s]];
                            }

                            output[[t_offset, e]] = y_t;
                        }
                    }
                }

                black_box(output)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_buffer_pool,
    bench_attention_context,
    bench_matrix_ops,
    bench_softmax,
    bench_activations,
    bench_layer_norm,
    bench_multihead_attention,
    bench_ssm_scan,
);

criterion_main!(benches);
