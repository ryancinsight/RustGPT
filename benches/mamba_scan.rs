use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llm::layers::ssm::{Mamba, MambaConfig};
use ndarray::Array2;

fn bench_mamba_forward_enhanced_scan(c: &mut Criterion) {
    let t = 2048usize;
    let d = 128usize;

    let input = Array2::from_shape_fn((t, d), |(ti, j)| {
        ((ti as f32 * 0.01 + j as f32 * 0.02).sin() * 0.5).tanh()
    });

    let mut layer_seq = Mamba::new_with_config(d, 3, MambaConfig::default());
    let mut layer_par = Mamba::new_with_config(d, 3, MambaConfig::enhanced());

    // Warm up to allocate caches and projections.
    let _ = layer_seq.forward_enhanced(&input);
    let _ = layer_par.forward_enhanced(&input);

    c.bench_function("mamba_forward_enhanced_sequential_scan", |b| {
        b.iter(|| {
            let out = layer_seq.forward_enhanced(black_box(&input));
            black_box(out)
        })
    });

    c.bench_function("mamba_forward_enhanced_parallel_scan", |b| {
        b.iter(|| {
            let out = layer_par.forward_enhanced(black_box(&input));
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_mamba_forward_enhanced_scan);
criterion_main!(benches);
