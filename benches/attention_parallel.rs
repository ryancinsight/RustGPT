use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use ndarray::Array2;
use llm::attention::poly_attention::PolyAttention;

fn bench_attention_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_parallel_vs_baseline");
    let n = 256usize; let d = 256usize;
    let mut pa = PolyAttention::new(d, 8, 3, n, Some(n));
    pa.set_parallel_batch_size(32);
    pa.set_parallel_timeout_ms(0);
    let input = Array2::<f32>::zeros((n, d));
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("parallel_forward", |b| {
        b.iter(|| { let _ = pa.forward_impl(&input, false); });
    });
    group.bench_function("baseline_forward", |b| {
        b.iter(|| { let _ = pa.forward_impl_baseline(&input, false); });
    });
    group.finish();
}

criterion_group!(benches, bench_attention_parallel);
criterion_main!(benches);