use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llm::richards::{RichardsCurve, Variant};

fn bench_update_scaling(c: &mut Criterion) {
    let mut curve = RichardsCurve::new_learnable(Variant::Sigmoid);
    // Make it "heavy"
    curve.enable_per_feature_transform(1024); // Allocates gamma/bias arrays

    // Fill grad_norm_history
    for i in 0..100 {
        curve.grad_norm_history.push(i as f64);
    }

    // Ensure optimizer is initialized (it is in new_learnable)

    // Set scale and shift to fixed values to trigger the optimization path
    curve.scale = Some(1.0);
    curve.shift = Some(0.0);

    c.bench_function("update_scaling_from_max_abs", |b| {
        b.iter(|| {
            // max_abs_x value doesn't matter much for allocation cost, but let's use a value that triggers update
            let updated = curve.update_scaling_from_max_abs(black_box(2.0));
            black_box(updated);
        })
    });
}

criterion_group!(benches, bench_update_scaling);
criterion_main!(benches);
