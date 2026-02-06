use std::time::Instant;

use llm::domain::attention::poly_attention::PolyAttention;
use ndarray::Array2;

fn main() {
    let mut attn = PolyAttention::new(256, 8, 3, 256, Some(256));
    let n = 256usize;
    let d = 256usize;
    let input = Array2::<f32>::zeros((n, d));
    for _ in 0..10 {
        let _ = attn.forward_impl_baseline(&input, true);
    }
    let iters = 200;
    let start_b = Instant::now();
    for _ in 0..iters {
        let _ = attn.forward_impl_baseline(&input, true);
    }
    let eb = start_b.elapsed().as_secs_f64();
    for _ in 0..10 {
        let _ = attn.forward_impl(&input, true);
    }
    let start_o = Instant::now();
    for _ in 0..iters {
        let _ = attn.forward_impl(&input, true);
    }
    let eo = start_o.elapsed().as_secs_f64();
    let tokens = (n * iters) as f64;
    println!(
        "baseline_tps={}, optimized_tps={}, speedup_pct={}",
        tokens / eb,
        tokens / eo,
        ((tokens / eo) / (tokens / eb) - 1.0) * 100.0
    );
}
