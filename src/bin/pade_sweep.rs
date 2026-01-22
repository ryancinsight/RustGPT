use std::time::Instant;

use llm::pade::PadeExp;

fn horner(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc.mul_add(x, c))
}

fn pade_5_5(x: f64) -> f64 {
    const P: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
    const Q: [f64; 6] = [30240.0, -15120.0, 3360.0, -420.0, 30.0, -1.0];
    horner(&P, x) / horner(&Q, x)
}

fn max_rel_error_linear(min_x: f64, max_x: f64, n: usize) -> (f64, f64) {
    let step = (max_x - min_x) / (n.saturating_sub(1) as f64);
    let mut worst = 0.0;
    let mut worst_x = min_x;

    for i in 0..n {
        let x = min_x + (i as f64) * step;
        let a = PadeExp::exp(x);
        let b = x.exp();

        if a.is_finite() && b.is_finite() && b != 0.0 {
            let rel = ((a - b) / b).abs();
            if rel > worst {
                worst = rel;
                worst_x = x;
            }
        }
    }

    (worst, worst_x)
}

fn main() {
    let ranges = [
        (-0.15, 0.15, 20001, "ultra-small"),
        (-0.4, 0.4, 20001, "small"),
        (-0.8, 0.8, 20001, "medium"),
        (-1.2, 1.2, 20001, "large-ish"),
        (-20.0, 20.0, 20001, "range-reduction"),
        (-100.0, 0.0, 20001, "softmax-like negative"),
    ];

    println!("PadeExp sweep (compare to std::exp)");
    for (min_x, max_x, n, label) in ranges {
        let (max_rel, worst_x) = max_rel_error_linear(min_x, max_x, n);
        println!(
            "{label:>22}: x∈[{min_x:>7.3},{max_x:>7.3}] max_rel={max_rel:.3e} at x={worst_x:.6}"
        );
    }

    // Subnormal band sanity: exp(x) should be > 0 and < MIN_POSITIVE for part of it
    let x = -740.0;
    let y = PadeExp::exp(x);
    println!(
        "subnormal check: exp({x}) = {y:e} (MIN_POSITIVE={:e})",
        f64::MIN_POSITIVE
    );

    // Spot-check a few points
    for &x in &[-0.2f64, -0.15f64, -0.1f64, 0.1f64, 0.15f64, 0.2f64, 1.2f64] {
        let a = PadeExp::exp(x);
        let b = x.exp();
        let rel = if b != 0.0 { ((a - b) / b).abs() } else { 0.0 };
        let p55 = pade_5_5(x);
        let rel55 = if b != 0.0 { ((p55 - b) / b).abs() } else { 0.0 };
        println!("x={x:>6.3}  pade={a:.17e}  std={b:.17e}  rel={rel:.3e}  p55_rel={rel55:.3e}");
    }

    // Micro-benchmark (very rough)
    let iters: usize = 2_000_000;
    let mut acc = 0.0;
    let start = Instant::now();
    for i in 0..iters {
        let x = -10.0 + 20.0 * ((i as f64) / (iters as f64));
        acc += PadeExp::exp(x);
    }
    let dt = start.elapsed();
    let ns = dt.as_nanos() as f64 / (iters as f64);
    println!("pade exp avg: {ns:.2} ns/call (acc={acc:.3e})");
}
