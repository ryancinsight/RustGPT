use llm::domain::attention::poly_attention::PolyAttention;
use llm::domain::attention::position::config::{CoPEConfig, CoPEVariant};
use ndarray::Array2;

#[test]
fn parallel_vs_sequential_forward_match() {
    let cope_config = CoPEConfig {
        variant: CoPEVariant::Standard,
        max_pos: 64,
        window_size: Some(16),
    };
    let mut pa = PolyAttention::new(64, 4, 3, cope_config);
    pa.set_parallel_batch_size(16);
    pa.set_parallel_timeout_ms(0);
    let n = 32;
    let d = 64;
    let mut input = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            input[[i, j]] = ((i * j + 3) as f32 * 0.001).sin();
        }
    }
    let out_par = pa.forward_impl(&input, false);
    let out_seq = pa.forward_impl_baseline(&input, false);
    assert_eq!(out_par.shape(), out_seq.shape());
    let mut diff = 0.0f32;
    for (a, b) in out_par.iter().zip(out_seq.iter()) {
        diff += (a - b).abs();
    }
    assert!(diff < 1e-2);
}
