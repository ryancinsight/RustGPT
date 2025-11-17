use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use llm::Layer;
use llm::model_config::ModelConfig;
use llm::transformer::transformer_block::{TransformerBlock, TransformerBlockConfig};

fn bench_transformer_block_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer_block_forward");
    let configs = vec![
        (128usize, 256usize, 8usize, 3usize, 256usize),
        (256usize, 512usize, 8usize, 3usize, 512usize),
        (512usize, 1024usize, 8usize, 3usize, 512usize),
    ];

    for (embed_dim, hidden_dim, num_heads, poly_degree, seq_len) in configs {
        let tcfg = TransformerBlockConfig {
            embed_dim,
            hidden_dim,
            num_heads,
            poly_degree,
            max_pos: seq_len.saturating_sub(1),
            window_size: Some(seq_len),
            use_moe: false,
            moe_config: None,
            head_selection: llm::mixtures::HeadSelectionStrategy::Fixed { num_active: num_heads },
            use_adaptive_window: false,
            min_window_size: seq_len,
            max_window_size: seq_len,
            window_adaptation_strategy: llm::model_config::WindowAdaptationStrategy::Fixed,
            entropy_ema_alpha: 0.2,
        };
        let mut block = TransformerBlock::new(tcfg);
        let input = Array2::<f32>::zeros((seq_len, embed_dim));

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{}-n{}-h{}-p{}", embed_dim, seq_len, num_heads, poly_degree)),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let _out = block.forward(&input);
                });
            },
        );
    }
    group.finish();
}

fn bench_attention_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_only_forward");
    let cfg = ModelConfig::transformer(256, 512, 3, 512, Some(512), Some(8));
    let mut block = TransformerBlock::from_model_config(&cfg, 0);
    let input = Array2::<f32>::zeros((512, 256));

    group.throughput(Throughput::Elements(512));
    group.bench_function("attention_forward", |b| {
        b.iter(|| {
            let _ = block.attention.forward(&input);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_transformer_block_forward, bench_attention_only);
criterion_main!(benches);