use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llm::{
    Layer,
    mixtures::HeadSelectionStrategy,
    transformer::diffusion_block::{
        DiffusionBlock,
        DiffusionBlockConfig,
        DiffusionPredictionTarget,
        NoiseSchedule,
    },
};
use ndarray::Array2;

fn bench_forward(c: &mut Criterion) {
    let config = DiffusionBlockConfig {
        embed_dim: 128,
        hidden_dim: 256,
        num_heads: 8,
        poly_degree: 3,
        max_pos: 127,
        window_size: None,
        use_moe: false,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed { num_active: 8 },
        time_embed_dim: 128,
        num_timesteps: 1000,
        noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
        causal_attention: false,
        discrete_masked: false,
        mask_token_id: None,
        prediction_target: DiffusionPredictionTarget::default(),
    };
    let mut block = DiffusionBlock::new(config);
    block.set_timestep(500);
    let input = Array2::<f32>::zeros((32, 128));
    c.bench_function("diffusion_block_forward", |b| {
        b.iter(|| {
            let out = block.forward(black_box(&input));
            black_box(out)
        })
    });
}

fn bench_sample(c: &mut Criterion) {
    let config = DiffusionBlockConfig {
        embed_dim: 64,
        hidden_dim: 128,
        num_heads: 4,
        poly_degree: 3,
        max_pos: 63,
        window_size: None,
        use_moe: false,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
        time_embed_dim: 64,
        num_timesteps: 200,
        noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
        causal_attention: false,
        discrete_masked: false,
        mask_token_id: None,
        prediction_target: DiffusionPredictionTarget::default(),
    };
    let mut block = DiffusionBlock::new(config);
    c.bench_function("diffusion_block_sample_50", |b| {
        b.iter(|| {
            let x = block.sample(black_box((8, 64)), black_box(Some(50)));
            black_box(x)
        })
    });
}

criterion_group!(benches, bench_forward, bench_sample);
criterion_main!(benches);
