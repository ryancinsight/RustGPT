use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use llm::{
    Layer,
    layers::{
        diffusion::{
            DiffusionBlock, DiffusionBlockConfig, DiffusionPredictionTarget, DiffusionSampler,
            EDM_SIGMA_DATA_DEFAULT, NoiseSchedule,
        },
        transformer::{TransformerBlock, TransformerBlockConfig},
    },
    mixtures::HeadSelectionStrategy,
    model_config::{DiffusionTimestepStrategy, TemporalMixingType, WindowAdaptationStrategy},
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
        use_adaptive_window: false,
        use_moe: false,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed { num_active: 8 },
        time_embed_dim: 128 * 4,
        num_timesteps: 1000,
        noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
        causal_attention: false,
        timestep_strategy: DiffusionTimestepStrategy::Uniform,
        temporal_mixing: TemporalMixingType::Attention,
        use_advanced_adaptive_residuals: true,
        discrete_masked: false,
        mask_token_id: None,
        prediction_target: DiffusionPredictionTarget::default(),
        edm_sigma_data: EDM_SIGMA_DATA_DEFAULT,
        sampler: DiffusionSampler::DDIM { eta: 0.0 },
        guidance: None,
        loss_weighting: Default::default(),
        use_p2_weighting: false,
        use_snr_weighting: false,
        adaptive_guidance: false,
        min_guidance_scale: 1.0,
        max_guidance_scale: 10.0,
        ddim_steps_policy: Default::default(),
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

fn bench_forward_vs_transformer(c: &mut Criterion) {
    let seq_len = 32usize;
    let embed_dim = 128usize;
    let hidden_dim = 256usize;
    let num_heads = 8usize;

    let diffusion_config = DiffusionBlockConfig {
        embed_dim,
        hidden_dim,
        num_heads,
        poly_degree: 3,
        max_pos: 127,
        window_size: None,
        use_adaptive_window: false,
        use_moe: false,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed {
            num_active: num_heads,
        },
        time_embed_dim: embed_dim * 4,
        num_timesteps: 1000,
        noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
        causal_attention: false,
        timestep_strategy: DiffusionTimestepStrategy::Uniform,
        temporal_mixing: TemporalMixingType::Attention,
        use_advanced_adaptive_residuals: true,
        discrete_masked: false,
        mask_token_id: None,
        prediction_target: DiffusionPredictionTarget::default(),
        edm_sigma_data: EDM_SIGMA_DATA_DEFAULT,
        sampler: DiffusionSampler::DDIM { eta: 0.0 },
        guidance: None,
        loss_weighting: Default::default(),
        use_p2_weighting: false,
        use_snr_weighting: false,
        adaptive_guidance: false,
        min_guidance_scale: 1.0,
        max_guidance_scale: 10.0,
        ddim_steps_policy: Default::default(),
    };

    let transformer_config = TransformerBlockConfig {
        embed_dim,
        hidden_dim,
        num_heads,
        poly_degree: 3,
        max_pos: 127,
        window_size: None,
        use_moe: false,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed {
            num_active: num_heads,
        },
        temporal_mixing: TemporalMixingType::Attention,
        use_adaptive_window: false,
        min_window_size: 16,
        max_window_size: 128,
        window_adaptation_strategy: WindowAdaptationStrategy::Fixed,
        entropy_ema_alpha: 0.1,
        use_advanced_adaptive_residuals: true,
    };

    let mut diffusion_block = DiffusionBlock::new(diffusion_config);
    diffusion_block.set_timestep(500);
    let mut transformer_block = TransformerBlock::new(transformer_config);

    let input = Array2::<f32>::zeros((seq_len, embed_dim));

    let mut group = c.benchmark_group("block_forward_tokens");
    group.throughput(Throughput::Elements(seq_len as u64));

    group.bench_function(
        BenchmarkId::new(
            "diffusion_block_forward",
            format!("seq{seq_len}_d{embed_dim}"),
        ),
        |b| {
            b.iter(|| {
                let out = diffusion_block.forward(black_box(&input));
                black_box(out)
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "transformer_block_forward",
            format!("seq{seq_len}_d{embed_dim}"),
        ),
        |b| {
            b.iter(|| {
                let out = transformer_block.forward(black_box(&input));
                black_box(out)
            })
        },
    );

    group.finish();
}

fn bench_sample(c: &mut Criterion) {
    let config = DiffusionBlockConfig {
        embed_dim: 64,
        hidden_dim: 128,
        num_heads: 4,
        poly_degree: 3,
        max_pos: 63,
        window_size: None,
        use_adaptive_window: false,
        use_moe: false,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
        time_embed_dim: 64 * 4,
        num_timesteps: 200,
        noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
        causal_attention: false,
        timestep_strategy: DiffusionTimestepStrategy::Uniform,
        temporal_mixing: TemporalMixingType::Attention,
        use_advanced_adaptive_residuals: true,
        sampler: DiffusionSampler::DDIM { eta: 0.0 },
        discrete_masked: false,
        mask_token_id: None,
        prediction_target: DiffusionPredictionTarget::default(),
        edm_sigma_data: EDM_SIGMA_DATA_DEFAULT,
        guidance: None,
        loss_weighting: Default::default(),
        use_p2_weighting: false,
        use_snr_weighting: false,
        adaptive_guidance: false,
        min_guidance_scale: 1.0,
        max_guidance_scale: 10.0,
        ddim_steps_policy: Default::default(),
    };
    let mut block = DiffusionBlock::new(config);
    c.bench_function("diffusion_block_sample_50", |b| {
        b.iter(|| {
            let x = block.sample(black_box((8, 64)), black_box(Some(50)));
            black_box(x)
        })
    });
}

criterion_group!(
    benches,
    bench_forward,
    bench_forward_vs_transformer,
    bench_sample
);
criterion_main!(benches);
