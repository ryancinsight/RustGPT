use llm::domain::layers::transformer::{TransformerBlock, TransformerBlockConfig};
use llm::domain::mixtures::moh::HeadSelectionStrategy;
use llm::domain::models::config::{TemporalMixingType, TitanMemoryConfig, WindowAdaptationStrategy};
use llm::domain::network::Layer;
use llm::domain::richards::adaptive::AdaptiveScalar;
use ndarray::Array2;
use rand::Rng;

use llm::domain::layers::components::common::TemporalMixingLayer;

fn run_consistency_test(mixing_type: TemporalMixingType, use_moe: bool, threshold: f32) {
    let embed_dim = 16;
    let hidden_dim = 32;
    let num_heads = 2;
    let seq_len = 20;
    
    let config = TransformerBlockConfig {
        embed_dim,
        hidden_dim,
        num_heads,
        poly_degree: 3,
        max_pos: 100,
        window_size: Some(4), // Short window
        use_moe,
        moe_config: None,
        head_selection: HeadSelectionStrategy::Fixed { num_active: num_heads },
        moh_threshold_modulation: AdaptiveScalar::default(),
        temporal_mixing: mixing_type,
        use_adaptive_window: false,
        min_window_size: 2,
        max_window_size: 10,
        window_adaptation_strategy: WindowAdaptationStrategy::Fixed,
        entropy_ema_alpha: 0.1,
        use_advanced_adaptive_residuals: false, // Start with basic
        titan_memory: TitanMemoryConfig {
            enabled: true, // Test Titan Memory
            segment_len: 1, // Force segment length 1 for streaming consistency check
            scale: 0.1,
            eta: 0.5,
            decay: 0.1,
            ..TitanMemoryConfig::default()
        },
        eprop_adaptor: None,
    };

    let block = TransformerBlock::new(config);

    let mut rng = rand::rng();
    let input_data: Vec<f32> = (0..seq_len * embed_dim).map(|_| rng.random_range(-0.5..0.5)).collect();
    let input = Array2::from_shape_vec((seq_len, embed_dim), input_data).unwrap();

    // 1. Batch Forward
    let mut block_batch = block.clone();
    let batch_output = block_batch.forward(&input);

    // Capture global normalization parameters from batch run
    let norm1_overrides = block_batch.pre_attention_norm().cached_adjusted_richards().map(|r| {
        (r.temperature, r.m, r.beta)
    });
    let norm2_overrides = block_batch.pre_ffn_norm().cached_adjusted_richards().map(|r| {
        (r.temperature, r.m, r.beta)
    });

    // Capture MoH overrides from batch run (for SSMs)
    let moh_overrides = match &block_batch.temporal_mixing().temporal_mixing {
        TemporalMixingLayer::RgLruMoH(rglru) => rglru.get_last_max_abs_z(),
        TemporalMixingLayer::MambaMoH(mamba) => mamba.get_last_max_abs_z(),
        _ => None,
    };

    // 2. Streaming Forward
    let mut block_streaming = block.clone();
    let mut stream_outputs = Vec::new();
    for i in 0..seq_len {
        let input_step = input.row(i).to_owned();
        let out_step = block_streaming.forward_step_with_overrides(
            &input_step,
            norm1_overrides,
            norm2_overrides,
            moh_overrides.clone(),
        );
        stream_outputs.push(out_step);
    }

    // 3. Compare
    let mut max_diff = 0.0;
    for i in 0..seq_len {
        let batch_row = batch_output.row(i);
        let stream_row = &stream_outputs[i];
        
        let diff = (&batch_row - stream_row).mapv(|x: f32| x.abs()).sum();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max diff: {}", max_diff);
    assert!(max_diff < threshold, "Streaming output diverged from batch output. Max diff: {}", max_diff);
}

#[test]
fn test_transformer_block_streaming_consistency_attention() {
    run_consistency_test(TemporalMixingType::Attention, false, 1e-4);
}

#[test]
fn test_transformer_block_streaming_consistency_rglru() {
    run_consistency_test(TemporalMixingType::RgLru, false, 1e-4);
}

#[test]
fn test_transformer_block_streaming_consistency_rglru_moh() {
    run_consistency_test(TemporalMixingType::RgLru, true, 1e-4);
}

#[test]
fn test_transformer_block_streaming_consistency_mamba() {
    run_consistency_test(TemporalMixingType::Mamba, false, 1e-4);
}

#[test]
fn test_transformer_block_streaming_consistency_mamba_moh() {
    run_consistency_test(TemporalMixingType::Mamba, true, 1e-4);
}

#[test]
fn test_transformer_block_streaming_consistency_mamba2() {
    run_consistency_test(TemporalMixingType::Mamba2, false, 1e-4);
}

#[test]
fn test_transformer_block_streaming_consistency_mamba2_moh() {
    run_consistency_test(TemporalMixingType::Mamba2, true, 1e-4);
}

#[test]
fn test_transformer_block_streaming_consistency_titans() {
    run_consistency_test(TemporalMixingType::Titans, false, 1e-4);
}
