//! Shared block-core assembly for transformer-like layer stacks.
//!
//! This consolidates construction of the common pre-attention norm, temporal
//! mixing, pre-FFN norm, and feedforward wrappers used by both Transformer and
//! Diffusion blocks.

use crate::domain::{
    layers::components::{
        common::{CommonLayerConfig, CommonLayers},
        feedforward::SharedFeedforward,
        temporal_processing::SharedTemporalProcessing,
    },
    richards::RichardsNorm,
};

/// Shared core layers used by transformer and diffusion blocks.
#[derive(Debug)]
pub struct SharedBlockCore {
    pub pre_attention_norm: RichardsNorm,
    pub temporal_mixing: SharedTemporalProcessing,
    pub pre_ffn_norm: RichardsNorm,
    pub feedforward: SharedFeedforward,
}

/// Build the shared block core from a common layer config.
#[inline]
pub fn build_shared_block_core(
    common_config: &CommonLayerConfig,
    window_size: Option<usize>,
    use_adaptive_window: bool,
) -> SharedBlockCore {
    let layers = CommonLayers::new(common_config);
    SharedBlockCore {
        pre_attention_norm: layers.pre_attention_norm,
        temporal_mixing: SharedTemporalProcessing::new(
            layers.temporal_mixing,
            window_size,
            use_adaptive_window,
        ),
        pre_ffn_norm: layers.pre_ffn_norm,
        feedforward: SharedFeedforward::new(layers.feedforward),
    }
}
