// Quick test to verify PolyAttention workspace consolidation compiles
// This is just a smoke test - not meant to be comprehensive

#[cfg(test)]
mod test_poly_workspace {
    use rustgpt::domain::attention::PolyAttention;
    use rustgpt::domain::attention::position::config::{CoPEConfig, CoPEVariant};
    use rustgpt::domain::layers::components::WorkspaceManaged;
    use rustgpt::domain::layers::components::StreamingWorkspaceManaged;

    #[test]
    fn test_poly_attention_workspace_managed() {
        let mut pa = PolyAttention::new(
            32,
            4,
            3,
            CoPEConfig {
                variant: CoPEVariant::Standard,
                max_pos: 256,
                window_size: Some(64),
            },
        );

        // Test WorkspaceManaged trait
        pa.ensure_capacity(2, 64, 32);
        let stats = pa.workspace_stats();
        assert_eq!(stats.buffer_count, 0); // May be 0 if unified_workspace initializes empty

        // Test that we can clear
        pa.clear_workspace();
    }

    #[test]
    fn test_poly_attention_streaming_managed() {
        let mut pa = PolyAttention::new(
            32,
            4,
            3,
            CoPEConfig {
                variant: CoPEVariant::Standard,
                max_pos: 256,
                window_size: Some(64),
            },
        );

        // Test StreamingWorkspaceManaged trait
        pa.init_streaming(1, 32).unwrap();
        assert!(pa.is_streaming());

        // Test reset
        pa.reset_streaming_state();
        assert!(pa.is_streaming()); // Still streaming after reset
    }
}
