use llm::{llm::LLM, Layer};

#[test]
fn versioned_model_binary_roundtrip_smoke() {
    let llm = LLM::default();

    let path = std::env::temp_dir().join(format!(
        "rustgpt_versioned_roundtrip_{}_{}.rgpt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let path_str = path.to_str().expect("temp path should be valid UTF-8");

    llm.save_versioned(path_str, Some("test".to_string()))
        .expect("save_versioned should succeed");

    let loaded = LLM::load_versioned(path_str).expect("load_versioned should succeed");

    // Best effort cleanup.
    let _ = std::fs::remove_file(&path);

    assert_eq!(loaded.vocab.size(), llm.vocab.size());
    assert_eq!(loaded.network.len(), llm.network.len());

    // Pinpoint any layer-level mismatch (helps catch ambiguous serde decoding).
    for (idx, (a, b)) in llm.network.iter().zip(loaded.network.iter()).enumerate() {
        let a_type = a.layer_type();
        let b_type = b.layer_type();
        assert_eq!(a_type, b_type, "layer_type mismatch at index {idx}");

        let a_params = a.parameters();
        let b_params = b.parameters();
        assert_eq!(
            a_params, b_params,
            "parameters mismatch at index {idx} ({a_type}): {a_params} != {b_params}"
        );
    }

    assert_eq!(loaded.total_parameters(), llm.total_parameters());
}
