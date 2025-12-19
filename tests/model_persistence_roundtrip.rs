use llm::{llm::LLM, model_persistence::VersionedModel, Layer};

#[test]
fn versioned_model_binary_roundtrip_smoke() {
    let llm = LLM::default();

    // "binary" payload now uses MessagePack (see VersionedModel::from_llm)
    let versioned = VersionedModel::from_llm(&llm, "binary", Some("test".to_string()))
        .expect("binary serialization should succeed");

    let loaded = versioned
        .to_llm("binary")
        .expect("binary deserialization should succeed");

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
