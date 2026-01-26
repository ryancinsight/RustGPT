use criterion::{criterion_group, criterion_main, Criterion};
use llm::models::llm::LLM;
use llm::Vocab;
use llm::model_config::ModelConfig;
use llm::model_builder::build_network;

fn bench_generation(c: &mut Criterion) {
    let mut config = ModelConfig::default();
    config.max_seq_len = 128;
    config.embedding_dim = 64;
    config.hidden_dim = 128;
    config.num_layers = 2;
    config.num_heads = Some(4);

    // Vocab::default() usually has a few words.
    let vocab = Vocab::default();
    let network = build_network(&config, &vocab);
    let mut llm = LLM::new(vocab, network);

    // Switch to inference mode for layers (if any)
    llm.set_trm_inference_mode();

    let input_text = "hello world";

    c.bench_function("generate_50_tokens", |b| {
        b.iter(|| {
            // We want to measure the generation loop overhead
            // predict_with_limit tokenizes and runs forward
            llm.predict_with_limit(input_text, 50);
        })
    });
}

criterion_group!(benches, bench_generation);
criterion_main!(benches);
