use std::time::Instant;
use ndarray::Array2;
use llm::Layer;
use llm::model_config::ModelConfig;
use llm::transformer::transformer_block::TransformerBlock;

fn main() {
    let cfg = ModelConfig::transformer(256, 512, 3, 512, Some(256), Some(8));
    let mut block = TransformerBlock::from_model_config(&cfg, 0);
    let n = 256usize;
    let d = 256usize;
    let input = Array2::<f32>::zeros((n, d));
    let warmup = 10;
    for _ in 0..warmup { let _ = block.forward(&input); }
    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters { let _ = block.forward(&input); }
    let elapsed = start.elapsed().as_secs_f64();
    let tokens = (n * iters) as f64;
    let tps = tokens / elapsed;
    println!("throughput_tokens_per_sec={}, elapsed_seconds={}", tps, elapsed);
}