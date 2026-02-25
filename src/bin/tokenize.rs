use clap::Parser;
use llm::{
    application::encoding::Vocab,
    common::errors::Result,
    infrastructure::persistence::{
        dataset::{Dataset, DatasetType},
        rkyv_dataset::tokenize_and_save,
    },
};

/// Pre-tokenize a training corpus and write an rkyv archive for zero-copy training.
///
/// Run this once before training:
/// ```shell
/// cargo run --bin tokenize -- \
///     --pretrain-data data/pretraining_data.json \
///     --chat-data data/chat_training_data.json \
///     --output data/corpus.rkyv
/// ```
#[derive(Parser, Debug)]
#[command(name = "tokenize")]
#[command(about = "Pre-tokenize training data and write an rkyv archive for zero-copy training")]
struct Args {
    /// Path to pretraining data JSON
    #[arg(long, default_value = "data/pretraining_data.json")]
    pretrain_data: String,

    /// Path to chat/instruction data JSON
    #[arg(long, default_value = "data/chat_training_data.json")]
    chat_data: String,

    /// Output path for the rkyv archive
    #[arg(long, default_value = "data/corpus.rkyv")]
    output: String,

    /// Number of rayon parallel chunks (default: auto)
    #[arg(long)]
    chunk_size: Option<std::num::NonZeroUsize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    tracing::info!("Loading dataset from {} / {}", args.pretrain_data, args.chat_data);
    let dataset = Dataset::new(args.pretrain_data, args.chat_data, DatasetType::JSON)?;

    // Build vocabulary from all texts
    let mut all_texts = Vec::new();
    all_texts.extend(dataset.pretraining_data.iter().cloned());
    all_texts.extend(dataset.chat_training_data.iter().cloned());

    tracing::info!(
        total_examples = all_texts.len(),
        "Building vocabulary and tokenizing corpus"
    );
    let vocab = Vocab::build_from_texts(all_texts.iter());

    tokenize_and_save(&all_texts, &vocab, &args.output, args.chunk_size)?;

    println!(
        "Tokenized {} examples → {}",
        all_texts.len(),
        args.output
    );

    Ok(())
}
