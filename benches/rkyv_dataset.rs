use criterion::{Criterion, criterion_group, criterion_main, BenchmarkId, Throughput, black_box};
use llm::{application::encoding::Vocab, infrastructure::persistence::rkyv_dataset::{tokenize_and_save, MemoryMappedDataset, BatchedArchiveIter}};
use std::num::NonZeroUsize;
use tempfile::NamedTempFile;

fn bench_rkyv_dataset(c: &mut Criterion) {
    let vocab_texts = vec!["hello world foo bar baz qux rust gpt benchmark test iterator".to_string()];
    let vocab = Vocab::build_from_texts(vocab_texts.iter());
    let num_examples = 10_000;
    let mut texts = Vec::with_capacity(num_examples);
    for i in 0..num_examples {
        texts.push(format!("hello world foo bar {} rust gpt benchmark test iterator", i));
    }

    let tmp_file = NamedTempFile::new().unwrap();
    tokenize_and_save(&texts, &vocab, tmp_file.path(), None).unwrap();
    
    let ds = MemoryMappedDataset::open(tmp_file.path()).unwrap();
    
    // Count total length for throughput
    let mut total_tokens = 0_usize;
    for seq in ds.iter_examples() {
        total_tokens += seq.len();
    }
    
    let mut group = c.benchmark_group("rkyv_pipeline");
    group.throughput(Throughput::Elements(total_tokens as u64));

    // Legacy approach: parse strings in loop
    group.bench_function("legacy_tokenize_on_fly", |b| {
        b.iter(|| {
            let mut acc = 0;
            for text in black_box(&texts) {
                let seq = vocab.tokenize(text);
                acc += seq.len();
            }
            black_box(acc);
        })
    });
    
    // New approach: stream memory-mapped zero copies (iter_examples)
    group.bench_function("zero_copy_iter_examples", |b| {
        b.iter(|| {
            let mut acc = 0;
            for seq in black_box(&ds).iter_examples() {
                acc += seq.len();
            }
            black_box(acc);
        })
    });

    // Batching zero-copy (e.g. MicroBatch)
    group.bench_function("zero_copy_batched", |b| {
        b.iter(|| {
            let mut acc = 0;
            let batch_iter = BatchedArchiveIter::new(black_box(&ds), 32);
            for batch in batch_iter {
                for seq in batch {
                    acc += seq.len();
                }
            }
            black_box(acc);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_rkyv_dataset);
criterion_main!(benches);
