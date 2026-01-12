use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use llm::{SimpleTokenizer, Vocab};

fn bench_tokenize(c: &mut Criterion) {
    let tokenizer = SimpleTokenizer::new();

    // Small-ish vocab covering typical tokens in our simple tokenizer.
    let vocab = Vocab::new(vec![
        "hello", "world", "this", "is", "rust", "a", "b", "c", ",", ".", "!", "?", "<unk>", "</s>",
        "<mask>",
    ]);

    let texts = [
        "hello, world!",
        "this is rust.",
        "a,b,c",
        "hello world </s>",
        "unknown-token ??? hello",
        "mix: hello,world! a,b,c </s> <mask>",
    ];

    let mut group = c.benchmark_group("encoding_tokenize");
    for (i, text) in texts.iter().enumerate() {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("tokenize", i), text, |b, t| {
            b.iter(|| {
                let out = tokenizer.tokenize(black_box(t), black_box(&vocab));
                black_box(out)
            })
        });
    }
    group.finish();
}

fn bench_tokenize_into(c: &mut Criterion) {
    let tokenizer = SimpleTokenizer::new();

    let vocab = Vocab::new(vec![
        "hello", "world", "this", "is", "rust", "a", "b", "c", ",", ".", "!", "?", "<unk>", "</s>",
        "<mask>",
    ]);

    let text = "mix: hello,world! a,b,c </s> <mask> unknown unknown";

    let mut group = c.benchmark_group("encoding_tokenize_into");
    group.throughput(Throughput::Bytes(text.len() as u64));

    // Compare the in-place API (reused Vec) which should have fewer allocations.
    group.bench_function("tokenize_into_reuse_vec", |b| {
        let mut out = Vec::<usize>::with_capacity(256);
        b.iter(|| {
            tokenizer.tokenize_into(black_box(text), black_box(&vocab), black_box(&mut out));
            black_box(&out);
        })
    });

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let vocab = Vocab::new(vec![
        "hello", "world", "this", "is", "rust", "a", "b", "c", ",", ".", "!", "?", "<unk>", "</s>",
        "<mask>",
    ]);

    let token_ids: Vec<usize> = vec![
        vocab.encode("hello").unwrap(),
        vocab.encode(",").unwrap(),
        vocab.encode("world").unwrap(),
        vocab.encode("!").unwrap(),
        vocab.encode("</s>").unwrap(),
        9_999_999, // out-of-range on purpose; should fall back to <unk>
    ];

    let mut group = c.benchmark_group("encoding_decode");
    group.throughput(Throughput::Elements(token_ids.len() as u64));
    group.bench_function("decode_tokens_to_string", |b| {
        b.iter(|| {
            let s = vocab.decode_tokens_to_string(black_box(&token_ids));
            black_box(s)
        })
    });
    group.finish();
}

criterion_group!(benches, bench_tokenize, bench_tokenize_into, bench_decode);
criterion_main!(benches);
