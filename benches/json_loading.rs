use std::io::Write;
use criterion::{Criterion, criterion_group, criterion_main};
use llm::{Dataset, DatasetType};
use tempfile::NamedTempFile;

fn create_json_file(rows: usize) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("failed to create temp file");
    write!(file, "[").unwrap();
    for i in 0..rows {
        if i > 0 {
            write!(file, ",").unwrap();
        }
        // Create a reasonably long string to simulate real data
        let text = format!("This is row number {} with some dummy text to make it longer. It needs to be long enough to make memory allocation significant.", i);
        serde_json::to_writer(&file, &serde_json::json!({"text": text})).unwrap();
    }
    write!(file, "]").unwrap();
    file
}

fn bench_json_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_loading");

    // Create a file with 10k rows
    let json_file = create_json_file(10_000);
    let path = json_file.path().to_str().unwrap().to_string();

    group.bench_function("json_loading_10k_rows", |b| {
        b.iter(|| Dataset::new(path.clone(), path.clone(), DatasetType::JSON).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_json_loading);
criterion_main!(benches);
