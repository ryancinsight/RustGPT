use std::io::Write;

use criterion::{Criterion, criterion_group, criterion_main};
use llm::infrastructure::persistence::dataset::{Dataset, DatasetType};
use tempfile::NamedTempFile;

fn create_csv_file(rows: usize) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("failed to create temp file");
    for i in 0..rows {
        writeln!(file, "{},{},{},{},{}", i, i + 1, i + 2, i + 3, i + 4)
            .expect("failed to write to file");
    }
    file
}

fn bench_csv_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_loading");

    let csv_file = create_csv_file(10_000);
    let path = csv_file.path().to_str().unwrap().to_string();

    group.bench_function("csv_loading_10k_rows", |b| {
        b.iter(|| Dataset::new(path.clone(), path.clone(), DatasetType::CSV).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_csv_loading);
criterion_main!(benches);
