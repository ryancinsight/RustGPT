use std::{
    fs,
    io::{BufRead, Seek},
};

use csv::ReaderBuilder;

use crate::errors::{ModelError, Result};

pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

#[allow(clippy::upper_case_acronyms)]
pub enum DatasetType {
    JSON,
    CSV,
}

impl Dataset {
    pub fn new(
        pretraining_data_path: String,
        chat_training_data_path: String,
        type_of_data: DatasetType,
    ) -> Result<Self> {
        let pretraining_data: Vec<String>;
        let chat_training_data: Vec<String>;

        match type_of_data {
            DatasetType::CSV => {
                pretraining_data = get_data_from_csv(&pretraining_data_path)?;
                chat_training_data = get_data_from_csv(&chat_training_data_path)?;
            }
            DatasetType::JSON => {
                pretraining_data = get_data_from_json(&pretraining_data_path)?;
                chat_training_data = get_data_from_json(&chat_training_data_path)?;
            }
        }

        Ok(Dataset {
            pretraining_data,
            chat_training_data,
        })
    }
}

#[derive(serde::Deserialize)]
struct TextRow {
    text: String,
}

fn get_data_from_json(path: &str) -> Result<Vec<String>> {
    // File size validation
    let metadata = fs::metadata(path).map_err(ModelError::from)?;
    if metadata.len() > crate::MAX_FILE_SIZE {
        return Err(ModelError::InvalidInput {
            message: format!(
                "File size {} exceeds maximum allowed size {}",
                metadata.len(),
                crate::MAX_FILE_SIZE
            ),
        });
    }

    // convert json file to Vec<String>
    let file = fs::File::open(path).map_err(ModelError::from)?;
    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);

    match serde_json::from_reader::<_, Vec<String>>(&mut reader) {
        Ok(strict) => Ok(strict),
        Err(_) => {
            reader.seek(std::io::SeekFrom::Start(0))?;

            // Optimization: Try to parse as array of objects with "text" field directly
            // This avoids the overhead of parsing into generic Value enums
            if let Ok(rows) = serde_json::from_reader::<_, Vec<TextRow>>(&mut reader) {
                return Ok(rows.into_iter().map(|r| r.text).collect());
            }

            reader.seek(std::io::SeekFrom::Start(0))?;

            let parsed = serde_json::from_reader::<_, Vec<serde_json::Value>>(&mut reader);
            if let Ok(vals) = parsed {
                let mut out: Vec<String> = Vec::new();
                for v in vals {
                    match v {
                        serde_json::Value::String(s) => out.push(s),
                        serde_json::Value::Object(map) => {
                            if let Some(serde_json::Value::String(s)) = map.get("text") {
                                out.push(s.clone());
                            }
                        }
                        _ => {}
                    }
                }
                if !out.is_empty() {
                    return Ok(out);
                }
            }

            reader.seek(std::io::SeekFrom::Start(0))?;

            let mut items = Vec::new();
            for line in (&mut reader).lines() {
                let line = line.map_err(ModelError::from)?;
                let t = line.trim();
                if t.is_empty() || t == "," || t == "[" || t == "]" {
                    continue;
                }
                if t.starts_with('"') {
                    let mut s = t.trim_end_matches(',').to_string();
                    if s.starts_with('"') && s.ends_with('"') {
                        s = s[1..s.len() - 1].to_string();
                    }
                    items.push(s);
                }
            }
            if items.is_empty() {
                reader.seek(std::io::SeekFrom::Start(0))?;
                serde_json::from_reader::<_, Vec<String>>(&mut reader).map_err(|e| {
                    ModelError::Serialization {
                        source: Box::new(e),
                    }
                })
            } else {
                tracing::warn!(
                    path = path,
                    count = items.len(),
                    "Loaded JSON via relaxed parser (found formatting artifacts)"
                );
                Ok(items)
            }
        }
    }
}

fn get_data_from_csv(path: &str) -> Result<Vec<String>> {
    // File size validation
    let metadata = fs::metadata(path).map_err(ModelError::from)?;
    if metadata.len() > crate::MAX_FILE_SIZE {
        return Err(ModelError::InvalidInput {
            message: format!(
                "File size {} exceeds maximum allowed size {}",
                metadata.len(),
                crate::MAX_FILE_SIZE
            ),
        });
    }

    // convert csv file to Vec<String>
    let file = fs::File::open(path).map_err(ModelError::from)?;
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result.map_err(|e| ModelError::DatasetLoad {
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
        })?;
        // Each record is a row, join all columns into a single string
        let capacity =
            record.iter().map(|s| s.len()).sum::<usize>() + record.len().saturating_sub(1);
        let mut line = String::with_capacity(capacity);
        for (i, field) in record.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(field);
        }
        data.push(line);
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn test_parse_array_of_strings() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "[\"a\",\"b\",\"c\"]").unwrap();
        let path = f.path().to_str().unwrap();
        let data = get_data_from_json(path).unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0], "a");
    }

    #[test]
    fn test_parse_array_of_objects() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "[{{\"text\":\"hello\"}},{{\"text\":\"world\"}}]").unwrap();
        let path = f.path().to_str().unwrap();
        let data = get_data_from_json(path).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], "hello");
        assert_eq!(data[1], "world");
    }
}
