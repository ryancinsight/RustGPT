use std::fs;

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
    let data_json_raw = fs::read_to_string(path).map_err(ModelError::from)?;

    // First attempt: strict JSON parsing
    match serde_json::from_str::<Vec<String>>(&data_json_raw) {
        Ok(strict) => return Ok(strict),
        Err(_) => {
            // Fallback: relaxed line-based parsing to handle comma-only lines or split commas
            let mut items = Vec::new();
            for line in data_json_raw.lines() {
                let t = line.trim();
                if t.is_empty() || t == "," || t == "[" || t == "]" { continue; }
                // Accept lines like "...", or "...",
                if t.starts_with('"') {
                    let mut s = t.trim_end_matches(',').to_string();
                    // Remove surrounding quotes
                    if s.starts_with('"') && s.ends_with('"') {
                        s = s[1..s.len()-1].to_string();
                    }
                    items.push(s);
                }
            }
            if items.is_empty() {
                // If still empty, return original error for visibility
                return serde_json::from_str::<Vec<String>>(&data_json_raw).map_err(|e| ModelError::Serialization { source: Box::new(e) });
            }
            tracing::warn!(path = path, count = items.len(), "Loaded JSON via relaxed parser (found formatting artifacts)");
            Ok(items)
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
        data.push(record.iter().collect::<Vec<_>>().join(","));
    }
    Ok(data)
}
