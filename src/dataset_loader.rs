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
            pretraining_data: pretraining_data.clone(),
            chat_training_data: chat_training_data.clone(),
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
    let data_json = fs::read_to_string(path).map_err(ModelError::from)?;
    let data: Vec<String> =
        serde_json::from_str(&data_json).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;
    Ok(data)
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
