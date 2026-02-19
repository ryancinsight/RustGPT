use std::{
    collections::HashSet,
    fs,
    io::{BufRead, Seek},
    path::Path,
};

use csv::ReaderBuilder;

use crate::common::errors::{ModelError, Result};

/// Multi-modal dataset supporting text, image, video, and audio data
pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
    /// Image training data (captions and conversations)
    pub image_training_data: Vec<ImageExample>,
    /// Video training data (captions and conversations)
    pub video_training_data: Vec<VideoExample>,
    /// Speech/audio training data (transcripts and conversations)
    pub speech_training_data: Vec<SpeechExample>,
}

/// Image training example with caption and optional conversations
#[derive(Debug, Clone)]
pub struct ImageExample {
    pub image_id: String,
    pub caption: String,
    pub objects: Vec<String>,
    pub conversations: Vec<ConversationTurn>,
}

/// Video training example with caption and optional conversations
#[derive(Debug, Clone)]
pub struct VideoExample {
    pub video_id: String,
    pub duration_seconds: f32,
    pub num_frames: usize,
    pub caption: String,
    pub actions: Vec<String>,
    pub conversations: Vec<ConversationTurn>,
}

/// Speech/audio training example with transcript and optional conversations
#[derive(Debug, Clone)]
pub struct SpeechExample {
    pub audio_id: String,
    pub duration_seconds: f32,
    pub transcript: String,
    pub speaker: String,
    pub language: String,
    pub conversations: Vec<ConversationTurn>,
}

/// A single conversation turn
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub from: String,
    pub value: String,
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
        let mut pretraining_data: Vec<String>;
        let mut chat_training_data: Vec<String>;
        let is_json = matches!(type_of_data, DatasetType::JSON);

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

        if is_json {
            let rust_path = "data/rust_programming_training_data.json";
            if chat_training_data_path != rust_path && Path::new(rust_path).exists() {
                let rust_data = get_data_from_json(rust_path)?;
                chat_training_data.extend(rust_data);
            }
            let tool_path = "data/tool_calling_training_data.json";
            if chat_training_data_path != tool_path && Path::new(tool_path).exists() {
                let tool_data = get_data_from_json(tool_path)?;
                chat_training_data.extend(tool_data);
            }
        }

        pretraining_data = normalize_and_dedup_text_data(pretraining_data);
        chat_training_data = normalize_and_dedup_text_data(chat_training_data);

        Ok(Dataset {
            pretraining_data,
            chat_training_data,
            image_training_data: Vec::new(),
            video_training_data: Vec::new(),
            speech_training_data: Vec::new(),
        })
    }

    /// Load multi-modal datasets from JSON files
    pub fn with_multimodal(
        pretraining_data_path: String,
        chat_training_data_path: String,
        type_of_data: DatasetType,
    ) -> Result<Self> {
        let mut dataset = Self::new(pretraining_data_path, chat_training_data_path, type_of_data)?;

        // Load image training data if available
        let image_path = "data/image_training_data.json";
        if Path::new(image_path).exists() {
            dataset.image_training_data = load_image_data(image_path)?;
            tracing::info!(
                count = dataset.image_training_data.len(),
                "Loaded image training data"
            );
        }

        // Load video training data if available
        let video_path = "data/video_training_data.json";
        if Path::new(video_path).exists() {
            dataset.video_training_data = load_video_data(video_path)?;
            tracing::info!(
                count = dataset.video_training_data.len(),
                "Loaded video training data"
            );
        }

        // Load speech training data if available
        let speech_path = "data/speech_training_data.json";
        if Path::new(speech_path).exists() {
            dataset.speech_training_data = load_speech_data(speech_path)?;
            tracing::info!(
                count = dataset.speech_training_data.len(),
                "Loaded speech training data"
            );
        }

        Ok(dataset)
    }

    /// Load multi-modal datasets with actual data from MNIST and Speech Commands
    pub fn with_real_multimodal_data(
        pretraining_data_path: String,
        chat_training_data_path: String,
        type_of_data: DatasetType,
        mnist_max_samples: Option<usize>,
        speech_max_per_class: Option<usize>,
    ) -> Result<Self> {
        let mut dataset =
            Self::with_multimodal(pretraining_data_path, chat_training_data_path, type_of_data)?;

        // Load MNIST image data
        let mnist_dir = "data/mnist";
        if Path::new(mnist_dir).exists() {
            match crate::infrastructure::persistence::mnist_loader::load_mnist_training_data(
                mnist_dir,
                mnist_max_samples,
            ) {
                Ok(mut image_examples) => {
                    tracing::info!(count = image_examples.len(), "Loaded MNIST training data");
                    dataset.image_training_data.append(&mut image_examples);
                }
                Err(e) => {
                    tracing::warn!("Failed to load MNIST data: {}", e);
                }
            }
        }

        // Load Speech Commands audio data
        let speech_dir = "data/speech_commands";
        if Path::new(speech_dir).exists() {
            match crate::infrastructure::persistence::speech_loader::load_speech_training_data(
                speech_dir,
                speech_max_per_class,
            ) {
                Ok(mut speech_examples) => {
                    tracing::info!(
                        count = speech_examples.len(),
                        "Loaded Speech Commands training data"
                    );
                    dataset.speech_training_data.append(&mut speech_examples);
                }
                Err(e) => {
                    tracing::warn!("Failed to load Speech Commands data: {}", e);
                }
            }
        }

        tracing::info!(
            images = dataset.image_training_data.len(),
            speech = dataset.speech_training_data.len(),
            video = dataset.video_training_data.len(),
            "Total multimodal training data loaded"
        );

        Ok(dataset)
    }

    /// Get all text data from all modalities for pretraining
    pub fn get_all_text_data(&self) -> Vec<String> {
        let mut all_text = Vec::new();
        all_text.extend(self.pretraining_data.clone());

        // Add image captions
        for img in &self.image_training_data {
            all_text.push(img.caption.clone());
            for conv in &img.conversations {
                all_text.push(conv.value.clone());
            }
        }

        // Add video captions
        for vid in &self.video_training_data {
            all_text.push(vid.caption.clone());
            for conv in &vid.conversations {
                all_text.push(conv.value.clone());
            }
        }

        // Add speech transcripts
        for aud in &self.speech_training_data {
            all_text.push(aud.transcript.clone());
            for conv in &aud.conversations {
                all_text.push(conv.value.clone());
            }
        }

        all_text
    }

    /// Check if multi-modal data is available
    pub fn has_multimodal_data(&self) -> bool {
        !self.image_training_data.is_empty()
            || !self.video_training_data.is_empty()
            || !self.speech_training_data.is_empty()
    }
}

fn normalize_text_entry(raw: &str) -> Option<String> {
    // Normalize whitespace and trim relaxed JSON artifacts.
    let normalized = raw
        .replace('\r', " ")
        .replace('\n', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    let mut text = normalized.trim().to_string();
    if text.is_empty() {
        return None;
    }

    // Enforce sentence boundary token for consistency across corpora.
    if !text.ends_with("</s>") {
        text.push_str(" </s>");
    }

    Some(text)
}

fn normalize_and_dedup_text_data(data: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::<String>::new();
    let mut out = Vec::with_capacity(data.len());

    for item in data {
        if let Some(text) = normalize_text_entry(&item) {
            let key = text.to_lowercase();
            if seen.insert(key) {
                out.push(text);
            }
        }
    }

    out
}

#[derive(serde::Deserialize)]
struct TextRow {
    text: String,
}

/// JSON structure for image training data
#[derive(serde::Deserialize)]
struct ImageDataJson {
    #[serde(default)]
    examples: Vec<ImageExampleJson>,
}

#[derive(serde::Deserialize)]
struct ImageExampleJson {
    image_id: String,
    caption: String,
    #[serde(default)]
    objects: Vec<String>,
    #[serde(default)]
    conversations: Vec<ConversationJson>,
}

/// JSON structure for video training data
#[derive(serde::Deserialize)]
struct VideoDataJson {
    #[serde(default)]
    examples: Vec<VideoExampleJson>,
}

#[derive(serde::Deserialize)]
struct VideoExampleJson {
    video_id: String,
    #[serde(default)]
    duration_seconds: f32,
    #[serde(default)]
    num_frames: usize,
    caption: String,
    #[serde(default)]
    actions: Vec<String>,
    #[serde(default)]
    conversations: Vec<ConversationJson>,
}

/// JSON structure for speech training data
#[derive(serde::Deserialize)]
struct SpeechDataJson {
    #[serde(default)]
    examples: Vec<SpeechExampleJson>,
}

#[derive(serde::Deserialize)]
struct SpeechExampleJson {
    audio_id: String,
    #[serde(default)]
    duration_seconds: f32,
    transcript: String,
    #[serde(default)]
    speaker: String,
    #[serde(default)]
    language: String,
    #[serde(default)]
    conversations: Vec<ConversationJson>,
}

#[derive(serde::Deserialize)]
struct ConversationJson {
    from: String,
    value: String,
}

fn load_image_data(path: &str) -> Result<Vec<ImageExample>> {
    let file = fs::File::open(path).map_err(ModelError::from)?;
    let reader = std::io::BufReader::with_capacity(1024 * 1024, file);

    let data: ImageDataJson =
        serde_json::from_reader(reader).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;

    Ok(data
        .examples
        .into_iter()
        .map(|ex| ImageExample {
            image_id: ex.image_id,
            caption: ex.caption,
            objects: ex.objects,
            conversations: ex
                .conversations
                .into_iter()
                .map(|c| ConversationTurn {
                    from: c.from,
                    value: c.value,
                })
                .collect(),
        })
        .collect())
}

fn load_video_data(path: &str) -> Result<Vec<VideoExample>> {
    let file = fs::File::open(path).map_err(ModelError::from)?;
    let reader = std::io::BufReader::with_capacity(1024 * 1024, file);

    let data: VideoDataJson =
        serde_json::from_reader(reader).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;

    Ok(data
        .examples
        .into_iter()
        .map(|ex| VideoExample {
            video_id: ex.video_id,
            duration_seconds: ex.duration_seconds,
            num_frames: ex.num_frames,
            caption: ex.caption,
            actions: ex.actions,
            conversations: ex
                .conversations
                .into_iter()
                .map(|c| ConversationTurn {
                    from: c.from,
                    value: c.value,
                })
                .collect(),
        })
        .collect())
}

fn load_speech_data(path: &str) -> Result<Vec<SpeechExample>> {
    let file = fs::File::open(path).map_err(ModelError::from)?;
    let reader = std::io::BufReader::with_capacity(1024 * 1024, file);

    let data: SpeechDataJson =
        serde_json::from_reader(reader).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;

    Ok(data
        .examples
        .into_iter()
        .map(|ex| SpeechExample {
            audio_id: ex.audio_id,
            duration_seconds: ex.duration_seconds,
            transcript: ex.transcript,
            speaker: ex.speaker,
            language: ex.language,
            conversations: ex
                .conversations
                .into_iter()
                .map(|c| ConversationTurn {
                    from: c.from,
                    value: c.value,
                })
                .collect(),
        })
        .collect())
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

    #[test]
    fn test_normalize_text_entry_adds_eos_and_collapses_whitespace() {
        let got = normalize_text_entry("  User: Hi\nAssistant: Hello   ").unwrap();
        assert_eq!(got, "User: Hi Assistant: Hello </s>");
    }

    #[test]
    fn test_normalize_and_dedup_text_data_preserves_first_entry() {
        let data = vec![
            "One line".to_string(),
            " one   line </s> ".to_string(),
            "Another".to_string(),
        ];
        let out = normalize_and_dedup_text_data(data);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], "One line </s>");
        assert_eq!(out[1], "Another </s>");
    }
}
