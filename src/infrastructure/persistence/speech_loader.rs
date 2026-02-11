//! Speech Commands dataset loader for audio training.
//!
//! Loads Google Mini Speech Commands audio files (.wav) and converts them
//! to AudioSample for multimodal training.

use std::fs::{self, File};
use std::io::{Read, BufReader};
use std::path::Path;

use crate::common::errors::{ModelError, Result};
use crate::domain::multimodal::audio::AudioSample;
use crate::infrastructure::persistence::dataset::SpeechExample;
use crate::infrastructure::persistence::loader::DatasetLoader;

/// Command categories in the mini speech commands dataset
pub const SPEECH_COMMANDS: &[&str] = &[
    "down", "go", "left", "no", "right", "stop", "up", "yes",
];

/// Configuration for speech data loading
#[derive(Debug, Clone)]
pub struct SpeechConfig {
    /// Target sample rate (Hz)
    pub target_sample_rate: usize,
    /// Maximum duration in seconds
    pub max_duration_secs: f32,
    /// Whether to normalize audio
    pub normalize: bool,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16000,
            max_duration_secs: 1.0,
            normalize: true,
        }
    }
}

/// Speech dataset loader
pub struct SpeechLoader {
    config: SpeechConfig,
    max_samples_per_class: Option<usize>,
}

impl SpeechLoader {
    pub fn new(config: SpeechConfig, max_samples_per_class: Option<usize>) -> Self {
        Self {
            config,
            max_samples_per_class,
        }
    }
}

impl DatasetLoader for SpeechLoader {
    type Item = Vec<SpeechExample>;

    fn load<P: AsRef<Path>>(&self, source: P) -> Result<Self::Item> {
        load_speech_commands(source, &self.config, self.max_samples_per_class)
    }
}

/// Load speech commands dataset from directory
pub fn load_speech_commands<P: AsRef<Path>>(
    data_dir: P,
    config: &SpeechConfig,
    max_samples_per_class: Option<usize>,
) -> Result<Vec<SpeechExample>> {
    let base_path = data_dir.as_ref().join("mini_speech_commands");
    
    if !base_path.exists() {
        return Err(ModelError::InvalidInput {
            message: format!("Speech commands directory not found: {:?}", base_path),
        });
    }
    
    let mut examples = Vec::new();
    
    for command in SPEECH_COMMANDS {
        let command_path = base_path.join(command);
        if !command_path.exists() {
            tracing::warn!("Command directory not found: {:?}", command_path);
            continue;
        }
        
        let entries = fs::read_dir(&command_path).map_err(|e| ModelError::InvalidInput {
            message: format!("Failed to read directory {:?}: {}", command_path, e),
        })?;
        
        let mut count = 0;
        for entry in entries {
            if let Some(limit) = max_samples_per_class {
                if count >= limit {
                    break;
                }
            }
            
            let entry = entry.map_err(|e| ModelError::InvalidInput {
                message: format!("Failed to read entry: {}", e),
            })?;
            
            let path = entry.path();
            if path.extension().map(|e| e == "wav").unwrap_or(false) {
                match load_wav_file(&path, config) {
                    Ok(audio_sample) => {
                        let example = create_speech_example(&path, command, audio_sample);
                        examples.push(example);
                        count += 1;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load {:?}: {}", path, e);
                    }
                }
            }
        }
        
        tracing::info!("Loaded {} samples for command '{}'", count, command);
    }
    
    tracing::info!("Total speech examples loaded: {}", examples.len());
    Ok(examples)
}

/// Load a WAV file and convert to AudioSample
fn load_wav_file<P: AsRef<Path>>(path: P, config: &SpeechConfig) -> Result<AudioSample> {
    let file = File::open(&path).map_err(|e| ModelError::InvalidInput {
        message: format!("Failed to open WAV file: {}", e),
    })?;
    
    let mut reader = BufReader::new(file);
    
    // Read WAV header
    let header = WavHeader::read(&mut reader)?;
    
    // Verify format
    if header.audio_format != 1 {
        return Err(ModelError::InvalidInput {
            message: format!("Unsupported WAV format: {}", header.audio_format),
        });
    }
    
    // Read sample data
    let mut sample_data = vec![0u8; header.data_size];
    reader.read_exact(&mut sample_data).map_err(|e| ModelError::InvalidInput {
        message: format!("Failed to read WAV data: {}", e),
    })?;
    
    // Convert to f32 samples
    let mut samples: Vec<f32> = match header.bits_per_sample {
        8 => sample_data.iter().map(|&s| (s as f32 - 128.0) / 128.0).collect(),
        16 => {
            sample_data
                .chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32;
                    sample / 32768.0
                })
                .collect()
        }
        24 => {
            sample_data
                .chunks_exact(3)
                .map(|chunk| {
                    let sample = if chunk[2] & 0x80 != 0 {
                        // Negative - sign extend
                        ((chunk[0] as i32) | ((chunk[1] as i32) << 8) | ((chunk[2] as i32) << 16) | (-16777216i32)) as f32
                    } else {
                        ((chunk[0] as i32) | ((chunk[1] as i32) << 8) | ((chunk[2] as i32) << 16)) as f32
                    };
                    sample / 8388608.0
                })
                .collect()
        }
        32 => {
            sample_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }
        b => {
            return Err(ModelError::InvalidInput {
                message: format!("Unsupported bits per sample: {}", b),
            });
        }
    };
    
    // Resample if necessary
    if header.sample_rate != config.target_sample_rate as u32 {
        samples = resample(&samples, header.sample_rate, config.target_sample_rate as u32);
    }
    
    // Pad or truncate to target duration
    let target_samples = (config.target_sample_rate as f32 * config.max_duration_secs) as usize;
    if samples.len() > target_samples {
        samples.truncate(target_samples);
    } else if samples.len() < target_samples {
        samples.resize(target_samples, 0.0);
    }
    
    let mut audio_sample = AudioSample::new(samples, config.target_sample_rate);
    audio_sample.transcript = Some(
        path.as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
    );
    
    if config.normalize {
        audio_sample.normalize();
    }
    
    Ok(audio_sample)
}

/// Simple resampling using linear interpolation
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    
    let ratio = to_rate as f32 / from_rate as f32;
    let new_len = (samples.len() as f32 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);
    
    for i in 0..new_len {
        let src_idx = i as f32 / ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = src_idx - idx0 as f32;
        
        let v0 = samples.get(idx0).copied().unwrap_or(0.0);
        let v1 = samples.get(idx1).copied().unwrap_or(0.0);
        
        resampled.push(v0 * (1.0 - frac) + v1 * frac);
    }
    
    resampled
}

/// WAV file header structure
#[derive(Debug)]
struct WavHeader {
    audio_format: u16,
    #[allow(dead_code)]
    num_channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    data_size: usize,
}

impl WavHeader {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let mut buffer = [0u8; 44]; // Standard WAV header size
        reader.read_exact(&mut buffer).map_err(|e| ModelError::InvalidInput {
            message: format!("Failed to read WAV header: {}", e),
        })?;
        
        // Verify "RIFF" signature
        if &buffer[0..4] != b"RIFF" {
            return Err(ModelError::InvalidInput {
                message: "Invalid WAV file: missing RIFF signature".to_string(),
            });
        }
        
        // Verify "WAVE" format
        if &buffer[8..12] != b"WAVE" {
            return Err(ModelError::InvalidInput {
                message: "Invalid WAV file: missing WAVE signature".to_string(),
            });
        }
        
        // Find "fmt " chunk
        let mut offset = 12;
        if &buffer[offset..offset+4] != b"fmt " {
            // Handle JUNK chunk or other chunks before fmt
            while offset < 40 {
                let chunk_id = &buffer[offset..offset+4];
                let chunk_size = u32::from_le_bytes([
                    buffer[offset+4], buffer[offset+5], buffer[offset+6], buffer[offset+7]
                ]) as usize;
                
                if chunk_id == b"fmt " {
                    break;
                }
                offset += 8 + chunk_size;
                if offset + 8 > buffer.len() {
                    break;
                }
            }
        }
        
        // Parse format chunk
        let audio_format = u16::from_le_bytes([buffer[offset+8], buffer[offset+9]]);
        let num_channels = u16::from_le_bytes([buffer[offset+10], buffer[offset+11]]);
        let sample_rate = u32::from_le_bytes([
            buffer[offset+12], buffer[offset+13], buffer[offset+14], buffer[offset+15]
        ]);
        let bits_per_sample = u16::from_le_bytes([buffer[offset+22], buffer[offset+23]]);
        
        // Find "data" chunk
        let mut data_offset = 36;
        while data_offset < 40 {
            if &buffer[data_offset..data_offset+4] == b"data" {
                break;
            }
            data_offset += 1;
        }
        
        if data_offset >= 40 {
            // Data chunk might be later, try to read more
            return Err(ModelError::InvalidInput {
                message: "Could not find data chunk in WAV header".to_string(),
            });
        }
        
        let data_size = u32::from_le_bytes([
            buffer[data_offset+4], buffer[data_offset+5], buffer[data_offset+6], buffer[data_offset+7]
        ]) as usize;
        
        Ok(Self {
            audio_format,
            num_channels,
            sample_rate,
            bits_per_sample,
            data_size,
        })
    }
}

/// Create a SpeechExample from an audio sample
fn create_speech_example<P: AsRef<Path>>(
    path: P,
    command: &str,
    audio: AudioSample,
) -> SpeechExample {
    let file_name = path.as_ref()
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    
    SpeechExample {
        audio_id: format!("speech_{}_{}", command, file_name),
        duration_seconds: audio.duration,
        transcript: command.to_string(),
        speaker: format!("unknown_{}", command),
        language: "en".to_string(),
        conversations: vec![
            crate::infrastructure::persistence::dataset::ConversationTurn {
                from: "human".to_string(),
                value: "What word is spoken in this audio?".to_string(),
            },
            crate::infrastructure::persistence::dataset::ConversationTurn {
                from: "gpt".to_string(),
                value: format!("The spoken word is '{}'.", command),
            },
        ],
    }
}

/// Load speech training data with automatic discovery
pub fn load_speech_training_data(
    data_dir: &str,
    max_samples_per_class: Option<usize>,
) -> Result<Vec<SpeechExample>> {
    let config = SpeechConfig::default();
    load_speech_commands(data_dir, &config, max_samples_per_class)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resample() {
        let samples = vec![0.0f32, 0.5, 1.0, 0.5, 0.0];
        let resampled = resample(&samples, 16000, 8000);
        
        // Should have roughly half the samples
        assert!(resampled.len() >= 2 && resampled.len() <= 4);
        
        // First and last samples should be similar
        assert!((resampled[0] - samples[0]).abs() < 0.01);
    }
    
    #[test]
    fn test_speech_config_default() {
        let config = SpeechConfig::default();
        assert_eq!(config.target_sample_rate, 16000);
        assert_eq!(config.max_duration_secs, 1.0);
        assert!(config.normalize);
    }
    
    #[test]
    fn test_create_speech_example() {
        use std::path::PathBuf;
        
        let audio = AudioSample::new(vec![0.0; 16000], 16000);
        let path = PathBuf::from("/data/speech_commands/down/test.wav");
        let example = create_speech_example(&path, "down", audio);
        
        assert_eq!(example.transcript, "down");
        assert_eq!(example.language, "en");
        assert!(example.audio_id.contains("down"));
        assert!(example.audio_id.contains("test"));
        assert_eq!(example.conversations.len(), 2);
    }
}