//! Audio/Speech processing module for multi-modal training.
//!
//! Implements audio understanding through spectrogram-based patch embedding,
//! supporting both waveform and spectral representations for speech and sound.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::common::errors::Result;
use crate::domain::multimodal::patch::{get_1d_sincos_pos_embed, PatchEmbed};

/// Configuration for audio processing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: usize,
    /// Maximum duration in seconds
    pub max_duration: f32,
    /// FFT window size for spectrogram
    pub n_fft: usize,
    /// Hop length for spectrogram
    pub hop_length: usize,
    /// Number of mel frequency bins (0 to disable mel scale)
    pub n_mels: usize,
    /// Patch size along time dimension
    pub temporal_patch_size: usize,
    /// Patch size along frequency dimension
    pub freq_patch_size: usize,
    /// Target embedding dimension
    pub embedding_dim: usize,
    /// Use time-frequency position embeddings
    #[serde(default = "default_true")]
    pub use_position_embeddings: bool,
    /// Pre-emphasis coefficient (0.0 to disable)
    #[serde(default = "preemphasis_default")]
    pub preemphasis: f32,
    /// Normalize audio to [-1, 1]
    #[serde(default = "default_true")]
    pub normalize_audio: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            max_duration: 30.0,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            temporal_patch_size: 16,
            freq_patch_size: 10,
            embedding_dim: 768,
            use_position_embeddings: true,
            preemphasis: 0.97,
            normalize_audio: true,
        }
    }
}

impl AudioConfig {
    /// Calculate maximum number of samples
    pub fn max_samples(&self) -> usize {
        (self.sample_rate as f32 * self.max_duration) as usize
    }

    /// Calculate number of time frames in spectrogram
    pub fn num_time_frames(&self) -> usize {
        let max_samples = self.max_samples();
        (max_samples / self.hop_length) + 1
    }

    /// Calculate frequency dimension
    pub fn freq_dim(&self) -> usize {
        if self.n_mels > 0 {
            self.n_mels
        } else {
            self.n_fft / 2 + 1
        }
    }

    /// Calculate number of temporal patches
    pub fn num_temporal_patches(&self) -> usize {
        self.num_time_frames() / self.temporal_patch_size
    }

    /// Calculate number of frequency patches
    pub fn num_freq_patches(&self) -> usize {
        self.freq_dim() / self.freq_patch_size
    }

    /// Calculate total number of patches
    pub fn total_patches(&self) -> usize {
        self.num_temporal_patches() * self.num_freq_patches()
    }

    /// Calculate the flattened patch dimension
    pub fn patch_dim(&self) -> usize {
        self.temporal_patch_size * self.freq_patch_size
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "sample_rate must be > 0".to_string(),
            });
        }
        if self.max_duration <= 0.0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "max_duration must be > 0".to_string(),
            });
        }
        if self.n_fft == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "n_fft must be > 0".to_string(),
            });
        }
        if self.hop_length == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "hop_length must be > 0".to_string(),
            });
        }
        if self.temporal_patch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "temporal_patch_size must be > 0".to_string(),
            });
        }
        if self.freq_patch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "freq_patch_size must be > 0".to_string(),
            });
        }
        let time_frames = self.num_time_frames();
        if time_frames % self.temporal_patch_size != 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "num_time_frames ({}) must be divisible by temporal_patch_size ({})",
                    time_frames, self.temporal_patch_size
                ),
            });
        }
        let freq_dim = self.freq_dim();
        if freq_dim % self.freq_patch_size != 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "freq_dim ({}) must be divisible by freq_patch_size ({})",
                    freq_dim, self.freq_patch_size
                ),
            });
        }
        if self.embedding_dim == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "embedding_dim must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// A single audio sample
#[derive(Debug, Clone)]
pub struct AudioSample {
    /// Raw waveform samples
    pub waveform: Vec<f32>,
    /// Sample rate
    pub sample_rate: usize,
    /// Optional transcript or description
    pub transcript: Option<String>,
    /// Duration in seconds (may be estimated)
    pub duration: f32,
}

impl AudioSample {
    /// Create a new audio sample
    pub fn new(waveform: Vec<f32>, sample_rate: usize) -> Self {
        let duration = waveform.len() as f32 / sample_rate as f32;
        Self {
            waveform,
            sample_rate,
            transcript: None,
            duration,
        }
    }

    /// Create a new audio sample with transcript
    pub fn with_transcript(
        waveform: Vec<f32>,
        sample_rate: usize,
        transcript: String,
    ) -> Self {
        let duration = waveform.len() as f32 / sample_rate as f32;
        Self {
            waveform,
            sample_rate,
            transcript: Some(transcript),
            duration,
        }
    }

    /// Resample to target sample rate using linear interpolation
    pub fn resample(&self, target_rate: usize) -> Vec<f32> {
        if self.sample_rate == target_rate {
            return self.waveform.clone();
        }

        let ratio = target_rate as f32 / self.sample_rate as f32;
        let new_len = (self.waveform.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f32 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(self.waveform.len() - 1);
            let frac = src_idx - idx0 as f32;

            let v0 = self.waveform.get(idx0).copied().unwrap_or(0.0);
            let v1 = self.waveform.get(idx1).copied().unwrap_or(0.0);

            resampled.push(v0 * (1.0 - frac) + v1 * frac);
        }

        resampled
    }

    /// Apply pre-emphasis filter: y[t] = x[t] - alpha * x[t-1]
    pub fn preemphasis(&self, alpha: f32) -> Vec<f32> {
        if alpha == 0.0 || self.waveform.is_empty() {
            return self.waveform.clone();
        }

        let mut emphasized = Vec::with_capacity(self.waveform.len());
        emphasized.push(self.waveform[0]);

        for i in 1..self.waveform.len() {
            emphasized.push(self.waveform[i] - alpha * self.waveform[i - 1]);
        }

        emphasized
    }

    /// Normalize waveform to [-1, 1]
    pub fn normalize(&mut self) {
        let max_abs = self
            .waveform
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if max_abs > 0.0 {
            for x in &mut self.waveform {
                *x /= max_abs;
            }
        }
    }

    /// Pad or truncate to target length
    pub fn pad_or_truncate(&mut self, target_samples: usize) {
        if self.waveform.len() > target_samples {
            self.waveform.truncate(target_samples);
        } else if self.waveform.len() < target_samples {
            self.waveform.resize(target_samples, 0.0);
        }
        self.duration = self.waveform.len() as f32 / self.sample_rate as f32;
    }
}

/// Audio encoder using spectrogram-based patch embedding
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioEncoder {
    pub config: AudioConfig,
    pub patch_embed: PatchEmbed,
    #[serde(skip, default)]
    pub position_embeddings: Option<Array2<f32>>,
}

impl AudioEncoder {
    /// Create a new audio encoder
    pub fn new(config: AudioConfig) -> Result<Self> {
        config.validate()?;

        let patch_dim = config.patch_dim();
        let patch_embed = PatchEmbed::new(patch_dim, config.embedding_dim)?;

        let position_embeddings = if config.use_position_embeddings {
            let num_patches = config.total_patches();
            Some(get_1d_sincos_pos_embed(config.embedding_dim, num_patches))
        } else {
            None
        };

        Ok(Self {
            config,
            patch_embed,
            position_embeddings,
        })
    }

    /// Encode an audio sample into embeddings
    pub fn encode(&self, sample: &AudioSample) -> Result<Array2<f32>> {
        // Resample if necessary
        let waveform = if sample.sample_rate != self.config.sample_rate {
            sample.resample(self.config.sample_rate)
        } else {
            sample.waveform.clone()
        };

        // Pre-emphasis
        let waveform = if self.config.preemphasis > 0.0 {
            let temp_sample = AudioSample::new(waveform, self.config.sample_rate);
            temp_sample.preemphasis(self.config.preemphasis)
        } else {
            waveform
        };

        // Compute spectrogram
        let spectrogram = self.compute_spectrogram(&waveform)?;

        // Extract patches from spectrogram
        let patches = self.extract_patches(&spectrogram)?;

        // Embed patches
        let embeddings = self.patch_embed.forward(&patches);

        // Add position embeddings
        if let Some(ref pos_emb) = self.position_embeddings {
            Ok(embeddings + pos_emb)
        } else {
            Ok(embeddings)
        }
    }

    /// Compute magnitude spectrogram
    fn compute_spectrogram(&self, waveform: &[f32]) -> Result<Array2<f32>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let max_samples = self.config.max_samples();

        // Truncate or pad waveform
        let mut signal = waveform.to_vec();
        if signal.len() > max_samples {
            signal.truncate(max_samples);
        } else if signal.len() < max_samples {
            signal.resize(max_samples, 0.0);
        }

        let num_frames = (signal.len() - n_fft) / hop_length + 1;
        let freq_bins = n_fft / 2 + 1;

        let mut spectrogram = Array2::zeros((num_frames, freq_bins));

        // Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / n_fft as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        // Compute STFT
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;

            // Extract and window frame
            let frame: Vec<f32> = signal[start..start + n_fft]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Pad to power of 2 if necessary (simplified - using naive DFT for clarity)
            // In production, use FFT
            for freq_idx in 0..freq_bins {
                let freq = freq_idx as f32;
                let mut real = 0.0f32;
                let mut imag = 0.0f32;

                for (t, &sample) in frame.iter().enumerate() {
                    let angle = -2.0 * std::f32::consts::PI * freq * t as f32 / n_fft as f32;
                    real += sample * angle.cos();
                    imag += sample * angle.sin();
                }

                // Magnitude
                spectrogram[[frame_idx, freq_idx]] = (real * real + imag * imag).sqrt();
            }
        }

        // Convert to mel scale if configured
        if self.config.n_mels > 0 {
            spectrogram = self.to_mel_scale(&spectrogram)?;
        }

        // Log scale
        spectrogram.mapv_inplace(|x| (x + 1e-10).ln());

        Ok(spectrogram)
    }

    /// Convert linear spectrogram to mel scale
    fn to_mel_scale(&self, spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        let n_mels = self.config.n_mels;
        let num_frames = spectrogram.nrows();
        let freq_bins = spectrogram.ncols();

        // Create mel filterbank
        let mel_fb = self.create_mel_filterbank(freq_bins, n_mels);

        // Apply filterbank
        let mut mel_spec = Array2::zeros((num_frames, n_mels));
        for frame in 0..num_frames {
            for mel in 0..n_mels {
                let mut sum = 0.0f32;
                for freq in 0..freq_bins {
                    sum += spectrogram[[frame, freq]] * mel_fb[mel][freq];
                }
                mel_spec[[frame, mel]] = sum;
            }
        }

        Ok(mel_spec)
    }

    /// Create mel filterbank
    fn create_mel_filterbank(&self, n_freqs: usize, n_mels: usize) -> Vec<Vec<f32>> {
        let sample_rate = self.config.sample_rate as f32;
        let f_min = 0.0f32;
        let f_max = sample_rate / 2.0;

        // Mel scale conversion
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);

        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];

        for mel in 0..n_mels {
            let f_left = hz_points[mel];
            let f_center = hz_points[mel + 1];
            let f_right = hz_points[mel + 2];

            for freq in 0..n_freqs {
                let f = freq as f32 * sample_rate / (2.0 * (n_freqs - 1) as f32);

                if f >= f_left && f <= f_center {
                    filterbank[mel][freq] = (f - f_left) / (f_center - f_left);
                } else if f > f_center && f <= f_right {
                    filterbank[mel][freq] = (f_right - f) / (f_right - f_center);
                }
            }
        }

        filterbank
    }

    /// Extract patches from spectrogram
    fn extract_patches(&self, spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        let num_time_frames = spectrogram.nrows();
        let freq_bins = spectrogram.ncols();

        let p_t = self.config.temporal_patch_size;
        let p_f = self.config.freq_patch_size;

        let num_t_patches = num_time_frames / p_t;
        let num_f_patches = freq_bins / p_f;
        let total_patches = num_t_patches * num_f_patches;
        let patch_dim = p_t * p_f;

        let mut patches = Array2::zeros((total_patches, patch_dim));

        let mut patch_idx = 0;
        for tp in 0..num_t_patches {
            for fp in 0..num_f_patches {
                let mut patch_data = Vec::with_capacity(patch_dim);

                for t in 0..p_t {
                    for f in 0..p_f {
                        let time_idx = tp * p_t + t;
                        let freq_idx = fp * p_f + f;
                        patch_data.push(spectrogram[[time_idx, freq_idx]]);
                    }
                }

                patches
                    .row_mut(patch_idx)
                    .assign(&Array1::from_vec(patch_data));
                patch_idx += 1;
            }
        }

        Ok(patches)
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        self.config.total_patches()
    }
}

fn default_true() -> bool {
    true
}

fn preemphasis_default() -> f32 {
    0.97
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(duration_secs: f32, sample_rate: usize) -> AudioSample {
        let num_samples = (duration_secs * sample_rate as f32) as usize;
        let waveform: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();
        AudioSample::new(waveform, sample_rate)
    }

    #[test]
    fn test_audio_config_validation() {
        // Use a config with compatible patch sizes
        let config = AudioConfig {
            max_duration: 1.0,
            sample_rate: 8000,
            n_fft: 256,
            hop_length: 128,
            n_mels: 64, // 64 bins divides evenly by freq_patch_size of 8
            temporal_patch_size: 63, // 63 time frames / 63 = 1
            freq_patch_size: 8,
            embedding_dim: 64,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        let bad_config = AudioConfig {
            sample_rate: 0,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_audio_sample_operations() {
        let mut sample = create_test_audio(1.0, 16000);
        assert_eq!(sample.waveform.len(), 16000);

        sample.normalize();
        let max_abs = sample.waveform.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((max_abs - 1.0).abs() < 1e-6 || max_abs == 0.0);

        sample.pad_or_truncate(8000);
        assert_eq!(sample.waveform.len(), 8000);
    }

    #[test]
    fn test_resample() {
        let sample = create_test_audio(1.0, 16000);
        let resampled = sample.resample(8000);
        assert_eq!(resampled.len(), 8000);
    }

    #[test]
    fn test_audio_encoder_creation() {
        // Config with compatible divisibility:
        // max_samples = 8000 * 1.0 = 8000
        // num_time_frames = (8000 / 128) + 1 = 63
        // freq_dim = n_fft / 2 + 1 = 129 (without mel)
        // Use n_mels = 64 which divides evenly by freq_patch_size = 8
        let config = AudioConfig {
            max_duration: 1.0,
            sample_rate: 8000,
            n_fft: 256,
            hop_length: 128,
            n_mels: 64, // Use mel scale so freq_dim = 64, which divides by 8
            temporal_patch_size: 63, // 63 time frames divides by 63
            freq_patch_size: 8,
            embedding_dim: 64,
            ..Default::default()
        };

        let encoder = AudioEncoder::new(config).unwrap();
        assert!(encoder.output_dim() > 0);
    }

    #[test]
    fn test_spectrogram_computation() {
        // max_samples = 8000 * 0.5 = 4000
        // num_time_frames = (4000 / 128) + 1 = 31 + 1 = 32
        // For mel spectrogram: freq_dim = n_mels = 64
        // We need freq_dim (64) to divide by freq_patch_size (8), and time_frames (32) by temporal_patch_size
        let config = AudioConfig {
            max_duration: 0.5,
            sample_rate: 8000,
            n_fft: 256,
            hop_length: 128,
            n_mels: 64, // Use mel scale: freq_dim = 64, divides by 8
            temporal_patch_size: 32, // 32 time frames divides by 32
            freq_patch_size: 8,
            embedding_dim: 32,
            ..Default::default()
        };

        let encoder = AudioEncoder::new(config).unwrap();
        let sample = create_test_audio(0.5, 8000);

        let waveform = sample.waveform.clone();
        let spec = encoder.compute_spectrogram(&waveform).unwrap();

        assert!(spec.nrows() > 0);
        assert!(spec.ncols() > 0);
    }
}
