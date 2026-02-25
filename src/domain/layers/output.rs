use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{common::rng::get_rng, domain::network::Layer, infrastructure::optimizer::adam::Adam};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OutputProjection {
    pub w_out: Array2<f32>, // Weight matrix (no bias - modern LLM practice)
    pub optimizer: Adam,
    pub cached_input: Option<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    pub gpu_device: Option<std::sync::Arc<std::sync::Mutex<crate::domain::compute::GpuDevice>>>,
    #[serde(skip_serializing, skip_deserializing)]
    gpu_weight_cache: Option<crate::domain::compute::GpuBuffer>,
    #[serde(skip_serializing, skip_deserializing)]
    gpu_cached_input: Option<crate::domain::compute::GpuBuffer>,
    #[serde(skip_serializing, skip_deserializing)]
    gpu_cached_input_shape: Option<(usize, usize)>,
}

impl OutputProjection {
    /// Initialize output layer with random weights (no bias - modern LLM practice)
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = get_rng();
        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        OutputProjection {
            w_out: Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng)),
            optimizer: Adam::new((embedding_dim, vocab_size)),
            cached_input: None,
            gpu_device: None,
            gpu_weight_cache: None,
            gpu_cached_input: None,
            gpu_cached_input_shape: None,
        }
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    /// Forward pass: project embeddings to vocab logits (no bias)
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.gpu_device.is_some() {
            return self.forward_gpu(input).unwrap_or_else(|err| {
                panic!(
                    "OutputProjection GPU forward failed (GPU attached, no fallback): {}",
                    err
                )
            });
        }

        // input shape is [sequence_length, embedding_dim]
        self.cached_input = Some(input.clone());
        input.dot(&self.w_out) // shape is [sequence_length, vocab_size]
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.gpu_device.is_some() {
            let (grad_input, grad_w_out) =
                self.compute_gradients_gpu(output_grads).unwrap_or_else(|err| {
                    panic!(
                        "OutputProjection GPU compute_gradients failed (GPU attached, no fallback): {}",
                        err
                    )
                });
            return (grad_input, vec![grad_w_out]);
        }

        // grads shape is [sequence_length, vocab_size]
        let input = self.cached_input.as_ref().unwrap();
        let grad_w_out = input.t().dot(output_grads);
        let grad_input = output_grads.dot(&self.w_out.t());

        (grad_input, vec![grad_w_out])
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::common::errors::Result<()> {
        if param_grads.is_empty() {
            return Err(crate::common::errors::ModelError::GradientError {
                message: "OutputProjection expected 1 parameter gradient (weights), got 0"
                    .to_string(),
            });
        }
        let mut grad = param_grads[0].clone();
        grad.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
        let gnorm: f32 = grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let wnorm = self.weight_norm().max(1e-6);
        let clip = 5.0f32;
        let mut scale = (wnorm / gnorm.max(1e-6)).clamp(0.5, 2.0);
        if gnorm.is_finite() && gnorm > clip && gnorm > 0.0 {
            scale *= clip / gnorm;
        }
        grad.mapv_inplace(|x| x * scale);
        self.optimizer.step(&mut self.w_out, &grad, lr);
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        self.sync_gpu_weight_cache_after_update()?;
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.gpu_device.is_some() {
            return self.backward_gpu(grads, lr).unwrap_or_else(|err| {
                panic!(
                    "OutputProjection GPU backward failed (GPU attached, no fallback): {}",
                    err
                )
            });
        }

        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Unwrap is safe: backward is only called from training loop which validates inputs
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.w_out.len()
    }

    fn weight_norm(&self) -> f32 {
        let sumsq = self.w_out.iter().map(|&w| w * w).sum::<f32>();
        sumsq.sqrt()
    }

    fn zero_gradients(&mut self) {
        // OutputProjection doesn't maintain internal gradient state
        // Gradients are computed on-demand
    }
}

// ============================================================================
// GPU Component Implementation
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl crate::domain::compute::GpuComponent for OutputProjection {
    fn set_gpu_device(
        &mut self,
        device: std::sync::Arc<std::sync::Mutex<crate::domain::compute::GpuDevice>>,
    ) {
        self.clear_gpu_runtime_cache();
        self.gpu_device = Some(device);
    }

    fn enable_gpu_auto_detect(&mut self) -> crate::common::errors::Result<()> {
        self.clear_gpu_runtime_cache();
        let device = crate::domain::compute::GpuDevice::auto_detect()?;
        self.gpu_device = Some(std::sync::Arc::new(std::sync::Mutex::new(device)));
        Ok(())
    }

    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some()
    }

    fn gpu_backend_name(&self) -> Option<&'static str> {
        self.gpu_device
            .as_ref()
            .and_then(|device_arc| match device_arc.lock() {
                Ok(device) => Some(device.backend().as_str()),
                Err(_) => None,
            })
    }

    fn gpu_device(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<crate::domain::compute::GpuDevice>>> {
        self.gpu_device.clone()
    }

    fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        _seq_len: usize,
    ) -> crate::common::errors::Result<()> {
        if let Some(device_arc) = &self.gpu_device {
            let mut device =
                device_arc
                    .lock()
                    .map_err(|_| crate::common::errors::ModelError::Backend {
                        message:
                            "Failed to lock GPU device for OutputProjection capacity allocation"
                                .to_string(),
                    })?;
            let vocab_size = self.w_out.ncols();
            let _ = device.allocate_f32(batch_size * embed_dim)?; // input
            let _ = device.allocate_f32(embed_dim * vocab_size)?; // weights
            let _ = device.allocate_f32(batch_size * vocab_size)?; // output (logits)
            Ok(())
        } else {
            Err(crate::common::errors::ModelError::Backend {
                message: "GPU device not attached to OutputProjection. Call enable_gpu_auto_detect() first.".to_string(),
            })
        }
    }
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl OutputProjection {
    fn clear_gpu_runtime_cache(&mut self) {
        if let Some(device_arc) = &self.gpu_device {
            if let Ok(mut device) = device_arc.lock() {
                if let Some(buf) = self.gpu_weight_cache.take() {
                    device.deallocate(buf);
                }
                if let Some(buf) = self.gpu_cached_input.take() {
                    device.deallocate(buf);
                }
            } else {
                self.gpu_weight_cache = None;
                self.gpu_cached_input = None;
            }
        } else {
            self.gpu_weight_cache = None;
            self.gpu_cached_input = None;
        }
        self.gpu_cached_input_shape = None;
    }

    fn ensure_gpu_weight_cache_with_device(
        &mut self,
        device: &mut crate::domain::compute::GpuDevice,
    ) -> crate::common::errors::Result<crate::domain::compute::GpuBuffer> {
        if self.gpu_weight_cache.is_none() {
            let (embed_dim, vocab_size) = self.w_out.dim();
            let mut gpu_weights = device.allocate_f32(embed_dim * vocab_size)?;
            let w_slice =
                self.w_out
                    .as_slice()
                    .ok_or_else(|| crate::common::errors::ModelError::Backend {
                        message: "OutputProjection weights not contiguous".to_string(),
                    })?;
            device.upload(w_slice, &mut gpu_weights)?;
            self.gpu_weight_cache = Some(gpu_weights);
        }
        Ok(self.gpu_weight_cache.expect("weight cache initialized"))
    }

    fn sync_gpu_weight_cache_after_update(&mut self) -> crate::common::errors::Result<()> {
        let Some(device_arc) = &self.gpu_device else {
            return Ok(());
        };
        let Some(mut gpu_weights) = self.gpu_weight_cache else {
            return Ok(());
        };
        let mut device =
            device_arc
                .lock()
                .map_err(|_| crate::common::errors::ModelError::Backend {
                    message: "Failed to lock GPU device for OutputProjection weight-cache sync"
                        .to_string(),
                })?;
        let w_slice =
            self.w_out
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "OutputProjection weights not contiguous".to_string(),
                })?;
        device.upload(w_slice, &mut gpu_weights)?;
        self.gpu_weight_cache = Some(gpu_weights);
        Ok(())
    }

    /// GPU-accelerated forward pass: logits = input @ w_out
    ///
    /// This is a critical hot-path operation since w_out is vocab-sized.
    /// GEMM on GPU avoids transferring the large weight matrix.
    pub fn forward_gpu(
        &mut self,
        input: &Array2<f32>,
    ) -> crate::common::errors::Result<Array2<f32>> {
        let device_arc = crate::domain::compute::require_gpu_or_error(
            &self.gpu_device,
            "OutputProjection::forward_gpu",
        )?;
        let mut device =
            device_arc
                .lock()
                .map_err(|_| crate::common::errors::ModelError::Backend {
                    message: "Failed to lock GPU device for OutputProjection forward".to_string(),
                })?;

        let (batch_size, embed_dim) = input.dim();
        let vocab_size = self.w_out.ncols();

        if let Some(buf) = self.gpu_cached_input.take() {
            device.deallocate(buf);
        }
        self.gpu_cached_input_shape = None;

        // Upload input (retained on GPU for backward)
        let mut gpu_input = device.allocate_f32(batch_size * embed_dim)?;
        let input_slice =
            input
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "OutputProjection input not contiguous".to_string(),
                })?;
        device.upload(input_slice, &mut gpu_input)?;

        let gpu_weights = self.ensure_gpu_weight_cache_with_device(&mut device)?;

        // GEMM: output = input @ w_out
        let mut gpu_output = device.allocate_f32(batch_size * vocab_size)?;
        device.gemm_f32(
            1.0,
            &gpu_input,
            &gpu_weights,
            0.0,
            &mut gpu_output,
            batch_size,
            vocab_size,
            embed_dim,
            false,
            false,
        )?;

        // Download result
        let mut result = vec![0.0f32; batch_size * vocab_size];
        device.download(&gpu_output, &mut result)?;

        // Cache input for backward
        self.cached_input = Some(input.clone());
        self.gpu_cached_input = Some(gpu_input);
        self.gpu_cached_input_shape = Some((batch_size, embed_dim));
        device.deallocate(gpu_output);

        Array2::from_shape_vec((batch_size, vocab_size), result).map_err(|e| {
            crate::common::errors::ModelError::Backend {
                message: format!("Failed to reshape OutputProjection GPU output: {}", e),
            }
        })
    }

    /// GPU-accelerated gradient computation used by `compute_gradients()`.
    ///
    /// Returns `(grad_input, grad_w_out)` without applying optimizer updates.
    fn compute_gradients_gpu(
        &self,
        output_grads: &Array2<f32>,
    ) -> crate::common::errors::Result<(Array2<f32>, Array2<f32>)> {
        let device_arc = crate::domain::compute::require_gpu_or_error(
            &self.gpu_device,
            "OutputProjection::compute_gradients_gpu",
        )?;
        let mut device =
            device_arc
                .lock()
                .map_err(|_| crate::common::errors::ModelError::Backend {
                    message: "Failed to lock GPU device for OutputProjection compute_gradients"
                        .to_string(),
                })?;

        let input = self
            .cached_input
            .as_ref()
            .ok_or_else(|| crate::common::errors::ModelError::Backend {
                message:
                    "OutputProjection::compute_gradients_gpu requires cached input from forward pass"
                        .to_string(),
            })?;

        let (batch_size, vocab_size) = output_grads.dim();
        let embed_dim = self.w_out.nrows();

        let mut gpu_output_grads = device.allocate_f32(batch_size * vocab_size)?;
        let grads_slice =
            output_grads
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "OutputProjection output_grads not contiguous".to_string(),
                })?;
        device.upload(grads_slice, &mut gpu_output_grads)?;

        let mut gpu_weights = device.allocate_f32(embed_dim * vocab_size)?;
        let weights_slice =
            self.w_out
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "OutputProjection weights not contiguous".to_string(),
                })?;
        device.upload(weights_slice, &mut gpu_weights)?;

        let mut gpu_grad_input = device.allocate_f32(batch_size * embed_dim)?;
        device.gemm_f32(
            1.0,
            &gpu_output_grads,
            &gpu_weights,
            0.0,
            &mut gpu_grad_input,
            batch_size,
            embed_dim,
            vocab_size,
            false,
            true,
        )?;

        let mut gpu_input = device.allocate_f32(batch_size * embed_dim)?;
        let input_slice =
            input
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "OutputProjection cached_input not contiguous".to_string(),
                })?;
        device.upload(input_slice, &mut gpu_input)?;

        let mut gpu_grad_w = device.allocate_f32(embed_dim * vocab_size)?;
        device.gemm_f32(
            1.0,
            &gpu_input,
            &gpu_output_grads,
            0.0,
            &mut gpu_grad_w,
            embed_dim,
            vocab_size,
            batch_size,
            true,
            false,
        )?;

        let mut grad_input_vec = vec![0.0f32; batch_size * embed_dim];
        let mut grad_w_vec = vec![0.0f32; embed_dim * vocab_size];
        device.download(&gpu_grad_input, &mut grad_input_vec)?;
        device.download(&gpu_grad_w, &mut grad_w_vec)?;

        let grad_input = Array2::from_shape_vec((batch_size, embed_dim), grad_input_vec).map_err(
            |e| crate::common::errors::ModelError::Backend {
                message: format!(
                    "Failed to reshape OutputProjection GPU grad_input in compute_gradients: {}",
                    e
                ),
            },
        )?;
        let grad_w_out = Array2::from_shape_vec((embed_dim, vocab_size), grad_w_vec).map_err(
            |e| crate::common::errors::ModelError::Backend {
                message: format!(
                    "Failed to reshape OutputProjection GPU grad_w_out in compute_gradients: {}",
                    e
                ),
            },
        )?;

        Ok((grad_input, grad_w_out))
    }

    /// GPU-accelerated backward pass.
    ///
    /// Computes grad_input = output_grads @ w_out^T and grad_w = input^T @ output_grads.
    pub fn backward_gpu(
        &mut self,
        output_grads: &Array2<f32>,
        lr: f32,
    ) -> crate::common::errors::Result<Array2<f32>> {
        let _ = crate::domain::compute::require_gpu_or_error(
            &self.gpu_device,
            "OutputProjection::backward_gpu",
        )?;
        let (grad_input, param_grads) = self.compute_gradients_gpu_from_backward(output_grads)?;
        if !param_grads.is_empty() {
            self.apply_gradients(&param_grads, lr)?;
        }
        Ok(grad_input)
    }

    fn compute_gradients_gpu_from_backward(
        &mut self,
        output_grads: &Array2<f32>,
    ) -> crate::common::errors::Result<(Array2<f32>, Vec<Array2<f32>>)> {
        let device_arc = crate::domain::compute::require_gpu_or_error(
            &self.gpu_device,
            "OutputProjection::compute_gradients_gpu_from_backward",
        )?;
        let mut device =
            device_arc
                .lock()
                .map_err(|_| crate::common::errors::ModelError::Backend {
                    message: "Failed to lock GPU device for OutputProjection backward".to_string(),
                })?;

        let (batch_size, vocab_size) = output_grads.dim();
        let embed_dim = self.w_out.nrows();

        let mut gpu_grads = device.allocate_f32(batch_size * vocab_size)?;
        let grads_slice =
            output_grads
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "OutputProjection output_grads not contiguous".to_string(),
                })?;
        device.upload(grads_slice, &mut gpu_grads)?;

        let gpu_weights = self.ensure_gpu_weight_cache_with_device(&mut device)?;

        let mut gpu_grad_input = device.allocate_f32(batch_size * embed_dim)?;
        device.gemm_f32(
            1.0,
            &gpu_grads,
            &gpu_weights,
            0.0,
            &mut gpu_grad_input,
            batch_size,
            embed_dim,
            vocab_size,
            false,
            true,
        )?;

        let cached_gpu_input = match (self.gpu_cached_input, self.gpu_cached_input_shape) {
            (Some(buf), Some((b, e))) if b == batch_size && e == embed_dim => Some(buf),
            _ => None,
        };

        let mut temp_gpu_input = None;
        let input_buf = if let Some(buf) = cached_gpu_input {
            buf
        } else if let Some(cached_input) = &self.cached_input {
            let mut gpu_cached_input = device.allocate_f32(batch_size * embed_dim)?;
            let cached_slice = cached_input.as_slice().ok_or_else(|| {
                crate::common::errors::ModelError::Backend {
                    message: "OutputProjection cached_input not contiguous".to_string(),
                }
            })?;
            device.upload(cached_slice, &mut gpu_cached_input)?;
            temp_gpu_input = Some(gpu_cached_input);
            gpu_cached_input
        } else {
            return Err(crate::common::errors::ModelError::Backend {
                message: "OutputProjection::backward_gpu requires cached input from forward pass"
                    .to_string(),
            });
        };

        let mut gpu_grad_w = device.allocate_f32(embed_dim * vocab_size)?;
        device.gemm_f32(
            1.0,
            &input_buf,
            &gpu_grads,
            0.0,
            &mut gpu_grad_w,
            embed_dim,
            vocab_size,
            batch_size,
            true,
            false,
        )?;

        let mut grad_input_vec = vec![0.0f32; batch_size * embed_dim];
        let mut grad_w_vec = vec![0.0f32; embed_dim * vocab_size];
        device.download(&gpu_grad_input, &mut grad_input_vec)?;
        device.download(&gpu_grad_w, &mut grad_w_vec)?;

        device.deallocate(gpu_grads);
        device.deallocate(gpu_grad_input);
        device.deallocate(gpu_grad_w);
        if let Some(buf) = temp_gpu_input {
            device.deallocate(buf);
        }
        if let Some(buf) = self.gpu_cached_input.take() {
            device.deallocate(buf);
        }
        self.gpu_cached_input_shape = None;

        let grad_input =
            Array2::from_shape_vec((batch_size, embed_dim), grad_input_vec).map_err(|e| {
                crate::common::errors::ModelError::Backend {
                    message: format!("Failed to reshape OutputProjection backward output: {}", e),
                }
            })?;
        let grad_w =
            Array2::from_shape_vec((embed_dim, vocab_size), grad_w_vec).map_err(|e| {
                crate::common::errors::ModelError::Backend {
                    message: format!("Failed to reshape weight gradients: {}", e),
                }
            })?;
        Ok((grad_input, vec![grad_w]))
    }
}
