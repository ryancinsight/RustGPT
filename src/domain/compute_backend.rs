use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::common::errors::{ModelError, Result};

/// Concrete execution backend selected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ComputeBackend {
    /// CPU backend (always available).
    #[default]
    Cpu,
    /// NVIDIA CUDA backend.
    Cuda,
    /// Apple Metal backend.
    Metal,
    /// Vulkan compute backend.
    Vulkan,
    /// Intel NPU via Vulkan/WGPU adapter selection.
    Npu,
}

impl ComputeBackend {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Npu => "npu",
        }
    }

    #[inline]
    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Fail fast when a code path still has CPU-only kernels.
    #[inline]
    pub fn require_cpu_implemented(self, op_name: &str) {
        if self.is_gpu() {
            panic!(
                "Backend '{}' selected for '{}', but this path does not have GPU kernels yet. \
                 No automatic CPU fallback is allowed once a GPU backend is selected.",
                self.as_str(),
                op_name
            );
        }
    }
}

/// User/runtime backend selection preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ComputeBackendPreference {
    /// Prefer GPU automatically; fall back to CPU only when no GPU is detected at runtime.
    ///
    /// If a GPU is detected but unavailable due to build-feature mismatch, resolution fails
    /// so the GPU issue is not hidden.
    AutoGpu,
    /// Force CPU execution.
    #[default]
    Cpu,
    /// Require CUDA.
    Cuda,
    /// Require Metal.
    Metal,
    /// Require Vulkan.
    Vulkan,
    /// Require Intel NPU via Vulkan/WGPU backend.
    Npu,
}

impl ComputeBackendPreference {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AutoGpu => "auto-gpu",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Npu => "npu",
        }
    }
}

/// Resolve an effective backend from a preference.
///
/// `AutoGpu` prefers GPU. It falls back to CPU only when no runtime GPU backend is detected.
/// If runtime GPU backends are detected but not compiled into this binary, this returns an error.
pub fn resolve_compute_backend(preference: ComputeBackendPreference) -> Result<ComputeBackend> {
    let preference = env_backend_override().unwrap_or(preference);
    match preference {
        ComputeBackendPreference::Cpu => Ok(ComputeBackend::Cpu),
        ComputeBackendPreference::Cuda => require_backend(ComputeBackend::Cuda, "CUDA"),
        ComputeBackendPreference::Metal => require_backend(ComputeBackend::Metal, "Metal"),
        ComputeBackendPreference::Vulkan => require_backend(ComputeBackend::Vulkan, "Vulkan"),
        ComputeBackendPreference::Npu => {
            require_backend(ComputeBackend::Npu, "Intel NPU (Vulkan/WGPU)")
        }
        ComputeBackendPreference::AutoGpu => {
            let runtime_detected = detect_available_gpu_backends_runtime();
            let detected = detect_available_gpu_backends();
            if let Some(&backend) = detected.first() {
                Ok(backend)
            } else if !runtime_detected.is_empty() {
                let detected_names = runtime_detected
                    .iter()
                    .map(|backend| backend.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(ModelError::Backend {
                    message: format!(
                        "Automatic GPU detection found runtime backend(s): {}. \
                         This binary was not built with matching GPU feature flags. \
                         Enable one of: --features gpu-cuda, gpu-metal, gpu-wgpu. \
                         Falling back to CPU would hide a GPU setup issue, so resolution failed.",
                        detected_names
                    ),
                })
            } else {
                Ok(ComputeBackend::Cpu)
            }
        }
    }
}

/// Resolve `AutoGpu` with strict no-fallback semantics.
///
/// Unlike `resolve_compute_backend(ComputeBackendPreference::AutoGpu)`, this helper
/// never returns CPU. It returns an error when no usable GPU backend is available.
pub fn resolve_compute_backend_strict_auto_gpu() -> Result<ComputeBackend> {
    let backend = resolve_compute_backend(ComputeBackendPreference::AutoGpu)?;
    if backend.is_gpu() {
        Ok(backend)
    } else {
        Err(ModelError::Backend {
            message: "Strict auto-GPU detection requires an available GPU backend. \
                      CPU fallback is disabled for troubleshooting."
                .to_string(),
        })
    }
}

/// Resolve strict Intel NPU execution (no fallback).
///
/// This helper requires an Intel NPU-capable adapter via the WGPU backend.
/// It never falls back to non-NPU GPU or CPU backends.
pub fn resolve_compute_backend_strict_auto_npu() -> Result<ComputeBackend> {
    require_backend(ComputeBackend::Npu, "Intel NPU (Vulkan/WGPU)")
}

/// Detect supported GPU backends in priority order.
///
/// Priority: CUDA > Metal > Vulkan.
pub fn detect_available_gpu_backends() -> Vec<ComputeBackend> {
    detect_available_gpu_backends_runtime()
        .into_iter()
        .filter(|backend| backend_feature_enabled(*backend))
        .collect()
}

/// Detect GPU backends available at runtime without considering compile-time feature flags.
///
/// Priority: CUDA > Metal > Vulkan.
pub fn detect_available_gpu_backends_runtime() -> Vec<ComputeBackend> {
    let mut backends = Vec::new();
    if detect_cuda() {
        backends.push(ComputeBackend::Cuda);
    }
    if detect_metal() {
        backends.push(ComputeBackend::Metal);
    }
    if detect_vulkan() {
        backends.push(ComputeBackend::Vulkan);
    }
    backends
}

/// Detect GPU backends that are both runtime-available AND compile-time enabled.
///
/// This is the preferred function for tests and code that needs to actually use the backend.
/// Priority: CUDA > Metal > Vulkan.
pub fn detect_available_and_compiled_gpu_backends() -> Vec<ComputeBackend> {
    detect_available_gpu_backends_runtime()
        .into_iter()
        .filter(|b| backend_feature_enabled(*b))
        .collect()
}

#[inline]
fn require_backend(backend: ComputeBackend, display_name: &str) -> Result<ComputeBackend> {
    if !backend_feature_enabled(backend) {
        return Err(ModelError::Backend {
            message: format!(
                "Requested backend '{}' is not compiled in this binary. \
                 Enable {}. No fallback is enabled.",
                display_name,
                backend_feature_hint(backend)
            ),
        });
    }

    let runtime_supported = match backend {
        ComputeBackend::Cpu => true,
        ComputeBackend::Cuda => detect_cuda(),
        ComputeBackend::Metal => detect_metal(),
        ComputeBackend::Vulkan => detect_vulkan(),
        ComputeBackend::Npu => detect_intel_npu_runtime(),
    };

    if !runtime_supported {
        Err(ModelError::Backend {
            message: format!(
                "Requested backend '{}' is unavailable on this machine. No fallback is enabled.",
                display_name
            ),
        })
    } else {
        Ok(backend)
    }
}

#[inline]
fn backend_feature_enabled(backend: ComputeBackend) -> bool {
    match backend {
        ComputeBackend::Cpu => true,
        ComputeBackend::Cuda => cfg!(feature = "gpu-cuda"),
        ComputeBackend::Metal => cfg!(all(feature = "gpu-metal", target_os = "macos")),
        ComputeBackend::Vulkan => cfg!(feature = "gpu-wgpu") || cfg!(feature = "wgpu"),
        ComputeBackend::Npu => cfg!(feature = "gpu-wgpu") || cfg!(feature = "wgpu"),
    }
}

#[inline]
fn backend_feature_hint(backend: ComputeBackend) -> &'static str {
    match backend {
        ComputeBackend::Cpu => "--features cpu",
        ComputeBackend::Cuda => "`--features gpu-cuda`",
        ComputeBackend::Metal => "`--features gpu-metal`",
        ComputeBackend::Vulkan => "`--features gpu-wgpu`",
        ComputeBackend::Npu => "`--features gpu-wgpu`",
    }
}

#[inline]
fn env_backend_override() -> Option<ComputeBackendPreference> {
    let raw = std::env::var("RUSTGPT_GPU_BACKEND").ok()?;
    parse_backend_preference(raw.trim())
}

#[inline]
fn parse_backend_preference(raw: &str) -> Option<ComputeBackendPreference> {
    match raw.to_ascii_lowercase().as_str() {
        "auto" | "auto-gpu" | "autogpu" => Some(ComputeBackendPreference::AutoGpu),
        // NPU routing uses WGPU/Vulkan with strict adapter-level NPU prioritization.
        "npu" | "intel-npu" | "intel_npu" => Some(ComputeBackendPreference::Npu),
        "cpu" => Some(ComputeBackendPreference::Cpu),
        "cuda" => Some(ComputeBackendPreference::Cuda),
        "metal" => Some(ComputeBackendPreference::Metal),
        "vulkan" => Some(ComputeBackendPreference::Vulkan),
        _ => None,
    }
}

#[inline]
fn detect_cuda() -> bool {
    if gpu_visibility_disabled() {
        return false;
    }

    // Most reliable probe on workstations with NVIDIA drivers.
    if command_probe("nvidia-smi", &["-L"]) {
        return true;
    }
    // Toolkit-only environments.
    command_probe("nvcc", &["--version"])
}

#[inline]
fn detect_metal() -> bool {
    #[cfg(target_os = "macos")]
    {
        if command_probe("system_profiler", &["SPDisplaysDataType"]) {
            return true;
        }
    }
    false
}

#[inline]
fn detect_vulkan() -> bool {
    // Try WGPU API first (preferred for cross-platform, doesn't need external tools)
    #[cfg(feature = "wgpu")]
    {
        if detect_wgpu_direct() {
            return true;
        }
    }

    // Fallback to external tool probe for non-WGPU systems
    command_probe("vulkaninfo", &["--summary"])
}

/// Detect WGPU/Vulkan availability using WGPU's native API.
/// This doesn't require external tools like `vulkaninfo`.
///
/// If WGPU feature is compiled, we assume GPU support is available.
/// Actual GPU initialization will fail if GPU is not present at runtime.
#[cfg(feature = "wgpu")]
#[inline]
fn detect_wgpu_direct() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .any(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
}

#[cfg(feature = "wgpu")]
#[inline]
fn is_intel_npu_adapter(info: &wgpu::AdapterInfo) -> bool {
    let name = info.name.to_ascii_lowercase();
    let intel = info.vendor == 0x8086 || name.contains("intel");
    let npu_like = name.contains(" npu")
        || name.ends_with("npu")
        || name.contains("neural")
        || name.contains("ai boost");
    intel && npu_like
}

/// Detect whether an Intel NPU-capable adapter is available at runtime.
#[inline]
pub fn detect_intel_npu_runtime() -> bool {
    #[cfg(feature = "wgpu")]
    {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        return instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .any(|adapter| {
                let info = adapter.get_info();
                info.device_type != wgpu::DeviceType::Cpu && is_intel_npu_adapter(&info)
            });
    }

    #[allow(unreachable_code)]
    false
}

#[inline]
fn gpu_visibility_disabled() -> bool {
    fn hidden(var_name: &str) -> bool {
        std::env::var(var_name)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v.is_empty() || v == "-1" || v == "none" || v == "void"
            })
            .unwrap_or(false)
    }

    hidden("CUDA_VISIBLE_DEVICES") || hidden("NVIDIA_VISIBLE_DEVICES")
}

#[inline]
fn command_probe(program: &str, args: &[&str]) -> bool {
    let out = Command::new(program).args(args).output();
    match out {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_preference_always_resolves() {
        let resolved = resolve_compute_backend(ComputeBackendPreference::Cpu).unwrap();
        assert_eq!(resolved, ComputeBackend::Cpu);
    }

    #[test]
    fn parse_preference_aliases() {
        assert_eq!(
            parse_backend_preference("auto"),
            Some(ComputeBackendPreference::AutoGpu)
        );
        assert_eq!(
            parse_backend_preference("AUTO-GPU"),
            Some(ComputeBackendPreference::AutoGpu)
        );
        assert_eq!(
            parse_backend_preference("intel-npu"),
            Some(ComputeBackendPreference::Npu)
        );
        assert_eq!(
            parse_backend_preference("cuda"),
            Some(ComputeBackendPreference::Cuda)
        );
        assert_eq!(
            parse_backend_preference("metal"),
            Some(ComputeBackendPreference::Metal)
        );
        assert_eq!(
            parse_backend_preference("vulkan"),
            Some(ComputeBackendPreference::Vulkan)
        );
        assert_eq!(
            parse_backend_preference("cpu"),
            Some(ComputeBackendPreference::Cpu)
        );
        assert_eq!(parse_backend_preference("invalid"), None);
    }

    #[test]
    fn npu_preference_resolves_or_reports_strict_error() {
        match resolve_compute_backend(ComputeBackendPreference::Npu) {
            Ok(backend) => assert_eq!(backend, ComputeBackend::Npu),
            Err(err) => {
                let msg = err.to_string().to_ascii_lowercase();
                assert!(
                    msg.contains("npu")
                        || msg.contains("vulkan")
                        || msg.contains("gpu")
                        || msg.contains("fallback"),
                    "NPU resolution error should mention backend constraints, got: {}",
                    msg
                );
            }
        }
    }

    #[test]
    fn filtered_backends_are_subset_of_runtime_detection() {
        let runtime = detect_available_gpu_backends_runtime();
        let filtered = detect_available_gpu_backends();
        for backend in filtered {
            assert!(
                runtime.contains(&backend),
                "Filtered backend '{}' was not present in runtime detection",
                backend.as_str()
            );
        }
    }

    #[test]
    fn auto_gpu_resolution_policy_matches_runtime_and_features() {
        let runtime = detect_available_gpu_backends_runtime();
        let feature_filtered = detect_available_gpu_backends();
        let resolved = resolve_compute_backend(ComputeBackendPreference::AutoGpu);

        if let Some(&backend) = feature_filtered.first() {
            let actual = resolved.expect("AutoGpu should resolve to first available GPU backend");
            assert_eq!(actual, backend);
        } else if !runtime.is_empty() {
            assert!(
                resolved.is_err(),
                "AutoGpu must error when runtime GPU exists but matching feature flags are missing"
            );
        } else {
            let actual = resolved.expect("AutoGpu should resolve to CPU when no GPU is detected");
            assert_eq!(actual, ComputeBackend::Cpu);
        }
    }

    #[test]
    fn strict_auto_gpu_never_returns_cpu() {
        match resolve_compute_backend_strict_auto_gpu() {
            Ok(backend) => assert!(backend.is_gpu()),
            Err(err) => {
                let msg = err.to_string().to_ascii_lowercase();
                assert!(
                    msg.contains("gpu") || msg.contains("fallback"),
                    "strict auto-gpu error should mention GPU/fallback, got: {}",
                    msg
                );
            }
        }
    }

    #[test]
    fn strict_auto_npu_requires_npu_or_errors() {
        match resolve_compute_backend_strict_auto_npu() {
            Ok(backend) => assert_eq!(backend, ComputeBackend::Npu),
            Err(err) => {
                let msg = err.to_string().to_ascii_lowercase();
                assert!(
                    msg.contains("npu")
                        || msg.contains("gpu")
                        || msg.contains("fallback")
                        || msg.contains("compiled"),
                    "strict auto-npu error should mention backend constraints, got: {}",
                    msg
                );
            }
        }
    }
}
