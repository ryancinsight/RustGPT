# GPU Utilization Diagnostic - Root Cause Analysis

## Problem Statement
GPU utilization remains at ~0% even with `--features gpu-wgpu` and strict no-fallback enforcement.

## Root Cause Found
**GPU Detection Mechanism is Command-Based**

The GPU detection (`detect_vulkan()`) relies on external programs:
```rust
fn detect_vulkan() -> bool {
    command_probe("vulkaninfo", &["--summary"])
}

fn command_probe(program: &str, args: &[&str]) -> bool {
    let out = Command::new(program).args(args).output();
    match out {
        Ok(output) => output.status.success(),
        Err(_) => false,  // ← Returns false if program not in PATH
    }
}
```

**On Windows**: If `vulkaninfo` is not in `PATH`, GPU detection fails silently.

## Current GPU Detection Flow
1. `enable_gpu_auto_detect()` in RichardsGlu calls `GpuDevice::auto_detect()`
2. `auto_detect()` calls `detect_available_gpu_backends()` 
3. `detect_available_gpu_backends()` runs `detect_vulkan()` which runs `Command::new("vulkaninfo")`
4. If `vulkaninfo` not in PATH → returns `false` → GPU detection fails → layer stays on CPU

## Solution Needed
Replace external command-based detection with direct WGPU API detection.

### Option A: Use WGPU's built-in adapter enumeration (BEST)
```rust
#[cfg(feature = "wgpu")]
fn detect_wgpu() -> bool {
    use wgpu::{Instance, Backends};
    
    let instance = Instance::new(Backends::all());
    let adapters: Vec<_> = futures::executor::block_on(async {
        instance.enumerate_adapters(Backends::all()).collect()
    });
    
    !adapters.is_empty()
}
```

### Option B: Check environment variables directly
```rust
fn detect_wgpu_env() -> bool {
    // WGPU will work if ANY GPU is available
    // We can check by trying to create an instance
    true // Assume available, let runtime initialization fail if not
}
```

## Recommended Fix
Add a new `detect_wgpu()` function that directly probes WGPU instead of relying on `vulkaninfo` CLI tool.

## Testing the Fix
1. After implementing `detect_wgpu()` with WGPU API
2. Run: `RUST_LOG=debug cargo run --release --features gpu-wgpu --bin main`
3. Look for log: "GPU initialization successful" or "GPU forward succeeded"
4. Check Task Manager GPU usage while training runs

## Files to Modify
- `src/domain/compute_backend.rs`: Add `detect_wgpu()` function
- `src/domain/compute_backend.rs`: Update `detect_available_gpu_backends_runtime()` to use it

## Why GPU Still Works After Detection Fails
Even though detection fails, `GpuDevice::new(Vulkan)` is called and tries to initialize WGPU anyway. If WGPU initializes successfully at that point, GPU works. But detection failure means we never TRY that backend.

The strict no-fallback training code logs this:
```
RichardsGlu GPU initialization failed (will use CPU)
```

So GPU layer initialization is silently failing, causing all forward passes to use CPU.
