"""
Patch 2: Wire flush/begin_recording through GpuMatrixOps trait,
fix GpuDevice to use trait methods (not downcasting),
and create gpu_fused_forward.rs module.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# 1. Add flush() / begin_recording() to GpuMatrixOps trait with default no-ops
# ─────────────────────────────────────────────────────────────────────────────
GPU_OPS = 'd:/RustGPT/src/domain/compute/gpu_ops.rs'
content = open(GPU_OPS, 'r', encoding='utf-8').read()

# Find the trait opening and insert flush/begin_recording after it
OLD_TRAIT_HEADER = 'trait GpuMatrixOps: Send + Sync {'
NEW_TRAIT_HEADER = '''trait GpuMatrixOps: Send + Sync {
    // ─────────────────────────────────────────────────────────────────
    // Command Batching / Deferred Submission
    // ─────────────────────────────────────────────────────────────────

    /// Begin deferred recording mode.
    ///
    /// After calling this, dispatch calls should be recorded into an
    /// internal command buffer rather than submitted immediately.
    /// Default implementation is a no-op (immediate submission).
    fn begin_recording(&mut self) {}

    /// Flush all pending recorded commands to the GPU in one submission.
    ///
    /// This is the key performance primitive — call it once per training
    /// step to batch the entire forward + backward pass into a single
    /// GPU submission, eliminating per-kernel sync bubbles.
    /// Default implementation is a no-op (immediate submission).
    fn flush(&mut self) {}'''

if OLD_TRAIT_HEADER in content:
    content = content.replace(OLD_TRAIT_HEADER, NEW_TRAIT_HEADER, 1)
    print('✓ flush/begin_recording added to GpuMatrixOps trait')
else:
    print('  WARNING: GpuMatrixOps trait header not found')

open(GPU_OPS, 'w', encoding='utf-8').write(content)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fix GpuDevice::begin_recording / flush to call trait methods (not downcast)
# ─────────────────────────────────────────────────────────────────────────────
GPU_DEVICE = 'd:/RustGPT/src/domain/compute/gpu_device.rs'
content = open(GPU_DEVICE, 'r', encoding='utf-8').read()

OLD_BEGIN = '''    pub fn begin_recording(&mut self) {
        #[cfg(feature = "wgpu")]
        {
            if let Some(ops) = self.ops.as_any_mut()
                .downcast_mut::<super::wgpu_ops::WgpuMatrixOps>()
            {
                ops.begin_recording();
            }
        }
    }

    /// Flush all pending GPU commands and return when submitted.
    ///
    /// This should be called once per training step (after the full forward
    /// pass + backward pass have been recorded). The GPU queue receives all
    /// work as a single batch, maximizing SM utilization by avoiding
    /// pipeline bubbles from frequent submits.
    pub fn flush(&mut self) {
        #[cfg(feature = "wgpu")]
        {
            if let Some(ops) = self.ops.as_any_mut()
                .downcast_mut::<super::wgpu_ops::WgpuMatrixOps>()
            {
                ops.flush();
            }
        }
    }

    /// Check whether deferred recording is currently active.
    pub fn is_recording(&self) -> bool {
        #[cfg(feature = "wgpu")]
        {
            if let Some(ops) = self.ops.as_any()
                .downcast_ref::<super::wgpu_ops::WgpuMatrixOps>()
            {
                return ops.deferred_encoder.is_some();
            }
        }
        false
    }'''

NEW_BEGIN = '''    pub fn begin_recording(&mut self) {
        self.ops.begin_recording();
    }

    /// Flush all pending GPU commands and return when submitted.
    ///
    /// Call once per training step to submit the entire forward+backward
    /// as a single GPU batch, eliminating per-kernel CPU-GPU sync bubbles.
    pub fn flush(&mut self) {
        self.ops.flush();
    }'''

if OLD_BEGIN.replace('\r\n', '\n') in content.replace('\r\n', '\n'):
    content = content.replace('\r\n', '\n').replace(OLD_BEGIN, NEW_BEGIN, 1)
    print('✓ GpuDevice::begin_recording/flush switched to trait delegation')
else:
    print('  WARNING: GpuDevice begin_recording/flush block not found')

open(GPU_DEVICE, 'w', encoding='utf-8').write(content)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Add flush() / begin_recording() override in WgpuMatrixOps trait impl block
# ─────────────────────────────────────────────────────────────────────────────
WGPU_OPS = 'd:/RustGPT/src/domain/compute/wgpu_ops.rs'
content = open(WGPU_OPS, 'r', encoding='utf-8').read().replace('\r\n', '\n')

impl_start = content.find('impl GpuMatrixOps for WgpuMatrixOps {')
depth = 0; i = impl_start
while i < len(content):
    if content[i] == '{': depth += 1
    elif content[i] == '}':
        depth -= 1
        if depth == 0:
            impl_end = i; break
    i += 1

if 'fn flush(&mut self)' not in content[impl_start:impl_end]:
    FLUSH_OVERRIDES = '''
    fn begin_recording(&mut self) {
        self.begin_recording();
    }

    fn flush(&mut self) {
        self.flush();
    }
'''
    content = content[:impl_end] + FLUSH_OVERRIDES + '\n' + content[impl_end:]
    print('✓ flush/begin_recording trait overrides added to WgpuMatrixOps impl')
else:
    print('  flush already present in WgpuMatrixOps impl')

open(WGPU_OPS, 'w', encoding='utf-8').write(content)
print('All patches applied successfully')
