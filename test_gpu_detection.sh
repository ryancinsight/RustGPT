#!/bin/bash
# Quick GPU detection test
# Runs training for 10 iterations to verify GPU kernels are callable

echo "=== GPU Detection & Kernel Verification Test ==="
echo "Building with gpu-wgpu feature..."
cargo build --bin main --features gpu-wgpu --release 2>&1 | grep -E "(Compiling|Finished|error)"

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✅ Build successful. Running GPU detection test..."
echo ""
echo "Expected: GPU device detection, WGPU initialization, successful kernel calls"
echo ""

# Run for brief period to test GPU
timeout 30 ./target/release/main.exe 2>&1 | head -200
