use llm::domain::compute::GpuDevice;

fn main() {
    println!("Detecting GPU devices...");
    match GpuDevice::auto_detect() {
        Ok(device) => println!("✓ GPU detected: {}", device.backend()),
        Err(e) => println!("✗ GPU detection failed: {}", e),
    }
}
