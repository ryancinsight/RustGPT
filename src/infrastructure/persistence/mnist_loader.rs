//! MNIST dataset loader for image training.
//!
//! Loads MNIST handwritten digit images from the original binary format
//! and converts them to ImageSample for multimodal training.

use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;

use flate2::read::GzDecoder;

use crate::common::errors::{ModelError, Result};
use crate::domain::multimodal::image::ImageSample;
use crate::infrastructure::persistence::dataset::ImageExample;

/// MNIST image dimensions
pub const MNIST_IMAGE_SIZE: usize = 28;
pub const MNIST_NUM_CLASSES: usize = 10;

/// Load MNIST training images and labels
pub fn load_mnist_train(data_dir: &str) -> Result<(Vec<ImageSample>, Vec<u8>)> {
    let images_path = Path::new(data_dir).join("train-images.gz");
    let labels_path = Path::new(data_dir).join("train-labels.gz");
    
    let images = load_mnist_images(&images_path)?;
    let labels = load_mnist_labels(&labels_path)?;
    
    if images.len() != labels.len() {
        return Err(ModelError::InvalidInput {
            message: format!(
                "MNIST image/label count mismatch: {} vs {}",
                images.len(),
                labels.len()
            ),
        });
    }
    
    Ok((images, labels))
}

/// Load MNIST test images and labels
pub fn load_mnist_test(data_dir: &str) -> Result<(Vec<ImageSample>, Vec<u8>)> {
    let images_path = Path::new(data_dir).join("t10k-images.gz");
    let labels_path = Path::new(data_dir).join("t10k-labels.gz");
    
    let images = load_mnist_images(&images_path)?;
    let labels = load_mnist_labels(&labels_path)?;
    
    if images.len() != labels.len() {
        return Err(ModelError::InvalidInput {
            message: format!(
                "MNIST test image/label count mismatch: {} vs {}",
                images.len(),
                labels.len()
            ),
        });
    }
    
    Ok((images, labels))
}

/// Load MNIST images from gzip file
fn load_mnist_images<P: AsRef<Path>>(path: P) -> Result<Vec<ImageSample>> {
    let file = File::open(&path).map_err(ModelError::from)?;
    let decoder = GzDecoder::new(file);
    let mut reader = BufReader::new(decoder);
    
    // Read header
    let mut header = [0u8; 16];
    reader.read_exact(&mut header).map_err(|e| ModelError::InvalidInput {
        message: format!("Failed to read MNIST image header: {}", e),
    })?;
    
    // Verify magic number (0x00000803 for images)
    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if magic != 0x00000803 {
        return Err(ModelError::InvalidInput {
            message: format!("Invalid MNIST image magic number: {}", magic),
        });
    }
    
    let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let num_rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let num_cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;
    
    tracing::info!(
        "Loading MNIST images: {} images, {}x{} pixels",
        num_images,
        num_rows,
        num_cols
    );
    
    // Read image data
    let mut images = Vec::with_capacity(num_images);
    let mut buffer = vec![0u8; num_rows * num_cols];
    
    for i in 0..num_images {
        reader.read_exact(&mut buffer).map_err(|e| ModelError::InvalidInput {
            message: format!("Failed to read MNIST image {}: {}", i, e),
        })?;
        
        // Convert to f32 and normalize to [0, 1]
        let pixels: Vec<f32> = buffer.iter().map(|&p| p as f32 / 255.0).collect();
        
        let mut sample = ImageSample::new(pixels, num_rows, num_cols, 1);
        sample.label = Some(format!("mnist_digit_{}", i));
        
        images.push(sample);
    }
    
    Ok(images)
}

/// Load MNIST labels from gzip file
fn load_mnist_labels<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let file = File::open(&path).map_err(ModelError::from)?;
    let decoder = GzDecoder::new(file);
    let mut reader = BufReader::new(decoder);
    
    // Read header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).map_err(|e| ModelError::InvalidInput {
        message: format!("Failed to read MNIST label header: {}", e),
    })?;
    
    // Verify magic number (0x00000801 for labels)
    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if magic != 0x00000801 {
        return Err(ModelError::InvalidInput {
            message: format!("Invalid MNIST label magic number: {}", magic),
        });
    }
    
    let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    // Read label data
    let mut labels = vec![0u8; num_labels];
    reader.read_exact(&mut labels).map_err(|e| ModelError::InvalidInput {
        message: format!("Failed to read MNIST labels: {}", e),
    })?;
    
    Ok(labels)
}

/// Convert MNIST images and labels to ImageExample for training
pub fn mnist_to_image_examples(images: Vec<ImageSample>, labels: Vec<u8>) -> Vec<ImageExample> {
    images
        .into_iter()
        .zip(labels.into_iter())
        .enumerate()
        .map(|(idx, (mut image, label))| {
            let digit = label as usize;
            let caption = format!("A handwritten digit {}", digit);
            image.label = Some(caption.clone());
            
            ImageExample {
                image_id: format!("mnist_{:06}", idx),
                caption,
                objects: vec![format!("digit_{}", digit)],
                conversations: vec![
                    crate::infrastructure::persistence::dataset::ConversationTurn {
                        from: "human".to_string(),
                        value: "What digit is shown in this image?".to_string(),
                    },
                    crate::infrastructure::persistence::dataset::ConversationTurn {
                        from: "gpt".to_string(),
                        value: format!("The digit is {}.", digit),
                    },
                ],
            }
        })
        .collect()
}

/// Load MNIST dataset and convert to training examples
pub fn load_mnist_training_data(data_dir: &str, max_samples: Option<usize>) -> Result<Vec<ImageExample>> {
    let (images, labels) = load_mnist_train(data_dir)?;
    
    let (images, labels) = match max_samples {
        Some(n) if n < images.len() => {
            let images: Vec<_> = images.into_iter().take(n).collect();
            let labels: Vec<_> = labels.into_iter().take(n).collect();
            (images, labels)
        }
        _ => (images, labels),
    };
    
    tracing::info!("Loaded {} MNIST training samples", images.len());
    
    Ok(mnist_to_image_examples(images, labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_mnist_header_parsing() {
        // Create a minimal MNIST image file
        let mut file = NamedTempFile::new().unwrap();
        
        // Magic number (0x00000803)
        file.write_all(&[0x00, 0x00, 0x08, 0x03]).unwrap();
        // Number of images: 2
        file.write_all(&[0x00, 0x00, 0x00, 0x02]).unwrap();
        // Rows: 28
        file.write_all(&[0x00, 0x00, 0x00, 0x1c]).unwrap();
        // Columns: 28
        file.write_all(&[0x00, 0x00, 0x00, 0x1c]).unwrap();
        // 2 images * 28 * 28 = 1568 bytes of pixel data
        file.write_all(&[128u8; 1568]).unwrap();
        
        let path = file.path();
        let images = load_mnist_images(path).unwrap();
        
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].height, 28);
        assert_eq!(images[0].width, 28);
        assert_eq!(images[0].channels, 1);
    }
    
    #[test]
    fn test_mnist_to_examples() {
        let mut image1 = ImageSample::new(vec![0.5; 784], 28, 28, 1);
        image1.label = Some("test1".to_string());
        
        let mut image2 = ImageSample::new(vec![0.7; 784], 28, 28, 1);
        image2.label = Some("test2".to_string());
        
        let images = vec![image1, image2];
        let labels = vec![5u8, 3u8];
        
        let examples = mnist_to_image_examples(images, labels);
        
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].objects, vec!["digit_5"]);
        assert_eq!(examples[1].objects, vec!["digit_3"]);
        assert!(examples[0].caption.contains("5"));
        assert!(examples[1].caption.contains("3"));
    }
}