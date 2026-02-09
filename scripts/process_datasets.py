#!/usr/bin/env python3
"""
Process real datasets (MNIST, Speech Commands) into RustGPT multi-modal format.
"""
import gzip
import json
import os
import struct
import zipfile
from pathlib import Path

DATA_DIR = Path("data")

def read_mnist_images(filepath):
    """Read MNIST images from gzip file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = []
        for i in range(min(num, 100)):  # Limit to 100 samples for demo
            img_data = f.read(rows * cols)
            images.append({
                'index': i,
                'rows': rows,
                'cols': cols,
                'data': list(img_data)  # Raw pixel data (0-255)
            })
        return images

def read_mnist_labels(filepath):
    """Read MNIST labels from gzip file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = []
        for i in range(min(num, 100)):
            labels.append(struct.unpack('B', f.read(1))[0])
        return labels

def process_mnist():
    """Process MNIST dataset into image_training_data.json format."""
    mnist_dir = DATA_DIR / "mnist"
    
    # Standard MNIST filenames
    train_images_path = mnist_dir / "train-images-idx3-ubyte.gz"
    train_labels_path = mnist_dir / "train-labels-idx1-ubyte.gz"
    
    if not train_images_path.exists() or not train_labels_path.exists():
        print(f"MNIST files not found. Expected:")
        print(f"  - {train_images_path}")
        print(f"  - {train_labels_path}")
        return
    
    print(f"Processing {train_images_path}...")
    images = read_mnist_images(train_images_path)
    labels = read_mnist_labels(train_labels_path)
    
    # Create image training examples
    examples = []
    digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    for i, (img, label) in enumerate(zip(images, labels)):
        examples.append({
            "image_id": f"mnist_train_{i:05d}",
            "caption": f"Handwritten digit {digit_names[label]} ({label})",
            "objects": [digit_names[label]],
            "conversations": [
                {"from": "human", "value": f"What digit is shown in this image?"},
                {"from": "gpt", "value": f"This is the digit {digit_names[label]} ({label})."},
                {"from": "human", "value": f"Describe this image."},
                {"from": "gpt", "value": f"This is a 28x28 grayscale image of the handwritten digit {digit_names[label]}."}
            ],
            "metadata": {
                "dataset": "MNIST",
                "label": int(label),
                "label_name": digit_names[label],
                "rows": img['rows'],
                "cols": img['cols'],
                "raw_data": img['data']  # Raw pixel values for actual processing
            }
        })
    
    output = {"examples": examples}
    output_path = DATA_DIR / "image_training_data_real.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Created {output_path} with {len(examples)} real MNIST images")

def process_speech_commands():
    """Process Speech Commands dataset into speech_training_data.json format."""
    speech_dir = DATA_DIR / "speech_commands" / "mini_speech_commands"
    
    if not speech_dir.exists():
        # Try extracting first
        zip_path = DATA_DIR / "speech_commands" / "mini_speech_commands.zip"
        if zip_path.exists():
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR / "speech_commands")
    
    if not speech_dir.exists():
        print(f"Speech commands directory not found: {speech_dir}")
        return
    
    # Get all command categories (subdirectories)
    commands = [d.name for d in speech_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
    print(f"Found speech command categories: {commands}")
    
    examples = []
    max_per_command = 10  # Limit samples per command
    
    for command in commands:
        command_dir = speech_dir / command
        wav_files = list(command_dir.glob("*.wav"))
        
        for i, wav_file in enumerate(wav_files[:max_per_command]):
            # Get file size as proxy for duration
            file_size = wav_file.stat().st_size
            # Rough estimate: 16kHz, 16-bit, mono = 32000 bytes per second
            duration = file_size / 32000
            
            examples.append({
                "audio_id": f"speech_{command}_{i:03d}",
                "duration_seconds": round(duration, 2),
                "transcript": command,
                "speaker": f"unknown_{command}_{i}",
                "language": "en",
                "conversations": [
                    {"from": "human", "value": "What word is spoken in this audio?"},
                    {"from": "gpt", "value": f"The spoken word is '{command}'."},
                    {"from": "human", "value": f"Transcribe this audio clip."},
                    {"from": "gpt", "value": f"{command}"}
                ],
                "metadata": {
                    "dataset": "Google Mini Speech Commands",
                    "command": command,
                    "file_path": str(wav_file.relative_to(DATA_DIR)),
                    "sample_rate": 16000,
                    "format": "wav"
                }
            })
    
    output = {"examples": examples}
    output_path = DATA_DIR / "speech_training_data_real.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Created {output_path} with {len(examples)} real speech samples")

def main():
    print("Processing real datasets for RustGPT...")
    print("=" * 50)
    
    process_mnist()
    print()
    process_speech_commands()
    
    print()
    print("=" * 50)
    print("Dataset processing complete!")
    print()
    print("Files created:")
    print(f"  - {DATA_DIR / 'image_training_data_real.json'}")
    print(f"  - {DATA_DIR / 'speech_training_data_real.json'}")
    print()
    print("To use these datasets, update your dataset loader or rename:")
    print("  mv data/image_training_data_real.json data/image_training_data.json")
    print("  mv data/speech_training_data_real.json data/speech_training_data.json")

if __name__ == "__main__":
    main()
