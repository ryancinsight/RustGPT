#!/usr/bin/env python3
"""
Download real datasets for RustGPT training.
This script downloads:
1. TinyStories (for pretraining) - Subset of training data
2. Alpaca-Cleaned (for chat tuning) - Full dataset (~52k instruction pairs)

Requires: pip install datasets pandas pyarrow
"""
import json
import os
import sys
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
PRETRAINING_FILE = DATA_DIR / "pretraining_data.json"
CHAT_FILE = DATA_DIR / "chat_training_data.json"

# Limit pretraining data to avoid OOM or excessively long downloads
# 500,000 stories is approx 1GB of text, which is a good "Minimum Viable" size
# The full TinyStories is >2M examples.
MAX_PRETRAIN_EXAMPLES = 500_000

def ensure_dependencies():
    try:
        import datasets
        import pandas
    except ImportError:
        print("Missing dependencies!")
        print("Please run: pip install datasets pandas pyarrow")
        sys.exit(1)

def save_json(data, filepath):
    print(f"Saving {len(data)} examples to {filepath}...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"Saved {filepath} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"Error saving {filepath}: {e}")

def download_tinystories():
    from datasets import load_dataset
    print(f"\n=== Downloading TinyStories (Pretraining Data) ===")
    print(f"Targeting {MAX_PRETRAIN_EXAMPLES} stories...")
    
    try:
        # Use streaming to avoid downloading the whole dataset before slicing
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        
        texts = []
        count = 0
        for item in dataset:
            texts.append(item['text'])
            count += 1
            if count % 10000 == 0:
                print(f"Downloaded {count} stories...", end='\r')
            if count >= MAX_PRETRAIN_EXAMPLES:
                break
        print(f"\nDownloaded {count} stories total.")
        
        save_json(texts, PRETRAINING_FILE)
        
    except Exception as e:
        print(f"Error downloading TinyStories: {e}")

def download_alpaca():
    from datasets import load_dataset
    print("\n=== Downloading Alpaca-Cleaned (Chat Tuning Data) ===")
    
    try:
        # Alpaca is small enough to download fully
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        print(f"Downloaded {len(dataset)} conversations.")
    except Exception as e:
        print(f"Error downloading Alpaca: {e}")
        return

    formatted_chats = []
    for item in dataset:
        instruction = item['instruction']
        input_text = item['input']
        output = item['output']
        
        # Alpaca format to RustGPT chat format
        # RustGPT typically uses "User: ... \nAssistant: ... </s>"
        if input_text:
            text = f"User: {instruction}\nInput: {input_text}\nAssistant: {output} </s>"
        else:
            text = f"User: {instruction}\nAssistant: {output} </s>"
            
        formatted_chats.append(text)

    save_json(formatted_chats, CHAT_FILE)

def main():
    ensure_dependencies()
    DATA_DIR.mkdir(exist_ok=True)
    
    download_tinystories()
    download_alpaca()
    
    print("\nDone! You can now run training.")

if __name__ == "__main__":
    main()
