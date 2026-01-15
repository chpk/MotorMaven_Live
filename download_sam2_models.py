#!/usr/bin/env python3
"""
Download SAM2.1 Model Checkpoints
=================================
Downloads all SAM2.1 model variants for segmentation.
"""

import os
import urllib.request
import sys

# SAM2.1 Model URLs
SAM2_MODELS = {
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "filename": "sam2.1_hiera_tiny.pt",
        "config": "sam2.1_hiera_t.yaml",
        "params": "38.9M",
        "fps": "91.5"
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "filename": "sam2.1_hiera_small.pt",
        "config": "sam2.1_hiera_s.yaml",
        "params": "46M",
        "fps": "85.6"
    },
    "base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "filename": "sam2.1_hiera_base_plus.pt",
        "config": "sam2.1_hiera_b+.yaml",
        "params": "80.8M",
        "fps": "64.8"
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_l.yaml",
        "params": "224.4M",
        "fps": "39.7"
    }
}

# HuggingFace SAM models (alternative, works with transformers)
HUGGINGFACE_SAM_MODELS = {
    "sam-vit-base": {
        "model_id": "facebook/sam-vit-base",
        "params": "94M",
        "description": "SAM ViT-Base (HuggingFace)"
    },
    "sam-vit-large": {
        "model_id": "facebook/sam-vit-large",
        "params": "308M",
        "description": "SAM ViT-Large (HuggingFace)"
    },
    "sam-vit-huge": {
        "model_id": "facebook/sam-vit-huge",
        "params": "636M",
        "description": "SAM ViT-Huge (HuggingFace)"
    },
    "sam2-hiera-tiny": {
        "model_id": "facebook/sam2-hiera-tiny",
        "params": "38.9M",
        "description": "SAM2 Hiera Tiny (HuggingFace)"
    },
    "sam2-hiera-small": {
        "model_id": "facebook/sam2-hiera-small",
        "params": "46M",
        "description": "SAM2 Hiera Small (HuggingFace)"
    },
    "sam2-hiera-base-plus": {
        "model_id": "facebook/sam2-hiera-base-plus",
        "params": "80.8M",
        "description": "SAM2 Hiera Base+ (HuggingFace)"
    },
    "sam2-hiera-large": {
        "model_id": "facebook/sam2-hiera-large",
        "params": "224.4M",
        "description": "SAM2 Hiera Large (HuggingFace)"
    }
}


def get_checkpoints_dir():
    """Get the checkpoints directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir


def download_with_progress(url: str, filepath: str):
    """Download file with progress bar."""
    print(f"Downloading: {os.path.basename(filepath)}")
    print(f"From: {url}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = '█' * filled + '░' * (bar_len - filled)
            size_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)')
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print("\n[OK] Download complete!")
        return True
    except Exception as e:
        print(f"\n[ERR] Download failed: {e}")
        return False


def download_model(model_name: str):
    """Download a specific SAM2.1 model."""
    if model_name not in SAM2_MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(SAM2_MODELS.keys())}")
        return False
    
    model_info = SAM2_MODELS[model_name]
    checkpoints_dir = get_checkpoints_dir()
    filepath = os.path.join(checkpoints_dir, model_info["filename"])
    
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[OK] {model_name} already exists ({size_mb:.1f} MB)")
        return True
    
    return download_with_progress(model_info["url"], filepath)


def download_all():
    """Download all SAM2.1 models."""
    print("=" * 60)
    print("SAM2.1 Model Downloader")
    print("=" * 60)
    
    for name, info in SAM2_MODELS.items():
        print(f"\n[{name.upper()}]: {info['params']} params, {info['fps']} FPS")
        download_model(name)
    
    print("\n" + "=" * 60)
    print("[OK] All downloads complete!")
    print("=" * 60)


def list_models():
    """List all available models."""
    print("\nSAM2.1 Checkpoint Models (Official):")
    print("-" * 50)
    
    checkpoints_dir = get_checkpoints_dir()
    
    for name, info in SAM2_MODELS.items():
        filepath = os.path.join(checkpoints_dir, info["filename"])
        status = "[OK]" if os.path.exists(filepath) else "[--]"
        print(f"{status} {name:12} | {info['params']:8} | {info['fps']:6} FPS | {info['filename']}")
    
    print("\nHuggingFace SAM Models (Auto-download):")
    print("-" * 50)
    
    for name, info in HUGGINGFACE_SAM_MODELS.items():
        print(f"   {name:20} | {info['params']:8} | {info['model_id']}")


def get_available_models():
    """Get list of available models for UI selection."""
    models = []
    checkpoints_dir = get_checkpoints_dir()
    
    # Add HuggingFace models (always available)
    for name, info in HUGGINGFACE_SAM_MODELS.items():
        models.append({
            "name": name,
            "display": f"{info['description']} ({info['params']})",
            "type": "huggingface",
            "model_id": info["model_id"],
            "params": info["params"]
        })
    
    # Add downloaded checkpoint models
    for name, info in SAM2_MODELS.items():
        filepath = os.path.join(checkpoints_dir, info["filename"])
        if os.path.exists(filepath):
            models.append({
                "name": f"sam2.1-{name}",
                "display": f"SAM2.1 {name.title()} ({info['params']}, {info['fps']} FPS)",
                "type": "checkpoint",
                "checkpoint": filepath,
                "config": info["config"],
                "params": info["params"]
            })
    
    return models


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM2.1 Model Downloader")
    parser.add_argument("--model", type=str, help="Download specific model (tiny/small/base_plus/large)")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.all:
        download_all()
    elif args.model:
        download_model(args.model)
    else:
        print("SAM2.1 Model Downloader")
        print("-" * 40)
        print("Usage:")
        print("  --list          List all models")
        print("  --all           Download all models")
        print("  --model NAME    Download specific model")
        print("\nAvailable models: tiny, small, base_plus, large")
        print("\nExample:")
        print("  python download_sam2_models.py --all")
        print("  python download_sam2_models.py --model large")
