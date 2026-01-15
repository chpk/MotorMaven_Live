#!/usr/bin/env python3
"""
Setup script for Grounded SAM2
Downloads and caches HuggingFace models for faster startup.

Run this once after installing dependencies:
    python setup_sam2.py
"""

import os
import sys

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     PyTorch version: {torch.__version__}")
        else:
            print("[WARN] CUDA not available - will use CPU (slower)")
        return cuda_available
    except ImportError:
        print("[ERR] PyTorch not installed")
        return False


def download_grounding_dino():
    """Pre-download Grounding DINO model."""
    print("\nDownloading Grounding DINO (tiny)...")
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        
        model_id = "IDEA-Research/grounding-dino-tiny"
        print(f"   Model: {model_id}")
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        
        print("[OK] Grounding DINO downloaded!")
        return True
    except Exception as e:
        print(f"[ERR] Failed to download Grounding DINO: {e}")
        return False


def download_sam2():
    """Pre-download SAM2 model."""
    print("\nDownloading SAM2 (tiny)...")
    try:
        # Try SAM2 first
        try:
            from transformers import Sam2Model, Sam2Processor
            
            model_id = "facebook/sam2-hiera-tiny"
            print(f"   Model: {model_id}")
            
            processor = Sam2Processor.from_pretrained(model_id)
            model = Sam2Model.from_pretrained(model_id)
            
            print("[OK] SAM2 downloaded!")
            return True
        except ImportError:
            # Fall back to SAM1
            print("   SAM2 not available in transformers, trying SAM1...")
            from transformers import SamModel, SamProcessor
            
            model_id = "facebook/sam-vit-base"
            print(f"   Fallback model: {model_id}")
            
            processor = SamProcessor.from_pretrained(model_id)
            model = SamModel.from_pretrained(model_id)
            
            print("[OK] SAM1 (fallback) downloaded!")
            return True
    except Exception as e:
        print(f"[ERR] Failed to download SAM: {e}")
        return False


def test_models():
    """Quick test of models."""
    print("\nTesting models...")
    try:
        import numpy as np
        from PIL import Image
        
        # Create test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pil_img = Image.fromarray(test_img)
        
        # Test Grounding DINO
        print("   Testing Grounding DINO...")
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        
        inputs = processor(images=pil_img, text="person. phone.", return_tensors="pt")
        outputs = model(**inputs)
        print("   [OK] Grounding DINO working!")
        
        print("\n[OK] All models ready!")
        return True
    except Exception as e:
        print(f"[ERR] Model test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  Grounded SAM2 Setup")
    print("=" * 60)
    
    # Check CUDA
    cuda_ok = check_cuda()
    
    # Download models
    dino_ok = download_grounding_dino()
    sam_ok = download_sam2()
    
    # Test
    if dino_ok and sam_ok:
        test_models()
    
    print("\n" + "=" * 60)
    if dino_ok and sam_ok:
        print("  [OK] Setup complete! Run: python app_gradio.py")
    else:
        print("  [WARN] Some models failed to download")
        print("  Try: pip install transformers torch torchvision --upgrade")
    print("=" * 60)


if __name__ == "__main__":
    main()
