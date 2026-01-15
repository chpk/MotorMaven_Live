#!/usr/bin/env python3
"""
Grounded SAM Tracker - PRECISE HYBRID APPROACH
===============================================
Uses Gemini for:
1. SPECIFIC visual descriptions (color, shape, material)
2. Approximate bounding box coordinates

Then uses SAM for precise segmentation with both:
- Text prompts via Grounding DINO
- Bounding box hints from Gemini

Works for: generic objects, text/labels, custom/unseen objects, specific parts
"""

import threading
import queue
import time
import base64
import io
import json
import re
import os
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM Tracker] PyTorch: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class DetectedObject:
    instance_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class GeminiDetection:
    """Detection result from Gemini with description and bbox."""
    description: str  # Specific visual description
    bbox: Optional[Tuple[float, float, float, float]] = None  # Normalized 0-1: x1, y1, x2, y2
    confidence: float = 0.8


class PreciseGeminiExtractor:
    """
    Extracts specific objects with accurate bounding boxes.
    Uses intent classification to determine if bbox is needed.
    """
    
    # Objects that Grounding DINO handles well - NO bbox needed from Gemini
    GENERIC_OBJECTS = {
        "phone", "smartphone", "mobile", "cellphone", "iphone", "android",
        "wallet", "purse", "bag", "backpack", "handbag",
        "bottle", "cup", "mug", "glass", "can",
        "laptop", "computer", "keyboard", "mouse", "monitor", "screen",
        "pen", "pencil", "marker", "eraser", "book", "notebook", "paper",
        "chair", "table", "desk", "sofa", "couch", "bed",
        "door", "window", "wall", "floor", "ceiling",
        "car", "vehicle", "truck", "bus", "motorcycle", "bicycle",
        "person", "man", "woman", "child", "face", "hand",
        "cat", "dog", "bird", "animal",
        "plant", "flower", "tree", "leaf",
        "food", "fruit", "apple", "banana", "orange",
        "clock", "watch", "tv", "television", "remote",
        "shoe", "shirt", "pants", "hat", "glasses",
        "box", "container", "package",
        "light", "lamp", "fan", "ac", "air conditioner",
        "refrigerator", "fridge", "microwave", "oven", "stove",
    }
    
    # Objects that need Gemini bbox (custom/industrial/text)
    CUSTOM_KEYWORDS = {
        "motor", "terminal", "connector", "wire", "cable", "circuit",
        "screw", "bolt", "nut", "washer", "gear", "bearing",
        "valve", "pump", "sensor", "switch", "button", "knob",
        "label", "sign", "text", "writing", "logo", "brand", "sticker",
        "warning", "caution", "serial", "number", "code", "barcode", "qr",
        "part", "component", "module", "unit", "assembly",
        "positive", "negative", "ground", "power", "input", "output",
        "red wire", "blue wire", "black wire", "yellow wire",
        "specific", "particular", "this", "that",
    }
    
    SYSTEM_PROMPT = """You are an object identifier. Identify ONLY the objects the user mentions.

OUTPUT FORMAT (JSON only):
{"objects":[
  {"description":"simple_name", "type":"generic|custom", "bbox":[x1,y1,x2,y2]}
]}

RULES:
1. "description": Simple 1-2 word name (phone, wallet, motor, terminal)
2. "type": 
   - "generic" for common objects (phone, wallet, cup, chair, person)
   - "custom" for industrial/text/specific items (motor, terminal, label, specific text)
3. "bbox": ONLY provide bbox for "custom" type objects. For "generic", use null.

BBOX FORMAT (only for custom objects):
- Normalized 0.0-1.0 coordinates: [left, top, right, bottom]
- Be VERY precise for custom/industrial objects

Return JSON only. No explanation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._lock = threading.Lock()
    
    def _ensure_client(self):
        if self.client is None:
            with self._lock:
                if self.client is None:
                    try:
                        from google import genai
                        self.client = genai.Client(api_key=self.api_key)
                        print("[GeminiExtractor] Initialized")
                    except Exception as e:
                        print(f"[GeminiExtractor] Init failed: {e}")
    
    def is_generic_object(self, description: str) -> bool:
        """Check if object is generic (DINO can handle) or custom (needs bbox)."""
        desc_lower = description.lower().strip()
        
        # Check against generic objects list
        for generic in self.GENERIC_OBJECTS:
            if generic in desc_lower or desc_lower in generic:
                return True
        
        # Check if it contains custom keywords
        for custom in self.CUSTOM_KEYWORDS:
            if custom in desc_lower:
                return False
        
        # Default: if it's a simple common noun, it's probably generic
        simple_words = desc_lower.split()
        if len(simple_words) <= 2:
            # Single/double word objects are often generic
            return True
        
        return False
    
    def extract(self, frame: np.ndarray, text: str, frame_size: Tuple[int, int]) -> List[GeminiDetection]:
        """
        Extract objects mentioned by user. Only provides bbox for custom/text/unseen objects.
        """
        self._ensure_client()
        if self.client is None:
            print("[GeminiExtractor] No client - using fallback")
            return self._fallback(text, frame_size)
        
        try:
            # Convert frame to JPEG (lower quality for faster processing)
            pil_img = Image.fromarray(frame)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=60)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            
            prompt = f"""{self.SYSTEM_PROMPT}

User's request: "{text}"

IMPORTANT:
- Identify ONLY the specific objects the user is asking about
- Do NOT include random background objects
- For common objects (phone, wallet, cup, person), set type="generic" and bbox=null
- For industrial/text/specific items (motor, terminal, label, specific text), set type="custom" with precise bbox
- Be conservative - only include objects actually mentioned by the user"""
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    prompt
                ]
            )
            
            result = response.text.strip()
            print(f"[GeminiExtractor] Response: {result[:300]}")
            
            # Parse JSON response
            detections = self._parse_response(result)
            
            # Process detections with intent classification
            valid_detections = []
            for d in detections:
                is_generic = self.is_generic_object(d.description)
                
                if is_generic:
                    # Generic object - DINO will handle it, no bbox needed
                    d.bbox = None
                    print(f"[GeminiExtractor] '{d.description}' (generic - DINO will detect)")
                elif d.bbox is not None:
                    # Custom object with bbox - validate coordinates
                    x1, y1, x2, y2 = d.bbox
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))
                    
                    # Ensure proper ordering and minimum size
                    if x2 <= x1: x1, x2 = x2, x1
                    if y2 <= y1: y1, y2 = y2, y1
                    if x2 - x1 < 0.05: x2 = min(1.0, x1 + 0.1)
                    if y2 - y1 < 0.05: y2 = min(1.0, y1 + 0.1)
                    
                    d.bbox = (x1, y1, x2, y2)
                    print(f"[GeminiExtractor] '{d.description}' (custom) bbox={d.bbox}")
                else:
                    # Custom but no bbox - will use fallback
                    print(f"[GeminiExtractor] '{d.description}' (custom, no bbox)")
                
                valid_detections.append(d)
            
            if valid_detections:
                print(f"[GeminiExtractor] Found {len(valid_detections)} object(s)")
                return valid_detections
            else:
                return self._fallback(text, frame_size)
            
        except Exception as e:
            print(f"[GeminiExtractor] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback(text, frame_size)
    
    def _parse_response(self, response: str) -> List[GeminiDetection]:
        """Parse Gemini's JSON response - only keeps bbox for custom objects."""
        detections = []
        
        try:
            # Clean up response - remove markdown code blocks
            clean = response.strip()
            clean = re.sub(r'^```json\s*', '', clean)
            clean = re.sub(r'^```\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)
            clean = clean.strip()
            
            # Try to parse JSON
            data = None
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                # Try to find and fix JSON
                json_match = re.search(r'\{.*\}', clean, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            if data:
                # Handle various formats
                objects = data.get("objects", data) if isinstance(data, dict) else data
                if not isinstance(objects, list):
                    objects = [objects]
                
                for obj in objects[:5]:  # Limit to 5 objects to reduce noise
                    if isinstance(obj, dict):
                        desc = obj.get("description", obj.get("name", "")).strip()
                        obj_type = obj.get("type", "generic").lower()
                        bbox = obj.get("bbox", obj.get("bounding_box", obj.get("box")))
                        
                        if not desc:
                            continue
                        
                        # Only use bbox for custom objects, not generic ones
                        valid_bbox = None
                        if obj_type == "custom" and bbox:
                            valid_bbox = self._parse_bbox(bbox)
                        
                        detections.append(GeminiDetection(
                            description=desc,
                            bbox=valid_bbox,
                            confidence=0.85 if obj_type == "custom" else 0.9
                        ))
            
            # Fallback: extract description with regex if JSON parsing failed
            if not detections:
                # Look for "description": "..." pattern
                desc_matches = re.findall(r'"description"\s*:\s*"([^"]+)"', response)
                for desc in desc_matches[:3]:
                    if desc and len(desc) > 2:
                        # No bbox for fallback - let DINO handle it
                        detections.append(GeminiDetection(
                            description=desc.strip(),
                            bbox=None,
                            confidence=0.7
                        ))
            
            # Last fallback: plain text extraction
            if not detections:
                for line in response.strip().split('\n')[:3]:
                    line = line.strip().strip('-').strip('*').strip()
                    if line and len(line) > 3 and '"' not in line and '{' not in line:
                        detections.append(GeminiDetection(
                            description=line[:40],
                            bbox=(0.2, 0.2, 0.8, 0.8)
                        ))
        
        except Exception as e:
            print(f"[GeminiExtractor] Parse error: {e}")
        
        return detections
    
    def _parse_bbox(self, bbox) -> Optional[Tuple[float, float, float, float]]:
        """Parse bbox from various formats - handles malformed arrays."""
        if not bbox:
            return None
        
        try:
            # Handle list format - Gemini sometimes returns 6 values, take first 4
            if isinstance(bbox, (list, tuple)):
                if len(bbox) >= 4:
                    # Take exactly 4 values
                    vals = [float(v) for v in bbox[:4]]
                    x1, y1, x2, y2 = vals
                elif len(bbox) == 2:
                    # Maybe center point - expand to region
                    cx, cy = float(bbox[0]), float(bbox[1])
                    x1, y1 = cx - 0.15, cy - 0.15
                    x2, y2 = cx + 0.15, cy + 0.15
                else:
                    return None
            # Handle dict format
            elif isinstance(bbox, dict):
                x1 = float(bbox.get("x1", bbox.get("left", bbox.get("xmin", 0))))
                y1 = float(bbox.get("y1", bbox.get("top", bbox.get("ymin", 0))))
                x2 = float(bbox.get("x2", bbox.get("right", bbox.get("xmax", 1))))
                y2 = float(bbox.get("y2", bbox.get("bottom", bbox.get("ymax", 1))))
            else:
                return None
            
            # If coords look like pixels (>1), normalize to 0-1
            if x2 > 1.5 or y2 > 1.5:
                max_val = max(x1, y1, x2, y2)
                if max_val > 10:
                    # Pixel coords - normalize assuming ~1000px max
                    scale = max_val if max_val < 2000 else 1000
                    x1 /= scale
                    y1 /= scale
                    x2 /= scale
                    y2 /= scale
            
            # Clamp to valid range
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            
            # Ensure proper ordering (swap if needed)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            
            # Ensure minimum size
            if x2 - x1 < 0.05:
                cx = (x1 + x2) / 2
                x1, x2 = cx - 0.1, cx + 0.1
            if y2 - y1 < 0.05:
                cy = (y1 + y2) / 2
                y1, y2 = cy - 0.1, cy + 0.1
            
            # Re-clamp after adjustment
            x1 = max(0.0, min(0.95, x1))
            y1 = max(0.0, min(0.95, y1))
            x2 = max(0.05, min(1.0, x2))
            y2 = max(0.05, min(1.0, y2))
            
            return (x1, y1, x2, y2)
            
        except (ValueError, TypeError, KeyError) as e:
            print(f"[GeminiExtractor] Bbox parse failed: {e}")
            return None
    
    def _fallback(self, text: str, frame_size: Tuple[int, int] = None) -> List[GeminiDetection]:
        """Extract simple object keyword from user's text."""
        text_lower = text.lower()
        
        # Default center bbox for unknown objects
        default_bbox = (0.2, 0.2, 0.8, 0.8)
        
        # Simple keyword extraction
        keywords = [
            (r"phone|smartphone|mobile|cell", "phone"),
            (r"wallet|purse", "wallet"),
            (r"key|keys", "key"),
            (r"cup|mug|glass", "cup"),
            (r"pen|pencil", "pen"),
            (r"mouse", "mouse"),
            (r"bottle|water", "bottle"),
            (r"watch", "watch"),
            (r"ring", "ring"),
            (r"card|credit", "card"),
            (r"book", "book"),
            (r"remote", "remote"),
            (r"laptop|computer", "laptop"),
        ]
        
        for pattern, obj_name in keywords:
            if re.search(pattern, text_lower):
                return [GeminiDetection(description=obj_name, bbox=default_bbox, confidence=0.7)]
        
        # If user asks "what is this/in my hand" - use generic object
        if "what" in text_lower or "this" in text_lower or "hand" in text_lower:
            return [GeminiDetection(description="object", bbox=default_bbox, confidence=0.5)]
        
        return []  # No objects to track


class HybridSAMTracker:
    """
    Hybrid tracker that supports:
    - SAM/SAM2: Uses Grounding DINO + SAM for segmentation (visual prompts)
    - SAM3: Uses text prompts directly for semantic segmentation (no DINO needed!)
    
    SAM3 is more powerful - understands natural language and semantic relations.
    """
    
    # Available SAM models
    SAM_MODELS = {
        # SAM1 models (HuggingFace)
        "sam-vit-base": {"type": "sam1", "id": "facebook/sam-vit-base", "params": "94M"},
        "sam-vit-large": {"type": "sam1", "id": "facebook/sam-vit-large", "params": "308M"},
        "sam-vit-huge": {"type": "sam1", "id": "facebook/sam-vit-huge", "params": "636M"},
        # SAM2 models (HuggingFace)
        "sam2-hiera-tiny": {"type": "sam2", "id": "facebook/sam2-hiera-tiny", "params": "38.9M"},
        "sam2-hiera-small": {"type": "sam2", "id": "facebook/sam2-hiera-small", "params": "46M"},
        "sam2-hiera-base-plus": {"type": "sam2", "id": "facebook/sam2-hiera-base-plus", "params": "80.8M"},
        "sam2-hiera-large": {"type": "sam2", "id": "facebook/sam2-hiera-large", "params": "224.4M"},
        # SAM3 models (HuggingFace) - Text-based segmentation!
        "sam3": {"type": "sam3", "id": "facebook/sam3", "params": "~1B", "text_based": True},
    }
    
    # HuggingFace token for SAM3 (gated model) - set via environment variable
    # Export HF_TOKEN=your_token or enter in UI
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    
    def __init__(self, api_key: str = None, sam_model: str = "sam2-hiera-small"):
        self.device = DEVICE
        self.api_key = api_key
        self.sam_model_name = sam_model
        self.is_sam3 = "sam3" in sam_model.lower()
        
        # Models
        self.dino_model = None
        self.dino_processor = None
        self.sam_model = None
        self.sam_processor = None
        
        # Gemini extractor (only needed for SAM1/SAM2, not SAM3)
        if api_key and not self.is_sam3:
            print(f"[Tracker] Creating Gemini extractor with API key ({len(api_key)} chars)")
            self.gemini_extractor = PreciseGeminiExtractor(api_key)
        elif self.is_sam3:
            print("[Tracker] SAM3 mode - using direct text prompts (no Gemini extractor needed)")
            self.gemini_extractor = None
        else:
            print("[Tracker] WARNING: No API key provided - Gemini extractor disabled!")
            self.gemini_extractor = None
        
        # State
        self.current_prompt = ""
        self.current_gemini_detections: List[GeminiDetection] = []
        self.prompt_lock = threading.Lock()
        self.models_loaded = False
        
        # Detection settings
        self.detection_threshold = 0.15
        self.detection_interval = 5
        self.frame_count = 0
        self.last_detections: List[DetectedObject] = []
        
        # Colors
        self.colors = [
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
        ]
        
        # Background thread
        self.extraction_queue = queue.Queue(maxsize=2)
        self.extraction_thread = None
        self._running = False
    
    @classmethod
    def get_available_models(cls) -> List[Dict]:
        """Get list of available SAM models for UI selection."""
        models = []
        for name, info in cls.SAM_MODELS.items():
            models.append({
                "name": name,
                "display": f"{name} ({info['params']})",
                "params": info["params"]
            })
        return models
    
    def set_sam_model(self, model_name: str):
        """Change the SAM model (requires reload)."""
        if model_name in self.SAM_MODELS:
            self.sam_model_name = model_name
            self.is_sam3 = "sam3" in model_name.lower()
            self.models_loaded = False
            self.sam_model = None
            self.sam_processor = None
            self.dino_model = None
            self.dino_processor = None
            print(f"[Tracker] SAM model set to: {model_name} (SAM3={self.is_sam3})")
        else:
            print(f"[Tracker] Unknown model: {model_name}")
    
    def load_models(self) -> bool:
        """Load Grounding DINO (for SAM1/SAM2) and selected SAM model."""
        if self.models_loaded:
            return True
        
        try:
            import warnings
            
            sam_info = self.SAM_MODELS.get(self.sam_model_name)
            if not sam_info:
                print(f"[Tracker] Unknown SAM model: {self.sam_model_name}, using default")
                sam_info = self.SAM_MODELS["sam2-hiera-small"]
                self.sam_model_name = "sam2-hiera-small"
                self.is_sam3 = False
            
            # SAM3 uses text prompts directly - no need for Grounding DINO!
            if self.is_sam3:
                return self._load_sam3_models(sam_info)
            else:
                return self._load_sam2_models(sam_info)
            
        except Exception as e:
            print(f"[Tracker] Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_sam3_models(self, sam_info: dict) -> bool:
        """Load SAM3 model - text-based segmentation (no DINO needed!)."""
        import warnings
        
        print(f"[Tracker] Loading SAM3 ({sam_info['params']}) - Text-based segmentation!")
        print("[Tracker] SAM3 understands natural language directly - no Grounding DINO needed")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                from transformers import Sam3Processor, Sam3Model
                
                # Login to HuggingFace for gated model access
                try:
                    from huggingface_hub import login
                    login(token=self.HF_TOKEN, add_to_git_credential=False)
                    print("[Tracker] HuggingFace login successful")
                except Exception as login_err:
                    print(f"[Tracker] HuggingFace login warning: {login_err}")
                
                # Load SAM3 model
                self.sam_processor = Sam3Processor.from_pretrained(
                    sam_info["id"], 
                    token=self.HF_TOKEN
                )
                self.sam_model = Sam3Model.from_pretrained(
                    sam_info["id"], 
                    token=self.HF_TOKEN
                ).to(self.device)
                self.sam_model.eval()
                
            self.models_loaded = True
            print(f"[Tracker] SAM3 loaded! Use text prompts like 'phone', 'all red objects', etc.")
            return True
            
        except ImportError as e:
            print(f"[Tracker] SAM3 not available in transformers: {e}")
            print("[Tracker] Try: pip install transformers>=4.40.0")
            return False
        except Exception as e:
            print(f"[Tracker] SAM3 load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_sam2_models(self, sam_info: dict) -> bool:
        """Load Grounding DINO + SAM1/SAM2 models."""
        import warnings
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        
        # Load Grounding DINO (needed for SAM1/SAM2)
        print("[Tracker] Loading Grounding DINO...")
        self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(self.device)
        self.dino_model.eval()
        
        print(f"[Tracker] Loading SAM: {self.sam_model_name} ({sam_info['params']})...")
        
        # Load SAM1 or SAM2 from HuggingFace
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            if sam_info["type"] == "sam2":
                # SAM2 models
                from transformers import Sam2Model, Sam2Processor
                self.sam_processor = Sam2Processor.from_pretrained(sam_info["id"])
                self.sam_model = Sam2Model.from_pretrained(sam_info["id"]).to(self.device)
            else:
                # Original SAM1 models
                from transformers import SamModel, SamProcessor
                self.sam_processor = SamProcessor.from_pretrained(sam_info["id"])
                self.sam_model = SamModel.from_pretrained(sam_info["id"]).to(self.device)
            
            self.sam_model.eval()
        
        self.models_loaded = True
        print(f"[Tracker] All models loaded! (SAM: {self.sam_model_name})")
        return True
    
    def set_prompt(self, prompt: str):
        with self.prompt_lock:
            if prompt != self.current_prompt:
                self.current_prompt = prompt
                self.last_detections = []
                self.frame_count = 0
                if prompt:
                    print(f"[Tracker] Prompt: '{prompt}'")
    
    def clear_prompt(self):
        """Clear the current prompt - stop highlighting."""
        with self.prompt_lock:
            if self.current_prompt:
                print("[Tracker] Prompt cleared")
            self.current_prompt = ""
            self.current_gemini_detections = []
            self.last_detections = []
            self.frame_count = 0
    
    def set_gemini_detections(self, detections: List[GeminiDetection]):
        """Set detections from Gemini - track ALL objects the user asked for."""
        with self.prompt_lock:
            self.current_gemini_detections = detections if detections else []
            
            if detections:
                # Create DINO-friendly prompt (lowercase, period-separated)
                descs = [d.description.lower().strip() for d in detections]
                self.current_prompt = ". ".join(descs) + "."
                self.last_detections = []  # Force re-detection
                self.frame_count = 0
                
                # Log with bbox info
                for d in detections:
                    bbox_str = f"bbox={d.bbox}" if d.bbox else "no bbox"
                    print(f"[Tracker] → {d.description} ({bbox_str})")
                print(f"[Tracker] DINO prompt: '{self.current_prompt}'")
    
    def get_prompt(self) -> str:
        with self.prompt_lock:
            return self.current_prompt
    
    def extract_keywords_async(self, frame: np.ndarray, text: str):
        try:
            self.extraction_queue.put_nowait((frame.copy(), text))
        except queue.Full:
            pass
    
    def _extraction_thread_func(self):
        while self._running:
            try:
                frame, text = self.extraction_queue.get(timeout=0.5)
                print(f"[Tracker] Processing extraction request...")
                
                if self.gemini_extractor:
                    h, w = frame.shape[:2]
                    print(f"[Tracker] Calling Gemini extractor with frame {w}x{h}...")
                    detections = self.gemini_extractor.extract(frame, text, (w, h))
                    
                    if detections:
                        # Found objects - update segmentation
                        self.set_gemini_detections(detections)
                        print(f"[Tracker] Found {len(detections)} objects")
                    else:
                        # No detections - keep previous prompt (don't clear immediately)
                        # Only log, don't clear
                        print("[Tracker] No new objects detected")
                else:
                    print("[Tracker] ERROR: Gemini extractor not initialized (no API key?)")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Tracker] Extraction error: {e}")
                import traceback
                traceback.print_exc()
    
    def start(self):
        self._running = True
        self.extraction_thread = threading.Thread(target=self._extraction_thread_func, daemon=True)
        self.extraction_thread.start()
    
    def stop(self):
        self._running = False
        if self.extraction_thread:
            self.extraction_thread.join(timeout=2.0)
    
    @torch.inference_mode()
    def detect_with_dino(self, image: np.ndarray, prompt: str) -> List[DetectedObject]:
        """Detect specific object using Grounding DINO."""
        if not self.models_loaded or not prompt:
            return []
        
        try:
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            pil_image = Image.fromarray(image)
            h, w = image.shape[:2]
            
            inputs = self.dino_processor(
                images=pil_image, 
                text=prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.dino_model(**inputs)
            
            target_sizes = torch.tensor([[h, w]], device=self.device)
            
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=target_sizes,
            )[0]
            
            detected = []
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results.get("text_labels", results.get("labels", []))
            
            if len(boxes) > 0:
                sorted_indices = torch.argsort(scores, descending=True)
                
                # Return ALL high-confidence detections (up to 10)
                for idx in sorted_indices[:10]:
                    i = idx.item()
                    score = scores[i].item()
                    
                    # Lower threshold to catch more objects
                    if score < 0.15:
                        continue
                    
                    box = boxes[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    label = labels[i] if i < len(labels) else "object"
                    
                    detected.append(DetectedObject(
                        instance_id=len(detected) + 1,
                        class_name=str(label),
                        bbox=(x1, y1, x2, y2),
                        confidence=score
                    ))
            
            if detected:
                print(f"[DINO] Found {len(detected)}: {[d.class_name for d in detected]}")
            
            return detected
            
        except Exception as e:
            print(f"[Tracker] DINO error: {e}")
            return []
    
    def detect_with_gemini_bbox(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Use Gemini's bounding boxes ONLY for custom/unseen objects.
        Only returns objects that have valid bbox (custom objects).
        Generic objects should be handled by DINO, not here.
        """
        detected = []
        
        with self.prompt_lock:
            gemini_dets = list(self.current_gemini_detections)  # Copy
        
        if not gemini_dets:
            return detected
        
        h, w = image.shape[:2]
        
        for i, gd in enumerate(gemini_dets):
            # ONLY use Gemini bbox if:
            # 1. It's a custom object (not generic)
            # 2. It has a valid bbox from Gemini
            is_generic = self._is_generic_object(gd.description) if hasattr(self, '_is_generic_object') else False
            
            if is_generic:
                # Skip generic objects - DINO should handle them
                continue
            
            if not gd.bbox or len(gd.bbox) < 4:
                # No valid bbox - can't use Gemini detection
                continue
            
            # Convert normalized (0-1) to absolute pixel coordinates
            x1 = int(gd.bbox[0] * w)
            y1 = int(gd.bbox[1] * h)
            x2 = int(gd.bbox[2] * w)
            y2 = int(gd.bbox[3] * h)
            
            # Ensure valid bounds with minimum size
            x1 = max(0, min(w - 30, x1))
            y1 = max(0, min(h - 30, y1))
            x2 = max(x1 + 30, min(w, x2))
            y2 = max(y1 + 30, min(h, y2))
            
            # Skip if bbox is too large (probably hallucinated)
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            if bbox_area > frame_area * 0.7:  # More than 70% of frame is suspicious
                print(f"[Tracker] Skipping large bbox for '{gd.description}'")
                continue
            
            detected.append(DetectedObject(
                instance_id=len(detected) + 1,
                class_name=gd.description[:30],
                bbox=(x1, y1, x2, y2),
                confidence=gd.confidence
            ))
        
        if detected:
            print(f"[Gemini] {len(detected)} bbox(es): {[d.class_name for d in detected]}")
        
        return detected
    
    @torch.inference_mode()
    def segment(self, image: np.ndarray, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Generate masks using SAM with proper format for both SAM1 and SAM2."""
        if not self.models_loaded or not detections:
            return detections
        
        try:
            pil_image = Image.fromarray(image)
            h, w = image.shape[:2]
            is_sam2 = "sam2" in self.sam_model_name
            
            for det in detections:
                try:
                    x1, y1, x2, y2 = det.bbox
                    
                    # Ensure valid bbox coordinates (pixel values)
                    x1 = max(0, min(w - 1, int(x1)))
                    y1 = max(0, min(h - 1, int(y1)))
                    x2 = max(x1 + 10, min(w, int(x2)))
                    y2 = max(y1 + 10, min(h, int(y2)))
                    
                    # Use center point as prompt (more reliable than boxes for HF models)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # Input points format: [[[x, y]]] for single point
                    input_points = [[[float(cx), float(cy)]]]
                    input_labels = [[1]]  # 1 = foreground point
                    
                    inputs = self.sam_processor(
                        images=pil_image,
                        input_points=input_points,
                        input_labels=input_labels,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.sam_model(**inputs)
                    
                    # Post-process masks
                    masks = self.sam_processor.image_processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"].cpu(),
                        inputs["reshaped_input_sizes"].cpu()
                    )
                    
                    if masks and len(masks) > 0 and len(masks[0]) > 0:
                        mask_tensor = masks[0][0]
                        
                        # Handle different output shapes
                        if len(mask_tensor.shape) == 3:
                            # Multiple mask predictions - take best one (usually last or middle)
                            # SAM predicts 3 masks with different granularity
                            scores = outputs.iou_scores[0][0] if hasattr(outputs, 'iou_scores') else None
                            if scores is not None:
                                best_idx = torch.argmax(scores).item()
                            else:
                                best_idx = mask_tensor.shape[0] - 1  # Take last (largest mask)
                            mask = mask_tensor[best_idx].numpy()
                        else:
                            mask = mask_tensor.numpy()
                        
                        det.mask = mask > 0.5
                        
                except Exception as seg_err:
                    # Fallback: create simple rectangular mask from bbox
                    try:
                        mask = np.zeros((h, w), dtype=bool)
                        x1, y1, x2, y2 = det.bbox
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(x2)), min(h, int(y2))
                        mask[y1:y2, x1:x2] = True
                        det.mask = mask
                    except:
                        pass
                    continue
            
            return detections
            
        except Exception as e:
            print(f"[Tracker] SAM error: {e}")
            return detections
    
    @torch.inference_mode()
    def detect_with_sam3(self, frame: np.ndarray, text_prompt: str) -> List[DetectedObject]:
        """
        SAM3 text-based detection - segments ALL instances of the concept!
        Much simpler than SAM2: just pass text, get masks.
        """
        if not self.models_loaded or not self.is_sam3:
            return []
        
        try:
            pil_image = Image.fromarray(frame)
            h, w = frame.shape[:2]
            
            # SAM3 takes text prompt directly
            inputs = self.sam_processor(
                images=pil_image, 
                text=text_prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.sam_model(**inputs)
            
            # Post-process to get instance masks
            results = self.sam_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=[(h, w)]
            )[0]
            
            detections = []
            masks = results.get("masks", [])
            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            
            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                if score < 0.3:  # Skip low confidence
                    continue
                
                # Convert box to integers
                if hasattr(box, 'cpu'):
                    box = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # Convert mask to numpy
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                
                det = DetectedObject(
                    instance_id=i + 1,
                    class_name=text_prompt[:30],
                    bbox=(x1, y1, x2, y2),
                    mask=mask_np > 0.5,
                    confidence=float(score)
                )
                detections.append(det)
            
            if detections:
                print(f"[SAM3] Found {len(detections)} '{text_prompt}' instance(s)")
            
            return detections
            
        except Exception as e:
            print(f"[SAM3] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectedObject]]:
        """
        Process frame with smart detection.
        
        For SAM3: Uses text prompts directly (semantic understanding)
        For SAM1/SAM2: Uses DINO + SAM (visual prompts)
        """
        prompt = self.get_prompt()
        
        # NO prompt = NO highlighting
        if not prompt or not self.models_loaded:
            return frame, []
        
        self.frame_count += 1
        
        # Re-detect periodically or when no detections
        need_detection = (
            not self.last_detections or 
            self.frame_count % 25 == 1  # Less frequent to reduce noise
        )
        
        if need_detection:
            # SAM3: Use text-based detection (much simpler!)
            if self.is_sam3:
                detections = self._process_frame_sam3(frame, prompt)
            else:
                detections = self._process_frame_sam2(frame, prompt)
            
            if detections:
                self.last_detections = detections
                print(f"[Tracker] Total: {len(detections)} object(s)")
            else:
                self.last_detections = []
        else:
            detections = self.last_detections
            
            # Update masks periodically for tracking (only for SAM2)
            if not self.is_sam3 and self.frame_count % 10 == 0 and detections:
                detections = self.segment(frame, detections)
                self.last_detections = detections
        
        annotated = self.annotate_frame(frame, detections)
        return annotated, detections
    
    def _process_frame_sam3(self, frame: np.ndarray, prompt: str) -> List[DetectedObject]:
        """Process frame using SAM3 text-based segmentation."""
        # Parse prompt - may contain multiple objects like "phone. wallet."
        # SAM3 can handle each concept
        objects = [obj.strip() for obj in prompt.replace(".", " ").split() if obj.strip()]
        objects = list(set(objects))[:5]  # Limit to 5 unique objects
        
        all_detections = []
        for obj_name in objects:
            if len(obj_name) < 2:
                continue
            detections = self.detect_with_sam3(frame, obj_name)
            all_detections.extend(detections)
        
        return all_detections
    
    def _process_frame_sam2(self, frame: np.ndarray, prompt: str) -> List[DetectedObject]:
        """Process frame using DINO + SAM2 (visual prompts)."""
        # Step 1: Try DINO for everything
        detections = self.detect_with_dino(frame, prompt)
        dino_found = {d.class_name.lower() for d in detections}
        
        if detections:
            print(f"[DINO] Found: {list(dino_found)}")
        
        # Step 2: Check if there are CUSTOM objects that DINO missed
        with self.prompt_lock:
            gemini_dets = list(self.current_gemini_detections)
        
        # Only add Gemini bbox for CUSTOM objects that DINO couldn't find
        for gd in gemini_dets:
            desc_lower = gd.description.lower()
            
            # Skip if DINO already found something similar
            found_match = any(
                desc_lower in found or found in desc_lower 
                for found in dino_found
            )
            if found_match:
                continue
            
            # Only use Gemini bbox if:
            # 1. It's a CUSTOM object (not generic)
            # 2. It has a valid bbox
            # 3. DINO didn't find it
            is_custom = not self._is_generic_object(gd.description)
            has_bbox = gd.bbox is not None and len(gd.bbox) == 4
            
            if is_custom and has_bbox:
                h, w = frame.shape[:2]
                x1 = int(gd.bbox[0] * w)
                y1 = int(gd.bbox[1] * h)
                x2 = int(gd.bbox[2] * w)
                y2 = int(gd.bbox[3] * h)
                
                # Validate bbox
                x1 = max(0, min(w - 20, x1))
                y1 = max(0, min(h - 20, y1))
                x2 = max(x1 + 20, min(w, x2))
                y2 = max(y1 + 20, min(h, y2))
                
                det = DetectedObject(
                    instance_id=len(detections) + 1,
                    class_name=gd.description[:30],
                    bbox=(x1, y1, x2, y2),
                    confidence=0.7
                )
                detections.append(det)
                print(f"[Tracker] Added custom bbox: {gd.description}")
        
        # Segment all detections
        if detections:
            detections = self.segment(frame, detections)
        
        return detections
    
    def _is_generic_object(self, description: str) -> bool:
        """Check if object is generic (DINO handles well) or custom (needs Gemini bbox)."""
        desc_lower = description.lower().strip()
        
        # Generic objects that DINO handles well
        generic_objects = {
            "phone", "smartphone", "mobile", "wallet", "purse", "bag",
            "bottle", "cup", "mug", "glass", "laptop", "computer", "keyboard",
            "mouse", "monitor", "pen", "pencil", "book", "paper", "chair",
            "table", "desk", "door", "window", "car", "person", "man", "woman",
            "cat", "dog", "plant", "flower", "clock", "watch", "tv", "remote",
            "shoe", "shirt", "hat", "glasses", "box", "light", "lamp", "fan",
        }
        
        for generic in generic_objects:
            if generic in desc_lower:
                return True
        
        return False
    
    def annotate_frame(self, frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """Annotate frame with segmentation masks ONLY (no labels)."""
        if not detections:
            return frame
        
        annotated = frame.copy()
        
        for i, det in enumerate(detections):
            color = self.colors[i % len(self.colors)]
            
            # Draw segmentation mask with semi-transparent overlay
            if det.mask is not None:
                mask_overlay = annotated.copy()
                mask_overlay[det.mask] = color
                annotated = cv2.addWeighted(annotated, 0.6, mask_overlay, 0.4, 0)
                
                # Draw contour outline
                mask_uint8 = (det.mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, color, 2)
            else:
                # No mask - draw thin bounding box only
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # NO LABELS - user requested clean segmentation only
        
        return annotated


# Backwards compatibility
GroundedSAMTracker = HybridSAMTracker
GroundedSAM2Tracker = HybridSAMTracker
KeywordExtractor = PreciseGeminiExtractor


if __name__ == "__main__":
    print("Testing Precise Gemini Extractor...")
    
    # Test fallback
    extractor = PreciseGeminiExtractor(api_key=None)
    tests = [
        "What's in my hand?",
        "Show me the positive terminal",
        "Find the flat head screwdriver",
        "Highlight the warning sign",
    ]
    
    for t in tests:
        results = extractor._fallback(t)
        print(f"  '{t}' → {[r.description for r in results]}")
