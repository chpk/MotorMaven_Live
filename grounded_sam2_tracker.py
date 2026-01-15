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
    description: str  # Specific visual description (for SAM3: make it descriptive!)
    bbox: Optional[Tuple[float, float, float, float]] = None  # Normalized 0-1: x1, y1, x2, y2
    confidence: float = 0.8
    object_type: str = "generic"  # "generic" or "custom" - custom objects need box prompts


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
    
    SYSTEM_PROMPT = """You are an object identifier for video segmentation. Extract ONLY the specific objects the user wants to see highlighted.

OUTPUT FORMAT (JSON only):
{"objects":[
  {"description":"object_name", "type":"generic|custom", "bbox":[x1,y1,x2,y2]}
]}

RULES:
1. "description": Simple, clear object name that a segmentation model can understand
   - Use common nouns: "phone", "laptop", "cup", "person", "hand"
   - For specific items: "phone case", "coffee mug", "red wire"
   - Keep it short (1-3 words max)

2. "type": 
   - "generic" for common visible objects (phone, person, cup, chair, laptop, hand)
   - "custom" for text/labels, industrial parts, or objects not visually obvious

3. "bbox": ONLY provide for "custom" type. Use null for "generic".
   - Format: [left, top, right, bottom] normalized 0.0-1.0
   - Be precise for custom objects

IMPORTANT:
- ONLY include objects the user explicitly asks about
- If user says "highlight the phone", return: {"objects":[{"description":"phone","type":"generic","bbox":null}]}
- If user says "show me what's in my hand", identify what they're holding
- Don't add objects the user didn't mention

Return JSON only."""

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
                            confidence=0.85 if obj_type == "custom" else 0.9,
                            object_type=obj_type  # Store the type!
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
        
        # Gemini extractor - ALWAYS needed to understand what user wants to track
        # SAM3 handles segmentation, but we need Gemini to extract object names from conversation
        if api_key:
            print(f"[Tracker] Creating Gemini extractor with API key ({len(api_key)} chars)")
            self.gemini_extractor = PreciseGeminiExtractor(api_key)
            if self.is_sam3:
                print("[Tracker] SAM3 mode - Gemini extracts objects, SAM3 segments them")
        else:
            print("[Tracker] WARNING: No API key provided - Gemini extractor disabled!")
            self.gemini_extractor = None
        
        # State
        self.current_prompt = ""
        self.current_gemini_detections: List[GeminiDetection] = []
        self.prompt_lock = threading.Lock()
        self.models_loaded = False
        
        # SAM3 Video Tracker State
        self.sam3_video_model = None
        self.sam3_predictor = None
        self.sam3_inference_state = None
        self.sam3_tracked_objects = {}  # obj_id -> description
        self.sam3_next_obj_id = 1
        self.sam3_frame_width = 0
        self.sam3_frame_height = 0
        self.sam3_temp_dir = None
        
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
    
    def __del__(self):
        """Cleanup resources."""
        self._cleanup_sam3_temp_dir()
    
    def _cleanup_sam3_temp_dir(self):
        """Remove SAM3 temporary video directory."""
        try:
            if self.sam3_temp_dir and os.path.exists(self.sam3_temp_dir):
                import shutil
                shutil.rmtree(self.sam3_temp_dir)
                self.sam3_temp_dir = None
        except Exception:
            pass  # Silently ignore cleanup errors during shutdown
    
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
        """Load SAM3 VIDEO PREDICTOR for real-time tracking with TEXT prompts."""
        import warnings
        
        print(f"[Tracker] Loading SAM3 VIDEO PREDICTOR ({sam_info['params']})")
        print("[Tracker] SAM3 supports: TEXT PROMPTS, point prompts, box prompts")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                from sam3.model_builder import build_sam3_video_predictor
                
                print("[Tracker] Building SAM3 video predictor (this may take a moment)...")
                self.sam3_predictor = build_sam3_video_predictor(gpus_to_use=[0])
                
                # Session management for real-time streaming
                self.sam3_session_id = None
                self.sam3_session_frame_dir = None
                self.sam3_current_prompts = {}  # obj_id -> description
                self.sam3_next_obj_id = 1
                self.sam3_last_masks = {}  # obj_id -> mask
                
                self.models_loaded = True
                print("[Tracker] SAM3 VIDEO PREDICTOR loaded!")
                print("[Tracker] Use descriptive TEXT prompts: 'black phone', 'red box', 'white motor with grills'")
                return True
                
        except ImportError as e:
            print(f"[Tracker] SAM3 not installed: {e}")
            print("[Tracker] Install: cd sam3 && pip install -e .")
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
    
    def _init_sam3_session(self, frames: List[np.ndarray]):
        """Initialize SAM3 video session for tracking."""
        import tempfile
        import uuid
        
        # Close any existing session
        if self.sam3_session_id:
            try:
                self.sam3_predictor.close_session(self.sam3_session_id)
            except:
                pass
        
        # Cleanup old frame directory
        self._cleanup_sam3_temp_dir()
        
        # Create new frame directory
        self.sam3_temp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        
        # Save frames to directory
        for i, frame in enumerate(frames):
            frame_path = os.path.join(self.sam3_temp_dir, f"{i:05d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Start session with frame directory
        session_result = self.sam3_predictor.start_session(
            resource_path=self.sam3_temp_dir
        )
        
        # Extract session_id from result (may be dict or string)
        if isinstance(session_result, dict):
            self.sam3_session_id = session_result.get('session_id', session_result)
        else:
            self.sam3_session_id = session_result
        
        # Clear tracked objects
        self.sam3_current_prompts = {}
        self.sam3_next_obj_id = 1
        self.sam3_last_masks = {}
        
        print(f"[SAM3] Session started: {self.sam3_session_id}")
        return self.sam3_session_id
    
    def _add_sam3_text_prompt(self, description: str, frame_idx: int = 0) -> Optional[dict]:
        """Add object to SAM3 using TEXT prompt - returns immediate detection result!"""
        if not self.sam3_session_id:
            return None
        
        try:
            obj_id = self.sam3_next_obj_id
            self.sam3_next_obj_id += 1
            
            # Use SAM3 text prompt API - returns immediate segmentation!
            response = self.sam3_predictor.add_prompt(
                session_id=self.sam3_session_id,
                frame_idx=frame_idx,
                text=description,  # THE TEXT PROMPT!
                obj_id=obj_id
            )
            
            if response:
                self.sam3_current_prompts[obj_id] = description
                print(f"[SAM3] Text prompt: '{description}' (ID={obj_id})")
                return {
                    'obj_id': obj_id,
                    'description': description,
                    'response': response
                }
            
            return None
            
        except Exception as e:
            print(f"[SAM3] Text prompt error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _add_sam3_box_prompt(self, description: str, bbox: Tuple[float, float, float, float], 
                             frame_idx: int = 0) -> Optional[dict]:
        """Add object using bounding box prompt (for custom objects) - returns immediate result!"""
        if not self.sam3_session_id:
            return None
        
        try:
            obj_id = self.sam3_next_obj_id
            self.sam3_next_obj_id += 1
            
            # Convert bbox (normalized 0-1) to list format
            x1, y1, x2, y2 = bbox
            boxes = [[x1, y1, x2, y2]]
            box_labels = [1]  # Positive box
            
            response = self.sam3_predictor.add_prompt(
                session_id=self.sam3_session_id,
                frame_idx=frame_idx,
                bounding_boxes=boxes,
                bounding_box_labels=box_labels,
                obj_id=obj_id
            )
            
            if response:
                self.sam3_current_prompts[obj_id] = description
                print(f"[SAM3] Box prompt: '{description}' (ID={obj_id})")
                return {
                    'obj_id': obj_id,
                    'description': description,
                    'response': response
                }
            
            return None
            
        except Exception as e:
            print(f"[SAM3] Box prompt error: {e}")
            return None
    
    @torch.inference_mode()
    def detect_with_sam3_video(self, frame: np.ndarray, objects: List[GeminiDetection]) -> List[DetectedObject]:
        """
        SAM3 VIDEO detection using TEXT PROMPTS!
        
        SAM3 understands natural language and returns immediate segmentation:
        - "black phone case"
        - "white motor with cooling grills" 
        - "red warning label"
        - "ABB logo"
        - "green circuit board"
        
        Each text prompt returns a segmentation mask directly.
        """
        if not self.models_loaded or not self.is_sam3:
            return []
        
        if not objects:
            return []
        
        try:
            h, w = frame.shape[:2]
            
            # Initialize session with current frame
            self._init_sam3_session([frame])
            
            detections = []
            
            # Process each object with SAM3 text prompts
            for obj in objects:
                desc = obj.description.strip()
                
                # Use TEXT prompt - returns immediate segmentation!
                if obj.object_type == "custom" and obj.bbox:
                    # Custom object with bbox - use box prompt
                    result = self._add_sam3_box_prompt(desc, obj.bbox, frame_idx=0)
                else:
                    # Use text prompt - SAM3's main feature!
                    result = self._add_sam3_text_prompt(desc, frame_idx=0)
                
                if not result or 'response' not in result:
                    continue
                
                # Extract detection from immediate response
                response = result['response']
                obj_id = result['obj_id']
                description = result['description']
                
                outputs = response.get('outputs', {})
                masks = outputs.get('out_binary_masks', [])
                probs = outputs.get('out_probs', [])
                
                if masks is None or len(masks) == 0:
                    continue
                
                # Get the first (and usually only) detection
                mask = masks[0] if len(masks) > 0 else None
                prob = probs[0] if len(probs) > 0 else 0.9
                
                if mask is None:
                    continue
                
                # Convert to numpy
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                else:
                    mask = np.array(mask)
                
                # Ensure 2D
                while len(mask.shape) > 2:
                    mask = mask[0]
                
                mask = mask.astype(bool)
                
                # Resize if needed
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                
                # Skip empty masks
                if mask.sum() == 0:
                    print(f"[SAM3] '{description}' - no mask found")
                    continue
                
                # Get bbox from mask
                ys, xs = np.where(mask)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                
                det = DetectedObject(
                    instance_id=obj_id,
                    class_name=description[:30],
                    bbox=(x1, y1, x2, y2),
                    mask=mask,
                    confidence=float(prob)
                )
                detections.append(det)
                self.sam3_last_masks[obj_id] = mask
                print(f"[SAM3] '{description}': bbox=({x1},{y1},{x2},{y2}), conf={prob:.2f}")
            
            if detections:
                print(f"[SAM3] Total: {len(detections)} object(s) detected")
            
            return detections
            
        except Exception as e:
            print(f"[SAM3] Video detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def clear_sam3_tracking(self):
        """Clear all SAM3 tracked objects and close session."""
        if hasattr(self, 'sam3_predictor') and self.sam3_session_id:
            try:
                self.sam3_predictor.close_session(self.sam3_session_id)
            except:
                pass
        self.sam3_session_id = None
        self.sam3_current_prompts = {}
        self.sam3_next_obj_id = 1
        self.sam3_last_masks = {}
        self._cleanup_sam3_temp_dir()
        print("[SAM3] Tracking cleared")
    
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
    
    def _enhance_prompt_for_sam3(self, obj_name: str) -> str:
        """
        Enhance object prompt for SAM3 with descriptive attributes.
        
        SAM3 works best with descriptive, attribute-based prompts like:
        - "black phone case" instead of "phone"
        - "white motor with cooling grills" instead of "motor"
        - "green circuit board with components" instead of "PCB"
        - "red warning label" instead of "label"
        """
        # Map generic names to more descriptive prompts
        descriptor_map = {
            # Electronics
            "phone": "black phone device",
            "laptop": "gray laptop computer",
            "monitor": "black computer monitor screen",
            "keyboard": "gray computer keyboard",
            "mouse": "black computer mouse",
            "cable": "black cable wire",
            "charger": "black charging adapter",
            "pcb": "green circuit board with electronic components",
            "circuit board": "green printed circuit board",
            
            # Industrial
            "motor": "gray electric motor with cooling fins",
            "pump": "metallic industrial pump",
            "valve": "metal valve component",
            "pipe": "metal pipe tubing",
            "conveyor": "industrial conveyor belt system",
            "sensor": "small electronic sensor module",
            "button": "colored control button",
            "switch": "electrical switch toggle",
            "gauge": "circular gauge meter display",
            
            # Tools/Objects
            "tool": "metal hand tool",
            "box": "cardboard box container",
            "bottle": "plastic bottle container",
            "cup": "drinking cup or mug",
            "pen": "writing pen or pencil",
            "book": "rectangular book or document",
            "paper": "white paper sheet",
            "label": "printed label sticker",
            
            # Furniture
            "table": "flat surface table",
            "chair": "seating chair",
            "shelf": "storage shelf rack",
            "cabinet": "storage cabinet unit",
            
            # Safety
            "sign": "warning or information sign",
            "light": "illuminated light fixture",
            "warning": "yellow warning sign",
            "logo": "company logo branding",
        }
        
        obj_lower = obj_name.lower().strip()
        words = obj_lower.split()
        
        # Color/attribute words to preserve
        color_words = ["red", "blue", "green", "black", "white", "gray", "grey", 
                       "yellow", "orange", "brown", "silver", "metallic", "shiny"]
        attribute_words = ["large", "small", "big", "tall", "round", "square", 
                          "left", "right", "top", "bottom", "front", "back"]
        
        # Extract user-specified attributes
        user_color = None
        user_attr = None
        base_obj = []
        
        for word in words:
            if word in color_words:
                user_color = word
            elif word in attribute_words:
                user_attr = word
            else:
                base_obj.append(word)
        
        base_obj_str = " ".join(base_obj)
        
        # If user specified color/attribute, use it with enhanced base
        if user_color or user_attr or len(words) > 1:
            # Already descriptive - enhance base object only
            if base_obj_str in descriptor_map:
                enhanced_base = descriptor_map[base_obj_str]
                # Replace default color with user color if specified
                if user_color:
                    # Remove default color from enhanced
                    for c in color_words:
                        enhanced_base = enhanced_base.replace(c + " ", "")
                    return f"{user_color} {enhanced_base}".strip()
                return f"{user_attr} {enhanced_base}".strip() if user_attr else enhanced_base
            return obj_name  # Already descriptive enough
        
        # Check for exact match on single word
        if obj_lower in descriptor_map:
            return descriptor_map[obj_lower]
        
        # Check for partial match
        for key, desc in descriptor_map.items():
            if key in obj_lower or obj_lower in key:
                return desc
        
        # Default: add generic descriptor
        return f"{obj_name} object"
    
    def _interpret_custom_object(self, obj_name: str, frame: np.ndarray = None) -> str:
        """
        Interpret custom/unseen object and generate descriptive prompt.
        
        For unfamiliar objects, transform the name into a visual description:
        - "PCB" -> "green circuit board with electronic components"
        - "ABB motor" -> "gray industrial motor with ABB logo"
        - "T1 text" -> "text label showing T1"
        """
        obj_lower = obj_name.lower().strip()
        
        # Special interpretations for common industrial/technical terms
        interpretations = {
            "abb": "ABB branded industrial equipment",
            "siemens": "Siemens branded equipment",
            "allen bradley": "Allen Bradley control equipment",
            "rockwell": "Rockwell automation equipment",
            "plc": "programmable logic controller box",
            "hmi": "human machine interface touchscreen",
            "vfd": "variable frequency drive unit",
            "relay": "electrical relay module",
            "contactor": "electrical contactor switch",
            "terminal": "electrical terminal block",
            "fuse": "electrical fuse component",
            "breaker": "circuit breaker switch",
            "transformer": "electrical transformer unit",
        }
        
        # Check for brand/technical terms
        for term, interpretation in interpretations.items():
            if term in obj_lower:
                return interpretation
        
        # Pattern: "XX text" or "XX label" -> interpret as text/label
        if "text" in obj_lower or "label" in obj_lower:
            parts = obj_lower.replace("text", "").replace("label", "").strip()
            if parts:
                return f"text or label showing {parts}"
        
        # Pattern: "XX logo" -> interpret as logo
        if "logo" in obj_lower:
            parts = obj_lower.replace("logo", "").strip()
            if parts:
                return f"{parts} company logo branding"
        
        return self._enhance_prompt_for_sam3(obj_name)
    
    def _process_frame_sam3(self, frame: np.ndarray, prompt: str) -> List[DetectedObject]:
        """
        Process frame using SAM3 VIDEO TRACKER with descriptive prompts.
        
        SAM3 Video Strategy:
        1. Get objects from Gemini detections (extracted from user speech)
        2. Enhance prompts with descriptive attributes
        3. For generic objects: add via point prompt at expected location
        4. For custom/unseen objects: interpret and use box prompt
        5. Track all objects across frames using SAM3 video propagation
        """
        h, w = frame.shape[:2]
        
        # Get Gemini-extracted objects
        with self.prompt_lock:
            gemini_dets = list(self.current_gemini_detections)
        
        # Prepare object list for SAM3 video tracking
        objects_to_track = []
        
        # If no Gemini detections, parse the prompt directly
        if not gemini_dets and prompt:
            # Parse "phone. wallet." style prompt
            parsed_objects = [obj.strip() for obj in prompt.replace(".", " ").split() if obj.strip()]
            parsed_objects = list(set(parsed_objects))[:5]
            
            for obj_name in parsed_objects:
                if len(obj_name) < 2:
                    continue
                
                # Enhance with descriptive prompt
                enhanced_desc = self._enhance_prompt_for_sam3(obj_name)
                
                objects_to_track.append(GeminiDetection(
                    description=enhanced_desc,
                    confidence=0.8,
                    object_type="generic",
                    bbox=None
                ))
        else:
            # Use Gemini-extracted objects
            seen_objects = set()
            
            for gd in gemini_dets[:5]:  # Limit to 5 objects
                obj_name = gd.description.lower().strip()
                
                # Skip duplicates
                if obj_name in seen_objects:
                    continue
                seen_objects.add(obj_name)
                
                # Check if it's a custom/unseen object
                is_custom = gd.object_type == "custom" or not self._is_generic_object(obj_name)
                
                if is_custom:
                    # Interpret custom object for better SAM3 understanding
                    enhanced_desc = self._interpret_custom_object(obj_name, frame)
                    print(f"[SAM3] Custom object '{obj_name}' -> '{enhanced_desc}'")
                else:
                    # Enhance generic object description
                    enhanced_desc = self._enhance_prompt_for_sam3(obj_name)
                
                objects_to_track.append(GeminiDetection(
                    description=enhanced_desc,
                    confidence=gd.confidence,
                    object_type=gd.object_type,
                    bbox=gd.bbox
                ))
        
        # Use SAM3 video tracking
        return self.detect_with_sam3_video(frame, objects_to_track)
    
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
