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
    Works for both common objects AND custom/industrial objects.
    """
    
    SYSTEM_PROMPT = """You are a precise object detector. Identify ALL objects mentioned by the user.

TASK: Look at the image and find EVERY object the user is asking about.

OUTPUT FORMAT (JSON only):
{"objects":[
  {"description":"simple_name", "bbox":[x1, y1, x2, y2]},
  {"description":"simple_name", "bbox":[x1, y1, x2, y2]}
]}

CRITICAL RULES:
1. Include ALL objects the user mentions - if they say "phone and wallet", include BOTH
2. Use simple 1-2 word descriptions: "phone", "wallet", "red mug", "motor", "terminal"
3. BBOX must be accurate normalized coordinates [0.0-1.0]: [left, top, right, bottom]

BBOX EXAMPLES:
- Left object: [0.05, 0.2, 0.4, 0.8]
- Right object: [0.6, 0.2, 0.95, 0.8]  
- Center object: [0.25, 0.25, 0.75, 0.75]
- Top-left small: [0.05, 0.05, 0.3, 0.35]

For CUSTOM objects (motors, terminals, screws, text/signs, labels):
- Provide ACCURATE bounding boxes since detection models may not recognize them
- Be very precise with bbox coordinates

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
    
    def extract(self, frame: np.ndarray, text: str, frame_size: Tuple[int, int]) -> List[GeminiDetection]:
        """
        Extract ALL objects mentioned by the user with accurate bboxes.
        """
        self._ensure_client()
        if self.client is None:
            print("[GeminiExtractor] No client - using fallback")
            return self._fallback(text, frame_size)
        
        try:
            # Convert frame to JPEG
            pil_img = Image.fromarray(frame)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=75)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            
            prompt = f"""{self.SYSTEM_PROMPT}

User's request: "{text}"

Instructions:
- Find ALL objects mentioned in the user's request
- If user says "highlight X and Y" - include BOTH X and Y
- If user asks "what is this" - identify the main object being shown
- Provide accurate bounding boxes for each object
- For custom/industrial/text objects, be extra precise with bbox

Return JSON with ALL requested objects."""
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    prompt
                ]
            )
            
            result = response.text.strip()
            print(f"[GeminiExtractor] Raw response: {result[:400]}")
            
            # Parse JSON response
            detections = self._parse_response(result)
            
            # Validate and fix bboxes
            valid_detections = []
            for d in detections:
                if d.bbox is None:
                    d.bbox = (0.2, 0.2, 0.8, 0.8)  # Default center region
                    print(f"[GeminiExtractor] '{d.description}' - no bbox, using default")
                else:
                    # Validate bbox values are in 0-1 range
                    x1, y1, x2, y2 = d.bbox
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))
                    if x2 > x1 and y2 > y1:
                        d.bbox = (x1, y1, x2, y2)
                        print(f"[GeminiExtractor] '{d.description}' at bbox={d.bbox}")
                    else:
                        d.bbox = (0.2, 0.2, 0.8, 0.8)
                        print(f"[GeminiExtractor] '{d.description}' - invalid bbox, using default")
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
        """Parse Gemini's JSON response - robust extraction with fallbacks."""
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
                
                for obj in objects[:10]:  # Allow up to 10 objects
                    if isinstance(obj, dict):
                        desc = obj.get("description", obj.get("name", "")).strip()
                        bbox = obj.get("bbox", obj.get("bounding_box", obj.get("box")))
                        
                        if not desc:
                            continue
                        
                        # Parse bbox
                        valid_bbox = self._parse_bbox(bbox)
                        
                        detections.append(GeminiDetection(
                            description=desc,
                            bbox=valid_bbox,
                            confidence=0.85
                        ))
            
            # Fallback: extract description with regex if JSON parsing failed
            if not detections:
                # Look for "description": "..." pattern
                desc_matches = re.findall(r'"description"\s*:\s*"([^"]+)"', response)
                for desc in desc_matches[:3]:
                    if desc and len(desc) > 2:
                        detections.append(GeminiDetection(
                            description=desc.strip(),
                            bbox=(0.2, 0.2, 0.8, 0.8),  # Default center bbox
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
    Hybrid tracker that combines:
    1. Gemini for specific descriptions + approximate bboxes
    2. Grounding DINO for text-based detection
    3. SAM/SAM2 for precise segmentation (multiple model options)
    
    Supports multiple SAM model variants for accuracy/speed tradeoff.
    """
    
    # Available SAM models
    SAM_MODELS = {
        # HuggingFace models (auto-download)
        "sam-vit-base": {"type": "hf", "id": "facebook/sam-vit-base", "params": "94M"},
        "sam-vit-large": {"type": "hf", "id": "facebook/sam-vit-large", "params": "308M"},
        "sam-vit-huge": {"type": "hf", "id": "facebook/sam-vit-huge", "params": "636M"},
        "sam2-hiera-tiny": {"type": "hf", "id": "facebook/sam2-hiera-tiny", "params": "38.9M"},
        "sam2-hiera-small": {"type": "hf", "id": "facebook/sam2-hiera-small", "params": "46M"},
        "sam2-hiera-base-plus": {"type": "hf", "id": "facebook/sam2-hiera-base-plus", "params": "80.8M"},
        "sam2-hiera-large": {"type": "hf", "id": "facebook/sam2-hiera-large", "params": "224.4M"},
    }
    
    def __init__(self, api_key: str = None, sam_model: str = "sam2-hiera-small"):
        self.device = DEVICE
        self.api_key = api_key
        self.sam_model_name = sam_model
        
        # Models
        self.dino_model = None
        self.dino_processor = None
        self.sam_model = None
        self.sam_processor = None
        
        # Gemini extractor
        if api_key:
            print(f"[Tracker] Creating Gemini extractor with API key ({len(api_key)} chars)")
            self.gemini_extractor = PreciseGeminiExtractor(api_key)
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
            self.models_loaded = False
            self.sam_model = None
            self.sam_processor = None
            print(f"[Tracker] SAM model set to: {model_name}")
        else:
            print(f"[Tracker] Unknown model: {model_name}")
    
    def load_models(self) -> bool:
        """Load Grounding DINO and selected SAM model."""
        if self.models_loaded:
            return True
        
        try:
            import warnings
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            
            # Load Grounding DINO
            print("[Tracker] Loading Grounding DINO...")
            self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(self.device)
            self.dino_model.eval()
            
            # Load selected SAM model
            sam_info = self.SAM_MODELS.get(self.sam_model_name)
            if not sam_info:
                print(f"[Tracker] Unknown SAM model: {self.sam_model_name}, using default")
                sam_info = self.SAM_MODELS["sam2-hiera-small"]
                self.sam_model_name = "sam2-hiera-small"
            
            print(f"[Tracker] Loading SAM: {self.sam_model_name} ({sam_info['params']})...")
            
            # Load from HuggingFace
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                if "sam2" in self.sam_model_name:
                    # SAM2 models
                    from transformers import Sam2Model, Sam2Processor
                    self.sam_processor = Sam2Processor.from_pretrained(sam_info["id"])
                    self.sam_model = Sam2Model.from_pretrained(sam_info["id"]).to(self.device)
                else:
                    # Original SAM models
                    from transformers import SamModel, SamProcessor
                    self.sam_processor = SamProcessor.from_pretrained(sam_info["id"])
                    self.sam_model = SamModel.from_pretrained(sam_info["id"]).to(self.device)
                
                self.sam_model.eval()
            
            self.models_loaded = True
            print(f"[Tracker] All models loaded! (SAM: {self.sam_model_name})")
            return True
            
        except Exception as e:
            print(f"[Tracker] Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
        Use Gemini's bounding boxes for custom/unseen objects.
        Critical for: motors, terminals, text, signs, industrial parts
        """
        detected = []
        
        with self.prompt_lock:
            gemini_dets = list(self.current_gemini_detections)  # Copy
        
        if not gemini_dets:
            return detected
        
        h, w = image.shape[:2]
        
        for i, gd in enumerate(gemini_dets):
            # Use Gemini's bbox if available, otherwise use center region
            if gd.bbox and len(gd.bbox) >= 4:
                bbox = gd.bbox
            else:
                # Default center region for objects without bbox
                bbox = (0.2, 0.2, 0.8, 0.8)
            
            # Convert normalized (0-1) to absolute pixel coordinates
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            # Ensure valid bounds
            x1 = max(0, min(w - 10, x1))
            y1 = max(0, min(h - 10, y1))
            x2 = max(x1 + 30, min(w, x2))
            y2 = max(y1 + 30, min(h, y2))
            
            detected.append(DetectedObject(
                instance_id=i + 1,
                class_name=gd.description[:40],
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
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectedObject]]:
        """
        Process frame with hybrid detection.
        
        Strategy:
        1. Try DINO for common objects
        2. Use Gemini bbox for custom/unseen objects
        3. Combine both if needed
        """
        prompt = self.get_prompt()
        
        # NO prompt = NO highlighting
        if not prompt or not self.models_loaded:
            return frame, []
        
        self.frame_count += 1
        
        # Re-detect periodically or when no detections
        need_detection = (
            not self.last_detections or 
            self.frame_count % 20 == 1  # More frequent updates
        )
        
        if need_detection:
            # Strategy 1: Try DINO
            detections = self.detect_with_dino(frame, prompt)
            
            # Strategy 2: Get Gemini bboxes for comparison/fallback
            with self.prompt_lock:
                gemini_count = len(self.current_gemini_detections)
            
            # If DINO found fewer than Gemini suggested, add Gemini bboxes
            if len(detections) < gemini_count:
                gemini_dets = self.detect_with_gemini_bbox(frame)
                
                # Add Gemini detections that DINO missed
                existing_labels = {d.class_name.lower() for d in detections}
                for gd in gemini_dets:
                    if gd.class_name.lower() not in existing_labels:
                        detections.append(gd)
                        print(f"[Tracker] Added Gemini bbox: {gd.class_name}")
            
            # If still no detections, use only Gemini
            if not detections:
                detections = self.detect_with_gemini_bbox(frame)
                print(f"[Tracker] Using Gemini bboxes only: {len(detections)}")
            
            if detections:
                detections = self.segment(frame, detections)
                self.last_detections = detections
                print(f"[Tracker] Total: {len(detections)} object(s)")
        else:
            detections = self.last_detections
            
            # Update masks for tracking
            if self.frame_count % 8 == 0 and detections:
                detections = self.segment(frame, detections)
                self.last_detections = detections
        
        annotated = self.annotate_frame(frame, detections)
        return annotated, detections
    
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
