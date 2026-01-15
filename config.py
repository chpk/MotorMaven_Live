"""
Configuration file for Gemini Live API Multimodal Application.
Store all API keys, endpoints, and model configurations here.
"""

# =============================================================================
# API CREDENTIALS
# =============================================================================
import os

# Get API key from environment variable (set GEMINI_API_KEY in your environment)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Please set it as an environment variable.")
    print("Example: export GEMINI_API_KEY='your-api-key-here'")

GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# System instruction for the AI assistant
SYSTEM_INSTRUCTION = """You are a helpful, friendly, and knowledgeable AI assistant. 
You engage in natural conversations, answer questions accurately, and provide helpful information.
You can see video from the user's camera and hear their voice in real-time.
Keep your responses concise, natural, and conversational."""

# Response modalities - "AUDIO" for voice output, "TEXT" for text output
RESPONSE_MODALITIES = ["AUDIO"]

# Available voice options for Gemini
VOICE_OPTIONS = [
    "Puck",      # Default male voice
    "Charon",    # Male voice
    "Kore",      # Female voice
    "Fenrir",    # Male voice
    "Aoede",     # Female voice
]
DEFAULT_VOICE = "Puck"

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================
# Audio format settings
AUDIO_FORMAT = "paInt16"  # 16-bit audio
CHANNELS = 1  # Mono audio

# Sample rates (Hz) - Gemini requires these specific rates
# If your device doesn't support these, the app will resample
SEND_SAMPLE_RATE = 16000    # Rate for sending audio to Gemini
RECEIVE_SAMPLE_RATE = 24000  # Rate for receiving audio from Gemini

# Fallback sample rates to try if device doesn't support target rate
FALLBACK_SAMPLE_RATES = [48000, 44100, 32000, 22050, 16000, 8000]

# Buffer size for audio chunks
CHUNK_SIZE = 512  # Smaller chunks for lower latency

# Audio queue settings
MIC_QUEUE_MAX_SIZE = 10  # Maximum items in microphone queue

# =============================================================================
# VIDEO CONFIGURATION
# =============================================================================
# Video resolution for capturing (higher = sharper but more data)
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# Video frame rate - higher = better context for Gemini
VIDEO_FPS = 2  # Frames per second to send (Gemini Live API recommends 1-2 FPS)

# JPEG quality for video frames (1-100)
VIDEO_JPEG_QUALITY = 95  # High quality for sharp text and clear display

# =============================================================================
# SESSION CONFIGURATION  
# =============================================================================
# Connection timeout in seconds
CONNECTION_TIMEOUT = 30

# Reconnection settings
MAX_RECONNECTION_ATTEMPTS = 3
RECONNECTION_DELAY = 2  # seconds

# =============================================================================
# GROUNDED SAM2 CONFIGURATION
# =============================================================================
# Enable/disable Grounded SAM2 segmentation
ENABLE_GROUNDED_SAM2 = True

# Grounding DINO model (tiny is faster, small is more accurate)
# Options: "IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# SAM model (using SAM1 for stability - SAM2 has API issues in transformers)
# Options: "facebook/sam-vit-base", "facebook/sam-vit-large", "facebook/sam-vit-huge"
SAM2_MODEL = "facebook/sam-vit-base"

# Detection settings
DETECTION_BOX_THRESHOLD = 0.25  # Confidence threshold for object detection
DETECTION_TEXT_THRESHOLD = 0.25  # Text matching threshold
DETECTION_INTERVAL_FRAMES = 5  # Re-detect every N frames (lower = more responsive)

# Use Gemini for keyword extraction (True = smarter, False = faster rule-based)
USE_GEMINI_KEYWORD_EXTRACTION = False

# =============================================================================
# UI CONFIGURATION (for Gradio)
# =============================================================================
UI_TITLE = "🎙️ Gemini Live - Real-time Voice & Video Assistant"
UI_DESCRIPTION = """
Real-time multimodal conversation with Google Gemini.
Speak naturally and share your camera - get instant voice responses!
"""
SERVER_PORT = 7860
