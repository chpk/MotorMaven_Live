# MotorMaven Live

Real-time multimodal AI assistant powered by Google Gemini Live API with intelligent object segmentation using SAM3/SAM2.

## Features

- **Voice Conversation**: Natural voice chat with Gemini Live API
- **Video Understanding**: Camera feed sent to Gemini for visual context
- **SAM3 Text Segmentation**: Direct text-based object segmentation (NEW!)
- **Grounded SAM2**: Automatic object detection using Grounding DINO + SAM
- **Voice Responses**: Gemini responds with natural voice
- **Text Overlay**: Responses displayed on video feed
- **Multiple SAM Models**: Choose SAM3, SAM2, or SAM1 models for different use cases

## Architecture

```
User Speech/Video --> Gemini Live API --> AI Response (Voice + Text)
                           |
                           v
                    Object Detection
                           |
            +--------------+---------------+
            |                              |
            v                              v
   SAM3 (Text Prompts)          DINO + SAM2 (Visual Prompts)
            |                              |
            +--------------+---------------+
                           |
                           v
                  Segmentation Overlay
```

### SAM3 vs SAM2

| Feature | SAM3 | SAM2 |
|---------|------|------|
| Input | Text prompts ("phone", "all red objects") | Visual prompts (points, boxes) |
| Detection | Built-in text understanding | Requires Grounding DINO |
| Capability | Segments ALL instances of a concept | Segments specific object from prompt |
| Use Case | Natural language queries | Precise single-object tracking |

## Requirements

### Hardware
- Microphone
- Webcam (optional but recommended)
- NVIDIA GPU with CUDA support (recommended for SAM2)
- 8GB+ RAM (16GB+ recommended for larger SAM models)

### Software
- Python 3.10 or 3.11
- CUDA 11.8+ (for GPU acceleration)
- Linux (tested on Ubuntu 22.04)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chpk/MotorMaven_Live.git
cd MotorMaven_Live
```

### 2. Create Conda Environment

```bash
conda create -n motormaven_live python=3.11
conda activate motormaven_live
```

### 3. Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Gemini API Key

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

You have two options to provide the API key:

**Option A: Via Web UI (Recommended for quick testing)**
1. Launch the application: `python app_gradio.py`
2. Open http://localhost:7860
3. Expand the "Settings" accordion
4. Enter your API key in the "Gemini API Key" field
5. Click "Start Call"

**Option B: Via Environment Variable (Recommended for regular use)**
```bash
# Linux/macOS
export GEMINI_API_KEY="your-api-key-here"

# Add to ~/.bashrc for persistence
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

Note: If you provide an API key in the UI, it will override the environment variable for that session.

### 6. Install SAM3 (Optional)

SAM3 uses Meta's native package for text-based segmentation:

```bash
pip install 'git+https://github.com/facebookresearch/sam3.git'
```

SAM3 features:
- Direct text prompts ("phone", "all red objects")
- Semantic understanding of concepts
- Automatic multi-instance detection

Note: SAM2 works without SAM3 installed.

### 7. Download SAM Models (Optional)

The models will auto-download from HuggingFace on first run. For faster startup, pre-download:

```bash
# Pre-download Grounding DINO and SAM models
python setup_sam2.py

# Or download specific SAM2.1 checkpoints
python download_sam2_models.py --all
python download_sam2_models.py --model large  # Download specific model
python download_sam2_models.py --list         # List available models
```

## Available SAM Models

### SAM3 (Recommended)
| Model | Parameters | Description | Speed |
|-------|-----------|-------------|-------|
| SAM3 | ~1B | Text prompts, semantic understanding | Medium |

SAM3 is the recommended model - it understands natural language directly:
- "segment all phones"
- "highlight red objects"
- "find the laptop"

### SAM2 (Visual Prompts)
| Model | Parameters | Description | Speed |
|-------|-----------|-------------|-------|
| SAM2 Hiera Tiny | 38.9M | Fastest, good for real-time | Very Fast |
| SAM2 Hiera Small | 46M | Balanced speed/accuracy | Fast |
| SAM2 Hiera Base+ | 80.8M | Better accuracy | Medium |
| SAM2 Hiera Large | 224.4M | Best accuracy | Slower |

### SAM1 (Original)
| Model | Parameters | Description | Speed |
|-------|-----------|-------------|-------|
| SAM ViT-Base | 94M | Original SAM, stable | Medium |
| SAM ViT-Large | 308M | High quality | Slow |
| SAM ViT-Huge | 636M | Highest quality | Very Slow |

## Usage

### Web Interface (Recommended)

```bash
python app_gradio.py
```

Open http://localhost:7860 in your browser.

### Command Line Interface

```bash
# Basic usage
python app_cli.py

# List available devices
python app_cli.py --list-devices

# Specify devices
python app_cli.py --mic 0 --speaker 1 --camera 0

# Audio only (no video)
python app_cli.py --no-video

# Change AI voice
python app_cli.py --voice Charon
```

### Voice Options

- Puck (default)
- Charon
- Kore
- Fenrir
- Aoede

## Configuration

Edit `config.py` to customize:

```python
# Video settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 2
VIDEO_JPEG_QUALITY = 90

# Audio settings
AUDIO_INPUT_SAMPLE_RATE = 48000
AUDIO_OUTPUT_SAMPLE_RATE = 48000

# SAM settings
ENABLE_GROUNDED_SAM2 = True
DETECTION_INTERVAL_FRAMES = 30
```

## How It Works

1. **Voice Capture**: Your speech is captured and sent to Gemini Live API
2. **Video Stream**: Camera frames are sent at 2 FPS when you're speaking
3. **AI Processing**: Gemini processes both audio and video for context
4. **Object Detection**: When you mention objects, they are extracted from speech
5. **Segmentation**:
   - **SAM3 Mode**: Text prompt sent directly to SAM3 for semantic segmentation
   - **SAM2 Mode**: Grounding DINO detects objects, then SAM2 segments them
6. **Response**: AI responds with voice and text overlay on video

### Example Interactions

- "What do you see?"
- "Highlight the phone on the table"
- "Track my coffee mug"
- "What objects are in front of me?"
- "Segment the keyboard and mouse"

## Troubleshooting

### No Audio Input
```bash
# Check available devices
python app_cli.py --list-devices
# Try different microphone index
python app_cli.py --mic 1
```

### CUDA Out of Memory
- Use a smaller SAM model (Tiny or Small)
- Reduce VIDEO_WIDTH and VIDEO_HEIGHT in config.py

### Models Not Loading
```bash
# Re-run setup to download models
python setup_sam2.py
# Or clear HuggingFace cache and retry
rm -rf ~/.cache/huggingface/hub
python setup_sam2.py
```

### API Connection Issues
- Verify your GEMINI_API_KEY is set correctly
- Check internet connection
- Ensure you have API access to Gemini Live

## Project Structure

```
motormaven_live/
|-- app_gradio.py           # Main Gradio web interface
|-- app_cli.py              # Command line interface
|-- config.py               # Configuration settings
|-- grounded_sam2_tracker.py # SAM2 + Grounding DINO integration
|-- setup_sam2.py           # Model download helper
|-- download_sam2_models.py # SAM2.1 checkpoint downloader
|-- requirements.txt        # Python dependencies
|-- checkpoints/            # SAM2.1 model checkpoints
|-- README.md               # This file
```

## Dependencies

- google-genai: Gemini Live API client
- gradio: Web interface
- pyaudio: Audio capture/playback
- opencv-python: Video capture
- transformers: Grounding DINO and SAM models
- torch: Deep learning framework
- numpy: Numerical operations
- Pillow: Image processing

## License

MIT License

## Acknowledgments

- [Google Gemini](https://deepmind.google/technologies/gemini/) for the Live API
- [Meta AI](https://segment-anything.com/) for SAM, SAM2, and SAM3
- [IDEA Research](https://github.com/IDEA-Research/Grounding-DINO) for Grounding DINO
- [Gradio](https://gradio.app/) for the web interface framework
- [HuggingFace](https://huggingface.co/) for model hosting and transformers library