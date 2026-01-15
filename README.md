# MotorMaven Live

Real-time multimodal AI assistant powered by Google Gemini Live API with intelligent object segmentation using SAM3 Video Tracker with text prompts.

## Features

- **Voice Conversation**: Natural voice chat with Gemini Live API
- **Video Understanding**: Camera feed sent to Gemini for visual context
- **SAM3 Text Prompts**: Describe objects in natural language - "black phone", "red warning label", "white motor with grills"
- **Descriptive Detection**: Auto-enhances prompts for better detection accuracy
- **Custom Object Interpretation**: Converts technical terms like "PCB" to "green circuit board"
- **Real-time Segmentation**: Immediate mask generation from text descriptions
- **Fallback to SAM2**: Visual prompt-based detection when needed

## Architecture

```
User Speech --> Gemini Live API --> Extract Object Descriptions
                      |
                      v
              Prompt Enhancement
      ("phone" --> "black phone device")
                      |
           +----------+----------+
           |                     |
           v                     v
      SAM3 Video            DINO + SAM2
    (Text Prompts)       (Visual Prompts)
           |                     |
           +----------+----------+
                      |
                      v
          Segmentation Overlay on Video
```

## SAM3 vs SAM2

| Feature | SAM3 (Recommended) | SAM2 |
|---------|-------------------|------|
| Input | Text prompts: "black phone", "red box" | Visual prompts (points, boxes) |
| Detection | Built-in text understanding | Requires Grounding DINO |
| Descriptive | "white motor with cooling grills" | Generic "motor" only |
| Multi-instance | Segments ALL matching objects | Single object per prompt |
| Custom Objects | "ABB logo" interpreted automatically | Needs manual bbox |

### Example Descriptive Prompts

SAM3 works best with descriptive, attribute-based prompts:

```
"black phone case"           (not just "phone")
"red warning label"          (not just "label")
"white motor with grills"    (not just "motor")
"green circuit board"        (not just "PCB")
"ABB logo text"              (brand-specific)
"brown wooden table"         (material + color)
```

The system automatically enhances simple prompts:

```
"phone" --> "black phone device"
"motor" --> "gray electric motor with cooling fins"
"pcb"   --> "green circuit board with electronic components"
```

## Requirements

### Hardware

- Microphone
- Webcam (optional but recommended)
- NVIDIA GPU with CUDA support (required for SAM3)
- 12GB+ VRAM recommended for SAM3

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

### 4. Install SAM3 (Required for Text Prompts)

SAM3 requires installation from Meta's repository:

```bash
# Clone SAM3 repository
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# Install in editable mode (recommended)
pip install -e .

# Return to project directory
cd ..
```

SAM3 is a gated model on HuggingFace. Complete the following steps:

1. Create an account at [HuggingFace](https://huggingface.co/)
2. Accept the SAM3 license at [facebook/sam3](https://huggingface.co/facebook/sam3)
3. Create an access token at [HuggingFace Tokens](https://huggingface.co/settings/tokens)
4. Login to HuggingFace CLI:

```bash
huggingface-cli login
# Enter your token when prompted
```

Or set the environment variable:

```bash
export HF_TOKEN="your-huggingface-token"
```

### 5. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 6. Set Up Gemini API Key

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Option A: Environment Variable (Recommended)**

```bash
export GEMINI_API_KEY="your-api-key-here"

# Add to ~/.bashrc for persistence
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Option B: Via Web UI**

1. Launch: `python app_gradio.py`
2. Open http://localhost:7860
3. Enter API key in Settings, then click Gemini API Key field

### 7. Download SAM2 Models (Optional Fallback)

For SAM2 fallback support:

```bash
python setup_sam2.py
# Or download specific checkpoints
python download_sam2_models.py --all
```

## Usage

### Web Interface (Recommended)

```bash
python app_gradio.py
```

Open http://localhost:7860 in your browser.

### Quick Start

1. Start the app: `python app_gradio.py`
2. Verify SAM3: Check Settings and confirm SAM Model shows "SAM3 (~1B) - TEXT PROMPTS"
3. Click "Start Call"
4. Speak: "Highlight the black phone on the table"
5. See results: SAM3 segments matching objects with colored overlays

### Command Line Interface

```bash
# Basic usage
python app_cli.py

# List devices
python app_cli.py --list-devices

# Specify devices
python app_cli.py --mic 0 --speaker 1 --camera 0

# Audio only
python app_cli.py --no-video
```

## Available Models

### SAM3 (Default - Text Prompts)

| Model | Parameters | Features |
|-------|-----------|----------|
| SAM3 | ~1B | Text prompts, semantic understanding, multi-instance |

### SAM2 (Visual Prompts)

| Model | Parameters | Speed |
|-------|-----------|-------|
| SAM2 Hiera Tiny | 38.9M | Very Fast |
| SAM2 Hiera Small | 46M | Fast |
| SAM2 Hiera Base+ | 80.8M | Medium |
| SAM2 Hiera Large | 224.4M | Slower |

### SAM1 (Original)

| Model | Parameters | Speed |
|-------|-----------|-------|
| SAM ViT-Base | 94M | Medium |
| SAM ViT-Large | 308M | Slow |
| SAM ViT-Huge | 636M | Very Slow |

## Configuration

Edit `config.py` to customize:

```python
# Video settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 2

# SAM settings
ENABLE_GROUNDED_SAM2 = True
DETECTION_INTERVAL_FRAMES = 30
```

## How It Works

### SAM3 Text Prompt Pipeline

1. **Voice Input**: "Highlight the red warning label"
2. **Gemini Extraction**: Extracts object description
3. **Prompt Enhancement**: Adds visual attributes if needed
4. **SAM3 Detection**: Text prompt produces immediate segmentation mask
5. **Overlay**: Colored mask displayed on video feed

### Custom Object Interpretation

For technical or unseen objects, the system interprets requests:

```
"PCB"       --> "green circuit board with electronic components"
"ABB motor" --> "gray industrial motor with ABB branding"
"T1 text"   --> "text label showing T1"
"VFD"       --> "variable frequency drive unit"
"HMI"       --> "human machine interface touchscreen"
```

### Example Interactions

- "What do you see?" - Gemini describes the scene
- "Highlight the black phone" - SAM3 segments phone
- "Find all red objects" - SAM3 segments red items
- "Track the white motor with cooling fins" - Detailed segmentation
- "Where is the ABB logo?" - Brand-specific detection

## Troubleshooting

### SAM3 Not Loading

```bash
# Verify SAM3 installation
python -c "from sam3.model_builder import build_sam3_video_predictor; print('SAM3 OK')"

# If not found, reinstall:
cd sam3 && pip install -e . && cd ..
```

### HuggingFace Authentication Error

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN="your-token"
```

### CUDA Out of Memory

- SAM3 requires approximately 12GB VRAM
- Use SAM2 Tiny or Small for lower VRAM GPUs
- Reduce VIDEO_WIDTH and VIDEO_HEIGHT in config.py

### No Objects Detected

- Use descriptive prompts: "black phone" not just "phone"
- Check the SAM Prompt field shows your object
- Try more specific descriptions

### Audio Issues

```bash
# List devices
python app_cli.py --list-devices
# Try different mic
python app_cli.py --mic 1
```

## Project Structure

```
motormaven_live/
├── app_gradio.py           # Gradio web interface
├── app_cli.py              # Command line interface
├── config.py               # Configuration
├── grounded_sam2_tracker.py # SAM3/SAM2 tracker with text prompts
├── setup_sam2.py           # Model download helper
├── download_sam2_models.py # SAM2 checkpoint downloader
├── requirements.txt        # Dependencies
├── checkpoints/            # SAM2 model checkpoints
└── README.md              
```

## Dependencies

- google-genai: Gemini Live API
- sam3: SAM3 video predictor with text prompts
- gradio: Web interface
- transformers: Grounding DINO, SAM2
- torch: Deep learning
- opencv-python: Video capture
- pyaudio: Audio I/O

## License

MIT License

## Acknowledgments

- [Google Gemini](https://deepmind.google/technologies/gemini/) for the Live API
- [Meta AI SAM3](https://github.com/facebookresearch/sam3) for text-based segmentation
- [Meta AI SAM2](https://github.com/facebookresearch/segment-anything-2) for video segmentation
- [IDEA Research](https://github.com/IDEA-Research/Grounding-DINO) for Grounding DINO
- [Gradio](https://gradio.app/) for the web interface
- [HuggingFace](https://huggingface.co/) for model hosting
