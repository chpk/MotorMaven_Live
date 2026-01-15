#!/usr/bin/env python3
"""
Gemini Live API - Command Line Real-time Voice & Video Assistant
=================================================================
Reliable CLI application for real-time multimodal conversations with Gemini.
Supports audio AND video input.

Usage:
    python app_cli.py

Requirements:
    - Microphone access
    - Camera access (optional)
    - Speaker/headphones for audio output
    - Valid Gemini API key in config.py
"""

import asyncio
import sys
import signal
import argparse
from typing import Optional

try:
    import pyaudio
except ImportError:
    print("Error: pyaudio not installed.")
    print("Run: pip install pyaudio")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Warning: opencv-python not installed. Video will be disabled.")
    print("Run: pip install opencv-python")
    cv2 = None

try:
    from google import genai
except ImportError:
    print("Error: google-genai package not installed.")
    print("Run: pip install google-genai")
    sys.exit(1)

from config import (
    GEMINI_API_KEY,
    MODEL,
    SYSTEM_INSTRUCTION,
    RESPONSE_MODALITIES,
    CHANNELS,
    SEND_SAMPLE_RATE,
    RECEIVE_SAMPLE_RATE,
    CHUNK_SIZE,
    MIC_QUEUE_MAX_SIZE,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    VIDEO_JPEG_QUALITY,
    DEFAULT_VOICE,
)


class GeminiLiveSession:
    """
    Manages a real-time voice AND video conversation session with Gemini Live API.
    """
    
    def __init__(self, enable_video: bool = True, voice: str = DEFAULT_VOICE):
        # Initialize Gemini client
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Audio configuration
        self.format = pyaudio.paInt16
        self.channels = CHANNELS
        self.send_rate = SEND_SAMPLE_RATE
        self.receive_rate = RECEIVE_SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        
        # Video configuration
        self.enable_video = enable_video and cv2 is not None
        self.video_width = VIDEO_WIDTH
        self.video_height = VIDEO_HEIGHT
        self.video_fps = VIDEO_FPS
        self.jpeg_quality = VIDEO_JPEG_QUALITY
        
        # Voice
        self.voice = voice
        
        # PyAudio
        self.pya: Optional[pyaudio.PyAudio] = None
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        # Video capture
        self.video_capture = None
        
        # Async queues
        self.audio_send_queue: asyncio.Queue = asyncio.Queue(maxsize=MIC_QUEUE_MAX_SIZE)
        self.audio_play_queue: asyncio.Queue = asyncio.Queue()
        self.video_send_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
        
        # Session state
        self.is_running = False
        self.live_session = None
        
        # Model configuration
        self.model = MODEL
        self.config = {
            "response_modalities": RESPONSE_MODALITIES,
            "system_instruction": SYSTEM_INSTRUCTION,
        }
        
        # Add voice config
        if self.voice:
            self.config["speech_config"] = {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": self.voice
                    }
                }
            }
    
    def list_audio_devices(self):
        """List available audio devices."""
        pya = pyaudio.PyAudio()
        
        print("\nAudio Input Devices:")
        for i in range(pya.get_device_count()):
            info = pya.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                print(f"   [{i}] {info['name']}")
        
        print("\nAudio Output Devices:")
        for i in range(pya.get_device_count()):
            info = pya.get_device_info_by_index(i)
            if info.get('maxOutputChannels', 0) > 0:
                print(f"   [{i}] {info['name']}")
        
        pya.terminate()
    
    def list_video_devices(self):
        """List available video devices."""
        if cv2 is None:
            print("\nVideo: OpenCV not installed")
            return
        
        print("\nVideo Devices:")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"   [{i}] Camera {i}")
            cap.release()
    
    def initialize_audio(self, mic_index: int = None, speaker_index: int = None) -> bool:
        """Initialize PyAudio."""
        try:
            self.pya = pyaudio.PyAudio()
            
            # Get default devices if not specified
            if mic_index is None:
                mic_info = self.pya.get_default_input_device_info()
                mic_index = int(mic_info['index'])
            else:
                mic_info = self.pya.get_device_info_by_index(mic_index)
            
            if speaker_index is None:
                speaker_info = self.pya.get_default_output_device_info()
                speaker_index = int(speaker_info['index'])
            else:
                speaker_info = self.pya.get_device_info_by_index(speaker_index)
            
            self.mic_index = mic_index
            self.speaker_index = speaker_index
            
            print(f"✓ Microphone: {mic_info.get('name', 'Unknown')}")
            print(f"✓ Speaker: {speaker_info.get('name', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"✗ Audio initialization failed: {e}")
            return False
    
    def initialize_video(self, camera_index: int = 0) -> bool:
        """Initialize video capture."""
        if not self.enable_video:
            return True
        
        try:
            self.video_capture = cv2.VideoCapture(camera_index)
            
            if not self.video_capture.isOpened():
                print(f"✗ Cannot open camera {camera_index}")
                return False
            
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
            
            ret, _ = self.video_capture.read()
            if not ret:
                print("✗ Cannot read from camera")
                return False
            
            print(f"✓ Camera {camera_index} initialized ({self.video_width}x{self.video_height})")
            return True
            
        except Exception as e:
            print(f"✗ Video initialization failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception:
                pass
            self.input_stream = None
        
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None
        
        if self.pya:
            try:
                self.pya.terminate()
            except Exception:
                pass
            self.pya = None
        
        if self.video_capture:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None
    
    async def capture_audio(self):
        """Capture audio from microphone."""
        try:
            self.input_stream = await asyncio.to_thread(
                self.pya.open,
                format=self.format,
                channels=self.channels,
                rate=self.send_rate,
                input=True,
                input_device_index=self.mic_index,
                frames_per_buffer=self.chunk_size,
            )
            
            print("[MIC] Microphone active")
            
            while self.is_running:
                try:
                    data = await asyncio.to_thread(
                        self.input_stream.read,
                        self.chunk_size,
                        exception_on_overflow=False
                    )
                    
                    try:
                        self.audio_send_queue.put_nowait({
                            "data": data,
                            "mime_type": "audio/pcm"
                        })
                    except asyncio.QueueFull:
                        try:
                            self.audio_send_queue.get_nowait()
                            self.audio_send_queue.put_nowait({
                                "data": data,
                                "mime_type": "audio/pcm"
                            })
                        except asyncio.QueueEmpty:
                            pass
                            
                except Exception as e:
                    if self.is_running:
                        await asyncio.sleep(0.01)
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"✗ Audio capture error: {e}")
    
    async def capture_video(self):
        """Capture video frames from camera."""
        if not self.enable_video or self.video_capture is None:
            return
        
        try:
            import time
            frame_interval = 1.0 / self.video_fps
            last_frame_time = 0
            
            print("[CAM] Camera active")
            
            while self.is_running:
                current_time = time.time()
                
                if current_time - last_frame_time < frame_interval:
                    await asyncio.sleep(0.01)
                    continue
                
                try:
                    ret, frame = await asyncio.to_thread(self.video_capture.read)
                    
                    if not ret:
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Encode as JPEG
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    _, jpeg_data = cv2.imencode('.jpg', frame, encode_params)
                    
                    try:
                        self.video_send_queue.put_nowait({
                            "data": jpeg_data.tobytes(),
                            "mime_type": "image/jpeg"
                        })
                    except asyncio.QueueFull:
                        try:
                            self.video_send_queue.get_nowait()
                            self.video_send_queue.put_nowait({
                                "data": jpeg_data.tobytes(),
                                "mime_type": "image/jpeg"
                            })
                        except asyncio.QueueEmpty:
                            pass
                    
                    last_frame_time = current_time
                    
                except Exception:
                    if self.is_running:
                        await asyncio.sleep(0.01)
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"✗ Video capture error: {e}")
    
    async def send_to_gemini(self):
        """Send audio and video to Gemini."""
        try:
            while self.is_running:
                # Send audio
                try:
                    audio_msg = await asyncio.wait_for(
                        self.audio_send_queue.get(),
                        timeout=0.05
                    )
                    if self.live_session:
                        await self.live_session.send_realtime_input(audio=audio_msg)
                except asyncio.TimeoutError:
                    pass
                
                # Send video
                if self.enable_video:
                    try:
                        video_msg = self.video_send_queue.get_nowait()
                        if self.live_session:
                            await self.live_session.send_realtime_input(video=video_msg)
                    except asyncio.QueueEmpty:
                        pass
                        
        except asyncio.CancelledError:
            pass
    
    async def receive_from_gemini(self):
        """Receive responses from Gemini."""
        try:
            while self.is_running:
                try:
                    if not self.live_session:
                        await asyncio.sleep(0.1)
                        continue
                    
                    turn = self.live_session.receive()
                    
                    async for response in turn:
                        if not self.is_running:
                            break
                        
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                # Audio
                                if part.inline_data and isinstance(part.inline_data.data, bytes):
                                    self.audio_play_queue.put_nowait(part.inline_data.data)
                                
                                # Text
                                if hasattr(part, 'text') and part.text:
                                    print(f"\n[AI] Gemini: {part.text}")
                    
                    # Clear queue on turn end
                    while not self.audio_play_queue.empty():
                        try:
                            self.audio_play_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                            
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self.is_running:
                        await asyncio.sleep(0.1)
                        
        except asyncio.CancelledError:
            pass
    
    async def play_audio(self):
        """Play audio responses."""
        try:
            self.output_stream = await asyncio.to_thread(
                self.pya.open,
                format=self.format,
                channels=self.channels,
                rate=self.receive_rate,
                output=True,
                output_device_index=self.speaker_index,
            )
            
            print("[SPK] Speaker active")
            
            while self.is_running:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_play_queue.get(),
                        timeout=0.1
                    )
                    await asyncio.to_thread(self.output_stream.write, audio_data)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    if self.is_running:
                        await asyncio.sleep(0.01)
                        
        except asyncio.CancelledError:
            pass
    
    async def run(self, mic_index: int = None, speaker_index: int = None, camera_index: int = 0):
        """Main session loop."""
        print("\n" + "=" * 65)
        print("  GEMINI LIVE - Real-time Voice & Video Assistant")
        print("=" * 65 + "\n")
        
        # Initialize audio
        if not self.initialize_audio(mic_index, speaker_index):
            print("\n✗ Failed to initialize audio.")
            return
        
        # Initialize video
        if not self.initialize_video(camera_index):
            if self.enable_video:
                print("⚠ Video disabled, continuing with audio only")
                self.enable_video = False
        
        print(f"\nConnecting to Gemini ({self.model})...")
        if self.voice:
            print(f"Voice: {self.voice}")
        
        try:
            async with self.client.aio.live.connect(
                model=self.model,
                config=self.config
            ) as session:
                self.live_session = session
                self.is_running = True
                
                print("✓ Connected successfully!")
                print("\n" + "-" * 65)
                if self.enable_video:
                    print("  Speak and show your camera to Gemini!")
                else:
                    print("  Speak to have a conversation with Gemini!")
                print("  Press Ctrl+C to end the session")
                print("-" * 65 + "\n")
                
                # Create tasks
                tasks = [
                    self.capture_audio(),
                    self.send_to_gemini(),
                    self.receive_from_gemini(),
                    self.play_audio(),
                ]
                
                if self.enable_video:
                    tasks.append(self.capture_video())
                
                async with asyncio.TaskGroup() as tg:
                    for task in tasks:
                        tg.create_task(task)
                    
        except asyncio.CancelledError:
            print("\n\nSession cancelled...")
        except Exception as e:
            print(f"\n✗ Connection error: {e}")
            print("\nTroubleshooting:")
            print("  1. Check your internet connection")
            print("  2. Verify your API key in config.py")
            print("  3. Ensure the model name is correct")
        finally:
            self.is_running = False
            self.live_session = None
            self.cleanup()
            print("\n✓ Session ended. Goodbye!\n")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\n\nInterrupt received, shutting down...")
    sys.exit(0)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gemini Live CLI - Voice & Video Assistant")
    parser.add_argument('--list-devices', action='store_true', help='List available audio/video devices')
    parser.add_argument('--mic', type=int, default=None, help='Microphone device index')
    parser.add_argument('--speaker', type=int, default=None, help='Speaker device index')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--no-video', action='store_true', help='Disable video')
    parser.add_argument('--voice', type=str, default=DEFAULT_VOICE, 
                       choices=['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'],
                       help='AI voice selection')
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    session = GeminiLiveSession(
        enable_video=not args.no_video,
        voice=args.voice
    )
    
    if args.list_devices:
        session.list_audio_devices()
        session.list_video_devices()
        return
    
    await session.run(
        mic_index=args.mic,
        speaker_index=args.speaker,
        camera_index=args.camera
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n[ERR] Fatal error: {e}")
        sys.exit(1)
