#!/usr/bin/env python3
"""
Gemini Live + Grounded SAM - Real-time Multimodal AI
====================================================
RESTORED WORKING VERSION with proper async handling.
"""

import asyncio
import threading
import queue
import time
import sys
import traceback
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import pyaudio
except ImportError:
    print("Error: pyaudio not installed")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed")
    sys.exit(1)

try:
    import gradio as gr
except ImportError:
    print("Error: gradio not installed")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("Error: google-genai not installed")
    sys.exit(1)

from config import (
    GEMINI_API_KEY, MODEL, SYSTEM_INSTRUCTION, RESPONSE_MODALITIES,
    CHANNELS, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, CHUNK_SIZE,
    VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, VIDEO_JPEG_QUALITY,
    VOICE_OPTIONS, DEFAULT_VOICE, SERVER_PORT, ENABLE_GROUNDED_SAM2,
)

# Import SAM tracker
SAM_AVAILABLE = False
if ENABLE_GROUNDED_SAM2:
    try:
        from grounded_sam2_tracker import HybridSAMTracker as GroundedSAMTracker
        import torch
        SAM_AVAILABLE = True
        print(f"[OK] SAM Tracker available (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"[WARN] SAM Tracker not available: {e}")


class SessionState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class DeviceInfo:
    index: int
    name: str
    is_input: bool = True


# =============================================================================
# UTILITIES
# =============================================================================

def get_supported_rate(pya, idx, is_input, target):
    rates = [target, 48000, 44100, 32000, 22050, 16000]
    for r in rates:
        try:
            if is_input:
                ok = pya.is_format_supported(r, input_device=idx, input_channels=CHANNELS, input_format=pyaudio.paInt16)
            else:
                ok = pya.is_format_supported(r, output_device=idx, output_channels=CHANNELS, output_format=pyaudio.paInt16)
            if ok:
                return r
        except:
            continue
    return target


def resample(data, from_rate, to_rate):
    if from_rate == to_rate:
        return data
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    ratio = to_rate / from_rate
    new_len = int(len(arr) * ratio)
    if new_len == 0:
        return data
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, new_len)
    return np.interp(x_new, x_old, arr).astype(np.int16).tobytes()


def get_audio_devices():
    inputs, outputs = [], []
    try:
        pya = pyaudio.PyAudio()
        for i in range(pya.get_device_count()):
            try:
                info = pya.get_device_info_by_index(i)
                name = info.get('name', f'Device {i}')
                if info.get('maxInputChannels', 0) > 0:
                    inputs.append(DeviceInfo(i, name, True))
                if info.get('maxOutputChannels', 0) > 0:
                    outputs.append(DeviceInfo(i, name, False))
            except:
                continue
        pya.terminate()
    except:
        pass
    return inputs, outputs


def get_video_devices():
    devices = []
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                devices.append(DeviceInfo(i, f"Camera {i}", True))
            cap.release()
        except:
            continue
    return devices


def draw_overlay(frame, text):
    """Draw slim, elegant text overlay at bottom - Google Meet style."""
    if not text or frame is None:
        return frame
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Word wrap with more chars per line
    words = text.split()
    lines, line = [], ""
    max_chars = w // 10  # More chars per line for slimmer look
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line += (" " if line else "") + word
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    lines = lines[-2:]  # Only last 2 lines for slim look
    
    if not lines:
        return frame
    
    # Slim semi-transparent bar at bottom
    line_height = 24
    padding = 8
    total_h = len(lines) * line_height + padding * 2
    y_start = h - total_h
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Draw text with thin font
    for i, ln in enumerate(lines):
        y = y_start + padding + (i + 1) * line_height - 4
        # Shadow for readability
        cv2.putText(frame, ln, (12, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame


def draw_status(frame, connected, ai_speaking, user_speaking, sam_prompt):
    """Draw minimal status indicators - Google Meet style."""
    if frame is None:
        return frame
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Top bar with status (slim)
    if connected or ai_speaking or user_speaking:
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Connection dot (top right) - small
    color = (0, 255, 0) if connected else (100, 100, 100)
    cv2.circle(frame, (w - 16, 16), 6, color, -1)
    
    # User speaking indicator (top left) - pulsing mic icon area
    if user_speaking:
        cv2.circle(frame, (16, 16), 6, (100, 200, 255), -1)
        cv2.putText(frame, "Listening", (28, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)
    
    # AI speaking indicator 
    if ai_speaking:
        cv2.circle(frame, (w - 80, 16), 6, (0, 255, 200), -1)
        cv2.putText(frame, "AI", (w - 70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1, cv2.LINE_AA)
    
    # SAM prompt - small text at top center if active
    if sam_prompt:
        text = sam_prompt[:25]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    
    return frame


# =============================================================================
# IMPROVED VAD - More patient, adaptive threshold
# =============================================================================

class ImprovedVAD:
    """
    Voice Activity Detection with:
    - Adaptive noise floor calibration
    - Minimum speaking duration (won't end too fast)
    - Longer silence required to end speech
    - Initial warmup period
    """
    
    def __init__(self):
        # Thresholds
        self.base_threshold = 200  # Lower base threshold
        self.threshold_multiplier = 1.5  # How much above noise floor
        self.noise_floor = 150  # Starting noise floor estimate
        
        # Timing (in seconds)
        self.min_speech_duration = 2.0  # Minimum time before we can end speech
        self.silence_to_end = 1.5  # Silence duration to end speech
        self.warmup_duration = 1.0  # Initial warmup (just calibrate, don't detect)
        
        # State
        self.is_speaking = False
        self.speech_start = 0
        self.silence_start = 0
        self.session_start = 0
        self.is_warmed_up = False
        
        # Noise calibration
        self.noise_samples = []
        self.max_noise_samples = 30
        
        # Pre-speech buffer
        self.buffer = []
        self.max_buffer = 10  # Keep more pre-speech audio
    
    def start_session(self):
        """Call when session starts to begin warmup."""
        self.session_start = time.time()
        self.is_warmed_up = False
        self.noise_samples.clear()
        print(f"[VAD] Session started, warming up for {self.warmup_duration}s...")
    
    def _update_noise_floor(self, energy):
        """Update noise floor estimate with recent quiet samples."""
        self.noise_samples.append(energy)
        if len(self.noise_samples) > self.max_noise_samples:
            self.noise_samples.pop(0)
        
        if len(self.noise_samples) >= 5:
            # Use median of quietest 50% of samples
            sorted_samples = sorted(self.noise_samples)
            quiet_half = sorted_samples[:len(sorted_samples)//2 + 1]
            self.noise_floor = max(100, np.median(quiet_half))
    
    def _get_threshold(self):
        """Get current detection threshold based on noise floor."""
        return max(self.base_threshold, self.noise_floor * self.threshold_multiplier)
    
    def process(self, data: bytes) -> Tuple[bool, bytes, bool]:
        """
        Process audio chunk.
        Returns: (should_send, audio_data, speech_ended)
        """
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        energy = np.sqrt(np.mean(arr ** 2)) if len(arr) > 0 else 0
        now = time.time()
        
        # Check warmup
        if not self.is_warmed_up:
            if now - self.session_start < self.warmup_duration:
                # During warmup, just calibrate noise floor
                self._update_noise_floor(energy)
                self.buffer.append(data)
                if len(self.buffer) > self.max_buffer:
                    self.buffer.pop(0)
                return False, b'', False
            else:
                self.is_warmed_up = True
                threshold = self._get_threshold()
                print(f"[VAD] Warmup complete. Noise floor: {self.noise_floor:.0f}, Threshold: {threshold:.0f}")
        
        # Keep buffer for pre-speech audio
        self.buffer.append(data)
        if len(self.buffer) > self.max_buffer:
            self.buffer.pop(0)
        
        threshold = self._get_threshold()
        is_voice = energy > threshold
        
        if is_voice:
            # Voice detected
            if not self.is_speaking:
                # Start of speech
                self.is_speaking = True
                self.speech_start = now
                self.silence_start = 0
                # Include buffered audio
                result = b''.join(self.buffer)
                self.buffer.clear()
                print(f"[VAD] Speech started (energy: {energy:.0f}, threshold: {threshold:.0f})")
                return True, result, False
            else:
                # Continuing speech - reset silence timer
                self.silence_start = 0
                return True, data, False
        else:
            # No voice detected
            if not self.is_speaking:
                # Not speaking - update noise floor
                self._update_noise_floor(energy)
                return False, b'', False
            else:
                # Was speaking - check if we should end
                if self.silence_start == 0:
                    self.silence_start = now
                
                speech_duration = now - self.speech_start
                silence_duration = now - self.silence_start
                
                # Don't end speech too early
                if speech_duration < self.min_speech_duration:
                    # Keep listening, haven't spoken long enough
                    return True, data, False
                
                # Check if enough silence to end
                if silence_duration >= self.silence_to_end:
                    self.is_speaking = False
                    print(f"[VAD] Speech ended after {speech_duration:.1f}s (silence: {silence_duration:.1f}s)")
                    return True, data, True
                
                # Still in grace period, keep sending
                return True, data, False
    
    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_start = 0
        self.silence_start = 0
        self.buffer.clear()
        # Don't reset noise calibration
    
    def full_reset(self):
        """Full reset including calibration."""
        self.reset()
        self.noise_samples.clear()
        self.noise_floor = 150
        self.is_warmed_up = False


# =============================================================================
# MAIN SESSION - PROVEN ASYNC PATTERN
# =============================================================================

class GeminiLiveSession:
    def __init__(self):
        # API key - can be set via UI or environment variable
        self.api_key = GEMINI_API_KEY
        self.client = None  # Will be initialized when starting
        
        # Audio
        self.pya = None
        self.input_stream = None
        self.output_stream = None
        self.actual_input_rate = SEND_SAMPLE_RATE
        self.actual_output_rate = RECEIVE_SAMPLE_RATE
        
        # Video
        self.video_cap = None
        
        # Devices
        self.mic_idx = 0
        self.speaker_idx = 0
        self.camera_idx = 0
        self.voice = DEFAULT_VOICE
        
        # VAD
        self.vad = ImprovedVAD()
        
        # SAM
        self.sam_tracker = None
        self.sam_enabled = False
        self.sam_prompt = ""
        self.sam_model = "sam2-hiera-small"  # Default model
        
        # Queues
        self.audio_in_q = asyncio.Queue()
        self.video_in_q = asyncio.Queue()
        self.audio_out_q = queue.Queue()
        self.log_q = queue.Queue()
        
        # State
        self.state = SessionState.IDLE
        self.is_running = False
        self.session_thread = None
        
        # Locks
        self.lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        # Status
        self.user_speaking = False
        self.gemini_speaking = False
        self.cooldown_until = 0  # Don't listen until this time (prevents feedback)
        
        # Frames
        self.display_frame = None
        self.latest_jpeg = None
        
        # Stats
        self.frames_sent = 0
        
        # Response
        self.response_text = ""
        self.response_until = 0
        
        # Transcript
        self.transcript = []
    
    def log(self, msg, level="info"):
        icons = {"info": "[INFO]", "success": "[OK]", "warning": "[WARN]", "error": "[ERR]", 
                 "audio": "[MIC]", "video": "[CAM]", "speech": "[SPEECH]", "sam": "[SAM]"}
        try:
            self.log_q.put_nowait(f"[{time.strftime('%H:%M:%S')}] {icons.get(level, '[LOG]')} {msg}")
        except:
            pass
    
    def get_logs(self):
        logs = []
        while not self.log_q.empty():
            try:
                logs.append(self.log_q.get_nowait())
            except:
                break
        return "\n".join(logs)
    
    def get_status(self):
        base = {
            SessionState.IDLE: "⚫ Ready",
            SessionState.CONNECTING: "🟡 Connecting...",
            SessionState.CONNECTED: f"🟢 Connected | Frames: {self.frames_sent}",
            SessionState.ERROR: "🔴 Error",
        }.get(self.state, "Unknown")
        
        with self.lock:
            if self.gemini_speaking:
                return "AI Speaking..."
            if self.user_speaking:
                return "Listening..."
        
        if self.sam_prompt:
            return f"{base} | SAM: {self.sam_prompt[:20]}..."
        return base
    
    def get_transcript(self):
        return "\n".join(self.transcript[-20:])
    
    def set_devices(self, mic, speaker, camera, voice, sam_enabled, sam_model="sam2-hiera-small", api_key=""):
        self.mic_idx = mic
        self.speaker_idx = speaker
        self.camera_idx = camera
        self.voice = voice
        self.sam_enabled = sam_enabled and SAM_AVAILABLE
        self.sam_model = sam_model
        # Use UI-provided API key if available, otherwise fall back to environment variable
        if api_key and api_key.strip():
            self.api_key = api_key.strip()
            self.log("Using API key from UI", "info")
        elif GEMINI_API_KEY:
            self.api_key = GEMINI_API_KEY
            self.log("Using API key from environment variable", "info")
        else:
            self.api_key = ""
    
    def _init_audio(self):
        try:
            self.pya = pyaudio.PyAudio()
            
            mic_info = self.pya.get_device_info_by_index(self.mic_idx)
            self.actual_input_rate = get_supported_rate(self.pya, self.mic_idx, True, SEND_SAMPLE_RATE)
            self.log(f"Mic: {mic_info.get('name', 'Unknown')} @ {self.actual_input_rate}Hz", "success")
            
            spk_info = self.pya.get_device_info_by_index(self.speaker_idx)
            self.actual_output_rate = get_supported_rate(self.pya, self.speaker_idx, False, RECEIVE_SAMPLE_RATE)
            self.log(f"Speaker: {spk_info.get('name', 'Unknown')} @ {self.actual_output_rate}Hz", "success")
            
            return True
        except Exception as e:
            self.log(f"Audio init failed: {e}", "error")
            return False
    
    def _init_video(self):
        try:
            self.video_cap = cv2.VideoCapture(self.camera_idx)
            if not self.video_cap.isOpened():
                self.log(f"Cannot open camera {self.camera_idx}", "error")
                return False
            
            self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            
            ret, _ = self.video_cap.read()
            if not ret:
                self.log("Cannot read camera", "error")
                return False
            
            self.log(f"Camera {self.camera_idx} ready", "success")
            return True
        except Exception as e:
            self.log(f"Video init failed: {e}", "error")
            return False
    
    def _init_sam(self):
        if not self.sam_enabled:
            return
        
        try:
            self.log(f"Loading SAM: {self.sam_model}...", "sam")
            self.sam_tracker = GroundedSAMTracker(
                api_key=GEMINI_API_KEY,
                sam_model=self.sam_model
            )
            
            if self.sam_tracker.load_models():
                self.sam_tracker.start()
                self.log(f"SAM loaded: {self.sam_model}", "success")
            else:
                self.log("SAM load failed", "warning")
                self.sam_enabled = False
        except Exception as e:
            self.log(f"SAM error: {e}", "warning")
            import traceback
            traceback.print_exc()
            self.sam_enabled = False
    
    def _cleanup(self):
        for stream in [self.input_stream, self.output_stream]:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
        self.input_stream = None
        self.output_stream = None
        
        if self.pya:
            try:
                self.pya.terminate()
            except:
                pass
            self.pya = None
        
        if self.video_cap:
            try:
                self.video_cap.release()
            except:
                pass
            self.video_cap = None
        
        if self.sam_tracker:
            try:
                self.sam_tracker.stop()
            except:
                pass
            self.sam_tracker = None
        
        self.vad.reset()
    
    # =========================================================================
    # AUDIO CAPTURE (runs in thread, puts to async queue)
    # =========================================================================
    
    def _audio_capture_loop(self, loop):
        """Captures audio and puts into async queue."""
        try:
            chunk_size = int(CHUNK_SIZE * self.actual_input_rate / SEND_SAMPLE_RATE)
            
            self.input_stream = self.pya.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=self.actual_input_rate,
                input=True,
                input_device_index=self.mic_idx,
                frames_per_buffer=chunk_size,
            )
            
            self.log("Microphone started", "audio")
            
            # Start VAD warmup (calibrates noise floor)
            self.vad.start_session()
            self.log("VAD calibrating... speak in 1 second", "audio")
            
            while self.is_running:
                try:
                    data = self.input_stream.read(chunk_size, exception_on_overflow=False)
                    
                    # CHECK: Don't process audio while Gemini is speaking or in cooldown
                    # This PREVENTS feedback loop where app hears its own output
                    with self.lock:
                        ai_active = self.gemini_speaking
                        in_cooldown = time.time() < self.cooldown_until
                    
                    if ai_active or in_cooldown:
                        # Discard audio - don't even process through VAD
                        # This prevents feedback from speakers
                        continue
                    
                    if self.actual_input_rate != SEND_SAMPLE_RATE:
                        data = resample(data, self.actual_input_rate, SEND_SAMPLE_RATE)
                    
                    should_send, audio, speech_ended = self.vad.process(data)
                    
                    with self.lock:
                        was_speaking = self.user_speaking
                        self.user_speaking = self.vad.is_speaking
                        
                        if self.user_speaking and not was_speaking:
                            self.log("Speaking...", "speech")
                        elif not self.user_speaking and was_speaking:
                            self.log("Finished speaking", "speech")
                            # Add marker to transcript for user speech
                            self.transcript.append("[User spoke]")
                    
                    if should_send and audio:
                        asyncio.run_coroutine_threadsafe(
                            self.audio_in_q.put({"data": audio, "mime_type": "audio/pcm"}),
                            loop
                        )
                    
                    # When user finishes speaking, use a generic request to analyze the frame
                    # The actual object extraction will happen when Gemini responds
                    if speech_ended and self.sam_enabled and self.sam_tracker:
                        with self.frame_lock:
                            frame = self.display_frame.copy() if self.display_frame is not None else None
                        if frame is not None:
                            # Use a generic prompt - Gemini will see the frame and describe what's visible
                            # that the user might be asking about
                            self.sam_tracker.extract_keywords_async(frame, "What object is the user pointing at or asking about? Identify the main objects visible.")
                    
                except Exception as e:
                    time.sleep(0.01)
                    
        except Exception as e:
            self.log(f"Audio capture error: {e}", "error")
            traceback.print_exc()
    
    # =========================================================================
    # VIDEO CAPTURE (runs in thread)
    # =========================================================================
    
    def _video_capture_loop(self, loop):
        """Captures video frames."""
        try:
            self.log("Video capture started", "video")
            last_send = 0
            send_interval = 1.0 / VIDEO_FPS
            
            while self.is_running:
                try:
                    ret, frame = self.video_cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # SAM processing - always process if enabled and has prompt
                    if self.sam_enabled and self.sam_tracker:
                        prompt = self.sam_tracker.get_prompt()
                        if prompt:
                            self.sam_prompt = prompt
                            frame_rgb, _ = self.sam_tracker.process_frame(frame_rgb)
                        else:
                            self.sam_prompt = ""
                    
                    # Get state for overlay
                    with self.lock:
                        resp_text = self.response_text if time.time() < self.response_until else ""
                        ai_spk = self.gemini_speaking
                        user_spk = self.user_speaking
                    
                    # Draw overlays
                    connected = self.state == SessionState.CONNECTED
                    display = draw_status(frame_rgb, connected, ai_spk, user_spk, self.sam_prompt)
                    if resp_text:
                        display = draw_overlay(display, resp_text)
                    
                    with self.frame_lock:
                        self.display_frame = display
                    
                    # Store frame directly for display (no compression for sharp quality)
                    with self.frame_lock:
                        self.display_frame = display.copy()
                    
                    # Send to Gemini when user is speaking (not during AI turn or cooldown)
                    now = time.time()
                    with self.lock:
                        speaking = self.user_speaking
                        ai_active = self.gemini_speaking
                        in_cooldown = now < self.cooldown_until
                    
                    # Only send video when user is speaking AND AI is not active
                    if speaking and not ai_active and not in_cooldown and now - last_send >= send_interval:
                        _, send_jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                                                     [cv2.IMWRITE_JPEG_QUALITY, VIDEO_JPEG_QUALITY])
                        asyncio.run_coroutine_threadsafe(
                            self.video_in_q.put({"data": send_jpeg.tobytes(), "mime_type": "image/jpeg"}),
                            loop
                        )
                        last_send = now
                    
                except Exception as e:
                    time.sleep(0.01)
                
                time.sleep(0.03)
                    
        except Exception as e:
            self.log(f"Video error: {e}", "error")
            traceback.print_exc()
    
    # =========================================================================
    # AUDIO PLAYBACK (runs in thread)
    # =========================================================================
    
    def _audio_play_loop(self):
        """Plays audio from Gemini."""
        try:
            buffer_size = int(CHUNK_SIZE * self.actual_output_rate / RECEIVE_SAMPLE_RATE) * 4
            
            self.output_stream = self.pya.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=self.actual_output_rate,
                output=True,
                output_device_index=self.speaker_idx,
                frames_per_buffer=buffer_size,
            )
            
            self.log("Audio playback started", "audio")
            
            while self.is_running:
                try:
                    audio = self.audio_out_q.get(timeout=0.1)
                    
                    with self.lock:
                        self.gemini_speaking = True
                    
                    if self.actual_output_rate != RECEIVE_SAMPLE_RATE:
                        audio = resample(audio, RECEIVE_SAMPLE_RATE, self.actual_output_rate)
                    
                    self.output_stream.write(audio)
                    
                except queue.Empty:
                    with self.lock:
                        self.gemini_speaking = False
                except Exception as e:
                    time.sleep(0.01)
                    
        except Exception as e:
            self.log(f"Playback error: {e}", "error")
            traceback.print_exc()
    
    # =========================================================================
    # ASYNC SEND TO GEMINI
    # =========================================================================
    
    async def _send_audio(self, session):
        """Send audio to Gemini."""
        while self.is_running:
            try:
                msg = await asyncio.wait_for(self.audio_in_q.get(), timeout=0.1)
                await session.send_realtime_input(audio=msg)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self.log(f"Send audio error: {e}", "warning")
                await asyncio.sleep(0.1)
    
    async def _send_video(self, session):
        """Send video to Gemini."""
        while self.is_running:
            try:
                msg = await asyncio.wait_for(self.video_in_q.get(), timeout=0.1)
                await session.send_realtime_input(video=msg)
                self.frames_sent += 1
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self.log(f"Send video error: {e}", "warning")
                await asyncio.sleep(0.1)
    
    # =========================================================================
    # ASYNC RECEIVE FROM GEMINI
    # =========================================================================
    
    async def _receive(self, session):
        """Receive responses from Gemini."""
        while self.is_running:
            try:
                turn = session.receive()
                turn_text = ""
                
                self.log("Gemini responding...", "info")
                
                async for response in turn:
                    if not self.is_running:
                        break
                    
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            # Audio
                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                self.audio_out_q.put(part.inline_data.data)
                            
                            # Text
                            if hasattr(part, 'text') and part.text:
                                text = part.text.strip()
                                if text:
                                    turn_text += text + " "
                                    with self.lock:
                                        self.response_text = turn_text.strip()
                                        self.response_until = time.time() + 10
                                    self.transcript.append(f"[AI] {text}")
                
                # Wait for audio to finish playing
                while not self.audio_out_q.empty():
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.5)  # Extra wait for audio to fully play
                
                with self.lock:
                    self.gemini_speaking = False
                    # SET COOLDOWN: Don't listen for 2 seconds after AI finishes
                    # This prevents feedback loop from speaker echo/reverb
                    self.cooldown_until = time.time() + 2.0
                
                self.log("Response complete (cooldown 2s)", "success")
                
                # Extract SAM keywords from AI response - this tells us what user asked about
                # The cooldown prevents feedback loops
                if turn_text and self.sam_enabled and self.sam_tracker:
                    with self.frame_lock:
                        frame = self.display_frame.copy() if self.display_frame is not None else None
                    if frame is not None:
                        # Use Gemini's response to understand what objects were discussed
                        self.sam_tracker.extract_keywords_async(frame, f"Based on this conversation: '{turn_text[:200]}' - identify the objects being discussed.")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log(f"Receive error: {e}", "warning")
                await asyncio.sleep(0.1)
    
    # =========================================================================
    # MAIN ASYNC SESSION
    # =========================================================================
    
    async def _run_session(self):
        self.state = SessionState.CONNECTING
        self.log("Initializing...", "info")
        
        if not self._init_audio():
            self.state = SessionState.ERROR
            return
        
        if not self._init_video():
            self.state = SessionState.ERROR
            self._cleanup()
            return
        
        self._init_sam()
        
        # Build config
        system = SYSTEM_INSTRUCTION
        if self.sam_enabled:
            system += "\nYou can see video with object segmentation."
        
        config = {
            "response_modalities": RESPONSE_MODALITIES,
            "system_instruction": system,
        }
        
        if self.voice:
            config["speech_config"] = {"voice_config": {"prebuilt_voice_config": {"voice_name": self.voice}}}
        
        self.log(f"Connecting to {MODEL}...", "info")
        
        try:
            async with self.client.aio.live.connect(model=MODEL, config=config) as session:
                self.state = SessionState.CONNECTED
                self.log(f"Connected! Voice: {self.voice}, SAM: {'ON' if self.sam_enabled else 'OFF'}", "success")
                
                loop = asyncio.get_running_loop()
                
                # Start capture threads
                audio_thread = threading.Thread(target=self._audio_capture_loop, args=(loop,), daemon=True)
                video_thread = threading.Thread(target=self._video_capture_loop, args=(loop,), daemon=True)
                play_thread = threading.Thread(target=self._audio_play_loop, daemon=True)
                
                audio_thread.start()
                video_thread.start()
                play_thread.start()
                
                self.log("All systems running!", "success")
                
                # Run async tasks
                await asyncio.gather(
                    self._send_audio(session),
                    self._send_video(session),
                    self._receive(session),
                )
                
        except asyncio.CancelledError:
            self.log("Session cancelled", "info")
        except Exception as e:
            self.log(f"Session error: {e}", "error")
            traceback.print_exc()
            self.state = SessionState.ERROR
        finally:
            self.is_running = False
            self._cleanup()
            self.state = SessionState.IDLE
            self.log("Session ended", "info")
    
    def _session_thread_func(self):
        asyncio.run(self._run_session())
    
    def start(self):
        if self.is_running:
            return "Already running"
        
        # Check for API key
        if not self.api_key:
            self.log("ERROR: No API key provided!", "error")
            return "Error: No API key. Enter key in Settings or set GEMINI_API_KEY environment variable."
        
        # Initialize Gemini client with the API key
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.log("Gemini client initialized", "success")
        except Exception as e:
            self.log(f"Failed to initialize client: {e}", "error")
            return f"Error: Failed to initialize Gemini client: {e}"
        
        self.is_running = True
        self.transcript.clear()
        self.response_text = ""
        self.gemini_speaking = False
        self.user_speaking = False
        self.cooldown_until = 0
        self.frames_sent = 0
        self.sam_prompt = ""
        self.vad.full_reset()
        
        # Clear queues
        while not self.audio_out_q.empty():
            try:
                self.audio_out_q.get_nowait()
            except:
                break
        while not self.log_q.empty():
            try:
                self.log_q.get_nowait()
            except:
                break
        
        self.session_thread = threading.Thread(target=self._session_thread_func, daemon=True)
        self.session_thread.start()
        return "Starting..."
    
    def stop(self):
        self.is_running = False
        if self.session_thread and self.session_thread.is_alive():
            self.session_thread.join(timeout=3.0)
        self._cleanup()
        self.state = SessionState.IDLE
        return "Stopped"
    
    def get_frame(self):
        with self.frame_lock:
            if self.display_frame is not None:
                # Ensure frame is in correct format for Gradio (RGB uint8)
                frame = self.display_frame.copy()
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                return frame
            return None
    
    def set_sam_prompt(self, prompt):
        if self.sam_enabled and self.sam_tracker:
            self.sam_tracker.set_prompt(prompt)
            self.sam_prompt = prompt


# =============================================================================
# GRADIO UI
# =============================================================================

session = GeminiLiveSession()
mic_devices, speaker_devices = get_audio_devices()
camera_devices = get_video_devices()

mic_choices = [(d.name, d.index) for d in mic_devices] if mic_devices else [("No mic", 0)]
speaker_choices = [(d.name, d.index) for d in speaker_devices] if speaker_devices else [("No speaker", 0)]
camera_choices = [(d.name, d.index) for d in camera_devices] if camera_devices else [("No camera", 0)]


def create_ui():
    """Google Meet style UI - Video focused with minimal controls."""
    
    # Custom CSS for video-call style UI - MAXIMIZED VIDEO
    css = """
    .gradio-container {
        max-width: 100% !important;
        padding: 8px !important;
        background: #0a0a0a !important;
    }
    .main-video {
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        background: #000 !important;
        display: flex !important;
        justify-content: center !important;
    }
    .main-video img {
        border-radius: 8px;
        width: auto !important;
        height: auto !important;
        max-width: none !important;
        max-height: none !important;
        object-fit: none !important;
        image-rendering: -webkit-optimize-contrast !important;
        image-rendering: crisp-edges !important;
    }
    .control-bar {
        background: linear-gradient(180deg, transparent, rgba(0,0,0,0.8));
        border-radius: 0 0 12px 12px;
        padding: 8px 16px;
        margin-top: -8px;
    }
    .status-pill {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
    }
    .btn-call {
        border-radius: 50px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
    }
    .btn-start {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        border: none !important;
    }
    .btn-stop {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        border: none !important;
    }
    .settings-accordion {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        margin-top: 8px !important;
    }
    .transcript-box {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        font-family: 'Segoe UI', sans-serif !important;
        margin-top: 8px !important;
    }
    footer {
        display: none !important;
    }
    """
    
    with gr.Blocks(title="Gemini Live", css=css, theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
    )) as demo:
        
        # Hidden state for logs
        logs_state = gr.State("")
        
        # =====================================================================
        # MAIN VIDEO AREA - Full width, centered
        # =====================================================================
        with gr.Column():
            # Video feed - NATIVE RESOLUTION (no scaling)
            video = gr.Image(
                label=None,
                type="numpy",
                show_label=False,
                container=False,
                elem_classes=["main-video"]
            )
            
            # Control bar below video
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    status = gr.Textbox(
                        value="⚫ Ready to start",
                        interactive=False,
                        show_label=False,
                        container=False,
                        elem_classes=["status-pill"]
                    )
                
                with gr.Column(scale=2):
                    with gr.Row():
                        start_btn = gr.Button(
                            "Start Call",
                            variant="primary",
                            elem_classes=["btn-call", "btn-start"],
                            scale=2
                        )
                        stop_btn = gr.Button(
                            "End",
                            variant="secondary",
                            elem_classes=["btn-call", "btn-stop"],
                            scale=1
                        )
                
                with gr.Column(scale=1):
                    sam_cb = gr.Checkbox(
                        label="SAM",
                        value=SAM_AVAILABLE,
                        interactive=SAM_AVAILABLE,
                        container=False
                    )
        
        # =====================================================================
        # SETTINGS - Collapsible accordion (minimized by default)
        # =====================================================================
        with gr.Accordion("Settings", open=False, elem_classes=["settings-accordion"]):
            # API Key row
            with gr.Row():
                api_key_status = "Set via environment" if GEMINI_API_KEY else "Not set"
                api_key = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="Enter your Gemini API key (or set GEMINI_API_KEY env var)",
                    type="password",
                    scale=3,
                    info=f"Status: {api_key_status}. Get your key from https://aistudio.google.com/apikey"
                )
            
            with gr.Row():
                mic = gr.Dropdown(
                    choices=mic_choices,
                    value=mic_choices[0][1],
                    label="Microphone",
                    scale=1
                )
                speaker = gr.Dropdown(
                    choices=speaker_choices,
                    value=speaker_choices[0][1],
                    label="Speaker",
                    scale=1
                )
                camera = gr.Dropdown(
                    choices=camera_choices,
                    value=camera_choices[0][1],
                    label="Camera",
                    scale=1
                )
                voice = gr.Dropdown(
                    choices=VOICE_OPTIONS,
                    value=DEFAULT_VOICE,
                    label="AI Voice",
                    scale=1
                )
            
            with gr.Row():
                # SAM Model Selection
                sam_model_choices = [
                    ("SAM2 Tiny (38.9M) - Fastest", "sam2-hiera-tiny"),
                    ("SAM2 Small (46M) - Balanced", "sam2-hiera-small"),
                    ("SAM2 Base+ (80.8M) - Better Accuracy", "sam2-hiera-base-plus"),
                    ("SAM2 Large (224.4M) - Best Accuracy", "sam2-hiera-large"),
                    ("SAM ViT-Base (94M)", "sam-vit-base"),
                    ("SAM ViT-Large (308M)", "sam-vit-large"),
                    ("SAM ViT-Huge (636M) - Highest Quality", "sam-vit-huge"),
                ]
                sam_model = gr.Dropdown(
                    choices=sam_model_choices,
                    value="sam2-hiera-small",
                    label="SAM Model",
                    scale=2,
                    info="Larger models = better accuracy, slower speed"
                )
                sam_prompt_box = gr.Textbox(
                    label="SAM Prompt (auto-generated)",
                    placeholder="Object detection prompt...",
                    interactive=True,
                    scale=2
                )
        
        # =====================================================================
        # CONVERSATION - Below the fold
        # =====================================================================
        with gr.Accordion("Conversation", open=True, elem_classes=["transcript-box"]):
            transcript = gr.Textbox(
                label=None,
                show_label=False,
                lines=6,
                interactive=False,
                placeholder="Conversation will appear here...",
                container=False
            )
        
        # =====================================================================
        # LOGS - Hidden by default
        # =====================================================================
        with gr.Accordion("📋 Debug Logs", open=False):
            logs = gr.Textbox(
                label=None,
                show_label=False,
                lines=8,
                interactive=False,
                container=False
            )
        
        # =====================================================================
        # EVENT HANDLERS
        # =====================================================================
        
        def on_start(m, s, c, v, sam, sam_m, key):
            session.set_devices(m, s, c, v, sam, sam_m, key)
            return session.start()
        
        def on_stop():
            return session.stop()
        
        def on_prompt_change(p):
            session.set_sam_prompt(p)
        
        def update_ui(current_logs):
            st = session.get_status()
            new_logs = session.get_logs()
            tr = session.get_transcript()
            frame = session.get_frame()
            prompt = session.sam_prompt
            
            # Append new logs
            if new_logs:
                current_logs = (current_logs + "\n" + new_logs) if current_logs else new_logs
            
            # Limit log lines
            lines = current_logs.split("\n") if current_logs else []
            if len(lines) > 50:
                current_logs = "\n".join(lines[-50:])
            
            return st, current_logs or "", tr or "", frame, prompt
        
        # Button clicks
        start_btn.click(on_start, [mic, speaker, camera, voice, sam_cb, sam_model, api_key], [])
        stop_btn.click(on_stop, [], [])
        sam_prompt_box.change(on_prompt_change, [sam_prompt_box], [])
        
        # Timer for UI updates
        timer = gr.Timer(0.12)
        timer.tick(update_ui, [logs], [status, logs, transcript, video, sam_prompt_box])
        
        # Initial load
        demo.load(
            lambda: (session.get_status(), "", "", None, ""),
            outputs=[status, logs, transcript, video, sam_prompt_box]
        )
    
    return demo


def main():
    print("\n" + "=" * 60)
    print("  GEMINI LIVE - Video Call Style UI")
    print("=" * 60)
    print(f"Devices: {len(mic_devices)} mics | {len(speaker_devices)} speakers | {len(camera_devices)} cameras")
    print(f"SAM: {'Available' if SAM_AVAILABLE else 'Not Available'}")
    print(f"URL: http://localhost:{SERVER_PORT}\n")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=SERVER_PORT,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
