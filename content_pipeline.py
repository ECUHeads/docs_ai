"""
Automated content generation pipeline.

Orchestrates a multi-stage pipeline for generating multimedia content:
    1. Script Generation — Uses Qwen model to create scripts from knowledge graph context.
    2. Audio Synthesis — Converts script text to speech using Omnivoice.
    3. Image Generation — Creates illustrative images using ComfyUI.

Each stage loads its model via VRAMModelContext for safe GPU memory management,
ensuring only one large model resides in VRAM at a time.

Requirements:
    - Independent and testable stages
    - Pipeline can resume from failed stage
    - Absolute file paths for all outputs
    - UTF-8 encoding for all text files
    - Progress logging at each stage
"""

import asyncio
import aiohttp
import logging
import re
import os
import uuid
import time
import wave
import struct
import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, List, Any, Dict

from vram_manager import VRAMManager, ModelType, VRAMModelContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voice Configuration
# ---------------------------------------------------------------------------

@dataclass
class VoiceConfig:
    """Configuration for voice synthesis parameters.

    Attributes:
        voice_id: Identifier for the voice to use (e.g. "th_female_1", "en_male_1").
        speech_rate: Speed of speech (0.5 = half speed, 1.0 = normal, 2.0 = double).
        pitch: Pitch adjustment in semitones (-12 to +12, 0 = unchanged).
        language: Language code for voice selection ("th" or "en").
        volume_gain: Volume gain in dB (0.0 = unchanged).
    """

    # Voice selection
    voice_id: str = "default"
    language: str = "th"

    # Speech parameters
    speech_rate: float = 1.0
    pitch: float = 0.0
    volume_gain: float = 0.0

    # Available voices per language
    VOICES_THAI: Dict[str, str] = field(default_factory=lambda: {
        "th_female_1": "Thai female voice 1 (warm, clear)",
        "th_female_2": "Thai female voice 2 (soft, gentle)",
        "th_male_1": "Thai male voice 1 (deep, authoritative)",
        "th_male_2": "Thai male voice 2 (young, energetic)",
        "default": "th_female_1",
    })

    VOICES_ENGLISH: Dict[str, str] = field(default_factory=lambda: {
        "en_female_1": "English female voice 1 (clear, professional)",
        "en_female_2": "English female voice 2 (warm, conversational)",
        "en_male_1": "English male voice 1 (deep, narrative)",
        "en_male_2": "English male voice 2 (young, dynamic)",
        "default": "en_female_1",
    })

    def get_voice_for_language(self, language: str) -> str:
        """Select the appropriate voice ID for the given language.

        Args:
            language: Language code ("th" or "en").

        Returns:
            Voice ID string suitable for the requested language.
        """
        if language.lower() == "th":
            voices = self.VOICES_THAI
        else:
            voices = self.VOICES_ENGLISH

        # If voice_id is already set and valid for this language, use it
        if self.voice_id in voices:
            return self.voice_id

        # Fall back to language-specific default
        return voices.get("default", "default")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ContentPackage:
    """Container for all assets produced by the content pipeline.

    Attributes:
        topic: The subject or theme of the generated content.
        language: Output language code (e.g. \"th\", \"en\").
        script: Generated script text.
        audio_path: Absolute path to synthesized audio file.
        image_path: Absolute path to generated image file.
        success: Whether the pipeline completed without errors.
        error: Error message if pipeline failed.
        created_at: ISO-8601 timestamp of creation.
    """
    topic: str
    language: str
    script: str = ""
    audio_path: str = ""
    image_path: str = ""
    success: bool = True
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_visual_prompt(self) -> Optional[str]:
        """Extract visual_concept references from the script.

        Parses the script for lines matching the pattern
        ``visual_concept: <description>`` and returns the first match.
        Falls back to None if no concept is found.

        Returns:
            The visual concept string or None.
        """
        if not self.script:
            return None

        # Look for visual_concept references in the script
        pattern = re.compile(r"visual_concept\s*:\s*(.+)", re.IGNORECASE)
        match = pattern.search(self.script)
        if match:
            return match.group(1).strip()

        return None


@dataclass
class PipelineStage:
    """A single executable stage in the content pipeline.

    Attributes:
        name: Human-readable stage identifier.
        execute_func: Async callable that takes a ContentPackage and
                      returns an updated ContentPackage.
    """
    name: str
    execute_func: Callable


# ---------------------------------------------------------------------------
# Placeholder model loaders (stub implementations)
# ---------------------------------------------------------------------------

async def load_qwen_model() -> Any:
    """Load the Qwen 3.6 model for script generation.

    Returns:
        Loaded Qwen model instance with a ``generate_script()`` method.

    Raises:
        NotImplementedError: If Qwen integration is not yet implemented.
    """
    logger.info("Loading Qwen model...")
    # Stub – replace with actual model loading logic
    class _QwenStub:
        def generate_script(self, topic: str, context: str = "", language: str = "th") -> str:
            return f"[{language.upper()} Script] Topic: {topic}\nContext: {context[:200] if context else 'none'}"

    return _QwenStub()


# ---------------------------------------------------------------------------
# Omnivoice TTS Integration
# ---------------------------------------------------------------------------

# Maximum characters per synthesis chunk to avoid OOM.
# Omnivoice models typically handle ~200-400 chars per batch comfortably.
_OMNIVOICE_MAX_CHUNK_CHARS = 300

# Punctuation boundaries for Thai/English sentence splitting.
_SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?।])\s+|(?<=\n)|(?<=\.|\!|\?)|(?<=۔)"
)


class OmnivoiceTTS:
    """Omnivoice-based text-to-speech synthesizer.

    Supports Thai and English speech synthesis with configurable voice,
    rate, pitch, and volume. Handles long texts by splitting into chunks
    to avoid out-of-memory errors.

    Attributes:
        model: The underlying Omnivoice model instance.
        sample_rate: Audio sample rate in Hz (default 24000).
        device: Compute device string (e.g. "cuda:0", "cpu").
    """

    def __init__(
        self,
        model: Any = None,
        sample_rate: int = 24000,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        language: str = "th",
        voice_config: Optional[VoiceConfig] = None,
        output_dir: str = "output/audio/",
        sample_rate: int = 24000,
    ) -> str:
        """Synthesize speech from text.

        Splits long texts into sentence-level chunks to avoid OOM,
        synthesizes each chunk, and concatenates the resulting audio
        into a single WAV file.

        Args:
            text: Input text to synthesize.
            language: Language code ("th" or "en").
            voice_config: Optional voice configuration for rate/pitch/voice selection.
            output_dir: Directory for output audio file.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Absolute path to the generated WAV audio file.

        Raises:
            RuntimeError: If synthesis fails (pipeline catches this).
        """
        if voice_config is None:
            voice_config = VoiceConfig(language=language)

        voice_id = voice_config.get_voice_for_language(language)
        effective_rate = sample_rate if sample_rate else self.sample_rate

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename
        filename = f"audio_{uuid.uuid4().hex[:8]}_{voice_id}.wav"
        output_path = os.path.join(output_dir, filename)

        start_time = time.time()

        try:
            # Split text into chunks for memory safety
            chunks = _split_text_into_chunks(text, max_chars=_OMNIVOICE_MAX_CHUNK_CHARS)
            logger.info(
                f"Splitting text into {len(chunks)} chunk(s) "
                f"({len(text)} chars total) for Omnivoice synthesis"
            )

            # Synthesize each chunk and collect audio segments
            audio_segments = []
            for idx, chunk in enumerate(chunks):
                logger.debug(f"Synthesizing chunk {idx + 1}/{len(chunks)}: {chunk[:60]}...")
                segment = self._synthesize_chunk(
                    text=chunk,
                    language=language,
                    voice_id=voice_id,
                    speech_rate=voice_config.speech_rate,
                    pitch=voice_config.pitch,
                    volume_gain=voice_config.volume_gain,
                    sample_rate=effective_rate,
                )
                audio_segments.append(segment)

            # Concatenate segments and save
            concatenated = _concat_audio_segments(audio_segments, effective_rate)
            _save_wav(concatenated, effective_rate, output_path)

            elapsed = time.time() - start_time
            file_size = os.path.getsize(output_path)

            logger.info(
                f"Omnivoice synthesis completed in {elapsed:.2f}s — "
                f"file: {os.path.basename(output_path)}, "
                f"size: {file_size / 1024:.1f} KB, "
                f"chunks: {len(chunks)}"
            )

            return os.path.abspath(output_path)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Omnivoice synthesis failed after {elapsed:.2f}s: {e}")
            raise RuntimeError(f"Audio synthesis failed: {e}") from e

    def cleanup(self) -> None:
        """Release model resources and clear GPU memory."""
        try:
            if self.model is not None:
                # Attempt to move model to CPU before deletion
                if hasattr(self.model, "cpu"):
                    self.model = self.model.cpu()
                del self.model
                self.model = None

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("OmnivoiceTTS resources cleaned up.")
        except Exception as e:
            logger.warning(f"Error during OmnivoiceTTS cleanup: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _synthesize_chunk(
        self,
        text: str,
        language: str,
        voice_id: str,
        speech_rate: float,
        pitch: float,
        volume_gain: float,
        sample_rate: int,
    ) -> bytes:
        """Synthesize a single text chunk into raw PCM audio bytes.

        If the real Omnivoice model is available, use it for inference.
        Otherwise, fall back to generating a silent WAV placeholder so
        the pipeline does not crash.

        Returns:
            Raw PCM audio bytes (little-endian 16-bit signed).
        """
        # --- Real Omnivoice inference path ---
        if self.model is not None:
            try:
                result = self.model.infer(
                    text=text,
                    language=language,
                    speaker=voice_id,
                    speed=speech_rate,
                    pitch=pitch,
                    volume=volume_gain,
                )
                # Handle various return formats from Omnivoice
                if isinstance(result, dict):
                    audio = result.get("audio", result.get("waveform", None))
                    sr = result.get("sample_rate", sample_rate)
                elif hasattr(result, "audio"):
                    audio = result.audio
                    sr = getattr(result, "sample_rate", sample_rate)
                else:
                    audio = result
                    sr = sample_rate

                # Convert numpy array to raw bytes if needed
                import numpy as np
                if isinstance(audio, np.ndarray):
                    # Normalize to [-1, 1] then convert to int16
                    audio = np.clip(audio, -1.0, 1.0)
                    audio = (audio * 32767).astype(np.int16)
                    return audio.tobytes()
                elif isinstance(audio, bytes):
                    return audio
            except Exception as e:
                logger.warning(f"Omnivoice model inference failed, using fallback: {e}")

        # --- Fallback: generate silence placeholder ---
        # Generate ~0.5 seconds of silence proportional to text length
        duration_seconds = max(0.3, len(text) * 0.04 / speech_rate)
        num_samples = int(duration_seconds * sample_rate)
        silent = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        return silent


# ---------------------------------------------------------------------------
# Audio utility helpers
# ---------------------------------------------------------------------------

def _split_text_into_chunks(text: str, max_chars: int = _OMNIVOICE_MAX_CHUNK_CHARS) -> List[str]:
    """Split text into sentence-bounded chunks respecting a max character limit.

    Prioritizes splitting at sentence boundaries (periods, exclamation marks,
    question marks, newlines). Falls back to hard splits if a single sentence
    exceeds ``max_chars``.

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence boundaries
    sentences = _SENTENCE_SPLIT_PATTERN.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # If still no chunks (single huge sentence), do hard split
    if not chunks:
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i + max_chars])

    return chunks


def _concat_audio_segments(segments: List[bytes], sample_rate: int) -> bytes:
    """Concatenate multiple PCM audio segments into a single byte string.

    Args:
        segments: List of raw PCM byte strings (16-bit LE).
        sample_rate: Sample rate (used only for metadata, not conversion).

    Returns:
        Concatenated raw PCM bytes.
    """
    return b"".join(segments)


def _save_wav(audio_bytes: bytes, sample_rate: int, output_path: str) -> None:
    """Save raw PCM bytes as a WAV file.

    Args:
        audio_bytes: Raw 16-bit little-endian PCM audio data.
        sample_rate: Audio sample rate in Hz.
        output_path: File path for the output WAV.
    """
    num_channels = 1
    sampwidth = 2  # 16-bit

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)


async def load_omnivoice_model() -> OmnivoiceTTS:
    """Load the Omnivoice TTS model.

    Attempts to import and initialize the Omnivoice library with its
    pretrained weights. Falls back to a stub instance if the library
    is not installed, allowing the pipeline to continue in degraded mode.

    Returns:
        An ``OmnivoiceTTS`` instance ready for synthesis.
    """
    logger.info("Loading Omnivoice TTS model...")

    model = None
    try:
        # Try to import Omnivoice
        from omnivoice import Omnivoice  # type: ignore
        logger.info("Omnivoice library found, loading model weights...")

        # Initialize Omnivoice with default pretrained config
        model = Omnivoice.from_pretrained(
            checkpoint_path="./cache/omnivoice_large.pt",
            config_path="./cache/config.json",
        )

        if hasattr(model, "to"):
            device = "cuda" if _cuda_available() else "cpu"
            model = model.to(device)
            logger.info(f"Omnivoice model loaded on device: {device}")
        else:
            logger.info("Omnivoice model loaded (device-agnostic).")

    except ImportError:
        logger.warning(
            "omnivoice package not installed. "
            "TTS will use silent-audio fallback. "
            "Install with: pip install omnivoice"
        )
    except Exception as e:
        logger.warning(f"Failed to load Omnivoice model weights: {e}")
        logger.warning("TTS will use silent-audio fallback.")

    device = "cuda" if _cuda_available() else "cpu"
    tts = OmnivoiceTTS(model=model, sample_rate=24000, device=device)
    logger.info(f"OmnivoiceTTS initialized (model={'real' if model else 'fallback'}, device={device})")
    return tts


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# ComfyUI Integration
# ---------------------------------------------------------------------------

# Supported model configurations for ComfyUI workflows.
_COMFYUI_MODELS: Dict[str, Dict[str, str]] = {
    "sdxl": {
        "checkpoint": "sd_xl_base_1.0.safetensors",
        "vae": "taesd.xl",
    },
    "sd15": {
        "checkpoint": "v1-5-pruned-emaonly.safetensors",
        "vae": "vae-ft-mse-840000-ema.safetensors",
    },
}

# Default model family.
_DEFAULT_COMFYUI_MODEL = "sdxl"

# Maximum time (seconds) to wait for a single image generation.
_COMFYUI_GENERATION_TIMEOUT = 300  # 5 minutes


class ComfyUIClient:
    """Async client for the ComfyUI API.

    Submits text-to-image prompts to a running ComfyUI server, polls for
    completion, and saves the generated image to disk.

    Attributes:
        api_url: Base URL of the ComfyUI API (e.g. http://localhost:8188).
        session: Underlying aiohttp client session.
        model_family: Model family key ("sdxl" or "sd15").
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8188",
        model_family: str = _DEFAULT_COMFYUI_MODEL,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self.model_family = model_family.lower() if model_family.lower() in _COMFYUI_MODELS else _DEFAULT_COMFYUI_MODEL

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create and return an aiohttp ClientSession."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_COMFYUI_GENERATION_TIMEOUT)
            )
        return self.session

    async def generate(
        self,
        prompt: str,
        output_dir: str = "output/images/",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        model_family: Optional[str] = None,
    ) -> str:
        """Generate an image from a text prompt via the ComfyUI API.

        Builds a ComfyUI workflow JSON, submits it to the server, waits for
        completion, and saves the resulting image.

        Args:
            prompt: Text prompt for image generation.
            output_dir: Directory for the output image.
            width: Image width in pixels.
            height: Image height in pixels.
            steps: Number of sampling steps.
            cfg: Classifier-free guidance scale.
            seed: Optional random seed (auto-generated if None).
            model_family: Override model family ("sdxl" or "sd15").

        Returns:
            Absolute path to the generated image file.

        Raises:
            RuntimeError: If ComfyUI is unreachable or generation fails.
        """
        session = await self._ensure_session()
        family = (model_family or self.model_family).lower()
        model_config = _COMFYUI_MODELS.get(family, _COMFYUI_MODELS[_DEFAULT_COMFYUI_MODEL])

        if seed is None:
            seed = int(uuid.uuid4().hex[:8], 16)

        filename = f"image_{uuid.uuid4().hex[:8]}.png"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        start_time = time.time()
        logger.info(
            f"ComfyUI generation started - prompt: {prompt[:80]}..., "
            f"size: {width}x{height}, steps: {steps}, cfg: {cfg}, "
            f"seed: {seed}, model: {family}"
        )

        try:
            workflow = self._build_workflow(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
                model_config=model_config,
            )

            async with session.post(
                f"{self.api_url}/prompt",
                json={"prompt": workflow, "client_id": str(uuid.uuid4())},
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise RuntimeError(
                        f"ComfyUI /prompt returned HTTP {response.status}: {error_body}"
                    )
                result = await response.json()
                prompt_id = result["prompt_id"]

            image_bytes = await self._wait_for_result(prompt_id)
            await self._save_image(image_bytes, output_path)

            elapsed = time.time() - start_time
            file_size = os.path.getsize(output_path)
            logger.info(
                f"ComfyUI generation completed in {elapsed:.2f}s - "
                f"file: {os.path.basename(output_path)}, "
                f"size: {file_size / 1024:.1f} KB"
            )

            return os.path.abspath(output_path)

        except aiohttp.ClientConnectorError as e:
            elapsed = time.time() - start_time
            logger.error(f"ComfyUI connection failed after {elapsed:.2f}s: {e}")
            raise RuntimeError(f"Cannot connect to ComfyUI at {self.api_url}: {e}") from e
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"ComfyUI generation timed out after {elapsed:.2f}s")
            raise RuntimeError(
                f"ComfyUI generation exceeded {_COMFYUI_GENERATION_TIMEOUT}s timeout"
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"ComfyUI generation failed after {elapsed:.2f}s: {e}")
            raise RuntimeError(f"Image generation failed: {e}") from e

    def _build_workflow(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        model_config: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build a ComfyUI workflow JSON for text-to-image generation.

        Creates a standard checkpoint-loader + CLIP-text-encode + KSampler
        + VAEDecode workflow compatible with SDXL and SD 1.5 checkpoints.

        Args:
            prompt: Positive text prompt.
            width: Image width.
            height: Image height.
            steps: Sampling steps.
            cfg: CFG scale.
            seed: Random seed.
            model_config: Model configuration dict from _COMFYUI_MODELS.

        Returns:
            Workflow dict for ComfyUI's /prompt endpoint.
        """
        is_sdxl = self.model_family == "sdxl"

        if is_sdxl:
            return {
                "3": {
                    "class_type": "KSampler",
                    "inputs": {
                        "cfg": cfg,
                    "denoise": 1.0,
                    "latent_image": ["2", 0],
                    "model": ["5", 0],
                    "negative": ["8", 0],
                    "positive": ["10", 0],
                    "sampler_name": "euler",
                    "seed": seed,
                    "steps": steps,
                },
                },
                "5": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": model_config["checkpoint"]},
                },
                "7": {
                    "class_type": "CLIPSetLastLayer",
                    "inputs": {"clip": ["5", 1], "last_layer": -1},
                },
                "8": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["7", 0],
                        "text": "worst quality, low quality, blurry, distorted, ugly, deformed",
                    },
                },
                "10": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["7", 0], "text": prompt},
                },
                "2": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {
                        "batch_size": 1,
                        "height": height // 8,
                        "width": width // 8,
                    },
                },
                "4": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["3", 0], "vae": ["5", 2]},
                },
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["4", 0]},
                },
            }

        # SD 1.5 workflow (simpler single-CLIP path)
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "seed": seed,
                    "steps": steps,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": model_config["checkpoint"]},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": height // 8,
                    "width": width // 8,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": prompt},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": "worst quality, low quality, blurry, distorted, ugly, deformed",
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"images": ["8", 0]},
            },
        }

    async def _wait_for_result(self, prompt_id: str) -> bytes:
        """Poll ComfyUI for the generation result of a submitted prompt.

        Uses the history endpoint to check completion and retrieve image data.

        Args:
            prompt_id: Prompt ID returned by /prompt endpoint.

        Returns:
            Raw PNG image bytes.

        Raises:
            RuntimeError: If generation fails or times out.
        """
        session = await self._ensure_session()
        start_time = time.time()
        poll_interval = 2.0

        while time.time() - start_time < _COMFYUI_GENERATION_TIMEOUT:
            await asyncio.sleep(poll_interval)

            async with session.get(f"{self.api_url}/history/{prompt_id}") as response:
                if response.status != 200:
                    continue
                history = await response.json()

            if prompt_id not in history:
                continue

            prompt_data = history[prompt_id]
            status = prompt_data.get("status", {})
            status_str = status.get("status_str", "")

            if status_str == "error":
                messages = prompt_data.get("messages", [])
                error_msg = "; ".join(str(m) for m in messages)
                raise RuntimeError(f"ComfyUI generation failed: {error_msg}")

            outputs = prompt_data.get("outputs", {})
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    image_info = node_output["images"][0]
                    image_filename = image_info.get("filename", "")
                    subfolder = image_info.get("subfolder", "")
                    output_type = image_info.get("type", "output")

                    async with session.get(
                        f"{self.api_url}/view?filename={image_filename}"
                        f"&subfolder={subfolder}&type={output_type}"
                    ) as img_response:
                        if img_response.status == 200:
                            return await img_response.read()

        raise RuntimeError(
            f"ComfyUI generation timed out after {_COMFYUI_GENERATION_TIMEOUT}s "
            f"(prompt_id: {prompt_id})"
        )

    async def _save_image(self, data: bytes, path: str) -> None:
        """Save image bytes to disk, creating parent directories if needed.

        Args:
            data: Raw image bytes (PNG).
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        logger.debug(f"Image saved to {path} ({len(data)} bytes)")

    async def verify_connection(self) -> bool:
        """Verify connectivity to the ComfyUI server.

        Returns:
            True if the server responds, False otherwise.
        """
        session = await self._ensure_session()
        try:
            async with session.get(f"{self.api_url}/system_stats") as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            try:
                async with session.get(f"{self.api_url}/") as response:
                    return response.status < 400
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return False

    async def cleanup(self) -> None:
        """Close the underlying HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.info("ComfyUIClient session closed.")


async def load_comfyui_api() -> ComfyUIClient:
    """Initialize and verify a ComfyUI API connection.

    Connects to the ComfyUI server (default: http://localhost:8188),
    verifies the connection is alive, and returns a ready-to-use client.

    Returns:
        A ``ComfyUIClient`` instance with an active HTTP session.

    Raises:
        RuntimeError: If the ComfyUI server cannot be reached.
    """
    api_url = os.environ.get("COMFYUI_API_URL", "http://localhost:8188")
    model_family = os.environ.get("COMFYUI_MODEL_FAMILY", _DEFAULT_COMFYUI_MODEL)

    logger.info(f"Connecting to ComfyUI API at {api_url} (model: {model_family})...")
    client = ComfyUIClient(api_url=api_url, model_family=model_family)

    try:
        is_connected = await client.verify_connection()
        if is_connected:
            logger.info("ComfyUI API connection verified successfully.")
        else:
            logger.warning(
                f"ComfyUI server at {api_url} did not respond to health check. "
                "The client will attempt connections on first use."
            )
    except Exception as e:
        logger.warning(
            f"ComfyUI health check failed: {e}. "
            "The client will attempt connections on first use."
        )

    return client


# ---------------------------------------------------------------------------
# Hybrid search helper
# ---------------------------------------------------------------------------

def _get_search_context(topic: str) -> dict:
    """Run hybrid search and return context for script generation.

    Args:
        topic: Search query / content topic.

    Returns:
        Dict with ``context_block`` key containing assembled context string.
    """
    try:
        from hybrid_search import get_hybrid_search
        engine = get_hybrid_search()
        results = engine.search(topic)
        context_block = results.get("context_block", "") if isinstance(results, dict) else ""
        return {"context_block": context_block}
    except Exception as e:
        logger.warning(f"Hybrid search unavailable: {e}")
        return {"context_block": ""}


# ---------------------------------------------------------------------------
# Content Pipeline
# ---------------------------------------------------------------------------

class ContentPipeline:
    """Full automated content generation pipeline.

    Executes three sequential stages:
        1. **Script Generation** – Qwen model creates a script from the topic
           and knowledge graph context.
        2. **Audio Synthesis** – Omnivoice converts the script to speech.
        3. **Image Generation** – ComfyUI generates an illustrative image.

    Each stage loads its model through ``VRAMModelContext`` so that only one
    large model occupies VRAM at any given time.

    Attributes:
        vram_manager: Singleton VRAM manager instance.
        stages: Ordered list of ``PipelineStage`` objects.
    """

    def __init__(self) -> None:
        self.vram_manager = VRAMManager()
        self.stages: List[PipelineStage] = [
            PipelineStage("script_generation", self._generate_script),
            PipelineStage("audio_synthesis", self._synthesize_audio),
            PipelineStage("image_generation", self._generate_image),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self, topic: str, language: str = "th", resume_from: Optional[str] = None
    ) -> ContentPackage:
        """Execute the full content pipeline: script → audio → image.

        Args:
            topic: Content topic / subject.
            language: Output language code (\"th\" or \"en\").
            resume_from: Stage name to resume from (skips earlier stages).

        Returns:
            ``ContentPackage`` with all generated assets.
        """
        package = ContentPackage(topic=topic, language=language)

        start_index = 0
        if resume_from:
            for i, stage in enumerate(self.stages):
                if stage.name == resume_from:
                    start_index = i
                    logger.info(f"Resuming pipeline from stage: {stage.name}")
                    break
            else:
                logger.warning(f"Unknown resume stage '{resume_from}', starting from beginning")

        for stage in self.stages[start_index:]:
            logger.info(f"Starting pipeline stage: {stage.name}")

            try:
                result = await stage.execute_func(package)
                package = result  # Each stage returns updated package

                if not package.success:
                    logger.error(f"Pipeline failed at stage: {stage.name}")
                    break

                logger.info(f"Stage {stage.name} completed successfully")

            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                package.success = False
                package.error = str(e)
                break

        return package

    # ------------------------------------------------------------------
    # Pipeline Stages
    # ------------------------------------------------------------------

    async def _generate_script(self, package: ContentPackage) -> ContentPackage:
        """Stage 1 – Generate script using Qwen 3.6.

        Loads the Qwen model via VRAM context manager, queries the knowledge
        graph for contextual information, and generates a language-appropriate
        script.

        Args:
            package: ContentPackage with ``topic`` and ``language`` set.

        Returns:
            Updated ContentPackage with ``script`` populated.
        """
        # Gather context from knowledge graph before loading model
        search_results = _get_search_context(package.topic)

        async with VRAMModelContext(
            self.vram_manager, ModelType.QWEN, load_qwen_model
        ) as qwen:
            script = qwen.generate_script(
                topic=package.topic,
                context=search_results["context_block"],
                language=package.language,
            )

            package.script = script

            # Save script to file for persistence
            script_path = self._save_script(package)
            logger.info(f"Script saved to {script_path}")

        return package

    async def _synthesize_audio(self, package: ContentPackage) -> ContentPackage:
        """Stage 2 – Synthesize audio using Omnivoice.

        Loads the Omnivoice model via VRAM context manager and converts the
        script text into a speech audio file. Uses VoiceConfig for language-
        specific voice selection with configurable rate, pitch, and volume.

        Errors during synthesis are caught and recorded on the package rather
        than crashing the pipeline.

        Args:
            package: ContentPackage with ``script`` populated.

        Returns:
            Updated ContentPackage with ``audio_path`` set (or error recorded).
        """
        if not package.script:
            logger.error("Cannot synthesize audio: no script available")
            package.success = False
            package.error = "No script available for audio synthesis"
            return package

        try:
            async with VRAMModelContext(
                self.vram_manager, ModelType.OMNIVOICE, load_omnivoice_model
            ) as omnivoice:
                # Build voice configuration from package language
                voice_config = VoiceConfig(language=package.language)

                audio_dir = os.path.abspath("output/audio")
                audio_path = omnivoice.synthesize(
                    text=package.script,
                    language=package.language,
                    voice_config=voice_config,
                    output_dir=audio_dir,
                )

                package.audio_path = os.path.abspath(audio_path)

        except RuntimeError as e:
            # Synthesis-specific failure — record on package, do not crash
            logger.error(f"Audio synthesis failed: {e}")
            package.success = False
            package.error = str(e)
        except Exception as e:
            # Unexpected failure
            logger.error(f"Unexpected error during audio synthesis: {e}")
            package.success = False
            package.error = f"Audio synthesis error: {e}"

        return package

    async def _generate_image(self, package: ContentPackage) -> ContentPackage:
        """Stage 3 – Generate image using ComfyUI.

        Loads the ComfyUI API client via VRAM context manager and generates
        an illustrative image based on a visual prompt extracted from the
        script or topic.

        Args:
            package: ContentPackage with ``script`` and ``topic`` set.

        Returns:
            Updated ContentPackage with ``image_path`` set.
        """
        async with VRAMModelContext(
            self.vram_manager, ModelType.COMFYUI, load_comfyui_api
        ) as comfyui:
            image_prompt = package.get_visual_prompt() or f"Illustration of {package.topic}"

            image_dir = os.path.abspath("output/images")
            image_path = comfyui.generate(
                prompt=image_prompt,
                output_dir=image_dir,
            )

            package.image_path = os.path.abspath(image_path)

        return package

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_script(self, package: ContentPackage) -> str:
        """Save the generated script to a UTF-8 text file.

        Args:
            package: ContentPackage with ``script`` populated.

        Returns:
            Absolute path to the saved script file.
        """
        script_dir = os.path.abspath("output/scripts")
        os.makedirs(script_dir, exist_ok=True)

        # Sanitize topic for filename
        safe_topic = re.sub(r"[^\w\s-]", "", package.topic).strip().replace(" ", "_")
        filename = f"{safe_topic}_{package.language}_{package.created_at[:19].replace(':', '-')}.txt"
        path = os.path.join(script_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(package.script)

        return os.path.abspath(path)
