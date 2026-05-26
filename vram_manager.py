import asyncio
import os
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict, List

import torch

logger = logging.getLogger(__name__)


class VRAMState(Enum):
    IDLE = "idle"
    QWEN_LOADED = "qwen_loaded"
    OMNIVOICE_LOADED = "omnivoice_loaded"
    COMFYUI_LOADED = "comfyui_loaded"


class ModelType(Enum):
    QWEN = "qwen"
    OMNIVOICE = "omnivoice"
    COMFYUI = "comfyui"


class VRAMConflictError(Exception):
    pass


class VRAMManager:
    _instance = None
    _state: VRAMState = VRAMState.IDLE
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> "VRAMManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.safe_limit_gb = int(os.getenv("VRAM_SAFE_LIMIT_GB", "22"))
        self.sequential_mode = os.getenv("VRAM_SEQUENTIAL_MODE", "true").lower() == "true"

        # Monitoring attributes
        self._monitoring = False
        self._monitor_interval = 10
        self._monitor_task: Optional[asyncio.Task] = None
        self._vram_history: List[Dict] = []

    async def load_model(self, model_type: ModelType, load_func: Callable) -> Any:
        """Load a model with VRAM safety checks.

        Args:
            model_type: Which model to load.
            load_func: Async callable that loads the model.

        Returns:
            Loaded model instance.

        Raises:
            VRAMConflictError: If another model is already loaded in sequential mode.
        """
        async with self._lock:
            if self._state != VRAMState.IDLE and self.sequential_mode:
                raise VRAMConflictError(
                    f"Cannot load {model_type.value}: {self._state.value} is active"
                )

            # Check available VRAM
            available_gb = self._get_available_vram()
            if available_gb < self.safe_limit_gb:
                logger.warning(
                    f"Low VRAM: {available_gb:.1f}GB available, "
                    f"{self.safe_limit_gb}GB required"
                )

            # Load model via provided async callable
            model = await load_func()

            # Update state to reflect the loaded model type
            self._state = VRAMState(f"{model_type.value}_LOADED")

            # Log VRAM usage after loading
            used_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            logger.info(f"Loaded {model_type.value}: {used_gb:.1f}GB VRAM used")

            return model

    async def unload_model(self) -> None:
        """Unload current model and clean up CUDA cache."""
        async with self._lock:
            if self._state == VRAMState.IDLE:
                logger.warning("No model to unload")
                return

            model_name = self._state.value.replace("_LOADED", "")
            logger.info(f"Unloading {model_name}")

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            self._state = VRAMState.IDLE
            logger.info("Model unloaded, VRAM state: IDLE")

    def _get_available_vram(self) -> float:
        """Get available VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0

        total = torch.cuda.get_device_properties(0).total_mem
        allocated = torch.cuda.memory_allocated()
        available = (total - allocated) / (1024 ** 3)
        return available

    def get_state(self) -> VRAMState:
        """Return the current VRAM state."""
        return self._state

    def get_vram_stats(self) -> Dict[str, float]:
        """Return current VRAM statistics."""
        if not torch.cuda.is_available():
            return {"available_gb": 0.0, "used_gb": 0.0, "total_gb": 0.0}

        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        used = torch.cuda.memory_allocated() / (1024 ** 3)
        return {
            "total_gb": total,
            "used_gb": used,
            "available_gb": total - used,
            "state": self._state.value,
        }

    async def start_monitor(self, interval_seconds: int = 10, callback: Callable = None) -> None:
        """Start background VRAM monitoring."""
        self._monitoring = True
        self._monitor_interval = interval_seconds

        async def monitor_loop():
            while self._monitoring:
                stats = self.get_vram_stats()
                logger.debug(f"VRAM Monitor: {stats}")

                # Record usage to history
                self._record_vram_usage()

                # Check threshold
                if stats["used_gb"] > self.safe_limit_gb:
                    logger.warning(
                        f"VRAM threshold exceeded: {stats['used_gb']:.1f}GB > {self.safe_limit_gb}GB"
                    )

                    # Trigger auto-unload if in sequential mode
                    if self.sequential_mode and self._state != VRAMState.IDLE:
                        logger.warning("Auto-unloading model due to VRAM threshold")
                        await self.unload_model()

                    # Call alert callback
                    if callback:
                        await callback(stats, "threshold_exceeded")

                await asyncio.sleep(interval_seconds)

        self._monitor_task = asyncio.create_task(monitor_loop())

    async def stop_monitor(self) -> None:
        """Stop background VRAM monitoring."""
        self._monitoring = False
        if hasattr(self, '_monitor_task') and self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    def get_vram_history(self) -> List[Dict]:
        """Return VRAM usage history for diagnostics."""
        return self._vram_history

    def _record_vram_usage(self) -> None:
        """Record current VRAM usage to history."""
        import time
        self._vram_history.append({
            "timestamp": time.time(),
            "stats": self.get_vram_stats()
        })
        # Keep last 1000 entries
        if len(self._vram_history) > 1000:
            self._vram_history = self._vram_history[-1000:]


class VRAMModelContext:
    """Async context manager for safe model loading/unloading."""

    def __init__(self, vram_manager: "VRAMManager", model_type: ModelType, load_func: Callable):
        self.manager = vram_manager
        self.model_type = model_type
        self.load_func = load_func
        self.model = None

    async def __aenter__(self):
        self.model = await self.manager.load_model(self.model_type, self.load_func)
        return self.model

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manager.unload_model()
        self.model = None
        return False
