# handlers/__init__.py
#
# Lazy handler registry — each handler is imported only when first accessed.
# This avoids loading torch, transformers, outetts, etc. at startup, cutting
# import time from 6+ minutes to <1 second and reducing RAM by ~1.5 GB.

import importlib
import logging

logger = logging.getLogger("CrispTTS.handlers")

# Registry: handler_key → (module_path, function_name)
# Module paths are relative to the handlers package.
_HANDLER_REGISTRY = {
    "edge": (".edge_handler", "synthesize_with_edge_tts"),
    "piper": (".piper_handler", "synthesize_with_piper_local"),
    "orpheus_gguf": (".orpheus_gguf_handler", "synthesize_with_orpheus_gguf_local"),
    "orpheus_lm_studio": (".orpheus_api_handler", "synthesize_with_orpheus_lm_studio"),
    "orpheus_ollama": (".orpheus_api_handler", "synthesize_with_orpheus_ollama"),
    "outetts": (".outetts_handler", "synthesize_with_outetts_local"),
    "speecht5": (".speecht5_handler", "synthesize_with_speecht5_transformers"),
    "nemo_fastpitch": (".nemo_handler", "synthesize_with_fastpitch_nemo"),
    "coqui_tts": (".coqui_tts_handler", "synthesize_with_coqui_tts"),
    "orpheus_kartoffel": (".kartoffel", "synthesize_with_orpheus_kartoffel"),
    "llasa_hybrid": (".llasa_hybrid_handler", "synthesize_with_llasa_hybrid"),
    "mlx_audio": (".mlx_audio_handler", "synthesize_with_mlx_audio"),
    "llasa_german_transformers": (
        ".llasa_german_transformers_handler", "synthesize_with_llasa_german_transformers"),
    "llasa_multilingual_transformers": (
        ".llasa_multilingual_transformers_handler", "synthesize_with_llasa_multilingual_transformers"),
    "llasa_hf_transformers": (".llasa_hf_transformers_handler", "synthesize_with_llasa_hf_transformers"),
    "f5_tts": (".f5_tts_handler", "synthesize_with_f5_tts"),
    "tts_cpp": (".tts_cpp_handler", "synthesize_with_tts_cpp"),
    "kokoro_onnx": (".kokoro_onnx_handler", "synthesize_with_kokoro_onnx"),
    "zonos": (".zonos_handler", "synthesize_with_zonos"),
    "chatterbox": (".chatterbox_handler", "synthesize_with_chatterbox"),
    "crispasr": (".crispasr_handler", "synthesize_with_crispasr"),
}


class _LazyHandlerDict(dict):
    """Dict that imports handler functions on first access.

    Supports iteration (keys/values/items) by returning only
    already-loaded handlers — avoids eagerly loading everything
    when code iterates over ALL_HANDLERS.
    """

    def __init__(self):
        super().__init__()
        self._failed = set()  # keys that failed to import

    def __getitem__(self, key):
        # Already loaded?
        if key in self.keys():
            return super().__getitem__(key)
        # Known but not yet loaded?
        if key in _HANDLER_REGISTRY and key not in self._failed:
            func = _import_handler(key)
            if func is not None:
                self[key] = func
                return func
            self._failed.add(key)
        return None

    def get(self, key, default=None):
        result = self[key]
        return result if result is not None else default

    def __contains__(self, key):
        return key in _HANDLER_REGISTRY or super().__contains__(key)

    def all_keys(self):
        """Return all registered handler keys (loaded or not)."""
        return list(_HANDLER_REGISTRY.keys())


def _import_handler(key):
    """Import a single handler by key. Returns the function or None."""
    if key not in _HANDLER_REGISTRY:
        return None
    module_path, func_name = _HANDLER_REGISTRY[key]
    try:
        mod = importlib.import_module(module_path, package="handlers")
        func = getattr(mod, func_name, None)
        if func is not None:
            logger.debug("Handler '%s' imported from %s.", key, module_path)
        else:
            logger.warning("Handler '%s': function '%s' not found in %s.", key, func_name, module_path)
        return func
    except ImportError as e:
        logger.info("Handler '%s' unavailable (missing dependency): %s", key, e)
        return None
    except Exception as e:
        logger.warning("Handler '%s' import failed: %s", key, e)
        return None


ALL_HANDLERS = _LazyHandlerDict()

# Pre-load only the crispasr handler — it has no heavy deps and is the most
# commonly used handler. All others load on demand.
_crispasr_func = _import_handler("crispasr")
if _crispasr_func:
    ALL_HANDLERS["crispasr"] = _crispasr_func

__all__ = ["ALL_HANDLERS"]

logger.debug("Handler registry initialized (lazy). %d handlers registered.", len(_HANDLER_REGISTRY))
