# handlers/__init__.py

import logging

logger = logging.getLogger("CrispTTS.handlers")

# --- Attempt to import all handler functions ---
# Initialize all to None so they can be checked before adding to ALL_HANDLERS

synthesize_with_edge_tts = None
try:
    from .edge_handler import synthesize_with_edge_tts
    logger.debug("EdgeTTS handler imported.")
except ImportError as e:
    logger.warning(f"Could not import EdgeTTS handler (edge_handler.py): {e}. This handler will be unavailable.")

synthesize_with_piper_local = None
try:
    from .piper_handler import synthesize_with_piper_local
    logger.debug("Piper handler imported.")
except ImportError as e:
    logger.warning(f"Could not import Piper handler (piper_handler.py): {e}. This handler will be unavailable.")

synthesize_with_orpheus_gguf_local = None
try:
    from .orpheus_gguf_handler import synthesize_with_orpheus_gguf_local
    logger.debug("Orpheus GGUF handler imported.")
except ImportError as e:
    logger.warning(f"Could not import Orpheus GGUF handler (orpheus_gguf_handler.py): {e}. This handler will be unavailable.")

synthesize_with_orpheus_lm_studio = None
synthesize_with_orpheus_ollama = None
try:
    from .orpheus_api_handler import synthesize_with_orpheus_lm_studio, synthesize_with_orpheus_ollama
    logger.debug("Orpheus API handlers (LM Studio, Ollama) imported.")
except ImportError as e:
    logger.warning(f"Could not import Orpheus API handlers (orpheus_api_handler.py): {e}. These handlers will be unavailable.")

synthesize_with_outetts_local = None
try:
    from .outetts_handler import synthesize_with_outetts_local
    logger.debug("OuteTTS handler imported.")
except ImportError as e:
    logger.warning(f"Could not import OuteTTS handler (outetts_handler.py): {e}. This handler will be unavailable.")

synthesize_with_speecht5_transformers = None
try:
    from .speecht5_handler import synthesize_with_speecht5_transformers
    logger.debug("SpeechT5 handler imported.")
except ImportError as e:
    logger.warning(f"Could not import SpeechT5 handler (speecht5_handler.py): {e}. This handler will be unavailable.")

synthesize_with_fastpitch_nemo = None
try:
    from .nemo_handler import synthesize_with_fastpitch_nemo
    logger.debug("NeMo FastPitch handler imported.")
except ImportError as e:
    logger.warning(f"Could not import NeMo FastPitch handler (nemo_handler.py): {e}. This handler will be unavailable.")

synthesize_with_mlx_audio = None # General handler for all mlx-audio models
try:
    from .mlx_audio_handler import synthesize_with_mlx_audio
    logger.info("MLX Audio handler (generic) imported successfully.") # Changed to INFO as it's a key integration
except ImportError as e:
    logger.warning(f"Could not import MLX Audio handler (mlx_audio_handler.py): {e}. This handler will be unavailable.")
except Exception as e_mlx_other:
    logger.error(f"An unexpected error occurred during import of mlx_audio_handler.py: {e_mlx_other}", exc_info=True)

synthesize_with_zonos = None
try:
    from .zonos_handler import synthesize_with_zonos
    logger.info("Zonos handler imported successfully.")
except ImportError as e:
    logger.warning(f"Could not import Zonos handler (zonos_handler.py): {e}. This handler will be unavailable.")

synthesize_with_coqui_tts = None # General handler for all Coqui TTS models
try:
    from .coqui_tts_handler import synthesize_with_coqui_tts
    logger.info("Coqui TTS handler (generic) imported successfully.") # Changed to INFO
except ImportError as e:
    logger.warning(f"Could not import Coqui TTS handler (coqui_tts_handler.py): {e}. This handler will be unavailable.")
except Exception as e_coqui_other:
    logger.error(f"An unexpected error occurred during import of coqui_tts_handler.py: {e_coqui_other}", exc_info=True)

synthesize_with_orpheus_kartoffel = None
try:
    from .kartoffel import synthesize_with_orpheus_kartoffel
    if synthesize_with_orpheus_kartoffel:
        logger.info("Orpheus Kartoffel (Transformers) handler imported SUCCESSFULLY.")
    else: # This case should ideally not happen if the import itself was successful
        logger.warning("Orpheus Kartoffel handler file imported, BUT 'synthesize_with_orpheus_kartoffel' function is None (unexpected).")
        synthesize_with_orpheus_kartoffel = None 
except ImportError as e_imp_kartoffel: 
    logger.warning(f"Could not import Orpheus Kartoffel handler (kartoffel.py) due to ImportError: {e_imp_kartoffel}", exc_info=False) # Reduced log level for common missing optional deps
except Exception as e_other_kartoffel: 
    logger.error(f"An UNEXPECTED error occurred during import of kartoffel.py: {e_other_kartoffel}", exc_info=True)

synthesize_with_llasa_hybrid_func = None
try:
    from .llasa_hybrid_handler import synthesize_with_llasa_hybrid
    synthesize_with_llasa_hybrid_func = synthesize_with_llasa_hybrid 
    if synthesize_with_llasa_hybrid_func:
        logger.info("LLaSA Hybrid handler imported SUCCESSFULLY.")
    else: # This case should ideally not happen
        logger.warning("LLaSA Hybrid handler file imported, but function is None (unexpected).")
except ImportError as e_imp_llasa: 
    logger.warning(f"Could not import LLaSA Hybrid handler (llasa_hybrid_handler.py) due to ImportError: {e_imp_llasa}", exc_info=False)
except Exception as e_other_llasa: 
    logger.error(f"An UNEXPECTED error during import of llasa_hybrid_handler.py: {e_other_llasa}", exc_info=True)

# NEW: Separate LLaSA handlers for German and Multilingual models
synthesize_with_llasa_german_transformers_func = None
try:
    from .llasa_german_transformers_handler import synthesize_with_llasa_german_transformers
    synthesize_with_llasa_german_transformers_func = synthesize_with_llasa_german_transformers
    if synthesize_with_llasa_german_transformers_func:
        logger.info("LLaSA German Transformers handler imported SUCCESSFULLY.")
    else:
        logger.warning("LLaSA German Transformers handler file imported, but function is None.")
except ImportError as e_imp_llasa_german:
    logger.warning(f"Could not import LLaSA German Transformers handler due to ImportError: {e_imp_llasa_german}", exc_info=False)
except Exception as e_other_llasa_german:
    logger.error(f"An UNEXPECTED error during import of LLaSA German Transformers handler: {e_other_llasa_german}", exc_info=True)

# NEW: Separate LLaSA handlers for German and Multilingual models
synthesize_with_llasa_hf_transformers = None
try:
    from .llasa_hf_transformers_handler import synthesize_with_llasa_hf_transformers
    synthesize_with_llasa_hf_transformers = synthesize_with_llasa_hf_transformers
    if synthesize_with_llasa_hf_transformers:
        logger.info("LLaSA HF Transformers handler imported SUCCESSFULLY.")
    else:
        logger.warning("LLaSA HF Transformers handler file imported, but function is None.")
except ImportError as e_imp_llasa_german:
    logger.warning(f"Could not import LLaSA HF Transformers handler due to ImportError: {e_imp_llasa_german}", exc_info=False)
except Exception as e_other_llasa_german:
    logger.error(f"An UNEXPECTED error during import of LLaSA HF Transformers handler: {e_other_llasa_german}", exc_info=True)

synthesize_with_llasa_multilingual_transformers_func = None
try:
    from .llasa_multilingual_transformers_handler import synthesize_with_llasa_multilingual_transformers
    synthesize_with_llasa_multilingual_transformers_func = synthesize_with_llasa_multilingual_transformers
    if synthesize_with_llasa_multilingual_transformers_func:
        logger.info("LLaSA Multilingual Transformers handler imported SUCCESSFULLY.")
    else:
        logger.warning("LLaSA Multilingual Transformers handler file imported, but function is None.")
except ImportError as e_imp_llasa_multi:
    logger.warning(f"Could not import LLaSA Multilingual Transformers handler due to ImportError: {e_imp_llasa_multi}", exc_info=False)
except Exception as e_other_llasa_multi:
    logger.error(f"An UNEXPECTED error during import of LLaSA Multilingual Transformers handler: {e_other_llasa_multi}", exc_info=True)

# F5-TTS Handler
synthesize_with_f5_tts = None
try:
    from .f5_tts_handler import synthesize_with_f5_tts
    if synthesize_with_f5_tts:
        logger.info("F5-TTS handler imported SUCCESSFULLY.")
    else:
        logger.warning("F5-TTS handler file imported, but function is None.")
        synthesize_with_f5_tts = None
except ImportError as e_imp_f5:
    logger.warning(f"Could not import F5-TTS handler due to ImportError: {e_imp_f5}", exc_info=False)
except Exception as e_other_f5:
    logger.error(f"An UNEXPECTED error during import of f5_tts_handler.py: {e_other_f5}", exc_info=True)

synthesize_with_kokoro_onnx = None
try:
    from .kokoro_onnx_handler import synthesize_with_kokoro_onnx
    logger.info("Kokoro ONNX handler imported successfully.")
except ImportError as e:
    logger.warning(f"Could not import Kokoro ONNX handler (kokoro_onnx_handler.py): {e}. This handler will be unavailable.")

# tts.cpp handler
synthesize_with_tts_cpp = None
try:
    from .tts_cpp_handler import synthesize_with_tts_cpp
    logger.info("TTS.cpp handler imported successfully.")
except ImportError as e:
    logger.warning(f"Could not import TTS.cpp handler (tts_cpp_handler.py): {e}. This handler will be unavailable.")
except Exception as e_tts_cpp_other:
    logger.error(f"An unexpected error occurred during import of tts_cpp_handler.py: {e_tts_cpp_other}", exc_info=True)

# Add to your ALL_HANDLERS dictionary:
# "f5_tts": synthesize_with_f5_tts,
# --- Standardized Handler Keys ---
# These keys should be used in config.py's "handler_function_key"
ALL_HANDLERS = {
    "edge": synthesize_with_edge_tts,
    "piper": synthesize_with_piper_local,
    # Orpheus GGUF models can share a handler if logic is identical based on config
    "orpheus_gguf": synthesize_with_orpheus_gguf_local, # Use this for both "orpheus_lex_au" and "orpheus_sauerkraut"
    "orpheus_lm_studio": synthesize_with_orpheus_lm_studio,
    "orpheus_ollama": synthesize_with_orpheus_ollama,
    # OuteTTS models can share a handler if logic differentiates via config
    "outetts": synthesize_with_outetts_local, # Use this for "oute_llamacpp" and "oute_hf"
    "speecht5": synthesize_with_speecht5_transformers, 
    "nemo_fastpitch": synthesize_with_fastpitch_nemo, 
    "coqui_tts": synthesize_with_coqui_tts, # Generic handler for all Coqui TTS API models
    "orpheus_kartoffel": synthesize_with_orpheus_kartoffel,
    "llasa_hybrid": synthesize_with_llasa_hybrid_func,
    "mlx_audio": synthesize_with_mlx_audio, # Single key for all mlx-audio handled models
    # Separate handlers for different LLaSA architectures
    "llasa_german_transformers": synthesize_with_llasa_german_transformers_func,
    "llasa_multilingual_transformers": synthesize_with_llasa_multilingual_transformers_func,
    "llasa_hf_transformers": synthesize_with_llasa_hf_transformers,
    "f5_tts": synthesize_with_f5_tts,
    "tts_cpp": synthesize_with_tts_cpp,
    "kokoro_onnx": synthesize_with_kokoro_onnx,
    "zonos": synthesize_with_zonos,
}

# Remove entries where the handler function is None (due to import failure)
ALL_HANDLERS = {k: v for k, v in ALL_HANDLERS.items() if v is not None}

# __all__ should ideally list the function names themselves if you intend to import them directly elsewhere,
# but for this structure, exporting ALL_HANDLERS is key.
__all__ = list(ALL_HANDLERS.keys()) # Export the valid handler keys
__all__.append("ALL_HANDLERS")

logger.info(f"TTS Handlers package initialized. Mapped and available handler keys: {list(ALL_HANDLERS.keys())}")