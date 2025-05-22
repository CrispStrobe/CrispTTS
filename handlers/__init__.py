# handlers/__init__.py

import logging

logger = logging.getLogger("CrispTTS.handlers")

# These imports should be okay as they are relative within the 'handlers' package

try:
    from .edge_handler import synthesize_with_edge_tts
    logger.debug("EdgeTTS handler imported.")
except ImportError as e:
    logger.warning(f"Could not import EdgeTTS handler (edge_handler.py): {e}. This handler will be unavailable.")
    synthesize_with_edge_tts = None
try:
    from .piper_handler import synthesize_with_piper_local
    logger.debug("Piper handler imported.")
except ImportError as e:
    logger.warning(f"Could not import Piper handler (piper_handler.py): {e}. This handler will be unavailable.")
    synthesize_with_piper_local = None
try:
    from .orpheus_gguf_handler import synthesize_with_orpheus_gguf_local
    logger.debug("Orpheus GGUF handler imported.")
except ImportError as e:
    logger.warning(f"Could not import Orpheus GGUF handler (orpheus_gguf_handler.py): {e}. This handler will be unavailable.")
    synthesize_with_orpheus_gguf_local = None
try:
    from .orpheus_api_handler import synthesize_with_orpheus_lm_studio, synthesize_with_orpheus_ollama
    logger.debug("Orpheus API handlers (LM Studio, Ollama) imported.")
except ImportError as e:
    logger.warning(f"Could not import Orpheus API handlers (orpheus_api_handler.py): {e}. These handlers will be unavailable.")
    synthesize_with_orpheus_lm_studio = None
    synthesize_with_orpheus_ollama = None
try:
    from .outetts_handler import synthesize_with_outetts_local
    logger.debug("OuteTTS handler imported.")
except ImportError as e:
    logger.warning(f"Could not import OuteTTS handler (outetts_handler.py): {e}. This handler will be unavailable.")
    synthesize_with_outetts_local = None
try:
    from .speecht5_handler import synthesize_with_speecht5_transformers
    logger.debug("SpeechT5 handler imported.")
except ImportError as e:
    logger.warning(f"Could not import SpeechT5 handler (speecht5_handler.py): {e}. This handler will be unavailable.")
    synthesize_with_speecht5_transformers = None
try:
    from .nemo_handler import synthesize_with_fastpitch_nemo
    logger.debug("NeMo FastPitch handler imported.")
except ImportError as e:
    logger.warning(f"Could not import NeMo FastPitch handler (nemo_handler.py): {e}. This handler will be unavailable.")
    synthesize_with_fastpitch_nemo = None
try:
    from .mlx_audio_handler import synthesize_with_mlx_audio
    logger.debug("MLX Audio handler imported.")
except ImportError as e:
    logger.warning(f"Could not import MLX Audio handler (mlx_audio_handler.py): {e}. This handler will be unavailable.")
    synthesize_with_mlx_audio = None
try:
    from .coqui_tts_handler import synthesize_with_coqui_tts
    logger.debug("Coqui TTS handler imported.")
except ImportError as e:
    logger.warning(f"Could not import Coqui TTS handler (coqui_tts_handler.py): {e}. This handler will be unavailable.")
    synthesize_with_coqui_tts = None

# FIXED: Import from kartoffel.py (not .kartoffel)
synthesize_with_orpheus_kartoffel = None # Initialize
try:
    from .kartoffel import synthesize_with_orpheus_kartoffel  # FIXED: matches filename
    if synthesize_with_orpheus_kartoffel:
        logger.info("Orpheus Kartoffel (Transformers) handler imported SUCCESSFULLY.")
    else:
        logger.warning("Orpheus Kartoffel handler file imported, BUT 'synthesize_with_orpheus_kartoffel' function is None.")
        synthesize_with_orpheus_kartoffel = None 
except ImportError as e_imp_kartoffel: 
    logger.error(f"INIT.PY: Could not import Orpheus Kartoffel handler due to ImportError: {e_imp_kartoffel}", exc_info=True)
    synthesize_with_orpheus_kartoffel = None
except Exception as e_other_kartoffel: 
    logger.error(f"INIT.PY: An UNEXPECTED error occurred during import of kartoffel.py: {e_other_kartoffel}", exc_info=True)
    synthesize_with_orpheus_kartoffel = None

# FIXED: Import from llasa_hybrid_handler.py (not .llasa_hybrid_handler)
synthesize_with_llasa_hybrid_func = None # Initialize
try:
    from .llasa_hybrid_handler import synthesize_with_llasa_hybrid  # FIXED: matches filename
    synthesize_with_llasa_hybrid_func = synthesize_with_llasa_hybrid 
    if synthesize_with_llasa_hybrid_func:
        logger.info("LLaSA Hybrid handler imported SUCCESSFULLY.")
    else:
        logger.warning("LLaSA Hybrid handler file imported, but function is None.")
except ImportError as e_imp_llasa: 
    logger.error(f"INIT.PY: Could not import LLaSA Hybrid handler due to ImportError: {e_imp_llasa}", exc_info=True)
except Exception as e_other_llasa: 
    logger.error(f"INIT.PY: An UNEXPECTED error during import of llasa_hybrid_handler.py: {e_other_llasa}", exc_info=True)

ALL_HANDLERS = {
    "edge": synthesize_with_edge_tts,
    "piper_local": synthesize_with_piper_local,
    "orpheus_lex_au": synthesize_with_orpheus_gguf_local,
    "orpheus_sauerkraut": synthesize_with_orpheus_gguf_local,
    "orpheus_lm_studio": synthesize_with_orpheus_lm_studio,
    "orpheus_ollama": synthesize_with_orpheus_ollama,
    "oute_llamacpp": synthesize_with_outetts_local,
    "oute_hf": synthesize_with_outetts_local,
    "speecht5_german_transformers": synthesize_with_speecht5_transformers,
    "fastpitch_german_nemo": synthesize_with_fastpitch_nemo,
    "mlx_audio_kokoro_de": synthesize_with_mlx_audio,
    "mlx_audio_csm_clone": synthesize_with_mlx_audio,
    "mlx_audio_outetts_q4": synthesize_with_mlx_audio,
    "coqui_tts_thorsten_ddc": synthesize_with_coqui_tts,
    "coqui_tts_thorsten_vits": synthesize_with_coqui_tts,
    "coqui_tts_thorsten_dca": synthesize_with_coqui_tts,
    "orpheus_kartoffel_natural": synthesize_with_orpheus_kartoffel,
    "llasa_hybrid": synthesize_with_llasa_hybrid_func, 
    "llasa_hybrid_de_clone": synthesize_with_llasa_hybrid_func, 
}

ALL_HANDLERS = {k: v for k, v in ALL_HANDLERS.items() if v is not None}

__all__ = [name for name, func in ALL_HANDLERS.items() if func is not None] # Export only function names
__all__.append("ALL_HANDLERS")

logger.info(f"TTS Handlers package initialized. Mapped and available handlers: {list(ALL_HANDLERS.keys())}")