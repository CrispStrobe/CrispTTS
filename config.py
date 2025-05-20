# config.py

import os
from pathlib import Path
import logging # Use logging for config-level warnings if needed

logger_cfg = logging.getLogger("CrispTTS.config")

# --- Attempt to import PyTorch for type hints if available ---
try:
    import torch
    TORCH_AVAILABLE_FOR_CONFIG = True
except ImportError:
    torch = None # type: ignore
    TORCH_AVAILABLE_FOR_CONFIG = False

# --- Attempt to import OuteTTS enums ---
OUTETTS_AVAILABLE_FOR_CONFIG = False
OuteTTSModels_Enum = None # type: ignore
OuteTTSBackend_Enum = None # type: ignore
OuteTTSLlamaCppQuantization_Enum = None # type: ignore
OuteTTSInterfaceVersion_Enum = None # type: ignore

if TORCH_AVAILABLE_FOR_CONFIG: # OuteTTS depends on PyTorch
    try:
        from outetts import (
            Models as _OuteTTSModels_Enum,
            Backend as _OuteTTSBackend_Enum,
            LlamaCppQuantization as _OuteTTSLlamaCppQuantization_Enum,
            InterfaceVersion as _OuteTTSInterfaceVersion_Enum
        )
        OuteTTSModels_Enum = _OuteTTSModels_Enum
        OuteTTSBackend_Enum = _OuteTTSBackend_Enum
        OuteTTSLlamaCppQuantization_Enum = _OuteTTSLlamaCppQuantization_Enum
        OuteTTSInterfaceVersion_Enum = _OuteTTSInterfaceVersion_Enum
        OUTETTS_AVAILABLE_FOR_CONFIG = True
    except ImportError:
        logger_cfg.info("OuteTTS library not found during config load. OuteTTS enums will be placeholders.")
    except AttributeError:
        logger_cfg.warning("OuteTTS import has AttributeError (likely protobuf). OuteTTS enums will be placeholders.")

# --- Global TTS Constants ---
ORPHEUS_SAMPLE_RATE = 24000

# Voice Lists
ORPHEUS_AVAILABLE_VOICES_BASE = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
ORPHEUS_GERMAN_VOICES = ["jana", "thomas", "max"]
SAUERKRAUT_VOICES = ["Tom", "Anna", "Max", "Lena"]
ORPHEUS_DEFAULT_VOICE = "jana"

EDGE_TTS_ALL_GERMAN_VOICES = [
    "de-AT-IngridNeural", "de-AT-JonasNeural", "de-DE-AmalaNeural", "de-DE-ConradNeural",
    "de-DE-KatjaNeural", "de-DE-KillianNeural", "de-CH-JanNeural", "de-CH-LeniNeural",
]
EDGE_TTS_DEFAULT_GERMAN_VOICE = "de-DE-KatjaNeural"

# --- Piper German Voices ---
# Paths are relative to the root of the piper-voices repository for the specified version (e.g., v1.0.0)
PIPER_GERMAN_VOICE_PATHS = [
    "de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx",
    "de/de_DE/karlsson/low/de_DE-karlsson-low.onnx",
    "de/de_DE/kerstin/low/de_DE-kerstin-low.onnx",
    "de/de_DE/mls/medium/de_DE-mls-medium.onnx",
    "de/de_DE/pavoque/low/de_DE-pavoque-low.onnx",
    "de/de_DE/ramona/low/de_DE-ramona-low.onnx",
    "de/de_DE/thorsten/low/de_DE-thorsten-low.onnx",
    "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx", # Current default
    "de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
    "de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx",
]
PIPER_DEFAULT_GERMAN_VOICE_PATH = "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"

COQUI_THORSTEN_TACOTRON2_DCA = "tts_models/de/thorsten/tacotron2-DCA"

# API Endpoint Defaults
LM_STUDIO_API_URL_DEFAULT = "http://127.0.0.1:1234/v1/completions"
OLLAMA_API_URL_DEFAULT = "http://localhost:11434/api/generate"
LM_STUDIO_HEADERS = {"Content-Type": "application/json"}


# --- MLX Audio Specific Configs ---
MLX_AUDIO_KOKORO_MODEL_ID = "prince-canuma/Kokoro-82M"
MLX_AUDIO_CSM_MODEL_ID = "mlx-community/csm-1b"
MLX_AUDIO_KOKORO_LANG_CODE_GERMAN = "de" # must verify this
MLX_AUDIO_KOKORO_DEFAULT_VOICE = "af_heart"
MLX_AUDIO_KOKORO_VOICES = ["af_heart", "af_nova", "af_bella", "bf_emma"]
MLX_AUDIO_OUTETTS_ONNX_REPO_ID = "OuteAI/Llama-OuteTTS-1.0-1B-ONNX"

# --- OuteTTS Internal Data Structures (Fallback if not directly queryable from library) ---
OUTETTS_INTERNAL_MODEL_INFO_DATA = {}
if OUTETTS_AVAILABLE_FOR_CONFIG and OuteTTSModels_Enum and OuteTTSInterfaceVersion_Enum:
    try:
        OUTETTS_INTERNAL_MODEL_INFO_DATA = {
            OuteTTSModels_Enum.VERSION_0_1_SIZE_350M: {"max_seq_length": 4096, "interface_version": OuteTTSInterfaceVersion_Enum.V1},
            OuteTTSModels_Enum.VERSION_0_2_SIZE_500M: {"max_seq_length": 4096, "interface_version": OuteTTSInterfaceVersion_Enum.V2},
            OuteTTSModels_Enum.VERSION_0_3_SIZE_500M: {"max_seq_length": 4096, "interface_version": OuteTTSInterfaceVersion_Enum.V2},
            OuteTTSModels_Enum.VERSION_0_3_SIZE_1B: {"max_seq_length": 4096, "interface_version": OuteTTSInterfaceVersion_Enum.V2},
            OuteTTSModels_Enum.VERSION_1_0_SIZE_1B: {"max_seq_length": 8192, "interface_version": OuteTTSInterfaceVersion_Enum.V3},
        }
    except AttributeError:
        logger_cfg.warning("Could not fully populate OUTETTS_INTERNAL_MODEL_INFO_DATA due to OuteTTS enum mismatch.")

OUTETTS_VERSION_STRING_MODEL_CONFIG_DATA = {
    "0.1": {"max_seq_length": 4096, "interface_version_enum_val": OuteTTSInterfaceVersion_Enum.V1 if OuteTTSInterfaceVersion_Enum else "V1_STR_FALLBACK"},
    "0.2": {"max_seq_length": 4096, "interface_version_enum_val": OuteTTSInterfaceVersion_Enum.V2 if OuteTTSInterfaceVersion_Enum else "V2_STR_FALLBACK"},
    "0.3": {"max_seq_length": 4096, "interface_version_enum_val": OuteTTSInterfaceVersion_Enum.V2 if OuteTTSInterfaceVersion_Enum else "V2_STR_FALLBACK"},
    "1.0": {"max_seq_length": 8192, "interface_version_enum_val": OuteTTSInterfaceVersion_Enum.V3 if OuteTTSInterfaceVersion_Enum else "V3_STR_FALLBACK"},
}

# --- Main Model Configuration Dictionary ---
GERMAN_TTS_MODELS = {
    "edge": {
        # "handler_function_key": "edge", # Redundant if model_id is the same and used as key in ALL_HANDLERS
        "default_voice_id": EDGE_TTS_DEFAULT_GERMAN_VOICE,
        "available_voices": EDGE_TTS_ALL_GERMAN_VOICES,
        "notes": "MS Edge TTS (cloud). Output: MP3. Internet required."
    },
    "piper_local": {
        # "handler_function_key": "piper_local", # Redundant
        "piper_voice_repo_id": "rhasspy/piper-voices",
        "default_model_path_in_repo": PIPER_DEFAULT_GERMAN_VOICE_PATH,
        "available_voices": PIPER_GERMAN_VOICE_PATHS,
        "requires_hf_token": False,
        "notes": "Local Piper TTS (ONNX). Downloads model/voice from Hugging Face. Output: WAV."
    },
    "orpheus_lex_au": {
        # "handler_function_key": "orpheus_gguf", # REMOVED: ALL_HANDLERS key is "orpheus_lex_au"
        "model_repo_id": "lex-au/Orpheus-3b-FT-Q4_K_M.gguf", # lex-au/Orpheus-3b-FT-Q4_K_M.gguf
        "model_filename": "Orpheus-3b-FT-Q4_K_M.gguf", # Orpheus-3b-German-FT-Q8_0.gguf 
        "requires_hf_token": False,
        "default_voice_id": "jana",
        "available_voices": ORPHEUS_GERMAN_VOICES,
        "notes": "Local Orpheus GGUF (lex-au). Uses llama-cpp-python & decoder.py. Output: WAV."
    },
    "orpheus_sauerkraut": {
        # "handler_function_key": "orpheus_gguf", # REMOVED: ALL_HANDLERS key is "orpheus_sauerkraut"
        "model_repo_id": "VAGOsolutions/SauerkrautTTS-Preview-0.1-Q4_K_M-GGUF",
        "model_filename": "sauerkrauttts_preview_0_1.Q4_K_M.gguf",
        "requires_hf_token": False,
        "default_voice_id": "Tom",
        "available_voices": SAUERKRAUT_VOICES,
        "notes": "Local SauerkrautTTS (Orpheus GGUF). Uses llama-cpp-python & decoder.py. Output: WAV."
    },
    "orpheus_lm_studio": {
        # "handler_function_key": "orpheus_lm_studio", # Redundant
        "api_url": LM_STUDIO_API_URL_DEFAULT,
        "gguf_model_name_in_api": "SauerkrautTTS-Preview-0.1",
        "default_voice_id": "Tom",
        "available_voices": SAUERKRAUT_VOICES + ORPHEUS_GERMAN_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE,
        "notes": "Orpheus via LM Studio API. Requires decoder.py. Output: WAV."
    },
    "orpheus_ollama": {
        # "handler_function_key": "orpheus_ollama", # Redundant
        "api_url": OLLAMA_API_URL_DEFAULT,
        "ollama_model_name": "legraphista/Orpheus",
        "default_voice_id": "tara",
        "available_voices": ORPHEUS_AVAILABLE_VOICES_BASE,
        # ORPHEUS_GERMAN_VOICES + SAUERKRAUT_VOICES + 
        "notes": "Orpheus via Ollama API. Requires decoder.py. Output: WAV."
    },
    "oute_llamacpp": {
        # "handler_function_key": "outetts", # REMOVED: ALL_HANDLERS key is "oute_llamacpp"
        "outetts_model_enum": OuteTTSModels_Enum.VERSION_1_0_SIZE_1B if OUTETTS_AVAILABLE_FOR_CONFIG else "VERSION_1_0_SIZE_1B_STR_FALLBACK",
        "backend_to_use": OuteTTSBackend_Enum.LLAMACPP if OUTETTS_AVAILABLE_FOR_CONFIG else "LLAMACPP_STR_FALLBACK",
        "quantization_to_use": OuteTTSLlamaCppQuantization_Enum.FP16 if OUTETTS_AVAILABLE_FOR_CONFIG else "FP16_STR_FALLBACK",
        "default_voice_id": "./german.wav",
        "available_voices": ["./german.wav"],
        "test_default_speakers": ["EN-FEMALE-1-NEUTRAL"],
        "notes": "Local OuteTTS (LlamaCPP backend). Custom WAV or OuteTTS default ID."
    },
    "oute_hf": {
        "outetts_model_version_str": "1.0",
        "onnx_repo_id": "OuteAI/Llama-OuteTTS-1.0-1B-ONNX", # Source of ONNX files
        "tokenizer_path": "OuteAI/Llama-OuteTTS-1.0-1B",   # Source of base model config/tokenizer
        "onnx_filename_options": ["onnx/model_int8.onnx"],
        # "onnx/model_q4f16.onnx", "onnx/model_q4.onnx", 
        # "onnx_subfolder" is implicit in onnx_filename_options
        "wavtokenizer_model_path": "onnx-community/WavTokenizer-large-speech-75token_decode",
        "interface_version_enum": OuteTTSInterfaceVersion_Enum.V3 if OUTETTS_AVAILABLE_FOR_CONFIG else "V3_STR_FALLBACK",
        "backend_to_use": OuteTTSBackend_Enum.HF if OUTETTS_AVAILABLE_FOR_CONFIG else "HF_STR_FALLBACK",
        "language": "de",
        "torch_dtype_for_hf_wrapper": torch.float16 if TORCH_AVAILABLE_FOR_CONFIG else "torch.float32_STR_FALLBACK",
        "default_voice_id": "./german.wav", # IMPORTANT: Must be < 15-20s
        "available_voices": ["./german.wav"],
        "test_default_speakers": ["EN-FEMALE-1-NEUTRAL"],
        "notes": "OuteTTS (HF ONNX). Ref WAV will be shortened <15s. Output: WAV. Clears /var/folders/ on Mac on failure."
    },
    "mlx_audio_kokoro_de": {
        # "handler_function_key": "mlx_audio", # REMOVED: ALL_HANDLERS key is "mlx_audio_kokoro_de"
        "mlx_model_path": MLX_AUDIO_KOKORO_MODEL_ID,
        "default_voice_id": MLX_AUDIO_KOKORO_DEFAULT_VOICE,
        "available_voices": MLX_AUDIO_KOKORO_VOICES,
        "lang_code": MLX_AUDIO_KOKORO_LANG_CODE_GERMAN,
        "sample_rate": 24000,
        "notes": "mlx-audio (Kokoro model) for Apple Silicon. German lang_code needed. Voice may be non-German."
    },
    "mlx_audio_csm_clone": {
        # "handler_function_key": "mlx_audio", # REMOVED: ALL_HANDLERS key is "mlx_audio_csm_clone"
        "mlx_model_path": MLX_AUDIO_CSM_MODEL_ID,
        "default_voice_id": "./german_csm_reference.wav",
        "available_voices": ["./german_csm_reference.wav"],
        "lang_code": "de",
        "sample_rate": 24000,
        "notes": "mlx-audio (CSM model) for voice cloning on Apple Silicon. Requires German reference WAV."
    },
    "mlx_audio_outetts_q4": {
        # "handler_function_key": "mlx_audio", # REMOVED: ALL_HANDLERS key is "mlx_audio_outetts_q4"
        "mlx_model_path": MLX_AUDIO_OUTETTS_ONNX_REPO_ID,
        "onnx_filename_options_for_mlx": [
            "onnx/model_q4f16.onnx",
            "onnx/model_q4.onnx",
            "onnx/model_int8.onnx",
        ],
        "onnx_subfolder_for_mlx": "onnx",
        "tokenizer_path_for_mlx_outetts": "OuteAI/Llama-OuteTTS-1.0-1B",
        "lang_code": "de",
        "default_voice_id": "./german_outetts_reference.wav",
        "sample_rate": 24000,
        "notes": "OuteTTS Llama 1B ONNX (Q4 attempt) run via mlx-audio framework. Requires reference audio."
    },
    "speecht5_german_transformers": {
        # "handler_function_key": "speecht5", # REMOVED: ALL_HANDLERS key is "speecht5_german_transformers"
        "model_id": "sjdata/speecht5_finetuned_common_voice_11_de",
        "vocoder_id": "microsoft/speecht5_hifigan",
        "speaker_embeddings_repo": "Matthijs/cmu-arctic-xvectors",
        "default_speaker_embedding_index": 7306,
        "notes": "Local SpeechT5 (German fine-tune) via Transformers. Output: WAV."
    },
    "fastpitch_german_nemo": {
        # "handler_function_key": "nemo", # REMOVED: ALL_HANDLERS key is "fastpitch_german_nemo"
        "spectrogram_model_repo_id": "inOXcrm/German_multispeaker_FastPitch_nemo",
        "spectrogram_model_filename": "German_multispeaker_FastPitch_nemo.nemo",
        "vocoder_model_name": "tts_de_hui_hifigan_ft_fastpitch_multispeaker_5",
        "default_speaker_id": 0,
        "available_voices": [i for i in range(10)],
        "notes": "Local FastPitch (German) via NeMo. Output: WAV."
    },
    "coqui_tts_thorsten_ddc": {
        # "handler_function_key": "coqui_tts", # Already correct or redundant if model_id is key in ALL_HANDLERS
        "coqui_model_name": "tts_models/de/thorsten/tacotron2-DDC",
        "default_coqui_speaker": None,
        "language": "de",
        "sample_rate": 22050,
        "available_voices": ["default_speaker"], # ADDED to allow test loop to pick it up
        "notes": "Coqui TTS (Tacotron2 DDC) for German (Thorsten dataset). Uses 'TTS' library."
    },
    "coqui_tts_thorsten_vits": {
        # "handler_function_key": "coqui_tts",
        "coqui_model_name": "tts_models/de/thorsten/vits",
        "default_coqui_speaker": None,
        "language": "de",
        "sample_rate": 22050,
        "available_voices": ["default_speaker"], # ADDED
        "notes": "Coqui TTS (VITS model) for German (Thorsten dataset). Uses 'TTS' library."
    },
    "coqui_tts_thorsten_dca": {
        # "handler_function_key": "coqui_tts",
        "coqui_model_name": COQUI_THORSTEN_TACOTRON2_DCA,
        "default_coqui_speaker": None,
        "language": "de",
        "sample_rate": 22050,
        "available_voices": ["default_speaker"], # ADDED
        "notes": "Coqui TTS (Tacotron2-DCA) for German (Thorsten dataset). Uses 'TTS' library. May require espeak/gruut."
    },
}