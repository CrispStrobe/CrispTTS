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

DEFAULT_GERMAN_REF_WAV = "./german.wav"

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
        logger_cfg.debug("Successfully imported OuteTTS enums.")
    except ImportError:
        logger_cfg.info("OuteTTS library not found during config load. OuteTTS enums will be placeholders.")
    except AttributeError: # Catches error if protobufs for OuteTTS are not compiled
        logger_cfg.warning("OuteTTS import has AttributeError (likely protobuf issue). OuteTTS enums will be placeholders.")


# --- Global TTS Constants ---
ORPHEUS_SAMPLE_RATE = 24000

# Voice Lists
ORPHEUS_AVAILABLE_VOICES_BASE = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
ORPHEUS_GERMAN_VOICES = ["jana", "thomas", "max"]
SAUERKRAUT_VOICES = ["Tom", "Anna", "Max", "Lena"]
ORPHEUS_DEFAULT_VOICE = "jana" # General fallback for Orpheus-like models if not specified

EDGE_TTS_ALL_GERMAN_VOICES = [
    "de-AT-IngridNeural", "de-AT-JonasNeural", "de-DE-AmalaNeural", "de-DE-ConradNeural",
    "de-DE-KatjaNeural", "de-DE-KillianNeural", "de-CH-JanNeural", "de-CH-LeniNeural",
]
EDGE_TTS_DEFAULT_GERMAN_VOICE = "de-DE-KatjaNeural"

# --- Piper German Voices ---
PIPER_GERMAN_VOICE_PATHS = [
    "de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx",
    "de/de_DE/karlsson/low/de_DE-karlsson-low.onnx",
    "de/de_DE/kerstin/low/de_DE-kerstin-low.onnx",
    "de/de_DE/mls/medium/de_DE-mls-medium.onnx",
    "de/de_DE/pavoque/low/de_DE-pavoque-low.onnx",
    "de/de_DE/ramona/low/de_DE-ramona-low.onnx",
    "de/de_DE/thorsten/low/de_DE-thorsten-low.onnx",
    "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
    "de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
    "de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx",
]
PIPER_DEFAULT_GERMAN_VOICE_PATH = "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"

# --- Coqui Specific Constants ---
COQUI_THORSTEN_TACOTRON2_DCA = "tts_models/de/thorsten/tacotron2-DCA" # Used in your existing config

# API Endpoint Defaults
LM_STUDIO_API_URL_DEFAULT = "http://127.0.0.1:1234/v1/completions"
OLLAMA_API_URL_DEFAULT = "http://localhost:11434/api/generate"
LM_STUDIO_HEADERS = {"Content-Type": "application/json"}

# --- Kartoffel Orpheus (Transformers-based) Specific Configs ---
KARTORPHEUS_NATURAL_MODEL_ID = "SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1"
KARTORPHEUS_SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
KARTORPHEUS_NATURAL_SPEAKERS = [
    "Jakob", "Anton", "Julian", "Jan", "Alexander", "Emil", "Ben", 
    "Elias", "Felix", "Jonas", "Noah", "Maximilian", "Sophie", 
    "Marie", "Mia", "Maria", "Sophia", "Lina", "Lea"
]
KARTORPHEUS_NATURAL_DEFAULT_SPEAKER = "Julian"

KARTORPHEUS_PROMPT_START_TOKEN_ID = 128259
KARTORPHEUS_PROMPT_END_TOKEN_IDS = [128009, 128260]
KARTORPHEUS_GENERATION_EOS_TOKEN_ID = 128258
KARTORPHEUS_AUDIO_START_MARKER_TOKEN_ID = 128257
KARTORPHEUS_AUDIO_END_MARKER_TOKEN_ID = 128258
KARTORPHEUS_AUDIO_TOKEN_OFFSET = 128266

# --- Zonos (Zyphra) Specific Constants ---
ZONOS_TRANSFORMER_MODEL_ID = "Zyphra/Zonos-v0.1-transformer"
ZONOS_HYBRID_MODEL_ID = "Zyphra/Zonos-v0.1-hybrid"

# --- LLaSA Specific Configs ---
LLASA_MLX_LLM_MODEL_ID = "nhe-ai/Llasa-1B-Multilingual-mlx-4Bit"

LLASA_CHAT_TEMPLATE_TOKENIZER_ID = "HKUSTAudio/Llasa-1B-Multilingual" 

LLASA_XCODEC2_VOCODER_MODEL_ID = "HKUSTAudio/xcodec2"
LLASA_GERMAN_TRANSFORMERS_MODEL_ID = "MultiLlasa/Llasa-1B-Multilingual-German"
LLASA_MULTILINGUAL_HF_MODEL_ID = "HKUSTAudio/Llasa-1B-Multilingual"

LLASA_WHISPER_MODEL_ID_FOR_TRANSCRIPTION = "openai/whisper-large-v3-turbo" # Or your preferred Whisper model

MLX_AUDIO_KOKORO_MODEL_ID = "mlx-community/Kokoro-82M-bf16"
MLX_AUDIO_KOKORO_LANG_CODE = "en-us" # For espeak G2P
MLX_AUDIO_KOKORO_DEFAULT_VOICE = "af_heart" # Hypothetical, or use "af_heart" for testing
MLX_AUDIO_KOKORO_VOICES = ["af_heart", "af_nova", "bf_emma", ] 
# en-us: af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
# en-us: am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck
# en-gb: bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis

MLX_AUDIO_CSM_MODEL_ID = "mlx-community/csm-1b-8bit" # Sesame
MLX_AUDIO_OUTETTS_MAIN_REPO_ID = "mlx-community/Llama-OuteTTS-1.0-1B-4bit" # For mlx-audio OuteTTS handler
MLX_AUDIO_SPARK_REPO_ID = "mlx-community/Spark-TTS-0.5B-8bit"
MLX_AUDIO_BARK_REPO_ID = "mlx-community/bark-small" # or: mlx_bark, but needs config.json fix
MLX_AUDIO_DIA_REPO_ID = "mlx-community/Dia-1.6B-4bit"
MLX_AUDIO_ORPHEUS_LLAMA_REPO_ID = "mlx-community/orpheus-3b-0.1-ft-4bit" # mlx-audio's Llama variant

# This was for your OuteTTS HF ONNX handler, separate from mlx-audio OuteTTS
OUTETTS_HF_REPO_ID = "OuteAI/Llama-OuteTTS-1.0-1B-ONNX"

# --- TTS.cpp Specific Constants ---
# Assumes TTS.cpp was cloned and built inside the CrispTTS project directory.
# Adjust this path to wherever your TTS.cpp 'cli' executable is located.
TTS_CPP_EXECUTABLE_PATH = "./TTS.cpp/build/cli" 

# Example paths to GGUF models. User must download these separately.
# See TTS.cpp docs for links to GGUF files for each model.
TTS_CPP_KOKORO_GGUF_PATH = "./models/tts_cpp/kokoro-82m-f32.gguf"
TTS_CPP_DIA_GGUF_PATH = "./models/tts_cpp/dia-256-f32.gguf"

# Kokoro has specific voices
KOKORO_VOICES = [
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole',
    'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir',
    'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 'bf_emma',
    'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora',
    'em_alex', 'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara',
    'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora',
    'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi'
]
KOKORO_DEFAULT_VOICE = "bf_emma" # British English Female

# --- Kokoro-ONNX Specific Constants ---
# NOTE: User must manually download these files from the kokoro-onnx GitHub releases page.
# These paths are placeholders and must be updated.
KOKORO_ONNX_MODEL_PATH = "./models/kokoro_onnx/kokoro-v0_19.int8.onnx"
KOKORO_ONNX_VOICES_PATH = "./models/kokoro_onnx/voices-v1.0.bin"

KOKORO_ONNX_VOICES = ["af_sarah", "af_alloy", "bf_emma", "bm_daniel", "jf_gongitsune"] # Example voices
KOKORO_ONNX_DEFAULT_VOICE = "af_sarah"

# --- OuteTTS Internal Data Structures (Fallback if not directly queryable from library) ---
# (This section from your config.py is kept as is)
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
    "zonos_transformer_de": {
        "handler_function_key": "zonos",
        "model_repo_id": ZONOS_TRANSFORMER_MODEL_ID,
        # reference audio file for the voice
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV],
        "language": "de", # espeak language code for German
        "sample_rate": 44100, # Zonos native sample rate
        "requires_hf_token": False,
        "notes": "Zyphra Zonos v0.1 (Transformer). High-quality, requires PyTorch & espeak-ng. Voice cloning via reference WAV."
    },
    "edge": {
        "handler_function_key": "edge",
        "default_voice_id": EDGE_TTS_DEFAULT_GERMAN_VOICE,
        "available_voices": EDGE_TTS_ALL_GERMAN_VOICES,
        "notes": "MS Edge TTS (cloud). Output: MP3. Internet required."
    },
    "orpheus_kartoffel_natural": {
        "handler_function_key": "orpheus_kartoffel", # Standardized key
        "model_repo_id": KARTORPHEUS_NATURAL_MODEL_ID,
        "tokenizer_repo_id": KARTORPHEUS_NATURAL_MODEL_ID,
        "snac_model_id": KARTORPHEUS_SNAC_MODEL_ID,
        "default_voice_id": KARTORPHEUS_NATURAL_DEFAULT_SPEAKER,
        "available_voices": KARTORPHEUS_NATURAL_SPEAKERS,
        "language": "de",
        "sample_rate": ORPHEUS_SAMPLE_RATE,
        "requires_hf_token": True,
        "prompt_start_token_id": KARTORPHEUS_PROMPT_START_TOKEN_ID,
        "prompt_end_token_ids": KARTORPHEUS_PROMPT_END_TOKEN_IDS,
        "generation_eos_token_id": KARTORPHEUS_GENERATION_EOS_TOKEN_ID,
        "audio_start_marker_token_id": KARTORPHEUS_AUDIO_START_MARKER_TOKEN_ID,
        "audio_end_marker_token_id": KARTORPHEUS_AUDIO_END_MARKER_TOKEN_ID,
        "audio_token_offset": KARTORPHEUS_AUDIO_TOKEN_OFFSET,
        "notes": "Kartoffel Orpheus-3B (Natural German) via Transformers & SNAC. Gated model."
    },
    "piper_local": {
        "handler_function_key": "piper", # Standardized key
        "piper_voice_repo_id": "rhasspy/piper-voices",
        "default_model_path_in_repo": PIPER_DEFAULT_GERMAN_VOICE_PATH,
        "available_voices": PIPER_GERMAN_VOICE_PATHS,
        "requires_hf_token": False,
        "notes": "Local Piper TTS (ONNX). Downloads model/voice. Output: WAV."
    },
    "orpheus_lex_au": {
        "handler_function_key": "orpheus_gguf", # Standardized key for GGUF
        "model_repo_id": "lex-au/Orpheus-3b-FT-Q4_K_M.gguf",
        "model_filename": "Orpheus-3b-FT-Q4_K_M.gguf",
        "requires_hf_token": False,
        "default_voice_id": ORPHEUS_DEFAULT_VOICE,
        "available_voices": ORPHEUS_GERMAN_VOICES,
        "notes": "Local Orpheus GGUF (lex-au). Uses llama-cpp-python & custom decoder. Output: WAV."
    },
    "orpheus_sauerkraut": {
        "handler_function_key": "orpheus_gguf", # Standardized key for GGUF
        "model_repo_id": "VAGOsolutions/SauerkrautTTS-Preview-0.1-Q4_K_M-GGUF",
        "model_filename": "sauerkrauttts_preview_0_1.Q4_K_M.gguf",
        "requires_hf_token": False,
        "default_voice_id": "Tom",
        "available_voices": SAUERKRAUT_VOICES,
        "notes": "Local SauerkrautTTS GGUF. Uses llama-cpp-python & custom decoder. Output: WAV."
    },
    "orpheus_lm_studio": {
        "handler_function_key": "orpheus_lm_studio",
        "api_url": LM_STUDIO_API_URL_DEFAULT,
        "gguf_model_name_in_api": "SauerkrautTTS-Preview-0.1",
        "default_voice_id": "Tom",
        "available_voices": SAUERKRAUT_VOICES + ORPHEUS_GERMAN_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE,
        "notes": "Orpheus via LM Studio API. Requires custom decoder. Output: WAV."
    },
    "orpheus_ollama": {
        "handler_function_key": "orpheus_ollama",
        "api_url": OLLAMA_API_URL_DEFAULT,
        "ollama_model_name": "legraphista/Orpheus",
        "default_voice_id": "tara",
        "available_voices": ORPHEUS_AVAILABLE_VOICES_BASE,
        "notes": "Orpheus via Ollama API. Requires custom decoder. Output: WAV."
    },
    "oute_llamacpp": {
        "handler_function_key": "outetts", # Standardized key for OuteTTS
        "outetts_model_enum": OuteTTSModels_Enum.VERSION_1_0_SIZE_1B if OUTETTS_AVAILABLE_FOR_CONFIG else "VERSION_1_0_SIZE_1B_STR_FALLBACK",
        "backend_to_use": OuteTTSBackend_Enum.LLAMACPP if OUTETTS_AVAILABLE_FOR_CONFIG else "LLAMACPP_STR_FALLBACK",
        "quantization_to_use": OuteTTSLlamaCppQuantization_Enum.FP16 if OUTETTS_AVAILABLE_FOR_CONFIG else "FP16_STR_FALLBACK",
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV],
        "test_default_speakers": ["EN-FEMALE-1-NEUTRAL"],
        "notes": "Local OuteTTS (LlamaCPP backend). Custom WAV or OuteTTS default ID."
    },
    "oute_hf": {
        "handler_function_key": "outetts", # Standardized key for OuteTTS
        "outetts_model_version_str": "1.0",
        "onnx_repo_id": OUTETTS_HF_REPO_ID,
        "tokenizer_path": "OuteAI/Llama-OuteTTS-1.0-1B",
        "onnx_filename_options": ["onnx/model_int8.onnx"],
        "wavtokenizer_model_path": "onnx-community/WavTokenizer-large-speech-75token_decode",
        "interface_version_enum": OuteTTSInterfaceVersion_Enum.V3 if OUTETTS_AVAILABLE_FOR_CONFIG else "V3_STR_FALLBACK",
        "backend_to_use": OuteTTSBackend_Enum.HF if OUTETTS_AVAILABLE_FOR_CONFIG else "HF_STR_FALLBACK",
        "language": "de",
        "torch_dtype_for_hf_wrapper": torch.float16 if TORCH_AVAILABLE_FOR_CONFIG else "torch.float16_STR_FALLBACK",
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV],
        "test_default_speakers": ["EN-FEMALE-1-NEUTRAL"],
        "notes": "OuteTTS (HF ONNX). Ref WAV ideally <15s. Output: WAV."
    },
    "speecht5_german_transformers": {
        "handler_function_key": "speecht5", # Standardized key
        "model_id": "sjdata/speecht5_finetuned_common_voice_11_de",
        "vocoder_id": "microsoft/speecht5_hifigan",
        "speaker_embeddings_repo": "Matthijs/cmu-arctic-xvectors",
        "default_speaker_embedding_index": 7306,
        "available_voices": ["7306", "path/to/your/custom_xvector.pt"],
        "notes": "Local SpeechT5 (German fine-tune) via Transformers. Output: WAV."
    },
    "fastpitch_german_nemo": {
        "handler_function_key": "nemo_fastpitch", # Standardized key
        "spectrogram_model_repo_id": "inOXcrm/German_multispeaker_FastPitch_nemo",
        "spectrogram_model_filename": "German_multispeaker_FastPitch_nemo.nemo",
        "vocoder_model_name": "tts_de_hui_hifigan_ft_fastpitch_multispeaker_5",
        "default_speaker_id": 0,
        "available_voices": [str(i) for i in range(10)],
        "notes": "Local FastPitch (German) via NeMo. Output: WAV."
    },
    # --- Kokoro-ONNX Entry ---
    "kokoro_onnx": {
        "crisptts_model_id": "kokoro_onnx",
        "handler_function_key": "kokoro_onnx",
        "onnx_model_path": KOKORO_ONNX_MODEL_PATH,
        "voices_bin_path": KOKORO_ONNX_VOICES_PATH,
        "default_voice_id": KOKORO_ONNX_DEFAULT_VOICE,
        "available_voices": KOKORO_ONNX_VOICES,
        "language": "en-us",  # Language must be specified for kokoro-onnx
        "default_speed": 1.0,
        "sample_rate": 24000, # This will be overwritten by the sample_rate from the library
        "notes": "kokoro-onnx. Requires manual download of ONNX model and voices.bin file."
    },
    # --- TTS.cpp Model Entries ---
    "tts_cpp_kokoro": {
        "crisptts_model_id": "tts_cpp_kokoro",
        "handler_function_key": "tts_cpp",
        "tts_cpp_executable_path": TTS_CPP_EXECUTABLE_PATH,
        "gguf_model_path": TTS_CPP_KOKORO_GGUF_PATH,
        "default_voice_id": KOKORO_DEFAULT_VOICE,
        "available_voices": KOKORO_VOICES,
        "use_metal": True, # Metal acceleration is supported for Kokoro
        "sample_rate": 24000, # Kokoro sample rate
        "notes": "TTS.cpp (Kokoro). Uses local GGUF model. Requires building TTS.cpp locally."
    },
    "tts_cpp_dia": {
        "crisptts_model_id": "tts_cpp_dia",
        "handler_function_key": "tts_cpp",
        "tts_cpp_executable_path": TTS_CPP_EXECUTABLE_PATH,
        "gguf_model_path": TTS_CPP_DIA_GGUF_PATH,
        "default_voice_id": None, # Dia does not use named voices
        "available_voices": [],
        "use_metal": True, # Metal acceleration is supported for Dia
        "sample_rate": 44100, # Dia sample rate
        "notes": "TTS.cpp (Dia). Uses local GGUF. Recommended params: '{\"temperature\": 1.3, \"topk\": 35}'."
    },
    "coqui_tts_thorsten_ddc": {
        "handler_function_key": "coqui_tts", # Standardized Coqui key
        "coqui_model_name": "tts_models/de/thorsten/tacotron2-DDC",
        "default_coqui_speaker": None,
        "language": "de",
        "sample_rate": 22050,
        "available_voices": ["default_speaker"],
        "notes": "Coqui TTS (Tacotron2 DDC) for German. Uses 'TTS' library."
    },
    "coqui_tts_thorsten_vits": {
        "handler_function_key": "coqui_tts",
        "coqui_model_name": "tts_models/de/thorsten/vits",
        "default_coqui_speaker": None,
        "language": "de",
        "sample_rate": 22050,
        "available_voices": ["default_speaker"],
        "notes": "Coqui TTS (VITS model) for German. Uses 'TTS' library."
    },
    "coqui_tts_thorsten_dca": {
        "handler_function_key": "coqui_tts",
        "coqui_model_name": COQUI_THORSTEN_TACOTRON2_DCA,
        "default_coqui_speaker": None,
        "language": "de",
        "sample_rate": 22050,
        "available_voices": ["default_speaker"],
        "notes": "Coqui TTS (Tacotron2-DCA) for German. Uses 'TTS' library. May require espeak/gruut."
    },
    "coqui_xtts_v2_de_clone": {
        "handler_function_key": "coqui_tts", # XTTS also uses the Coqui TTS API
        "coqui_model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "default_coqui_speaker": None, 
        "language": "de",      
        "sample_rate": 24000,  
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV], 
        "notes": "Coqui XTTS v2 (multilingual, 24kHz). Requires a speaker_wav for voice cloning. Ref WAV ~6-20s."
    },
    "coqui_css10_de_vits": {
        "handler_function_key": "coqui_tts",
        "coqui_model_name": "tts_models/de/css10/vits-neon", 
        "default_coqui_speaker": None,
        "language": "de",
        "sample_rate": 22050, # VITS typical sample rate
        "available_voices": ["default_speaker"], # Placeholder for single-speaker
        "notes": "Coqui TTS: German CSS10 VITS model (single-speaker, Neon variant)."
    },
    "coqui_vctk_en_vits": {
        "handler_function_key": "coqui_tts",
        "coqui_model_name": "tts_models/en/vctk/vits",
        "default_coqui_speaker": "p225", 
        "available_voices": ["p225", "p228", "p232", "p249"],
        "language": "en",
        "sample_rate": 22050, 
        "notes": "Coqui TTS: English VCTK VITS (multi-speaker). Use VCTK speaker IDs."
    },
    "llasa_hybrid_de_clone": {
        "handler_function_key": "llasa_hybrid",
        "llm_model_id": LLASA_MLX_LLM_MODEL_ID,
        "chat_tokenizer_id": LLASA_CHAT_TEMPLATE_TOKENIZER_ID, 
        "codec_model_id": LLASA_XCODEC2_VOCODER_MODEL_ID,
        "language": "de",
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV], 
        "sample_rate": 16000,
        "requires_hf_token": False, 
        "notes": "LLaSA 1B Hybrid (MLX LLM + PyTorch XCodec2). Uses ref WAV for cloning. 16kHz. Experimental."
    },
    "llasa_hybrid_de_zeroshot": { 
        "handler_function_key": "llasa_hybrid",
        "llm_model_id": LLASA_MLX_LLM_MODEL_ID, # e.g., "nhe-ai/Llasa-1B-Multilingual-mlx-4Bit"
        "chat_tokenizer_id": LLASA_CHAT_TEMPLATE_TOKENIZER_ID, # e.g., "HKUSTAudio/Llasa-1B-Multilingual"
        "codec_model_id": LLASA_XCODEC2_VOCODER_MODEL_ID, # PyTorch-based XCodec2
        "language": "de", # Or None if truly multilingual and language is in text
        "default_voice_id": None, # Indicates zero-shot
        "available_voices": [],
        "sample_rate": 16000,
        "requires_hf_token": False,
        "notes": "LLaSA Hybrid (MLX LLM + PyTorch XCodec2). Zero-shot German TTS. 16kHz."
    },
    "llasa_german_transformers_clone": {
        "crisptts_model_id": "llasa_german_transformers_clone",
        "handler_function_key": "llasa_german_transformers",  # NEW: Use German-specific handler
        "llm_model_id": LLASA_GERMAN_TRANSFORMERS_MODEL_ID,
        "tokenizer_id": LLASA_GERMAN_TRANSFORMERS_MODEL_ID,
        "codec_model_id": LLASA_XCODEC2_VOCODER_MODEL_ID,
        "whisper_model_id_for_transcription": LLASA_WHISPER_MODEL_ID_FOR_TRANSCRIPTION,
        "language": "de",
        "default_voice_id": DEFAULT_GERMAN_REF_WAV, # e.g., "./german.wav"
        "available_voices": [DEFAULT_GERMAN_REF_WAV],
        "sample_rate": 16000,
        "requires_hf_token": False,
        "notes": "LLaSA German (German-specific handler + XCodec2). Voice cloning via reference WAV. Output: 16kHz WAV."
    },
    "llasa_german_transformers_zeroshot": {
        "crisptts_model_id": "llasa_german_transformers_zeroshot",
        "handler_function_key": "llasa_german_transformers",  # NEW: Use German-specific handler
        "llm_model_id": LLASA_GERMAN_TRANSFORMERS_MODEL_ID,
        "tokenizer_id": LLASA_GERMAN_TRANSFORMERS_MODEL_ID,
        "codec_model_id": LLASA_XCODEC2_VOCODER_MODEL_ID,
        "language": "de",
        "default_voice_id": None,
        "available_voices": [],
        "sample_rate": 16000,
        "requires_hf_token": False,
        "notes": "LLaSA German (German-specific handler + XCodec2). Zero-shot TTS. Output: 16kHz WAV."
    },
    "llasa_multilingual_hf_clone": {
        "crisptts_model_id": "llasa_multilingual_hf_clone",
        "handler_function_key": "llasa_multilingual_transformers",  # NEW: Use Multilingual-specific handler
        "llm_model_id": LLASA_MULTILINGUAL_HF_MODEL_ID,
        "tokenizer_id": LLASA_MULTILINGUAL_HF_MODEL_ID,
        "codec_model_id": LLASA_XCODEC2_VOCODER_MODEL_ID,
        "language": None, 
        "default_voice_id": "./german.wav",  # Changed from "./my_bark_output.wav"
        "available_voices": ["./german.wav"], 
        "sample_rate": 16000,
        "requires_hf_token": False,
        "notes": "LLaSA Multilingual (Multilingual-specific handler + XCodec2). Voice cloning via reference WAV. Output: 16kHz WAV."
    },
    "llasa_multilingual_hf_zeroshot": {
        "crisptts_model_id": "llasa_multilingual_hf_zeroshot",
        "handler_function_key": "llasa_multilingual_transformers",  # NEW: Use Multilingual-specific handler
        "llm_model_id": LLASA_MULTILINGUAL_HF_MODEL_ID,
        "tokenizer_id": LLASA_MULTILINGUAL_HF_MODEL_ID,
        "codec_model_id": LLASA_XCODEC2_VOCODER_MODEL_ID,
        "language": None,
        "default_voice_id": None,
        "available_voices": [],
        "sample_rate": 16000,
        "requires_hf_token": False,
        "notes": "LLaSA Multilingual (Multilingual-specific handler + XCodec2). Zero-shot TTS. Output: 16kHz WAV."
    },

    # --- MLX-AUDIO MODEL ENTRIES using "mlx_audio" handler_function_key ---
    "mlx_audio_kokoro_de": { # Your original ID for Kokoro
        "handler_function_key": "mlx_audio",
        "mlx_model_path": MLX_AUDIO_KOKORO_MODEL_ID,
        "default_voice_id": MLX_AUDIO_KOKORO_DEFAULT_VOICE,
        "available_voices": MLX_AUDIO_KOKORO_VOICES,
        "lang_code": MLX_AUDIO_KOKORO_LANG_CODE,
        "sample_rate": 24000,
        "notes": "mlx-audio (Kokoro model) for Apple Silicon. Lang: German ('de' with espeak). Uses Apple Silicon MLX."
    },
    "mlx_audio_csm_clone": { # Your existing ID for CSM/Sesame
        "handler_function_key": "mlx_audio",
        "mlx_model_path": MLX_AUDIO_CSM_MODEL_ID,
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV],
        # "lang_code": "de", # Optional for CSM, often inferred
        "sample_rate": 24000,
        "notes": "mlx-audio (CSM/Sesame) for voice cloning. Ref WAV & ref_text needed. Uses Apple Silicon MLX."
    },
    "mlx_audio_outetts_clone": { # Your existing ID for OuteTTS via mlx-audio
        "handler_function_key": "mlx_audio",
        "mlx_model_path": MLX_AUDIO_OUTETTS_MAIN_REPO_ID,
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV],
        "sample_rate": 24000,
        "notes": "mlx-audio (OuteTTS Llama) for voice cloning. Ref WAV & ref_text needed. Uses Apple Silicon MLX."
    },
    # --- NEW MLX-AUDIO ENTRIES with standardized IDs ---
    "mlx_audio_spark_clone": {
        "handler_function_key": "mlx_audio",
        "mlx_model_path": MLX_AUDIO_SPARK_REPO_ID,
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV],
        "sample_rate": 16000,
        "notes": "mlx-audio (Spark model) for voice cloning. Ref WAV & ref_text needed. Uses Apple Silicon MLX."
    },
    "mlx_audio_spark_control": {
        "handler_function_key": "mlx_audio",
        "mlx_model_path": MLX_AUDIO_SPARK_REPO_ID,
        "default_voice_id": "female", 
        "available_voices": ["female", "male"],
        "default_speed": 1.0, 
        "default_pitch": 1.0, 
        "sample_rate": 16000,
        "notes": "mlx-audio (Spark model) with controllable attributes. Uses Apple Silicon MLX."
    },
    "mlx_audio_bark_de": {
        "handler_function_key": "mlx_audio",
        "mlx_model_path": "mlx-community/bark-small", # For the main MLX model
        "default_voice_id": "v2/de_speaker_3",        # This voice will be fetched from "suno/bark-small" by the patch
        "available_voices": [
            "v2/de_speaker_0", "v2/de_speaker_1", "v2/de_speaker_2", 
            "v2/de_speaker_3", "v2/de_speaker_4", "v2/de_speaker_5", 
            "v2/de_speaker_6", "v2/de_speaker_7", "v2/de_speaker_8", 
            "v2/de_speaker_9", "v2/en_speaker_1" # Example
        ],
        "lang_code": "de", 
        "sample_rate": 24000,
        "notes": "mlx-audio (Bark) with main model from mlx-community/bark-small and voices from suno/bark-small (via patch)."
    },
    # --- F5-TTS entries, experimentally including non-standard repos ---
    "f5_tts_multilingual": {
        "handler_function_key": "f5_tts",
        "model_repo_id": "cstr/aihpi_f5_german_mlx_q4",  # lucasnewman/f5-tts-mlx # This is a working MLX model
        "use_mlx": True,  # This model is compatible with MLX
        "language": "multilingual",
        "default_voice_id": "./german.wav",
        "available_voices": ["./german.wav"],
        "sample_rate": 24000, 
        "default_steps": 32,
        "default_cfg_strength": 2.0,
        "requires_hf_token": False,
        "notes": "F5-TTS MLX multilingual model. Works reliably with f5-tts-mlx library."
    },
    "f5_tts_german": {
        "handler_function_key": "f5_tts",
        "model_repo_id": "cstr/aihpi_f5_german_mlx_q4",  # lucasnewman/f5-tts-mlx # Use the working repository
        "use_mlx": True,
        "language": "de",
        "default_voice_id": "./german.wav",
        "available_voices": ["./german.wav"],
        "sample_rate": 24000,
        "default_steps": 32,
        "default_cfg_strength": 2.0,
        "requires_hf_token": False,
        "notes": "F5-TTS MLX using the main working repository. Reliable German TTS with voice cloning."
    },
    "f5_tts_german_hpi": {
        "handler_function_key": "f5_tts",
        "model_repo_id": "aihpi/F5-TTS-German",
        "use_mlx": False,  # Force PyTorch backend
        "language": "de",
        # Specify the exact checkpoint file within the repo's subdirectory
        "checkpoint_filename": "F5TTS_Base/model_420000.safetensors", 
        "default_voice_id": "./german.wav",
        "available_voices": ["./german.wav"],
        "sample_rate": 24000,
        "default_steps": 32,
        "default_cfg_strength": 2.0,
        "requires_hf_token": False,
        "notes": "F5-TTS German by HPI. Uses PyTorch backend with a specific checkpoint file."
    },
    "f5_tts_german_marduk": {
        "handler_function_key": "f5_tts",
        "model_repo_id": "marduk-ra/F5-TTS-German",
        "use_mlx": False, # Force PyTorch backend, as MLX fails
        "language": "de",
        # Specify the exact checkpoint file at the repo root
        "checkpoint_filename": "f5_tts_german_1010000.safetensors",
        "default_voice_id": "./german.wav",
        "available_voices": ["./german.wav"],
        "sample_rate": 24000,
        "default_steps": 64,
        "default_cfg_strength": 2.0,
        "requires_hf_token": False,
        "notes": "EXPERIMENTAL: F5-TTS German by marduk-ra. Uses PyTorch backend with a specific checkpoint file."
    },
    "f5_tts_german_eamag": {
        "handler_function_key": "f5_tts",
        "model_repo_id": "eamag/f5-tts-mlx-german",
        "use_mlx": True, # Let this attempt MLX first, so the fallback can be triggered
        "language": "de", 
        "default_voice_id": "./german.wav",
        "available_voices": ["./german.wav"],
        "sample_rate": 24000,
        "default_steps": 32,
        "default_cfg_strength": 2.0,
        "requires_hf_token": False,
        "notes": "EXPERIMENTAL: F5-TTS German MLX by eamag. Known to have architecture incompatibility. Will attempt MLX and should fall back to PyTorch."
    },
    # --- mlx audio endpoint test ---
    "mlx_audio_orpheus_llama": { # For the Orpheus-Llama in mlx-audio
        "handler_function_key": "mlx_audio",
        "mlx_model_path": MLX_AUDIO_ORPHEUS_LLAMA_REPO_ID,
        "default_voice_id": "zac", # Orpheus-style voice name
        "available_voices": ORPHEUS_AVAILABLE_VOICES_BASE + ORPHEUS_GERMAN_VOICES,
        "sample_rate": ORPHEUS_SAMPLE_RATE,
        "notes": "mlx-audio (Orpheus-Llama). Uses string voice names. Uses Apple Silicon MLX."
    },
    # --- experimental (not yet working) models ---
    "mlx_audio_dia_clone": {
        "crisptts_model_id": "mlx_audio_dia_clone", # Add this for better logging
        "handler_function_key": "mlx_audio",
        "mlx_model_path": MLX_AUDIO_DIA_REPO_ID,
        "default_voice_id": DEFAULT_GERMAN_REF_WAV,
        "available_voices": [DEFAULT_GERMAN_REF_WAV],
        "whisper_model_id_for_transcription": LLASA_WHISPER_MODEL_ID_FOR_TRANSCRIPTION,
        "language_for_whisper": "de",
        "sample_rate": 44100,
        # "ref_audio_max_duration_ms": 10000, # Optional: to enforce shorter ref for Dia
        "notes": "mlx-audio (Dia model) for voice cloning..."
    },
    
}