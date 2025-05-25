# CrispTTS/handlers/mlx_audio_handler.py
import logging
import platform
import os
from pathlib import Path
import json
import shutil
import tempfile
import gc
import numpy as np

# CrispTTS utils (ensure SuppressOutput is still safe to import here)
from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.mlx_audio")

# --- Conditional Imports for mlx-audio and pydub ---
MLX_AUDIO_AVAILABLE = False
generate_audio_mlx_func = None
mx_core_module = None

PYDUB_AVAILABLE_FOR_TRIM = False
AudioSegment_pydub = None
pydub_mediainfo_func = None # For audio format detection if needed by transcription

# --- Conditional Imports FOR WHISPER TRANSCRIPTION ---
TORCH_FOR_WHISPER_AVAILABLE = False
torch_whisper_module = None
TORCHAUDIO_FOR_WHISPER_AVAILABLE = False
torchaudio_whisper_module = None
TRANSFORMERS_PIPELINE_FOR_WHISPER_AVAILABLE = False
transformers_pipeline_whisper_func = None


logger_init = logging.getLogger("CrispTTS.handlers.mlx_audio.init")

try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo # Optional for transcription pre-processing
    AudioSegment_pydub = AudioSegment
    pydub_mediainfo_func = mediainfo
    PYDUB_AVAILABLE_FOR_TRIM = True # Also indicates pydub is generally available
    logger_init.info("pydub and pydub.utils.mediainfo imported for MLX-Audio handler.")
except ImportError:
    logger_init.info("pydub not found. Reference audio trimming/conversion for MLX-Audio will be limited.")

try:
    from mlx_audio.tts.generate import generate_audio as generate_audio_mlx_imp
    import mlx.core as mx_core_imported

    generate_audio_mlx_func = generate_audio_mlx_imp
    mx_core_module = mx_core_imported
    MLX_AUDIO_AVAILABLE = True

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        logger_init.info("mlx-audio library and mlx.core loaded (Apple Silicon detected).")
    else:
        logger_init.warning("mlx-audio library loaded, but non-Apple Silicon detected. MLX ops will run on CPU.")

except ImportError:
    logger_init.warning("mlx-audio library not found or failed to import. MLX Audio handler will be non-functional.")
except Exception as e:
    logger_init.error(f"An error occurred during mlx-audio or mlx.core import: {e}. Handler non-functional.", exc_info=True)

# Whisper-related imports for this handler's transcription capability
try:
    import torch as torch_for_whisper_imp
    torch_whisper_module = torch_for_whisper_imp
    TORCH_FOR_WHISPER_AVAILABLE = True
    logger_init.info("PyTorch (for Whisper) loaded in mlx_audio_handler.")
except ImportError:
    logger_init.warning("PyTorch (for Whisper) not found. Transcription of ref_audio will not be available in mlx_audio_handler.")

if TORCH_FOR_WHISPER_AVAILABLE:
    try:
        import torchaudio as torchaudio_for_whisper_imp
        torchaudio_whisper_module = torchaudio_for_whisper_imp
        TORCHAUDIO_FOR_WHISPER_AVAILABLE = True
        logger_init.info("Torchaudio (for Whisper) loaded in mlx_audio_handler.")
    except ImportError:
        logger_init.warning("Torchaudio (for Whisper) not found. ref_audio loading/resampling for transcription might fail.")

    try:
        from transformers import pipeline as pipeline_hf_func
        transformers_pipeline_whisper_func = pipeline_hf_func
        TRANSFORMERS_PIPELINE_FOR_WHISPER_AVAILABLE = True
        logger_init.info("Transformers pipeline (for Whisper) loaded in mlx_audio_handler.")
    except ImportError:
        logger_init.warning("Transformers pipeline (for Whisper) not found. Transcription of ref_audio will not be available.")


# --- Monkey Patch Section for mlx_audio.tts.models.bark.pipeline._load_voice_prompt ---
_original_load_voice_prompt = None
_mlx_bark_patch_applied_flag = False
BARK_VOICE_PROMPT_REPO_ID = "suno/bark-small"
PATCH_IMPORTS_SUCCESSFUL = False
try:
    import mlx_audio.tts.models.bark.pipeline as bark_pipeline_module_for_patch
    from huggingface_hub import hf_hub_download
    import numpy as np_for_patch # Keep aliased for patch
    import os as os_for_patch # Aliased os for the patch
    PATCH_IMPORTS_SUCCESSFUL = True
    logger_init.debug("Successfully imported modules for mlx-audio Bark patch.")
except ImportError as e_patch_imp:
    logger_init.warning(f"Failed to import modules needed for mlx-audio Bark patch: {e_patch_imp}. Patch will not be applied if Bark is used.")

def _patched_load_voice_prompt(voice_prompt_input):
    # (Your existing _patched_load_voice_prompt function - assumed correct based on previous context)
    # This function uses os_for_patch, np_for_patch, hf_hub_download, bark_pipeline_module_for_patch
    logger.debug(f"[Patched _load_voice_prompt] Called. Voice prompts from '{BARK_VOICE_PROMPT_REPO_ID}'.")
    if isinstance(voice_prompt_input, str) and voice_prompt_input.endswith(".npz"):
        if os_for_patch.path.exists(voice_prompt_input):
            logger.info(f"[Patched] Loading direct .npz: {voice_prompt_input}")
            try: return np_for_patch.load(voice_prompt_input)
            except Exception as e: raise ValueError(f"Could not load .npz: {voice_prompt_input}") from e
        else:
            try:
                logger.info(f"[Patched] Downloading .npz '{voice_prompt_input}' from '{BARK_VOICE_PROMPT_REPO_ID}'.")
                cached_path = hf_hub_download(repo_id=BARK_VOICE_PROMPT_REPO_ID, filename=voice_prompt_input)
                return np_for_patch.load(cached_path)
            except Exception as e: raise ValueError(f"Failed to load/download .npz: {voice_prompt_input}") from e
    elif isinstance(voice_prompt_input, dict):
        assert all(k in voice_prompt_input for k in ["semantic_prompt", "coarse_prompt", "fine_prompt"])
        return voice_prompt_input
    elif isinstance(voice_prompt_input, str):
        normalized_voice_name = os_for_patch.path.join(*voice_prompt_input.split(os_for_patch.path.sep))
        if hasattr(bark_pipeline_module_for_patch, 'ALLOWED_PROMPTS') and \
           normalized_voice_name not in bark_pipeline_module_for_patch.ALLOWED_PROMPTS:
            logger.warning(f"[Patched] Voice '{normalized_voice_name}' not in ALLOWED_PROMPTS. Using NPY load from '{BARK_VOICE_PROMPT_REPO_ID}'.")
        
        base_stem = ""
        relative_dir = "speaker_embeddings"
        if normalized_voice_name == "announcer": base_stem = "announcer"
        elif normalized_voice_name.startswith(f"v2{os_for_patch.path.sep}"):
            base_stem = normalized_voice_name.split(os_for_patch.path.sep, 1)[1]
            relative_dir = os_for_patch.path.join(relative_dir, "v2")
        else: base_stem = normalized_voice_name
        
        prompts_dict = {}
        try:
            for key, suffix in {"semantic_prompt": "semantic_prompt.npy", "coarse_prompt": "coarse_prompt.npy", "fine_prompt": "fine_prompt.npy"}.items():
                filename = f"{base_stem}_{suffix}" if base_stem else suffix
                path_in_repo = os_for_patch.path.join(relative_dir, filename)
                logger.info(f"[Patched] Downloading NPY: '{path_in_repo}' from '{BARK_VOICE_PROMPT_REPO_ID}' for '{key}'")
                cached_path = hf_hub_download(repo_id=BARK_VOICE_PROMPT_REPO_ID, filename=path_in_repo)
                prompts_dict[key] = np_for_patch.load(cached_path)
            if not all(k in prompts_dict for k in ["semantic_prompt", "coarse_prompt", "fine_prompt"]):
                raise ValueError("Missing NPY components.")
            return prompts_dict
        except Exception as e: raise ValueError(f"Error processing NPY prompts for '{voice_prompt_input}': {e}") from e
    raise ValueError(f"Voice prompt format unrecognized: {type(voice_prompt_input)}")


def apply_mlx_audio_bark_pipeline_patch_if_needed():
    global _mlx_bark_patch_applied_flag, _original_load_voice_prompt
    if not PATCH_IMPORTS_SUCCESSFUL:
        logger.warning("[Patch] Cannot apply Bark patch, essential modules failed import (hf_hub_download, np, os, or bark_pipeline_module).")
        return
    if not _mlx_bark_patch_applied_flag:
        try:
            if bark_pipeline_module_for_patch and hasattr(bark_pipeline_module_for_patch, '_load_voice_prompt'):
                if bark_pipeline_module_for_patch._load_voice_prompt is _patched_load_voice_prompt:
                    logger.debug("Bark patch already applied.")
                    _mlx_bark_patch_applied_flag = True; return
                logger.info(f"Applying patch to mlx_audio Bark voice loading. Prompts from: {BARK_VOICE_PROMPT_REPO_ID}")
                _original_load_voice_prompt = bark_pipeline_module_for_patch._load_voice_prompt
                bark_pipeline_module_for_patch._load_voice_prompt = _patched_load_voice_prompt
                _mlx_bark_patch_applied_flag = True
            else: logger.warning("[Patch] Could not apply Bark patch: _load_voice_prompt not found in mlx_audio.tts.models.bark.pipeline.")
        except Exception as e: logger.error(f"[Patch] Error applying Bark patch: {e}", exc_info=True)
# --- End Monkey Patch Section ---

def _detect_audio_format_pydub_inline(audio_path_str: str) -> str | None:
    """Detects audio format using pydub. Fallback to extension if pydub fails."""
    if not PYDUB_AVAILABLE_FOR_TRIM or not pydub_mediainfo_func: # PYDUB_AVAILABLE_FOR_TRIM implies AudioSegment is there
        logger.warning("mlx-audio: pydub.utils.mediainfo not available for format detection.")
        ext = Path(audio_path_str).suffix.lower().lstrip('.')
        return ext if ext else None # Return extension or None if no extension
    try:
        info = pydub_mediainfo_func(audio_path_str)
        format_name = info.get('format_name', '')
        # Prioritize common formats from format_name
        if any(fmt in format_name for fmt in ['mp4', 'm4a']): return 'mp4' # Covers m4a, mov etc. reported as mp4
        if 'mp3' in format_name: return 'mp3'
        if 'ogg' in format_name: return 'ogg' # Covers opus in ogg
        if 'flac' in format_name: return 'flac'
        if 'wav' in format_name: return 'wav'
        if 'webm' in format_name: return 'webm' # Covers opus in webm
        # Fallback to the first part of format_name or file extension
        return format_name.split(',')[0].strip() if format_name else Path(audio_path_str).suffix.lower().lstrip('.')
    except Exception as e:
        logger.error(f"mlx-audio: pydub mediainfo error for {audio_path_str}: {e}. Falling back to extension.", exc_info=False)
        ext = Path(audio_path_str).suffix.lower().lstrip('.')
        return ext if ext else None

def _convert_audio_for_whisper_inline(audio_path_str: str, target_sr: int = 16000) -> tuple[Path | None, str | None]:
    """Converts audio to 16kHz mono WAV for Whisper, returns Path to temp WAV or None and error string."""
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        return None, f"Audio file not found: {audio_path_str}"
    
    temp_wav_file_obj = None # To ensure it's always cleaned up
    try:
        # Create a named temporary file that we can pass by path
        with tempfile.NamedTemporaryFile(suffix='.wav', prefix="crisptts_whisper_prep_", delete=False) as tmp_f:
            temp_wav_path_str = tmp_f.name
        temp_wav_file_obj = Path(temp_wav_path_str)

        if PYDUB_AVAILABLE_FOR_TRIM and AudioSegment_pydub: # Use pydub if available
            logger.debug(f"mlx-audio: Converting '{audio_path}' for Whisper using pydub.")
            audio_format = _detect_audio_format_pydub_inline(str(audio_path)) or audio_path.suffix.lstrip(".").lower()
            if not audio_format:
                raise ValueError(f"Could not determine audio format for '{audio_path_str}' using pydub or extension.")
            
            sound = AudioSegment_pydub.from_file(str(audio_path), format=audio_format)
            sound = sound.set_channels(1).set_frame_rate(target_sr).set_sample_width(2) # 16-bit PCM
            sound.export(str(temp_wav_file_obj), format="wav")
        elif TORCHAUDIO_FOR_WHISPER_AVAILABLE and torchaudio_whisper_module and torch_whisper_module: # Fallback to torchaudio
            logger.debug(f"mlx-audio: Converting '{audio_path}' for Whisper using torchaudio (pydub not preferred/available).")
            waveform, sr = torchaudio_whisper_module.load(str(audio_path))
            if sr != target_sr:
                resampler = torchaudio_whisper_module.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
            if waveform.size(0) > 1: # Ensure mono
                waveform = torch_whisper_module.mean(waveform, dim=0, keepdim=True)
            torchaudio_whisper_module.save(str(temp_wav_file_obj), waveform, sample_rate=target_sr, bits_per_sample=16)
        else:
            if temp_wav_file_obj and temp_wav_file_obj.exists(): temp_wav_file_obj.unlink(missing_ok=True) # Clean up before error
            return None, "Neither pydub nor torchaudio is available for audio conversion for Whisper."
        
        logger.info(f"mlx-audio: Successfully converted '{audio_path_str}' to temporary WAV for Whisper: {temp_wav_file_obj}")
        return temp_wav_file_obj, None

    except Exception as e:
        logger.error(f"mlx-audio: Audio conversion for Whisper failed for '{audio_path_str}': {e}", exc_info=True)
        if temp_wav_file_obj and temp_wav_file_obj.exists(): # Ensure cleanup on error
            try: temp_wav_file_obj.unlink(missing_ok=True)
            except OSError: pass
        return None, str(e)

def _transcribe_ref_audio_for_mlx(
    audio_path_str: str,
    whisper_model_id: str,
    hf_token: str | None,
    target_device_str: str, # "cpu", "cuda", "mps"
    language_hint: str | None = None
) -> tuple[str | None, str | None]:
    """
    Transcribes reference audio using a Whisper pipeline.
    Returns (transcribed_text, error_message). error_message is None on success.
    """
    if not all([TORCH_FOR_WHISPER_AVAILABLE, TORCHAUDIO_FOR_WHISPER_AVAILABLE, TRANSFORMERS_PIPELINE_FOR_WHISPER_AVAILABLE]):
        return None, "Whisper dependencies (torch, torchaudio, transformers pipeline) not available in mlx_audio_handler."

    if not Path(audio_path_str).exists():
        return None, f"Reference audio for transcription not found: {audio_path_str}"

    logger.info(f"mlx-audio: Preparing to transcribe '{audio_path_str}' with Whisper model '{whisper_model_id}'. Target device: {target_device_str}, Lang hint: {language_hint}")

    temp_wav_for_whisper, conversion_error = _convert_audio_for_whisper_inline(audio_path_str, target_sr=16000)
    if conversion_error or not temp_wav_for_whisper:
        return None, f"Audio conversion for Whisper failed: {conversion_error}"

    whisper_pipe_instance = None
    transcribed_text = None
    error_msg = None

    try:
        pipeline_device_arg = target_device_str
        if target_device_str == "cuda": pipeline_device_arg = 0 # transformers pipeline convention

        logger.debug(f"mlx-audio: Initializing Whisper pipeline '{whisper_model_id}' on device arg: '{pipeline_device_arg}'")
        whisper_pipe_instance = transformers_pipeline_whisper_func(
            task="automatic-speech-recognition",
            model=whisper_model_id,
            torch_dtype=torch_whisper_module.float16 if target_device_str != "cpu" else torch_whisper_module.float32,
            device=pipeline_device_arg,
            token=hf_token,
            framework="pt"
        )
        
        generate_whisper_kwargs = {"return_timestamps": True} # For robust long-form transcription
        if language_hint:
            generate_whisper_kwargs["language"] = language_hint.lower() # Whisper expects lang codes e.g. "en", "de"

        logger.debug(f"mlx-audio: Calling Whisper pipeline with generate_kwargs: {generate_whisper_kwargs} on temp file: {temp_wav_for_whisper}")
        with SuppressOutput(suppress_stderr=True, suppress_stdout=not logger.isEnabledFor(logging.DEBUG)):
            transcription_result = whisper_pipe_instance(str(temp_wav_for_whisper), generate_kwargs=generate_whisper_kwargs)
        
        full_text = transcription_result.get("text", "").strip()
        if not full_text and "chunks" in transcription_result and isinstance(transcription_result["chunks"], list):
            logger.debug("mlx-audio: Stitching text from Whisper chunks as top-level 'text' is empty.")
            full_text = " ".join([chunk.get('text',"").strip() for chunk in transcription_result["chunks"]]).strip()
        
        if full_text:
            transcribed_text = full_text
            logger.info(f"mlx-audio: Whisper transcription successful (first 100 chars): '{transcribed_text[:100]}...'")
        else:
            error_msg = "Whisper returned empty transcription."
            logger.warning(f"mlx-audio: {error_msg}")
            
    except Exception as e_wspr:
        error_msg = f"Whisper transcription pipeline failed: {e_wspr}"
        logger.error(f"mlx-audio: {error_msg}", exc_info=True)
    finally:
        if temp_wav_for_whisper and temp_wav_for_whisper.exists():
            try: temp_wav_for_whisper.unlink()
            except OSError as e_del: logger.warning(f"mlx-audio: Could not delete temp Whisper WAV {temp_wav_for_whisper}: {e_del}")
        del whisper_pipe_instance
        if TORCH_FOR_WHISPER_AVAILABLE and torch_whisper_module and target_device_str != "cpu":
            if torch_whisper_module.cuda.is_available(): torch_whisper_module.cuda.empty_cache()
            elif hasattr(torch_whisper_module.backends, "mps") and torch_whisper_module.backends.mps.is_available() and hasattr(torch_whisper_module.mps, "empty_cache"):
                try: torch_whisper_module.mps.empty_cache()
                except Exception: pass
        gc.collect()
        
    return transcribed_text, error_msg


def _trim_ref_audio_if_needed(ref_audio_path: Path, max_duration_ms: int, temp_dir_for_trimmed_audio: Path) -> tuple[Path, Path | None]:
    # (Your existing _trim_ref_audio_if_needed function - seems fine)
    if not PYDUB_AVAILABLE_FOR_TRIM or not AudioSegment_pydub:
        logger.info(f"mlx-audio: pydub not available, cannot check/trim reference audio: {ref_audio_path}. Using as is.")
        return ref_audio_path, None # Return original path, no temp file created
    try:
        logger.debug(f"mlx-audio: Checking/trimming ref audio '{ref_audio_path}' for max duration {max_duration_ms}ms.")
        audio_segment = AudioSegment_pydub.from_file(str(ref_audio_path))
        if len(audio_segment) > max_duration_ms:
            logger.info(f"mlx-audio: Reference audio '{ref_audio_path}' ({len(audio_segment)/1000.0:.1f}s) is > {max_duration_ms/1000.0:.1f}s. Trimming.")
            trimmed_segment = audio_segment[:max_duration_ms]
            # Use a unique name for the temporary trimmed file in the provided temp_dir
            temp_trimmed_filename = f"trimmed_ref_{ref_audio_path.stem}{ref_audio_path.suffix}"
            path_to_newly_trimmed_file = temp_dir_for_trimmed_audio / temp_trimmed_filename
            
            file_format = ref_audio_path.suffix.lstrip('.').lower() or 'wav' # Default to wav if no suffix
            trimmed_segment.export(str(path_to_newly_trimmed_file), format=file_format)
            logger.info(f"mlx-audio: Using trimmed temporary reference audio: {path_to_newly_trimmed_file}")
            return path_to_newly_trimmed_file, path_to_newly_trimmed_file # Return new path and path to delete
        else:
            logger.debug(f"mlx-audio: Reference audio '{ref_audio_path}' is within length limits. Using original.")
            return ref_audio_path, None # Return original path, no temp file created
    except Exception as e:
        logger.warning(f"mlx-audio: Error processing/trimming reference audio '{ref_audio_path}': {e}. Using original path.", exc_info=True)
        return ref_audio_path, None # Fallback to original path


def synthesize_with_mlx_audio(
    crisptts_model_config: dict,
    text: str,
    voice_id_or_path_override: str | None, # Can be a voice name (for Bark/Kokoro) or path to ref audio/NPZ
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    global _mlx_bark_patch_applied_flag # For Bark patch

    crisptts_specific_model_id = crisptts_model_config.get('crisptts_model_id', 'mlx_audio_unknown')
    logger.info(f"mlx-audio: Starting synthesis for model '{crisptts_specific_model_id}'.")

    if not MLX_AUDIO_AVAILABLE or not generate_audio_mlx_func:
        logger.error(f"mlx-audio ({crisptts_specific_model_id}): mlx-audio library or generate_audio function not available. Skipping.")
        return

    mlx_model_repo_id_or_path = crisptts_model_config.get("mlx_model_path")
    if not mlx_model_repo_id_or_path:
        logger.error(f"mlx-audio ({crisptts_specific_model_id}): 'mlx_model_path' not in config. Skipping.")
        return

    # Apply Bark patch if relevant
    if "bark" in crisptts_specific_model_id.lower() or ("bark" in str(mlx_model_repo_id_or_path).lower()):
        apply_mlx_audio_bark_pipeline_patch_if_needed()

    # --- Determine voice input type for mlx-audio's generate_audio ---
    # effective_voice_input_str can be:
    # 1. Path to an audio file (WAV, MP3 for cloning by CSM, OuteTTS, Spark, Dia)
    # 2. Path to an NPZ file (for Bark pre-computed voice prompts)
    # 3. A string voice name (for Bark history prompts, Kokoro voices, Spark control voices, Orpheus voices)
    # 4. None (for zero-shot if supported, or model's internal default)
    effective_voice_input_str = voice_id_or_path_override or crisptts_model_config.get("default_voice_id")
    
    actual_ref_audio_path_for_cloning = None # Path to a valid audio file (WAV, MP3 etc.) for cloning
    voice_name_or_npz_path_for_mlx = None    # String (voice name or path to NPZ) for mlx-audio's `voice` param
    is_cloning_intent = False                # True if actual_ref_audio_path_for_cloning is set

    if effective_voice_input_str:
        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Raw voice input: '{effective_voice_input_str}'")
        p_input = Path(effective_voice_input_str)
        resolved_p_input = p_input if p_input.is_absolute() else (Path.cwd() / p_input).resolve()
        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Attempting to resolve voice input at: '{resolved_p_input}'")

        if resolved_p_input.exists() and resolved_p_input.is_file():
            if resolved_p_input.suffix.lower() == ".npz": # Bark .npz voice prompt
                voice_name_or_npz_path_for_mlx = str(resolved_p_input)
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): Using NPZ voice prompt file: {voice_name_or_npz_path_for_mlx}")
            else: # It's another existing file, assume it's for cloning (e.g. WAV, MP3)
                  # This applies to CSM, OuteTTS, Spark-Clone, Dia-Clone
                actual_ref_audio_path_for_cloning = str(resolved_p_input)
                is_cloning_intent = True
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): Cloning intent. Reference audio file found: {actual_ref_audio_path_for_cloning}")
        else:
            # File doesn't exist. Treat effective_voice_input_str as a voice name string.
            # This applies to Bark history prompts, Kokoro, Spark-Control, Orpheus-Llama
            voice_name_or_npz_path_for_mlx = effective_voice_input_str
            logger.info(f"mlx-audio ({crisptts_specific_model_id}): Path '{resolved_p_input}' not found or not a file. Treating '{effective_voice_input_str}' as a voice name/preset for 'voice' parameter.")
    else:
        logger.info(f"mlx-audio ({crisptts_specific_model_id}): No voice_id or default_voice_id provided. Model will run in zero-shot or use its internal default voice if applicable.")
        # Both actual_ref_audio_path_for_cloning and voice_name_or_npz_path_for_mlx remain None

    # Initialize kwargs for mlx-audio's generate_audio
    mlx_gen_kwargs = {
        "speed": crisptts_model_config.get("default_speed", 1.0), # For Spark-Control
        "temperature": crisptts_model_config.get("default_temperature", 0.7), # For Bark, OuteTTS
        "verbose": logger.isEnabledFor(logging.DEBUG),
        "play": False, "join_audio": True, "audio_format": "wav",
        # These will be refined based on input type
        "voice": voice_name_or_npz_path_for_mlx, # Will be None if cloning
        "ref_audio": None,                       # Will be set if cloning
        "ref_text": None,
    }
    if lang_code := crisptts_model_config.get("lang_code"): # For Kokoro
        mlx_gen_kwargs["lang_code"] = lang_code
    if pitch_val := crisptts_model_config.get("default_pitch"): # For Spark-Control
        mlx_gen_kwargs["pitch"] = float(pitch_val)

    # --- Parameter Overrides & Ref Text Handling for Cloning ---
    custom_ref_text_from_params = None
    if model_params_override:
        try:
            cli_params_json = json.loads(model_params_override)
            custom_ref_text_from_params = cli_params_json.get("ref_text") # For cloning models

            # Common mlx-audio generate_audio parameters
            valid_keys = ["speed", "temperature", "top_p", "top_k", "repetition_penalty",
                          "streaming_interval", "pitch", "gender", "stt_model", "lang_code",
                          "cfg_scale", "sentence_split_method"] # Add any other relevant ones
            for key, value in cli_params_json.items():
                if key == "ref_text": continue
                if key in valid_keys:
                    if key in ["speed", "temperature", "top_p", "repetition_penalty", "streaming_interval", "pitch", "cfg_scale"]:
                        mlx_gen_kwargs[key] = float(value)
                    elif key == "top_k": mlx_gen_kwargs[key] = int(value)
                    elif key == "sentence_split_method" and isinstance(value, str) and value.lower() in ["none", "null", "passthrough"]:
                        mlx_gen_kwargs[key] = None # Explicitly set to None
                    else: mlx_gen_kwargs[key] = str(value) # For gender, stt_model, lang_code
                    logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Overrode kwarg '{key}' to '{mlx_gen_kwargs[key]}'")
        except Exception as e_parse:
            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Error parsing --model-params: {e_parse}")

    # Populate ref_audio and ref_text if cloning
    if is_cloning_intent and actual_ref_audio_path_for_cloning:
        mlx_gen_kwargs["ref_audio"] = actual_ref_audio_path_for_cloning # This might be trimmed later
        mlx_gen_kwargs["voice"] = None # Ensure 'voice' is None if using 'ref_audio'

        if custom_ref_text_from_params:
            mlx_gen_kwargs["ref_text"] = custom_ref_text_from_params
            logger.info(f"mlx-audio ({crisptts_specific_model_id}): Using ref_text from params: '{custom_ref_text_from_params[:100]}...'")
        else:
            whisper_model_id = crisptts_model_config.get("whisper_model_id_for_transcription")
            if whisper_model_id: # Check if transcription is configured for this model
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): No explicit ref_text. Transcribing ref_audio '{actual_ref_audio_path_for_cloning}' with local Whisper utility.")
                
                # Determine device for Whisper (try to keep on CPU if MLX is on MPS to avoid conflicts)
                whisper_target_device_for_util = "cpu" 
                if TORCH_FOR_WHISPER_AVAILABLE and torch_whisper_module:
                    if torch_whisper_module.cuda.is_available():
                        whisper_target_device_for_util = "cuda"
                    elif hasattr(torch_whisper_module.backends, "mps") and torch_whisper_module.backends.mps.is_available():
                        # If MLX is on Apple Silicon (MPS), run Whisper on CPU to potentially avoid device contention
                        # or if user has a non-MPS PyTorch but MLX is MPS.
                        if platform.machine() == "arm64" and platform.system() == "Darwin":
                             logger.info("mlx-audio: MLX likely running on MPS. Directing Whisper transcription to CPU.")
                        else: # General MPS case for PyTorch
                            whisper_target_device_for_util = "mps"
                
                lang_hint_for_whisper_util = crisptts_model_config.get("language_for_whisper") # e.g., "de"
                
                transcribed_text, transcription_error = _transcribe_ref_audio_for_mlx(
                    audio_path_str=actual_ref_audio_path_for_cloning,
                    whisper_model_id=whisper_model_id,
                    hf_token=os.getenv("HF_TOKEN"), # Assuming HF_TOKEN might be needed by pipeline
                    target_device_str=whisper_target_device_for_util,
                    language_hint=lang_hint_for_whisper_util
                )

                if transcription_error:
                    logger.error(f"mlx-audio ({crisptts_specific_model_id}): Transcription failed: {transcription_error}. Passing placeholder ref_text.")
                    mlx_gen_kwargs["ref_text"] = " " # Pass a non-None placeholder
                elif transcribed_text is not None and transcribed_text.strip():
                    mlx_gen_kwargs["ref_text"] = transcribed_text
                    # Logger message for successful transcription is already in _transcribe_ref_audio_for_mlx
                else:
                    logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Transcription returned empty or None. Passing placeholder ref_text.")
                    mlx_gen_kwargs["ref_text"] = " " # Pass a non-None placeholder
            else:
                logger.warning(f"mlx-audio ({crisptts_specific_model_id}): No ref_text in params and Whisper not configured for this model in config. `mlx-audio` library might attempt its own transcription if model requires it (e.g., for Dia).")
                # For Dia and similar models that NEED ref_text, mlx-audio will try to transcribe.
                # For models like CSM/OuteTTS in mlx-audio that can take ref_audio + ref_text OR just ref_audio (and infer text from audio),
                # setting ref_text to None might be fine. For consistency and to avoid mlx-audio's internal STT if we *tried* and failed:
                if "dia" not in crisptts_specific_model_id.lower(): # If not Dia, can pass None
                     mlx_gen_kwargs["ref_text"] = None
                else: # For Dia, it's better to pass something to avoid its internal STT if ours failed.
                     mlx_gen_kwargs["ref_text"] = " " # Or some other placeholder

    # --- Actual Synthesis ---
    generated_audio_data_bytes = None
    final_saved_path_str = None # Will store the path of the successfully saved audio file
    
    with tempfile.TemporaryDirectory(prefix="crisptts_mlx_handler_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        path_to_temp_trimmed_ref_to_delete = None

        # Trim ref_audio if it's set for cloning, do this *after* transcription if original was used for transcription
        if mlx_gen_kwargs.get("ref_audio"):
            original_ref_audio_path_for_trim = Path(mlx_gen_kwargs["ref_audio"]) # Path already resolved
            # Max duration for ref audio (e.g., OuteTTS in mlx-audio might have 10s limit, Spark 30s)
            # This should ideally be model-specific from config if limits are known.
            max_duration_ms_for_trim = crisptts_model_config.get("ref_audio_max_duration_ms", 15000) # Default 15s
            
            path_to_use_for_ref, path_to_temp_trimmed_ref_to_delete = _trim_ref_audio_if_needed(
                original_ref_audio_path_for_trim,
                max_duration_ms=max_duration_ms_for_trim,
                temp_dir_for_trimmed_audio=temp_dir_path # Store trimmed file in this session's temp dir
            )
            mlx_gen_kwargs["ref_audio"] = str(path_to_use_for_ref) # Update with potentially trimmed path

        # Prepare output path within the temp directory
        temp_file_basename = "mlx_synth_output" # mlx-audio will append .wav or .mp3
        mlx_gen_kwargs["file_prefix"] = temp_file_basename # generate_audio will save as file_prefix.audio_format

        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir_path) # generate_audio often saves in CWD
            
            # Filter out None kwargs unless explicitly allowed (like sentence_split_method=None)
            # This avoids passing `voice=None` if `ref_audio` is used, or `ref_audio=None` if `voice` is used.
            final_kwargs_for_generate = {
                k: v for k, v in mlx_gen_kwargs.items() 
                if v is not None or k == "sentence_split_method" # Allow sentence_split_method to be explicitly None
            }
            # Ensure that if it's cloning, 'voice' is not in final_kwargs
            if is_cloning_intent and "voice" in final_kwargs_for_generate:
                del final_kwargs_for_generate["voice"]
            # Ensure that if it's using a voice name/NPZ, 'ref_audio' and 'ref_text' are not in final_kwargs
            # (unless the specific mlx-audio model supports voice name + ref_text, which is unusual)
            if voice_name_or_npz_path_for_mlx and not is_cloning_intent:
                if "ref_audio" in final_kwargs_for_generate: del final_kwargs_for_generate["ref_audio"]
                if "ref_text" in final_kwargs_for_generate: del final_kwargs_for_generate["ref_text"]


            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Calling generate_audio with text len {len(text)}, effective kwargs: {json.dumps(final_kwargs_for_generate, default=str)}")
            
            # Suppress prolific stdout from mlx-audio unless debug is on for this handler
            with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=not logger.isEnabledFor(logging.DEBUG)):
                generate_audio_mlx_func(model_path=mlx_model_repo_id_or_path, text=text, **final_kwargs_for_generate)
            
            # Determine the expected output file name based on audio_format kwarg
            output_format_ext = final_kwargs_for_generate.get("audio_format", "wav")
            expected_temp_output_file = temp_dir_path / f"{temp_file_basename}.{output_format_ext}"
            
            if expected_temp_output_file.exists() and expected_temp_output_file.stat().st_size > 100:
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): Successfully generated: {expected_temp_output_file}")
                generated_audio_data_bytes = expected_temp_output_file.read_bytes()
            else:
                logger.error(f"mlx-audio ({crisptts_specific_model_id}): Output file not created or empty in temp dir: {expected_temp_output_file}")

        except Exception as e_synth_mlx:
            logger.error(f"mlx-audio ({crisptts_specific_model_id}): Synthesis process failed: {e_synth_mlx}", exc_info=True)
        finally:
            os.chdir(original_cwd) # Change back to original CWD
            if path_to_temp_trimmed_ref_to_delete and path_to_temp_trimmed_ref_to_delete.exists():
                try:
                    path_to_temp_trimmed_ref_to_delete.unlink()
                    logger.debug(f"mlx-audio: Deleted temporary trimmed ref audio: {path_to_temp_trimmed_ref_to_delete}")
                except OSError as e_del_temp_ref:
                    logger.warning(f"mlx-audio: Could not delete temporary trimmed ref audio {path_to_temp_trimmed_ref_to_delete}: {e_del_temp_ref}")

    if generated_audio_data_bytes:
        logger.info(f"mlx-audio ({crisptts_specific_model_id}): {len(generated_audio_data_bytes)} bytes of audio data retrieved.")
        if output_file_str:
            # Save the generated audio bytes to the user-specified output_file_str
            # The save_audio util can handle bytes directly if input_format is specified
            output_format_ext = mlx_gen_kwargs.get("audio_format", "wav")
            final_saved_path_obj = Path(output_file_str).with_suffix(f".{output_format_ext}")
            try:
                save_audio(
                    audio_data_or_path=generated_audio_data_bytes,
                    output_filepath_str=str(final_saved_path_obj),
                    source_is_path=False,
                    input_format=f"{output_format_ext}_bytes", # e.g., "wav_bytes", "mp3_bytes"
                    sample_rate=crisptts_model_config.get("sample_rate") # Useful if input is PCM
                )
                # Logger message for successful save is in save_audio util
                final_saved_path_str = str(final_saved_path_obj) # For benchmark reporting if needed
            except Exception as e_save_final:
                 logger.error(f"mlx-audio ({crisptts_specific_model_id}): Error saving final audio using save_audio util: {e_save_final}", exc_info=True)
        
        if play_direct:
            output_format_ext = mlx_gen_kwargs.get("audio_format", "wav")
            play_audio(
                audio_path_or_data=generated_audio_data_bytes,
                is_path=False,
                input_format=f"{output_format_ext}_bytes", # Let play_audio handle these byte formats
                sample_rate=crisptts_model_config.get("sample_rate") # Useful if input is PCM
            )
    else:
        logger.warning(f"mlx-audio ({crisptts_specific_model_id}): No audio generated for '{mlx_model_repo_id_or_path}'.")

    # MLX cache clear
    if MLX_AUDIO_AVAILABLE and mx_core_module and hasattr(mx_core_module, 'clear_cache'):
        try: mx_core_module.clear_cache(); logger.debug("mlx-audio: Cleared MLX cache.")
        except Exception as e: logger.warning(f"mlx-audio: Error clearing MLX cache: {e}")
    gc.collect()
    logger.info(f"mlx-audio ({crisptts_specific_model_id}): Synthesis function finished.")