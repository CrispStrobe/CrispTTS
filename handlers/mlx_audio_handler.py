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

# CrispTTS utils
from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.mlx_audio")

# --- Conditional Imports ---
MLX_AUDIO_AVAILABLE = False
generate_audio_mlx_func = None
mx_core_module = None # For mx.array type check and mx.clear_cache()

PYDUB_AVAILABLE_FOR_TRIM = False
AudioSegment_pydub = None

try:
    from pydub import AudioSegment
    AudioSegment_pydub = AudioSegment
    PYDUB_AVAILABLE_FOR_TRIM = True
except ImportError:
    # Logger for init phase, might be different from the main handler logger
    logger_mlx_init_trim = logging.getLogger("CrispTTS.handlers.mlx_audio.init.trim")
    logger_mlx_init_trim.info("pydub not found. Reference audio trimming for MLX-Audio will not be available.")

try:
    from mlx_audio.tts.generate import generate_audio as generate_audio_mlx_imp
    import mlx.core as mx_core_imported # Import for type checking and utilities
    
    generate_audio_mlx_func = generate_audio_mlx_imp
    mx_core_module = mx_core_imported
    MLX_AUDIO_AVAILABLE = True
    
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        logger.info("mlx-audio library and mlx.core loaded (Apple Silicon detected).")
    else:
        # mlx currently only supports Apple Silicon for GPU operations
        logger.warning("mlx-audio library loaded, but non-Apple Silicon detected. MLX operations will run on CPU and might be slow or unsupported.")
        
except ImportError:
    logger.warning("mlx-audio library not found or failed to import. MLX Audio handler will be non-functional.")
    generate_audio_mlx_func = None
    mx_core_module = None
    MLX_AUDIO_AVAILABLE = False
except Exception as e:
    logger.error(f"An error occurred during mlx-audio or mlx.core import: {e}. Handler non-functional.", exc_info=True)
    generate_audio_mlx_func = None
    mx_core_module = None
    MLX_AUDIO_AVAILABLE = False


def _convert_mlx_audio_to_pcm_s16le_bytes(mlx_audio_array: 'mx_core_module.array', target_sample_rate: int) -> bytes:
    """Converts an mlx.core.array (expected float32, [-1, 1]) to 16-bit PCM bytes."""
    if not MLX_AUDIO_AVAILABLE or not mx_core_module or not isinstance(mlx_audio_array, mx_core_module.array):
        logger.error("Cannot convert audio: MLX array or mx_core_module not available.")
        return b""
    
    try:
        audio_np_float32 = np.array(mlx_audio_array.astype(mx_core_module.float32))
        audio_np_int16 = (np.clip(audio_np_float32, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_np_int16.tobytes()
    except Exception as e:
        logger.error(f"Error converting MLX audio array to PCM bytes: {e}", exc_info=True)
        return b""


def _trim_ref_audio_if_needed(ref_audio_path: Path, max_duration_ms: int, temp_dir_for_trimmed_audio: Path) -> tuple[Path, Path | None]:
    """Trims audio if longer than max_duration_ms. Saves to temp_dir_for_trimmed_audio.
    Returns path to (potentially trimmed) audio and path of new temp file if created for deletion."""
    if not PYDUB_AVAILABLE_FOR_TRIM or not AudioSegment_pydub:
        logger.info(f"pydub not available, cannot trim reference audio: {ref_audio_path}. Using as is.")
        return ref_audio_path, None
    try:
        audio_segment = AudioSegment_pydub.from_file(str(ref_audio_path))
        if len(audio_segment) > max_duration_ms:
            logger.info(f"Reference audio '{ref_audio_path}' ({len(audio_segment)/1000.0:.1f}s) is > {max_duration_ms/1000.0:.1f}s. Trimming.")
            trimmed_segment = audio_segment[:max_duration_ms]
            # Ensure temp_dir_for_trimmed_audio exists
            temp_dir_for_trimmed_audio.mkdir(parents=True, exist_ok=True)
            temp_trimmed_filename = f"trimmed_ref_{ref_audio_path.stem}{ref_audio_path.suffix}"
            path_to_newly_trimmed_file = temp_dir_for_trimmed_audio / temp_trimmed_filename
            file_format = ref_audio_path.suffix.lstrip('.').lower() or 'wav'
            trimmed_segment.export(str(path_to_newly_trimmed_file), format=file_format)
            logger.info(f"Using trimmed temporary reference audio: {path_to_newly_trimmed_file}")
            return path_to_newly_trimmed_file, path_to_newly_trimmed_file # Return new path and path to delete
        else:
            logger.debug(f"Reference audio '{ref_audio_path}' is within length limits. Using original.")
            return ref_audio_path, None # Return original path, nothing to delete
    except Exception as e:
        logger.warning(f"Error processing/trimming reference audio '{ref_audio_path}': {e}. Using original path.", exc_info=True)
        return ref_audio_path, None


def synthesize_with_mlx_audio(
    crisptts_model_config: dict,
    text: str,
    voice_id_or_path_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    if not MLX_AUDIO_AVAILABLE or not generate_audio_mlx_func:
        logger.error("mlx-audio handler: library or generate_audio function not available. Skipping.")
        return

    mlx_model_repo_id_or_path = crisptts_model_config.get("mlx_model_path")
    if not mlx_model_repo_id_or_path:
        logger.error("mlx-audio: 'mlx_model_path' (HF repo ID or local path) not in config. Skipping.")
        return

    voice_input_for_mlx = voice_id_or_path_override or crisptts_model_config.get("default_voice_id")
    ref_audio_path_for_mlx_str = None
    voice_name_for_mlx_str = None
    path_to_temp_trimmed_ref_to_delete = None

    if voice_input_for_mlx:
        p_voice_input = Path(voice_input_for_mlx)
        # Check if it looks like a file path first (more specific)
        is_likely_file_path = p_voice_input.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg'] or p_voice_input.is_file()

        if is_likely_file_path:
            ref_audio_path_actual = p_voice_input.resolve()
            if not p_voice_input.is_absolute(): # If relative, resolve against CrispTTS project root
                project_root = Path(__file__).resolve().parent.parent
                ref_audio_path_actual = (project_root / p_voice_input).resolve()

            if ref_audio_path_actual.exists():
                # Use a temporary directory within the main temp_dir_manager context for trimmed files
                # This requires temp_dir_manager to be initialized before this block.
                # Let's handle trimming inside the main try-finally block that creates temp_dir_manager.
                ref_audio_path_for_mlx_str = str(ref_audio_path_actual) # Tentatively set, might be trimmed
                # Trimming logic moved into the main try block
            else:
                logger.error(f"mlx-audio: Reference audio file '{voice_input_for_mlx}' not found at '{ref_audio_path_actual}'. Skipping.")
                return
        else: # Treat as a voice name string
            voice_name_for_mlx_str = str(voice_input_for_mlx)
    else:
        logger.warning("mlx-audio: No voice ID or reference audio path provided. Some models might require one or use an internal default.")

    lang_code_for_mlx = crisptts_model_config.get("lang_code")
    target_sample_rate = crisptts_model_config.get("sample_rate", 24000)
    output_audio_format = "wav"

    mlx_gen_kwargs = {
        "speed": crisptts_model_config.get("default_speed", 1.0),
        "temperature": crisptts_model_config.get("default_temperature", 0.7),
        "verbose": logger.isEnabledFor(logging.DEBUG),
        "play": False,
        "join_audio": True,
        "audio_format": output_audio_format,
        "voice": voice_name_for_mlx_str,
        "ref_audio": None, # Will be set after potential trimming
        "ref_text": None,  # Default, can be overridden by model_params
    }
    if lang_code_for_mlx:
        mlx_gen_kwargs["lang_code"] = lang_code_for_mlx
    
    if model_params_override:
        try:
            cli_params_json = json.loads(model_params_override)
            for key, value in cli_params_json.items():
                if key in ["speed", "temperature", "top_p", "top_k", "repetition_penalty", "streaming_interval", "pitch"]: # Added pitch
                    try: mlx_gen_kwargs[key] = float(value)
                    except ValueError: logger.warning(f"mlx-audio: Could not convert param '{key}' value '{value}' to float.")
                elif key in ["ref_text", "gender", "stt_model", "lang_code"]:
                    mlx_gen_kwargs[key] = str(value)
                # Add other specific params if mlx-audio's generate_audio supports them
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"mlx-audio: Error parsing --model-params '{model_params_override}': {e}")

    generated_audio_data_bytes = None
    final_saved_path_str = None # Path to the final audio file (either temp or user-specified)
    actual_generated_temp_file_path_obj = None # Path to the file mlx-audio created in temp dir

    with tempfile.TemporaryDirectory(prefix="crisptts_mlx_handler_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        
        # Handle reference audio trimming now that temp_dir_path is available
        if ref_audio_path_for_mlx_str: # If it was identified as a file path earlier
            path_to_use_for_ref, path_to_temp_trimmed_ref_to_delete = _trim_ref_audio_if_needed(
                Path(ref_audio_path_for_mlx_str),
                max_duration_ms=15000, # Example, make configurable if needed per model
                temp_dir_for_trimmed_audio=temp_dir_path # Use current temp dir for trimmed files
            )
            mlx_gen_kwargs["ref_audio"] = str(path_to_use_for_ref)
            mlx_gen_kwargs["voice"] = None # Ensure voice is None if ref_audio is used
            logger.info(f"mlx-audio: Using reference audio: {mlx_gen_kwargs['ref_audio']}")
        elif voice_name_for_mlx_str:
             mlx_gen_kwargs["voice"] = voice_name_for_mlx_str
             mlx_gen_kwargs["ref_audio"] = None
             logger.info(f"mlx-audio: Using voice name: {mlx_gen_kwargs['voice']}")


        temp_file_basename = "mlx_synth_output"
        mlx_gen_kwargs["file_prefix"] = temp_file_basename
        expected_temp_file_path_obj = temp_dir_path / f"{temp_file_basename}.{output_audio_format}"
        
        original_cwd = Path.cwd()
        os.chdir(temp_dir_path)

        logger.info(f"mlx-audio: Synthesizing with model '{mlx_model_repo_id_or_path}'...")
        logger.debug(f"mlx-audio: Calling generate_audio with effective kwargs: {json.dumps(mlx_gen_kwargs, default=str)}")

        final_audio_array_from_result = None
        generation_summary = []

        try:
            for result in generate_audio_mlx_func(model_path=mlx_model_repo_id_or_path, text=text, **mlx_gen_kwargs):
                result_log = (
                    f"Segment {result.segment_idx}: Duration: {result.audio_duration}, "
                    f"Samples: {result.samples}, Tokens: {result.token_count}, "
                    f"RTF: {result.real_time_factor:.2f}x, ProcTime: {result.processing_time_seconds:.2f}s, "
                    f"Mem: {result.peak_memory_usage:.2f}GB"
                )
                logger.info(f"mlx-audio: {result_log}")
                generation_summary.append(result_log)
                if result.audio is not None:
                    final_audio_array_from_result = result.audio # Assume last one is the joined one

            if expected_temp_file_path_obj.exists() and expected_temp_file_path_obj.stat().st_size > 100:
                actual_generated_temp_file_path_obj = expected_temp_file_path_obj
                generated_audio_data_bytes = actual_generated_temp_file_path_obj.read_bytes()
                logger.info(f"mlx-audio: Successfully generated and read temp audio file: {actual_generated_temp_file_path_obj}")
            elif final_audio_array_from_result is not None:
                logger.info("mlx-audio: Generated audio array from result. File not found/empty. Using array for output.")
                generated_audio_data_bytes = _convert_mlx_audio_to_pcm_s16le_bytes(final_audio_array_from_result, target_sample_rate)
            else:
                logger.error(f"mlx-audio: Synthesis for '{mlx_model_repo_id_or_path}' did not produce usable file or audio array.")
        
        except Exception as e:
            logger.error(f"mlx-audio: Synthesis failed for model '{mlx_model_repo_id_or_path}': {e}", exc_info=True)
        finally:
            os.chdir(original_cwd)
            if path_to_temp_trimmed_ref_to_delete and path_to_temp_trimmed_ref_to_delete.exists():
                try: path_to_temp_trimmed_ref_to_delete.unlink()
                except OSError: logger.warning(f"Could not delete temp trimmed ref: {path_to_temp_trimmed_ref_to_delete}")

    # After temp_dir_manager context exits (temp dir is cleaned up)
    if generated_audio_data_bytes:
        if output_file_str:
            # If actual_generated_temp_file_path_obj was set and then its parent temp dir deleted,
            # we must save from generated_audio_data_bytes.
            final_saved_path_obj = Path(output_file_str).with_suffix(f".{output_audio_format}")
            save_audio(generated_audio_data_bytes, str(final_saved_path_obj), 
                       source_is_path=False, input_format="pcm_s16le", sample_rate=target_sample_rate)
            final_saved_path_str = str(final_saved_path_obj)
        
        if play_direct:
            if final_saved_path_str and Path(final_saved_path_str).exists(): # If saved to user path
                play_audio(final_saved_path_str, is_path=True)
            else: # Play from bytes if not saved to a persistent user path
                play_audio(generated_audio_data_bytes, is_path=False, input_format="pcm_s16le", sample_rate=target_sample_rate)
    else:
        logger.warning(f"mlx-audio: No audio was generated or an error occurred for model '{mlx_model_repo_id_or_path}'.")
            
    gc.collect()
    if MLX_AUDIO_AVAILABLE and mx_core_module and hasattr(mx_core_module, 'clear_cache'):
        try:
            mx_core_module.clear_cache()
            logger.debug("mlx-audio: Cleared MLX cache.")
        except Exception as e_mlx_clear:
            logger.warning(f"mlx-audio: Error clearing MLX cache: {e_mlx_clear}")