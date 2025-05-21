# handlers/mlx_audio_handler.py
import logging
import platform
import os
from pathlib import Path
import json
import shutil
import tempfile
import gc

from utils import save_audio, play_audio, SuppressOutput

# Attempt to import pydub for trimming, use a flag
PYDUB_AVAILABLE_FOR_TRIM = False
AudioSegment_pydub = None
try:
    from pydub import AudioSegment
    AudioSegment_pydub = AudioSegment
    PYDUB_AVAILABLE_FOR_TRIM = True
except ImportError:
    logger_mlx_init = logging.getLogger("CrispTTS.handlers.mlx_audio.init")
    logger_mlx_init.warning("pydub not found. Reference audio trimming for MLX-Audio will not be available.")

logger = logging.getLogger("CrispTTS.handlers.mlx_audio")

MLX_AUDIO_AVAILABLE = False
generate_audio_mlx_func = None

try:
    from mlx_audio.tts.generate import generate_audio as generate_audio_mlx_imp
    generate_audio_mlx_func = generate_audio_mlx_imp
    MLX_AUDIO_AVAILABLE = True
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        logger.info("mlx-audio library loaded (Apple Silicon detected).")
    else:
        logger.info("mlx-audio library loaded (Non-Apple Silicon or generic import).")
except ImportError:
    logger.info("mlx-audio library not found. MLX Audio handler will be non-functional.")
    generate_audio_mlx_func = None
    MLX_AUDIO_AVAILABLE = False
except Exception as e:
    logger.warning(f"An error occurred during mlx-audio import: {e}. Handler non-functional.", exc_info=True)
    generate_audio_mlx_func = None
    MLX_AUDIO_AVAILABLE = False

def _trim_ref_audio_if_needed(ref_audio_path: Path, max_duration_ms: int, temp_dir_for_trimmed_audio: Path) -> tuple[Path, Path | None]:
    """Trims audio if longer than max_duration_ms. Saves to temp_dir_for_trimmed_audio.
    Returns path to (potentially trimmed) audio and path of new temp file if created for deletion."""
    if not PYDUB_AVAILABLE_FOR_TRIM or not AudioSegment_pydub:
        logger.warning(f"pydub not available, cannot trim reference audio: {ref_audio_path}. Using as is.")
        return ref_audio_path, None
    try:
        audio_segment = AudioSegment_pydub.from_file(str(ref_audio_path))
        if len(audio_segment) > max_duration_ms:
            logger.info(f"Reference audio '{ref_audio_path}' ({len(audio_segment)/1000.0:.1f}s) is > {max_duration_ms/1000.0:.1f}s. Trimming.")
            trimmed_segment = audio_segment[:max_duration_ms]
            temp_trimmed_filename = f"trimmed_ref_{ref_audio_path.stem}{ref_audio_path.suffix}"
            temp_dir_for_trimmed_audio.mkdir(parents=True, exist_ok=True)
            path_to_newly_trimmed_file = temp_dir_for_trimmed_audio / temp_trimmed_filename
            file_format = ref_audio_path.suffix.lstrip('.') or 'wav'
            trimmed_segment.export(str(path_to_newly_trimmed_file), format=file_format)
            logger.info(f"Using trimmed temporary reference audio: {path_to_newly_trimmed_file}")
            return path_to_newly_trimmed_file, path_to_newly_trimmed_file
        else:
            logger.debug(f"Reference audio '{ref_audio_path}' is within length limits.")
            return ref_audio_path, None
    except Exception as e:
        logger.warning(f"Error processing/trimming reference audio '{ref_audio_path}': {e}. Using original path.", exc_info=True)
        return ref_audio_path, None

def synthesize_with_mlx_audio(model_config, text, voice_id_or_path_override, model_params_override, output_file_str, play_direct):
    if not MLX_AUDIO_AVAILABLE or not generate_audio_mlx_func:
        logger.error("mlx-audio handler: library or generate_audio function not available. Skipping.")
        return

    mlx_model_path_config = model_config.get("mlx_model_path")
    if not mlx_model_path_config:
        logger.error("mlx-audio: 'mlx_model_path' not specified in configuration.")
        return

    voice_input = voice_id_or_path_override or model_config.get("default_voice_id")
    if not voice_input:
        logger.error("mlx-audio: Voice ID or reference audio path not specified.")
        return

    lang_code = model_config.get("lang_code")
    default_speed = model_config.get("default_speed", 1.0)
    default_temperature = model_config.get("default_temperature", 0.7)
    output_format = "wav"

    gen_params_from_config = {"speed": default_speed, "temperature": default_temperature}
    gen_params_runtime = {}

    if model_params_override:
        try:
            cli_params = json.loads(model_params_override)
            if "speed" in cli_params: gen_params_runtime["speed"] = float(cli_params["speed"])
            if "temperature" in cli_params: gen_params_runtime["temperature"] = float(cli_params["temperature"])
            if "lang_code" in cli_params: lang_code = cli_params["lang_code"]
            if "ref_text" in cli_params and cli_params["ref_text"]:
                 gen_params_runtime["ref_text"] = cli_params["ref_text"]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"mlx-audio: Error parsing --model-params '{model_params_override}': {e}")

    final_gen_params = {**gen_params_from_config, **gen_params_runtime}

    temp_dir_manager = None
    actual_generated_file_path = None
    path_to_temp_trimmed_ref = None 

    try:
        temp_dir_manager = tempfile.TemporaryDirectory(prefix="crisptts_mlx_")
        temp_context_dir = Path(temp_dir_manager.name)
        temp_file_base_name_in_ctx = "mlx_synth_output"

        generate_kwargs = {
            "text": text,
            "model_path": mlx_model_path_config,
            "speed": final_gen_params["speed"],
            "temperature": final_gen_params["temperature"],
            "lang_code": lang_code,
            "file_prefix": temp_file_base_name_in_ctx,
            "audio_format": output_format,
            "join_audio": True,
            "verbose": logger.isEnabledFor(logging.DEBUG),
            "play": False
        }
        if "ref_text" in final_gen_params:
            generate_kwargs["ref_text"] = final_gen_params["ref_text"]
            logger.info(f"mlx-audio: Using provided ref_text: '{final_gen_params['ref_text'][:50]}...'")

        is_ref_audio_path = Path(voice_input).is_file() and Path(voice_input).suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']
        
        if is_ref_audio_path:
            ref_audio_file_orig = Path(voice_input)
            if not ref_audio_file_orig.is_absolute():
                project_root = Path(__file__).resolve().parent.parent 
                ref_audio_file_orig = (project_root / ref_audio_file_orig).resolve()
            
            if ref_audio_file_orig.exists():
                MAX_REF_AUDIO_DURATION_MS_MLX = 15000 
                path_for_mlx_ref_audio, path_to_temp_trimmed_ref = _trim_ref_audio_if_needed(
                    ref_audio_file_orig, MAX_REF_AUDIO_DURATION_MS_MLX, temp_context_dir
                )
                logger.info(f"mlx-audio: Using '{path_for_mlx_ref_audio}' as reference audio for MLX.")
                generate_kwargs["ref_audio"] = str(path_for_mlx_ref_audio)
                generate_kwargs["voice"] = None  # **** CRITICAL FIX: Explicitly None ****
            else:
                logger.error(f"mlx-audio: Reference audio file not found: {ref_audio_file_orig}")
                # Ensure cleanup if we exit early
                if path_to_temp_trimmed_ref and path_to_temp_trimmed_ref.exists(): path_to_temp_trimmed_ref.unlink(missing_ok=True)
                if temp_dir_manager: temp_dir_manager.cleanup()
                return
        else: 
            logger.info(f"mlx-audio: Using pre-defined voice ID: '{voice_input}'.")
            generate_kwargs["voice"] = voice_input
            generate_kwargs["ref_audio"] = None # **** CRITICAL FIX: Explicitly None ****

        logger.debug(f"mlx-audio: Final generate_kwargs: {json.dumps(generate_kwargs, default=str)}")

        original_cwd = Path.cwd()
        os.chdir(temp_context_dir) 

        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), 
                            suppress_stderr=not logger.isEnabledFor(logging.DEBUG)):
            generate_audio_mlx_func(**generate_kwargs)
        
        expected_temp_file = temp_context_dir / f"{temp_file_base_name_in_ctx}.{output_format}"
        if expected_temp_file.exists() and expected_temp_file.stat().st_size > 100:
            actual_generated_file_path = expected_temp_file
            logger.info(f"mlx-audio: Synthesis successful. Temporary audio at: {actual_generated_file_path}")
        else:
            logger.error(f"mlx-audio: No audio file found at '{expected_temp_file}' or file is too small. Check mlx-audio output with debug.")
    
    except FileNotFoundError as e_fnf:
        logger.error(f"mlx-audio: Model assets error for '{mlx_model_path_config}'. Error: {e_fnf}", exc_info=True)
    except RuntimeError as e_runtime:
        logger.error(f"mlx-audio: Runtime error for '{mlx_model_path_config}'. Error: {e_runtime}", exc_info=True)
    except ValueError as e_val:
        logger.error(f"mlx-audio: ValueError for '{mlx_model_path_config}': {e_val}", exc_info=True)
    except Exception as e:
        logger.error(f"mlx-audio: Unexpected error for '{mlx_model_path_config}': {e}", exc_info=True)
    finally:
        if 'original_cwd' in locals() and Path.cwd() != original_cwd :
            os.chdir(original_cwd)

    if actual_generated_file_path and actual_generated_file_path.exists():
        final_output_path_to_play = actual_generated_file_path
        if output_file_str:
            target_output_file = Path(output_file_str).with_suffix(f".{output_format}")
            target_output_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(actual_generated_file_path), str(target_output_file))
                logger.info(f"mlx-audio: Audio moved to {target_output_file}")
                final_output_path_to_play = target_output_file
            except Exception as e_move:
                logger.error(f"mlx-audio: Failed to move temp audio: {e_move}")
        
        if play_direct:
            play_audio(str(final_output_path_to_play), is_path=True)
            
    elif not actual_generated_file_path :
         logger.warning(f"mlx-audio: Synthesis ran but no audio path determined for '{mlx_model_path_config}'.")

    if path_to_temp_trimmed_ref and path_to_temp_trimmed_ref.exists():
        try:
            path_to_temp_trimmed_ref.unlink()
            logger.debug(f"mlx-audio: Deleted temporary trimmed reference: {path_to_temp_trimmed_ref}")
        except Exception as e_del_trim:
            logger.warning(f"mlx-audio: Failed to delete temp trimmed ref: {e_del_trim}")

    if temp_dir_manager:
        try:
            temp_dir_manager.cleanup()
            logger.debug("mlx-audio: Cleaned up main temporary directory.")
        except Exception as e_clean:
            logger.warning(f"mlx-audio: Error cleaning main temp dir {temp_dir_manager.name}: {e_clean}", exc_info=True)
    gc.collect()