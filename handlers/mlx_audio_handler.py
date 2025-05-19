# handlers/mlx_audio_handler.py

import logging
import platform
import os
from pathlib import Path
import json
import shutil
import tempfile
import gc

# Relative imports
from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.mlx_audio")

MLX_AUDIO_AVAILABLE = False
generate_audio_mlx_func = None
HF_HUB_AVAILABLE_FOR_MLX_HANDLER = False
hf_hub_download_mlx = None
hf_fs_mlx = None

try:
    from huggingface_hub import hf_hub_download, HfFileSystem
    HF_HUB_AVAILABLE_FOR_MLX_HANDLER = True
    hf_hub_download_mlx = hf_hub_download
    hf_fs_mlx = HfFileSystem()
except ImportError:
    logger.warning("huggingface_hub not installed. MLX Audio handler might fail to download models.")

try:
    from mlx_audio.tts.generate import generate_audio as generate_audio_mlx_imp
    generate_audio_mlx_func = generate_audio_mlx_imp
    MLX_AUDIO_AVAILABLE = True
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        logger.info("mlx-audio library loaded (Apple Silicon detected).")
    else:
        logger.info("mlx-audio library loaded (Non-Apple Silicon or generic import). Performance/compatibility may vary.")
except ImportError:
    logger.info("mlx-audio library not found. MLX Audio handler will be non-functional.")
except Exception as e:
    logger.warning(f"An error occurred during mlx-audio import: {e}. Handler non-functional.")


def synthesize_with_mlx_audio(model_config, text, voice_id_or_path_override, model_params_override, output_file_str, play_direct):
    if not MLX_AUDIO_AVAILABLE or not generate_audio_mlx_func:
        logger.error("mlx-audio handler: library or generate_audio function not available. Skipping.")
        return

    mlx_model_path_config = model_config.get("mlx_model_path") # For Kokoro, CSM (HF ID)
    onnx_repo_id_for_mlx_outetts = model_config.get("onnx_repo_id") # For OuteTTS-via-MLX from specific ONNX repo

    if not mlx_model_path_config and not onnx_repo_id_for_mlx_outetts:
        logger.error("mlx-audio: 'mlx_model_path' (for Kokoro/CSM) or 'onnx_repo_id' (for OuteTTS-via-MLX) not in config.")
        return

    voice_input = voice_id_override or model_config.get("default_voice_id")
    if not voice_input:
        logger.error("mlx-audio: Voice ID or reference audio path not specified.")
        return

    lang_code = model_config.get("lang_code") # Can be None
    default_speed = model_config.get("default_speed", 1.0)
    default_temperature = model_config.get("default_temperature", 0.5) # mlx_audio uses temperature
    target_sample_rate = model_config.get("sample_rate", 24000)
    output_format = "wav" # Standardize

    gen_params = {"speed": default_speed, "temperature": default_temperature}
    if model_params_override:
        try:
            cli_params = json.loads(model_params_override)
            gen_params.update({k: cli_params[k] for k in ["speed", "temperature"] if k in cli_params})
            if "lang_code" in cli_params: lang_code = cli_params["lang_code"]
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"mlx-audio: Error parsing --model-params '{model_params_override}': {e}")

    temp_dir_manager = tempfile.TemporaryDirectory(prefix="crisptts_mlx_")
    temp_output_dir_path = Path(temp_dir_manager.name)
    temp_file_prefix = "mlx_synth_temp"
    
    actual_model_to_load_for_mlx = mlx_model_path_config # For Kokoro, CSM
    
    # --- Logic for OuteTTS-via-MLX using specific ONNX files ---
    # This assumes mlx_audio.tts.generate.generate_audio can take a local path to a directory
    # containing an ONNX model and its associated files.
    if onnx_repo_id_for_mlx_outetts and model_config.get("handler_function_key") == "mlx_audio_outetts_q4": # Specific key
        if not HF_HUB_AVAILABLE_FOR_MLX_HANDLER:
            logger.error("mlx-audio (OuteTTS ONNX): huggingface_hub needed to download ONNX files. Skipping.")
            temp_dir_manager.cleanup(); return

        onnx_files_to_try = model_config.get("onnx_filename_options_for_mlx", [])
        onnx_subfolder_in_repo = model_config.get("onnx_subfolder", "onnx")
        
        successfully_downloaded_onnx_dir = None
        for onnx_file_relative in onnx_files_to_try:
            full_path_in_repo = f"{onnx_subfolder_in_repo}/{Path(onnx_file_relative).name}" if onnx_subfolder_in_repo else Path(onnx_file_relative).name
            
            # Create a unique subdirectory within the temp dir for this specific ONNX attempt
            current_attempt_local_dir = temp_output_dir_path / Path(onnx_file_relative).stem
            current_attempt_local_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"mlx-audio (OuteTTS ONNX): Attempting to download '{full_path_in_repo}' from '{onnx_repo_id}' to '{current_attempt_local_dir}'.")
            try:
                hf_hub_download_mlx(repo_id=onnx_repo_id, filename=full_path_in_repo,
                                   local_dir=str(current_attempt_local_dir), local_dir_use_symlinks=False, token=os.getenv("HF_TOKEN"))
                
                # Attempt to download corresponding .onnx_data file
                onnx_data_file_relative = full_path_in_repo + "_data"
                if hf_fs_mlx and hf_fs_mlx.exists(f"{onnx_repo_id}/{onnx_data_file_relative}"):
                    hf_hub_download_mlx(repo_id=onnx_repo_id, filename=onnx_data_file_relative,
                                       local_dir=str(current_attempt_local_dir), local_dir_use_symlinks=False, token=os.getenv("HF_TOKEN"))
                    logger.info(f"mlx-audio (OuteTTS ONNX): Downloaded '{Path(onnx_data_file_relative).name}'.")
                
                actual_model_to_load_for_mlx = str(current_model_attempt_dir) # Point to the directory containing the ONNX
                successfully_downloaded_onnx_dir = current_model_attempt_dir
                break # Successfully downloaded, proceed with this one
            except Exception as e_dl:
                logger.warning(f"mlx-audio (OuteTTS ONNX): Failed to download '{full_path_in_repo}': {e_dl}. Trying next if available.")
                shutil.rmtree(current_attempt_local_dir, ignore_errors=True) # Clean up failed attempt dir
        
        if not successfully_downloaded_onnx_dir:
            logger.error(f"mlx-audio (OuteTTS ONNX): Failed to download any specified ONNX model from {onnx_repo_id}.")
            temp_dir_manager.cleanup(); return
        logger.info(f"mlx-audio (OuteTTS ONNX): Will use local ONNX model at: {actual_model_to_load_for_mlx}")


    generate_kwargs = {
        "text": text, "model_path": actual_model_to_load_for_mlx,
        "speed": gen_params["speed"], "temperature": gen_params.get("temperature"),
        "lang_code": lang_code, "file_prefix": temp_file_prefix,
        "output_path": str(temp_output_dir_path), "audio_format": output_format,
        "sample_rate": target_sample_rate, "join_audio": True,
        "verbose": logger.isEnabledFor(logging.DEBUG)
    }

    is_ref_audio_path = Path(voice_input).is_file() and Path(voice_input).suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']
    if is_ref_audio_path: # For CSM or OuteTTS-via-MLX with reference audio
        logger.info(f"mlx-audio: Using '{voice_input}' as reference audio.")
        generate_kwargs["ref_audio_path"] = str(voice_input)
    else: # For Kokoro-style predefined voices
        logger.info(f"mlx-audio: Using pre-defined voice ID: '{voice_input}'.")
        generate_kwargs["voice"] = voice_input

    logger.debug(f"mlx-audio: Final generate_kwargs: {generate_kwargs}")
    
    actual_generated_file_path = None
    try:
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            synthesis_result = generate_audio_mlx_func(**generate_kwargs)

        if isinstance(synthesis_result, list) and synthesis_result: actual_generated_file_path = Path(synthesis_result[0])
        elif isinstance(synthesis_result, str) and Path(synthesis_result).exists(): actual_generated_file_path = Path(synthesis_result)
        else:
            found_files = list(temp_output_dir_path.glob(f"{temp_file_prefix}*.{output_format}"))
            if found_files: actual_generated_file_path = found_files[0]
        
        if not actual_generated_file_path or not actual_generated_file_path.exists():
            logger.error(f"mlx-audio: No audio file found/generated in '{temp_output_dir_path}' with prefix '{temp_file_prefix}'."); return

        logger.info(f"mlx-audio: Synthesis successful. Temporary audio at: {actual_generated_file_path}")
        final_output_path = None
        if output_file_str:
            final_output_path = Path(output_file_str).with_suffix(f".{output_format}")
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(actual_generated_file_path), str(final_output_path))
            logger.info(f"mlx-audio: Audio moved to {final_output_path}")
        else: final_output_path = actual_generated_file_path

        if play_direct and final_output_path and final_output_path.exists():
            play_audio(str(final_output_path), is_path=True)

    except FileNotFoundError as e_fnf: logger.error(f"mlx-audio: Model assets error for '{actual_model_to_load_for_mlx}': {e_fnf}", exc_info=True)
    except RuntimeError as e_runtime: logger.error(f"mlx-audio: Runtime error (possibly MLX backend/device issue for '{actual_model_to_load_for_mlx}'): {e_runtime}", exc_info=True)
    except Exception as e: logger.error(f"mlx-audio: Synthesis failed for '{actual_model_to_load_for_mlx}': {e}", exc_info=True)
    finally:
        if temp_dir_manager:
            try: temp_dir_manager.cleanup(); logger.debug("mlx-audio: Cleaned up temporary directory.")
            except Exception as e_clean: logger.warning(f"mlx-audio: Failed to cleanup temporary directory: {e_clean}")
        gc.collect()