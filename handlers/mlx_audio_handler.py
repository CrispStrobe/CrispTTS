# CrispTTS/handlers/mlx_audio_handler.py
import logging
import platform
import os # Main os import
from pathlib import Path
import json
import shutil
import tempfile
import gc
import numpy as np # Main numpy for the handler

# CrispTTS utils
from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.mlx_audio")

# --- Conditional Imports for mlx-audio and pydub ---
MLX_AUDIO_AVAILABLE = False
generate_audio_mlx_func = None
mx_core_module = None

PYDUB_AVAILABLE_FOR_TRIM = False
AudioSegment_pydub = None

try:
    from pydub import AudioSegment
    AudioSegment_pydub = AudioSegment
    PYDUB_AVAILABLE_FOR_TRIM = True
except ImportError:
    logger.info("pydub not found. Reference audio trimming for MLX-Audio will not be available.")

try:
    from mlx_audio.tts.generate import generate_audio as generate_audio_mlx_imp
    import mlx.core as mx_core_imported
    
    generate_audio_mlx_func = generate_audio_mlx_imp
    mx_core_module = mx_core_imported
    MLX_AUDIO_AVAILABLE = True
    
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        logger.info("mlx-audio library and mlx.core loaded (Apple Silicon detected).")
    else:
        logger.warning("mlx-audio library loaded, but non-Apple Silicon detected. MLX ops will run on CPU.")
        
except ImportError:
    logger.warning("mlx-audio library not found or failed to import. MLX Audio handler will be non-functional.")
except Exception as e:
    logger.error(f"An error occurred during mlx-audio or mlx.core import: {e}. Handler non-functional.", exc_info=True)

# --- Monkey Patch Section for mlx_audio.tts.models.bark.pipeline._load_voice_prompt ---
_original_load_voice_prompt = None
_mlx_bark_patch_applied_flag = False 

BARK_VOICE_PROMPT_REPO_ID = "suno/bark-small" # Dedicated repository for voice prompts

try:
    # These imports are for the patch logic
    import mlx_audio.tts.models.bark.pipeline as bark_pipeline_module_for_patch
    from huggingface_hub import hf_hub_download
    import numpy as np_for_patch 
    import os as os_for_patch # Aliased os for the patch
    PATCH_IMPORTS_SUCCESSFUL = True
except ImportError as e_patch_imp:
    logger.warning(f"Failed to import modules needed for mlx-audio Bark patch: {e_patch_imp}. Patch will not be applied.")
    PATCH_IMPORTS_SUCCESSFUL = False


def _patched_load_voice_prompt(voice_prompt_input):
    logger.debug(f"[Patched _load_voice_prompt] Called with voice_prompt_input: '{voice_prompt_input}'. Voice prompts will be sourced from '{BARK_VOICE_PROMPT_REPO_ID}'.")

    if isinstance(voice_prompt_input, str) and voice_prompt_input.endswith(".npz"):
        if os_for_patch.path.exists(voice_prompt_input): 
            logger.info(f"[Patched] Loading voice prompt from direct .npz path: {voice_prompt_input}")
            try:
                return np_for_patch.load(voice_prompt_input)
            except Exception as e:
                logger.error(f"[Patched] Failed to load direct .npz file '{voice_prompt_input}': {e}", exc_info=True)
                raise ValueError(f"Could not load .npz file: {voice_prompt_input}") from e
        else:
            try: 
                logger.info(f"[Patched] Direct .npz path not found. Attempting to download '{voice_prompt_input}' as filename from repo '{BARK_VOICE_PROMPT_REPO_ID}'.")
                cached_path = hf_hub_download(repo_id=BARK_VOICE_PROMPT_REPO_ID, filename=voice_prompt_input)
                return np_for_patch.load(cached_path)
            except Exception as e:
                logger.error(f"[Patched] Failed to load .npz '{voice_prompt_input}' (tried as direct path and as filename in '{BARK_VOICE_PROMPT_REPO_ID}'): {e}", exc_info=True)
                raise ValueError(f"Voice prompt .npz file not found/accessible: {voice_prompt_input}") from e
    
    elif isinstance(voice_prompt_input, dict):
        logger.debug("[Patched] Loading voice prompt from pre-loaded dict.")
        assert "semantic_prompt" in voice_prompt_input, "Missing 'semantic_prompt' in dict"
        assert "coarse_prompt" in voice_prompt_input, "Missing 'coarse_prompt' in dict"
        assert "fine_prompt" in voice_prompt_input, "Missing 'fine_prompt' in dict"
        return voice_prompt_input

    elif isinstance(voice_prompt_input, str):
        normalized_voice_name = os_for_patch.path.join(*voice_prompt_input.split(os_for_patch.path.sep))
        
        if hasattr(bark_pipeline_module_for_patch, 'ALLOWED_PROMPTS') and \
           normalized_voice_name not in bark_pipeline_module_for_patch.ALLOWED_PROMPTS:
            logger.warning(f"[Patched] Voice name '{normalized_voice_name}' is not in mlx-audio's original ALLOWED_PROMPTS set. Proceeding with load attempt based on NPY structure from '{BARK_VOICE_PROMPT_REPO_ID}'.")

        base_filename_stem = ""
        relative_npy_dir_in_voice_repo = "speaker_embeddings" 

        if normalized_voice_name == "announcer":
            base_filename_stem = "announcer"
        elif normalized_voice_name.startswith(f"v2{os_for_patch.path.sep}"):
            base_filename_stem = normalized_voice_name.split(os_for_patch.path.sep, 1)[1]
            relative_npy_dir_in_voice_repo = os_for_patch.path.join(relative_npy_dir_in_voice_repo, "v2")
        else:
            base_filename_stem = normalized_voice_name
            logger.info(f"[Patched] Non-v2 voice '{normalized_voice_name}'. Expecting .npy files under '{relative_npy_dir_in_voice_repo}' in '{BARK_VOICE_PROMPT_REPO_ID}'.")

        logger.debug(f"[Patched] Voice name parts: base_stem='{base_filename_stem}', relative_dir_in_voice_repo='{relative_npy_dir_in_voice_repo}'")

        prompt_key_to_file_suffix_map = {
            "semantic_prompt": "semantic_prompt.npy",
            "coarse_prompt": "coarse_prompt.npy",
            "fine_prompt": "fine_prompt.npy"
        }
        loaded_prompts_dict = {}

        try:
            for dict_key, npy_file_suffix in prompt_key_to_file_suffix_map.items():
                npy_filename_leaf = f"{base_filename_stem}_{npy_file_suffix}"
                path_to_npy_in_voice_repo = os_for_patch.path.join(relative_npy_dir_in_voice_repo, npy_filename_leaf)
                
                logger.info(f"[Patched] Attempting to download NPY: '{path_to_npy_in_voice_repo}' from repo '{BARK_VOICE_PROMPT_REPO_ID}'")
                
                cached_npy_path = hf_hub_download(
                    repo_id=BARK_VOICE_PROMPT_REPO_ID,
                    filename=path_to_npy_in_voice_repo
                )
                logger.info(f"[Patched] NPY '{path_to_npy_in_voice_repo}' found at cache: {cached_npy_path}")
                loaded_prompts_dict[dict_key] = np_for_patch.load(cached_npy_path)
            
            if not all(pt_key in loaded_prompts_dict for pt_key in prompt_key_to_file_suffix_map.keys()):
                missing_keys = [k for k in prompt_key_to_file_suffix_map.keys() if k not in loaded_prompts_dict]
                raise ValueError(f"Failed to load all required .npy components for voice '{voice_prompt_input}'. Missing for dict keys: {missing_keys}")
            
            logger.info(f"[Patched] Successfully loaded and combined NPY prompts for '{voice_prompt_input}' from '{BARK_VOICE_PROMPT_REPO_ID}'.")
            return loaded_prompts_dict

        except Exception as e:
            logger.error(f"[Patched] Failed to download or load .npy files for voice '{voice_prompt_input}' from '{BARK_VOICE_PROMPT_REPO_ID}' (tried base '{base_filename_stem}' in '{relative_npy_dir_in_voice_repo}'): {e}", exc_info=True)
            raise ValueError(f"Error processing NPY voice prompts for '{voice_prompt_input}'. Error: {e}") from e
    else:
        raise ValueError(f"Voice prompt format unrecognized by patch: {type(voice_prompt_input)}")

def apply_mlx_audio_bark_pipeline_patch_if_needed():
    global _mlx_bark_patch_applied_flag, _original_load_voice_prompt
    
    if not PATCH_IMPORTS_SUCCESSFUL:
        logger.warning("[Patch] Cannot apply Bark voice patch, essential modules for patch failed to import.")
        return

    if not _mlx_bark_patch_applied_flag:
        try:
            if bark_pipeline_module_for_patch and hasattr(bark_pipeline_module_for_patch, '_load_voice_prompt'):
                logger.info(f"Applying monkey patch to mlx_audio.tts.models.bark.pipeline._load_voice_prompt. Voice prompts WILL BE FETCHED FROM: {BARK_VOICE_PROMPT_REPO_ID}")
                _original_load_voice_prompt = bark_pipeline_module_for_patch._load_voice_prompt
                
                bark_pipeline_module_for_patch._load_voice_prompt = _patched_load_voice_prompt
                _mlx_bark_patch_applied_flag = True
                logger.info(f"Successfully applied monkey patch for Bark voice prompt loading in mlx-audio. Prompts source: {BARK_VOICE_PROMPT_REPO_ID}")
            else:
                logger.warning("Could not apply Bark voice prompt patch: mlx_audio.tts.models.bark.pipeline module or _load_voice_prompt attribute not found.")
        except Exception as e_patch:
            logger.error(f"Error during application of mlx-audio Bark voice prompt patch: {e_patch}", exc_info=True)

# --- End Monkey Patch Section ---

def _convert_mlx_audio_to_pcm_s16le_bytes(mlx_audio_array: 'mx_core_module.array', target_sample_rate: int) -> bytes:
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
    if not PYDUB_AVAILABLE_FOR_TRIM or not AudioSegment_pydub:
        logger.info(f"pydub not available, cannot trim reference audio: {ref_audio_path}. Using as is.")
        return ref_audio_path, None
    try:
        audio_segment = AudioSegment_pydub.from_file(str(ref_audio_path))
        if len(audio_segment) > max_duration_ms:
            logger.info(f"Reference audio '{ref_audio_path}' ({len(audio_segment)/1000.0:.1f}s) is > {max_duration_ms/1000.0:.1f}s. Trimming.")
            trimmed_segment = audio_segment[:max_duration_ms]
            temp_dir_for_trimmed_audio.mkdir(parents=True, exist_ok=True)
            temp_trimmed_filename = f"trimmed_ref_{ref_audio_path.stem}{ref_audio_path.suffix}"
            path_to_newly_trimmed_file = temp_dir_for_trimmed_audio / temp_trimmed_filename
            file_format = ref_audio_path.suffix.lstrip('.').lower() or 'wav'
            trimmed_segment.export(str(path_to_newly_trimmed_file), format=file_format)
            logger.info(f"Using trimmed temporary reference audio: {path_to_newly_trimmed_file}")
            return path_to_newly_trimmed_file, path_to_newly_trimmed_file
        else:
            logger.debug(f"Reference audio '{ref_audio_path}' is within length limits. Using original.")
            return ref_audio_path, None
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
    global _mlx_bark_patch_applied_flag 

    if not MLX_AUDIO_AVAILABLE or not generate_audio_mlx_func:
        logger.error("mlx-audio handler: library or generate_audio function not available. Skipping.")
        return

    mlx_model_repo_id_or_path = crisptts_model_config.get("mlx_model_path")
    if not mlx_model_repo_id_or_path:
        logger.error("mlx-audio: 'mlx_model_path' (HF repo ID or local path) not in config. Skipping.")
        return

    if "bark" in mlx_model_repo_id_or_path.lower(): 
        apply_mlx_audio_bark_pipeline_patch_if_needed()
    
    voice_input_for_mlx = voice_id_or_path_override or crisptts_model_config.get("default_voice_id")
    ref_audio_path_for_mlx_str = None
    voice_name_for_mlx_str = None 
                                  
    if voice_input_for_mlx:
        p_voice_input = Path(voice_input_for_mlx)
        is_likely_file_path = p_voice_input.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.npz'] or \
                              (p_voice_input.exists() and p_voice_input.is_file())

        if is_likely_file_path:
            ref_audio_path_actual = p_voice_input 
            if not p_voice_input.is_absolute():
                project_root = Path(__file__).resolve().parent.parent
                ref_audio_path_actual = (project_root / p_voice_input).resolve()

            if ref_audio_path_actual.exists():
                if ref_audio_path_actual.suffix.lower() == ".npz":
                    voice_name_for_mlx_str = str(ref_audio_path_actual)
                    logger.info(f"mlx-audio: Using direct path to NPZ voice prompt: {voice_name_for_mlx_str}")
                else: 
                    ref_audio_path_for_mlx_str = str(ref_audio_path_actual)
            else:
                logger.error(f"mlx-audio: Reference/voice file '{voice_input_for_mlx}' (resolved to '{ref_audio_path_actual}') not found. Skipping.")
                return
        else: 
            voice_name_for_mlx_str = str(voice_input_for_mlx)
    else:
        logger.warning("mlx-audio: No voice ID or reference audio path provided.")

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
        "ref_audio": None, 
        "ref_text": None,
    }
    if lang_code_for_mlx:
        mlx_gen_kwargs["lang_code"] = lang_code_for_mlx
    
    if model_params_override:
        try:
            cli_params_json = json.loads(model_params_override)
            valid_override_keys = [
                "speed", "temperature", "top_p", "top_k", "repetition_penalty", 
                "streaming_interval", "pitch", "gender", "stt_model", "lang_code", "ref_text"
            ]
            for key, value in cli_params_json.items():
                if key in valid_override_keys:
                    try:
                        if key in ["speed", "temperature", "top_p", "repetition_penalty", "streaming_interval", "pitch"]:
                            mlx_gen_kwargs[key] = float(value)
                        elif key == "top_k":
                             mlx_gen_kwargs[key] = int(value)
                        else: 
                            mlx_gen_kwargs[key] = str(value)
                    except ValueError:
                        logger.warning(f"mlx-audio: Could not convert param '{key}' value '{value}' to expected type. Using as string.")
                        mlx_gen_kwargs[key] = str(value) 
                else:
                    logger.warning(f"mlx-audio: Ignoring unknown parameter '{key}' from --model-params.")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"mlx-audio: Error parsing --model-params '{model_params_override}': {e}")

    generated_audio_data_bytes = None
    final_saved_path_str = None
    
    with tempfile.TemporaryDirectory(prefix="crisptts_mlx_handler_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        path_to_temp_trimmed_ref_to_delete = None 

        if ref_audio_path_for_mlx_str: 
            path_to_use_for_ref, path_to_temp_trimmed_ref_to_delete = _trim_ref_audio_if_needed(
                Path(ref_audio_path_for_mlx_str),
                max_duration_ms=15000, 
                temp_dir_for_trimmed_audio=temp_dir_path
            )
            mlx_gen_kwargs["ref_audio"] = str(path_to_use_for_ref)
            mlx_gen_kwargs["voice"] = None 
            logger.info(f"mlx-audio: Using reference audio for cloning: {mlx_gen_kwargs['ref_audio']}")
        elif voice_name_for_mlx_str:
             mlx_gen_kwargs["voice"] = voice_name_for_mlx_str 
             mlx_gen_kwargs["ref_audio"] = None 
             logger.info(f"mlx-audio: Using voice name/preset: {mlx_gen_kwargs['voice']}")
        
        temp_file_basename = "mlx_synth_output" 
        mlx_gen_kwargs["file_prefix"] = temp_file_basename 
        
        original_cwd = Path.cwd()
        os.chdir(temp_dir_path) 
        logger.debug(f"mlx-audio: Changed CWD to temp dir: {temp_dir_path}")

        logger.info(f"mlx-audio: Synthesizing with main model from '{mlx_model_repo_id_or_path}'...")
        logger.debug(f"mlx-audio: Calling generate_audio with effective kwargs: {json.dumps(mlx_gen_kwargs, default=str)}")
        
        try:
            # Call generate_audio_mlx_func. It will save the file to the CWD (temp_dir_path)
            # and print its own verbose output. It returns None.
            # No iteration here.
            generate_audio_mlx_func(model_path=mlx_model_repo_id_or_path, text=text, **mlx_gen_kwargs)
            
            # After the call, the file should exist in temp_dir_path (which was the CWD for mlx-audio)
            expected_temp_output_file = Path(f"{temp_file_basename}.{output_audio_format}") 
            
            if expected_temp_output_file.exists() and expected_temp_output_file.stat().st_size > 100:
                logger.info(f"mlx-audio: Successfully generated audio file in temp dir: {expected_temp_output_file.resolve()} (relative to temp CWD)")
                generated_audio_data_bytes = expected_temp_output_file.read_bytes()
            else:
                # This else block will be hit if mlx-audio's internal saving failed or produced empty file.
                # The verbose log from mlx-audio (like "Audio: 1 samples") might give clues.
                logger.error(f"mlx-audio: Synthesis for '{mlx_model_repo_id_or_path}' did not produce a usable file in the temp directory ('{expected_temp_output_file}'), or file was too small.")
                generated_audio_data_bytes = None

        except Exception as e:
            logger.error(f"mlx-audio: Synthesis process failed for model '{mlx_model_repo_id_or_path}': {e}", exc_info=True)
            generated_audio_data_bytes = None 
        finally:
            os.chdir(original_cwd) 
            logger.debug(f"mlx-audio: Restored CWD to: {original_cwd}")
            if path_to_temp_trimmed_ref_to_delete and path_to_temp_trimmed_ref_to_delete.exists():
                try:
                    path_to_temp_trimmed_ref_to_delete.unlink()
                    logger.debug(f"mlx-audio: Deleted temp trimmed reference: {path_to_temp_trimmed_ref_to_delete}")
                except OSError as e_del_trim:
                    logger.warning(f"mlx-audio: Could not delete temp trimmed ref '{path_to_temp_trimmed_ref_to_delete}': {e_del_trim}")
    
    if generated_audio_data_bytes:
        logger.info(f"mlx-audio: {len(generated_audio_data_bytes)} bytes of audio data retrieved.")
        if output_file_str:
            final_saved_path_obj = Path(output_file_str).with_suffix(f".{output_audio_format}")
            # Save the bytes we captured (which should be WAV format as per mlx-audio's saving)
            # Re-saving it ensures it's in the user's desired location.
            # If generated_audio_data_bytes is raw PCM, specify input_format.
            # Since mlx-audio saves as .wav, we are reading .wav bytes.
            try:
                with open(final_saved_path_obj, 'wb') as f_out:
                    f_out.write(generated_audio_data_bytes)
                logger.info(f"Audio saved to {final_saved_path_obj}")
                final_saved_path_str = str(final_saved_path_obj)
            except Exception as e_save:
                 logger.error(f"Error saving final audio to {final_saved_path_obj}: {e_save}", exc_info=True)
        
        if play_direct:
            # If we have a final saved path, play that. Otherwise, play from bytes.
            if final_saved_path_str and Path(final_saved_path_str).exists():
                play_audio(final_saved_path_str, is_path=True)
            else: # Play from bytes if not saved or save failed
                play_audio(generated_audio_data_bytes, is_path=False, input_format="wav_bytes", sample_rate=target_sample_rate)
    else:
        logger.warning(f"mlx-audio: No audio was generated or retrieved for model '{mlx_model_repo_id_or_path}'.")

    gc.collect()
    if MLX_AUDIO_AVAILABLE and mx_core_module and hasattr(mx_core_module, 'clear_cache'):
        try:
            mx_core_module.clear_cache()
            logger.debug("mlx-audio: Cleared MLX cache.")
        except Exception as e_mlx_clear:
            logger.warning(f"mlx-audio: Error clearing MLX cache: {e_mlx_clear}")