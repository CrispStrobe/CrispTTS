# CrispTTS/handlers/mlx_audio_handler.py
import logging
import platform
import os
from pathlib import Path
import json
import shutil # For copying files if needed, though current logic reads bytes
import tempfile
import gc
import numpy as np

# CrispTTS utils
from utils import save_audio, play_audio, SuppressOutput # Assuming these are in ../utils.py relative to handlers/

logger = logging.getLogger("CrispTTS.handlers.mlx_audio")

# --- Conditional Imports for mlx-audio and pydub ---
MLX_AUDIO_AVAILABLE = False
generate_audio_mlx_func = None
mx_core_module = None

PYDUB_AVAILABLE_FOR_TRIM = False
AudioSegment_pydub = None

logger_init = logging.getLogger("CrispTTS.handlers.mlx_audio.init") # Separate logger for init phase

try:
    from pydub import AudioSegment
    AudioSegment_pydub = AudioSegment
    PYDUB_AVAILABLE_FOR_TRIM = True
    logger_init.info("pydub imported successfully for MLX-Audio reference audio trimming.")
except ImportError:
    logger_init.info("pydub not found. Reference audio trimming for MLX-Audio will not be available.")

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

# --- Monkey Patch Section for mlx_audio.tts.models.bark.pipeline._load_voice_prompt ---
_original_load_voice_prompt = None
_mlx_bark_patch_applied_flag = False

BARK_VOICE_PROMPT_REPO_ID = "suno/bark-small" # Dedicated repository for voice prompts

PATCH_IMPORTS_SUCCESSFUL = False
try:
    # These imports are for the patch logic
    import mlx_audio.tts.models.bark.pipeline as bark_pipeline_module_for_patch
    from huggingface_hub import hf_hub_download
    import numpy as np_for_patch
    import os as os_for_patch # Aliased os for the patch
    PATCH_IMPORTS_SUCCESSFUL = True
    logger_init.debug("Successfully imported modules for mlx-audio Bark patch.")
except ImportError as e_patch_imp:
    logger_init.warning(f"Failed to import modules needed for mlx-audio Bark patch: {e_patch_imp}. Patch will not be applied if Bark is used.")


def _patched_load_voice_prompt(voice_prompt_input):
    # This function is the patched version, sourced from your previous input.
    # Logger calls within this function should use the main mlx_audio handler logger.
    logger.debug(f"[Patched _load_voice_prompt] Called with voice_prompt_input type: '{type(voice_prompt_input)}'. Voice prompts sourced from '{BARK_VOICE_PROMPT_REPO_ID}'.")

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
                logger.info(f"[Patched] Direct .npz path '{voice_prompt_input}' not found. Attempting download from repo '{BARK_VOICE_PROMPT_REPO_ID}'.")
                cached_path = hf_hub_download(repo_id=BARK_VOICE_PROMPT_REPO_ID, filename=voice_prompt_input) # voice_prompt_input is filename here
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

    elif isinstance(voice_prompt_input, str): # Is a voice name string
        normalized_voice_name = os_for_patch.path.join(*voice_prompt_input.split(os_for_patch.path.sep))
        logger.debug(f"[Patched] Processing voice name string: '{voice_prompt_input}', normalized: '{normalized_voice_name}'")

        if hasattr(bark_pipeline_module_for_patch, 'ALLOWED_PROMPTS') and \
           normalized_voice_name not in bark_pipeline_module_for_patch.ALLOWED_PROMPTS:
            logger.warning(f"[Patched] Voice name '{normalized_voice_name}' is not in mlx-audio's original ALLOWED_PROMPTS set. Proceeding with NPY load from '{BARK_VOICE_PROMPT_REPO_ID}'.")

        base_filename_stem = ""
        relative_npy_dir_in_voice_repo = "speaker_embeddings"

        if normalized_voice_name == "announcer": # Special case in original mlx-audio
            base_filename_stem = "announcer"
            # original mlx-audio expects announcer NPYs at the root of its internal assets,
            # but suno/bark-small has them under speaker_embeddings/
        elif normalized_voice_name.startswith(f"v2{os_for_patch.path.sep}"):
            base_filename_stem = normalized_voice_name.split(os_for_patch.path.sep, 1)[1]
            relative_npy_dir_in_voice_repo = os_for_patch.path.join(relative_npy_dir_in_voice_repo, "v2")
        else: # e.g. "en_speaker_0"
            base_filename_stem = normalized_voice_name
            # No change to relative_npy_dir_in_voice_repo needed for non-v2, non-announcer if they are in speaker_embeddings/ directly

        logger.debug(f"[Patched] Determined NPY lookup: base_stem='{base_filename_stem}', relative_dir_in_voice_repo='{relative_npy_dir_in_voice_repo}'")

        prompt_key_to_file_suffix_map = {
            "semantic_prompt": "semantic_prompt.npy",
            "coarse_prompt": "coarse_prompt.npy",
            "fine_prompt": "fine_prompt.npy"
        }
        loaded_prompts_dict = {}

        try:
            for dict_key, npy_file_suffix in prompt_key_to_file_suffix_map.items():
                # Construct the full path as expected in the suno/bark-small repo structure
                npy_filename_leaf = f"{base_filename_stem}_{npy_file_suffix}" if base_filename_stem else npy_file_suffix # Handle cases like just "semantic_prompt.npy" if base is empty
                if base_filename_stem == "announcer" and relative_npy_dir_in_voice_repo == "speaker_embeddings":
                     # Announcer files in suno/bark-small are directly under speaker_embeddings/
                     # e.g. speaker_embeddings/announcer_semantic_prompt.npy
                     # The original mlx-audio might look for just "announcer_semantic_prompt.npy" in its assets.
                     # Our patch aims to get them from the standard suno/bark-small structure.
                     path_to_npy_in_voice_repo = os_for_patch.path.join(relative_npy_dir_in_voice_repo, npy_filename_leaf)
                else:
                     path_to_npy_in_voice_repo = os_for_patch.path.join(relative_npy_dir_in_voice_repo, npy_filename_leaf)


                logger.info(f"[Patched] Attempting to download NPY: '{path_to_npy_in_voice_repo}' from repo '{BARK_VOICE_PROMPT_REPO_ID}' for key '{dict_key}'")

                cached_npy_path = hf_hub_download(
                    repo_id=BARK_VOICE_PROMPT_REPO_ID,
                    filename=path_to_npy_in_voice_repo # This must be the exact relative path in the HF repo
                )
                logger.info(f"[Patched] NPY '{path_to_npy_in_voice_repo}' downloaded to cache: {cached_npy_path}")
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
        logger.error(f"[Patched] Voice prompt format unrecognized: {type(voice_prompt_input)}. Expected str (path or name) or dict.")
        raise ValueError(f"Voice prompt format unrecognized by patch: {type(voice_prompt_input)}")

def apply_mlx_audio_bark_pipeline_patch_if_needed():
    global _mlx_bark_patch_applied_flag, _original_load_voice_prompt

    if not PATCH_IMPORTS_SUCCESSFUL:
        logger.warning("[Patch Application] Cannot apply Bark voice patch, essential modules for patch (e.g., bark_pipeline_module_for_patch, hf_hub_download) failed to import earlier.")
        return

    if not _mlx_bark_patch_applied_flag:
        try:
            if bark_pipeline_module_for_patch and hasattr(bark_pipeline_module_for_patch, '_load_voice_prompt'):
                if bark_pipeline_module_for_patch._load_voice_prompt is _patched_load_voice_prompt:
                    logger.debug("[Patch Application] Bark voice prompt patch already applied.")
                    _mlx_bark_patch_applied_flag = True # Ensure flag is set if somehow re-entered
                    return

                logger.info(f"Applying monkey patch to mlx_audio.tts.models.bark.pipeline._load_voice_prompt. Voice prompts WILL BE FETCHED FROM: {BARK_VOICE_PROMPT_REPO_ID}")
                _original_load_voice_prompt = bark_pipeline_module_for_patch._load_voice_prompt
                bark_pipeline_module_for_patch._load_voice_prompt = _patched_load_voice_prompt
                _mlx_bark_patch_applied_flag = True
                logger.info(f"Successfully applied monkey patch for Bark voice prompt loading in mlx-audio. Prompts source: {BARK_VOICE_PROMPT_REPO_ID}")
            else:
                logger.warning("Could not apply Bark voice prompt patch: mlx_audio.tts.models.bark.pipeline module or _load_voice_prompt attribute not found/accessible.")
        except Exception as e_patch:
            logger.error(f"Error during application of mlx-audio Bark voice prompt patch: {e_patch}", exc_info=True)
    else:
        logger.debug("[Patch Application] Bark voice prompt patch was already applied in a previous call.")

# --- End Monkey Patch Section ---

def _convert_mlx_audio_to_pcm_s16le_bytes(mlx_audio_array: 'mx_core_module.array', target_sample_rate: int) -> bytes:
    # This function seems unused if mlx-audio directly saves to WAV and we read bytes from that file.
    # Keeping it in case a future mlx-audio version returns raw arrays.
    if not MLX_AUDIO_AVAILABLE or not mx_core_module or not isinstance(mlx_audio_array, mx_core_module.array):
        logger.error("Cannot convert audio: MLX array or mx_core_module not available.")
        return b""
    try:
        logger.debug(f"Converting MLX audio array (shape {mlx_audio_array.shape}, dtype {mlx_audio_array.dtype}) to PCM S16LE bytes.")
        audio_np_float32 = np.array(mlx_audio_array.astype(mx_core_module.float32)) # Ensure float32 before scaling
        audio_np_int16 = (np.clip(audio_np_float32, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_np_int16.tobytes()
    except Exception as e:
        logger.error(f"Error converting MLX audio array to PCM bytes: {e}", exc_info=True)
        return b""

def _trim_ref_audio_if_needed(ref_audio_path: Path, max_duration_ms: int, temp_dir_for_trimmed_audio: Path) -> tuple[Path, Path | None]:
    if not PYDUB_AVAILABLE_FOR_TRIM or not AudioSegment_pydub:
        logger.info(f"mlx-audio: pydub not available, cannot check/trim reference audio: {ref_audio_path}. Using as is.")
        return ref_audio_path, None
    try:
        logger.debug(f"mlx-audio: Checking/trimming ref audio '{ref_audio_path}' for max duration {max_duration_ms}ms.")
        audio_segment = AudioSegment_pydub.from_file(str(ref_audio_path))
        if len(audio_segment) > max_duration_ms:
            logger.info(f"mlx-audio: Reference audio '{ref_audio_path}' ({len(audio_segment)/1000.0:.1f}s) is > {max_duration_ms/1000.0:.1f}s. Trimming.")
            trimmed_segment = audio_segment[:max_duration_ms]
            # temp_dir_for_trimmed_audio should already exist (created by with tempfile.TemporaryDirectory)
            temp_trimmed_filename = f"trimmed_ref_{ref_audio_path.stem}{ref_audio_path.suffix}"
            path_to_newly_trimmed_file = temp_dir_for_trimmed_audio / temp_trimmed_filename
            file_format = ref_audio_path.suffix.lstrip('.').lower() or 'wav' # Default to wav if no suffix
            trimmed_segment.export(str(path_to_newly_trimmed_file), format=file_format)
            logger.info(f"mlx-audio: Using trimmed temporary reference audio: {path_to_newly_trimmed_file}")
            return path_to_newly_trimmed_file, path_to_newly_trimmed_file # Return path to temp file and mark it for deletion
        else:
            logger.debug(f"mlx-audio: Reference audio '{ref_audio_path}' is within length limits. Using original.")
            return ref_audio_path, None # Return original path, no temp file to delete by this function
    except Exception as e:
        logger.warning(f"mlx-audio: Error processing/trimming reference audio '{ref_audio_path}': {e}. Using original path.", exc_info=True)
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

    crisptts_specific_model_id = crisptts_model_config.get('crisptts_model_id', 'mlx_audio_unknown') # For clearer logs
    logger.info(f"mlx-audio: Starting synthesis for model '{crisptts_specific_model_id}'.")
    logger.debug(f"mlx-audio: Input text (first 100 chars): '{text[:100]}...'")
    logger.debug(f"mlx-audio: Voice ID/Path override: '{voice_id_or_path_override}'")
    logger.debug(f"mlx-audio: Model params override: '{model_params_override}'")

    if not MLX_AUDIO_AVAILABLE or not generate_audio_mlx_func:
        logger.error(f"mlx-audio ({crisptts_specific_model_id}): Core library or generate_audio function not available. Skipping synthesis.")
        return

    mlx_model_repo_id_or_path = crisptts_model_config.get("mlx_model_path")
    if not mlx_model_repo_id_or_path:
        logger.error(f"mlx-audio ({crisptts_specific_model_id}): 'mlx_model_path' (HF repo ID or local path) not found in config. Skipping.")
        return
    logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Using mlx_model_path: '{mlx_model_repo_id_or_path}'")

    # Apply Bark patch specifically if a Bark model is identified
    # Model IDs for Bark in config.py are like "mlx_audio_bark_de"
    if "bark" in crisptts_specific_model_id.lower() or ("bark" in mlx_model_repo_id_or_path.lower()):
        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Bark model detected, ensuring patch is applied.")
        apply_mlx_audio_bark_pipeline_patch_if_needed()
    elif _mlx_bark_patch_applied_flag: # If patch was applied but this isn't Bark, it's fine, patch is specific.
        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Not a Bark model, but patch might have been applied earlier (this is generally OK).")


    voice_input_for_mlx = voice_id_or_path_override or crisptts_model_config.get("default_voice_id")
    ref_audio_path_for_mlx_str = None
    voice_name_for_mlx_str = None

    if voice_input_for_mlx:
        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Raw voice_input_for_mlx: '{voice_input_for_mlx}'")
        p_voice_input = Path(voice_input_for_mlx)
        # Check if it's an existing file path with a common audio/prompt suffix
        is_file_path_candidate = p_voice_input.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.npz']

        if is_file_path_candidate:
            # If it looks like a file path, resolve it absolutely
            resolved_voice_path = p_voice_input if p_voice_input.is_absolute() else (Path.cwd() / p_voice_input).resolve()
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Voice input '{voice_input_for_mlx}' resolved to '{resolved_voice_path}'")
            if resolved_voice_path.exists() and resolved_voice_path.is_file():
                if resolved_voice_path.suffix.lower() == ".npz": # Bark .npz voice prompt
                    voice_name_for_mlx_str = str(resolved_voice_path)
                    logger.info(f"mlx-audio ({crisptts_specific_model_id}): Using direct path to NPZ voice prompt: {voice_name_for_mlx_str}")
                else: # Other audio file for cloning
                    ref_audio_path_for_mlx_str = str(resolved_voice_path)
                    logger.info(f"mlx-audio ({crisptts_specific_model_id}): Identified as audio file for cloning: {ref_audio_path_for_mlx_str}")
            else:
                # If it looked like a file but doesn't exist, treat as a voice name
                logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Voice input '{voice_input_for_mlx}' looked like a file path but not found at '{resolved_voice_path}'. Treating as a voice name/preset.")
                voice_name_for_mlx_str = str(voice_input_for_mlx)
        else: # Not a recognized file suffix, treat as voice name/preset
            voice_name_for_mlx_str = str(voice_input_for_mlx)
            logger.info(f"mlx-audio ({crisptts_specific_model_id}): Using as voice name/preset: {voice_name_for_mlx_str}")
    else:
        logger.warning(f"mlx-audio ({crisptts_specific_model_id}): No voice ID or reference audio path specified or defaulted. Some models may require this.")

    lang_code_for_mlx = crisptts_model_config.get("lang_code")
    target_sample_rate = crisptts_model_config.get("sample_rate", 24000)
    output_audio_format = "wav" # mlx-audio generate_audio saves as WAV by default

    # Default kwargs for mlx-audio's generate_audio
    mlx_gen_kwargs = {
        "speed": crisptts_model_config.get("default_speed", 1.0),
        "temperature": crisptts_model_config.get("default_temperature", 0.7), # Will be overridden by model_params if specified
        "verbose": logger.isEnabledFor(logging.DEBUG), # mlx-audio's verbosity tied to our DEBUG level
        "play": False, # We handle playback with our own utils
        "join_audio": True, # For multi-sentence inputs, join into one file
        "audio_format": output_audio_format,
        "voice": None, # Will be set if voice_name_for_mlx_str is determined
        "ref_audio": None, # Will be set if ref_audio_path_for_mlx_str is determined
        "ref_text": None, # Default to None, populate if ref_audio is used and ref_text is available
    }

    if lang_code_for_mlx:
        mlx_gen_kwargs["lang_code"] = lang_code_for_mlx
        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Using lang_code: {lang_code_for_mlx}")


    # Populate voice or ref_audio for mlx_gen_kwargs
    if ref_audio_path_for_mlx_str:
        mlx_gen_kwargs["ref_audio"] = ref_audio_path_for_mlx_str # This will be path to original or trimmed temp file
        mlx_gen_kwargs["voice"] = None # Explicitly None if ref_audio is used
        logger.info(f"mlx-audio ({crisptts_specific_model_id}): Will use reference audio for cloning: {ref_audio_path_for_mlx_str}")
        # Set ref_text if cloning
        default_ref_text_from_config = crisptts_model_config.get("default_ref_text")
        if default_ref_text_from_config:
            mlx_gen_kwargs["ref_text"] = default_ref_text_from_config
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Using default_ref_text from config: '{default_ref_text_from_config[:50]}...'")
    elif voice_name_for_mlx_str:
        mlx_gen_kwargs["voice"] = voice_name_for_mlx_str
        mlx_gen_kwargs["ref_audio"] = None
        logger.info(f"mlx-audio ({crisptts_specific_model_id}): Using voice name/preset: {voice_name_for_mlx_str}")


    # Apply CLI --model-params overrides
    if model_params_override:
        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Applying model_params_override: {model_params_override}")
        try:
            cli_params_json = json.loads(model_params_override)
            # Expanded list of keys that mlx-audio's generic generate_audio might accept and pass through,
            # or that influence its behavior for Dia.
            valid_override_keys = [
                "speed", "temperature", "top_p", "top_k", "repetition_penalty",
                "streaming_interval", "pitch", "gender", "stt_model", "lang_code",
                "ref_text",
                "cfg_scale",  # Crucial for Dia
                "sentence_split_method" # To control how text is passed to Dia
                # Add other relevant params if mlx-audio's generate_audio supports them for Dia
            ]
            for key, value in cli_params_json.items():
                if key in valid_override_keys:
                    try:
                        if key in ["speed", "temperature", "top_p", "repetition_penalty", "streaming_interval", "pitch", "cfg_scale"]:
                            mlx_gen_kwargs[key] = float(value)
                        elif key == "top_k":
                            mlx_gen_kwargs[key] = int(value)
                        elif key == "sentence_split_method" and isinstance(value, str) and value.lower() in ["none", "null", "passthrough"]:
                            # Assuming "none", "null", or "passthrough" means disable mlx-audio's splitting
                            mlx_gen_kwargs[key] = None # Or the specific value generate_audio expects for no splitting
                            logger.info(f"mlx-audio ({crisptts_specific_model_id}): Setting sentence_split_method to None/passthrough for Dia.")
                        else: # string parameters
                            mlx_gen_kwargs[key] = str(value)
                        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Overrode kwarg '{key}' to '{mlx_gen_kwargs[key]}'")
                    except ValueError:
                        if key == "sentence_split_method" and value is None:
                             mlx_gen_kwargs[key] = None
                             logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Overrode kwarg '{key}' to None")
                        else:
                            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Could not convert param '{key}' value '{value}' to expected type. Using as string: '{str(value)}'.")
                            mlx_gen_kwargs[key] = str(value)
                else:
                    logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Ignoring unknown parameter '{key}' from --model-params.")
        except (json.JSONDecodeError, TypeError) as e_parse:
            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Error parsing --model-params JSON '{model_params_override}': {e_parse}")

    generated_audio_data_bytes = None
    final_saved_path_str = None
    path_to_temp_trimmed_ref_to_delete_final = None

    with tempfile.TemporaryDirectory(prefix="crisptts_mlx_handler_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Created temporary directory: {temp_dir_path}")

        # If ref_audio is a path, trim it if necessary into the temp_dir
        if mlx_gen_kwargs.get("ref_audio"):
            original_ref_audio_path = Path(mlx_gen_kwargs["ref_audio"])
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Preparing ref_audio '{original_ref_audio_path}' for mlx-audio call.")
            path_to_use_for_ref, path_to_temp_trimmed_ref_to_delete_final = _trim_ref_audio_if_needed(
                original_ref_audio_path,
                max_duration_ms=15000, # Example max duration, adjust if needed
                temp_dir_for_trimmed_audio=temp_dir_path
            )
            mlx_gen_kwargs["ref_audio"] = str(path_to_use_for_ref) # Update kwarg to point to potentially trimmed file
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Effective ref_audio path for generate_audio: {mlx_gen_kwargs['ref_audio']}")

        temp_file_basename = "mlx_synth_output"
        mlx_gen_kwargs["file_prefix"] = temp_file_basename # mlx-audio will save as temp_file_basename.wav

        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir_path)
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Changed CWD to temp dir: {temp_dir_path}")

            logger.info(f"mlx-audio ({crisptts_specific_model_id}): Synthesizing with main model from '{mlx_model_repo_id_or_path}'.")
            # Filter out None values from kwargs to prevent errors with generate_audio if some optional args are not set
            final_mlx_gen_kwargs = {k: v for k, v in mlx_gen_kwargs.items() if v is not None}
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Calling generate_audio with effective kwargs: {json.dumps(final_mlx_gen_kwargs, default=str)}")

            # mlx-audio's generate_audio is not async and handles its own printing
            generate_audio_mlx_func(model_path=mlx_model_repo_id_or_path, text=text, **final_mlx_gen_kwargs)

            expected_temp_output_file = temp_dir_path / f"{temp_file_basename}.{output_audio_format}"
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Checking for output file at: {expected_temp_output_file}")

            if expected_temp_output_file.exists() and expected_temp_output_file.stat().st_size > 100: # Basic check for non-empty file
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): Successfully generated audio file in temp dir: {expected_temp_output_file}")
                generated_audio_data_bytes = expected_temp_output_file.read_bytes()
            else:
                logger.error(f"mlx-audio ({crisptts_specific_model_id}): Synthesis did not produce a usable file in the temp directory ('{expected_temp_output_file}'), or file was too small. Check mlx-audio's own logs.")
        except Exception as e_synth_mlx:
            logger.error(f"mlx-audio ({crisptts_specific_model_id}): Synthesis process failed: {e_synth_mlx}", exc_info=True)
        finally:
            os.chdir(original_cwd)
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Restored CWD to: {original_cwd}")
            if path_to_temp_trimmed_ref_to_delete_final and path_to_temp_trimmed_ref_to_delete_final.exists():
                try:
                    path_to_temp_trimmed_ref_to_delete_final.unlink()
                    logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Deleted temp trimmed reference: {path_to_temp_trimmed_ref_to_delete_final}")
                except OSError as e_del_trim:
                    logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Could not delete temp trimmed ref '{path_to_temp_trimmed_ref_to_delete_final}': {e_del_trim}")
            # The entire temp_dir_path will be cleaned up automatically by TemporaryDirectory context manager

    if generated_audio_data_bytes:
        logger.info(f"mlx-audio ({crisptts_specific_model_id}): {len(generated_audio_data_bytes)} bytes of audio data retrieved.")
        if output_file_str:
            final_saved_path_obj = Path(output_file_str).with_suffix(f".{output_audio_format}")
            final_saved_path_obj.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(final_saved_path_obj, 'wb') as f_out:
                    f_out.write(generated_audio_data_bytes)
                logger.info(f"Audio saved to {final_saved_path_obj}") # Use utils.save_audio if conversion is needed
                final_saved_path_str = str(final_saved_path_obj)
            except Exception as e_save_final:
                 logger.error(f"mlx-audio ({crisptts_specific_model_id}): Error saving final audio to {final_saved_path_obj}: {e_save_final}", exc_info=True)
        else: # No output_file_str, but we have bytes (e.g., for direct play or test mode without specific file saving)
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Audio generated but no output_file_str provided for saving outside temp.")


        if play_direct and save_audio_util and play_audio_util : # Check if utils are available
            if final_saved_path_str and Path(final_saved_path_str).exists():
                play_audio_util(final_saved_path_str, is_path=True)
            else: # Play from bytes if not saved or save failed (input_format="wav_bytes" assumes bytes are a full wav file)
                play_audio_util(generated_audio_data_bytes, is_path=False, input_format="wav_bytes", sample_rate=target_sample_rate)
        elif play_direct:
            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Play direct requested but utils (play_audio) not available.")

    else:
        logger.warning(f"mlx-audio ({crisptts_specific_model_id}): No audio was generated or retrieved for model '{mlx_model_repo_id_or_path}'.")

    if MLX_AUDIO_AVAILABLE and mx_core_module and hasattr(mx_core_module, 'clear_cache'):
        try:
            mx_core_module.clear_cache() # Clear MLX's internal cache
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Cleared MLX cache.")
        except Exception as e_mlx_clear:
            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Error clearing MLX cache: {e_mlx_clear}")
    gc.collect()
    logger.info(f"mlx-audio ({crisptts_specific_model_id}): Synthesis function finished.")