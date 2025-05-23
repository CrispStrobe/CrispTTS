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
from utils import save_audio, play_audio, SuppressOutput # Assuming these are in ../utils.py relative to handlers/

logger = logging.getLogger("CrispTTS.handlers.mlx_audio")

# --- Conditional Imports for mlx-audio and pydub ---
MLX_AUDIO_AVAILABLE = False
generate_audio_mlx_func = None
mx_core_module = None

PYDUB_AVAILABLE_FOR_TRIM = False
AudioSegment_pydub = None

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
        logger.warning("[Patch] Cannot apply, essential modules failed import.")
        return
    if not _mlx_bark_patch_applied_flag:
        try:
            if bark_pipeline_module_for_patch and hasattr(bark_pipeline_module_for_patch, '_load_voice_prompt'):
                if bark_pipeline_module_for_patch._load_voice_prompt is _patched_load_voice_prompt:
                    _mlx_bark_patch_applied_flag = True; return
                logger.info(f"Applying patch to mlx_audio Bark voice loading. Prompts from: {BARK_VOICE_PROMPT_REPO_ID}")
                _original_load_voice_prompt = bark_pipeline_module_for_patch._load_voice_prompt
                bark_pipeline_module_for_patch._load_voice_prompt = _patched_load_voice_prompt
                _mlx_bark_patch_applied_flag = True
            else: logger.warning("[Patch] Could not apply: _load_voice_prompt not found.")
        except Exception as e: logger.error(f"[Patch] Error applying: {e}", exc_info=True)
# --- End Monkey Patch Section ---

def _trim_ref_audio_if_needed(ref_audio_path: Path, max_duration_ms: int, temp_dir_for_trimmed_audio: Path) -> tuple[Path, Path | None]:
    # (Your existing _trim_ref_audio_if_needed function - seems fine)
    if not PYDUB_AVAILABLE_FOR_TRIM or not AudioSegment_pydub:
        logger.info(f"mlx-audio: pydub not available, cannot check/trim reference audio: {ref_audio_path}. Using as is.")
        return ref_audio_path, None
    try:
        logger.debug(f"mlx-audio: Checking/trimming ref audio '{ref_audio_path}' for max duration {max_duration_ms}ms.")
        audio_segment = AudioSegment_pydub.from_file(str(ref_audio_path))
        if len(audio_segment) > max_duration_ms:
            logger.info(f"mlx-audio: Reference audio '{ref_audio_path}' ({len(audio_segment)/1000.0:.1f}s) is > {max_duration_ms/1000.0:.1f}s. Trimming.")
            trimmed_segment = audio_segment[:max_duration_ms]
            temp_trimmed_filename = f"trimmed_ref_{ref_audio_path.stem}{ref_audio_path.suffix}"
            path_to_newly_trimmed_file = temp_dir_for_trimmed_audio / temp_trimmed_filename
            file_format = ref_audio_path.suffix.lstrip('.').lower() or 'wav'
            trimmed_segment.export(str(path_to_newly_trimmed_file), format=file_format)
            logger.info(f"mlx-audio: Using trimmed temporary reference audio: {path_to_newly_trimmed_file}")
            return path_to_newly_trimmed_file, path_to_newly_trimmed_file
        else:
            logger.debug(f"mlx-audio: Reference audio '{ref_audio_path}' is within length limits. Using original.")
            return ref_audio_path, None
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

    if "bark" in crisptts_specific_model_id.lower() or ("bark" in str(mlx_model_repo_id_or_path).lower()):
        apply_mlx_audio_bark_pipeline_patch_if_needed()

    # Determine effective reference audio path for cloning
    effective_ref_audio_path_str = voice_id_or_path_override or crisptts_model_config.get("default_voice_id")
    actual_ref_audio_path = None
    is_cloning_intent = False
    if effective_ref_audio_path_str:
        p = Path(effective_ref_audio_path_str)
        resolved_p = p if p.is_absolute() else (Path.cwd() / p).resolve()
        if resolved_p.exists() and resolved_p.is_file():
            actual_ref_audio_path = str(resolved_p)
            is_cloning_intent = True # If a valid ref audio file is found, intent is cloning
            logger.info(f"mlx-audio ({crisptts_specific_model_id}): Cloning intent. Using reference audio: {actual_ref_audio_path}")
        else:
            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Reference audio path '{resolved_p}' not found. Model may run in zero-shot mode if supported, or fail.")
            # If the config *requires* cloning (e.g. "dia_clone"), this could be an error state later.
            # For now, allow generate_audio to handle it.

    # Initialize kwargs for mlx-audio's generate_audio
    mlx_gen_kwargs = {
        "speed": crisptts_model_config.get("default_speed", 1.0),
        "temperature": crisptts_model_config.get("default_temperature", 0.7),
        "verbose": logger.isEnabledFor(logging.DEBUG),
        "play": False, "join_audio": True, "audio_format": "wav",
        "voice": None, "ref_audio": None, "ref_text": None,
    }
    if lang_code := crisptts_model_config.get("lang_code"):
        mlx_gen_kwargs["lang_code"] = lang_code

    # --- Parameter Overrides & Ref Text Handling ---
    custom_ref_text_from_params = None
    if model_params_override:
        try:
            cli_params_json = json.loads(model_params_override)
            custom_ref_text_from_params = cli_params_json.get("ref_text")

            valid_keys = ["speed", "temperature", "top_p", "top_k", "repetition_penalty",
                          "streaming_interval", "pitch", "gender", "stt_model", "lang_code",
                          "cfg_scale", "sentence_split_method"]
            for key, value in cli_params_json.items():
                if key == "ref_text": continue
                if key in valid_keys:
                    if key in ["speed", "temperature", "top_p", "repetition_penalty", "streaming_interval", "pitch", "cfg_scale"]:
                        mlx_gen_kwargs[key] = float(value)
                    elif key == "top_k": mlx_gen_kwargs[key] = int(value)
                    elif key == "sentence_split_method" and isinstance(value, str) and value.lower() in ["none", "null", "passthrough"]:
                        mlx_gen_kwargs[key] = None
                    else: mlx_gen_kwargs[key] = str(value)
                    logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Overrode kwarg '{key}' to '{mlx_gen_kwargs[key]}'")
        except Exception as e_parse:
            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Error parsing --model-params: {e_parse}")

    # Populate ref_audio, ref_text, or voice for mlx_gen_kwargs
    if is_cloning_intent and actual_ref_audio_path:
        mlx_gen_kwargs["ref_audio"] = actual_ref_audio_path # This might be trimmed later

        if custom_ref_text_from_params:
            mlx_gen_kwargs["ref_text"] = custom_ref_text_from_params
            logger.info(f"mlx-audio ({crisptts_specific_model_id}): Using ref_text from params: '{custom_ref_text_from_params[:100]}...'")
        else:
            whisper_model_id = crisptts_model_config.get("whisper_model_id_for_transcription")
            if whisper_model_id and TORCH_FOR_WHISPER_AVAILABLE and TRANSFORMERS_PIPELINE_FOR_WHISPER_AVAILABLE and TORCHAUDIO_FOR_WHISPER_AVAILABLE:
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): Transcribing ref_audio '{actual_ref_audio_path}' with Whisper: {whisper_model_id}")
                pt_device = "cpu"
                if torch_whisper_module.cuda.is_available(): pt_device = "cuda"
                elif hasattr(torch_whisper_module.backends, "mps") and torch_whisper_module.backends.mps.is_available(): pt_device = "mps"
                
                whisper_pipeline_device_arg = pt_device
                if pt_device == "cuda": whisper_pipeline_device_arg = 0 # transformers pipeline convention

                whisper_pipe_instance = None
                try:
                    whisper_pipe_instance = transformers_pipeline_whisper_func(
                        task="automatic-speech-recognition", model=whisper_model_id,
                        torch_dtype=torch_whisper_module.float16 if pt_device != "cpu" else torch_whisper_module.float32,
                        device=whisper_pipeline_device_arg, token=os.getenv("HF_TOKEN"), framework="pt"
                    )
                    
                    wf, sr = torchaudio_whisper_module.load(actual_ref_audio_path)
                    if sr != 16000: # Whisper expects 16kHz
                        resampler = torchaudio_whisper_module.transforms.Resample(orig_freq=sr, new_freq=16000)
                        wf = resampler(wf)
                    if wf.size(0) > 1: wf = torch_whisper_module.mean(wf, dim=0, keepdim=True)

                    lang_hint_for_whisper = crisptts_model_config.get("language_for_whisper") # e.g., "de"
                    transcription_kwargs = {"language": lang_hint_for_whisper} if lang_hint_for_whisper else {}
                    
                    with SuppressOutput(suppress_stderr=True, suppress_stdout=not logger.isEnabledFor(logging.DEBUG)):
                        transcription_result = whisper_pipe_instance(wf.squeeze(0).cpu().numpy(), generate_kwargs=transcription_kwargs)
                    
                    obtained_ref_text = transcription_result["text"].strip()
                    if obtained_ref_text:
                        mlx_gen_kwargs["ref_text"] = obtained_ref_text
                        logger.info(f"mlx-audio ({crisptts_specific_model_id}): Whisper transcription for ref_audio: '{obtained_ref_text}' (Lang hint: {lang_hint_for_whisper})")
                    else:
                        logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Whisper returned empty transcription. Using placeholder.")
                        mlx_gen_kwargs["ref_text"] = "Reference audio content." # Fallback
                except Exception as e_wspr:
                    logger.error(f"mlx-audio ({crisptts_specific_model_id}): Whisper transcription failed: {e_wspr}. Using placeholder.", exc_info=True)
                    mlx_gen_kwargs["ref_text"] = "Transcription failed." # Fallback
                finally:
                    del whisper_pipe_instance
                    if TORCH_FOR_WHISPER_AVAILABLE and torch_whisper_module.cuda.is_available():
                        torch_whisper_module.cuda.empty_cache()
            else:
                logger.warning(f"mlx-audio ({crisptts_specific_model_id}): No ref_text provided and Whisper not configured/available in handler. `mlx-audio` library might attempt its own transcription if model requires it (e.g., Dia).")
                # If ref_text is critical for Dia and not obtainable, it might still fail or use mlx-audio's internal.
                # Set to a generic placeholder if you want to avoid mlx-audio's internal transcription attempt completely.
                # mlx_gen_kwargs["ref_text"] = "Reference audio provided without explicit transcription."
    elif voice_name_for_mlx_str: # For models like Bark that use voice names/NPZ paths
        mlx_gen_kwargs["voice"] = voice_name_for_mlx_str
    # If neither cloning nor voice name, it's zero-shot for models that support it. ref_audio/ref_text/voice remain None.

    # --- Actual Synthesis ---
    generated_audio_data_bytes = None
    final_saved_path_str = None
    
    with tempfile.TemporaryDirectory(prefix="crisptts_mlx_handler_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        path_to_temp_trimmed_ref_to_delete = None

        if mlx_gen_kwargs.get("ref_audio"): # If ref_audio path is set (cloning)
            original_ref_audio_path_for_trim = Path(mlx_gen_kwargs["ref_audio"])
            # Dia expects 44.1kHz, max_duration could be specific to model.
            # Using a generic 15s for now as in other parts of your code.
            max_duration_ms_for_trim = 15000
            if "dia" in crisptts_specific_model_id.lower():
                # Dia might handle longer, or have its own internal limits when encoding.
                # For now, keeping the 15s trim consistent unless Dia has specific requirements.
                logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Dia model - considering ref audio length for DAC encoding.")

            path_to_use_for_ref, path_to_temp_trimmed_ref_to_delete = _trim_ref_audio_if_needed(
                original_ref_audio_path_for_trim,
                max_duration_ms=max_duration_ms_for_trim,
                temp_dir_for_trimmed_audio=temp_dir_path
            )
            mlx_gen_kwargs["ref_audio"] = str(path_to_use_for_ref) # Update with potentially trimmed path

        temp_file_basename = "mlx_synth_output"
        mlx_gen_kwargs["file_prefix"] = temp_file_basename

        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir_path) # mlx-audio generate_audio often saves in CWD
            final_kwargs_for_generate = {k: v for k, v in mlx_gen_kwargs.items() if v is not None or k == "sentence_split_method"}
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Calling generate_audio with text len {len(text)}, effective kwargs: {json.dumps(final_kwargs_for_generate, default=str)}")
            
            generate_audio_mlx_func(model_path=mlx_model_repo_id_or_path, text=text, **final_kwargs_for_generate)
            
            expected_temp_output_file = temp_dir_path / f"{temp_file_basename}.wav" # Assuming wav
            if expected_temp_output_file.exists() and expected_temp_output_file.stat().st_size > 100:
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): Successfully generated: {expected_temp_output_file}")
                generated_audio_data_bytes = expected_temp_output_file.read_bytes()
            else:
                logger.error(f"mlx-audio ({crisptts_specific_model_id}): Output file not created or empty in temp dir: {expected_temp_output_file}")
        except Exception as e_synth_mlx:
            logger.error(f"mlx-audio ({crisptts_specific_model_id}): Synthesis process failed: {e_synth_mlx}", exc_info=True)
        finally:
            os.chdir(original_cwd)
            if path_to_temp_trimmed_ref_to_delete and path_to_temp_trimmed_ref_to_delete.exists():
                try: path_to_temp_trimmed_ref_to_delete.unlink()
                except OSError as e: logger.warning(f"Could not delete temp trimmed ref: {e}")

    if generated_audio_data_bytes:
        # ... (your existing save and play logic using utils.save_audio, utils.play_audio) ...
        logger.info(f"mlx-audio ({crisptts_specific_model_id}): {len(generated_audio_data_bytes)} bytes of audio data retrieved.")
        if output_file_str:
            final_saved_path_obj = Path(output_file_str).with_suffix(f".{mlx_gen_kwargs.get('audio_format', 'wav')}")
            final_saved_path_obj.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(final_saved_path_obj, 'wb') as f_out: f_out.write(generated_audio_data_bytes)
                logger.info(f"Audio saved to {final_saved_path_obj}")
                final_saved_path_str = str(final_saved_path_obj)
            except Exception as e_save_final:
                 logger.error(f"mlx-audio ({crisptts_specific_model_id}): Error saving final audio: {e_save_final}", exc_info=True)
        
        if play_direct:
            # utils.play_audio expects bytes to be raw PCM or a full file format
            # Assuming generate_audio_mlx_func produces a full WAV file in bytes
            play_audio(generated_audio_data_bytes, is_path=False, input_format="wav_bytes", sample_rate=crisptts_model_config.get("sample_rate"))
    else:
        logger.warning(f"mlx-audio ({crisptts_specific_model_id}): No audio generated for '{mlx_model_repo_id_or_path}'.")

    if MLX_AUDIO_AVAILABLE and mx_core_module and hasattr(mx_core_module, 'clear_cache'):
        try: mx_core_module.clear_cache(); logger.debug("mlx-audio: Cleared MLX cache.")
        except Exception as e: logger.warning(f"mlx-audio: Error clearing MLX cache: {e}")
    gc.collect()
    logger.info(f"mlx-audio ({crisptts_specific_model_id}): Synthesis function finished.")