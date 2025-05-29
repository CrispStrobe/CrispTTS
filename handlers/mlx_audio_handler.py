# CrispTTS/handlers/mlx_audio_handler.py
import logging
import platform
import os
from pathlib import Path
import json
import tempfile
import gc
import numpy as np
import shutil # Import shutil for cleaning up temp dirs if needed

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
BARK_VOICE_PROMPT_REPO_ID = "suno/bark-small" #
PATCH_IMPORTS_SUCCESSFUL = False
try:
    import mlx_audio.tts.models.bark.pipeline as bark_pipeline_module_for_patch
    from huggingface_hub import hf_hub_download
    # numpy as np_for_patch is already imported as np
    # os as os_for_patch is already imported as os
    PATCH_IMPORTS_SUCCESSFUL = True
    logger_init.debug("Successfully imported modules for mlx-audio Bark patch.")
except ImportError as e_patch_imp:
    logger_init.warning(f"Failed to import modules needed for mlx-audio Bark patch: {e_patch_imp}. Patch will not be applied if Bark is used.")

def _patched_load_voice_prompt(voice_prompt_input): #
    logger.debug(f"[Patched _load_voice_prompt] Called. Voice prompts from '{BARK_VOICE_PROMPT_REPO_ID}'.") #
    if isinstance(voice_prompt_input, str) and voice_prompt_input.endswith(".npz"): #
        if os.path.exists(voice_prompt_input): #
            logger.info(f"[Patched] Loading direct .npz: {voice_prompt_input}") #
            try: return np.load(voice_prompt_input) #
            except Exception as e: raise ValueError(f"Could not load .npz: {voice_prompt_input}") from e #
        else:
            try:
                logger.info(f"[Patched] Downloading .npz '{voice_prompt_input}' from '{BARK_VOICE_PROMPT_REPO_ID}'.") #
                cached_path = hf_hub_download(repo_id=BARK_VOICE_PROMPT_REPO_ID, filename=voice_prompt_input) #
                return np.load(cached_path) #
            except Exception as e: raise ValueError(f"Failed to load/download .npz: {voice_prompt_input}") from e #
    elif isinstance(voice_prompt_input, dict): #
        assert all(k in voice_prompt_input for k in ["semantic_prompt", "coarse_prompt", "fine_prompt"]) #
        return voice_prompt_input #
    elif isinstance(voice_prompt_input, str): #
        normalized_voice_name = os.path.join(*voice_prompt_input.split(os.path.sep)) #
        if hasattr(bark_pipeline_module_for_patch, 'ALLOWED_PROMPTS') and \
           normalized_voice_name not in bark_pipeline_module_for_patch.ALLOWED_PROMPTS: #
            logger.warning(f"[Patched] Voice '{normalized_voice_name}' not in ALLOWED_PROMPTS. Using NPY load from '{BARK_VOICE_PROMPT_REPO_ID}'.") #
        
        base_stem = "" #
        relative_dir = "speaker_embeddings" #
        if normalized_voice_name == "announcer": base_stem = "announcer" #
        elif normalized_voice_name.startswith(f"v2{os.path.sep}"): #
            base_stem = normalized_voice_name.split(os.path.sep, 1)[1] #
            relative_dir = os.path.join(relative_dir, "v2") #
        else: base_stem = normalized_voice_name #
        
        prompts_dict = {} #
        try:
            for key, suffix in {"semantic_prompt": "semantic_prompt.npy", "coarse_prompt": "coarse_prompt.npy", "fine_prompt": "fine_prompt.npy"}.items(): #
                filename = f"{base_stem}_{suffix}" if base_stem else suffix #
                path_in_repo = os.path.join(relative_dir, filename) #
                logger.info(f"[Patched] Downloading NPY: '{path_in_repo}' from '{BARK_VOICE_PROMPT_REPO_ID}' for '{key}'") #
                cached_path = hf_hub_download(repo_id=BARK_VOICE_PROMPT_REPO_ID, filename=path_in_repo) #
                prompts_dict[key] = np.load(cached_path) #
            if not all(k in prompts_dict for k in ["semantic_prompt", "coarse_prompt", "fine_prompt"]): #
                raise ValueError("Missing NPY components.") #
            return prompts_dict #
        except Exception as e: raise ValueError(f"Error processing NPY prompts for '{voice_prompt_input}': {e}") from e #
    raise ValueError(f"Voice prompt format unrecognized: {type(voice_prompt_input)}") #


def apply_mlx_audio_bark_pipeline_patch_if_needed(): #
    global _mlx_bark_patch_applied_flag, _original_load_voice_prompt #
    if not PATCH_IMPORTS_SUCCESSFUL: #
        logger.warning("[Patch] Cannot apply Bark patch, essential modules failed import.") #
        return #
    if not _mlx_bark_patch_applied_flag: #
        try:
            if bark_pipeline_module_for_patch and hasattr(bark_pipeline_module_for_patch, '_load_voice_prompt'): #
                if bark_pipeline_module_for_patch._load_voice_prompt is _patched_load_voice_prompt: #
                    logger.debug("Bark patch already applied.") #
                    _mlx_bark_patch_applied_flag = True; return #
                logger.info(f"Applying patch to mlx_audio Bark voice loading. Prompts from: {BARK_VOICE_PROMPT_REPO_ID}") #
                _original_load_voice_prompt = bark_pipeline_module_for_patch._load_voice_prompt #
                bark_pipeline_module_for_patch._load_voice_prompt = _patched_load_voice_prompt #
                _mlx_bark_patch_applied_flag = True #
            else: logger.warning("[Patch] Could not apply Bark patch: _load_voice_prompt not found in mlx_audio.tts.models.bark.pipeline.") #
        except Exception as e: logger.error(f"[Patch] Error applying Bark patch: {e}", exc_info=True) #
# --- End Monkey Patch Section ---

def _detect_audio_format_pydub_inline(audio_path_str: str) -> str | None: #
    if not PYDUB_AVAILABLE_FOR_TRIM or not pydub_mediainfo_func: #
        logger.warning("mlx-audio: pydub.utils.mediainfo not available for format detection.") #
        ext = Path(audio_path_str).suffix.lower().lstrip('.') #
        return ext if ext else None #
    try:
        info = pydub_mediainfo_func(audio_path_str) #
        format_name = info.get('format_name', '') #
        if any(fmt in format_name for fmt in ['mp4', 'm4a']): return 'mp4' #
        if 'mp3' in format_name: return 'mp3' #
        if 'ogg' in format_name: return 'ogg' #
        if 'flac' in format_name: return 'flac' #
        if 'wav' in format_name: return 'wav' #
        if 'webm' in format_name: return 'webm' #
        return format_name.split(',')[0].strip() if format_name else Path(audio_path_str).suffix.lower().lstrip('.') #
    except Exception as e: #
        logger.error(f"mlx-audio: pydub mediainfo error for {audio_path_str}: {e}. Falling back to extension.", exc_info=False) #
        ext = Path(audio_path_str).suffix.lower().lstrip('.') #
        return ext if ext else None #

def _convert_audio_for_whisper_inline(audio_path_str: str, target_sr: int = 16000) -> tuple[Path | None, str | None]: #
    audio_path = Path(audio_path_str) #
    if not audio_path.exists(): return None, f"Audio file not found: {audio_path_str}" #
    temp_wav_file_obj = None #
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', prefix="crisptts_whisper_prep_", delete=False) as tmp_f: #
            temp_wav_path_str = tmp_f.name #
        temp_wav_file_obj = Path(temp_wav_path_str) #
        if PYDUB_AVAILABLE_FOR_TRIM and AudioSegment_pydub: #
            logger.debug(f"mlx-audio: Converting '{audio_path}' for Whisper using pydub.") #
            audio_format = _detect_audio_format_pydub_inline(str(audio_path)) or audio_path.suffix.lstrip(".").lower() #
            if not audio_format: raise ValueError(f"Could not determine audio format for '{audio_path_str}'.") #
            sound = AudioSegment_pydub.from_file(str(audio_path), format=audio_format) #
            sound = sound.set_channels(1).set_frame_rate(target_sr).set_sample_width(2) #
            sound.export(str(temp_wav_file_obj), format="wav") #
        elif TORCHAUDIO_FOR_WHISPER_AVAILABLE and torchaudio_whisper_module and torch_whisper_module: #
            logger.debug(f"mlx-audio: Converting '{audio_path}' for Whisper using torchaudio.") #
            waveform, sr = torchaudio_whisper_module.load(str(audio_path)) #
            if sr != target_sr: #
                resampler = torchaudio_whisper_module.transforms.Resample(orig_freq=sr, new_freq=target_sr) #
                waveform = resampler(waveform) #
            if waveform.size(0) > 1: waveform = torch_whisper_module.mean(waveform, dim=0, keepdim=True) #
            torchaudio_whisper_module.save(str(temp_wav_file_obj), waveform, sample_rate=target_sr, bits_per_sample=16) #
        else:
            if temp_wav_file_obj and temp_wav_file_obj.exists(): temp_wav_file_obj.unlink(missing_ok=True) #
            return None, "Neither pydub nor torchaudio for Whisper audio conversion." #
        logger.info(f"mlx-audio: Converted '{audio_path_str}' to temp WAV for Whisper: {temp_wav_file_obj}") #
        return temp_wav_file_obj, None #
    except Exception as e: #
        error_msg = f"Audio conversion for Whisper failed for '{audio_path_str}': {e}" #
        logger.error(f"mlx-audio: {error_msg}", exc_info=True) #
        if temp_wav_file_obj and temp_wav_file_obj.exists(): #
            try: temp_wav_file_obj.unlink(missing_ok=True) #
            except OSError: pass #
        return None, error_msg #

def _transcribe_ref_audio_for_mlx(
    audio_path_str: str, whisper_model_id: str, hf_token: str | None,
    target_device_str: str, language_hint: str | None = None
) -> tuple[str | None, str | None]: #
    if not all([TORCH_FOR_WHISPER_AVAILABLE, TORCHAUDIO_FOR_WHISPER_AVAILABLE, TRANSFORMERS_PIPELINE_FOR_WHISPER_AVAILABLE]): #
        return None, "Whisper dependencies not available in mlx_audio_handler." #
    if not Path(audio_path_str).exists(): #
        return None, f"Ref audio for transcription not found: {audio_path_str}" #

    logger.info(f"mlx-audio: Transcribing '{audio_path_str}' with Whisper '{whisper_model_id}'. Device: {target_device_str}, Lang: {language_hint}") #
    temp_wav_for_whisper, prep_err = _convert_audio_for_whisper_inline(audio_path_str) #
    if prep_err or not temp_wav_for_whisper: #
        return None, f"Audio prep for Whisper failed: {prep_err}" #

    whisper_pipe, transcribed_text, err_msg = None, None, None #
    try:
        pipe_device = target_device_str if target_device_str != "cuda" else 0 #
        whisper_pipe = transformers_pipeline_whisper_func( #
            task="automatic-speech-recognition", model=whisper_model_id, #
            torch_dtype=torch_whisper_module.float16 if target_device_str != "cpu" else torch_whisper_module.float32, #
            device=pipe_device, token=hf_token, framework="pt" #
        )
        gen_kwargs = {"return_timestamps": True} #
        if language_hint: gen_kwargs["language"] = language_hint.lower() #
        
        with SuppressOutput(suppress_stderr=True, suppress_stdout=not logger.isEnabledFor(logging.DEBUG)): #
            result = whisper_pipe(str(temp_wav_for_whisper), generate_kwargs=gen_kwargs) #
        
        full_text = result.get("text", "").strip() #
        if not full_text and "chunks" in result and isinstance(result["chunks"], list): #
            full_text = " ".join([chunk.get('text',"").strip() for chunk in result["chunks"]]).strip() #
        if full_text: #
            transcribed_text = full_text #
            logger.info(f"mlx-audio: Whisper transcription (first 100): '{transcribed_text[:100]}...'") #
        else: err_msg = "Whisper returned empty transcription." #
    except Exception as e: #
        err_msg = f"Whisper pipeline failed: {e}" #
        logger.error(f"mlx-audio: {err_msg}", exc_info=True) #
    finally: #
        if temp_wav_for_whisper and temp_wav_for_whisper.exists(): #
            try: temp_wav_for_whisper.unlink() #
            except OSError as e_del: logger.warning(f"mlx-audio: Could not delete temp Whisper WAV {temp_wav_for_whisper}: {e_del}") #
        del whisper_pipe #
        if TORCH_FOR_WHISPER_AVAILABLE and torch_whisper_module and target_device_str != "cpu": #
            if torch_whisper_module.cuda.is_available(): torch_whisper_module.cuda.empty_cache() #
            elif hasattr(torch_whisper_module.backends, "mps") and torch_whisper_module.backends.mps.is_available() and hasattr(torch_whisper_module.mps, "empty_cache"): #
                try: torch_whisper_module.mps.empty_cache() #
                except Exception: pass #
        gc.collect() #
    return transcribed_text, err_msg #

def _trim_ref_audio_if_needed(ref_audio_path: Path, max_duration_ms: int, temp_dir_for_trimmed_audio: Path) -> tuple[Path, Path | None]: #
    if not PYDUB_AVAILABLE_FOR_TRIM or not AudioSegment_pydub: #
        logger.info(f"mlx-audio: pydub not available, cannot trim ref audio: {ref_audio_path}.") #
        return ref_audio_path, None #
    try:
        audio_segment = AudioSegment_pydub.from_file(str(ref_audio_path)) #
        if len(audio_segment) > max_duration_ms: #
            trimmed_segment = audio_segment[:max_duration_ms] #
            temp_trimmed_filename = f"trimmed_ref_{ref_audio_path.stem}{ref_audio_path.suffix}" #
            path_to_newly_trimmed_file = temp_dir_for_trimmed_audio / temp_trimmed_filename #
            file_format = ref_audio_path.suffix.lstrip('.').lower() or 'wav' #
            trimmed_segment.export(str(path_to_newly_trimmed_file), format=file_format) #
            logger.info(f"mlx-audio: Trimmed ref audio from {len(audio_segment)/1000.0:.1f}s to {len(trimmed_segment)/1000.0:.1f}s. Using: {path_to_newly_trimmed_file}") #
            return path_to_newly_trimmed_file, path_to_newly_trimmed_file #
        return ref_audio_path, None #
    except Exception as e: #
        logger.warning(f"mlx-audio: Error trimming ref audio '{ref_audio_path}': {e}. Using original.", exc_info=True) #
        return ref_audio_path, None #

def synthesize_with_mlx_audio(
    crisptts_model_config: dict,
    text: str,
    voice_id_or_path_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
): #
    global _mlx_bark_patch_applied_flag #

    crisptts_specific_model_id = crisptts_model_config.get('crisptts_model_id', 'mlx_audio_unknown') #
    logger.info(f"mlx-audio: Starting synthesis for model '{crisptts_specific_model_id}'.") #

    if not MLX_AUDIO_AVAILABLE or not generate_audio_mlx_func: #
        logger.error(f"mlx-audio ({crisptts_specific_model_id}): mlx-audio library or generate_audio function not available. Skipping.") #
        return #

    mlx_model_repo_id_or_path = crisptts_model_config.get("mlx_model_path") #
    if not mlx_model_repo_id_or_path: #
        logger.error(f"mlx-audio ({crisptts_specific_model_id}): 'mlx_model_path' not in config. Skipping.") #
        return #

    if "bark" in crisptts_specific_model_id.lower() or ("bark" in str(mlx_model_repo_id_or_path).lower()): #
        apply_mlx_audio_bark_pipeline_patch_if_needed() #

    effective_voice_input_str = voice_id_or_path_override or crisptts_model_config.get("default_voice_id") #
    actual_ref_audio_path_for_cloning = None #
    voice_name_or_npz_path_for_mlx = None #
    is_cloning_intent = False #

    if effective_voice_input_str: #
        p_input = Path(effective_voice_input_str) #
        resolved_p_input = p_input if p_input.is_absolute() else (Path.cwd() / p_input).resolve() #
        if resolved_p_input.exists() and resolved_p_input.is_file(): #
            if resolved_p_input.suffix.lower() == ".npz": #
                voice_name_or_npz_path_for_mlx = str(resolved_p_input) #
            else: #
                actual_ref_audio_path_for_cloning = str(resolved_p_input) #
                is_cloning_intent = True #
        else: #
            voice_name_or_npz_path_for_mlx = effective_voice_input_str # Treat as name if not a file #
    
    logger.info(f"mlx-audio ({crisptts_specific_model_id}): Cloning intent: {is_cloning_intent}. Voice/NPZ: '{voice_name_or_npz_path_for_mlx}'. Ref audio for cloning: '{actual_ref_audio_path_for_cloning}'.") #

    mlx_gen_kwargs = { #
        "speed": crisptts_model_config.get("default_speed", 1.0), #
        "temperature": crisptts_model_config.get("default_temperature", 0.7), #
        "verbose": logger.isEnabledFor(logging.DEBUG), "play": False, "join_audio": True, "audio_format": "wav", #
        "voice": voice_name_or_npz_path_for_mlx, "ref_audio": None, "ref_text": None, #
    }
    if lang_code := crisptts_model_config.get("lang_code"): mlx_gen_kwargs["lang_code"] = lang_code #
    if pitch_val := crisptts_model_config.get("default_pitch"): mlx_gen_kwargs["pitch"] = float(pitch_val) #

    custom_ref_text_from_params = None #
    if model_params_override: #
        try:
            cli_params_json = json.loads(model_params_override) #
            custom_ref_text_from_params = cli_params_json.get("ref_text") #
            
            if "max_tokens" in cli_params_json: #
                cli_params_json["max_length_override"] = cli_params_json.pop("max_tokens") #
            
            valid_keys = ["speed", "temperature", "top_p", "top_k", "repetition_penalty", "streaming_interval",  #
                          "pitch", "gender", "stt_model", "lang_code", "cfg_scale", "sentence_split_method",  #
                          "max_length_override"] #

            for key, value in cli_params_json.items(): #
                if key == "ref_text": continue #
                if key in valid_keys: #
                    if key == "max_length_override": #
                        mlx_gen_kwargs[key] = int(value) #
                        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Set '{key}' to '{int(value)}' from model_params.") #
                    elif key in ["speed", "temperature", "top_p", "repetition_penalty", "streaming_interval", "pitch", "cfg_scale"]: #
                        mlx_gen_kwargs[key] = float(value) #
                        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Overrode kwarg '{key}' to '{float(value)}'") #
                    elif key == "top_k": #
                        mlx_gen_kwargs[key] = int(value) #
                        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Overrode kwarg '{key}' to '{int(value)}'") #
                    elif key == "sentence_split_method" and isinstance(value, str) and value.lower() in ["none", "null", "passthrough"]: #
                        mlx_gen_kwargs[key] = None #
                        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Set kwarg '{key}' to None") #
                    else: # string type parameters #
                        mlx_gen_kwargs[key] = str(value) #
                        logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Overrode kwarg '{key}' to '{str(value)}'") #
        except Exception as e_parse: #
            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Error parsing --model-params: {e_parse}") #

    # *** MODIFICATION START ***
    temp_transcription_ref_dir = None # To hold the directory for transcription ref audio
    audio_path_for_transcription = actual_ref_audio_path_for_cloning # Default to original

    if is_cloning_intent and actual_ref_audio_path_for_cloning: #
        if not custom_ref_text_from_params and crisptts_model_config.get("whisper_model_id_for_transcription"):
            # Create a temporary directory for the potentially trimmed transcription reference
            temp_transcription_ref_dir = tempfile.TemporaryDirectory(prefix="crisptts_mlx_transcribe_ref_")
            trimmed_for_transcription_path, _ = _trim_ref_audio_if_needed(
                Path(actual_ref_audio_path_for_cloning),
                15000,  # Trim to 15 seconds for Whisper transcription
                Path(temp_transcription_ref_dir.name)
            )
            if trimmed_for_transcription_path and trimmed_for_transcription_path.exists():
                audio_path_for_transcription = str(trimmed_for_transcription_path)
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): Using trimmed audio '{audio_path_for_transcription}' for Whisper transcription.")
            else:
                logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Trimming for transcription failed or not needed, using original '{actual_ref_audio_path_for_cloning}'. This might be too long.")
        
        # This part of the logic remains similar, but uses audio_path_for_transcription
        mlx_gen_kwargs["voice"] = None  #

        if custom_ref_text_from_params: #
            mlx_gen_kwargs["ref_text"] = custom_ref_text_from_params #
            logger.info(f"mlx-audio ({crisptts_specific_model_id}): Using ref_text from params: '{custom_ref_text_from_params[:100]}...'") #
        else: #
            whisper_model_id_cfg = crisptts_model_config.get("whisper_model_id_for_transcription") #
            if whisper_model_id_cfg: #
                logger.info(f"mlx-audio ({crisptts_specific_model_id}): No explicit ref_text. Transcribing ref_audio '{audio_path_for_transcription}' with local Whisper utility.") #
                whisper_target_device = "cpu" #
                if TORCH_FOR_WHISPER_AVAILABLE and torch_whisper_module: #
                    if torch_whisper_module.cuda.is_available(): whisper_target_device = "cuda" #
                    elif hasattr(torch_whisper_module.backends, "mps") and torch_whisper_module.backends.mps.is_available(): #
                        if platform.machine() == "arm64" and platform.system() == "Darwin": logger.info("mlx-audio: MLX on MPS, Whisper to CPU.") #
                        else: whisper_target_device = "mps" #
                
                transcribed_text, trans_err = _transcribe_ref_audio_for_mlx( #
                    audio_path_for_transcription, whisper_model_id_cfg, os.getenv("HF_TOKEN"), #
                    whisper_target_device, crisptts_model_config.get("language_for_whisper") #
                )
                if trans_err or not transcribed_text: #
                    logger.error(f"mlx-audio ({crisptts_specific_model_id}): Transcription failed: {trans_err}. Using placeholder.") #
                    mlx_gen_kwargs["ref_text"] = " " #
                else: #
                    mlx_gen_kwargs["ref_text"] = transcribed_text #
            else: #
                logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Cloning but Whisper not configured for this model. Dia might use internal STT or fail if ref_text is required.") #
                mlx_gen_kwargs["ref_text"] = " " if "dia" in crisptts_specific_model_id.lower() else None #
    # *** MODIFICATION END ***

    generated_audio_data_bytes = None #
    # This temp_dir_path is for the *output* of mlx-audio and for the *ref_audio* that mlx-audio itself consumes
    with tempfile.TemporaryDirectory(prefix="crisptts_mlx_handler_") as temp_dir_str: #
        temp_dir_path = Path(temp_dir_str) #
        path_to_temp_trimmed_ref_for_generate_func = None # Renamed for clarity

        # Prepare the ref_audio that mlx-audio's generate() function will use
        if is_cloning_intent and actual_ref_audio_path_for_cloning:
             # This ensures mlx_gen_kwargs["ref_audio"] is set for the generate_audio_mlx_func call
            max_dur_ms = crisptts_model_config.get("ref_audio_max_duration_ms", 15000) #
            if "dia" in crisptts_specific_model_id.lower() and max_dur_ms > 12000: #
                logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Reducing ref audio trim for Dia to 12s from {max_dur_ms/1000.0:.1f}s.") #
                max_dur_ms = 12000 #
            
            # Trim the original ref audio for the mlx generate() function, storing in temp_dir_path
            path_to_use_for_generate, path_to_temp_trimmed_ref_for_generate_func = _trim_ref_audio_if_needed( #
                Path(actual_ref_audio_path_for_cloning), max_dur_ms, temp_dir_path #
            )
            mlx_gen_kwargs["ref_audio"] = str(path_to_use_for_generate) #
        elif voice_name_or_npz_path_for_mlx : # If not cloning but using a voice name/NPZ
            mlx_gen_kwargs["ref_audio"] = None # Ensure ref_audio is None
            mlx_gen_kwargs["ref_text"] = None  # Ensure ref_text is None
            mlx_gen_kwargs["voice"] = voice_name_or_npz_path_for_mlx # Ensure voice is set
        # If not cloning and no voice_name_or_npz_path_for_mlx, it's zero-shot or uses model's default voice.
        # mlx_gen_kwargs["voice"] would be None or a default from config.

        temp_file_basename = "mlx_synth_output" #
        mlx_gen_kwargs["file_prefix"] = temp_file_basename #
        original_cwd = Path.cwd() #
        try:
            os.chdir(temp_dir_path) #
            # Ensure no conflicting voice/ref_audio parameters are passed to mlx-audio
            final_kwargs_for_generate = {k: v for k, v in mlx_gen_kwargs.items() if v is not None or k == "sentence_split_method"} #
            
            if is_cloning_intent: #
                if "voice" in final_kwargs_for_generate: del final_kwargs_for_generate["voice"] #
            elif voice_name_or_npz_path_for_mlx: #
                 if "ref_audio" in final_kwargs_for_generate: del final_kwargs_for_generate["ref_audio"] #
                 if "ref_text" in final_kwargs_for_generate: del final_kwargs_for_generate["ref_text"] #
            # If neither, it's likely a model that doesn't need voice/ref_audio (e.g. Kokoro with lang_code)

            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Calling generate_audio with text: '{text[:50]}...', kwargs: {json.dumps(final_kwargs_for_generate, default=str)}") #
            with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=not logger.isEnabledFor(logging.DEBUG)): #
                generate_audio_mlx_func(model_path=mlx_model_repo_id_or_path, text=text, **final_kwargs_for_generate) #
            
            output_format_ext = final_kwargs_for_generate.get("audio_format", "wav") #
            expected_temp_output_file = temp_dir_path / f"{temp_file_basename}.{output_format_ext}" #
            if expected_temp_output_file.exists() and expected_temp_output_file.stat().st_size > 100: #
                generated_audio_data_bytes = expected_temp_output_file.read_bytes() #
            else: #
                logger.error(f"mlx-audio ({crisptts_specific_model_id}): Output file not created/empty: {expected_temp_output_file}") #
        except Exception as e_synth_mlx: #
            logger.error(f"mlx-audio ({crisptts_specific_model_id}): Synthesis failed: {e_synth_mlx}", exc_info=True) #
        finally: #
            os.chdir(original_cwd) #
            # path_to_temp_trimmed_ref_for_generate_func is created within temp_dir_path, so it's auto-cleaned by TemporaryDirectory
            # No need to manually delete path_to_temp_trimmed_ref_for_generate_func here.
            # if path_to_temp_trimmed_ref_to_delete and path_to_temp_trimmed_ref_to_delete.exists():
            #     try: path_to_temp_trimmed_ref_to_delete.unlink()
            #     except OSError as e_del: logger.warning(f"mlx-audio: Could not delete temp ref: {e_del}")

    # Cleanup the temporary directory used for transcription ref audio, if it was created
    if temp_transcription_ref_dir:
        try:
            temp_transcription_ref_dir.cleanup()
            logger.debug(f"mlx-audio ({crisptts_specific_model_id}): Cleaned up temp directory for transcription ref: {temp_transcription_ref_dir.name}")
        except Exception as e_cleanup:
            logger.warning(f"mlx-audio ({crisptts_specific_model_id}): Error cleaning up temp transcription ref dir {temp_transcription_ref_dir.name}: {e_cleanup}")


    if generated_audio_data_bytes: #
        if output_file_str: #
            out_fmt = mlx_gen_kwargs.get('audio_format', 'wav') #
            final_path = Path(output_file_str).with_suffix(f".{out_fmt}") #
            try:
                save_audio(generated_audio_data_bytes, str(final_path), False, f"{out_fmt}_bytes", crisptts_model_config.get("sample_rate")) #
            except Exception as e_save: logger.error(f"mlx-audio ({crisptts_specific_model_id}): Error saving: {e_save}", exc_info=True) #
        if play_direct: #
            play_audio(generated_audio_data_bytes, False, f"{mlx_gen_kwargs.get('audio_format', 'wav')}_bytes", crisptts_model_config.get("sample_rate")) #
    else: #
        logger.warning(f"mlx-audio ({crisptts_specific_model_id}): No audio bytes generated for '{mlx_model_repo_id_or_path}'.") #

    if MLX_AUDIO_AVAILABLE and mx_core_module and hasattr(mx_core_module, 'clear_cache'): #
        try: mx_core_module.clear_cache(); logger.debug("mlx-audio: Cleared MLX cache.") #
        except Exception as e: logger.warning(f"mlx-audio: Error clearing MLX cache: {e}") #
    gc.collect() #
    logger.info(f"mlx-audio ({crisptts_specific_model_id}): Handler finished.") #