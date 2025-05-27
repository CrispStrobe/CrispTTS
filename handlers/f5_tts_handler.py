# handlers/f5_tts_handler.py

# print("DEBUG: f5_tts_handler.py: Top of file reached.", flush=True) # Commented out

import logging
import os
from pathlib import Path
import gc
import json
import platform
import tempfile

logger_init = logging.getLogger("CrispTTS.handlers.f5_tts.init")

TORCH_AVAILABLE = False
torch = None
IS_MPS = False
IS_CUDA = False
F5_TTS_STANDARD_API_AVAILABLE = False
F5TTSStandardAPIClass = None
F5_TTS_MLX_AVAILABLE = False
generate_mlx_func = None
SOUNDFILE_AVAILABLE = False
sf = None
NUMPY_AVAILABLE = False
np = None
LIBROSA_AVAILABLE = False
librosa = None
HF_HUB_AVAILABLE = False
hf_hub_download_func = None
TRANSFORMERS_PIPELINE_AVAILABLE = False
transformers_pipeline_func = None

try:
    # print("DEBUG: f5_tts_handler.py: Attempting to import torch...", flush=True)
    import torch
    TORCH_AVAILABLE = True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): IS_MPS = True
    if torch.cuda.is_available(): IS_CUDA = True
    logger_init.info("PyTorch imported successfully for F5-TTS handler.")
    # print("DEBUG: f5_tts_handler.py: torch imported successfully.", flush=True)
except ImportError as e_torch:
    logger_init.critical(f"CRITICAL: PyTorch import FAILED. All F5 backends will be unavailable. Error: {e_torch}")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        from f5_tts.api import F5TTS
        F5TTSStandardAPIClass = F5TTS
        F5_TTS_STANDARD_API_AVAILABLE = True
        logger_init.info("Standard f5_tts.api.F5TTS class imported successfully.")
    except ImportError as e_f5_std:
        logger_init.warning(f"Standard f5_tts.api.F5TTS import failed. Standard PyTorch backend unavailable. Error: {e_f5_std}")
        F5_TTS_STANDARD_API_AVAILABLE = False
    except Exception as e_f5_std_other:
        logger_init.error(f"Unexpected error during f5_tts.api.F5TTS import. Standard PyTorch backend unavailable. Error: {e_f5_std_other}")
        F5_TTS_STANDARD_API_AVAILABLE = False

    try:
        if IS_MPS:
            from f5_tts_mlx.generate import generate
            generate_mlx_func = generate
            F5_TTS_MLX_AVAILABLE = True
            logger_init.info("f5_tts_mlx.generate function imported successfully.")
        else:
            F5_TTS_MLX_AVAILABLE = False
    except ImportError as e_f5_mlx:
        logger_init.warning(f"f5_tts_mlx library import failed. MLX backend unavailable. Error: {e_f5_mlx}")
        F5_TTS_MLX_AVAILABLE = False
    except Exception as e_f5_mlx_other:
        logger_init.error(f"Unexpected error during f5_tts_mlx.generate import. MLX backend unavailable. Error: {e_f5_mlx_other}")
        F5_TTS_MLX_AVAILABLE = False
    
    try:
        from transformers import pipeline
        transformers_pipeline_func = pipeline
        TRANSFORMERS_PIPELINE_AVAILABLE = True
        logger_init.info("Transformers pipeline imported (for Whisper).")
    except ImportError as e_transformers:
        logger_init.warning(f"Transformers pipeline import failed. Whisper unavailable. Error: {e_transformers}")
        TRANSFORMERS_PIPELINE_AVAILABLE = False
else:
    logger_init.critical("PyTorch is NOT available. All F5-TTS functionality is disabled.")
    F5_TTS_STANDARD_API_AVAILABLE = False
    F5_TTS_MLX_AVAILABLE = False
    TRANSFORMERS_PIPELINE_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    hf_hub_download_func = hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError as e_hf_hub: logger_init.warning(f"huggingface_hub import failed. Error: {e_hf_hub}"); HF_HUB_AVAILABLE = False
try:
    import soundfile as sf_imported; sf = sf_imported; SOUNDFILE_AVAILABLE = True
except ImportError: logger_init.warning("SoundFile import failed.")
try:
    import numpy as np_imported; np = np_imported; NUMPY_AVAILABLE = True
except ImportError: logger_init.warning("NumPy import failed.")
try:
    import librosa as librosa_imported; librosa = librosa_imported; LIBROSA_AVAILABLE = True
except ImportError: logger_init.warning("Librosa import failed.")

# print("DEBUG: f5_tts_handler.py: Initial imports section finished.", flush=True) # Commented out

from utils import save_audio, play_audio, SuppressOutput
logger = logging.getLogger("CrispTTS.handlers.f5_tts")

_transcription_cache = {}

def _ensure_hf_files_for_standard_f5(model_name_or_path: str, hf_token: str | None, log_prefix: str) -> bool:
    # The f5_tts.api.F5TTS class handles its own downloads via cached_path.
    # This helper might only be useful if specific pre-caching of vocab is needed outside the class's own logic.
    # For now, it's mostly a placeholder as the main class should manage this.
    if not (HF_HUB_AVAILABLE and callable(hf_hub_download_func)): return False
    try:
        if "/" in model_name_or_path: # Likely an HF repo ID
            logger.debug(f"{log_prefix}Ensuring vocab.txt (if exists) for standard F5 model '{model_name_or_path}'.")
            # The F5TTS class will look for configs/model.yaml, then try to get vocab from there or default.
            # vocab.txt at root is not standard for all SWivid/F5-TTS derived models, but doesn't hurt to try.
            hf_hub_download_func(repo_id=model_name_or_path, filename="vocab.txt", token=hf_token, local_files_only=False, resume_download=True, force_download=False, ignore_errors=True)
        return True
    except Exception: return False


def _transcribe_ref_audio_with_whisper(
    audio_path_str: str, whisper_model_id: str, hf_token: str | None, log_prefix: str
) -> str | None:
    global _transcription_cache
    resolved_audio_path = str(Path(audio_path_str).resolve()) 
    if resolved_audio_path in _transcription_cache:
        cached_transcription = _transcription_cache[resolved_audio_path]
        if cached_transcription is not None: # Not a cached failure
            logger.info(f"{log_prefix}Using cached transcription for '{resolved_audio_path}'.")
            return cached_transcription
        else: # Cached failure
            logger.warning(f"{log_prefix}Skipping transcription for '{resolved_audio_path}' (previous failure).")
            return None

    if not (TRANSFORMERS_PIPELINE_AVAILABLE and callable(transformers_pipeline_func) and TORCH_AVAILABLE):
        logger.warning(f"{log_prefix}Whisper dependencies not met."); return None
        
    whisper_pipeline_instance = None; transcribed_text = None
    try:
        whisper_device = "cpu"; torch_dtype = torch.float32
        if torch.cuda.is_available(): whisper_device = "cuda:0"; torch_dtype = torch.float16
        logger.info(f"{log_prefix}Initializing Whisper ('{whisper_model_id}') on device '{whisper_device}'.")
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            whisper_pipeline_instance = transformers_pipeline_func(
                task="automatic-speech-recognition", model=whisper_model_id,
                torch_dtype=torch_dtype, device=whisper_device, token=hf_token, framework="pt"
            )
        audio_input_bytes = Path(resolved_audio_path).read_bytes()
        logger.info(f"{log_prefix}Transcribing '{resolved_audio_path}'...")
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            transcription_result = whisper_pipeline_instance(audio_input_bytes, generate_kwargs={"return_timestamps": True})
        transcribed_text = transcription_result["text"].strip() if transcription_result and "text" in transcription_result and isinstance(transcription_result["text"], str) else None
        if transcribed_text: logger.info(f"{log_prefix}Transcription: '{transcribed_text[:100]}...'")
        else: logger.warning(f"{log_prefix}Whisper returned empty for '{resolved_audio_path}'.")
    except RuntimeError as e_rt:
        if "tensorflow" in str(e_rt).lower(): logger.error(f"{log_prefix}Whisper failed (TF issue). Error: {e_rt}", exc_info=logger.isEnabledFor(logging.DEBUG))
        else: logger.error(f"{log_prefix}Whisper runtime error ('{resolved_audio_path}'): {e_rt}", exc_info=True)
    except Exception as e:
        logger.error(f"{log_prefix}Whisper failed ('{resolved_audio_path}'): {e}", exc_info=True)
    finally:
        _transcription_cache[resolved_audio_path] = transcribed_text # Cache success or None
        del whisper_pipeline_instance; gc.collect()
        if TORCH_AVAILABLE:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif IS_MPS: torch.mps.empty_cache()
    return transcribed_text

def synthesize_with_f5_tts(
    model_config: dict, text: str, voice_id_override: str | None,
    model_params_override: str | None, output_file_str: str | None, play_direct: bool
):
    crisptts_model_id_log = model_config.get('model_repo_id', 'f5_tts')
    log_prefix = f"F5-TTS ({crisptts_model_id_log}): "
    logger.info(f"{log_prefix}Starting synthesis for model ID '{crisptts_model_id_log}'.")

    # Dependency checks remain the same
    if not all([NUMPY_AVAILABLE, SOUNDFILE_AVAILABLE, LIBROSA_AVAILABLE, HF_HUB_AVAILABLE]):
        missing_deps = [n for avail, n in [
            (NUMPY_AVAILABLE, "NumPy"), (SOUNDFILE_AVAILABLE, "SoundFile"), 
            (LIBROSA_AVAILABLE, "Librosa"), (HF_HUB_AVAILABLE, "huggingface_hub")] if not avail]
        logger.error(f"{log_prefix}Missing general dependencies: {', '.join(missing_deps)}. Aborting.")
        return

    # Backend selection logic remains the same
    use_mlx_cfg = model_config.get("use_mlx", False)
    can_use_mlx = F5_TTS_MLX_AVAILABLE and IS_MPS and callable(generate_mlx_func)
    can_use_pytorch = F5_TTS_STANDARD_API_AVAILABLE and callable(F5TTSStandardAPIClass)
    
    backend_name, device_str = None, "cpu"
    if use_mlx_cfg and can_use_mlx:
        backend_name, device_str = "F5-TTS-MLX", "mps"
        logger.info(f"{log_prefix}Selected MLX backend as preferred.")
    elif can_use_pytorch:
        backend_name = "F5-TTS (PyTorch Standard)"
        if IS_CUDA: device_str = "cuda"
        elif IS_MPS: device_str = "mps"
        logger.info(f"{log_prefix}Selected PyTorch Standard backend on {device_str}.")
    else:
        logger.error(f"{log_prefix}No suitable F5-TTS backend available. Aborting.")
        return

    # Parameter and path setup remains the same
    hf_token = os.getenv("HF_TOKEN")
    actual_ref_text_for_model = None
    if model_params_override:
        try: actual_ref_text_for_model = json.loads(model_params_override).get("ref_text")
        except: pass
    
    gen_params = { "steps": 32, "nfe_step": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0, "speed": 1.0, "estimate_duration": backend_name == "F5-TTS-MLX", "duration": None, "fix_duration": None }
    # ... (gen_params setup remains the same)

    ref_audio_path_orig = Path(voice_id_override or model_config.get("default_voice_id")).resolve()
    if not ref_audio_path_orig.exists():
        logger.error(f"{log_prefix}Reference audio '{ref_audio_path_orig}' not found. Aborting."); return

    output_target_path = Path(output_file_str).with_suffix(".wav") if output_file_str else None
    if output_target_path: output_target_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_audio_session_dir_obj = None
    f5_standard_model_instance = None
    # Determine which backends to run
    run_mlx_backend = use_mlx_cfg and can_use_mlx
    run_pytorch_backend = not run_mlx_backend
    wav_np_array = None
    sr_from_infer = model_config.get("sample_rate", 24000)

    try:
        # --- MLX Backend Attempt ---
        if run_mlx_backend:
            try:
                # ... (The entire MLX synthesis block remains the same as the previous fix)
                logger.info(f"{log_prefix}Attempting synthesis with MLX backend.")
                final_ref_audio_path_for_model_str = str(ref_audio_path_orig)
                
                if not (librosa and np and sf): raise RuntimeError("Missing MLX dependencies")
                temp_audio_session_dir_obj = tempfile.TemporaryDirectory(prefix="crisptts_f5mlx_ref_")
                processed_ref_path = Path(temp_audio_session_dir_obj.name) / f"processed_{ref_audio_path_orig.stem}.wav"
                max_ref_dur_s = float(model_config.get("mlx_ref_max_duration_s", 15.0))
                y_trim, sr_orig = librosa.load(str(ref_audio_path_orig), sr=None, duration=max_ref_dur_s, mono=True)
                y_proc = librosa.resample(y_trim, orig_sr=sr_orig, target_sr=24000) if sr_orig != 24000 else y_trim
                sf.write(str(processed_ref_path), y_proc, 24000)
                final_ref_audio_path_for_model_str = str(processed_ref_path)

                if not actual_ref_text_for_model:
                    actual_ref_text_for_model = _transcribe_ref_audio_with_whisper(final_ref_audio_path_for_model_str, "openai/whisper-base", hf_token, log_prefix)
                
                mlx_args = { "generation_text": text, "model_name": crisptts_model_id_log, "ref_audio_path": final_ref_audio_path_for_model_str, "ref_audio_text": actual_ref_text_for_model or "", "output_path": str(output_target_path) if output_target_path else None, **gen_params }
                logger.debug(f"{log_prefix}MLX generate args: {mlx_args}")
                generate_mlx_func(**mlx_args)
                
                if not (output_target_path and output_target_path.exists() and output_target_path.stat().st_size > 100):
                    raise RuntimeError("MLX ran but did not produce a valid output file.")
                
                run_pytorch_backend = False # Success! Do not run fallback.

            except (ValueError, RuntimeError) as e_mlx:
                logger.warning(f"{log_prefix}MLX backend failed: '{e_mlx}'. Attempting PyTorch fallback.")
                if can_use_pytorch:
                    run_pytorch_backend = True
                else:
                    logger.error(f"{log_prefix}Cannot fallback as PyTorch backend is not available. Aborting.")
                    raise e_mlx
        
        # --- PyTorch Backend (Primary or Fallback) ---
        if run_pytorch_backend:
            logger.info(f"{log_prefix}Executing synthesis with PyTorch Standard backend.")
            if not actual_ref_text_for_model:
                actual_ref_text_for_model = _transcribe_ref_audio_with_whisper(str(ref_audio_path_orig), "openai/whisper-base", hf_token, log_prefix)
            
            pytorch_ref_text = actual_ref_text_for_model or "Reference transcription unavailable."

            # --- NEW: Logic to handle specific checkpoint files ---
            checkpoint_path_local = None
            if model_config.get("checkpoint_filename"):
                try:
                    chkpt_file = model_config["checkpoint_filename"]
                    logger.info(f"{log_prefix}Downloading specific checkpoint: {chkpt_file}")
                    checkpoint_path_local = hf_hub_download_func(
                        repo_id=crisptts_model_id_log,
                        filename=chkpt_file,
                        token=hf_token
                    )
                except Exception as e_hf:
                    logger.error(f"{log_prefix}Failed to download checkpoint '{model_config['checkpoint_filename']}'. Error: {e_hf}", exc_info=True)
                    raise  # Re-raise the exception to stop synthesis
            
            # The base model to load is always the original F5-TTS, we just supply a fine-tuned checkpoint
            model_arg_api = "SWivid/F5-TTS" 
            
            f5_standard_model_instance = F5TTSStandardAPIClass(
                model=model_arg_api, 
                ckpt_file=checkpoint_path_local, # Use the downloaded specific checkpoint
                device=device_str
            )
            
            # ... (rest of the pytorch infer call remains the same)
            wav_np_array, sr_from_infer, _ = f5_standard_model_instance.infer(
                ref_file=str(ref_audio_path_orig), ref_text=pytorch_ref_text, gen_text=text,
                file_wave=str(output_target_path) if output_target_path else None,
                nfe_step=int(gen_params.get("nfe_step", 32)), cfg_strength=float(gen_params.get("cfg_strength")),
                sway_sampling_coef=float(gen_params.get("sway_sampling_coef")),
                speed=float(gen_params.get("speed"))
            )

    except Exception as e:
        logger.error(f"{log_prefix}An unhandled error occurred during synthesis: {e}", exc_info=True)
    finally:
        # Cleanup logic remains the same
        del f5_standard_model_instance; gc.collect()
        if temp_audio_session_dir_obj:
            try: temp_audio_session_dir_obj.cleanup()
            except Exception as e_clean: logger.warning(f"{log_prefix}Error cleaning temp dir: {e_clean}")
        if TORCH_AVAILABLE:
            if device_str == "cuda": torch.cuda.empty_cache()
            elif device_str == "mps" and IS_MPS: torch.mps.empty_cache()

        # Playback logic (moved inside finally to ensure it runs)
        if play_direct:
            if output_target_path and output_target_path.exists():
                play_audio(str(output_target_path), is_path=True)
            elif wav_np_array is not None and hasattr(wav_np_array, 'size') and wav_np_array.size > 0:
                play_audio(wav_np_array, False, "numpy", sr_from_infer)

        logger.info(f"{log_prefix}Handler finished.")