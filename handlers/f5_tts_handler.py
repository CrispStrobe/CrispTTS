# handlers/f5_tts_handler.py

# --- ABSOLUTE FIRST LINES FOR DIAGNOSTICS ---
print("DEBUG: f5_tts_handler.py: Top of file reached.", flush=True)
# --- END DIAGNOSTICS ---

import logging
import os
from pathlib import Path
import gc
import json
import platform
import tempfile

# --- Logger for initial imports ---
logger_init = logging.getLogger("CrispTTS.handlers.f5_tts.init")

# --- Conditional Imports ---
TORCH_AVAILABLE = False
torch = None
IS_MPS = False
IS_CUDA = False

F5_TTS_STANDARD_API_AVAILABLE = False # For SWivid/F5-TTS
F5TTSStandardAPIClass = None          # Will hold f5_tts.api.F5TTS

F5_TTS_MLX_AVAILABLE = False
generate_mlx_func = None              # Will hold f5_tts_mlx.generate.generate

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

# --- Attempt Direct Imports ---
try:
    print("DEBUG: f5_tts_handler.py: Attempting to import torch...", flush=True)
    import torch
    TORCH_AVAILABLE = True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): IS_MPS = True
    if torch.cuda.is_available(): IS_CUDA = True
    logger_init.info("PyTorch imported successfully for F5-TTS handler.")
    print("DEBUG: f5_tts_handler.py: torch imported successfully.", flush=True)
except ImportError as e_torch:
    logger_init.critical(f"CRITICAL: PyTorch import FAILED for F5-TTS handler. All F5 backends will be unavailable. Error: {e_torch}")
    print(f"CRITICAL DEBUG: f5_tts_handler.py: torch import FAILED: {e_torch}", flush=True)
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        from f5_tts.api import F5TTS # Correct import path for SWivid/F5-TTS
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
        if IS_MPS: # f5-tts-mlx is primarily for MLX on Apple Silicon
            from f5_tts_mlx.generate import generate
            generate_mlx_func = generate
            F5_TTS_MLX_AVAILABLE = True
            logger_init.info("f5_tts_mlx.generate function imported successfully.")
        else:
            logger_init.debug("Not an MPS device, f5-tts-mlx import not attempted.")
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
        logger_init.info("Transformers pipeline imported successfully (for Whisper).")
    except ImportError as e_transformers:
        logger_init.warning(f"Transformers pipeline import failed. Whisper transcription unavailable. Error: {e_transformers}")
        TRANSFORMERS_PIPELINE_AVAILABLE = False
else: # TORCH_AVAILABLE is False
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

print("DEBUG: f5_tts_handler.py: Initial imports section finished.", flush=True)

from utils import save_audio, play_audio, SuppressOutput
logger = logging.getLogger("CrispTTS.handlers.f5_tts")

_transcription_cache = {}

def _ensure_hf_files_for_standard_f5(model_name_or_path: str, hf_token: str | None, log_prefix: str) -> bool:
    """Ensures vocab.txt is present for standard F5TTS API if it relies on HF"""
    if not (HF_HUB_AVAILABLE and callable(hf_hub_download_func)):
        logger.warning(f"{log_prefix}huggingface_hub not available for _ensure_hf_files_for_standard_f5.")
        return False
    try: # The F5TTS API might look for vocab.txt relative to a downloaded checkpoint.
          # This function is more of a best-effort if direct repo_id is used.
        if "/" in model_name_or_path: # Likely an HF repo ID
            logger.debug(f"{log_prefix}Ensuring vocab.txt for standard F5 model '{model_name_or_path}'.")
            hf_hub_download_func(repo_id=model_name_or_path, filename="vocab.txt", token=hf_token, local_files_only=False, resume_download=True, force_download=False)
        return True
    except Exception as e:
        logger.info(f"{log_prefix}Info: Could not pre-cache vocab.txt for '{model_name_or_path}': {e}. F5TTS API will attempt loading.", exc_info=False)
        return False


def _transcribe_ref_audio_with_whisper(
    audio_path_str: str, whisper_model_id: str, hf_token: str | None, log_prefix: str
) -> str | None:
    # ... (implementation remains the same as previous version)
    global _transcription_cache
    resolved_audio_path = str(Path(audio_path_str).resolve()) 

    if resolved_audio_path in _transcription_cache:
        cached_transcription = _transcription_cache[resolved_audio_path]
        if cached_transcription is not None:
            logger.info(f"{log_prefix}Using cached transcription for '{resolved_audio_path}'.")
            return cached_transcription
        else:
            logger.warning(f"{log_prefix}Skipping transcription for '{resolved_audio_path}' due to previous failure (cached as None).")
            return None

    if not (TRANSFORMERS_PIPELINE_AVAILABLE and callable(transformers_pipeline_func)):
        logger.warning(f"{log_prefix}Transformers pipeline for Whisper not available/callable."); return None
    if not TORCH_AVAILABLE:
        logger.warning(f"{log_prefix}PyTorch not available for Whisper."); return None
        
    whisper_pipeline_instance = None
    transcribed_text = None
    try:
        whisper_device = "cpu"
        if torch.cuda.is_available(): whisper_device = "cuda:0"
        
        logger.info(f"{log_prefix}Initializing Whisper ('{whisper_model_id}') on device '{whisper_device}'.")
        
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            whisper_pipeline_instance = transformers_pipeline_func(
                task="automatic-speech-recognition", model=whisper_model_id,
                torch_dtype=torch.float16 if whisper_device.startswith("cuda") else torch.float32,
                device=whisper_device, token=hf_token, framework="pt"
            )
        
        audio_input_bytes = Path(resolved_audio_path).read_bytes()
        logger.info(f"{log_prefix}Transcribing '{resolved_audio_path}'...")

        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            transcription_result = whisper_pipeline_instance(
                audio_input_bytes,
                generate_kwargs={"return_timestamps": True}
            )
        
        transcribed_text = transcription_result["text"].strip() if transcription_result and "text" in transcription_result and isinstance(transcription_result["text"], str) else None
        
        if transcribed_text:
            logger.info(f"{log_prefix}Transcription: '{transcribed_text[:100]}...'")
            _transcription_cache[resolved_audio_path] = transcribed_text
        else:
            logger.warning(f"{log_prefix}Whisper returned empty for '{resolved_audio_path}'.")
            _transcription_cache[resolved_audio_path] = None
        return transcribed_text
    except RuntimeError as e_rt:
        if "tensorflow" in str(e_rt).lower():
            logger.error(f"{log_prefix}Whisper failed (TensorFlow issue). Error: {e_rt}", exc_info=logger.isEnabledFor(logging.DEBUG))
        else:
            logger.error(f"{log_prefix}Whisper runtime error ('{resolved_audio_path}'): {e_rt}", exc_info=True)
        _transcription_cache[resolved_audio_path] = None; return None
    except Exception as e:
        logger.error(f"{log_prefix}Whisper failed ('{resolved_audio_path}'): {e}", exc_info=True)
        _transcription_cache[resolved_audio_path] = None; return None
    finally:
        del whisper_pipeline_instance; gc.collect()
        if TORCH_AVAILABLE:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif IS_MPS: torch.mps.empty_cache()

def synthesize_with_f5_tts(
    model_config: dict, text: str, voice_id_override: str | None,
    model_params_override: str | None, output_file_str: str | None, play_direct: bool
):
    crisptts_model_id_log = model_config.get('model_repo_id', model_config.get('model_name_for_api', 'f5_tts')) # Prefer repo_id for logging
    log_prefix = f"F5-TTS ({crisptts_model_id_log}): "
    logger.info(f"{log_prefix}Starting synthesis.")

    # General dependencies (np, sf, librosa) are checked first. Torch is checked per backend.
    if not all([NUMPY_AVAILABLE, SOUNDFILE_AVAILABLE, LIBROSA_AVAILABLE]):
        missing_deps = [n for avail, n in [(NUMPY_AVAILABLE, "NumPy"), (SOUNDFILE_AVAILABLE, "SoundFile"), (LIBROSA_AVAILABLE, "Librosa")] if not avail]
        logger.error(f"{log_prefix}Missing core general dependencies: {', '.join(missing_deps)}. Aborting.")
        return

    use_mlx_cfg = model_config.get("use_mlx", False)
    # Backend readiness checks
    can_use_mlx = use_mlx_cfg and TORCH_AVAILABLE and F5_TTS_MLX_AVAILABLE and IS_MPS and callable(generate_mlx_func)
    can_use_pytorch = TORCH_AVAILABLE and F5_TTS_STANDARD_API_AVAILABLE and callable(F5TTSStandardAPIClass)

    backend_name, device_str = None, "cpu"

    if use_mlx_cfg:
        if can_use_mlx:
            backend_name, device_str = "F5-TTS-MLX", "mps"
            logger.info(f"{log_prefix}Selected MLX backend on MPS device.")
        else:
            logger.error(f"{log_prefix}MLX backend requested but not usable. MLX Lib Imported: {F5_TTS_MLX_AVAILABLE}, Torch Available: {TORCH_AVAILABLE}, Is MPS: {IS_MPS}, generate_mlx_func loaded: {callable(generate_mlx_func)}. Aborting.")
            return
    elif can_use_pytorch:
        backend_name = "F5-TTS (PyTorch Standard)"
        if IS_CUDA: device_str = "cuda"
        elif IS_MPS: device_str = "mps"
        logger.info(f"{log_prefix}Selected standard PyTorch backend on device: {device_str}.")
    else:
        logger.error(f"{log_prefix}No suitable F5-TTS backend. MLX possible: {can_use_mlx}, PyTorch Standard API possible: {can_use_pytorch}. Check initial import logs (especially for PyTorch and f5_tts.api). Aborting.")
        return

    hf_token = os.getenv("HF_TOKEN") if model_config.get("requires_hf_token") else None
    
    # --- Parse Generation Parameters ---
    # For standard F5TTS API, relevant params are: nfe_step, cfg_strength, sway_sampling_coef, speed, fix_duration
    # For MLX F5TTS, relevant params are: steps, cfg_strength, sway_sampling_coef, speed, estimate_duration, duration
    gen_params = { 
        "steps": model_config.get("default_steps", 32 if backend_name == "F5-TTS (PyTorch Standard)" else 8), # Different defaults
        "cfg_strength": model_config.get("default_cfg_strength", 2.0),
        "sway_sampling_coef": model_config.get("default_sway_sampling_coef", -1.0), 
        "speed": model_config.get("default_speed", 1.0), 
        "ref_text": None, # Populated later
        "estimate_duration": backend_name == "F5-TTS-MLX", # Default to True for MLX to avoid padding errors
        "duration": None # For MLX if estimate_duration is False
    }
    # Specific alias for PyTorch standard model
    if backend_name == "F5-TTS (PyTorch Standard)":
        gen_params["nfe_step"] = gen_params.pop("steps") 
        gen_params["fix_duration"] = None # Equivalent of MLX duration

    if model_params_override:
        try: 
            parsed_cli_params = json.loads(model_params_override)
            allowed_keys = set(gen_params.keys()).union({"ref_text", "estimate_duration", "duration", "nfe_step", "fix_duration"})
            
            # Handle aliasing for PyTorch standard steps
            if backend_name == "F5-TTS (PyTorch Standard)" and "steps" in parsed_cli_params:
                parsed_cli_params["nfe_step"] = parsed_cli_params.pop("steps")
            
            for k,v in parsed_cli_params.items():
                if k in allowed_keys:
                    gen_params[k] = v
                    logger.info(f"{log_prefix}Overriding param '{k}' to '{v}' from CLI.")
        except: logger.warning(f"{log_prefix}Could not parse --model-params.")

    ref_audio_path_orig = Path(voice_id_override or model_config.get("default_voice_id")).resolve()
    if not ref_audio_path_orig.exists():
        logger.error(f"{log_prefix}Reference audio '{ref_audio_path_orig}' not found. Aborting."); return

    final_ref_audio_path_for_model_str = str(ref_audio_path_orig)
    temp_audio_session_dir_obj = None 
    
    # --- Reference Audio Processing ---
    # For MLX: needs trimming, 24kHz, and transcription of the processed audio.
    # For Standard PyTorch: uses original (or user-provided path), but still needs transcription if ref_text not given.
    
    actual_ref_text = gen_params.get("ref_text") # Get text if user provided it via --model-params

    if backend_name == "F5-TTS-MLX":
        if not (librosa and np and sf): 
            logger.error(f"{log_prefix}Librosa/NumPy/SoundFile not available for MLX ref audio processing. Aborting."); return
        
        temp_audio_session_dir_obj = tempfile.TemporaryDirectory(prefix="crisptts_f5mlx_ref_")
        processed_ref_audio_for_mlx_path = Path(temp_audio_session_dir_obj.name) / f"processed_{ref_audio_path_orig.stem}.wav"
        
        try:
            max_ref_duration_s = float(model_config.get("mlx_ref_max_duration_s", 15.0))
            logger.info(f"{log_prefix}Loading and trimming original ref audio '{ref_audio_path_orig}' to max {max_ref_duration_s}s for MLX.")
            y_trimmed, sr_orig = librosa.load(str(ref_audio_path_orig), sr=None, duration=max_ref_duration_s, mono=True)
            logger.info(f"{log_prefix}Ref audio trimmed to {librosa.get_duration(y=y_trimmed, sr=sr_orig):.2f}s (original SR: {sr_orig}Hz).")
            
            target_sr_mlx = 24000
            y_processed = librosa.resample(y_trimmed, orig_sr=sr_orig, target_sr=target_sr_mlx) if sr_orig != target_sr_mlx else y_trimmed
            
            sf.write(str(processed_ref_audio_for_mlx_path), y_processed, target_sr_mlx)
            final_ref_audio_path_for_model_str = str(processed_ref_audio_for_mlx_path)
            logger.info(f"{log_prefix}Processed ref audio for MLX (trimmed & 24kHz): {final_ref_audio_path_for_model_str}")

            if not actual_ref_text: # If not provided via --model-params
                whisper_model_id = model_config.get("whisper_model_id_for_transcription", "openai/whisper-base")
                logger.info(f"{log_prefix}Transcribing processed MLX ref audio '{final_ref_audio_path_for_model_str}'.")
                actual_ref_text = _transcribe_ref_audio_with_whisper(final_ref_audio_path_for_model_str, whisper_model_id, hf_token, log_prefix)
        
        except Exception as e_process_ref:
            logger.error(f"{log_prefix}Error processing ref audio for MLX: {e_process_ref}. Using original.", exc_info=True)
            final_ref_audio_path_for_model_str = str(ref_audio_path_orig) # Fallback
            if not actual_ref_text: # Transcribe original if processing failed and no text yet
                actual_ref_text = _transcribe_ref_audio_with_whisper(str(ref_audio_path_orig), model_config.get("whisper_model_id_for_transcription", "openai/whisper-base"), hf_token, log_prefix)
        
        actual_ref_text = actual_ref_text or "Reference transcription failed or unavailable." # Fallback for MLX

    elif backend_name == "F5-TTS (PyTorch Standard)":
        if not actual_ref_text: # If not provided via --model-params for standard model
            whisper_model_id = model_config.get("whisper_model_id_for_transcription", "openai/whisper-base")
            logger.info(f"{log_prefix}No ref_text for Standard PyTorch. Transcribing original ref audio '{ref_audio_path_orig}'.")
            actual_ref_text = _transcribe_ref_audio_with_whisper(str(ref_audio_path_orig), whisper_model_id, hf_token, log_prefix)
        actual_ref_text = actual_ref_text or "Reference transcription failed or unavailable." # Fallback for Standard

    f5_standard_model_instance = None # For PyTorch standard model
    
    # Define where output will be saved, if at all by the handler/library
    output_target_path = None
    if output_file_str:
        output_target_path = Path(output_file_str).with_suffix(".wav")
        output_target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"{log_prefix}Synthesizing with {backend_name} for model '{crisptts_model_id_log}'...")
        if backend_name == "F5-TTS-MLX":
            if not callable(generate_mlx_func):
                 logger.error(f"{log_prefix}MLX generate_mlx_func not callable. Aborting."); return

            mlx_args = { "generation_text": text, "model_name": crisptts_model_id_log, 
                         "ref_audio_path": final_ref_audio_path_for_model_str,
                         "ref_audio_text": actual_ref_text, "steps": int(gen_params["steps"]), 
                         "cfg_strength": float(gen_params["cfg_strength"]), "speed": float(gen_params["speed"]), 
                         "sway_sampling_coef": float(gen_params["sway_sampling_coef"]),
                         "quantization_bits": model_config.get("quantization_bits"),
                         "estimate_duration": bool(gen_params["estimate_duration"]), # Use from gen_params
                         "output_path": str(output_target_path) if output_target_path else None
                        }
            if mlx_args["quantization_bits"] is None: del mlx_args["quantization_bits"]
            if not mlx_args["estimate_duration"] and "duration" in gen_params and gen_params["duration"] is not None:
                 try: mlx_args["duration"] = float(gen_params["duration"])
                 except (ValueError, TypeError): logger.warning(f"{log_prefix}Invalid 'duration' in model_params for MLX, 'estimate_duration' might be used by library if it's True.")
            elif mlx_args["estimate_duration"]:
                 mlx_args.pop("duration", None)

            logger.debug(f"{log_prefix}MLX generate args: {mlx_args}")
            try:
                # generate_mlx_func will save to mlx_args["output_path"] if provided
                # It returns the audio tensor which we won't use for saving if output_path is set.
                # It plays internally if output_path is None.
                _ = generate_mlx_func(**mlx_args) # We don't need the returned tensor if lib saves
                
                if output_target_path and output_target_path.exists() and output_target_path.stat().st_size > 100:
                    logger.info(f"{log_prefix}MLX library saved audio to {output_target_path}")
                elif output_target_path: # output_path was given but file not created/empty
                     logger.warning(f"{log_prefix}MLX library was given output_path '{output_target_path}' but file was not created or is empty.")
                # If no output_target_path, MLX lib plays internally if play_direct is desired by user (handled by lib)
                
            except ValueError as e_val_mlx:
                if "Received parameters not in model" in str(e_val_mlx):
                    logger.error(f"{log_prefix}FATAL: MLX model '{crisptts_model_id_log}' (likely its duration_v2.safetensors) is incompatible with installed f5_tts_mlx. Advise: Manually intervene. Error: {e_val_mlx}", exc_info=False)
                elif "Reference audio must have a sample rate of 24kHz" in str(e_val_mlx):
                     logger.error(f"{log_prefix}MLX model requires 24kHz reference audio. Resampling failed or was bypassed. Error: {e_val_mlx}", exc_info=False)
                elif "Invalid high padding size" in str(e_val_mlx):
                    logger.error(f"{log_prefix}MLX failed (padding error). This may be due to the combination of a (trimmed) 15s reference audio and its (potentially long) transcription leading to an internal duration estimate that's too short for the reference itself. Try a shorter reference audio (e.g., 5-10s). Error: {e_val_mlx}", exc_info=False)
                else: 
                    logger.error(f"{log_prefix}MLX generate_mlx_func raised an unexpected ValueError.", exc_info=True)
                return
        
        else: # Standard PyTorch Backend (f5_tts.api.F5TTS)
            if not (F5TTSStandardAPIClass and TORCH_AVAILABLE and librosa and np and sf): 
                 logger.error(f"{log_prefix}PyTorch Standard backend dependencies not met. Aborting."); return

            # The 'model' parameter for F5TTSStandardAPIClass can be a repo_id like "SWivid/F5-TTS"
            # or a local name if configs are present, e.g. "F5TTS_v1_Base".
            # It internally uses cached_path for HF models.
            # ckpt_file can be a specific filename within the repo if model_config provides it.
            model_arg_for_api = model_config.get("model_name_for_api", crisptts_model_id_log)
            _ensure_hf_files_for_standard_f5(model_arg_for_api, hf_token, log_prefix)

            f5_standard_model_instance = F5TTSStandardAPIClass(
                model=model_arg_for_api, 
                ckpt_file=model_config.get("checkpoint_path"), # Can be path or HF filename
                vocab_file=model_config.get("vocab_file_path"), # Optional
                device=device_str
                # vocoder_local_path is another option if not using default from API class
            )
            
            # The F5TTS.infer method expects nfe_step, not steps
            # It also expects fix_duration, not duration
            nfe_step_val = int(gen_params.get("nfe_step", gen_params.get("steps", 32))) # Prioritize nfe_step
            fix_duration_val = gen_params.get("fix_duration")
            if fix_duration_val is not None:
                try: fix_duration_val = float(fix_duration_val)
                except (ValueError, TypeError): fix_duration_val = None

            logger.debug(f"{log_prefix}Calling Standard F5TTS.infer with ref_file='{final_ref_audio_path_for_model_str}', ref_text='{actual_ref_text[:50]}...', gen_text='{text[:50]}...', nfe_step={nfe_step_val}, fix_duration={fix_duration_val}")

            # F5TTS.infer saves if file_wave is provided, and returns (wav, sr, spec)
            wav_np_array, sr_from_infer, _ = f5_standard_model_instance.infer(
                ref_file=final_ref_audio_path_for_model_str, # Original reference for standard
                ref_text=actual_ref_text,
                gen_text=text,
                file_wave=str(output_target_path) if output_target_path else None,
                nfe_step=nfe_step_val,
                cfg_strength=float(gen_params["cfg_strength"]),
                sway_sampling_coef=float(gen_params["sway_sampling_coef"]),
                speed=float(gen_params["speed"]),
                fix_duration=fix_duration_val
            )

            if output_target_path and output_target_path.exists() and output_target_path.stat().st_size > 100:
                 logger.info(f"{log_prefix}Standard F5-TTS API saved audio to {output_target_path}")
            elif wav_np_array is not None and wav_np_array.size > 0 :
                 logger.info(f"{log_prefix}Standard F5-TTS API returned audio data. Shape: {wav_np_array.shape}")
                 if output_target_path: # if file_wave was None but user requested output_file_str
                     save_audio(wav_np_array, str(output_target_path), False, "numpy", sr_from_infer)
                     logger.info(f"{log_prefix}Audio saved by handler to {output_target_path}")
            else:
                logger.error(f"{log_prefix}Standard F5-TTS API: Empty audio output or file not saved."); return
        
        # Playback logic (common for both backends if audio was produced/saved)
        if play_direct:
            if output_target_path and output_target_path.exists():
                logger.info(f"{log_prefix}Playing saved output file: {output_target_path}")
                play_audio(str(output_target_path), is_path=True) # play_audio can take a path
            elif backend_name == "F5-TTS-MLX" and not output_target_path:
                # MLX plays internally if output_path was None for its generate()
                logger.info(f"{log_prefix}MLX library handled playback internally.")
            elif backend_name == "F5-TTS (PyTorch Standard)" and wav_np_array is not None and wav_np_array.size > 0 and not output_target_path:
                # PyTorch returned audio data, and no file was saved by lib (because file_wave was None)
                logger.info(f"{log_prefix}Playing PyTorch generated audio from memory.")
                play_audio(wav_np_array, False, "numpy", sr_from_infer)

    except Exception as e: # General catch-all for synthesis part
        logger.error(f"{log_prefix}Synthesis error: {e}", exc_info=True)
    finally:
        del f5_standard_model_instance; gc.collect() # Safe if None
        if temp_audio_session_dir_obj:
            try: temp_audio_session_dir_obj.cleanup()
            except Exception as e_clean: logger.warning(f"{log_prefix}Error cleaning up temp audio dir: {e_clean}")
            
        if TORCH_AVAILABLE:
            if device_str == "cuda": torch.cuda.empty_cache()
            elif device_str == "mps" and IS_MPS: torch.mps.empty_cache()
        logger.info(f"{log_prefix}Handler finished.")