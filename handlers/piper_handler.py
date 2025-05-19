# handlers/piper_handler.py

import json
import logging
import os
import tempfile
import wave
from pathlib import Path
import gc

# Use relative imports for project modules
from utils import save_audio, play_audio

logger = logging.getLogger("CrispTTS.handlers.piper")

PIPER_TTS_AVAILABLE_IN_HANDLER = False
PiperVoice_h = None
HF_HUB_AVAILABLE_IN_HANDLER = False
hf_hub_download_h = None

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE_IN_HANDLER = True
    hf_hub_download_h = hf_hub_download
except ImportError:
    logger.warning("huggingface_hub not installed. Piper model downloading will fail.")

if HF_HUB_AVAILABLE_IN_HANDLER:
    try:
        from piper.voice import PiperVoice as PiperVoice_imp
        PiperVoice_h = PiperVoice_imp
        PIPER_TTS_AVAILABLE_IN_HANDLER = True
    except ImportError:
        logger.info("'piper-tts' not installed. Piper handler will not be functional.")


def synthesize_with_piper_local(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    if not PIPER_TTS_AVAILABLE_IN_HANDLER or not PiperVoice_h:
        logger.error("Piper-tts library not available. Skipping Piper local synthesis.")
        return
    if not HF_HUB_AVAILABLE_IN_HANDLER or not hf_hub_download_h:
        logger.error("huggingface_hub library not available. Cannot download Piper models. Skipping.")
        return

    logger.debug(f"Piper Local - Text: '{text[:50]}...'")
    model_path_in_repo = None
    config_path_in_repo = None

    if voice_id_override:
        if isinstance(voice_id_override, str) and voice_id_override.startswith('{'):
            try:
                override_details = json.loads(voice_id_override)
                model_path_in_repo = override_details.get("model")
                config_path_in_repo = override_details.get("config")
                logger.debug("Piper - Using dictionary voice override.")
            except json.JSONDecodeError:
                logger.warning(f"Piper - Failed to parse JSON voice ID '{voice_id_override}'. Treating as direct model path.")
                model_path_in_repo = voice_id_override
        elif isinstance(voice_id_override, str): # Assuming direct path or repo relative path
            model_path_in_repo = voice_id_override
    
    if not model_path_in_repo: # Fallback to default from config if override not provided
        model_path_in_repo = model_config.get("default_model_path_in_repo")

    # Auto-derive config path if not explicitly provided
    if model_path_in_repo and not config_path_in_repo:
        if model_path_in_repo.endswith(".onnx.json"):
            config_path_in_repo = model_path_in_repo
            model_path_in_repo = model_path_in_repo.replace(".onnx.json", ".onnx")
        elif model_path_in_repo.endswith(".onnx"):
            config_path_in_repo = model_path_in_repo + ".json"
        else: # Assume it's a base name, append .onnx and .onnx.json
             config_path_in_repo = model_path_in_repo + ".onnx.json"
             model_path_in_repo = model_path_in_repo + ".onnx"

    piper_voice_repo_id = model_config.get("piper_voice_repo_id", "rhasspy/piper-voices")
    if not model_path_in_repo or not config_path_in_repo:
        logger.error(f"Piper - Cannot determine model/config paths. Model: '{model_path_in_repo}', Config: '{config_path_in_repo}'.")
        return

    # Use a unique cache directory for piper models within the project's cache
    model_cache_dir_base = Path.home() / ".cache" / "crisptts_cache" / "piper_models"
    model_cache_dir_base.mkdir(parents=True, exist_ok=True)
    
    # Construct local paths maintaining subdirectory structure from repo path
    # Path(model_path_in_repo).parent will give the subdirectories
    local_piper_model_path_target_dir = model_cache_dir_base / Path(model_path_in_repo).parent
    local_piper_config_path_target_dir = model_cache_dir_base / Path(config_path_in_repo).parent
    
    local_piper_model_path_target_dir.mkdir(parents=True, exist_ok=True)
    local_piper_config_path_target_dir.mkdir(parents=True, exist_ok=True)

    voice_obj = None
    try:
        logger.info(f"Piper - Ensuring model '{model_path_in_repo}' from '{piper_voice_repo_id}'...")
        dl_model_path_str = hf_hub_download_h(
            repo_id=piper_voice_repo_id, filename=model_path_in_repo,
            local_dir=str(local_piper_model_path_target_dir), # Download into the target parent dir
            local_dir_use_symlinks=False, token=os.getenv("HF_TOKEN"), repo_type="model"
        )
        # The downloaded file will be local_piper_model_path_target_dir / Path(model_path_in_repo).name
        local_piper_model_path = Path(dl_model_path_str) # hf_hub_download returns the full path to the file
        logger.info(f"Piper - Model available at: {local_piper_model_path}")

        logger.info(f"Piper - Ensuring config '{config_path_in_repo}' from '{piper_voice_repo_id}'...")
        dl_config_path_str = hf_hub_download_h(
            repo_id=piper_voice_repo_id, filename=config_path_in_repo,
            local_dir=str(local_piper_config_path_target_dir),
            local_dir_use_symlinks=False, token=os.getenv("HF_TOKEN"), repo_type="model"
        )
        local_piper_config_path = Path(dl_config_path_str)
        logger.info(f"Piper - Config available at: {local_piper_config_path}")

        if not local_piper_model_path.exists() or not local_piper_config_path.exists():
             logger.error("Piper - Model or config file missing after download attempt.")
             return

        logger.info(f"Piper - Loading voice: Model='{local_piper_model_path}', Config='{local_piper_config_path}'")
        voice_obj = PiperVoice_h.load(str(local_piper_model_path), config_path=str(local_piper_config_path))
        logger.info("Piper - Voice loaded.")

        audio_data = None
        # Synthesize to a temporary in-memory buffer
        with tempfile.SpooledTemporaryFile() as audio_bytes_io:
            with wave.open(audio_bytes_io, 'wb') as wf:
                # Piper's synthesize writes WAV data directly to the file-like object wf
                voice_obj.synthesize(text, wf)
            audio_bytes_io.seek(0) # Rewind to the beginning of the buffer
            audio_data = audio_bytes_io.read()

        if audio_data:
            logger.info(f"Piper - Synthesis successful, {len(audio_data)} bytes generated.")
            effective_output_file_wav = Path(output_file_str).with_suffix(".wav") if output_file_str else None
            # Piper voice config usually has sample_rate
            current_sample_rate = getattr(getattr(voice_obj, 'config', None), 'sample_rate', 22050) 

            if effective_output_file_wav:
                save_audio(audio_data, str(effective_output_file_wav), source_is_path=False, input_format="wav", sample_rate=current_sample_rate)
            if play_direct:
                play_audio(audio_data, is_path=False, input_format="wav", sample_rate=current_sample_rate)
        else:
            logger.warning("Piper - No audio data generated.")

    except Exception as e:
        logger.error(f"Piper - Synthesis failed: {e}", exc_info=True)
    finally:
        del voice_obj
        gc.collect()