# handlers/chatterbox_handler.py

import logging
import os
from pathlib import Path
import gc
import json
import tempfile
import sys

# Conditional imports
TORCH_AVAILABLE_IN_HANDLER = False
CHATTERBOX_AVAILABLE_IN_HANDLER = False
SOUNDFILE_AVAILABLE_IN_HANDLER = False
HF_HUB_AVAILABLE_IN_HANDLER = False
SAFETENSORS_AVAILABLE_IN_HANDLER = False

torch_chatterbox = None
ChatterboxTTS = None
sf_chatterbox = None
hf_hub_download_chatterbox = None
load_file_chatterbox = None

logger_init = logging.getLogger("CrispTTS.handlers.chatterbox.init")

try:
    import torch as torch_imp
    torch_chatterbox = torch_imp
    TORCH_AVAILABLE_IN_HANDLER = True
except ImportError as e:
    print(f"Chatterbox Handler INIT ERROR: PyTorch import failed: {e}", file=sys.stderr)
    logger_init.info("PyTorch not found. Chatterbox handler will be non-functional.")

if TORCH_AVAILABLE_IN_HANDLER:
    try:
        from chatterbox.tts import ChatterboxTTS as ChatterboxTTS_imp
        ChatterboxTTS = ChatterboxTTS_imp
        CHATTERBOX_AVAILABLE_IN_HANDLER = True
    except ImportError as e:
        print(f"Chatterbox Handler INIT ERROR: chatterbox-tts import failed: {e}", file=sys.stderr)
        logger_init.info("chatterbox-tts library not found. Install with: pip install chatterbox-tts")

try:
    import soundfile as sf_imp
    sf_chatterbox = sf_imp
    SOUNDFILE_AVAILABLE_IN_HANDLER = True
except ImportError as e:
    print(f"Chatterbox Handler INIT ERROR: SoundFile import failed: {e}", file=sys.stderr)
    logger_init.info("SoundFile library not found.")

try:
    from huggingface_hub import hf_hub_download
    hf_hub_download_chatterbox = hf_hub_download
    HF_HUB_AVAILABLE_IN_HANDLER = True
except ImportError as e:
    print(f"Chatterbox Handler INIT ERROR: huggingface_hub import failed: {e}", file=sys.stderr)
    logger_init.info("huggingface_hub library not found.")

try:
    from safetensors.torch import load_file
    load_file_chatterbox = load_file
    SAFETENSORS_AVAILABLE_IN_HANDLER = True
except ImportError as e:
    print(f"Chatterbox Handler INIT ERROR: safetensors import failed: {e}", file=sys.stderr)
    logger_init.info("safetensors library not found.")

try:
    from utils import save_audio, play_audio, SuppressOutput, _prepare_oute_speaker_ref
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Chatterbox Handler CRITICAL INIT ERROR: Failed to import from 'utils': {e}", file=sys.stderr)
    UTILS_AVAILABLE = False

logger = logging.getLogger("CrispTTS.handlers.chatterbox")

def synthesize_with_chatterbox(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'kartoffelbox_unknown')

    # Check dependencies
    if not all([TORCH_AVAILABLE_IN_HANDLER, CHATTERBOX_AVAILABLE_IN_HANDLER, 
                SOUNDFILE_AVAILABLE_IN_HANDLER, HF_HUB_AVAILABLE_IN_HANDLER,
                SAFETENSORS_AVAILABLE_IN_HANDLER, UTILS_AVAILABLE]):
        logger.error(f"Chatterbox ({crisptts_model_id_for_log}): Missing core dependencies. Skipping.")
        return
    
    if not all([torch_chatterbox, ChatterboxTTS, sf_chatterbox, 
                hf_hub_download_chatterbox, load_file_chatterbox]):
        logger.error(f"Chatterbox ({crisptts_model_id_for_log}): Critical modules not loaded. Skipping.")
        return

    # Get model configuration
    model_repo_id = model_config.get("model_repo_id")
    t3_checkpoint_file = model_config.get("t3_checkpoint_file", "t3_cfg.safetensors")
    sample_rate = model_config.get("sample_rate", 22050)
    
    if not model_repo_id:
        logger.error(f"Chatterbox ({crisptts_model_id_for_log}): Missing model_repo_id in config.")
        return

    # Handle voice input - could be a reference audio file or None for zero-shot
    reference_audio_path = None
    if voice_id_override:
        voice_path = Path(voice_id_override)
        if not voice_path.is_absolute():
            # Try relative to project root
            project_root = Path(__file__).resolve().parent.parent
            voice_path = (project_root / voice_id_override).resolve()
        
        if voice_path.exists() and voice_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
            reference_audio_path = str(voice_path)
            logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Using reference audio: {reference_audio_path}")
        else:
            logger.warning(f"Chatterbox ({crisptts_model_id_for_log}): Reference audio not found: {voice_id_override}")

    # Parse model parameters
    generation_params = {
        "exaggeration": 0.5,
        "temperature": 0.6,
        "cfg_weight": 0.3
    }
    
    if model_params_override:
        try:
            cli_params = json.loads(model_params_override)
            generation_params.update(cli_params)
        except json.JSONDecodeError:
            logger.warning(f"Chatterbox ({crisptts_model_id_for_log}): Could not parse --model-params: {model_params_override}")

    logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Synthesizing with model '{model_repo_id}'")
    
    chatterbox_model = None
    temp_ref_audio = None

    try:
        # Determine device
        if torch_chatterbox.cuda.is_available():
            device = "cuda"
        elif hasattr(torch_chatterbox.backends, "mps") and torch_chatterbox.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Using device: {device}")

        # Initialize base Chatterbox model
        logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Loading base ChatterboxTTS model...")
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
        
        # Download and apply German patch
        logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Downloading German patch from {model_repo_id}...")
        hf_token = os.getenv("HF_TOKEN")
        checkpoint_path = hf_hub_download_chatterbox(
            repo_id=model_repo_id, 
            filename=t3_checkpoint_file,
            token=hf_token
        )
        
        logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Applying German patch...")
        t3_state = load_file_chatterbox(checkpoint_path, device="cpu")
        chatterbox_model.t3.load_state_dict(t3_state)
        logger.info(f"Chatterbox ({crisptts_model_id_for_log}): German patch applied successfully.")

        # Prepare reference audio if provided
        if reference_audio_path:
            # Use the same audio preparation logic as OuteTTS for consistency
            processed_ref_path, temp_ref_audio = _prepare_oute_speaker_ref(
                reference_audio_path, crisptts_model_id_for_log
            )
            if processed_ref_path:
                reference_audio_path = str(processed_ref_path)
            else:
                logger.warning(f"Chatterbox ({crisptts_model_id_for_log}): Failed to process reference audio, proceeding without it.")
                reference_audio_path = None

        # Generate speech
        logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Generating speech...")
        logger.debug(f"Generation params: {generation_params}")
        
        with torch_chatterbox.inference_mode():
            if reference_audio_path:
                wav = chatterbox_model.generate(
                    text,
                    audio_prompt_path=reference_audio_path,
                    **generation_params
                )
            else:
                # Zero-shot generation without reference audio
                wav = chatterbox_model.generate(
                    text,
                    **generation_params
                )
        
        # Process output audio
        if wav is None or wav.numel() == 0:
            logger.error(f"Chatterbox ({crisptts_model_id_for_log}): Generation returned empty audio.")
            return

        # Convert to numpy and ensure correct format
        audio_numpy = wav.squeeze().cpu().numpy()
        if audio_numpy.ndim == 0:
            logger.error(f"Chatterbox ({crisptts_model_id_for_log}): Generated audio is scalar.")
            return

        logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Synthesis successful, audio shape: {audio_numpy.shape}")

        # Save output
        if output_file_str:
            output_path = Path(output_file_str).with_suffix(".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                sf_chatterbox.write(str(output_path), audio_numpy, samplerate=chatterbox_model.sr)
                logger.info(f"Chatterbox ({crisptts_model_id_for_log}): Audio saved to {output_path}")
            except Exception as e_save:
                logger.error(f"Chatterbox ({crisptts_model_id_for_log}): Failed to save to {output_path}: {e_save}", exc_info=True)

        # Play if requested
        if play_direct and audio_numpy.size > 0:
            audio_int16 = (audio_numpy * 32767).astype('int16')
            play_audio(audio_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=chatterbox_model.sr)

    except Exception as e:
        logger.error(f"Chatterbox ({crisptts_model_id_for_log}): Synthesis failed: {e}", exc_info=True)
    finally:
        # Cleanup
        if temp_ref_audio:
            try:
                Path(temp_ref_audio).unlink(missing_ok=True)
            except Exception as e_cleanup:
                logger.warning(f"Chatterbox ({crisptts_model_id_for_log}): Failed to cleanup temp file: {e_cleanup}")
        
        if chatterbox_model is not None:
            del chatterbox_model
        
        if TORCH_AVAILABLE_IN_HANDLER and torch_chatterbox:
            if torch_chatterbox.cuda.is_available():
                torch_chatterbox.cuda.empty_cache()
            if hasattr(torch_chatterbox.backends, "mps") and torch_chatterbox.backends.mps.is_available() and hasattr(torch_chatterbox.mps, "empty_cache"):
                try:
                    torch_chatterbox.mps.empty_cache()
                except Exception:
                    pass
        gc.collect()

logger.info("Chatterbox handler module loaded successfully.")