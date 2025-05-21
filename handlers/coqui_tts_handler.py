# handlers/coqui_tts_handler.py

import logging
import os
from pathlib import Path
# import tempfile # Not strictly needed as Coqui TTS handles its own caching
import gc
import json # For parsing model_params_override

# Import numpy for audio processing
import numpy as np

# Conditional imports for PyTorch and Coqui TTS
TORCH_AVAILABLE_IN_HANDLER = False
torch_coqui = None
IS_MPS_IN_HANDLER_COQUI = False # For Apple Silicon MPS
COQUI_TTS_AVAILABLE = False
CoquiTTS = None # This will be the TTS class from TTS.api

try:
    import torch as torch_imp
    torch_coqui = torch_imp
    TORCH_AVAILABLE_IN_HANDLER = True
    if hasattr(torch_coqui.backends, "mps") and torch_coqui.backends.mps.is_available():
        IS_MPS_IN_HANDLER_COQUI = True
except ImportError:
    # This logger is for messages during the import phase of this module
    logging.getLogger("CrispTTS.handlers.coqui_tts.init").info(
        "PyTorch not found. Coqui TTS handler will be limited."
    )

if TORCH_AVAILABLE_IN_HANDLER:
    try:
        from TTS.api import TTS as CoquiTTS_API
        CoquiTTS = CoquiTTS_API
        COQUI_TTS_AVAILABLE = True
    except ImportError:
        logging.getLogger("CrispTTS.handlers.coqui_tts.init").info(
            "Coqui TTS library ('TTS') not found. Coqui TTS handler will be non-functional. "
            "Install with: pip install TTS"
        )
    except Exception as e: # Catch any other error during TTS import
        logging.getLogger("CrispTTS.handlers.coqui_tts.init").error(
            f"Coqui TTS library import failed with an unexpected error: {e}. Handler will be non-functional.",
            exc_info=True
        )

# Relative imports from your project's utils
from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.coqui_tts")

def synthesize_with_coqui_tts(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    if not COQUI_TTS_AVAILABLE or not CoquiTTS:
        # This check is redundant if the logger messages above already fired, but good for safety.
        logger.error("Coqui TTS handler called, but 'TTS' library or its dependencies are not available. Skipping.")
        return

    coqui_model_name = model_config.get("coqui_model_name")
    if not coqui_model_name:
        logger.error("Coqui TTS: 'coqui_model_name' not specified in the model configuration. Skipping.")
        return

    # --- Voice/Speaker Argument Handling ---
    effective_voice_input = voice_id_override
    
    # Check if this model is configured as single-speaker and if a placeholder is used
    is_single_speaker_model = model_config.get("default_coqui_speaker") is None
    if is_single_speaker_model and effective_voice_input == "default_speaker":
        logger.info(f"Coqui TTS: Single-speaker model ('{coqui_model_name}'). Placeholder '{effective_voice_input}' received, will use intrinsic model speaker.")
        effective_voice_input = None # This ensures speaker_arg becomes None for TTS().tts()

    # effective_voice_input is now either an actual speaker_id, a path, or None.
    # default_coqui_speaker from config is None for Thorsten models.
    speaker_arg_for_tts = effective_voice_input or model_config.get("default_coqui_speaker")
    speaker_wav_arg_for_tts = None

    if speaker_arg_for_tts and isinstance(speaker_arg_for_tts, str) and \
       (Path(speaker_arg_for_tts).is_file() and Path(speaker_arg_for_tts).suffix.lower() in ['.wav', '.mp3']):
        # If speaker_arg_for_tts is a path to a WAV/MP3 file for voice cloning
        speaker_wav_arg_for_tts = str(Path(speaker_arg_for_tts).resolve())
        speaker_arg_for_tts = None # Clear speaker_arg_for_tts as we're using speaker_wav_arg_for_tts
        logger.info(f"Coqui TTS: Using reference speaker WAV: {speaker_wav_arg_for_tts}")
    elif speaker_arg_for_tts: # If speaker_arg_for_tts is a string (could be a speaker_id for multi-speaker)
        logger.info(f"Coqui TTS: Using speaker ID/name: {speaker_arg_for_tts}")
    else: # speaker_arg_for_tts is None (means use intrinsic voice for single-speaker, or default for multi-speaker if model has one)
        logger.info(f"Coqui TTS ({coqui_model_name}): No specific speaker parameter; using model's default/intrinsic speaker.")
    
    # --- Other Parameters ---
    language_arg = model_config.get("language", "de") # Default to German
    coqui_vocoder_name = model_config.get("coqui_vocoder_name") # Optional

    tts_constructor_kwargs = {} # For TTS()
    # tts_method_kwargs = {} # For .tts() - Coqui TTS .tts() mainly takes text, speaker, speaker_wav, language

    if model_params_override:
        try:
            cli_params = json.loads(model_params_override)
            if "language" in cli_params: 
                language_arg = cli_params["language"]
            # Example: if Coqui TTS().tts() supported speed directly
            # if "speed" in cli_params: tts_method_kwargs['speed'] = cli_params.get('speed')
        except json.JSONDecodeError:
            logger.warning(f"Coqui TTS: Could not parse --model-params JSON: {model_params_override}")

    logger.debug(f"Coqui TTS - Model: {coqui_model_name}, Speaker Name/ID to be used: {speaker_arg_for_tts}, Speaker WAV to be used: {speaker_wav_arg_for_tts}, Language: {language_arg}")

    tts_instance = None
    try:
        # Determine device
        if torch_coqui and torch_coqui.cuda.is_available():
            device = "cuda"
        elif torch_coqui and IS_MPS_IN_HANDLER_COQUI: # Check if torch_coqui is not None
            device = "mps"
        else:
            device = "cpu"
        
        logger.info(f"Coqui TTS: Attempting to initialize model '{coqui_model_name}' on device '{device}'. This may take some time...")

        tts_constructor_kwargs['model_name'] = coqui_model_name
        if coqui_vocoder_name:
            tts_constructor_kwargs['vocoder_name'] = coqui_vocoder_name
        
        # TTS class constructor can take `gpu` (bool or str for device id) or automatically uses available GPU.
        # Let's rely on auto-detection or force CPU if no GPU, then move model if needed.
        # Newer TTS versions might handle device selection better in the constructor.
        # Forcing CPU then moving to GPU/MPS can sometimes avoid initialization issues on certain setups.
        
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), 
                            suppress_stderr=not logger.isEnabledFor(logging.DEBUG)): # Suppress verbose loading
            tts_instance = CoquiTTS(**tts_constructor_kwargs, progress_bar=False) # gpu= (device!="cpu")
            if device != "cpu" and hasattr(tts_instance, 'to'): # Ensure model is on the target device
                 tts_instance.to(device)

        logger.info(f"Coqui TTS: Model '{coqui_model_name}' loaded. Current device: {tts_instance.device if hasattr(tts_instance, 'device') else 'unknown'}")

        # Prepare arguments for the .tts() method
        synthesis_args = {"text": text}
        if speaker_arg_for_tts: 
            synthesis_args["speaker"] = speaker_arg_for_tts
        if speaker_wav_arg_for_tts: 
            synthesis_args["speaker_wav"] = speaker_wav_arg_for_tts
        
        # Language handling for .tts() method
        model_is_multilingual = hasattr(tts_instance, 'is_multi_lingual') and tts_instance.is_multi_lingual
        if language_arg:
            if model_is_multilingual:
                if language_arg in tts_instance.languages:
                    synthesis_args["language"] = language_arg
                else:
                    logger.warning(f"Coqui TTS: Language '{language_arg}' not in model's supported languages: {tts_instance.languages}. Attempting without language arg.")
            elif coqui_model_name and (f"/{language_arg}/" in coqui_model_name or coqui_model_name.startswith(f"{language_arg}/")):
                 logger.debug(f"Coqui TTS: Model name suggests it's monolingual ('{language_arg}'). Not passing explicit language arg to .tts().")
            else: # Not explicitly multilingual, but language_arg is provided. Pass it cautiously.
                 logger.debug(f"Coqui TTS: Model not reported as multilingual. Passing language '{language_arg}' to .tts() method.")
                 synthesis_args["language"] = language_arg

        logger.info(f"Coqui TTS: Synthesizing speech with args: {synthesis_args}")
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), 
                            suppress_stderr=not logger.isEnabledFor(logging.DEBUG)):
            wav_output = tts_instance.tts(**synthesis_args) 
        
        if wav_output is None or (isinstance(wav_output, (list, np.ndarray)) and not np.asarray(wav_output).size):
            logger.error("Coqui TTS: Synthesis did not return any audio data or returned empty data.")
            return

        if isinstance(wav_output, list): # Older TTS versions might return list of floats
            audio_numpy = np.array(wav_output, dtype=np.float32)
        elif isinstance(wav_output, np.ndarray):
            audio_numpy = wav_output.astype(np.float32)
        else:
            logger.error(f"Coqui TTS: Synthesis returned unexpected data type: {type(wav_output)}")
            return
        
        # Ensure audio is 1D (mono)
        if audio_numpy.ndim > 1: 
            audio_numpy = audio_numpy.squeeze() # Remove single dimensions
            if audio_numpy.ndim > 1: # If still >1D (e.g. stereo [2, N] or [N, 2]), take mean
                 logger.warning(f"Coqui TTS: Output is multi-channel ({audio_numpy.shape}), taking mean for mono.")
                 # Check which dimension is smaller (likely channels)
                 channel_dim = np.argmin(audio_numpy.shape)
                 audio_numpy = audio_numpy.mean(axis=channel_dim)

        # Normalize and convert to 16-bit PCM
        audio_normalized = np.clip(audio_numpy, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        current_sample_rate = model_config.get("sample_rate", 22050) # Fallback from config
        try: # Try to get actual sample rate from TTS instance
            if hasattr(tts_instance, 'synthesizer') and hasattr(tts_instance.synthesizer, 'output_sample_rate') and tts_instance.synthesizer.output_sample_rate:
                current_sample_rate = tts_instance.synthesizer.output_sample_rate
            elif hasattr(tts_instance, 'model') and hasattr(tts_instance.model, 'config') and \
                 hasattr(tts_instance.model.config, 'audio') and 'sample_rate' in tts_instance.model.config.audio and tts_instance.model.config.audio['sample_rate']:
                current_sample_rate = tts_instance.model.config.audio['sample_rate']
            elif hasattr(tts_instance, 'config') and hasattr(tts_instance.config, 'audio') and 'sample_rate' in tts_instance.config.audio and tts_instance.config.audio['sample_rate']:
                current_sample_rate = tts_instance.config.audio['sample_rate']

            logger.info(f"Coqui TTS: Using sample rate: {current_sample_rate}Hz")
        except Exception as e_sr:
            logger.warning(f"Coqui TTS: Could not reliably determine sample rate from model, using config fallback {current_sample_rate}Hz. Error: {e_sr}")


        logger.info(f"Coqui TTS: Synthesis successful, {len(audio_bytes)} bytes generated.")
        if output_file_str:
            effective_output_file_wav = Path(output_file_str).with_suffix(".wav")
            save_audio(audio_bytes, str(effective_output_file_wav), source_is_path=False, input_format="pcm_s16le", sample_rate=current_sample_rate)
        if play_direct:
            play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=current_sample_rate)

    except RuntimeError as e_runtime:
        # Check if device is available for more specific MPS error messages
        current_device_for_log = "unknown"
        if 'device' in locals(): current_device_for_log = device

        if "Attempting to deserialize object on a CUDA device" in str(e_runtime) and current_device_for_log == "mps":
            logger.error(f"Coqui TTS ({coqui_model_name}): Model loading error on MPS. Try with CPU or ensure CUDA if model needs it. Error: {e_runtime}", exc_info=True)
        else:
            logger.error(f"Coqui TTS ({coqui_model_name}): Runtime error during synthesis: {e_runtime}", exc_info=True)
    except Exception as e:
        logger.error(f"Coqui TTS ({coqui_model_name}): Synthesis failed: {e}", exc_info=True)
    finally:
        del tts_instance # Ensure model is released
        if TORCH_AVAILABLE_IN_HANDLER and torch_coqui and torch_coqui.cuda.is_available():
            torch_coqui.cuda.empty_cache()
            logger.debug("Coqui TTS: Cleared CUDA cache.")
        gc.collect()