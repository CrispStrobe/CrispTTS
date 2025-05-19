# handlers/coqui_tts_handler.py

import logging
import os
from pathlib import Path
import tempfile # For potential temporary storage if needed, though TTS lib handles caching
import gc

# Conditional imports
TORCH_AVAILABLE_IN_HANDLER = False
torch_coqui = None
IS_MPS_IN_HANDLER_COQUI = False
COQUI_TTS_AVAILABLE = False
CoquiTTS = None

try:
    import torch as torch_imp
    torch_coqui = torch_imp
    TORCH_AVAILABLE_IN_HANDLER = True
    IS_MPS_IN_HANDLER_COQUI = hasattr(torch_coqui.backends, "mps") and torch_coqui.backends.mps.is_available()
except ImportError:
    pass # Will be checked before use

if TORCH_AVAILABLE_IN_HANDLER:
    try:
        from TTS.api import TTS as CoquiTTS_API
        CoquiTTS = CoquiTTS_API
        COQUI_TTS_AVAILABLE = True
    except ImportError:
        logging.getLogger("CrispTTS.handlers.coqui_tts").info(
            "Coqui TTS library ('TTS') not found. Coqui TTS handler will be non-functional. "
            "Install with: pip install TTS"
        )
    except Exception as e:
        logging.getLogger("CrispTTS.handlers.coqui_tts").warning(
            f"Coqui TTS import failed with an unexpected error: {e}. Handler will be non-functional."
        )

# Relative imports
from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.coqui_tts")

def synthesize_with_coqui_tts(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    if not COQUI_TTS_AVAILABLE or not CoquiTTS:
        logger.error("Coqui TTS handler called, but the 'TTS' library is not available. Skipping.")
        return

    # --- Get configuration from model_config ---
    # `model_name` should be a Coqui TTS model string, e.g., "tts_models/de/thorsten/tacotron2-DDC"
    coqui_model_name = model_config.get("coqui_model_name")
    if not coqui_model_name:
        logger.error("Coqui TTS: 'coqui_model_name' not specified in the model configuration.")
        return

    # Coqui TTS voice parameters are often handled differently:
    # - Single-speaker models: No specific voice_id needed beyond the model_name.
    # - Multi-speaker models: Use `speaker=speaker_name` or `speaker_wav="path/to/ref.wav"` in `tts.tts()`.
    # `voice_id_override` can be used for speaker name or path to speaker WAV.
    speaker_arg = voice_id_override or model_config.get("default_coqui_speaker") # Can be speaker name or path
    speaker_wav_arg = None
    if speaker_arg and (Path(speaker_arg).is_file() and Path(speaker_arg).suffix.lower() in ['.wav', '.mp3']):
        speaker_wav_arg = str(Path(speaker_arg).resolve())
        speaker_arg = None # Don't pass both speaker and speaker_wav to tts()
        logger.info(f"Coqui TTS: Using reference speaker WAV: {speaker_wav_arg}")
    elif speaker_arg:
        logger.info(f"Coqui TTS: Using speaker ID/name: {speaker_arg}")
    else:
        logger.info("Coqui TTS: No specific speaker ID or WAV provided; using model's default speaker.")

    # Vocoder (optional, Coqui TTS often bundles vocoders or has defaults)
    coqui_vocoder_name = model_config.get("coqui_vocoder_name")
    
    # model_params_override (JSON string) can be used for `language` if model is multilingual
    # or other TTS() constructor or tts() method parameters.
    language_arg = model_config.get("language", "de") # Default to German for these models
    tts_constructor_kwargs = {}
    tts_method_kwargs = {}

    if model_params_override:
        try:
            cli_params = json.loads(model_params_override)
            if "language" in cli_params: language_arg = cli_params["language"]
            # Add other specific Coqui TTS params here if needed, separating constructor vs. method args
            # e.g., tts_method_kwargs['speed'] = cli_params.get('speed', 1.0) - if Coqui TTS supports speed
        except json.JSONDecodeError:
            logger.warning(f"Coqui TTS: Could not parse --model-params JSON: {model_params_override}")

    logger.debug(f"Coqui TTS - Model: {coqui_model_name}, SpeakerArg: {speaker_arg}, SpeakerWAV: {speaker_wav_arg}, Lang: {language_arg}")

    tts_instance = None
    try:
        if not TORCH_AVAILABLE_IN_HANDLER:
            logger.error("Coqui TTS requires PyTorch, but it's not available.")
            return

        device = "cuda" if torch_coqui.cuda.is_available() else \
                 ("mps" if IS_MPS_IN_HANDLER_COQUI else "cpu")
        logger.info(f"Coqui TTS: Initializing model '{coqui_model_name}' on device '{device}'. This might take some time...")

        tts_constructor_kwargs['model_name'] = coqui_model_name
        if coqui_vocoder_name:
            tts_constructor_kwargs['vocoder_name'] = coqui_vocoder_name
        # Newer Coqui TTS might prefer language in tts() method for some models.
        # If model is explicitly for one language (e.g. in model_name), language arg might not be needed.
        
        # Suppress Coqui TTS's own progress bars and verbose loading messages
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            tts_instance = CoquiTTS(**tts_constructor_kwargs, progress_bar=False)
            tts_instance.to(device) # Move model to device
        logger.info(f"Coqui TTS: Model '{coqui_model_name}' loaded successfully.")

        # Prepare arguments for the .tts() method
        synthesis_args = {"text": text}
        if speaker_arg: synthesis_args["speaker"] = speaker_arg
        if speaker_wav_arg: synthesis_args["speaker_wav"] = speaker_wav_arg
        if language_arg and tts_instance.is_multi_lingual: # Only pass language if model supports it
            synthesis_args["language"] = language_arg
        
        logger.info(f"Coqui TTS: Synthesizing speech... Args: {synthesis_args}")
        # Coqui TTS tts() method returns a list (audio buffer as numpy array)
        # Some versions might return a numpy array directly.
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            wav_list = tts_instance.tts(**synthesis_args)
        
        if not wav_list or not isinstance(wav_list, (list, np.ndarray)):
            logger.error("Coqui TTS: Synthesis did not return expected audio data.")
            return
        
        # If it's a list, assume it's a list of samples (older versions).
        # Newer versions might return a single numpy array.
        audio_numpy = np.array(wav_list, dtype=np.float32) if isinstance(wav_list, list) else wav_list
        if audio_numpy.ndim > 1: # Flatten if multi-channel, assuming we want mono
            audio_numpy = audio_numpy.mean(axis=0) if audio_numpy.shape[0] < audio_numpy.shape[1] else audio_numpy.mean(axis=1)

        # Convert float32 numpy array to 16-bit PCM bytes for saving/playing
        audio_int16 = (audio_numpy * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Coqui TTS models typically output at 22050 Hz or 24000 Hz or 16000Hz.
        # The actual sample rate should be obtained from the model/synthesizer if possible.
        # tts_instance.synthesizer.output_sample_rate or tts_instance.configs.sample_rate (older versions)
        # For modern TTS, it's often tts_instance.synthesizer.tts_config.audio['sample_rate']
        # Or it might be fixed based on the model.
        # Let's try to get it from the instance, default if not found.
        current_sample_rate = getattr(tts_instance, 'sample_rate', None) # For some direct TTS objects
        if not current_sample_rate and hasattr(tts_instance, 'synthesizer'):
            if hasattr(tts_instance.synthesizer, 'output_sample_rate'):
                current_sample_rate = tts_instance.synthesizer.output_sample_rate
            elif hasattr(tts_instance.synthesizer, 'tts_config') and \
                 hasattr(tts_instance.synthesizer.tts_config, 'audio') and \
                 'sample_rate' in tts_instance.synthesizer.tts_config.audio:
                current_sample_rate = tts_instance.synthesizer.tts_config.audio['sample_rate']
        
        if not current_sample_rate:
            current_sample_rate = model_config.get("sample_rate", 22050) # Fallback sample rate
            logger.warning(f"Coqui TTS: Could not determine sample rate from model, using default/config: {current_sample_rate}Hz")
        else:
            logger.info(f"Coqui TTS: Determined model sample rate: {current_sample_rate}Hz")


        logger.info(f"Coqui TTS: Synthesis successful, {len(audio_bytes)} bytes generated.")
        if output_file_str:
            effective_output_file_wav = Path(output_file_str).with_suffix(".wav")
            save_audio(audio_bytes, str(effective_output_file_wav), source_is_path=False, input_format="pcm_s16le", sample_rate=current_sample_rate)
        if play_direct:
            play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=current_sample_rate)

    except RuntimeError as e_runtime:
        if "Attempting to deserialize object on a CUDA device" in str(e_runtime) and device == "mps":
            logger.error(f"Coqui TTS: Model loading error on MPS. Try running with CPU or ensure CUDA availability if model requires it. Error: {e_runtime}")
        else:
            logger.error(f"Coqui TTS: Runtime error during synthesis: {e_runtime}", exc_info=True)
    except Exception as e:
        logger.error(f"Coqui TTS: Synthesis failed: {e}", exc_info=True)
    finally:
        del tts_instance # Ensure model is released
        if TORCH_AVAILABLE_IN_HANDLER and torch_coqui and torch_coqui.cuda.is_available():
            torch_coqui.cuda.empty_cache()
            logger.debug("Coqui TTS: Cleared CUDA cache.")
        gc.collect()