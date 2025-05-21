# handlers/coqui_tts_handler.py

import logging
import os
from pathlib import Path
import gc
import json
import numpy as np
import collections # For defaultdict fix
import builtins    # For builtins.dict fix

# Conditional imports for PyTorch and Coqui TTS
TORCH_AVAILABLE_IN_HANDLER = False
torch_coqui = None
IS_MPS_IN_HANDLER_COQUI = False 
COQUI_TTS_AVAILABLE = False
CoquiTTS_API_CLASS = None 

CoquiRAdam_CLASS = None 
RADAM_IMPORTED_SUCCESSFULLY = False
TORCH_SERIALIZATION_AVAILABLE = False

logger_init = logging.getLogger("CrispTTS.handlers.coqui_tts.init")

try:
    import torch as torch_imp
    torch_coqui = torch_imp
    TORCH_AVAILABLE_IN_HANDLER = True
    if hasattr(torch_coqui.backends, "mps") and torch_coqui.backends.mps.is_available():
        IS_MPS_IN_HANDLER_COQUI = True
    
    import torch.serialization 
    TORCH_SERIALIZATION_AVAILABLE = True
    
    try:
        from TTS.utils.radam import RAdam 
        CoquiRAdam_CLASS = RAdam
        RADAM_IMPORTED_SUCCESSFULLY = True
        logger_init.info("Successfully imported TTS.utils.radam.RAdam for Coqui model fix.")
    except ImportError:
        logger_init.info("TTS.utils.radam.RAdam not found. Fix for older Coqui models (like DCA) may not apply if needed.")

except ImportError:
    logger_init.info(
        "PyTorch (or a component like torch.serialization) not found. Coqui TTS handler will be limited."
    )

if TORCH_AVAILABLE_IN_HANDLER:
    try:
        from TTS.api import TTS as CoquiTTS_Module
        CoquiTTS_API_CLASS = CoquiTTS_Module
        COQUI_TTS_AVAILABLE = True
    except ImportError:
        logger_init.info(
            "Coqui TTS library ('TTS') not found. Coqui TTS handler will be non-functional. "
            "Install with: pip install TTS"
        )
    except Exception as e: 
        logger_init.error(
            f"Coqui TTS library import failed: {e}. Handler will be non-functional.",
            exc_info=True
        )

from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.coqui_tts")

# List of model names known to potentially require these unpickle fixes
MODELS_REQUIRING_UNPICKLE_FIXES = [
    "tts_models/de/thorsten/tacotron2-DCA"
    # Add other older model IDs here if they show similar unpickling errors for basic types
]

def synthesize_with_coqui_tts(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'coqui_tts_unknown') 

    if not COQUI_TTS_AVAILABLE or not CoquiTTS_API_CLASS:
        logger.error(f"Coqui TTS handler ({crisptts_model_id_for_log}): 'TTS' library not available. Skipping.")
        return
    if not torch_coqui:
        logger.error(f"Coqui TTS handler ({crisptts_model_id_for_log}): PyTorch not available. Skipping.")
        return

    coqui_model_name = model_config.get("coqui_model_name")
    if not coqui_model_name:
        logger.error(f"Coqui TTS ({crisptts_model_id_for_log}): 'coqui_model_name' not in config. Skipping.")
        return

    effective_voice_input = voice_id_override
    is_single_speaker_model_type = model_config.get("default_coqui_speaker") is None
    
    if is_single_speaker_model_type and effective_voice_input == "default_speaker":
        logger.info(f"Coqui TTS ({coqui_model_name}): Single-speaker model. Placeholder '{effective_voice_input}' -> using intrinsic speaker.")
        effective_voice_input = None 

    speaker_arg_for_tts = effective_voice_input or model_config.get("default_coqui_speaker")
    speaker_wav_arg_for_tts = None

    if speaker_arg_for_tts and isinstance(speaker_arg_for_tts, str) and \
       (Path(speaker_arg_for_tts).is_file() and Path(speaker_arg_for_tts).suffix.lower() in ['.wav', '.mp3']):
        ref_audio_file_orig = Path(speaker_arg_for_tts)
        if not ref_audio_file_orig.is_absolute():
            project_root = Path(__file__).resolve().parent.parent 
            ref_audio_file_orig = (project_root / ref_audio_file_orig).resolve()
        
        if ref_audio_file_orig.exists():
            speaker_wav_arg_for_tts = str(ref_audio_file_orig)
            speaker_arg_for_tts = None 
            logger.info(f"Coqui TTS ({coqui_model_name}): Using reference speaker WAV: {speaker_wav_arg_for_tts}")
        else:
            logger.error(f"Coqui TTS ({coqui_model_name}): Ref speaker WAV not found: {ref_audio_file_orig}.")
            speaker_arg_for_tts = None 
            
    elif speaker_arg_for_tts: 
        logger.info(f"Coqui TTS ({coqui_model_name}): Using speaker ID/name: {speaker_arg_for_tts}")
    else: 
        logger.info(f"Coqui TTS ({coqui_model_name}): No specific speaker parameter; using model's intrinsic/default speaker.")
    
    config_lang = model_config.get("language") 
    cli_lang = None
    if model_params_override:
        try:
            cli_params = json.loads(model_params_override)
            if "language" in cli_params: cli_lang = cli_params["language"]
        except json.JSONDecodeError:
            logger.warning(f"Coqui TTS ({coqui_model_name}): Could not parse --model-params: {model_params_override}")
    
    language_to_use_if_multilingual = cli_lang or config_lang 
    
    coqui_vocoder_name = model_config.get("coqui_vocoder_name")
    tts_constructor_kwargs = {}

    logger.debug(f"Coqui TTS ({coqui_model_name}) - Params before TTS init: Speaker ID='{speaker_arg_for_tts}', Speaker WAV='{speaker_wav_arg_for_tts}', Language (intended if multi)='{language_to_use_if_multilingual}'")

    tts_instance = None
    try:
        if coqui_model_name in MODELS_REQUIRING_UNPICKLE_FIXES:
            globals_to_add_for_pickle = []
            if RADAM_IMPORTED_SUCCESSFULLY and CoquiRAdam_CLASS:
                globals_to_add_for_pickle.append(CoquiRAdam_CLASS)
            if hasattr(collections, 'defaultdict'): 
                globals_to_add_for_pickle.append(collections.defaultdict)
            if hasattr(builtins, 'dict'): # Explicitly add builtins.dict
                 globals_to_add_for_pickle.append(builtins.dict)
            
            if globals_to_add_for_pickle and TORCH_SERIALIZATION_AVAILABLE and hasattr(torch_coqui, 'serialization'):
                try:
                    torch_coqui.serialization.add_safe_globals(globals_to_add_for_pickle)
                    added_names = [c.__name__ if hasattr(c, '__name__') else str(c) for c in globals_to_add_for_pickle]
                    logger.info(f"Coqui TTS: Added {len(globals_to_add_for_pickle)} item(s) ({added_names}) to torch safe globals for {coqui_model_name}.")
                except Exception as e_safe_global:
                    logger.warning(f"Coqui TTS: Failed to add items to torch safe globals for {coqui_model_name}: {e_safe_global}. Model loading might still fail.", exc_info=True)
            elif coqui_model_name in MODELS_REQUIRING_UNPICKLE_FIXES:
                 logger.warning(f"Coqui TTS: Model {coqui_model_name} may require unpickle fixes, but not all needed classes could be prepared or torch.serialization unavailable.")

        target_device = "cpu"
        if torch_coqui.cuda.is_available(): target_device = "cuda"
        elif IS_MPS_IN_HANDLER_COQUI: target_device = "mps"
        
        logger.info(f"Coqui TTS: Target device: {target_device} for '{coqui_model_name}'.")

        tts_constructor_kwargs['model_name'] = coqui_model_name
        if coqui_vocoder_name: tts_constructor_kwargs['vocoder_name'] = coqui_vocoder_name
        
        force_cpu_init = (target_device == "mps") # Always init on CPU if target is MPS for older TTS lib
        constructor_gpu_param = not force_cpu_init if target_device != "cpu" else False

        if force_cpu_init and target_device == "mps":
            logger.info(f"Coqui TTS ({coqui_model_name}): Target MPS. Using CPU init (gpu=False) strategy to avoid CUDA asserts, then will attempt move to MPS.")
        
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=not logger.isEnabledFor(logging.DEBUG)):
            tts_instance = CoquiTTS_API_CLASS(gpu=constructor_gpu_param, **tts_constructor_kwargs, progress_bar=False)
            
            if constructor_gpu_param is False and target_device != "cpu" and hasattr(tts_instance, 'to'):
                logger.info(f"Coqui TTS: Attempting to move model {coqui_model_name} from CPU to {target_device} post-init.")
                tts_instance.to(target_device)
            elif constructor_gpu_param and hasattr(tts_instance, 'to') and hasattr(tts_instance, 'device') and str(tts_instance.device) != target_device:
                 if target_device != "cpu" or (target_device == "cpu" and str(tts_instance.device) != "cpu"):
                    logger.info(f"Coqui TTS: Model loaded on {tts_instance.device}, ensuring it's on target {target_device}.")
                    tts_instance.to(target_device)

        current_device_str = str(tts_instance.device) if hasattr(tts_instance, 'device') else 'device attribute not found'
        logger.info(f"Coqui TTS: Model '{coqui_model_name}' loaded. Effective device: {current_device_str}")

        synthesis_args = {"text": text}
        if speaker_arg_for_tts: synthesis_args["speaker"] = speaker_arg_for_tts
        if speaker_wav_arg_for_tts: synthesis_args["speaker_wav"] = speaker_wav_arg_for_tts
        
        model_is_multilingual = hasattr(tts_instance, 'is_multi_lingual') and tts_instance.is_multi_lingual
        
        if language_to_use_if_multilingual: 
            if model_is_multilingual:
                model_languages = getattr(tts_instance, 'languages', [])
                if language_to_use_if_multilingual in model_languages:
                    synthesis_args["language"] = language_to_use_if_multilingual
                    logger.debug(f"Coqui TTS ({coqui_model_name}): Passing language '{language_to_use_if_multilingual}'.")
                else:
                    logger.warning(f"Coqui TTS ({coqui_model_name}): Language '{language_to_use_if_multilingual}' not in model languages: {model_languages}. Omitting 'language' from .tts() call.")
            else: 
                logger.debug(f"Coqui TTS ({coqui_model_name}): Model is not multilingual. Not passing 'language' argument (specified: '{language_to_use_if_multilingual}').")
        else:
            logger.debug(f"Coqui TTS ({coqui_model_name}): No language specified for synthesis.")
        
        logger.info(f"Coqui TTS: Synthesizing speech with args: {synthesis_args}")
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=not logger.isEnabledFor(logging.DEBUG)):
            wav_output = tts_instance.tts(**synthesis_args) 
        
        if wav_output is None or (isinstance(wav_output, (list, np.ndarray)) and not np.asarray(wav_output).size):
            logger.error(f"Coqui TTS ({coqui_model_name}): Synthesis returned no audio data or empty data.")
            return

        if isinstance(wav_output, list): audio_numpy = np.array(wav_output, dtype=np.float32)
        elif isinstance(wav_output, np.ndarray): audio_numpy = wav_output.astype(np.float32)
        else:
            logger.error(f"Coqui TTS ({coqui_model_name}): Synthesis returned unexpected data type: {type(wav_output)}")
            return
        
        if audio_numpy.ndim > 1: 
            audio_numpy = audio_numpy.squeeze()
            if audio_numpy.ndim > 1:
                 channel_dim = np.argmin(audio_numpy.shape)
                 audio_numpy = audio_numpy.mean(axis=channel_dim)

        audio_normalized = np.clip(audio_numpy, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        current_sample_rate = model_config.get("sample_rate", 22050)
        try:
            sr_from_model = None
            if hasattr(tts_instance, 'synthesizer') and tts_instance.synthesizer and hasattr(tts_instance.synthesizer, 'output_sample_rate') and tts_instance.synthesizer.output_sample_rate:
                sr_from_model = tts_instance.synthesizer.output_sample_rate
            elif hasattr(tts_instance, 'config') and tts_instance.config and hasattr(tts_instance.config, 'audio') and isinstance(tts_instance.config.audio, dict) and 'sample_rate' in tts_instance.config.audio and tts_instance.config.audio['sample_rate']:
                sr_from_model = tts_instance.config.audio['sample_rate']
            
            if sr_from_model and isinstance(sr_from_model, int) and sr_from_model > 0:
                current_sample_rate = sr_from_model
            logger.info(f"Coqui TTS: Using sample rate: {current_sample_rate}Hz")
        except Exception as e_sr:
            logger.warning(f"Coqui TTS: Could not get SR from model, using config fallback {current_sample_rate}Hz. Error: {e_sr}")

        logger.info(f"Coqui TTS ({coqui_model_name}): Synthesis successful, {len(audio_bytes)} bytes generated.")
        if output_file_str:
            effective_output_file_wav = Path(output_file_str).with_suffix(".wav")
            save_audio(audio_bytes, str(effective_output_file_wav), source_is_path=False, input_format="pcm_s16le", sample_rate=current_sample_rate)
        if play_direct:
            play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=current_sample_rate)

    except AssertionError as e_assert:
        if "CUDA is not availabe on this machine" in str(e_assert): 
            logger.error(f"Coqui TTS ({coqui_model_name}): Failed due to CUDA assertion by the TTS library. Error: {e_assert}", exc_info=True)
        else: 
            logger.error(f"Coqui TTS ({coqui_model_name}): Assertion error: {e_assert}", exc_info=True)
    except _pickle.UnpicklingError as e_pickle: # Catch specific UnpicklingError
        logger.error(f"Coqui TTS ({coqui_model_name}): Unpickling error during model loading. This often happens with older models and newer PyTorch versions. Error: {e_pickle}", exc_info=True)
    except RuntimeError as e_runtime:
        device_for_log = "unknown"
        if 'target_device' in locals() and target_device: device_for_log = target_device
        if "Attempting to deserialize object on a CUDA device" in str(e_runtime) and device_for_log == "mps":
            logger.error(f"Coqui TTS ({coqui_model_name}): MPS loading error: {e_runtime}", exc_info=True)
        else:
            logger.error(f"Coqui TTS ({coqui_model_name}): Runtime error: {e_runtime}", exc_info=True)
    except Exception as e:
        logger.error(f"Coqui TTS ({coqui_model_name}): Synthesis failed: {e}", exc_info=True)
    finally:
        if 'tts_instance' in locals() and tts_instance is not None:
            if hasattr(tts_instance, 'model') and tts_instance.model and hasattr(tts_instance.model, 'cpu'):
                try: tts_instance.model.cpu()
                except Exception as e_cpu: logger.debug(f"Coqui TTS: Error moving model to CPU for cleanup: {e_cpu}")
            del tts_instance
            tts_instance = None
        
        if TORCH_AVAILABLE_IN_HANDLER and torch_coqui:
            if torch_coqui.cuda.is_available(): torch_coqui.cuda.empty_cache()
            if IS_MPS_IN_HANDLER_COQUI and hasattr(torch_coqui.mps, "empty_cache"):
                try: torch_coqui.mps.empty_cache()
                except Exception as e_mps_clear: logger.debug(f"Coqui TTS: Error clearing MPS cache: {e_mps_clear}")
        gc.collect()