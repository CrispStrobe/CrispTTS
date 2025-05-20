# handlers/outetts_handler.py

import logging
import json
from pathlib import Path
import os
import gc
import platform
import time

# Conditional Imports (same as before)
OUTETTS_AVAILABLE = False
OuteTTSInterface_h = None
OuteTTSModelConfig_h = None
OuteTTSModels_h = None
OuteTTSBackend_h = None
OuteTTSLlamaCppQuantization_h = None
OuteTTSGenerationConfig_h = None
OuteTTSGenerationType_h = None
OuteTTSSamplerConfig_h = None
OuteTTSInterfaceVersion_h = None

TORCH_AVAILABLE_IN_HANDLER = False
torch_oute = None
IS_MPS_HANDLER = False

logger_init = logging.getLogger("CrispTTS.handlers.outetts_init")

try:
    import torch
    torch_oute = torch
    TORCH_AVAILABLE_IN_HANDLER = True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        IS_MPS_HANDLER = True
    logger_init.info(f"PyTorch loaded for OuteTTS handler. MPS available: {IS_MPS_HANDLER}")
except ImportError:
    logger_init.info("PyTorch not found. OuteTTS functionality, especially HF backend, might be limited or unavailable.")

if TORCH_AVAILABLE_IN_HANDLER:
    try:
        from outetts import (
            Interface as _Interface, ModelConfig as _ModelConfig, Models as _Models, Backend as _Backend,
            LlamaCppQuantization as _LlamaCppQuantization, GenerationConfig as _GenerationConfig,
            GenerationType as _GenerationType, SamplerConfig as _SamplerConfig, InterfaceVersion as _InterfaceVersion
        )
        OuteTTSInterface_h, OuteTTSModelConfig_h, OuteTTSModels_h, OuteTTSBackend_h = _Interface, _ModelConfig, _Models, _Backend
        OuteTTSLlamaCppQuantization_h, OuteTTSGenerationConfig_h = _LlamaCppQuantization, _GenerationConfig
        OuteTTSGenerationType_h, OuteTTSSamplerConfig_h, OuteTTSInterfaceVersion_h = _GenerationType, _SamplerConfig, _InterfaceVersion
        OUTETTS_AVAILABLE = True
        logger_init.info("OuteTTS library and its components loaded successfully.")
    except ImportError:
        logger_init.warning("OuteTTS library not found (ImportError). OuteTTS handler will be non-functional.")
    except AttributeError as e_attr:
        logger_init.warning(f"OuteTTS library import failed (AttributeError: {e_attr}). Check components/protobuf. Non-functional.")
    except Exception as e_outetts_other:
        logger_init.warning(f"OuteTTS library import failed ({e_outetts_other}). Non-functional.")
else:
    logger_init.warning("PyTorch not available, OuteTTS library not loaded. OuteTTS handler non-functional.")

from utils import _prepare_oute_speaker_ref, SuppressOutput # Removed save_audio, play_audio as OuteTTS object handles it
from config import OUTETTS_VERSION_STRING_MODEL_CONFIG_DATA

logger = logging.getLogger("CrispTTS.handlers.outetts")

def synthesize_with_outetts_local(model_config, text, voice_id_or_path_override, model_params_override, output_file_str, play_direct):
    # Get the actual model ID key from main.py for more accurate logging
    # This assumes main.py adds this key to model_config before calling.
    # If not, it will default. A better way is to pass model_id as a separate arg to the handler.
    # For now, using the provided model_id from main args directly in log messages.
    # The 'model_id_for_logs' variable will take the true model_id (e.g. "oute_hf") later.

    if not OUTETTS_AVAILABLE or not all([
        OuteTTSInterface_h, OuteTTSModelConfig_h, OuteTTSModels_h, OuteTTSBackend_h,
        OuteTTSGenerationConfig_h, OuteTTSGenerationType_h, OuteTTSSamplerConfig_h, OuteTTSInterfaceVersion_h
    ]):
        logger.error(f"OuteTTS (Model: {model_config.get('crisptts_model_id', 'N/A')}) - Library or essential components are not available. Skipping synthesis.")
        return

    # Use a passed model_id for logging if available, otherwise a generic one.
    # It's better if main.py passes args.model_id to this handler.
    # For this fix, I'll assume model_config might contain a 'crisptts_model_id' field added by main.py.
    # If not, the log will be less specific.
    crisptts_model_id = model_config.get('crisptts_model_id', 'outetts_unknown_id')
    logger.debug(f"OuteTTS ({crisptts_model_id}) - Text: '{text[:50]}...' | Voice/Ref: {voice_id_or_path_override}")

    backend_choice_from_config = model_config.get("backend_to_use")
    actual_backend_enum = None
    if isinstance(backend_choice_from_config, OuteTTSBackend_h):
        actual_backend_enum = backend_choice_from_config
    elif isinstance(backend_choice_from_config, str):
        if "LLAMACPP" in backend_choice_from_config.upper(): actual_backend_enum = OuteTTSBackend_h.LLAMACPP
        elif "HF" in backend_choice_from_config.upper(): actual_backend_enum = OuteTTSBackend_h.HF
        else:
            logger.error(f"OuteTTS ({crisptts_model_id}) - Unknown backend string: {backend_choice_from_config}.")
            return
    else:
        logger.error(f"OuteTTS ({crisptts_model_id}) - Invalid 'backend_to_use' type: {type(backend_choice_from_config)}.")
        return

    outetts_model_enum_from_config = model_config.get("outetts_model_enum")
    actual_model_enum = None # This will hold the OuteTTS.Models enum member
    if isinstance(outetts_model_enum_from_config, OuteTTSModels_h):
        actual_model_enum = outetts_model_enum_from_config
    elif isinstance(outetts_model_enum_from_config, str):
        try:
            model_enum_name = outetts_model_enum_from_config.replace("_STR_FALLBACK", "")
            actual_model_enum = getattr(OuteTTSModels_h, model_enum_name)
        except AttributeError:
            logger.warning(f"OuteTTS ({crisptts_model_id}) - Could not map model string '{outetts_model_enum_from_config}' to OuteTTS.Models enum. Will proceed if backend allows.")
            actual_model_enum = None # Explicitly set to None
    elif outetts_model_enum_from_config is None:
        logger.info(f"OuteTTS ({crisptts_model_id}) - 'outetts_model_enum' is None in config. This is acceptable for HF backend if other info is present.")
        actual_model_enum = None
    else: # Catch other invalid types
        logger.error(f"OuteTTS ({crisptts_model_id}) - Invalid 'outetts_model_enum' type in config: {type(outetts_model_enum_from_config)}. Expected OuteTTS.Models enum, string, or None.")
        return

    logger.info(f"OuteTTS ({crisptts_model_id}) - Resolved Target Backend: {actual_backend_enum}, Attempted Model Enum: {actual_model_enum}")

    interface_config_obj, outetts_interface = None, None
    temp_speaker_file_to_delete, speaker_profile_obj, output_audio_obj = None, None, None

    try:
        logger.info(f"OuteTTS ({crisptts_model_id}) - Preparing ModelConfig...")
        model_config_constructor_params = {"backend": actual_backend_enum}

        if actual_backend_enum == OuteTTSBackend_h.LLAMACPP:
            if not actual_model_enum: # LlamaCPP backend *requires* a valid model enum.
                logger.error(f"OuteTTS ({crisptts_model_id}) - LlamaCPP backend: 'outetts_model_enum' is missing or invalid in config. Cannot proceed.")
                return
            model_config_constructor_params["model"] = actual_model_enum
            if not OuteTTSLlamaCppQuantization_h:
                logger.error(f"OuteTTS ({crisptts_model_id}) - OuteTTS.LlamaCppQuantization enum not available. Cannot configure LlamaCPP.")
                return
            quant_cfg = model_config.get("quantization_to_use")
            if isinstance(quant_cfg, OuteTTSLlamaCppQuantization_h): model_config_constructor_params["quantization"] = quant_cfg
            elif isinstance(quant_cfg, str): model_config_constructor_params["quantization"] = getattr(OuteTTSLlamaCppQuantization_h, quant_cfg.replace("_STR_FALLBACK", ""))
            else: model_config_constructor_params["quantization"] = OuteTTSLlamaCppQuantization_h.FP16
            logger.debug(f"OuteTTS ({crisptts_model_id}) - LlamaCPP auto_config params: {model_config_constructor_params}")
            interface_config_obj = OuteTTSModelConfig_h.auto_config(**model_config_constructor_params)

        elif actual_backend_enum == OuteTTSBackend_h.HF:
            full_hf_model_id_for_outetts = None
            if actual_model_enum and hasattr(actual_model_enum, 'value'): # Primary: use model_enum if available
                hf_model_id_base = actual_model_enum.value
                full_hf_model_id_for_outetts = f"OuteAI/{hf_model_id_base}"
                logger.info(f"OuteTTS ({crisptts_model_id}) - HF: Derived 'full_hf_model_id' from 'actual_model_enum.value': {full_hf_model_id_for_outetts}")
            else: # Fallback: try to use tokenizer_path from config as the full HF model ID
                tokenizer_path_from_config = model_config.get("tokenizer_path")
                if tokenizer_path_from_config and isinstance(tokenizer_path_from_config, str) and "/" in tokenizer_path_from_config:
                    full_hf_model_id_for_outetts = tokenizer_path_from_config
                    logger.info(f"OuteTTS ({crisptts_model_id}) - HF: Using 'full_hf_model_id' directly from 'tokenizer_path' config: {full_hf_model_id_for_outetts}")
                else:
                    logger.error(f"OuteTTS ({crisptts_model_id}) - HF: Cannot determine Hugging Face model ID. 'outetts_model_enum' is invalid AND 'tokenizer_path' is not a valid HF ID string ('{tokenizer_path_from_config}').")
                    return
            
            model_config_constructor_params["model"] = full_hf_model_id_for_outetts # For HF, 'model' is the string ID

            model_version_str = model_config.get("outetts_model_version_str", "1.0")
            crisptts_version_cfg = OUTETTS_VERSION_STRING_MODEL_CONFIG_DATA.get(model_version_str, {})
            interface_ver_val = crisptts_version_cfg.get("interface_version_enum_val", "V3_STR_FALLBACK") # Default to V3 string
            actual_interface_version = OuteTTSInterfaceVersion_h.V3 # Default
            if isinstance(interface_ver_val, OuteTTSInterfaceVersion_h): actual_interface_version = interface_ver_val
            elif isinstance(interface_ver_val, str):
                try: actual_interface_version = getattr(OuteTTSInterfaceVersion_h, interface_ver_val.replace("_STR_FALLBACK", ""))
                except AttributeError: logger.warning(f"OuteTTS ({crisptts_model_id}) - Invalid interface version string '{interface_ver_val}'. Defaulting to V3.")
            
            model_config_constructor_params["interface_version"] = actual_interface_version

            if IS_MPS_HANDLER and TORCH_AVAILABLE_IN_HANDLER:
                logger.info(f"OuteTTS ({crisptts_model_id}) - Configuring HF backend specifically for MPS. Model ID: {full_hf_model_id_for_outetts}")
                interface_config_obj = OuteTTSModelConfig_h(
                    model_path=full_hf_model_id_for_outetts,
                    tokenizer_path=full_hf_model_id_for_outetts, # OuteTTS uses this for HF models
                    interface_version=actual_interface_version,
                    backend=OuteTTSBackend_h.HF,
                    device="mps",
                    dtype=torch_oute.float32
                )
            else: # Non-MPS HF (CPU or CUDA)
                logger.debug(f"OuteTTS ({crisptts_model_id}) - HF non-MPS auto_config params: {model_config_constructor_params}")
                interface_config_obj = OuteTTSModelConfig_h.auto_config(**model_config_constructor_params)
        else: # Should not be reached if initial backend check is correct
            logger.error(f"OuteTTS ({crisptts_model_id}) - Unhandled backend for ModelConfig: {actual_backend_enum}")
            return

        if not interface_config_obj:
            logger.error(f"OuteTTS ({crisptts_model_id}) - Failed to create OuteTTSModelConfig object.")
            return
        logger.info(f"OuteTTS ({crisptts_model_id}) - ModelConfig prepared: Device='{interface_config_obj.device}', Dtype='{interface_config_obj.dtype}'")

        logger.info(f"OuteTTS ({crisptts_model_id}) - Initializing OuteTTS Interface (suppressing verbose lib output)...")
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            outetts_interface = OuteTTSInterface_h(config=interface_config_obj)
        logger.info(f"OuteTTS ({crisptts_model_id}) - Interface initialized.")

        speaker_input_arg = voice_id_or_path_override
        is_likely_internal_id = isinstance(speaker_input_arg, str) and \
                                not (speaker_input_arg.lower().endswith(".wav") or Path(speaker_input_arg).is_file())

        if is_likely_internal_id:
            logger.info(f"OuteTTS ({crisptts_model_id}) - Loading default speaker profile ID: '{speaker_input_arg}'")
            try:
                speaker_profile_obj = outetts_interface.load_default_speaker(speaker_input_arg)
            except Exception as e_load_default:
                logger.error(f"OuteTTS ({crisptts_model_id}) - Failed to load default speaker ID '{speaker_input_arg}': {e_load_default}. Check available IDs.")
                return
        else:
            logger.info(f"OuteTTS ({crisptts_model_id}) - Preparing custom speaker WAV: '{speaker_input_arg}'")
            processed_speaker_wav_path, temp_speaker_file_to_delete = _prepare_oute_speaker_ref(speaker_input_arg, crisptts_model_id)
            if not processed_speaker_wav_path:
                logger.error(f"OuteTTS ({crisptts_model_id}) - Failed to prepare speaker WAV from: {speaker_input_arg}")
                if temp_speaker_file_to_delete: Path(temp_speaker_file_to_delete).unlink(missing_ok=True)
                return
            logger.info(f"OuteTTS ({crisptts_model_id}) - Creating speaker profile from: {processed_speaker_wav_path} (suppressing verbose lib output)...")
            try:
                with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
                    speaker_profile_obj = outetts_interface.create_speaker(str(processed_speaker_wav_path.resolve()))
            except Exception as e_create_speaker:
                logger.error(f"OuteTTS ({crisptts_model_id}) - Failed to create speaker profile from '{processed_speaker_wav_path}': {e_create_speaker}")
                if temp_speaker_file_to_delete: Path(temp_speaker_file_to_delete).unlink(missing_ok=True)
                return

        if not speaker_profile_obj:
            logger.error(f"OuteTTS ({crisptts_model_id}) - Speaker profile not loaded/created.")
            return

        sampler_values = {"temperature":0.4, "repetition_penalty":1.1, "top_k":40, "top_p":0.9, "min_p":0.05}
        max_gen_len = 8192
        if model_params_override:
            try:
                cli_params = json.loads(model_params_override)
                for k, v in cli_params.items():
                    if hasattr(OuteTTSSamplerConfig_h(), k): sampler_values[k] = v
                    elif k == "max_length": max_gen_len = int(v)
            except (json.JSONDecodeError, ValueError) as e_json:
                logger.warning(f"OuteTTS ({crisptts_model_id}) - Error parsing --model-params '{model_params_override}': {e_json}")

        sampler_obj = OuteTTSSamplerConfig_h(**sampler_values)
        gen_cfg_obj = OuteTTSGenerationConfig_h(text=text, generation_type=OuteTTSGenerationType_h.CHUNKED,
                                             speaker=speaker_profile_obj, sampler_config=sampler_obj, max_length=max_gen_len)
        logger.debug(f"OuteTTS ({crisptts_model_id}) - GenerationConfig: {gen_cfg_obj}, SamplerConfig: {sampler_obj}")

        logger.info(f"OuteTTS ({crisptts_model_id}) - Generating speech...")
        start_synth_time = time.time()
        output_audio_obj = outetts_interface.generate(config=gen_cfg_obj)
        logger.info(f"OuteTTS ({crisptts_model_id}) - Speech generated in {time.time() - start_synth_time:.2f}s.")

        if not output_audio_obj:
            logger.error(f"OuteTTS ({crisptts_model_id}) - Generation returned None.")
            return

        if output_file_str:
            out_path = Path(output_file_str).with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"OuteTTS ({crisptts_model_id}) - Saving audio to {out_path}...")
            output_audio_obj.save(str(out_path))
        if play_direct:
            logger.info(f"OuteTTS ({crisptts_model_id}) - Playing audio...")
            output_audio_obj.play()

    except Exception as e_main_synth:
        logger.error(f"OuteTTS ({crisptts_model_id}) - Synthesis process error: {e_main_synth}", exc_info=True)
        if actual_backend_enum == OuteTTSBackend_h.HF and platform.system() == "Darwin" and \
           any(msg in str(e_main_synth).lower() for msg in ["no such file", "can't load tokenizer", "can't load model"]):
            logger.warning(f"OuteTTS ({crisptts_model_id}) - macOS HF ONNX error hint: Clear ~/.cache/huggingface/hub and any OuteTTS cache.")
    finally:
        logger.debug(f"OuteTTS ({crisptts_model_id}) - Entering finally block for cleanup.")
        del outetts_interface, speaker_profile_obj, output_audio_obj, interface_config_obj
        gc.collect()
        if TORCH_AVAILABLE_IN_HANDLER and torch_oute.cuda.is_available(): torch_oute.cuda.empty_cache()
        if temp_speaker_file_to_delete:
            try: Path(temp_speaker_file_to_delete).unlink(missing_ok=True)
            except OSError as e_del: logger.warning(f"OuteTTS ({crisptts_model_id}) - Failed to delete temp speaker WAV {temp_speaker_file_to_delete}: {e_del}")
        logger.debug(f"OuteTTS ({crisptts_model_id}) - Cleanup finished.")