# handlers/outetts_handler.py

import logging
import json
from pathlib import Path
import os
import gc
import platform
import time

# --- Conditional Imports ---
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
    logger_init.info("PyTorch not found. OuteTTS functionality might be limited.")

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

from utils import _prepare_oute_speaker_ref, SuppressOutput
from config import OUTETTS_VERSION_STRING_MODEL_CONFIG_DATA

logger = logging.getLogger("CrispTTS.handlers.outetts")

def synthesize_with_outetts_local(model_config, text, voice_id_or_path_override, model_params_override, output_file_str, play_direct):
    crisptts_model_id = model_config.get('crisptts_model_id', 'outetts_unknown_id')

    if not OUTETTS_AVAILABLE or not all([
        OuteTTSInterface_h, OuteTTSModelConfig_h, OuteTTSModels_h, OuteTTSBackend_h,
        OuteTTSGenerationConfig_h, OuteTTSGenerationType_h, OuteTTSSamplerConfig_h, OuteTTSInterfaceVersion_h
    ]):
        logger.error(f"OuteTTS ({crisptts_model_id}) - Library or essential components missing. Skipping.")
        return

    logger.debug(f"OuteTTS ({crisptts_model_id}) - Text: '{text[:50]}...' | Voice/Ref: {voice_id_or_path_override}")

    backend_choice_from_config = model_config.get("backend_to_use")
    actual_backend_enum = None
    if isinstance(backend_choice_from_config, OuteTTSBackend_h): actual_backend_enum = backend_choice_from_config
    elif isinstance(backend_choice_from_config, str):
        if "LLAMACPP" in backend_choice_from_config.upper(): actual_backend_enum = OuteTTSBackend_h.LLAMACPP
        elif "HF" in backend_choice_from_config.upper(): actual_backend_enum = OuteTTSBackend_h.HF
        else: logger.error(f"OuteTTS ({crisptts_model_id}) - Unknown backend string: '{backend_choice_from_config}'."); return
    else: logger.error(f"OuteTTS ({crisptts_model_id}) - Invalid 'backend_to_use' type: {type(backend_choice_from_config)}."); return

    outetts_model_enum_from_config = model_config.get("outetts_model_enum")
    actual_model_enum = None
    if OuteTTSModels_h:
        if isinstance(outetts_model_enum_from_config, OuteTTSModels_h): actual_model_enum = outetts_model_enum_from_config
        elif isinstance(outetts_model_enum_from_config, str):
            try: actual_model_enum = getattr(OuteTTSModels_h, outetts_model_enum_from_config.replace("_STR_FALLBACK", ""))
            except AttributeError: logger.warning(f"OuteTTS ({crisptts_model_id}) - Could not map model string '{outetts_model_enum_from_config}' to enum.")
        elif outetts_model_enum_from_config is None: logger.info(f"OuteTTS ({crisptts_model_id}) - 'outetts_model_enum' is None in config (expected for HF).")
        else: logger.error(f"OuteTTS ({crisptts_model_id}) - Unexpected 'outetts_model_enum' type: {type(outetts_model_enum_from_config)}."); return
    else: logger.warning(f"OuteTTS ({crisptts_model_id}) - OuteTTS.Models enum class not available.")

    logger.info(f"OuteTTS ({crisptts_model_id}) - Resolved Backend: {actual_backend_enum}, Derived Model Enum: {actual_model_enum}")

    interface_config_obj, outetts_interface = None, None
    temp_speaker_file_to_delete, speaker_profile_obj, output_audio_obj = None, None, None

    try:
        logger.info(f"OuteTTS ({crisptts_model_id}) - Preparing ModelConfig...")
        model_config_constructor_params = {"backend": actual_backend_enum}

        if actual_backend_enum == OuteTTSBackend_h.LLAMACPP:
            if not actual_model_enum: logger.error(f"OuteTTS ({crisptts_model_id}) - LlamaCPP: 'outetts_model_enum' is required."); return
            model_config_constructor_params["model"] = actual_model_enum
            if not OuteTTSLlamaCppQuantization_h: logger.error(f"OuteTTS ({crisptts_model_id}) - LlamaCppQuantization enum NA."); return
            quant_cfg = model_config.get("quantization_to_use")
            if isinstance(quant_cfg, OuteTTSLlamaCppQuantization_h): model_config_constructor_params["quantization"] = quant_cfg
            elif isinstance(quant_cfg, str): model_config_constructor_params["quantization"] = getattr(OuteTTSLlamaCppQuantization_h, quant_cfg.replace("_STR_FALLBACK", ""))
            else: model_config_constructor_params["quantization"] = OuteTTSLlamaCppQuantization_h.FP16
            logger.debug(f"OuteTTS ({crisptts_model_id}) - LlamaCPP auto_config params: {model_config_constructor_params}")
            interface_config_obj = OuteTTSModelConfig_h.auto_config(**model_config_constructor_params)

        elif actual_backend_enum == OuteTTSBackend_h.HF:
            full_hf_model_id = None
            if actual_model_enum and hasattr(actual_model_enum, 'value'):
                full_hf_model_id = f"OuteAI/{actual_model_enum.value}"
                logger.info(f"OuteTTS ({crisptts_model_id}) - HF: Model ID from enum: {full_hf_model_id}")
            else:
                tokenizer_path = model_config.get("tokenizer_path")
                if tokenizer_path and isinstance(tokenizer_path, str) and "/" in tokenizer_path:
                    full_hf_model_id = tokenizer_path
                    logger.info(f"OuteTTS ({crisptts_model_id}) - HF: Model ID from tokenizer_path: {full_hf_model_id}")
                else: logger.error(f"OuteTTS ({crisptts_model_id}) - HF: Cannot determine HF model ID."); return
            model_config_constructor_params["model"] = full_hf_model_id

            model_ver_str = model_config.get("outetts_model_version_str", "1.0")
            crisptts_ver_cfg = OUTETTS_VERSION_STRING_MODEL_CONFIG_DATA.get(model_ver_str, {})
            iface_ver_val = model_config.get("interface_version_enum", crisptts_ver_cfg.get("interface_version_enum_val", "V3_STR_FALLBACK"))
            actual_iface_ver = OuteTTSInterfaceVersion_h.V3
            if isinstance(iface_ver_val, OuteTTSInterfaceVersion_h): actual_iface_ver = iface_ver_val
            elif isinstance(iface_ver_val, str):
                try: actual_iface_ver = getattr(OuteTTSInterfaceVersion_h, iface_ver_val.replace("_STR_FALLBACK", ""))
                except AttributeError: logger.warning(f"OuteTTS ({crisptts_model_id}) - Invalid HF interface version string '{iface_ver_val}'. Defaulting to V3.")
            model_config_constructor_params["interface_version"] = actual_iface_ver

            if IS_MPS_HANDLER and TORCH_AVAILABLE_IN_HANDLER:
                logger.info(f"OuteTTS ({crisptts_model_id}) - Configuring HF backend for MPS. Model ID: {full_hf_model_id}")
                # Prioritize float16 for MPS to reduce memory, respecting config if it specifically asks for float32
                dtype_from_config_str = model_config.get("torch_dtype_for_hf_wrapper", "torch.float16_STR_FALLBACK") # Default to float16 string
                mps_dtype = torch_oute.float16 # Default for MPS
                if isinstance(dtype_from_config_str, torch_oute.dtype): # if config already provided torch.dtype object
                    mps_dtype = dtype_from_config_str
                elif isinstance(dtype_from_config_str, str):
                    if "float32" in dtype_from_config_str.lower(): # If config string explicitly asks for float32
                        mps_dtype = torch_oute.float32
                    # else it remains float16 (from default or if "float16" in string)
                logger.info(f"OuteTTS ({crisptts_model_id}) - MPS using dtype: {mps_dtype}")
                interface_config_obj = OuteTTSModelConfig_h(
                    model_path=full_hf_model_id, tokenizer_path=full_hf_model_id,
                    interface_version=actual_iface_ver, backend=OuteTTSBackend_h.HF,
                    device="mps", dtype=mps_dtype
                )
            else: # Non-MPS HF
                logger.debug(f"OuteTTS ({crisptts_model_id}) - HF non-MPS auto_config using params: {model_config_constructor_params}")
                interface_config_obj = OuteTTSModelConfig_h.auto_config(**model_config_constructor_params)
        else: logger.error(f"OuteTTS ({crisptts_model_id}) - Unhandled backend: {actual_backend_enum}"); return

        if not interface_config_obj: logger.error(f"OuteTTS ({crisptts_model_id}) - Failed to create ModelConfig."); return
        logger.info(f"OuteTTS ({crisptts_model_id}) - ModelConfig: Device='{interface_config_obj.device}', Dtype='{interface_config_obj.dtype}'")

        logger.info(f"OuteTTS ({crisptts_model_id}) - Initializing Interface (suppressing verbose lib output)...")
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            outetts_interface = OuteTTSInterface_h(config=interface_config_obj)
        logger.info(f"OuteTTS ({crisptts_model_id}) - Interface initialized.")

        speaker_input_arg = voice_id_or_path_override
        is_internal_id = isinstance(speaker_input_arg, str) and \
                         not (speaker_input_arg.lower().endswith(".wav") or Path(speaker_input_arg).is_file())

        if is_internal_id:
            logger.info(f"OuteTTS ({crisptts_model_id}) - Loading default speaker ID: '{speaker_input_arg}'")
            try: speaker_profile_obj = outetts_interface.load_default_speaker(speaker_input_arg)
            except Exception as e: logger.error(f"OuteTTS ({crisptts_model_id}) - Failed to load speaker ID '{speaker_input_arg}': {e}"); return
        else:
            logger.info(f"OuteTTS ({crisptts_model_id}) - Preparing custom WAV: '{speaker_input_arg}'")
            processed_wav, temp_speaker_file_to_delete = _prepare_oute_speaker_ref(speaker_input_arg, crisptts_model_id)
            if not processed_wav:
                logger.error(f"OuteTTS ({crisptts_model_id}) - Failed to prepare WAV: {speaker_input_arg}")
                if temp_speaker_file_to_delete: Path(temp_speaker_file_to_delete).unlink(missing_ok=True)
                return
            logger.info(f"OuteTTS ({crisptts_model_id}) - Creating speaker profile from: {processed_wav} (suppressing verbose lib output)...")
            try:
                with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
                    speaker_profile_obj = outetts_interface.create_speaker(str(processed_wav.resolve()))
            except Exception as e:
                logger.error(f"OuteTTS ({crisptts_model_id}) - Failed to create profile from '{processed_wav}': {e}")
                if temp_speaker_file_to_delete: Path(temp_speaker_file_to_delete).unlink(missing_ok=True)
                return

        if not speaker_profile_obj: logger.error(f"OuteTTS ({crisptts_model_id}) - Speaker profile not loaded/created."); return

        sampler_vals = {"temperature":0.4, "repetition_penalty":1.1, "top_k":40, "top_p":0.9, "min_p":0.05}
        max_len = interface_config_obj.max_sequence_length if hasattr(interface_config_obj, 'max_sequence_length') else 8192
        if model_params_override:
            try:
                cli_params = json.loads(model_params_override)
                for k, v_param in cli_params.items():
                    if hasattr(OuteTTSSamplerConfig_h(), k): sampler_vals[k] = v_param
                    elif k == "max_length": max_len = int(v_param)
            except (json.JSONDecodeError, ValueError) as e: logger.warning(f"OuteTTS ({crisptts_model_id}) - Error parsing --model-params: {e}")

        sampler_obj = OuteTTSSamplerConfig_h(**sampler_vals)
        gen_cfg_obj = OuteTTSGenerationConfig_h(text=text, generation_type=OuteTTSGenerationType_h.CHUNKED,
                                             speaker=speaker_profile_obj, sampler_config=sampler_obj, max_length=max_len)
        logger.debug(f"OuteTTS ({crisptts_model_id}) - GenConfig: {gen_cfg_obj}, SamplerConfig: {sampler_obj}")

        logger.info(f"OuteTTS ({crisptts_model_id}) - Generating speech...")
        start_time = time.time()
        output_audio_obj = outetts_interface.generate(config=gen_cfg_obj)
        logger.info(f"OuteTTS ({crisptts_model_id}) - Speech generated in {time.time() - start_time:.2f}s.")

        if not output_audio_obj: logger.error(f"OuteTTS ({crisptts_model_id}) - Generation returned None."); return

        if output_file_str:
            out_p = Path(output_file_str).with_suffix(".wav")
            out_p.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"OuteTTS ({crisptts_model_id}) - Saving audio to {out_p}...")
            output_audio_obj.save(str(out_p))
        if play_direct:
            logger.info(f"OuteTTS ({crisptts_model_id}) - Playing audio...")
            output_audio_obj.play()

    except Exception as e_synth:
        logger.error(f"OuteTTS ({crisptts_model_id}) - Synthesis error: {e_synth}", exc_info=True)
        if actual_backend_enum == OuteTTSBackend_h.HF and platform.system() == "Darwin" and \
           any(m in str(e_synth).lower() for m in ["no such file", "can't load tokenizer", "can't load model", "mpsgraph"]):
            logger.warning(f"OuteTTS ({crisptts_model_id}) - macOS HF/MPS error hint: Clear mps_graph_compiler cache and/or Hugging Face cache.")
    finally:
        logger.debug(f"OuteTTS ({crisptts_model_id}) - Cleanup...")
        del output_audio_obj, speaker_profile_obj, outetts_interface, interface_config_obj
        gc.collect()
        if TORCH_AVAILABLE_IN_HANDLER:
            if IS_MPS_HANDLER and hasattr(torch_oute, "mps") and hasattr(torch_oute.mps, "empty_cache"):
                try: torch_oute.mps.empty_cache(); logger.debug(f"OuteTTS ({crisptts_model_id}) - MPS cache cleared.")
                except Exception as e: logger.warning(f"OuteTTS ({crisptts_model_id}) - Failed to clear MPS cache: {e}")
            elif torch_oute.cuda.is_available() and hasattr(torch_oute.cuda, "empty_cache"):
                torch_oute.cuda.empty_cache(); logger.debug(f"OuteTTS ({crisptts_model_id}) - CUDA cache cleared.")
        if temp_speaker_file_to_delete:
            try: Path(temp_speaker_file_to_delete).unlink(missing_ok=True)
            except OSError as e: logger.warning(f"OuteTTS ({crisptts_model_id}) - Failed to delete temp WAV {temp_speaker_file_to_delete}: {e}")
        logger.debug(f"OuteTTS ({crisptts_model_id}) - Cleanup finished.")