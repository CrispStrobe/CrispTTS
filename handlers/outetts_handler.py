# handlers/outetts_handler.py

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
import gc

# Relative imports
from config import (
    OUTETTS_INTERNAL_MODEL_INFO_DATA, # Fallback data
    OUTETTS_VERSION_STRING_MODEL_CONFIG_DATA
)
from utils import SuppressOutput, save_audio, play_audio, _prepare_oute_speaker_ref

logger = logging.getLogger("CrispTTS.handlers.outetts")

# Conditional imports for OuteTTS and its dependencies
OUTETTS_AVAILABLE_IN_HANDLER = False
OuteTTSInterface_h, OuteTTSModelConfig_h, OuteTTSModels_h, OuteTTSBackend_h = (None,) * 4
OuteTTSLlamaCppQuantization_h, OuteTTSGenerationConfig_h, OuteTTSGenerationType_h, OuteTTSSamplerConfig_h = (None,) * 4
OuteTTSInterfaceVersion_h = None
HFModelConfig_v1_h, HFModelConfig_v2_h, HFModelConfig_v3_h = None, None, None # For specific versioned configs

TORCH_AVAILABLE_IN_HANDLER = False
torch_h = None # type: ignore
IS_MPS_IN_HANDLER = False

try:
    import torch as torch_imp
    torch_h = torch_imp
    TORCH_AVAILABLE_IN_HANDLER = True
    IS_MPS_IN_HANDLER = hasattr(torch_h.backends, "mps") and torch_h.backends.mps.is_available()
except ImportError:
    pass # Will be checked before use

if TORCH_AVAILABLE_IN_HANDLER:
    try:
        import outetts
        from outetts import (Interface, ModelConfig, Models, Backend, LlamaCppQuantization,
                             GenerationConfig, GenerationType, SamplerConfig, InterfaceVersion)
        OuteTTSInterface_h, OuteTTSModelConfig_h, OuteTTSModels_h, OuteTTSBackend_h = Interface, ModelConfig, Models, Backend
        OuteTTSLlamaCppQuantization_h = LlamaCppQuantization
        OuteTTSGenerationConfig_h, OuteTTSGenerationType_h, OuteTTSSamplerConfig_h = GenerationConfig, GenerationType, SamplerConfig
        OuteTTSInterfaceVersion_h = InterfaceVersion
        OUTETTS_AVAILABLE_IN_HANDLER = True
        # Try to import versioned configs if OuteTTS exposes them this way
        try: from outetts.version.v1.interface import HFModelConfig as HFModelConfig_v1_imp; HFModelConfig_v1_h = HFModelConfig_v1_imp
        except ImportError: pass
        try: from outetts.version.v2.interface import HFModelConfig as HFModelConfig_v2_imp; HFModelConfig_v2_h = HFModelConfig_v2_imp
        except ImportError: pass
        try: from outetts.version.v3.interface import HFModelConfig as HFModelConfig_v3_imp; HFModelConfig_v3_h = HFModelConfig_v3_imp
        except ImportError: pass
    except ImportError:
        logger.info("'outetts' library not installed. OuteTTS handler will not be functional.")
    except AttributeError:
        logger.warning("OuteTTS import has AttributeError (likely protobuf). OuteTTS handler unavailable.")
    except Exception as e:
        logger.warning(f"OuteTTS general import error: {e}. OuteTTS handler unavailable.")

try:
    from huggingface_hub import hf_hub_download, HfFileSystem
    HF_HUB_AVAILABLE_FOR_OUTETTS = True
    hf_fs_outetts = HfFileSystem()
except ImportError:
    HF_HUB_AVAILABLE_FOR_OUTETTS = False
    hf_hub_download = None # type: ignore
    hf_fs_outetts = None # type: ignore
    if OUTETTS_AVAILABLE_IN_HANDLER: # Only warn if OuteTTS itself is available but hub isn't
        logger.warning("huggingface_hub not installed. OuteTTS model downloading (especially for ONNX) might fail.")


def synthesize_with_outetts_local(model_config, text, voice_id_or_path, model_params_override, output_file_str, play_direct):
    if not OUTETTS_AVAILABLE_IN_HANDLER:
        logger.error("OuteTTS library not available for this handler. Skipping OuteTTS synthesis.")
        return

    backend_choice_from_config = model_config.get("backend_to_use")
    # Map string fallbacks from config (if OuteTTS wasn't available during config.py load) to actual enums
    backend_choice = None
    if isinstance(backend_choice_from_config, str):
        if backend_choice_from_config == "LLAMACPP_STR_FALLBACK" and OuteTTSBackend_h: backend_choice = OuteTTSBackend_h.LLAMACPP
        elif backend_choice_from_config == "HF_STR_FALLBACK" and OuteTTSBackend_h: backend_choice = OuteTTSBackend_h.HF
        else: logger.error(f"OuteTTS: Unknown backend string '{backend_choice_from_config}'."); return
    else: # Assumed to be enum if not string
        backend_choice = backend_choice_from_config


    outetts_model_enum_cfg = model_config.get("outetts_model_enum")
    outetts_model_enum_for_cpp = None
    if isinstance(outetts_model_enum_cfg, str) and OuteTTSModels_h:
        try: outetts_model_enum_for_cpp = getattr(OuteTTSModels_h, outetts_model_enum_cfg.replace("_STR_FALLBACK",""))
        except AttributeError: logger.error(f"OuteTTS: Bad model enum string '{outetts_model_enum_cfg}'."); return
    else:
        outetts_model_enum_for_cpp = outetts_model_enum_cfg

    outetts_model_version_str = model_config.get("outetts_model_version_str")

    if not backend_choice: logger.error("OuteTTS: backend_to_use missing."); return
    if backend_choice == OuteTTSBackend_h.LLAMACPP and not outetts_model_enum_for_cpp:
        logger.error("OuteTTS LlamaCPP: outetts_model_enum missing."); return
    if backend_choice == OuteTTSBackend_h.HF and not outetts_model_version_str:
        logger.error(f"OuteTTS HF: 'outetts_model_version_str' missing."); return

    logger.debug(f"OuteTTS Local - Text:'{text[:50]}...', Backend:{backend_choice}, Voice/Ref:{voice_id_or_path}")

    interface = None; speaker = None; cfg_obj = None
    temp_onnx_model_dir_obj = None; temp_trimmed_file_to_delete = None

    try:
        logger.info("OuteTTS - Preparing ModelConfig...")
        device_to_use = "cuda" if TORCH_AVAILABLE_IN_HANDLERS and torch_h.cuda.is_available() else \
                        ("mps" if IS_MPS_IN_HANDLER else "cpu")

        _model_max_seq_length = 8192 # Default
        # Try to get from version string mapping first, then enum mapping
        if outetts_model_version_str and outetts_model_version_str in OUTETTS_VERSION_STRING_MODEL_CONFIG_DATA:
            _model_max_seq_length = OUTETTS_VERSION_STRING_MODEL_CONFIG_DATA[outetts_model_version_str].get("max_seq_length", _model_max_seq_length)
        elif outetts_model_enum_for_cpp and outetts_model_enum_for_cpp in OUTETTS_INTERNAL_MODEL_INFO_DATA:
             _model_max_seq_length = OUTETTS_INTERNAL_MODEL_INFO_DATA[outetts_model_enum_for_cpp].get("max_seq_length", _model_max_seq_length)
        logger.debug(f"OuteTTS - Model family max_seq_length for GenerationConfig: {_model_max_seq_length}")

        if backend_choice == OuteTTSBackend_h.LLAMACPP:
            quant_cfg_val = model_config.get("quantization_to_use", OuteTTSLlamaCppQuantization_h.FP16 if OuteTTSLlamaCppQuantization_h else "FP16_STR_FALLBACK")
            quantization = None
            if isinstance(quant_cfg_val, str) and OuteTTSLlamaCppQuantization_h:
                try: quantization = getattr(OuteTTSLlamaCppQuantization_h, quant_cfg_val.replace("_STR_FALLBACK",""))
                except AttributeError: logger.error(f"OuteTTS LlamaCPP: Bad quant string '{quant_cfg_val}'."); return
            else: quantization = quant_cfg_val
            if not quantization: logger.error("OuteTTS LlamaCPP: Quantization could not be determined."); return

            logger.info(f"OuteTTS LlamaCPP - auto_config. Model:{outetts_model_enum_for_cpp}, Quant:{quantization}")
            with SuppressOutput(suppress_stdout=True, suppress_stderr=True):
                cfg_obj = OuteTTSModelConfig_h.auto_config(model=outetts_model_enum_for_cpp, backend=OuteTTSBackend_h.LLAMACPP, quantization=quantization)
            if not cfg_obj: logger.error("OuteTTS LlamaCPP - auto_config failed."); return
            
            logger.info("OuteTTS - Initializing LlamaCPP Interface...")
            with SuppressOutput(suppress_stdout=True, suppress_stderr=True):
                interface = OuteTTSInterface_h(config=cfg_obj)

        elif backend_choice == OuteTTSBackend_h.HF:
            if not HF_HUB_AVAILABLE_IN_HANDLERS or not hf_hub_download:
                logger.error("OuteTTS HF: huggingface_hub not available for model download. Skipping.")
                return

            onnx_repo_id = model_config.get("onnx_repo_id")
            onnx_filename_options = model_config.get("onnx_filename_options", [])
            onnx_subfolder = model_config.get("onnx_subfolder", "onnx")
            tokenizer_path = model_config.get("tokenizer_path")
            language_code = model_config.get("language")
            torch_dtype_cfg = model_config.get("torch_dtype_for_hf_wrapper")
            torch_dtype_for_wrapper = torch_dtype_cfg if torch_dtype_cfg is not None else (torch_h.float32 if TORCH_AVAILABLE_IN_HANDLERS else None)
            
            interface_version_enum_cfg = model_config.get("interface_version_enum")
            interface_version_enum = None
            if isinstance(interface_version_enum_cfg, str) and OuteTTSInterfaceVersion_h:
                try: interface_version_enum = getattr(OuteTTSInterfaceVersion_h, interface_version_enum_cfg.replace("_STR_FALLBACK",""))
                except AttributeError: logger.error(f"OuteTTS HF: Bad interface version string '{interface_version_enum_cfg}'."); return
            else: interface_version_enum = interface_version_enum_cfg

            if not all([outetts_model_version_str, onnx_repo_id, onnx_filename_options, tokenizer_path, language_code, interface_version_enum]):
                logger.error("OuteTTS HF ONNX: Incomplete config."); return

            successful_model_load = False
            temp_onnx_model_dir_obj = tempfile.TemporaryDirectory(prefix="outetts_onnx_")
            local_onnx_dir_path_base = Path(temp_onnx_model_dir_obj.name)

            for onnx_filename_in_repo_subpath in onnx_filename_options:
                current_model_attempt_dir = local_onnx_dir_path_base / Path(onnx_filename_in_repo_subpath).stem
                current_model_attempt_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"OuteTTS HF - Attempting ONNX: {onnx_repo_id}/{onnx_filename_in_repo_subpath}")
                try:
                    hf_hub_download(repo_id=onnx_repo_id, filename=onnx_filename_in_repo_subpath,
                                    local_dir=str(current_model_attempt_dir), local_dir_use_symlinks=False, token=os.getenv("HF_TOKEN"))
                    onnx_data_file_in_repo = onnx_filename_in_repo_subpath + "_data"
                    if hf_fs_handler and hf_fs_handler.exists(f"{onnx_repo_id}/{onnx_data_file_in_repo}"): # Check with hf_fs from this module
                        hf_hub_download(repo_id=onnx_repo_id, filename=onnx_data_file_in_repo,
                                        local_dir=str(current_model_attempt_dir), local_dir_use_symlinks=False, token=os.getenv("HF_TOKEN"))
                        logger.info(f"OuteTTS HF - Downloaded {Path(onnx_data_file_in_repo).name}.")
                    
                    ConfigClass = OuteTTSModelConfig_h # Fallback
                    if outetts_model_version_str == "1.0": ConfigClass = HFModelConfig_v3_h or OuteTTSModelConfig_h
                    elif outetts_model_version_str == "0.3": ConfigClass = HFModelConfig_v2_h or OuteTTSModelConfig_h
                    elif outetts_model_version_str in ["0.1", "0.2"]: ConfigClass = HFModelConfig_v1_h or OuteTTSModelConfig_h
                    
                    cfg_obj = ConfigClass(
                        model_path=str(current_model_attempt_dir), tokenizer_path=tokenizer_path, language=language_code,
                        dtype=torch_dtype_for_wrapper, # For PyTorch wrapper elements
                        interface_version=interface_version_enum, backend=OuteTTSBackend_h.HF,
                        device=device_to_use, max_seq_length=_model_max_seq_length
                    )
                    logger.info("OuteTTS - Initializing HF Interface with local ONNX...")
                    with SuppressOutput(suppress_stdout=True, suppress_stderr=True):
                        if hasattr(outetts, "InterfaceHF"):
                            interface = outetts.InterfaceHF(model_version=outetts_model_version_str, cfg=cfg_obj)
                        else:
                            interface = OuteTTSInterface_h(config=cfg_obj)
                    logger.info(f"OuteTTS HF - Successfully initialized with: {onnx_filename_in_repo_subpath}")
                    successful_model_load = True; break
                except Exception as e_load:
                    logger.warning(f"OuteTTS HF - Failed for ONNX '{onnx_filename_in_repo_subpath}': {e_load}", exc_info=True)
                    if interface: del interface; interface = None
                    if cfg_obj: del cfg_obj; cfg_obj = None; gc.collect()
            
            if not successful_model_load:
                logger.error(f"OuteTTS HF - All ONNX model options failed for {onnx_repo_id}.")
                if temp_onnx_model_dir_obj: temp_onnx_model_dir_obj.cleanup(); temp_onnx_model_dir_obj = None
                return
        else: logger.error(f"OuteTTS - Unexpected backend: {backend_choice}"); return

        if not interface: logger.error("OuteTTS Interface init failed."); return

        # Speaker Loading (using logger)
        is_default_speaker_id = isinstance(voice_id_or_path, str) and not (Path(voice_id_or_path).is_file() and voice_id_or_path.lower().endswith((".wav", ".json")))
        if is_default_speaker_id:
            logger.info(f"OuteTTS - Loading default speaker: {voice_id_or_path}")
            try: speaker = interface.load_default_speaker(voice_id_or_path)
            except Exception as e: logger.error(f"OuteTTS - Failed to load default speaker '{voice_id_or_path}': {e}"); return
        else:
            speaker_path = Path(voice_id_or_path)
            if not speaker_path.exists(): logger.error(f"OuteTTS - Custom speaker file DNE: {voice_id_or_path}"); return
            if speaker_path.suffix.lower() == ".json":
                logger.info(f"OuteTTS - Loading speaker from JSON: {speaker_path}"); speaker = interface.load_speaker(str(speaker_path))
            elif speaker_path.suffix.lower() == ".wav":
                processed_speaker_ref_path_obj, temp_trimmed_file_to_delete = _prepare_oute_speaker_ref(voice_id_or_path, f"OuteTTS ({backend_choice})")
                if not processed_speaker_ref_path_obj: return
                logger.info(f"OuteTTS - Creating speaker from WAV: {processed_speaker_ref_path_obj}")
                with SuppressOutput(suppress_stdout=True, suppress_stderr=True): speaker = interface.create_speaker(str(processed_speaker_ref_path_obj))
            else: logger.error(f"OuteTTS - Invalid speaker file: {voice_id_or_path}"); return
        if not speaker: logger.error("OuteTTS - Speaker profile not loaded/created."); return
        logger.info(f"OuteTTS - Speaker ready: {voice_id_or_path}")

        # Generation
        sampler_config_dict = {"temperature": 0.4, "repetition_penalty": 1.1, "top_k": 40, "top_p": 0.9, "min_p": 0.05}
        gen_max_len = _model_max_seq_length
        if hasattr(interface, 'config') and hasattr(interface.config, 'max_seq_length') and interface.config.max_seq_length:
             gen_max_len = interface.config.max_seq_length
        if model_params_override:
            try:
                cli_params = json.loads(model_params_override)
                for k, v_cli in cli_params.items():
                    if hasattr(OuteTTSSamplerConfig_h, k): sampler_config_dict[k] = v_cli
                    elif k == "max_length": gen_max_len = min(v_cli, gen_max_len)
            except json.JSONDecodeError: logger.warning(f"OuteTTS - Could not parse --model-params: {model_params_override}")
        
        sampler_config = OuteTTSSamplerConfig_h(**sampler_config_dict) if OuteTTSSamplerConfig_h else sampler_config_dict
        generation_config_obj = OuteTTSGenerationConfig_h(text=text, generation_type=OuteTTSGenerationType_h.CHUNKED, speaker=speaker, sampler_config=sampler_config, max_length=gen_max_len) if OuteTTSGenerationConfig_h and OuteTTSGenerationType_h else {}

        logger.info(f"OuteTTS - Generating speech (max_length: {gen_max_len})...")
        start_time = time.time()
        with SuppressOutput(suppress_stdout=True, suppress_stderr=True): output_audio = interface.generate(config=generation_config_obj)
        end_time = time.time(); generation_duration = end_time - start_time
        
        if output_audio and hasattr(output_audio, 'audio_tensor') and output_audio.audio_tensor is not None and output_audio.audio_tensor.numel() > 0:
            logger.info(f"OuteTTS - Speech generated in {generation_duration:.2f}s.")
            effective_output_file_wav = Path(output_file_str).with_suffix(".wav") if output_file_str else None
            if effective_output_file_wav: save_audio(output_audio.audio_bytes, str(effective_output_file_wav), source_is_path=False, input_format="wav", sample_rate=output_audio.sample_rate)
            if play_direct: output_audio.play()
        else:
            logger.warning("OuteTTS - Generation produced no audio data.")
            if output_file_str and Path(output_file_str).exists(): Path(output_file_str).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"OuteTTS - Error for backend {backend_choice}: {e}", exc_info=True)
    finally:
        if temp_trimmed_file_to_delete and Path(temp_trimmed_file_to_delete).exists():
            try: os.remove(temp_trimmed_file_to_delete); logger.debug("OuteTTS - Deleted temp speaker audio.")
            except Exception as e: logger.warning(f"OuteTTS - Failed to delete temp speaker audio: {e}")
        if temp_onnx_model_dir_obj:
            try: temp_onnx_model_dir_obj.cleanup(); logger.debug("OuteTTS HF - Cleaned up temp ONNX dir.")
            except Exception as e: logger.warning(f"OuteTTS HF - Failed to cleanup temp ONNX dir: {e}")
        if interface: del interface
        if speaker: del speaker
        if cfg_obj: del cfg_obj
        gc.collect()
        if TORCH_AVAILABLE_IN_HANDLERS and torch_h and backend_choice == OuteTTSBackend_h.HF and torch_h.cuda.is_available():
            torch_h.cuda.empty_cache(); logger.debug("OuteTTS HF - Cleared CUDA cache.")
        logger.debug("OuteTTS - Resources cleanup attempted.")