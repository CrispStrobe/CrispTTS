#!/usr/bin/env python3
# CrispTTS - main.py
# Main Command-Line Interface for the Text-to-Speech Synthesizer (Modularized with Overrides)

import sys 
from pathlib import Path 
import time 
import argparse 
import os 
import json 
import logging 
import types 
import importlib 

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ["GGML_METAL_NDEBUG"] = "1" 

_main_mp_logger = logging.getLogger("CrispTTS.main_monkey_patch")
if not _main_mp_logger.handlers:
    _mp_handler = logging.StreamHandler(sys.stderr)
    _mp_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - MONKEY_PATCH: %(message)s')
    _mp_handler.setFormatter(_mp_formatter)
    _main_mp_logger.addHandler(_mp_handler)
    _main_mp_logger.setLevel(logging.INFO) 
    _main_mp_logger.propagate = False

class _CrispTTSDummyTritonConfig:
    def __init__(self, *args, **kwargs):
        _main_mp_logger.debug(f"DummyTritonConfig initialized with args: {args}, kwargs: {kwargs}") # Changed to DEBUG
        pass

def _apply_triton_config_monkey_patch_for_vllm():
    patch_applied_summary = [] 
    try:
        if 'triton' in sys.modules:
            triton_module = sys.modules['triton']
            placeholder_type_name = str(type(triton_module))
            
            is_likely_vllm_placeholder = (
                "vllm" in placeholder_type_name and 
                "TritonPlaceholder" in placeholder_type_name and
                hasattr(triton_module, '_dummy_decorator') 
            )

            if is_likely_vllm_placeholder:
                patch_applied_summary.append("vLLM's TritonPlaceholder detected")

                if not hasattr(triton_module, 'Config'):
                    _main_mp_logger.debug("Adding 'Config' attribute to vLLM's TritonPlaceholder.") # DEBUG
                    setattr(triton_module, 'Config', _CrispTTSDummyTritonConfig)
                
                if not hasattr(triton_module, 'cdiv'):
                    _main_mp_logger.debug("Adding 'cdiv' attribute to vLLM's TritonPlaceholder.") # DEBUG
                    setattr(triton_module, 'cdiv', lambda x, y: (x + y - 1) // y)

                if hasattr(triton_module, 'language'):
                    triton_lang_module = triton_module.language
                    lang_placeholder_type_name = str(type(triton_lang_module))
                    if "vllm" in lang_placeholder_type_name and "TritonLanguagePlaceholder" in lang_placeholder_type_name:
                        _main_mp_logger.debug("vLLM's TritonLanguagePlaceholder found. Patching missing attributes.") # DEBUG
                        
                        class _DummyDtypePlaceholder:
                            def __init__(self, name_): self.name = name_
                            def __repr__(self): return f"tl.{self.name}"
                            def to(self, target_device_type_str):
                                _main_mp_logger.debug(f"DummyDtypePlaceholder {self.name}.to({target_device_type_str}) called.")
                                return self

                        dtypes_to_add = {name: _DummyDtypePlaceholder(name) for name in 
                                         ["int1", "int8", "int16", "int32", "uint8", "uint16", 
                                          "uint32", "uint64", "float8e4nv", "float8e5", 
                                          "float16", "bfloat16", "float32", "float64"]}
                        
                        for dtype_name, dtype_obj in dtypes_to_add.items():
                            if not hasattr(triton_lang_module, dtype_name):
                                _main_mp_logger.debug(f"Adding dtype '{dtype_name}' to TritonLanguagePlaceholder.") # DEBUG
                                setattr(triton_lang_module, dtype_name, dtype_obj)
                        
                        if getattr(triton_lang_module, 'constexpr', 'NOT_SET') is None:
                            _main_mp_logger.debug("Patching 'constexpr' in TritonLanguagePlaceholder to be an identity function.") # DEBUG
                            setattr(triton_lang_module, 'constexpr', lambda x: x)
                        
                        current_dtype_attr = getattr(triton_lang_module, 'dtype', 'NOT_SET')
                        if current_dtype_attr is None or not callable(current_dtype_attr):
                             _main_mp_logger.debug("Patching 'dtype' in TritonLanguagePlaceholder to be a dummy factory.") # DEBUG
                             setattr(triton_lang_module, 'dtype', 
                                     lambda name_str_or_obj: dtypes_to_add.get(str(name_str_or_obj), _DummyDtypePlaceholder(str(name_str_or_obj))) 
                                     if isinstance(name_str_or_obj, str) else name_str_or_obj)
                        
                        dummy_attrs = {
                            'PROGRAM_ID': lambda axis: 0, 'make_block_ptr': lambda *a, **kw: None,
                            'load': lambda *a, **kw: None, 'store': lambda *a, **kw: None,
                            'dot': lambda *a, **kw: None
                        }
                        for attr_name, attr_val in dummy_attrs.items():
                            if not hasattr(triton_lang_module, attr_name):
                                _main_mp_logger.debug(f"Adding dummy '{attr_name}' to TritonLanguagePlaceholder.") # DEBUG
                                setattr(triton_lang_module, attr_name, attr_val)
                    else:
                        _main_mp_logger.debug(f"sys.modules['triton'].language (type: {lang_placeholder_type_name}) is not vLLM's placeholder.")
                else:
                     _main_mp_logger.debug("TritonPlaceholder does not have 'language' attribute.") # DEBUG
            
        if patch_applied_summary:
            _main_mp_logger.info(f"Triton placeholder patch: {patch_applied_summary[0]} (see DEBUG for details if enabled)")
        else:
            _main_mp_logger.debug("No Triton placeholder patch applied.")

    except Exception as e_mp:
        print(f"CRITICAL MONKEY PATCH ERROR: {e_mp}", file=sys.stderr)
        _main_mp_logger.error(f"Error during Triton monkey patching: {e_mp}", exc_info=True)

ALL_HANDLERS = {} 
_HANDLERS_LOADED = False

from config import (
    GERMAN_TTS_MODELS, LM_STUDIO_API_URL_DEFAULT, OLLAMA_API_URL_DEFAULT    
)
from utils import (
    get_text_from_input, list_available_models, get_voice_info,
    PYDUB_AVAILABLE as UTILS_PYDUB_AVAILABLE, 
    SOUNDFILE_AVAILABLE as UTILS_SOUNDFILE_AVAILABLE
)

_apply_triton_config_monkey_patch_for_vllm()

logger = logging.getLogger("CrispTTS.main")

def _load_handlers_if_needed():
    global ALL_HANDLERS, _HANDLERS_LOADED
    if not _HANDLERS_LOADED:
        logger.debug("Attempting to dynamically load TTS handlers...")
        try:
            from handlers import ALL_HANDLERS as loaded_handlers_dict # This imports handlers/__init__.py
            ALL_HANDLERS.update(loaded_handlers_dict) 
            _HANDLERS_LOADED = True
            # Log at INFO that handlers are loaded, details of each handler import can be DEBUG in handlers/__init__.py
            logger.info(f"TTS handlers dynamically loaded. Available keys: {list(ALL_HANDLERS.keys())}")
        except ImportError as e:
            logger.critical(f"CRITICAL ERROR: Failed to import from 'handlers' package: {e}", exc_info=True)
        except KeyError as e_key: 
            logger.critical(f"CRITICAL ERROR: 'ALL_HANDLERS' map not found or incomplete in handlers package: {e_key}", exc_info=True)
        except Exception as e_load:
            logger.critical(f"Unexpected critical error during dynamic handler loading: {e_load}", exc_info=True)
    return ALL_HANDLERS

def _apply_cli_overrides_to_config(model_config_dict, model_id_key, cli_args):
    config_to_modify = model_config_dict.copy() 
    if cli_args.override_main_model_repo:
        repo_override = cli_args.override_main_model_repo
        updated = False
        if model_id_key in ["orpheus_lex_au", "orpheus_sauerkraut"] and "model_repo_id" in config_to_modify:
            config_to_modify["model_repo_id"] = repo_override; updated = True
        elif model_id_key == "piper_local" and "piper_voice_repo_id" in config_to_modify and not cli_args.override_piper_voices_repo:
            config_to_modify["piper_voice_repo_id"] = repo_override; updated = True
        elif model_id_key == "oute_hf" and "onnx_repo_id" in config_to_modify: 
            config_to_modify["onnx_repo_id"] = repo_override; updated = True
        elif model_id_key.startswith("mlx_audio") and "mlx_model_path" in config_to_modify:
            config_to_modify["mlx_model_path"] = repo_override; updated = True
        elif model_id_key == "speecht5_german_transformers" and "model_id" in config_to_modify:
            config_to_modify["model_id"] = repo_override; updated = True
        elif model_id_key == "fastpitch_german_nemo" and "spectrogram_model_repo_id" in config_to_modify:
            config_to_modify["spectrogram_model_repo_id"] = repo_override; updated = True
        elif model_id_key == "orpheus_kartoffel_natural" and "model_repo_id" in config_to_modify: # Added for Kartoffel
            config_to_modify["model_repo_id"] = repo_override; updated = True


        if updated: logger.info(f"Overriding main model repo for '{model_id_key}' to: {repo_override}")
        elif model_id_key not in ["edge", "orpheus_lm_studio", "orpheus_ollama"]: 
            logger.debug(f"No primary repo key found to override for '{model_id_key}' with '{repo_override}'. Check config keys.")

    if cli_args.override_model_filename:
        fn_override = cli_args.override_model_filename; updated = False
        if model_id_key in ["orpheus_lex_au", "orpheus_sauerkraut"] and "model_filename" in config_to_modify:
            config_to_modify["model_filename"] = fn_override; updated = True
        elif model_id_key == "fastpitch_german_nemo" and "spectrogram_model_filename" in config_to_modify:
            config_to_modify["spectrogram_model_filename"] = fn_override; updated = True
        if updated: logger.info(f"Overriding model filename for '{model_id_key}' to: {fn_override}")

    if cli_args.override_tokenizer_repo:
        tok_override = cli_args.override_tokenizer_repo
        if ("oute_hf" == model_id_key or "oute_llamacpp" == model_id_key) and "tokenizer_path" in config_to_modify:
            config_to_modify["tokenizer_path"] = tok_override; logger.info(f"Overriding 'tokenizer_path' for '{model_id_key}' to: {tok_override}")
        elif model_id_key == "orpheus_kartoffel_natural" and "tokenizer_repo_id" in config_to_modify: # Added for Kartoffel
            config_to_modify["tokenizer_repo_id"] = tok_override; logger.info(f"Overriding 'tokenizer_repo_id' for '{model_id_key}' to: {tok_override}")
        # Ensure key "tokenizer_path_for_mlx_outetts" exists if this model ID is used, or handle more gracefully
        elif "mlx_audio_outetts_clone" == model_id_key and "tokenizer_path_for_mlx_outetts" in config_to_modify: 
             config_to_modify["tokenizer_path_for_mlx_outetts"] = tok_override
             logger.info(f"Overriding 'tokenizer_path_for_mlx_outetts' for '{model_id_key}' to: {tok_override}")

    if cli_args.override_vocoder_repo:
        voc_override = cli_args.override_vocoder_repo
        if model_id_key == "speecht5_german_transformers" and "vocoder_id" in config_to_modify:
            config_to_modify["vocoder_id"] = voc_override; logger.info(f"Overriding 'vocoder_id' for '{model_id_key}' to: {voc_override}")
        elif model_id_key == "fastpitch_german_nemo" and "vocoder_model_name" in config_to_modify:
            config_to_modify["vocoder_model_name"] = voc_override; logger.info(f"Overriding 'vocoder_model_name' for '{model_id_key}' to: {voc_override}")

    if cli_args.override_speaker_embed_repo:
        spk_embed_override = cli_args.override_speaker_embed_repo
        if model_id_key == "speecht5_german_transformers" and "speaker_embeddings_repo" in config_to_modify:
            config_to_modify["speaker_embeddings_repo"] = spk_embed_override
            logger.info(f"Overriding 'speaker_embeddings_repo' for '{model_id_key}' to: {spk_embed_override}")
            
    if cli_args.override_piper_voices_repo and model_id_key == "piper_local":
        config_to_modify["piper_voice_repo_id"] = cli_args.override_piper_voices_repo
        logger.info(f"Overriding 'piper_voice_repo_id' for '{model_id_key}' to: {cli_args.override_piper_voices_repo}")
    return config_to_modify

def test_all_models(text_to_synthesize, base_output_dir_str, cli_args):
    # Deferred imports for benchmark utilities
    soundfile_for_benchmark = None
    pydub_for_benchmark = False # Changed to boolean flag
    AudioSegment_benchmark_imp = None 
    if UTILS_SOUNDFILE_AVAILABLE:
        try:
            import soundfile as sf_benchmark_imp
            soundfile_for_benchmark = sf_benchmark_imp
        except ImportError: 
            logger.debug("Soundfile for benchmark could not be imported.")
            pass # Keep as None
    if UTILS_PYDUB_AVAILABLE:
        try:
            from pydub import AudioSegment as AudioSegment_bm_imp 
            AudioSegment_benchmark_imp = AudioSegment_bm_imp 
            pydub_for_benchmark = True 
        except ImportError: 
            logger.debug("Pydub for benchmark could not be imported.")
            pass # Keep as False

    current_all_handlers = _load_handlers_if_needed()
    if not _HANDLERS_LOADED or not current_all_handlers : 
        logger.critical("Cannot run test_all_models: Handlers failed to load.")
        return

    test_all_speakers_flag = cli_args.test_all_speakers
    logger.info(f"--- Starting Test for All Models ({'All Configured Speakers/Voices' if test_all_speakers_flag else 'Default Speakers/Voices Only'}) ---")
    if cli_args.skip_models:
        logger.info(f"Skipping models based on --skip-models: {', '.join(cli_args.skip_models)}")
    logger.info(f"Input text: \"{text_to_synthesize[:100]}...\"")
    base_output_dir = Path(base_output_dir_str)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {base_output_dir.resolve()}")
    logger.info("------------------------------------")

    benchmark_results = [] 

    for model_id, config_entry_original in GERMAN_TTS_MODELS.items():
        if model_id in cli_args.skip_models:
            logger.info(f"\n>>> Skipping Model (CLI --skip-models): {model_id} <<<")
            benchmark_results.append({
                "model_id": model_id, "voice_id": "N/A", "status": "SKIPPED (CLI)",
                "gen_time_sec": "N/A", "file_size_bytes": "N/A",
                "audio_duration_sec": "N/A", "output_file": "N/A"
            })
            logger.info("------------------------------------")
            continue

        handler_key = config_entry_original.get("handler_function_key", model_id)
        handler_func = current_all_handlers.get(handler_key)
        
        current_model_status = "SKIPPED (No Handler)"
        current_gen_time_sec = None
        current_file_size_bytes = None
        current_audio_duration_sec = None
        current_output_path = None
        current_voice_id_tested = "N/A"

        if not handler_func:
            logger.warning(f"\n>>> No handler found for Model ID: {model_id} (handler key: {handler_key}). Skipping. <<<")
            benchmark_results.append({ "model_id": model_id, "voice_id": current_voice_id_tested, "status": current_model_status,"gen_time_sec": "N/A", "file_size_bytes": "N/A","audio_duration_sec": "N/A", "output_file": "N/A"})
            logger.info("------------------------------------")
            continue
            
        current_config_for_handler = _apply_cli_overrides_to_config(config_entry_original, model_id, cli_args)
        
        if model_id == "orpheus_lm_studio":
            current_config_for_handler["api_url"] = cli_args.lm_studio_api_url
            if cli_args.gguf_model_name_in_api: current_config_for_handler["gguf_model_name_in_api"] = cli_args.gguf_model_name_in_api
        elif model_id == "orpheus_ollama":
            current_config_for_handler["api_url"] = cli_args.ollama_api_url
            if cli_args.ollama_model_name: current_config_for_handler["ollama_model_name"] = cli_args.ollama_model_name
        
        voices_to_test_this_run = []
        if test_all_speakers_flag: 
            if current_config_for_handler.get("available_voices"): voices_to_test_this_run.extend(current_config_for_handler.get("available_voices"))
            if "oute" in model_id and current_config_for_handler.get("test_default_speakers"): voices_to_test_this_run.extend(current_config_for_handler.get("test_default_speakers"))
            if not voices_to_test_this_run: # Add default if list is still empty after checking available_voices
                default_v_candidate = None
                if model_id.startswith("coqui_"): # Coqui specific default
                    coqui_default_speaker_id = current_config_for_handler.get('default_coqui_speaker')
                    if coqui_default_speaker_id and str(coqui_default_speaker_id).strip():
                        default_v_candidate = str(coqui_default_speaker_id)
                
                if not default_v_candidate: # Generic defaults
                    default_v_candidate = (current_config_for_handler.get('default_voice_id') or 
                                           current_config_for_handler.get('default_model_path_in_repo'))
                if not default_v_candidate: # Index/ID based defaults
                    idx_val = current_config_for_handler.get('default_speaker_embedding_index')
                    if idx_val is not None: default_v_candidate = str(idx_val)
                    else:
                        idx_val_speaker = current_config_for_handler.get('default_speaker_id')
                        if idx_val_speaker is not None: default_v_candidate = str(idx_val_speaker)
                
                if default_v_candidate and str(default_v_candidate).strip(): 
                    voices_to_test_this_run.append(str(default_v_candidate))
                # Fallback for Coqui single-speaker models if still nothing found
                elif model_id.startswith("coqui_") and current_config_for_handler.get("default_coqui_speaker") is None and current_config_for_handler.get("available_voices") == ["default_speaker"]:
                    voices_to_test_this_run.append("default_speaker")
                    logger.info(f"Coqui single-speaker model '{model_id}' for --test-all-speakers, using placeholder 'default_speaker'.")
        else: # Not test_all_speakers_flag, so get only the single default voice OR handle zero-shot
            default_v_to_add = None
            
            # NEW: Check for explicit zero-shot configuration for default test mode
            # A model is considered zero-shot for this purpose if its default_voice_id is None
            # AND it has no specific available_voices listed (indicating it doesn't rely on a predefined voice set)
            # AND it's not a type of model that inherently requires a speaker/voice even in default (e.g. some Coqui models)
            is_primarily_zero_shot_type = (
                current_config_for_handler.get('default_voice_id') is None and
                not current_config_for_handler.get('available_voices')
            )
            # Add specific model IDs here if they are zero-shot but might have other default fields that confuse the logic below
            # For example, if a model is zero-shot but happens to have a 'default_speaker_id' for some other purpose.
            if model_id in ["llasa_hybrid_de_zeroshot", "llasa_german_transformers_zeroshot", "llasa_multilingual_hf_zeroshot"]: # Be explicit for known zero-shot LLaSA
                is_primarily_zero_shot_type = True


            if is_primarily_zero_shot_type:
                logger.info(f"Model '{model_id}': Identified as zero-shot or configured for such a test. Proceeding with 'None' as voice_id for default test.")
                voices_to_test_this_run.append(None) # Use None to signify zero-shot for the handler
            else:
                # Original logic to find a default_v_to_add
                # Priority 1: Coqui-specific default speaker ID
                if model_id.startswith("coqui_"):
                    coqui_default_speaker_id = current_config_for_handler.get('default_coqui_speaker')
                    if coqui_default_speaker_id and str(coqui_default_speaker_id).strip():
                        default_v_to_add = str(coqui_default_speaker_id)
                        logger.debug(f"Model '{model_id}': Using 'default_coqui_speaker': {default_v_to_add} for default test run.")
                
                # Priority 2: Standard default voice/path keys
                if not default_v_to_add:
                    std_default_keys = ['default_voice_id', 'default_model_path_in_repo']
                    for key_cfg in std_default_keys: # Renamed key to key_cfg to avoid conflict
                        val = current_config_for_handler.get(key_cfg)
                        # Ensure val is not None and, if string, not empty after stripping
                        if val is not None and (not isinstance(val, str) or str(val).strip()):
                            default_v_to_add = str(val)
                            logger.debug(f"Model '{model_id}': Found default via '{key_cfg}': {default_v_to_add}")
                            break
                
                # Priority 3: Index/ID based defaults
                if not default_v_to_add:
                    idx_val_embed = current_config_for_handler.get('default_speaker_embedding_index')
                    if idx_val_embed is not None: # Check for None explicitly for numeric 0
                        default_v_to_add = str(idx_val_embed)
                        logger.debug(f"Model '{model_id}': Found default via 'default_speaker_embedding_index': {default_v_to_add}")
                    else:
                        idx_val_speaker = current_config_for_handler.get('default_speaker_id')
                        if idx_val_speaker is not None: # Check for None explicitly for numeric 0
                            default_v_to_add = str(idx_val_speaker)
                            logger.debug(f"Model '{model_id}': Found default via 'default_speaker_id': {default_v_to_add}")
                
                # Priority 4: Fallback for Coqui single-speaker models using "default_speaker" placeholder
                if not default_v_to_add and \
                   model_id.startswith("coqui_") and \
                   current_config_for_handler.get("default_coqui_speaker") is None and \
                   current_config_for_handler.get("available_voices") == ["default_speaker"]:
                    default_v_to_add = "default_speaker" # This specific string might be handled by Coqui handler
                    logger.info(f"Coqui single-speaker model '{model_id}': Using placeholder 'default_speaker' for default test run.")
                
                if default_v_to_add is not None and (not isinstance(default_v_to_add, str) or str(default_v_to_add).strip()):
                    voices_to_test_this_run.append(str(default_v_to_add))
                elif not voices_to_test_this_run: # Only log if it's still empty and not identified as zero-shot
                    logger.debug(f"Model '{model_id}': No default voice/speaker could be determined for default speaker test mode (and not flagged as zero-shot for this test).")

        unique_voices_to_test = list(dict.fromkeys(voices_to_test_this_run)) # Handles [None] correctly

        #voices_to_test_this_run = [v for v in voices_to_test_this_run if v is not None and str(v).strip()]
        
        if not unique_voices_to_test: # This will now be false if voices_to_test_this_run contains [None]
            current_model_status = "SKIPPED (No Voice/Default Identified)" # Clarify message
            logger.info(f"\n>>> Skipping Model: {model_id} ({current_model_status} for this mode) <<<")
            benchmark_results.append({
                "model_id": model_id, "voice_id": "N/A", "status": current_model_status,
                "gen_time_sec": "N/A", "file_size_bytes": "N/A",
                "audio_duration_sec": "N/A", "output_file": "N/A"
            })
            logger.info("------------------------------------")
            continue
        
        for voice_idx, voice_id_for_test in enumerate(unique_voices_to_test):
            current_voice_id_tested = str(voice_id_for_test)
            speaker_suffix_for_file = ""
            if test_all_speakers_flag and len(unique_voices_to_test) > 1:
                sanitized_voice_id = str(voice_id_for_test).replace('/', '_').replace('\\','_').replace(':','-')
                sanitized_voice_id = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in sanitized_voice_id)
                speaker_suffix_for_file = f"_voice_{sanitized_voice_id[:30]}"

            output_suffix = ".wav" 
            if model_id == "edge": output_suffix = ".mp3"
            # Ensure model_id in filename is also sanitized for special characters like '/'
            sanitized_model_id_for_filename = model_id.replace('/', '_').replace(':','-')
            output_filename = base_output_dir / f"test_output_{sanitized_model_id_for_filename}{speaker_suffix_for_file}{output_suffix}"
            current_output_path = output_filename

            logger.info(f"\n>>> Testing Model: {model_id} (Voice/Speaker: {voice_id_for_test}) <<<")
            
            start_time_model_test = time.time()
            current_gen_time_sec = None 
            current_file_size_bytes = None
            current_audio_duration_sec = None
            try:
                handler_func(current_config_for_handler, text_to_synthesize, str(voice_id_for_test), cli_args.model_params, str(output_filename), False)
                current_gen_time_sec = time.time() - start_time_model_test

                if output_filename.exists() and output_filename.stat().st_size > 100: 
                    current_model_status = "SUCCESS"
                    logger.info(f"SUCCESS: Output for {model_id} (Voice: {voice_id_for_test}) saved to {output_filename}")
                    current_file_size_bytes = output_filename.stat().st_size
                    try:
                        if output_filename.suffix.lower() == ".wav" and soundfile_for_benchmark:
                            data, samplerate = soundfile_for_benchmark.read(str(output_filename))
                            if samplerate > 0: current_audio_duration_sec = len(data) / samplerate
                        elif output_filename.suffix.lower() == ".mp3" and pydub_for_benchmark and AudioSegment_benchmark_imp:
                            audio_seg = AudioSegment_benchmark_imp.from_file(str(output_filename))
                            current_audio_duration_sec = len(audio_seg) / 1000.0
                    except Exception as e_dur: logger.warning(f"Could not determine audio duration for {output_filename}: {e_dur}")
                else:
                    current_model_status = "FAIL (No/Small File)"; logger.warning(f"NOTE: Synthesis for {model_id} (Voice: {voice_id_for_test}) ran. Output file '{output_filename}' not created or is empty/too small.")
            except Exception as e_test_model:
                if current_gen_time_sec is None: current_gen_time_sec = time.time() - start_time_model_test
                current_model_status = "ERROR"; logger.error(f"ERROR: Testing {model_id} (Voice: {voice_id_for_test}) failed: {e_test_model}", exc_info=True)

            benchmark_results.append({ "model_id": model_id, "voice_id": current_voice_id_tested, "status": current_model_status, "gen_time_sec": f"{current_gen_time_sec:.2f}s" if current_gen_time_sec is not None else "N/A", "file_size_bytes": current_file_size_bytes if current_file_size_bytes is not None else "N/A", "audio_duration_sec": f"{current_audio_duration_sec:.2f}s" if current_audio_duration_sec is not None else "N/A", "output_file": str(current_output_path.name) if current_output_path else "N/A" })
            if test_all_speakers_flag and len(unique_voices_to_test) > 1 and voice_idx < len(unique_voices_to_test) -1 : logger.info("---") 
        logger.info("------------------------------------")

    logger.info("--- Test for All Models Finished ---")
    logger.info("\n--- BENCHMARK SUMMARY ---")
    if benchmark_results:
        # Calculate column widths
        max_len = lambda key_str, default_len: max(default_len, max(len(str(r[key_str])) for r in benchmark_results if r[key_str] is not None and str(r[key_str]).strip() != ""))
        
        col_model = max_len("model_id", len("Model ID"))
        col_voice = max_len("voice_id", len("Voice/Speaker"))
        col_status = max_len("status", len("Status"))
        col_gentime = max_len("gen_time_sec", len("Gen Time"))
        col_size = max_len("file_size_bytes", len("Size (Bytes)"))
        col_duration = max_len("audio_duration_sec", len("Audio (s)"))
        col_file = max_len("output_file", len("File"))

        header_parts = [f" {'Model ID'.ljust(col_model)} ", f" {'Voice/Speaker'.ljust(col_voice)} ", f" {'Status'.ljust(col_status)} ", f" {'Gen Time'.rjust(col_gentime)} ", f" {'Size (Bytes)'.rjust(col_size)} ", f" {'Audio (s)'.rjust(col_duration)} ", f" {'File'.ljust(col_file)} "]
        header = f"|{'|'.join(header_parts)}|"
        sep_parts = [f"{'-'*(col_model+2)}", f"{'-'*(col_voice+2)}", f"{'-'*(col_status+2)}", f"{'-'*(col_gentime+2)}", f"{'-'*(col_size+2)}", f"{'-'*(col_duration+2)}", f"{'-'*(col_file+2)}"]
        separator = f"|{'|'.join(sep_parts)}|"
        logger.info(separator); logger.info(header); logger.info(separator)
        
        for r in benchmark_results:
            row_parts = [f" {str(r['model_id']).ljust(col_model)} ", f" {str(r['voice_id']).ljust(col_voice)} ", f" {str(r['status']).ljust(col_status)} ", f" {str(r['gen_time_sec']).rjust(col_gentime)} ", f" {str(r['file_size_bytes']).rjust(col_size)} ", f" {str(r['audio_duration_sec']).rjust(col_duration)} ", f" {str(r['output_file']).ljust(col_file)} "]
            logger.info(f"|{'|'.join(row_parts)}|")
        logger.info(separator)
    else: logger.info("No benchmark results to display.")

def run_synthesis(args):
    current_all_handlers = _load_handlers_if_needed()
    if not _HANDLERS_LOADED or not current_all_handlers:
        logger.critical("Cannot run synthesis: Handlers failed to load.")
        return

    text_to_synthesize = get_text_from_input(args.input_text, args.input_file) 
    if not text_to_synthesize: 
        logger.error("No input text resolved for synthesis.")
        return
    text_to_synthesize = text_to_synthesize[:3000]

    model_config_base = GERMAN_TTS_MODELS.get(args.model_id)
    if not model_config_base: 
        logger.error(f"Invalid model ID '{args.model_id}' passed to run_synthesis.")
        return

    logger.info(f"Synthesizing with: {args.model_id}")
    logger.info(f"Input (start): '{text_to_synthesize[:70]}...'")

    current_config_for_handler = _apply_cli_overrides_to_config(model_config_base, args.model_id, args)
    if args.model_id == "orpheus_lm_studio":
        current_config_for_handler["api_url"] = args.lm_studio_api_url
        if args.gguf_model_name_in_api: current_config_for_handler["gguf_model_name_in_api"] = args.gguf_model_name_in_api
    elif args.model_id == "orpheus_ollama":
        current_config_for_handler["api_url"] = args.ollama_api_url
        if args.ollama_model_name: current_config_for_handler["ollama_model_name"] = args.ollama_model_name
        if "USER MUST SET" in current_config_for_handler.get("ollama_model_name","") or not current_config_for_handler.get("ollama_model_name"):
            logger.error(f"For {args.model_id}, Ollama model name not set. Use --ollama-model-name or set in config."); return
    
    effective_voice_id = args.german_voice_id
    if not effective_voice_id: 
        default_v = current_config_for_handler.get('default_voice_id') or \
                    current_config_for_handler.get('default_model_path_in_repo') or \
                    str(current_config_for_handler.get('default_speaker_embedding_index', '')) or \
                    str(current_config_for_handler.get('default_speaker_id', ''))
        effective_voice_id = default_v if (isinstance(default_v, Path) or (isinstance(default_v, str) and default_v.strip())) else None
    if not effective_voice_id and not (args.model_id.startswith("coqui_tts") and current_config_for_handler.get("default_coqui_speaker") is None):
        logger.error(f"No voice ID specified and no default could be determined for model {args.model_id}."); return

    handler_key = current_config_for_handler.get("handler_function_key", args.model_id)
    handler_func = current_all_handlers.get(handler_key)

    if handler_func:
        try:
            handler_func(current_config_for_handler, text_to_synthesize, str(effective_voice_id) if effective_voice_id is not None else None, args.model_params, args.output_file, args.play_direct)
        except Exception as e_synth:
            logger.error(f"Synthesis failed for model {args.model_id}: {e_synth}", exc_info=True)
    else:
        logger.error(f"No synthesis handler function found for model ID: {args.model_id} (handler key: {handler_key})")

def main_cli_entrypoint():
    parser = argparse.ArgumentParser(description="CrispTTS: Modular German Text-to-Speech Synthesizer", formatter_class=argparse.RawTextHelpFormatter)
    action_group = parser.add_argument_group(title="Primary Actions")
    input_group = parser.add_mutually_exclusive_group(required=False) 
    action_group.add_argument("--list-models", action="store_true", help="List all configured TTS models.")
    action_group.add_argument("--voice-info", type=str, metavar="MODEL_ID", help="Display voice/speaker info for a specific MODEL_ID.")
    action_group.add_argument("--test-all", action="store_true", help="Test all models with default voices. Requires --input-text or --input-file.")
    action_group.add_argument("--test-all-speakers", action="store_true", help="Test all models with ALL configured voices. Requires --input-text or --input-file.")
    action_group.add_argument("--skip-models", type=str, nargs='*', default=[], help="List of model IDs (space-separated) to skip during --test-all or --test-all-speakers.")


    synth_group = parser.add_argument_group(title="Synthesis Options (used with --model-id or --test-all*)")
    input_group.add_argument("--input-text", type=str, help="Text to synthesize.")
    input_group.add_argument("--input-file", type=str, help="Path to input file (txt, md, html, pdf, epub).")
    
    model_choices = list(GERMAN_TTS_MODELS.keys()) if GERMAN_TTS_MODELS else []
    synth_group.add_argument("--model-id", type=str, choices=model_choices, default=None, help="Select TTS model ID. Required for single synthesis if not using an action flag.")
    synth_group.add_argument("--output-file", type=str, help="Path to save synthesized audio (for single synthesis).")
    synth_group.add_argument("--output-dir", type=str, default="tts_test_outputs", help="Directory for --test-all* outputs (default: tts_test_outputs).")
    synth_group.add_argument("--play-direct", action="store_true", help="Play audio directly after synthesis (not with --test-all*).")
    synth_group.add_argument("--german-voice-id", type=str, help="Override default voice/speaker for the selected model.")
    synth_group.add_argument("--model-params", type=str, help="JSON string of model-specific parameters (e.g., '{\"temperature\":0.7}').")
    
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set console logging level (default: INFO).")

    override_group = parser.add_argument_group(title="Runtime Model Path/Repo Overrides (for selected --model-id or during --test-all*)")
    override_group.add_argument("--override-main-model-repo", type=str, metavar="REPO_OR_PATH", help="Override main model repository ID or path.")
    override_group.add_argument("--override-model-filename", type=str, metavar="FILENAME", help="Override specific model filename within the main repo.")
    override_group.add_argument("--override-tokenizer-repo", type=str, metavar="REPO_OR_PATH", help="Override tokenizer repository ID or path.")
    override_group.add_argument("--override-vocoder-repo", type=str, metavar="REPO_OR_NAME", help="Override vocoder repository ID or name.")
    override_group.add_argument("--override-speaker-embed-repo", type=str, metavar="REPO_ID", help="Override speaker embeddings repository ID.")
    override_group.add_argument("--override-piper-voices-repo", type=str, metavar="REPO_ID", help="Override main repository ID for Piper voices.")

    api_group = parser.add_argument_group(title="API Backend Overrides (also in config.py)")
    api_group.add_argument("--lm-studio-api-url", type=str, default=LM_STUDIO_API_URL_DEFAULT if 'LM_STUDIO_API_URL_DEFAULT' in globals() else "http://127.0.0.1:1234/v1/completions", help=f"Override LM Studio API URL.")
    api_group.add_argument("--gguf-model-name-in-api", type=str, help="Override model name for LM Studio API (from config or this flag).")
    api_group.add_argument("--ollama-api-url", type=str, default=OLLAMA_API_URL_DEFAULT if 'OLLAMA_API_URL_DEFAULT' in globals() else "http://localhost:11434/api/generate", help=f"Override Ollama API URL.")
    api_group.add_argument("--ollama-model-name", type=str, help="Override model name/tag for Ollama (from config or this flag).")

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    cli_numeric_log_level = getattr(logging, args.loglevel.upper(), logging.INFO)

    for name in list(logging.root.manager.loggerDict.keys()) + ["root", _main_mp_logger.name]:
        lgr = logging.getLogger(name)
        if lgr.name.startswith("CrispTTS") or lgr.name == "root":
            if lgr.level == 0 or lgr.level > cli_numeric_log_level: lgr.setLevel(cli_numeric_log_level)
            if lgr.name == _main_mp_logger.name:
                for handler_obj in lgr.handlers: # handler_obj instead of handler
                    if handler_obj.level == 0 or handler_obj.level > cli_numeric_log_level: handler_obj.setLevel(cli_numeric_log_level)
        
        if cli_numeric_log_level == logging.INFO:
            # List of prefixes for third-party loggers to quieten
            noisy_prefixes = ["vllm", "transformers", "huggingface_hub", "torch", "pydub", "soundfile", 
                              "mlx_audio", "nemo_collections", "datasets", "matplotlib", "PIL", "git", "wandb",
                              "numba", "urllib3", "filelock", "fsspec", "gruut", "charset_normalizer",
                              "torchaudio", "TTS.utils", "TTS.tts.utils", "TTS.vc.utils"] # Added more based on logs
            # TTS often logs model downloads at INFO, which is fine, but internal steps can be noisy.
            # Some libraries like 'TTS' might have sub-loggers that are too verbose.
            # Example: 'TTS.tts.models.xtts' or 'TTS.vocoder.models'
            
            is_noisy_third_party = any(lgr.name.startswith(prefix) for prefix in noisy_prefixes)
            is_crisptts_sub_logger = lgr.name.startswith("CrispTTS.") and lgr.name != "CrispTTS.main" # Allow main app INFO

            if is_noisy_third_party and not is_crisptts_sub_logger : # Don't silence our main app logger's INFO
                if lgr.getEffectiveLevel() < logging.WARNING:
                    lgr.setLevel(logging.WARNING)
                    # Debug log from main logger, not the one being quieted
                    logger.debug(f"Set logger '{lgr.name}' to WARNING to reduce noise at INFO level.")


    logger.info(f"Effective logging level for CrispTTS and sub-loggers set to: {args.loglevel.upper()}")
    _main_mp_logger.debug(f"Monkey patch logger effective level is: {logging.getLevelName(_main_mp_logger.getEffectiveLevel())}")


    if args.list_models:
        list_available_models(GERMAN_TTS_MODELS); return
    if args.voice_info:
        if not args.voice_info in GERMAN_TTS_MODELS: logger.error(f"Model ID '{args.voice_info}' for --voice-info not found."); return
        get_voice_info(args.voice_info, GERMAN_TTS_MODELS); return

    text_to_process = get_text_from_input(args.input_text, args.input_file)
    if not text_to_process:
        if args.test_all or args.test_all_speakers: parser.error("--test-all or --test-all-speakers requires --input-text or --input-file.")
        elif args.model_id : logger.error("No text input provided for synthesis via --input-text or --input-file.")
        else: parser.print_help()
        return

    if args.test_all or args.test_all_speakers:
        _load_handlers_if_needed() # Load handlers before testing all
        if not _HANDLERS_LOADED: logger.critical("Failed to load handlers. Aborting --test-all / --test-all-speakers."); return
        test_text = text_to_process[:500] if len(text_to_process) > 500 else text_to_process 
        logger.info(f"--- Applying Test Mode: {'All Speakers' if args.test_all_speakers else 'Default Speaker Only'} ---")
        if any([args.override_main_model_repo, args.override_model_filename, args.override_tokenizer_repo, args.override_vocoder_repo, args.override_speaker_embed_repo, args.override_piper_voices_repo]):
            logger.warning("CLI repo/path overrides are active and will apply to all compatible models during --test-all(-speakers).")
        test_all_models(test_text, args.output_dir, args)
        return

    if not args.model_id: parser.error("A --model-id is required for synthesis if not using an action flag."); return
    
    _load_handlers_if_needed() 
    if not _HANDLERS_LOADED: logger.critical(f"Failed to load handlers. Aborting synthesis for model '{args.model_id}'."); return
    run_synthesis(args)

if __name__ == "__main__":
    _torch_available_main = False; _is_mps_main = False # Keep these checks light and local if only for info
    try: import torch; _torch_available_main = True
    except ImportError: pass
    if _torch_available_main:
        try: _is_mps_main = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception: pass # Catch any error during mps check
    # The logger isn't configured yet if main_cli_entrypoint hasn't run.
    # So, these debug logs won't show with default config.
    # print(f"DEBUG (pre-log): Torch available: {_torch_available_main}, MPS available: {_is_mps_main}")
    main_cli_entrypoint()