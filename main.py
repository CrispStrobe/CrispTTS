#!/usr/bin/env python3
# CrispTTS - main.py
# Main Command-Line Interface for the Text-to-Speech Synthesizer (Modularized with Overrides)

import sys # KEEP AT TOP
from pathlib import Path # KEEP AT TOP
import time # KEEP AT TOP

# --- Add project root to sys.path to ensure relative imports work ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Path Adjustment ---

import argparse # KEEP
import os # KEEP
import json # KEEP
import logging # KEEP
import types # NEEDED FOR MONKEY PATCH TYPE CHECKING

# --- Environment Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false" # KEEP
os.environ["GGML_METAL_NDEBUG"] = "1" # KEEP

# === START MONKEY PATCH FOR VLLM TRITON PLACEHOLDER ===
# Setup a specific logger for the monkey patch actions that can log early
_main_mp_logger = logging.getLogger("CrispTTS.main_monkey_patch")
if not _main_mp_logger.handlers:
    _mp_handler = logging.StreamHandler(sys.stderr)
    _mp_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - MONKEY_PATCH: %(message)s')
    _mp_handler.setFormatter(_mp_formatter)
    _main_mp_logger.addHandler(_mp_handler)
    _main_mp_logger.setLevel(logging.INFO) # Ensure INFO messages from patch are visible
    _main_mp_logger.propagate = False # Don't send to root logger if already handled

class _CrispTTSDummyTritonConfig:
    """A dummy class to stand in for triton.Config."""
    def __init__(self, *args, **kwargs):
        _main_mp_logger.debug(f"DummyTritonConfig initialized with args: {args}, kwargs: {kwargs}")
        pass

def _apply_triton_config_monkey_patch_for_vllm():
    """
    Checks if vLLM's TritonPlaceholder is in sys.modules and, if so,
    adds dummy 'Config', 'cdiv', and various triton.language attributes
    to it, to prevent AttributeErrors when torchao tries to access them.
    """
    # Logger is already defined globally in main.py as _main_mp_logger
    # _main_mp_logger = logging.getLogger("CrispTTS.main_monkey_patch") # Redundant if already global

    try:
        if 'triton' in sys.modules:
            triton_module = sys.modules['triton']
            placeholder_type_name = str(type(triton_module))
            
            # Check if it's vLLM's placeholder by its specific type name and a known unique attribute
            is_likely_vllm_placeholder = (
                "vllm" in placeholder_type_name and 
                "TritonPlaceholder" in placeholder_type_name and
                hasattr(triton_module, '_dummy_decorator') 
            )

            if is_likely_vllm_placeholder:
                _main_mp_logger.info("vLLM's TritonPlaceholder detected in sys.modules.")

                # Patch triton.Config
                if not hasattr(triton_module, 'Config'):
                    _main_mp_logger.info("Adding 'Config' attribute to vLLM's TritonPlaceholder.")
                    setattr(triton_module, 'Config', _CrispTTSDummyTritonConfig) # _CrispTTSDummyTritonConfig needs to be defined globally in main.py
                else:
                    _main_mp_logger.info("vLLM's TritonPlaceholder already has 'Config'.")

                # Patch triton.cdiv
                if not hasattr(triton_module, 'cdiv'):
                    _main_mp_logger.info("Adding 'cdiv' attribute to vLLM's TritonPlaceholder.")
                    setattr(triton_module, 'cdiv', lambda x, y: (x + y - 1) // y)
                else:
                    _main_mp_logger.info("vLLM's TritonPlaceholder already has 'cdiv'.")

                # Patch attributes within triton.language
                if hasattr(triton_module, 'language'):
                    triton_lang_module = triton_module.language
                    lang_placeholder_type_name = str(type(triton_lang_module))

                    is_likely_vllm_lang_placeholder = (
                        "vllm" in lang_placeholder_type_name and
                        "TritonLanguagePlaceholder" in lang_placeholder_type_name
                    )

                    if is_likely_vllm_lang_placeholder:
                        _main_mp_logger.info("vLLM's TritonLanguagePlaceholder found. Patching missing attributes.")
                        
                        class _DummyDtypePlaceholder:
                            def __init__(self, name_): self.name = name_
                            def __repr__(self): return f"tl.{self.name}" # Mimic Triton dtype repr
                            # Add .to() method as it might be called by tl.constexpr(dtype.to(device_type))
                            def to(self, target_device_type_str): # target_device_type_str e.g. "device_type"
                                _main_mp_logger.debug(f"DummyDtypePlaceholder {self.name}.to({target_device_type_str}) called.")
                                return self # Return self, as it's just a type descriptor

                        dtypes_to_add = {
                            "int1": _DummyDtypePlaceholder("int1"), # Often used for masks
                            "int8": _DummyDtypePlaceholder("int8"),
                            "int16": _DummyDtypePlaceholder("int16"),
                            "int32": _DummyDtypePlaceholder("int32"), # Error was here
                            "uint8": _DummyDtypePlaceholder("uint8"),
                            "uint16": _DummyDtypePlaceholder("uint16"),
                            "uint32": _DummyDtypePlaceholder("uint32"),
                            "uint64": _DummyDtypePlaceholder("uint64"),
                            "float8e4nv": _DummyDtypePlaceholder("float8e4nv"), # Common in new models
                            "float8e5": _DummyDtypePlaceholder("float8e5"),   # Common in new models
                            "float16": _DummyDtypePlaceholder("float16"),
                            "bfloat16": _DummyDtypePlaceholder("bfloat16"),
                            "float32": _DummyDtypePlaceholder("float32"),
                            "float64": _DummyDtypePlaceholder("float64"), # Proactive
                            # tl.int64 is already defined in vllm's placeholder
                        }
                        
                        for dtype_name, dtype_obj in dtypes_to_add.items():
                            if not hasattr(triton_lang_module, dtype_name):
                                _main_mp_logger.info(f"Adding dtype '{dtype_name}' to TritonLanguagePlaceholder.")
                                setattr(triton_lang_module, dtype_name, dtype_obj)
                        
                        # Patch tl.constexpr (vllm has it as None)
                        # In real Triton, it's a decorator/function. For type hints/annotations, identity works.
                        # If torchao uses it as `tl.constexpr[X]`, this simple lambda won't work.
                        # `torchao/kernel/intmm_triton.py` uses `ACC_TYPE: tl.constexpr = tl.int32`
                        # This means tl.constexpr is used as a type hint for assignment, not a call.
                        # However, it can also be used as `@triton.jit def kernel(..., X: tl.constexpr):`
                        # An identity function is a safer bet than None if it's ever called.
                        if getattr(triton_lang_module, 'constexpr', 'NOT_SET') is None:
                            _main_mp_logger.info("Patching 'constexpr' in TritonLanguagePlaceholder to be an identity function.")
                            setattr(triton_lang_module, 'constexpr', lambda x: x)
                        
                        # Patch tl.dtype (vllm has it as None)
                        # In real Triton, tl.dtype is a class/factory: e.g. tl.dtype("int32")
                        # Or it can be a dictionary: tl.dtype = {"int32": tl.int32_t, ...}
                        # Making it a factory for our dummy dtypes is safer.
                        current_dtype_attr = getattr(triton_lang_module, 'dtype', 'NOT_SET')
                        if current_dtype_attr is None or not callable(current_dtype_attr):
                             _main_mp_logger.info("Patching 'dtype' in TritonLanguagePlaceholder to be a dummy factory.")
                             setattr(triton_lang_module, 'dtype', 
                                     lambda name_str_or_obj: dtypes_to_add.get(str(name_str_or_obj), _DummyDtypePlaceholder(str(name_str_or_obj))) 
                                     if isinstance(name_str_or_obj, str) 
                                     else name_str_or_obj # If an object is passed, return it (e.g. tl.int32)
                                    )
                        
                        # Other potentially needed triton.language attributes by torchao's intmm_triton.py:
                        # tl.dot, tl.zeros, tl.arange, tl.sum, tl.sigmoid, tl.exp, tl.clamp
                        # tl.load, tl.store, tl.make_block_ptr, tl.advance, tl.PROGRAM_ID, tl.num_programs
                        # For now, only adding what's explicitly errored or highly likely.
                        if not hasattr(triton_lang_module, 'PROGRAM_ID'):
                            _main_mp_logger.info("Adding dummy 'PROGRAM_ID' to TritonLanguagePlaceholder.")
                            setattr(triton_lang_module, 'PROGRAM_ID', lambda axis: 0) # Needs to be callable

                        if not hasattr(triton_lang_module, 'make_block_ptr'):
                            _main_mp_logger.info("Adding dummy 'make_block_ptr' to TritonLanguagePlaceholder.")
                            # This is complex. For now, a placeholder that doesn't crash on call.
                            setattr(triton_lang_module, 'make_block_ptr', lambda *args, **kwargs: None)

                        if not hasattr(triton_lang_module, 'load'):
                            _main_mp_logger.info("Adding dummy 'load' to TritonLanguagePlaceholder.")
                            setattr(triton_lang_module, 'load',  lambda *args, **kwargs: None) # Assuming it returns something like a tensor

                        if not hasattr(triton_lang_module, 'store'):
                            _main_mp_logger.info("Adding dummy 'store' to TritonLanguagePlaceholder.")
                            setattr(triton_lang_module, 'store', lambda *args, **kwargs: None)

                        if not hasattr(triton_lang_module, 'dot'):
                            _main_mp_logger.info("Adding dummy 'dot' to TritonLanguagePlaceholder.")
                            setattr(triton_lang_module, 'dot', lambda *args, **kwargs: None) # Assuming it returns something

                    else:
                        _main_mp_logger.debug(f"sys.modules['triton'].language (type: {lang_placeholder_type_name}) is not the target vLLM placeholder.")
                else:
                    _main_mp_logger.warning("vLLM's TritonPlaceholder does not have a 'language' attribute to patch.")
            # else:
            #     _main_mp_logger.debug(f"sys.modules['triton'] (type: {placeholder_type_name}) is present but not target vLLM placeholder, or already patched.")
        # else:
        #     _main_mp_logger.debug("'triton' not in sys.modules when attempting patch. This may be okay if vLLM imports later.")
    except Exception as e_mp:
        print(f"CRITICAL MONKEY PATCH ERROR: {e_mp}", file=sys.stderr)
        _main_mp_logger.error(f"Error during Triton monkey patching: {e_mp}", exc_info=True)

# === END MONKEY PATCH DEFINITIONS ===

# --- Project-Specific Imports ---
try:
    from config import (
        GERMAN_TTS_MODELS,
        LM_STUDIO_API_URL_DEFAULT, 
        OLLAMA_API_URL_DEFAULT    
    )
    from utils import (
        get_text_from_input,
        list_available_models,
        get_voice_info,
        PYDUB_AVAILABLE as UTILS_PYDUB_AVAILABLE,
        SOUNDFILE_AVAILABLE as UTILS_SOUNDFILE_AVAILABLE
    )

    # --- Call the monkey patch here ---
    # This timing is crucial: after potential vLLM import via config/utils,
    # and before handlers trigger transformers/torchao.
    _apply_triton_config_monkey_patch_for_vllm()

    from handlers import ALL_HANDLERS 

except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import from project modules (config, utils, handlers): {e}", file=sys.stderr)
    print("Please ensure these modules/packages are correctly structured and in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)
except KeyError as e_key: 
    print(f"CRITICAL ERROR: 'ALL_HANDLERS' map not found or incomplete in handlers package: {e_key}", file=sys.stderr)
    sys.exit(1)

# --- Imports for Benchmark ---
if UTILS_SOUNDFILE_AVAILABLE:
    try:
        import soundfile as sf
        SOUNDFILE_FOR_BENCHMARK = True
    except ImportError:
        SOUNDFILE_FOR_BENCHMARK = False
else:
    SOUNDFILE_FOR_BENCHMARK = False

if UTILS_PYDUB_AVAILABLE:
    try:
        from pydub import AudioSegment
        PYDUB_FOR_BENCHMARK = True
    except ImportError:
        PYDUB_FOR_BENCHMARK = False
else:
    PYDUB_FOR_BENCHMARK = False


# --- Conditional Library Availability Checks (for user feedback at startup) ---
TORCH_AVAILABLE_MAIN = False
IS_MPS_MAIN = False
try:
    import torch
    TORCH_AVAILABLE_MAIN = True
    IS_MPS_MAIN = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except ImportError:
    pass 

logger = logging.getLogger("CrispTTS.main")


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
        
        if updated:
            logger.info(f"Overriding main model repo for '{model_id_key}' to: {repo_override}")
        elif model_id_key not in ["edge", "orpheus_lm_studio", "orpheus_ollama"]: 
            logger.debug(f"No primary repo key found to override for '{model_id_key}' with '{repo_override}'. Check config keys.")


    if cli_args.override_model_filename:
        fn_override = cli_args.override_model_filename
        updated = False
        if model_id_key in ["orpheus_lex_au", "orpheus_sauerkraut"] and "model_filename" in config_to_modify:
            config_to_modify["model_filename"] = fn_override; updated = True
        elif model_id_key == "fastpitch_german_nemo" and "spectrogram_model_filename" in config_to_modify:
            config_to_modify["spectrogram_model_filename"] = fn_override; updated = True
        
        if updated:
            logger.info(f"Overriding model filename for '{model_id_key}' to: {fn_override}")

    if cli_args.override_tokenizer_repo:
        tok_override = cli_args.override_tokenizer_repo
        if ("oute_hf" == model_id_key or "oute_llamacpp" == model_id_key) and "tokenizer_path" in config_to_modify:
            config_to_modify["tokenizer_path"] = tok_override
            logger.info(f"Overriding 'tokenizer_path' for '{model_id_key}' to: {tok_override}")
        elif "mlx_audio_outetts_q4" == model_id_key and "tokenizer_path_for_mlx_outetts" in config_to_modify:
            config_to_modify["tokenizer_path_for_mlx_outetts"] = tok_override
            logger.info(f"Overriding 'tokenizer_path_for_mlx_outetts' for '{model_id_key}' to: {tok_override}")


    if cli_args.override_vocoder_repo:
        voc_override = cli_args.override_vocoder_repo
        if model_id_key == "speecht5_german_transformers" and "vocoder_id" in config_to_modify:
            config_to_modify["vocoder_id"] = voc_override
            logger.info(f"Overriding 'vocoder_id' for '{model_id_key}' to: {voc_override}")
        elif model_id_key == "fastpitch_german_nemo" and "vocoder_model_name" in config_to_modify:
            config_to_modify["vocoder_model_name"] = voc_override
            logger.info(f"Overriding 'vocoder_model_name' for '{model_id_key}' to: {voc_override}")

    if cli_args.override_speaker_embed_repo:
        spk_embed_override = cli_args.override_speaker_embed_repo
        if model_id_key == "speecht5_german_transformers" and "speaker_embeddings_repo" in config_to_modify:
            config_to_modify["speaker_embeddings_repo"] = spk_embed_override
            logger.info(f"Overriding 'speaker_embeddings_repo' for '{model_id_key}' to: {spk_embed_override}")
            
    if cli_args.override_piper_voices_repo and model_id_key == "piper_local":
        config_to_modify["piper_voice_repo_id"] = cli_args.override_piper_voices_repo
        logger.info(f"Overriding 'piper_voice_repo_id' for '{model_id_key}' specifically to: {cli_args.override_piper_voices_repo}")

    return config_to_modify


def test_all_models(text_to_synthesize, base_output_dir_str, cli_args):
    test_all_speakers_flag = cli_args.test_all_speakers
    logger.info(f"--- Starting Test for All Models ({'All Configured Speakers/Voices' if test_all_speakers_flag else 'Default Speakers/Voices Only'}) ---")
    logger.info(f"Input text: \"{text_to_synthesize[:100]}...\"")
    base_output_dir = Path(base_output_dir_str)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {base_output_dir.resolve()}")
    logger.info("------------------------------------")

    benchmark_results = [] 

    for model_id, config_entry_original in GERMAN_TTS_MODELS.items():
        handler_key = config_entry_original.get("handler_function_key", model_id)
        handler_func = ALL_HANDLERS.get(handler_key)
        
        current_model_status = "SKIPPED (No Handler)"
        current_gen_time_sec = None
        current_file_size_bytes = None
        current_audio_duration_sec = None
        current_output_path = None
        current_voice_id_tested = "N/A"


        if not handler_func:
            logger.warning(f"\n>>> No handler found for Model ID: {model_id} (handler key: {handler_key}). Skipping. <<<")
            benchmark_results.append({
                "model_id": model_id, "voice_id": current_voice_id_tested, "status": current_model_status,
                "gen_time_sec": "N/A", "file_size_bytes": "N/A",
                "audio_duration_sec": "N/A", "output_file": "N/A"
            })
            logger.info("------------------------------------")
            continue
            
        current_config_for_handler = _apply_cli_overrides_to_config(config_entry_original, model_id, cli_args)
        
        if model_id == "orpheus_lm_studio":
            current_config_for_handler["api_url"] = cli_args.lm_studio_api_url
            if cli_args.gguf_model_name_in_api:
                current_config_for_handler["gguf_model_name_in_api"] = cli_args.gguf_model_name_in_api
        elif model_id == "orpheus_ollama":
            current_config_for_handler["api_url"] = cli_args.ollama_api_url
            if cli_args.ollama_model_name:
                current_config_for_handler["ollama_model_name"] = cli_args.ollama_model_name
        
        voices_to_test_this_run = []
        if test_all_speakers_flag:
            if current_config_for_handler.get("available_voices"):
                voices_to_test_this_run.extend(current_config_for_handler.get("available_voices"))
            if "oute" in model_id and current_config_for_handler.get("test_default_speakers"):
                voices_to_test_this_run.extend(current_config_for_handler.get("test_default_speakers"))
            
            if not voices_to_test_this_run:
                default_v_candidate = (
                    current_config_for_handler.get('default_voice_id') or
                    current_config_for_handler.get('default_model_path_in_repo')
                )
                if not default_v_candidate:
                    idx_val = current_config_for_handler.get('default_speaker_embedding_index')
                    if idx_val is not None: default_v_candidate = str(idx_val)
                    else:
                        idx_val_speaker = current_config_for_handler.get('default_speaker_id')
                        if idx_val_speaker is not None: default_v_candidate = str(idx_val_speaker)
                
                if default_v_candidate and str(default_v_candidate).strip():
                    voices_to_test_this_run.append(str(default_v_candidate))
                elif model_id.startswith("coqui_tts") and current_config_for_handler.get("default_coqui_speaker") is None and current_config_for_handler.get("available_voices") == ["default_speaker"]:
                    voices_to_test_this_run.append("default_speaker")
                    logger.info(f"Coqui single-speaker model '{model_id}' for --test-all-speakers, using placeholder 'default_speaker'.")


        else: 
            default_v_to_add = None 
            std_default_keys = ['default_voice_id', 'default_model_path_in_repo']
            for key in std_default_keys:
                val = current_config_for_handler.get(key)
                if val and str(val).strip(): 
                    default_v_to_add = str(val)
                    logger.debug(f"Model '{model_id}': Found default via '{key}': {default_v_to_add}")
                    break
            
            if not default_v_to_add and model_id.startswith("coqui_tts"):
                coqui_default_speaker_id = current_config_for_handler.get('default_coqui_speaker')
                if coqui_default_speaker_id and str(coqui_default_speaker_id).strip():
                    default_v_to_add = str(coqui_default_speaker_id)
                    logger.info(f"Model '{model_id}': Using 'default_coqui_speaker': {default_v_to_add} for default test run.")
            
            if not default_v_to_add:
                idx_val_embed = current_config_for_handler.get('default_speaker_embedding_index')
                if idx_val_embed is not None: 
                    default_v_to_add = str(idx_val_embed)
                    logger.debug(f"Model '{model_id}': Found default via 'default_speaker_embedding_index': {default_v_to_add}")
                else:
                    idx_val_speaker = current_config_for_handler.get('default_speaker_id')
                    if idx_val_speaker is not None: 
                        default_v_to_add = str(idx_val_speaker)
                        logger.debug(f"Model '{model_id}': Found default via 'default_speaker_id': {default_v_to_add}")
            
            if not default_v_to_add and \
                model_id.startswith("coqui_tts") and \
                current_config_for_handler.get("default_coqui_speaker") is None: 
                if current_config_for_handler.get("available_voices") == ["default_speaker"]:
                    default_v_to_add = "default_speaker" 
                    logger.info(f"Coqui single-speaker model '{model_id}': Using placeholder 'default_speaker' for default test run.")
            
            if default_v_to_add and str(default_v_to_add).strip(): 
                voices_to_test_this_run.append(str(default_v_to_add))
            elif not voices_to_test_this_run: 
                    logger.debug(f"Model '{model_id}': No default voice/speaker could be determined for --test-all mode based on its config keys.")
        
        
        voices_to_test_this_run = [v for v in voices_to_test_this_run if v is not None and str(v).strip()] 
        seen_voices = set()
        unique_voices_to_test = []
        if voices_to_test_this_run: 
            unique_voices_to_test = [v for v in voices_to_test_this_run if not (str(v) in seen_voices or seen_voices.add(str(v)))]

        if not unique_voices_to_test:
            current_model_status = "SKIPPED (No Voice)"
            logger.info(f"\n>>> Skipping Model: {model_id} (No voices to test configured/found for this mode) <<<")
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
            output_filename = base_output_dir / f"test_output_{model_id.replace('/', '_').replace(':','-')}{speaker_suffix_for_file}{output_suffix}"
            current_output_path = output_filename

            logger.info(f"\n>>> Testing Model: {model_id} (Voice/Speaker: {voice_id_for_test}) <<<")
            
            start_time = time.time()
            try:
                handler_func(
                    current_config_for_handler,
                    text_to_synthesize,
                    str(voice_id_for_test),
                    cli_args.model_params,
                    str(output_filename),
                    False 
                )
                current_gen_time_sec = time.time() - start_time

                if output_filename.exists() and output_filename.stat().st_size > 100: 
                    current_model_status = "SUCCESS"
                    logger.info(f"SUCCESS: Output for {model_id} (Voice: {voice_id_for_test}) saved to {output_filename}")
                    current_file_size_bytes = output_filename.stat().st_size
                    try:
                        if output_filename.suffix.lower() == ".wav" and SOUNDFILE_FOR_BENCHMARK:
                            data, samplerate = sf.read(str(output_filename))
                            if samplerate > 0:
                                current_audio_duration_sec = len(data) / samplerate
                        elif output_filename.suffix.lower() == ".mp3" and PYDUB_FOR_BENCHMARK:
                            audio_seg = AudioSegment.from_file(str(output_filename))
                            current_audio_duration_sec = len(audio_seg) / 1000.0
                    except Exception as e_dur:
                        logger.warning(f"Could not determine audio duration for {output_filename}: {e_dur}")
                else:
                    current_model_status = "FAIL (No/Small File)"
                    logger.warning(f"NOTE: Synthesis for {model_id} (Voice: {voice_id_for_test}) ran. Output file '{output_filename}' not created or is empty/too small.")
            
            except Exception as e_test_model:
                if current_gen_time_sec is None: 
                    current_gen_time_sec = time.time() - start_time
                current_model_status = "ERROR"
                logger.error(f"ERROR: Testing {model_id} (Voice: {voice_id_for_test}) failed: {e_test_model}", exc_info=True)


            benchmark_results.append({
                "model_id": model_id, 
                "voice_id": current_voice_id_tested, 
                "status": current_model_status,
                "gen_time_sec": f"{current_gen_time_sec:.2f}s" if current_gen_time_sec is not None else "N/A",
                "file_size_bytes": current_file_size_bytes if current_file_size_bytes is not None else "N/A",
                "audio_duration_sec": f"{current_audio_duration_sec:.2f}s" if current_audio_duration_sec is not None else "N/A",
                "output_file": str(current_output_path.name) if current_output_path else "N/A"
            })
            
            if test_all_speakers_flag and len(unique_voices_to_test) > 1 and voice_idx < len(unique_voices_to_test) -1 :
                logger.info("---") 
        logger.info("------------------------------------")

    logger.info("--- Test for All Models Finished ---")
    logger.info("\n--- BENCHMARK SUMMARY ---")
    if benchmark_results:
        col_model = max(len("Model ID"), max(len(r["model_id"]) for r in benchmark_results if r["model_id"]))
        col_voice = max(len("Voice/Speaker"), max(len(str(r["voice_id"])) for r in benchmark_results if r["voice_id"]))
        col_status = max(len("Status"), max(len(r["status"]) for r in benchmark_results if r["status"]))
        col_gentime = max(len("Gen Time"), max(len(str(r["gen_time_sec"])) for r in benchmark_results if r["gen_time_sec"]))
        col_size = max(len("Size (Bytes)"), max(len(str(r["file_size_bytes"])) for r in benchmark_results if r["file_size_bytes"]))
        col_duration = max(len("Audio (s)"), max(len(str(r["audio_duration_sec"])) for r in benchmark_results if r["audio_duration_sec"]))
        col_file = max(len("File"), max(len(r["output_file"]) for r in benchmark_results if r["output_file"]))

        header_parts = [
            f" {'Model ID'.ljust(col_model)} ",
            f" {'Voice/Speaker'.ljust(col_voice)} ",
            f" {'Status'.ljust(col_status)} ",
            f" {'Gen Time'.rjust(col_gentime)} ",
            f" {'Size (Bytes)'.rjust(col_size)} ",
            f" {'Audio (s)'.rjust(col_duration)} ",
            f" {'File'.ljust(col_file)} "
        ]
        header = f"|{'|'.join(header_parts)}|"
        
        sep_parts = [
            f"{'-'*(col_model+2)}",
            f"{'-'*(col_voice+2)}",
            f"{'-'*(col_status+2)}",
            f"{'-'*(col_gentime+2)}",
            f"{'-'*(col_size+2)}",
            f"{'-'*(col_duration+2)}",
            f"{'-'*(col_file+2)}"
        ]
        separator = f"|{'|'.join(sep_parts)}|"
        
        logger.info(separator)
        logger.info(header)
        logger.info(separator)
        
        for r in benchmark_results:
            row_parts = [
                f" {r['model_id'].ljust(col_model)} ",
                f" {str(r['voice_id']).ljust(col_voice)} ",
                f" {r['status'].ljust(col_status)} ",
                f" {str(r['gen_time_sec']).rjust(col_gentime)} ",
                f" {str(r['file_size_bytes']).rjust(col_size)} ",
                f" {str(r['audio_duration_sec']).rjust(col_duration)} ",
                f" {r['output_file'].ljust(col_file)} "
            ]
            logger.info(f"|{'|'.join(row_parts)}|")
        logger.info(separator)
    else:
        logger.info("No benchmark results to display.")


def run_synthesis(args):
    text_to_synthesize = get_text_from_input(args.input_text, args.input_file)
    if not text_to_synthesize:
        if not args.input_text and not args.input_file: 
            logger.error("No input provided. Use --input-text or --input-file for synthesis.")
        return 

    text_to_synthesize = text_to_synthesize[:3000]

    if not args.model_id: 
        logger.error("--model-id is required for single synthesis.")
        return

    model_config_base = GERMAN_TTS_MODELS.get(args.model_id)
    if not model_config_base:
        logger.error(f"Invalid model ID: {args.model_id}. Use --list-models to see available IDs.")
        return

    logger.info(f"Synthesizing with: {args.model_id}")
    logger.info(f"Input (start): '{text_to_synthesize[:70]}...'")

    current_config_for_handler = _apply_cli_overrides_to_config(model_config_base, args.model_id, args)

    if args.model_id == "orpheus_lm_studio":
        current_config_for_handler["api_url"] = args.lm_studio_api_url
        if args.gguf_model_name_in_api:
            current_config_for_handler["gguf_model_name_in_api"] = args.gguf_model_name_in_api
    elif args.model_id == "orpheus_ollama":
        current_config_for_handler["api_url"] = args.ollama_api_url
        if args.ollama_model_name:
            current_config_for_handler["ollama_model_name"] = args.ollama_model_name
        if "USER MUST SET" in current_config_for_handler.get("ollama_model_name","") or not current_config_for_handler.get("ollama_model_name"):
            logger.error(f"For {args.model_id}, Ollama model name is not set. Use --ollama-model-name or set in config.")
            return
    
    effective_voice_id = args.german_voice_id
    if not effective_voice_id: 
        default_v = current_config_for_handler.get('default_voice_id') or \
                    current_config_for_handler.get('default_model_path_in_repo') or \
                    str(current_config_for_handler.get('default_speaker_embedding_index', '')) or \
                    str(current_config_for_handler.get('default_speaker_id', ''))
        effective_voice_id = default_v if (isinstance(default_v, Path) or (isinstance(default_v, str) and default_v.strip())) else None


    if not effective_voice_id and not (args.model_id.startswith("coqui_tts") and current_config_for_handler.get("default_coqui_speaker") is None):
        logger.error(f"No voice ID specified and no default could be determined for model {args.model_id}.")
        return

    handler_key = current_config_for_handler.get("handler_function_key", args.model_id)
    handler_func = ALL_HANDLERS.get(handler_key)

    if handler_func:
        try:
            handler_func(
                current_config_for_handler,
                text_to_synthesize,
                str(effective_voice_id) if effective_voice_id is not None else None, 
                args.model_params,
                args.output_file,
                args.play_direct
            )
        except Exception as e_synth:
            logger.error(f"Synthesis failed for model {args.model_id}: {e_synth}", exc_info=True)
    else:
        logger.error(f"No synthesis handler function found for model ID: {args.model_id} (handler key: {handler_key})")


def main_cli_entrypoint():
    parser = argparse.ArgumentParser(
        description="CrispTTS: Modular German Text-to-Speech Synthesizer",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    action_group = parser.add_argument_group(title="Primary Actions")
    input_group = parser.add_mutually_exclusive_group(required=False) 

    action_group.add_argument("--list-models", action="store_true", help="List all configured TTS models.")
    action_group.add_argument("--voice-info", type=str, metavar="MODEL_ID", help="Display voice/speaker info for a specific MODEL_ID.")
    action_group.add_argument("--test-all", action="store_true", help="Test all models with default voices. Requires --input-text or --input-file.")
    action_group.add_argument("--test-all-speakers", action="store_true", help="Test all models with ALL configured voices. Requires --input-text or --input-file.")

    synth_group = parser.add_argument_group(title="Synthesis Options (used with --model-id or --test-all*)")
    input_group.add_argument("--input-text", type=str, help="Text to synthesize.")
    input_group.add_argument("--input-file", type=str, help="Path to input file (txt, md, html, pdf, epub).")
    
    # Ensure GERMAN_TTS_MODELS is available for choices, or handle its potential unavailability if config fails to load
    model_choices = list(GERMAN_TTS_MODELS.keys()) if GERMAN_TTS_MODELS else []
    model_choices.append(None) # Add None as a valid choice for argparse

    synth_group.add_argument("--model-id", type=str, choices=model_choices, default=None,      
                                help="Select TTS model ID. Required for single synthesis if not using an action flag.")
    synth_group.add_argument("--output-file", type=str, help="Path to save synthesized audio (for single synthesis).")
    synth_group.add_argument("--output-dir", type=str, default="tts_test_outputs", help="Directory for --test-all* outputs (default: tts_test_outputs).")
    synth_group.add_argument("--play-direct", action="store_true", help="Play audio directly after synthesis (not with --test-all*).")
    synth_group.add_argument("--german-voice-id", type=str, help="Override default voice/speaker for the selected model.")
    synth_group.add_argument("--model-params", type=str, help="JSON string of model-specific parameters (e.g., '{\"temperature\":0.7}').")
    
    parser.add_argument(
        "--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set console logging level (default: INFO)."
    )

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

    # Setup root logger AFTER parsing args for loglevel
    # This will apply to the main_mp_logger as well if it wasn't set to propagate=False
    logging.basicConfig(
        level=args.loglevel.upper(), # Set initial level from CLI
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True 
    )
    
    # Get the numeric representation of the desired log level from CLI args
    cli_numeric_log_level = getattr(logging, args.loglevel.upper(), logging.INFO)

    # Ensure all existing loggers (including root and any created by libraries)
    # are set to at least the verbosity specified by the CLI.
    # This also correctly sets the level for _main_mp_logger.
    loggers_to_update = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers_to_update.append(logging.root) # include the root logger
    
    for lgr in loggers_to_update:
        if lgr.level == 0 or lgr.level > cli_numeric_log_level: # if not set or less verbose
            lgr.setLevel(cli_numeric_log_level)
        # For the monkey patch logger, ensure its handler also respects the level
        if lgr.name == "CrispTTS.main_monkey_patch":
            for handler in lgr.handlers:
                 if handler.level == 0 or handler.level > cli_numeric_log_level:
                    handler.setLevel(cli_numeric_log_level)

    # Log the effective level
    logger.info(f"Effective logging level for CrispTTS and sub-loggers set to: {args.loglevel.upper()}")
    # Confirm the monkey_patch logger's level as well, using its effective level
    _main_mp_logger.info(f"Monkey patch logger effective level is: {logging.getLevelName(_main_mp_logger.getEffectiveLevel())}")

    if args.list_models:
        list_available_models(GERMAN_TTS_MODELS)
        return
    if args.voice_info:
        if not args.voice_info in GERMAN_TTS_MODELS:
            logger.error(f"Model ID '{args.voice_info}' for --voice-info not found. Use --list-models.")
            return
        get_voice_info(args.voice_info, GERMAN_TTS_MODELS)
        return

    text_to_process = get_text_from_input(args.input_text, args.input_file)
    if not text_to_process:
        if args.test_all or args.test_all_speakers:
            parser.error("--test-all or --test-all-speakers requires --input-text or --input-file.")
        elif args.model_id : 
            logger.error("No text input provided for synthesis via --input-text or --input-file.")
        else: 
            parser.print_help()
        return

    if args.test_all or args.test_all_speakers:
        test_text = text_to_process[:500] if len(text_to_process) > 500 else text_to_process 
        logger.info(f"--- Applying Test Mode: {'All Speakers' if args.test_all_speakers else 'Default Speaker Only'} ---")
        if args.override_main_model_repo or args.override_tokenizer_repo or args.override_vocoder_repo or args.override_speaker_embed_repo or args.override_piper_voices_repo or args.override_model_filename:
            logger.warning("CLI repo/path overrides are active and will apply to all compatible models during --test-all(-speakers). This might not be intended for all models.")
        test_all_models(test_text, args.output_dir, args)
        return

    if not args.model_id: 
        parser.error("A --model-id is required for synthesis if not using an action flag like --list-models or --test-all.")
        return
    
    run_synthesis(args)


if __name__ == "__main__":
    main_cli_entrypoint()