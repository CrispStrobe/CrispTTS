#!/usr/bin/env python3
# CrispTTS - main.py
# Main Command-Line Interface for the Text-to-Speech Synthesizer (Modularized with Overrides)

import argparse
import logging
import os
import sys
import time
from pathlib import Path

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
                    triton_module.Config = _CrispTTSDummyTritonConfig

                if not hasattr(triton_module, 'cdiv'):
                    _main_mp_logger.debug("Adding 'cdiv' attribute to vLLM's TritonPlaceholder.") # DEBUG
                    triton_module.cdiv = lambda x, y: (x + y - 1) // y

                if hasattr(triton_module, 'language'):
                    triton_lang_module = triton_module.language
                    lang_placeholder_type_name = str(type(triton_lang_module))
                    if ("vllm" in lang_placeholder_type_name
                            and "TritonLanguagePlaceholder" in lang_placeholder_type_name):
                        _main_mp_logger.debug("vLLM's TritonLanguagePlaceholder found. Patching missing attributes.") # DEBUG  # noqa: E501

                        class _DummyDtypePlaceholder:
                            def __init__(self, name_): self.name = name_
                            def __repr__(self): return f"tl.{self.name}"
                            def to(self, target_device_type_str):
                                _main_mp_logger.debug(f"DummyDtypePlaceholder {self.name}.to({target_device_type_str}) called.")  # noqa: E501
                                return self

                        dtypes_to_add = {name: _DummyDtypePlaceholder(name) for name in
                                         ["int1", "int8", "int16", "int32", "uint8", "uint16",
                                          "uint32", "uint64", "float8e4nv", "float8e5",
                                          "float16", "bfloat16", "float32", "float64"]}

                        for dtype_name, dtype_obj in dtypes_to_add.items():
                            if not hasattr(triton_lang_module, dtype_name):
                                _main_mp_logger.debug(f"Adding dtype '{dtype_name}' to TritonLanguagePlaceholder.") # DEBUG  # noqa: E501
                                setattr(triton_lang_module, dtype_name, dtype_obj)

                        if getattr(triton_lang_module, 'constexpr', 'NOT_SET') is None:
                            _main_mp_logger.debug("Patching 'constexpr' in TritonLanguagePlaceholder to be an identity function.") # DEBUG  # noqa: E501
                            triton_lang_module.constexpr = lambda x: x

                        current_dtype_attr = getattr(triton_lang_module, 'dtype', 'NOT_SET')
                        if current_dtype_attr is None or not callable(current_dtype_attr):
                             _main_mp_logger.debug("Patching 'dtype' in TritonLanguagePlaceholder to be a dummy factory.") # DEBUG  # noqa: E501
                             triton_lang_module.dtype = lambda name_str_or_obj: dtypes_to_add.get(str(name_str_or_obj),
                                 _DummyDtypePlaceholder(str(name_str_or_obj))) if isinstance(name_str_or_obj,
                                 str) else name_str_or_obj

                        dummy_attrs = {
                            'PROGRAM_ID': lambda axis: 0, 'make_block_ptr': lambda *a, **kw: None,
                            'load': lambda *a, **kw: None, 'store': lambda *a, **kw: None,
                            'dot': lambda *a, **kw: None
                        }
                        for attr_name, attr_val in dummy_attrs.items():
                            if not hasattr(triton_lang_module, attr_name):
                                _main_mp_logger.debug(f"Adding dummy '{attr_name}' to TritonLanguagePlaceholder.") # DEBUG  # noqa: E501
                                setattr(triton_lang_module, attr_name, attr_val)
                    else:
                        _main_mp_logger.debug(f"sys.modules['triton'].language (type: {lang_placeholder_type_name}) is not vLLM's placeholder.")  # noqa: E501
                else:
                     _main_mp_logger.debug("TritonPlaceholder does not have 'language' attribute.") # DEBUG

        if patch_applied_summary:
            _main_mp_logger.info(f"Triton placeholder patch: {patch_applied_summary[0]} (see DEBUG for details if enabled)")  # noqa: E501
        else:
            _main_mp_logger.debug("No Triton placeholder patch applied.")

    except Exception as e_mp:
        print(f"CRITICAL MONKEY PATCH ERROR: {e_mp}", file=sys.stderr)
        _main_mp_logger.error(f"Error during Triton monkey patching: {e_mp}", exc_info=True)

ALL_HANDLERS = None
_HANDLERS_LOADED = False

from config import GERMAN_TTS_MODELS, LM_STUDIO_API_URL_DEFAULT, OLLAMA_API_URL_DEFAULT  # noqa: E402
from utils import PYDUB_AVAILABLE as UTILS_PYDUB_AVAILABLE  # noqa: E402
from utils import SOUNDFILE_AVAILABLE as UTILS_SOUNDFILE_AVAILABLE  # noqa: E402
from utils import (  # noqa: E402
    get_text_from_input,
    get_voice_info,
    list_available_models,
    resolve_written_output,
)

_apply_triton_config_monkey_patch_for_vllm()

logger = logging.getLogger("CrispTTS.main")

def _load_handlers_if_needed():
    global ALL_HANDLERS, _HANDLERS_LOADED
    if not _HANDLERS_LOADED:
        logger.debug("Loading lazy handler registry...")
        try:
            from handlers import ALL_HANDLERS as lazy_registry
            ALL_HANDLERS = lazy_registry
            _HANDLERS_LOADED = True
            logger.info("Handler registry loaded (lazy — handlers import on first use).")
        except ImportError as e:
            logger.critical("Failed to import handlers package: %s", e, exc_info=True)
        except Exception as e_load:
            logger.critical("Unexpected error loading handler registry: %s", e_load, exc_info=True)
    return ALL_HANDLERS

def _apply_cli_overrides_to_config(model_config_dict, model_id_key, cli_args):
    config_to_modify = model_config_dict.copy()
    if cli_args.override_main_model_repo:
        repo_override = cli_args.override_main_model_repo
        updated = False
        if model_id_key in ["orpheus_lex_au", "orpheus_sauerkraut"] and "model_repo_id" in config_to_modify:
            config_to_modify["model_repo_id"] = repo_override
            updated = True
        elif (model_id_key == "piper_local"
                and "piper_voice_repo_id" in config_to_modify and not cli_args.override_piper_voices_repo):
            config_to_modify["piper_voice_repo_id"] = repo_override
            updated = True
        elif model_id_key == "oute_hf" and "onnx_repo_id" in config_to_modify:
            config_to_modify["onnx_repo_id"] = repo_override
            updated = True
        elif model_id_key.startswith("mlx_audio") and "mlx_model_path" in config_to_modify:
            config_to_modify["mlx_model_path"] = repo_override
            updated = True
        elif model_id_key == "speecht5_german_transformers" and "model_id" in config_to_modify:
            config_to_modify["model_id"] = repo_override
            updated = True
        elif model_id_key == "fastpitch_german_nemo" and "spectrogram_model_repo_id" in config_to_modify:
            config_to_modify["spectrogram_model_repo_id"] = repo_override
            updated = True
        elif model_id_key == "orpheus_kartoffel_natural" and "model_repo_id" in config_to_modify: # Added for Kartoffel
            config_to_modify["model_repo_id"] = repo_override
            updated = True


        if updated:
            logger.info(f"Overriding main model repo for '{model_id_key}' to: {repo_override}")
        elif model_id_key not in ["edge", "orpheus_lm_studio", "orpheus_ollama"]:
            logger.debug(f"No primary repo key found to override for '{model_id_key}' with '{repo_override}'. Check config keys.")  # noqa: E501

    if cli_args.override_model_filename:
        fn_override = cli_args.override_model_filename
        updated = False
        if model_id_key in ["orpheus_lex_au", "orpheus_sauerkraut"] and "model_filename" in config_to_modify:
            config_to_modify["model_filename"] = fn_override
            updated = True
        elif model_id_key == "fastpitch_german_nemo" and "spectrogram_model_filename" in config_to_modify:
            config_to_modify["spectrogram_model_filename"] = fn_override
            updated = True
        if updated:
            logger.info(f"Overriding model filename for '{model_id_key}' to: {fn_override}")

    if cli_args.override_tokenizer_repo:
        tok_override = cli_args.override_tokenizer_repo
        if ("oute_hf" == model_id_key or "oute_llamacpp" == model_id_key) and "tokenizer_path" in config_to_modify:
            config_to_modify["tokenizer_path"] = tok_override
            logger.info(f"Overriding 'tokenizer_path' for '{model_id_key}' to: {tok_override}")
        elif (model_id_key == "orpheus_kartoffel_natural"
                and "tokenizer_repo_id" in config_to_modify):  # Added for Kartoffel
            config_to_modify["tokenizer_repo_id"] = tok_override
            logger.info(f"Overriding 'tokenizer_repo_id' for '{model_id_key}' to: {tok_override}")
        # Ensure key "tokenizer_path_for_mlx_outetts" exists if this model ID is used, or handle more gracefully
        elif "mlx_audio_outetts_clone" == model_id_key and "tokenizer_path_for_mlx_outetts" in config_to_modify:
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
    logger.info(f"--- Starting Test for All Models ({'All Configured Speakers/Voices' if test_all_speakers_flag else 'Default Speakers/Voices Only'}) ---")  # noqa: E501
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
            logger.warning(f"\n>>> No handler found for Model ID: {model_id} (handler key: {handler_key}). Skipping. <<<")  # noqa: E501
            benchmark_results.append({ "model_id": model_id, "voice_id": current_voice_id_tested,
                "status": current_model_status, "gen_time_sec": "N/A", "file_size_bytes": "N/A",
                "audio_duration_sec": "N/A", "output_file": "N/A"})
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
                    if idx_val is not None:
                        default_v_candidate = str(idx_val)
                    else:
                        idx_val_speaker = current_config_for_handler.get('default_speaker_id')
                        if idx_val_speaker is not None:
                            default_v_candidate = str(idx_val_speaker)

                if default_v_candidate and str(default_v_candidate).strip():
                    voices_to_test_this_run.append(str(default_v_candidate))
                # Fallback for Coqui single-speaker models if still nothing found
                elif (model_id.startswith("coqui_")
                        and current_config_for_handler.get("default_coqui_speaker") is None
                        and current_config_for_handler.get("available_voices") == ["default_speaker"]):
                    voices_to_test_this_run.append("default_speaker")
                    logger.info(f"Coqui single-speaker model '{model_id}' for --test-all-speakers, using placeholder 'default_speaker'.")  # noqa: E501
        else: # Not test_all_speakers_flag, so get only the single default voice OR handle zero-shot
            default_v_to_add = None

            # NEW: Check for explicit zero-shot configuration for default test mode
            # A model is considered zero-shot for this purpose if its default_voice_id is None
            # AND it has no specific available_voices listed (indicating it doesn't rely on a predefined voice set)
            # AND it's not a type of model that inherently requires a speaker/voice even in default (e.g. some Coqui models)  # noqa: E501
            is_primarily_zero_shot_type = (
                current_config_for_handler.get('default_voice_id') is None and
                not current_config_for_handler.get('available_voices')
            )
            # Add specific model IDs here if they are zero-shot but might have other default fields that confuse the logic below  # noqa: E501
            # For example, if a model is zero-shot but happens to have a 'default_speaker_id' for some other purpose.
            if model_id in ["llasa_hybrid_de_zeroshot", "llasa_german_transformers_zeroshot",
                "llasa_multilingual_hf_zeroshot"]: # Be explicit for known zero-shot LLaSA
                is_primarily_zero_shot_type = True


            if is_primarily_zero_shot_type:
                logger.info(f"Model '{model_id}': Identified as zero-shot or configured for such a test. Proceeding with 'None' as voice_id for default test.")  # noqa: E501
                voices_to_test_this_run.append(None) # Use None to signify zero-shot for the handler
            else:
                # Original logic to find a default_v_to_add
                # Priority 1: Coqui-specific default speaker ID
                if model_id.startswith("coqui_"):
                    coqui_default_speaker_id = current_config_for_handler.get('default_coqui_speaker')
                    if coqui_default_speaker_id and str(coqui_default_speaker_id).strip():
                        default_v_to_add = str(coqui_default_speaker_id)
                        logger.debug(f"Model '{model_id}': Using 'default_coqui_speaker': {default_v_to_add} for default test run.")  # noqa: E501

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
                        logger.debug(f"Model '{model_id}': Found default via 'default_speaker_embedding_index': {default_v_to_add}")  # noqa: E501
                    else:
                        idx_val_speaker = current_config_for_handler.get('default_speaker_id')
                        if idx_val_speaker is not None: # Check for None explicitly for numeric 0
                            default_v_to_add = str(idx_val_speaker)
                            logger.debug(f"Model '{model_id}': Found default via 'default_speaker_id': {default_v_to_add}")  # noqa: E501

                # Priority 4: Fallback for Coqui single-speaker models using "default_speaker" placeholder
                if not default_v_to_add and \
                   model_id.startswith("coqui_") and \
                   current_config_for_handler.get("default_coqui_speaker") is None and \
                   current_config_for_handler.get("available_voices") == ["default_speaker"]:
                    default_v_to_add = "default_speaker" # This specific string might be handled by Coqui handler
                    logger.info(f"Coqui single-speaker model '{model_id}': Using placeholder 'default_speaker' for default test run.")  # noqa: E501

                if default_v_to_add is not None and (not isinstance(default_v_to_add,
                    str) or str(default_v_to_add).strip()):
                    voices_to_test_this_run.append(str(default_v_to_add))
                elif not voices_to_test_this_run: # Only log if it's still empty and not identified as zero-shot
                    logger.debug(f"Model '{model_id}': No default voice/speaker could be determined for default speaker test mode (and not flagged as zero-shot for this test).")  # noqa: E501

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
            if model_id == "edge":
                output_suffix = ".mp3"
            # Ensure model_id in filename is also sanitized for special characters like '/'
            sanitized_model_id_for_filename = model_id.replace('/', '_').replace(':','-')
            output_filename = base_output_dir / f"test_output_{sanitized_model_id_for_filename}{speaker_suffix_for_file}{output_suffix}"  # noqa: E501
            current_output_path = output_filename

            logger.info(f"\n>>> Testing Model: {model_id} (Voice/Speaker: {voice_id_for_test}) <<<")

            # --- Voice-cloning consent gate (same rules as real synthesis) ---
            _is_voice_cloning = False
            _needs_disclosure = False
            try:
                from watermark import log_consent_attestation as _test_log_consent
                from watermark import requires_consent as _test_requires_consent
                from watermark import requires_spoken_disclosure as _test_needs_disclosure
                from watermark import resolve_speaker_identity as _test_speaker_identity
                _voice_str = str(voice_id_for_test) if voice_id_for_test else None
                _is_voice_cloning = _test_requires_consent(
                    model_id, handler_key, _voice_str,
                    model_config=current_config_for_handler)
                _needs_disclosure = _test_needs_disclosure(
                    _is_voice_cloning,
                    _test_speaker_identity(current_config_for_handler,
                                           getattr(cli_args, 'speaker_identity', None)),
                    model_id=model_id)
                if _is_voice_cloning and not getattr(cli_args, 'i_have_rights', False):
                    current_model_status = "SKIPPED (Consent Required)"
                    logger.info(f"Skipping voice-cloning model '{model_id}': --i-have-rights not set.")
                    benchmark_results.append({"model_id": model_id, "voice_id": current_voice_id_tested,
                        "status": current_model_status, "gen_time_sec": "N/A", "file_size_bytes": "N/A",
                        "audio_duration_sec": "N/A", "output_file": "N/A"})
                    continue
                if _is_voice_cloning:
                    # The attestation is logged where it is honoured, not only
                    # in run_synthesis — otherwise --test-all clones voices
                    # with no entry in the audit trail.
                    _test_log_consent(model_id, _voice_str,
                                      source="CLI --i-have-rights flag (--test-all)")
            except ImportError:
                # Fails closed, as in run_synthesis: an unknown cloning status
                # is treated as cloning, not as permission.
                logger.error("watermark module unavailable — cannot check the consent "
                             "gate; skipping '%s'.", model_id)
                if not os.environ.get("CRISPTTS_ALLOW_UNMARKED"):
                    benchmark_results.append({"model_id": model_id, "voice_id": current_voice_id_tested,
                        "status": "SKIPPED (Consent Gate Unavailable)", "gen_time_sec": "N/A",
                        "file_size_bytes": "N/A", "audio_duration_sec": "N/A", "output_file": "N/A"})
                    continue

            # --- Marking preflight, before any model is loaded ---
            _test_marking_policy = None
            try:
                from watermark import MarkingError as _TestMarkingError
                from watermark import preflight_marking as _test_preflight
                try:
                    _test_marking_policy = _test_preflight(
                        str(output_filename),
                        handler_key=handler_key,
                        no_watermark=getattr(cli_args, 'no_watermark', False),
                        allow_unmarked=getattr(cli_args, 'allow_unmarked', False),
                        responsibility_accepted=getattr(cli_args, 'accept_marking_responsibility', False),
                        no_spoken_disclaimer=getattr(cli_args, 'no_spoken_disclaimer', False),
                        c2pa_cert=getattr(cli_args, 'c2pa_cert', None),
                        c2pa_key=getattr(cli_args, 'c2pa_key', None),
                    )
                except _TestMarkingError as e_test_pre:
                    logger.error("Skipping '%s': %s", model_id, e_test_pre)
                    benchmark_results.append({"model_id": model_id, "voice_id": current_voice_id_tested,
                        "status": "SKIPPED (Unmarkable)", "gen_time_sec": "N/A",
                        "file_size_bytes": "N/A", "audio_duration_sec": "N/A", "output_file": "N/A"})
                    continue
            except ImportError:
                if not os.environ.get("CRISPTTS_ALLOW_UNMARKED"):
                    logger.error("watermark module unavailable — refusing to synthesize "
                                 "unmarkable audio in --test-all.")
                    return benchmark_results

            start_time_model_test = time.time()
            current_gen_time_sec = None
            current_file_size_bytes = None
            current_audio_duration_sec = None
            try:
                handler_func(current_config_for_handler, text_to_synthesize, str(voice_id_for_test),
                    cli_args.model_params, str(output_filename), False)
                current_gen_time_sec = time.time() - start_time_model_test

                # Follow the audio, as run_synthesis does: a handler that wrote
                # its own container would otherwise leave an unmarked file on
                # disk and be recorded as having produced nothing.
                _written = resolve_written_output(str(output_filename), since=start_time_model_test)
                if _written and _written != str(output_filename):
                    output_filename = Path(_written)
                    current_output_path = output_filename

                if output_filename.exists() and output_filename.stat().st_size > 100:
                    current_model_status = "SUCCESS"
                    logger.info(f"SUCCESS: Output for {model_id} (Voice: {voice_id_for_test}) saved to {output_filename}")  # noqa: E501

                    # Spoken AI disclosure, then marking — the same sequence and
                    # the same fail-closed rules as run_synthesis. --test-all
                    # writes real cloned audio to disk, so it gets the real
                    # obligations, not a relaxed subset.
                    try:
                        from watermark import DisclosureError, MarkingError, mark_audio_file
                        if (_needs_disclosure
                                and not getattr(cli_args, 'no_spoken_disclaimer', False)):
                            from watermark import prepend_disclaimer_file
                            prepend_disclaimer_file(
                                str(output_filename),
                                language=current_config_for_handler.get("language"),
                                disclosure_lang=getattr(cli_args, 'disclosure_lang', None))
                        mark_audio_file(
                            str(output_filename),
                            handler_key=handler_key,
                            allow_unmarked=True if getattr(cli_args, 'allow_unmarked', False) else None,
                            c2pa_cert=getattr(cli_args, 'c2pa_cert', None),
                            c2pa_key=getattr(cli_args, 'c2pa_key', None),
                            policy=_test_marking_policy,
                            model_id=model_id,
                        )
                    except DisclosureError as e_td:
                        logger.error("Test output has no AI disclosure: %s", e_td)
                        current_model_status = "FAILED (No Disclosure)"
                        try:
                            output_filename.unlink()
                        except OSError:
                            pass
                    except MarkingError as e_tw:
                        logger.error("Test output could not be marked: %s", e_tw)
                        current_model_status = "FAILED (Unmarked)"
                        try:
                            output_filename.unlink()
                        except OSError:
                            pass
                    except ImportError:
                        if os.environ.get("CRISPTTS_ALLOW_UNMARKED"):
                            logger.warning("watermark module unavailable — test output is UNMARKED.")
                        else:
                            logger.error("watermark module unavailable — discarding unmarked "
                                         "test output for %s.", model_id)
                            current_model_status = "FAILED (Unmarked)"
                            try:
                                output_filename.unlink()
                            except OSError:
                                pass

                    if not output_filename.exists():
                        # Discarded because it could not be marked.
                        benchmark_results.append({"model_id": model_id, "voice_id": current_voice_id_tested,
                            "status": current_model_status, "gen_time_sec": f"{current_gen_time_sec:.2f}",
                            "file_size_bytes": "N/A", "audio_duration_sec": "N/A", "output_file": "N/A"})
                        continue

                    current_file_size_bytes = output_filename.stat().st_size
                    try:
                        if output_filename.suffix.lower() == ".wav" and soundfile_for_benchmark:
                            data, samplerate = soundfile_for_benchmark.read(str(output_filename))
                            if samplerate > 0:
                                current_audio_duration_sec = len(data) / samplerate
                        elif (output_filename.suffix.lower() == ".mp3"
                                and pydub_for_benchmark and AudioSegment_benchmark_imp):
                            audio_seg = AudioSegment_benchmark_imp.from_file(str(output_filename))
                            current_audio_duration_sec = len(audio_seg) / 1000.0
                    except Exception as e_dur:
                        logger.warning(f"Could not determine audio duration for {output_filename}: {e_dur}")
                else:
                    current_model_status = "FAIL (No/Small File)"
                    logger.warning(f"NOTE: Synthesis for {model_id} (Voice: {voice_id_for_test}) ran. Output file '{output_filename}' not created or is empty/too small.")  # noqa: E501
            except Exception as e_test_model:
                if current_gen_time_sec is None:
                    current_gen_time_sec = time.time() - start_time_model_test
                current_model_status = "ERROR"
                logger.error(f"ERROR: Testing {model_id} (Voice: {voice_id_for_test}) failed: {e_test_model}",
                    exc_info=True)

            benchmark_results.append({ "model_id": model_id, "voice_id": current_voice_id_tested, "status": current_model_status, "gen_time_sec": f"{current_gen_time_sec:.2f}s" if current_gen_time_sec is not None else "N/A", "file_size_bytes": current_file_size_bytes if current_file_size_bytes is not None else "N/A", "audio_duration_sec": f"{current_audio_duration_sec:.2f}s" if current_audio_duration_sec is not None else "N/A", "output_file": str(current_output_path.name) if current_output_path else "N/A" })  # noqa: E501
            if test_all_speakers_flag and len(unique_voices_to_test) > 1 and voice_idx < len(unique_voices_to_test) -1 :
                logger.info("---")
        logger.info("------------------------------------")

    logger.info("--- Test for All Models Finished ---")
    logger.info("\n--- BENCHMARK SUMMARY ---")
    if benchmark_results:
        # Calculate column widths
        def max_len(key_str, default_len):
            return max(default_len, max(len(str(r[key_str])) for r in benchmark_results if r[key_str] is not None and str(r[key_str]).strip() != ""))  # noqa: E501

        col_model = max_len("model_id", len("Model ID"))
        col_voice = max_len("voice_id", len("Voice/Speaker"))
        col_status = max_len("status", len("Status"))
        col_gentime = max_len("gen_time_sec", len("Gen Time"))
        col_size = max_len("file_size_bytes", len("Size (Bytes)"))
        col_duration = max_len("audio_duration_sec", len("Audio (s)"))
        col_file = max_len("output_file", len("File"))

        header_parts = [f" {'Model ID'.ljust(col_model)} ", f" {'Voice/Speaker'.ljust(col_voice)} ",
            f" {'Status'.ljust(col_status)} ", f" {'Gen Time'.rjust(col_gentime)} ",
            f" {'Size (Bytes)'.rjust(col_size)} ", f" {'Audio (s)'.rjust(col_duration)} ",
            f" {'File'.ljust(col_file)} "]
        header = f"|{'|'.join(header_parts)}|"
        sep_parts = [f"{'-'*(col_model+2)}", f"{'-'*(col_voice+2)}", f"{'-'*(col_status+2)}", f"{'-'*(col_gentime+2)}",
            f"{'-'*(col_size+2)}", f"{'-'*(col_duration+2)}", f"{'-'*(col_file+2)}"]
        separator = f"|{'|'.join(sep_parts)}|"
        logger.info(separator)
        logger.info(header)
        logger.info(separator)

        for r in benchmark_results:
            row_parts = [f" {str(r['model_id']).ljust(col_model)} ", f" {str(r['voice_id']).ljust(col_voice)} ",
                f" {str(r['status']).ljust(col_status)} ", f" {str(r['gen_time_sec']).rjust(col_gentime)} ",
                f" {str(r['file_size_bytes']).rjust(col_size)} ",
                f" {str(r['audio_duration_sec']).rjust(col_duration)} ", f" {str(r['output_file']).ljust(col_file)} "]
            logger.info(f"|{'|'.join(row_parts)}|")
        logger.info(separator)
    else:
        logger.info("No benchmark results to display.")

def _discard_output(args, temp_play_file=None):
    """Delete synthesized audio that could not be AI-provenance marked.

    Under EU AI Act Art. 50(2) synthetic audio must carry a machine-readable
    mark, so an output we failed to mark must not be left on disk where it
    could be mistaken for a compliant file.
    """
    for path in {getattr(args, "output_file", None), temp_play_file}:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
                logger.info("Discarded unmarked output: %s", path)
            except OSError as e:
                logger.warning("Could not discard unmarked output %s: %s", path, e)
    if temp_play_file:
        args.output_file = None


def _read_pcm(path):
    """Read any handler-written container as mono float32 PCM.

    soundfile covers WAV/FLAC and, with a recent libsndfile, MP3; pydub covers
    the rest. Returns ``(None, None)`` if neither can read it.
    """
    try:
        import soundfile as sf_read
        data, sr = sf_read.read(path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        return data, sr
    except Exception as e_sf:
        logger.debug("soundfile could not read %s (%s); trying pydub.", path, e_sf)

    try:
        import numpy as np
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path).set_channels(1)
        data = np.array(seg.get_array_of_samples(), dtype=np.float32)
        peak = float(1 << (8 * seg.sample_width - 1))
        return data / peak, seg.frame_rate
    except Exception as e_pd:
        logger.warning("Could not read audio from %s: %s", path, e_pd)
        return None, None


def _synthesize_ssml_segments(args, segments, output_path):
    """Render SSML segments individually and crossfade them into one file.

    Called in place of the handler, so the combined result carries on through
    the shared disclosure and marking block rather than bypassing it.

    Segments are rendered with ``_ssml_segment`` set, which suppresses the
    per-segment spoken disclosure and the per-segment marking. Both belong on
    the combined file instead:

    - The disclosure has to be one sentence at the front. Repeated before
      every segment the crossfade would partly bury all but the first.
    - Marking a segment is work that is thrown away: the combined file is
      embedded and verified in its own right regardless, and each segment
      would otherwise cost its own C2PA signature. It is also a hazard —
      marking *discards* a file it cannot mark, so a segment that failed
      verification would vanish from the combined audio instead of failing
      the run.

    Nothing unmarked escapes: each temp segment is deleted in ``finally``, and
    the caller marks the combined file with the usual fail-closed handling.

    Returns:
        True if ``output_path`` now holds the combined audio. False means
        nothing was written and the caller should fall back to synthesizing
        the markup-free text.
    """
    import tempfile

    import numpy as np
    try:
        import soundfile as sf_ssml
    except ImportError:
        logger.warning("SSML: soundfile is required to combine segments; "
                       "synthesizing the text without its markup instead.")
        return False
    from utils import crossfade_segments

    logger.info("SSML: %d segments detected", len(segments))
    audio_parts = []
    sr_out = None
    for seg in segments:
        if seg.silence_ms > 0 and sr_out:
            audio_parts.append(np.zeros(int(sr_out * seg.silence_ms / 1000), dtype=np.float32))
        if not seg.text.strip():
            continue
        fd_seg, tmp_seg = tempfile.mkstemp(suffix=".wav")
        os.close(fd_seg)
        written = tmp_seg
        try:
            seg_args = argparse.Namespace(**vars(args))
            seg_args._ssml_segment = True
            seg_args.input_text = seg.text
            seg_args.input_file = None  # input_text wins, but be explicit
            seg_args.output_file = tmp_seg
            seg_args.play_direct = False
            seg_args.batch = False
            seg_args.translate = False  # the text is already translated
            seg_args.verify = False  # verify the combined output, not fragments
            if seg.speed != 1.0:
                seg_args.speech_speed = seg.speed
            seg_started = time.time()
            run_synthesis(seg_args)
            # The handler may have written its own container next to the temp
            # path, exactly as it does for a real output.
            written = resolve_written_output(tmp_seg, since=seg_started) or tmp_seg
            if os.path.isfile(written) and os.path.getsize(written) > 100:
                data_seg, sr_seg = _read_pcm(written)
                if data_seg is not None:
                    audio_parts.append(data_seg)
                    sr_out = sr_out or sr_seg
        finally:
            for leftover in {tmp_seg, written}:
                if leftover and os.path.exists(leftover):
                    os.unlink(leftover)

    if not audio_parts or sr_out is None:
        logger.warning("SSML: no segment produced audio; "
                       "synthesizing the text without its markup instead.")
        return False

    combined = crossfade_segments(audio_parts, sample_rate=sr_out)
    sf_ssml.write(output_path, combined, sr_out, subtype="PCM_16")
    logger.info("SSML: combined %d segments → %s", len(audio_parts), output_path)
    return True


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

    # --- Pre-synthesis translation (CrispASR integration) ---
    if getattr(args, 'translate', False):
        try:
            from handlers.crispasr_handler import translate_text_with_crispasr
            original_text = text_to_synthesize
            text_to_synthesize = translate_text_with_crispasr(
                text_to_synthesize,
                source_lang=args.translate_from,
                target_lang=args.translate_to,
                backend=args.translate_backend,
            )
            logger.info("Translation (%s->%s): '%s...' -> '%s...'",
                        args.translate_from, args.translate_to,
                        original_text[:50], text_to_synthesize[:50])
        except Exception as e_tr:
            logger.warning("Translation failed, using original text: %s", e_tr)

    # --- SSML-lite preprocessing (parsing only) ---
    # Multi-segment SSML used to synthesize, crossfade, write and *return*
    # right here — a second exit from run_synthesis that never reached the
    # disclosure and marking block below. The combined file kept only whatever
    # audio watermark happened to survive concatenation, with nothing checking
    # that it still cleared threshold; its LIST/INFO metadata and its C2PA
    # manifest were gone; the marking preflight never saw the real output
    # format; and --play-direct played the result unverified.
    #
    # Segments are now rendered in place of the handler call further down
    # instead, which puts the combined output through the same
    # trim/normalize/resample/disclose/mark/play sequence as every other
    # output and leaves run_synthesis with a single exit again.
    _ssml_segments = None
    if not getattr(args, '_ssml_segment', False):
        try:
            from ssml import has_ssml, parse_ssml
            if has_ssml(text_to_synthesize):
                segments = parse_ssml(text_to_synthesize)
                if len(segments) > 1:
                    _ssml_segments = segments
                    # Fallback for the case where there is nowhere to render
                    # segments to: speak the words without the markup rather
                    # than reading the tags out loud.
                    text_to_synthesize = " ".join(
                        s.text.strip() for s in segments if s.text.strip())
                elif len(segments) == 1:
                    text_to_synthesize = segments[0].text
                    if segments[0].speed != 1.0:
                        args.speech_speed = segments[0].speed
        except ImportError:
            pass

    model_config_base = GERMAN_TTS_MODELS.get(args.model_id)
    if not model_config_base:
        logger.error(f"Invalid model ID '{args.model_id}' passed to run_synthesis.")
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
        if "USER MUST SET" in current_config_for_handler.get("ollama_model_name",
            "") or not current_config_for_handler.get("ollama_model_name"):
            logger.error(f"For {args.model_id}, Ollama model name not set. Use --ollama-model-name or set in config.")
            return

    effective_voice_id = args.german_voice_id

    # NEW: Check for zero-shot model configuration (same logic as in test_all_models)
    is_zero_shot_model = (
        current_config_for_handler.get('default_voice_id') is None and
        not current_config_for_handler.get('available_voices')
    )
    # Add specific model IDs that are zero-shot
    if args.model_id in ["llasa_hybrid_de_zeroshot", "llasa_german_transformers_zeroshot",
                         "llasa_multilingual_hf_zeroshot", "kartoffelbox_zeroshot"]:
        is_zero_shot_model = True

    if not effective_voice_id and not is_zero_shot_model:
        default_v = current_config_for_handler.get('default_voice_id') or \
                    current_config_for_handler.get('default_model_path_in_repo') or \
                    str(current_config_for_handler.get('default_speaker_embedding_index', '')) or \
                    str(current_config_for_handler.get('default_speaker_id', ''))
        effective_voice_id = default_v if (isinstance(default_v, Path) or (isinstance(default_v,
            str) and default_v.strip())) else None

    # Updated condition to allow None for zero-shot models
    if not effective_voice_id and not is_zero_shot_model and not (args.model_id.startswith("coqui_tts") and current_config_for_handler.get("default_coqui_speaker") is None):  # noqa: E501
        logger.error(f"No voice ID specified and no default could be determined for model {args.model_id}.")
        return

    # For zero-shot models, effective_voice_id can be None
    if is_zero_shot_model and not effective_voice_id:
        logger.info(f"Zero-shot model '{args.model_id}': proceeding without voice ID (zero-shot synthesis)")
        effective_voice_id = None

    # --- Inject CLI synthesis overrides into handler config ---
    if getattr(args, 'speech_speed', 1.0) != 1.0:
        current_config_for_handler["_cli_speech_speed"] = args.speech_speed
    if getattr(args, 'trim_silence', False):
        current_config_for_handler["_cli_trim_silence"] = True
    if getattr(args, 'tts_steps', None) is not None:
        current_config_for_handler["_cli_tts_steps"] = args.tts_steps
    if getattr(args, 'tts_language', None):
        current_config_for_handler["language"] = args.tts_language  # override config language
    if getattr(args, 'pitch_shift', 0.0) != 0.0:
        current_config_for_handler["_cli_pitch_shift"] = args.pitch_shift
    if getattr(args, 'instruct', None):
        current_config_for_handler["instruct"] = args.instruct  # override config instruct
    if getattr(args, 'ref_text', None):
        current_config_for_handler["reference_text"] = args.ref_text
    if getattr(args, 'no_spoken_disclaimer', False):
        current_config_for_handler["_cli_no_spoken_disclaimer"] = True
    if getattr(args, 'lexicon', None):
        current_config_for_handler["_cli_lexicon"] = args.lexicon

    handler_key = current_config_for_handler.get("handler_function_key", args.model_id)
    handler_func = current_all_handlers.get(handler_key)

    if handler_func:
        # --- Voice-cloning consent gate ---
        _is_voice_cloning = False
        _needs_disclosure = False
        try:
            from watermark import (
                log_consent_attestation,
                requires_consent,
                requires_spoken_disclosure,
                resolve_speaker_identity,
            )
            _is_voice_cloning = requires_consent(args.model_id, handler_key, effective_voice_id,
                                                 model_config=current_config_for_handler)
            # Cloning is not the only route to a deep fake: a model whose preset
            # voice is an identifiable person produces one too. See Art. 3(60).
            _needs_disclosure = requires_spoken_disclosure(
                _is_voice_cloning,
                resolve_speaker_identity(current_config_for_handler,
                                         getattr(args, 'speaker_identity', None)),
                model_id=args.model_id)
            if _is_voice_cloning and not getattr(args, 'i_have_rights', False):
                logger.error(
                    "Model '%s' involves voice cloning. You must pass --i-have-rights to attest "
                    "that you have the consent of the speaker whose voice this clones, "
                    "or that it is your own voice.", args.model_id)
                return
            if _is_voice_cloning:
                log_consent_attestation(args.model_id, effective_voice_id)
        except ImportError:
            # Fails closed, like the marking gate below. The consent check is
            # the control standing between this tool and cloning someone's
            # voice without their agreement; silently skipping it because a
            # module is missing is the one outcome it must never have.
            logger.error(
                "watermark module unavailable — cannot determine whether '%s' clones a "
                "voice, so synthesis is refused. Reinstall CrispTTS, or set "
                "CRISPTTS_ALLOW_UNMARKED=1 to bypass every provenance control.",
                args.model_id)
            if not os.environ.get("CRISPTTS_ALLOW_UNMARKED"):
                return

        # --- Marking preflight: refuse BEFORE generating anything ---
        # Gating here rather than after synthesis means unmarkable audio is
        # never produced at all, and no model is loaded to produce it.
        _marking_policy = None
        try:
            from watermark import MarkingError, preflight_marking
            try:
                _marking_policy = preflight_marking(
                    args.output_file,
                    handler_key=handler_key,
                    no_watermark=getattr(args, 'no_watermark', False),
                    allow_unmarked=getattr(args, 'allow_unmarked', False),
                    responsibility_accepted=getattr(args, 'accept_marking_responsibility', False),
                    no_spoken_disclaimer=getattr(args, 'no_spoken_disclaimer', False),
                    c2pa_cert=getattr(args, 'c2pa_cert', None),
                    c2pa_key=getattr(args, 'c2pa_key', None),
                )
            except MarkingError as e_pre:
                logger.error("%s", e_pre)
                return
        except ImportError:
            if not os.environ.get("CRISPTTS_ALLOW_UNMARKED"):
                logger.error("watermark module unavailable — refusing to synthesize unmarkable audio. "
                             "Pass --allow-unmarked --accept-marking-responsibility to override.")
                return

        # Playback must happen AFTER marking, never during synthesis — otherwise
        # the user hears audio that was never marked. So we always let the
        # handler write a file and play it ourselves once marking is done.
        #
        # This includes CrispASR. It used to be exempt, on the grounds that its
        # binary marks internally — but that is a claim about another program's
        # behaviour, and mark_audio_file() deliberately refuses to take it on
        # trust (measured: CrispTTS's detector reads 0.44, its noise floor, on
        # crispasr kokoro output). Exempting the playback path meant the one
        # place audio reached a listener was also the one place nothing was
        # verified. Real --stream is the sole exception below, because
        # incremental playback cannot wait for a completed file.
        _streaming = bool(getattr(args, 'stream', False)) and handler_key == "crispasr"
        _temp_play_file = None
        _play_after_marking = False
        _effective_output = args.output_file
        if args.play_direct and not _streaming:
            _play_after_marking = True
            if not args.output_file:
                import tempfile
                fd, _temp_play_file = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                _effective_output = _temp_play_file
                args.output_file = _temp_play_file
        elif args.play_direct and _streaming:
            # Streaming plays audio as it is produced, so there is no completed
            # file to verify first. Nothing unverified is *written* — any
            # --output-file is still marked and gated below — but what the
            # listener hears is whatever the CrispASR binary emitted.
            logger.warning(
                "--stream --play-direct plays audio as it is synthesized, so what you "
                "hear is not verified against the marking gate; it carries only the "
                "CrispASR binary's own watermark. Any --output-file is still verified. "
                "Drop --stream for playback that is marked and verified first.")

        try:
            # SSML segments are rendered in place of the handler call. The
            # combined file then continues through the shared post-processing
            # below — the marking preflight above already ran against the real
            # output path, so an unmarkable target was refused before this.
            _ssml_handled = False
            _synth_started = time.time()
            if _ssml_segments and _effective_output:
                _ssml_handled = _synthesize_ssml_segments(
                    args, _ssml_segments, _effective_output)

            if _ssml_handled:
                pass  # combined SSML output written
            # Use streaming handler if --stream and crispasr backend
            elif getattr(args, 'stream', False) and handler_key == "crispasr":
                from handlers.crispasr_handler import synthesize_with_crispasr_streaming
                synthesize_with_crispasr_streaming(
                    current_config_for_handler, text_to_synthesize, effective_voice_id,
                    args.model_params, _effective_output, args.play_direct)
            else:
                handler_func(current_config_for_handler, text_to_synthesize, effective_voice_id, args.model_params,
                    _effective_output, False if _play_after_marking else args.play_direct)

            # --- Follow the audio the handler actually wrote ---
            # Handlers routinely override the requested container, so the file
            # to trim, disclaim, mark and play is not necessarily the one that
            # was asked for. Rebinding args.output_file here means every step
            # below — including _discard_output() on failure — acts on the real
            # output instead of quietly skipping a path that does not exist.
            _actual_output = resolve_written_output(_effective_output, since=_synth_started)
            if _actual_output and _actual_output != _effective_output:
                # Anything left at the requested path is an empty stub — the
                # resolver only looks past a path that holds real audio. Drop
                # it so it cannot be mistaken for unmarked output, but only
                # once it is certain the two paths are not the same file.
                try:
                    _stale = (os.path.isfile(_effective_output)
                              and not os.path.samefile(_effective_output, _actual_output))
                except OSError:
                    _stale = False
                if _stale:
                    try:
                        os.unlink(_effective_output)
                    except OSError:
                        pass
                if _temp_play_file and _temp_play_file == _effective_output:
                    _temp_play_file = _actual_output
                args.output_file = _actual_output
                _effective_output = _actual_output

            # --- Post-synthesis silence trimming (Python fallback for non-crispasr) ---
            if getattr(args, 'trim_silence', False) and args.output_file and os.path.isfile(args.output_file):
                if handler_key != "crispasr":  # crispasr handles it via --tts-trim-silence
                    from utils import trim_silence_file
                    trim_silence_file(args.output_file)

            # --- Post-synthesis normalization ---
            if getattr(args, 'normalize', False) and args.output_file and os.path.isfile(args.output_file):
                try:
                    import soundfile as sf_norm

                    from utils import normalize_audio
                    data_n, sr_n = sf_norm.read(args.output_file, dtype="float32")
                    if data_n.ndim > 1:
                        data_n = data_n[:, 0]
                    data_n = normalize_audio(data_n)
                    sf_norm.write(args.output_file, data_n, sr_n, subtype="PCM_16")
                    logger.info("Audio normalized to -3 dB peak.")
                except Exception as e_norm:
                    logger.warning("Normalization failed: %s", e_norm)

            # --- Post-synthesis resampling ---
            if getattr(args, 'output_sample_rate', None) and args.output_file and os.path.isfile(args.output_file):
                try:
                    import soundfile as sf_rs

                    from utils import resample_audio
                    data, sr = sf_rs.read(args.output_file, dtype="float32")
                    if data.ndim > 1:
                        data = data[:, 0]
                    if sr != args.output_sample_rate:
                        data = resample_audio(data, sr, args.output_sample_rate)
                        sf_rs.write(args.output_file, data, args.output_sample_rate, subtype="PCM_16")
                        logger.info("Resampled output: %d Hz → %d Hz", sr, args.output_sample_rate)
                except Exception as e_rs:
                    logger.warning("Could not resample output: %s", e_rs)

            # --- Spoken disclaimer for voice-cloned audio (Art. 50(4)) ---
            # Format-agnostic; runs before marking so the disclaimer itself
            # ends up inside the watermarked audio. Fails closed, like marking:
            # a deepfake delivered without its disclosure is the failure mode
            # this exists to prevent, so a silent skip is not acceptable.
            #
            # Skipped for an SSML segment: that temp file is a fragment of the
            # output, never the output. Its parent call disclaims and marks the
            # combined file. See _synthesize_ssml_segments().
            if (_needs_disclosure and args.output_file and os.path.isfile(args.output_file)
                    and not getattr(args, 'no_spoken_disclaimer', False)
                    and not getattr(args, '_ssml_segment', False)):
                try:
                    from watermark import DisclosureError, prepend_disclaimer_file
                except ImportError:
                    logger.error("watermark module unavailable — cannot add the spoken AI "
                                 "disclosure to voice-cloned audio. Pass "
                                 "--no-spoken-disclaimer --accept-marking-responsibility "
                                 "to take on the disclosure duty yourself.")
                    _discard_output(args, _temp_play_file)
                    return
                try:
                    prepend_disclaimer_file(
                        args.output_file,
                        language=current_config_for_handler.get("language"),
                        disclosure_lang=getattr(args, 'disclosure_lang', None))
                except DisclosureError as e_disc:
                    logger.error("%s", e_disc)
                    _discard_output(args, _temp_play_file)
                    return

            # --- AI-provenance marking (EU AI Act Art. 50(2)) ---
            # Deliberately the LAST step: silence trimming, normalization,
            # resampling and the spoken disclaimer all rewrite the file, so
            # marking earlier would be stripped or weakened. Fails closed —
            # if the output cannot be marked it is not delivered at all.
            #
            # Skipped for an SSML segment, which is a temp fragment rather than
            # a delivered output; its parent call marks the combined file, and
            # marking a fragment that failed verification would silently delete
            # it. See _synthesize_ssml_segments().
            if (args.output_file and os.path.isfile(args.output_file)
                    and not getattr(args, '_ssml_segment', False)):
                _allow_unmarked = True if getattr(args, 'allow_unmarked', False) else None
                try:
                    from watermark import MarkingError, mark_audio_file
                except ImportError:
                    if _allow_unmarked or os.environ.get("CRISPTTS_ALLOW_UNMARKED"):
                        logger.warning("watermark module unavailable — output is UNMARKED.")
                    else:
                        logger.error(
                            "watermark module unavailable — refusing to deliver unmarked "
                            "synthetic audio. Pass --allow-unmarked to override.")
                        _discard_output(args, _temp_play_file)
                        return
                else:
                    try:
                        mark_audio_file(
                            args.output_file,
                            handler_key=handler_key,
                            allow_unmarked=_allow_unmarked,
                            c2pa_cert=getattr(args, 'c2pa_cert', None),
                            c2pa_key=getattr(args, 'c2pa_key', None),
                            policy=_marking_policy,
                            model_id=args.model_id,
                        )
                    except MarkingError as e_mark:
                        logger.error("%s", e_mark)
                        _discard_output(args, _temp_play_file)
                        return

            # --- Playback (only ever of marked audio) ---
            if _play_after_marking and args.output_file and os.path.isfile(args.output_file):
                try:
                    from utils import play_audio
                    play_audio(args.output_file, is_path=True)
                except Exception as e_play:
                    logger.warning("Playback failed: %s", e_play)
            if _temp_play_file:
                try:
                    os.unlink(_temp_play_file)
                except OSError:
                    pass
                args.output_file = None  # restore original

            # --- Post-synthesis ASR verification (CrispASR integration) ---
            if getattr(args, 'verify', False) and args.output_file and os.path.isfile(args.output_file):
                try:
                    from handlers.crispasr_handler import verify_tts_with_asr
                    logger.info("Running ASR verification on TTS output...")
                    result = verify_tts_with_asr(
                        args.output_file, text_to_synthesize,
                        asr_backend=args.verify_backend,
                    )
                    if "error" in result:
                        logger.warning("ASR verification failed: %s", result["error"])
                    else:
                        logger.info("ASR roundtrip result: '%s'", result["asr_text"])
                        logger.info("Word overlap similarity: %.1f%%", result["similarity"] * 100)
                        print("\n--- ASR Verification ---")
                        print(f"Original:   {result['original_text']}")
                        print(f"ASR output: {result['asr_text']}")
                        print(f"Similarity: {result['similarity']*100:.1f}%")
                except Exception as e_verify:
                    logger.warning("ASR verification error: %s", e_verify)

        except Exception as e_synth:
            logger.error(f"Synthesis failed for model {args.model_id}: {e_synth}", exc_info=True)
    else:
        logger.error(f"No synthesis handler function found for model ID: {args.model_id} (handler key: {handler_key})")

def _probe_crispasr_backends(models_dict):
    """Probe CrispASR backends to check which models are available."""
    from handlers.crispasr_handler import _find_crispasr
    exe = _find_crispasr()
    if not exe:
        print("\n[Check] CrispASR binary not found — cannot probe backends.")
        return

    print(f"\n[Check] Probing CrispASR backends (binary: {exe})...")
    crispasr_models = {
        mid: cfg for mid, cfg in models_dict.items()
        if cfg.get("handler_function_key") == "crispasr"
    }
    for mid, cfg in crispasr_models.items():
        backend = cfg.get("crispasr_backend", "?")
        model_path = cfg.get("crispasr_model_path", "auto")
        try:
            import subprocess
            result = subprocess.run(  # noqa: S603
                [exe, "-m", model_path, "--backend", backend,
                 "--auto-download", "--dry-run"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"  {mid:40s} [{backend:20s}]  READY")
            else:
                # Check if model just needs downloading
                if "downloading" in (result.stderr or "").lower() or "resolving" in (result.stderr or "").lower():
                    print(f"  {mid:40s} [{backend:20s}]  NEEDS DOWNLOAD")
                else:
                    print(f"  {mid:40s} [{backend:20s}]  UNAVAILABLE")
        except subprocess.TimeoutExpired:
            print(f"  {mid:40s} [{backend:20s}]  TIMEOUT")
        except Exception:
            print(f"  {mid:40s} [{backend:20s}]  ERROR")


def _validate_config():
    """Check GERMAN_TTS_MODELS entries for common misconfigurations."""
    from config import GERMAN_TTS_MODELS
    issues = []
    for mid, cfg in GERMAN_TTS_MODELS.items():
        if not isinstance(cfg, dict):
            issues.append(f"  {mid}: config is not a dict")
            continue
        if "handler_function_key" not in cfg:
            issues.append(f"  {mid}: missing 'handler_function_key'")
        if cfg.get("handler_function_key") == "crispasr" and "crispasr_backend" not in cfg:
            issues.append(f"  {mid}: crispasr handler but missing 'crispasr_backend'")
        sr = cfg.get("sample_rate")
        if sr is not None and (not isinstance(sr, int) or sr <= 0):
            issues.append(f"  {mid}: invalid sample_rate={sr}")
    if issues:
        logger.warning("Config validation warnings:\n%s", "\n".join(issues))
    return len(issues) == 0


def main_cli_entrypoint():
    parser = argparse.ArgumentParser(description="CrispTTS: Modular German Text-to-Speech Synthesizer",
        formatter_class=argparse.RawTextHelpFormatter)
    action_group = parser.add_argument_group(title="Primary Actions")
    input_group = parser.add_mutually_exclusive_group(required=False)
    action_group.add_argument("--list-models", action="store_true", help="List all configured TTS models.")
    action_group.add_argument("--check", action="store_true",
        help="With --list-models: probe CrispASR backends to show availability status.")
    action_group.add_argument("--voice-info", type=str, metavar="MODEL_ID",
        help="Display voice/speaker info for a specific MODEL_ID.")
    action_group.add_argument("--test-all", action="store_true",
        help="Test all models with default voices. Requires --input-text or --input-file.")
    action_group.add_argument("--test-all-speakers", action="store_true",
        help="Test all models with ALL configured voices. Requires --input-text or --input-file.")
    action_group.add_argument("--skip-models", type=str, nargs='*', default=[],
        help="List of model IDs (space-separated) to skip during --test-all or --test-all-speakers.")
    action_group.add_argument("--detect-watermark", type=str, metavar="AUDIO_FILE",
        help="Detect AI-generated watermark in a WAV file and report confidence.")
    action_group.add_argument("--list-disclosure-langs", action="store_true",
        help="List the languages the spoken AI disclosure is available in.")
    action_group.add_argument("--consent-log-erase", type=str, nargs="?", const="",
        metavar="SUBJECT",
        help="Erase consent audit entries (GDPR Art. 17). With a SUBJECT (a\n"
             "reference-audio path or ref_sha256 digest) only matching lines are\n"
             "removed; with no argument the whole log is erased.")
    action_group.add_argument("--consent-log-prune", action="store_true",
        help="Drop consent audit entries past the retention window and exit.")
    action_group.add_argument("--consent-log-verify", action="store_true",
        help="Check the consent audit log's hash chain and anchor for tampering.")
    action_group.add_argument("--cache-stats", action="store_true",
        help="Show synthesis cache statistics (size, entries).")
    action_group.add_argument("--cache-clear", action="store_true",
        help="Clear the synthesis cache.")
    action_group.add_argument("--server", action="store_true",
        help="Run as HTTP server with OpenAI-compatible /v1/audio/speech endpoint.")
    action_group.add_argument("--server-host", type=str, default="127.0.0.1",
        help="Server bind address (default: 127.0.0.1).")
    action_group.add_argument("--server-port", type=int, default=8880,
        help="Server port (default: 8880).")
    action_group.add_argument("--server-rate-limit", type=int, default=10,
        help="Max synthesis requests per minute per IP (default: 10, 0=unlimited).")
    action_group.add_argument("--warm-up", type=str, default=None, metavar="MODEL_ID",
        help="Pre-synthesize with this model at server startup to warm caches.")


    synth_group = parser.add_argument_group(title="Synthesis Options (used with --model-id or --test-all*)")
    input_group.add_argument("--input-text", type=str, help="Text to synthesize.")
    input_group.add_argument("--input-file", type=str, help="Path to input file (txt, md, html, pdf, epub).")

    model_choices = list(GERMAN_TTS_MODELS.keys()) if GERMAN_TTS_MODELS else []
    synth_group.add_argument("--model-id", type=str, choices=model_choices, default=None,
        help="Select TTS model ID. Required for single synthesis if not using an action flag.")
    synth_group.add_argument("--backend", type=str, default=None, metavar="NAME",
        help="Shortcut: select a CrispASR backend by name (e.g., kokoro, piper, dots-tts).\n"
             "Equivalent to --model-id crispasr_<name> but more convenient.")
    synth_group.add_argument("--output-file", type=str, help="Path to save synthesized audio (for single synthesis).")
    synth_group.add_argument("--output-dir", type=str, default="tts_test_outputs",
        help="Directory for --test-all* outputs (default: tts_test_outputs).")
    synth_group.add_argument("--play-direct", action="store_true",
        help="Play audio directly after synthesis (not with --test-all*).")
    synth_group.add_argument("--german-voice-id", type=str,
        help="Override default voice/speaker for the selected model.")
    synth_group.add_argument("--model-params", type=str,
        help="JSON string of model-specific parameters (e.g., '{\"temperature\":0.7}').")
    synth_group.add_argument("--speech-speed", type=float, default=1.0,
        help="Speech rate multiplier (>1 = faster, <1 = slower, default: 1.0).")
    synth_group.add_argument("--trim-silence", action="store_true",
        help="Trim leading/trailing silence from synthesized audio.")
    synth_group.add_argument("--tts-steps", type=int, default=None, metavar="N",
        help="DPM-Solver++ inference steps for diffusion models (default: backend-specific).")
    synth_group.add_argument("--tts-language", type=str, default=None, metavar="LANG",
        help="Override language for multilingual models (e.g., de, en, zh, ja).")
    synth_group.add_argument("--pitch-shift", type=float, default=0.0, metavar="HZ",
        help="Pitch shift in Hz (positive = higher, negative = lower, default: 0).")
    synth_group.add_argument("--instruct", type=str, default=None, metavar="TEXT",
        help="Natural-language voice/style description for VoiceDesign models (e.g., qwen3-tts).")
    synth_group.add_argument("--output-sample-rate", type=int, default=None, metavar="HZ",
        help="Resample output audio to this sample rate (e.g., 16000, 22050, 44100).")
    synth_group.add_argument("--stream", action="store_true",
        help="Stream audio playback during synthesis (crispasr backends only).")
    synth_group.add_argument("--ref-text", type=str, default=None, metavar="TEXT",
        help="Transcript of the reference voice audio for inline voice cloning (TADA, dots-tts).")
    synth_group.add_argument("--no-spoken-disclaimer", action="store_true",
        help="Skip the AI-disclosure spoken prefix on voice-cloned audio.")
    synth_group.add_argument("--disclosure-lang", type=str, default=None, metavar="LANG",
        help="Language for the spoken AI disclosure on voice-cloned audio (e.g. 'de',\n"
             "'en', 'zh'). Defaults to the model's declared language. Required for a\n"
             "meaningful disclosure with multilingual models, whose output language\n"
             "depends on the input text rather than the model. --list-disclosure-langs\n"
             "shows what is available.")
    synth_group.add_argument("--speaker-identity", type=str, default=None,
        choices=["real_person", "synthetic", "unknown"],
        help="Whether a fixed-speaker model's preset voice belongs to an identifiable\n"
             "person. 'real_person' makes the output a deep fake under EU AI Act\n"
             "Art. 3(60) and prepends the spoken disclosure, as voice cloning does.\n"
             "Overrides the model's declared speaker_identity; use it when you know\n"
             "more about a voice than the config does.")
    synth_group.add_argument("--lexicon", type=str, default=None, metavar="TSV_PATH",
        help="Path to a word→phoneme TSV file for custom pronunciation (CrispASR backends).")
    synth_group.add_argument("--batch", action="store_true",
        help="Batch mode: split input at blank lines, produce numbered output files\n"
             "(e.g., output_001.wav, output_002.wav, ...). Requires --output-dir.")
    synth_group.add_argument("--jobs", type=int, default=1, metavar="N",
        help="Concurrent synthesis jobs for --batch mode (default: 1).")
    synth_group.add_argument("--normalize", action="store_true",
        help="Peak-normalize output audio to -3 dB for consistent volume.")

    # CrispASR integration options
    crispasr_group = parser.add_argument_group(title="CrispASR Integration")
    crispasr_group.add_argument("--verify", action="store_true",
        help="Run ASR on TTS output for roundtrip quality verification (requires crispasr binary).")
    crispasr_group.add_argument("--verify-backend", type=str, default="parakeet",
        help="ASR backend for --verify (default: parakeet).")
    crispasr_group.add_argument("--translate", action="store_true",
        help="Translate input text before synthesis (e.g., EN→DE via m2m100).")
    crispasr_group.add_argument("--translate-from", type=str, default="en",
        help="Source language for --translate (default: en).")
    crispasr_group.add_argument("--translate-to", type=str, default="de",
        help="Target language for --translate (default: de).")
    crispasr_group.add_argument("--translate-backend", type=str, default="m2m100",
        help="Translation backend for --translate (default: m2m100).")

    # Watermarking / provenance options
    wm_group = parser.add_argument_group(title="Watermarking & Provenance")
    wm_group.add_argument("--no-watermark", action="store_true",
        help="Disable ALL AI-provenance marking: audio watermark, container metadata\n"
             "and C2PA (debug only). EU AI Act Art. 50(2) requires synthetic audio to\n"
             "be machine-readably marked; responsibility for unmarked output is yours.")
    wm_group.add_argument("--allow-unmarked", action="store_true",
        help="Deliver output even if marking fails or is not detectable.\n"
             "Without this, an output that cannot be marked is deleted and\n"
             "synthesis exits non-zero. Equivalent to CRISPTTS_ALLOW_UNMARKED=1.")
    wm_group.add_argument("--accept-marking-responsibility", action="store_true",
        help="Required alongside any provenance opt-out (--no-watermark,\n"
             "--allow-unmarked, --no-spoken-disclaimer). Affirms that you accept\n"
             "the AI-content marking and disclosure duty for this output.\n"
             "Logged as a [MARKING] audit line, like --i-have-rights.")
    wm_group.add_argument("--watermark-model", type=str, metavar="GGUF_PATH",
        help="Path to AudioSeal GGUF model for neural watermarking (optional upgrade).")
    wm_group.add_argument("--i-have-rights", action="store_true",
        help="Attest that you have consent of the speaker whose voice is being cloned,\n"
             "or that it is your own voice. Required for voice-cloning models.")
    wm_group.add_argument("--c2pa-cert", type=str, metavar="PEM_PATH",
        help="Path to X.509 PEM certificate for C2PA content credentials signing.")
    wm_group.add_argument("--c2pa-key", type=str, metavar="PEM_PATH",
        help="Path to PEM private key for C2PA content credentials signing.")

    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set console logging level (default: INFO).")

    override_group = parser.add_argument_group(title="Runtime Model Path/Repo Overrides (for selected --model-id or during --test-all*)")  # noqa: E501
    override_group.add_argument("--override-main-model-repo", type=str, metavar="REPO_OR_PATH",
        help="Override main model repository ID or path.")
    override_group.add_argument("--override-model-filename", type=str, metavar="FILENAME",
        help="Override specific model filename within the main repo.")
    override_group.add_argument("--override-tokenizer-repo", type=str, metavar="REPO_OR_PATH",
        help="Override tokenizer repository ID or path.")
    override_group.add_argument("--override-vocoder-repo", type=str, metavar="REPO_OR_NAME",
        help="Override vocoder repository ID or name.")
    override_group.add_argument("--override-speaker-embed-repo", type=str, metavar="REPO_ID",
        help="Override speaker embeddings repository ID.")
    override_group.add_argument("--override-piper-voices-repo", type=str, metavar="REPO_ID",
        help="Override main repository ID for Piper voices.")

    api_group = parser.add_argument_group(title="API Backend Overrides (also in config.py)")
    api_group.add_argument("--lm-studio-api-url", type=str, default=LM_STUDIO_API_URL_DEFAULT if 'LM_STUDIO_API_URL_DEFAULT' in globals() else "http://127.0.0.1:1234/v1/completions", help="Override LM Studio API URL.")  # noqa: E501
    api_group.add_argument("--gguf-model-name-in-api", type=str,
        help="Override model name for LM Studio API (from config or this flag).")
    api_group.add_argument("--ollama-api-url", type=str, default=OLLAMA_API_URL_DEFAULT if 'OLLAMA_API_URL_DEFAULT' in globals() else "http://localhost:11434/api/generate", help="Override Ollama API URL.")  # noqa: E501
    api_group.add_argument("--ollama-model-name", type=str,
        help="Override model name/tag for Ollama (from config or this flag).")

    args = parser.parse_args()

    # --- Resolve --backend shortcut to --model-id ---
    if getattr(args, 'backend', None) and not args.model_id:
        backend_name = args.backend.replace("-", "_")
        candidate = f"crispasr_{backend_name}"
        if candidate in GERMAN_TTS_MODELS:
            args.model_id = candidate
        else:
            # Try with original name (e.g. "dots-tts" → "crispasr_dots_tts")
            candidate2 = f"crispasr_{args.backend.replace('-', '_')}_tts"
            if candidate2 in GERMAN_TTS_MODELS:
                args.model_id = candidate2
            else:
                # Search for any model with matching crispasr_backend value
                for mid, cfg in GERMAN_TTS_MODELS.items():
                    if cfg.get("crispasr_backend") == args.backend:
                        args.model_id = mid
                        break
                else:
                    print(f"Error: No CrispASR backend matching '{args.backend}'. "
                          f"Use --list-models to see available models.")
                    return

    logging.basicConfig(level=args.loglevel.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', force=True)
    cli_numeric_log_level = getattr(logging, args.loglevel.upper(), logging.INFO)

    for name in list(logging.root.manager.loggerDict.keys()) + ["root", _main_mp_logger.name]:
        lgr = logging.getLogger(name)
        if lgr.name.startswith("CrispTTS") or lgr.name == "root":
            if lgr.level == 0 or lgr.level > cli_numeric_log_level:
                lgr.setLevel(cli_numeric_log_level)
            if lgr.name == _main_mp_logger.name:
                for handler_obj in lgr.handlers: # handler_obj instead of handler
                    if handler_obj.level == 0 or handler_obj.level > cli_numeric_log_level:
                        handler_obj.setLevel(cli_numeric_log_level)

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
            is_crisptts_sub_logger = (
                lgr.name.startswith("CrispTTS.") and lgr.name != "CrispTTS.main"
            )  # Allow main app INFO

            if is_noisy_third_party and not is_crisptts_sub_logger : # Don't silence our main app logger's INFO
                if lgr.getEffectiveLevel() < logging.WARNING:
                    lgr.setLevel(logging.WARNING)
                    # Debug log from main logger, not the one being quieted
                    logger.debug(f"Set logger '{lgr.name}' to WARNING to reduce noise at INFO level.")


    logger.info(f"Effective logging level for CrispTTS and sub-loggers set to: {args.loglevel.upper()}")
    _main_mp_logger.debug(f"Monkey patch logger effective level is: {logging.getLevelName(_main_mp_logger.getEffectiveLevel())}")  # noqa: E501


    # --- Config validation ---
    _validate_config()

    # --- Watermarking setup ---
    # Neural watermark backends (WavMark/AudioSeal) are lazy-loaded on first
    # watermark_embed() call to avoid 200MB+ model load on --list-models etc.
    # Only explicit --watermark-model triggers eager loading.
    if args.no_watermark:
        # The "output will be UNMARKED" warning is deliberately NOT emitted
        # here. At this point we do not yet know whether the opt-out will be
        # honoured: preflight refuses it without an attestation, and the
        # watermark floor overrides it when nothing else marks the output.
        # mark_audio_file() warns if and when marking is actually skipped.
        os.environ["CRISPTTS_NO_WATERMARK"] = "1"
    if getattr(args, 'allow_unmarked', False):
        os.environ["CRISPTTS_ALLOW_UNMARKED"] = "1"
    if getattr(args, 'accept_marking_responsibility', False):
        os.environ["CRISPTTS_ACCEPT_MARKING_RESPONSIBILITY"] = "1"

    if not args.no_watermark and args.watermark_model:
        try:
            from watermark import load_audioseal_model
            if load_audioseal_model(args.watermark_model):
                logger.info("AudioSeal neural watermark active (crispasr GGUF).")
            else:
                logger.info("GGUF load failed; neural watermark will lazy-load on first use.")
        except ImportError:
            logger.debug("watermark module not available.")

    # C2PA certificate setup
    if getattr(args, 'c2pa_cert', None):
        os.environ["C2PA_CERT_PATH"] = args.c2pa_cert
    if getattr(args, 'c2pa_key', None):
        os.environ["C2PA_KEY_PATH"] = args.c2pa_key

    if args.server:
        from server import run_server
        run_server(args.server_host, args.server_port,
                   rate_limit=getattr(args, 'server_rate_limit', 10),
                   warm_up=getattr(args, 'warm_up', None))
        return

    if getattr(args, 'cache_stats', False):
        try:
            import cache as _cache
            _cache.configure(enabled=True)
            total_size = 0
            n_entries = 0
            for name in os.listdir(_cache._cache_dir):
                path = os.path.join(_cache._cache_dir, name)
                if os.path.isfile(path):
                    total_size += os.path.getsize(path)
                    n_entries += 1
            print(f"Cache directory: {_cache._cache_dir}")
            print(f"Entries: {n_entries}")
            print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
            print(f"Max size: {_cache._max_bytes / 1024 / 1024:.0f} MB")
        except Exception as e:
            print(f"Cache stats error: {e}")
        return

    if getattr(args, 'cache_clear', False):
        try:
            import shutil

            import cache as _cache
            if os.path.isdir(_cache._cache_dir):
                n = len(os.listdir(_cache._cache_dir))
                shutil.rmtree(_cache._cache_dir)
                os.makedirs(_cache._cache_dir, exist_ok=True)
                print(f"Cleared {n} cached entries from {_cache._cache_dir}")
            else:
                print("Cache directory does not exist.")
        except Exception as e:
            print(f"Cache clear error: {e}")
        return

    if args.detect_watermark:
        try:
            from watermark import describe_detection
            report = describe_detection(args.detect_watermark)
            if report is None:
                print(f"Could not read audio file: {args.detect_watermark}")
            else:
                print(f"File:       {args.detect_watermark}")
                print(f"Backend:    {report['backend']}")
                print(f"Confidence: {report['confidence']:.4f} "
                      f"(detected at >= {report['threshold']:.2f})")
                print(f"Result:     {report['verdict']}")
                if not report["verdict"].startswith("AI-GENERATED"):
                    print(f"Note:       {report['caveat']}")
                if report["backend"] == "spread_spectrum":
                    print("            The built-in detector is a convenience check. Install "
                          "a neural\n            backend (pip install 'crisptts[robust]') when "
                          "the answer matters.")
        except ImportError:
            logger.error("watermark module not available.")
        return

    if getattr(args, 'list_disclosure_langs', False):
        try:
            from watermark import (
                DEFAULT_DISCLAIMER_LANG,
                DISCLAIMER_TEXTS,
                bundled_disclosure_path,
            )
            print("Spoken AI-disclosure languages (--disclosure-lang):\n")
            for code in sorted(DISCLAIMER_TEXTS):
                offline = "bundled" if bundled_disclosure_path(code) else "needs a TTS backend"
                default = "  (default)" if code == DEFAULT_DISCLAIMER_LANG else ""
                print(f"  {code}  [{offline}]{default}")
                print(f"        {DISCLAIMER_TEXTS[code]}")
            print(f"\n{len(DISCLAIMER_TEXTS)} languages. 'bundled' ones work offline with "
                  "no TTS backend installed.")
        except ImportError:
            logger.error("watermark module not available.")
        return

    if getattr(args, 'consent_log_prune', False):
        try:
            from watermark import consent_log_path, prune_audit_log
            removed = prune_audit_log()
            print(f"Pruned {removed} expired entr{'y' if removed == 1 else 'ies'} "
                  f"from {consent_log_path()}")
        except ImportError:
            logger.error("watermark module not available.")
        return

    if getattr(args, 'consent_log_verify', False):
        try:
            from watermark import anchor_path, consent_log_path, verify_audit_chain
            report = verify_audit_chain()
            print(f"Log:      {consent_log_path()}")
            print(f"Anchor:   {anchor_path()}")
            print(f"Entries:  {report['entries']}")
            if report.get("legacy"):
                print(f"Legacy:   {report['legacy']} entr"
                      f"{'y' if report['legacy'] == 1 else 'ies'} predate hash chaining "
                      "(v0.9.10) —\n          they are covered from the next append onward.")
            if report["rebuilds"]:
                print(f"Rebuilds: {report['rebuilds']} (retention prune or Art. 17 erasure — "
                      "lawful, and recorded)")
            if report["ok"]:
                print("Result:   chain intact — no undocumented change detected")
            else:
                print("Result:   CHAIN BROKEN")
                for issue in report["issues"]:
                    print(f"  - {issue}")
                print("\nThis is tamper-evidence, not tamper-proofing: anyone who can write\n"
                      "the file can rebuild the chain. A break means the log is no longer\n"
                      "reliable evidence of what was attested, not that it definitely was\n"
                      "edited maliciously.")
        except ImportError:
            logger.error("watermark module not available.")
        return

    if getattr(args, 'consent_log_erase', None) is not None:
        try:
            from watermark import consent_log_path, erase_audit_log
            subject = args.consent_log_erase or None
            removed = erase_audit_log(subject)
            scope = f"matching {subject!r}" if subject else "(entire log)"
            print(f"Erased {removed} entr{'y' if removed == 1 else 'ies'} {scope} "
                  f"from {consent_log_path()}")
        except ImportError:
            logger.error("watermark module not available.")
        return

    if args.list_models:
        list_available_models(GERMAN_TTS_MODELS)
        if getattr(args, 'check', False):
            _probe_crispasr_backends(GERMAN_TTS_MODELS)
        return
    if args.voice_info:
        if args.voice_info not in GERMAN_TTS_MODELS:
            logger.error(f"Model ID '{args.voice_info}' for --voice-info not found.")
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
        _load_handlers_if_needed() # Load handlers before testing all
        if not _HANDLERS_LOADED:
            logger.critical("Failed to load handlers. Aborting --test-all / --test-all-speakers.")
            return
        test_text = text_to_process[:500] if len(text_to_process) > 500 else text_to_process
        logger.info(f"--- Applying Test Mode: {'All Speakers' if args.test_all_speakers else 'Default Speaker Only'} ---")  # noqa: E501
        if any([args.override_main_model_repo, args.override_model_filename, args.override_tokenizer_repo,
            args.override_vocoder_repo, args.override_speaker_embed_repo, args.override_piper_voices_repo]):
            logger.warning("CLI repo/path overrides are active and will apply to all compatible models during --test-all(-speakers).")  # noqa: E501
        test_all_models(test_text, args.output_dir, args)
        return

    if not args.model_id:
        parser.error("A --model-id is required for synthesis if not using an action flag.")
        return

    _load_handlers_if_needed()
    if not _HANDLERS_LOADED:
        logger.critical(f"Failed to load handlers. Aborting synthesis for model '{args.model_id}'.")
        return

    # --- Batch mode: split at blank lines, produce numbered files ---
    batch_text = None
    if getattr(args, 'batch', False):
        batch_text = get_text_from_input(
            getattr(args, 'input_text', None), getattr(args, 'input_file', None))
    if batch_text:
        paragraphs = [p.strip() for p in batch_text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            logger.info("Batch mode: only 1 paragraph found, running single synthesis.")
            run_synthesis(args)
            return
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".mp3" if args.model_id == "edge" else ".wav"
        n_jobs = getattr(args, 'jobs', 1)
        logger.info("Batch mode: %d paragraphs → %s/ (jobs=%d)", len(paragraphs), output_dir, n_jobs)

        def _synth_one(item):
            i, para = item
            batch_args = argparse.Namespace(**vars(args))
            batch_args.input_text = para
            batch_args.output_file = str(output_dir / f"output_{i:03d}{ext}")
            batch_args.batch = False
            batch_args.play_direct = False
            logger.info("Batch [%d/%d]: %s", i, len(paragraphs), para[:60])
            try:
                run_synthesis(batch_args)
                if os.path.isfile(batch_args.output_file) and os.path.getsize(batch_args.output_file) > 100:
                    return True
                logger.warning("Batch [%d/%d]: no output produced.", i, len(paragraphs))
                return False
            except Exception as e_batch:
                logger.error("Batch [%d/%d] failed: %s", i, len(paragraphs), e_batch)
                return False

        items = list(enumerate(paragraphs, 1))
        if n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                results = list(pool.map(_synth_one, items))
        else:
            results = [_synth_one(item) for item in items]

        batch_ok = sum(results)
        batch_fail = len(results) - batch_ok
        logger.info("Batch complete: %d/%d succeeded, %d failed in %s/",
                     batch_ok, len(paragraphs), batch_fail, output_dir)
        return

    run_synthesis(args)

if __name__ == "__main__":
    # Nothing heavyweight belongs here. This block used to `import torch` and
    # probe `torch.backends.mps` to set two locals, whose only consumer was a
    # debug print that had already been commented out — so every invocation,
    # including `--help` and `--list-models`, paid a full torch import and
    # threw the answer away. Measured with `-X importtime`: torch was 16.7 s of
    # cumulative import time, and it was the entire startup cost.
    #
    # Note this cost only ever appeared when running `main.py` as a script, not
    # via `import main`, which is why the import profile of the module looked
    # clean. Backends that need torch import it themselves, lazily.
    main_cli_entrypoint()
