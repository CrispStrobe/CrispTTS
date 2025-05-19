#!/usr/bin/env python3
# CrispTTS - main.py
# Main Command-Line Interface for the Text-to-Speech Synthesizer (Modularized with Overrides)

import sys
from pathlib import Path
# import time # Not strictly needed in main.py if timing is in handlers

# --- Add project root to sys.path to ensure relative imports work ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Path Adjustment ---

import argparse
import os
import json
import logging

# --- Environment Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GGML_METAL_NDEBUG"] = "1" # Attempt to silence Metal's verbose llama.cpp logging

# --- Project-Specific Imports ---
try:
    from config import (
        GERMAN_TTS_MODELS,
        LM_STUDIO_API_URL_DEFAULT, # For CLI default
        OLLAMA_API_URL_DEFAULT     # For CLI default
    )
    from utils import (
        get_text_from_input,
        list_available_models,
        get_voice_info
    )
    from handlers import ALL_HANDLERS # Expects a map like {"model_id_key": handler_function}
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import from project modules (config, utils, handlers): {e}", file=sys.stderr)
    print("Please ensure these modules/packages are correctly structured and in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)
except KeyError as e_key:
    print(f"CRITICAL ERROR: 'ALL_HANDLERS' map not found or incomplete in handlers package: {e_key}", file=sys.stderr)
    sys.exit(1)


# --- Conditional Library Availability Checks (for user feedback at startup) ---
TORCH_AVAILABLE_MAIN = False
IS_MPS_MAIN = False
try:
    import torch
    TORCH_AVAILABLE_MAIN = True
    IS_MPS_MAIN = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except ImportError:
    pass # Handled by individual handlers if torch is critical for them

# Global logger for this file
logger = logging.getLogger("CrispTTS.main")


def _apply_cli_overrides_to_config(model_config_dict, model_id_key, cli_args):
    """
    Modifies a copy of the model_config_dict based on CLI override arguments.
    Returns the modified config dictionary.
    """
    config_to_modify = model_config_dict.copy() # Work on a copy

    if cli_args.override_main_model_repo:
        repo_override = cli_args.override_main_model_repo
        updated = False
        if model_id_key in ["orpheus_lex_au", "orpheus_sauerkraut"] and "model_repo_id" in config_to_modify:
            config_to_modify["model_repo_id"] = repo_override; updated = True
        elif model_id_key == "piper_local" and "piper_voice_repo_id" in config_to_modify and not cli_args.override_piper_voices_repo: # General override if specific piper one not used
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
        elif model_id_key not in ["edge", "orpheus_lm_studio", "orpheus_ollama"]: # Don't log for non-repo models
             logger.warning(f"No primary repo key found to override for '{model_id_key}' with '{repo_override}'. Check config keys.")


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
        # OuteTTS and some MLX configurations might use 'tokenizer_path'
        if ("oute_hf" == model_id_key or "mlx_audio_outetts_q4" == model_id_key) and "tokenizer_path" in config_to_modify:
            config_to_modify["tokenizer_path"] = tok_override
            logger.info(f"Overriding 'tokenizer_path' for '{model_id_key}' to: {tok_override}")
        elif "oute_llamacpp" == model_id_key and "tokenizer_path" in config_to_modify: # If LlamaCPP backend for OuteTTS uses separate tokenizer repo
            config_to_modify["tokenizer_path"] = tok_override
            logger.info(f"Overriding 'tokenizer_path' for '{model_id_key}' to: {tok_override}")


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

    for model_id, config_entry_original in GERMAN_TTS_MODELS.items():
        # Determine the key for ALL_HANDLERS map
        handler_key = config_entry_original.get("handler_function_key", model_id)
        handler_func = ALL_HANDLERS.get(handler_key)
        
        if not handler_func:
            logger.warning(f"\n>>> No handler found for Model ID: {model_id} (handler key: {handler_key}). Skipping. <<<")
            logger.info("------------------------------------")
            continue
            
        # Apply CLI overrides for this specific model_id to a copy of its config
        current_config_for_handler = _apply_cli_overrides_to_config(config_entry_original, model_id, cli_args)
        
        # API URL overrides specific to test_all (already in your script)
        if model_id == "orpheus_lm_studio":
            current_config_for_handler["api_url"] = cli_args.lm_studio_api_url # Key used by orpheus_api_handler
            if cli_args.gguf_model_name_in_api:
                current_config_for_handler["gguf_model_name_in_api"] = cli_args.gguf_model_name_in_api
        elif model_id == "orpheus_ollama":
            current_config_for_handler["api_url"] = cli_args.ollama_api_url # Key used by orpheus_api_handler
            if cli_args.ollama_model_name:
                current_config_for_handler["ollama_model_name"] = cli_args.ollama_model_name
        # Piper repo override is now part of _apply_cli_overrides_to_config


        voices_to_test_this_run = []
        if test_all_speakers_flag:
            if current_config_for_handler.get("available_voices"):
                voices_to_test_this_run.extend(current_config_for_handler.get("available_voices"))
            if "oute" in model_id and current_config_for_handler.get("test_default_speakers"):
                voices_to_test_this_run.extend(current_config_for_handler.get("test_default_speakers"))
            if not voices_to_test_this_run:
                default_v = current_config_for_handler.get('default_voice_id') or \
                            current_config_for_handler.get('default_model_path_in_repo') or \
                            str(current_config_for_handler.get('default_speaker_embedding_index', '')) or \
                            str(current_config_for_handler.get('default_speaker_id', ''))
                if default_v: voices_to_test_this_run.append(default_v)
        else:
            default_v = current_config_for_handler.get('default_voice_id') or \
                        current_config_for_handler.get('default_model_path_in_repo') or \
                        str(current_config_for_handler.get('default_speaker_embedding_index', '')) or \
                        str(current_config_for_handler.get('default_speaker_id', ''))
            if default_v: voices_to_test_this_run.append(default_v)

        voices_to_test_this_run = [v for v in voices_to_test_this_run if v is not None and str(v).strip()]
        seen_voices = set()
        unique_voices_to_test = [v for v in voices_to_test_this_run if not (str(v) in seen_voices or seen_voices.add(str(v)))]

        if not unique_voices_to_test:
            if model_id.startswith("oute") and Path("./german.wav").exists() and str(Path("./german.wav")) not in voices_to_test_this_run :
                logger.info(f"Model '{model_id}' - No specific voices, adding ./german.wav for test.")
                unique_voices_to_test.append(str(Path("./german.wav")))
            if not unique_voices_to_test: # Check again
                logger.info(f"\n>>> Skipping Model: {model_id} (No voices to test configured/found) <<<")
                logger.info("------------------------------------")
                continue
        
        for voice_id_for_test in unique_voices_to_test:
            speaker_suffix_for_file = ""
            if test_all_speakers_flag and len(unique_voices_to_test) > 1:
                sanitized_voice_id = str(voice_id_for_test).replace('/', '_').replace('\\','_').replace(':','-')
                sanitized_voice_id = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in sanitized_voice_id)
                speaker_suffix_for_file = f"_voice_{sanitized_voice_id[:30]}"

            output_suffix = ".wav" # Default to WAV for most local models
            if model_id == "edge": output_suffix = ".mp3"

            output_filename = base_output_dir / f"test_output_{model_id.replace('/', '_').replace(':','-')}{speaker_suffix_for_file}{output_suffix}"

            logger.info(f"\n>>> Testing Model: {model_id} (Voice/Speaker: {voice_id_for_test}) <<<")
            
            try:
                handler_func(
                    current_config_for_handler, # Pass the potentially modified config
                    text_to_synthesize,
                    str(voice_id_for_test),
                    cli_args.model_params, # Pass general model_params from CLI
                    str(output_filename),
                    False # play_direct is false for test_all
                )
                if output_filename.exists() and output_filename.stat().st_size > 100: # Basic check
                    logger.info(f"SUCCESS: Output for {model_id} (Voice: {voice_id_for_test}) saved to {output_filename}")
                else:
                    logger.warning(f"NOTE: Synthesis for {model_id} (Voice: {voice_id_for_test}) ran. Output file '{output_filename}' not created or is empty/too small.")
            except Exception as e_test_model:
                logger.error(f"ERROR: Testing {model_id} (Voice: {voice_id_for_test}) failed: {e_test_model}", exc_info=True)
            
            if test_all_speakers_flag and len(unique_voices_to_test) > 1 :
                logger.info("---")
        logger.info("------------------------------------")
    logger.info("--- Test for All Models Finished ---")


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

    # Apply CLI overrides to a copy of the base config
    current_config_for_handler = _apply_cli_overrides_to_config(model_config_base, args.model_id, args)

    # API URL overrides (specific to these models)
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
        effective_voice_id = default_v if str(default_v).strip() else None
    
    if not effective_voice_id:
        logger.error(f"No voice ID specified and no default could be determined for model {args.model_id}.")
        return

    handler_key = current_config_for_handler.get("handler_function_key", args.model_id)
    handler_func = ALL_HANDLERS.get(handler_key)

    if handler_func:
        try:
            handler_func(
                current_config_for_handler,
                text_to_synthesize,
                str(effective_voice_id),
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
    action_group.add_argument("--test-all", action="store_true", help="Test all models with default voices using --input-text or --input-file.")
    action_group.add_argument("--test-all-speakers", action="store_true", help="Test all models with ALL configured voices using --input-text or --input-file.")

    synth_group = parser.add_argument_group(title="Synthesis Options")
    input_group.add_argument("--input-text", type=str, help="Text to synthesize.")
    input_group.add_argument("--input-file", type=str, help="Path to input file (txt, md, html, pdf, epub).")
    
    synth_group.add_argument("--model-id", type=str, choices=list(GERMAN_TTS_MODELS.keys()) + [None], default=None, 
                             help="Select TTS model ID. Required for synthesis if not using an action flag.")
    synth_group.add_argument("--output-file", type=str, help="Path to save synthesized audio.")
    synth_group.add_argument("--output-dir", type=str, default="tts_test_outputs", help="Directory for --test-all* outputs.")
    synth_group.add_argument("--play-direct", action="store_true", help="Play audio directly (not with --test-all*).")
    synth_group.add_argument("--german-voice-id", type=str, help="Override default voice/speaker for the selected model.")
    synth_group.add_argument("--model-params", type=str, help="JSON string of model-specific parameters (e.g., '{\"temperature\":0.7}').")
    
    parser.add_argument(
        "--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set console logging level (default: INFO)."
    )

    # --- Override Arguments ---
    override_group = parser.add_argument_group(title="Runtime Model Path/Repo Overrides (for selected --model-id)")
    override_group.add_argument("--override-main-model-repo", type=str, metavar="REPO_OR_PATH",
                                help="Override main model repository ID or path (e.g., GGUF repo, ONNX repo, SpeechT5 base ID, MLX ID).")
    override_group.add_argument("--override-model-filename", type=str, metavar="FILENAME",
                                help="Override specific model filename within the main repo (e.g., for GGUF, NeMo).")
    override_group.add_argument("--override-tokenizer-repo", type=str, metavar="REPO_OR_PATH",
                                help="Override tokenizer repository ID or path (e.g., for OuteTTS, MLX).")
    override_group.add_argument("--override-vocoder-repo", type=str, metavar="REPO_OR_NAME",
                                help="Override vocoder repository ID or name (e.g., for SpeechT5, NeMo).")
    override_group.add_argument("--override-speaker-embed-repo", type=str, metavar="REPO_ID",
                                help="Override speaker embeddings repository ID (e.g., for SpeechT5).")
    override_group.add_argument("--override-piper-voices-repo", type=str, metavar="REPO_ID",
                                help="Override main repository ID for Piper voices.")

    api_group = parser.add_argument_group(title="API Backend Overrides (also in config.py)")
    api_group.add_argument("--lm-studio-api-url", type=str, default=LM_STUDIO_API_URL_DEFAULT, help=f"Override LM Studio API URL.")
    api_group.add_argument("--gguf-model-name-in-api", type=str, help="Override model name for LM Studio API.")
    api_group.add_argument("--ollama-api-url", type=str, default=OLLAMA_API_URL_DEFAULT, help=f"Override Ollama API URL.")
    api_group.add_argument("--ollama-model-name", type=str, help="Override model name/tag for Ollama.")

    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel.upper())
    logger.info(f"Logging level set to: {args.loglevel.upper()}")

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
        elif args.model_id:
             logger.error("No text input provided for synthesis via --input-text or --input-file.")
        else: parser.print_help()
        return

    if args.test_all or args.test_all_speakers:
        test_text = text_to_process[:500] if len(text_to_process) > 500 else text_to_process
        logger.info(f"--- Applying Test Mode: {'All Speakers' if args.test_all_speakers else 'Default Speaker Only'} ---")
        if args.override_main_model_repo or args.override_tokenizer_repo or args.override_vocoder_repo or args.override_speaker_embed_repo or args.override_piper_voices_repo or args.override_model_filename:
            logger.warning("CLI repo/path overrides are active and will apply to all compatible models during --test-all(-speakers). This might not be intended for all models.")
        test_all_models(test_text, args.output_dir, args)
        return

    if not args.model_id:
        parser.error("A --model-id is required for synthesis if not using an action flag.")
        return
    
    run_synthesis(args)


if __name__ == "__main__":
    # Initial logging (before CLI parsing can change loglevel for these specific messages)
    if any(GERMAN_TTS_MODELS[m].get("requires_hf_token", False) for m in GERMAN_TTS_MODELS if m in GERMAN_TTS_MODELS) and not os.getenv("HF_TOKEN"): # Check m in GERMAN_TTS_MODELS
        logging.info("Some models configured might require a Hugging Face token (HF_TOKEN env var).")

    if TORCH_AVAILABLE_MAIN:
        if torch.cuda.is_available(): device_message = f"PyTorch detected CUDA: {torch.cuda.get_device_name(0)}"
        elif IS_MPS_MAIN: device_message = "PyTorch detected MPS (Apple Metal)."
        else: device_message = "PyTorch detected CPU."
    else: device_message = "PyTorch not available."
    logging.info(device_message) # This will use root logger's default level before CLI parsing

    main_cli_entrypoint()