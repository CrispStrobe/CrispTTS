#!/usr/bin/env python3
# CrispTTS - main.py
# Main Command-Line Interface for the Text-to-Speech Synthesizer (Modularized with Overrides)

import sys
from pathlib import Path
import time # For benchmarking

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
os.environ["GGML_METAL_NDEBUG"] = "1" 

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
        PYDUB_AVAILABLE as UTILS_PYDUB_AVAILABLE, # Get availability from utils
        SOUNDFILE_AVAILABLE as UTILS_SOUNDFILE_AVAILABLE # Get availability from utils
    )
    from handlers import ALL_HANDLERS 
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import from project modules (config, utils, handlers): {e}", file=sys.stderr)
    print("Please ensure these modules/packages are correctly structured and in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)
except KeyError as e_key: # Should not happen if __init__.py in handlers is correct
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
    # ... (this function remains the same as in your provided code) ...
    config_to_modify = model_config_dict.copy() 

    if cli_args.override_main_model_repo:
        repo_override = cli_args.override_main_model_repo
        updated = False
        if model_id_key in ["orpheus_lex_au", "orpheus_sauerkraut"] and "model_repo_id" in config_to_modify:
            config_to_modify["model_repo_id"] = repo_override; updated = True
        elif model_id_key == "piper_local" and "piper_voice_repo_id" in config_to_modify and not cli_args.override_piper_voices_repo:
            config_to_modify["piper_voice_repo_id"] = repo_override; updated = True
        elif model_id_key == "oute_hf" and "onnx_repo_id" in config_to_modify: # Assuming oute_hf can have its onnx_repo_id overridden
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
        # OuteTTS and some MLX configurations might use 'tokenizer_path' or 'tokenizer_path_for_mlx_outetts'
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

    benchmark_results = [] # For storing benchmark data

    for model_id, config_entry_original in GERMAN_TTS_MODELS.items():
        handler_key = config_entry_original.get("handler_function_key", model_id)
        handler_func = ALL_HANDLERS.get(handler_key)
        
        # Initialize benchmark fields for this model_id (will be refined per voice)
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
            # This part is for --test-all-speakers, uses "available_voices"
            if current_config_for_handler.get("available_voices"):
                voices_to_test_this_run.extend(current_config_for_handler.get("available_voices"))
            if "oute" in model_id and current_config_for_handler.get("test_default_speakers"):
                voices_to_test_this_run.extend(current_config_for_handler.get("test_default_speakers"))
            
            # Fallback to a single default if --test-all-speakers found no specific voices
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
                    # If --test-all-speakers but Coqui only has "default_speaker", use it
                    voices_to_test_this_run.append("default_speaker")
                    logger.info(f"Coqui single-speaker model '{model_id}' for --test-all-speakers, using placeholder 'default_speaker'.")


        else: # This is for --test-all (test_all_speakers_flag is False), test only default voice
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
            # **** START MODIFICATION FOR COQUI DEFAULT TEST ****
            elif model_id.startswith("coqui_tts") and current_config_for_handler.get("default_coqui_speaker") is None:
                # For single-speaker Coqui models in --test-all mode,
                # if available_voices is ["default_speaker"], use that string.
                # The handler will interpret "default_speaker" as "use intrinsic speaker".
                if current_config_for_handler.get("available_voices") == ["default_speaker"]:
                    voices_to_test_this_run.append("default_speaker")
                    logger.info(f"Coqui single-speaker model '{model_id}', using placeholder 'default_speaker' for default test run.")
            # **** END MODIFICATION FOR COQUI DEFAULT TEST ****
        
        # Ensure unique_voices_to_test is populated correctly:
        voices_to_test_this_run = [v for v in voices_to_test_this_run if v is not None and str(v).strip()] # Filter out None or empty strings
        seen_voices = set()
        unique_voices_to_test = []
        if voices_to_test_this_run: # Only proceed if there's something to make unique
            unique_voices_to_test = [v for v in voices_to_test_this_run if not (str(v) in seen_voices or seen_voices.add(str(v)))]

        if not unique_voices_to_test:
            # This block is now the final check if NO voices were found by any logic above
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

                if output_filename.exists() and output_filename.stat().st_size > 100: # Basic check for output
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
                if current_gen_time_sec is None: # Ensure gen_time is recorded even if error occurs mid-handler
                    current_gen_time_sec = time.time() - start_time
                current_model_status = "ERROR"
                # Log full error for main log, but not for benchmark summary for cleaner table
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
                logger.info("---") # Separator between voices for the same model
        logger.info("------------------------------------")

    # --- Print Benchmark Table ---
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
    # ... (this function remains the same) ...
    text_to_synthesize = get_text_from_input(args.input_text, args.input_file)
    if not text_to_synthesize:
        if not args.input_text and not args.input_file: # Only error if neither was given for explicit synthesis
            logger.error("No input provided. Use --input-text or --input-file for synthesis.")
        return # If text extraction failed, get_text_from_input logs it.

    # Limit text length for individual synthesis too, if desired
    text_to_synthesize = text_to_synthesize[:3000] # Example limit

    if not args.model_id: # Should be caught by argparser if not test_all etc.
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
    if not effective_voice_id: # Determine default if not overridden
        default_v = current_config_for_handler.get('default_voice_id') or \
                    current_config_for_handler.get('default_model_path_in_repo') or \
                    str(current_config_for_handler.get('default_speaker_embedding_index', '')) or \
                    str(current_config_for_handler.get('default_speaker_id', ''))
        # Handle cases where default_v might be an empty string after str() conversion
        effective_voice_id = default_v if (isinstance(default_v, Path) or (isinstance(default_v, str) and default_v.strip())) else None


    if not effective_voice_id and not (args.model_id.startswith("coqui_tts") and current_config_for_handler.get("default_coqui_speaker") is None):
        # Coqui single-speaker models might not need an explicit voice_id if default_coqui_speaker is None
        # and the handler can cope.
        logger.error(f"No voice ID specified and no default could be determined for model {args.model_id}.")
        return

    handler_key = current_config_for_handler.get("handler_function_key", args.model_id)
    handler_func = ALL_HANDLERS.get(handler_key)

    if handler_func:
        try:
            handler_func(
                current_config_for_handler,
                text_to_synthesize,
                str(effective_voice_id) if effective_voice_id is not None else None, # Pass None if no voice ID
                args.model_params,
                args.output_file,
                args.play_direct
            )
        except Exception as e_synth:
            logger.error(f"Synthesis failed for model {args.model_id}: {e_synth}", exc_info=True)
    else:
        logger.error(f"No synthesis handler function found for model ID: {args.model_id} (handler key: {handler_key})")


def main_cli_entrypoint():
    # ... (argparse setup remains the same) ...
    parser = argparse.ArgumentParser(
        description="CrispTTS: Modular German Text-to-Speech Synthesizer",
        formatter_class=argparse.RawTextHelpFormatter # Keep this for good help text
    )
    action_group = parser.add_argument_group(title="Primary Actions")
    # input_group requires one or the other for synthesis/test, but not for list/info
    input_group = parser.add_mutually_exclusive_group(required=False) 

    action_group.add_argument("--list-models", action="store_true", help="List all configured TTS models.")
    action_group.add_argument("--voice-info", type=str, metavar="MODEL_ID", help="Display voice/speaker info for a specific MODEL_ID.")
    action_group.add_argument("--test-all", action="store_true", help="Test all models with default voices. Requires --input-text or --input-file.")
    action_group.add_argument("--test-all-speakers", action="store_true", help="Test all models with ALL configured voices. Requires --input-text or --input-file.")

    synth_group = parser.add_argument_group(title="Synthesis Options (used with --model-id or --test-all*)")
    input_group.add_argument("--input-text", type=str, help="Text to synthesize.")
    input_group.add_argument("--input-file", type=str, help="Path to input file (txt, md, html, pdf, epub).")
    
    synth_group.add_argument("--model-id", type=str, choices=list(GERMAN_TTS_MODELS.keys()) + [None], default=None,  
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
    # ... (override arguments remain the same) ...
    override_group.add_argument("--override-main-model-repo", type=str, metavar="REPO_OR_PATH", help="Override main model repository ID or path.")
    override_group.add_argument("--override-model-filename", type=str, metavar="FILENAME", help="Override specific model filename within the main repo.")
    override_group.add_argument("--override-tokenizer-repo", type=str, metavar="REPO_OR_PATH", help="Override tokenizer repository ID or path.")
    override_group.add_argument("--override-vocoder-repo", type=str, metavar="REPO_OR_NAME", help="Override vocoder repository ID or name.")
    override_group.add_argument("--override-speaker-embed-repo", type=str, metavar="REPO_ID", help="Override speaker embeddings repository ID.")
    override_group.add_argument("--override-piper-voices-repo", type=str, metavar="REPO_ID", help="Override main repository ID for Piper voices.")


    api_group = parser.add_argument_group(title="API Backend Overrides (also in config.py)")
    # ... (API arguments remain the same) ...
    api_group.add_argument("--lm-studio-api-url", type=str, default=LM_STUDIO_API_URL_DEFAULT, help=f"Override LM Studio API URL (default: {LM_STUDIO_API_URL_DEFAULT}).")
    api_group.add_argument("--gguf-model-name-in-api", type=str, help="Override model name for LM Studio API (from config or this flag).")
    api_group.add_argument("--ollama-api-url", type=str, default=OLLAMA_API_URL_DEFAULT, help=f"Override Ollama API URL (default: {OLLAMA_API_URL_DEFAULT}).")
    api_group.add_argument("--ollama-model-name", type=str, help="Override model name/tag for Ollama (from config or this flag).")


    args = parser.parse_args()

    # Setup root logger
    # Basic config for the root logger. Individual module loggers will inherit this.
    # Format includes the logger name to identify messages from different modules.
    logging.basicConfig(
        level=args.loglevel.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Re-apply level to the root logger specifically, in case it was already configured by an import
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

    # Input text is required for --test-all, --test-all-speakers, or direct synthesis with --model-id
    text_to_process = get_text_from_input(args.input_text, args.input_file)
    if not text_to_process:
        if args.test_all or args.test_all_speakers:
            parser.error("--test-all or --test-all-speakers requires --input-text or --input-file.")
        elif args.model_id : # Only if --model-id was specified for synthesis
             logger.error("No text input provided for synthesis via --input-text or --input-file.")
        else: # No action flag and no model_id for synthesis
            parser.print_help()
        return

    if args.test_all or args.test_all_speakers:
        test_text = text_to_process[:500] if len(text_to_process) > 500 else text_to_process # Limit test text length
        logger.info(f"--- Applying Test Mode: {'All Speakers' if args.test_all_speakers else 'Default Speaker Only'} ---")
        if args.override_main_model_repo or args.override_tokenizer_repo or args.override_vocoder_repo or args.override_speaker_embed_repo or args.override_piper_voices_repo or args.override_model_filename:
            logger.warning("CLI repo/path overrides are active and will apply to all compatible models during --test-all(-speakers). This might not be intended for all models.")
        test_all_models(test_text, args.output_dir, args)
        return

    if not args.model_id: # If not test_all and no model_id, it's an invalid combo
        parser.error("A --model-id is required for synthesis if not using an action flag like --list-models or --test-all.")
        return
    
    run_synthesis(args)


if __name__ == "__main__":
    # These initial logs will use the default root logger level until CLI parsing sets it.
    # We now configure basicConfig inside main_cli_entrypoint after args are parsed.
    # For very early messages before that, they might have a different default format/level.
    
    # Pre-flight checks / Info messages (will use root logger's default before main_cli_entrypoint config)
    # if any(GERMAN_TTS_MODELS[m].get("requires_hf_token", False) for m in GERMAN_TTS_MODELS if m in GERMAN_TTS_MODELS):
    #     if not os.getenv("HF_TOKEN"):
    #         logging.info("Hint: Some models configured might require a Hugging Face token (HF_TOKEN env var).")

    # device_message = "PyTorch "
    # if TORCH_AVAILABLE_MAIN:
    #     if torch.cuda.is_available(): device_message += f"detected CUDA: {torch.cuda.get_device_name(0)}"
    #     elif IS_MPS_MAIN: device_message += "detected MPS (Apple Metal)."
    #     else: device_message += "detected CPU."
    # else: 
    #     device_message += "not available."
    # logging.info(device_message)
    
    main_cli_entrypoint()