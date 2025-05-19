# handlers/orpheus_gguf_handler.py

import json
import logging
import os
from pathlib import Path
import gc
import sys # ADDED for sys.modules check

# Use relative imports for project modules
from config import ORPHEUS_SAMPLE_RATE, ORPHEUS_DEFAULT_VOICE, ORPHEUS_GERMAN_VOICES # Import constants
from utils import (
    SuppressOutput, play_audio,
    orpheus_format_prompt, _orpheus_master_token_processor_and_decoder
)

logger = logging.getLogger("CrispTTS.handlers.orpheus_gguf")

# Conditional imports for LlamaCPP and HuggingFace Hub
LLAMA_CPP_AVAILABLE_IN_HANDLER = False # Corrected variable name (singular)
LlamaForHandler = None
HF_HUB_AVAILABLE_IN_HANDLER = False # Corrected variable name (singular)
hf_hub_download_h = None
IS_MPS_FOR_HANDLER_GGUF = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE_IN_HANDLER = True # Corrected
    hf_hub_download_h = hf_hub_download
except ImportError:
    logger.warning("huggingface_hub not installed. Orpheus GGUF model downloading will fail.")

if HF_HUB_AVAILABLE_IN_HANDLER: # LlamaCPP is useful only if models can be fetched
    try:
        from llama_cpp import Llama as Llama_imp
        import llama_cpp as llama_cpp_module
        LlamaForHandler = Llama_imp
        LLAMA_CPP_AVAILABLE_IN_HANDLER = True # Corrected
        try:
            import torch
            IS_MPS_FOR_HANDLER_GGUF = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            pass
    except ImportError:
        logger.info("llama-cpp-python not installed. Orpheus GGUF handler will not be functional.")


def synthesize_with_orpheus_gguf_local(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    if not LLAMA_CPP_AVAILABLE_IN_HANDLER or not LlamaForHandler: # Corrected
        logger.error("llama-cpp-python not available. Skipping Orpheus GGUF local synthesis.")
        return
    if not HF_HUB_AVAILABLE_IN_HANDLER or not hf_hub_download_h: # Corrected
        logger.error("huggingface_hub not available. Cannot download Orpheus GGUF models. Skipping.")
        return

    voice = voice_id_override or model_config.get('default_voice_id', ORPHEUS_DEFAULT_VOICE)
    logger.debug(f"Orpheus GGUF Local - Text: '{text[:50]}...', Voice: {voice}")

    model_repo_id = model_config.get("model_repo_id")
    model_filename = model_config.get("model_filename")

    if not model_repo_id or not model_filename:
        logger.error("Orpheus GGUF - model_repo_id or model_filename missing in configuration.")
        return

    model_cache_dir = Path.home() / ".cache" / "crisptts_gguf_models"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = model_cache_dir / model_filename
    llm = None

    try:
        if not local_model_path.exists():
            logger.info(f"Orpheus GGUF - Downloading {model_filename} from {model_repo_id}...")
            hf_hub_download_h(
                repo_id=model_repo_id, filename=model_filename,
                local_dir=str(model_cache_dir), token=os.getenv("HF_TOKEN"), repo_type="model"
            )
        logger.info(f"Orpheus GGUF - Model found/downloaded: {local_model_path}")

        available_voices = model_config.get('available_voices', ORPHEUS_GERMAN_VOICES)
        formatted_prompt = orpheus_format_prompt(text, voice, available_voices)

        cli_params = json.loads(model_params_override) if model_params_override else {}
        temperature = cli_params.get("temperature", 0.5)
        top_p = cli_params.get("top_p", 0.9)
        max_tokens_gen = cli_params.get("max_tokens", 3072)
        repetition_penalty = cli_params.get("repetition_penalty", 1.1)
        n_gpu_layers = cli_params.get("n_gpu_layers", -1 if IS_MPS_FOR_HANDLER_GGUF else 0)

        logger.info("Orpheus GGUF - Loading GGUF model (C-level output suppressed by global handler and verbose=False)...")
        with SuppressOutput(suppress_stdout=True, suppress_stderr=True):
            llm = LlamaForHandler(
                model_path=str(local_model_path),
                verbose=False,
                n_gpu_layers=n_gpu_layers,
                n_ctx=2048,
                logits_all=True
            )
        logger.info(f"Orpheus GGUF - Model loaded: {local_model_path}")

        def _llama_cpp_text_stream_generator_local():
            logger.debug(f"Orpheus GGUF Local - llama.cpp prompt: {formatted_prompt}")
            stream = llm.create_completion(
                prompt=formatted_prompt, max_tokens=max_tokens_gen,
                temperature=temperature, top_p=top_p,
                repeat_penalty=repetition_penalty, stream=True
            )
            full_raw_output_for_debug = ""
            for output_chunk in stream:
                token_text = output_chunk['choices'][0]['text']
                full_raw_output_for_debug += token_text
                yield token_text
            logger.debug(f"Orpheus GGUF Local - Full raw output (first 200): '{full_raw_output_for_debug[:200]}{'...' if len(full_raw_output_for_debug)>200 else ''}'")
            logger.debug("Orpheus GGUF Local - llama-cpp-python stream finished.")

        effective_output_file_wav_str = str(Path(output_file_str).with_suffix(".wav")) if output_file_str else None
        
        audio_bytes = _orpheus_master_token_processor_and_decoder(
            _llama_cpp_text_stream_generator_local(),
            output_file_wav_str=effective_output_file_wav_str
            # REMOVED: orpheus_sample_rate=ORPHEUS_SAMPLE_RATE
        )

        if audio_bytes:
            if play_direct:
                if effective_output_file_wav_str and Path(effective_output_file_wav_str).exists():
                    play_audio(effective_output_file_wav_str, is_path=True)
                else:
                    play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=ORPHEUS_SAMPLE_RATE)
        else:
            if effective_output_file_wav_str and Path(effective_output_file_wav_str).exists():
                Path(effective_output_file_wav_str).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Orpheus GGUF - Synthesis failed: {e}", exc_info=True)
    finally:
        if llm is not None:
            del llm
            logger.debug("Orpheus GGUF - Llama model object deleted.")
        gc.collect()
        # Corrected variable name LLAMA_CPP_AVAILABLE_IN_HANDLER
        if LLAMA_CPP_AVAILABLE_IN_HANDLER and 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            sys.modules['torch'].cuda.empty_cache()
            logger.debug("Orpheus GGUF - Cleared CUDA cache (if torch was used by LlamaCPP).")