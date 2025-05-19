# handlers/orpheus_api_handler.py

import json
import logging
from pathlib import Path
import requests
import gc

# Use relative imports
from config import (
    ORPHEUS_SAMPLE_RATE, ORPHEUS_DEFAULT_VOICE, ORPHEUS_GERMAN_VOICES,
    SAUERKRAUT_VOICES, ORPHEUS_AVAILABLE_VOICES_BASE,
    LM_STUDIO_API_URL_DEFAULT, LM_STUDIO_HEADERS, OLLAMA_API_URL_DEFAULT
)
from utils import (
    play_audio, orpheus_format_prompt,
    _orpheus_master_token_processor_and_decoder
)

logger = logging.getLogger("CrispTTS.handlers.orpheus_api")

def synthesize_with_orpheus_lm_studio(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    voice = voice_id_override or model_config.get('default_voice_id', ORPHEUS_DEFAULT_VOICE)
    logger.debug(f"Orpheus LM Studio - Text: '{text[:50]}...', Voice: {voice}")

    # Get available voices from model_config or fallback to combined list from config.py
    available_voices = model_config.get('available_voices', ORPHEUS_GERMAN_VOICES + SAUERKRAUT_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE)
    api_url = model_config.get("api_url", LM_STUDIO_API_URL_DEFAULT) # Use specific from GERMAN_TTS_MODELS if set, else default

    cli_params = json.loads(model_params_override) if model_params_override else {}
    model_name_in_api = cli_params.get("gguf_model_name_in_api", model_config.get("gguf_model_name_in_api", "SauerkrautTTS-Preview-0.1"))
    
    # Use ORPHEUS_DEFAULT_VOICE from config.py as the ultimate fallback for formatting
    formatted_prompt = orpheus_format_prompt(text, voice, available_voices) # Util from utils.py
    logger.debug(f"Orpheus LM Studio - Formatted prompt for API: {formatted_prompt}")

    def _lm_studio_generate_raw_token_text_stream():
        logger.debug(f"Orpheus LM Studio - Requesting from API: {api_url} for model '{model_name_in_api}'")
        payload = {
            "model": model_name_in_api,
            "prompt": formatted_prompt,
            "max_tokens": cli_params.get("max_tokens", 1200), # Max tokens for the API response
            "temperature": cli_params.get("temperature", 0.5),
            "top_p": cli_params.get("top_p", 0.9),
            "repeat_penalty": cli_params.get("repetition_penalty", 1.1),
            "stream": True
        }
        logger.debug(f"Orpheus LM Studio - API Payload: {json.dumps(payload)}")
        response = None # Define for finally block
        try:
            response = requests.post(api_url, headers=LM_STUDIO_HEADERS, json=payload, stream=True, timeout=120)
            response.raise_for_status() # Check for HTTP errors
            
            full_raw_output_for_debug = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            logger.debug("Orpheus LM Studio - Stream [DONE] received.")
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices'] and 'text' in data['choices'][0]:
                                token_text_chunk = data['choices'][0].get('text', '')
                                full_raw_output_for_debug += token_text_chunk
                                if token_text_chunk:
                                    yield token_text_chunk
                        except json.JSONDecodeError:
                            logger.warning(f"Orpheus LM Studio - Could not decode JSON line: {data_str}")
            logger.debug(f"Orpheus LM Studio - Full raw API output (first 200 chars): '{full_raw_output_for_debug[:200]}{'...' if len(full_raw_output_for_debug)>200 else ''}'")
        except requests.exceptions.RequestException as e_req:
            logger.error(f"Orpheus LM Studio - API connection/request failed: {e_req}")
            # yield "" # Ensure generator terminates if error occurs before any yield
        except Exception as e_stream:
            logger.error(f"Orpheus LM Studio - Error processing stream: {e_stream}", exc_info=True)
        finally:
            if response:
                response.close() # Ensure the response is closed
            logger.debug("Orpheus LM Studio - API stream processing finished or errored.")

    effective_output_file_wav_str = str(Path(output_file_str).with_suffix(".wav")) if output_file_str else None
    audio_bytes = _orpheus_master_token_processor_and_decoder(
        _lm_studio_generate_raw_token_text_stream(),
        output_file_wav_str=effective_output_file_wav_str,
        orpheus_sample_rate=ORPHEUS_SAMPLE_RATE
    )

    if audio_bytes:
        if play_direct:
            play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=ORPHEUS_SAMPLE_RATE)
    else:
        logger.warning("Orpheus LM Studio - No audio bytes generated from API stream.")
    gc.collect()


def synthesize_with_orpheus_ollama(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    voice = voice_id_override or model_config.get('default_voice_id', ORPHEUS_DEFAULT_VOICE)
    logger.debug(f"Orpheus Ollama - Text: '{text[:50]}...', Voice: {voice}")

    available_voices = model_config.get('available_voices', ORPHEUS_GERMAN_VOICES + SAUERKRAUT_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE)
    formatted_prompt = orpheus_format_prompt(text, voice, available_voices) # Util from utils.py

    cli_params = json.loads(model_params_override) if model_params_override else {}
    ollama_model_name = cli_params.get("ollama_model_name", model_config.get("ollama_model_name", "orpheus-german-tts:latest"))
    ollama_api_url = cli_params.get("ollama_api_url", model_config.get("api_url", OLLAMA_API_URL_DEFAULT))

    if "USER MUST SET" in ollama_model_name or not ollama_model_name: # Check from config.py
        logger.error(f"Orpheus Ollama - Model name not configured properly (current: '{ollama_model_name}').")
        return

    def _ollama_text_stream_generator():
        logger.debug(f"Orpheus Ollama - Requesting from API: {ollama_api_url} for model: {ollama_model_name}")
        payload = {
            "model": ollama_model_name,
            "prompt": formatted_prompt,
            "stream": True,
            "options": {
                "temperature": cli_params.get("temperature", 0.5),
                "top_p": cli_params.get("top_p", 0.9),
                "repeat_penalty": cli_params.get("repeat_penalty", 1.1)
            }
        }
        logger.debug(f"Orpheus Ollama - API Payload: {json.dumps(payload, indent=2)}")
        response = None
        try:
            response = requests.post(ollama_api_url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            full_raw_output_for_debug = ""
            for line in response.iter_lines():
                if line:
                    try:
                        line_str = line.decode('utf-8')
                        data = json.loads(line_str)
                        if data.get("error"): logger.error(f"Orpheus Ollama - API error: {data['error']}"); break
                        token_text_chunk = data.get('response', '') # Ollama uses 'response'
                        full_raw_output_for_debug += token_text_chunk
                        if token_text_chunk: yield token_text_chunk
                        if data.get("done", False) and data.get("done") is True: # Explicitly check for True
                            logger.debug("Orpheus Ollama - Ollama API stream 'done'.")
                            break
                    except json.JSONDecodeError: logger.warning(f"Orpheus Ollama - Could not decode JSON line: {line_str}")
            logger.debug(f"Orpheus Ollama - Full raw API output (first 200 chars): '{full_raw_output_for_debug[:200]}{'...' if len(full_raw_output_for_debug)>200 else ''}'")
        except requests.exceptions.RequestException as e_req:
            logger.error(f"Orpheus Ollama - API connection/request failed: {e_req}")
            if hasattr(e_req, 'response') and e_req.response is not None:
                 logger.error(f"Ollama Response status: {e_req.response.status_code}, Text: {e_req.response.text[:500]}")
            # yield "" # Ensure generator terminates
        except Exception as e_stream:
            logger.error(f"Orpheus Ollama - Error processing stream: {e_stream}", exc_info=True)
        finally:
            if response: response.close()
            logger.debug("Orpheus Ollama - API stream processing finished or errored.")

    effective_output_file_wav_str = str(Path(output_file_str).with_suffix(".wav")) if output_file_str else None
    audio_bytes = _orpheus_master_token_processor_and_decoder(
        _ollama_text_stream_generator(),
        output_file_wav_str=effective_output_file_wav_str,
        orpheus_sample_rate=ORPHEUS_SAMPLE_RATE
    )

    if audio_bytes:
        if play_direct: play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=ORPHEUS_SAMPLE_RATE)
    else:
        logger.warning("Orpheus Ollama - No audio bytes generated from API stream.")
    gc.collect()