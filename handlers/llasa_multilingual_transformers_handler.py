# handlers/llasa_multilingual_transformers_handler.py
# Follows HKUSTAudio/Llasa-1B-Multilingual & MultiLlasa/Llasa-1B-Multilingual-German repository implementation examples.

import logging
import os
from pathlib import Path
import gc
import json
import tempfile # For temporary audio file for Whisper
from typing import Optional, List, Tuple # For type hinting

# Conditional imports
TORCH_AVAILABLE = False
torch_llasa_multi = None
TORCHAUDIO_AVAILABLE = False
torchaudio_llasa_multi = None
IS_MPS_LLASA_MULTI = False
IS_CUDA_LLASA_MULTI = False

TRANSFORMERS_AVAILABLE = False
AutoTokenizer_llasa_multi, AutoModelForCausalLM_llasa_multi, pipeline_llasa_multi = None, None, None

XCODEC2_AVAILABLE = False
XCodec2Model_llasa_multi = None

SOUNDFILE_AVAILABLE = False
sf_llasa_multi = None

NUMPY_AVAILABLE = False
numpy_llasa_multi = None

# CrispTTS Utils
from utils import save_audio, play_audio, SuppressOutput

logger_init = logging.getLogger("CrispTTS.handlers.llasa_multilingual_transformers.init")
logger = logging.getLogger("CrispTTS.handlers.llasa_multilingual_transformers")

try:
    import torch
    torch_llasa_multi = torch
    TORCH_AVAILABLE = True
    if hasattr(torch_llasa_multi.backends, "mps") and torch_llasa_multi.backends.mps.is_available():
        IS_MPS_LLASA_MULTI = True
    if torch_llasa_multi.cuda.is_available():
        IS_CUDA_LLASA_MULTI = True
    logger_init.info("PyTorch loaded for LLaSA Multilingual Transformers handler.")

    import torchaudio
    torchaudio_llasa_multi = torchaudio
    TORCHAUDIO_AVAILABLE = True
    logger_init.info("Torchaudio loaded for LLaSA Multilingual Transformers handler (ref audio processing).")

except ImportError:
    logger_init.warning("PyTorch or Torchaudio not found. LLaSA Multilingual Transformers handler will be non-functional.")

if TORCH_AVAILABLE: # Only attempt these if torch is available
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        AutoTokenizer_llasa_multi = AutoTokenizer
        AutoModelForCausalLM_llasa_multi = AutoModelForCausalLM
        pipeline_llasa_multi = pipeline # Assign the imported pipeline function
        TRANSFORMERS_AVAILABLE = True
        logger_init.info("Transformers (AutoTokenizer, AutoModelForCausalLM, pipeline) loaded.")
    except ImportError:
        TRANSFORMERS_AVAILABLE = False # Explicitly set to False on import error
        pipeline_llasa_multi = None    # Ensure it's None
        logger_init.warning("Transformers library (or its `pipeline` component) not found. Transcription will be unavailable.")

    try:
        from xcodec2.modeling_xcodec2 import XCodec2Model
        XCodec2Model_llasa_multi = XCodec2Model
        XCODEC2_AVAILABLE = True
        logger_init.info("XCodec2 loaded for LLaSA Multilingual Transformers handler.")
    except ImportError:
        XCODEC2_AVAILABLE = False # Explicitly set
        logger_init.warning("XCodec2 library not found. LLaSA Multilingual Transformers handler will be non-functional.")

try:
    import soundfile as sf
    sf_llasa_multi = sf
    SOUNDFILE_AVAILABLE = True
    logger_init.info("SoundFile loaded for LLaSA Multilingual Transformers handler (saving output).")
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger_init.warning("SoundFile library not found. Saving audio output might fail or use alternatives.")

try:
    import numpy as np
    numpy_llasa_multi = np
    NUMPY_AVAILABLE = True
    logger_init.info("NumPy loaded.")
except ImportError:
    NUMPY_AVAILABLE = False
    logger_init.warning("NumPy not found. Audio data manipulation might fail.")

# Global cache for Whisper pipeline
_llasa_multilingual_whisper_pipeline_cache = {}


def _ids_to_speech_tokens_llasa_multi(speech_ids: list[int]) -> list[str]:
    speech_tokens_str_list = []
    for speech_id in speech_ids:
        speech_tokens_str_list.append(f"<|s_{speech_id}|>")
    return speech_tokens_str_list

def _extract_speech_ids_llasa_multi(token_strings_list: list[str]) -> list[int]:
    numeric_speech_ids = []
    if not token_strings_list:
        return numeric_speech_ids
    for token_str in token_strings_list:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num_str_content = token_str[4:-2]
                num = int(num_str_content)
                numeric_speech_ids.append(num)
            except ValueError:
                logger.debug(f"LLaSA Multi: Could not parse speech ID from token: '{token_str}'")
    return numeric_speech_ids

def _prepare_audio_for_whisper_local(
    audio_path_str: str,
    target_sr: int = 16000,
    log_prefix: str = "LLaSA Multi (Whisper Prep): "
) -> Tuple[Optional[Path], Optional[str]]:
    if not (TORCHAUDIO_AVAILABLE and torchaudio_llasa_multi and TORCH_AVAILABLE and torch_llasa_multi):
        return None, "Torchaudio/PyTorch not available for audio preparation."

    original_audio_path = Path(audio_path_str)
    if not original_audio_path.exists() or not original_audio_path.is_file():
        return None, f"Audio file for Whisper prep not found: {original_audio_path}"

    temp_wav_file: Optional[Path] = None
    try:
        logger.debug(f"{log_prefix}Preparing '{original_audio_path}' for Whisper (target SR: {target_sr}Hz).")
        waveform, sr = torchaudio_llasa_multi.load(str(original_audio_path))
        logger.debug(f"{log_prefix}Loaded '{original_audio_path}'. Original SR: {sr}, Shape: {waveform.shape}")

        if sr != target_sr:
            logger.debug(f"{log_prefix}Resampling from {sr}Hz to {target_sr}Hz.")
            resampler = torchaudio_llasa_multi.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            logger.debug(f"{log_prefix}Converting to mono from {waveform.size(0)} channels.")
            waveform = torch_llasa_multi.mean(waveform, dim=0, keepdim=True)

        fd, temp_wav_path_str = tempfile.mkstemp(suffix=".wav", prefix="crisptts_llasa_whisper_prep_")
        os.close(fd)
        temp_wav_file = Path(temp_wav_path_str)
        torchaudio_llasa_multi.save(str(temp_wav_file), waveform, sample_rate=target_sr, bits_per_sample=16)
        logger.info(f"{log_prefix}Converted and saved temporary audio for Whisper: {temp_wav_file}")
        return temp_wav_file, None
    except Exception as e:
        error_msg = f"Audio preparation for Whisper failed for '{original_audio_path}': {e}"
        logger.error(f"{log_prefix}{error_msg}", exc_info=True)
        if temp_wav_file and temp_wav_file.exists():
            try: temp_wav_file.unlink(missing_ok=True)
            except OSError: pass
        return None, error_msg


def synthesize_with_llasa_multilingual_transformers(
    model_config: dict,
    text: str,
    voice_id_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    global _llasa_multilingual_whisper_pipeline_cache

    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'llasa_multilingual_hf_unknown')
    log_prefix = f"LLaSA MultiHF ({crisptts_model_id_for_log}): "

    logger.info(f"{log_prefix}Starting synthesis. Target text (first 50): '{text[:50]}...'")
    logger.debug(f"{log_prefix}Voice/Ref audio path override from main: '{voice_id_override}'")

    if not all([TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, XCODEC2_AVAILABLE, SOUNDFILE_AVAILABLE, NUMPY_AVAILABLE, TORCHAUDIO_AVAILABLE]):
        missing = [
            cond_name for cond, cond_name in [
                (TORCH_AVAILABLE, "PyTorch"), (TRANSFORMERS_AVAILABLE, "Transformers"),
                (XCODEC2_AVAILABLE, "XCodec2"), (SOUNDFILE_AVAILABLE, "SoundFile"),
                (NUMPY_AVAILABLE, "NumPy"), (TORCHAUDIO_AVAILABLE, "Torchaudio")
            ] if not cond
        ]
        logger.error(f"{log_prefix}Missing one or more critical dependencies: {', '.join(missing)}. Skipping.")
        return

    llm_model_id = model_config.get("llm_model_id")
    tokenizer_id = model_config.get("tokenizer_id", llm_model_id)
    codec_model_id = model_config.get("codec_model_id", "HKUSTAudio/xcodec2")

    if not all([llm_model_id, tokenizer_id, codec_model_id]):
        logger.error(f"{log_prefix}Core model IDs (LLM, Tokenizer, Codec) not fully configured. Skipping.")
        return

    hf_token = os.getenv("HF_TOKEN") if model_config.get("requires_hf_token") else None
    
    device_str = "cpu"
    if IS_CUDA_LLASA_MULTI: device_str = "cuda"
    elif IS_MPS_LLASA_MULTI: device_str = "mps"
    logger.info(f"{log_prefix}Using device: {device_str}")

    llasa_tokenizer_inst, llasa_llm_inst, codec_model_inst = None, None, None
    whisper_pipeline_inst = None
    temp_whisper_audio_file_to_delete: Optional[Path] = None
    processed_ref_waveform_for_trimming: Optional[torch_llasa_multi.Tensor] = None

    try:
        logger.info(f"{log_prefix}Loading LLaSA Tokenizer from '{tokenizer_id}'...")
        llasa_tokenizer_inst = AutoTokenizer_llasa_multi.from_pretrained(tokenizer_id, token=hf_token, trust_remote_code=True)
        
        logger.info(f"{log_prefix}Loading LLaSA LLM from '{llm_model_id}' to device '{device_str}'...")
        llasa_llm_inst = AutoModelForCausalLM_llasa_multi.from_pretrained(llm_model_id, token=hf_token, trust_remote_code=True)
        llasa_llm_inst.to(device_str).eval()

        logger.info(f"{log_prefix}Loading XCodec2 model from '{codec_model_id}' to device '{device_str}'...")
        codec_model_inst = XCodec2Model_llasa_multi.from_pretrained(codec_model_id, token=hf_token)
        codec_model_inst.to(device_str).eval()
        logger.info(f"{log_prefix}All primary models loaded successfully.")

        is_cloning_mode = False
        speech_ids_prefix_str_list = []
        input_text_for_llm = text
        
        actual_ref_audio_path_str = voice_id_override or model_config.get("default_voice_id")
        if actual_ref_audio_path_str and isinstance(actual_ref_audio_path_str, str) and actual_ref_audio_path_str.strip().lower() != 'none':
            resolved_ref_path = Path(actual_ref_audio_path_str.strip())
            if not resolved_ref_path.is_absolute():
                resolved_ref_path = (Path.cwd() / resolved_ref_path).resolve()
            
            logger.debug(f"{log_prefix}Checking for reference audio file at resolved path: {resolved_ref_path}")
            if resolved_ref_path.exists() and resolved_ref_path.is_file():
                is_cloning_mode = True
                logger.info(f"{log_prefix}Cloning Mode ENABLED. Reference audio: {resolved_ref_path}")

                logger.debug(f"{log_prefix}Loading reference audio with torchaudio from: {resolved_ref_path}")
                waveform_ref, sr_ref = torchaudio_llasa_multi.load(str(resolved_ref_path))
                logger.debug(f"{log_prefix}Ref audio loaded. Original SR: {sr_ref}, Shape: {waveform_ref.shape}")

                max_ref_secs = model_config.get("ref_audio_max_duration_s", 15.0)
                target_ref_sr_for_xcodec = 16000

                if waveform_ref.size(0) > 1:
                    waveform_ref = torch_llasa_multi.mean(waveform_ref, dim=0, keepdim=True)
                if sr_ref != target_ref_sr_for_xcodec:
                    resampler = torchaudio_llasa_multi.transforms.Resample(orig_freq=sr_ref, new_freq=target_ref_sr_for_xcodec)
                    waveform_ref = resampler(waveform_ref)
                    sr_ref = target_ref_sr_for_xcodec
                
                if waveform_ref.shape[1] / sr_ref > max_ref_secs:
                    waveform_ref = waveform_ref[:, :int(sr_ref * max_ref_secs)]
                waveform_ref = torch_llasa_multi.nn.functional.pad(waveform_ref, (0, int(sr_ref * 0.5)), "constant", 0)
                logger.debug(f"{log_prefix}Ref audio processed for XCodec. Final shape: {waveform_ref.shape}")
                
                processed_ref_waveform_for_trimming = waveform_ref.clone().to(device_str)

                ref_transcription = "Reference audio transcription placeholder."
                custom_ref_text = None
                if model_params_override:
                    try: custom_ref_text = json.loads(model_params_override).get("reference_text")
                    except (json.JSONDecodeError, TypeError): pass
                
                if custom_ref_text and isinstance(custom_ref_text, str) and custom_ref_text.strip():
                    ref_transcription = custom_ref_text.strip()
                    logger.info(f"{log_prefix}Using provided reference text: '{ref_transcription[:100]}...'")
                else:
                    whisper_model_id_cfg = model_config.get("whisper_model_id_for_transcription")
                    
                    # --- ADDED DEBUG LOGS ---
                    logger.debug(f"{log_prefix}Whisper Check: whisper_model_id_cfg = '{whisper_model_id_cfg}' (type: {type(whisper_model_id_cfg)})")
                    logger.debug(f"{log_prefix}Whisper Check: pipeline_llasa_multi is {'NOT None and callable' if callable(pipeline_llasa_multi) else ('None or not callable')} (type: {type(pipeline_llasa_multi)})")
                    logger.debug(f"{log_prefix}Whisper Check: TRANSFORMERS_AVAILABLE = {TRANSFORMERS_AVAILABLE}")
                    # --- END ADDED DEBUG LOGS ---

                    if whisper_model_id_cfg and pipeline_llasa_multi and callable(pipeline_llasa_multi): # Ensure pipeline_llasa_multi is callable
                        logger.info(f"{log_prefix}Preparing to transcribe ref audio '{resolved_ref_path}' with Whisper: '{whisper_model_id_cfg}'")
                        
                        temp_whisper_audio_file_to_delete, prep_error = _prepare_audio_for_whisper_local(
                            str(resolved_ref_path), target_sr=16000, log_prefix=f"{log_prefix}WhisperPrep: "
                        )
                        if prep_error or not temp_whisper_audio_file_to_delete:
                            logger.error(f"{log_prefix}Failed to prepare audio for Whisper: {prep_error}. Using placeholder transcription.")
                            ref_transcription = "Reference audio preparation failed for Whisper."
                        else:
                            logger.debug(f"{log_prefix}Ref audio prepared for Whisper at temp path: {temp_whisper_audio_file_to_delete}")
                            whisper_pipeline_device_arg = device_str
                            if device_str == "cuda": whisper_pipeline_device_arg = 0
                            
                            cache_key = (whisper_model_id_cfg, whisper_pipeline_device_arg, str(torch_llasa_multi.float16 if device_str != "cpu" else torch_llasa_multi.float32))
                            if cache_key in _llasa_multilingual_whisper_pipeline_cache:
                                whisper_pipeline_inst = _llasa_multilingual_whisper_pipeline_cache[cache_key]
                            else:
                                whisper_pipeline_inst = pipeline_llasa_multi(
                                    "automatic-speech-recognition", model=whisper_model_id_cfg,
                                    torch_dtype=torch_llasa_multi.float16 if device_str != "cpu" else torch_llasa_multi.float32,
                                    device=whisper_pipeline_device_arg, token=hf_token
                                )
                                _llasa_multilingual_whisper_pipeline_cache[cache_key] = whisper_pipeline_inst
                            
                            lang_hint_for_whisper = model_config.get("language")
                            whisper_gen_kwargs = {"language": lang_hint_for_whisper.lower()} if lang_hint_for_whisper else {}
                            whisper_gen_kwargs["return_timestamps"] = True

                            logger.debug(f"{log_prefix}Calling Whisper ASR pipeline (model: {whisper_model_id_cfg}) with generate_kwargs: {whisper_gen_kwargs}")
                            with SuppressOutput(suppress_stderr=True, suppress_stdout=not logger.isEnabledFor(logging.DEBUG)):
                                transcription_output = whisper_pipeline_inst(str(temp_whisper_audio_file_to_delete), generate_kwargs=whisper_gen_kwargs)
                            
                            transcribed_text_str = transcription_output.get("text", "").strip()
                            if not transcribed_text_str and "chunks" in transcription_output:
                                transcribed_text_str = " ".join([chunk.get('text',"").strip() for chunk in transcription_output["chunks"]]).strip()
                            
                            if transcribed_text_str:
                                ref_transcription = transcribed_text_str
                                logger.info(f"{log_prefix}Whisper transcription: '{ref_transcription[:100]}...'")
                            else:
                                logger.warning(f"{log_prefix}Whisper transcription returned empty. Using placeholder.")
                                ref_transcription = "Transcription was empty."
                    else: # This is the path taken if condition whisper_model_id_cfg and pipeline_llasa_multi is false
                        logger.warning(f"{log_prefix}No custom reference text AND (Whisper model not configured OR Transformers pipeline not available). Using placeholder transcription.")
                        ref_transcription = "Reference audio (transcription N/A)."
                
                input_text_for_llm = f"{ref_transcription} {text}"
                with torch_llasa_multi.no_grad():
                    vq_codes_from_ref_tensor = codec_model_inst.encode_code(input_waveform=processed_ref_waveform_for_trimming.to(device_str))
                numeric_speech_ids_from_ref = vq_codes_from_ref_tensor[0, 0, :].tolist()
                speech_ids_prefix_str_list = _ids_to_speech_tokens_llasa_multi(numeric_speech_ids_from_ref)
                logger.info(f"{log_prefix}Generated {len(speech_ids_prefix_str_list)} speech prefix tokens from reference audio.")
            else:
                is_cloning_mode = False; input_text_for_llm = text
                if "clone" in crisptts_model_id_for_log.lower():
                    logger.error(f"{log_prefix}Cloning intended, but ref audio '{resolved_ref_path}' not valid. Aborting."); return
                logger.info(f"{log_prefix}Ref audio path '{resolved_ref_path}' not valid. Operating in Zero-Shot mode.")
        else:
            is_cloning_mode = False; input_text_for_llm = text
            logger.info(f"{log_prefix}No reference audio path. Operating in Zero-Shot mode.")
        
        formatted_text_for_llm = f"<|TEXT_UNDERSTANDING_START|>{input_text_for_llm}<|TEXT_UNDERSTANDING_END|>"
        assistant_start_content = "<|SPEECH_GENERATION_START|>"
        if is_cloning_mode and speech_ids_prefix_str_list:
            assistant_start_content += "".join(speech_ids_prefix_str_list)
        
        chat_prompt_list = [
            {"role": "user", "content": f"Convert the text to speech:{formatted_text_for_llm}"},
            {"role": "assistant", "content": assistant_start_content}
        ]
        logger.debug(f"{log_prefix}Chat prompt for LLM (assistant starts with '{assistant_start_content[:60]}...'): {json.dumps(chat_prompt_list, ensure_ascii=False, indent=2)[:500]}...")

        input_ids_for_llm = llasa_tokenizer_inst.apply_chat_template(
            chat_prompt_list, tokenize=True, return_tensors='pt', continue_final_message=True
        ).to(device_str)

        generation_params = {
            "max_length": 2048,
            "eos_token_id": llasa_tokenizer_inst.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>'),
            "do_sample": True, "top_p": 1.0, "temperature": 0.8
        }
        default_min_new = 4 if is_cloning_mode else model_config.get("min_new_tokens_zeroshot", 1) 
        generation_params["min_new_tokens"] = model_config.get("min_new_tokens", default_min_new)
        
        if model_params_override:
            try:
                cli_gen_params = json.loads(model_params_override)
                valid_hf_keys = {"temperature", "top_p", "top_k", "do_sample", "num_beams", 
                                 "repetition_penalty", "max_new_tokens", "min_new_tokens", "max_length"}
                for k_param, v_param in cli_gen_params.items():
                    if k_param in valid_hf_keys: generation_params[k_param] = v_param
            except (json.JSONDecodeError, TypeError): pass # Already logged warning if needed
        
        if generation_params["eos_token_id"] == llasa_tokenizer_inst.unk_token_id:
            generation_params["eos_token_id"] = llasa_tokenizer_inst.eos_token_id

        logger.info(f"{log_prefix}Starting LLM generation with params: {generation_params}")
        with torch_llasa_multi.no_grad(), SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            llm_output_ids_tensor = llasa_llm_inst.generate(
                input_ids_for_llm,
                pad_token_id=llasa_tokenizer_inst.eos_token_id,
                **generation_params
            )
        
        start_index_for_generated_part = input_ids_for_llm.shape[1]
        generated_ids_after_prompt = llm_output_ids_tensor[0][start_index_for_generated_part:]
        if generated_ids_after_prompt.numel() > 0 and generated_ids_after_prompt[-1] == generation_params["eos_token_id"]:
            generated_ids_after_prompt = generated_ids_after_prompt[:-1]
        
        generated_token_strings_from_llm = llasa_tokenizer_inst.convert_ids_to_tokens(generated_ids_after_prompt.tolist())
        final_numeric_speech_ids_for_xcodec = _extract_speech_ids_llasa_multi(generated_token_strings_from_llm)
        
        if not final_numeric_speech_ids_for_xcodec:
            logger.error(f"{log_prefix}No valid numeric speech IDs extracted. Cannot decode."); return
        logger.info(f"{log_prefix}Extracted {len(final_numeric_speech_ids_for_xcodec)} numeric speech IDs for XCodec2.")

        xcodec_input_tensor = torch_llasa_multi.tensor(final_numeric_speech_ids_for_xcodec, device=device_str).unsqueeze(0).unsqueeze(0)
        with torch_llasa_multi.no_grad():
            generated_waveform_decoded = codec_model_inst.decode_code(xcodec_input_tensor)
        
        if generated_waveform_decoded is None or generated_waveform_decoded.numel() == 0:
            logger.error(f"{log_prefix}XCodec2 decoding returned None or empty tensor."); return

        if is_cloning_mode and processed_ref_waveform_for_trimming is not None:
            trim_reconstructed_audio = True
            if model_params_override:
                try: trim_reconstructed_audio = json.loads(model_params_override).get("trim_reconstructed_prompt", True)
                except (json.JSONDecodeError, TypeError): pass
            if trim_reconstructed_audio:
                num_samples_in_original_ref_prompt = processed_ref_waveform_for_trimming.shape[1]
                if generated_waveform_decoded.shape[2] >= num_samples_in_original_ref_prompt:
                    generated_waveform_decoded = generated_waveform_decoded[:, :, num_samples_in_original_ref_prompt:]
                else:
                    logger.warning(f"{log_prefix}Requested to trim re-synthesized prompt, but decoded audio too short.")
        
        audio_output_np_final = generated_waveform_decoded.squeeze().cpu().numpy()
        if audio_output_np_final.size == 0:
            logger.error(f"{log_prefix}Final audio is empty after XCodec2 and potential trimming."); return

        final_duration_s = audio_output_np_final.shape[0] / 16000.0
        logger.info(f"{log_prefix}Final audio generated. Duration: {final_duration_s:.2f}s at 16000Hz.")

        if output_file_str:
            output_path_wav = Path(output_file_str).with_suffix(".wav")
            output_path_wav.parent.mkdir(parents=True, exist_ok=True)
            try:
                sf_llasa_multi.write(str(output_path_wav), audio_output_np_final, samplerate=16000)
                logger.info(f"{log_prefix}Audio saved to {output_path_wav}")
            except Exception as e_save:
                logger.error(f"{log_prefix}Failed to save audio: {e_save}", exc_info=True)
        
        if play_direct:
            try:
                audio_int16_bytes = (numpy_llasa_multi.clip(audio_output_np_final, -1.0, 1.0) * 32767).astype(numpy_llasa_multi.int16).tobytes()
                play_audio(audio_int16_bytes, is_path=False, input_format="pcm_s16le", sample_rate=16000)
            except Exception as e_play:
                logger.error(f"{log_prefix}Failed to play audio: {e_play}", exc_info=True)
        logger.info(f"{log_prefix}Synthesis process completed.")

    except Exception as e_synth:
        logger.error(f"{log_prefix}Error during synthesis: {e_synth}", exc_info=True)
    finally:
        logger.info(f"{log_prefix}Entering 'finally' for cleanup.")
        del llasa_tokenizer_inst, llasa_llm_inst, codec_model_inst, whisper_pipeline_inst
        if temp_whisper_audio_file_to_delete and temp_whisper_audio_file_to_delete.exists():
            try:
                temp_whisper_audio_file_to_delete.unlink(missing_ok=True)
                logger.debug(f"{log_prefix}Deleted temp Whisper audio: {temp_whisper_audio_file_to_delete}")
            except OSError as e_del_temp:
                 logger.warning(f"{log_prefix}Could not delete temp Whisper audio '{temp_whisper_audio_file_to_delete}': {e_del_temp}")

        if TORCH_AVAILABLE and torch_llasa_multi:
            if device_str == "cuda": torch_llasa_multi.cuda.empty_cache()
            elif device_str == "mps" and hasattr(torch_llasa_multi.mps, "empty_cache"):
                try: torch_llasa_multi.mps.empty_cache()
                except Exception: pass
            logger.debug(f"{log_prefix}PyTorch cache cleared (if applicable).")
        gc.collect()
        logger.info(f"{log_prefix}Resource cleanup finished.")