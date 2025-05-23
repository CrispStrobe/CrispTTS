# handlers/llasa_hf_transformers_handler.py

import logging
import os
from pathlib import Path
import gc
import json
import tempfile

# Conditional imports
TORCH_AVAILABLE = False
torch_llasa_hf = None
IS_MPS_LLASA_HF = False
TRANSFORMERS_AVAILABLE = False
AutoTokenizer_llasa_hf, AutoModelForCausalLM_llasa_hf, pipeline_llasa_hf = None, None, None
XCODEC2_AVAILABLE = False
XCodec2Model_llasa_hf = None
SOUNDFILE_AVAILABLE = False
sf_llasa_hf = None
TORCHAUDIO_AVAILABLE = False
torchaudio_llasa_hf = None
numpy_llasa_hf = None

logger_init = logging.getLogger("CrispTTS.handlers.llasa_hf_transformers.init")

try:
    import torch
    torch_llasa_hf = torch
    TORCH_AVAILABLE = True
    if hasattr(torch_llasa_hf.backends, "mps") and torch_llasa_hf.backends.mps.is_available():
        IS_MPS_LLASA_HF = True
    import torchaudio
    torchaudio_llasa_hf = torchaudio
    TORCHAUDIO_AVAILABLE = True
    logger_init.info("PyTorch and Torchaudio loaded for LLaSA HF Transformers handler.")
except ImportError:
    logger_init.warning("PyTorch or Torchaudio not found. LLaSA HF Transformers handler will be non-functional.")

if TORCH_AVAILABLE:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        AutoTokenizer_llasa_hf = AutoTokenizer
        AutoModelForCausalLM_llasa_hf = AutoModelForCausalLM
        pipeline_llasa_hf = pipeline
        TRANSFORMERS_AVAILABLE = True
        logger_init.info("Transformers library loaded.")
    except ImportError:
        logger_init.warning("Transformers library not found. LLaSA HF Transformers handler will be non-functional.")

    try:
        from xcodec2.modeling_xcodec2 import XCodec2Model
        XCodec2Model_llasa_hf = XCodec2Model
        XCODEC2_AVAILABLE = True
        logger_init.info("XCodec2 library loaded.")
    except ImportError:
        logger_init.warning("XCodec2 library not found. LLaSA HF Transformers handler will be non-functional.")

try:
    import soundfile as sf
    sf_llasa_hf = sf
    SOUNDFILE_AVAILABLE = True
    logger_init.info("SoundFile library loaded.")
except ImportError:
    logger_init.warning("SoundFile library not found. Saving audio will not be possible with this handler directly.")

try:
    import numpy as np
    numpy_llasa_hf = np
except ImportError:
    logger_init.warning("NumPy not found. Some audio operations might fail.")


from utils import save_audio, play_audio, SuppressOutput # Ensure these utils are robust

logger = logging.getLogger("CrispTTS.handlers.llasa_hf_transformers")

def _extract_speech_ids_llasa_hf(token_strings_list: list[str]) -> list[int]:
    speech_ids = []
    if not token_strings_list:
        return speech_ids
    for token_str in token_strings_list:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            except ValueError:
                logger.warning(f"LLaSA HF: Could not parse speech ID from token: {token_str}")
        # else: # Potentially noisy if many non-speech tokens are present
        #     logger.debug(f"LLaSA HF: Non-speech or unexpected token encountered: {token_str}")
    return speech_ids

def _ids_to_speech_tokens_llasa_hf(speech_ids: list[int]) -> list[str]:
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def synthesize_with_llasa_hf_transformers(
    model_config: dict,
    text: str, # Target text for synthesis
    voice_id_override: str | None, # Path to reference audio for cloning, or None for zero-shot
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'llasa_hf_transformers')

    if not all([TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, XCODEC2_AVAILABLE, SOUNDFILE_AVAILABLE, TORCHAUDIO_AVAILABLE, numpy_llasa_hf]):
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Missing one or more core dependencies. Skipping.")
        return

    llm_model_id = model_config.get("llm_model_id")
    tokenizer_id = model_config.get("tokenizer_id", llm_model_id)
    codec_model_id = model_config.get("codec_model_id")
    
    if not all([llm_model_id, tokenizer_id, codec_model_id]):
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Core model IDs (llm, tokenizer, codec) not fully configured. Skipping.")
        return

    hf_token = os.getenv("HF_TOKEN") if model_config.get("requires_hf_token") else None

    if torch_llasa_hf.cuda.is_available(): device = "cuda"
    elif IS_MPS_LLASA_HF: device = "mps"
    else: device = "cpu"
    logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Using device: {device}")

    llasa_tokenizer, llasa_model, codec_model, whisper_pipe = None, None, None, None
    prompt_waveform_for_trimming = None # To store reference waveform for potential trimming

    try:
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Loading LLaSA tokenizer from '{tokenizer_id}'...")
        llasa_tokenizer = AutoTokenizer_llasa_hf.from_pretrained(tokenizer_id, token=hf_token, trust_remote_code=True)

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Loading LLaSA LLM from '{llm_model_id}'...")
        llasa_model = AutoModelForCausalLM_llasa_hf.from_pretrained(llm_model_id, token=hf_token, trust_remote_code=True)
        llasa_model.to(device).eval()

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Loading XCodec2 model from '{codec_model_id}'...")
        codec_model = XCodec2Model_llasa_hf.from_pretrained(codec_model_id, token=hf_token)
        codec_model.to(device).eval()
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Models loaded.")

        # Determine mode: cloning or zero-shot
        # `voice_id_override` is the path to reference audio. If not provided or invalid, it's zero-shot.
        # The config `default_voice_id` could also specify a reference for cloning modes.
        ref_audio_path_str = voice_id_override or model_config.get("default_voice_id")
        is_cloning_mode = False
        if ref_audio_path_str:
            if Path(ref_audio_path_str).exists() and Path(ref_audio_path_str).is_file():
                is_cloning_mode = True
                logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Cloning mode enabled with reference audio: {ref_audio_path_str}")
            else:
                logger.warning(f"LLaSA HF ({crisptts_model_id_for_log}): Reference audio path '{ref_audio_path_str}' not found. Falling back to zero-shot TTS if applicable, or may fail if model requires cloning.")
                # If the config implies cloning (e.g., "llasa_german_transformers_clone") but ref audio is missing, it should ideally error or clearly state zero-shot.
                # For a "zeroshot" config, ref_audio_path_str would be None.
                if "clone" in crisptts_model_id_for_log.lower(): # If it's a cloning config entry
                     logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Model configured for cloning but reference audio path is invalid: {ref_audio_path_str}. Cannot proceed with cloning.")
                     return # Or raise an error

        speech_tokens_prefix_str_list = []
        input_text_for_llm = text # Target text for zero-shot

        if is_cloning_mode:
            ref_audio_path = Path(ref_audio_path_str)
            waveform, sr = torchaudio_llasa_hf.load(ref_audio_path)
            prompt_waveform_for_trimming = waveform.clone() # Keep original for trimming length

            if sr != 16000:
                resampler = torchaudio_llasa_hf.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            if waveform.size(0) > 1: waveform = torch_llasa_hf.mean(waveform, dim=0, keepdim=True)
            
            max_ref_secs = 15.0 # As per LLaSA examples
            current_ref_secs = waveform.shape[1] / 16000.0
            if current_ref_secs > max_ref_secs:
                logger.info(f"LLaSA HF: Reference audio ({current_ref_secs:.1f}s) > {max_ref_secs}s. Trimming.")
                waveform = waveform[:, :int(16000 * max_ref_secs)]
                # prompt_waveform_for_trimming also needs to be trimmed if it's used for length calculation for audio trimming
                prompt_waveform_for_trimming = waveform.clone() 


            ref_transcription = "Reference audio." # Default placeholder
            custom_ref_text = None
            if model_params_override:
                try:
                    params_json = json.loads(model_params_override)
                    custom_ref_text = params_json.get("reference_text")
                except json.JSONDecodeError: pass
            
            if custom_ref_text:
                ref_transcription = custom_ref_text
                logger.info(f"LLaSA HF: Using provided reference text: '{ref_transcription[:100]}...'")
            else:
                whisper_model_id = model_config.get("whisper_model_id_for_transcription")
                if whisper_model_id:
                    logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Transcribing reference audio with Whisper model '{whisper_model_id}'...")
                    if not pipeline_llasa_hf: # Ensure pipeline import was successful
                        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Transformers pipeline utility not available. Cannot transcribe.")
                        # Decide on fallback: use placeholder or error out
                        ref_transcription = "Referenz Audio." 
                    else:
                        try:
                            # whisper_device = 0 if device == "cuda" else -1
                            # if device == "mps": whisper_device = "mps"
                            whisper_device = device # MPS, CUDA, or CPU string
                            if device == "cuda": whisper_device = 0 # pipeline expects int for CUDA

                            whisper_pipe = pipeline_llasa_hf(
                                task="automatic-speech-recognition", # Ensure task is specified
                                model=whisper_model_id,
                                torch_dtype=torch_llasa_hf.float16 if device != "cpu" else torch_llasa_hf.float32,
                                device=whisper_device, # Pass "mps", 0 (for cuda:0), or "cpu"
                                token=hf_token,
                                framework="pt"  
                            )
                            with SuppressOutput(suppress_stderr=True, suppress_stdout=not logger.isEnabledFor(logging.DEBUG)):
                                transcription_result = whisper_pipe(waveform.squeeze(0).cpu().numpy(), generate_kwargs={"language": model_config.get("language") or "german"}) # Use configured lang or default to German
                            ref_transcription = transcription_result["text"].strip()
                            logger.info(f"LLaSA HF: Transcription: '{ref_transcription}'")
                        except Exception as e_whisper:
                            logger.error(f"LLaSA HF: Whisper transcription failed: {e_whisper}. Using placeholder.", exc_info=True)
                            ref_transcription = "Referenz Audio." # Fallback
                else:
                    logger.warning("LLaSA HF: No reference_text provided and no whisper_model_id configured for transcription. Using placeholder reference text.")

            input_text_for_llm = f"{ref_transcription} {text}"

            with torch_llasa_hf.no_grad():
                vq_codes_prompt = codec_model.encode_code(input_waveform=waveform.to(device))
            speech_ids_from_ref_numeric = vq_codes_prompt[0, 0, :].tolist()
            speech_tokens_prefix_str_list = _ids_to_speech_tokens_llasa_hf(speech_ids_from_ref_numeric)
        
        # Common part for both modes
        formatted_llm_text_input = f"<|TEXT_UNDERSTANDING_START|>{input_text_for_llm}<|TEXT_UNDERSTANDING_END|>"
        
        assistant_content = "<|SPEECH_GENERATION_START|>"
        if is_cloning_mode:
            assistant_content += "".join(speech_tokens_prefix_str_list)

        chat_messages = [
            {"role": "user", "content": f"Convert the text to speech:{formatted_llm_text_input}"},
            {"role": "assistant", "content": assistant_content}
        ]

        input_ids = llasa_tokenizer.apply_chat_template(
            chat_messages, tokenize=True, return_tensors='pt', continue_final_message=True
        ).to(device)

        gen_params = {"temperature": 0.8, "top_p": 1.0, "do_sample": True, "max_length": 2048, "min_new_tokens": 4}
        if model_params_override:
            try:
                cli_gen_params = json.loads(model_params_override)
                valid_hf_gen_params = {"temperature", "top_p", "top_k", "do_sample", "num_beams", "repetition_penalty", "max_new_tokens", "min_new_tokens", "max_length"}
                for k, v in cli_gen_params.items():
                    if k in valid_hf_gen_params: gen_params[k] = v
            except json.JSONDecodeError:
                logger.warning(f"LLaSA HF ({crisptts_model_id_for_log}): Could not parse --model-params for generation: {model_params_override}")
        
        speech_end_token_id = llasa_tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        if speech_end_token_id == llasa_tokenizer.unk_token_id:
            logger.warning("LLaSA HF: <|SPEECH_GENERATION_END|> token not found. Using EOS token.")
            speech_end_token_id = llasa_tokenizer.eos_token_id

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Generating speech tokens with params: {gen_params}...")
        with torch_llasa_hf.no_grad():
            generated_output_ids_tensor = llasa_model.generate(input_ids, eos_token_id=speech_end_token_id, **gen_params)
        
        # Slicing logic based on official LLaSA examples
        if is_cloning_mode:
            # For cloning, the example suggests this slice to get the model's generation of the prefix + new content
            # Here, `len(speech_tokens_prefix_str_list)` is the number of <|s_XXX|> tokens in the prefix
            slice_start_index = input_ids.shape[1] - len(speech_tokens_prefix_str_list)
            if slice_start_index < 0: # Should not happen if prefix is part of input_ids
                logger.warning(f"LLaSA HF: Cloning slice_start_index negative ({slice_start_index}). Defaulting to start of assistant generation.")
                slice_start_index = input_ids.shape[1] # Fallback to get only purely new tokens
            
            # Ensure slice_start_index is not greater than the length of generated_output_ids_tensor[0]
            slice_start_index = min(slice_start_index, generated_output_ids_tensor.shape[1])

            extracted_ids_for_decode = generated_output_ids_tensor[0][slice_start_index:]
        else: # Zero-shot
            extracted_ids_for_decode = generated_output_ids_tensor[0][input_ids.shape[1]:]

        # Remove EOS if present at the end
        if extracted_ids_for_decode.numel() > 0 and extracted_ids_for_decode[-1] == speech_end_token_id:
             extracted_ids_for_decode = extracted_ids_for_decode[:-1]
        
        generated_token_strings = llasa_tokenizer.convert_ids_to_tokens(extracted_ids_for_decode.tolist())
        final_speech_ids_numeric = _extract_speech_ids_llasa_hf(generated_token_strings)
        
        if not final_speech_ids_numeric:
            logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): No valid speech IDs extracted. Output may be empty or non-speech.")
            return

        logger.info(f"LLaSA HF: Extracted {len(final_speech_ids_numeric)} speech IDs for vocoder.")
        speech_tokens_tensor_for_codec = torch_llasa_hf.tensor(final_speech_ids_numeric, device=device).unsqueeze(0).unsqueeze(0)
        
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Decoding speech IDs with XCodec2...")
        with torch_llasa_hf.no_grad():
            generated_waveform = codec_model.decode_code(speech_tokens_tensor_for_codec)

        # Optional: Trim the re-synthesized prompt audio if cloning
        if is_cloning_mode and prompt_waveform_for_trimming is not None:
            trim_reconstructed = False
            if model_params_override:
                try: trim_reconstructed = json.loads(model_params_override).get("trim_reconstructed_prompt", False)
                except: pass
            
            if trim_reconstructed:
                num_prompt_samples = prompt_waveform_for_trimming.shape[1] # Original reference audio length
                if generated_waveform.shape[2] > num_prompt_samples:
                    logger.info(f"LLaSA HF: Trimming {num_prompt_samples} samples of re-synthesized prompt from output audio.")
                    generated_waveform = generated_waveform[:, :, num_prompt_samples:]
                else:
                    logger.warning(f"LLaSA HF: Trim requested, but generated audio ({generated_waveform.shape[2]}) not longer than prompt ({num_prompt_samples}). Not trimming.")

        audio_output_np = generated_waveform.squeeze().cpu().numpy()
        
        if audio_output_np.size == 0:
            logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): XCodec2 decoding resulted in empty audio.")
            return

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Synthesis successful. Output audio shape: {audio_output_np.shape}")
        output_sample_rate = model_config.get("sample_rate", 16000)

        if output_file_str:
            effective_output_path = Path(output_file_str).with_suffix(".wav")
            effective_output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"LLaSA HF: Saving audio to {effective_output_path}")
            sf_llasa_hf.write(str(effective_output_path), audio_output_np, samplerate=output_sample_rate)
        
        if play_direct:
            logger.info(f"LLaSA HF: Playing audio...")
            audio_int16 = (numpy_llasa_hf.clip(audio_output_np, -1.0, 1.0) * 32767).astype(numpy_llasa_hf.int16)
            play_audio(audio_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=output_sample_rate)

    except Exception as e:
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Synthesis failed: {e}", exc_info=True)
    finally:
        logger.debug(f"LLaSA HF ({crisptts_model_id_for_log}): Cleaning up models.")
        del llasa_tokenizer, llasa_model, codec_model, whisper_pipe
        if TORCH_AVAILABLE and torch_llasa_hf:
            if device == "cuda": torch_llasa_hf.cuda.empty_cache()
            elif device == "mps" and hasattr(torch_llasa_hf.mps, "empty_cache"):
                try: torch_llasa_hf.mps.empty_cache()
                except Exception as e_mps_clear: logger.debug(f"LLaSA HF: Error clearing MPS cache: {e_mps_clear}")
        gc.collect()
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Handler finished.")