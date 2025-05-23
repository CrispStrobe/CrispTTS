# handlers/llasa_german_transformers_handler.py

import logging
import os
from pathlib import Path
import gc
import json
import tempfile

# Conditional imports for torch, transformers, xcodec, soundfile, torchaudio
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

logger_init = logging.getLogger("CrispTTS.handlers.llasa_german_transformers.init")

try:
    import torch
    torch_llasa_hf = torch
    TORCH_AVAILABLE = True
    if hasattr(torch_llasa_hf.backends, "mps") and torch_llasa_hf.backends.mps.is_available():
        IS_MPS_LLASA_HF = True
    import torchaudio
    torchaudio_llasa_hf = torchaudio
    TORCHAUDIO_AVAILABLE = True
    logger_init.info("PyTorch and Torchaudio loaded for LLaSA German Transformers handler.")
except ImportError:
    logger_init.warning("PyTorch or Torchaudio not found. LLaSA German Transformers handler will be non-functional.")

if TORCH_AVAILABLE:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        AutoTokenizer_llasa_hf = AutoTokenizer
        AutoModelForCausalLM_llasa_hf = AutoModelForCausalLM
        pipeline_llasa_hf = pipeline
        TRANSFORMERS_AVAILABLE = True
        logger_init.info("Transformers library loaded.")
    except ImportError:
        logger_init.warning("Transformers library not found. LLaSA German Transformers handler will be non-functional.")

    try:
        from xcodec2.modeling_xcodec2 import XCodec2Model
        XCodec2Model_llasa_hf = XCodec2Model
        XCODEC2_AVAILABLE = True
        logger_init.info("XCodec2 library loaded.")
    except ImportError:
        logger_init.warning("XCodec2 library not found. LLaSA German Transformers handler will be non-functional.")

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


from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.llasa_german_transformers")

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
        # else:
            # logger.debug(f"LLaSA HF: Non-speech token encountered during extraction: {token_str}")
    return speech_ids

def _ids_to_speech_tokens_llasa_hf(speech_ids: list[int]) -> list[str]:
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def synthesize_with_llasa_german_transformers(
    model_config: dict,
    text: str,
    voice_id_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'llasa_german_transformers_clone')

    if not all([TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, XCODEC2_AVAILABLE, SOUNDFILE_AVAILABLE, TORCHAUDIO_AVAILABLE, numpy_llasa_hf]):
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Missing one or more core dependencies (Torch, Transformers, XCodec2, SoundFile, Torchaudio, NumPy). Skipping.")
        return

    llm_model_id = model_config.get("llm_model_id")
    tokenizer_id = model_config.get("tokenizer_id", llm_model_id)
    codec_model_id = model_config.get("codec_model_id")
    whisper_model_id = model_config.get("whisper_model_id_for_transcription") # Optional

    if not all([llm_model_id, tokenizer_id, codec_model_id]):
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Core model IDs (llm, tokenizer, codec) not fully configured. Skipping.")
        return

    hf_token = os.getenv("HF_TOKEN") if model_config.get("requires_hf_token") else None

    # Determine device
    if torch_llasa_hf.cuda.is_available():
        device = "cuda"
    elif IS_MPS_LLASA_HF:
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Using device: {device}")

    llasa_tokenizer, llasa_model, codec_model, whisper_pipe = None, None, None, None

    try:
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Loading LLaSA tokenizer from '{tokenizer_id}'...")
        llasa_tokenizer = AutoTokenizer_llasa_hf.from_pretrained(tokenizer_id, token=hf_token, trust_remote_code=True)

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Loading LLaSA LLM from '{llm_model_id}'...")
        llasa_model = AutoModelForCausalLM_llasa_hf.from_pretrained(llm_model_id, token=hf_token, trust_remote_code=True)
        llasa_model.to(device).eval()

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Loading XCodec2 model from '{codec_model_id}'...")
        codec_model = XCodec2Model_llasa_hf.from_pretrained(codec_model_id, token=hf_token)
        codec_model.to(device).eval()
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): All models loaded.")

        # Reference audio processing (for voice cloning)
        ref_audio_path_str = voice_id_override or model_config.get("default_voice_id")
        if not ref_audio_path_str or not Path(ref_audio_path_str).exists():
            logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Reference audio path '{ref_audio_path_str}' not found or not specified. Voice cloning requires it. Skipping.")
            return
        
        ref_audio_path = Path(ref_audio_path_str)
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Processing reference audio: {ref_audio_path}")

        waveform, sample_rate = torchaudio_llasa_hf.load(ref_audio_path)

        # Resample if necessary (target for LLaSA/XCodec2 is 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio_llasa_hf.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch_llasa_hf.mean(waveform, dim=0, keepdim=True)

        # Trim audio (e.g., to first 15 seconds as in example, make configurable if needed)
        max_ref_secs = 15.0
        if waveform.shape[1] / sample_rate > max_ref_secs:
            logger.info(f"LLaSA HF: Reference audio is longer than {max_ref_secs}s. Trimming.")
            waveform = waveform[:, :int(sample_rate * max_ref_secs)]
            # Optional: pad to ensure min length or add silence, example adds 0.5s if trimmed
            waveform = torch_llasa_hf.nn.functional.pad(waveform, (0, int(sample_rate * 0.5)), "constant", 0)


        # Transcription (can be made optional or configurable)
        ref_transcription = "Reference audio transcription." # Default placeholder
        if model_params_override: # Check if ref_text is provided in model_params
            try:
                params_json = json.loads(model_params_override)
                if params_json.get("reference_text"):
                    ref_transcription = params_json["reference_text"]
                    logger.info(f"LLaSA HF: Using provided reference text: '{ref_transcription[:100]}...'")
            except json.JSONDecodeError:
                 logger.warning(f"LLaSA HF ({crisptts_model_id_for_log}): Could not parse --model-params as JSON: {model_params_override}")


        if ref_transcription == "Reference audio transcription." and whisper_model_id: # If still placeholder and whisper is configured
            logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Transcribing reference audio with Whisper model '{whisper_model_id}'...")
            if not pipeline_llasa_hf:
                logger.error("Transformers pipeline not available for Whisper.")
                return
            try:
                # Ensure whisper pipe is on the same device if possible, or handle CPU explicitly
                whisper_device = 0 if device == "cuda" else -1 # pipeline device: 0 for cuda:0, -1 for CPU
                if device == "mps": whisper_device = "mps"

                whisper_pipe = pipeline_llasa_hf(
                    "automatic-speech-recognition",
                    model=whisper_model_id,
                    torch_dtype=torch_llasa_hf.float16 if device != "cpu" else torch_llasa_hf.float32, # float16 for GPU, float32 for CPU
                    device=whisper_device,
                    token=hf_token
                )
                # Whisper pipeline expects numpy array for audio input
                with SuppressOutput(suppress_stderr=True, suppress_stdout=not logger.isEnabledFor(logging.DEBUG)):
                    transcription_result = whisper_pipe(waveform.squeeze(0).cpu().numpy(), generate_kwargs={"language": "german"})
                ref_transcription = transcription_result["text"].strip()
                logger.info(f"LLaSA HF: Transcription: '{ref_transcription}'")
            except Exception as e_whisper:
                logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Whisper transcription failed: {e_whisper}. Using placeholder.", exc_info=True)
                ref_transcription = "Referenz Audio." # Fallback German placeholder
        
        # Encode reference audio to VQ codes
        prompt_wav_for_codec = waveform.to(device) # Ensure on correct device for codec
        with torch_llasa_hf.no_grad():
            vq_codes_prompt = codec_model.encode_code(input_waveform=prompt_wav_for_codec) # Shape: [1, num_codebooks, T]
        
        # The example uses vq_code_prompt[0, 0, :]. This implies using only the first codebook.
        # LLaSA might expect flattened codes from multiple codebooks or specific handling.
        # For now, following example:
        # speech_ids_from_ref = vq_codes_prompt[0, 0, :].tolist() # IDs from the first codebook

        # Let's assume a more general approach if multiple codebooks are used:
        # Flatten codes from all codebooks, or interleave them if that's what the model expects
        # The LLaSA paper or xcodec2 documentation should clarify how these are fed.
        # For now, sticking to example's simplicity if it works.
        # Assuming vq_codes_prompt is [Batch, Codebook, Time]. The example's usage:
        # outputs = model.generate(input_ids, max_length=2048, speech_ids_prefix=vq_code_prompt)
        # This suggests passing the whole tensor. Let's verify LLaSA's generate signature.
        # The provided example's HF page for MultiLlasa shows:
        # speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt[0,0,:].tolist()) # This is string join
        # assistant_content = "<|SPEECH_GENERATION_START|>" + "".join(speech_ids_prefix)
        # This indicates it expects a string of speech tokens, not the raw tensor.

        speech_ids_from_ref_numeric = vq_codes_prompt[0, 0, :].tolist() # Numeric IDs from the first codebook
        speech_tokens_prefix_str_list = _ids_to_speech_tokens_llasa_hf(speech_ids_from_ref_numeric)
        speech_tokens_prefix_joined = "".join(speech_tokens_prefix_str_list)

        # Prepare text for LLM
        combined_input_text = f"{ref_transcription} {text}" # As per example
        formatted_llm_text_input = f"<|TEXT_UNDERSTANDING_START|>{combined_input_text}<|TEXT_UNDERSTANDING_END|>"
        
        chat_messages = [
            {"role": "user", "content": f"Convert the text to speech:{formatted_llm_text_input}"},
            {"role": "assistant", "content": f"<|SPEECH_GENERATION_START|>{speech_tokens_prefix_joined}"}
        ]

        input_ids = llasa_tokenizer.apply_chat_template(
            chat_messages,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True # Important for assistant to continue generation
        ).to(device)

        # Generation parameters
        gen_params = {"temperature": 0.8, "top_p": 1.0, "do_sample": True, "max_length": 2048, "min_new_tokens": 4} # Defaults from example
        if model_params_override:
            try:
                cli_gen_params = json.loads(model_params_override)
                # Filter for valid generate() arguments or handle specific ones
                valid_hf_gen_params = {"temperature", "top_p", "top_k", "do_sample", "num_beams", "repetition_penalty", "max_new_tokens", "min_new_tokens"}
                for k, v in cli_gen_params.items():
                    if k in valid_hf_gen_params:
                        gen_params[k] = v
                    elif k == "max_length": # max_length for generate includes prompt
                        gen_params[k] = v
                logger.info(f"LLaSA HF: Using generation params: {gen_params}")
            except json.JSONDecodeError:
                logger.warning(f"LLaSA HF ({crisptts_model_id_for_log}): Could not parse --model-params for generation: {model_params_override}")
        
        speech_end_token_id = llasa_tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        if speech_end_token_id == llasa_tokenizer.unk_token_id: # Check if token exists
            logger.warning("LLaSA HF: <|SPEECH_GENERATION_END|> token not found in tokenizer. Generation might not stop correctly.")
            speech_end_token_id = llasa_tokenizer.eos_token_id # Fallback

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Generating speech tokens...")
        with torch_llasa_hf.no_grad():
            generated_output_ids = llasa_model.generate(
                input_ids,
                eos_token_id=speech_end_token_id,
                **gen_params
            )
        
        # Extract only the newly generated part, excluding the prompt and the final EOS token
        # The prefix itself is part of input_ids because of continue_final_message=True
        # The length of the assistant's prefix (speech_tokens_prefix_joined) needs to be known in terms of token IDs.
        # input_ids already contains the tokenized version of speech_tokens_prefix_joined.
        # So, generated_ids = generated_output_ids[0][input_ids.shape[1]:-1] should be correct if eos_token_id is properly handled by generate.
        
        # If speech_tokens_prefix_joined was added to assistant content, input_ids includes it.
        # The generated output starts *after* input_ids.
        # The example code handles the prefix differently for the final extraction:
        # generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix_tokenized_for_len_calc) : -1]
        # This implies speech_ids_prefix was added to the prompt *before* tokenization for length calculation.
        # Let's use the simpler and more standard: generated_ids = outputs[0][input_ids.shape[1]:-1]
        # If the generate call includes the prefix already, this slice is what we want.

        generated_ids_only = generated_output_ids[0][input_ids.shape[1]:]
        # Remove EOS if present at the end
        if generated_ids_only.numel() > 0 and generated_ids_only[-1] == speech_end_token_id:
             generated_ids_only = generated_ids_only[:-1]
        
        logger.debug(f"LLaSA HF: Generated token IDs (length {generated_ids_only.numel()}). Example: {generated_ids_only[:10].tolist()}")

        # Decode generated token IDs to speech token strings
        generated_token_strings = llasa_tokenizer.convert_ids_to_tokens(generated_ids_only.tolist())
        final_speech_ids_numeric = _extract_speech_ids_llasa_hf(generated_token_strings)
        
        if not final_speech_ids_numeric:
            logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): No valid speech IDs extracted from generation. Output might be empty or non-speech.")
            return

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Extracted {len(final_speech_ids_numeric)} speech IDs for vocoder.")
        
        # Prepare for vocoder: needs to be [B, 1, T_codes] for single codebook if using XCodec2 directly with extracted IDs.
        # Or [B, N_codebooks, T_codes] if multi-codebook is expected by some internal LLaSA wrapper for XCodec.
        # The example implies passing the extracted numeric IDs to codec_model.decode_code.
        # These extracted IDs usually correspond to the first codebook.
        speech_tokens_tensor_for_codec = torch_llasa_hf.tensor(final_speech_ids_numeric, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, T_codes]
        
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Decoding speech IDs with XCodec2...")
        with torch_llasa_hf.no_grad():
            # The example's final decode `gen_wav = Codec_model.decode_code(speech_tokens)`
            # where speech_tokens is `torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)`
            # This means the extracted IDs are treated as codes for a single codebook.
            generated_waveform = codec_model.decode_code(speech_tokens_tensor_for_codec) # Expected output [B, 1, T_samples]

        # The example for speaker cloning also does: gen_wav = gen_wav[:, :, prompt_wav.shape[1]:]
        # This is to remove the re-synthesis of the prompt audio.
        # This implies the LLM might re-generate the prefix tokens.
        # Let's check if the generated tokens start with the prefix tokens.
        # For now, we'll try without this slicing, as it complicates things if not perfectly aligned.
        # If the output audio starts with the reference audio, this slice is needed.

        audio_output_np = generated_waveform.squeeze().cpu().numpy()
        
        if audio_output_np.size == 0:
            logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): XCodec2 decoding resulted in empty audio.")
            return

        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Synthesis successful. Output audio shape: {audio_output_np.shape}")

        output_sample_rate = model_config.get("sample_rate", 16000) # Should be 16000 for XCodec2

        if output_file_str:
            effective_output_path = Path(output_file_str).with_suffix(".wav")
            effective_output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"LLaSA HF: Saving audio to {effective_output_path}")
            sf_llasa_hf.write(str(effective_output_path), audio_output_np, samplerate=output_sample_rate)
        
        if play_direct:
            logger.info(f"LLaSA HF: Playing audio...")
            # Convert float32 numpy to int16 bytes for play_audio utility
            audio_int16 = (numpy_llasa_hf.clip(audio_output_np, -1.0, 1.0) * 32767).astype(numpy_llasa_hf.int16)
            play_audio(audio_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=output_sample_rate)

    except Exception as e:
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Synthesis failed: {e}", exc_info=True)
    finally:
        logger.debug(f"LLaSA HF ({crisptts_model_id_for_log}): Cleaning up models.")
        del llasa_tokenizer, llasa_model, codec_model, whisper_pipe
        if TORCH_AVAILABLE and torch_llasa_hf:
            if device == "cuda":
                torch_llasa_hf.cuda.empty_cache()
            elif device == "mps" and hasattr(torch_llasa_hf.mps, "empty_cache"):
                try:
                    torch_llasa_hf.mps.empty_cache()
                except Exception as e_mps_clear:
                    logger.debug(f"LLaSA HF: Error clearing MPS cache: {e_mps_clear}")
        gc.collect()
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Handler finished.")