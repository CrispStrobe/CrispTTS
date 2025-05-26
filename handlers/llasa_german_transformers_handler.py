# handlers/llasa_german_transformers_handler.py
# Specifically for German fine-tuned LLaSA models like MultiLlasa/Llasa-1B-Multilingual-German

import logging
import os
from pathlib import Path
import gc
import json
import tempfile

# Conditional imports
TORCH_AVAILABLE = False
torch_llasa_german = None
IS_MPS_LLASA_HF = False
IS_CUDA_LLASA_HF = False
TRANSFORMERS_AVAILABLE = False
AutoTokenizer_llasa_german, AutoModelForCausalLM_llasa_german, pipeline_llasa_german = None, None, None
XCODEC2_AVAILABLE = False
XCodec2Model_llasa_german = None
SOUNDFILE_AVAILABLE = False
sf_llasa_german = None
TORCHAUDIO_AVAILABLE = False
torchaudio_llasa_german = None
NUMPY_AVAILABLE = False
numpy_llasa_german = None

logger_init = logging.getLogger("CrispTTS.handlers.llasa_german_transformers.init")

try:
    import torch
    torch_llasa_german = torch; TORCH_AVAILABLE = True
    if hasattr(torch_llasa_german.backends, "mps") and torch_llasa_german.backends.mps.is_available(): 
        IS_MPS_LLASA_HF = True
    if torch_llasa_german.cuda.is_available():
        IS_CUDA_LLASA_HF = True
    logger_init.info("PyTorch loaded for LLaSA German.")
except ImportError: logger_init.warning("PyTorch not found for LLaSA German.")

if TORCH_AVAILABLE:
    try:
        import torchaudio
        torchaudio_llasa_german = torchaudio; TORCHAUDIO_AVAILABLE = True
        logger_init.info("Torchaudio loaded for LLaSA German.")
    except ImportError: logger_init.warning("Torchaudio not found for LLaSA German.")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        AutoTokenizer_llasa_german, AutoModelForCausalLM_llasa_german, pipeline_llasa_german = AutoTokenizer, AutoModelForCausalLM, pipeline
        TRANSFORMERS_AVAILABLE = True; logger_init.info("Transformers loaded for LLaSA German.")
    except ImportError: logger_init.warning("Transformers not found for LLaSA German.")
    try:
        from xcodec2.modeling_xcodec2 import XCodec2Model
        XCodec2Model_llasa_german = XCodec2Model; XCODEC2_AVAILABLE = True
        logger_init.info("XCodec2 loaded for LLaSA German.")
    except ImportError: logger_init.warning("XCodec2 not found for LLaSA German.")

try:
    import soundfile as sf
    sf_llasa_german = sf; SOUNDFILE_AVAILABLE = True
    logger_init.info("SoundFile loaded for LLaSA German.")
except ImportError: logger_init.warning("SoundFile not found for LLaSA German.")
try:
    import numpy as np
    numpy_llasa_german = np; NUMPY_AVAILABLE = True
except ImportError: logger_init.warning("NumPy not found for LLaSA German.")

from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.llasa_german_transformers")

def extract_speech_ids(speech_tokens_str):
    """EXACT blueprint implementation for German models"""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            logger.debug(f"Unexpected token: {token_str}")
    return speech_ids

def ids_to_speech_tokens(speech_ids):
    """EXACT blueprint implementation for German models"""
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def synthesize_with_llasa_german_transformers(
    model_config: dict,
    text: str,
    voice_id_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    """German LLaSA model handler - follows blueprint exactly"""
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'llasa_german_transformers')
    logger.info(f"LLaSA German ({crisptts_model_id_for_log}): Starting synthesis. Target text: '{text[:50]}...'")

    if not all([TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, XCODEC2_AVAILABLE, SOUNDFILE_AVAILABLE, NUMPY_AVAILABLE]):
        logger.error(f"LLaSA German ({crisptts_model_id_for_log}): Missing core dependencies. Skipping.")
        return
    
    llm_model_id = model_config.get("llm_model_id")
    tokenizer_id = model_config.get("tokenizer_id", llm_model_id)
    codec_model_id = model_config.get("codec_model_id")
    
    if not all([llm_model_id, tokenizer_id, codec_model_id]):
        logger.error(f"LLaSA German ({crisptts_model_id_for_log}): Core model IDs not fully configured. Skipping.")
        return

    hf_token = os.getenv("HF_TOKEN") if model_config.get("requires_hf_token") else None
    
    # Device selection
    if torch_llasa_german.cuda.is_available():
        device = "cuda"
    elif IS_MPS_LLASA_HF:
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"LLaSA German ({crisptts_model_id_for_log}): Using device: {device}")

    # EXACT blueprint variable names
    tokenizer = None
    model = None  
    Codec_model = None
    whisper_turbo_pipe = None

    try:
        # EXACT blueprint model loading
        llasa_model_name = llm_model_id
        logger.info(f"LLaSA German: Loading tokenizer model: {llasa_model_name}")
        tokenizer = AutoTokenizer_llasa_german.from_pretrained(llasa_model_name)
        
        logger.info(f"LLaSA German: Loading llm model: {llasa_model_name}")
        model = AutoModelForCausalLM_llasa_german.from_pretrained(llasa_model_name)
        model.to(device)
        
        logger.info(f"LLaSA German: Loading codec model: {codec_model_id}")
        codec_model_path = codec_model_id
        Codec_model = XCodec2Model_llasa_german.from_pretrained(codec_model_path)
        Codec_model.to(device)
        
        logger.info(f"LLaSA German: All primary models loaded.")

        # EXACT blueprint audio processing
        sample_audio_path = voice_id_override or model_config.get("default_voice_id")
        sample_audio_text = None  # Blueprint default
        target_text = text  # Blueprint variable name
        is_cloning_mode = False
        speech_ids_prefix = []

        # Handle zero-shot case
        if isinstance(sample_audio_path, str) and sample_audio_path.lower() == 'none':
            sample_audio_path = None
            logger.debug(f"LLaSA German: Voice ID override was string 'None', converted to Python None for zero-shot mode")

        if sample_audio_path and sample_audio_path.lower() != 'none':
            resolved_ref_path = Path(sample_audio_path).resolve()
            
            if resolved_ref_path.exists() and resolved_ref_path.is_file():
                is_cloning_mode = True
                logger.info(f"LLaSA German ({crisptts_model_id_for_log}): Cloning mode activated. Reference audio: {resolved_ref_path}")

                # Check for custom reference text
                if model_params_override:
                    try: 
                        parsed_params = json.loads(model_params_override)
                        sample_audio_text = parsed_params.get("reference_text")
                        if sample_audio_text:
                            logger.info(f"LLaSA German: Using provided reference text: '{sample_audio_text[:100]}...'")
                    except: 
                        pass
                
                # EXACT blueprint audio loading and processing
                waveform, sample_rate = torchaudio_llasa_german.load(sample_audio_path)
                
                max_secs = 15
                if len(waveform[0]) / sample_rate > 15:
                    logger.warning("LLaSA German: Reference audio longer than 15.0s. Trimming to first 15.0s.")
                    waveform = waveform[:, : sample_rate * 15]
                    waveform = torch_llasa_german.nn.functional.pad(waveform, (0, int(sample_rate * 0.5)), "constant", 0)

                if waveform.size(0) > 1:
                    waveform = torch_llasa_german.mean(waveform, dim=0, keepdim=True)

                prompt_wav = torchaudio_llasa_german.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
                
                # EXACT blueprint transcription handling
                if sample_audio_text is None:
                    logger.info("LLaSA German: Attempting to transcribe audio with Whisper...")
                    transcription = None
                    
                    try:
                        whisper_model_id = model_config.get("whisper_model_id_for_transcription", "openai/whisper-large-v3-turbo")
                        
                        whisper_turbo_pipe = pipeline_llasa_german(
                            "automatic-speech-recognition",
                            model=whisper_model_id,
                            torch_dtype=torch_llasa_german.float16 if device == "cuda" else torch_llasa_german.float32,
                            device=device if device != "mps" else "cpu",
                        )
                        
                        # Use original waveform for Whisper (not resampled)
                        transcription = whisper_turbo_pipe(waveform[0].cpu().numpy())["text"].strip()
                        logger.info(f"LLaSA German: Whisper transcription result: {transcription[:100]}")
                        
                    except Exception as whisper_error:
                        logger.warning(f"LLaSA German: Whisper transcription failed: {whisper_error}")
                        logger.info("LLaSA German: Falling back to default German transcription")
                        transcription = None
                    
                    # Fallback to default German transcription if Whisper fails
                    if not transcription or not transcription.strip():
                        transcription = "Das ist eine deutsche Sprachprobe für die Stimmenklonierung."
                        logger.info("LLaSA German: Using default German transcription (fallback)")
                else:
                    transcription = sample_audio_text
                    logger.info(f"LLaSA German: Using provided transcription: {transcription[:100]}")

                # EXACT blueprint text length check and combination
                if len(target_text) == 0:
                    raise ValueError("Target text must be provided!")
                elif len(target_text) > 500:
                    logger.warning("LLaSA German: Text is too long; trimming to first 500 characters.")
                    target_text = target_text[:500]

                # EXACT blueprint text combination
                input_text = transcription + " " + target_text
                logger.debug(f"LLaSA German: Combined input text: '{input_text[:100]}...'")
                
                # EXACT blueprint reference encoding
                logger.info(f"LLaSA German: Encoding reference audio...")
                with torch_llasa_german.no_grad():
                    vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav.to(device))
                    vq_code_prompt = vq_code_prompt[0, 0, :]
                    speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt.tolist())
                    logger.debug(f"LLaSA German: Encoded {len(speech_ids_prefix)} reference speech tokens")
            else:
                logger.error(f"LLaSA German ({crisptts_model_id_for_log}): Configured for cloning, but ref audio not found. Aborting.")
                return 
        else:
            # Zero-shot mode
            input_text = target_text
            speech_ids_prefix = []
            logger.info(f"LLaSA German ({crisptts_model_id_for_log}): No reference audio provided. Operating in zero-shot mode.")

        # EXACT blueprint text formatting and generation
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
        
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + "".join(speech_ids_prefix)}
        ]

        input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt", continue_final_message=True)
        input_ids = input_ids.to(device)
        speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        # EXACT blueprint generation parameters
        gen_params = {
            "max_length": 2048, 
            "eos_token_id": speech_end_id,
            "do_sample": True,
            "top_p": 1,
            "temperature": 0.8,
            "min_new_tokens": 4,  # Fix so the model does not directly stop 
        }

        # Override with user parameters if provided
        if model_params_override:
            try:
                cli_gen_params = json.loads(model_params_override)
                valid_hf_keys = {"temperature", "top_p", "top_k", "do_sample", "num_beams", 
                                 "repetition_penalty", "max_new_tokens", "min_new_tokens", "max_length"}
                for k, v_param in cli_gen_params.items():
                    if k in valid_hf_keys: 
                        gen_params[k] = v_param
            except: 
                logger.warning(f"LLaSA German ({crisptts_model_id_for_log}): Could not parse --model-params for generation.")
        
        logger.info(f"LLaSA German: Generating speech tokens...")
        
        with torch_llasa_german.no_grad():
            outputs = model.generate(input_ids, **gen_params)

            # EXACT blueprint token extraction for German models
            generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix) : -1]
            logger.debug(f"LLaSA German: German model extraction - input_ids.shape[1]: {input_ids.shape[1]}, speech_ids_prefix length: {len(speech_ids_prefix)}")
            logger.debug(f"LLaSA German: Extracted generated_ids shape: {generated_ids.shape}")
            
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            logger.debug(f"LLaSA German: Decoded {len(speech_tokens)} speech token strings")
            
            # EXACT blueprint speech ID extraction
            speech_tokens = extract_speech_ids(speech_tokens)
            logger.info(f"LLaSA German: Extracted {len(speech_tokens)} speech IDs for XCodec2")
            
            if not speech_tokens:
                logger.error(f"LLaSA German ({crisptts_model_id_for_log}): No speech tokens extracted!")
                return
            
            # EXACT blueprint tensor conversion and decoding
            speech_tokens = torch_llasa_german.tensor(speech_tokens).to(device).unsqueeze(0).unsqueeze(0)
            gen_wav = Codec_model.decode_code(speech_tokens)
            
            # EXACT blueprint audio trimming for German models
            if is_cloning_mode:
                gen_wav = gen_wav[:, :, prompt_wav.shape[1] :]
                logger.info(f"LLaSA German: Trimmed reference audio portion")

        # Extract final audio
        audio_output_np = gen_wav[0, 0, :].cpu().numpy()
        
        if audio_output_np.size == 0:
            logger.error(f"LLaSA German ({crisptts_model_id_for_log}): Generated empty audio!")
            return

        logger.info(f"LLaSA German: Synthesis successful. Output shape: {audio_output_np.shape}")

        # Save output - EXACT blueprint sample rate
        if output_file_str:
            output_path = Path(output_file_str).with_suffix(".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf_llasa_german.write(str(output_path), audio_output_np, 16000)
            logger.info(f"LLaSA German: Audio saved to {output_path}")
        
        if play_direct:
            logger.info(f"LLaSA German: Playing audio...")
            audio_int16 = (numpy_llasa_german.clip(audio_output_np, -1.0, 1.0) * 32767).astype(numpy_llasa_german.int16)
            play_audio(audio_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=16000)

    except Exception as e:
        logger.error(f"LLaSA German ({crisptts_model_id_for_log}): Synthesis failed: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.debug(f"LLaSA German ({crisptts_model_id_for_log}): Cleaning up...")
        
        if model is not None:
            try:
                model.cpu()
                del model
            except:
                pass
                
        if Codec_model is not None:
            try:
                Codec_model.cpu() 
                del Codec_model
            except:
                pass
                
        if tokenizer is not None:
            try:
                del tokenizer
            except:
                pass
                
        if whisper_turbo_pipe is not None:
            try:
                del whisper_turbo_pipe
            except:
                pass
        
        # Clear device caches
        if TORCH_AVAILABLE and torch_llasa_german:
            if device == "cuda":
                torch_llasa_german.cuda.empty_cache()
            elif device == "mps" and hasattr(torch_llasa_german.mps, "empty_cache"):
                try: 
                    torch_llasa_german.mps.empty_cache()
                except: 
                    pass
        
        gc.collect()
        logger.info(f"LLaSA German ({crisptts_model_id_for_log}): Handler finished and cleaned up.")