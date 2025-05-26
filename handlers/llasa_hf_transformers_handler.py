# handlers/llasa_hf_transformers_handler.py

import logging
import os
from pathlib import Path
import gc
import json
import tempfile
import re

# Conditional imports (same as your last provided version)
TORCH_AVAILABLE = False
torch_llasa_hf = None
IS_MPS_LLASA_HF = False
IS_CUDA_LLASA_HF = False
TRANSFORMERS_AVAILABLE = False
AutoTokenizer_llasa_hf, AutoModelForCausalLM_llasa_hf, pipeline_llasa_hf = None, None, None
XCODEC2_AVAILABLE = False
XCodec2Model_llasa_hf = None
SOUNDFILE_AVAILABLE = False
sf_llasa_hf = None
TORCHAUDIO_AVAILABLE = False
torchaudio_llasa_hf = None
NUMPY_AVAILABLE = False
numpy_llasa_hf = None
PYDUB_FOR_HANDLER_AVAILABLE = False
AudioSegment_pydub_handler = None
pydub_mediainfo_func_handler = None

logger_init = logging.getLogger("CrispTTS.handlers.llasa_hf_transformers.init")

try:
    import torch
    torch_llasa_hf = torch; TORCH_AVAILABLE = True
    if hasattr(torch_llasa_hf.backends, "mps") and torch_llasa_hf.backends.mps.is_available(): 
        IS_MPS_LLASA_HF = True
    if torch_llasa_hf.cuda.is_available():
        IS_CUDA_LLASA_HF = True
    logger_init.info("PyTorch loaded for LLaSA HF.")
except ImportError: logger_init.warning("PyTorch not found for LLaSA HF.")
if TORCH_AVAILABLE:
    try:
        import torchaudio
        torchaudio_llasa_hf = torchaudio; TORCHAUDIO_AVAILABLE = True
        logger_init.info("Torchaudio loaded for LLaSA HF.")
    except ImportError: logger_init.warning("Torchaudio not found for LLaSA HF.")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        AutoTokenizer_llasa_hf, AutoModelForCausalLM_llasa_hf, pipeline_llasa_hf = AutoTokenizer, AutoModelForCausalLM, pipeline
        TRANSFORMERS_AVAILABLE = True; logger_init.info("Transformers loaded for LLaSA HF.")
    except ImportError: logger_init.warning("Transformers not found for LLaSA HF.")
    try:
        from xcodec2.modeling_xcodec2 import XCodec2Model
        XCodec2Model_llasa_hf = XCodec2Model; XCODEC2_AVAILABLE = True
        logger_init.info("XCodec2 loaded for LLaSA HF.")
    except ImportError: logger_init.warning("XCodec2 not found for LLaSA HF.")
try:
    from pydub import AudioSegment as PydubAudioSegment_llasa_imp
    from pydub.utils import mediainfo as pydub_mediainfo_llasa_imp
    AudioSegment_pydub_handler, pydub_mediainfo_func_handler = PydubAudioSegment_llasa_imp, pydub_mediainfo_llasa_imp
    PYDUB_FOR_HANDLER_AVAILABLE = True; logger_init.info("pydub imported for LLaSA HF.")
except ImportError: logger_init.info("pydub not found for LLaSA HF.")
try:
    import soundfile as sf
    sf_llasa_hf = sf; SOUNDFILE_AVAILABLE = True
    logger_init.info("SoundFile loaded for LLaSA HF.")
except ImportError: logger_init.warning("SoundFile not found for LLaSA HF.")
try:
    import numpy as np
    numpy_llasa_hf = np; NUMPY_AVAILABLE = True
except ImportError: logger_init.warning("NumPy not found for LLaSA HF.")

from utils import save_audio, play_audio, SuppressOutput

logger = logging.getLogger("CrispTTS.handlers.llasa_hf_transformers")

def _get_optimal_device():
    """Get the best available device with proper MPS/CUDA detection"""
    if IS_MPS_LLASA_HF:
        return torch_llasa_hf.device("mps")
    elif IS_CUDA_LLASA_HF:
        return torch_llasa_hf.device("cuda")
    else:
        return torch_llasa_hf.device("cpu")

def _get_optimal_dtype(device):
    """Get optimal dtype for the device, ensuring consistency"""
    if device.type == "mps":
        # Use float32 for MPS to avoid mixed precision issues
        return torch_llasa_hf.float32
    elif device.type == "cuda":
        return torch_llasa_hf.float16
    else:
        return torch_llasa_hf.float32

def extract_speech_ids(speech_tokens_str):
    """EXACT blueprint implementation"""
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
    """EXACT blueprint implementation"""
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def synthesize_with_llasa_hf_transformers(
    model_config: dict,
    text: str,
    voice_id_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'llasa_hf_transformers_unknown')
    logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Starting synthesis. Target text: '{text[:50]}...'")

    if not all([TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, XCODEC2_AVAILABLE, SOUNDFILE_AVAILABLE, NUMPY_AVAILABLE]):
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Missing core dependencies. Skipping.")
        return
    
    llm_model_id = model_config.get("llm_model_id")
    tokenizer_id = model_config.get("tokenizer_id", llm_model_id)
    codec_model_id = model_config.get("codec_model_id")
    
    if not all([llm_model_id, tokenizer_id, codec_model_id]):
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Core model IDs not fully configured. Skipping.")
        return

    hf_token = os.getenv("HF_TOKEN") if model_config.get("requires_hf_token") else None
    
    # Use the device detection with proper MPS support
    device = _get_optimal_device()
    dtype = _get_optimal_dtype(device)
    device_str = str(device)
    logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Using device: {device}, dtype: {dtype}")

    # Initialize models to None for proper cleanup
    tokenizer = None
    model = None  
    Codec_model = None
    whisper_turbo_pipe = None
    prompt_wav = None  # Store for audio trimming

    try:
        # EXACT blueprint model loading - variable names must match!
        llasa_model_name = llm_model_id
        
        # Load tokenizer with pad token setup
        logger.info(f"LLaSA HF: Loading tokenizer model: {llasa_model_name}")
        tokenizer = AutoTokenizer_llasa_hf.from_pretrained(llasa_model_name)
        
        # Ensure tokenizer has pad token to avoid attention mask warnings
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug("LLaSA HF: Set pad_token to eos_token")
            else:
                tokenizer.add_special_tokens({'pad_token': '<pad>'})
                logger.debug("LLaSA HF: Added <pad> token")
        
        logger.info(f"LLaSA HF: Loading llm model: {llasa_model_name}")
        model = AutoModelForCausalLM_llasa_hf.from_pretrained(llasa_model_name)
        model.to(device)
        
        logger.info(f"LLaSA HF: Loading codec model: {codec_model_id}")
        codec_model_path = codec_model_id
        Codec_model = XCodec2Model_llasa_hf.from_pretrained(codec_model_path)
        Codec_model.to(device)
        
        logger.info(f"LLaSA HF: All primary models loaded.")

        ref_audio_path_str_config = voice_id_override or model_config.get("default_voice_id")
        is_cloning_mode = False
        speech_ids_prefix = []
        target_text = text  # Blueprint variable name

        # Handle special case where voice_id_override is string "None" (from zero-shot test)
        if isinstance(ref_audio_path_str_config, str) and ref_audio_path_str_config.lower() == 'none':
            ref_audio_path_str_config = None
            logger.debug(f"LLaSA HF: Voice ID override was string 'None', converted to Python None for zero-shot mode")

        if ref_audio_path_str_config and ref_audio_path_str_config.lower() != 'none':
            p_input = Path(ref_audio_path_str_config)
            resolved_ref_path = p_input if p_input.is_absolute() else (Path.cwd() / p_input).resolve()
            
            logger.info(f"LLaSA HF: Attempting to use reference audio at resolved path: {resolved_ref_path}")
            if resolved_ref_path.exists() and resolved_ref_path.is_file() and os.access(str(resolved_ref_path), os.R_OK):
                is_cloning_mode = True
                logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Cloning mode activated. Reference audio: {resolved_ref_path}")

                if not TORCHAUDIO_AVAILABLE:
                    logger.error("LLaSA HF: Torchaudio not available, cannot process reference audio for cloning. Aborting.")
                    return

                # EXACT blueprint audio processing
                sample_audio_path = str(resolved_ref_path)
                sample_audio_text = None  # Blueprint default
                
                # Check for custom reference text
                if model_params_override:
                    try: 
                        parsed_params = json.loads(model_params_override)
                        sample_audio_text = parsed_params.get("reference_text")
                        if sample_audio_text:
                            logger.info(f"LLaSA HF: Using provided reference text: '{sample_audio_text[:100]}...'")
                    except: 
                        pass
                
                # EXACT blueprint audio loading and processing
                waveform, sample_rate = torchaudio_llasa_hf.load(sample_audio_path)
                
                max_secs = 15
                if len(waveform[0]) / sample_rate > 15:
                    logger.warning("LLaSA HF: Reference audio longer than 15.0s. Trimming to first 15.0s.")
                    waveform = waveform[:, : sample_rate * 15]
                    waveform = torch_llasa_hf.nn.functional.pad(waveform, (0, int(sample_rate * 0.5)), "constant", 0)

                if waveform.size(0) > 1:
                    waveform = torch_llasa_hf.mean(waveform, dim=0, keepdim=True)

                prompt_wav = torchaudio_llasa_hf.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
                
                # EXACT blueprint transcription handling with fallback
                if sample_audio_text is None:
                    logger.info("LLaSA HF: Attempting to transcribe audio with Whisper...")
                    transcription = None
                    
                    try:
                        whisper_model_id = model_config.get("whisper_model_id_for_transcription", "openai/whisper-large-v3-turbo")
                        
                        # Try to create Whisper pipeline with error handling
                        whisper_turbo_pipe = pipeline_llasa_hf(
                            "automatic-speech-recognition",
                            model=whisper_model_id,
                            torch_dtype=dtype,
                            device=device_str if device_str != "mps" else "cpu",  # Whisper doesn't support MPS well
                        )
                        
                        # Use original waveform for Whisper (not resampled)
                        transcription = whisper_turbo_pipe(waveform[0].cpu().numpy())["text"].strip()
                        logger.info(f"LLaSA HF: Whisper transcription result: {transcription[:100]}")
                        
                    except Exception as whisper_error:
                        logger.warning(f"LLaSA HF: Whisper transcription failed: {whisper_error}")
                        logger.info("LLaSA HF: Falling back to default German transcription")
                        transcription = None
                    
                    # Fallback to default German transcription if Whisper fails
                    if not transcription or not transcription.strip():
                        transcription = "Das ist eine deutsche Sprachprobe für die Stimmenklonierung."
                        logger.info("LLaSA HF: Using default German transcription (fallback)")
                else:
                    transcription = sample_audio_text
                    logger.info(f"LLaSA HF: Using provided transcription: {transcription[:100]}")

                # EXACT blueprint text length check and combination
                if len(target_text) == 0:
                    raise ValueError("Target text must be provided!")
                elif len(target_text) > 500:
                    logger.warning("LLaSA HF: Text is too long; trimming to first 500 characters.")
                    target_text = target_text[:500]

                # EXACT blueprint text combination - both models use same approach
                input_text = transcription + " " + target_text
                logger.debug(f"LLaSA HF: Combined input text: '{input_text[:100]}...'")
                
                # EXACT blueprint reference encoding
                logger.info(f"LLaSA HF: Encoding reference audio...")
                with torch_llasa_hf.no_grad():
                    vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav.to(device))
                    vq_code_prompt = vq_code_prompt[0, 0, :]
                    speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt.tolist())
                    logger.debug(f"LLaSA HF: Encoded {len(speech_ids_prefix)} reference speech tokens")
            else:
                is_cloning_mode = False
                input_text = target_text
                if "clone" in crisptts_model_id_for_log.lower():
                    logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Configured for cloning, but ref audio not found. Aborting.")
                    return 
                logger.warning(f"LLaSA HF ({crisptts_model_id_for_log}): Ref audio not found. Operating in zero-shot mode.")
        else:
            is_cloning_mode = False
            input_text = target_text
            speech_ids_prefix = []
            logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): No reference audio provided. Operating in zero-shot mode.")

        # EXACT blueprint text formatting and generation
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
        
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + "".join(speech_ids_prefix)}
        ]

        input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt", continue_final_message=True)
        input_ids = input_ids.to(device)
        
        # Create attention mask to avoid warnings
        attention_mask = torch_llasa_hf.ones_like(input_ids)
        
        speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        # EXACT blueprint generation parameters with model-specific tuning
        gen_params = {
            "max_length": 2048, 
            "eos_token_id": speech_end_id,
            "do_sample": True,
            "top_p": 1,
            "temperature": 0.8,
            "min_new_tokens": 4,  # Fix so the model does not directly stop
            "attention_mask": attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
        }
        
        # Adjust parameters based on model type
        if "multilingual" in crisptts_model_id_for_log.lower():
            # Multilingual models may need more aggressive generation settings
            gen_params["min_new_tokens"] = 50  # Force more generation for multilingual
            gen_params["max_length"] = 3000    # Allow longer sequences
            logger.debug("LLaSA HF: Using enhanced generation parameters for multilingual model")

        # Override with user parameters if provided
        if model_params_override:
            try:
                cli_gen_params = json.loads(model_params_override)
                valid_hf_keys = {"temperature", "top_p", "top_k", "do_sample", "num_beams", 
                                 "repetition_penalty", "max_new_tokens", "min_new_tokens", "max_length"}
                for k, v_param in cli_gen_params.items():
                    if k in valid_hf_keys: 
                        gen_params[k] = v_param
                        logger.debug(f"LLaSA HF: Overriding {k} = {v_param}")
            except: 
                logger.warning(f"LLaSA HF ({crisptts_model_id_for_log}): Could not parse --model-params for generation.")
        
        logger.info(f"LLaSA HF: Generating speech tokens...")
        
        with torch_llasa_hf.no_grad():
            outputs = model.generate(input_ids, **gen_params)

            # EXACT blueprint token extraction - THIS IS CRITICAL!
            # Different handling for German vs Multilingual models
            if "german" in crisptts_model_id_for_log.lower():
                # German model: use exact blueprint extraction
                generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix) : -1]
                logger.debug(f"LLaSA HF: German model extraction - input_ids.shape[1]: {input_ids.shape[1]}, speech_ids_prefix length: {len(speech_ids_prefix)}")
            else:
                # Multilingual model: different extraction pattern
                if is_cloning_mode and speech_ids_prefix:
                    # For multilingual cloning, don't subtract prefix length
                    generated_ids = outputs[0][input_ids.shape[1] : -1]
                    logger.debug(f"LLaSA HF: Multilingual cloning extraction - input_ids.shape[1]: {input_ids.shape[1]}")
                else:
                    # For multilingual zero-shot
                    generated_ids = outputs[0][input_ids.shape[1] : -1]
                    logger.debug(f"LLaSA HF: Multilingual zero-shot extraction - input_ids.shape[1]: {input_ids.shape[1]}")
            
            logger.debug(f"LLaSA HF: Extracted generated_ids shape: {generated_ids.shape}")
            
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            logger.debug(f"LLaSA HF: Decoded {len(speech_tokens)} speech token strings")
            logger.debug(f"LLaSA HF: First 10 speech tokens: {speech_tokens[:10]}")
            
            # EXACT blueprint speech ID extraction
            speech_tokens = extract_speech_ids(speech_tokens)
            logger.info(f"LLaSA HF: Extracted {len(speech_tokens)} speech IDs for XCodec2")
            
            if not speech_tokens:
                logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): No speech tokens extracted!")
                logger.debug(f"LLaSA HF: Raw decoded tokens were: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[:20]}")
                return
            
            # EXACT blueprint tensor conversion and decoding
            speech_tokens_tensor = torch_llasa_hf.tensor(speech_tokens).to(device).unsqueeze(0).unsqueeze(0)
            gen_wav = Codec_model.decode_code(speech_tokens_tensor)
            
            # EXACT blueprint audio trimming - FIXED for multilingual
            if is_cloning_mode and prompt_wav is not None:
                original_shape = gen_wav.shape
                ref_samples = prompt_wav.shape[1]
                
                # For multilingual models, the generation behavior is different
                if "german" in crisptts_model_id_for_log.lower():
                    # German model: trim reference as in blueprint
                    if gen_wav.shape[2] > ref_samples:
                        gen_wav = gen_wav[:, :, ref_samples:]
                        logger.info(f"LLaSA HF: German model trimming - Original: {original_shape}, After trim: {gen_wav.shape}")
                    else:
                        logger.warning(f"LLaSA HF: German model - generated audio too short for trimming")
                else:
                    # Multilingual model: no trimming needed - it doesn't concatenate reference
                    logger.info(f"LLaSA HF: Multilingual model - no trimming needed (doesn't include reference in output)")
            else:
                logger.debug(f"LLaSA HF: No audio trimming needed (zero-shot mode)")

        # Extract final audio
        audio_output_np = gen_wav[0, 0, :].cpu().numpy()
        
        # Additional quality check
        if audio_output_np.size == 0:
            logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Generated empty audio!")
            return
        
        # Check for silence/very quiet audio which might indicate issues
        audio_rms = numpy_llasa_hf.sqrt(numpy_llasa_hf.mean(audio_output_np**2))
        logger.debug(f"LLaSA HF: Audio RMS level: {audio_rms:.6f}")
        
        if audio_rms < 0.001:
            logger.warning(f"LLaSA HF: Generated audio has very low RMS ({audio_rms:.6f}) - may be mostly silence")

        logger.info(f"LLaSA HF: Synthesis successful. Output shape: {audio_output_np.shape}, Duration: {len(audio_output_np)/16000:.2f}s")

        # Save output - EXACT blueprint sample rate
        if output_file_str:
            output_path = Path(output_file_str).with_suffix(".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf_llasa_hf.write(str(output_path), audio_output_np, 16000)  # Blueprint uses 16000
            logger.info(f"LLaSA HF: Audio saved to {output_path}")
        
        if play_direct:
            logger.info(f"LLaSA HF: Playing audio...")
            audio_int16 = (numpy_llasa_hf.clip(audio_output_np, -1.0, 1.0) * 32767).astype(numpy_llasa_hf.int16)
            play_audio(audio_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=16000)

    except Exception as e:
        logger.error(f"LLaSA HF ({crisptts_model_id_for_log}): Synthesis failed: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.debug(f"LLaSA HF ({crisptts_model_id_for_log}): Cleaning up...")
        
        # Clear models from memory
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
        if TORCH_AVAILABLE and torch_llasa_hf:
            if device.type == "cuda":
                torch_llasa_hf.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch_llasa_hf.mps, "empty_cache"):
                try: 
                    torch_llasa_hf.mps.empty_cache()
                except Exception as e_mps: 
                    logger.debug(f"LLaSA HF: Error clearing MPS cache: {e_mps}")
        
        gc.collect()
        logger.info(f"LLaSA HF ({crisptts_model_id_for_log}): Handler finished and cleaned up.")

def clear_llasa_hf_cache():
    """Clear any remaining caches"""
    if TORCH_AVAILABLE and torch_llasa_hf:
        if torch_llasa_hf.cuda.is_available():
            torch_llasa_hf.cuda.empty_cache()
        if hasattr(torch_llasa_hf.backends, "mps") and torch_llasa_hf.backends.mps.is_available():
            try:
                torch_llasa_hf.mps.empty_cache()
            except:
                pass
    gc.collect()
    logger.info("LLaSA HF: Caches cleared.")