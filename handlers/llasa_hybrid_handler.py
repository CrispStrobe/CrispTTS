# handlers/llasa_hybrid_handler.py
"""
Robust LLaSA Hybrid Handler for CrispTTS
Combines MLX LLM with PyTorch XCodec2 for speech synthesis with voice cloning
"""

import logging
import os
from pathlib import Path
import gc
import json
import numpy as np
import sys
import re
import tempfile
import shutil
from typing import Optional, List, Tuple, Union

# --- Conditional Imports with Robust Error Handling ---
MLX_LM_AVAILABLE = False
XCODEC2_AVAILABLE = False
TRANSFORMERS_FOR_LLASA_AVAILABLE = False
TORCH_FOR_LLASA_AVAILABLE = False
SF_FOR_LLASA_AVAILABLE = False

# Import placeholders
mlx_lm_load, mlx_lm_generate_string_output = None, None 
mlx_core = None
XCodec2Model_llasa = None 
AutoTokenizer_llasa_chat = None 
torch_llasa = None 
sf_llasa = None 

logger_init = logging.getLogger("CrispTTS.handlers.llasa_hybrid.init")

# MLX imports
try:
    import mlx.core as mx_imported
    from mlx_lm import load as mlx_lm_load_imported
    from mlx_lm import generate as mlx_lm_generate_imported
    
    mlx_core = mx_imported
    mlx_lm_load = mlx_lm_load_imported
    mlx_lm_generate_string_output = mlx_lm_generate_imported
    MLX_LM_AVAILABLE = True
    logger_init.info("MLX and mlx-lm imported successfully for LLaSA Handler.")
except ImportError as e:
    print(f"LLaSA Handler INIT ERROR: MLX/mlx-lm import failed: {e}", file=sys.stderr)
    logger_init.warning("MLX/mlx-lm not found. LLaSA Hybrid handler will be non-functional.")
except Exception as e:
    print(f"LLaSA Handler INIT ERROR: Unexpected MLX error: {e}", file=sys.stderr)
    logger_init.error(f"Unexpected MLX import error: {e}")

# XCodec2 imports
try:
    from xcodec2.modeling_xcodec2 import XCodec2Model as XCodec2Model_imported
    XCodec2Model_llasa = XCodec2Model_imported
    XCODEC2_AVAILABLE = True
    logger_init.info("XCodec2 imported successfully for LLaSA Handler.")
except ImportError as e:
    print(f"LLaSA Handler INIT ERROR: XCodec2 import failed: {e}", file=sys.stderr)
    logger_init.warning("XCodec2 library not found. LLaSA Hybrid handler will be non-functional.")
except Exception as e:
    print(f"LLaSA Handler INIT ERROR: Unexpected XCodec2 error: {e}", file=sys.stderr)
    logger_init.error(f"Unexpected XCodec2 import error: {e}")

# Transformers imports
try:
    from transformers import AutoTokenizer as AutoTokenizer_imported
    AutoTokenizer_llasa_chat = AutoTokenizer_imported
    TRANSFORMERS_FOR_LLASA_AVAILABLE = True
    logger_init.info("Transformers (for LLaSA Chat Tokenizer) imported successfully.")
except ImportError as e:
    print(f"LLaSA Handler INIT ERROR: Transformers import failed: {e}", file=sys.stderr)
    logger_init.warning("Transformers (for LLaSA Chat Tokenizer) not found. LLaSA Hybrid handler may fail.")
except Exception as e:
    print(f"LLaSA Handler INIT ERROR: Unexpected Transformers error: {e}", file=sys.stderr)
    logger_init.error(f"Unexpected Transformers import error: {e}")

# PyTorch imports
try:
    import torch as torch_imported
    torch_llasa = torch_imported
    TORCH_FOR_LLASA_AVAILABLE = True
    logger_init.info("PyTorch (for LLaSA XCodec2) imported successfully.")
except ImportError as e:
    print(f"LLaSA Handler INIT ERROR: PyTorch import failed: {e}", file=sys.stderr)
    logger_init.warning("PyTorch (for LLaSA XCodec2) not found. LLaSA Hybrid handler will be non-functional.")
except Exception as e:
    print(f"LLaSA Handler INIT ERROR: Unexpected PyTorch error: {e}", file=sys.stderr)
    logger_init.error(f"Unexpected PyTorch import error: {e}")

# SoundFile imports
try:
    import soundfile as sf_imported
    sf_llasa = sf_imported
    SF_FOR_LLASA_AVAILABLE = True
    logger_init.info("SoundFile imported successfully for LLaSA Handler.")
except ImportError as e:
    print(f"LLaSA Handler INIT ERROR: SoundFile import failed: {e}", file=sys.stderr)
    logger_init.warning("SoundFile not found. LLaSA Hybrid handler cannot save audio.")
except Exception as e:
    print(f"LLaSA Handler INIT ERROR: Unexpected SoundFile error: {e}", file=sys.stderr)
    logger_init.error(f"Unexpected SoundFile import error: {e}")

# Project utils imports
try:
    from utils import save_audio, play_audio, SuppressOutput
    UTILS_AVAILABLE = True
    logger_init.info("Project utils imported successfully for LLaSA Handler.")
except ImportError as e:
    print(f"LLaSA Handler CRITICAL INIT ERROR: Failed to import from 'utils': {e}", file=sys.stderr)
    logger_init.error(f"Failed to import project utils: {e}")
    UTILS_AVAILABLE = False

logger = logging.getLogger("CrispTTS.handlers.llasa_hybrid")

# --- Helper Functions ---
def _validate_dependencies() -> Tuple[bool, List[str]]:
    """Validate all required dependencies are available."""
    missing_deps = []
    
    if not MLX_LM_AVAILABLE:
        missing_deps.append("MLX/mlx-lm")
    if not XCODEC2_AVAILABLE:
        missing_deps.append("XCodec2")
    if not TRANSFORMERS_FOR_LLASA_AVAILABLE:
        missing_deps.append("Transformers")
    if not TORCH_FOR_LLASA_AVAILABLE:
        missing_deps.append("PyTorch")
    if not SF_FOR_LLASA_AVAILABLE:
        missing_deps.append("SoundFile")
    if not UTILS_AVAILABLE:
        missing_deps.append("Project Utils")
    
    return len(missing_deps) == 0, missing_deps

def _validate_model_objects() -> Tuple[bool, List[str]]:
    """Validate all imported model objects are available."""
    missing_objects = []
    
    if not mlx_lm_load:
        missing_objects.append("mlx_lm.load")
    if not mlx_lm_generate_string_output:
        missing_objects.append("mlx_lm.generate")
    if not XCodec2Model_llasa:
        missing_objects.append("XCodec2Model")
    if not AutoTokenizer_llasa_chat:
        missing_objects.append("AutoTokenizer")
    if not torch_llasa:
        missing_objects.append("torch")
    if not sf_llasa:
        missing_objects.append("soundfile")
    
    return len(missing_objects) == 0, missing_objects

def _llasa_ids_to_speech_tokens_str(speech_ids: List[int]) -> List[str]:
    """Convert speech IDs to speech token strings."""
    if not speech_ids:
        return []
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids if isinstance(speech_id, int)]

def _llasa_extract_speech_ids_from_str_list(speech_tokens_list_of_str: List[str]) -> List[int]:
    """Extract speech IDs from speech token strings."""
    speech_ids = []
    for token_str in speech_tokens_list_of_str:
        if not isinstance(token_str, str):
            continue
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            try:
                speech_id = int(num_str)
                speech_ids.append(speech_id)
            except ValueError:
                logger.warning(f"LLaSA: Could not parse int from speech token: {token_str}")
    return speech_ids

def _get_pytorch_device() -> Tuple[str, Optional['torch.device']]:
    """Get the best available PyTorch device."""
    if not torch_llasa:
        return "cpu", None
    
    device_str = "cpu"
    if torch_llasa.cuda.is_available():
        device_str = "cuda"
        logger.info("LLaSA: Using CUDA for PyTorch operations")
    elif hasattr(torch_llasa.backends, "mps") and torch_llasa.backends.mps.is_available():
        device_str = "mps"
        logger.info("LLaSA: Using MPS for PyTorch operations")
    else:
        logger.info("LLaSA: Using CPU for PyTorch operations")
    
    try:
        device = torch_llasa.device(device_str)
        return device_str, device
    except Exception as e:
        logger.warning(f"LLaSA: Failed to create device '{device_str}': {e}. Falling back to CPU.")
        return "cpu", torch_llasa.device("cpu")

def _validate_and_prepare_reference_audio(ref_wav_path_str: str, project_root: Path) -> Optional[Tuple[Path, np.ndarray, int]]:
    """Validate and prepare reference audio file."""
    if not ref_wav_path_str or not isinstance(ref_wav_path_str, str):
        return None
    
    ref_wav_path_actual = Path(ref_wav_path_str)
    if not ref_wav_path_actual.is_absolute():
        ref_wav_path_actual = (project_root / ref_wav_path_actual).resolve()
    
    if not ref_wav_path_actual.exists():
        logger.error(f"LLaSA Hybrid: Reference WAV file not found: {ref_wav_path_actual}")
        return None
    
    if not ref_wav_path_actual.is_file():
        logger.error(f"LLaSA Hybrid: Reference path is not a file: {ref_wav_path_actual}")
        return None
    
    if ref_wav_path_actual.suffix.lower() not in ['.wav', '.flac', '.mp3']:
        logger.warning(f"LLaSA Hybrid: Reference file may not be a supported audio format: {ref_wav_path_actual}")
    
    try:
        prompt_wav_samples, sr_prompt = sf_llasa.read(str(ref_wav_path_actual), dtype='float32')
        
        if prompt_wav_samples.size == 0:
            logger.error(f"LLaSA Hybrid: Reference audio file is empty: {ref_wav_path_actual}")
            return None
        
        # Handle stereo to mono conversion
        if prompt_wav_samples.ndim > 1:
            prompt_wav_samples = np.mean(prompt_wav_samples, axis=1)
            logger.info("LLaSA Hybrid: Converted stereo reference audio to mono")
        
        # Validate sample rate
        if sr_prompt != 16000:
            logger.warning(f"LLaSA Hybrid: Reference WAV SR is {sr_prompt}Hz, XCodec2 expects 16000Hz. Results may vary.")
        
        # Validate duration (reasonable limits)
        duration_seconds = len(prompt_wav_samples) / sr_prompt
        if duration_seconds > 30:
            logger.warning(f"LLaSA Hybrid: Reference audio is {duration_seconds:.1f}s long, which may be too long for effective voice cloning.")
        elif duration_seconds < 1:
            logger.warning(f"LLaSA Hybrid: Reference audio is only {duration_seconds:.1f}s long, which may be too short for effective voice cloning.")
        
        logger.info(f"LLaSA Hybrid: Reference audio loaded: {duration_seconds:.1f}s at {sr_prompt}Hz")
        return ref_wav_path_actual, prompt_wav_samples, sr_prompt
        
    except Exception as e:
        logger.error(f"LLaSA Hybrid: Failed to load reference audio '{ref_wav_path_actual}': {e}")
        return None

def _cleanup_resources(*resources):
    """Safely cleanup resources."""
    for resource in resources:
        if resource is not None:
            try:
                del resource
            except Exception as e:
                logger.debug(f"LLaSA: Error during resource cleanup: {e}")
    
    # Cleanup PyTorch cache
    if TORCH_FOR_LLASA_AVAILABLE and torch_llasa:
        try:
            if hasattr(torch_llasa, 'cuda') and torch_llasa.cuda.is_available():
                torch_llasa.cuda.empty_cache()
            if hasattr(torch_llasa, 'mps') and hasattr(torch_llasa.backends, "mps") and torch_llasa.backends.mps.is_available():
                if hasattr(torch_llasa.mps, 'empty_cache'):
                    torch_llasa.mps.empty_cache()
        except Exception as e:
            logger.debug(f"LLaSA: Error during PyTorch cache cleanup: {e}")
    
    # Cleanup MLX cache
    if MLX_LM_AVAILABLE and mlx_core:
        try:
            if hasattr(mlx_core, 'clear_cache'):
                mlx_core.clear_cache()
        except Exception as e:
            logger.debug(f"LLaSA: Error during MLX cache cleanup: {e}")
    
    gc.collect()

# --- Main Synthesis Function ---
def synthesize_with_llasa_hybrid(
    model_config: dict, 
    text_to_synthesize: str, 
    voice_id_override: Optional[str], 
    model_params_override: Optional[str], 
    output_file_str: Optional[str], 
    play_direct: bool
) -> None:
    """
    Synthesize speech using LLaSA Hybrid approach.
    
    Args:
        model_config: Configuration dictionary for the model
        text_to_synthesize: Text to convert to speech
        voice_id_override: Optional voice/reference audio override
        model_params_override: Optional JSON string of model parameters
        output_file_str: Optional output file path
        play_direct: Whether to play audio directly after synthesis
    """
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'llasa_hybrid_unknown')
    
    # Validate dependencies
    deps_valid, missing_deps = _validate_dependencies()
    if not deps_valid:
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Missing core dependencies: {', '.join(missing_deps)}. Skipping.")
        return
    
    # Validate model objects
    objects_valid, missing_objects = _validate_model_objects()
    if not objects_valid:
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Missing critical objects: {', '.join(missing_objects)}. Skipping.")
        return
    
    # Validate input text
    if not text_to_synthesize or not isinstance(text_to_synthesize, str) or not text_to_synthesize.strip():
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Invalid or empty input text.")
        return
    
    # Get configuration
    llm_model_id_cfg = model_config.get("llm_model_id")
    chat_tokenizer_id_cfg = model_config.get("chat_tokenizer_id")
    codec_model_id_cfg = model_config.get("codec_model_id")
    ref_speaker_wav_path_str = voice_id_override or model_config.get("default_voice_id")
    sample_rate = model_config.get("sample_rate", 16000)
    
    # Validate configuration
    if not all([llm_model_id_cfg, chat_tokenizer_id_cfg, codec_model_id_cfg]):
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Missing required config - LLM: {llm_model_id_cfg}, Chat Tokenizer: {chat_tokenizer_id_cfg}, Codec: {codec_model_id_cfg}. Skipping.")
        return
    
    logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Starting synthesis with LLM '{llm_model_id_cfg}', Codec '{codec_model_id_cfg}'.")
    
    # Get PyTorch device
    pt_device_str, pt_device = _get_pytorch_device()
    if pt_device is None:
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Failed to get PyTorch device. Skipping.")
        return
    
    logger.info(f"LLaSA Hybrid: XCodec2 (PyTorch) using device: {pt_device_str}. MLX LLM uses default MLX device.")
    
    # Initialize model variables
    llasa_llm_mlx_model = None
    llasa_llm_mlx_tokenizer = None 
    chat_template_hf_tokenizer = None 
    codec_model_pt_inst = None
    
    try:
        # Load models with robust error handling
        logger.info("LLaSA Hybrid: Loading models...")
        
        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=not logger.isEnabledFor(logging.DEBUG)):
            try:
                logger.debug(f"LLaSA Hybrid: Loading MLX LLM '{llm_model_id_cfg}' & its tokenizer via mlx_lm.load()...")
                llasa_llm_mlx_model, llasa_llm_mlx_tokenizer = mlx_lm_load(llm_model_id_cfg)
                if not llasa_llm_mlx_model or not llasa_llm_mlx_tokenizer:
                    raise ValueError("MLX model or tokenizer is None after loading")
                logger.debug("LLaSA Hybrid: MLX LLM loaded successfully.")
            except Exception as e:
                logger.error(f"LLaSA Hybrid: Failed to load MLX LLM '{llm_model_id_cfg}': {e}")
                return
            
            try:
                logger.debug(f"LLaSA Hybrid: Loading Chat Template Tokenizer '{chat_tokenizer_id_cfg}' from Hugging Face...")
                chat_template_hf_tokenizer = AutoTokenizer_llasa_chat.from_pretrained(chat_tokenizer_id_cfg)
                if not chat_template_hf_tokenizer:
                    raise ValueError("Chat template tokenizer is None after loading")
                logger.debug("LLaSA Hybrid: Chat template tokenizer loaded successfully.")
            except Exception as e:
                logger.error(f"LLaSA Hybrid: Failed to load chat template tokenizer '{chat_tokenizer_id_cfg}': {e}")
                return
            
            try:
                logger.debug(f"LLaSA Hybrid: Loading PyTorch XCodec2 Model '{codec_model_id_cfg}'...")
                codec_model_pt_inst = XCodec2Model_llasa.from_pretrained(codec_model_id_cfg).to(pt_device).eval()
                if not codec_model_pt_inst:
                    raise ValueError("XCodec2 model is None after loading")
                logger.debug("LLaSA Hybrid: XCodec2 model loaded successfully.")
            except Exception as e:
                logger.error(f"LLaSA Hybrid: Failed to load XCodec2 model '{codec_model_id_cfg}': {e}")
                return
        
        logger.info("LLaSA Hybrid: All models and tokenizers loaded successfully.")
        
        # Prepare voice cloning prompt
        assistant_content_prefix_str = "<|SPEECH_GENERATION_START|>"
        speech_ids_prefix_for_llm_strings = []
        
        if ref_speaker_wav_path_str:
            project_root = Path(__file__).resolve().parent.parent
            ref_audio_result = _validate_and_prepare_reference_audio(ref_speaker_wav_path_str, project_root)
            
            if ref_audio_result:
                ref_wav_path_actual, prompt_wav_samples, sr_prompt = ref_audio_result
                
                try:
                    logger.info(f"LLaSA Hybrid: Processing reference WAV for voice cloning...")
                    prompt_wav_tensor_pt = torch_llasa.from_numpy(prompt_wav_samples).float().unsqueeze(0).to(pt_device)
                    
                    with torch_llasa.no_grad(), SuppressOutput():
                        vq_codes_from_prompt_pt = codec_model_pt_inst.encode_code(input_waveform=prompt_wav_tensor_pt)
                    
                    if vq_codes_from_prompt_pt is not None and vq_codes_from_prompt_pt.numel() > 0:
                        prompt_speech_ids_integers = vq_codes_from_prompt_pt[0, 0, :].tolist()
                        speech_ids_prefix_for_llm_strings = _llasa_ids_to_speech_tokens_str(prompt_speech_ids_integers)
                        assistant_content_prefix_str += "".join(speech_ids_prefix_for_llm_strings)
                        logger.info(f"LLaSA Hybrid: Generated {len(speech_ids_prefix_for_llm_strings)} speech prefix tokens from reference WAV.")
                    else:
                        logger.warning("LLaSA Hybrid: XCodec2 encoding returned empty codes. Proceeding without voice cloning.")
                        
                except Exception as e:
                    logger.error(f"LLaSA Hybrid: Failed to process reference audio: {e}")
                    logger.info("LLaSA Hybrid: Proceeding without voice cloning.")
            else:
                logger.warning("LLaSA Hybrid: Reference audio validation failed. Proceeding without voice cloning.")
        
        # Prepare chat template
        user_content = f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text_to_synthesize}<|TEXT_UNDERSTANDING_END|>"
        chat_for_llasa_template = [
            {"role": "user", "content": user_content}, 
            {"role": "assistant", "content": assistant_content_prefix_str}
        ]
        
        try:
            prompt_string_for_llm = chat_template_hf_tokenizer.apply_chat_template(
                chat_for_llasa_template, tokenize=False, add_generation_prompt=True
            )
            if not prompt_string_for_llm:
                raise ValueError("Chat template application returned empty string")
            logger.debug(f"LLaSA Hybrid: Chat template applied successfully. Prompt length: {len(prompt_string_for_llm)} chars")
        except Exception as e:
            logger.error(f"LLaSA Hybrid: Failed to apply chat template: {e}")
            return
        
        # Calculate token limits
        speech_gen_end_token_str = '<|SPEECH_GENERATION_END|>'
        
        try:
            prompt_tokens_for_llm = llasa_llm_mlx_tokenizer.encode(prompt_string_for_llm)
            max_new_tokens_llm = 2048 - len(prompt_tokens_for_llm) - 10  # Safety margin
            max_new_tokens_llm = max(50, min(max_new_tokens_llm, 1800))  # Clamp to reasonable bounds
            logger.debug(f"LLaSA Hybrid: Prompt tokens: {len(prompt_tokens_for_llm)}, Max new tokens: {max_new_tokens_llm}")
        except Exception as e:
            logger.error(f"LLaSA Hybrid: Failed to tokenize prompt: {e}")
            return
        
        # Parse generation parameters
        llm_gen_params = {"temp": 0.8, "top_p": 1.0}  # Defaults
        if model_params_override:
            try:
                cli_params = json.loads(model_params_override)
                if "temperature" in cli_params:
                    llm_gen_params["temp"] = float(cli_params["temperature"])
                if "top_p" in cli_params:
                    llm_gen_params["top_p"] = float(cli_params["top_p"])
                logger.debug(f"LLaSA Hybrid: Using custom generation parameters: {llm_gen_params}")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"LLaSA Hybrid: Could not parse --model-params '{model_params_override}': {e}. Using defaults.")
        
        # Generate speech tokens
        logger.info(f"LLaSA Hybrid: Generating speech tokens (max_new_tokens={max_new_tokens_llm})...")
        
        try:
            generated_completion_str = mlx_lm_generate_string_output(
                model=llasa_llm_mlx_model, 
                tokenizer=llasa_llm_mlx_tokenizer,
                prompt=prompt_string_for_llm, 
                max_tokens=max_new_tokens_llm, 
                eos_token_texts=[speech_gen_end_token_str], 
                verbose=False,
                **llm_gen_params
            )
            
            if not generated_completion_str:
                logger.error("LLaSA Hybrid: LLM generation returned empty string.")
                return
            
            logger.debug(f"LLaSA Hybrid: Generated {len(generated_completion_str)} characters of output.")
            
        except Exception as e:
            logger.error(f"LLaSA Hybrid: LLM generation failed: {e}")
            return
        
        # Extract speech tokens
        try:
            speech_tokens_as_strings_list = re.findall(r"(<\|s_\d+\|>)", generated_completion_str)
            logger.debug(f"LLaSA Hybrid: Found {len(speech_tokens_as_strings_list)} speech token matches.")
            
            if not speech_tokens_as_strings_list:
                logger.error("LLaSA Hybrid: No speech tokens found in LLM output. Generation may have failed.")
                return
            
            speech_integer_ids = _llasa_extract_speech_ids_from_str_list(speech_tokens_as_strings_list)
            if not speech_integer_ids:
                logger.error("LLaSA Hybrid: No valid integer speech IDs extracted from tokens.")
                return
            
            logger.info(f"LLaSA Hybrid: Extracted {len(speech_integer_ids)} speech IDs for XCodec2 decoding.")
            
        except Exception as e:
            logger.error(f"LLaSA Hybrid: Failed to extract speech tokens: {e}")
            return
        
        # Decode with XCodec2
        try:
            speech_tokens_tensor_pt = torch_llasa.tensor(speech_integer_ids, device=pt_device).unsqueeze(0).unsqueeze(0)
            
            logger.info("LLaSA Hybrid: Decoding speech with XCodec2...")
            with torch_llasa.no_grad(), SuppressOutput():
                gen_wav_pt = codec_model_pt_inst.decode_code(speech_tokens_tensor_pt)
            
            if gen_wav_pt is None:
                logger.error("LLaSA Hybrid: XCodec2 decoding returned None.")
                return
            
            audio_numpy = gen_wav_pt[0, 0, :].cpu().numpy()
            
            if audio_numpy.size == 0:
                logger.error("LLaSA Hybrid: XCodec2 decoding resulted in empty audio.")
                return
            
            # Validate audio quality
            if np.all(audio_numpy == 0):
                logger.warning("LLaSA Hybrid: Generated audio is all zeros (silence).")
            elif np.max(np.abs(audio_numpy)) < 0.001:
                logger.warning("LLaSA Hybrid: Generated audio has very low amplitude.")
            
            logger.info(f"LLaSA Hybrid: Synthesis successful - {audio_numpy.shape[0]} samples at {sample_rate}Hz ({audio_numpy.shape[0]/sample_rate:.2f}s)")
            
        except Exception as e:
            logger.error(f"LLaSA Hybrid: XCodec2 decoding failed: {e}")
            return
        
        # Save audio output
        if output_file_str:
            try:
                output_path_obj = Path(output_file_str).with_suffix(".wav")
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                sf_llasa.write(str(output_path_obj), audio_numpy, samplerate=sample_rate)
                logger.info(f"LLaSA Hybrid: Audio saved to {output_path_obj}")
                
                # Validate saved file
                if output_path_obj.stat().st_size < 1000:
                    logger.warning(f"LLaSA Hybrid: Saved audio file is very small ({output_path_obj.stat().st_size} bytes).")
                    
            except Exception as e_save:
                logger.error(f"LLaSA Hybrid: Failed to save audio to '{output_file_str}': {e_save}")
        
        # Play audio directly
        if play_direct:
            try:
                audio_int16 = (np.clip(audio_numpy, -1.0, 1.0) * 32767).astype(np.int16)
                play_audio(audio_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=sample_rate)
                logger.info("LLaSA Hybrid: Audio playback completed.")
            except Exception as e_play:
                logger.error(f"LLaSA Hybrid: Failed to play audio: {e_play}")
        
        logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Synthesis completed successfully.")
        
    except Exception as e:
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Unexpected synthesis error: {e}", exc_info=True)
    finally:
        # Cleanup resources
        logger.debug("LLaSA Hybrid: Cleaning up resources...")
        _cleanup_resources(
            llasa_llm_mlx_model, 
            llasa_llm_mlx_tokenizer, 
            chat_template_hf_tokenizer, 
            codec_model_pt_inst
        )

# --- Additional Utility Functions ---
def validate_llasa_installation() -> Tuple[bool, dict]:
    """
    Validate LLaSA installation and return status report.
    
    Returns:
        Tuple of (is_fully_functional, status_report)
    """
    status_report = {
        "mlx_lm": MLX_LM_AVAILABLE,
        "xcodec2": XCODEC2_AVAILABLE,
        "transformers": TRANSFORMERS_FOR_LLASA_AVAILABLE,
        "torch": TORCH_FOR_LLASA_AVAILABLE,
        "soundfile": SF_FOR_LLASA_AVAILABLE,
        "utils": UTILS_AVAILABLE,
        "fully_functional": False
    }
    
    # Check if all dependencies are available
    all_deps_available = all([
        MLX_LM_AVAILABLE,
        XCODEC2_AVAILABLE, 
        TRANSFORMERS_FOR_LLASA_AVAILABLE,
        TORCH_FOR_LLASA_AVAILABLE,
        SF_FOR_LLASA_AVAILABLE,
        UTILS_AVAILABLE
    ])
    
    # Check if all objects are properly imported
    objects_valid, _ = _validate_model_objects()
    
    status_report["fully_functional"] = all_deps_available and objects_valid
    
    return status_report["fully_functional"], status_report

def test_llasa_basic_functionality() -> bool:
    """
    Test basic LLaSA functionality without full model loading.
    
    Returns:
        True if basic functionality works, False otherwise
    """
    try:
        # Test basic imports and object creation
        if not MLX_LM_AVAILABLE or not torch_llasa:
            return False
        
        # Test device creation
        device_str, device = _get_pytorch_device()
        if device is None:
            return False
        
        # Test basic tensor operations
        test_tensor = torch_llasa.tensor([1, 2, 3], device=device)
        if test_tensor.sum().item() != 6:
            return False
        
        # Test speech token conversion
        test_ids = [100, 200, 300]
        test_tokens = _llasa_ids_to_speech_tokens_str(test_ids)
        recovered_ids = _llasa_extract_speech_ids_from_str_list(test_tokens)
        if recovered_ids != test_ids:
            return False
        
        logger.info("LLaSA Handler: Basic functionality test passed.")
        return True
        
    except Exception as e:
        logger.error(f"LLaSA Handler: Basic functionality test failed: {e}")
        return False

def get_llasa_system_info() -> dict:
    """
    Get system information relevant to LLaSA operation.
    
    Returns:
        Dictionary with system information
    """
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "dependencies": {}
    }
    
    # Check PyTorch info
    if torch_llasa:
        info["torch_version"] = torch_llasa.__version__
        info["cuda_available"] = torch_llasa.cuda.is_available()
        info["mps_available"] = (
            hasattr(torch_llasa.backends, "mps") and 
            torch_llasa.backends.mps.is_available()
        )
        if torch_llasa.cuda.is_available():
            info["cuda_device_count"] = torch_llasa.cuda.device_count()
            info["cuda_device_name"] = torch_llasa.cuda.get_device_name(0) if torch_llasa.cuda.device_count() > 0 else "Unknown"
    
    # Check MLX info  
    if mlx_core:
        try:
            info["mlx_available"] = True
            # Try to get MLX device info if available
            if hasattr(mlx_core, 'default_device'):
                info["mlx_default_device"] = str(mlx_core.default_device())
        except Exception:
            info["mlx_available"] = True  # Available but can't get details
    
    # Check other dependencies
    if sf_llasa:
        info["soundfile_version"] = getattr(sf_llasa, '__version__', 'unknown')
    
    if AutoTokenizer_llasa_chat:
        try:
            import transformers
            info["transformers_version"] = transformers.__version__
        except (ImportError, AttributeError):
            info["transformers_version"] = "unknown"
    
    return info

def create_llasa_config_template() -> dict:
    """
    Create a template configuration for LLaSA models.
    
    Returns:
        Template configuration dictionary
    """
    template = {
        "handler_function_key": "llasa_hybrid",
        "llm_model_id": "nhe-ai/Llasa-1B-Multilingual-mlx-4Bit",
        "chat_tokenizer_id": "HKUSTAudio/Llasa-1B-Multilingual",
        "codec_model_id": "HKUSTAudio/xcodec2",
        "language": "de",
        "default_voice_id": "./german.wav",
        "available_voices": ["./german.wav"],
        "sample_rate": 16000,
        "requires_hf_token": False,
        "notes": "LLaSA 1B Hybrid (MLX LLM + PyTorch XCodec2). Uses ref WAV for cloning. 16kHz output. Requires Apple Silicon for optimal MLX performance."
    }
    return template

# --- Module Initialization Log ---
def _log_initialization_status():
    """Log the initialization status of the LLaSA handler."""
    is_functional, status = validate_llasa_installation()
    
    logger.info("=" * 60)
    logger.info("LLaSA Hybrid Handler Initialization Status")
    logger.info("=" * 60)
    logger.info(f"Overall Status: {'✓ FUNCTIONAL' if is_functional else '✗ NON-FUNCTIONAL'}")
    logger.info("-" * 60)
    
    for component, available in status.items():
        if component != "fully_functional":
            status_symbol = "✓" if available else "✗"
            logger.info(f"{component:15s}: {status_symbol} {'Available' if available else 'Missing'}")
    
    logger.info("-" * 60)
    
    if is_functional:
        logger.info("LLaSA Handler is ready for use.")
        # Run basic functionality test
        basic_test_passed = test_llasa_basic_functionality()
        if basic_test_passed:
            logger.info("✓ Basic functionality test passed.")
        else:
            logger.warning("✗ Basic functionality test failed.")
    else:
        missing_deps = [k for k, v in status.items() if not v and k != "fully_functional"]
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.warning("Install missing dependencies to enable LLaSA functionality.")
    
    logger.info("=" * 60)

# Run initialization logging when module is imported
_log_initialization_status()

# --- Export marker for debugging ---
print(f"--- LLaSA Hybrid Handler Module Loaded Successfully (llasa_hybrid_handler.py) ---")