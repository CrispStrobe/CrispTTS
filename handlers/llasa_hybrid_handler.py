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
from typing import Optional, List, Tuple, Union, Callable

# --- Conditional Imports with Robust Error Handling ---
MLX_LM_AVAILABLE = False
XCODEC2_AVAILABLE = False
TRANSFORMERS_FOR_LLASA_AVAILABLE = False
TORCH_FOR_LLASA_AVAILABLE = False
SF_FOR_LLASA_AVAILABLE = False
SCIPY_AVAILABLE_FOR_LLASA = False
UTILS_AVAILABLE = False # For CrispTTS utils
PYDUB_LLASA_AVAILABLE = False
TORCHAUDIO_AVAILABLE_FOR_LLASA = False

mlx_lm_load, mlx_lm_generate_str_func = None, None
make_sampler_func = None
mlx_core = None
XCodec2Model_llasa = None
AutoTokenizer_llasa_chat = None
torch_llasa = None
torchaudio_llasa_transforms = None
torchaudio = None # For direct torchaudio import
sf_llasa = None
scipy_resample_func = None
save_audio_util, play_audio_util, SuppressOutput_util = None, None, None
AudioSegment_llasa_pydub = None

logger_init = logging.getLogger("CrispTTS.handlers.llasa_hybrid.init")
logger = logging.getLogger("CrispTTS.handlers.llasa_hybrid")

try:
    import mlx.core as mx_imported
    from mlx_lm import load as mlx_lm_load_imported
    from mlx_lm.generate import generate as mlx_lm_generate_str_imported
    from mlx_lm.sample_utils import make_sampler as make_sampler_imported

    mlx_core = mx_imported
    mlx_lm_load = mlx_lm_load_imported
    mlx_lm_generate_str_func = mlx_lm_generate_str_imported
    make_sampler_func = make_sampler_imported
    MLX_LM_AVAILABLE = True
    logger_init.info("MLX, mlx-lm (load, generate), and make_sampler imported successfully for LLaSA Handler.")
except ImportError as e:
    logger_init.warning(f"MLX/mlx-lm or critical components not found. LLaSA Hybrid handler will be non-functional. Error: {e}")
except Exception as e:
    logger_init.error(f"Unexpected MLX/mlx-lm import error: {e}", exc_info=True)

try:
    from xcodec2.modeling_xcodec2 import XCodec2Model as XCodec2Model_imported
    XCodec2Model_llasa = XCodec2Model_imported
    XCODEC2_AVAILABLE = True
    logger_init.info("XCodec2 imported successfully for LLaSA Handler.")
except ImportError as e:
    logger_init.warning(f"XCodec2 library not found. LLaSA Hybrid handler will be non-functional. Error: {e}")
except Exception as e:
    logger_init.error(f"Unexpected XCodec2 import error: {e}", exc_info=True)

try:
    from transformers import AutoTokenizer as AutoTokenizer_imported
    AutoTokenizer_llasa_chat = AutoTokenizer_imported
    TRANSFORMERS_FOR_LLASA_AVAILABLE = True
    logger_init.info("Transformers (for LLaSA Chat Tokenizer) imported successfully.")
except ImportError as e:
    logger_init.warning(f"Transformers (for LLaSA Chat Tokenizer) not found. LLaSA Hybrid handler may fail. Error: {e}")
except Exception as e:
    logger_init.error(f"Unexpected Transformers import error: {e}", exc_info=True)

try:
    import torch as torch_imported
    import torchaudio as torchaudio_imported_module 
    torch_llasa = torch_imported
    torchaudio = torchaudio_imported_module
    if hasattr(torchaudio_imported_module, 'transforms'):
        torchaudio_llasa_transforms = torchaudio_imported_module.transforms
        TORCHAUDIO_AVAILABLE_FOR_LLASA = True
    else:
        logger_init.warning("torchaudio.transforms not found, torchaudio resampling fallback unavailable.")
        TORCHAUDIO_AVAILABLE_FOR_LLASA = False
    TORCH_FOR_LLASA_AVAILABLE = True
    logger_init.info(f"PyTorch imported. Torchaudio for resampling: {'Available' if TORCHAUDIO_AVAILABLE_FOR_LLASA else 'Unavailable'}.")
except ImportError:
    logger_init.warning("PyTorch or torchaudio not found. LLaSA XCodec2 or resampling fallback will be non-functional.")
    if 'torch_llasa' in locals() and torch_llasa is None: TORCH_FOR_LLASA_AVAILABLE = False # type: ignore
    TORCHAUDIO_AVAILABLE_FOR_LLASA = False
except Exception as e:
    logger_init.error(f"Unexpected PyTorch/torchaudio import error: {e}", exc_info=True)
    TORCH_FOR_LLASA_AVAILABLE = False
    TORCHAUDIO_AVAILABLE_FOR_LLASA = False

try:
    import soundfile as sf_imported
    sf_llasa = sf_imported
    SF_FOR_LLASA_AVAILABLE = True
    logger_init.info("SoundFile imported successfully for LLaSA Handler.")
except ImportError as e:
    logger_init.warning(f"SoundFile not found. LLaSA Hybrid handler cannot save/load audio. Error: {e}")
except Exception as e:
    logger_init.error(f"Unexpected SoundFile import error: {e}", exc_info=True)

try:
    from scipy.signal import resample as scipy_resample_imported
    scipy_resample_func = scipy_resample_imported
    SCIPY_AVAILABLE_FOR_LLASA = True
    logger_init.info("Scipy (for resampling) imported successfully for LLaSA Handler.")
except ImportError:
    logger_init.info("Scipy not found. Reference audio resampling will use torchaudio or basic methods if needed.")

try:
    from utils import save_audio, play_audio, SuppressOutput
    save_audio_util = save_audio
    play_audio_util = play_audio
    SuppressOutput_util = SuppressOutput
    UTILS_AVAILABLE = True
    logger_init.info("Project utils (save_audio, play_audio, SuppressOutput) imported successfully for LLaSA Handler.")
except ImportError as e:
    logger_init.error(f"Failed to import project utils: {e}. LLaSA handler save/play functionality will be affected.")
    def _placeholder_suppress_output(*args, **kwargs): # type: ignore
        class DummyContextManager:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): return False
        return DummyContextManager()
    SuppressOutput_util = _placeholder_suppress_output # type: ignore

try:
    from pydub import AudioSegment as AudioSegment_llasa_pydub_imp
    AudioSegment_llasa_pydub = AudioSegment_llasa_pydub_imp
    PYDUB_LLASA_AVAILABLE = True
    logger_init.info("pydub imported successfully for LLaSA audio trimming.")
except ImportError:
    logger_init.info("pydub not available for LLaSA handler. Reference audio trimming will use numpy slicing if pydub is selected but unavailable.")


def _validate_and_prepare_reference_audio(
    ref_wav_path_str: str,
    project_root: Path,
    target_sr: int = 16000,
    max_duration_s: int = 15
) -> Optional[Tuple[str, np.ndarray, int, Optional[Path]]]:
    logger.debug(f"LLaSA Prep Ref Audio: Input path: '{ref_wav_path_str}', Target SR: {target_sr}, Max Duration: {max_duration_s}s")
    if not SF_FOR_LLASA_AVAILABLE or not sf_llasa:
        logger.error("LLaSA Prep Ref Audio: SoundFile library (sf_llasa) is not available. Cannot process reference audio.")
        return None
    if not ref_wav_path_str or not isinstance(ref_wav_path_str, str):
        logger.error(f"LLaSA Prep Ref Audio: Invalid reference WAV path string provided: {ref_wav_path_str}")
        return None

    ref_wav_path_actual = Path(ref_wav_path_str)
    if not ref_wav_path_actual.is_absolute():
        resolved_path = (project_root / ref_wav_path_actual).resolve()
        logger.debug(f"LLaSA Prep Ref Audio: Relative path '{ref_wav_path_str}' resolved to '{resolved_path}' against project root '{project_root}'.")
        ref_wav_path_actual = resolved_path

    if not ref_wav_path_actual.exists() or not ref_wav_path_actual.is_file():
        logger.error(f"LLaSA Prep Ref Audio: Reference WAV file not found or is not a file: {ref_wav_path_actual}")
        return None

    temp_trimmed_file_to_delete: Optional[Path] = None

    try:
        logger.debug(f"LLaSA Prep Ref Audio: Loading audio from: {ref_wav_path_actual}")
        prompt_wav_samples, sr_prompt = sf_llasa.read(str(ref_wav_path_actual), dtype='float32', always_2d=False)
        logger.debug(f"LLaSA Prep Ref Audio: Loaded audio shape: {prompt_wav_samples.shape}, Original SR: {sr_prompt}Hz")

        if prompt_wav_samples.size == 0:
            logger.error(f"LLaSA Prep Ref Audio: Loaded audio from '{ref_wav_path_actual}' is empty.")
            return None
        if prompt_wav_samples.ndim > 1:
            prompt_wav_samples = np.mean(prompt_wav_samples, axis=1)
            logger.info(f"LLaSA Prep Ref Audio: Converted multi-channel audio from '{ref_wav_path_actual}' to mono. New shape: {prompt_wav_samples.shape}")

        current_duration_s = len(prompt_wav_samples) / sr_prompt
        logger.debug(f"LLaSA Prep Ref Audio: Duration after mono conversion (if any): {current_duration_s:.2f}s at {sr_prompt}Hz.")

        if current_duration_s > max_duration_s:
            logger.info(f"LLaSA Prep Ref Audio: Reference audio ({current_duration_s:.1f}s) > {max_duration_s}s. Attempting to trim.")
            if PYDUB_LLASA_AVAILABLE and AudioSegment_llasa_pydub:
                try:
                    audio_segment = AudioSegment_llasa_pydub.from_file(str(ref_wav_path_actual))
                    trimmed_segment = audio_segment[:max_duration_s * 1000]
                    with tempfile.NamedTemporaryFile(suffix=ref_wav_path_actual.suffix or ".wav", prefix="llasa_trimmed_pydub_", delete=False) as tmp_f:
                        temp_trimmed_file_to_delete = Path(tmp_f.name)
                    trimmed_segment.export(str(temp_trimmed_file_to_delete), format=ref_wav_path_actual.suffix.lstrip('.') or 'wav')
                    prompt_wav_samples, sr_prompt = sf_llasa.read(str(temp_trimmed_file_to_delete), dtype='float32', always_2d=False)
                    if prompt_wav_samples.ndim > 1: prompt_wav_samples = np.mean(prompt_wav_samples, axis=1)
                    current_duration_s = len(prompt_wav_samples) / sr_prompt
                    logger.info(f"LLaSA Prep Ref Audio: Successfully trimmed with pydub to {current_duration_s:.1f}s. Temp file: {temp_trimmed_file_to_delete}")
                except Exception as e_pydub_trim:
                    logger.warning(f"LLaSA Prep Ref Audio: pydub trimming failed ('{e_pydub_trim}'). Falling back to numpy slice.", exc_info=logger.isEnabledFor(logging.DEBUG))
                    if temp_trimmed_file_to_delete and temp_trimmed_file_to_delete.exists(): temp_trimmed_file_to_delete.unlink(missing_ok=True)
                    temp_trimmed_file_to_delete = None
                    prompt_wav_samples, sr_prompt = sf_llasa.read(str(ref_wav_path_actual), dtype='float32', always_2d=False)
                    if prompt_wav_samples.ndim > 1: prompt_wav_samples = np.mean(prompt_wav_samples, axis=1)
                    current_duration_s = len(prompt_wav_samples) / sr_prompt
                    num_samples_to_keep = int(max_duration_s * sr_prompt)
                    if len(prompt_wav_samples) > num_samples_to_keep:
                        prompt_wav_samples = prompt_wav_samples[:num_samples_to_keep]
                        current_duration_s = len(prompt_wav_samples) / sr_prompt
                        logger.info(f"LLaSA Prep Ref Audio: Trimmed using numpy slice to {current_duration_s:.1f}s (after pydub fail).")
            else:
                logger.info("LLaSA Prep Ref Audio: pydub not available, using numpy slice for trimming.")
                num_samples_to_keep = int(max_duration_s * sr_prompt)
                if len(prompt_wav_samples) > num_samples_to_keep:
                    prompt_wav_samples = prompt_wav_samples[:num_samples_to_keep]
                    current_duration_s = len(prompt_wav_samples) / sr_prompt
                    logger.info(f"LLaSA Prep Ref Audio: Trimmed using numpy slice to {current_duration_s:.1f}s.")
        else:
            logger.debug(f"LLaSA Prep Ref Audio: No trimming needed (duration {current_duration_s:.1f}s <= {max_duration_s}s).")

        if sr_prompt != target_sr:
            logger.info(f"LLaSA Prep Ref Audio: Resampling audio from {sr_prompt}Hz to {target_sr}Hz.")
            if SCIPY_AVAILABLE_FOR_LLASA and scipy_resample_func:
                num_target_samples = int(len(prompt_wav_samples) * (target_sr / float(sr_prompt)))
                if num_target_samples > 0:
                    prompt_wav_samples = scipy_resample_func(prompt_wav_samples, num_target_samples)
                    sr_prompt = target_sr
                    logger.debug(f"LLaSA Prep Ref Audio: Resampled with Scipy. New shape: {prompt_wav_samples.shape}, new SR: {sr_prompt}Hz.")
                else: logger.warning(f"LLaSA Prep Ref Audio: Scipy resampling would result in 0 samples. Using original SR {sr_prompt}Hz.")
            elif TORCHAUDIO_AVAILABLE_FOR_LLASA and torchaudio_llasa_transforms and torch_llasa:
                logger.info("LLaSA Prep Ref Audio: Scipy unavailable, trying torchaudio for resampling.")
                try:
                    resampler = torchaudio_llasa_transforms.Resample(orig_freq=sr_prompt, new_freq=target_sr)
                    tensor_audio = torch_llasa.from_numpy(prompt_wav_samples.copy()).float()
                    if tensor_audio.ndim == 1: tensor_audio = tensor_audio.unsqueeze(0)
                    resampled_tensor = resampler(tensor_audio)
                    prompt_wav_samples = resampled_tensor.squeeze(0).numpy()
                    sr_prompt = target_sr
                    logger.info(f"LLaSA Prep Ref Audio: Resampled successfully with torchaudio.")
                except Exception as e_torchaudio_resample:
                    logger.error(f"LLaSA Prep Ref Audio: torchaudio resampling failed: {e_torchaudio_resample}. Using original SR {sr_prompt}Hz.", exc_info=True)
            else:
                logger.warning(f"LLaSA Prep Ref Audio: Neither Scipy nor torchaudio available for resampling. Using original SR {sr_prompt}Hz. XCodec2 expects {target_sr}Hz; quality may be impacted.")
        else:
            logger.debug(f"LLaSA Prep Ref Audio: Audio already at target SR {target_sr}Hz.")

        final_duration_s = len(prompt_wav_samples) / sr_prompt
        logger.info(f"LLaSA Prep Ref Audio: Final processed reference audio: {final_duration_s:.2f}s at {sr_prompt}Hz.")
        if final_duration_s < 0.5: logger.warning(f"LLaSA Prep Ref Audio: Final reference audio is very short ({final_duration_s:.1f}s). Cloning quality may be poor.")

        return str(ref_wav_path_actual), prompt_wav_samples, sr_prompt, temp_trimmed_file_to_delete

    except Exception as e_audio_proc:
        logger.error(f"LLaSA Prep Ref Audio: Critical error during loading/processing of reference audio '{ref_wav_path_actual}': {e_audio_proc}", exc_info=True)
        if temp_trimmed_file_to_delete and temp_trimmed_file_to_delete.exists():
            try: temp_trimmed_file_to_delete.unlink(missing_ok=True)
            except OSError: pass
        return None

def _validate_dependencies() -> Tuple[bool, List[str]]:
    missing_deps = []
    if not (MLX_LM_AVAILABLE and mlx_lm_load and mlx_lm_generate_str_func and make_sampler_func):
        missing_deps.append("MLX/mlx-lm (load, generate, make_sampler)")
    if not XCODEC2_AVAILABLE: missing_deps.append("XCodec2")
    if not TRANSFORMERS_FOR_LLASA_AVAILABLE: missing_deps.append("Transformers (for LLaSA chat)")
    if not TORCH_FOR_LLASA_AVAILABLE: missing_deps.append("PyTorch (for LLaSA XCodec2)")
    if not SF_FOR_LLASA_AVAILABLE: missing_deps.append("SoundFile")
    if not UTILS_AVAILABLE: missing_deps.append("CrispTTS Project Utils")
    return len(missing_deps) == 0, missing_deps

def _validate_model_objects() -> Tuple[bool, List[str]]:
    missing_objects = []
    if not mlx_lm_load: missing_objects.append("mlx_lm.load function")
    if not mlx_lm_generate_str_func: missing_objects.append("mlx_lm.generate.generate function")
    if not make_sampler_func: missing_objects.append("mlx_lm.sample_utils.make_sampler function")
    if not XCodec2Model_llasa: missing_objects.append("XCodec2Model class")
    if not AutoTokenizer_llasa_chat: missing_objects.append("AutoTokenizer class (for LLaSA chat)")
    if not torch_llasa: missing_objects.append("torch module (for LLaSA)")
    if not sf_llasa: missing_objects.append("soundfile module (for LLaSA)")
    return len(missing_objects) == 0, missing_objects

def _llasa_ids_to_speech_tokens_str(speech_ids: List[int]) -> List[str]:
    if not speech_ids: return []
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids if isinstance(speech_id, int)]

def _llasa_extract_speech_ids_from_str_list(speech_tokens_text: str) -> List[int]:
    speech_ids = []
    if not isinstance(speech_tokens_text, str):
        logger.warning(f"LLaSA Extract Speech IDs: Expected string input, got {type(speech_tokens_text)}")
        return []
    matches = re.findall(r"<\|s_(\d+)\|>", speech_tokens_text)
    for num_str in matches:
        try: speech_ids.append(int(num_str))
        except ValueError: logger.warning(f"LLaSA Extract Speech IDs: Could not parse int from '{num_str}' in '{speech_tokens_text[:100]}...'")
    logger.debug(f"LLaSA Extract Speech IDs: Input (first 100): '{speech_tokens_text[:100]}...', Extracted IDs: {len(speech_ids)} (e.g., {speech_ids[:10]})")
    return speech_ids

def _get_pytorch_device() -> Tuple[str, Optional['torch_llasa.device']]: # type: ignore
    if not torch_llasa: logger.error("LLaSA Get PyTorch Device: PyTorch (torch_llasa) not available."); return "cpu", None
    device_str = "cpu"
    if torch_llasa.cuda.is_available(): device_str = "cuda"
    elif hasattr(torch_llasa.backends, "mps") and torch_llasa.backends.mps.is_available(): device_str = "mps"
    logger.info(f"LLaSA: Determined PyTorch device: {device_str} for XCodec2.")
    try: return device_str, torch_llasa.device(device_str)
    except Exception as e: logger.warning(f"LLaSA: Failed to create torch.device('{device_str}'): {e}. Falling back to CPU."); return "cpu", torch_llasa.device("cpu")

def _cleanup_resources(*resources_to_del: object) -> None:
    logger.debug(f"LLaSA Cleanup: Attempting to delete {len(resources_to_del)} resource(s).")
    for idx, resource in enumerate(resources_to_del):
        if resource is not None:
            res_name = f"resource_{idx}"
            logger.debug(f"LLaSA Cleanup: Deleting {res_name} (type: {type(resource)})...")
            try: del resource
            except Exception as e: logger.debug(f"LLaSA Cleanup: Error during deletion of {res_name}: {e}")
    if TORCH_FOR_LLASA_AVAILABLE and torch_llasa:
        try:
            if torch_llasa.cuda.is_available(): torch_llasa.cuda.empty_cache(); logger.debug("LLaSA Cleanup: PyTorch CUDA cache cleared.")
            if TORCHAUDIO_AVAILABLE_FOR_LLASA and hasattr(torch_llasa.backends, "mps") and torch_llasa.backends.mps.is_available() and hasattr(torch_llasa.mps, "empty_cache"):
                torch_llasa.mps.empty_cache(); logger.debug("LLaSA Cleanup: PyTorch MPS cache cleared.")
        except Exception as e: logger.debug(f"LLaSA Cleanup: Error during PyTorch cache cleanup: {e}")
    if MLX_LM_AVAILABLE and mlx_core and hasattr(mlx_core, 'clear_cache'):
        try: mlx_core.clear_cache(); logger.debug("LLaSA Cleanup: MLX cache cleared.")
        except Exception as e: logger.debug(f"LLaSA Cleanup: Error during MLX cache cleanup: {e}")
    gc.collect()
    logger.debug("LLaSA Cleanup: Garbage collection called.")

def synthesize_with_llasa_hybrid(
    model_config: dict,
    text_to_synthesize: str,
    voice_id_override: Optional[str],
    model_params_override: Optional[str],
    output_file_str: Optional[str],
    play_direct: bool,
    attempt_no_cloning_fallback: bool = True,
    _is_fallback_attempt: bool = False
) -> None:
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'llasa_hybrid_unknown')
    log_prefix = f"LLaSA Hybrid ({crisptts_model_id_for_log}{' Fallback NoClone' if _is_fallback_attempt else ''}): "

    logger.info(f"{log_prefix}Starting synthesis process.")
    logger.debug(f"{log_prefix}Input text (first 100): '{text_to_synthesize[:100]}...'")
    logger.debug(f"{log_prefix}Voice ID/Path: '{voice_id_override}'")
    logger.debug(f"{log_prefix}Model params: '{model_params_override}'")

    deps_valid, missing_deps = _validate_dependencies()
    if not deps_valid: logger.error(f"{log_prefix}Missing dependencies: {', '.join(missing_deps)}. Skip."); return
    objects_valid, missing_objects = _validate_model_objects()
    if not objects_valid: logger.error(f"{log_prefix}Missing Python objects: {', '.join(missing_objects)}. Skip."); return
    if not text_to_synthesize or not text_to_synthesize.strip(): logger.error(f"{log_prefix}Invalid/empty text. Skip."); return

    llm_model_id_cfg = model_config.get("llm_model_id")
    chat_tokenizer_id_cfg = model_config.get("chat_tokenizer_id")
    codec_model_id_cfg = model_config.get("codec_model_id")
    ref_speaker_wav_path_str = None if _is_fallback_attempt else (voice_id_override or model_config.get("default_voice_id"))
    target_sample_rate = model_config.get("sample_rate", 16000)
    MIN_SPEECH_TOKENS_THRESHOLD = model_config.get("min_speech_tokens_threshold", 75)

    logger.debug(f"{log_prefix}Config: LLM='{llm_model_id_cfg}', ChatTok='{chat_tokenizer_id_cfg}', Codec='{codec_model_id_cfg}', RefWAV='{ref_speaker_wav_path_str}', TargetSR={target_sample_rate}Hz")
    if not all([llm_model_id_cfg, chat_tokenizer_id_cfg, codec_model_id_cfg]): logger.error(f"{log_prefix}Missing critical model IDs in config. Skip."); return

    _, pt_device_obj = _get_pytorch_device()
    if pt_device_obj is None: logger.error(f"{log_prefix}Could not init PyTorch device. Skip."); return

    llasa_llm_mlx_model, llasa_llm_mlx_tokenizer, chat_template_hf_tokenizer, codec_model_pt_inst = None, None, None, None
    temp_trimmed_ref_audio_file_to_delete: Optional[Path] = None

    try:
        logger.info(f"{log_prefix}Loading models/tokenizers...")
        suppress_ext_logs = not logger.isEnabledFor(logging.DEBUG)
        with SuppressOutput_util(suppress_stdout=suppress_ext_logs, suppress_stderr=suppress_ext_logs): # type: ignore
            try:
                llasa_llm_mlx_model, llasa_llm_mlx_tokenizer = mlx_lm_load(llm_model_id_cfg) # type: ignore
                if not (llasa_llm_mlx_model and llasa_llm_mlx_tokenizer): raise ValueError("MLX model/tokenizer is None")
            except Exception as e: logger.error(f"{log_prefix}MLX LLM load failed: {e}", exc_info=True); return
            try:
                chat_template_hf_tokenizer = AutoTokenizer_llasa_chat.from_pretrained(chat_tokenizer_id_cfg) # type: ignore
                if not chat_template_hf_tokenizer: raise ValueError("Chat tokenizer is None")
            except Exception as e: logger.error(f"{log_prefix}Chat Tokenizer load failed: {e}", exc_info=True); return
            try:
                codec_model_pt_inst = XCodec2Model_llasa.from_pretrained(codec_model_id_cfg).to(pt_device_obj).eval() # type: ignore
                if not codec_model_pt_inst: raise ValueError("XCodec2 model is None")
            except Exception as e: logger.error(f"{log_prefix}XCodec2 load failed: {e}", exc_info=True); return
        logger.info(f"{log_prefix}All models/tokenizers loaded.")

        assistant_content_prefix_str = "<|SPEECH_GENERATION_START|>"
        if ref_speaker_wav_path_str:
            logger.debug(f"{log_prefix}Processing ref WAV for cloning: '{ref_speaker_wav_path_str}'")
            project_root = Path(__file__).resolve().parent.parent
            ref_audio_data = _validate_and_prepare_reference_audio(ref_speaker_wav_path_str, project_root, target_sample_rate, 15)
            if ref_audio_data:
                _, samples, sr, temp_trimmed_ref_audio_file_to_delete = ref_audio_data
                if sr == target_sample_rate:
                    try:
                        logger.info(f"{log_prefix}Encoding ref WAV ({len(samples)/sr:.1f}s) with XCodec2...")
                        vq_codes = codec_model_pt_inst.encode_code(torch_llasa.from_numpy(samples).float().unsqueeze(0).to(pt_device_obj))[0,0,:].tolist() # type: ignore
                        assistant_content_prefix_str += "".join(_llasa_ids_to_speech_tokens_str(vq_codes))
                        logger.info(f"{log_prefix}Generated {len(vq_codes)} speech prefix tokens for cloning.")
                    except Exception as e: logger.error(f"{log_prefix}Failed to encode ref audio: {e}", exc_info=True)
                else: logger.warning(f"{log_prefix}Ref audio SR ({sr}Hz) != target ({target_sample_rate}Hz). Skipping cloning.")
            else: logger.warning(f"{log_prefix}Ref audio processing failed. No cloning prefix.")
        else: logger.info(f"{log_prefix}No ref WAV or fallback mode. Using basic assistant prefix.")

        user_content = f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text_to_synthesize}<|TEXT_UNDERSTANDING_END|>"
        chat = [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content_prefix_str}]
        try:
            prompt_str = chat_template_hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) # type: ignore
            if not prompt_str: raise ValueError("Empty prompt from chat template.")
        except Exception as e: logger.error(f"{log_prefix}Chat template failed: {e}", exc_info=True); return
        logger.debug(f"{log_prefix}Full prompt to LLM (first 300): '{prompt_str[:300]}...'")

        num_prompt_toks = len(llasa_llm_mlx_tokenizer.encode(prompt_str)) # type: ignore
        max_ctx_cfg = getattr(llasa_llm_mlx_model, 'config', {}) # type: ignore
        max_ctx = getattr(max_ctx_cfg, 'max_position_embeddings', 2048) # type: ignore

        desired_max_speech_tokens = 1200 
        max_new = max(75, min(max_ctx - num_prompt_toks - 20, desired_max_speech_tokens)) 
        if max_new <= 0: logger.error(f"{log_prefix}Max new tokens {max_new} too low. Prompt: {num_prompt_toks} for ctx {max_ctx}. Skip."); return

        temp_val, top_p_val, min_p_val, top_k_val = 0.6, 0.9, 0.05, 40 # Adjusted defaults: temp slightly higher to avoid repetition
        if model_params_override:
            try:
                p = json.loads(model_params_override)
                temp_val = float(p.get("temperature", temp_val))
                top_p_val = float(p.get("top_p", top_p_val))
                min_p_val = float(p.get("min_p", min_p_val))
                top_k_val = int(p.get("top_k", top_k_val))
            except: logger.warning(f"{log_prefix}Could not parse model_params for sampling. Using defaults.")
        
        sampler = make_sampler_func(temp=temp_val, top_p=top_p_val, min_p=min_p_val, top_k=top_k_val) # type: ignore
        logger.info(f"{log_prefix}LLM Gen: max_new_tokens={max_new}. Sampler: temp={temp_val}, top_p={top_p_val}, min_p={min_p_val}, top_k={top_k_val}.")

        llm_output = mlx_lm_generate_str_func( # type: ignore
            model=llasa_llm_mlx_model,
            tokenizer=llasa_llm_mlx_tokenizer,
            prompt=prompt_str,
            max_tokens=max_new, # This should be the kwarg for mlx_lm.generate.generate
            sampler=sampler,
            verbose=logger.isEnabledFor(logging.DEBUG)
        )
        if not llm_output: logger.error(f"{log_prefix}LLM returned empty. Skip."); return
        logger.debug(f"{log_prefix}Raw LLM output length {len(llm_output)}. Starts: '{llm_output[:100]}...', Ends: '...{llm_output[-100:]}'")
        
        completion = llm_output
        if llm_output.startswith(prompt_str):
            completion = llm_output[len(prompt_str):]
            logger.debug(f"{log_prefix}Stripped echoed prompt. Completion length: {len(completion)}.")
        else:
            logger.debug(f"{log_prefix}LLM output did not start with prompt. Assuming full output is completion.")
        
        eos_str = '<|SPEECH_GENERATION_END|>'
        if eos_str in completion:
            completion = completion.split(eos_str, 1)[0]
            logger.debug(f"{log_prefix}Truncated completion at '{eos_str}'. Final part length: {len(completion)}.")
        else:
            logger.debug(f"{log_prefix}Custom EOS '{eos_str}' not in completion. Using full completion (may lead to run-on audio if LLM didn't stop).")
        if not completion.strip(): logger.error(f"{log_prefix}Completion empty after processing. Skip."); return

        speech_ids = _llasa_extract_speech_ids_from_str_list(completion)
        logger.info(f"{log_prefix}Extracted {len(speech_ids)} speech IDs.")

        if len(speech_ids) < MIN_SPEECH_TOKENS_THRESHOLD:
            logger.warning(f"{log_prefix}Extracted only {len(speech_ids)} speech IDs (< threshold {MIN_SPEECH_TOKENS_THRESHOLD}).")
            if attempt_no_cloning_fallback and ref_speaker_wav_path_str and not _is_fallback_attempt:
                logger.info(f"{log_prefix}Output too short with voice cloning. RETRYING WITHOUT CLONING.")
                if temp_trimmed_ref_audio_file_to_delete and temp_trimmed_ref_audio_file_to_delete.exists():
                    try: temp_trimmed_ref_audio_file_to_delete.unlink()
                    except OSError: pass
                _cleanup_resources(llasa_llm_mlx_model,llasa_llm_mlx_tokenizer,chat_template_hf_tokenizer,codec_model_pt_inst)
                return synthesize_with_llasa_hybrid(model_config, text_to_synthesize, None, model_params_override, output_file_str, play_direct, False, True)
            else: logger.error(f"{log_prefix}Output too short. No further fallbacks. Skip audio decode."); return

        speech_tensor = torch_llasa.tensor(speech_ids, device=pt_device_obj).unsqueeze(0).unsqueeze(0) # type: ignore
        logger.info(f"{log_prefix}Decoding {len(speech_ids)} speech IDs with XCodec2...")
        with torch_llasa.no_grad(), SuppressOutput_util(suppress_stdout=suppress_ext_logs, suppress_stderr=suppress_ext_logs): # type: ignore
            gen_wav_pt = codec_model_pt_inst.decode_code(speech_tensor) # type: ignore
        if gen_wav_pt is None: logger.error(f"{log_prefix}XCodec2 decode returned None. Skip."); return
        audio_np = gen_wav_pt[0,0,:].cpu().numpy()
        if audio_np.size == 0: logger.error(f"{log_prefix}XCodec2 decode empty. Skip."); return
        
        dur = audio_np.shape[0]/target_sample_rate
        logger.info(f"{log_prefix}XCodec2 decode OK. Audio: {dur:.2f}s at {target_sample_rate}Hz.")

        if output_file_str and UTILS_AVAILABLE and save_audio_util:
            try:
                out_path = Path(output_file_str).with_suffix(".wav")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                s16le_bytes = (np.clip(audio_np.astype(np.float32),-1.0,1.0)*32767).astype(np.int16).tobytes()
                save_audio_util(s16le_bytes, str(out_path), False, "pcm_s16le", target_sample_rate)
                logger.info(f"{log_prefix}Audio saved to {out_path}")
            except Exception as e: logger.error(f"{log_prefix}Save audio failed: {e}", exc_info=True)
        if play_direct and UTILS_AVAILABLE and play_audio_util:
            try:
                s16le_bytes = (np.clip(audio_np.astype(np.float32),-1.0,1.0)*32767).astype(np.int16).tobytes()
                play_audio_util(s16le_bytes, False, "pcm_s16le", target_sample_rate)
            except Exception as e: logger.error(f"{log_prefix}Play audio failed: {e}", exc_info=True)
        logger.info(f"{log_prefix}Synthesis completed{' (fallback)' if _is_fallback_attempt else ''}.")

    except Exception as e: logger.error(f"{log_prefix}Unexpected error in main block: {e}", exc_info=True)
    finally:
        logger.info(f"{log_prefix}Entering finally: cleanup.")
        _cleanup_resources(llasa_llm_mlx_model,llasa_llm_mlx_tokenizer,chat_template_hf_tokenizer,codec_model_pt_inst)
        if temp_trimmed_ref_audio_file_to_delete and temp_trimmed_ref_audio_file_to_delete.exists():
            try: temp_trimmed_ref_audio_file_to_delete.unlink(); logger.debug(f"{log_prefix}Deleted temp ref: {temp_trimmed_ref_audio_file_to_delete}")
            except OSError as e_del: logger.warning(f"{log_prefix}Could not delete temp ref '{temp_trimmed_ref_audio_file_to_delete}': {e_del}")
        logger.info(f"{log_prefix}Resource cleanup finished.")

def validate_llasa_installation() -> Tuple[bool, dict]:
    is_functional, missing = _validate_dependencies()
    objects_ok, missing_obj = _validate_model_objects()
    overall_functional = is_functional and objects_ok
    status_report = {
        "dependencies_met": is_functional, "missing_dependencies": missing,
        "python_objects_loaded": objects_ok, "missing_python_objects": missing_obj,
        "overall_functional": overall_functional,
        "details": {
            "mlx_lm_and_utils": MLX_LM_AVAILABLE, "xcodec2": XCODEC2_AVAILABLE,
            "transformers_llasa": TRANSFORMERS_FOR_LLASA_AVAILABLE, "torch_llasa": TORCH_FOR_LLASA_AVAILABLE,
            "soundfile_llasa": SF_FOR_LLASA_AVAILABLE, "scipy_llasa": SCIPY_AVAILABLE_FOR_LLASA,
            "crisptts_utils": UTILS_AVAILABLE, "pydub_llasa": PYDUB_LLASA_AVAILABLE,
            "torchaudio_llasa": TORCHAUDIO_AVAILABLE_FOR_LLASA
        }
    }
    return overall_functional, status_report

def test_llasa_basic_functionality() -> bool:
    logger.debug("LLaSA Basic Functionality Test: Starting...")
    try:
        if not all([MLX_LM_AVAILABLE, TORCH_FOR_LLASA_AVAILABLE, XCODEC2_AVAILABLE, TRANSFORMERS_FOR_LLASA_AVAILABLE, SF_FOR_LLASA_AVAILABLE]):
            logger.warning("LLaSA Basic Functionality Test: Skipping due to missing core dependencies.")
            return False
        _, pt_device = _get_pytorch_device()
        if pt_device is None: logger.error("LLaSA Basic Functionality Test: Failed to get PyTorch device."); return False
        if torch_llasa.tensor([1,2,3], device=pt_device).sum().item()!=6: logger.error("LLaSA Basic Functionality Test: PyTorch tensor sum mismatch."); return False # type: ignore
        logger.debug("LLaSA Basic Functionality Test: PyTorch tensor operations seem OK.")

        test_ids = [100,200,300]; test_tokens_str = "".join(_llasa_ids_to_speech_tokens_str(test_ids))
        if _llasa_extract_speech_ids_from_str_list(test_tokens_str) != test_ids: logger.error("LLaSA Basic Functionality Test: Speech token utils failed."); return False
        logger.debug("LLaSA Basic Functionality Test: Speech token string utils seem OK.")

        logger.info("LLaSA Handler: Basic functionality tests passed.")
        return True
    except Exception as e: logger.error(f"LLaSA Handler: Basic functionality test failed: {e}", exc_info=True); return False

def _log_initialization_status_detailed():
    is_functional, status_dict = validate_llasa_installation()
    title = "LLaSA Hybrid Handler Initialization Status"; logger.info(f"\n{'=' * 60}\n{title:^60}\n{'=' * 60}")
    logger.info(f"Overall Status: {'✓ FUNCTIONAL' if is_functional else '✗ NON-FUNCTIONAL'}")
    
    for key_check in ["dependencies_met", "python_objects_loaded"]:
        key_title = key_check.replace('_', ' ').title()
        is_met_or_loaded = status_dict.get(key_check, False)
        logger.info(f"  - {key_title:<25}: {'✓ Yes' if is_met_or_loaded else '✗ No'}")
        
        missing_list_key = ""
        if key_check == "dependencies_met":
            missing_list_key = "missing_dependencies"
        elif key_check == "python_objects_loaded":
            missing_list_key = "missing_python_objects"
        
        if not is_met_or_loaded: # Only show missing items if the category itself is 'No'
            missing_items = status_dict.get(missing_list_key, [])
            if isinstance(missing_items, list) and missing_items:
                logger.info(f"    Missing:   {', '.join(missing_items)}")
            elif missing_items: 
                 logger.info(f"    Missing items (unexpected format for {missing_list_key}): {missing_items}")

    logger.info("-" * 60 + "\nDependency Details:")
    details = status_dict.get("details", {})
    if isinstance(details, dict): 
        for comp, avail in details.items(): 
            logger.info(f"  - {comp:<25}: {'✓ Available' if avail else '✗ Missing/Unavailable'}")
    else:
        logger.warning(f"LLaSA Init: 'details' in status_report is not a dictionary: {details}")

    logger.info("-" * 60)

    if is_functional:
        logger.info("LLaSA Hybrid Handler appears ready for use.")
        if not test_llasa_basic_functionality(): 
            logger.warning("LLaSA: Basic functionality test FAILED despite dependencies appearing met. Check detailed logs for test_llasa_basic_functionality.")
    else: 
        logger.warning("LLaSA Hybrid Handler: NOT ready. Check missing components/errors above.")
    logger.info("=" * 60)

if __name__ != "__main__":
    _log_initialization_status_detailed()
    if not UTILS_AVAILABLE:
        logger_init.critical("LLaSA Hybrid Handler CRITICAL: Project 'utils.py' components not imported. Key functionalities will fail.")