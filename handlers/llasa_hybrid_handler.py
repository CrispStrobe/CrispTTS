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

mlx_lm_load, mlx_lm_generate_str_func, mlx_lm_stream_generate_func = None, None, None # Renamed for clarity
make_sampler_func = None
mlx_core = None
XCodec2Model_llasa = None
AutoTokenizer_llasa_chat = None
torch_llasa = None
sf_llasa = None
scipy_resample_func = None
save_audio_util, play_audio_util, SuppressOutput_util = None, None, None


logger_init = logging.getLogger("CrispTTS.handlers.llasa_hybrid.init")
logger = logging.getLogger("CrispTTS.handlers.llasa_hybrid")

try:
    import mlx.core as mx_imported
    from mlx_lm import load as mlx_lm_load_imported
    from mlx_lm.generate import generate as mlx_lm_generate_str_imported # For full string output
    from mlx_lm.generate import stream_generate as mlx_lm_stream_generate_imported # For token-by-token streaming if needed later
    from mlx_lm.sample_utils import make_sampler as make_sampler_imported

    mlx_core = mx_imported
    mlx_lm_load = mlx_lm_load_imported
    mlx_lm_generate_str_func = mlx_lm_generate_str_imported
    mlx_lm_stream_generate_func = mlx_lm_stream_generate_imported
    make_sampler_func = make_sampler_imported
    MLX_LM_AVAILABLE = True
    logger_init.info("MLX, mlx-lm, and make_sampler imported successfully for LLaSA Handler.")
except ImportError as e:
    logger_init.warning(f"MLX/mlx-lm or make_sampler not found. LLaSA Hybrid handler will be non-functional. Error: {e}")
except Exception as e:
    logger_init.error(f"Unexpected MLX import error: {e}", exc_info=True)

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
    torch_llasa = torch_imported
    TORCH_FOR_LLASA_AVAILABLE = True
    logger_init.info("PyTorch (for LLaSA XCodec2) imported successfully.")
except ImportError as e:
    logger_init.warning(f"PyTorch (for LLaSA XCodec2) not found. LLaSA Hybrid handler will be non-functional. Error: {e}")
except Exception as e:
    logger_init.error(f"Unexpected PyTorch import error: {e}", exc_info=True)

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
    logger_init.warning("Scipy not found. Reference audio resampling for LLaSA will not be available if needed.")

try:
    from utils import save_audio, play_audio, SuppressOutput
    save_audio_util = save_audio
    play_audio_util = play_audio
    SuppressOutput_util = SuppressOutput
    UTILS_AVAILABLE = True
    logger_init.info("Project utils imported successfully for LLaSA Handler.")
except ImportError as e:
    logger_init.error(f"Failed to import project utils: {e}. LLaSA handler save/play functionality will be affected.")


_LLASA_PYDUB_AVAILABLE = False
_AudioSegment_llasa_pydub = None
try:
    from pydub import AudioSegment as _AudioSegment_llasa_pydub_imp
    _AudioSegment_llasa_pydub = _AudioSegment_llasa_pydub_imp
    _LLASA_PYDUB_AVAILABLE = True
    logger_init.info("pydub imported successfully for LLaSA audio trimming.")
except ImportError:
    logger_init.info("pydub not available for LLaSA handler. Reference audio trimming will use numpy slicing if necessary.")


def _validate_and_prepare_reference_audio(ref_wav_path_str: str, project_root: Path, target_sr: int = 16000, max_duration_s: int = 15) -> Optional[Tuple[str, np.ndarray, int, Optional[Path]]]:
    logger.debug(f"LLaSA Hybrid Prep Ref Audio: Input path string: '{ref_wav_path_str}', Target SR: {target_sr}, Max Duration: {max_duration_s}s")
    if not ref_wav_path_str or not isinstance(ref_wav_path_str, str):
        logger.error("LLaSA Hybrid Prep Ref Audio: Invalid or empty reference WAV path string provided.")
        return None
    if not SF_FOR_LLASA_AVAILABLE:
        logger.error("LLaSA Hybrid Prep Ref Audio: SoundFile library not available for audio processing.")
        return None

    ref_wav_path_actual = Path(ref_wav_path_str)
    if not ref_wav_path_actual.is_absolute():
        ref_wav_path_actual = (project_root / ref_wav_path_actual).resolve()
    logger.debug(f"LLaSA Hybrid Prep Ref Audio: Resolved absolute path: {ref_wav_path_actual}")

    if not ref_wav_path_actual.exists() or not ref_wav_path_actual.is_file():
        logger.error(f"LLaSA Hybrid Prep Ref Audio: Reference WAV not found or not a file: {ref_wav_path_actual}")
        return None

    temp_trimmed_file_to_delete: Optional[Path] = None

    try:
        logger.debug(f"LLaSA Hybrid Prep Ref Audio: Loading reference audio from: {ref_wav_path_actual}")
        prompt_wav_samples, sr_prompt = sf_llasa.read(str(ref_wav_path_actual), dtype='float32')

        if prompt_wav_samples.size == 0:
            logger.error(f"LLaSA Hybrid Prep Ref Audio: Reference audio is empty: {ref_wav_path_actual}")
            return None
        if prompt_wav_samples.ndim > 1:
            prompt_wav_samples = np.mean(prompt_wav_samples, axis=1)
            logger.info("LLaSA Hybrid Prep Ref Audio: Converted reference audio to mono.")

        current_duration_s = len(prompt_wav_samples) / sr_prompt
        logger.debug(f"LLaSA Hybrid Prep Ref Audio: Original duration: {current_duration_s:.2f}s at {sr_prompt}Hz.")

        path_to_load_for_resample_str = str(ref_wav_path_actual)
        samples_for_resample = prompt_wav_samples
        sr_for_resample = sr_prompt

        if current_duration_s > max_duration_s:
            logger.info(f"LLaSA Hybrid Prep Ref Audio: Reference audio ({current_duration_s:.1f}s) > {max_duration_s}s. Attempting to trim.")
            trimmed_successfully = False
            if _LLASA_PYDUB_AVAILABLE and _AudioSegment_llasa_pydub:
                try:
                    audio_segment = _AudioSegment_llasa_pydub.from_file(str(ref_wav_path_actual))
                    trimmed_segment = audio_segment[:max_duration_s * 1000]

                    temp_fd, temp_trimmed_path_str = tempfile.mkstemp(suffix=ref_wav_path_actual.suffix or ".wav", prefix="llasa_trimmed_ref_")
                    os.close(temp_fd)
                    temp_trimmed_file_to_delete = Path(temp_trimmed_path_str) # Store Path object
                    path_to_load_for_resample_str = str(temp_trimmed_file_to_delete)

                    file_format = (ref_wav_path_actual.suffix.lstrip('.').lower() if ref_wav_path_actual.suffix else 'wav')
                    trimmed_segment.export(path_to_load_for_resample_str, format=file_format)

                    samples_for_resample, sr_for_resample = sf_llasa.read(path_to_load_for_resample_str, dtype='float32')
                    if samples_for_resample.ndim > 1: samples_for_resample = np.mean(samples_for_resample, axis=1)
                    current_duration_s = len(samples_for_resample) / sr_for_resample
                    logger.info(f"LLaSA Hybrid Prep Ref Audio: Trimmed using pydub to {current_duration_s:.1f}s. Temp file: {path_to_load_for_resample_str}")
                    trimmed_successfully = True
                except Exception as e_pydub_trim:
                    logger.warning(f"LLaSA Hybrid Prep Ref Audio: pydub trimming failed: {e_pydub_trim}. Using numpy slice if possible.", exc_info=True)
                    if temp_trimmed_file_to_delete and temp_trimmed_file_to_delete.exists():
                        temp_trimmed_file_to_delete.unlink(missing_ok=True)
                    temp_trimmed_file_to_delete = None
                    path_to_load_for_resample_str = str(ref_wav_path_actual)
                    samples_for_resample, sr_for_resample = sf_llasa.read(path_to_load_for_resample_str, dtype='float32')
                    if samples_for_resample.ndim > 1: samples_for_resample = np.mean(samples_for_resample, axis=1)
                    current_duration_s = len(samples_for_resample) / sr_for_resample # Re-evaluate duration of original

            if not trimmed_successfully: # Fallback to numpy slice if pydub failed or unavailable
                num_samples_to_keep = int(max_duration_s * sr_for_resample)
                if len(samples_for_resample) > num_samples_to_keep:
                    samples_for_resample = samples_for_resample[:num_samples_to_keep]
                    current_duration_s = len(samples_for_resample) / sr_for_resample
                    logger.info(f"LLaSA Hybrid Prep Ref Audio: Trimmed using numpy slice to {current_duration_s:.1f}s.")
                else:
                    logger.debug("LLaSA Hybrid Prep Ref Audio: Numpy slice not needed as audio already within max duration after initial load.")


            prompt_wav_samples = samples_for_resample
            sr_prompt = sr_for_resample
        else:
            logger.debug("LLaSA Hybrid Prep Ref Audio: Reference audio already within max duration.")


        if sr_prompt != target_sr:
            logger.info(f"LLaSA Hybrid Prep Ref Audio: Resampling ref audio from {sr_prompt}Hz to {target_sr}Hz.")
            if SCIPY_AVAILABLE_FOR_LLASA and scipy_resample_func:
                num_samples_target_sr = int(len(prompt_wav_samples) * (target_sr / sr_prompt))
                if num_samples_target_sr > 0:
                    prompt_wav_samples = scipy_resample_func(prompt_wav_samples, num_samples_target_sr)
                    sr_prompt = target_sr
                    logger.debug(f"LLaSA Hybrid Prep Ref Audio: Resampled with scipy. New duration: {len(prompt_wav_samples)/sr_prompt:.2f}s")
                else:
                    logger.warning(f"LLaSA Hybrid Prep Ref Audio: Resampling would result in 0 samples. Using original SR {sr_prompt}Hz.")
            else:
                logger.warning(f"LLaSA Hybrid Prep Ref Audio: Scipy not available for resampling. Using SR {sr_prompt}Hz, but {target_sr}Hz expected by XCodec2. Quality may be affected.")
        else:
            logger.debug(f"LLaSA Hybrid Prep Ref Audio: Reference audio already at target SR {target_sr}Hz.")

        final_duration_s = len(prompt_wav_samples) / sr_prompt
        logger.info(f"LLaSA Hybrid Prep Ref Audio: Final reference audio processed: {final_duration_s:.1f}s at {sr_prompt}Hz.")
        if final_duration_s < 0.5:
            logger.warning(f"LLaSA Hybrid Prep Ref Audio: Processed reference audio is very short ({final_duration_s:.1f}s). Voice cloning quality may be poor.")

        return str(ref_wav_path_actual), prompt_wav_samples, sr_prompt, temp_trimmed_file_to_delete

    except Exception as e:
        logger.error(f"LLaSA Hybrid Prep Ref Audio: Failed to load/process ref audio '{ref_wav_path_actual}': {e}", exc_info=True)
        if temp_trimmed_file_to_delete and temp_trimmed_file_to_delete.exists():
            try: temp_trimmed_file_to_delete.unlink(missing_ok=True)
            except OSError as e_del: logger.warning(f"LLaSA: Could not delete temp trimmed ref audio {temp_trimmed_file_to_delete} during exception handling: {e_del}")
        return None


def _validate_dependencies() -> Tuple[bool, List[str]]:
    missing_deps = []
    if not MLX_LM_AVAILABLE: missing_deps.append("MLX/mlx-lm or make_sampler")
    if not XCODEC2_AVAILABLE: missing_deps.append("XCodec2")
    if not TRANSFORMERS_FOR_LLASA_AVAILABLE: missing_deps.append("Transformers (for LLaSA chat)")
    if not TORCH_FOR_LLASA_AVAILABLE: missing_deps.append("PyTorch (for LLaSA XCodec2)")
    if not SF_FOR_LLASA_AVAILABLE: missing_deps.append("SoundFile")
    if not UTILS_AVAILABLE: missing_deps.append("CrispTTS Project Utils (save_audio, play_audio)")
    # Scipy is optional for now, but good to list
    if not SCIPY_AVAILABLE_FOR_LLASA: logger_init.debug("Note: Scipy (for resampling) is optional but recommended if ref audio SR mismatch.")
    return len(missing_deps) == 0, missing_deps

def _validate_model_objects() -> Tuple[bool, List[str]]:
    missing_objects = [];
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

def _llasa_extract_speech_ids_from_str_list(speech_tokens_list_of_str: List[str]) -> List[int]:
    speech_ids = []
    for token_str in speech_tokens_list_of_str:
        if not isinstance(token_str, str): continue
        # Regex to find all occurrences of <|s_NUMBER|>
        matches = re.findall(r"<\|s_(\d+)\|>", token_str)
        for num_str in matches:
            try:
                speech_ids.append(int(num_str))
            except ValueError:
                logger.warning(f"LLaSA: Could not parse int from speech token segment: '{num_str}' in '{token_str}'")
    logger.debug(f"LLaSA Extract Speech IDs: Input strings: {speech_tokens_list_of_str}, Extracted IDs: {speech_ids}")
    return speech_ids


def _get_pytorch_device() -> Tuple[str, Optional['torch_llasa.device']]: # type: ignore
    if not torch_llasa:
        logger.error("LLaSA Get PyTorch Device: PyTorch (torch_llasa) not available.")
        return "cpu", None # Fallback, though an error should have been raised earlier
    device_str = "cpu"
    if torch_llasa.cuda.is_available(): device_str = "cuda"
    elif hasattr(torch_llasa.backends, "mps") and torch_llasa.backends.mps.is_available(): device_str = "mps"
    logger.info(f"LLaSA: Determined PyTorch device: {device_str} for XCodec2 operations.")
    try:
        pt_device_obj = torch_llasa.device(device_str)
        logger.debug(f"LLaSA: Successfully created torch.device object for '{device_str}'.")
        return device_str, pt_device_obj
    except Exception as e:
        logger.warning(f"LLaSA: Failed to create torch.device('{device_str}'): {e}. Falling back to CPU.")
        return "cpu", torch_llasa.device("cpu")

def _cleanup_resources(*resources_to_del):
    logger.debug(f"LLaSA Cleanup: Attempting to delete {len(resources_to_del)} resource(s).")
    for idx, resource in enumerate(resources_to_del):
        if resource is not None:
            res_name = f"resource_{idx}"
            try:
                # Try to get a more descriptive name if possible (e.g., from a known variable name if passed as locals())
                # This is complex to do robustly, so a generic name is used.
                logger.debug(f"LLaSA Cleanup: Deleting {res_name} (type: {type(resource)})...")
                del resource
            except Exception as e:
                logger.debug(f"LLaSA Cleanup: Error during deletion of {res_name}: {e}")

    if TORCH_FOR_LLASA_AVAILABLE and torch_llasa:
        try:
            if torch_llasa.cuda.is_available():
                torch_llasa.cuda.empty_cache()
                logger.debug("LLaSA Cleanup: PyTorch CUDA cache cleared.")
            if hasattr(torch_llasa.backends, "mps") and torch_llasa.backends.mps.is_available() and hasattr(torch_llasa.mps, "empty_cache"):
                torch_llasa.mps.empty_cache()
                logger.debug("LLaSA Cleanup: PyTorch MPS cache cleared.")
        except Exception as e:
            logger.debug(f"LLaSA Cleanup: Error during PyTorch cache cleanup: {e}")

    if MLX_LM_AVAILABLE and mlx_core and hasattr(mlx_core, 'clear_cache'):
        try:
            mlx_core.clear_cache()
            logger.debug("LLaSA Cleanup: MLX cache cleared.")
        except Exception as e:
            logger.debug(f"LLaSA Cleanup: Error during MLX cache cleanup: {e}")
    gc.collect()
    logger.debug("LLaSA Cleanup: Garbage collection called.")


def synthesize_with_llasa_hybrid(
    model_config: dict,
    text_to_synthesize: str,
    voice_id_override: Optional[str],
    model_params_override: Optional[str],
    output_file_str: Optional[str],
    play_direct: bool
) -> None:
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'llasa_hybrid_unknown')
    logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Starting synthesis process.")
    logger.debug(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Input text (first 100 chars): '{text_to_synthesize[:100]}...'")
    logger.debug(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Voice ID override: '{voice_id_override}'")
    logger.debug(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Model params override: '{model_params_override}'")
    logger.debug(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Output file string: '{output_file_str}'")
    logger.debug(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Play direct: {play_direct}")

    deps_valid, missing_deps = _validate_dependencies()
    if not deps_valid:
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Missing critical dependencies: {', '.join(missing_deps)}. Skipping synthesis.")
        return
    objects_valid, missing_objects = _validate_model_objects()
    if not objects_valid:
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Critical Python objects/functions not imported: {', '.join(missing_objects)}. Skipping synthesis.")
        return
    if not text_to_synthesize or not isinstance(text_to_synthesize, str) or not text_to_synthesize.strip():
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Provided text is invalid or empty. Skipping synthesis.")
        return

    llm_model_id_cfg = model_config.get("llm_model_id")
    chat_tokenizer_id_cfg = model_config.get("chat_tokenizer_id")
    codec_model_id_cfg = model_config.get("codec_model_id")
    ref_speaker_wav_path_str = voice_id_override or model_config.get("default_voice_id")
    target_sample_rate = model_config.get("sample_rate", 16000)

    logger.debug(f"LLaSA Hybrid ({crisptts_model_id_for_log}): LLM: '{llm_model_id_cfg}', Chat Tokenizer: '{chat_tokenizer_id_cfg}', Codec: '{codec_model_id_cfg}', Ref WAV path: '{ref_speaker_wav_path_str}', Target SR: {target_sample_rate}Hz")

    if not all([llm_model_id_cfg, chat_tokenizer_id_cfg, codec_model_id_cfg]):
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Missing one or more critical model IDs in configuration (llm_model_id, chat_tokenizer_id, codec_model_id). Skipping.")
        return

    pt_device_str, pt_device_obj = _get_pytorch_device()
    if pt_device_obj is None:
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Could not initialize PyTorch device. Skipping.")
        return

    llasa_llm_mlx_model, llasa_llm_mlx_tokenizer, chat_template_hf_tokenizer, codec_model_pt_inst = None, None, None, None
    temp_trimmed_ref_audio_file_to_delete: Optional[Path] = None # For cleanup

    try:
        logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Loading models and tokenizers...")
        # Suppress Hugging Face/MLX download/conversion logs unless CrispTTS DEBUG is on for LLaSA handler.
        suppress_external_logs = not logger.isEnabledFor(logging.DEBUG)
        with SuppressOutput_util(suppress_stdout=suppress_external_logs, suppress_stderr=suppress_external_logs):
            try:
                logger.debug(f"LLaSA Hybrid: Loading MLX LLM '{llm_model_id_cfg}'...")
                llasa_llm_mlx_model, llasa_llm_mlx_tokenizer = mlx_lm_load(llm_model_id_cfg)
                if not llasa_llm_mlx_model or not llasa_llm_mlx_tokenizer:
                    raise ValueError(f"MLX LLM model or tokenizer for '{llm_model_id_cfg}' loaded as None.")
                logger.debug(f"LLaSA Hybrid: MLX LLM '{llm_model_id_cfg}' and its tokenizer loaded successfully.")
            except Exception as e:
                logger.error(f"LLaSA Hybrid: Failed to load MLX LLM '{llm_model_id_cfg}': {e}", exc_info=True)
                return

            try:
                logger.debug(f"LLaSA Hybrid: Loading Chat Template HF Tokenizer '{chat_tokenizer_id_cfg}'...")
                chat_template_hf_tokenizer = AutoTokenizer_llasa_chat.from_pretrained(chat_tokenizer_id_cfg)
                if not chat_template_hf_tokenizer:
                    raise ValueError(f"Chat template HF tokenizer for '{chat_tokenizer_id_cfg}' loaded as None.")
                logger.debug(f"LLaSA Hybrid: Chat Template HF Tokenizer '{chat_tokenizer_id_cfg}' loaded successfully.")
            except Exception as e:
                logger.error(f"LLaSA Hybrid: Failed to load Chat Template HF Tokenizer '{chat_tokenizer_id_cfg}': {e}", exc_info=True)
                return

            try:
                logger.debug(f"LLaSA Hybrid: Loading XCodec2 PyTorch Model '{codec_model_id_cfg}' to device '{pt_device_str}'...")
                codec_model_pt_inst = XCodec2Model_llasa.from_pretrained(codec_model_id_cfg).to(pt_device_obj).eval()
                if not codec_model_pt_inst:
                    raise ValueError(f"XCodec2 PyTorch model for '{codec_model_id_cfg}' loaded as None.")
                logger.debug(f"LLaSA Hybrid: XCodec2 PyTorch Model '{codec_model_id_cfg}' loaded successfully to '{pt_device_str}'.")
            except Exception as e:
                logger.error(f"LLaSA Hybrid: Failed to load XCodec2 PyTorch Model '{codec_model_id_cfg}': {e}", exc_info=True)
                return
        logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): All models and tokenizers loaded.")

        assistant_content_prefix_str = "<|SPEECH_GENERATION_START|>"
        if ref_speaker_wav_path_str:
            logger.debug(f"LLaSA Hybrid: Processing reference speaker WAV: '{ref_speaker_wav_path_str}'")
            project_root = Path(__file__).resolve().parent.parent # Assuming handler is in handlers/ and project root is one level up
            ref_audio_result = _validate_and_prepare_reference_audio(ref_speaker_wav_path_str, project_root, target_sr=target_sample_rate, max_duration_s=15)

            if ref_audio_result:
                _, prompt_wav_samples, sr_prompt_final, temp_trimmed_ref_audio_file_to_delete = ref_audio_result
                if sr_prompt_final == target_sample_rate:
                    try:
                        logger.info(f"LLaSA Hybrid: Encoding reference WAV ({len(prompt_wav_samples)/sr_prompt_final:.1f}s at {sr_prompt_final}Hz) with XCodec2...")
                        prompt_wav_tensor_pt = torch_llasa.from_numpy(prompt_wav_samples).float().unsqueeze(0).to(pt_device_obj)
                        logger.debug(f"LLaSA Hybrid: Reference WAV tensor shape for XCodec2: {prompt_wav_tensor_pt.shape}")
                        with torch_llasa.no_grad(), SuppressOutput_util(suppress_stdout=suppress_external_logs, suppress_stderr=suppress_external_logs):
                            vq_codes_from_prompt_pt = codec_model_pt_inst.encode_code(input_waveform=prompt_wav_tensor_pt)

                        if vq_codes_from_prompt_pt is not None and vq_codes_from_prompt_pt.numel() > 0:
                            prompt_speech_ids_integers = vq_codes_from_prompt_pt[0, 0, :].tolist() # Assuming [B, N_CODEBOOKS, T_CODES] -> take first codebook
                            speech_prefix_tokens_str_list = _llasa_ids_to_speech_tokens_str(prompt_speech_ids_integers)
                            assistant_content_prefix_str += "".join(speech_prefix_tokens_str_list)
                            logger.info(f"LLaSA Hybrid: Generated {len(prompt_speech_ids_integers)} speech prefix tokens from reference audio for voice cloning.")
                            logger.debug(f"LLaSA Hybrid: Speech prefix (first 10 tokens): {speech_prefix_tokens_str_list[:10]}")
                        else:
                            logger.warning("LLaSA Hybrid: XCodec2 encoding of reference audio resulted in empty or None codes. Proceeding without voice cloning.")
                    except Exception as e_ref_enc:
                        logger.error(f"LLaSA Hybrid: Failed to process/encode reference audio with XCodec2: {e_ref_enc}", exc_info=True)
                        logger.info("LLaSA Hybrid: Proceeding without voice cloning due to reference audio processing error.")
                else:
                    logger.warning(f"LLaSA Hybrid: Reference audio SR ({sr_prompt_final}Hz) does not match target SR ({target_sample_rate}Hz) after processing. Skipping voice cloning.")
            else:
                logger.warning("LLaSA Hybrid: Reference audio validation/processing failed. Proceeding without voice cloning.")
        else:
            logger.info("LLaSA Hybrid: No reference speaker WAV provided. Proceeding with default/generic voice.")

        user_content = f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text_to_synthesize}<|TEXT_UNDERSTANDING_END|>"
        chat_for_llasa_template = [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content_prefix_str}]
        logger.debug(f"LLaSA Hybrid: Chat template structure to be applied: {json.dumps(chat_for_llasa_template, indent=2)[:500]}...")

        try:
            prompt_string_for_llm = chat_template_hf_tokenizer.apply_chat_template(chat_for_llasa_template, tokenize=False, add_generation_prompt=True)
            if not prompt_string_for_llm:
                raise ValueError("Applying chat template resulted in an empty string.")
            logger.debug(f"LLaSA Hybrid: Full prompt string for LLM (first 300 chars): '{prompt_string_for_llm[:300]}...'")
        except Exception as e_template:
            logger.error(f"LLaSA Hybrid: Failed to apply chat template: {e_template}", exc_info=True)
            return

        # --- Max tokens for LLM generation ---
        # We need to consider the length of the prompt tokens when setting max_new_tokens.
        # The prompt_string_for_llm already contains the (potentially long) speech prefix from cloning.
        try:
            prompt_tokens_for_llm_list = llasa_llm_mlx_tokenizer.encode(prompt_string_for_llm)
            num_prompt_tokens = len(prompt_tokens_for_llm_list)
            logger.debug(f"LLaSA Hybrid: Number of tokens in the full prompt to LLM: {num_prompt_tokens}")
        except Exception as e_enc:
            logger.error(f"LLaSA Hybrid: Failed to encode full prompt string for length calculation: {e_enc}", exc_info=True)
            return

        # Max new tokens calculation logic from mlx-lm example
        # Roughly, 1 sec audio is ~75 speech tokens. A typical short sentence might be 3-7 seconds.
        # Let's aim for max ~15 seconds of audio -> 15 * 75 = 1125 speech tokens.
        # LLaSA's max context seems to be 2048.
        desired_max_speech_tokens = 1200 # Generous upper bound for speech tokens
        max_context_len_llm = getattr(llasa_llm_mlx_model, 'config', {}).get('max_position_embeddings', 2048)
        
        max_new_tokens_for_llm = max_context_len_llm - num_prompt_tokens - 20 # Buffer for EOS, etc.
        max_new_tokens_for_llm = min(max_new_tokens_for_llm, desired_max_speech_tokens) # Don't generate excessively long audio
        max_new_tokens_for_llm = max(50, max_new_tokens_for_llm) # Ensure at least some tokens can be generated

        if max_new_tokens_for_llm <= 0:
            logger.error(f"LLaSA Hybrid: Calculated max_new_tokens_for_llm ({max_new_tokens_for_llm}) is too low or negative. Prompt might be too long ({num_prompt_tokens} tokens for context {max_context_len_llm}). Cannot generate.")
            return
        logger.debug(f"LLaSA Hybrid: Max new tokens for LLM generation: {max_new_tokens_for_llm}")
        
        # --- Sampling Parameters ---
        current_temp = 0.8
        current_top_p = 1.0
        # Add other sampling defaults if make_sampler needs them
        min_p_val = 0.0
        top_k_val = 0 # 0 means disabled in mlx-lm

        if model_params_override:
            try:
                cli_params = json.loads(model_params_override)
                current_temp = float(cli_params.get("temperature", current_temp))
                current_top_p = float(cli_params.get("top_p", current_top_p))
                min_p_val = float(cli_params.get("min_p", min_p_val))
                top_k_val = int(cli_params.get("top_k", top_k_val))
                logger.debug(f"LLaSA Hybrid: Using custom generation parameters from CLI: temp={current_temp}, top_p={current_top_p}, min_p={min_p_val}, top_k={top_k_val}")
            except Exception as e_parse_params:
                logger.warning(f"LLaSA Hybrid: Could not parse --model-params '{model_params_override}': {e_parse_params}. Using default sampling parameters.")
        
        llasa_sampler = make_sampler_func(
            temp=current_temp,
            top_p=current_top_p,
            min_p=min_p_val,
            top_k=top_k_val
            # Add other params like min_tokens_to_keep, xtc settings if you plan to support them
        )
        logger.debug(f"LLaSA Hybrid: Sampler configured: {llasa_sampler}")

        speech_gen_end_token_str = '<|SPEECH_GENERATION_END|>'
        eos_token_id_for_llm = None
        try:
            # Ensure not to add bos/eos when encoding just the EOS string itself.
            # The `encode` method of HF tokenizers typically has `add_special_tokens`
            encoded_custom_eos_list = llasa_llm_mlx_tokenizer.encode(speech_gen_end_token_str, add_special_tokens=False)
            if encoded_custom_eos_list and len(encoded_custom_eos_list) > 0:
                eos_token_id_for_llm = encoded_custom_eos_list[0]
                # Check if it's UNK
                if hasattr(llasa_llm_mlx_tokenizer, 'unk_token_id') and eos_token_id_for_llm == llasa_llm_mlx_tokenizer.unk_token_id:
                    logger.warning(f"LLaSA Hybrid: Custom EOS '{speech_gen_end_token_str}' encoded to UNK token. Fallback to tokenizer's default EOS might be used by mlx-lm.")
                    eos_token_id_for_llm = None # Let mlx-lm handle default if custom is UNK
                else:
                     logger.debug(f"LLaSA Hybrid: Using custom EOS token ID for LLM generation: {eos_token_id_for_llm} (for string '{speech_gen_end_token_str}')")
            else:
                 logger.warning(f"LLaSA Hybrid: Encoding custom EOS '{speech_gen_end_token_str}' resulted in empty list. Fallback to tokenizer's default EOS may occur in mlx-lm.")
        except Exception as e_enc_eos:
            logger.warning(f"LLaSA Hybrid: Error encoding custom EOS string '{speech_gen_end_token_str}': {e_enc_eos}. mlx-lm might use its default EOS.")


        logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Generating speech tokens with LLM (max_new_tokens={max_new_tokens_for_llm}). Sampler params: temp={current_temp}, top_p={current_top_p}, min_p={min_p_val}, top_k={top_k_val}")
        
        generated_llm_output_string = ""
        try:
            # Use the mlx_lm.generate.generate function that returns the full string
            logger.debug(f"LLaSA Hybrid: Determined custom EOS token ID for LLM generation (if supported by API): {eos_token_id_for_llm} (for string '{speech_gen_end_token_str}')")

            generated_llm_output_string = mlx_lm_generate_str_func(
                model=llasa_llm_mlx_model,
                tokenizer=llasa_llm_mlx_tokenizer,
                prompt=prompt_string_for_llm,
                max_tokens=max_new_tokens_for_llm,
                sampler=llasa_sampler,
                # eos_token_id=eos_token_id_for_llm, # <-- REMOVE/COMMENT THIS
                verbose=logger.isEnabledFor(logging.DEBUG)
            )
            if not generated_llm_output_string:
                logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): LLM generation returned an empty string.")
                return
            logger.debug(f"LLaSA Hybrid: Raw LLM output string (first 300 chars): '{generated_llm_output_string[:300]}...'")

            if speech_gen_end_token_str in generated_llm_output_string:
                logger.debug(f"LLaSA Hybrid: Custom EOS string '{speech_gen_end_token_str}' found in LLM output. Truncating completion based on it.")
                # Ensure you split the part *after* the prompt from the LLM output first
                completion_part_before_eos_truncate = generated_llm_output_string # Placeholder for actual completion part
                if generated_llm_output_string.startswith(prompt_string_for_llm):
                    completion_part_before_eos_truncate = generated_llm_output_string[len(prompt_string_for_llm):]
                else: # Fallback split, might not be perfect
                    if assistant_content_prefix_str in generated_llm_output_string:
                        parts = generated_llm_output_string.split(assistant_content_prefix_str, 1)
                        if len(parts) > 1: completion_part_before_eos_truncate = parts[1]

                # Now truncate this completion part
                completion_part_str = completion_part_before_eos_truncate.split(speech_gen_end_token_str, 1)[0]
                logger.debug(f"LLaSA Hybrid: Completion part truncated at custom EOS. New length: {len(completion_part_str)}")
            else:
                logger.debug(f"LLaSA Hybrid: Custom EOS string '{speech_gen_end_token_str}' NOT found in LLM output. Will use output as is (might be truncated by max_tokens).")
                # Logic to get completion_part_str without custom EOS truncation
                if generated_llm_output_string.startswith(prompt_string_for_llm):
                    completion_part_str = generated_llm_output_string[len(prompt_string_for_llm):]
                else:
                    if assistant_content_prefix_str in generated_llm_output_string:
                        parts = generated_llm_output_string.split(assistant_content_prefix_str, 1)
                        if len(parts) > 1: completion_part_str = parts[1]
                        else: completion_part_str = generated_llm_output_string
                    else: completion_part_str = generated_llm_output_string
                    
        except Exception as e_llm_gen:
            logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): LLM generation failed: {e_llm_gen}", exc_info=True)
            return

        # Extract only the newly generated part if the LLM output includes the prompt
        # This behavior can vary based on the LLM and tokenizer's chat template application.
        # A robust way is to find the start of the assistant's _actual_ generated content.
        # The prompt_string_for_llm already includes the assistant_content_prefix_str.
        # We need the content *after* that prefix within the LLM's full output.
        
        completion_part_str = ""
        if generated_llm_output_string.startswith(prompt_string_for_llm):
            completion_part_str = generated_llm_output_string[len(prompt_string_for_llm):]
            logger.debug("LLaSA Hybrid: Stripped prompt from LLM output to get completion part.")
        else:
            # Fallback: try to find assistant_content_prefix_str if stripping full prompt fails
            # This might happen if the LLM slightly reformats the prompt.
            split_marker = assistant_content_prefix_str
            if assistant_content_prefix_str in generated_llm_output_string:
                parts = generated_llm_output_string.split(split_marker, 1)
                if len(parts) > 1:
                    completion_part_str = parts[1]
                    logger.debug(f"LLaSA Hybrid: Extracted completion part by splitting on assistant prefix.")
                else:
                    completion_part_str = generated_llm_output_string # Could not reliably split
                    logger.warning("LLaSA Hybrid: Could not split LLM output on assistant prefix, using full output. May contain prompt.")
            else:
                completion_part_str = generated_llm_output_string # Could not find prefix
                logger.warning(f"LLaSA Hybrid: Assistant prefix '{split_marker}' not found in LLM output. Using full output string. This may contain the prompt.")

        logger.debug(f"LLaSA Hybrid: Extracted completion part (first 200 chars): '{completion_part_str[:200]}...'")
        if not completion_part_str.strip():
            logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Extracted completion from LLM is empty or whitespace.")
            return

        try:
            # Speech tokens are of the form <|s_NUMBER|>
            # Regex findall should capture all such tokens.
            speech_tokens_as_strings_list = re.findall(r"(<\|s_\d+\|>)", completion_part_str)
            if not speech_tokens_as_strings_list:
                logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): No speech tokens (e.g., <|s_...|>) found in the LLM's generated completion part: '{completion_part_str[:200]}...'")
                return
            logger.debug(f"LLaSA Hybrid: Found {len(speech_tokens_as_strings_list)} speech token strings (e.g., <|s_...|>) in completion.")

            speech_integer_ids = _llasa_extract_speech_ids_from_str_list(speech_tokens_as_strings_list)
            if not speech_integer_ids:
                logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): No valid integer speech IDs extracted from the found token strings.")
                return
            logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Extracted {len(speech_integer_ids)} speech integer IDs for XCodec2 decoding.")
            logger.debug(f"LLaSA Hybrid: First 10 extracted speech integer IDs: {speech_integer_ids[:10]}")
        except Exception as e_extract_speech_ids:
            logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Failed to extract speech tokens/IDs from LLM output: {e_extract_speech_ids}", exc_info=True)
            return

        try:
            speech_tokens_tensor_pt = torch_llasa.tensor(speech_integer_ids, device=pt_device_obj).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, NumSpeechTokens]
            logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Decoding {len(speech_integer_ids)} speech IDs with XCodec2...")
            logger.debug(f"LLaSA Hybrid: XCodec2 input tensor shape: {speech_tokens_tensor_pt.shape}, device: {speech_tokens_tensor_pt.device}")

            with torch_llasa.no_grad(), SuppressOutput_util(suppress_stdout=suppress_external_logs, suppress_stderr=suppress_external_logs):
                gen_wav_pt = codec_model_pt_inst.decode_code(speech_tokens_tensor_pt)

            if gen_wav_pt is None:
                logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): XCodec2 decoding returned None.")
                return
            logger.debug(f"LLaSA Hybrid: XCodec2 output tensor shape: {gen_wav_pt.shape}")
            audio_numpy = gen_wav_pt[0, 0, :].cpu().numpy() # Shape: [NumSamples]

            if audio_numpy.size == 0:
                logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): XCodec2 decoding resulted in an empty audio array.")
                return
            if np.all(np.abs(audio_numpy) < 1e-6): # Check if audio is effectively silent
                logger.warning(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Generated audio is silent or near-silent.")
            elif np.max(np.abs(audio_numpy)) < 0.001:
                logger.warning(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Generated audio has very low amplitude (max abs: {np.max(np.abs(audio_numpy)):.2e}).")

            audio_duration_s = audio_numpy.shape[0] / target_sample_rate
            logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): XCodec2 decoding successful. Generated {audio_numpy.shape[0]} audio samples ({audio_duration_s:.2f}s) at {target_sample_rate}Hz.")

        except Exception as e_xcodec_decode:
            logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): XCodec2 decoding failed: {e_xcodec_decode}", exc_info=True)
            return

        if output_file_str and UTILS_AVAILABLE and save_audio_util:
            try:
                output_path_obj = Path(output_file_str).with_suffix(".wav")
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"LLaSA Hybrid: Preparing to save audio. Output path: {output_path_obj}, Sample rate: {target_sample_rate}Hz.")
                # Ensure audio_numpy is float32, then convert to int16 bytes for save_audio's PCM path
                audio_to_save_float32 = audio_numpy.astype(np.float32)
                audio_int16_bytes = (np.clip(audio_to_save_float32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                save_audio_util(audio_int16_bytes, str(output_path_obj), source_is_path=False, input_format="pcm_s16le", sample_rate=target_sample_rate)
                if not output_path_obj.exists() or output_path_obj.stat().st_size < 1000: # Check if file was actually saved and is not tiny
                    logger.warning(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Saved audio file '{output_path_obj}' is missing or very small after save attempt.")
                else:
                    logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Audio successfully saved to {output_path_obj}")
            except Exception as e_save:
                logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Failed to save audio to '{output_path_obj}': {e_save}", exc_info=True)

        if play_direct and UTILS_AVAILABLE and play_audio_util:
            try:
                logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Attempting direct playback of generated audio...")
                audio_to_play_float32 = audio_numpy.astype(np.float32)
                audio_int16_bytes_play = (np.clip(audio_to_play_float32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                play_audio_util(audio_int16_bytes_play, is_path=False, input_format="pcm_s16le", sample_rate=target_sample_rate)
                logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Playback finished (or initiated if non-blocking).")
            except Exception as e_play:
                logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Failed to play audio directly: {e_play}", exc_info=True)

        logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Synthesis process completed.")

    except Exception as e_outer:
        logger.error(f"LLaSA Hybrid ({crisptts_model_id_for_log}): An unexpected error occurred in the main synthesis try-block: {e_outer}", exc_info=True)
    finally:
        logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Entering finally block for resource cleanup.")
        _cleanup_resources(llasa_llm_mlx_model, llasa_llm_mlx_tokenizer, chat_template_hf_tokenizer, codec_model_pt_inst)
        if temp_trimmed_ref_audio_file_to_delete and temp_trimmed_ref_audio_file_to_delete.exists():
            try:
                temp_trimmed_ref_audio_file_to_delete.unlink()
                logger.debug(f"LLaSA Hybrid Cleanup: Successfully deleted temporary trimmed reference audio: {temp_trimmed_ref_audio_file_to_delete}")
            except OSError as e_del_temp:
                logger.warning(f"LLaSA Hybrid Cleanup: Could not delete temporary trimmed reference audio '{temp_trimmed_ref_audio_file_to_delete}': {e_del_temp}")
        logger.info(f"LLaSA Hybrid ({crisptts_model_id_for_log}): Resource cleanup finished.")


def validate_llasa_installation() -> Tuple[bool, dict]:
    is_functional, missing = _validate_dependencies()
    objects_ok, missing_obj = _validate_model_objects()
    overall_functional = is_functional and objects_ok

    status_report = {
        "dependencies_met": is_functional,
        "missing_dependencies": missing,
        "python_objects_loaded": objects_ok,
        "missing_python_objects": missing_obj,
        "overall_functional": overall_functional,
        "details": {
            "mlx_lm": MLX_LM_AVAILABLE,
            "xcodec2": XCODEC2_AVAILABLE,
            "transformers_llasa": TRANSFORMERS_FOR_LLASA_AVAILABLE,
            "torch_llasa": TORCH_FOR_LLASA_AVAILABLE,
            "soundfile_llasa": SF_FOR_LLASA_AVAILABLE,
            "scipy_llasa": SCIPY_AVAILABLE_FOR_LLASA,
            "crisptts_utils": UTILS_AVAILABLE,
            "pydub_llasa": _LLASA_PYDUB_AVAILABLE
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
        if pt_device is None:
            logger.error("LLaSA Basic Functionality Test: Failed to get PyTorch device.")
            return False
        if torch_llasa.tensor([1, 2, 3], device=pt_device).sum().item() != 6:
            logger.error("LLaSA Basic Functionality Test: PyTorch tensor sum mismatch.")
            return False
        logger.debug("LLaSA Basic Functionality Test: PyTorch tensor operations seem OK.")

        test_ids = [100, 200, 300]
        test_tokens_str_list = _llasa_ids_to_speech_tokens_str(test_ids)
        if _llasa_extract_speech_ids_from_str_list(test_tokens_str_list) != test_ids:
            logger.error("LLaSA Basic Functionality Test: Speech token string conversion/extraction failed.")
            return False
        logger.debug("LLaSA Basic Functionality Test: Speech token string utils seem OK.")

        logger.info("LLaSA Handler: Basic functionality tests passed.")
        return True
    except Exception as e:
        logger.error(f"LLaSA Handler: Basic functionality test failed with exception: {e}", exc_info=True)
        return False

# This function is not used by CrispTTS main directly, but good for standalone module testing
def get_llasa_system_info() -> dict:
    info = {"platform": sys.platform, "python_version": sys.version, "dependencies": {}}
    if torch_llasa:
        info["torch_version"] = torch_llasa.__version__
        info["cuda_available"] = torch_llasa.cuda.is_available()
        info["mps_available"] = hasattr(torch_llasa.backends, "mps") and torch_llasa.backends.mps.is_available()
        if torch_llasa.cuda.is_available() and torch_llasa.cuda.device_count() > 0:
            info["cuda_device_name"] = torch_llasa.cuda.get_device_name(0)
    if mlx_core:
        info["mlx_available"] = True
        if hasattr(mlx_core, 'default_device'):
             info["mlx_default_device"] = str(mlx_core.default_device())
    if sf_llasa: info["soundfile_version"] = getattr(sf_llasa, '__version__', 'unknown')
    if AutoTokenizer_llasa_chat:
        try:
            import transformers
            info["transformers_version"] = transformers.__version__
        except ImportError: info["transformers_version"] = "unknown (transformers not importable at info time)"
    if scipy_resample_func:
        try:
            import scipy
            info["scipy_version"] = scipy.__version__
        except ImportError: info["scipy_version"] = "unknown (scipy not importable at info time)"
    return info

def _log_initialization_status_detailed():
    is_functional, status_dict = validate_llasa_installation()
    title = "LLaSA Hybrid Handler Initialization Status"
    logger.info(f"\n{'=' * 60}\n{title:^60}\n{'=' * 60}")
    logger.info(f"Overall Status: {'✓ FUNCTIONAL' if is_functional else '✗ NON-FUNCTIONAL (see details below)'}")
    logger.info(f"  - Dependencies Met:       {'✓ Yes' if status_dict['dependencies_met'] else '✗ No'}")
    if not status_dict['dependencies_met']:
        logger.info(f"    Missing Dependencies:   {', '.join(status_dict['missing_dependencies'])}")
    logger.info(f"  - Python Objects Loaded:  {'✓ Yes' if status_dict['python_objects_loaded'] else '✗ No'}")
    if not status_dict['python_objects_loaded']:
        logger.info(f"    Missing Python Objects: {', '.join(status_dict['missing_python_objects'])}")

    logger.info("-" * 60 + "\nDependency Details:")
    for component, available in status_dict.get("details", {}).items():
        logger.info(f"  - {component:<25}: {'✓ Available' if available else '✗ Missing/Unavailable'}")
    logger.info("-" * 60)

    if is_functional:
        logger.info("LLaSA Hybrid Handler appears to be ready for use.")
        if test_llasa_basic_functionality(): # This will log its own success/failure
            pass # Already logged by the function
        else:
            logger.warning("LLaSA Hybrid Handler: Basic functionality test FAILED despite dependencies appearing met. Check detailed logs.")
    else:
        logger.warning("LLaSA Hybrid Handler: NOT ready due to missing components or failed checks. Please install/resolve the listed issues.")
    logger.info("=" * 60)


if __name__ != "__main__": # Standard module import, not run as script
    _log_initialization_status_detailed()
    if not UTILS_AVAILABLE:
        # This warning is critical if the handler is imported and expected to work with CrispTTS's utils
        logger.error("LLaSA Hybrid Handler CRITICAL: Project 'utils.py' (save_audio, play_audio) was not imported. Audio saving/playback will fail.")