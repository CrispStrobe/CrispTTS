# handlers/f5_tts_handler.py

import logging
import os
from pathlib import Path
import gc
import json
import shutil
import tempfile
import numpy as np

# Conditional Imports
TORCH_FOR_F5_HANDLER = False
torch_f5 = None
IS_MPS_FOR_F5_HANDLER = False

F5TTSStandardAPIClass = None # For standard PyTorch F5-TTS
F5_TTS_STANDARD_API_AVAILABLE = False

generate_mlx_func = None # For f5-tts-mlx
F5_TTS_MLX_GENERATE_AVAILABLE = False

transformers_pipeline_f5 = None # For Whisper transcription
TRANSFORMERS_PIPELINE_F5_AVAILABLE = False

torchaudio_f5 = None
TORCHAUDIO_F5_AVAILABLE = False

soundfile_f5 = None
SOUNDFILE_F5_AVAILABLE = False

PYDUB_FOR_F5_HANDLER = False
AudioSegment_pydub_f5 = None

OmegaConf_f5 = None # Not strictly needed for the fix, but part of the original conditional imports
OMEGACONF_F5_AVAILABLE = False

logger_init = logging.getLogger("CrispTTS.handlers.f5_tts.init")

try:
    import torch as torch_imp
    torch_f5 = torch_imp
    TORCH_FOR_F5_HANDLER = True
    if hasattr(torch_f5.backends, "mps") and torch_f5.backends.mps.is_available():
        IS_MPS_FOR_F5_HANDLER = True
    logger_init.info("PyTorch imported successfully for F5-TTS handler.")
except ImportError:
    logger_init.warning("PyTorch not found. F5-TTS handler will be limited.")

if TORCH_FOR_F5_HANDLER:
    try:
        from f5_tts.api import F5TTS as F5TTSStandardAPI_imp
        F5TTSStandardAPIClass = F5TTSStandardAPI_imp
        F5_TTS_STANDARD_API_AVAILABLE = True
        logger_init.info("Standard f5_tts.api.F5TTS class imported successfully.")
    except ImportError:
        logger_init.info("Could not import F5TTS from f5_tts.api. PyTorch backend for F5 will be unavailable.")

    try:
        from f5_tts_mlx.generate import generate as generate_mlx_imp
        generate_mlx_func = generate_mlx_imp
        F5_TTS_MLX_GENERATE_AVAILABLE = True
        logger_init.info("f5_tts_mlx.generate function imported successfully.")
    except ImportError:
        logger_init.info("Could not import 'generate' from 'f5_tts_mlx'. MLX backend for F5 will be unavailable.")

    try:
        from transformers import pipeline as pipeline_hf_func
        transformers_pipeline_f5 = pipeline_hf_func
        TRANSFORMERS_PIPELINE_F5_AVAILABLE = True
        logger_init.info("Transformers pipeline imported (for Whisper).")
    except ImportError:
        logger_init.warning("Transformers pipeline not found. Transcription for F5-TTS reference audio will not be available.")

    try:
        import torchaudio as ta_imp
        torchaudio_f5 = ta_imp
        TORCHAUDIO_F5_AVAILABLE = True
    except ImportError:
        logger_init.warning("Torchaudio not found. Required for some F5-TTS audio operations.")
        
    try:
        from omegaconf import OmegaConf as OmegaConf_imp #
        OmegaConf_f5 = OmegaConf_imp #
        OMEGACONF_F5_AVAILABLE = True #
    except ImportError: #
        logger_init.warning("OmegaConf not found. F5-TTS PyTorch backend may have issues loading configs.") #


try:
    import soundfile as sf_imp
    soundfile_f5 = sf_imp
    SOUNDFILE_F5_AVAILABLE = True
except ImportError:
    logger_init.warning("SoundFile not found. Saving audio for F5-TTS handler might fail.")

try:
    from pydub import AudioSegment #
    AudioSegment_pydub_f5 = AudioSegment #
    PYDUB_FOR_F5_HANDLER = True #
    logger_init.info("pydub imported for F5-TTS audio processing.") #
except ImportError: #
    logger_init.warning("pydub not found. Reference audio trimming/conversion for F5-TTS will be limited.") #


from utils import save_audio, play_audio, SuppressOutput, get_huggingface_cache_dir

logger = logging.getLogger("CrispTTS.handlers.f5_tts")
HF_CACHE_DIR = get_huggingface_cache_dir()

_transcription_cache = {}


def _prepare_ref_audio_for_f5(ref_audio_path: Path, target_dir: Path, max_duration_s: int = 28, target_sr: int = 24000) -> tuple[Path | None, bool]: #
    if not PYDUB_FOR_F5_HANDLER or not AudioSegment_pydub_f5: #
        logger.warning("F5-TTS: Pydub not available for reference audio preparation. Using original file (might fail if not WAV/suitable).")
        if ref_audio_path.suffix.lower() != ".wav":
             logger.error("F5-TTS: Pydub is unavailable, and reference audio is not WAV. Processing will likely fail.")
             return None, False
        return ref_audio_path, False #

    try:
        target_dir.mkdir(parents=True, exist_ok=True) #
        sound = AudioSegment_pydub_f5.from_file(ref_audio_path) #

        if len(sound) > max_duration_s * 1000: #
            sound = sound[:max_duration_s * 1000] #
            logger.info(f"F5-TTS: Reference audio trimmed to {max_duration_s} seconds.") #
        
        if sound.frame_rate != target_sr: #
            sound = sound.set_frame_rate(target_sr) #
        if sound.channels > 1: #
            sound = sound.set_channels(1) #
        sound = sound.set_sample_width(2) #

        processed_filename = f"processed_{ref_audio_path.stem}.wav" #
        final_path_for_f5 = target_dir / processed_filename #
        sound.export(str(final_path_for_f5), format="wav") #
        logger.debug(f"F5-TTS: Reference audio processed and saved to {final_path_for_f5}") #
        return final_path_for_f5, True #

    except Exception as e: #
        logger.error(f"F5-TTS: Error preparing reference audio '{ref_audio_path}': {e}", exc_info=True) #
        return None, False #


def _transcribe_ref_audio_with_whisper(audio_path_str: str, whisper_model_id_cfg: str,
                                     target_device_for_whisper: str, language_for_whisper: str | None,
                                     hf_token: str | None) -> tuple[str | None, str | None]: #
    global _transcription_cache #
    cache_key = (audio_path_str, whisper_model_id_cfg, language_for_whisper) #
    if cache_key in _transcription_cache: #
        logger.info(f"F5-TTS: Using cached transcription for '{Path(audio_path_str).name}'.") #
        return _transcription_cache[cache_key] #

    if not TRANSFORMERS_PIPELINE_F5_AVAILABLE or not transformers_pipeline_f5 or not TORCH_FOR_F5_HANDLER: #
        return None, "Whisper (transformers/torch) not available for transcription." #

    logger.info(f"F5-TTS: Initializing Whisper ('{whisper_model_id_cfg}') on device '{target_device_for_whisper}'.") #
    whisper_pipeline_instance = None #
    transcribed_text, err_msg = None, None #
    
    try:
        dtype_for_whisper = torch_f5.float16 if target_device_for_whisper != "cpu" else torch_f5.float32 #
        
        whisper_pipeline_instance = transformers_pipeline_f5( #
            task="automatic-speech-recognition", #
            model=whisper_model_id_cfg, #
            torch_dtype=dtype_for_whisper, #
            device=target_device_for_whisper if target_device_for_whisper != "cpu" else -1, #
            token=hf_token, #
            model_kwargs={"load_in_4bit": False, "load_in_8bit": False} #
        )
        logger.info(f"F5-TTS: Transcribing '{Path(audio_path_str).name}'...") #
        
        with open(audio_path_str, "rb") as f_audio: #
            audio_input_bytes = f_audio.read() #

        generate_kwargs = {"return_timestamps": True} #
        if language_for_whisper: #
            generate_kwargs["language"] = language_for_whisper.lower() #

        with SuppressOutput(suppress_stderr=True, suppress_stdout=not logger.isEnabledFor(logging.DEBUG)): #
            transcription_result = whisper_pipeline_instance(audio_input_bytes, generate_kwargs=generate_kwargs) #
        
        transcribed_text = transcription_result.get("text", "").strip() if transcription_result else "" #
        if not transcribed_text: #
            err_msg = "Whisper returned empty transcription." #
        else: #
            logger.info(f"F5-TTS: Transcription: '{transcribed_text[:100]}...'") #

    except Exception as e: #
        err_msg = f"Whisper failed ('{Path(audio_path_str).name}'): {e}" #
        logger.error(f"F5-TTS: {err_msg}", exc_info=True) #
        transcribed_text = None #
    finally: #
        del whisper_pipeline_instance #
        if TORCH_FOR_F5_HANDLER: #
            if torch_f5.cuda.is_available(): torch_f5.cuda.empty_cache() #
            if IS_MPS_FOR_F5_HANDLER and hasattr(torch_f5.mps, "empty_cache"): #
                try: torch_f5.mps.empty_cache() #
                except Exception: pass #
        gc.collect() #

    if transcribed_text and not err_msg: #
        _transcription_cache[cache_key] = (transcribed_text, None) #
    return transcribed_text, err_msg #


def synthesize_with_f5_tts(
    crisptts_model_config: dict,
    text: str,
    voice_id_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
): #
    model_repo_id = crisptts_model_config.get("model_repo_id") #
    display_model_id = model_repo_id or "f5_tts_unknown" #
    logger.info(f"F5-TTS ({display_model_id}): Starting synthesis for model ID '{model_repo_id}'.") #

    if not TORCH_FOR_F5_HANDLER or not soundfile_f5: #
        logger.error(f"F5-TTS ({display_model_id}): Core dependencies (PyTorch, SoundFile) missing. Skipping.") #
        return #

    use_mlx_preferred = crisptts_model_config.get("use_mlx", False) and F5_TTS_MLX_GENERATE_AVAILABLE #
    
    effective_ref_audio_path_str = voice_id_override or crisptts_model_config.get("default_voice_id", "./german.wav") #
    effective_ref_audio_path = Path(effective_ref_audio_path_str).resolve() #

    if not effective_ref_audio_path.exists(): #
        logger.error(f"F5-TTS ({display_model_id}): Reference audio path not found: {effective_ref_audio_path}. Skipping.") #
        return #

    language_code = crisptts_model_config.get("language", "de") #
    whisper_model_id = crisptts_model_config.get("whisper_model_id_for_transcription", "openai/whisper-base") #
    
    cli_params = {} #
    if model_params_override: #
        try: cli_params = json.loads(model_params_override) #
        except json.JSONDecodeError: logger.warning(f"F5-TTS ({display_model_id}): Could not parse --model-params: {model_params_override}") #

    effective_output_path_for_handler = Path(output_file_str).with_suffix(".wav") if output_file_str else None #
    
    synthesis_successful = False #
    temp_ref_audio_dir_obj = None #

    try:
        temp_ref_audio_dir_obj = Path(tempfile.mkdtemp(prefix="crisptts_f5_ref_")) #

        if use_mlx_preferred: #
            logger.info(f"F5-TTS ({display_model_id}): Selected MLX backend as preferred.") #
            if not generate_mlx_func: #
                logger.error(f"F5-TTS ({display_model_id}): MLX generate function not available. Skipping MLX attempt.") #
            else: #
                logger.info(f"F5-TTS ({display_model_id}): Attempting synthesis with MLX backend.") #
                
                prepared_ref_audio_path_mlx, _ = _prepare_ref_audio_for_f5( #
                    ref_audio_path=effective_ref_audio_path, #
                    target_dir=temp_ref_audio_dir_obj, #
                    max_duration_s=28, #
                    target_sr=16000 #
                )

                if not prepared_ref_audio_path_mlx or not prepared_ref_audio_path_mlx.exists(): #
                    logger.error(f"F5-TTS ({display_model_id}): MLX - Reference audio preparation failed for {effective_ref_audio_path}.") #
                    return #

                ref_audio_text, trans_err = _transcribe_ref_audio_with_whisper( #
                    audio_path_str=str(prepared_ref_audio_path_mlx), #
                    whisper_model_id_cfg=whisper_model_id, #
                    target_device_for_whisper="cpu", #
                    language_for_whisper=language_code if language_code != "multilingual" else None, #
                    hf_token=os.getenv("HF_TOKEN") #
                )
                if trans_err: #
                    logger.warning(f"F5-TTS ({display_model_id}): MLX - Whisper transcription failed: {trans_err}. Using placeholder.") #
                    ref_audio_text = ref_audio_text or "Reference audio transcription failed." #

                mlx_args = { #
                    "generation_text": text, #
                    "model_name": model_repo_id, #
                    "ref_audio_path": str(prepared_ref_audio_path_mlx), #
                    "ref_audio_text": ref_audio_text, #
                    "output_path": str(effective_output_path_for_handler) if effective_output_path_for_handler else "f5_mlx_output.wav", #
                    "steps": int(cli_params.get("steps", crisptts_model_config.get("default_steps", 32))), #
                    "cfg_strength": float(cli_params.get("cfg_strength", crisptts_model_config.get("default_cfg_strength", 2.0))), #
                    "sway_sampling_coef": float(cli_params.get("sway", -1.0)), #
                    "speed": float(cli_params.get("speed", 1.0)), #
                    "estimate_duration": cli_params.get("estimate_duration", True), #
                    "duration": cli_params.get("duration", None), #
                    # "fix_duration": cli_params.get("fix_duration", None), # REMOVED THIS LINE
                }
                logger.debug(f"F5-TTS ({display_model_id}): MLX generate args: {mlx_args}") #
                with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True): #
                    generate_mlx_func(**mlx_args) #
                
                output_path_to_check = Path(mlx_args["output_path"])
                if output_path_to_check.exists() and output_path_to_check.stat().st_size > 100: #
                    synthesis_successful = True #
                    # If effective_output_path_for_handler was None, but mlx_args["output_path"] (default) was created,
                    # we need to copy it to where the main function expects it if output_file_str was given.
                    # However, current logic sets mlx_args["output_path"] based on effective_output_path_for_handler.
                    # So if effective_output_path_for_handler is None, the file is "f5_mlx_output.wav"
                    if not effective_output_path_for_handler and output_file_str: # This case should not happen if output_file_str is given
                         logger.warning("F5-TTS MLX: Output was generated to default, but output_file_str was provided. This is unexpected.")
                    elif not effective_output_path_for_handler:
                         logger.info(f"F5-TTS MLX output saved to default path: {output_path_to_check.resolve()}")
                
        if not synthesis_successful: #
            if use_mlx_preferred: logger.warning(f"F5-TTS ({display_model_id}): MLX backend failed or produced no output. Attempting PyTorch Standard backend.") #
            else: logger.info(f"F5-TTS ({display_model_id}): Selected PyTorch Standard backend.") #

            if not F5_TTS_STANDARD_API_AVAILABLE or not F5TTSStandardAPIClass: #
                logger.error(f"F5-TTS ({display_model_id}): Standard F5TTS API class not available. Cannot use PyTorch backend. Skipping.") #
                return #
            if not OMEGACONF_F5_AVAILABLE: #
                logger.error(f"F5-TTS ({display_model_id}): OmegaConf not available. Cannot load configs for PyTorch backend. Skipping.") #
                return #

            logger.info(f"F5-TTS ({display_model_id}): Executing synthesis with PyTorch Standard backend.") #
            pytorch_device = "cuda" if torch_f5.cuda.is_available() else ("mps" if IS_MPS_FOR_F5_HANDLER else "cpu") #

            prepared_ref_audio_path_pytorch, _ = _prepare_ref_audio_for_f5( #
                ref_audio_path=effective_ref_audio_path, #
                target_dir=temp_ref_audio_dir_obj, #
                max_duration_s=28,  #
                target_sr=24000 #
            )
            if not prepared_ref_audio_path_pytorch or not prepared_ref_audio_path_pytorch.exists(): #
                logger.error(f"F5-TTS ({display_model_id}): PyTorch - Reference audio preparation failed for {effective_ref_audio_path}.") #
                return #

            transcribed_ref_text, trans_err = _transcribe_ref_audio_with_whisper( #
                audio_path_str=str(prepared_ref_audio_path_pytorch), #
                whisper_model_id_cfg=whisper_model_id, #
                target_device_for_whisper="cpu", #
                language_for_whisper=language_code if language_code != "multilingual" else None, #
                hf_token=os.getenv("HF_TOKEN") #
            )
            if trans_err: #
                logger.warning(f"F5-TTS ({display_model_id}): PyTorch - Whisper transcription failed: {trans_err}. Using placeholder.") #
                transcribed_ref_text = transcribed_ref_text or "Reference audio transcription failed." #

            f5_standard_model_instance = None #
            try:
                init_args_standard = { #
                    # "log_level": "ERROR", # REMOVED THIS LINE
                    "device": str(pytorch_device), #
                    "models_path": str(HF_CACHE_DIR), #
                }
                checkpoint_filename = crisptts_model_config.get("checkpoint_filename") #
                if not checkpoint_filename: #
                    # Only set 'model' if no specific checkpoint is being loaded,
                    # assuming model_repo_id is a name F5TTS api understands for config.
                    init_args_standard["model"] = model_repo_id #
                
                f5_standard_model_instance = F5TTSStandardAPIClass(**init_args_standard) #

                if checkpoint_filename: #
                    from huggingface_hub import hf_hub_download # Local import #
                    chkpt_path_obj = Path(checkpoint_filename) #
                    if not chkpt_path_obj.is_file(): #
                        logger.info(f"F5-TTS ({display_model_id}): Downloading specific checkpoint: {checkpoint_filename}") #
                        checkpoint_to_load = hf_hub_download( #
                            repo_id=model_repo_id, #
                            filename=checkpoint_filename, #
                            token=os.getenv("HF_TOKEN") #
                        )
                    else: #
                        checkpoint_to_load = str(chkpt_path_obj) #
                    
                    logger.info(f"F5-TTS ({display_model_id}): Loading state_dict from checkpoint: {checkpoint_to_load}") #
                    f5_standard_model_instance.model.load_state_dict(torch_f5.load(checkpoint_to_load, map_location=pytorch_device)["state_dict"]) #
                
                f5_standard_model_instance.model.eval() #

                f5_generation_params = { #
                    "steps": int(cli_params.get("steps", crisptts_model_config.get("default_steps", 32))), #
                    "cfg_strength": float(cli_params.get("cfg_strength", crisptts_model_config.get("default_cfg_strength", 2.0))), #
                    "temperature": float(cli_params.get("temperature", 1.0)), #
                }
                logger.debug(f"F5-TTS ({display_model_id}): PyTorch generate args: {f5_generation_params}") #

                audio_out_tensor = f5_standard_model_instance.generate_speech( #
                    text=text, #
                    condition_text=transcribed_ref_text, #
                    condition_wav_path=str(prepared_ref_audio_path_pytorch), #
                    **f5_generation_params #
                )
                
                if effective_output_path_for_handler: #
                    audio_numpy = audio_out_tensor.squeeze().cpu().numpy() #
                    soundfile_f5.write(str(effective_output_path_for_handler), audio_numpy, samplerate=f5_standard_model_instance.sr) #
                
                if effective_output_path_for_handler and effective_output_path_for_handler.exists() and effective_output_path_for_handler.stat().st_size > 100: #
                    synthesis_successful = True #
            finally: #
                del f5_standard_model_instance #

        if synthesis_successful: #
            logger.info(f"F5-TTS ({display_model_id}): Synthesis successful. Output at: {effective_output_path_for_handler}") #
            if play_direct and effective_output_path_for_handler: #
                play_audio(str(effective_output_path_for_handler), is_path=True) #
        else: #
            logger.error(f"F5-TTS ({display_model_id}): Synthesis failed with all attempted backends or output was empty.") #

    except Exception as e_main: #
        logger.error(f"F5-TTS ({display_model_id}): An unhandled error occurred during synthesis: {e_main}", exc_info=True) #
    finally: #
        if temp_ref_audio_dir_obj and temp_ref_audio_dir_obj.exists(): #
            shutil.rmtree(temp_ref_audio_dir_obj, ignore_errors=True) #
            logger.debug(f"F5-TTS ({display_model_id}): Cleaned up temp ref audio directory: {temp_ref_audio_dir_obj}") #

        if TORCH_FOR_F5_HANDLER: #
            if torch_f5.cuda.is_available(): torch_f5.cuda.empty_cache() #
            if IS_MPS_FOR_F5_HANDLER and hasattr(torch_f5.mps, "empty_cache"): #
                try: torch_f5.mps.empty_cache() #
                except Exception: pass #
        gc.collect() #
        logger.info(f"F5-TTS ({display_model_id}): Handler finished.") #