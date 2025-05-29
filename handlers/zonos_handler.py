# CrispTTS/handlers/zonos_handler.py

import logging
import json
from pathlib import Path
import gc
import platform # Add this import

# --- Conditional Imports ---
TORCH_AVAILABLE = False
torch_zonos = None
torchaudio_zonos = None

ZONOS_AVAILABLE = False
Zonos_class, make_cond_dict_func = None, None

logger_init = logging.getLogger("CrispTTS.handlers.zonos.init")

try:
    import torch
    import torchaudio
    torch_zonos = torch
    torchaudio_zonos = torchaudio
    TORCH_AVAILABLE = True
    logger_init.info("PyTorch and Torchaudio loaded for Zonos handler.")
except ImportError:
    logger_init.warning("PyTorch or Torchaudio not found. Zonos handler will be non-functional.")

if TORCH_AVAILABLE:
    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        Zonos_class = Zonos
        make_cond_dict_func = make_cond_dict
        ZONOS_AVAILABLE = True
        logger_init.info("Zonos TTS library imported successfully.")
    except ImportError:
        logger_init.warning("'zonos-tts' library not found. Zonos handler will be non-functional. Install with: (see Zonos repo for instructions)") # Updated install advice

from utils import save_audio, play_audio

logger = logging.getLogger("CrispTTS.handlers.zonos")

# --- Globals for Caching (to avoid reloading on every call) ---
CACHED_ZONOS_MODEL = None
CACHED_MODEL_ID = None
CACHED_SPEAKER_EMBEDDING = None
CACHED_SPEAKER_PATH = None

def synthesize_with_zonos(
    crisptts_model_config: dict,
    text: str,
    voice_id_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    """
    Synthesizes audio using the Zonos TTS library from Zyphra.
    """
    global CACHED_ZONOS_MODEL, CACHED_MODEL_ID, CACHED_SPEAKER_EMBEDDING, CACHED_SPEAKER_PATH

    if not ZONOS_AVAILABLE or not TORCH_AVAILABLE:
        logger.error("Zonos handler dependencies (zonos library, torch, torchaudio) not available. Skipping.")
        return

    model_repo_id = crisptts_model_config.get("model_repo_id")
    if not model_repo_id:
        logger.error("Zonos: 'model_repo_id' not in config. Skipping.")
        return

    try:
        device = "cuda" if torch_zonos.cuda.is_available() else ("mps" if hasattr(torch_zonos.backends, "mps") and torch_zonos.backends.mps.is_available() else "cpu")

        if CACHED_MODEL_ID != model_repo_id or CACHED_ZONOS_MODEL is None:
            logger.info(f"Zonos: Loading model '{model_repo_id}' to device '{device}'. This may take a moment...")
            if CACHED_ZONOS_MODEL is not None:
                del CACHED_ZONOS_MODEL # Try to free memory before loading new model
                gc.collect()
                if device == "cuda": torch_zonos.cuda.empty_cache()
                elif device == "mps" and hasattr(torch_zonos.mps, "empty_cache"): torch_zonos.mps.empty_cache()


            CACHED_ZONOS_MODEL = Zonos_class.from_pretrained(model_repo_id, device=device)
            CACHED_ZONOS_MODEL.requires_grad_(False).eval() # Ensure model is in eval mode
            CACHED_MODEL_ID = model_repo_id
            logger.info(f"Zonos: Model '{model_repo_id}' loaded successfully onto {CACHED_ZONOS_MODEL.device}.")
        else:
            logger.info(f"Zonos: Using cached model '{model_repo_id}'.")
            if str(CACHED_ZONOS_MODEL.device) != device: # Ensure cached model is on the correct current device
                logger.info(f"Zonos: Moving cached model from {CACHED_ZONOS_MODEL.device} to {device}.")
                CACHED_ZONOS_MODEL.to(device)
                if device == "cuda": torch_zonos.cuda.empty_cache() # Clean old device memory if possible
                elif device == "mps" and hasattr(torch_zonos.mps, "empty_cache"): torch_zonos.mps.empty_cache()


        ref_audio_path_str = voice_id_override or crisptts_model_config.get("default_voice_id")
        if not ref_audio_path_str:
            logger.error("Zonos requires a reference audio file. Please specify via --german-voice-id or in config.")
            return
        
        ref_audio_path = Path(ref_audio_path_str).resolve()
        if not ref_audio_path.exists():
            logger.error(f"Zonos: Reference audio file not found at '{ref_audio_path}'.")
            return

        if CACHED_SPEAKER_PATH != str(ref_audio_path) or CACHED_SPEAKER_EMBEDDING is None:
            logger.info(f"Zonos: Creating new speaker embedding from '{ref_audio_path.name}'.")
            wav, sr = torchaudio_zonos.load(ref_audio_path)
            CACHED_SPEAKER_EMBEDDING = CACHED_ZONOS_MODEL.make_speaker_embedding(wav, sr)
            # Zonos example uses bfloat16 for speaker embedding if available, otherwise float32.
            # Using float32 for broader compatibility, adjust if bfloat16 is preferred and available.
            target_speaker_dtype = torch_zonos.bfloat16 if hasattr(torch_zonos, 'bfloat16') and device != 'cpu' else torch_zonos.float32
            CACHED_SPEAKER_EMBEDDING = CACHED_SPEAKER_EMBEDDING.to(device, dtype=target_speaker_dtype)
            CACHED_SPEAKER_PATH = str(ref_audio_path)
            logger.info(f"Zonos: Speaker embedding created on {CACHED_SPEAKER_EMBEDDING.device} with dtype {CACHED_SPEAKER_EMBEDDING.dtype}.")
        else:
            logger.info(f"Zonos: Using cached speaker embedding from '{ref_audio_path.name}'.")
            if str(CACHED_SPEAKER_EMBEDDING.device) != device:
                 CACHED_SPEAKER_EMBEDDING = CACHED_SPEAKER_EMBEDDING.to(device)


        language = crisptts_model_config.get("language", "en-us") # Default from config
        cond_dict_args = {"text": text, "speaker": CACHED_SPEAKER_EMBEDDING, "language": language, "device": device}
        
        # Default conditioning parameters (can be overridden by model_params_override)
        # These are based on common parameters seen in Zonos's Gradio UI and examples.
        # Users can provide these via --model-params
        default_conditioning_params = {
            # "emotion": [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], # Example: mostly happy, a bit neutral
            # "vqscore_8": [0.78] * 8, # Example from Gradio
            "fmax": 24000.0,
            "pitch_std": 45.0,
            "speaking_rate": 15.0,
            "dnsmos_ovrl": 4.0,
            "speaker_noised": False
        }

        # Apply defaults first, then CLI overrides
        for key, val in default_conditioning_params.items():
            if key not in cond_dict_args: # Only if not already set by required args
                if isinstance(val, list): cond_dict_args[key] = torch_zonos.tensor(val, device=device)
                else: cond_dict_args[key] = val
        
        if model_params_override:
            try:
                cli_params = json.loads(model_params_override)
                allowed_params = list(default_conditioning_params.keys()) + ["emotion", "vqscore_8"] # ensure emotion/vqscore can be set fully
                for key, value in cli_params.items():
                    if key in allowed_params:
                        if isinstance(value, list): # For 'emotion' and 'vqscore_8'
                            cond_dict_args[key] = torch_zonos.tensor(value, device=device)
                        else: # For float/bool params
                            cond_dict_args[key] = value 
                logger.info(f"Zonos: Applied custom model parameters via --model-params: {list(cli_params.keys())}")
            except json.JSONDecodeError:
                logger.warning(f"Zonos: Could not parse --model-params JSON: {model_params_override}")

        final_cond_dict = make_cond_dict_func(**cond_dict_args)
        conditioning = CACHED_ZONOS_MODEL.prepare_conditioning(final_cond_dict)
        
        # Determine if torch.compile should be disabled
        # Disable on CPU or MPS (Mac) as Triton is primarily for NVIDIA GPUs
        # or if the user explicitly passes "disable_torch_compile": true in model_params
        disable_compile_flag = False
        if device in ["cpu", "mps"]:
            disable_compile_flag = True
            logger.info(f"Zonos: Disabling torch.compile as device is '{device}'.")
        if model_params_override: # Check if user wants to override this
            try:
                cli_p = json.loads(model_params_override)
                if "disable_torch_compile" in cli_p:
                    disable_compile_flag = bool(cli_p["disable_torch_compile"])
                    logger.info(f"Zonos: User override for disable_torch_compile: {disable_compile_flag}")
            except: pass


        logger.info("Zonos: Generating audio codes...")
        with torch_zonos.no_grad():
            codes = CACHED_ZONOS_MODEL.generate(
                conditioning,
                disable_torch_compile=disable_compile_flag # Pass the flag here
            )
            logger.info("Zonos: Decoding codes to waveform...")
            wav_tensor = CACHED_ZONOS_MODEL.autoencoder.decode(codes).cpu()

        sampling_rate = CACHED_ZONOS_MODEL.autoencoder.sampling_rate #
        logger.info(f"Zonos: Synthesis successful. Sample rate: {sampling_rate}Hz.")

        # --- MODIFICATION START: Handle potential batch dimension ---
        if wav_tensor.ndim == 3 and wav_tensor.shape[0] == 1:
            # Assuming batch size is 1, squeeze out the batch dimension
            # or take the first element: wav_tensor_to_save = wav_tensor[0]
            # For torchaudio.save, it expects [channels, samples] or [samples]
            # If Zonos returns [1, channels, samples], then wav_tensor[0] is [channels, samples]
            # If Zonos returns [1, samples] (mono), then wav_tensor[0] is [samples]
            wav_tensor_processed = wav_tensor[0] 
            logger.debug(f"Zonos: Processed wav_tensor shape for saving/playing: {wav_tensor_processed.shape}")
        elif wav_tensor.ndim == 2 or wav_tensor.ndim == 1:
            wav_tensor_processed = wav_tensor # Already in 2D or 1D format
            logger.debug(f"Zonos: wav_tensor already 2D/1D, shape: {wav_tensor_processed.shape}")
        else:
            logger.error(f"Zonos: wav_tensor has unexpected dimensions: {wav_tensor.ndim}. Shape: {wav_tensor.shape}")
            return
        # --- MODIFICATION END ---

        # --- 5. Save or Play Output ---
        if output_file_str:
            output_path = Path(output_file_str).with_suffix(".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Use the processed tensor for saving
            torchaudio_zonos.save(str(output_path), wav_tensor_processed, sampling_rate) #
            logger.info(f"Zonos: Audio saved to {output_path}")

        if play_direct:
            # play_audio utility expects a 1D tensor or path.
            # wav_tensor_processed could be [channels, samples] or [samples].
            # If it's [channels, samples], we might need to take the first channel or average.
            # For simplicity, assuming mono or taking the first channel if stereo.
            audio_for_play = wav_tensor_processed
            if audio_for_play.ndim == 2 and audio_for_play.shape[0] > 0 : # If [channels, samples]
                audio_for_play = audio_for_play[0] # Take the first channel
            
            play_audio(audio_for_play.squeeze(), is_path=False, sample_rate=sampling_rate)

    except Exception as e:
        logger.error(f"Zonos: Synthesis failed: {e}", exc_info=True)
    finally:
        # Do not delete cached model and speaker embedding to allow for faster subsequent runs.
        # Only clear GPU cache if CUDA was used.
        if device == "cuda" and TORCH_AVAILABLE and torch_zonos.cuda.is_available():
            torch_zonos.cuda.empty_cache()
        elif device == "mps" and TORCH_AVAILABLE and hasattr(torch_zonos.backends, "mps") and torch_zonos.backends.mps.is_available() and hasattr(torch_zonos.mps, "empty_cache"):
            torch_zonos.mps.empty_cache()
        gc.collect()