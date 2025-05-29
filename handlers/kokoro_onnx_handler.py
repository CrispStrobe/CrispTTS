# CrispTTS/handlers/kokoro_onnx_handler.py

import logging
import json
from pathlib import Path
import gc
import numpy as np

# CrispTTS utils
from utils import save_audio, play_audio

logger = logging.getLogger("CrispTTS.handlers.kokoro_onnx")

# --- Conditional Import for kokoro-onnx ---
KOKORO_ONNX_AVAILABLE = False
Kokoro_onnx_class = None

try:
    from kokoro_onnx import Kokoro
    Kokoro_onnx_class = Kokoro
    KOKORO_ONNX_AVAILABLE = True
except ImportError:
    logger.info("'kokoro-onnx' library not found. Kokoro ONNX handler will be non-functional. Install with: pip install kokoro-onnx")

def synthesize_with_kokoro_onnx(
    crisptts_model_config: dict,
    text: str,
    voice_id_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    """
    Synthesizes audio using the kokoro-onnx library.
    """
    if not KOKORO_ONNX_AVAILABLE or not Kokoro_onnx_class:
        logger.error("kokoro-onnx library not available. Skipping Kokoro ONNX synthesis.")
        return

    crisptts_specific_model_id = crisptts_model_config.get('crisptts_model_id', 'kokoro_onnx_unknown')
    logger.info(f"Kokoro-ONNX: Starting synthesis for model '{crisptts_specific_model_id}'.")

    # --- 1. Get and Validate Model Paths ---
    onnx_model_path_str = crisptts_model_config.get("onnx_model_path")
    voices_bin_path_str = crisptts_model_config.get("voices_bin_path")

    if not onnx_model_path_str or not voices_bin_path_str:
        logger.error(f"Kokoro-ONNX ({crisptts_specific_model_id}): 'onnx_model_path' or 'voices_bin_path' not defined in config. Skipping.")
        return

    onnx_model_path = Path(onnx_model_path_str).resolve()
    voices_bin_path = Path(voices_bin_path_str).resolve()

    if not onnx_model_path.exists() or not voices_bin_path.exists():
        logger.error(f"Kokoro-ONNX: Model file or voices file not found.")
        logger.error(f"  - Searched for ONNX model at: {onnx_model_path}")
        logger.error(f"  - Searched for voices file at: {voices_bin_path}")
        logger.error("  Please download these from the 'kokoro-onnx' GitHub releases and update the paths in config.py.")
        return

    # --- 2. Initialize Kokoro and Prepare Parameters ---
    kokoro_instance = None
    try:
        logger.info(f"Kokoro-ONNX: Loading model '{onnx_model_path.name}' and voices '{voices_bin_path.name}'.")
        # Initialize the Kokoro object with the model and voices paths
        kokoro_instance = Kokoro_onnx_class(str(onnx_model_path), str(voices_bin_path))

        # Determine voice, language, and speed
        effective_voice = voice_id_override or crisptts_model_config.get("default_voice_id")
        language = crisptts_model_config.get("language")
        speed = float(crisptts_model_config.get("default_speed", 1.0))

        if model_params_override:
            try:
                cli_params = json.loads(model_params_override)
                if "speed" in cli_params:
                    speed = float(cli_params["speed"])
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Kokoro-ONNX: Could not parse 'speed' from --model-params: {model_params_override}")

        if not effective_voice or not language:
            logger.error(f"Kokoro-ONNX ({crisptts_specific_model_id}): A 'voice' and 'language' must be specified. Skipping.")
            return

        # --- 3. Synthesize Audio ---
        logger.info(f"Kokoro-ONNX: Synthesizing with voice='{effective_voice}', lang='{language}', speed={speed:.1f}")
        # Call the create method to generate audio samples
        samples, sample_rate = kokoro_instance.create(
            text, voice=effective_voice, speed=speed, lang=language
        )

        if not isinstance(samples, np.ndarray) or samples.size == 0:
            logger.error("Kokoro-ONNX: Synthesis returned no audio data.")
            return
            
        logger.info(f"Kokoro-ONNX: Synthesis successful. Sample rate: {sample_rate}Hz.")

        # --- 4. Save or Play the Audio ---
        # Convert float samples to int16 bytes for our utils
        audio_int16 = (samples * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        if output_file_str:
            output_path = Path(output_file_str).with_suffix(".wav")
            save_audio(audio_bytes, str(output_path), source_is_path=False, input_format="pcm_s16le", sample_rate=sample_rate)

        if play_direct:
            play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=sample_rate)

    except Exception as e:
        logger.error(f"Kokoro-ONNX ({crisptts_specific_model_id}): Synthesis failed: {e}", exc_info=True)
    finally:
        del kokoro_instance
        gc.collect()