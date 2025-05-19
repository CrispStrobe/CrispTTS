# handlers/nemo_handler.py

import logging
from pathlib import Path
import os
import gc

# Conditional imports
TORCH_AVAILABLE_IN_HANDLER = False
torch_nemo = None
NEMO_TTS_AVAILABLE_IN_HANDLER = False
nemo_tts_models_h = None
torchaudio_module_h = None
HF_HUB_AVAILABLE_IN_HANDLER = False
hf_hub_download_h = None

try:
    import torch as torch_imp
    torch_nemo = torch_imp
    TORCH_AVAILABLE_IN_HANDLER = True
except ImportError: pass

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE_IN_HANDLER = True
    hf_hub_download_h = hf_hub_download
except ImportError: pass


if TORCH_AVAILABLE_IN_HANDLER and HF_HUB_AVAILABLE_IN_HANDLER:
    try:
        import nemo.collections.tts.models as nemo_models_imp
        import torchaudio as ta_imp
        nemo_tts_models_h = nemo_models_imp
        torchaudio_module_h = ta_imp
        NEMO_TTS_AVAILABLE_IN_HANDLER = True
    except ImportError:
        pass # Fail silently, checked in handler

# Use relative imports for project modules
from utils import save_audio, play_audio # Assuming these are in utils.py

logger = logging.getLogger("CrispTTS.handlers.nemo")

def synthesize_with_fastpitch_nemo(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    if not NEMO_TTS_AVAILABLE_IN_HANDLER or not nemo_tts_models_h or not torchaudio_module_h:
        logger.error("NeMo Toolkit or torchaudio not available. Skipping FastPitch synthesis.")
        return
    if not HF_HUB_AVAILABLE_IN_HANDLER or not hf_hub_download_h:
        logger.error("huggingface_hub not available. Cannot download NeMo models. Skipping.")
        return

    spectrogram_repo = model_config.get("spectrogram_model_repo_id")
    spectrogram_file = model_config.get("spectrogram_model_filename")
    vocoder_name = model_config.get("vocoder_model_name")

    if not all([spectrogram_repo, spectrogram_file, vocoder_name]):
        logger.error("NeMo FastPitch: Incomplete model configuration (spectrogram_repo/file or vocoder_name missing).")
        return

    logger.debug(f"FastPitch (NeMo) - Text: '{text[:50]}...', Spectrogram: {spectrogram_repo}/{spectrogram_file}")

    model_cache_dir = Path.home() / ".cache" / "crisptts_nemo_models"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    local_spectrogram_model_path = model_cache_dir / spectrogram_file
    spec_generator, vocoder = None, None

    try:
        if not local_spectrogram_model_path.exists():
            logger.info(f"FastPitch (NeMo) - Downloading spectrogram model {spectrogram_file} from {spectrogram_repo}...")
            hf_hub_download_h(
                repo_id=spectrogram_repo, filename=spectrogram_file,
                local_dir=str(model_cache_dir), local_dir_use_symlinks=False, # NeMo restore needs actual files
                token=os.getenv("HF_TOKEN"), repo_type="model"
            )
        logger.info(f"FastPitch (NeMo) - Spectrogram model available: {local_spectrogram_model_path}")

        logger.info("FastPitch (NeMo) - Loading models...")
        # Ensure strict=False if there are non-critical mismatches in newer NeMo versions
        spec_generator = nemo_tts_models_h.FastPitchModel.restore_from(str(local_spectrogram_model_path), strict=False)
        vocoder = nemo_tts_models_h.HifiGanModel.from_pretrained(model_name=vocoder_name)
        logger.info("FastPitch (NeMo) - Models loaded.")

        # NeMo models might need specific parsing/tokenization
        parsed_text = spec_generator.parse(text)
        
        speaker_id_to_use = int(voice_id_override) if voice_id_override and voice_id_override.isdigit() else model_config.get("default_speaker_id", 0)
        logger.info(f"FastPitch (NeMo) - Using speaker ID: {speaker_id_to_use}")

        device = "cuda" if torch_nemo.cuda.is_available() else ("mps" if (hasattr(torch_nemo.backends, "mps") and torch_nemo.backends.mps.is_available()) else "cpu")
        logger.info(f"FastPitch (NeMo) - Using device: {device}")
        spec_generator.to(device)
        vocoder.to(device)

        logger.info("FastPitch (NeMo) - Generating speech...")
        with torch_nemo.no_grad():
            # The speaker argument might be an integer or a torch.Tensor depending on the model
            # For multi-speaker FastPitch from NGC/HF, it's usually an integer speaker ID.
            spectrogram = spec_generator.generate_spectrogram(tokens=parsed_text, speaker=speaker_id_to_use)
            audio_tensor = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        
        # Ensure audio_tensor is on CPU and 1D or 2D [1, samples] for torchaudio
        audio_for_save = audio_tensor.detach().cpu().squeeze()
        if audio_for_save.ndim == 0: # Handle scalar tensor if it occurs
            audio_for_save = audio_for_save.unsqueeze(0)
        if audio_for_save.ndim == 1: # Ensure 2D for torchaudio (channels, time)
             audio_for_save = audio_for_save.unsqueeze(0)

        sampling_rate = vocoder.cfg.sample_rate

        effective_output_file_wav_str = str(Path(output_file_str).with_suffix(".wav")) if output_file_str else None
        if effective_output_file_wav_str:
            torchaudio_module_h.save(effective_output_file_wav_str, audio_for_save, sample_rate=sampling_rate)
            logger.info(f"FastPitch (NeMo) - Audio saved to {effective_output_file_wav_str}")
        
        if play_direct:
            # Convert to numpy array of float32, then to int16 bytes for play_audio's PCM path
            audio_np_float32 = audio_for_save.squeeze().numpy().astype(np.float32)
            audio_np_int16 = (audio_np_float32 * 32767).astype(np.int16)
            play_audio(audio_np_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=sampling_rate)
            
    except Exception as e:
        logger.error(f"FastPitch (NeMo) - Synthesis failed: {e}", exc_info=True)
    finally:
        del spec_generator, vocoder
        gc.collect()
        if TORCH_AVAILABLE_IN_HANDLER and torch_nemo.cuda.is_available():
            torch_nemo.cuda.empty_cache()