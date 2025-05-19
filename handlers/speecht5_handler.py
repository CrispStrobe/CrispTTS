# handlers/speecht5_handler.py

import logging
from pathlib import Path
import gc

# Conditional imports
TORCH_AVAILABLE_IN_HANDLER = False
torch_s5 = None
SpeechT5Processor_s5, SpeechT5ForTextToSpeech_s5, SpeechT5HifiGan_s5, load_dataset_s5 = (None,) * 4
SOUNDFILE_AVAILABLE_IN_HANDLER = False
sf_s5 = None

try:
    import torch as torch_imp
    torch_s5 = torch_imp
    TORCH_AVAILABLE_IN_HANDLER = True
except ImportError: pass

try:
    import soundfile as sf_imp
    sf_s5 = sf_imp
    SOUNDFILE_AVAILABLE_IN_HANDLER = True
except ImportError: pass

if TORCH_AVAILABLE_IN_HANDLER and SOUNDFILE_AVAILABLE_IN_HANDLER:
    try:
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset
        SpeechT5Processor_s5 = SpeechT5Processor
        SpeechT5ForTextToSpeech_s5 = SpeechT5ForTextToSpeech
        SpeechT5HifiGan_s5 = SpeechT5HifiGan
        load_dataset_s5 = load_dataset
    except ImportError:
        logger_s5_init = logging.getLogger("CrispTTS.handlers.speecht5_init") # Temp logger for init phase
        logger_s5_init.info("SpeechT5: 'transformers' or 'datasets' not installed. Handler will be non-functional.")


# Use relative imports for project modules
from utils import save_audio, play_audio # Assuming these are in utils.py

logger = logging.getLogger("CrispTTS.handlers.speecht5")

def synthesize_with_speecht5_transformers(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    if not TORCH_AVAILABLE_IN_HANDLER or not SpeechT5Processor_s5 or not SOUNDFILE_AVAILABLE_IN_HANDLER:
        logger.error("SpeechT5 dependencies (torch, transformers, datasets, soundfile) not fully available. Skipping.")
        return

    model_id = model_config.get("model_id")
    vocoder_id = model_config.get("vocoder_id")
    speaker_embeddings_repo = model_config.get("speaker_embeddings_repo")
    default_speaker_idx = model_config.get("default_speaker_embedding_index")

    if not all([model_id, vocoder_id, speaker_embeddings_repo, default_speaker_idx is not None]):
        logger.error("SpeechT5: Incomplete model configuration (model_id, vocoder_id, speaker_embeddings_repo, or default_speaker_embedding_index missing).")
        return

    logger.debug(f"SpeechT5 - Text: '{text[:50]}...', Model: {model_id}")

    processor, model, vocoder, speaker_embedding_to_use, inputs = None, None, None, None, None
    try:
        processor = SpeechT5Processor_s5.from_pretrained(model_id)
        model = SpeechT5ForTextToSpeech_s5.from_pretrained(model_id)
        vocoder = SpeechT5HifiGan_s5.from_pretrained(vocoder_id)
        logger.info("SpeechT5 - Models and processor loaded.")

        inputs = processor(text=text, return_tensors="pt")
        
        if voice_id_override:
            if Path(voice_id_override).is_file() and voice_id_override.lower().endswith((".pt", ".pth")):
                try: speaker_embedding_to_use = torch_s5.load(voice_id_override, map_location="cpu").unsqueeze(0); logger.info(f"SpeechT5 - Loaded custom speaker embedding: {voice_id_override}")
                except Exception as e: logger.warning(f"SpeechT5 - Failed to load custom speaker embedding '{voice_id_override}': {e}. Trying default/index.")
            if not speaker_embedding_to_use and voice_id_override.isdigit():
                try:
                    speaker_idx = int(voice_id_override)
                    embeddings_dataset = load_dataset_s5(speaker_embeddings_repo, split="validation", trust_remote_code=True)
                    speaker_embedding_to_use = torch_s5.tensor(embeddings_dataset[speaker_idx]["xvector"]).unsqueeze(0)
                    logger.info(f"SpeechT5 - Using speaker embedding index {speaker_idx} from {speaker_embeddings_repo}.")
                except Exception as e: logger.warning(f"SpeechT5 - Failed to load embedding index {voice_id_override}: {e}. Using default.")
        
        if speaker_embedding_to_use is None:
            try:
                embeddings_dataset = load_dataset_s5(speaker_embeddings_repo, split="validation", trust_remote_code=True)
                speaker_embedding_to_use = torch_s5.tensor(embeddings_dataset[default_speaker_idx]["xvector"]).unsqueeze(0)
                logger.info(f"SpeechT5 - Using default speaker index {default_speaker_idx}.")
            except Exception as e: logger.error(f"SpeechT5 - Failed to load default speaker embedding: {e}"); return

        device = "cuda" if torch_s5.cuda.is_available() else ("mps" if (hasattr(torch_s5.backends, "mps") and torch_s5.backends.mps.is_available()) else "cpu")
        logger.info(f"SpeechT5 - Using device: {device}")
        model.to(device); vocoder.to(device)
        input_ids = inputs["input_ids"].to(device)
        speaker_embedding_to_use = speaker_embedding_to_use.to(device)

        logger.info("SpeechT5 - Generating speech...")
        with torch_s5.no_grad():
            speech_tensor = model.generate_speech(input_ids, speaker_embedding_to_use, vocoder=vocoder)
        
        # Ensure audio is 1D float32 for soundfile/pydub processing
        audio_array_np = speech_tensor.cpu().numpy().astype(np.float32)
        if audio_array_np.ndim > 1: # Flatten if necessary, assuming mono
            audio_array_np = audio_array_np.squeeze()
            
        sampling_rate = 16000 # SpeechT5 default

        effective_output_file_wav_str = str(Path(output_file_str).with_suffix(".wav")) if output_file_str else None
        if effective_output_file_wav_str:
            # sf.write expects data to be float or int, matching subtype. Default is float.
            sf_s5.write(effective_output_file_wav_str, audio_array_np, samplerate=sampling_rate)
            logger.info(f"SpeechT5 - Audio saved to {effective_output_file_wav_str}")
        if play_direct:
            # play_audio utility expects bytes for PCM, so convert float numpy array to int16 bytes
            audio_int16 = (audio_array_np * 32767).astype(np.int16)
            play_audio(audio_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=sampling_rate)
            
    except Exception as e: logger.error(f"SpeechT5 - Synthesis failed: {e}", exc_info=True)
    finally:
        del processor, model, vocoder, speaker_embedding_to_use, inputs; gc.collect()
        if TORCH_AVAILABLE_IN_HANDLER and torch_s5.cuda.is_available(): torch_s5.cuda.empty_cache()