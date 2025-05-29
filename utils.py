# utils.py

import os
import sys
import io
import json
import wave
import shutil
from pathlib import Path
import logging
import numpy as np
import tempfile

# --- Project-Specific Imports (from other planned modules) ---
# These will be available once config.py is created and populated.
try:
    from config import ORPHEUS_SAMPLE_RATE as DEFAULT_ORPHEUS_SR
    from config import ORPHEUS_DEFAULT_VOICE, ORPHEUS_GERMAN_VOICES, SAUERKRAUT_VOICES, ORPHEUS_AVAILABLE_VOICES_BASE
except ImportError:
    # Fallbacks if config.py is not yet available or fully populated during initial setup
    logger = logging.getLogger("CrispTTS.utils_early") # Use a temporary logger if main one not set up
    logger.warning("config.py not found or incomplete; using hardcoded fallbacks for some utils constants.")
    DEFAULT_ORPHEUS_SR = 24000
    ORPHEUS_DEFAULT_VOICE = "jana"
    ORPHEUS_GERMAN_VOICES = ["jana", "thomas", "max"]
    SAUERKRAUT_VOICES = ["Tom", "Anna", "Max", "Lena"]
    ORPHEUS_AVAILABLE_VOICES_BASE = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


# --- Conditional Imports for Optional Features ---
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
try:
    import markdown
    try:
        from markdown.extensions.wikilinks import WikiLinkExtension
    except ImportError:
        WikiLinkExtension = None
except ImportError:
    markdown = None
    WikiLinkExtension = None
try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None
try:
    from ebooklib import epub
except ImportError:
    epub = None

# Audio libraries
PYDUB_AVAILABLE = False
SOUNDDEVICE_AVAILABLE = False
SOUNDFILE_AVAILABLE = False
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    PYDUB_AVAILABLE = True
except ImportError:
    pass
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    pass
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    pass

# --- Logger Setup ---
# Assumes logging is configured in main.py. If utils is imported before, this logger might not have full config.
logger = logging.getLogger("CrispTTS.utils")

# --- User's Orpheus Decoder (Import with Fallback) ---
try:
    from decoder import convert_to_audio as user_orpheus_decoder
    logger.info("Successfully imported 'convert_to_audio' from user's decoder.py for Orpheus utilities.")
except ImportError:
    logger.warning("'decoder.py' not found or 'convert_to_audio' not in it. "
                   "Orpheus audio decoding in utils will use a placeholder (no audio).")
    def user_orpheus_decoder_placeholder(multiframe, count):
        logger.warning("Using PLACEHOLDER orpheus_decoder_convert_to_audio. NO ACTUAL AUDIO WILL BE GENERATED.")
        return b'' # Return empty bytes
    user_orpheus_decoder = user_orpheus_decoder_placeholder
except Exception as e:
    logger.error(f"Importing 'convert_to_audio' from decoder.py failed: {e}. Using placeholder.")
    def user_orpheus_decoder_placeholder_exc(multiframe, count):
        logger.warning("Using PLACEHOLDER orpheus_decoder_convert_to_audio due to import error. NO AUDIO.")
        return b''
    user_orpheus_decoder = user_orpheus_decoder_placeholder_exc


# --- SuppressOutput Context Manager ---
class SuppressOutput:
    def __init__(self, suppress_stdout=True, suppress_stderr=False, use_stringio=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.use_stringio = use_stringio
        self._old_stdout = None
        self._old_stderr = None
        self._capture_buffer = None
        self._devnull_fp = None

    def __enter__(self):
        if self.use_stringio:
            self._capture_buffer = io.StringIO()
            if self.suppress_stdout:
                self._old_stdout = sys.stdout
                sys.stdout = self._capture_buffer
            if self.suppress_stderr:
                self._old_stderr = sys.stderr
                sys.stderr = self._capture_buffer
        else:
            try:
                self._devnull_fp = open(os.devnull, 'w', encoding='utf-8')
                if self.suppress_stdout:
                    self._old_stdout = sys.stdout
                    sys.stdout = self._devnull_fp
                if self.suppress_stderr:
                    self._old_stderr = sys.stderr
                    sys.stderr = self._devnull_fp
            except OSError as e:
                logger.warning(f"SuppressOutput: Failed to open os.devnull: {e}. Output will not be suppressed.")
                self._devnull_fp = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout and self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self.suppress_stderr and self._old_stderr is not None:
            sys.stderr = self._old_stderr

        if self._capture_buffer:
            self.captured_output = self._capture_buffer.getvalue()
            self._capture_buffer.close()
        if self._devnull_fp:
            self._devnull_fp.close()
        return False # Do not suppress exceptions


# --- Text Extraction Functions ---
def extract_text_from_txt(filepath: Path) -> str | None:
    try:
        return filepath.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading TXT file {filepath}: {e}")
        return None

def extract_text_from_md(filepath: Path) -> str | None:
    if not markdown: logger.error("Markdown library not available for .md extraction."); return None
    if not BeautifulSoup: logger.error("BeautifulSoup4 library not available for .md extraction (HTML parsing)."); return None
    try:
        md_text = filepath.read_text(encoding='utf-8')
        extensions = [WikiLinkExtension(base_url='', end_url='')] if WikiLinkExtension else [] # Basic wikilink
        html = markdown.markdown(md_text, extensions=extensions)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logger.error(f"Error reading Markdown file {filepath}: {e}")
        return None

def extract_text_from_html(filepath: Path) -> str | None:
    if not BeautifulSoup: logger.error("BeautifulSoup4 library not available for .html extraction."); return None
    try:
        html_content = filepath.read_text(encoding='utf-8')
        soup = BeautifulSoup(html_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logger.error(f"Error reading HTML file {filepath}: {e}")
        return None

def extract_text_from_pdf(filepath: Path) -> str | None:
    if not pdfium: logger.error("pypdfium2 library not available for .pdf extraction."); return None
    try:
        text_content = []
        doc = pdfium.PdfDocument(filepath) # type: ignore
        for i in range(len(doc)):
            page = doc.get_page(i)
            textpage = page.get_textpage()
            text_content.append(textpage.get_text_range())
            textpage.close()
            page.close()
        doc.close()
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Error reading PDF file {filepath}: {e}")
        return None

def extract_text_from_epub(filepath: Path) -> str | None:
    if not epub: logger.error("EbookLib library not available for .epub extraction."); return None
    if not BeautifulSoup: logger.error("BeautifulSoup4 library not available for .epub extraction (content parsing)."); return None
    try:
        book = epub.read_epub(filepath) # type: ignore
        text_parts = []
        for item in book.get_items_of_type(epub.ITEM_DOCUMENT): # type: ignore
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text_parts.append(soup.get_text(separator=' ', strip=True))
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error reading EPUB file {filepath}: {e}")
        return None

def get_text_from_input(input_text_direct: str | None, input_file_path_str: str | None) -> str | None:
    if input_text_direct:
        return input_text_direct
    if input_file_path_str:
        filepath = Path(input_file_path_str)
        if not filepath.exists():
            logger.error(f"Input file not found: {input_file_path_str}")
            return None
        ext = filepath.suffix.lower()
        if ext == '.txt': return extract_text_from_txt(filepath)
        elif ext == '.md': return extract_text_from_md(filepath)
        elif ext in ['.html', '.htm']: return extract_text_from_html(filepath)
        elif ext == '.pdf': return extract_text_from_pdf(filepath)
        elif ext == '.epub': return extract_text_from_epub(filepath)
        else:
            logger.error(f"Unsupported file type: {ext} for file {filepath}")
            return None
    logger.info("No text input provided (neither direct text nor file).")
    return None


# --- Audio Handling Utilities ---
def save_audio(audio_data_or_path, output_filepath_str: str, source_is_path=False, input_format=None, sample_rate=None):
    output_filepath = Path(output_filepath_str)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    target_format = output_filepath.suffix[1:].lower() if output_filepath.suffix else "mp3"

    if not audio_data_or_path and not source_is_path:
        logger.warning(f"No audio data provided to save_audio for {output_filepath}.")
        return

    try:
        if source_is_path:
            source_path = Path(audio_data_or_path)
            if not source_path.exists(): logger.error(f"Source audio path does not exist: {source_path}"); return
            if source_path.suffix.lower() == f".{target_format}": shutil.copyfile(source_path, output_filepath)
            elif PYDUB_AVAILABLE: AudioSegment.from_file(source_path).export(output_filepath, format=target_format)
            elif SOUNDFILE_AVAILABLE and target_format == "wav": data, sr = sf.read(source_path); sf.write(str(output_filepath), data, sr) # Ensure str for sf.write
            else: logger.error(f"Cannot convert {source_path.suffix} to {target_format}. Pydub needed."); return
        else: # audio_data_or_path contains audio bytes
            from io import BytesIO
            fmt = input_format if input_format else "wav"
            if fmt == "wav_bytes": fmt = "wav"

            if fmt == "pcm_s16le":
                current_sample_rate = sample_rate or DEFAULT_ORPHEUS_SR
                if PYDUB_AVAILABLE:
                    audio_segment = AudioSegment(data=audio_data_or_path, sample_width=2, frame_rate=current_sample_rate, channels=1)
                    audio_segment.export(output_filepath, format=target_format)
                elif SOUNDFILE_AVAILABLE and target_format == "wav":
                    audio_np_array = np.frombuffer(audio_data_or_path, dtype=np.int16)
                    sf.write(str(output_filepath), audio_np_array, current_sample_rate) # Ensure str
                else: logger.error("Cannot save raw PCM; Pydub or SoundFile (for WAV target) is required."); return
            elif PYDUB_AVAILABLE:
                AudioSegment.from_file(BytesIO(audio_data_or_path), format=fmt).export(output_filepath, format=target_format)
            elif SOUNDFILE_AVAILABLE and fmt == "wav" and target_format == "wav":
                data, sr_read = sf.read(BytesIO(audio_data_or_path))
                sf.write(str(output_filepath), data, sr_read if sr_read else (sample_rate or DEFAULT_ORPHEUS_SR))
            else: logger.error(f"Cannot save audio bytes of format '{fmt}' to '{target_format}'. Pydub or Soundfile (for WAV) needed."); return
        logger.info(f"Audio saved to {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving audio to {output_filepath}: {e}", exc_info=True)

def play_audio(audio_path_or_data, is_path=True, input_format=None, sample_rate=None):
    if is_path:
        if not PYDUB_AVAILABLE: logger.error("Pydub not available for file playback."); return
        audio_file_path = Path(audio_path_or_data)
        if not audio_file_path.exists(): logger.error(f"Audio file for playback does not exist: {audio_file_path}"); return
        try:
            logger.info(f"Playing audio from file: {audio_file_path}..."); sound = AudioSegment.from_file(audio_file_path); pydub_play(sound); logger.info("Playback finished.")
        except Exception as e: logger.error(f"Error playing audio file {audio_file_path}: {e}", exc_info=True)
    else: # audio_path_or_data contains audio bytes
        fmt = input_format if input_format else "wav"
        if fmt == "wav_bytes": fmt = "wav"

        if fmt == "pcm_s16le":
            current_sample_rate = sample_rate or DEFAULT_ORPHEUS_SR
            if not SOUNDDEVICE_AVAILABLE: logger.error("Sounddevice not available for raw PCM playback."); return
            try:
                logger.info(f"Playing raw PCM audio (sample rate: {current_sample_rate})..."); audio_np = np.frombuffer(audio_path_or_data, dtype=np.int16); sd.play(audio_np, samplerate=current_sample_rate, blocking=True); sd.wait(); logger.info("Playback finished.")
            except Exception as e: logger.error(f"Error playing PCM audio with sounddevice: {e}", exc_info=True)
        elif PYDUB_AVAILABLE:
            from io import BytesIO
            try:
                logger.info(f"Playing audio bytes (format: {fmt})..."); sound = AudioSegment.from_file(BytesIO(audio_path_or_data), format=fmt); pydub_play(sound); logger.info("Playback finished.")
            except Exception as e: logger.error(f"Error playing audio bytes with pydub: {e}", exc_info=True)
        else:
            logger.error(f"Cannot play audio bytes of format '{fmt}'. Pydub or Sounddevice (for PCM) needed.")


# --- Orpheus Specific Utilities ---
def orpheus_format_prompt(prompt_text, voice_name, available_voices_list):
    """Formats prompt for Orpheus models, using globally defined voice lists if needed."""
    voice_name_lower = voice_name.lower()
    found_match = None
    
    # Ensure available_voices_list is actually a list
    if not isinstance(available_voices_list, list):
        logger.warning(f"Orpheus: available_voices_list is not a list ({type(available_voices_list)}). Using default fallback.")
        available_voices_list = [] # Prevent error, will lead to fallback

    for v_avail in available_voices_list:
        if isinstance(v_avail, str) and v_avail.lower() == voice_name_lower:
            found_match = v_avail
            break
    
    chosen_voice = voice_name
    if found_match:
        if voice_name != found_match:
            logger.info(f"Orpheus voice '{voice_name}' matched (case-insensitively) as '{found_match}'.")
        chosen_voice = found_match
    elif voice_name not in available_voices_list: # Also check case-sensitively if not found case-insensitively
        fallback_voice = available_voices_list[0] if available_voices_list else ORPHEUS_DEFAULT_VOICE
        logger.warning(f"Orpheus voice '{voice_name}' not in available list {available_voices_list}. Using fallback '{fallback_voice}'.")
        chosen_voice = fallback_voice
    return f"<|audio|>{chosen_voice}: {prompt_text}<|eot_id|>"

def orpheus_turn_token_into_id(token_string: str, index: int) -> int | None:
    """Converts Orpheus custom token string to its ID."""
    token_string = token_string.strip()
    CUSTOM_TOKEN_PREFIX = "<custom_token_"
    if token_string.startswith(CUSTOM_TOKEN_PREFIX) and token_string.endswith(">"):
        try:
            return int(token_string[len(CUSTOM_TOKEN_PREFIX):-1]) - 10 - ((index % 7) * 4096)
        except ValueError:
            logger.warning(f"Could not parse token_id from Orpheus token: {token_string}")
            return None
    return None

def _orpheus_master_token_processor_and_decoder(raw_token_text_generator, output_file_wav_str=None):
    logger.debug("Orpheus Master Processor - Starting token processing and decoding.")
    all_audio_data = bytearray()
    wav_file = None
    
    if output_file_wav_str:
        output_p = Path(output_file_wav_str)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav_file = wave.open(str(output_p), "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16-bit
            wav_file.setframerate(DEFAULT_ORPHEUS_SR)
        except Exception as e:
            logger.error(f"Orpheus Master Processor - WAV open failed for {output_file_wav_str}: {e}")
            wav_file = None

    token_buffer = []
    token_count_for_decoder = 0
    audio_segment_count = 0
    id_conversion_idx = 0

    try:
        for text_chunk in raw_token_text_generator:
            if not isinstance(text_chunk, str):
                logger.warning(f"Orpheus Processor: Received non-string chunk: {type(text_chunk)}")
                continue
            
            current_pos = 0
            while True:
                start_custom = text_chunk.find("<custom_token_", current_pos)
                if start_custom == -1: break
                end_custom = text_chunk.find(">", start_custom)
                if end_custom == -1:
                    logger.debug(f"Orpheus Processor: Incomplete token at end of chunk: {text_chunk[start_custom:]}")
                    break 
                
                custom_token_str = text_chunk[start_custom : end_custom + 1]
                token_id = orpheus_turn_token_into_id(custom_token_str, id_conversion_idx)
                
                if token_id is not None and token_id > 0:
                    token_buffer.append(token_id)
                    token_count_for_decoder += 1
                    id_conversion_idx += 1
                    
                    if token_count_for_decoder % 7 == 0 and token_count_for_decoder >= 28:
                        buffer_to_process = token_buffer[-28:]
                        if len(buffer_to_process) == 28:
                            try:
                                audio_chunk_bytes = user_orpheus_decoder(buffer_to_process, token_count_for_decoder)
                                if audio_chunk_bytes and isinstance(audio_chunk_bytes, bytes) and len(audio_chunk_bytes) > 0:
                                    all_audio_data.extend(audio_chunk_bytes)
                                    audio_segment_count += 1
                                    if wav_file:
                                        try: wav_file.writeframes(audio_chunk_bytes)
                                        except Exception as e_write: logger.warning(f"Orpheus Processor - Failed to write audio frames: {e_write}")
                            except Exception as e_decode:
                                logger.error(f"Orpheus decoder error: {e_decode}", exc_info=True)
                current_pos = end_custom + 1
    except Exception as e_gen_loop:
        logger.error(f"Orpheus Processor - Error in token generation loop: {e_gen_loop}", exc_info=True)
    finally:
        if wav_file:
            try: wav_file.close()
            except Exception as e_close: logger.warning(f"Orpheus Processor - Failed to close WAV file {output_file_wav_str}: {e_close}")
        
    if not all_audio_data:
        logger.warning("Orpheus Processor - No audio data was generated/decoded.")
        if output_file_wav_str and Path(output_file_wav_str).exists(): # If an empty file was created
             Path(output_file_wav_str).unlink(missing_ok=True)


    duration_s = (len(all_audio_data) / (2 * DEFAULT_ORPHEUS_SR)) if DEFAULT_ORPHEUS_SR > 0 else 0
    logger.info(f"Orpheus Processor - Processed {audio_segment_count} audio segments. Total duration: {duration_s:.2f}s.")
    return bytes(all_audio_data)


# --- OuteTTS Specific Utilities ---
def _prepare_oute_speaker_ref(speaker_ref_path_str: str, model_id_for_log: str ="oute"):
    """
    Validates and optionally trims the OuteTTS speaker reference WAV file.
    Returns a tuple: (Path object to use for speaker creation, string path of temp file to delete or None).
    """
    speaker_ref_path_to_use = None
    temp_trimmed_audio_path_to_delete = None # Path of temp file if trimming occurs

    if not isinstance(speaker_ref_path_str, str) or not speaker_ref_path_str.strip():
        logger.warning(f"{model_id_for_log} - Speaker reference path is empty or not a string: '{speaker_ref_path_str}'. Attempting fallback to ./german.wav.")
        speaker_ref_path_str = "./german.wav" # Force fallback check

    # Handle placeholder for default and try ./german.wav explicitly if path seems like a placeholder or became ./german.wav
    if "path/to/your" in speaker_ref_path_str or speaker_ref_path_str == "./german.wav":
        german_wav_fallback = Path("./german.wav").resolve() # Resolve to make existence check robust
        if german_wav_fallback.exists() and german_wav_fallback.is_file():
            logger.info(f"{model_id_for_log} - Found '{german_wav_fallback}', using it as reference.")
            speaker_ref_path_input = german_wav_fallback
        else:
            logger.error(f"{model_id_for_log} - Default speaker placeholder used or './german.wav' fallback specified, but '{german_wav_fallback}' not found. Please provide a valid --german-voice-id (WAV path).")
            return None, None
    else:
        speaker_ref_path_input = Path(speaker_ref_path_str).resolve()

    if not speaker_ref_path_input.exists() or not speaker_ref_path_input.is_file() or speaker_ref_path_input.suffix.lower() != '.wav':
        logger.error(f"{model_id_for_log} - Speaker reference path '{speaker_ref_path_input}' is not a valid existing .wav file.")
        return None, None

    # At this point, speaker_ref_path_input should be a valid Path object to an existing .wav file
    if not PYDUB_AVAILABLE:
        logger.warning(f"{model_id_for_log} - Pydub not available. Cannot check/trim reference audio length. Using as is: '{speaker_ref_path_input}'. Max length for OuteTTS is ~14.5s.")
        return speaker_ref_path_input, None # Return original path, no temp file created

    try:
        logger.debug(f"{model_id_for_log} - Loading reference audio '{speaker_ref_path_input}' for duration check.")
        audio_segment = AudioSegment.from_file(str(speaker_ref_path_input))
        MAX_OUTE_REF_DURATION_MS = 14500

        if len(audio_segment) > MAX_OUTE_REF_DURATION_MS:
            logger.info(f"{model_id_for_log} - Reference audio '{speaker_ref_path_input}' ({len(audio_segment)/1000.0:.1f}s) is > {MAX_OUTE_REF_DURATION_MS/1000.0:.1f}s. Trimming to {MAX_OUTE_REF_DURATION_MS/1000.0:.1f}s.")
            trimmed_segment = audio_segment[:MAX_OUTE_REF_DURATION_MS]
            
            # Create a temporary file for the trimmed audio
            fd, temp_trimmed_audio_path_str = tempfile.mkstemp(suffix=".wav", prefix="trimmed_ref_")
            os.close(fd) # Close the file descriptor, pydub will reopen
            
            trimmed_segment.export(temp_trimmed_audio_path_str, format="wav")
            logger.info(f"{model_id_for_log} - Using trimmed temporary audio: {temp_trimmed_audio_path_str}")
            speaker_ref_path_to_use = Path(temp_trimmed_audio_path_str)
            temp_trimmed_audio_path_to_delete = temp_trimmed_audio_path_str # Store string path for deletion
        else:
            logger.info(f"{model_id_for_log} - Reference audio '{speaker_ref_path_input}' ({len(audio_segment)/1000.0:.1f}s) is within length limits. Using original.")
            speaker_ref_path_to_use = speaker_ref_path_input
            # temp_trimmed_audio_path_to_delete remains None
            
    except Exception as e_audio_proc:
        logger.warning(f"{model_id_for_log} - Error processing reference audio '{speaker_ref_path_input}' for length check/trim: {e_audio_proc}. Using original path without trimming attempt.")
        speaker_ref_path_to_use = speaker_ref_path_input # Fallback to original path
        # temp_trimmed_audio_path_to_delete remains None
            
    return speaker_ref_path_to_use, temp_trimmed_audio_path_to_delete

def get_huggingface_cache_dir() -> Path:
    """
    Determines the Hugging Face Hub cache directory.
    Prefers HF_HOME, then XDG_CACHE_HOME, then default ~/.cache/huggingface.
    """
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"])
    elif os.getenv("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "huggingface"
    else:
        return Path.home() / ".cache" / "huggingface"

# --- Informational Functions (to be called from main.py) ---
def list_available_models(models_config_dict):
    """Prints available TTS models from the provided configuration."""
    print("\nAvailable TTS Models (from config):")
    print("-------------------------------------------------")
    if not models_config_dict: print("No models configured."); return
    for model_id, config in models_config_dict.items():
        print(f"- {model_id}:")
        print(f"  Notes: {config.get('notes', 'N/A')}")
    print("-------------------------------------------------")

def get_voice_info(model_id_to_query, models_config_dict):
    """Prints detailed voice/speaker information for a specific model."""
    # Imports from config needed for default voice names if not in model_config
    from config import ORPHEUS_DEFAULT_VOICE as G_ORPHEUS_DEFAULT_VOICE # Avoid name clash
    
    print(f"\nVoice Information for Model: {model_id_to_query}")
    print("-------------------------------------")
    if model_id_to_query not in models_config_dict:
        print(f"Model ID '{model_id_to_query}' not found in configuration."); return

    config = models_config_dict[model_id_to_query]
    default_voice_display = "Not specified" # Fallback
    # Try to get a sensible default display name
    if config.get('default_voice_id'): default_voice_display = config.get('default_voice_id')
    elif config.get('default_model_path_in_repo'): default_voice_display = config.get('default_model_path_in_repo')
    elif config.get('default_speaker_embedding_index') is not None: default_voice_display = f"Embedding Index: {config.get('default_speaker_embedding_index')}"
    elif config.get('default_speaker_id') is not None: default_voice_display = f"Speaker ID: {config.get('default_speaker_id')}"
    elif "orpheus" in model_id_to_query: default_voice_display = G_ORPHEUS_DEFAULT_VOICE
    print(f"  Default Voice/Speaker: {default_voice_display}")

    if "available_voices" in config and config["available_voices"]:
        print("  Known available custom voices/speaker identifiers (use with --german-voice-id):")
        for voice in config["available_voices"]: print(f"    - {voice}")
    if "test_default_speakers" in config and config["test_default_speakers"] and "oute" in model_id_to_query:
        print("  OuteTTS internal default speaker IDs suggested for testing (use with --german-voice-id):")
        for voice in config["test_default_speakers"]: print(f"    - {voice}")
    
    # Model-specific guidance
    if model_id_to_query == "piper_local": print("  Piper Voice: Provide model path (e.g., 'de/de_DE/...') or JSON '{\"model\":\"...\", \"config\":\"...\"}' via --german-voice-id.")
    elif model_id_to_query == "speecht5_german_transformers": print("  SpeechT5 Voice: Provide embedding index (int) or .pt/.pth xvector path via --german-voice-id.")
    elif model_id_to_query == "fastpitch_german_nemo": print("  NeMo Voice: Provide speaker ID (int) via --german-voice-id.")
    elif "orpheus" in model_id_to_query: print("  Orpheus models may support emotion tags like <laugh> in input text.")
    elif model_id_to_query.startswith("oute"): print("  OuteTTS Voice: Use OuteTTS default speaker ID or path to .wav/.json speaker file via --german-voice-id.")
    print("-------------------------------------")