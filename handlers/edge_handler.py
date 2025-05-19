# handlers/edge_handler.py

import asyncio
import tempfile
import logging
import os
from pathlib import Path
import gc

# Use relative imports for project modules
from utils import save_audio, play_audio
# EdgeTTS is imported conditionally

logger = logging.getLogger("CrispTTS.handlers.edge")

EDGE_TTS_AVAILABLE_IN_HANDLER = False
try:
    import edge_tts
    EDGE_TTS_AVAILABLE_IN_HANDLER = True
except ImportError:
    logger.info("edge-tts library not found. EdgeTTS handler will not be functional.")
    edge_tts = None # Ensure it's defined for type checking if used

async def _synthesize_with_edge_tts_async_helper(text, voice_id, output_file_path_str: str):
    """Async helper for EdgeTTS synthesis."""
    if not EDGE_TTS_AVAILABLE_IN_HANDLER or not edge_tts:
        logger.error("EdgeTTS library is not available for async helper.")
        return None
    try:
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(output_file_path_str)
        return output_file_path_str
    except Exception as e:
        logger.error(f"EdgeTTS async synthesis error: {e}", exc_info=True)
        return None

def synthesize_with_edge_tts(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    if not EDGE_TTS_AVAILABLE_IN_HANDLER or not edge_tts:
        logger.error("EdgeTTS handler called, but edge-tts library is not available. Skipping.")
        return

    voice_id = voice_id_override or model_config.get("default_voice_id", "de-DE-KatjaNeural")
    logger.debug(f"EdgeTTS - Voice: {voice_id}, Text: '{text[:50]}...'")
    
    temp_mp3_path_obj = None # Will be a Path object
    success_path_str = None  # Will be a string path

    # model_params_override is not typically used by EdgeTTS in this basic setup
    if model_params_override:
        logger.warning("EdgeTTS handler received model_params_override, but does not use them directly.")

    try:
        # Create a temporary file that will be cleaned up
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmpfile_obj:
            temp_mp3_path_obj = Path(tmpfile_obj.name)
        
        # Ensure asyncio loop management
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_closed(): # pragma: no cover
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError: # pragma: no cover
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        success_path_str = loop.run_until_complete(
            _synthesize_with_edge_tts_async_helper(text, voice_id, str(temp_mp3_path_obj))
        )

        if success_path_str and Path(success_path_str).exists() and Path(success_path_str).stat().st_size > 100:
            target_output_file = Path(output_file_str).with_suffix(".mp3") if output_file_str else None
            if target_output_file:
                # save_audio expects source path as string if source_is_path
                save_audio(str(success_path_str), str(target_output_file), source_is_path=True)
            if play_direct:
                play_audio(str(success_path_str), is_path=True)
        else:
            logger.error("EdgeTTS synthesis failed or produced an empty file.")
            # Ensure temp_mp3_path_obj is unlinked if it still exists after a failure
            if temp_mp3_path_obj and temp_mp3_path_obj.exists():
                 temp_mp3_path_obj.unlink(missing_ok=True)
            temp_mp3_path_obj = None # Mark as handled

    except Exception as e:
        logger.error(f"Error in EdgeTTS synthesis process: {e}", exc_info=True)
    finally:
        if temp_mp3_path_obj and temp_mp3_path_obj.exists(): # Final cleanup check
            try:
                temp_mp3_path_obj.unlink(missing_ok=True)
            except OSError as e_final_rem:
                 logger.warning(f"Could not remove temp file {temp_mp3_path_obj} in EdgeTTS finally: {e_final_rem}")
        gc.collect()