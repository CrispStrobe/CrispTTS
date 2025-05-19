# decoder.py

import logging
import numpy as np
import torch

# Attempt to import SNAC and handle failure gracefully
SNAC_AVAILABLE = False
SNAC_model = None # SNAC class itself
snac_loaded_model = None # The loaded model instance
snac_device = "cpu" # Default to CPU

try:
    from snac import SNAC as SNAC_cls # Import the class
    SNAC_model = SNAC_cls
    SNAC_AVAILABLE = True
    logger_decoder_init = logging.getLogger("CrispTTS.decoder.init") # Logger for init phase

    if SNAC_AVAILABLE:
        logger_decoder_init.info("SNAC library found. Attempting to load SNAC model 'hubertsiuzdak/snac_24khz'...")
        try:
            # Model loading should happen once, ideally.
            # Making it global here means it loads when this module is imported.
            _snac_model_instance = SNAC_model.from_pretrained("hubertsiuzdak/snac_24khz").eval()

            if torch.cuda.is_available():
                snac_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # For Apple Silicon
                snac_device = "mps"
            else:
                snac_device = "cpu"
            
            snac_loaded_model = _snac_model_instance.to(snac_device)
            logger_decoder_init.info(f"SNAC model 'hubertsiuzdak/snac_24khz' loaded successfully to device: {snac_device}")
        except Exception as e:
            logger_decoder_init.error(f"Failed to load SNAC model 'hubertsiuzdak/snac_24khz': {e}", exc_info=True)
            SNAC_AVAILABLE = False # Mark as unavailable if model load fails
            snac_loaded_model = None

except ImportError:
    logging.getLogger("CrispTTS.decoder.init").warning( # Use separate logger for this phase
        "SNAC library not found. Orpheus decoding will not function. "
        "Please install it if you intend to use Orpheus models that require SNAC."
    )
except Exception as e_outer:
    logging.getLogger("CrispTTS.decoder.init").error(
        f"An unexpected error occurred during SNAC model initialization: {e_outer}", exc_info=True
    )
    SNAC_AVAILABLE = False
    snac_loaded_model = None


# Main logger for the rest of the module's functions
logger = logging.getLogger("CrispTTS.decoder")

def convert_to_audio(multiframe_tokens: list[int], total_token_count: int) -> bytes:
    """
    Converts a list of Orpheus/SNAC model's custom audio tokens into raw audio bytes
    using the pre-loaded SNAC model.

    Args:
        multiframe_tokens (list[int]): A list of integer token IDs.
                                       The Orpheus token processor in utils.py typically
                                       passes a buffer of 28 tokens.
        total_token_count (int): The total number of valid audio tokens processed so far.

    Returns:
        bytes: Raw audio data (16-bit PCM mono, 24000 Hz) or empty bytes if error.
    """
    if not SNAC_AVAILABLE or not snac_loaded_model:
        if convert_to_audio != placeholder_for_unavailable_snac: # Avoid recursion if this itself is the placeholder
             logger.error("SNAC model is not available or failed to load. Cannot decode Orpheus tokens.")
        return b'' # Return empty bytes if SNAC model isn't ready

    if not multiframe_tokens or len(multiframe_tokens) < 7: # SNAC processes tokens in groups of 7 for 3 codebooks
        # logger.debug(f"Decoder: Insufficient tokens ({len(multiframe_tokens)}) received for a frame.")
        return b''

    # Ensure multiframe_tokens contains only integers
    if not all(isinstance(token, int) for token in multiframe_tokens):
        logger.warning(f"Decoder: Received non-integer tokens in multiframe_tokens: {multiframe_tokens}")
        return b''
        
    num_frames_possible = len(multiframe_tokens) // 7
    if num_frames_possible == 0:
        # logger.debug(f"Decoder: Not enough tokens ({len(multiframe_tokens)}) for a full 7-token frame.")
        return b''
    
    # Process only complete frames
    valid_tokens_for_processing = multiframe_tokens[:num_frames_possible * 7]

    # Based on your uploaded decoder.py logic for SNAC
    # Codes are structured for 3 codebooks
    codes_0_list = []
    codes_1_list = []
    codes_2_list = []

    for j in range(num_frames_possible):
        i = 7 * j
        codes_0_list.append(valid_tokens_for_processing[i])
        
        codes_1_list.append(valid_tokens_for_processing[i+1])
        codes_1_list.append(valid_tokens_for_processing[i+4])
        
        codes_2_list.append(valid_tokens_for_processing[i+2])
        codes_2_list.append(valid_tokens_for_processing[i+3])
        codes_2_list.append(valid_tokens_for_processing[i+5])
        codes_2_list.append(valid_tokens_for_processing[i+6])

    try:
        codes_0 = torch.tensor(codes_0_list, device=snac_device, dtype=torch.int32).unsqueeze(0) # Add batch dim
        codes_1 = torch.tensor(codes_1_list, device=snac_device, dtype=torch.int32).unsqueeze(0)
        codes_2 = torch.tensor(codes_2_list, device=snac_device, dtype=torch.int32).unsqueeze(0)
        codes = [codes_0, codes_1, codes_2]

        # SNAC model might have specific range checks for tokens
        # The original decoder.py had: if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) ... return
        # This check might be important if your tokens are not already within the expected range.
        # For now, assuming tokens are valid as per `orpheus_turn_token_into_id` and SNAC's expectations.

        with torch.inference_mode():
            audio_hat = snac_loaded_model.decode(codes) # Expected shape: [1, 1, T]

        # The original decoder used a slice: audio_slice = audio_hat[:, :, 2048:4096]
        # This implies a fixed-size output from decode, and only a part is used.
        # This might be specific to the way tokens are generated or overlap-add is handled.
        # For a general decoder, you might use the whole output or adjust based on model.
        # If the SNAC model decode always produces a fixed output length per call regardless of input token count:
        # For example, if 28 tokens (4 SNAC frames) -> ~6144 samples, and you need a specific window.
        # The slice [:, :, 2048:4096] takes 2048 samples. At 24kHz, this is ~85ms.
        # This needs to be understood in context of how the Orpheus model produces tokens
        # and how SNAC is expected to reconstruct audio from them.
        
        # Using the slice from your provided decoder.py
        # Ensure audio_hat has the expected dimensions before slicing
        if audio_hat.ndim == 3 and audio_hat.shape[2] >= 4096:
            audio_slice = audio_hat[:, :, 2048:4096]
        elif audio_hat.ndim == 3 and audio_hat.shape[2] > 0 : # If output is shorter, take what's available
            logger.warning(f"SNAC output shorter than expected for slicing. Output shape: {audio_hat.shape}. Using available audio.")
            audio_slice = audio_hat 
        else:
            logger.error(f"SNAC output has unexpected shape: {audio_hat.shape}. Cannot process.")
            return b''


        detached_audio_cpu = audio_slice.detach().cpu()
        audio_numpy_float32 = detached_audio_cpu.numpy().astype(np.float32)
        
        # Normalize if needed (SNAC output is typically [-1, 1]) and convert to int16
        audio_numpy_int16 = (np.clip(audio_numpy_float32, -1.0, 1.0) * 32767).astype(np.int16)
        
        audio_bytes = audio_numpy_int16.tobytes()
        # logger.debug(f"Decoder: Successfully decoded {len(multiframe_tokens)} tokens into {len(audio_bytes)} audio bytes.")
        return audio_bytes

    except Exception as e:
        logger.error(f"Error during SNAC audio decoding: {e}", exc_info=True)
        return b''

# A placeholder function to check against if SNAC is completely unavailable at module load time.
# This is distinct from the `user_orpheus_decoder_placeholder` in `utils.py` which is a fallback
# if `decoder.py` or `convert_to_audio` itself cannot be imported from `decoder.py`.
def placeholder_for_unavailable_snac(multiframe_tokens: list[int], total_token_count: int) -> bytes:
    logger.critical(
            "CRITICAL: SNAC model (hubertsiuzdak/snac_24khz) required by decoder.py could not be loaded. "
            "Orpheus audio generation will not work. Please check SNAC installation and model availability."
    )
    return b''

# If SNAC failed to load at module import, redefine convert_to_audio to be the placeholder.
if not SNAC_AVAILABLE or not snac_loaded_model:
    logger.warning("Redefining 'convert_to_audio' to placeholder due to SNAC unavailability.")
    convert_to_audio = placeholder_for_unavailable_snac


if __name__ == "__main__":
    # Configure a basic logger for standalone testing of this decoder.py
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Standalone testing of decoder.py using SNAC...")

    if SNAC_AVAILABLE and snac_loaded_model:
        logger.info(f"SNAC model loaded on {snac_device} for testing.")
        # Example dummy tokens (7 tokens form one "SNAC frame" for the 3 codebooks)
        # The `orpheus_turn_token_into_id` from utils.py would produce these.
        # These are just placeholder IDs and won't produce meaningful audio.
        # The actual tokens are usually within a certain range (e.g., 0-4095 after decoding).
        # The structure from your `decoder.py` implies 7 input tokens make one SNAC frame.
        # convert_to_audio processes `num_frames_possible = len(multiframe_tokens) // 7`
        # and expects multiframe_tokens to be a flat list.
        
        # Test with 28 tokens (4 SNAC frames)
        # These are raw token IDs *after* `orpheus_turn_token_into_id`
        # (which subtracts 10 and the offset).
        # For SNAC, the actual values passed to model.decode are usually within [0, N_CODES_PER_CODEBOOK-1]
        # The `orpheus_turn_token_into_id` has a formula: int(num) - 10 - ((idx % 7) * 4096)
        # This suggests the original <custom_token_XXXX> numbers are large.
        # The SNAC model itself expects token IDs (usually 0-1023 or similar per codebook).
        # Your original `decoder.py` takes the output of `orpheus_turn_token_into_id` directly.
        # This implies the tokens from `orpheus_turn_token_into_id` are what SNAC expects *after* they are
        # structured into the three codebooks (codes_0, codes_1, codes_2).
        # The values in `multiframe_tokens` should be the already processed IDs from `orpheus_turn_token_into_id`.
        
        # Let's simulate tokens that might be valid for SNAC codebooks (e.g., small integers)
        # This is for testing the flow, not for meaningful audio.
        dummy_processed_tokens_1 = [100, 200, 300, 400, 150, 250, 350] * 4 # 28 tokens for 4 SNAC frames
        
        logger.info(f"Testing convert_to_audio with {len(dummy_processed_tokens_1)} dummy processed tokens...")
        audio_output_bytes = convert_to_audio(dummy_processed_tokens_1, len(dummy_processed_tokens_1))

        if audio_output_bytes:
            logger.info(f"Successfully decoded into {len(audio_output_bytes)} audio bytes.")
            # You could save this to a file to inspect (it will be gibberish with dummy tokens)
            try:
                test_output_path = Path("dummy_decoder_output.wav")
                with wave.open(str(test_output_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(24000) # Assuming SNAC output is 24kHz
                    wf.writeframes(audio_output_bytes)
                logger.info(f"Dummy audio saved to {test_output_path.resolve()}")
            except Exception as e:
                logger.error(f"Could not save dummy audio: {e}")
        else:
            logger.warning("convert_to_audio returned no bytes with dummy tokens (as expected if values out of range or other issue).")
    else:
        logger.error("SNAC model not available for standalone test.")