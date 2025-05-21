# handlers/orpheus_kartoffel_handler.py

import logging
import os
from pathlib import Path
import gc
import json 
from typing import List, Optional 
import numpy as np 
import sys # For stderr print on critical import fails

# Conditional imports
TORCH_AVAILABLE_IN_HANDLER = False
TRANSFORMERS_AVAILABLE_IN_HANDLER = False
SNAC_AVAILABLE_IN_HANDLER = False
SOUNDFILE_AVAILABLE_IN_HANDLER = False 

torch_kartoffel = None 
AutoModelForCausalLM_kartoffel, AutoTokenizer_kartoffel = None, None
SNAC_kartoffel_cls = None 
sf_kartoffel = None   

logger_init = logging.getLogger("CrispTTS.handlers.orpheus_kartoffel.init")

try:
    import torch as torch_imp
    torch_kartoffel = torch_imp
    TORCH_AVAILABLE_IN_HANDLER = True
except ImportError as e:
    print(f"Kartoffel Handler INIT ERROR: PyTorch import failed: {e}", file=sys.stderr)
    logger_init.info("PyTorch not found. Kartoffel Orpheus handler will be non-functional.")

if TORCH_AVAILABLE_IN_HANDLER: # Only try these if torch is available
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        AutoModelForCausalLM_kartoffel = AutoModelForCausalLM
        AutoTokenizer_kartoffel = AutoTokenizer
        TRANSFORMERS_AVAILABLE_IN_HANDLER = True
    except ImportError as e:
        print(f"Kartoffel Handler INIT ERROR: Transformers import failed: {e}", file=sys.stderr)
        logger_init.info("Transformers library not found. Kartoffel Orpheus handler will be non-functional.")

    try:
        from snac import SNAC # Import the main SNAC class
        SNAC_kartoffel_cls = SNAC 
        SNAC_AVAILABLE_IN_HANDLER = True
    except ImportError as e:
        print(f"Kartoffel Handler INIT ERROR: SNAC import failed: {e}", file=sys.stderr)
        logger_init.info("SNAC library not found. Kartoffel Orpheus handler will be non-functional.")
try:
    import soundfile as sf_imp
    sf_kartoffel = sf_imp
    SOUNDFILE_AVAILABLE_IN_HANDLER = True
except ImportError as e:
    print(f"Kartoffel Handler INIT ERROR: SoundFile import failed: {e}", file=sys.stderr)
    logger_init.info("SoundFile library not found. Kartoffel Orpheus handler cannot save audio.")

# --- MODIFIED IMPORT FOR UTILS ---
# Assuming utils.py is in the project root, which is in sys.path thanks to main.py
try:
    from utils import save_audio, play_audio, SuppressOutput 
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Kartoffel Handler CRITICAL INIT ERROR: Failed to import from 'utils': {e}", file=sys.stderr)
    UTILS_AVAILABLE = False
    # If utils are critical, this handler might become non-functional.
    # Depending on strictness, you might re-raise or just log and let later checks fail.
    # For now, we'll let later checks handle it if UTILS_AVAILABLE is False.

logger = logging.getLogger("CrispTTS.handlers.orpheus_kartoffel")

def _redistribute_codes_for_kartoffel(code_list: list[int], device: str) -> Optional[List['torch.Tensor']]:
    if not TORCH_AVAILABLE_IN_HANDLER or not torch_kartoffel: 
        return None
    if not code_list or len(code_list) % 7 != 0:
        logger.warning(f"Kartoffel: Code list length ({len(code_list)}) is not a multiple of 7. Cannot redistribute.")
        return None
    num_frames = len(code_list) // 7
    if num_frames == 0:
        logger.warning("Kartoffel: Not enough codes for any SNAC frames after filtering.")
        return None
        
    layer_1_codes, layer_2_codes, layer_3_codes = [], [], []
    for i in range(num_frames):
        base_idx = 7 * i
        try:
            layer_1_codes.append(code_list[base_idx])
            layer_2_codes.append(code_list[base_idx + 1] - 4096)
            layer_3_codes.append(code_list[base_idx + 2] - (2 * 4096))
            layer_3_codes.append(code_list[base_idx + 3] - (3 * 4096))
            layer_2_codes.append(code_list[base_idx + 4] - (4 * 4096))
            layer_3_codes.append(code_list[base_idx + 5] - (5 * 4096))
            layer_3_codes.append(code_list[base_idx + 6] - (6 * 4096))
        except IndexError:
            logger.error(f"Kartoffel: Index out of bounds during code redistribution at frame {i}.", exc_info=True)
            return None

    codes_for_snac = [
        torch_kartoffel.tensor(layer_1_codes, dtype=torch_kartoffel.int32, device=device).unsqueeze(0),
        torch_kartoffel.tensor(layer_2_codes, dtype=torch_kartoffel.int32, device=device).unsqueeze(0),
        torch_kartoffel.tensor(layer_3_codes, dtype=torch_kartoffel.int32, device=device).unsqueeze(0),
    ]
    return codes_for_snac

def synthesize_with_orpheus_kartoffel(model_config, text, voice_id_override, model_params_override, output_file_str, play_direct):
    crisptts_model_id_for_log = model_config.get('crisptts_model_id', 'orpheus_kartoffel_natural') 

    if not all([TORCH_AVAILABLE_IN_HANDLER, TRANSFORMERS_AVAILABLE_IN_HANDLER, 
                  SNAC_AVAILABLE_IN_HANDLER, SOUNDFILE_AVAILABLE_IN_HANDLER, UTILS_AVAILABLE]):
        logger.error(f"Kartoffel Orpheus ({crisptts_model_id_for_log}): Missing core dependencies (PyTorch, Transformers, SNAC, SoundFile, or project Utils). Skipping.")
        return
    if not torch_kartoffel or not AutoModelForCausalLM_kartoffel or \
       not AutoTokenizer_kartoffel or not SNAC_kartoffel_cls or not sf_kartoffel:
        logger.error(f"Kartoffel Orpheus ({crisptts_model_id_for_log}): Critical Python modules not loaded correctly. Skipping.")
        return

    hf_token = os.getenv("HF_TOKEN")
    if model_config.get("requires_hf_token", False) and not hf_token:
        logger.error(f"Kartoffel Orpheus ({model_config.get('model_repo_id')}): Model requires Hugging Face token (HF_TOKEN env var). Skipping.")
        return

    # ... (rest of the function from Step 32, ensuring all config keys are checked for None) ...
    # (Make sure to use the aliased module names: torch_kartoffel, AutoModelForCausalLM_kartoffel, etc.)
    
    kartoffel_model_id = model_config.get("model_repo_id")
    tokenizer_id = model_config.get("tokenizer_repo_id", kartoffel_model_id)
    snac_model_id_cfg = model_config.get("snac_model_id")
    chosen_voice = voice_id_override or model_config.get("default_voice_id")
    
    prompt_start_token_id = model_config.get("prompt_start_token_id")
    prompt_end_token_ids_list = model_config.get("prompt_end_token_ids")
    generation_eos_token_id = model_config.get("generation_eos_token_id")
    audio_start_marker_token_id = model_config.get("audio_start_marker_token_id")
    audio_end_marker_token_id = model_config.get("audio_end_marker_token_id")
    audio_token_offset = model_config.get("audio_token_offset")
    sample_rate = model_config.get("sample_rate", 24000)

    # Check all required config values are present
    required_configs = {
        "kartoffel_model_id": kartoffel_model_id, "tokenizer_id": tokenizer_id, 
        "snac_model_id_cfg": snac_model_id_cfg, "chosen_voice": chosen_voice,
        "prompt_start_token_id": prompt_start_token_id, "prompt_end_token_ids_list": prompt_end_token_ids_list,
        "generation_eos_token_id": generation_eos_token_id, "audio_start_marker_token_id": audio_start_marker_token_id,
        "audio_end_marker_token_id": audio_end_marker_token_id, "audio_token_offset": audio_token_offset
    }
    for key, val in required_configs.items():
        if val is None:
            logger.error(f"Kartoffel Orpheus ({crisptts_model_id_for_log}): Missing critical config value for '{key}'. Skipping.")
            return

    logger.info(f"Kartoffel Orpheus ({crisptts_model_id_for_log}): Synthesizing with model '{kartoffel_model_id}', voice '{chosen_voice}'.")
    
    loaded_kartoffel_model_inst = None
    loaded_tokenizer_inst = None
    loaded_snac_decoder_inst = None

    try:
        if torch_kartoffel.cuda.is_available(): device = "cuda"
        elif hasattr(torch_kartoffel.backends, "mps") and torch_kartoffel.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
        logger.info(f"Kartoffel Orpheus: Using device: {device}")

        with SuppressOutput(suppress_stdout=not logger.isEnabledFor(logging.DEBUG), suppress_stderr=True):
            loaded_tokenizer_inst = AutoTokenizer_kartoffel.from_pretrained(tokenizer_id, token=hf_token, trust_remote_code=True)
            loaded_kartoffel_model_inst = AutoModelForCausalLM_kartoffel.from_pretrained(
                kartoffel_model_id, torch_dtype=torch_kartoffel.float16,
                token=hf_token, trust_remote_code=True 
            ).to(device).eval()
            loaded_snac_decoder_inst = SNAC_kartoffel_cls.from_pretrained(snac_model_id_cfg).to(device).eval()
        logger.info("Kartoffel Orpheus: Models (Kartoffel, Tokenizer, SNAC) loaded.")
        
        full_prompt_text = f"{chosen_voice}: {text}"       
        input_ids_text = loaded_tokenizer_inst(full_prompt_text, return_tensors="pt").input_ids
        start_token_tensor = torch_kartoffel.tensor([[prompt_start_token_id]], dtype=torch_kartoffel.int64)
        end_tokens_tensor = torch_kartoffel.tensor([prompt_end_token_ids_list], dtype=torch_kartoffel.int64)
        
        input_ids = torch_kartoffel.cat([start_token_tensor, input_ids_text, end_tokens_tensor], dim=1).to(device)
        attention_mask = torch_kartoffel.ones_like(input_ids)

        gen_params = {
            "max_new_tokens": 4000, "do_sample": True, "temperature": 0.6, "top_p": 0.95,
            "repetition_penalty": 1.1, "num_return_sequences": 1,
            "eos_token_id": generation_eos_token_id, "use_cache": True,
            "pad_token_id": loaded_tokenizer_inst.eos_token_id if loaded_tokenizer_inst.eos_token_id is not None else generation_eos_token_id,
        }
        if model_params_override:
            try:
                cli_gen_params = json.loads(model_params_override)
                gen_params.update(cli_gen_params)
            except json.JSONDecodeError:
                logger.warning(f"Kartoffel Orpheus: Could not parse --model-params: {model_params_override}")

        logger.debug(f"Kartoffel Orpheus: Generating with params: {gen_params}")
        with torch_kartoffel.no_grad():
            generated_ids_output = loaded_kartoffel_model_inst.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_params
            )
        
        logger.debug("Kartoffel Orpheus: Post-processing generated token IDs...")
        token_indices = (generated_ids_output == audio_start_marker_token_id).nonzero(as_tuple=True)
        
        cropped_tensor = generated_ids_output 
        if token_indices[1].numel() > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids_output[:, last_occurrence_idx + 1 :]
            logger.debug(f"Kartoffel: Cropped at audio_start_marker token index {last_occurrence_idx}.")
        else:
            logger.warning(f"Kartoffel: Audio start marker token {audio_start_marker_token_id} not found. Using full generated sequence.")

        tokens_to_process = cropped_tensor[0]
        masked_row = tokens_to_process[tokens_to_process != audio_end_marker_token_id]
        row_length = masked_row.size(0)

        if row_length < 7:
            logger.error(f"Kartoffel Orpheus: Not enough audio tokens ({row_length}) after processing. Skipping.")
            return

        new_length_for_snac = (row_length // 7) * 7
        if new_length_for_snac == 0:
            logger.error("Kartoffel Orpheus: Zero audio tokens after ensuring multiple of 7. Skipping.")
            return
            
        trimmed_row_for_snac = masked_row[:new_length_for_snac]
        processed_code_list = [t.item() - audio_token_offset for t in trimmed_row_for_snac]
        logger.debug(f"Kartoffel Orpheus: Prepared {len(processed_code_list)} codes for SNAC ({len(processed_code_list)//7} frames).")

        snac_input_codes = _redistribute_codes_for_kartoffel(processed_code_list, device)
        
        if not snac_input_codes:
            logger.error("Kartoffel Orpheus: Failed to redistribute codes for SNAC. Skipping.")
            return

        logger.info("Kartoffel Orpheus: Decoding with SNAC...")
        with torch_kartoffel.no_grad():
            audio_hat = loaded_snac_decoder_inst.decode(snac_input_codes) 
        
        audio_numpy = audio_hat.detach().squeeze().cpu().numpy()
        
        if audio_numpy.ndim == 0 or audio_numpy.size == 0 :
            logger.error("Kartoffel Orpheus: SNAC decoding resulted in empty or scalar audio.")
            return

        logger.info(f"Kartoffel Orpheus: Synthesis successful, audio_numpy shape: {audio_numpy.shape}")

        if output_file_str:
            output_path_obj = Path(output_file_str).with_suffix(".wav")
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            try:
                sf_kartoffel.write(str(output_path_obj), audio_numpy, samplerate=sample_rate)
                logger.info(f"Kartoffel Orpheus: Audio saved to {output_path_obj}")
            except Exception as e_save:
                logger.error(f"Kartoffel Orpheus: Failed to save to {output_path_obj}: {e_save}", exc_info=True)

        if play_direct and audio_numpy.size > 0 :
            audio_to_play = audio_numpy.flatten() 
            audio_int16 = (np.clip(audio_to_play, -1.0, 1.0) * 32767).astype(np.int16)
            play_audio(audio_int16.tobytes(), is_path=False, input_format="pcm_s16le", sample_rate=sample_rate)

    except Exception as e:
        logger.error(f"Kartoffel Orpheus ({kartoffel_model_id}): Synthesis failed: {e}", exc_info=True)
    finally:
        # Ensure variables are deleted if they were assigned
        if 'loaded_tokenizer_inst' in locals() and loaded_tokenizer_inst is not None: del loaded_tokenizer_inst
        if 'loaded_kartoffel_model_inst' in locals() and loaded_kartoffel_model_inst is not None: del loaded_kartoffel_model_inst
        if 'loaded_snac_decoder_inst' in locals() and loaded_snac_decoder_inst is not None: del loaded_snac_decoder_inst
        if 'input_ids' in locals() : del input_ids
        if 'attention_mask' in locals(): del attention_mask
        if 'generated_ids_output' in locals(): del generated_ids_output
        
        if TORCH_AVAILABLE_IN_HANDLER and torch_kartoffel:
            if torch_kartoffel.cuda.is_available(): torch_kartoffel.cuda.empty_cache()
            if hasattr(torch_kartoffel.backends, "mps") and torch_kartoffel.backends.mps.is_available() and hasattr(torch_kartoffel.mps, "empty_cache"):
                try: torch_kartoffel.mps.empty_cache()
                except Exception: pass 
        gc.collect()

print(f"--- Orpheus Kartoffel Handler Module Parsed (orpheus_kartoffel_handler.py) ---")