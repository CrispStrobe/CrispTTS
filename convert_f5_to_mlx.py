# convert_f5_to_mlx_monkeypatched_v4_debug.py
import argparse
from pathlib import Path
import tempfile
import shutil
import os
import json
from typing import Optional, Tuple

# --- Early imports for monkeypatching & core functionality ---
import sys
from huggingface_hub import snapshot_download, hf_hub_download, HfApi # upload_file also useful

# --- Monkeypatch Definition ---
# This function will replace the original fetch_from_hub
def _patched_fetch_from_hub_impl_v4(hf_repo_or_path: str, quantization_bits: Optional[int] = None) -> Path:
    print(f"--- DDD: Executing MONKEYPATCHED fetch_from_hub v4 ---")
    print(f"DDD: Received path/repo_id: '{hf_repo_or_path}' (type: {type(hf_repo_or_path)})")
    local_path_candidate = Path(hf_repo_or_path)
    print(f"DDD: Path object: {local_path_candidate}")
    print(f"DDD: local_path_candidate.exists(): {local_path_candidate.exists()}")
    print(f"DDD: local_path_candidate.is_dir(): {local_path_candidate.is_dir()}")

    if local_path_candidate.is_dir():
        print(f"DDD: Monkeypatched fetch_from_hub: Detected local directory: {local_path_candidate}")
        return local_path_candidate

    print(f"DDD: Monkeypatched fetch_from_hub: Assuming '{hf_repo_or_path}' is an HF Repo ID. Proceeding to download...")
    allow_patterns = ["model_v1*.safetensors", "vocab.txt", "duration_v*.safetensors", "*.json"]
    try:
        downloaded_path_dir = Path(
            snapshot_download(
                repo_id=hf_repo_or_path,
                allow_patterns=allow_patterns,
            )
        )
        print(f"DDD: Monkeypatched fetch_from_hub: Downloaded content to directory: {downloaded_path_dir}")
        return downloaded_path_dir
    except Exception as e:
        print(f"DDD: Monkeypatched fetch_from_hub: Error during snapshot_download for '{hf_repo_or_path}': {e}")
        raise

# --- Apply Monkeypatch ---
try:
    import f5_tts_mlx.utils 
    sys.modules['f5_tts_mlx.utils'].fetch_from_hub = _patched_fetch_from_hub_impl_v4
    print("--- Monkeypatch applied to sys.modules['f5_tts_mlx.utils'].fetch_from_hub ---")

    # Attempt to patch cfm's direct reference if it has one (speculative)
    import f5_tts_mlx.cfm
    if hasattr(f5_tts_mlx.cfm, 'fetch_from_hub') and \
       callable(getattr(f5_tts_mlx.cfm, 'fetch_from_hub')) and \
       getattr(f5_tts_mlx.cfm, 'fetch_from_hub').__module__ == 'f5_tts_mlx.utils':
        f5_tts_mlx.cfm.fetch_from_hub = _patched_fetch_from_hub_impl_v4
        print("--- Monkeypatch also applied to 'fetch_from_hub' directly in 'f5_tts_mlx.cfm' module namespace (if found). ---")

except ModuleNotFoundError:
    print("CRITICAL ERROR: Could not find 'f5_tts_mlx.utils' or 'f5_tts_mlx.cfm' for monkeypatching.")
    exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during monkeypatching setup: {e}")
    exit(1)

# --- Standard Imports (should now use patched util if they import f5_tts_mlx.utils) ---
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import soundfile as sf
import numpy as np
import re

try:
    from f5_tts_mlx.cfm import F5TTS
    from f5_tts_mlx.utils import convert_char_to_pinyin
    from f5_tts_mlx.generate import SAMPLE_RATE, HOP_LENGTH
    from vocos_mlx import Vocos # For vocoder bypass
except ImportError as e:
    print(f"Error importing f5_tts_mlx components after monkeypatching: {e}")
    exit(1)

TARGET_RMS = 0.1

def download_hf_file_direct(repo_id: str, filename: str, hf_token: str = None, cache_dir: str = None) -> Path:
    print(f"Downloading specific file: {filename} from {repo_id}...")
    try:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token, cache_dir=cache_dir)
        print(f"Successfully downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {e}")
        raise

def convert_and_quantize_f5_model(
    source_pytorch_safetensors_path: Path,
    source_vocab_path: Path,
    target_output_mlx_model_dir: Path,
    quantization_bits: Optional[int] = None,
) -> Tuple[Path, Path]:
    print("\n--- Starting Model Conversion and Quantization ---")
    target_output_mlx_model_dir.mkdir(parents=True, exist_ok=True)

    if quantization_bits:
        output_model_filename = f"model_v1_{quantization_bits}b.safetensors"
    else:
        output_model_filename = "model_v1.safetensors"
    final_mlx_model_path = target_output_mlx_model_dir / output_model_filename

    with tempfile.TemporaryDirectory() as temp_conversion_input_dir:
        temp_dir_path = Path(temp_conversion_input_dir)
        
        temp_pytorch_weights_for_conversion = temp_dir_path / "model_v1.safetensors"
        shutil.copyfile(source_pytorch_safetensors_path, temp_pytorch_weights_for_conversion)
        shutil.copyfile(source_vocab_path, temp_dir_path / "vocab.txt")
        
        print(f"Attempting to load and convert PyTorch model from temp dir: {temp_dir_path}")
        mlx_model_converted = F5TTS.from_pretrained(
            str(temp_dir_path), 
            quantization_bits=None 
        )
        print("PyTorch model weights successfully loaded and converted to MLX structure.")

        if quantization_bits:
            print(f"Applying {quantization_bits}-bit quantization...")
            # Using the library-consistent predicate (from cfm.py)
            predicate = lambda p, m: (isinstance(m, nn.Linear) and hasattr(m, 'weight') and m.weight.shape[1] % 64 == 0)
            nn.quantize(
                mlx_model_converted,
                bits=quantization_bits,
                class_predicate=predicate,
            )
            mx.eval(mlx_model_converted.parameters())
            print("Quantization applied.")
        else:
            print("No quantization requested.")

        print(f"Saving converted MLX model to {final_mlx_model_path}...")
        flat_params = dict(tree_flatten(mlx_model_converted.parameters()))
        mx.save_safetensors(str(final_mlx_model_path), flat_params)

        final_vocab_path = target_output_mlx_model_dir / "vocab.txt"
        shutil.copyfile(source_vocab_path, final_vocab_path)
        print(f"Converted MLX model saved to {final_mlx_model_path}")
        print(f"Vocab file copied to {final_vocab_path}")

    return final_mlx_model_path, final_vocab_path

def perform_mlx_inference(
    converted_mlx_model_dir: Path,
    quantization_bits_for_loading: Optional[int],
    ref_audio_path_str: str,
    ref_text: str,
    gen_text: str,
    output_inference_wav_path: Path,
    steps: int = 8,
    cfg: float = 2.0,
    seed: int = 42
):
    print("\n--- Performing Test Inference ---")
    ref_audio_path = Path(ref_audio_path_str)
    if not ref_audio_path.exists():
        print(f"Reference audio {ref_audio_path} not found. Creating dummy silent audio for test.")
        ref_audio_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        sf.write(str(ref_audio_path), dummy_audio, SAMPLE_RATE)
        ref_text = "This is a silent reference audio for testing purposes."

    print(f"Loading converted MLX model from directory: {converted_mlx_model_dir}")
    f5_model_infer = F5TTS.from_pretrained(
        str(converted_mlx_model_dir),
        quantization_bits=quantization_bits_for_loading
    )
    mx.eval(f5_model_infer.parameters())

    print(f"Loading reference audio from: {ref_audio_path}")
    audio_data, sr = sf.read(str(ref_audio_path))

    if audio_data.ndim > 1:
        print(f"DEBUG: Reference audio original shape: {audio_data.shape}")
        audio_data = np.mean(audio_data, axis=1).astype(np.float32)
        print(f"DEBUG: Reference audio shape after mono conversion: {audio_data.shape}")
    elif audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    if sr != SAMPLE_RATE:
        print(f"Warning: Reference audio SR is {sr}, model expects {SAMPLE_RATE}. Resampling (basic).")
        original_num_frames = audio_data.shape[0]
        target_num_frames = int(original_num_frames * SAMPLE_RATE / sr)
        x_new = np.linspace(0, original_num_frames - 1, target_num_frames)
        x_original = np.arange(original_num_frames)
        audio_data = np.interp(x_new, x_original, audio_data).astype(np.float32)
        print(f"DEBUG: Reference audio shape after resampling: {audio_data.shape}")

    MAX_REF_AUDIO_DURATION_SEC = 15
    max_ref_samples = int(MAX_REF_AUDIO_DURATION_SEC * SAMPLE_RATE)
    if len(audio_data) > max_ref_samples:
        print(f"Reference audio is longer than {MAX_REF_AUDIO_DURATION_SEC}s ({len(audio_data)/SAMPLE_RATE:.2f}s). Truncating to {MAX_REF_AUDIO_DURATION_SEC}s.")
        audio_data = audio_data[:max_ref_samples]
        print(f"DEBUG: Reference audio shape after truncation: {audio_data.shape}")
    
    ref_audio_mx = mx.array(audio_data)

    rms = mx.sqrt(mx.mean(mx.square(ref_audio_mx)))
    if rms < TARGET_RMS:
        ref_audio_mx = ref_audio_mx * TARGET_RMS / rms

    print(f"Reference text for inference: '{ref_text}'")
    print(f"Text to generate: '{gen_text}'")

    combined_text_for_model_list = [ref_text + " " + gen_text]
    processed_text_for_model = convert_char_to_pinyin(combined_text_for_model_list)
    print(f"Processed text for model input: {processed_text_for_model}")

    ref_audio_len_frames = ref_audio_mx.shape[0] // HOP_LENGTH
    ref_text_len_bytes = len(ref_text.encode('utf-8')) + 3 * len(re.findall(r"[。，、；：？！.,]", ref_text))
    gen_text_len_bytes = len(gen_text.encode('utf-8')) + 3 * len(re.findall(r"[。，、；：？！.,]", gen_text))
    if ref_text_len_bytes == 0: ref_text_len_bytes = 1
    estimated_gen_frames = int(ref_audio_len_frames / ref_text_len_bytes * gen_text_len_bytes / 1.0)
    duration_frames = ref_audio_len_frames + max(10, estimated_gen_frames)
    print(f"Reference audio frames (after potential truncation): {ref_audio_len_frames}")
    print(f"Estimated total duration in frames for inference: {duration_frames}")

    print("Starting inference...")
    generated_wave_batched, _ = f5_model_infer.sample(
        cond=mx.expand_dims(ref_audio_mx, axis=0),
        text=processed_text_for_model,
        duration=duration_frames,
        steps=steps,
        cfg_strength=cfg,
        seed=seed
    )
    mx.eval(generated_wave_batched)
    print(f"DEBUG: Shape of generated_wave_batched from sample(): {generated_wave_batched.shape}")
    print(f"DEBUG: Dtype of generated_wave_batched: {generated_wave_batched.dtype}")

    generated_wave_full = None
    vocoder_bypass_used = False

    # --- MODIFIED CHECK for generated_wave_batched ---
    if not isinstance(generated_wave_batched, mx.array) or generated_wave_batched.size == 0:
        error_message = (f"f5_model_infer.sample() returned an empty or invalid output. "
                         f"Shape: {generated_wave_batched.shape}, type: {type(generated_wave_batched)}.")
        print(f"ERROR: {error_message}") # Changed to ERROR as this is more critical
        # Attempt vocoder bypass if the primary path failed badly
    elif generated_wave_batched.ndim == 1: # Output is already 1D (batch_size=1, squeezed)
        print("INFO: sample() returned 1D audio array, assuming batch_size=1 output.")
        generated_wave_full = generated_wave_batched # This is already the (audio_samples,)
    elif generated_wave_batched.ndim == 2 and generated_wave_batched.shape[0] == 1: # Expected (1, audio_samples)
        generated_wave_full = generated_wave_batched[0]
    else: # Unexpected shape
        error_message = (f"f5_model_infer.sample() returned an unexpected shape. "
                         f"Expected 1D or 2D [1, audio_samples], but got shape {generated_wave_batched.shape}.")
        print(f"WARNING: {error_message}") # This will now trigger the bypass

    if generated_wave_full is None: # If any of the above conditions led to bypass
        print("Attempting vocoder bypass strategy...")
        vocoder_bypass_used = True
        try:
            original_internal_vocoder = f5_model_infer._vocoder
            f5_model_infer._vocoder = None 
            print("INFO: Temporarily disabled internal vocoder in F5TTS instance to attempt raw mel output.")

            raw_mels_batched, _ = f5_model_infer.sample(
                cond=mx.expand_dims(ref_audio_mx, axis=0),
                text=processed_text_for_model,
                duration=duration_frames,
                steps=steps,
                cfg_strength=cfg,
                seed=seed
            )
            mx.eval(raw_mels_batched)
            f5_model_infer._vocoder = original_internal_vocoder
            print(f"DEBUG: Raw mels output shape from sample (with vocoder bypass): {raw_mels_batched.shape}")

            if raw_mels_batched.ndim != 3 or raw_mels_batched.shape[0] != 1:
                raise ValueError(f"Bypassed vocoder but did not get expected raw mels [1, frames, mels]. Shape: {raw_mels_batched.shape}")

            generated_mels_full_frames = raw_mels_batched[0] # Shape: [frames, mels]
            generated_mels_trimmed_frames = generated_mels_full_frames[ref_audio_len_frames:, :] 

            print("INFO: Vocoding externally using script's VocosMLX instance.")
            script_vocoder = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")
            mx.eval(script_vocoder.parameters())
            
            # --- CORRECTED MEL SHAPE FOR THIS VOCOS ---
            # Vocos lucasnewman/vocos-mel-24khz expects (batch, frames, mels) as per your test_vocos_mlx.py
            mels_for_script_vocoder = mx.expand_dims(generated_mels_trimmed_frames, axis=0) # [1, gen_frames, mels]
            mels_for_script_vocoder = mels_for_script_vocoder.astype(mx.float32)
            print(f"DEBUG: Shape of mels_for_script_vocoder for external vocoder: {mels_for_script_vocoder.shape}, Dtype: {mels_for_script_vocoder.dtype}")
            
            generated_wave_batched_external = script_vocoder.decode(mels_for_script_vocoder)
            mx.eval(generated_wave_batched_external)
            print(f"DEBUG: Shape of generated_wave_script_vocoded_batched: {generated_wave_batched_external.shape}")

            if generated_wave_batched_external.ndim == 1: # If vocos_mlx squeezed for batch 1
                 generated_wave_full = generated_wave_batched_external
            elif generated_wave_batched_external.ndim == 2 and generated_wave_batched_external.shape[0] == 1:
                 generated_wave_full = generated_wave_batched_external[0]
            else:
                 raise ValueError(f"External vocoding failed to produce 1D or [1,L] audio. Shape: {generated_wave_batched_external.shape}")
            
            # This generated_wave_full is already the generated part (mels were trimmed before vocoding)
            generated_wave_trimmed = generated_wave_full 
        except Exception as e_bypass:
            print(f"ERROR: Vocoder bypass strategy also failed: {e_bypass}")
            # Re-raise the original error that triggered the bypass, plus the bypass error
            original_trigger_error_msg = (f"f5_model_infer.sample() returned an unexpected output. "
                                          f"Expected a batched audio array [1, audio_samples], "
                                          f"but got shape {generated_wave_batched.shape} and type {type(generated_wave_batched)}.")
            raise ValueError(f"Both direct sample() output and vocoder bypass failed. Original trigger: {original_trigger_error_msg}") from e_bypass
    
    # If generated_wave_full is still None here, it means direct path also failed critically
    if generated_wave_full is None:
        raise ValueError("Failed to obtain valid audio waveform from model sample or bypass.")

    # Trimming logic (only if not bypassed and trimming was needed)
    if not vocoder_bypass_used: # If bypass was used, generated_wave_trimmed is already set correctly
        print(f"DEBUG: Shape of generated_wave_full (after batch index, before trim): {generated_wave_full.shape}")
        if generated_wave_full.ndim == 0:
            raise ValueError(f"generated_wave_full is scalar before trimming. Original batched shape: {generated_wave_batched.shape}")
        
        ref_len_samples = ref_audio_mx.shape[0]
        if ref_len_samples >= generated_wave_full.shape[0]:
            print(f"WARNING: Reference audio length ({ref_len_samples}) is >= generated full audio length ({generated_wave_full.shape[0]}). Generated part might be empty or very short.")
            actual_gen_start_index = min(ref_len_samples, generated_wave_full.shape[0] -1)
            actual_gen_start_index = max(0, actual_gen_start_index) # Ensure not negative
            generated_wave_trimmed = generated_wave_full[actual_gen_start_index:]
            if generated_wave_trimmed.size == 0 and generated_wave_full.size > 0 :
                 generated_wave_trimmed = generated_wave_full[-1:] 
            elif generated_wave_trimmed.size == 0 and generated_wave_full.size == 0:
                 generated_wave_trimmed = mx.array([], dtype=generated_wave_full.dtype)
            print(f"DEBUG: Trimming result (ref >= gen): {generated_wave_trimmed.shape}")
        else:
            generated_wave_trimmed = generated_wave_full[ref_len_samples:]
            print(f"DEBUG: Trimming result (ref < gen): {generated_wave_trimmed.shape}")

    output_inference_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_inference_wav_path), np.array(generated_wave_trimmed), SAMPLE_RATE)
    if vocoder_bypass_used:
        print(f"Inference output (with vocoder bypass) saved to: {output_inference_wav_path}")
    else:
        print(f"Inference output saved to: {output_inference_wav_path}")


def upload_to_hf_hub(
    local_folder: Path,
    hf_repo_id: str,
    hf_token: str = None,
    commit_message_prefix: str = "Upload converted F5 MLX model"
):
    print(f"\n--- Uploading to Hugging Face Hub: {hf_repo_id} ---")
    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id=hf_repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Could not create or ensure repo exists: {e}. Please create it manually if needed.")
        return

    print(f"Uploading files from {local_folder} to {hf_repo_id}...")
    api.upload_folder(
        folder_path=str(local_folder),
        repo_id=hf_repo_id,
        commit_message=commit_message_prefix,
        repo_type="model"
    )
    print("Upload process completed.")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch F5 TTS model to MLX, quantize, test, and upload.")
    parser.add_argument("--hf_repo_id_pytorch", type=str, required=True, help="HF Repo ID for the source PyTorch F5 model.")
    parser.add_argument("--pytorch_model_filename", type=str, required=True, help="Filename of the .safetensors PyTorch model in the repo.")
    parser.add_argument("--source_vocab_filename", type=str, default="vocab.txt", help="Filename of the vocab.txt in the PyTorch model repo (or separate if hf_vocab_repo_id is set).")
    parser.add_argument("--hf_vocab_repo_id", type=str, default=None, help="Optional: HF Repo ID for vocab.txt if it's separate from the model repo.")
    parser.add_argument("--output_dir", type=str, default="converted_f5_mlx_model", help="Directory to save the converted MLX model and vocab.")
    parser.add_argument("--quantization_bits", type=int, default=4, choices=[0, 4, 8], help="Bits for quantization (4 or 8). 0 for no quantization.")
    parser.add_argument("--skip_inference", action="store_true", help="Skip the inference test step.")
    parser.add_argument("--ref_audio_for_test", type=str, default="test_ref_audio.wav", help="Path to reference audio for the inference test. Will be created as dummy if not found.")
    parser.add_argument("--hf_upload_repo_id", type=str, default=None, help="Optional: HF Repo ID to upload the converted MLX model and vocab to.")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API token (or set HF_TOKEN env var).")
    args = parser.parse_args()

    target_output_mlx_dir = Path(args.output_dir)

    try:
        source_pytorch_weights_path = download_hf_file_direct(
            args.hf_repo_id_pytorch, args.pytorch_model_filename, args.hf_token
        )
        vocab_repo_id = args.hf_vocab_repo_id if args.hf_vocab_repo_id else args.hf_repo_id_pytorch
        source_vocab_path = download_hf_file_direct(
            vocab_repo_id, args.source_vocab_filename, args.hf_token
        )
    except Exception as e:
        print(f"Failed to download source files: {e}")
        return

    q_bits = args.quantization_bits if args.quantization_bits in [4, 8] else None

    try:
        _, _ = convert_and_quantize_f5_model(
            source_pytorch_safetensors_path=source_pytorch_weights_path,
            source_vocab_path=source_vocab_path,
            target_output_mlx_model_dir=target_output_mlx_dir,
            quantization_bits=q_bits,
        )
    except Exception as e:
        print(f"Error during model conversion/quantization: {e}")
        import traceback
        traceback.print_exc()
        return

    if not args.skip_inference:
        try:
            perform_mlx_inference(
                converted_mlx_model_dir=target_output_mlx_dir,
                quantization_bits_for_loading=q_bits,
                ref_audio_path_str=args.ref_audio_for_test,
                ref_text="Dies ist eine Testaufnahme für die Sprachreferenz.",
                gen_text="Hallo Welt, dies ist ein Test der Sprachgenerierung mit MLX.",
                output_inference_wav_path=target_output_mlx_dir / "test_inference_output.wav",
                seed=np.random.randint(0, 2**31 -1) # Use a random seed for each test run
            )
        except Exception as e:
            print(f"Error during inference test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping inference test.")

    if args.hf_upload_repo_id:
        upload_to_hf_hub(
            local_folder=target_output_mlx_dir,
            hf_repo_id=args.hf_upload_repo_id,
            hf_token=args.hf_token,
            commit_message_prefix=f"Add F5-TTS MLX model (q={q_bits}bit)" if q_bits else "Add F5-TTS MLX model (unquantized)"
        )
    else:
        print("Skipping upload to Hugging Face Hub.")

    print("\nScript finished successfully.")

if __name__ == "__main__":
    main()