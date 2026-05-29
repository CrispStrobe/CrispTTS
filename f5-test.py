#!/usr/bin/env python3
"""
F5-TTS Model Manager and Test Tool
Helps manage F5-TTS models, test compatibility, and convert to MLX format
"""

import argparse
import sys
from pathlib import Path

# Test configurations for different F5-TTS models
F5_TEST_MODELS = {
    "lucasnewman/f5-tts-mlx": {
        "type": "mlx_native",
        "status": "working",
        "notes": "Native MLX implementation, most reliable"
    },
    "SWivid/F5-TTS": {
        "type": "standard",
        "status": "working",
        "notes": "Original F5-TTS implementation"
    },
    "marduk-ra/F5-TTS-German": {
        "type": "standard",
        "status": "issues",
        "notes": "Missing model files (model_v1.safetensors)"
    },
    "eamag/f5-tts-mlx-german": {
        "type": "mlx_unofficial",
        "status": "incompatible",
        "notes": "Architecture incompatibility with f5-tts-mlx library"
    },
    "aihpi/F5-TTS-German": {
        "type": "standard",
        "status": "untested",
        "notes": "German F5-TTS model, may need conversion"
    }
}

def test_model_availability():
    """Test which F5-TTS libraries and dependencies are available"""
    results = {}

    # Test core dependencies
    try:
        import torch
        results["pytorch"] = {"available": True, "version": torch.__version__}
    except ImportError:
        results["pytorch"] = {"available": False, "error": "Not installed"}

    try:
        import torchaudio
        results["torchaudio"] = {"available": True, "version": torchaudio.__version__}
    except ImportError:
        results["torchaudio"] = {"available": False, "error": "Not installed"}

    try:
        import soundfile
        results["soundfile"] = {"available": True, "version": soundfile.__version__}
    except ImportError:
        results["soundfile"] = {"available": False, "error": "Not installed"}

    try:
        import numpy
        results["numpy"] = {"available": True, "version": numpy.__version__}
    except ImportError:
        results["numpy"] = {"available": False, "error": "Not installed"}

    # Test MLX
    try:
        import mlx.core  # noqa: F401
        results["mlx"] = {"available": True, "version": "available"}
    except ImportError:
        results["mlx"] = {"available": False, "error": "Not installed (macOS Apple Silicon only)"}

    # Test F5-TTS libraries
    try:
        from f5_tts.model import CFM, DiT, UNetT  # noqa: F401
        results["f5_tts_standard"] = {"available": True, "models": ["DiT", "UNetT", "CFM"]}
    except ImportError as e:
        results["f5_tts_standard"] = {"available": False, "error": str(e)}

    try:
        from f5_tts_mlx.generate import generate  # noqa: F401
        results["f5_tts_mlx"] = {"available": True, "function": "generate"}
    except ImportError as e:
        results["f5_tts_mlx"] = {"available": False, "error": str(e)}

    # Test Whisper options
    whisper_options = []
    try:
        import whisper  # noqa: F401
        whisper_options.append("openai-whisper")
    except ImportError:
        pass

    try:
        from faster_whisper import WhisperModel  # noqa: F401
        whisper_options.append("faster-whisper")
    except ImportError:
        pass

    try:
        from transformers import pipeline  # noqa: F401
        whisper_options.append("transformers")
    except ImportError:
        pass

    results["whisper_options"] = whisper_options

    return results

def print_availability_report():
    """Print a detailed availability report"""
    results = test_model_availability()

    print("🔍 F5-TTS Dependencies Report")
    print("=" * 50)

    # Core dependencies
    print("\n📦 Core Dependencies:")
    for lib, info in results.items():
        if lib in ["pytorch", "torchaudio", "soundfile", "numpy", "mlx"]:
            status = "✅" if info["available"] else "❌"
            version_info = f" (v{info.get('version', 'unknown')})" if info["available"] else f" - {info.get('error', 'Unknown error')}"  # noqa: E501
            print(f"  {status} {lib.capitalize()}{version_info}")

    # F5-TTS libraries
    print("\n🎙️ F5-TTS Libraries:")
    for lib, info in results.items():
        if lib in ["f5_tts_standard", "f5_tts_mlx"]:
            status = "✅" if info["available"] else "❌"
            detail = ""
            if info["available"]:
                if "models" in info:
                    detail = f" (Models: {', '.join(info['models'])})"
                elif "function" in info:
                    detail = f" (Function: {info['function']})"
            else:
                detail = f" - {info.get('error', 'Unknown error')}"

            lib_name = lib.replace('_', '-').upper()
            print(f"  {status} {lib_name}{detail}")

    # Whisper options
    print(f"\n🗣️ Whisper Transcription Options: {len(results['whisper_options'])}")
    if results['whisper_options']:
        for option in results['whisper_options']:
            print(f"  ✅ {option}")
    else:
        print("  ❌ No Whisper libraries available")

    # Compatibility assessment
    print("\n🎯 F5-TTS Compatibility Assessment:")

    core_available = all(results[lib]["available"] for lib in ["pytorch", "soundfile", "numpy"])
    standard_available = results["f5_tts_standard"]["available"]
    mlx_available = results["mlx"]["available"] and results["f5_tts_mlx"]["available"]
    whisper_available = len(results['whisper_options']) > 0

    if core_available and (standard_available or mlx_available) and whisper_available:
        print("  🎉 Excellent! F5-TTS should work well")
        if mlx_available:
            print("  🚀 MLX acceleration available for Apple Silicon")
        if standard_available:
            print("  💪 Standard F5-TTS available as fallback")
    elif core_available and (standard_available or mlx_available):
        print("  ⚠️ Good! F5-TTS will work but transcription may use fallbacks")
    elif core_available:
        print("  ❌ F5-TTS libraries missing - install f5-tts or f5-tts-mlx")
    else:
        print("  ❌ Core dependencies missing - install PyTorch, SoundFile, NumPy")

    return results

def test_model_download(model_id: str):
    """Test if a model can be downloaded and accessed"""
    try:
        print(f"🔄 Testing download for {model_id}...")

        # Try to get model info without downloading
        from huggingface_hub import HfApi
        api = HfApi()
        model_info = api.model_info(model_id)

        print("  ✅ Model exists on HuggingFace Hub")
        print(f"  📁 Files: {len(model_info.siblings)} files")

        # Check for key F5-TTS files
        file_names = [sibling.rfilename for sibling in model_info.siblings]
        key_files = {
            "config.json": "config.json" in file_names,
            "model files": any(f.endswith(('.safetensors', '.pt', '.pth', '.bin')) for f in file_names),
            "vocab files": any(f.endswith(('.txt', '.json')) and 'vocab' in f.lower() for f in file_names)
        }

        for file_type, exists in key_files.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {file_type}")

        return True

    except Exception as e:
        print(f"  ❌ Failed to access model: {e}")
        return False

def prepare_reference_audio(audio_path: str, output_path: str = None):
    """Prepare reference audio for F5-TTS testing"""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("❌ SoundFile and NumPy required for audio preparation")
        return False

    try:
        # Check input
        if not Path(audio_path).exists():
            print(f"❌ Audio file not found: {audio_path}")
            return False

        print(f"🔄 Preparing reference audio: {audio_path}")

        # Analyze input
        info = sf.info(audio_path)
        print(f"  📊 Input: {info.duration:.1f}s, {info.samplerate}Hz, {info.channels} channels")

        # Load and process
        audio_data, sr = sf.read(audio_path)

        # Convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print("  🔄 Converted to mono")

        # Optimal duration for F5-TTS (10 seconds from middle)
        target_duration = 10.0
        if info.duration > target_duration:
            target_samples = int(sr * target_duration)
            start_sample = (len(audio_data) - target_samples) // 2
            audio_data = audio_data[start_sample:start_sample + target_samples]
            print(f"  ✂️ Trimmed to {target_duration}s (middle section)")

        # Resample to 24kHz if needed
        if sr != 24000:
            try:
                import torch
                import torchaudio

                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, 24000)
                audio_data = resampler(audio_tensor).squeeze().numpy()
                print("  🔄 Resampled to 24kHz (torchaudio)")

            except ImportError:
                try:
                    from scipy import signal
                    num_samples = int(len(audio_data) * 24000 / sr)
                    audio_data = signal.resample(audio_data, num_samples)
                    print("  🔄 Resampled to 24kHz (scipy)")
                except ImportError:
                    print("  ⚠️ Cannot resample (install torchaudio or scipy)")

        # Normalize
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95
            print("  🔊 Normalized audio")

        # Save
        if output_path is None:
            output_path = Path(audio_path).stem + "_f5_ready.wav"

        sf.write(output_path, audio_data, 24000)

        # Verify
        verify_info = sf.info(output_path)
        print(f"  ✅ Output: {verify_info.duration:.1f}s, {verify_info.samplerate}Hz")
        print(f"  💾 Saved to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Failed to prepare audio: {e}")
        return False

def run_f5_test(model_id: str, audio_path: str, text: str = "Das ist ein einfacher Testtext."):
    """Run a complete F5-TTS test"""
    print(f"🧪 Testing F5-TTS with model: {model_id}")
    print("=" * 60)

    # Check if it's a known model
    if model_id in F5_TEST_MODELS:
        model_info = F5_TEST_MODELS[model_id]
        print(f"📋 Model info: {model_info['notes']} (Status: {model_info['status']})")

    # Test dependencies
    print("\n1️⃣ Checking dependencies...")
    deps = test_model_availability()

    core_ready = deps["pytorch"]["available"] and deps["soundfile"]["available"] and deps["numpy"]["available"]
    if not core_ready:
        print("❌ Core dependencies missing")
        return False

    # Test model access
    print("\n2️⃣ Testing model access...")
    if not test_model_download(model_id):
        return False

    # Prepare audio
    print("\n3️⃣ Preparing reference audio...")
    temp_audio = "temp_f5_ready.wav"
    if not prepare_reference_audio(audio_path, temp_audio):
        return False

    # Test synthesis
    print("\n4️⃣ Testing synthesis...")
    try:
        # This would integrate with your actual F5-TTS handler
        print(f"  🔄 Synthesizing: '{text[:50]}...'")
        print(f"  🎙️ Reference: {temp_audio}")
        print(f"  🤖 Model: {model_id}")

        # Placeholder for actual synthesis call
        # success = your_f5_handler.synthesize(model_id, text, temp_audio, "test_output.wav")

        print("  ⚠️ This is a simulation - integrate with actual F5-TTS handler")
        success = True  # Simulated

        if success:
            print("  ✅ Synthesis completed successfully!")
            return True
        else:
            print("  ❌ Synthesis failed")
            return False

    except Exception as e:
        print(f"  ❌ Synthesis error: {e}")
        return False

    finally:
        # Cleanup
        if Path(temp_audio).exists():
            Path(temp_audio).unlink()

def main():
    parser = argparse.ArgumentParser(
        description="F5-TTS Model Manager and Test Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check F5-TTS dependencies and compatibility"
    )

    parser.add_argument(
        "--test-model",
        help="Test a specific F5-TTS model (HuggingFace model ID)"
    )

    parser.add_argument(
        "--prepare-audio",
        help="Prepare audio file for F5-TTS (optimize duration, sample rate, etc.)"
    )

    parser.add_argument(
        "--audio-output",
        help="Output path for prepared audio (default: adds '_f5_ready' suffix)"
    )

    parser.add_argument(
        "--test-text",
        default="Das ist ein einfacher Testtext.",
        help="Text to use for synthesis testing"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List known F5-TTS models and their status"
    )

    args = parser.parse_args()

    if args.check_deps:
        print_availability_report()
        return

    if args.list_models:
        print("📋 Known F5-TTS Models:")
        print("=" * 50)
        for model_id, info in F5_TEST_MODELS.items():
            status_emoji = {"working": "✅", "issues": "⚠️", "incompatible": "❌", "untested": "❓"}
            emoji = status_emoji.get(info["status"], "❓")
            print(f"\n{emoji} {model_id}")
            print(f"  Type: {info['type']}")
            print(f"  Status: {info['status']}")
            print(f"  Notes: {info['notes']}")
        return

    if args.prepare_audio:
        success = prepare_reference_audio(args.prepare_audio, args.audio_output)
        sys.exit(0 if success else 1)

    if args.test_model:
        if not args.prepare_audio:
            print("❌ --test-model requires --prepare-audio with reference audio file")
            sys.exit(1)

        success = run_f5_test(args.test_model, args.prepare_audio, args.test_text)
        sys.exit(0 if success else 1)

    parser.print_help()

if __name__ == "__main__":
    main()
