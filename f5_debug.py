#!/usr/bin/env python3
"""
F5-TTS MLX Debug Script - Improved Version
Enhanced debugging to understand the None return issue and fix the handler
"""

import sys
import tempfile
import json
import inspect
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file for F5-TTS"""
    try:
        import numpy as np
        import soundfile as sf
        
        # Create 3 seconds of simple sine wave at 24kHz
        duration = 3.0
        sample_rate = 24000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Add some variation to make it more voice-like
        audio += np.sin(2 * np.pi * frequency * 2 * t) * 0.1
        audio += np.random.normal(0, 0.01, len(audio))  # Small amount of noise
        
        temp_file = tempfile.mktemp(suffix=".wav")
        sf.write(temp_file, audio, sample_rate)
        
        print(f"✅ Created test audio: {temp_file}")
        return temp_file
        
    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return None

def inspect_f5_mlx_function():
    """Inspect the F5-TTS MLX generate function to understand its behavior"""
    try:
        from f5_tts_mlx.generate import generate
        
        print("🔍 Inspecting F5-TTS MLX generate function...")
        
        # Get function signature
        sig = inspect.signature(generate)
        print(f"Function signature: {sig}")
        
        # Get docstring
        if generate.__doc__:
            print(f"Docstring: {generate.__doc__}")
        else:
            print("No docstring available")
        
        # Try to get source code (might not work for compiled modules)
        try:
            source = inspect.getsource(generate)
            print(f"Source code preview (first 500 chars):")
            print(source[:500] + "..." if len(source) > 500 else source)
        except:
            print("Source code not available (likely compiled)")
        
        return True
        
    except ImportError:
        print("❌ F5-TTS MLX not available for inspection")
        return False

def debug_f5_mlx_output_detailed():
    """Enhanced debugging of F5-TTS MLX output format"""
    try:
        from f5_tts_mlx.generate import generate
        import numpy as np
        
        print("🧪 Enhanced F5-TTS MLX debugging...")
        
        # Create test audio
        test_audio = create_test_audio()
        if not test_audio:
            return False
        
        # Test with very minimal parameters to isolate the issue
        print("🚀 Calling F5-TTS MLX with minimal params...")
        
        # Capture more details about the call
        generation_params = {
            'generation_text': "Hello, this is a test.",
            'model_name': "lucasnewman/f5-tts-mlx",
            'ref_audio_path': test_audio,
            'ref_audio_text': "This is a test reference.",
            'steps': 8,  # Very low steps for quick test
            'cfg_strength': 1.0,
            'speed': 1.0,
            'sway_sampling_coef': -1.0,
            'estimate_duration': False
        }
        
        print(f"Parameters: {json.dumps(generation_params, indent=2)}")
        
        try:
            # Call the function
            result = generate(**generation_params)
            
            print(f"✅ F5-TTS MLX call completed!")
            
            # Deep analysis of the result
            print(f"\n📊 Comprehensive Result Analysis:")
            print(f"  Result: {result}")
            print(f"  Type: {type(result)}")
            print(f"  Is None: {result is None}")
            print(f"  String representation: {str(result)}")
            print(f"  Repr: {repr(result)}")
            
            # Check for various attributes
            if result is not None:
                print(f"  Has __len__: {hasattr(result, '__len__')}")
                print(f"  Has __iter__: {hasattr(result, '__iter__')}")
                print(f"  Has shape: {hasattr(result, 'shape')}")
                print(f"  Has numpy: {hasattr(result, 'numpy')}")
                print(f"  Has __array__: {hasattr(result, '__array__')}")
                
                if hasattr(result, 'shape'):
                    print(f"  Shape: {result.shape}")
                elif hasattr(result, '__len__'):
                    try:
                        print(f"  Length: {len(result)}")
                    except Exception as e:
                        print(f"  Length error: {e}")
                
                if hasattr(result, 'dtype'):
                    print(f"  Dtype: {result.dtype}")
                
                # Try to get some information about the content
                if hasattr(result, '__iter__') and not isinstance(result, str):
                    try:
                        items = list(result)
                        print(f"  Iterable with {len(items)} items")
                        for i, item in enumerate(items[:3]):  # Show first 3 items
                            print(f"    [{i}]: {type(item)} - {item}")
                    except Exception as e:
                        print(f"  Iteration error: {e}")
            
            # Test various conversion strategies
            print(f"\n🔄 Conversion Strategy Tests:")
            
            if result is None:
                print("  ❌ Cannot convert None result")
                print("  💡 This suggests the F5-TTS MLX library has a bug where it")
                print("     performs the generation but fails to return the audio data")
                print("  💡 The model is likely working but the library's return mechanism is broken")
                return False
            
            # Strategy 1: Direct numpy conversion
            try:
                if hasattr(result, 'numpy'):
                    audio_data = result.numpy()
                    print(f"  ✅ result.numpy(): {type(audio_data)}, shape: {audio_data.shape}")
                    test_save_audio(audio_data, "strategy1_numpy.wav")
                else:
                    print(f"  ❌ No .numpy() method")
            except Exception as e:
                print(f"  ❌ result.numpy() failed: {e}")
            
            # Strategy 2: np.array conversion
            try:
                audio_data = np.array(result)
                print(f"  ✅ np.array(result): {type(audio_data)}, shape: {audio_data.shape}")
                if audio_data.size > 0:
                    test_save_audio(audio_data, "strategy2_array.wav")
                else:
                    print(f"    ⚠️ Array is empty")
            except Exception as e:
                print(f"  ❌ np.array(result) failed: {e}")
            
            # Strategy 3: Check if it's a tensor
            try:
                import torch
                if torch.is_tensor(result):
                    audio_data = result.detach().cpu().numpy()
                    print(f"  ✅ torch tensor conversion: {type(audio_data)}, shape: {audio_data.shape}")
                    test_save_audio(audio_data, "strategy3_torch.wav")
                else:
                    print(f"  ℹ️ Not a torch tensor")
            except Exception as e:
                print(f"  ❌ torch conversion failed: {e}")
            
            # Strategy 4: MLX array handling
            try:
                import mlx.core as mx
                if isinstance(result, mx.array):
                    audio_data = np.array(result)
                    print(f"  ✅ MLX array conversion: {type(audio_data)}, shape: {audio_data.shape}")
                    test_save_audio(audio_data, "strategy4_mlx.wav")
                else:
                    print(f"  ℹ️ Not an MLX array")
            except Exception as e:
                print(f"  ❌ MLX conversion failed: {e}")
            
            # Strategy 5: Check if it's a collection
            if isinstance(result, (tuple, list)):
                print(f"  📋 Result is {type(result)} with {len(result)} elements:")
                for i, item in enumerate(result):
                    print(f"    [{i}]: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"         Shape: {item.shape}")
                    if item is not None and hasattr(item, 'numpy'):
                        try:
                            item_data = item.numpy()
                            print(f"         Numpy shape: {item_data.shape}")
                            if item_data.size > 0:
                                test_save_audio(item_data, f"strategy5_item{i}.wav")
                        except Exception as e:
                            print(f"         Numpy conversion failed: {e}")
            
            return True
            
        finally:
            # Cleanup
            try:
                Path(test_audio).unlink()
            except:
                pass
        
    except ImportError:
        print("❌ F5-TTS MLX not available")
        return False
    except Exception as e:
        print(f"❌ Enhanced debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_audio(audio_data, filename):
    """Test saving audio data to file"""
    try:
        import soundfile as sf
        import numpy as np
        
        # Ensure it's a numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Ensure 1D
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Check if it has reasonable audio data
        if audio_data.size == 0:
            print(f"    ⚠️ {filename}: Empty audio data")
            return False
        
        duration = len(audio_data) / 24000
        if duration < 0.001:
            print(f"    ⚠️ {filename}: Audio too short ({duration:.6f}s)")
            return False
        
        # Normalize if needed
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        sf.write(filename, audio_data, 24000)
        
        file_size = Path(filename).stat().st_size
        print(f"    ✅ {filename}: Saved {duration:.3f}s, {file_size} bytes")
        
        # Quick audio analysis
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.abs(audio_data).max()
        print(f"       RMS: {rms:.4f}, Peak: {peak:.4f}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ {filename}: Save failed - {e}")
        return False

def investigate_f5_mlx_internals():
    """Try to investigate F5-TTS MLX internals"""
    try:
        import f5_tts_mlx
        print(f"🔍 F5-TTS MLX module investigation:")
        print(f"  Module file: {f5_tts_mlx.__file__}")
        print(f"  Module version: {getattr(f5_tts_mlx, '__version__', 'unknown')}")
        
        # Check what's in the module
        print(f"  Available attributes: {dir(f5_tts_mlx)}")
        
        # Check the generate module specifically
        try:
            import f5_tts_mlx.generate as gen_module
            print(f"  Generate module file: {gen_module.__file__}")
            print(f"  Generate module attributes: {dir(gen_module)}")
        except Exception as e:
            print(f"  Generate module inspection failed: {e}")
        
        return True
        
    except ImportError:
        print("❌ F5-TTS MLX not available for internal investigation")
        return False
    except Exception as e:
        print(f"❌ Internal investigation failed: {e}")
        return False

def test_with_german_audio_enhanced():
    """Enhanced test with german.wav file"""
    german_file = "./german.wav"
    
    if not Path(german_file).exists():
        print(f"❌ {german_file} not found")
        return False
    
    try:
        import soundfile as sf
        import numpy as np
        
        print(f"🎵 Enhanced testing with {german_file}")
        
        # Analyze the input file first
        audio_data, sr = sf.read(german_file)
        print(f"📊 Input file analysis:")
        print(f"  Duration: {len(audio_data)/sr:.2f}s")
        print(f"  Sample rate: {sr}Hz")
        print(f"  Channels: {audio_data.shape[1] if len(audio_data.shape) > 1 else 1}")
        print(f"  RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")
        print(f"  Peak: {np.abs(audio_data).max():.4f}")
        
        # Prepare the audio (same as in handler)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Take 5 seconds from middle
        if len(audio_data) > sr * 5:
            mid = len(audio_data) // 2
            half_duration = int(sr * 2.5)
            audio_data = audio_data[mid - half_duration:mid + half_duration]
        
        # Resample to 24kHz if needed
        if sr != 24000:
            try:
                import torch
                import torchaudio
                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, 24000)
                audio_data = resampler(audio_tensor).squeeze().numpy()
                sr = 24000
            except:
                print("⚠️ Cannot resample, using original sample rate")
        
        # Save prepared audio
        temp_audio = tempfile.mktemp(suffix="_prepared.wav")
        sf.write(temp_audio, audio_data, sr)
        
        print(f"📊 Prepared audio: {len(audio_data)/sr:.1f}s at {sr}Hz")
        
        # Test F5-TTS with more detailed monitoring
        from f5_tts_mlx.generate import generate
        
        print(f"🚀 Calling F5-TTS with german audio...")
        print("    Parameters:")
        params = {
            'generation_text': "Das ist ein einfacher deutscher Testtext für die Sprachsynthese.",
            'model_name': "lucasnewman/f5-tts-mlx",
            'ref_audio_path': temp_audio,
            'ref_audio_text': "Dies ist eine deutsche Referenzaufnahme für Tests.",
            'steps': 16,
            'cfg_strength': 1.5,
            'speed': 1.0
        }
        
        for key, value in params.items():
            print(f"      {key}: {value}")
        
        result = generate(**params)
        
        print(f"✅ Generation with german audio completed!")
        print(f"   Result: {result}")
        print(f"   Type: {type(result)}")
        print(f"   Is None: {result is None}")
        
        if result is None:
            print("❌ F5-TTS MLX returned None - this confirms the library bug")
            print("💡 The model appears to process the audio but fails to return results")
            print("💡 This is a known issue with the f5-tts-mlx library")
        else:
            # Try to save if we got something
            try:
                if hasattr(result, 'numpy'):
                    audio_out = result.numpy()
                else:
                    audio_out = np.array(result)
                
                if len(audio_out.shape) > 1:
                    audio_out = audio_out.flatten()
                
                if audio_out.size > 0:
                    sf.write("german_enhanced_output.wav", audio_out, 24000)
                    print(f"💾 Saved: german_enhanced_output.wav ({len(audio_out)/24000:.2f}s)")
                else:
                    print("❌ Generated audio is empty")
                    
            except Exception as save_error:
                print(f"❌ Save failed: {save_error}")
        
        # Cleanup
        Path(temp_audio).unlink()
        
        return result is not None
        
    except Exception as e:
        print(f"❌ Enhanced German audio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🛠️ F5-TTS MLX Enhanced Debug Tool")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python f5_debug_improved.py inspect    - Inspect F5-TTS MLX function")
        print("  python f5_debug_improved.py internals  - Investigate F5-TTS MLX internals")
        print("  python f5_debug_improved.py output     - Enhanced output format debugging")
        print("  python f5_debug_improved.py german     - Enhanced test with german.wav")
        print("  python f5_debug_improved.py all        - Run all tests")
        return
    
    command = sys.argv[1]
    
    if command == "inspect":
        inspect_f5_mlx_function()
    elif command == "internals":
        investigate_f5_mlx_internals()
    elif command == "output":
        debug_f5_mlx_output_detailed()
    elif command == "german":
        test_with_german_audio_enhanced()
    elif command == "all":
        print("🔍 Running all diagnostic tests...\n")
        investigate_f5_mlx_internals()
        print("\n" + "="*50 + "\n")
        inspect_f5_mlx_function()
        print("\n" + "="*50 + "\n")
        debug_f5_mlx_output_detailed()
        print("\n" + "="*50 + "\n")
        test_with_german_audio_enhanced()
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()