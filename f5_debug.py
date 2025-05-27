#!/usr/bin/env python3
"""
F5-TTS MLX Debug Script - Output Path Fix
Test using the output_path parameter to save files instead of relying on return values
"""

import sys
import tempfile
import json
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

def test_f5_mlx_with_output_path():
    """Test F5-TTS MLX using the output_path parameter"""
    try:
        from f5_tts_mlx.generate import generate
        import soundfile as sf
        
        print("🧪 Testing F5-TTS MLX with output_path parameter...")
        
        # Create test audio
        test_audio = create_test_audio()
        if not test_audio:
            return False
        
        # Create output file path
        output_file = "f5_output_path_test.wav"
        
        try:
            print("🚀 Calling F5-TTS MLX with output_path...")
            
            # Test with output_path parameter
            params = {
                'generation_text': "Hello, this is a test of the output path parameter.",
                'model_name': "lucasnewman/f5-tts-mlx",
                'ref_audio_path': test_audio,
                'ref_audio_text': "This is a test reference audio.",
                'steps': 8,
                'cfg_strength': 1.0,
                'speed': 1.0,
                'sway_sampling_coef': -1.0,
                'estimate_duration': False,
                'output_path': output_file  # KEY: This should make it save to file
            }
            
            print(f"Parameters: {json.dumps(params, indent=2)}")
            
            result = generate(**params)
            
            print(f"✅ F5-TTS MLX call completed!")
            print(f"   Return value: {result}")
            print(f"   Type: {type(result)}")
            
            # Check if output file was created
            output_path = Path(output_file)
            if output_path.exists():
                file_size = output_path.stat().st_size
                print(f"🎉 SUCCESS: Output file created!")
                print(f"   File: {output_file}")
                print(f"   Size: {file_size} bytes")
                
                # Analyze the saved audio
                try:
                    info = sf.info(output_file)
                    duration = info.duration
                    sample_rate = info.samplerate
                    print(f"   Duration: {duration:.2f}s")
                    print(f"   Sample Rate: {sample_rate}Hz")
                    print(f"   Channels: {info.channels}")
                    
                    # Read and analyze audio data
                    audio_data, sr = sf.read(output_file)
                    import numpy as np
                    rms = np.sqrt(np.mean(audio_data**2))
                    peak = np.abs(audio_data).max()
                    print(f"   RMS: {rms:.4f}")
                    print(f"   Peak: {peak:.4f}")
                    
                    if duration > 0.1 and rms > 0.001:
                        print("✅ Audio appears to be valid!")
                        return True
                    else:
                        print("⚠️ Audio may be too short or silent")
                        return False
                        
                except Exception as e:
                    print(f"   Audio analysis failed: {e}")
                    return False
            else:
                print("❌ FAILED: No output file was created")
                print("💡 The output_path parameter may not be working as expected")
                return False
        
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
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_german_with_output_path():
    """Test with german.wav using output_path"""
    german_file = "./german.wav"
    
    if not Path(german_file).exists():
        print(f"❌ {german_file} not found")
        return False
    
    try:
        import soundfile as sf
        import numpy as np
        
        print(f"🎵 Testing german.wav with output_path...")
        
        # Prepare the audio
        audio_data, sr = sf.read(german_file)
        
        # Convert to mono and trim
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
        
        # Create output file
        output_file = "german_output_path_test.wav"
        
        print(f"📊 Prepared audio: {len(audio_data)/sr:.1f}s at {sr}Hz")
        
        # Test F5-TTS
        from f5_tts_mlx.generate import generate
        
        params = {
            'generation_text': "Das ist ein einfacher deutscher Testtext für die Sprachsynthese mit output_path.",
            'model_name': "lucasnewman/f5-tts-mlx",
            'ref_audio_path': temp_audio,
            'ref_audio_text': "Dies ist eine deutsche Referenzaufnahme für Tests.",
            'steps': 16,
            'cfg_strength': 1.5,
            'speed': 1.0,
            'output_path': output_file  # KEY: Save to specific file
        }
        
        print(f"🚀 Calling F5-TTS with german audio and output_path...")
        result = generate(**params)
        
        print(f"✅ Generation completed!")
        print(f"   Return value: {result}")
        
        # Check output file
        output_path = Path(output_file)
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"🎉 SUCCESS: German output file created!")
            print(f"   File: {output_file}")
            print(f"   Size: {file_size} bytes")
            
            try:
                info = sf.info(output_file)
                duration = info.duration
                print(f"   Duration: {duration:.2f}s")
                
                if duration > 0.5:
                    print("✅ German synthesis appears successful!")
                    return True
                else:
                    print("⚠️ German audio may be too short")
                    return False
            except Exception as e:
                print(f"   Analysis failed: {e}")
                return False
        else:
            print("❌ FAILED: German output file not created")
            return False
        
        # Cleanup
        Path(temp_audio).unlink()
        
    except Exception as e:
        print(f"❌ German test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🛠️ F5-TTS MLX Output Path Fix Test")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python f5_debug_output_path.py test      - Test with output_path parameter")
        print("  python f5_debug_output_path.py german    - Test german.wav with output_path")
        print("  python f5_debug_output_path.py both      - Run both tests")
        return
    
    command = sys.argv[1]
    
    if command == "test":
        success = test_f5_mlx_with_output_path()
        if success:
            print("\n🎉 The output_path parameter fix WORKS!")
            print("💡 Update your handler to use output_path instead of trying to capture return values")
        else:
            print("\n❌ The output_path parameter test failed")
            
    elif command == "german":
        success = test_german_with_output_path()
        if success:
            print("\n🎉 German synthesis with output_path WORKS!")
        else:
            print("\n❌ German synthesis with output_path failed")
            
    elif command == "both":
        print("🔍 Running both tests...\n")
        
        test1 = test_f5_mlx_with_output_path()
        print("\n" + "="*30 + "\n")
        test2 = test_german_with_output_path()
        
        print(f"\n📊 Results Summary:")
        print(f"   Basic test: {'✅ PASS' if test1 else '❌ FAIL'}")
        print(f"   German test: {'✅ PASS' if test2 else '❌ FAIL'}")
        
        if test1 and test2:
            print("\n🎉 ALL TESTS PASSED!")
            print("💡 The F5-TTS handler should now work correctly with output_path parameter")
        else:
            print(f"\n⚠️ Some tests failed - check the output above")
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()