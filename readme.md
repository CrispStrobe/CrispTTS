# CrispTTS: Modular German Text-to-Speech Synthesizer

CrispTTS is a versatile command-line Text-to-Speech (TTS) tool designed for synthesizing German speech using a variety of popular local and cloud-based TTS engines. Its modular architecture allows for easy maintenance and straightforward addition of new TTS handlers.

### Part of the Crisp ecosystem

| Project | Role |
|---|---|
| **[Susurrus](https://github.com/CrispStrobe/Susurrus)** | Python GUI + CLI — 30+ ASR, 12 TTS, translation |
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | C++ ASR/TTS engine — 26+ ASR, 20+ TTS backends, ggml inference |
| **CrispTTS** | This repo — Python TTS CLI with 37+ handlers |
| **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** | Flutter transcription app — desktop + mobile |

NOTE: This is in experimental / work in progress state. Some Python-only models may be broken due to dependency conflicts. The CrispASR-based handlers (`crispasr_*`) are the most reliable — they use native C++ inference with no Python ML dependencies.

## Features

- **37+ TTS Engine Support**:
  - **CrispASR native C++ engines** (16 backends, auto-download, no Python ML deps):
    - Kokoro (multilingual, Apache 2.0)
    - Orpheus + Kartoffel-Orpheus DE (19 German speakers, llama3.2 license)
    - Qwen3-TTS (voice cloning + voice design, Apache 2.0)
    - Chatterbox (CFM synthesis, MIT)
    - VibeVoice TTS (voice cloning)
    - IndexTTS (zero-shot cloning, Apache 2.0)
    - VoxCPM2 (48 kHz, 30 languages, Apache 2.0)
    - F5-TTS (flow-matching, voice cloning, Apache 2.0)
    - MeloTTS (VITS2, 44.1 kHz, MIT)
    - Piper (250+ community voices, 30+ languages — faster than Python Piper)
    - BananaMind-TTS (Tacotron-lite + HiFi-GAN, en/de)
    - Dots.TTS (Qwen2.5 LLM + DiT + BigVGAN, 48 kHz, CAM++ voice cloning)
    - CosyVoice3 (multi-GGUF: LLM+flow+CAM+++HiFT, voice cloning)
    - CSM/Sesame (Llama backbone + Mimi codec, causal mode, voice cloning)
    - OmniVoice (Qwen3 masked iterative, 600+ languages, voice cloning)
    - MOSS-TTS-Local (4B transformer + codec-v2, 48 kHz)
  - Microsoft Edge TTS (cloud-based, requires `edge-tts`)
  - Coqui TTS (XTTS v2, VITS, etc.)
  - Piper (local ONNX, requires `piper-tts`)
  - Orpheus GGUF (local, requires `llama-cpp-python`)
  - Orpheus via LM Studio / Ollama API
  - OuteTTS (LlamaCPP or HF backend)
  - SpeechT5 (German fine-tune via HF Transformers)
  - FastPitch (German via NeMo)
  - mlx-audio (Bark, Kokoro, Dia — Apple Silicon)
  - LLaSA (hybrid, German, multilingual variants)
  - F5-TTS (MLX/PyTorch)
  - Kokoro ONNX (lightweight)
  - TTS.cpp (GGUF models)
  - Zonos (acoustic conditioning)
  - Chatterbox Python (Kartoffelbox)
- **AI Audio Watermarking & Provenance**:
  - WavMark neural watermark (MIT license — code + model weights; `pip install wavmark`)
  - Spread-spectrum watermark (always on, imperceptible, ~38 dB SNR)
  - AudioSeal neural watermark (optional upgrade via `pip install audioseal` or CrispASR GGUF)
  - WAV LIST/INFO, MP3 ID3v2, FLAC Vorbis comment, and Opus/OGG metadata marking audio as AI-generated
  - C2PA content credentials signed by default for WAV/MP3/FLAC/M4A (core dep;
    c2pa-audio / CrispASR used as fast paths when present)
  - Voice-cloning consent gate (`--i-have-rights` CLI / `"i_have_rights": true` API)
  - Spoken AI disclaimer prepended to voice-cloned audio, in all 24 EU official
    languages plus zh/ja/ko (`--list-disclosure-langs`), bundled for offline use
  - Persistent consent audit log at `~/.cache/crisptts/consent_audit.log`,
    owner-only, auto-pruned, with `--consent-log-erase` for GDPR Art. 17
- **CrispASR Integration**:
  - `--verify`: ASR roundtrip verification of TTS output quality
  - `--translate`: Pre-synthesis translation (EN→DE via m2m100/MadLad)
  - `--speech-speed`: Rate multiplier (maps to CrispASR `--pace`)
  - `--trim-silence`: Remove leading/trailing silence from output
  - `--tts-steps`: Diffusion model inference steps (quality vs speed)
  - `--tts-language`: Override language for multilingual models
  - `--pitch-shift`: Pitch shift in Hz for FastPitch backends
  - `--instruct`: Natural-language voice descriptions (Qwen3-TTS VoiceDesign)
  - `--stream`: Stream audio playback during synthesis
  - `--output-sample-rate`: Resample output to target sample rate
- **OpenAI-Compatible API Server** (`--server`):
  - `POST /v1/audio/speech` — drop-in replacement for OpenAI TTS
  - `GET /v1/audio/models` — list all configured models
  - Voice-cloning consent gate (returns 403 if `i_have_rights` missing)
  - All responses watermarked + metadata-injected
- **Text Input Flexibility**: Synthesize from CLI, `.txt`, `.md`, `.html`, `.pdf`, `.epub`
- **Smart Text Chunking**: Automatic sentence-boundary splitting for long texts
- **SSML-lite**: Supports `<break>`, `<prosody rate>`, `<say-as>`, `<phoneme>` tags in input text
- **Customizable Output**: Save audio to `.wav`, `.mp3`, `.flac`, `.m4a` or `.opus`
  (`.opus`/`.ogg` carry no C2PA manifest, so they require a neural watermark —
  `pip install 'crisptts[robust]'`)
- **Direct Playback**: Play synthesized audio immediately
- **Voice Selection**: Override default voices/speakers for most models
- **Model Parameter Tuning**: JSON-formatted parameters for fine-tuning
- **Comprehensive Testing**:
  - `--test-all`: Test all models with default voices
  - `--test-all-speakers`: Test all models with all configured voices
  - 433 unit and live tests
- **Modular Design**: `config.py` + `utils.py` + `handlers/` + `main.py`
- **Logging**: Configurable logging levels
- **Automatic Patching**: Runtime monkeypatches for library compatibility

## Project Structure

```
crisptts_project/
├── main.py                     # Main CLI application script
├── config.py                   # Model configurations and global constants
├── utils.py                    # Shared utility functions and classes
├── watermark.py                # Audio watermarking, metadata, consent gate, C2PA
├── chunking.py                 # Smart sentence-boundary text splitting
├── server.py                   # OpenAI-compatible HTTP API server
├── ssml.py                     # SSML-lite tag preprocessor
├── cache.py                    # Synthesis result caching (LRU)
├── decoder.py                  # User-provided decoder for Orpheus models (if used)
├── handlers/                   # Package for individual TTS engine handlers
│   ├── __init__.py             # Makes 'handlers' a package, exports handler functions
│   ├── crispasr_handler.py     # CrispASR native C++ TTS (10 backends)
│   ├── edge_handler.py         # Edge TTS cloud service handler
│   ├── piper_handler.py        # Piper TTS (ONNX) handler
│   ├── orpheus_gguf_handler.py # Local Orpheus GGUF model handler
│   ├── orpheus_api_handler.py  # Handlers for LM Studio and Ollama API
│   ├── outetts_handler.py      # OuteTTS model handler
│   ├── speecht5_handler.py     # SpeechT5 model handler
│   ├── nemo_handler.py         # NeMo FastPitch handler
│   ├── coqui_tts_handler.py    # Coqui TTS handler (for XTTS, VITS etc.)
│   ├── kartoffel_handler.py    # Orpheus "Kartoffel" Transformers handler
│   ├── kokoro_onnx_handler.py  # Kokoro (multilingual but no German) ONNX handler
│   ├── llasa_hybrid_handler.py # LLaSA Hybrid handler
│   ├── tts_cpp_handler.py      # TTS.cpp handler supporting GGUF models
│   ├── f5_tts_handler.py       # F5-TTS handler (MLX/PyTorch)
│   ├── zonos_handler.py        # Zonos acoustic conditioning handler
│   ├── chatterbox_handler.py   # Chatterbox/Kartoffelbox handler
│   └── mlx_audio_handler.py    # Handler for mlx-audio library (e.g., Bark)
├── tests/                      # Unit and integration tests
├── requirements.txt            # Python package dependencies
├── pyproject.toml              # Project metadata and build config
└── README.md                   # This documentation file
```

## Setup and Installation

### Prerequisites

- Python 3.10+ and `pip` for installing packages
- For `mlx-audio` based models: Apple Silicon Mac is required for GPU acceleration
- For `TTS.cpp` a C++ compiler and CMake are required to build the engine

### Installation Steps

1. **Clone/Download Files**
   ```bash
   git clone https://github.com/CrispStrobe/CrispTTS
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Optional feature groups:
   ```bash
   pip install crisptts[robust]          # RECOMMENDED: robust neural watermark (MIT)
   pip install crisptts[watermark-mit]   # WavMark neural watermark (MIT license)
   pip install crisptts[dev]             # ruff, bandit, pytest
   ```

   C2PA signing (`c2pa-python`), container metadata (`mutagen`), `soundfile`
   and `pydub` are **core** dependencies, not extras — without them the
   AI-provenance marking cannot be applied and CrispTTS fails closed rather
   than emitting unmarked synthetic audio. The `metadata` and `provenance`
   extras are kept only so existing install commands keep working.

   > **Note**: Some libraries like PyTorch, NeMo, LlamaCPP, and `mlx-audio` can have specific installation needs depending on your OS and hardware (e.g., CUDA for Nvidia GPUs, Metal for Apple Silicon). Please refer to their official documentation if you encounter issues.
   > Ensure you have `ffmpeg` installed and available in your system's PATH if you encounter issues with audio file format conversions or direct playback (some underlying libraries might need it).

4. **Install and Build Engine-Specific Dependencies** (required for certain handlers):

    For TTS.cpp: Clone and build the TTS.cpp project separately.
    ```bash
    git clone https://github.com/mmwillet/TTS.cpp.git
    cd TTS.cpp
    cmake -B build
    cmake --build build --config Release
    cd ..
    ```

    # Update the `tts_cpp_executable_path` in config.py to point to ./TTS.cpp/build/cli
    For kokoro-onnx: Install the Python package and download model files.
    ```bash
    pip install kokoro-onnx
    ```
    # Download the .onnx model and voices.bin from the kokoro-onnx GitHub releases page.
    # Update the paths in config.py to point to your downloaded files.

5. **Environment Variables** (Optional but Recommended):
   - **`HF_TOKEN`**: If you need to download models from gated or private Hugging Face repositories, set this environment variable with your Hugging Face API token:
     ```bash
     export HF_TOKEN="your_huggingface_token_here"
     ```
   - `GGML_METAL_NDEBUG=1`: Set automatically by `main.py` to reduce verbose Metal logs from `llama-cpp-python` on macOS.

## Configuration (`config.py`)

The `config.py` file is central to defining which TTS models are available and their default settings.

- **`GERMAN_TTS_MODELS` Dictionary**: This is the primary configuration structure. Each key is a unique `MODEL_ID` used in the CLI. The value is a dictionary containing:
  - `"handler_function_key"` (Optional, defaults to `MODEL_ID`): The key used to look up the synthesis function in `handlers.ALL_HANDLERS`
  - Specific parameters for that model (e.g., `model_repo_id`, `default_voice_id`, API URLs, `onnx_repo_id`, etc.)
  - `"notes"`: A brief description of the model

- **`mlx-audio` Bark Configuration Example**:
  To use the `mlx-audio` Bark model, your configuration might look like this, enabling the dual-source strategy for voice prompts (main model from `mlx-community`, voice NPYs from `suno`):
  ```python
  "mlx_audio_bark_de": {
      "handler_function_key": "mlx_audio",
      "mlx_model_path": "mlx-community/bark-small", # Main MLX model
      # Voice prompts will be fetched by the patched handler from "suno/bark-small"
      "default_voice_id": "v2/de_speaker_3", 
      "available_voices": ["v2/de_speaker_0", "v2/de_speaker_1", "v2/de_speaker_3", "..."],
      "lang_code": "de",
      "sample_rate": 24000,
      "notes": "mlx-audio (Bark) with main model from mlx-community/bark-small and voices from suno/bark-small (via patch)."
  },
  ```

- **Global Constants**: API URLs, default voice names, and sample rates are also defined here
- **Adding/Modifying Models**: To add a new variation of an existing engine or a completely new engine (after creating its handler), you would add a new entry to `GERMAN_TTS_MODELS`

## Usage (`main.py`)

All interactions are done through `main.py` from your project's root directory.

### Basic Command Structure

```bash
python main.py [ACTION_FLAG | --model-id <MODEL_ID> [OPTIONS]]
# or using the --backend shortcut for CrispASR engines:
python main.py --backend kokoro --input-text "Hello" --output-file out.wav
```

### CLI Reference

#### Primary Actions

| Flag | Description |
|------|-------------|
| `--list-models` | List all configured TTS models with their notes |
| `--voice-info MODEL_ID` | Show available voices/speakers for a model |
| `--test-all` | Test all models with default voices (requires `--input-text` or `--input-file`) |
| `--test-all-speakers` | Test all models with ALL configured voices |
| `--skip-models M1 M2 ...` | Skip specific model IDs during `--test-all` / `--test-all-speakers` |
| `--detect-watermark FILE` | Detect AI-generated watermark in a WAV file and report confidence |
| `--server` | Run as HTTP server with OpenAI-compatible endpoints |
| `--check` | With `--list-models`: probe CrispASR backends for availability |

#### Synthesis Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id MODEL_ID` | — | TTS model to use (see `--list-models` for choices) |
| `--backend NAME` | — | Shortcut for CrispASR backends (e.g., `kokoro`, `piper`, `dots-tts`) |
| `--input-text TEXT` | — | Text to synthesize (mutually exclusive with `--input-file`) |
| `--input-file PATH` | — | Input file: `.txt`, `.md`, `.html`, `.pdf`, `.epub` |
| `--output-file PATH` | — | Save audio to file (format detected from extension: `.wav`, `.mp3`, `.flac`, `.opus`) |
| `--output-dir DIR` | `tts_test_outputs` | Output directory for `--test-all` / `--test-all-speakers` |
| `--play-direct` | off | Play audio immediately after synthesis |
| `--german-voice-id ID` | model default | Override voice/speaker (name, ID, or path to `.wav` for cloning) |
| `--model-params JSON` | — | JSON string of model-specific parameters, e.g. `'{"temperature":0.7}'` |
| `--speech-speed FLOAT` | `1.0` | Speech rate multiplier (>1 = faster, <1 = slower) |
| `--trim-silence` | off | Remove leading/trailing silence from output |
| `--tts-steps N` | backend default | Diffusion/flow-matching inference steps (quality vs. speed) |
| `--tts-language LANG` | model default | Override language code for multilingual models (e.g. `de`, `en`, `zh`, `ja`) |
| `--pitch-shift HZ` | `0` | Pitch offset in Hz (positive = higher, negative = lower) |
| `--instruct TEXT` | — | Natural-language voice description for VoiceDesign models (Qwen3-TTS) |
| `--output-sample-rate HZ` | native | Resample output to target sample rate (e.g. `16000`, `22050`, `44100`) |
| `--stream` | off | Stream audio playback during synthesis (CrispASR backends only) |
| `--ref-text TEXT` | — | Transcript of reference voice audio for inline voice cloning (TADA, dots-tts) |
| `--no-spoken-disclaimer` | off | Skip the AI-disclosure spoken prefix on voice-cloned audio |
| `--lexicon TSV_PATH` | — | Custom word→phoneme TSV file for CrispASR pronunciation |
| `--batch` | off | Split input at blank lines, produce numbered output files |
| `--jobs N` | `1` | Concurrent synthesis jobs for `--batch` mode |
| `--normalize` | off | Peak-normalize output audio to -3 dB |

#### CrispASR Integration

| Flag | Default | Description |
|------|---------|-------------|
| `--verify` | off | Run ASR on output for roundtrip quality verification |
| `--verify-backend NAME` | `parakeet` | ASR backend for `--verify` (e.g. `parakeet`, `whisper`) |
| `--translate` | off | Translate input text before synthesis |
| `--translate-from LANG` | `en` | Source language for translation |
| `--translate-to LANG` | `de` | Target language for translation |
| `--translate-backend NAME` | `m2m100` | Translation backend (`m2m100` or `madlad`) |

#### Watermarking & Provenance

| Flag | Default | Description |
|------|---------|-------------|
| `--no-watermark` | off | Disable **all** marking layers: audio watermark, metadata, C2PA (debug only) |
| `--allow-unmarked` | off | Deliver output even if marking fails or is undetectable |
| `--accept-marking-responsibility` | off | Required for any provenance opt-out; logged as `[MARKING]` |
| `--watermark-model PATH` | — | Path to AudioSeal GGUF model for neural watermarking |
| `--i-have-rights` | off | Consent attestation for voice-cloning models (required) |
| `--disclosure-lang LANG` | model's language | Language of the spoken AI disclosure on cloned audio |
| `--list-disclosure-langs` | — | List the languages the spoken disclosure is available in |
| `--consent-log-prune` | — | Drop consent audit entries past the retention window |
| `--consent-log-erase [SUBJECT]` | — | Erase consent audit entries (GDPR Art. 17) |
| `--c2pa-cert PEM` | — | X.509 PEM certificate for C2PA content credentials |
| `--c2pa-key PEM` | — | PEM private key for C2PA content credentials |

#### Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | off | Start the HTTP API server |
| `--server-host ADDR` | `127.0.0.1` | Server bind address |
| `--server-port PORT` | `8880` | Server port |
| `--rate-limit N` | `10` | Max synthesis requests per minute per IP (0=unlimited) |
| `--warm-up MODEL_ID` | — | Pre-synthesize at startup to warm model caches |

#### Model-Specific Parameters (`--model-params`)

Parameters are passed as a JSON string. Available keys depend on the backend:

| Key | Backends | Description |
|-----|----------|-------------|
| `temperature` | Most LLM-based | Sampling temperature (higher = more varied) |
| `seed` | All CrispASR | Random seed for reproducible output |
| `top_p` | LLM-based | Nucleus sampling threshold |
| `repetition_penalty` | LLM-based | Penalize token repetition |
| `tts_steps` | Diffusion/flow | Number of inference steps |
| `speech_speed` | CrispASR | Rate multiplier (same as `--speech-speed`) |
| `pitch_shift` | FastPitch | Hz offset (same as `--pitch-shift`) |
| `top_k` | LLM-based | Top-K candidates |
| `min_p` | LLM-based | Min-P threshold |
| `cfg_weight` | Chatterbox | Classifier-free guidance weight |
| `cfg_scale` | Chatterbox, F5, TADA | CFG scale for acoustic conditioning |
| `exaggeration` | Chatterbox | Emotion exaggeration factor |
| `length_scale` | VITS | Duration scaling factor |
| `speaker_name` | Multi-speaker | Speaker name override |
| `speaker_id` | Piper | Multi-speaker model ID |
| `do_sample` | TADA | 0=greedy, 1=sample talker |
| `num_candidates` | TADA | Acoustic flow-matching candidates |
| `num_steps` | TADA, flow-matching | FM/diffusion inference steps |
| `noise_temp` | TADA | FM noise temperature |
| `noise_scale` | Piper | VITS variance |
| `noise_w` | Piper | Stochastic duration predictor |
| `max_speech_tokens` | Chatterbox | Max AR tokens |
| `tts_speed` | OmniVoice | Target-length speed estimate |

Example:
```bash
python main.py --model-id crispasr_chatterbox \
  --model-params '{"cfg_weight": 3.0, "exaggeration": 0.7, "temperature": 0.8}' \
  --input-text "Emotional speech test." --output-file chatterbox.wav
```

### Common Examples

**List all available models:**
```bash
python main.py --list-models
```

**Get information about voices for a specific model:**
```bash
python main.py --voice-info edge
python main.py --voice-info mlx_audio_bark_de
```

**Synthesize text using a specific model:**
```bash
python main.py --model-id edge --input-text "Hallo, wie geht es Ihnen heute?" --output-file hallo_edge.mp3 --play-direct
```

**Synthesize text using mlx-audio Bark (German):**
```bash
python main.py --model-id mlx_audio_bark_de --input-text "Das ist ein Test mit Bark auf Apple Silicon." --output-file bark_test_de.wav
```

**Use a specific German voice (if supported by the model):**
```bash
python main.py --model-id edge --input-text "Ein Test mit einer anderen Stimme." --german-voice-id de-DE-ConradNeural --output-file conrad_test.mp3
```
Check `--voice-info <MODEL_ID>` for available voice IDs/formats for that model.

**Synthesize text from a file:**
```bash
python main.py --model-id piper_local --input-file ./my_text.txt --output-file piper_output.wav
```
Supported input file types: `.txt`, `.md`, `.html`, `.pdf`, `.epub`.

**Use model-specific parameters (as a JSON string):**
```bash
python main.py --model-id orpheus_gguf --input-text "Ein Test." --model-params "{\"temperature\": 0.8, \"n_gpu_layers\": -1}" --output-file orpheus_custom.wav
```

**Test all configured models with default voices:**
```bash
python main.py --input-text "Dies ist ein kurzer Test für alle Modelle." --test-all --output-dir ./test_results
```

**Test all models with all their configured available voices/speakers:**
```bash
python main.py --input-text "Ein Test für alle Stimmen." --test-all-speakers --output-dir ./test_results_all_speakers
```

**Speech speed and pitch control:**
```bash
python main.py --model-id crispasr_kokoro --input-text "Schneller sprechen." --speech-speed 1.3 --output-file fast.wav
python main.py --model-id crispasr_kokoro --input-text "Höher." --pitch-shift 50 --output-file high.wav
```

**Silence trimming and resampling:**
```bash
python main.py --model-id crispasr_kokoro --input-text "Test." --trim-silence --output-sample-rate 16000 --output-file trimmed_16k.wav
```

**VoiceDesign — generate voices from text descriptions:**
```bash
python main.py --model-id crispasr_qwen3_tts_voicedesign --instruct "A calm elderly man" --input-text "Hallo" --output-file calm.wav
```

**Streaming playback (hear audio while it generates):**
```bash
python main.py --model-id crispasr_kokoro --input-text "Dies wird sofort abgespielt." --stream
```

**Run as OpenAI-compatible API server:**
```bash
python main.py --server --server-port 8880
# Then: curl -X POST http://localhost:8880/v1/audio/speech \
#   -H "Content-Type: application/json" \
#   -d '{"model":"crispasr_kokoro","input":"Hallo Welt","voice":"af_heart"}' \
#   --output speech.wav
```

**Voice cloning (with consent attestation):**
```bash
# CLI
python main.py --model-id coqui_xtts_v2_de_clone --i-have-rights \
  --input-text "Hallo" --german-voice-id ref_voice.wav --output-file cloned.wav

# API (include i_have_rights in request body)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"crispasr_f5_tts","input":"Hallo","voice":"ref.wav","i_have_rights":true}' \
  --output cloned.wav
```

**SSML-lite markup:**
```bash
python main.py --backend kokoro --output-file out.wav --input-text \
  'Hello. <break time="500ms"/> <prosody rate="fast">This part is fast.</prosody> Normal again.'
```

**Batch synthesis with parallel jobs:**
```bash
python main.py --backend kokoro --batch --jobs 4 \
  --input-file book.txt --output-dir chapters/
```

**Audio normalization:**
```bash
python main.py --backend kokoro --input-text "Test" --normalize --output-file normalized.wav
```

**Change Logging Level (for debugging):**
```bash
python main.py --model-id edge --input-text "Debug Test." --loglevel DEBUG
```

**Override API URLs (for API-based models like Orpheus LM Studio/Ollama):**
```bash
python main.py --model-id orpheus_lm_studio --input-text "Hallo API" --lm-studio-api-url http://localhost:5000/v1/completions
python main.py --model-id orpheus_ollama --input-text "Hallo Ollama" --ollama-api-url http://localhost:11223/api/generate --ollama-model-name my-orpheus-ollama-model
```

## Supported TTS Engines

Refer to the output of `python main.py --list-models` for the currently configured models and their notes. The script supports integration with:

- CrispASR native C++ (16 backends: Kokoro, Orpheus, Qwen3-TTS, Chatterbox, VibeVoice, IndexTTS, VoxCPM2, F5-TTS, MeloTTS, Piper, BananaMind, Dots.TTS, CosyVoice3, CSM/Sesame, OmniVoice, MOSS-TTS-Local)
- Microsoft Edge TTS
- Piper TTS
- Orpheus GGUF (via llama-cpp-python)
- Orpheus via LM Studio API
- Orpheus via Ollama API
- OuteTTS (LlamaCPP and Hugging Face ONNX backends)
- SpeechT5 (Hugging Face Transformers)
- FastPitch (NeMo / Hugging Face)
- Coqui TTS (XTTS, VITS, etc.)
- Orpheus "Kartoffel" (Transformers-based)
- LLaSA Hybrid (Experimental MLX + PyTorch)
- mlx-audio (e.g., Bark for Apple Silicon)
- F5-TTS (MLX/PyTorch)
- Zonos (acoustic conditioning)
- Chatterbox/Kartoffelbox (Python)

## Adding New TTS Handlers

The modular design makes it easy to add support for new TTS engines:

1. **Create a New Handler File**: In the `handlers/` directory, create a new Python file (e.g., `my_new_tts_handler.py`)

2. **Implement Synthesis Function**: Inside this file, write a function that takes the standard arguments: `model_config`, `text`, `voice_id_override`, `model_params_override`, `output_file_str`, `play_direct`. This function should handle all aspects of using the new TTS engine.

3. **Update `handlers/__init__.py`**: Import your new function and add it to the `ALL_HANDLERS` dictionary.

4. **Update `config.py`**: Add a new entry to `GERMAN_TTS_MODELS` for your new engine.

## `decoder.py` Requirement for Orpheus

For all Orpheus-based models (GGUF local, LM Studio API, Ollama API, Kartoffel), this project relies on a user-provided `decoder.py` file located in the project's root directory. This file must contain a function:

```python
def convert_to_audio(multiframe_tokens: list[int], total_token_count: int) -> bytes | None:
    # Your implementation here to convert Orpheus token IDs to raw PCM audio bytes
    # (16-bit, 24000 Hz, mono)
    # Return audio frame bytes, or None/empty bytes on error.
    pass
```

If this file or function is missing, Orpheus models will not produce audible output, and a placeholder will be used.

## Voice & model licensing

CrispTTS is a synthesis **tool** — it does **not** bundle or redistribute
any voice/model weights. Each model is downloaded at runtime from its
upstream repository into a local cache (Piper voices from
[`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices),
Coqui models via the `TTS` library, etc.). You obtain the weights directly
from the source, under that source's terms.

**You are responsible for honouring each voice's license** for whatever
you produce. Licenses vary per voice and are *not* uniform across
`rhasspy/piper-voices` — check the upstream `MODEL_CARD` (and, where it
only says "See URL", the underlying dataset), because the card fields are
self-reported. Notable cases among the German Piper voices CrispTTS lists:

- **thorsten**, **kerstin** — CC0 (public domain).
- **eva_k**, **karlsson**, **ramona** — [M-AILABS](https://github.com/i-celeste-aurora/m-ailabs-dataset),
  BSD-style (commercial OK; retain the copyright notice).
- **mls** — CC-BY 4.0 (attribution required).
- **pavoque** — **CC BY-NC-SA 4.0 (non-commercial)** — do not use the
  output commercially.

For a redistributable, pre-curated **permissive-only** GGUF set (the same
voices minus the non-commercial/restricted ones, converted for the
CrispASR/CrisperWeaver native runtime), see
[`cstr/piper-voices-GGUF`](https://huggingface.co/cstr/piper-voices-GGUF).

## Audio Watermarking & Provenance

CrispTTS automatically marks all synthesized audio as AI-generated using a multi-layered provenance system.

Every path that writes an audio file — CLI synthesis, `--batch`, `--test-all` and API server responses — marks it through the single `watermark.mark_audio_file()` entry point, in every supported format (WAV, MP3, FLAC, Opus/OGG). Marking is the **last** step, after trimming, normalization, resampling and the spoken disclaimer, so nothing downstream can strip it. CrispASR C++ backends watermark at the binary level; all other handlers are watermarked in Python post-synthesis.

Marking follows the file that was **written**, not the one that was requested. Most handlers force their own container regardless of the extension you ask for — the Edge handler writes `.mp3`, most local ones `.wav` — so `--output-file out.wav --model-id edge` produces `out.mp3`, and that is the file disclosed, marked, verified and played.

**Generation is gated on sufficient marking.** Three rules:

1. **Preflight.** Before any model is loaded, CrispTTS checks that the requested
   output *can* be marked — supported container, codec dependencies present. If
   not, synthesis is refused up front, so unmarkable audio is never produced.
2. **Verified, not assumed.** After marking, the watermark is read back, and
   the output is delivered only if at least one *robust* layer is confirmed:
   the watermark detected above threshold, or a C2PA manifest signed. Container
   metadata alone is never sufficient — any transcode strips it. So an
   undetectable watermark is fatal for FLAC and Opus, which cannot carry a
   manifest, while a WAV or MP3 may still ship on its manifest alone; the
   `MarkResult` reports which layers actually applied. This catches audio that
   is silent or otherwise unable to carry a mark, and it applies to CrispASR
   backends too: their binary-level watermark is verified rather than taken on
   trust.
3. **Watermark floor.** `--no-watermark` is honoured only when another robust
   layer (a C2PA manifest) still marks the output. When the container cannot
   carry one, the watermark is forced back on — no path can emit a fully
   unmarked AI file. This mirrors CrispASR's watertight-CLI guarantee.

Any provenance opt-out (`--no-watermark`, `--allow-unmarked`,
`--no-spoken-disclaimer`) additionally requires `--accept-marking-responsibility`,
which is recorded as a `[MARKING]` audit line next to `[CONSENT]`.

### Layers

| Layer | What | Status | Install |
|-------|------|--------|---------|
| **WavMark** | Neural watermark (MIT license, 16-bit payload, >38 dB SNR) | Auto-detected (preferred) | `pip install 'crisptts[robust]'` |
| **Spread-spectrum** | Frequency-domain watermark (32 bins, alpha=0.08) | Always active | Built-in (numpy) |
| **AudioSeal** | Neural watermark (Meta, 16-bit message, sample-rate aware) | Auto-detected | `pip install audioseal` |
| **WAV/MP3/FLAC/Opus metadata** | LIST/INFO, ID3v2, Vorbis comments — `AI_GENERATED=true` | Always active | Built-in (mutagen, core dep) |
| **C2PA credentials** | Signed provenance manifests (`trainedAlgorithmicMedia`) — self-signed unless you supply a certificate | **Always active** (WAV/MP3/FLAC/M4A) | Built-in (c2pa-python, core dep) |
| **Spoken disclaimer** | AI disclosure prepended to voice-cloned audio, in the model's language | Auto for cloning | Built-in |
| **Consent gate** | Voice-cloning attestation + persistent audit logging | Required for cloning | Built-in |

**Watermark backend priority**: WavMark (MIT) > AudioSeal (Python) > CrispASR GGUF > spread-spectrum (always-on fallback). Neural backends are lazy-loaded on first synthesis — `--list-models` and `--help` remain instant.

### Robustness of the built-in fallback

The built-in spread-spectrum watermark places a 32-bin comb inside the speech
band (~1.5–4.8 kHz), matching CrispASR's `wm_params`. Measured on 20 s of real
speech (`german.wav`, 44.1 kHz), detection threshold 0.65:

| Condition | Detection confidence |
|-----------|---------------------|
| Immediately after embedding | 0.84 |
| After a 44.1k → 16k → 44.1k resample | 0.78 |
| After an MP3 round-trip (128 kbps / 64 kbps) | 0.81 / 0.81 |
| After additive noise at 30 dB SNR | 0.81 |
| Unwatermarked human recording (false-positive check) | 0.44 |

Measured SNR of the embed is ~39.5 dB on speech.

These numbers supersede an earlier table measured on the pre-#260 wideband comb
(0.94 after embed but **0.63** after a resample — below threshold, i.e. the mark
was lost). Moving the comb into the speech band lowered the immediate reading
while making it survive resampling and transcoding, and made it ~6 dB quieter.
`CRISPASR_WATERMARK_LEGACY=1` restores the old band for A/B against older files;
detection always sweeps both, so previously-marked audio still verifies.

This is why **C2PA signing is on by default** rather than opt-in: for WAV,
MP3, FLAC and M4A output the signed manifest, not the spread-spectrum
watermark, is the durable and interoperable provenance layer. The watermark is
what survives having the manifest stripped; the manifest is what survives a
resample. Neither alone is sufficient for every downstream path, so both are
applied.

For Opus/OGG, which C2PA cannot sign, a neural backend is **required** rather
than recommended — see below. Install it also if audio in any container may be
transcoded *and* stripped; it is MIT-licensed and far more robust:

```bash
pip install 'crisptts[robust]'
```

CrispTTS logs a warning once per run when the spread-spectrum backend is the
only robust layer present. WavMark is an extra rather than a core dependency
because it pulls in PyTorch (~2 GB).

### C2PA content credentials

Every WAV and MP3 output is signed with a C2PA manifest asserting
`digitalSourceType: trainedAlgorithmicMedia` — the standard, machine-readable
claim that content is AI-generated — plus a `c2pa.training-mining` assertion
opting the audio out of AI training.

By default CrispTTS signs with the **bundled development certificate** in
`c2pa_dev_cert.py`. Its private key is public, by design: it is in the source
tree and in every wheel. A manifest signed with it proves the file has not been
altered since signing, but it does **not** attest to who produced the file and
will **not** validate against C2PA trust lists. CrispTTS never reports such a
signature as trusted — `MarkResult.c2pa_signer` is `"self-signed"`, and it logs
a warning once per run.

For a credential others can attribute to you, obtain a certificate from a
C2PA-recognised authority and pass `--c2pa-cert` / `--c2pa-key` (or set
`C2PA_CERT_PATH` / `C2PA_KEY_PATH`); the signer is then reported as
`"ca-issued"`. The certificate must be a **chain** (leaf followed by its CA)
and the key must be **PKCS#8** — c2pa-python rejects a bare self-signed leaf
and a SEC1 key. `scripts/make_dev_cert.sh` shows a working profile.

**WAV, MP3, FLAC and M4A all carry a manifest.** FLAC and M4A were excluded
until it turned out that only `sign_file()` refuses them — c2pa-python's
streaming `Builder.sign()` signs both, and the result reads back
`validation_state: Valid` with the `trainedAlgorithmicMedia` assertion intact.
CrispTTS uses the streaming path for every container, so those two gained a
manifest they were previously denied. A test signs each listed format for
real, so the set cannot drift back into overclaiming.

**Opus/OGG cannot.** c2pa-rs does not list it among its supported types at
all, and every format string tried (`opus`, `audio/opus`, `ogg`, `audio/ogg`,
`application/ogg`) returns `NotSupported`. A detached `.c2pa` sidecar can be
produced for it, but a manifest that travels as a separate file is a
provenance record you hold, not a mark on the content — CrispTTS does not
count it. So the audio watermark is Opus/OGG's only robust layer, which is why
`--no-watermark` is overridden for it, and why:

> **Opus and OGG output requires a neural watermark backend.** With only the
> built-in spread-spectrum comb installed, synthesis to those containers is
> refused up front — a fixed-key comb as the *sole* robust layer is not
> marking that is "robust as far as technically feasible". Install one with
> `pip install 'crisptts[robust]'`, choose a manifest-carrying container, or
> take the duty on yourself with
> `--allow-unmarked --accept-marking-responsibility`.

#### Signing backends

Three signers are tried in order, selectable with
`CRISPTTS_C2PA_BACKEND=auto|python|audio|crispasr|off` (default `auto`):

| Order | Backend | Availability |
|-------|---------|--------------|
| 1 | [`c2pa-audio`](https://github.com/CrispStrobe/c2pa-audio) | Native, fast. Not on PyPI — build from source |
| 2 | `c2pa-python` | **Always** — core dependency, and the only path where CrispTTS controls the manifest |

CrispASR is deliberately *not* a signing backend. Checked against crispasr
0.8.25: `--c2pa-cert` / `--c2pa-key` configure signing of its **own** synthesis
output and there is no flag that signs an existing file. What it does instead
is better — see *Upstream manifests* below.

**Every native result is verified before it is accepted.** c2pa-audio's
`sign_wav()` takes a certificate and a key but no manifest, so the library
decides its own assertions — and a manifest without `trainedAlgorithmicMedia`
marks a file as *unaltered* rather than as *AI-generated*. CrispTTS therefore
reads the manifest back after any native signer runs; if the AI assertion is
missing, that result is discarded and c2pa-python re-signs with a manifest that
carries it. `watermark.manifest_asserts_ai(path)` exposes the same check.

#### Upstream manifests are preserved, not overwritten

CrispASR signs its TTS output during synthesis, by default, with a manifest that
already asserts `trainedAlgorithmicMedia`. Every marking step rewrites the file,
and any rewrite breaks that manifest's hash — injecting the WAV `LIST/INFO`
chunk alone takes a CrispASR output from `validation_state: Valid` to `Invalid`.

So when a file already carries a manifest asserting AI generation, CrispTTS
leaves it exactly as it is: no metadata injection, no re-signing, reported as
`c2pa:preserved`. This keeps the upstream signer's identity (`softwareAgent:
CrispASR TTS`) instead of replacing it with ours, and removes the failure mode
where a broken-then-not-repaired manifest made an untampered file look tampered.

CrispTTS also no longer *claims* the CrispASR watermark as a layer. Measured on
crispasr 0.8.25 kokoro output, CrispTTS's spread-spectrum detector reads 0.44 —
its noise floor — so `audio-watermark:upstream` is reported only when
verification actually detects a mark.

### Voice cloning safety

Voice-cloning models require explicit consent attestation before synthesis is allowed:

- **CLI**: `--i-have-rights` flag required (synthesis blocked without it)
- **API**: `"i_have_rights": true` in request body (returns 403 without it)
- **Audit log**: written to stderr AND `~/.cache/crisptts/consent_audit.log`,
  including a SHA-256 digest of the reference recording. `--test-all` logs the
  attestation too, not just the gate check. See *Audit log retention* below.
- **Fails closed**: if the gate cannot be evaluated at all (the `watermark`
  module is missing), synthesis is refused rather than allowed through. An
  unknown cloning status is treated as cloning, not as permission.

**Detection**, strongest signal first:

1. **A reference recording as the voice** (`.wav`, `.mp3`, `.flac`, `.ogg`,
   `.opus`, `.m4a`). Handing the system somebody's voice to imitate is the
   cloning act itself, so this always gates — even for a model declared
   `voice_cloning: false`.
2. **The model's explicit `voice_cloning` key in `config.py`.** Every shipped
   model sets it, in both directions; a test asserts this so a new backend
   cannot be added without answering the question.
3. **Handler key / model-ID keywords**, as a fallback for user-supplied model
   dicts that predate the explicit key. This tier fails *open*, which is why
   tier 2 exists.

**Spoken disclosure** — prepended to cloned output, in **all 24 EU official
languages** plus Chinese, Japanese and Korean (27 total; `--list-disclosure-langs`
prints them). Art. 50 governs content placed on the EU market, so a disclosure
an EU audience can understand means any EU official language — a German
sentence in front of Greek audio discloses nothing to a Greek listener. The
German and English wording is kept identical to Susurrus's `disclosure.spoken`
string, so the Crisp projects disclose in the same words.

**Which language is used**, in order of precedence:

1. `--disclosure-lang` (CLI) or `"disclosure_lang"` (API) — an explicit choice
2. The model's declared `language` in `config.py`
3. German, the default — **with a warning**

Step 3 is a fallback, not a decision. Around half the shipped cloning
backends are multilingual (CosyVoice3, OmniVoice, IndexTTS, Qwen3-TTS,
LLaSA-Multilingual, OuteTTS, Spark, VoxCPM2, MOSS, VibeVoice, F5), and for
those the output language is a property of the *input text*, not of the model
— so it cannot be derived from the config at all. They declare
`"language": "multilingual"`, which CrispTTS treats as *unknown* rather than
silently substituting German, and warns that you should pass
`--disclosure-lang`. A test asserts every cloning model declares a `language`
key, so a new backend cannot skip the question.

Sources, in order:

1. CrispASR kokoro, if the binary is available — local, no network
2. Edge TTS, if installed — needs network
3. **A pre-rendered clip bundled in `crisptts_assets/`** — no backend, no model
   download, no network, no configuration

Tier 3 is why disclosure does not fail on an offline machine. It is a real
spoken sentence in the right language, so it counts as a disclosure. All 27
languages ship a clip (~2 MB total in the wheel), so offline disclosure works
in every one of them, not just German. Regenerate with
`python scripts/make_disclosure_assets.py` after editing `DISCLAIMER_TEXTS`;
a test fails if any language lacks a bundled clip.

Only if all three fail does CrispTTS fall back to a tone marker, which is
**refused**: three beeps are an audible signal, not a disclosure a listener can
understand. In that case the output is discarded rather than delivered, and
`--no-spoken-disclaimer --accept-marking-responsibility` is the way to take the
Art. 50(4) duty on explicitly.

### EU AI Act: what this tool does, and what you must still do

Article 50 of Regulation (EU) 2024/1689 sits in Chapter IV and **has applied
since 2 August 2026** under Art. 113 — it is in force now, not upcoming.
Article 4 (AI literacy, Chapter I) has applied since **2 February 2025**.
Releasing under an open-source licence does **not** exempt Article 50 —
Art. 2(12) expressly carves it back in. Check the current consolidated text
before relying on these dates; the Regulation has been subject to amendment
proposals since adoption.

**What CrispTTS does for you** (provider-side, Art. 50(2)):

- Marks every synthetic audio output in a machine-readable format
- Signs WAV/MP3 output with an interoperable C2PA manifest by default, so the
  provenance claim is readable by any C2PA verifier and not only by Crisp tools
- Fails closed rather than emitting unmarked audio
- Reports honestly what was applied (`MarkResult`, server `X-CrispTTS-*`
  headers), and never presents a bundled-certificate signature as trusted
- Gates voice cloning behind an attestation and logs it with a digest of the
  reference recording — and refuses to synthesize at all if that gate cannot
  be evaluated
- Never plays audio to a listener before it has been marked and verified
- Prepends a spoken AI disclosure to voice-cloned output, in any of the 24 EU
  official languages, and refuses to deliver cloned audio without one

**What remains your responsibility as the deployer** (Art. 50(4)):

- Disclosing that content is artificially generated where you publish it —
  a watermark is machine-readable, not a disclosure to the audience
- **Choosing the disclosure language.** CrispTTS defaults to the model's
  declared language and warns when it cannot determine one, but only you know
  what language your audience speaks. Pass `--disclosure-lang` when using a
  multilingual model — a disclosure the audience cannot understand does not
  discharge the Art. 50(4) duty
- Obtaining genuine consent from anyone whose voice you clone. The
  `--i-have-rights` flag is an unverified self-attestation; it records your
  claim, it does not establish a legal basis
- Checking the voice/model licences you use (see the licensing section above)
- GDPR: a cloned voice is personal data, and the consent audit log contains
  reference-audio paths — see *Audit log retention* below
- **Art. 4 (AI literacy)**: ensuring the people operating this tool understand
  what it does and what its output is. In practice, for CrispTTS, that means
  whoever runs it should have read this section

**Not applicable, having been checked:** CrispTTS performs no biometric
categorisation and no emotion *recognition* (Kartoffelbox's emotion control is
emotion *synthesis*), so Art. 5 prohibited practices do not engage; it is not
an Annex III high-risk system; and single-purpose TTS models are not
general-purpose AI models, so Chapter V obligations do not attach to the
models converted by `convert_f5_to_mlx.py`.

### Audit log retention

`[CONSENT]` and `[MARKING]` lines are written to
`~/.cache/crisptts/consent_audit.log`. That file records reference-audio
*paths*, which routinely contain personal names, so it is personal data:

- Created `0600`, owner-only, rather than at the umask default
- Entries older than **730 days** are pruned automatically on every append
  (GDPR Art. 5(1)(e) storage limitation). Set
  `CRISPTTS_CONSENT_LOG_RETENTION_DAYS` to change the window, `0` to disable
- `--consent-log-prune` prunes on demand
- `--consent-log-erase [SUBJECT]` handles an Art. 17 erasure request: with a
  reference-audio path or `ref_sha256` digest it removes only that speaker's
  lines; with no argument it erases the whole log

Lines with no parseable timestamp are kept — an unreadable record is not
evidence that it has expired.

C2PA manifests signed with the bundled development certificate prove the file
is unaltered since signing, but will **not** validate against C2PA trust lists.
Supply `--c2pa-cert` / `--c2pa-key` for a credential others can verify.

This is a summary of how the implementation is intended to map onto the
regulation, not legal advice.

### Compliance comparison across the Crisp ecosystem

| Feature | CrispTTS | CrispASR | CrisperWeaver |
|---------|----------|----------|---------------|
| Spread-spectrum watermark | numpy (Python) | C++ header-only | Dart LSB + native FFI |
| WavMark neural watermark (MIT) | Python (wavmark) | — | — |
| AudioSeal neural watermark | Python + crispasr GGUF | C++ ggml (GGUF) | via CrispASR FFI |
| WAV LIST/INFO metadata | ISFT + ICMT | ISFT + ICMT | ISFT + ICMT + IART + ICRD |
| MP3 ID3v2 tags | TXXX (AI_GENERATED) | TXXX (AI_GENERATED) | TXXX (AI_GENERATED) |
| FLAC/Opus metadata | Vorbis comments (mutagen) | — | — |
| C2PA content credentials | c2pa-python by default; c2pa-audio / CrispASR as fast paths, each verified | c2pa-c (compile-time) | — |
| Spoken AI disclaimer | CrispASR kokoro / Edge TTS / bundled clips, 27 languages (all 24 EU official); refuses if unavailable | Native TTS (cached) | Beep marker |
| Voice-cloning consent gate | CLI + API (403) | CLI + server JSON | GDPR Art. 9(2)(a) consent files |
| Consent audit logging | stderr + `consent_audit.log` | `[CONSENT]` stderr | `[CONSENT]` log + `.consent.json` |
| Post-embed verification | detect after save | detect after save | detect after embed |
| Watermark detection CLI | `--detect-watermark` | `--detect-watermark` | detect in service |
| Cross-project detection | Yes (shared PRNG key) | Yes (shared PRNG key) | Yes (via CrispASR FFI) |

### Marking-enforcement strength across the ecosystem

The layers above answer *what* is applied. This answers *what happens when it
doesn't work* — which is where the projects genuinely differ:

| Enforcement | CrispTTS | CrispASR | Susurrus | CrisperWeaver |
|---|---|---|---|---|
| Refuse **before** generating if unmarkable | Yes | — | — | — |
| Marking verified after embedding | Yes (gates) | — | — | — |
| Unmarkable output discarded | Yes | — | No (warns) | No |
| Watermark floor (opt-out only if C2PA carries it) | Yes | Yes | No | No |
| Opt-out requires marking attestation | Yes | Yes | No (single flag) | No |
| `[MARKING]` audit line | Yes | Yes | — | — |
| Non-WAV outputs marked | Yes | Yes | **No** (WAV only) | — |
| Silent no-op on short audio | Refused | — | — | Yes (<4608 samples) |
| Playback marked+verified before it is heard | Yes | — | — | — |
| Consent gate fails closed if unevaluable | Yes | — | — | — |
| Disclosure in all 24 EU official languages | Yes | — | — | — |
| Audit-log retention limit + erasure command | Yes | — | — | — |

CrispASR contributed the watermark floor and the attestation gate, which
CrispTTS adopts here. CrispTTS adds preflight refusal and verification-as-gate,
which the others do not yet have — in Susurrus and CrisperWeaver a failed or
skipped mark currently produces a warning and the audio still ships.

### Usage

```bash
# Default: spread-spectrum watermark + metadata (no extra deps)
python main.py --model-id edge --input-text "Hallo" --output-file out.mp3

# With WavMark neural watermark (MIT, preferred — survives transcoding)
pip install 'crisptts[robust]'
python main.py --model-id edge --input-text "Hallo" --output-file out.mp3

# With C2PA content credentials
pip install c2pa-python
python main.py --c2pa-cert cert.pem --c2pa-key key.pem --model-id edge --input-text "Hallo" --output-file out.mp3

# Voice-cloning models require consent attestation (spoken disclaimer auto-prepended)
python main.py --model-id coqui_xtts_v2_de_clone --i-have-rights --input-text "Hallo" --output-file out.wav

# Multilingual cloning model: say which language the disclosure should be in,
# because the model config cannot know what language your text is
python main.py --model-id crispasr_cosyvoice3_tts --i-have-rights \
  --disclosure-lang zh --input-text "你好" --output-file out.wav

# What languages can the spoken disclosure be in?
python main.py --list-disclosure-langs

# Detect watermark in existing audio
python main.py --detect-watermark out.wav

# GDPR housekeeping on the consent audit log
python main.py --consent-log-prune                 # drop entries past retention
python main.py --consent-log-erase /refs/alice.wav # Art. 17, one speaker
python main.py --consent-log-erase                 # Art. 17, everything

# Disable ALL marking layers (debug only — you take on the Art. 50 responsibility)
python main.py --no-watermark --model-id edge --input-text "Hallo" --output-file out.mp3

# Keep output even if marking fails (default is to discard it and exit non-zero)
python main.py --allow-unmarked --model-id edge --input-text "Hallo" --output-file out.mp3
```

### Detection (Python API)

```python
from watermark import watermark_detect
import soundfile as sf

pcm, sr = sf.read("out.wav", dtype="float32")
confidence = watermark_detect(pcm, sample_rate=sr)
print(f"Watermark confidence: {confidence:.3f}")  # >0.65 = AI-generated
```

### Cross-compatibility

The spread-spectrum watermark uses the same PRNG seed (`0x437269737041535F`), FFT parameters, and bin selection as CrispASR's C++ implementation and CrisperWeaver's native FFI path. Audio watermarked by any project in the ecosystem can be detected by the others.

## API Server

CrispTTS includes an OpenAI-compatible HTTP server for integration with applications that use the OpenAI TTS SDK.

```bash
# Start the server
python main.py --server --server-port 8880

# Or run directly
python server.py --host 0.0.0.0 --port 8880
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/speech` | Synthesize audio (OpenAI-compatible) |
| GET | `/v1/audio/models` | List available models and voices |
| GET | `/health` | Health check |

### Request format (POST /v1/audio/speech)

```json
{
  "model": "crispasr_kokoro",
  "input": "Hallo, wie geht es Ihnen?",
  "voice": "af_heart",
  "response_format": "wav",
  "speed": 1.0,
  "i_have_rights": false
}
```

The `i_have_rights` field is required (and must be `true`) for voice-cloning models. Omit it or set to `false` for non-cloning models.

Response: audio bytes with `Content-Type` and `Content-Disposition: attachment` headers. Features:
- All output marked, in every response format. Provenance headers report what
  was actually applied — `X-CrispTTS-Watermarked` is `true` only when marking
  really happened, alongside `X-CrispTTS-Watermark-Backend`,
  `X-CrispTTS-Watermark-Confidence` and `X-CrispTTS-Provenance-Layers`
- A response that cannot be marked is a `500`, never unmarked audio
- Voice-cloning models return 403 unless `i_have_rights` is set. The consent
  gate runs **before** the cache lookup, so cached audio cannot bypass it
- Concurrent requests handled via threaded server
- Rate limiting: 10 requests/minute/IP (configurable via `--rate-limit`)
- Synthesis caching: identical requests served from cache (`X-CrispTTS-Cache: hit`).
  Cache keys include the marking mode, so unmarked audio can never be served
  to a marking-enabled request
- Enhanced `/health`: reports loaded handlers, memory RSS, registered backends

## Troubleshooting & Notes

**espeak-ng for Kokoro**: The Kokoro backend requires `espeak-ng` for phonemization. Install via:
```bash
pip install py-espeak-ng     # installs espeak-ng CLI to ~/.local/bin
# or system-wide: apt install espeak-ng
```

**CrispASR voice paths**: The CrispASR binary auto-downloads models but voice packs need full paths for older binary versions. Use the cached path directly:
```bash
python main.py --model-id crispasr_kokoro \
  --german-voice-id ~/.cache/crispasr/kokoro-voice-af_heart.gguf \
  --input-text "Test" --output-file out.wav
```

**Missing Libraries**: If a specific TTS engine fails, ensure you have installed all its required libraries via `pip install -r requirements.txt` and any extra steps mentioned in their documentation.

**mlx-audio Bark Specifics**:
- This handler currently requires the main MLX model to be from a repository like `mlx-community/bark-small` (which should provide MLX-compatible `.safetensors` or model files)
- The voice prompts (speaker embeddings) are fetched from `suno/bark-small` by default (due to an included monkey patch in `mlx_audio_handler.py`) which has a comprehensive set of speaker prompts as separate `.npy` files. This dual-source setup is necessary because `mlx-community/bark-small` has limited voice prompt files in the required format
- If mlx-audio's `load_model` function reports "No safetensors found" for the main `mlx_model_path`, you may need to convert the target Bark model to MLX format using `python -m mlx_audio.tts.convert` and point `mlx_model_path` to the local converted directory. The voice prompt patch in the handler is designed to work with either an HF repo ID or a local path for `mlx_model_path` when determining how to fetch/locate the `.npy` prompts from `suno/bark-small` or a `speaker_embeddings` subfolder

**API Keys/Servers**: API-based models require the respective servers (LM Studio, Ollama) to be running and accessible.

**Model Downloads**: First-time use of a model that needs to be downloaded from Hugging Face Hub might take some time. Ensure you have an internet connection. Set `HF_TOKEN` for gated models.

**Verbose Output**: Use `--loglevel DEBUG` for detailed diagnostic information if you encounter issues.

**RAM Usage**: Local GGUF and large Transformer models can be memory-intensive. Ensure your system has sufficient RAM.

**Paths**: When providing paths for `--input-file`, `--output-file`, or speaker WAV files (`--german-voice-id`), use appropriate relative or absolute paths.
