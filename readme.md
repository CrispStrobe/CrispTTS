# CrispTTS: Modular German Text-to-Speech Synthesizer

CrispTTS is a versatile command-line Text-to-Speech (TTS) tool designed for synthesizing German speech using a variety of popular local and cloud-based TTS engines. Its modular architecture allows for easy maintenance and straightforward addition of new TTS handlers.

NOTE: This is in very early experimental / work in progress state. Currently these models are BROKEN: dia, llasa per xml, coqui_css10_de_vits, maybe nemo. (It proved a bit difficult to make up a consistent python environment with all libraries at once, and nemo was therefore postponed for now.) They might be fixed later if my time allows...

## Features

- **Multiple TTS Engine Support**:
  - Microsoft Edge TTS (Cloud-based, requires `edge-tts`)
  - Piper (Local, ONNX-based, requires `piper-tts`)
  - Orpheus 
    - GGUF (Local, requires `llama-cpp-python`)
    - Orpheus via LM Studio API (Requires running LM Studio server)
    - Orpheus via Ollama API (Requires running Ollama server with Orpheus model)
  - OuteTTS (Local, with LlamaCPP or Hugging Face Transformers backend, requires `outetts` and its dependencies)
  - SpeechT5 (Local, German fine-tune via Hugging Face Transformers)
  - FastPitch (Local, German via NeMo and Hugging Face)
  - `mlx-audio` for several models
    - Bark** (Local, optimized for Apple Silicon, requires `mlx-audio`):
      - Uses MLX-converted models (e.g., from `mlx-community/bark-small`)
      - Voice prompts are fetched from a separate repository (e.g., `suno/bark-small`) via an included monkeypatch, enabling a wide range of voices
- **Command-Line Interface**: Easy-to-use CLI for listing models, getting voice info, and performing synthesis
- **Text Input Flexibility**: Synthesize text directly from the command line or from various file types (`.txt`, `.md`, `.html`, `.pdf`, `.epub`)
- **Customizable Output**: Save audio to specified files (typically `.wav` or `.mp3`)
- **Direct Playback**: Option to play synthesized audio directly
- **Voice Selection**: Override default voices/speakers for most models
- **Model Parameter Tuning**: Pass JSON-formatted parameters to fine-tune model behavior
- **Comprehensive Testing**:
  - `--test-all`: Test all configured models with default voices
  - `--test-all-speakers`: Test all models with all their pre-configured available voices
- **Modular Design**:
  - `config.py`: Centralized model configurations and global settings
  - `utils.py`: Shared helper functions (text extraction, audio I/O, etc.)
  - `handlers/`: Dedicated Python package with individual modules for each TTS engine's logic
  - `main.py`: Main CLI entry point
- **Logging**: Configurable logging levels for debugging and monitoring
- **Automatic Patching**: Includes necessary runtime monkeypatches for some libraries (e.g., for `mlx-audio` Bark voice loading, VLLM Triton placeholder issues) to enhance compatibility and functionality

## Project Structure

```
crisptts_project/
├── main.py                     # Main CLI application script
├── config.py                   # Model configurations and global constants
├── utils.py                    # Shared utility functions and classes
├── decoder.py                  # User-provided decoder for Orpheus models (if used)
├── handlers/                   # Package for individual TTS engine handlers
│   ├── __init__.py             # Makes 'handlers' a package, exports handler functions
│   ├── edge_handler.py         # Edge TTS cloud service handler
│   ├── piper_handler.py        # Piper TTS (ONNX) handler
│   ├── orpheus_gguf_handler.py # Local Orpheus GGUF model handler
│   ├── orpheus_api_handler.py  # Handlers for LM Studio and Ollama API
│   ├── outetts_handler.py      # OuteTTS model handler
│   ├── speecht5_handler.py     # SpeechT5 model handler
│   ├── nemo_handler.py         # NeMo FastPitch handler
│   ├── coqui_tts_handler.py    # Coqui TTS handler (for XTTS, VITS etc.)
│   ├── kartoffel_handler.py    # Orpheus "Kartoffel" Transformers handler
│   ├── llasa_hybrid_handler.py # LLaSA Hybrid handler
│   └── mlx_audio_handler.py    # Handler for mlx-audio library (e.g., Bark)
├── requirements.txt            # Python package dependencies
└── README.md                   # This documentation file
```

## Setup and Installation

### Prerequisites

- Python and `pip` for installing packages
- For `mlx-audio` based models: Apple Silicon Mac is required for GPU acceleration

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
   A `requirements.txt` file is provided. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: Some libraries like PyTorch, NeMo, LlamaCPP, and `mlx-audio` can have specific installation needs depending on your OS and hardware (e.g., CUDA for Nvidia GPUs, Metal for Apple Silicon). Please refer to their official documentation if you encounter issues.
   > Ensure you have `ffmpeg` installed and available in your system's PATH if you encounter issues with audio file format conversions or direct playback (some underlying libraries might need it).

4. **Environment Variables** (Optional but Recommended):
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

## Troubleshooting & Notes

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