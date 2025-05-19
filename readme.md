# CrispTTS: Modular German Text-to-Speech Synthesizer

CrispTTS is a versatile command-line Text-to-Speech (TTS) tool designed for synthesizing German speech using a variety of popular and local TTS engines. Its modular architecture allows for easy maintenance and straightforward addition of new TTS handlers.

## Features

* **Multiple TTS Engine Support**:
    * Microsoft Edge TTS (Cloud-based, requires `edge-tts`)
    * Piper (Local, ONNX-based, requires `piper-tts`)
    * Orpheus GGUF (Local, requires `llama-cpp-python` and user-provided `decoder.py`)
    * Orpheus via LM Studio API (Requires running LM Studio server and user-provided `decoder.py`)
    * Orpheus via Ollama API (Requires running Ollama server with Orpheus model and user-provided `decoder.py`)
    * OuteTTS (Local, with LlamaCPP or Hugging Face Transformers backend, requires `outetts` and its dependencies)
    * SpeechT5 (Local, German fine-tune via Hugging Face Transformers)
    * FastPitch (Local, German via NeMo an
d Hugging Face)
* **Command-Line Interface**: Easy-to-use CLI for listing models, getting voice info, and performing synthesis.
* **Text Input Flexibility**: Synthesize text directly from the command line or from various file types (`.txt`, `.md`, `.html`, `.pdf`, `.epub`).
* **Customizable Output**: Save audio to specified files (typically `.wav` or `.mp3`).
* **Direct Playback**: Option to play synthesized audio directly.
* **Voice Selection**: Override default voices/speakers for most models.
* **Model Parameter Tuning**: Pass JSON-formatted parameters to fine-tune model behavior.
* **Comprehensive Testing**:
    * `--test-all`: Test all configured models with default voices.
    * `--test-all-speakers`: Test all models with all their pre-configured available voices.
* **Modular Design**:
    * `config.py`: Centralized model configurations and global settings.
    * `utils.py`: Shared helper functions (text extraction, audio I/O, etc.).
    * `handlers/`: Dedicated Python package with individual modules for each TTS engine's logic.
    * `main.py`: Main CLI entry point.
* **Logging**: Configurable logging levels for debugging and monitoring.

## Project Structure

crisptts_project/
|-- main.py                     # Main CLI application script
|-- config.py                   # Model configurations and global constants
|-- utils.py                    # Shared utility functions and classes
|-- decoder.py                  # User-provided decoder for Orpheus models
|-- handlers/                   # Package for individual TTS engine handlers
|   |-- init.py             # Makes 'handlers' a package, exports handler functions
|   |-- edge_handler.py
|   |-- piper_handler.py
|   |-- orpheus_gguf_handler.py
|   |-- orpheus_api_handler.py  # For LM Studio and Ollama
|   |-- outetts_handler.py
|   |-- speecht5_handler.py
|   |-- nemo_handler.py
|   |-- ... (other handlers can be added here)
|-- requirements.txt            # Python package dependencies
|-- README.md                   # This file


## Setup and Installation

1.  **Prerequisites**:
    * Python 3.9 or newer is recommended.
    * `pip` for installing packages.

2.  **Clone/Download Files**:
    * Place all the provided Python files (`main.py`, `config.py`, `utils.py`, and the `handlers/` directory with its contents) into a project directory (e.g., `crisptts_project/`).

3.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    A `requirements.txt` file is provided. Install the necessary packages:
    ```bash
    pip install -r requirements.txt
    ```
    Note: Some libraries like PyTorch, NeMo, and LlamaCPP can have specific installation needs depending on your OS and hardware (e.g., CUDA for Nvidia GPUs, Metal for Apple Silicon). Please refer to their official documentation if you encounter issues.

5.  **`decoder.py` for Orpheus**:
    * For Orpheus-based models (`orpheus_lex_au`, `orpheus_sauerkraut`, `orpheus_lm_studio`, `orpheus_ollama`), you **must** provide your own `decoder.py` file in the project root directory.
    * This file needs to contain a function `convert_to_audio(multiframe, count)` that takes Orpheus model output tokens and returns audio frame bytes. A placeholder is used if this file is missing, which will not produce audio.

6.  **Environment Variables** (Optional but Recommended):
    * **`HF_TOKEN`**: If you need to download models from gated or private Hugging Face repositories, set this environment variable with your Hugging Face API token.
        ```bash
        export HF_TOKEN="your_huggingface_token_here"
        ```
    * `GGML_METAL_NDEBUG=1`: Set automatically by `main.py` to reduce verbose Metal logs from `llama-cpp-python` on macOS.

## Configuration (`config.py`)

The `config.py` file is central to defining which TTS models are available and their default settings.

* **`GERMAN_TTS_MODELS` Dictionary**: This is the primary configuration structure. Each key is a unique `MODEL_ID` used in the CLI. The value is a dictionary containing:
    * `"handler_function_key"` (Optional, defaults to `MODEL_ID`): The key used to look up the synthesis function in `handlers.ALL_HANDLERS`.
    * Specific parameters for that model (e.g., `model_repo_id`, `default_voice_id`, API URLs, `onnx_repo_id`, etc.).
    * `"notes"`: A brief description of the model.
* **Global Constants**: API URLs, default voice names, and sample rates are also defined here.
* **Adding/Modifying Models**: To add a new variation of an existing engine or a completely new engine (after creating its handler), you would add a new entry to `GERMAN_TTS_MODELS`.

## Usage (`main.py`)

All interactions are done through `main.py` from your project's root directory.

**Basic Command Structure:**
```bash
python main.py [ACTION_FLAG | --model-id <MODEL_ID> [OPTIONS]]
Common Examples:

List all available models:

Bash
python main.py --list-models
 Get information about voices for a specific model:

Bash
python main.py --voice-info edge
python main.py --voice-info orpheus_lex_au
 Synthesize text using a specific model:

Bash
python main.py --model-id edge --input-text "Hallo, wie geht es Ihnen heute?" --output-file hallo_edge.mp3 --play-direct
 Use a specific German voice (if supported by the model):

Bash
python main.py --model-id edge --input-text "Ein Test mit einer anderen Stimme." --german-voice-id de-DE-ConradNeural --output-file conrad_test.mp3
 (Check --voice-info <MODEL_ID> for available voice IDs/formats for that model.)

Synthesize text from a file:

Bash
python main.py --model-id piper_local --input-file ./my_text.txt --output-file piper_output.wav
 (Supported input file types: .txt, .md, .html, .pdf, .epub)

Use model-specific parameters (as a JSON string):

Bash
python main.py --model-id orpheus_gguf_local --input-text "Ein Test." --model-params "{\"temperature\": 0.8, \"n_gpu_layers\": -1}" --output-file orpheus_custom.wav
 Test all configured models with default voices:
(Requires --input-text or --input-file)

Bash
python main.py --input-text "Dies ist ein kurzer Test für alle Modelle." --test-all --output-dir ./test_results
 Test all models with all their configured available voices/speakers:
(Requires --input-text or --input-file)

Bash
python main.py --input-text "Ein Test für alle Stimmen." --test-all-speakers --output-dir ./test_results_all_speakers
 Change Logging Level (for debugging):

Bash
python main.py --model-id edge --input-text "Debug Test." --loglevel DEBUG
 Override API URLs (for API-based models like Orpheus LM Studio/Ollama):

Bash
python main.py --model-id orpheus_lm_studio --input-text "Hallo API" --lm-studio-api-url http://localhost:5000/v1/completions
python main.py --model-id orpheus_ollama --input-text "Hallo Ollama" --ollama-api-url http://localhost:11223/api/generate --ollama-model-name my-orpheus-ollama-model
Supported TTS Engines (as configured)
Refer to the output of python main.py --list-models for the currently configured models and their notes. The script supports integration with:

Microsoft Edge TTS
Piper TTS
Orpheus GGUF (via llama-cpp-python)
Orpheus via LM Studio API
Orpheus via Ollama API
OuteTTS (LlamaCPP and Hugging Face ONNX backends)
SpeechT5 (Hugging Face Transformers)
FastPitch (NeMo / Hugging Face)
Adding New TTS Handlers
The modular design makes it easy to add support for new TTS engines:

Create a New Handler File: In the handlers/ directory, create a new Python file (e.g., my_new_tts_handler.py).
Implement Synthesis Function: Inside this file, write a function (e.g., synthesize_with_my_new_tts) that takes the standard arguments: model_config (from GERMAN_TTS_MODELS), text, voice_id_override, model_params_override, output_file_str, play_direct. This function should handle:
Importing its specific TTS library (ideally conditionally).
Downloading any necessary models.
Initializing the TTS engine.
Performing synthesis.
Saving the audio using utils.save_audio.
Optionally playing audio using utils.play_audio.
Cleaning up resources.
Using the logger for diagnostics.
Update handlers/__init__.py:
Import your new synthesis function: from .my_new_tts_handler import synthesize_with_my_new_tts
Add it to the ALL_HANDLERS dictionary, mapping a key (which will be used in config.py) to your function:
Python
ALL_HANDLERS = {
    # ... existing handlers ...
    "my_new_engine": synthesize_with_my_new_tts,
}
Add your function name to the __all__ list if desired.
Update config.py:
Add a new entry to the GERMAN_TTS_MODELS dictionary for your new engine or model variant.
Set the "handler_function_key" to the key you used in ALL_HANDLERS (e.g., "my_new_engine") or ensure the MODEL_ID itself matches the key.
Provide all necessary configuration parameters for your new handler.
decoder.py Requirement for Orpheus
For all Orpheus-based models (GGUF local, LM Studio API, Ollama API), this project relies on a user-provided decoder.py file located in the project's root directory. This file must contain a function:

Python
def convert_to_audio(multiframe_tokens: list[int], total_token_count: int) -> bytes | None:
    # Your implementation here to convert Orpheus token IDs to raw PCM audio bytes
    # (16-bit, 24000 Hz, mono)
    # Return audio frame bytes, or None/empty bytes on error.
    pass
If this file or function is missing, Orpheus models will not produce audible output, and a placeholder will be used.

Troubleshooting & Notes
Missing Libraries: If a specific TTS engine fails, ensure you have installed all its required libraries via pip install -r requirements.txt and any extra steps mentioned in their documentation.
API Keys/Servers: API-based models require the respective servers (LM Studio, Ollama) to be running and accessible. Some cloud services might need API keys (not currently implemented directly, but could be added).
Model Downloads: First-time use of a model that needs to be downloaded from Hugging Face Hub might take some time. Ensure you have an internet connection. Set HF_TOKEN for gated models.
Verbose Output: Use --loglevel DEBUG for detailed diagnostic information if you encounter issues.
RAM Usage: Local GGUF and large Transformer models can be memory-intensive. Ensure your system has sufficient RAM.
Paths: When providing paths for --input-file, --output-file, or speaker WAV files (--german-voice-id), use appropriate relative or absolute paths.