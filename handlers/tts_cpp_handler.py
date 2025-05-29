# CrispTTS/handlers/tts_cpp_handler.py

import logging
import json
import subprocess
import sys
from pathlib import Path
import shutil

# CrispTTS utils
from utils import play_audio

logger = logging.getLogger("CrispTTS.handlers.tts_cpp")

def synthesize_with_tts_cpp(
    crisptts_model_config: dict,
    text: str,
    voice_id_or_path_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool
):
    """
    Synthesizes audio using the TTS.cpp command-line tool.

    This handler constructs and executes a command for the TTS.cpp `cli` executable
    based on the provided configuration and arguments.
    """
    crisptts_specific_model_id = crisptts_model_config.get('crisptts_model_id', 'tts_cpp_unknown')
    logger.info(f"TTS.cpp: Starting synthesis for model '{crisptts_specific_model_id}'.")

    # --- 1. Get Path to TTS.cpp Executable ---
    cli_executable_path_str = crisptts_model_config.get("tts_cpp_executable_path")
    if not cli_executable_path_str:
        logger.error(f"TTS.cpp ({crisptts_specific_model_id}): 'tts_cpp_executable_path' not defined in config. Skipping.")
        return

    cli_executable_path = Path(cli_executable_path_str).resolve()
    if not cli_executable_path.exists() or not cli_executable_path.is_file():
        # Try to find it with shutil.which, which is more robust
        found_path = shutil.which(cli_executable_path.name)
        if found_path:
            cli_executable_path = Path(found_path)
            logger.info(f"TTS.cpp: Found executable '{cli_executable_path.name}' in PATH at: {cli_executable_path}")
        else:
            logger.error(f"TTS.cpp ({crisptts_specific_model_id}): Executable not found at '{cli_executable_path}'. Make sure TTS.cpp is compiled and the path is correct in config.py.")
            logger.error("Build TTS.cpp by running 'cmake -B build' and 'cmake --build build --config Release' in its directory.")
            return

    # --- 2. Get GGUF Model Path ---
    gguf_model_path_str = crisptts_model_config.get("gguf_model_path")
    if not gguf_model_path_str:
        logger.error(f"TTS.cpp ({crisptts_specific_model_id}): 'gguf_model_path' not defined in config. Skipping.")
        return
    
    gguf_model_path = Path(gguf_model_path_str).resolve()
    if not gguf_model_path.exists():
        logger.error(f"TTS.cpp ({crisptts_specific_model_id}): GGUF model file not found at '{gguf_model_path}'.")
        return

    # --- 3. Assemble the Command ---
    command = [
        str(cli_executable_path),
        "--model-path", str(gguf_model_path),
        "--prompt", text,
    ]

    # Handle output: either play directly or save to a file
    if play_direct:
        command.append("--play")
        # TTS.cpp plays directly, so we don't handle output file path
        logger.info("TTS.cpp: Will attempt to play audio directly via SDL2.")
    elif output_file_str:
        output_path = Path(output_file_str).with_suffix(".wav").resolve()
        command.extend(["--save-path", str(output_path)])
    else:
        # Default behavior if no output is specified: save to TTS.cpp.wav in the current directory
        default_save_path = Path.cwd() / "TTS.cpp.wav"
        command.extend(["--save-path", str(default_save_path)])
        logger.warning(f"TTS.cpp: No --output-file specified and --play-direct is false. Saving to default file: {default_save_path}")


    # Handle voice override (for Kokoro, etc.)
    effective_voice = voice_id_or_path_override or crisptts_model_config.get("default_voice_id")
    if effective_voice:
        command.extend(["--voice", effective_voice])

    # --- 4. Parse Model-Specific and CLI Override Parameters ---
    cli_params = {}
    if model_params_override:
        try:
            cli_params = json.loads(model_params_override)
        except json.JSONDecodeError:
            logger.warning(f"TTS.cpp: Could not parse --model-params JSON: {model_params_override}")

    # Set Dia-specific defaults if applicable and not overridden
    if "dia" in crisptts_specific_model_id.lower():
        if "temperature" not in cli_params: cli_params["temperature"] = 1.3
        if "topk" not in cli_params: cli_params["topk"] = 35
        logger.info(f"TTS.cpp: Applying Dia-specific defaults: temp={cli_params['temperature']}, topk={cli_params['topk']}")

    # Map parameters to TTS.cpp CLI arguments
    param_map = {
        "temperature": "--temperature",
        "repetition_penalty": "--repetition-penalty",
        "top_p": "--top-p",
        "topk": "--topk",
        "n_threads": "--n-threads",
        "max_tokens": "--max-tokens",
        "espeak_voice_id": "--espeak-voice-id",
    }
    for key, flag in param_map.items():
        if key in cli_params:
            command.extend([flag, str(cli_params[key])])

    # Handle boolean flags
    if cli_params.get("use_metal", crisptts_model_config.get("use_metal", False)):
        if sys.platform == "darwin": # Metal is only for macOS
            command.append("--use-metal")
        else:
            logger.warning("TTS.cpp: 'use_metal' is true but the system is not macOS. Ignoring.")

    # --- 5. Execute the Command ---
    logger.debug(f"TTS.cpp: Executing command: {' '.join(command)}")

    try:
        process = subprocess.run(
            command,
            check=True,        # Raise an exception for non-zero exit codes
            capture_output=True, # Capture stdout and stderr
            text=True            # Decode stdout/stderr as text
        )
        logger.info("TTS.cpp: Synthesis completed successfully.")
        if process.stdout:
            logger.debug(f"TTS.cpp stdout:\n{process.stdout}")
        if process.stderr:
            logger.debug(f"TTS.cpp stderr:\n{process.stderr}")

        # If not playing directly, and the user didn't specify an output file,
        # we can try to play the default TTS.cpp.wav if pydub is available.
        if not play_direct and not output_file_str and (Path.cwd() / "TTS.cpp.wav").exists():
             logger.info(f"TTS.cpp: Note - Audio saved to {Path.cwd() / 'TTS.cpp.wav'}")


    except FileNotFoundError:
        logger.error(f"TTS.cpp: Command failed because the executable was not found at '{cli_executable_path}'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"TTS.cpp: Synthesis process failed with exit code {e.returncode}.")
        logger.error(f"TTS.cpp command: {' '.join(e.cmd)}")
        if e.stdout:
            logger.error(f"TTS.cpp stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"TTS.cpp stderr:\n{e.stderr}")
    except Exception as e:
        logger.error(f"TTS.cpp: An unexpected error occurred: {e}", exc_info=True)