# CrispTTS/handlers/crispasr_handler.py
"""CrispASR TTS handler — access 14 C++ TTS engines via the crispasr binary.

Supported engines: kokoro, orpheus, qwen3-tts, chatterbox, vibevoice-tts,
indextts, voxcpm2-tts, f5-tts, melotts, piper, bananamind-tts, dots-tts,
cosyvoice3-tts, csm-tts. Each runs as native C++ inference through ggml,
offering fast synthesis without Python ML dependencies.

The binary automatically embeds a spread-spectrum watermark into all TTS
output — no additional watermarking needed on the Python side for these
backends.

Requires the crispasr binary on PATH, at CRISPASR_EXECUTABLE, or at
a common build location. Auto-downloads if not found.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import time

from utils import play_audio

logger = logging.getLogger("CrispTTS.handlers.crispasr")

_GITHUB_RELEASE_URL = "https://github.com/CrispStrobe/CrispASR/releases/latest/download"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "crisptts", "crispasr")
_SYNTHESIS_SEMAPHORE = threading.Semaphore(4)  # Limit concurrent streaming synthesis threads


def _find_crispasr():
    """Locate the crispasr binary."""
    env = os.environ.get("CRISPASR_EXECUTABLE")
    if env and os.path.isfile(env):
        return env

    candidates = [
        "crispasr",
        os.path.expanduser("~/.local/bin/crispasr"),
        "/usr/local/bin/crispasr",
    ]
    for base in [
        os.path.expanduser("~/whisper.cpp"),
        os.path.expanduser("~/CrispASR"),
    ]:
        candidates.append(os.path.join(base, "build", "bin", "crispasr"))

    # Check sibling directories relative to this repo
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for sibling in ["whisper.cpp", "CrispASR"]:
        candidates.append(os.path.join(repo_root, "..", sibling, "build", "bin", "crispasr"))

    exe_name = "crispasr.exe" if os.name == "nt" else "crispasr"
    candidates.append(os.path.join(_CACHE_DIR, exe_name))
    for sub in ("crispasr-linux-x86_64", "crispasr-macos", "crispasr-windows-x86_64"):
        candidates.append(os.path.join(_CACHE_DIR, sub, exe_name))

    for c in candidates:
        if c == "crispasr":
            found = shutil.which(c)
            if found:
                return found
        elif os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    # Auto-download
    logger.info("CrispASR binary not found, attempting download...")
    return _download_crispasr()


def _download_crispasr():
    """Download the latest CrispASR release."""
    import tarfile
    import urllib.request
    import zipfile

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine in ("x86_64", "amd64"):
        asset = "crispasr-linux-x86_64.tar.gz"
    elif system == "darwin":
        asset = "crispasr-macos.tar.gz"
    elif system == "windows":
        asset = "crispasr-windows-x86_64.zip"
    else:
        logger.warning("No pre-built CrispASR binary for %s/%s", system, machine)
        return None

    os.makedirs(_CACHE_DIR, exist_ok=True)
    exe_name = "crispasr.exe" if system == "windows" else "crispasr"
    cached_exe = os.path.join(_CACHE_DIR, exe_name)

    if os.path.isfile(cached_exe) and os.access(cached_exe, os.X_OK):
        return cached_exe

    url = f"{_GITHUB_RELEASE_URL}/{asset}"
    archive_path = os.path.join(_CACHE_DIR, asset)
    logger.info("Downloading CrispASR from %s", url)

    try:
        urllib.request.urlretrieve(url, archive_path)  # noqa: S310
    except Exception as e:
        logger.warning("Failed to download CrispASR: %s", e)
        return None

    try:
        if asset.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(_CACHE_DIR, filter="data")  # nosec B202
        elif asset.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.namelist():
                    if os.path.isabs(member) or ".." in member.split("/"):
                        raise ValueError(f"Unsafe zip member: {member}")
                zf.extractall(_CACHE_DIR)  # noqa: S202
    except Exception as e:
        logger.warning("Failed to extract CrispASR: %s", e)
        return None
    finally:
        if os.path.isfile(archive_path):
            os.remove(archive_path)

    for root, _dirs, files in os.walk(_CACHE_DIR):
        for f in files:
            if f == exe_name:
                path = os.path.join(root, f)
                if system != "windows":
                    os.chmod(path, 0o755)  # noqa: S103
                return path

    return None


def synthesize_with_crispasr(
    crisptts_model_config: dict,
    text: str,
    voice_id_or_path_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool,
):
    """Synthesize audio using CrispASR's native C++ TTS engines.

    Supported backends: kokoro, orpheus, qwen3-tts, chatterbox,
    vibevoice-tts, indextts, voxcpm2-tts, f5-tts, melotts, piper,
    bananamind-tts, dots-tts, cosyvoice3-tts, csm-tts.
    """
    model_id = crisptts_model_config.get("crisptts_model_id", "crispasr_unknown")
    backend = crisptts_model_config.get("crispasr_backend")
    logger.info("CrispASR TTS: Starting synthesis for '%s' (backend: %s)", model_id, backend)

    exe = _find_crispasr()
    if not exe:
        logger.error(
            "CrispASR binary not found. Set CRISPASR_EXECUTABLE, install from "
            "https://github.com/CrispStrobe/CrispASR, or place on PATH."
        )
        return

    # Model path — "auto" triggers auto-download from CrispASR registry
    model_path = crisptts_model_config.get("crispasr_model_path", "auto")

    # Output file
    output_path = output_file_str or "crispasr_tts_output.wav"
    if not output_path.endswith(".wav"):
        output_path += ".wav"

    # Build command
    cmd = [
        exe,
        "-m", model_path,
        "--tts", text,
        "--tts-output", output_path,
        "-t", str(min(os.cpu_count() or 4, 8)),
        "--auto-download",
    ]

    if backend:
        cmd.extend(["--backend", backend])

    # Voice
    voice = voice_id_or_path_override or crisptts_model_config.get("default_voice_id")
    if voice:
        if os.path.isfile(voice):
            cmd.extend(["--voice", voice])
        else:
            # Could be a speaker name (orpheus) or voice pack name
            cmd.extend(["--voice", voice])

    # Codec / companion model
    codec_model = crisptts_model_config.get("crispasr_codec_model")
    if codec_model:
        cmd.extend(["--codec-model", codec_model])

    # Reference text for voice cloning (qwen3-tts)
    ref_text = crisptts_model_config.get("reference_text")
    if ref_text:
        cmd.extend(["--ref-text", ref_text])

    # Instruct (qwen3-tts VoiceDesign)
    instruct = crisptts_model_config.get("instruct")
    if instruct:
        cmd.extend(["--instruct", instruct])

    # Language
    language = crisptts_model_config.get("language")
    if language:
        cmd.extend(["-l", language])

    # --- CLI-injected synthesis flags (from main.py) ---
    cli_speed = crisptts_model_config.get("_cli_speech_speed")
    if cli_speed and cli_speed != 1.0:
        cmd.extend(["--pace", str(cli_speed)])

    if crisptts_model_config.get("_cli_trim_silence"):
        cmd.append("--tts-trim-silence")

    cli_steps = crisptts_model_config.get("_cli_tts_steps")
    if cli_steps is not None:
        cmd.extend(["--tts-steps", str(cli_steps)])

    cli_pitch = crisptts_model_config.get("_cli_pitch_shift")
    if cli_pitch and cli_pitch != 0.0:
        cmd.extend(["--pitch-shift", str(cli_pitch)])

    if crisptts_model_config.get("_cli_no_spoken_disclaimer"):
        cmd.append("--no-spoken-disclaimer")

    cli_lexicon = crisptts_model_config.get("_cli_lexicon")
    if cli_lexicon:
        cmd.extend(["--lexicon", cli_lexicon])

    # Parse model params override
    if model_params_override:
        try:
            params = json.loads(model_params_override)
        except json.JSONDecodeError:
            logger.warning("Could not parse --model-params JSON: %s", model_params_override)
            params = {}

        param_map = {
            "temperature": "-tp",
            "seed": "--seed",
            "tts_steps": "--tts-steps",
            "top_p": "--top-p",
            "top_k": "--top-k",
            "min_p": "--min-p",
            "repetition_penalty": "--repetition-penalty",
            "cfg_weight": "--cfg-weight",
            "cfg_scale": "--tts-cfg-scale",
            "exaggeration": "--exaggeration",
            "length_scale": "--length-scale",
            "speaker_name": "--speaker-name",
            "speech_speed": "--pace",
            "pitch_shift": "--pitch-shift",
            "do_sample": "--tts-do-sample",
            "num_candidates": "--tts-num-candidates",
            "num_steps": "--tts-num-steps",
            "noise_temp": "--tts-noise-temp",
            "noise_scale": "--tts-noise-scale",
            "noise_w": "--tts-noise-w",
            "speaker_id": "--tts-speaker-id",
            "max_speech_tokens": "--tts-max-speech-tokens",
        }
        for key, flag in param_map.items():
            if key in params:
                cmd.extend([flag, str(params[key])])

    logger.info("CrispASR TTS: Running: %s", " ".join(cmd))

    t0 = time.time()
    try:
        result = subprocess.run(  # noqa: S603
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=300,
        )

        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.info("crispasr: %s", line)

        if result.returncode != 0:
            logger.error(
                "CrispASR TTS failed (code %d): %s",
                result.returncode, result.stderr[-500:] if result.stderr else "no output",
            )
            return

        elapsed = time.time() - t0
        logger.info("CrispASR TTS: Synthesis completed in %.1fs", elapsed)

        if os.path.isfile(output_path):
            logger.info("CrispASR TTS: Output saved to %s", output_path)
            if play_direct:
                play_audio(output_path, is_path=True)
        else:
            logger.error("CrispASR TTS: Output file not created at %s", output_path)

    except subprocess.TimeoutExpired:
        logger.error("CrispASR TTS: Synthesis timed out after 300s")
    except FileNotFoundError:
        logger.error("CrispASR TTS: Executable not found at '%s'", exe)
    except Exception as e:
        logger.error("CrispASR TTS: Unexpected error: %s", e, exc_info=True)


def synthesize_with_crispasr_streaming(
    crisptts_model_config: dict,
    text: str,
    voice_id_or_path_override: str | None,
    model_params_override: str | None,
    output_file_str: str | None,
    play_direct: bool,
):
    """Streaming TTS: plays audio as crispasr generates it.

    Runs crispasr with --tts-output pointing to a temp file, then
    starts playback as soon as the file appears while synthesis may
    still be running. Falls back to non-streaming if sounddevice
    is unavailable.
    """
    import struct
    import tempfile
    import threading

    try:
        import sounddevice as sd
    except ImportError:
        logger.warning("sounddevice not installed — falling back to non-streaming synthesis.")
        return synthesize_with_crispasr(
            crisptts_model_config, text, voice_id_or_path_override,
            model_params_override, output_file_str, play_direct,
        )

    model_id = crisptts_model_config.get("crisptts_model_id", "crispasr_unknown")
    backend = crisptts_model_config.get("crispasr_backend")
    logger.info("CrispASR TTS (streaming): Starting for '%s' (backend: %s)", model_id, backend)

    if not _SYNTHESIS_SEMAPHORE.acquire(timeout=30):
        logger.error("CrispASR TTS (streaming): Too many concurrent synthesis requests.")
        return

    exe = _find_crispasr()
    if not exe:
        _SYNTHESIS_SEMAPHORE.release()
        logger.error("CrispASR binary not found.")
        return

    model_path = crisptts_model_config.get("crispasr_model_path", "auto")

    # Use temp file for output, will stream-read it
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        exe, "-m", model_path,
        "--tts", text,
        "--tts-output", tmp_wav,
        "-t", str(min(os.cpu_count() or 4, 8)),
        "--auto-download",
    ]

    if backend:
        cmd.extend(["--backend", backend])
    voice = voice_id_or_path_override or crisptts_model_config.get("default_voice_id")
    if voice:
        cmd.extend(["--voice", voice])
    codec_model = crisptts_model_config.get("crispasr_codec_model")
    if codec_model:
        cmd.extend(["--codec-model", codec_model])
    language = crisptts_model_config.get("language")
    if language:
        cmd.extend(["-l", language])

    logger.info("CrispASR TTS (streaming): Running: %s", " ".join(cmd))

    synthesis_done = threading.Event()
    synth_error = [None]

    def _run_synthesis():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # noqa: S603
            if result.returncode != 0:
                synth_error[0] = result.stderr[-500:] if result.stderr else "unknown error"
        except Exception as e:
            synth_error[0] = str(e)
        finally:
            synthesis_done.set()

    synth_thread = threading.Thread(target=_run_synthesis, daemon=True)
    synth_thread.start()

    # Wait for the WAV file to appear (synthesis started writing)
    import time as _time
    for _ in range(600):  # up to 60 seconds
        if os.path.isfile(tmp_wav) and os.path.getsize(tmp_wav) > 44:
            break
        if synthesis_done.is_set():
            break
        _time.sleep(0.1)

    # Stream playback: read WAV header, then stream PCM chunks
    try:
        if os.path.isfile(tmp_wav) and os.path.getsize(tmp_wav) > 44:
            with open(tmp_wav, "rb") as f:
                header = f.read(44)
                if len(header) >= 28:
                    sr = struct.unpack_from("<I", header, 24)[0]
                else:
                    sr = 24000
                logger.info("CrispASR TTS (streaming): Playing at %d Hz...", sr)
                stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16")
                stream.start()
                try:
                    while not synthesis_done.is_set() or f.tell() < os.path.getsize(tmp_wav):
                        chunk = f.read(4096)
                        if chunk:
                            import numpy as np
                            samples = np.frombuffer(chunk, dtype=np.int16)
                            stream.write(samples.reshape(-1, 1))
                        elif not synthesis_done.is_set():
                            _time.sleep(0.05)
                        else:
                            # Read any remaining data
                            remaining = f.read()
                            if remaining:
                                samples = np.frombuffer(remaining, dtype=np.int16)
                                stream.write(samples.reshape(-1, 1))
                            break
                finally:
                    stream.stop()
                    stream.close()
    except Exception as e:
        logger.warning("Streaming playback error: %s", e)

    synth_thread.join(timeout=10)

    if synth_error[0]:
        logger.error("CrispASR TTS (streaming): Synthesis error: %s", synth_error[0])

    # Copy to final output if requested
    if output_file_str and os.path.isfile(tmp_wav):
        import shutil
        shutil.copy2(tmp_wav, output_file_str)
        logger.info("CrispASR TTS (streaming): Output saved to %s", output_file_str)

    # Cleanup temp
    try:
        os.unlink(tmp_wav)
    except OSError:
        pass
    finally:
        _SYNTHESIS_SEMAPHORE.release()


def verify_tts_with_asr(
    audio_path: str,
    original_text: str,
    asr_backend: str = "parakeet",
    asr_model: str = "auto",
) -> dict:
    """Run ASR on TTS output for roundtrip quality verification.

    Returns a dict with: asr_text, similarity, word_count, duration_s.
    """
    exe = _find_crispasr()
    if not exe:
        return {"error": "CrispASR binary not found"}

    cmd = [
        exe,
        "-m", asr_model,
        "--backend", asr_backend,
        "-f", audio_path,
        "-t", str(min(os.cpu_count() or 4, 8)),
        "-np",
        "--auto-download",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # noqa: S603
        if result.returncode != 0:
            return {"error": f"ASR failed (code {result.returncode})"}

        # Parse output lines — extract text (skip timestamp prefixes)
        import re
        lines = []
        for line in result.stdout.strip().splitlines():
            m = re.match(r"\[\d+:\d+:\d+\.\d+\s*-->\s*\d+:\d+:\d+\.\d+\]\s*(.*)", line)
            if m:
                lines.append(m.group(1).strip())
            elif line.strip():
                lines.append(line.strip())

        asr_text = " ".join(lines)

        # Simple word-overlap similarity
        orig_words = set(original_text.lower().split())
        asr_words = set(asr_text.lower().split())
        if orig_words:
            overlap = len(orig_words & asr_words)
            similarity = overlap / max(len(orig_words), len(asr_words))
        else:
            similarity = 0.0

        return {
            "asr_text": asr_text,
            "similarity": round(similarity, 3),
            "word_count": len(asr_text.split()),
            "original_text": original_text,
        }

    except Exception as e:
        return {"error": str(e)}


def translate_text_with_crispasr(
    text: str,
    source_lang: str = "en",
    target_lang: str = "de",
    model: str = "auto",
    backend: str = "m2m100",
) -> str:
    """Translate text using CrispASR's translation backends.

    Returns translated text, or the original text on failure.
    """
    exe = _find_crispasr()
    if not exe:
        logger.warning("CrispASR binary not found — cannot translate")
        return text

    cmd = [
        exe,
        "-m", model,
        "--backend", backend,
        "--text", text,
        "--tr-sl", source_lang,
        "--tr-tl", target_lang,
        "-t", str(min(os.cpu_count() or 4, 8)),
        "--auto-download",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # noqa: S603
        if result.returncode == 0 and result.stdout.strip():
            translated = result.stdout.strip()
            logger.info("Translation (%s→%s): '%s' → '%s'", source_lang, target_lang, text[:50], translated[:50])
            return translated
        else:
            logger.warning("Translation failed (code %d)", result.returncode)
            return text
    except Exception as e:
        logger.warning("Translation error: %s", e)
        return text
