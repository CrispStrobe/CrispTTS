#!/usr/bin/env python3
"""CrispTTS HTTP server — OpenAI-compatible /v1/audio/speech endpoint.

Provides a drop-in replacement for OpenAI's TTS API so applications
using the OpenAI SDK can switch to local synthesis without code changes.

Usage:
    python server.py [--host 0.0.0.0] [--port 8880]
    # Then: curl -X POST http://localhost:8880/v1/audio/speech \
    #   -H "Content-Type: application/json" \
    #   -d '{"model":"crispasr_kokoro","input":"Hello","voice":"af_heart"}' \
    #   --output speech.wav
"""

import json
import logging
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GERMAN_TTS_MODELS  # noqa: E402

logger = logging.getLogger("CrispTTS.server")

# Lazy handler loading
_handlers = None
_handlers_loaded = False


def _load_handlers():
    global _handlers, _handlers_loaded
    if not _handlers_loaded:
        try:
            from handlers import ALL_HANDLERS
            _handlers = ALL_HANDLERS
            _handlers_loaded = True
        except Exception as e:
            logger.error("Failed to load handlers: %s", e)
            _handlers = {}
    return _handlers


# --- Simple token-bucket rate limiter per client IP ---
_rate_limit_buckets: dict[str, list[float]] = {}
_rate_limit_max = 10  # requests per minute (configurable via run_server)
_rate_limit_window = 60.0  # seconds


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate limited."""
    import time as _time
    now = _time.time()
    bucket = _rate_limit_buckets.setdefault(client_ip, [])
    # Evict expired entries
    _rate_limit_buckets[client_ip] = [t for t in bucket if now - t < _rate_limit_window]
    bucket = _rate_limit_buckets[client_ip]
    if len(bucket) >= _rate_limit_max:
        return False
    bucket.append(now)
    return True


class TTSRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for OpenAI-compatible TTS API."""

    def log_message(self, format, *args):
        logger.info(format, *args)

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        self._send_json(code, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/audio/models" or self.path == "/v1/models":
            models = []
            for mid, cfg in GERMAN_TTS_MODELS.items():
                models.append({
                    "id": mid,
                    "object": "model",
                    "owned_by": "crisptts",
                    "backend": cfg.get("crispasr_backend", cfg.get("handler_function_key", "unknown")),
                    "voices": cfg.get("available_voices", []),
                })
            self._send_json(200, {"object": "list", "data": models})
        elif self.path == "/health" or self.path == "/":
            health = {"status": "ok", "server": "CrispTTS", "version": "0.7.0"}
            try:
                import resource
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                health["memory_rss_mb"] = round(rss_mb, 1)
            except (ImportError, AttributeError):
                pass
            handlers = _load_handlers()
            if handlers:
                health["loaded_handlers"] = list(handlers.keys())
                health["registered_handlers"] = (
                    handlers.all_keys() if hasattr(handlers, "all_keys") else list(handlers.keys())
                )
            self._send_json(200, health)
        else:
            self._send_error(404, f"Not found: {self.path}")

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/audio/speech":
            self._send_error(404, f"Not found: {self.path}")
            return

        # Rate limiting
        client_ip = self.client_address[0]
        if not _check_rate_limit(client_ip):
            self._send_error(429, "Rate limit exceeded. Try again later.")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error(400, "Empty request body")
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        # Parse OpenAI-compatible fields
        model = body.get("model")
        text = body.get("input", "")
        voice = body.get("voice")
        response_format = body.get("response_format", "wav")
        speed = body.get("speed", 1.0)
        i_have_rights = body.get("i_have_rights", False)

        if not model:
            self._send_error(400, "Missing 'model' field")
            return
        if not text:
            self._send_error(400, "Missing 'input' field")
            return
        if model not in GERMAN_TTS_MODELS:
            self._send_error(400, f"Unknown model: {model}. Use GET /v1/audio/models for available models.")
            return

        handlers = _load_handlers()
        model_config = GERMAN_TTS_MODELS[model].copy()
        handler_key = model_config.get("handler_function_key", model)
        handler_func = handlers.get(handler_key)

        if not handler_func:
            self._send_error(500, f"No handler available for model: {model}")
            return

        effective_voice = voice or model_config.get("default_voice_id")

        # --- Synthesis cache check ---
        try:
            import cache as _cache
            cached = _cache.lookup(model, effective_voice, text,
                                   json.dumps({"speed": speed}) if speed != 1.0 else None,
                                   f".{response_format}")
            if cached:
                with open(cached, "rb") as f_cached:
                    audio_data = f_cached.read()
                content_type = {"wav": "audio/wav", "mp3": "audio/mpeg",
                                "flac": "audio/flac", "opus": "audio/opus"}.get(response_format, "audio/wav")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(audio_data)))
                self.send_header("Content-Disposition",
                                 f'attachment; filename="tts_output.{response_format}"')
                self.send_header("X-CrispTTS-Model", model)
                self.send_header("X-CrispTTS-Watermarked", "true")
                self.send_header("X-CrispTTS-Cache", "hit")
                self.end_headers()
                self.wfile.write(audio_data)
                return
        except ImportError:
            pass

        # --- Voice-cloning consent gate ---
        _is_voice_cloning = False
        try:
            from watermark import log_consent_attestation, requires_consent
            _is_voice_cloning = requires_consent(model, handler_key, effective_voice)
            if _is_voice_cloning and not i_have_rights:
                self._send_error(403,
                    f"Model '{model}' involves voice cloning. Include "
                    '"i_have_rights": true in the request body to attest '
                    "that you have the consent of the speaker whose voice "
                    "is being cloned, or that it is your own voice.")
                return
            if _is_voice_cloning:
                log_consent_attestation(model, effective_voice, source="API i_have_rights field")
        except ImportError:
            pass

        # Apply speed
        if speed and speed != 1.0:
            model_config["_cli_speech_speed"] = speed

        # Synthesize to temp file
        suffix = f".{response_format}" if response_format in ("wav", "mp3", "flac", "opus") else ".wav"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        try:
            handler_func(
                model_config,
                text,
                effective_voice,
                json.dumps({"speech_speed": speed}) if speed != 1.0 else None,
                tmp_path,
                False,
            )

            if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) < 100:
                self._send_error(500, "Synthesis produced no output")
                return

            # --- Spoken disclaimer for voice-cloned audio (Art. 50(4)) ---
            if _is_voice_cloning and tmp_path.endswith(".wav"):
                try:
                    import soundfile as sf_disc

                    from watermark import prepend_disclaimer
                    data_disc, sr_disc = sf_disc.read(tmp_path, dtype="float32")
                    if data_disc.ndim > 1:
                        data_disc = data_disc[:, 0]
                    data_disc = prepend_disclaimer(data_disc, sample_rate=sr_disc)
                    sf_disc.write(tmp_path, data_disc, sr_disc, subtype="PCM_16")
                except Exception as e_disc:
                    logger.warning("Server disclaimer prepend failed: %s", e_disc)

            # --- Watermark & metadata injection ---
            if not os.environ.get("CRISPTTS_NO_WATERMARK"):
                try:
                    from watermark import (
                        inject_flac_metadata,
                        inject_mp3_metadata,
                        inject_opus_metadata,
                        inject_wav_metadata,
                        watermark_embed,
                    )

                    if tmp_path.endswith(".wav"):
                        if handler_key != "crispasr":
                            import soundfile as sf_srv
                            data_srv, sr_srv = sf_srv.read(tmp_path, dtype="float32")
                            if data_srv.ndim > 1:
                                data_srv = data_srv[:, 0]
                            data_srv = watermark_embed(data_srv, sample_rate=sr_srv)
                            sf_srv.write(tmp_path, data_srv, sr_srv, subtype="PCM_16")
                        with open(tmp_path, "rb") as f_srv:
                            wav_b = inject_wav_metadata(f_srv.read())
                        with open(tmp_path, "wb") as f_srv:
                            f_srv.write(wav_b)
                    elif tmp_path.endswith(".mp3"):
                        with open(tmp_path, "rb") as f_srv:
                            mp3_b = inject_mp3_metadata(f_srv.read())
                        with open(tmp_path, "wb") as f_srv:
                            f_srv.write(mp3_b)
                    elif tmp_path.endswith(".flac"):
                        inject_flac_metadata(tmp_path)
                    elif tmp_path.endswith(".opus"):
                        inject_opus_metadata(tmp_path)
                except ImportError:
                    logger.debug("watermark module not available in server.")
                except Exception as e_wm:
                    logger.warning("Server watermark embedding failed: %s", e_wm)

            with open(tmp_path, "rb") as f_out:
                audio_data = f_out.read()

            content_type = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
                "flac": "audio/flac",
                "opus": "audio/opus",
            }.get(response_format, "audio/wav")

            # Store in cache
            try:
                import cache as _cache
                _cache.store(model, effective_voice, text,
                             json.dumps({"speed": speed}) if speed != 1.0 else None,
                             tmp_path, f".{response_format}")
            except ImportError:
                pass

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(audio_data)))
            self.send_header("Content-Disposition",
                             f'attachment; filename="tts_output.{response_format}"')
            self.send_header("X-CrispTTS-Model", model)
            self.send_header("X-CrispTTS-Watermarked", "true")
            self.end_headers()
            self.wfile.write(audio_data)

        except Exception as e:
            logger.error("Synthesis error: %s", e, exc_info=True)
            self._send_error(500, f"Synthesis failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def run_server(host: str = "127.0.0.1", port: int = 8880, rate_limit: int = 10):
    """Start the CrispTTS HTTP server."""
    global _rate_limit_max
    _rate_limit_max = rate_limit
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Loading TTS handlers...")
    _load_handlers()
    logger.info("Starting CrispTTS server on %s:%d", host, port)
    logger.info("Endpoints:")
    logger.info("  POST /v1/audio/speech — synthesize audio (OpenAI-compatible)")
    logger.info("  GET  /v1/audio/models — list available models")
    logger.info("  GET  /health          — health check")
    class _ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = _ThreadedServer((host, port), TTSRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down.")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CrispTTS API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8880, help="Port (default: 8880)")
    parser.add_argument("--rate-limit", type=int, default=10,
                        help="Max synthesis requests per minute per IP (default: 10, 0=unlimited)")
    args = parser.parse_args()
    run_server(args.host, args.port, args.rate_limit)
