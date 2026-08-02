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
import time
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


def _cached_mark_state():
    """Provenance state for a cache hit.

    Cache keys include the marking mode (see ``cache._cache_key``), so an
    entry can only be hit under the same mode that produced it: a hit while
    marking is enabled is necessarily a marked file.
    """
    try:
        from watermark import MarkResult, marking_enabled
        if not marking_enabled():
            return MarkResult(marked=False, reason="disabled via CRISPTTS_NO_WATERMARK")
        import watermark as _wm
        return MarkResult(marked=True, backend=_wm._backend, layers=("cached",))
    except ImportError:
        return None


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

    def _send_marking_headers(self, mark_result):
        """Emit provenance headers that reflect what was actually applied.

        These headers are a machine-readable provenance claim, so they must
        never assert marking that did not happen — an unverified "true" is
        worse than an honest "false".
        """
        marked = bool(mark_result is not None and getattr(mark_result, "marked", False))
        self.send_header("X-CrispTTS-Watermarked", "true" if marked else "false")
        if marked:
            backend = getattr(mark_result, "backend", None)
            if backend:
                self.send_header("X-CrispTTS-Watermark-Backend", str(backend))
            confidence = getattr(mark_result, "confidence", None)
            if confidence is not None:
                self.send_header("X-CrispTTS-Watermark-Confidence", f"{confidence:.3f}")
            layers = getattr(mark_result, "layers", ())
            if layers:
                self.send_header("X-CrispTTS-Provenance-Layers", "+".join(layers))

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
            health = {"status": "ok", "server": "CrispTTS", "version": "0.9.1"}
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
        disclosure_lang = body.get("disclosure_lang")
        speaker_identity = body.get("speaker_identity")

        if not model:
            self._send_error(400, "Missing 'model' field")
            return
        if not text:
            self._send_error(400, "Missing 'input' field")
            return

        # Parse SSML tags if present (same as CLI)
        try:
            from ssml import has_ssml, parse_ssml
            if has_ssml(text):
                segments = parse_ssml(text)
                if len(segments) == 1:
                    text = segments[0].text
                    if segments[0].speed != 1.0:
                        speed = segments[0].speed
                # Multi-segment SSML: concatenate text, use first speed
                # (full multi-segment synthesis would need temp files + crossfade,
                #  which is complex for a single API response — keep it simple)
                elif segments:
                    text = " ".join(s.text for s in segments if s.text.strip())
                    speed = segments[0].speed
        except ImportError:
            pass

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

        # --- Voice-cloning consent gate ---
        # MUST run before the cache lookup: otherwise the first consenting
        # request warms the cache and every later caller receives cloned audio
        # without attesting and without an entry in the consent audit log.
        _is_voice_cloning = False
        _needs_disclosure = False
        try:
            from watermark import (
                log_consent_attestation,
                requires_consent,
                requires_spoken_disclosure,
                resolve_speaker_identity,
            )
            _is_voice_cloning = requires_consent(model, handler_key, effective_voice,
                                                 model_config=model_config)
            # A model whose preset voice is an identifiable person produces a
            # deep fake too (Art. 3(60)), so it gets the disclosure as well.
            _needs_disclosure = requires_spoken_disclosure(
                _is_voice_cloning,
                resolve_speaker_identity(model_config, speaker_identity),
                model_id=model)
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
            # Fails closed, like the marking gate below: an unknown cloning
            # status is treated as cloning, not as permission.
            logger.error("watermark module not available in server — refusing synthesis "
                         "because the voice-cloning consent gate cannot be evaluated.")
            self._send_error(500, "Voice-cloning consent gate unavailable; refusing to "
                                  "synthesize.")
            return

        # --- Synthesis cache check ---
        # The disclosure language is part of the key: cloned audio carries the
        # spoken disclosure *inside* it, so a clip disclosed in German is not a
        # valid response to a request that asked for English.
        _cache_params = {}
        if speed != 1.0:
            _cache_params["speed"] = speed
        if _needs_disclosure and disclosure_lang:
            _cache_params["disclosure_lang"] = disclosure_lang
        # Likewise: speaker_identity decides *whether* a disclosure is prepended,
        # so a clip made without one is not a valid response to a request that
        # asked for one.
        if speaker_identity:
            _cache_params["speaker_identity"] = speaker_identity
        _cache_params_json = json.dumps(_cache_params, sort_keys=True) if _cache_params else None

        try:
            import cache as _cache
            cached = _cache.lookup(model, effective_voice, text,
                                   _cache_params_json,
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
                # The cache key includes the marking mode (see cache._cache_key),
                # so a hit was produced under the mode in force right now.
                self._send_marking_headers(_cached_mark_state())
                self.send_header("X-CrispTTS-Cache", "hit")
                self.end_headers()
                self.wfile.write(audio_data)
                return
        except ImportError:
            pass

        # Apply speed
        if speed and speed != 1.0:
            model_config["_cli_speech_speed"] = speed

        # Synthesize to temp file
        suffix = f".{response_format}" if response_format in ("wav", "mp3", "flac", "opus") else ".wav"

        # --- Marking preflight: refuse before spending compute ---
        marking_policy = None
        try:
            from watermark import MarkingError, preflight_marking
            try:
                marking_policy = preflight_marking(f"x{suffix}", handler_key=handler_key)
            except MarkingError as e_pre:
                logger.error("Marking preflight refused %s: %s", response_format, e_pre)
                self._send_error(400, f"Cannot mark '{response_format}' output as AI-generated: {e_pre}")
                return
        except ImportError:
            pass

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        try:
            _synth_started = time.time()
            handler_func(
                model_config,
                text,
                effective_voice,
                json.dumps({"speech_speed": speed}) if speed != 1.0 else None,
                tmp_path,
                False,
            )

            # Follow the audio, as the CLI does: most handlers force their own
            # container regardless of the path they are handed, and reading
            # back only tmp_path turned that into "synthesis produced no
            # output" for every such backend. Unlike the CLI, the response has
            # a declared Content-Type, so convert rather than simply following
            # — returning MP3 bytes labelled audio/wav would be worse than the
            # error it replaces. A missing codec leaves tmp_path unwritten and
            # falls into the 500 below, so this stays fail-closed.
            from utils import resolve_written_output, save_audio
            _written = resolve_written_output(tmp_path, since=_synth_started)
            if _written and _written != tmp_path:
                if os.path.isfile(tmp_path):
                    os.unlink(tmp_path)  # the empty mkstemp stub
                save_audio(_written, tmp_path, source_is_path=True)
                try:
                    os.unlink(_written)
                except OSError:
                    pass

            if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) < 100:
                self._send_error(500, "Synthesis produced no output")
                return

            # --- Spoken disclaimer for voice-cloned audio (Art. 50(4)) ---
            # Applies to every response format, not only WAV. Fails closed:
            # a cloned voice without its disclosure is a 500, not a response.
            if _needs_disclosure:
                try:
                    from watermark import DisclosureError, prepend_disclaimer_file
                except ImportError:
                    logger.error("watermark module not available in server.")
                    self._send_error(500, "Cannot add the AI disclosure to voice-cloned "
                                          "audio; refusing to return undisclosed output.")
                    return
                try:
                    prepend_disclaimer_file(tmp_path,
                                            language=model_config.get("language"),
                                            disclosure_lang=disclosure_lang)
                except DisclosureError as e_disc:
                    logger.error("Server disclosure failed: %s", e_disc)
                    self._send_error(500, f"Cannot add the AI disclosure to voice-cloned "
                                          f"audio ({e_disc}); refusing to return "
                                          f"undisclosed output.")
                    return

            # --- AI-provenance marking (EU AI Act Art. 50(2)) ---
            # Single shared path with the CLI, so every response format gets a
            # real audio watermark — not just strippable container metadata.
            # Fails closed: an unmarkable response is a 500, not unmarked audio.
            mark_result = None
            try:
                from watermark import MarkingError, mark_audio_file
            except ImportError:
                logger.error("watermark module not available in server.")
                if not os.environ.get("CRISPTTS_ALLOW_UNMARKED"):
                    self._send_error(500, "Cannot mark synthetic audio (watermark module "
                                          "unavailable); refusing to return unmarked output.")
                    return
            else:
                try:
                    mark_result = mark_audio_file(tmp_path, handler_key=handler_key,
                                                  policy=marking_policy, model_id=model)
                except MarkingError as e_wm:
                    logger.error("Server marking failed: %s", e_wm)
                    self._send_error(500, f"Cannot mark synthetic audio as AI-generated "
                                          f"({e_wm}); refusing to return unmarked output.")
                    return

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
                             _cache_params_json,
                             tmp_path, f".{response_format}")
            except ImportError:
                pass

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(audio_data)))
            self.send_header("Content-Disposition",
                             f'attachment; filename="tts_output.{response_format}"')
            self.send_header("X-CrispTTS-Model", model)
            self._send_marking_headers(mark_result)
            self.send_header("X-CrispTTS-Cache", "miss")
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


def run_server(host: str = "127.0.0.1", port: int = 8880, rate_limit: int = 10,
               warm_up: str | None = None):
    """Start the CrispTTS HTTP server."""
    global _rate_limit_max
    _rate_limit_max = rate_limit
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Loading TTS handlers...")
    _load_handlers()
    # Optional warm-up: pre-synthesize a short phrase to load models
    if warm_up:
        handlers = _load_handlers()
        model_config = GERMAN_TTS_MODELS.get(warm_up)
        if model_config and handlers:
            handler_key = model_config.get("handler_function_key", warm_up)
            handler_func = handlers.get(handler_key)
            if handler_func:
                import tempfile
                fd, tmp = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    logger.info("Warm-up: synthesizing with %s...", warm_up)
                    handler_func(model_config.copy(), "Warm up.", None, None, tmp, False)
                    logger.info("Warm-up complete.")
                except Exception as e_wu:
                    logger.warning("Warm-up failed: %s", e_wu)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)

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
    parser.add_argument("--warm-up", type=str, default=None, metavar="MODEL_ID",
                        help="Pre-synthesize with this model at startup to warm caches.")
    args = parser.parse_args()
    run_server(args.host, args.port, args.rate_limit, args.warm_up)
