# PLAN: Leverage CrispASR features in CrispTTS

## Context

CrispTTS already calls the CrispASR binary for 9 TTS backends via `crispasr_handler.py`.
The handler builds a subprocess command, runs it, and collects the output WAV. Many CrispASR
features (speed control, streaming, voice design, silence trimming, etc.) are already
available in the binary but not exposed through CrispTTS's CLI or handler. This plan adds
them systematically, grouped into phases by dependency.

### Key integration point

**`handlers/crispasr_handler.py:synthesize_with_crispasr()`** (lines 132-265)
- Builds `cmd` array with flags like `--backend`, `--voice`, `--tts`, `--tts-output`
- Already has `param_map` (line 207-228) mapping JSON keys to CLI flags
- Already passes `--auto-download` for model registry

### What we need to touch per feature

1. **`main.py`** — add CLI flag (argparse)
2. **`crispasr_handler.py`** — pass flag to binary command
3. **`config.py`** — update model entries if needed
4. **`tests/test_watermark.py` or new test files** — unit + live tests

---

## Phase 1: Quick CLI flags (pass-through to crispasr binary)

These just need a new CLI flag in main.py, a line in crispasr_handler.py to append it
to the command, and tests.

### 1.1 Speech speed control (`--speech-speed`)

- **main.py**: Add `--speech-speed FLOAT` to synth_group (default 1.0)
- **crispasr_handler.py**: If `args.speech_speed != 1.0`, append `["--pace", str(args.speech_speed)]` to cmd
- **Also**: Add `"speech_speed"` → `"--pace"` to `param_map` so `--model-params '{"speech_speed":1.2}'` works too
- **Pass through**: `run_synthesis()` needs to pass `args.speech_speed` to handler via config dict
- **Tests**: Unit test that cmd array contains `--pace` when speed != 1.0; live test with crispasr binary

### 1.2 Silence trimming (`--trim-silence`)

- **main.py**: Add `--trim-silence` boolean flag
- **crispasr_handler.py**: If set, append `["--tts-trim-silence"]` to cmd
- **Also**: Pure-Python fallback in utils.py for non-crispasr handlers (RMS-based leading/trailing trim)
- **Tests**: Unit test for Python trim function; live test with crispasr binary

### 1.3 CFM inference steps (`--tts-steps`)

- **main.py**: Add `--tts-steps INT` (default None → backend default)
- **crispasr_handler.py**: Already in `param_map` as `"tts_steps"` → `"--tts-steps"`. Just add the CLI flag.
- **Tests**: Unit test param mapping; live test

### 1.4 Language selection (`--tts-language`)

- **main.py**: Add `--tts-language LANG` (e.g., de, en, zh, ja)
- **crispasr_handler.py**: Already passes `-l {language}` from config. Add CLI override: if `args.tts_language`, override config's `language` field.
- **Tests**: Unit test flag override; live test with kokoro multilingual

### 1.5 Pitch shift (`--pitch-shift`)

- **main.py**: Add `--pitch-shift FLOAT` (Hz offset, default 0)
- **crispasr_handler.py**: Append `["--pitch-shift", str(val)]` to cmd. Add to `param_map`.
- **Tests**: Unit + live

---

## Phase 2: Model config & voice features

### 2.1 Qwen3-TTS VoiceDesign (`--instruct`)

- **main.py**: Add `--instruct TEXT` for natural-language voice descriptions
- **crispasr_handler.py**: Already passes `--instruct` from config (line 202-205). Add CLI override.
- **config.py**: Add `crispasr_qwen3_tts_voicedesign` model entry with `instruct` field
- **Tests**: Unit test config; live test with qwen3-tts

### 2.2 VoxCPM2 voice cloning

- **config.py**: Update `crispasr_voxcpm2` entry to note voice cloning capability
- **crispasr_handler.py**: Already passes `--voice` for reference WAV. Ensure consent gate applies.
- **watermark.py**: Add `"synthesize_with_crispasr"` check in `requires_consent()` when voice is a .wav path
- **Tests**: Unit test consent gate triggers for .wav voice paths

### 2.3 FastPitch multi-speaker

- **config.py**: Update FastPitch entry with `available_voices` list (speaker names from CrispASR)
- **Tests**: Config validation

### 2.4 Orpheus speaker enumeration

- **config.py**: Ensure all 19 DE speakers + 5 EN speakers are in `available_voices`
- **Tests**: Config validation

---

## Phase 3: Audio processing (Python-side)

### 3.1 Smart text chunking

- **New file**: `chunking.py` — split text at sentence boundaries (`. ! ? ;`), yield chunks
- **main.py**: For long text (>500 chars), split into chunks, synthesize each, concatenate
- **Applies to**: All handlers, not just crispasr
- **Tests**: Unit test sentence splitting; integration test concatenation

### 3.2 Compressed output formats (Opus/FLAC)

- **utils.py**: Extend `save_audio()` to handle `.opus` and `.flac` via pydub/soundfile
- **main.py**: Accept `.opus` / `.flac` extensions in `--output-file`
- **crispasr_handler.py**: For crispasr backends, could also pass format flags to binary
- **Tests**: Unit test format detection; live test write+read opus/flac

### 3.3 Audio resampling utility

- **utils.py**: Add `resample_audio(pcm, from_sr, to_sr)` using scipy or the existing `_resample_linear` from watermark.py
- **Expose**: `--output-sample-rate INT` CLI flag
- **Tests**: Unit test resample quality

---

## Phase 4: Streaming synthesis

### 4.1 Streaming playback

- **crispasr_handler.py**: New function `synthesize_with_crispasr_streaming()` that:
  - Runs crispasr with `--tts-output -` (stdout) or writes to a FIFO
  - Reads PCM chunks from stdout in a thread
  - Plays via sounddevice as chunks arrive
- **main.py**: `--stream` flag enables streaming mode
- **Only for crispasr backends** initially (Python handlers don't support streaming)
- **Tests**: Unit test subprocess chunking; live test with kokoro

---

## Phase 5: OpenAI-compatible API server

### 5.1 `/v1/audio/speech` endpoint

- **New file**: `server.py` — FastAPI/Flask app
- **Endpoints**:
  - `POST /v1/audio/speech` — matches OpenAI spec (`model`, `input`, `voice`, `response_format`, `speed`)
  - `GET /v1/audio/models` — list available models
- **Maps to**: `run_synthesis()` internally
- **Watermarking**: Applied automatically via existing `save_audio()` pipeline
- **Tests**: Unit test endpoint routing; live test with curl

---

## File change summary

| File | Changes |
|------|---------|
| `main.py` | 7 new CLI flags, streaming mode, server mode |
| `crispasr_handler.py` | 5 new cmd flags, streaming function |
| `config.py` | Update voice lists, add voicedesign entry |
| `utils.py` | Silence trim, opus/flac support, resample |
| `watermark.py` | Consent gate for .wav voice paths |
| `chunking.py` | NEW — sentence-boundary text splitter |
| `server.py` | NEW — OpenAI-compatible API server |
| `tests/test_features.py` | NEW — unit + live tests for all features |

---

## Execution order

1. Phase 1 (5 CLI flags) — all independent, can batch
2. Phase 2 (4 config/voice features) — mostly config.py changes
3. Phase 3 (3 audio processing) — Python-side, independent of crispasr
4. Phase 4 (streaming) — depends on Phase 1 working
5. Phase 5 (server) — depends on everything else working

Each phase: implement → unit test → live test → commit → push.

---

## Status: ALL PHASES COMPLETE

All 5 phases implemented, tested, and pushed.

| Phase | Commit | Tests |
|-------|--------|-------|
| 1 | `bb5e451` | 26 pass |
| 2 | `8ae7f6f` | 34 pass |
| 3 | `c3f6691` | 46 pass |
| 4 | `1b4aee0` | 48 pass |
| 5 | `0f734d2` | 51 pass |

---

## Phase 6: Watermarking & Voice Cloning Safety (v0.4.0)

Implemented 2026-06-22.

### 6.1 Watermark embedding on all outputs

The critical gap: `watermark_embed()`, `inject_wav_metadata()`, `inject_mp3_metadata()`,
and `c2pa_sign_file()` were defined in `watermark.py` but never called on synthesized audio.

- **main.py**: Post-synthesis watermark + metadata injection for all handlers
  - CrispASR handlers skipped (binary already watermarks)
  - WAV/MP3/FLAC/Opus metadata injection on all outputs
  - C2PA signing if cert/key configured
  - Same pipeline in `test_all_models()` loop
- **server.py**: Same watermark pipeline for API responses
  - `X-CrispTTS-Watermarked: true` response header

### 6.2 WavMark neural watermark (MIT license)

Added as preferred neural backend over AudioSeal (CC-BY-NC model weights).

- **watermark.py**: `load_wavmark()`, `_embed_wavmark()`, `_detect_wavmark()`
- Fixed 16-bit "CT" payload for CrispTTS detection
- Sample-rate aware (resamples to 16 kHz for WavMark, applies delta at native rate)
- Priority: WavMark (MIT) > AudioSeal (Python) > CrispASR GGUF > spread-spectrum

### 6.3 New CrispASR TTS backends

- `crispasr_f5_tts` — F5-TTS flow-matching, 24 kHz, Apache 2.0
- `crispasr_melotts` — MeloTTS VITS2, 44.1 kHz, MIT
- `crispasr_piper` — Piper VITS via C++, 250+ voices, 30+ langs

### 6.4 Voice-cloning safety

- **Server consent gate**: `"i_have_rights": true` required in API request body for cloning models (returns 403 otherwise)
- **Expanded detection**: CrispASR cloning backends (`vibevoice`, `indextts`, `voxcpm2`, `qwen3_tts`) added to keyword set
- **Persistent audit log**: `~/.cache/crisptts/consent_audit.log` (not just stderr)
- **Spoken disclaimer**: CrispASR kokoro (local, first) > Edge TTS (cloud) > beep

### 6.5 FLAC/Opus metadata

- `inject_flac_metadata()` — Vorbis comments via mutagen
- `inject_opus_metadata()` — OggOpus tags via mutagen
- Wired into main.py and server.py

| Commit | Tests | CI |
|--------|-------|----|
| `01b4d41` | 199 pass | py3.10/3.11/3.12 + ruff ✓ |
| `10becea` | 212 pass | py3.10/3.11/3.12 + ruff ✓ |

Released as [v0.4.0](https://github.com/CrispStrobe/CrispTTS/releases/tag/v0.4.0).

---

## Phase 7: New CrispASR backends + TADA enhancements (v0.5.0)

Synced with CrispASR v0.8.7 (2026-07-04). CrispASR added 4 new TTS backends,
TADA gained inline voice cloning + forced alignment, and new per-request
tuning flags were added.

### 7.1 New backend configs in config.py

Add 4 new `crispasr_*` entries:

| Model ID | Backend | Sample Rate | Voice Cloning | Notes |
|----------|---------|-------------|---------------|-------|
| `crispasr_bananamind_tts` | `bananamind-tts` | 22050 | No | Tacotron-lite + HiFi-GAN, en/de |
| `crispasr_dots_tts` | `dots-tts` | 48000 | Yes (CAM++) | Qwen2.5 LLM + DiT + BigVGAN |
| `crispasr_cosyvoice3_tts` | `cosyvoice3-tts` | 24000 | Yes (baked) | Multi-GGUF: LLM+flow+CAM++HiFT |
| `crispasr_csm_tts` | `csm-tts` | 24000 | Yes (ref.wav) | Sesame CSM-1B, causal mode |

### 7.2 New CLI flags in main.py

Pass-through flags for TADA and new backends:

| Flag | Maps to | Purpose |
|------|---------|---------|
| `--ref-text TEXT` | `--ref-text` | Transcript for inline voice cloning |
| `--no-spoken-disclaimer` | `--no-spoken-disclaimer` | Skip AI disclaimer on cloned audio |

### 7.3 Expanded param_map in crispasr_handler.py

New keys in the `--model-params` JSON mapping:

| Key | Flag | Backends |
|-----|------|----------|
| `top_k` | `--top-k` | dots-tts, cosyvoice3, TADA |
| `min_p` | `--min-p` | LLM-based |
| `do_sample` | `--tts-do-sample` | TADA talker |
| `num_candidates` | `--tts-num-candidates` | TADA acoustic |
| `cfg_scale` | `--tts-cfg-scale` | chatterbox, f5, TADA |
| `num_steps` | `--tts-num-steps` | TADA flow-matching |
| `noise_temp` | `--tts-noise-temp` | TADA FM noise |
| `noise_scale` | `--tts-noise-scale` | piper VITS |
| `noise_w` | `--tts-noise-w` | piper stochastic duration |
| `speaker_id` | `--tts-speaker-id` | piper multi-speaker |
| `max_speech_tokens` | `--tts-max-speech-tokens` | chatterbox |

### 7.4 Voice-cloning keyword expansion

Add to `VOICE_CLONING_MODEL_KEYWORDS` in watermark.py:
- `dots`, `cosyvoice3`, `csm`, `tada`, `bananamind` (bananamind has no cloning
  but shares the handler — detected by `.wav` path heuristic)

### 7.5 Update handler docstring

`crispasr_handler.py`: 10 → 14 backends, document ref-text flag.

### 7.6 Unit tests

Mocked tests (no binary needed):
- Config validation for all 4 new backends
- `--ref-text` flag pass-through in command builder
- New param_map keys produce correct CLI flags
- Voice-cloning keyword detection for dots/cosyvoice3/csm
- `--no-spoken-disclaimer` pass-through

### 7.7 Live tests

Live tests need to work on 8 GB RAM VPS with no GPU:
- Use `--backend kokoro` (smallest model, ~82M params, auto-download)
- Short text input ("Test.") to minimize memory + time
- 30s timeout per synthesis to avoid hangs
- Skip GPU-heavy backends (dots-tts 48 kHz, cosyvoice3 multi-GGUF) in live tests
- Test `--ref-text` pass-through with a tiny WAV (generate sine wave)

### 7.8 README update

- Update engine count (31+ → 35+)
- Add new backends to the CrispASR native list
- Document `--ref-text` for inline voice cloning
- Update `--model-params` table with new keys

---

### Live test results (2026-06-07)

End-to-end pipeline verified with Kokoro backend:

```
pip install py-espeak-ng  # required for Kokoro phonemization
CRISPASR_EXECUTABLE=/mnt/volume1/CrispASR/build/bin/crispasr \
  python main.py --model-id crispasr_kokoro \
  --input-text "Hallo Welt, dies ist ein Live-Test der Sprachsynthese." \
  --german-voice-id ~/.cache/crispasr/kokoro-voice-af_heart.gguf \
  --output-file /tmp/full_pipeline_test.wav --trim-silence
```

Result:
- Synthesis: 3.73s audio @ 24 kHz (Kokoro, af_heart voice)
- Watermark: spread-spectrum applied (confidence 0.49)
- Metadata: LIST/INFO with "AI-generated audio" provenance
- Silence trimming: applied
- Pipeline: CrispTTS → crispasr binary → watermarked WAV ✓

---

## Phase 8: Performance optimizations (v0.5.0)

Implemented 2026-07-04.

### 8.1 Reduced file I/O in post-synthesis pipeline

WAV watermark embed + metadata injection was 4 I/O ops (read PCM, write PCM,
read bytes, write bytes). Combined into 3 ops by inlining the metadata read
into a single read→transform→write pass. MP3 path similarly streamlined.
Same fix applied to `test_all_models()` loop and `server.py`.

### 8.2 Lazy watermark model loading

Neural watermark backends (WavMark ~200 MB, AudioSeal ~150 MB) were loaded
eagerly at CLI startup — even for `--list-models` or `--help`. Now they
lazy-load on first `watermark_embed()` call via a guard in the dispatcher.
Only explicit `--watermark-model` still triggers eager loading.

### 8.3 Server fixes

- File handle leak: `open()` without context manager → `with` statement
- Added `Content-Disposition: attachment` header for proper file downloads

### 8.4 Streaming concurrency limit

Added `threading.Semaphore(4)` to cap concurrent streaming synthesis threads.
Returns error after 30s wait if all slots are occupied. Prevents unbounded
thread growth under load.

### 8.5 Subprocess stdout optimization

Synthesis subprocess: `capture_output=True` → `stdout=DEVNULL, stderr=PIPE`.
Only stderr is needed for error reporting. Avoids buffering potentially large
stdout on long synthesis runs.

| Commit | Tests | CI |
|--------|-------|----|
| `c981f3f` | 224 pass | py3.10/3.11/3.12 + ruff ✓ |

---

## Phase 9: Usability & reliability

### 9.1 Fix utils.py import hang on headless machines

`utils.py` imports pygame/sounddevice at module level, which blocks indefinitely
on headless machines (no audio hardware). This causes:
- Live tests to hang in pytest (pygame audio init blocks)
- Slow CLI startup on servers/VPS

**Fix**: Move pygame/sounddevice imports inside `play_audio()` and other
functions that actually need them. Guard with try/except at use-time, not
import-time.

**Files**: `utils.py`

### 9.2 `--backend` CLI shortcut

Users currently must remember `--model-id crispasr_kokoro` when they think
in CrispASR backend names (`kokoro`). Add `--backend NAME` as a shortcut
that auto-selects the matching `crispasr_*` config entry.

**Files**: `main.py` (argparse + dispatch logic)

### 9.3 Threaded HTTP server

Replace `HTTPServer` with `ThreadingHTTPServer` (stdlib). Current server
blocks on each request — a long synthesis blocks all other clients.

**Files**: `server.py` (one-line change + import)

### 9.4 Batch synthesis mode

`--input-file book.txt` currently produces one giant file. Add paragraph
splitting: `--batch` flag splits input at blank lines, produces numbered
output files (`output_001.wav`, `output_002.wav`, ...).

**Files**: `main.py`, `chunking.py`

### 9.5 Model availability probe

`--list-models --check` probes each CrispASR backend with a quick
`crispasr --backend X -m auto --dry-run` to show which backends are
actually available (model cached) vs. need downloading vs. unsupported.

**Files**: `main.py`, `handlers/crispasr_handler.py`

### 9.6 Config validation

Add a `validate_config()` function that checks all GERMAN_TTS_MODELS entries
at startup for required fields, valid handler keys, and correct types. Emit
clear warnings for misconfigured entries instead of failing at synthesis time.

**Files**: `config.py` or new `validate.py`, `main.py`

### 9.7 Pronunciation lexicon support

Pass custom word→phoneme mappings to CrispASR backends via
`--lexicon file.tsv` for domain-specific terms (medical, legal, brand names).

**Files**: `main.py` (argparse), `handlers/crispasr_handler.py` (pass-through)

### Status: ALL PHASE 9 ITEMS COMPLETE

| Task | Commit |
|------|--------|
| 9.1 Fix import hang | `3fbaf83` |
| 9.2 --backend shortcut | `3fbaf83` |
| 9.3 Threaded server | `3fbaf83` |
| 9.4 Batch synthesis | `3fbaf83` |
| 9.5 Model probe | `3fbaf83` |
| 9.6 Config validation | `3fbaf83` |
| 9.7 Lexicon support | `3fbaf83` |

224 tests passing (1 test updated for lazy-load semantics).

---

## Phase 10: Lazy handler registry + developer experience

### 10.1 Lazy handler registry

`handlers/__init__.py` eagerly imports all 21 handlers at module level.
This loads torch, transformers, outetts, nemo, etc. — ~2 GB RAM, 6+ minute
startup on the 8 GB VPS. Most sessions use only 1-2 handlers.

**Fix**: Replace eager imports with a lazy registry. Each handler is imported
only when its `handler_function_key` is first requested via `ALL_HANDLERS[key]`.

**Design**:
```python
class _LazyHandlerRegistry(dict):
    """Import handlers on first access, not at module load time."""
    _REGISTRY = {
        "edge": (".edge_handler", "synthesize_with_edge_tts"),
        "crispasr": (".crispasr_handler", "synthesize_with_crispasr"),
        ...
    }
    def __getitem__(self, key):
        if key not in self._loaded:
            module_path, func_name = self._REGISTRY[key]
            mod = importlib.import_module(module_path, package="handlers")
            self._loaded[key] = getattr(mod, func_name)
        return self._loaded.get(key)
```

**Impact**:
- Server starts instantly (imports only the requested handler)
- `--list-models` never loads torch
- RAM drops from ~2 GB to ~200 MB for single-handler use
- Test suite imports complete in seconds instead of 6 minutes

**Files**: `handlers/__init__.py`

### 10.2 Split test suite into fast/slow

Add `@pytest.mark.slow` to tests that trigger heavy imports (outetts,
torch model loading, handler registry tests). Default `pytest` runs only
fast tests; `pytest -m slow` or `pytest --run-slow` runs everything.

**Files**: `tests/test_handlers.py`, `tests/test_cli.py`, `pyproject.toml`

### 10.3 Server rate limiting

Simple in-memory token bucket per client IP. Default: 10 requests/minute,
configurable via `--rate-limit N`. Returns 429 Too Many Requests when
exceeded.

**Files**: `server.py`

### 10.4 Audio crossfade for chunked synthesis

When `chunking.py` splits long text, the handler synthesizes each chunk
separately. Add a short crossfade (~50 ms) between concatenated segments
to eliminate clicks/gaps at chunk boundaries.

**Files**: `utils.py` (new `crossfade_segments()` function), integration
in handlers that use chunking

### 10.5 Synthesis result caching

Hash `(model_id, voice, text, params)` → cached WAV path. Serves identical
requests from cache. LRU eviction by total cache size (default 500 MB,
configurable via `--cache-dir` / `--cache-max-mb`).

**Files**: `main.py` or new `cache.py`, `server.py`

### 10.6 Batch error recovery

If one paragraph fails in `--batch` mode, log the error and continue with
the next paragraph. Report a summary at the end showing which paragraphs
succeeded/failed.

**Files**: `main.py` (batch mode section)

### 10.7 Enhanced /health endpoint

Extend `/health` to report loaded handlers, memory usage (RSS), pending
requests, and uptime. Useful for monitoring in production.

**Files**: `server.py`

### Status: ALL PHASE 10 ITEMS COMPLETE

| Task | Commit |
|------|--------|
| 10.1 Lazy handler registry | `bf0fabc` |
| 10.2 Test suite markers | `bf0fabc` |
| 10.3 Server rate limiting | `bf0fabc` |
| 10.4 Audio crossfade | `bf0fabc` |
| 10.5 Synthesis caching | `bf0fabc` |
| 10.6 Batch error recovery | `bf0fabc` |
| 10.7 Enhanced /health | `bf0fabc` |

224 tests passing in 53s (was 350s before lazy registry).

---

## Phase 11: Test coverage for new features + v0.6.0 release

Phases 8-10 added caching, crossfade, lazy registry, rate limiting, batch
mode, --backend shortcut, config validation, and --lexicon — but none have
dedicated tests. This phase adds coverage, wires crossfade into chunking,
adds cache CLI commands, then cuts v0.6.0.

### 11.1 Tests for cache.py

- `_cache_key` determinism (same inputs → same key)
- `_cache_key` sensitivity (different text → different key)
- `lookup` returns None on miss
- `store` + `lookup` roundtrip
- `_evict_if_needed` drops oldest entries
- `configure` creates directory
- Disabled cache returns None

**Files**: `tests/test_cache.py`

### 11.2 Tests for crossfade_segments

- Empty list → empty array
- Single segment → returned unchanged
- Two segments → output shorter than sum (overlap region)
- Very short segments → concatenated without crash
- Crossfade doesn't clip values

**Files**: `tests/test_utils.py` (add to existing)

### 11.3 Tests for lazy handler registry

- `ALL_HANDLERS` contains "crispasr" immediately (pre-loaded)
- Accessing unknown key returns None
- `all_keys()` returns all 21 registered keys
- `__contains__` works for unloaded keys

**Files**: `tests/test_handlers.py` (add to existing)

### 11.4 Tests for rate limiting

- First request allowed
- 11th request within 60s blocked (429)
- Different IPs have independent buckets

**Files**: `tests/test_server.py` (new)

### 11.5 Tests for --backend shortcut

- `--backend kokoro` resolves to `crispasr_kokoro`
- `--backend dots-tts` resolves to `crispasr_dots_tts`
- `--backend nonexistent` produces error

**Files**: `tests/test_cli.py` (add to existing)

### 11.6 Tests for config validation

- Valid config produces no warnings
- Missing handler_function_key triggers warning
- CrispASR model missing crispasr_backend triggers warning

**Files**: `tests/test_config.py` (add to existing)

### 11.7 Wire crossfade into chunked synthesis

Call `crossfade_segments()` when concatenating chunked audio in the
synthesis pipeline. Currently chunks are just concatenated raw.

**Files**: `main.py` or handler-level integration

### 11.8 Cache CLI commands

Add `--cache-stats` and `--cache-clear` CLI actions for managing
the synthesis cache.

**Files**: `main.py`

### 11.9 Bump version and release v0.6.0

Update pyproject.toml to 0.6.0, create GitHub release with notes
covering Phases 8-11.

### Status: ALL PHASE 11 ITEMS COMPLETE

| Task | Details |
|------|---------|
| 11.1 Cache tests | 7 tests in test_cache.py |
| 11.2 Crossfade tests | 5 tests in test_utils.py |
| 11.3 Lazy registry tests | 5 tests in test_handlers.py |
| 11.4 Rate limit tests | 5 tests in test_server.py |
| 11.5 --backend tests | 3 tests in test_config.py |
| 11.6 Config validation tests | 3 tests in test_config.py |
| 11.7 Crossfade utility | Available; CrispASR handles its own chunking |
| 11.8 Cache CLI | --cache-stats, --cache-clear |
| 11.9 Version bump | v0.6.0 |

254 tests passing in ~60s.

---

## Phase 12: CrispASR v0.8.12 sync + ecosystem updates

Synced 2026-07-16. CrispASR added 2 new TTS backends, native C2PA signing
on-by-default, and AudioSeal wiring.

### 12.1 New backend configs

| Model ID | Backend | Sample Rate | Cloning | Notes |
|----------|---------|-------------|---------|-------|
| `crispasr_omnivoice_tts` | `omnivoice` | 24000 | Yes (HuBERT) | 600+ langs, masked iterative |
| `crispasr_moss_tts_local` | `moss-tts-local` | 48000 | No | 4B transformer, ~2.1 GB F16 |

### 12.2 Skip C2PA for CrispASR backends

CrispASR v0.8.8+ has native C2PA signing built in (self-signed by default).
Python-side `c2pa_sign_file()` is redundant for CrispASR handler outputs.

### 12.3 Add --tts-speed to param_map

New OmniVoice flag for target-length estimate.

### 12.4 Update voice-cloning keywords

Add `omnivoice` to VOICE_CLONING_MODEL_KEYWORDS.

### 12.5 Update ecosystem references

- CrispASR version: 0.8.12 (was 0.8.7)
- CrispASR TTS backends: 20+ (was 18+)
- Handler count: 37+ (was 35+)
- Handler docstring: 16 backends (was 14)

### 12.6 Tests

- Config validation for new backends
- Voice-cloning keyword detection for omnivoice
- C2PA skip logic for crispasr handler
- tts_speed param passthrough

### Status: ALL PHASE 12 ITEMS COMPLETE

257 tests passing in ~84s.

---

## Phase 13: EU AI Act compliance audit + c2pa-audio (v0.7.1)

> **SUPERSEDED by Phase 16.** The "all gaps closed" claim below was accurate
> for `dbbcb21` but is no longer true: `server.py` and `cache.py` landed
> afterwards and reopened several of these gaps on paths that did not exist
> when this audit was run. See Phase 16 for the current state.

Implemented 2026-07-16. Full Art. 50 audit identified 6 compliance gaps,
all closed in commit `dbbcb21`.

### 13.1 Compliance gaps closed

| Gap | Issue | Fix |
|-----|-------|-----|
| 1+5 | Streaming output had no metadata | WAV LIST/INFO injected after file copy |
| 2 | `--play-direct` without file skipped watermark | Temp file created, watermarked, then played |
| 8 | MP3 had metadata only, no audio watermark | decode→embed→re-encode via pydub |
| 10 | MP3 voice-cloning had no spoken disclaimer | Disclaimer prepended to MP3 output |
| 6 | Disclaimer failure logged at DEBUG (invisible) | Raised to INFO |
| 9 | mutagen missing logged at DEBUG | Raised to WARNING with install hint |

### 13.2 c2pa-audio native signing

Integrated [CrispStrobe/c2pa-audio](https://github.com/CrispStrobe/c2pa-audio)
(~160 KB, no Rust, no OpenSSL) as preferred C2PA signer. Falls back to
c2pa-python (~10 MB). Uses bundled self-signed cert by default — no
cert/key configuration needed for basic signing. Supports WAV, MP3, M4A.

### 13.3 Compliance tests

8 new tests covering all output paths:
- C2PA native signing fallback
- WAV watermark roundtrip detection
- WAV/MP3 metadata AI-generated tag injection
- Voice-cloning keyword coverage (all 8 cloning backends)
- Spoken disclaimer generation
- Consent audit log path
- Live: piper metadata injection

| Commit | Tests | CI |
|--------|-------|----|
| `dbbcb21` | 257 pass | py3.10/3.11/3.12 + ruff ✓ |
| `85c759a` | 265 pass | py3.10/3.11/3.12 + ruff ✓ |

Released as [v0.7.1](https://github.com/CrispStrobe/CrispTTS/releases/tag/v0.7.1).

---

## Phase 14: Remaining opportunities

### 14.1 SSML-lite preprocessing

Support a subset of SSML tags in input text, translated to backend-specific
controls before synthesis:

- `<break time="500ms"/>` → insert silence
- `<prosody rate="fast">` → map to `--speech-speed`
- `<say-as interpret-as="characters">ABC</say-as>` → spell out
- `<phoneme ph="...">` → pass to `--lexicon`-style phoneme override

**Design**: Parse SSML in a preprocessor (new `ssml.py`), strip tags and
emit a sequence of (text, params) tuples. Synthesize each segment with
its params, concatenate with `crossfade_segments()`.

**Files**: `ssml.py` (new), `main.py` (wire into synthesis), `utils.py` (silence insert)

### 14.2 Progress indication for long synthesis

Emit progress to stderr during synthesis:
- Batch mode: `[3/12 paragraphs]` (already partially done)
- Chunked synthesis: `[chunk 2/5]`
- Server: progress header not possible (HTTP), but log it

**Files**: `main.py`

### 14.3 Concurrent batch synthesis

`--batch --jobs N` synthesizes N paragraphs in parallel using a thread pool.
Default: 1 (sequential). Useful for CrispASR backends that are CPU-bound.

**Files**: `main.py`

### 14.4 Audio normalization

`--normalize` flag to apply peak or LUFS normalization to output audio.
Ensures consistent volume across different backends and voices.

**Files**: `utils.py` (new `normalize_audio()` function), `main.py` (CLI flag + post-processing)

### 14.5 Model warm-up for server

`--warm-up MODEL_ID` flag for the server to pre-synthesize a short phrase
at startup, ensuring the first real request isn't slow. Useful for backends
that lazy-load models.

**Files**: `server.py`

### Status: ALL PHASE 14 ITEMS COMPLETE

| Task | Commit |
|------|--------|
| 14.1 SSML-lite preprocessor | `2f84b8a` |
| 14.2 Progress indication | Already covered by batch + SSML logging |
| 14.3 Concurrent batch (--jobs) | `2f84b8a` |
| 14.4 Audio normalization (--normalize) | `2f84b8a` |
| 14.5 Server warm-up (--warm-up) | `2f84b8a` |

281 tests passing in ~34s.
Released as [v0.8.0](https://github.com/CrispStrobe/CrispTTS/releases/tag/v0.8.0).

---

## Phase 15: Consistency fixes

### 15.1 Server SSML parsing

Server API `/v1/audio/speech` now parses SSML tags in input text,
matching CLI behavior.

### 15.2 --warm-up wired through main.py

`--warm-up MODEL_ID` and `--server-rate-limit N` now available as
main.py CLI flags when using `--server`.

### 15.3 Versioned cache keys

Cache key hash includes the package version. Upgrades auto-invalidate
stale entries, preventing serving unwatermarked audio from old cache.

### Status: COMPLETE

| Commit | Tests |
|--------|-------|
| `dcb8545` | 285 pass |

---

## Phase 16: EU AI Act Art. 50 re-audit (v0.9.0)

Audited 2026-08-01, one day before Art. 50 becomes applicable (2 Aug 2026).
The Phase 13 claim that "all gaps are closed" is **stale**: the server
(`server.py`) and synthesis cache (`cache.py`) were added after `dbbcb21`
and reopened five of them, plus five new issues found on re-audit.

### 16.0 Regulatory context

| Item | Status |
|------|--------|
| Art. 50 applicable | **2 Aug 2026** — not deferred by the Digital Omnibus |
| Art. 50(2) marking, systems on market before 2 Aug 2026 | Grace period to **2 Dec 2026** |
| Annex III high-risk | Deferred to Dec 2027 — not applicable here (no biometric ID, no emotion recognition) |
| EUPL-1.2 / FOSS exemption (Art. 2(12)) | Does **not** exempt Art. 50 — expressly carved back in |
| Art. 50(4) deepfake disclosure | Binds the **deployer**, not this tool; we can only enable it |

Whether the maintainer is a "provider placing on the market" at all is
arguable for non-commercial FOSS distribution. Phase 16 assumes the
stricter reading and makes the tool sufficient for a deployer to comply.

### 16.1 Fail-closed marking + soundfile as a core dependency

**Problem.** `main.py:828`, `server.py:268`, `utils.py:321` wrap the whole
marking block in a `try` that also covers `import soundfile`. `soundfile`
is not a core dependency (`pyproject.toml` ships `requests` + `numpy`
only), so a default install raises ImportError, logs at **DEBUG**, and
emits a completely unmarked WAV. Every other failure logs a warning and
still ships the file. No fail-closed mode exists.

**Fix.** Promote `soundfile` to a core dependency. Add
`CRISPTTS_ALLOW_UNMARKED=1` / `--allow-unmarked` as the only escape hatch.
Default behaviour on marking failure: log ERROR, delete the output file,
exit non-zero (CLI) or return HTTP 500 (server).

**Files**: `pyproject.toml`, `watermark.py`, `main.py`, `server.py`, `utils.py`

### 16.2 Central `mark_audio_file()` — one marking path

**Problem.** Marking is reimplemented four times (`main.py:840` single
synthesis, `main.py:461` `--test-all`, `server.py:277`, `utils.py:343`),
each with different coverage. This is the root cause of 16.3–16.6.

**Fix.** One function in `watermark.py`:

```python
def mark_audio_file(path, *, handler_key=None, is_voice_cloning=False,
                    allow_unmarked=False) -> MarkResult
```

It embeds the PCM watermark at the file's **true** sample rate, injects
container metadata, optionally C2PA-signs, verifies the result, and returns
a structured `MarkResult` (backend used, confidence, layers applied). It is
idempotent — an already-marked file is detected and not re-marked. All four
call sites collapse onto it.

**Files**: `watermark.py`, `main.py`, `server.py`, `utils.py`

### 16.3 Audio watermark on compressed formats everywhere

**Problem.** `server.py:289` and `main.py:477` (`--test-all`) inject ID3 /
Vorbis tags but never call `watermark_embed`, unlike the CLI single-synthesis
path which does a proper decode→embed→re-encode (`main.py:855`). Tags do not
survive transcoding. `readme.md:543` ("All outputs are watermarked — CLI,
`--test-all`, and API server responses") is therefore inaccurate.

**Fix.** Folded into 16.2 — `mark_audio_file()` handles wav/mp3/flac/opus
uniformly with a real PCM embed.

### 16.4 No double embedding; idempotent metadata

**Problem.** The five handlers that use `save_audio` (edge, piper, coqui,
kokoro_onnx, mlx_audio) are watermarked at `utils.py:445` and then **again**
at `main.py:840`. Measured cost: SNR 33.6 dB → **27.5 dB**.
`inject_wav_metadata` (`watermark.py:525`) is not idempotent either, so
those files carry two LIST/INFO chunks — `inject_mp3_metadata` does guard
(`watermark.py:584`).

**Fix.** `save_audio` stops marking; marking becomes the caller's
responsibility via 16.2. Add an idempotency guard to `inject_wav_metadata`
and an `is_marked()` probe.

### 16.5 Correct sample rate to the neural backends

**Problem.** `utils.py:325` and `utils.py:355` call `wm.watermark_embed(data)`
with no `sample_rate`, defaulting to 24000 (`watermark.py:404`). A 16 kHz or
44.1 kHz file is watermarked as though it were 24 kHz while detection passes
the true rate — so WavMark/AudioSeal, the *preferred* backends, are the ones
silently broken on this path.

**Fix.** Folded into 16.2; the true rate is always read from the file.

### 16.6 Truthful server provenance header

**Problem.** `server.py:204` and `server.py:328` send
`X-CrispTTS-Watermarked: true` unconditionally — on cache hits, under
`CRISPTTS_NO_WATERMARK`, and when the embed raised and was swallowed at
`server.py:300`. A false machine-readable provenance claim is worse than none.

**Fix.** Derive the header from the `MarkResult` of 16.2. Add
`X-CrispTTS-Watermark-Backend` and `X-CrispTTS-Watermark-Confidence`.

### 16.7 Cache: consent gate first, marking state in the key

**Problem.** `server.py:187-210` runs the cache lookup **before** the consent
gate at `server.py:212-227`. Once any caller has synthesised a cloned phrase
with `i_have_rights`, every later caller receives it with no 403 and **no
attestation logged**. Separately, `cache.py:43` hashes only
model/voice/text/params/version, so audio produced once under
`--no-watermark` is served indefinitely to marking-enabled requests.

**Fix.** Move the consent gate above the cache lookup; log the attestation on
hits too. Add marking mode + disclaimer state to the cache key.

**Files**: `server.py`, `cache.py`

### 16.8 Robust-by-default watermark

**Problem.** Measured on a 20 s speech sample:

| Condition | Confidence |
|---|---|
| After embed | 0.938 |
| After 24k→16k→24k resample | **0.625** |
| Documented threshold | 0.65 |
| Real human recording (false-positive check) | 0.438 |

A plain resample drops the built-in watermark below its own detection
threshold. Measured SNR is 33.6 dB, not the ~38 dB claimed at
`readme.md:550`. The robust neural backends are optional extras, so the
default install ships the weakest layer — against Art. 50(2)'s "robust and
reliable **as far as technically feasible**".

**Decision.** WavMark pulls in `torch` (~2 GB), which is too heavy to force
on every install of a CLI. Instead: a one-time prominent WARNING when
synthesis runs on the bare spread-spectrum backend, a `robust` extra, install
docs recommending it, and honest measured numbers in the README. Making it a
hard dependency is a one-line change in `pyproject.toml` if that trade is
preferred later.

**Files**: `watermark.py`, `pyproject.toml`, `readme.md`

### 16.9 Coherent `--no-watermark` semantics

**Problem.** `--no-watermark` (`main.py:1094`) suppresses the PCM embed
(`watermark.py:417`) but `utils.py:359,374` still inject metadata and
C2PA-sign. For a provider under Art. 50(2), marking is not an end-user
preference in any case.

**Fix.** Make it disable **all** layers coherently, print a prominent
stderr warning that the output is unmarked and the user carries the Art. 50
responsibility, and document it as debug-only.

### 16.10 C2PA self-signed disclosure

**Problem.** `utils.py:374` signs with the c2pa-audio bundled **self-signed**
cert. Those manifests fail trust-list validation but are presented as
"signed provenance credentials".

**Fix.** Return signer identity from `c2pa_sign_file`, log
self-signed vs CA-issued distinctly, qualify the README.

### 16.11 Residual items

- `--play-direct` with `--output-file` plays pre-watermark audio
  (`main.py:750`) — always synthesize → mark → play.
- Consent log records no evidence (`watermark.py:695`) — add a SHA-256 of the
  reference audio.
- Keyword-based cloning detection (`watermark.py:666`) has zero misses today
  (verified: 32/61 models gated) but fails **open** for future backends — add
  an explicit `voice_cloning: true` config key with the keywords as fallback,
  plus a test asserting coverage.

### 16.12 Documentation corrections

- `readme.md:543` — "all outputs watermarked" claim, narrow to what is true.
- `readme.md:550` — 38 dB → measured 33.6 dB.
- `PLAN.md` Phase 13 — mark the "all gaps closed" claim as superseded.
- README: Art. 50 section stating what the tool does and what the **deployer**
  must still do (Art. 50(4) disclosure is theirs, not ours).

### Execution order

1. 16.2 central `mark_audio_file()` (unblocks 16.3, 16.4, 16.5)
2. 16.1 fail-closed + `soundfile` core dep
3. 16.9 coherent `--no-watermark`
4. Wire call sites: `utils.py`, `main.py` (single + `--test-all`), `server.py`
5. 16.7 cache + consent ordering
6. 16.6 truthful headers
7. 16.8 robustness warning + extra
8. 16.10 C2PA disclosure
9. 16.11 residual items
10. Tests for every item above
11. 16.12 docs

### Status: COMPLETE (v0.9.0)

| Item | Change | Verified by |
|------|--------|-------------|
| 16.1 | Fail-closed marking; `soundfile`/`pydub`/`mutagen` promoted to core deps; `--allow-unmarked` escape hatch | `test_fails_closed_on_unsupported_format`, `test_unmarkable_output_is_discarded` |
| 16.2 | `watermark.mark_audio_file()` — one marking path for all four call sites | `TestMarkAudioFile` (10 tests) |
| 16.3 | Real audio watermark on MP3/FLAC/Opus everywhere, not just metadata | `TestMp3Marking` |
| 16.4 | `save_audio()` no longer marks; `inject_wav_metadata` idempotent; `is_marked()` probe | `test_marking_is_idempotent`, `test_no_duplicate_metadata_chunks` |
| 16.5 | True sample rate always passed to the embed | `test_uses_true_sample_rate_not_default` (16k + 44.1k) |
| 16.6 | Provenance headers derived from `MarkResult` | `TestProvenanceHeaders` |
| 16.7 | Consent gate moved above cache lookup; marking mode in cache key | `TestConsentBeforeCache`, `TestMarkingCacheKey` |
| 16.8 | One-time weak-backend warning, `robust` extra, measured numbers documented | manual |
| 16.9 | `--no-watermark` disables every layer coherently | `test_no_watermark_env_disables_every_layer` |
| 16.10 | `c2pa_sign_file_ex()` reports self-signed vs CA-issued | `TestC2paSignerDisclosure` |
| 16.11 | Playback after marking; consent log records reference digest; explicit `voice_cloning` config key | `test_handler_never_plays_unmarked_audio`, `TestConsentAuditEvidence`, `TestConsentGateConfig` |
| 16.12 | README Art. 50 section, corrected claims; Phase 13 marked superseded | — |

**Measured after the fix** (20 s of speech, full pipeline through a
`save_audio` handler): detection confidence 0.938, SNR **33.6 dB** (was
27.5 dB while double-embedding), exactly one LIST/INFO chunk.

314 tests pass (31 new), ruff clean, bandit clean of medium/high findings.

---

## Phase 17: Gate generation on sufficient marking (v0.9.1)

Cross-checked against the sibling projects and adopted the strongest policy
from each. Phase 16 made marking *fail closed when it errored*; it still
shipped audio whose mark was applied but undetectable.

### 17.1 What the siblings do

| | Mechanism | Where |
|---|---|---|
| **CrispASR** | *Watermark floor* — if the container can't carry a C2PA manifest, `--no-watermark` is overridden so at least one robust mark remains | `examples/cli/crispasr_run.cpp:141` |
| **CrispASR** | *Marking attestation* — any provenance opt-out hard-refuses without `--accept-marking-responsibility`; emits a `[MARKING]` audit line | `crispasr_run.cpp:163` |
| **Susurrus** | Dependency-free declarative RIFF marker so a default install still marks; `--accept-marking-responsibility` as a single explicit opt-out; consent refused before any model loads | `utils/ai_marking.py`, `workers/tts/backends/base.py:64` |
| **CrisperWeaver** | LSB watermark + consent gate on the HTTP server | `lib/services/audio_watermark_service.dart` |

Gaps found in the siblings while reading them (not fixed here — different
repos): Susurrus `apply_provenance` only *warns* when both layers fail and is
WAV-only, so MP3 output is silently unmarked; CrisperWeaver's `embedWatermark`
returns the input unchanged for audio under 4 608 samples.

### 17.2 The hole this closes in CrispTTS

Marking reported success on audio that carried no detectable watermark:

| Input | Reported | Actually detectable |
|---|---|---|
| 0.02 s (under one FFT frame) | `marked=True` | 0.000 |
| 0.1 s | `marked=True` | 0.500 |
| digital silence | `marked=True` | 0.000 |

The embed silently no-ops below one FFT frame and on silence, leaving only
strippable container metadata. Detection is reliable from ~0.25 s upward
(0.69 at 0.25 s, 0.91 from 0.5 s) and is level-invariant.

### 17.3 Implemented

- `preflight_marking()` — refuses **before** synthesis when the output cannot
  be marked (unsupported container, missing codec dependency), so no model is
  loaded and no unmarkable audio is ever written. Wired into `run_synthesis`
  and the server's `do_POST`.
- Verification is now a **gate**: `mark_audio_file()` reads the mark back and
  refuses when confidence < 0.65 and no C2PA manifest was produced.
  Container metadata alone never counts as sufficient.
- CrispASR-backend output is **verified rather than trusted** — an old build or
  one run with its own `--no-watermark` no longer slips through on `handler_key`.
- Watermark floor ported from CrispASR, including `watermark_embed(force=True)`
  so the env opt-out cannot strip a forced mark.
- `--accept-marking-responsibility` gates every provenance opt-out; honoured
  opt-outs emit a `[MARKING]` audit line in CrispASR's format.

### Status: COMPLETE — 329 tests pass (15 new), ruff clean

### Known limitation — RESOLVED in 18.3

*Recorded as deliberately-not-closed at the time; superseded, kept for the
reasoning.* The built-in spread-spectrum watermark did not survive resampling
(0.63 against a 0.65 threshold). The assumption was that closing it meant
either a hard `torch` dependency or original signal-processing work on
rate-invariant embedding, so the mitigation was the `robust` extra plus a
loud warning and honest documentation — see 16.8.

Neither turned out to be necessary. Moving the comb into the speech band for
*interop* reasons (18.3) raised post-resample detection to 0.78 as a side
effect, because the bins now sit where the signal actually has energy. The
immediate-after-embed reading fell from 0.94 to 0.84 in exchange, and the
embed got ~6 dB quieter.

---

## Phase 18: Ecosystem interop, C2PA tiering, offline disclosure (v0.9.1)

Phases 16–17 made marking fail closed and gated generation on it. This phase
came out of actually *installing* CrispASR 0.8.25 and probing it, rather than
reasoning about what it probably does — which is how three of the four
defects below were found.

### 18.1 Tiered C2PA signers, each verified rather than trusted

`CRISPTTS_C2PA_BACKEND=auto|python|audio|crispasr|off`:

1. `c2pa-audio` if importable — native, built from source, not on PyPI
2. `c2pa-python` — always present, and the only path where CrispTTS controls
   the manifest

`c2pa-audio`'s `sign_wav()` takes a cert and a key but no manifest, so the
library picks its own assertions — and a manifest without
`trainedAlgorithmicMedia` marks a file as *unaltered* rather than as
*AI-generated*. That is exactly how the original `c2pa-audio` path came to
sign files carrying no AI claim. `manifest_asserts_ai()` now reads every
native result back; if the claim is absent the result is discarded and
`c2pa-python` re-signs.

### 18.2 CrispASR is not a signing backend, and its watermark is not assumed

Two claims about the sibling project turned out to be false when probed:

- A CrispASR signing tier was added by matching `--c2pa` in `--help`, which
  matches as a substring of `--c2pa-cert` and builds a command the binary
  rejects. Probed against 0.8.25: `--c2pa-cert`/`--c2pa-key` configure signing
  of its *own* synthesis output; no flag signs an existing file. Tier removed.
- `audio-watermark:upstream` was reported for any `handler_key == "crispasr"`
  output. Measured on 0.8.25 kokoro output, CrispTTS's detector read 0.44 —
  its noise floor. The layer is now reported only when verification actually
  detects a mark.

**Upstream manifests are preserved, not overwritten.** CrispASR signs during
synthesis with a manifest that already asserts `trainedAlgorithmicMedia`.
Every marking step rewrites the file and breaks that manifest's hash —
injecting the WAV `LIST/INFO` chunk alone takes it from `validation_state:
Valid` to `Invalid`. Re-signing afterwards hid that, at the cost of discarding
the upstream signer identity and leaving a tamper-looking file whenever the
re-sign failed. Such files are now left untouched, reported `c2pa:preserved`.

### 18.3 Watermark comb moved to CrispASR's speech band

The two projects could no longer read each other's marks: same key, same PRNG,
same FFT size and hop — only the comb placement differed. CrispASR moved theirs
in their #260 because spreading 32 bins across ~1.5–11.7 kHz put ~20 of them
where clean TTS speech is near-silent, making the comb audible as a tinny tone.
CrispTTS stayed on the old band.

| | CrispASR's detector | CrispTTS's detector |
|---|---|---|
| Before, on CrispASR kokoro output | 0.72 detected | **0.41 — missed** |
| After, both directions | 0.81 | 0.81 |

`wm_params()` now mirrors the C++ exactly (lo=`n_fft/16`, hi=`n_fft/5`,
alpha 0.05); `CRISPASR_WATERMARK_LEGACY=1` restores the old band and is read
by both projects. Detection sweeps both bands and takes the stronger reading,
so previously-marked audio still verifies. `alpha=0` is now a true no-op as
the C++ documents — the STFT round-trip previously perturbed the signal while
embedding nothing.

This is what resolved the Phase 17 known limitation; see the note there.

### 18.4 Bundled spoken disclosures — the offline tier

Pre-rendered clips in `crisptts_assets/`, mono 16 kHz FLAC, as tier 3 after
CrispASR kokoro and Edge TTS. The Art. 50(4) spoken disclosure now works with
no TTS backend, no model download and no network — previously the only remedy
for that configuration was to discard the cloned output. A bundled clip is a
real spoken sentence in the right language, so it counts as a disclosure; the
tone marker is still refused.

Disclosure wording aligned with Susurrus's `disclosure.spoken` for de/en, with
the other languages following the same "the following audio" phrasing — the
disclosure is prepended, so it describes what comes after it, not itself.

### 18.5 Dependency and CI floors

- `edge-tts >= 7.2`. Below that the Sec-MS-GEC auth token uses a scheme
  Microsoft has retired and every request 403s at the websocket handshake
  (measured minutes apart on one machine: 7.0.2 fails, 7.2.8 succeeds). The
  previous `>=6.0` floor let a fresh environment resolve to a version that
  cannot work. Survivable now that tier 3 exists, but a silently broken tier 2
  is still a broken tier.
- CI installs ffmpeg. The MP3 marking tests skip themselves without it, so the
  MP3 path — a core provenance path, and the reason pydub is a hard dependency
  — had never actually run in CI.

### Status: COMPLETE (v0.9.1)

---

## Phase 19: Art. 50 audit — disclosure language and fail-open gaps

Audited 2026-08-02, the day Art. 50 became applicable. Phases 16–18 hardened
*marking*; this audit looked at the paths around it. The marking core held up:
every file-writing path routes through `mark_audio_file()`, verification gates
rather than warns, preflight refuses before a model loads. Five findings, one
substantive.

### 19.1 The disclosure was always German (the defect)

The spoken Art. 50(4) disclosure took its language from a static config field
that was either `"de"` or absent, so **all 34 cloning models emitted a German
disclosure** — including CosyVoice3, OmniVoice, IndexTTS and
LLaSA-Multilingual. A German sentence in front of Mandarin audio discloses
nothing to that audience. The readme's claim of "in the language of the model
being used, 8 languages" was not true of any shipped configuration.

Twenty of the 34 declared no `language` at all, and the silent fallback made
that indistinguishable from a deliberate `"de"`.

- `resolve_disclaimer_lang()` returns `(lang, known)`. `"multilingual"`,
  `None` and unrecognised codes resolve as *unknown* and warn, naming
  `--disclosure-lang` and the Art. 50(4) duty, instead of becoming German.
- `--disclosure-lang` / `"disclosure_lang"` override, because a multilingual
  model's output language is a property of the input text, not of the model.
  It cannot be derived from config at all.
- Every cloning model declares a language; a test enforces it, so a new
  backend cannot skip the question the way twenty existing ones had.
- Coverage 8 → 27 languages: all 24 EU official, plus zh/ja/ko for the models
  that target them. Art. 50 governs the EU market, so "a disclosure the
  audience can understand" means any EU official language. Bundled offline
  clips for all 27 (~2 MB in the wheel), each verified with edge-tts and
  CrispASR both disabled.
- The server's cache key omitted the disclosure language, so an `en` request
  could be served a German-disclosed clip. Cloned audio carries its disclosure
  *inside* it, so the language has to participate in the key.

### 19.2 The consent gate failed open

`except ImportError: pass` at all three sites (CLI, `--test-all`, server) —
the one control standing between the tool and cloning someone's voice was also
the only one that vanished when a module was missing, while marking ten lines
below refused. An unknown cloning status is now treated as cloning, not as
permission.

### 19.3 CrispASR playback bypassed marking

`if args.play_direct and handler_key != "crispasr"`. Every other backend wrote
a file, marked it, verified, then played. CrispASR was exempt on the grounds
that its binary marks internally — the same assumption `mark_audio_file()`
already declines to make, and which 18.2 had just measured as unfounded. The
one place audio reached a listener was the one place nothing was verified.

Playback is now marked and verified first for every backend. Real `--stream`
remains exempt by necessity — incremental playback cannot wait for a completed
file — and now says so; any `--output-file` is still gated.

### 19.4 Audit log retention

The log records reference-audio *paths*, which routinely contain personal
names, so it is personal data. It was append-forever and umask-default.
Now `0600`, 730-day retention pruned on append (GDPR Art. 5(1)(e),
`CRISPTTS_CONSENT_LOG_RETENTION_DAYS` to tune), plus `--consent-log-prune` and
`--consent-log-erase [SUBJECT]` for Art. 17 — selective by reference-audio path
or `ref_sha256`, or the whole log. Lines with no parseable timestamp are kept:
an unreadable record is not evidence that it has expired.

### 19.5 Documentation

Art. 50 described as applying *now* rather than in the future. Art. 4 (AI
literacy, in force since 2 Feb 2025) added. Art. 5 / Annex III / Chapter V
recorded as checked-and-not-applicable, so the omissions are visible rather
than silent: no biometric categorisation, no emotion *recognition*
(Kartoffelbox's emotion control is synthesis), not Annex III, and
single-purpose TTS models are not GPAI, so Chapter V does not attach to models
converted by `convert_f5_to_mlx.py`.

### 19.6 A test that measured the environment

`test_streaming_still_marks_the_output_file` passed locally and failed on all
three CI Pythons. `run_synthesis` imports `synthesize_with_crispasr_streaming`
directly from the handler module, so the dispatch-table stub never intercepted
it and the test reached real code. Notably it was green locally *despite* no
`crispasr` binary on PATH, so local reasoning could not have caught it. It now
patches the streaming entry point and asserts the stub was reached.

### Status: COMPLETE — 410 tests pass (25 new), ruff clean, CI green on 3.10–3.12

### Deployer-side, and not closable in code

Publishing-point disclosure; genuine consent (`--i-have-rights` remains an
unverified self-attestation and says so); voice/model licence compliance; and
**choosing the disclosure language** — the tool now warns when it cannot
determine one, but only the deployer knows what language the audience speaks.

## Phase 20: Art. 50 audit — the two paths marking never reached

Audited 2026-08-02. Phase 19 concluded that "every file-writing path routes
through `mark_audio_file()`". That was checked by reading the call sites, and
every call site did indeed look right. Both defects below sit in the gap
between the call site and the file: marking was invoked on a path, and the
audio was somewhere else.

### 20.1 Marking followed the requested path, not the written one (the defect)

Sixteen of the twenty-three handlers force their own container regardless of
the extension asked for — `edge_handler` writes `.mp3`, most local ones
`with_suffix(".wav")`. Every post-synthesis step was guarded by
`os.path.isfile(args.output_file)`, and when the handler had written next door
that guard was simply false. Not an error, not a warning: the disclosure,
the marking, the verification and the fail-closed discard were all skipped in
silence.

Measured on the default cloud backend — `--model-id edge --output-file
out.wav` delivered `out.mp3` with watermark confidence **0.56** (below the
0.65 threshold, i.e. the detector's floor on unmarked audio), no ID3
`AI_GENERATED` tag and no C2PA manifest. Marking had reported nothing wrong
because it never ran.

`_resolve_written_output()` now finds the file the handler actually wrote —
same stem, any audio extension, newer than the moment synthesis started — and
`run_synthesis` and `--test-all` rebind to it before trimming, normalizing,
resampling, disclosing, marking, playing or discarding. The candidate set is
deliberately wider than what marking can handle: finding an output in an
unmarkable container has to end in `mark_audio_file()` refusing it, not in
never noticing the file. The `since` bound keeps a leftover from an earlier
run from being adopted as this run's output.

Two details that cost a debugging round each: `mkstemp` creates the requested
path *before* the handler runs, so an empty stub is left behind whenever the
handler writes elsewhere — the resolver requires >100 bytes, not mere
existence. And `Path("./out.wav")` normalizes to `"out.wav"`, so a resolver
returning `str(Path(...))` made the caller conclude the handler had written
somewhere else and delete the real output. It returns the caller's own string.

### 20.2 Multi-segment SSML was a second exit from run_synthesis

`<break>`/`<prosody>` input with more than one segment synthesized each
segment, crossfaded them, wrote the result and **returned** — above the
disclosure and marking block, which it therefore never reached.

The combined file kept only whatever audio watermark survived concatenation,
with nothing verifying it still cleared threshold (measured 0.78, so in
practice it did — but by luck, not by gate). Its LIST/INFO metadata and its
C2PA manifest were gone, since `sf.write` builds a fresh container. The
marking preflight had only ever seen the segments' temp `.wav`, so an
unmarkable target format was not refused. And `--play-direct` played the
result, contradicting 19.3 directly.

Segments are now rendered *in place of* the handler call rather than instead
of the rest of the function, so the combined output takes the same
trim/normalize/resample/disclose/mark/play path as any other output and
`run_synthesis` has one exit again. Segments carry `_ssml_segment`, which
suppresses per-segment disclosure (it belongs once at the front; repeated
before every segment the crossfade would partly bury all but the first) and
per-segment marking (thrown-away work — and a hazard, since marking *deletes*
what it cannot mark, so a segment failing verification would vanish from the
output instead of failing the run).

20.1 also fixed SSML with Edge, which had been silently falling back to
speaking the text without its markup: the segment loop looked for a `.wav`
that the handler had written as `.mp3`.

### 20.3 A README claim stronger than the code

"After marking, the watermark is read back. If it is not detectable above
threshold, the output is discarded." The code requires one *robust* layer —
watermark detected **or** C2PA signed. So an undetectable watermark is fatal
for FLAC and Opus, which cannot carry a manifest, while a WAV or MP3 may still
ship on its manifest alone. That is a defensible rule, and the next sentence
("container metadata alone is never sufficient") was always accurate, but as
written the passage promised a detectable watermark in every delivered file.
Corrected to state the rule the code implements.

### 20.4 The same assumption in the server, failing closed

`/v1/audio/speech` read back only its `mkstemp` path, so a handler that wrote
its own container produced "Synthesis produced no output" — a 500 rather than
unmarked audio, so not a compliance defect, but the Edge backend was
effectively unusable over HTTP.

The server converts rather than merely following: the response carries a
declared `Content-Type`, and returning MP3 bytes labelled `audio/wav` would be
worse than the error it replaces. `save_audio()` re-containers to the
requested format, and a missing codec leaves the path unwritten and falls into
the same 500 — still fail-closed. `resolve_written_output()` moved to
`utils.py` so both callers share one implementation.

Verified live: `{"model":"edge","response_format":"wav"}` now returns a real
`WAV PCM_16`, watermark 0.84, C2PA manifest present.

### Status: COMPLETE — 430 tests pass (20 new), ruff clean

Both defects were invisible to the 410 tests that preceded them, for the same
reason: every stub handler wrote to the path it was handed, which no shipped
handler reliably does, and no test combined SSML with marking. The new tests
use handlers that override the extension, and were confirmed to fail against
the previous `main.py`.

Two of the new tests pass against the old code by design — the stale-file
guard and the every-segment-survives check cover behaviour the rewrite
introduces rather than bugs it fixed, and say so.

### Deployer-side, and not closable in code

Unchanged from Phase 19.

## Phase 21: C2PA for FLAC and M4A; Opus gated on a neural watermark

Phase 20 left FLAC and Opus with the audio watermark as their only robust
layer, and the built-in spread-spectrum comb as that layer on a default
install. The question was whether C2PA could cover them. For two of the three
containers it can — the exclusion rested on a measurement of the wrong API.

### 21.1 FLAC and M4A were never actually unsignable

`C2PA_CAPABLE_EXTS` was `{".wav", ".mp3"}`, with a comment recording that
c2pa-rs advertises FLAC and M4A in `get_supported_mime_types()` but fails them
with `NotSupported: type is unsupported`. That is true — of
`Builder.sign_file()`, which is what the code called.

The streaming `Builder.sign(signer, format, source, dest)` signs both.
Measured on c2pa-python 0.37.2 with the bundled dev certificate:

| Container | `sign_file()` | `sign()` streaming |
|---|---|---|
| WAV | ok | ok |
| MP3 | ok | ok |
| FLAC | `NotSupported` | **ok** — `validation_state: Valid` |
| M4A | `NotSupported` | **ok** — `validation_state: Valid` |
| Opus/OGG | `NotSupported` | `NotSupported` |

Both signed files read back with `manifest_asserts_ai() == True`, the correct
`digitalSourceType: trainedAlgorithmicMedia`, the CrispTTS `softwareAgent`,
and audio still decodable afterwards. The stream API is a strict superset —
WAV and MP3 sign identically through it — so `_sign_with_c2pa_python()` now
uses it exclusively, and `C2PA_CAPABLE_EXTS` gains `.flac` and `.m4a`.

Widening that set is not cosmetic: it feeds the watermark floor, so an entry
that could not really sign would let `--no-watermark` be honoured for a file
carrying nothing but strippable metadata. The existing
`test_every_capable_ext_really_signs` signs every listed extension for real
and now additionally asserts the manifest carries the AI claim.

One trap worth recording: probing this with a `BytesIO` destination and
checking `.tell()` reports 0 bytes for every format, including ones that
signed perfectly — the library seeks back. Measured against real files
instead. `_sign_with_c2pa_python()` now treats a zero-byte result as failure
rather than success for the same reason.

### 21.2 Opus/OGG requires a neural watermark

Opus is not in c2pa-rs's supported types at all, and `opus`, `audio/opus`,
`ogg`, `audio/ogg` and `application/ogg` all return `NotSupported`. A detached
`.c2pa` sidecar *can* be produced (verified: 13 KB, signs fine via
`set_no_embed()`), but a manifest travelling as a separate file is a
provenance record the operator holds, not a mark on the content — it does not
count toward Art. 50(2), and the container cannot even carry a pointer to it,
since `set_remote_url` needs the same missing embed support.

So for Opus/OGG the watermark is the sole robust layer, and on a default
install that means the fixed-key comb. `preflight_marking()` now refuses those
containers unless a neural backend is installed, naming all three ways
forward: `pip install 'crisptts[robust]'`, a manifest-carrying container, or
`--allow-unmarked --accept-marking-responsibility`.

`neural_watermark_available()` is a package-presence check via
`importlib.util.find_spec`, deliberately not a load — preflight runs before
any model is pulled in and `--list-models` has to stay instant.

The gate sits inside the existing `handler_key != "crispasr"` block. CrispASR
embeds its own watermark in the binary, so a Python-package check says nothing
about what that path will produce; its Opus output stays gated by the
post-marking verification instead, which for a manifestless container already
requires the watermark to verify.

### Status: COMPLETE — 433 tests pass (3 new), ruff clean

Two pre-existing tests asserted the old behaviour (`output_carries_c2pa` false
for FLAC; `.opus` passing preflight unconditionally) and were updated rather
than deleted — both now state the rule the code implements and why.

## Phase 22: Art. 50 audit — the marker that stood in for evidence

Audited 2026-08-02. Phase 20 found two paths marking never reached. This one
found a path marking *did* reach and declined to do anything on, plus a gap in
what triggers the Art. 50(4) spoken disclosure at all.

### 22.1 A metadata string was accepted as proof of marking (the defect)

`mark_audio_file()` opened with `if is_marked(filepath): return
MarkResult(marked=True, layers=("already-marked",))`. `is_marked()` is a scan
for a **byte string in container metadata** — the strippable layer the same
function's own comment says "never qualifies". That early return sat above the
manifest check, the watermark embed, the C2PA signing and the verification
gate, so a file carrying only the marker was reported marked and delivered
with nothing else.

Reachable through `crispasr_handler.synthesize_with_crispasr_streaming()`,
which injected the WAV LIST/INFO chunk itself before returning. Measured,
`--model-id crispasr_kokoro --stream --output-file out.wav`:

| | delivered file |
|---|---|
| watermark confidence | **0.625** (threshold 0.65 — the detector's floor) |
| C2PA manifest | none |
| container marker | present |
| after a plain transcode | nothing left |

`mark_audio_file` returned `marked=True`. This is the same shape as 19.3 and
20.1: the gate was invoked and did not run. It also makes Phase 19's "any
`--output-file` is still gated" false as written.

The marker now sets `already_marked`, which only suppresses *redundant work*:
the PCM re-embed is skipped when `_existing_watermark_detectable()` measures a
watermark that is really there, and metadata injection is skipped because it is
already in the container. Signing and the verification gate run regardless.
The CrispASR streaming handler no longer marks at all — a handler writes audio,
marking has one owner — which removes the trigger as well as the fault.

Moving the check also fixed an ordering bug behind it: `is_marked` preceded the
`preserved_manifest` branch, so a file with both metadata and an AI-asserting
manifest was reported `already-marked` with its watermark never measured,
instead of `c2pa:preserved`.

### 22.2 The deepfake disclosure keyed on the mechanism, not the resemblance

The spoken Art. 50(4) disclosure fired on `voice_cloning` alone, so the 27
models declaring `voice_cloning: false` never received one. Several are
finetunes of named, identifiable people — `coqui_tts_thorsten_{ddc,vits,dca}`
(Thorsten Müller's published voice), `coqui_css10_de_vits`,
`coqui_vctk_en_vits` (109 recorded individuals), and the Piper community voice
catalogue used by `piper_local` and `crispasr_piper`.

Art. 3(60) defines a deep fake by what the output *resembles*, not by how the
resemblance was produced. A voice donor consenting to their recordings being
used for training is a licensing fact; it is not the audience knowing the audio
is synthetic.

Every non-cloning model now declares `speaker_identity`:
`real_person` (13) gets the disclosure exactly as cloning does, `synthetic` (7)
does not, `unknown` (7) warns once per model naming Art. 3(60) and
`--speaker-identity`. `unknown` deliberately does not force a disclosure —
the same reasoning as 19.1's `"multilingual"`: surface the question rather than
guess, in either direction. `--speaker-identity` / `"speaker_identity"`
overrides per run, and joins the server cache key for the same reason
`disclosure_lang` did in 19.1 — it decides whether a disclosure is *in* the
audio.

A test asserts every `voice_cloning: false` model declares a valid value, so a
new backend cannot skip the question the way all 27 of these did.

### 22.3 Art. 50(1) and 50(5) recorded

Phase 19.5 recorded Art. 5, Annex III and Chapter V as checked-and-not-
applicable. Art. 50(1) and 50(5) were neither implemented nor recorded.

- **50(1)** — not applicable, and now says so: CrispTTS holds no conversation.
  Embed it in something that talks to people and that system carries the duty.
- **50(5)** — the spoken disclosure is audio, so it does not reach a deaf or
  hard-of-hearing audience. Added to the deployer list: carry the disclosure
  sentence into captions and transcripts. It is the first thing in the audio,
  so a verbatim transcript already has it; the duty is not to strip it.

### Status: COMPLETE — 439 tests pass (6 new), ruff clean

22.1 was invisible to the 433 tests that preceded it for a specific reason:
`test_streaming_still_marks_the_output_file` patches `mark_audio_file` with a
stub, so it proved the call happened and could not see that the real one
returned early. The two new regression tests run the real marking path and were
both confirmed to fail against the previous code.

### Deployer-side, and not closable in code

Unchanged from Phase 19, plus: answering `speaker_identity` for the models
recorded as `unknown`, and carrying the disclosure into captions (50(5)).

### 22.4 The values, from each model's own documentation

The first pass classified from repo evidence alone and left 13 `unknown`.
Reading the upstream model cards settled six of them, all toward
`real_person` — the direction that adds a disclosure, which is why guessing
`synthetic` would have been the costly error:

| Model | Evidence |
|---|---|
| `orpheus_kartoffel_natural`, `crispasr_orpheus_de` | "fine-tuned primarily on natural human speech recordings" — podcasts, lectures, OER. The 19 speakers were *extracted* from those recordings ("not all speakers could be reconstructed"), so they are real people who spoke in public |
| `orpheus_sauerkraut`, `orpheus_lm_studio` | The card's speaker table: `Tom` and `Anna` include 1–3 h of original studio recordings each; `Max` and `Lena` are wholly synthetic. Default voice is `Tom` |
| `speecht5_german_transformers` | Fine-tuned on Common Voice German; the voice heard is whichever CMU ARCTIC x-vector is selected — seven identifiable recorded people (bdl, slt, jmk, awb, rms, clb, ksp), pseudonymous like VCTK's p225 |
| `fastpitch_german_nemo` | HUI-Audio-Corpus-German, whose narrators are named: Bernd, Hokuspokus, Friedrich, Eva, Karlsson, Sonja. Eva and Karlsson are the same donors as the Piper voices already marked `real_person` — the same corpus reached by another route |

Seven stay `unknown`, now with the check recorded in `config.py` so it is not
re-litigated: the four Orpheus entries serving Canopy Labs' base voices (100k+ h
of "permissive" audio disclosed, nothing about `tara`/`leah`/…), `edge`
(Microsoft's transparency note defines "voice talent" only for *custom* neural
voice), `crispasr_melotts` and `crispasr_bananamind_tts`.

A mixed model like SauerkrautTTS is classified by what it *can* speak as, not
by its safest voice: `speaker_identity` is per model, and the fail-safe
direction is the one that discloses.

## Phase 23: Art. 50 audit — the strength the band migration left behind (v0.9.3)

Audited 2026-08-02, hours after Phase 22. Phases 16–22 hardened *which paths*
get marked. This one asked what the mark actually is, and measured it instead
of reading the table. Two of the three findings are in documentation, and the
documentation was describing a watermark the tool does not embed.

### 23.1 The legacy alpha survived the move into the speech band (the defect)

`wm_params()` returns a per-band default strength: 0.05 for the speech band,
0.08 for the legacy wideband comb. `spread_spectrum_embed()` honours it — pass
`alpha=None` and you get the band's own value. But `watermark_embed()`, the
only function `mark_audio_file()` ever calls, declared:

```python
def watermark_embed(pcm, alpha: float = 0.08, ...)
```

and `mark_audio_file()` calls it without an alpha (`watermark.py:2622`). 0.08 is
the *legacy* band's strength. When #260 moved the comb into the speech band it
retuned the band and left the caller's default behind, so every file CrispTTS
has marked since ran 1.6x hotter than the band was designed for.

Measured over five 20 s segments of real speech, three sample rates:

| | alpha 0.05 (designed) | alpha 0.08 (in use) |
|---|---|---|
| SNR, mean | 19.6–25.1 dB | 15.5–22.1 dB |
| SNR, worst segment | 14.2 dB | 9.6 dB |
| Detection after 64 kbps MP3, worst | 0.750 | 0.844 |

Both clear the 0.65 threshold by a wide margin under every attack tested, so
the 3–4 dB was bought with nothing. Default is now `None`; the
`audioseal_crispasr` branch resolves the band default before crossing into the
C binding, which takes a float.

Two regression tests: one asserts `watermark_embed()` output is identical to
`spread_spectrum_embed(alpha=wm_params(...)[2])`, which keeps the two in step if
the band is retuned again; one asserts the default is quieter than 0.08. Both
confirmed to fail against the previous code.

Files marked by earlier versions stay valid — louder than intended, not weaker.

### 23.2 The SNR figure came from one lucky segment

`readme.md` claimed "~39.5 dB on speech" (and "~38 dB" in the feature list).
Measured across five segments the embed is **20–25 dB mean, 14–17 dB worst
case** at the designed alpha, and 9.6 dB worst case at the alpha actually in
use. 39.5 dB is reproducible only on segment 0 of `german.wav`.

The gap matters beyond accuracy: "imperceptible" was doing real work in the
Art. 50(2) argument, since inaudibility is what makes an always-on watermark
acceptable. At 10–17 dB on sparse passages that claim does not hold. The README
now says low-level rather than imperceptible and points at WavMark for anyone
who needs true inaudibility.

### 23.3 The comb's band is bin-indexed, so it moves with the sample rate

`wm_params()` computes `lo_bin = n_fft // 16`, `hi_bin = n_fft // 5` — pure bin
indices, with the comments ("~1.5 kHz @ 24 kHz") anchored to CrispASR's rate.
The occupied frequency range therefore scales with the file:

| Sample rate | Comb occupies |
|---|---|
| 16 kHz | ~1.0–3.2 kHz |
| 24 kHz | ~1.5–4.8 kHz (the design target) |
| 44.1 kHz | ~2.8–8.8 kHz |
| 48 kHz | ~3.0–9.6 kHz |

Measured on 44.1 kHz speech, only 31% of the watermark's residual energy lands
in the 1.5–4.8 kHz band the README named; 37% sits in 4.8–8 kHz and 25% above
8 kHz. Several shipped backends output 44.1 or 48 kHz (MeloTTS, VoxCPM2, MOSS,
Dots.TTS), so this is the common case, not an edge one.

**Not changed.** CrispASR addresses the comb by bin index too, so interop is
intact, and detection is unaffected at every rate tested. Pinning the band to
hertz would break bit-compatibility with existing marked files for a
perceptual gain that has not been shown to matter. Recorded and documented
rather than fixed; the README now gives the per-rate mapping instead of a
single wrong number.

### 23.4 Two compliance analyses that were assumed, never written down

Neither is a code change; both were missing from an audit trail that otherwise
records its reasoning.

- **Code of Practice on Transparency of AI-generated Content** (Art. 50(7)).
  Zero mentions across PLAN.md and readme.md before this phase. It is voluntary
  and CrispTTS is not a signatory, but it is the Commission's designated route
  to demonstrate Art. 50(2) compliance predictably. The Code asks for *layered*
  marking on the express ground that no single technique meets all four
  Art. 50(2) criteria — which is the architecture already built. README now
  carries the clause-by-clause mapping and names the two residual gaps, both of
  which are about third parties reading the mark: the untrusted certificate and
  a spread-spectrum watermark nobody else implements.

- **Whether Art. 50 binds this project at all.** Every prior phase assumed
  provider status. Art. 3(10) qualifies "making available" with *in the course
  of a commercial activity*, which a non-commercial FOSS release has a real
  argument for never crossing. The stricter reading stays — it is the safe one
  — but it is now recorded as an assumption with its counter-argument, not as
  settled fact. Also recorded: the Art. 50(2) grace period to 2 Dec 2026 for
  systems on the market before 2 Aug 2026, which covers CrispTTS.

### 23.5 The certificate limit, stated where it is relied on

`c2pa-python` is a core dependency because the watermark is Crisp-readable only
and Art. 50(2) asks for interoperability. The default signer is the bundled dev
certificate, so verifiers parse the manifest and report the signer untrusted:
interoperable in format, not in trust. The code already warned once per run and
distinguished `self-signed` from `ca-issued` in `MarkResult`, but the README
listed C2PA under "what CrispTTS does for you" with no caveat at the point of
reliance. Now cross-referenced, with the `--c2pa-cert` / `--c2pa-key` route
spelled out.

Not closable in code. It needs a certificate issued to a real identity.

### Deliberately not changed

The fail-open residuals from Phases 19–22 are design decisions, re-confirmed
rather than revisited: `speaker_identity: unknown` warns without forcing a
disclosure; `--stream --play-direct` plays audio that never passes the marking
gate, under a warning; and the `--allow-unmarked` / `--no-watermark` /
`--accept-marking-responsibility` / `--no-spoken-disclaimer` hatches each hand
the duty back to an operator who has said so explicitly.

### Status: COMPLETE — 441 tests pass (2 new), ruff clean

## Phase 24: the recommended extra could not be installed (v0.9.4)

Audited 2026-08-02, immediately after Phase 23. Phase 23 verified every number
in the spread-spectrum table by re-measuring it. The one figure left unverified
was WavMark's ">38 dB SNR", quoted from upstream. Verifying it required
installing the extra, which is how the first finding surfaced.

### 24.1 `wavmark>=0.3.0` matches no release that has ever existed (the defect)

Both the `robust` and `watermark-mit` extras pinned `wavmark>=0.3.0`. PyPI has
published exactly three versions: 0.0.1, 0.0.2, 0.0.3. The floor is a
transposition of 0.0.3, and nothing satisfies it:

```
$ pip install 'crisptts[robust]'
ERROR: Could not find a version that satisfies the requirement wavmark>=0.3.0
       (from versions: 0.0.1, 0.0.2, 0.0.3)
```

That command appears in **seven** places in the README, including the two that
matter most: the Opus/OGG remediation instructions. Opus and OGG carry no C2PA
manifest, so Phase 21 made a neural watermark *required* rather than optional
for those containers — and the documented way to obtain one did not resolve.
The failure was loud and at install time, and the marking gate still fails
closed, so no unmarked audio was produced. The effect was narrower and more
annoying: legal Opus output was unreachable by the documented route.

Fixed to `>=0.0.3` in both extras. A regression test parses `pyproject.toml`
and asserts the declared floor is satisfied by the installed distribution,
skipping where wavmark is absent (the default install). Verified against the
old pin: `(0,3,0) <= (0,0,3)` is false, so it fails.

wavmark 0.0.3 works correctly with the integration — `load_wavmark()` returns
True and the backend activates.

### 24.2 ">38 dB" reads upstream's number backwards

`wavmark.encode_watermark` is declared:

```python
def encode_watermark(model, signal, payload, pattern_bit_length=16,
                     min_snr=20, max_snr=38, show_progress=False)
```

38 dB is the **ceiling** of an iterative per-chunk SNR search, not a floor it
clears. Measured on 3 s of speech at 16 kHz, WavMark's native rate:

| | |
|---|---|
| SNR reported by upstream's own `info["snr"]` | 36.31 dB |
| SNR computed independently from the delta | 36.3 dB |

Both agree, and 36.3 dB is a real ~15 dB improvement on the built-in layer's
20–25 dB — the recommendation to install it for imperceptibility stands, with
the correct number. `_embed_wavmark()` discards the `info` dict that carries
the achieved SNR; logging it would make this measurable at runtime for free.

### 24.3 The cost that made the measurement hard to take

The reason 24.2 took three attempts to measure is a finding in itself. On CPU:

| Operation | Cost |
|---|---|
| One-time model load | ~21 s |
| Embed, 3 s of audio | 54 s (~18x realtime) |
| Embed, 2 s of audio | 99 s (under CPU contention) |
| **Detect, 3 s of audio** | **did not return within 10 minutes** |

Detection is the one that matters: `mark_audio_file()` runs it after every
embed as the verification gate, so every marked file pays it. With wavmark
installed the test suite went from 2.5 minutes to not completing 2% in 20
minutes.

Contributing factors, all recorded rather than fixed:

- `load_wavmark()` selects `cuda:0` or `cpu` and never checks
  `torch.backends.mps.is_available()`, so Apple Silicon always takes the CPU
  path.
- `wavmark.decode_watermark` sliding-window searches for the start-bit pattern,
  which is why detect is far worse than embed.
- Audio shorter than one 16 kHz chunk raises `AssertionError` upstream;
  `watermark_embed()` catches it and falls back to spread-spectrum, which is
  the correct behaviour and needs no change.

Not fixed because none of it can be fixed here without an upstream change or a
device-selection change that cannot be verified at a 10-minute-per-detect
iteration cost. The README now carries a measured cost warning telling readers
to benchmark on their own hardware, which is the honest form of a
recommendation this expensive.

### 24.4 What this says about the audit method

Phase 23's finding and this one have the same shape: a number in the docs that
nobody had reproduced. Phase 23 caught it by re-measuring the built-in layer;
this phase caught it by trying to install the thing being recommended. Neither
was visible to 441 passing tests, because tests assert behaviour and these were
claims *about* behaviour, made in prose.

### Status: COMPLETE — 441 pass, 8 skipped, ruff clean

The new pin guard is one of the 8 skips in a default install: it compares the
declared floor against the *installed* wavmark and has nothing to compare
against when the extra is absent. It runs, and fails against the old pin, on
any machine that has the extra — which is exactly where a broken floor matters.
