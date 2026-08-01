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

### Known limitation, deliberately not closed

The built-in spread-spectrum watermark still does not survive resampling
(0.63 against a 0.65 threshold). Closing that properly means either a hard
`torch` dependency or original signal-processing work on rate-invariant
embedding. The chosen mitigation is the `robust` extra plus a loud warning
and honest documentation — see 16.8. Revisit if CrispTTS is ever distributed
in a context where the maintainer is clearly a "provider placing on the
market" under Art. 3(9).
