# PLAN: what is open

Forward-looking only. Completed work is in `HISTORY.md` — read the phases named
below before touching their area, because most of them exist to record why an
obvious-looking change was wrong.

**State as of 2026-08-03, v0.9.14:** 472 tests pass, 7 skipped, ruff clean, CI
green on Python 3.10/3.11/3.12. Nothing is known-broken. Everything below is
improvement, investigation or a decision someone has to make.

---

## How to work on this repository

Four rules, each of which cost a phase in `HISTORY.md` to learn.

**1. Measure before you believe a claim, including one in these files.**
The largest defects found so far were all documented facts that nobody had
reproduced: an SNR figure taken from one lucky audio segment (23.2), a
dependency floor no release satisfied (24.1), a `--help` that imported torch
(27.1), a detector that was a coin flip (28.1), and a certificate limitation
that turned out not to be a limitation (31.2). If a number appears in prose
and not in a test, treat it as a rumour.

**2. Check what is installed before you benchmark.**
Phase 33.2 nearly concluded a bug was imaginary because AudioSeal was still
installed from an earlier phase's benchmarking and was silently handling the
case under test. A passing measurement taken in the wrong configuration is not
evidence. The configurations that matter are: default install (no neural
backend), `[robust]` (AudioSeal), and `[watermark-mit]` (WavMark).

**3. Watch CI, not just the local suite.**
CI runs 3.10–3.12; this machine runs 3.11. Three commits went out on a red
build because `tomllib` is 3.11+ and the local suite could not see it. And a
*green* 32-minute run turned out to be hiding an O(n²) regression (34.1) —
duration is a signal, not noise.

**4. In `watermark.py`, go slowly.**
Phases 30, 33 and 34 are three consecutive self-inflicted defects in the
consent-log code: added the chain, broke concurrency, broke performance. Each
was introduced by the previous fix. That file rewards paranoia.

---

## 1. Detector residuals — needs a wider corpus, not more tuning

**Where:** `watermark.py`, `_spread_spectrum_detect_band()`. Background in
HISTORY 28 and 32.3.

The detector is a per-frame t-statistic standardised against 15 decoy patterns,
with three conditions that must all hold (`t >= 3.0`, `z >= 1.0`,
`t_true > 0.70 * max(|t_decoy|)`). Measured on 79 unmarked / 237 marked clips:
**FP 1.3%, TP 98.3%**, against the old sign test's 8.6% / 97.0%.

Two residuals remain:

- One false positive: `crisptts_assets/disclosure_mt.flac` reads 0.766. Real
  TTS speech, not a pathological signal.
- Three false negatives, all `NEAR-SILENCE` synthetic clips (0.49–0.51).

**Do not tune the thresholds against the current corpus.** It is 79 unmarked
clips drawn from one recording plus the 27 bundled disclosure assets, and the
constants are already fitted to it — moving them further is curve-fitting to a
sample, not an improvement. The useful work is a bigger, more diverse null set
first: more speakers, more languages, music, and audio that has been through
real-world processing. Only then revisit the constants.

**Acceptance:** a null corpus of ≥500 clips from ≥5 distinct sources, with the
FP/TP table regenerated. Changing a constant without that is out of scope.

---

## 2. Seven models whose speaker identity is still unknown

**Where:** `config.py`, `speaker_identity` keys. Background in HISTORY 22.4 and
32.2.

`unknown` warns once per model and does **not** force a disclosure. That is
deliberate: `synthetic` is the value that silently removes an Art. 50(4)
disclosure, so claiming it without provider evidence is the costly error — see
32.2, where `mlx_audio_bark_de` had to be downgraded for exactly that reason.

Still open, each with the check already recorded in `config.py` so it is not
re-litigated:

| Model(s) | What was checked |
|---|---|
| `edge` | Microsoft's transparency note defines "voice talent" only under *custom* neural voice; the prebuilt section never says whose voice. Re-verified 2026-08-03. |
| `crispasr_melotts` | HF card, GitHub README, `docs/training.md` — the training guide explains how to train your own and discloses nothing about the shipped speakers. |
| `crispasr_orpheus`, `orpheus_lex_au`, `orpheus_ollama`, `mlx_audio_orpheus_llama` | Canopy Labs disclose 100k+ h of "permissive" audio and nothing about `tara`/`leah`/`jess`/`leo`/`dan`/`mia`/`zac`/`zoe`. |
| `mlx_audio_bark_de` | Was `synthetic` until 2026-08-03 on the strength of third-party write-ups. Neither the suno-ai/bark README nor the suno/bark model card says where the `v2/*_speaker_*` presets came from. Downgraded — see 32.2. |

**This is not a research task any more — the public documentation has been
read.** Closing these means asking the providers directly. If you do, record
the answer *and its source* in `config.py` next to the value.

Current distribution across the 27 non-cloning models: 14 `real_person`,
7 `unknown`, 6 `synthetic`. A test asserts every `voice_cloning: false` model
declares a valid value, so a new backend cannot skip the question.

**Do not** resolve them by inference from a third-party write-up. That is
precisely what produced the bark error.

---

## 3. Should the delivery gate depend on a statistical detector at all?

**Where:** `watermark.py`, `mark_audio_file()`. Background in HISTORY 29.2.

CrispASR's `docs/eu-ai-act.md` §6.7 states: *"embedding is unconditional and
the watertight floor does not consult the detector."* CrispTTS is different —
it verifies after embedding and **deletes the output** if verification fails.

So a detector error here is not a diagnostic inconvenience the way it is in
CrispASR: a false negative destroys a user's file. That asymmetry is why the
Phase 28 detector work mattered, and it is worth asking whether the design is
right at all.

Arguments both ways, unresolved:

- *For the gate:* it is the only thing that catches an embed that silently did
  nothing, which is a failure mode that has actually occurred here (HISTORY
  20.1, 22.1).
- *Against:* it makes output delivery contingent on a statistical test that can
  be wrong in both directions, and the cost of being wrong is deletion.

A middle option nobody has costed: keep the gate but make failure loud and
non-destructive — refuse to *report* the file as marked, warn hard, and leave
it on disk. That changes what `--allow-unmarked` means, so it needs thinking
through rather than implementing directly.

**Acceptance:** a decision recorded in HISTORY with its reasoning, whichever
way it goes. Not necessarily a code change.

---

## 4. Sibling projects

### 4a. Port the improved detector to CrispASR (was blocked; now clear)

CrispASR's built-in detector is the sign-agreement test CrispTTS replaced in
Phase 28. Their own `crispasr_watermark_stats.h` measures it at **4.8% false
positives** on 55 clips and concludes *"raising the bar cannot make this
instrument strong"* — which is true of the sign test and not true of the
statistic that replaced it. Their answer was to report an exact binomial
p-value with a three-way verdict; theirs is honest, ours is more sensitive.

Their `--detect-watermark` is a *diagnostic* and their marking floor does not
consult it, so the impact is an overclaim in a CLI message rather than lost
audio. Lower stakes than it was here.

**Was deferred for a good reason, which may recur.** Earlier on 2026-08-03 that
tree had uncommitted changes across four files and a dozen live worktrees;
editing someone else's mid-refactor tree risks clobbering work that is not
visible in git. It was clean when this plan was written, so the work is
available — but **re-check `git status` there before starting**, because that
repository is actively developed and the answer changes within a day.

Scope if unblocked: port per-frame t + decoy calibration into
`src/core/crispasr_watermark.h`, and retire or re-derive
`crispasr_wm_stats::p_value()`, whose binomial null is a property of the sign
test and does not survive the change. Their header already warns about exactly
this for AudioSeal scores.

### 4b. Susurrus disclosure locales

Susurrus ships 2 locales (`utils/translations/{en,de}.py`) against CrispTTS's
27 pre-rendered disclosure clips in `crisptts_assets/`. Art. 50(4) disclosure
in a language the audience does not speak does not discharge the duty.

CrispTTS's assets are rendered by `scripts/make_disclosure_assets.py` straight
from edge_tts and are not watermarked, so they are reusable. Susurrus's
enforcement is otherwise strong — it fails closed, marks mp3/flac/m4a/opus, and
has an attestation gate — so this is the one place it is clearly behind.

### 4c. CrisperWeaver short-audio no-op (unverified)

CrispTTS's `readme.md` comparison table records CrisperWeaver as silently
skipping the watermark below 4608 samples. **That table was stale about
Susurrus on four rows before it was re-checked on 2026-08-03, and
CrisperWeaver's rows have not been re-checked at all.** Verify against the
current code before treating it as a defect.

---

## 5. Decisions that are made — do not silently reopen

| Decision | Why | Revisit when |
|---|---|---|
| **Not signing the Art. 50(7) Code of Practice** | Adherence commits the project to a fixed public description of how it marks content, and the marking is still moving. A public claim that goes stale is the failure mode this repo keeps correcting. | The marking layers stop changing. |
| **The bundled C2PA certificate is fine for Art. 50(2)** | Measured: a default-signed file reads `validation_state: Valid` with `trainedAlgorithmicMedia` intact; the only failure is `signingCredential.untrusted`. Art. 50(2) asks for machine-readable marking, not proof of authorship. See HISTORY 31.2 — this was wrongly called the largest remaining gap for five releases. | Never, for compliance. Supply `--c2pa-cert` when you want **attribution**. |
| **AudioSeal over WavMark as the preferred backend** | Several times faster, and the only backend measured to survive an Opus round-trip. WavMark is ~7 dB quieter and stays available as `[watermark-mit]`. | New measurements, not new opinions. |
| **The comb's band is bin-indexed, so it moves with sample rate** | Pinning it to hertz would break bit-compatibility with existing marked files and with CrispASR, for an unproven perceptual gain. Documented per-rate in the README instead. | Someone demonstrates an audible cost. |
| **`speaker_identity` divergence with CrispASR on speecht5** | CrispTTS ships a default CMU ARCTIC x-vector set — seven identifiable people — so `real_person` is right *here*. CrispASR takes an arbitrary operator-supplied vector and cannot know. | Never; both are correct for their own product. |
| **`--stream --play-direct` plays audio the gate never saw** | Incremental playback cannot wait for a completed file. Warned loudly; any `--output-file` is still gated. | Someone wants gated streaming badly enough to buffer. |

---

## 6. Known limitations that are recorded, not bugs

- **Watermark detection needs Crisp tooling.** The spread-spectrum comb is not
  a standard anyone else implements. The C2PA manifest is the interoperable
  layer and is why `c2pa-python` is a core dependency.
- **Opus/OGG cannot carry a C2PA manifest.** c2pa-rs does not support the
  container, and the detached-sidecar route *reports success while writing a
  byte-identical copy of the input* (HISTORY 25.1 — verified, do not re-test
  hopefully). Those containers therefore require a neural watermark.
- **Sub-0.5 s clips cannot be watermark-verified.** Below `_DETECT_MIN_FRAMES`
  a real mark is indistinguishable from none. In the shipping configuration the
  C2PA manifest carries sufficiency, so short outputs still ship — see the test
  `test_short_output_still_ships_in_the_default_configuration`, which pins the
  *combination* because the combination is what users get.
- **WavMark is CPU-slow without an accelerator.** ~50× realtime on CPU at
  torch's default thread count. `load_wavmark()` now prefers CUDA → MPS → CPU.

---

## 7. Housekeeping

- `readme.md`'s cross-project comparison tables go stale silently and unfairly.
  They are dated and marked re-checked as of 2026-08-03. Re-read the sibling
  code before trusting or citing them.
- `crisptts_assets/*.flac` are edge_tts renderings and deliberately unmarked;
  they double as a null corpus for detector work (§1).
- The scratch benchmarks used across Phases 23–34 were not committed. If you
  redo detector work you will be rebuilding the harness; consider committing it
  under `scripts/` this time.
