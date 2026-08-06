# MMH3Tools

MiniMax H3 latent tooling for ComfyUI — latent-domain conditioning and correct AV
splicing for **chained long-form generation**.

Requires ComfyUI **v0.30.0+** (native H3 support).

## Why this exists

Three facts about H3 shape everything here:

1. **References are latents that are never denoised.** `PackedLayout` packs them
   with `update=False`, so they are re-injected at every sampling step as pure
   context. There is no shared region between chunks to blend.
2. **The stock reference node takes pixels** and calls `vae.encode()`. In a chain
   the previous chunk is already latent, so that roundtrip is generation loss
   compounding once per hop.
3. **Video and audio latents have different temporal axes.**

   | tensor | shape | temporal dim |
   |---|---|---|
   | video | `[B, 24, T, h, w]` | **2** |
   | audio | `[B, 32, 2, T40]` | **3** (dim 2 is stereo) |

   Generic nested-tensor helpers that assume one shared temporal dim will stack
   audio on its stereo axis — producing 4 channels at unchanged duration instead
   of a longer clip. It fails silently.

## Nodes

### Conditioning
- **MiniMax H3 Latent to Reference** — carry a chunk's tail forward as a
  `minimax_refs` block, no VAE roundtrip. `ref_downscale` is the cost lever:
  reference tokens are attended at *every* step, so 2× cuts their cost ~4×.
- **MiniMax H3 Latent Keyframe** — first/last frame anchor from a latent frame.
  Shares the *target* spatial grid, so the source must match generation
  dimensions exactly.
- **MiniMax H3 Image Keyframe** — the same anchor from a **still image**.
  Resizes and encodes internally, precisely because keyframe rows cannot be
  downscaled; a still encoded at the wrong size fails deep in the model with an
  unhelpful broadcast error. Both keyframe nodes *append*, so they compose with
  `MiniMaxH3ReferenceToVideo`, which has no keyframe inputs of its own — this is
  the only way to give the ref2va checkpoint a frame anchor.

  `frame_index` accepts `0`, `-1`, or an interior index. MiniMax's guide lists
  interior anchors as valid, but stock `PackedLayout` raises *"only first/last
  keyframe anchors are supported"*; the node warns rather than refusing, so it
  works as soon as that check is patched.

### Sequences
- **MiniMax H3 Reference (Multi-Prompt)** + **MMH3 Cond Select** — the stock
  reference node with N prompts. For a text-driven sequence with locked identity,
  every chunk shares one reference set and differs only in its prompt.

  The win is the **model swap**, not the encode. Qwen3-VL-32B and a 33B DiT can't
  be resident together in 32GB, and ComfyUI resolves outputs depth-first, so N
  chunks in a naive graph run `load TE → cond → evict → load DiT → sample → evict
  → …` N times. One node execution collapses that to a single swap for the whole
  sequence, and the references are resized and encoded once instead of N times.

  Per-prompt memoization means editing one prompt re-encodes only that prompt.
  Swapping a reference invalidates all of them.

  Note this design needs **no core patches** — those exist only to make keyframes
  coexist with references, and there are no keyframes here.

### Prompting
- **MMH3 Asset Plan** / **MMH3 Task System Prompt** — build a Context-IR system
  prompt for your own LLM node from the task type (or combination) and the
  assets in play, emitting only the relevant rule blocks. See
  `docs/context-ir-system-prompt.md` for the full spec these are derived from.

### Latent
- **MiniMax H3 Seed Overlap** — **prepends** overlap latents (multiples of 5, so
  17 frames each) to the target and masks them, giving frame-level seam
  continuity. Prepending rather than overwriting means the chunk keeps its full
  requested duration and the overlap is cut cleanly off afterwards.
- **MiniMax H3 Pack AV** — pair a video latent with an audio latent. Encoding real
  footage gives two *separate* plain latents (`VAEEncode` + `VAEEncodeAudio`) and
  nothing joins them. Audio is reconciled to `round(frames / 24 * 40)`. This is a
  **modality** join; Concat AV is a **time** join.
- **MiniMax H3 Find Divergence** — measures how many frames a continuation
  reproduces from its source, so the join can be trimmed at frame granularity.
- **MiniMax H3 Concat AV** — join two AV latents on the correct axes (video dim 2,
  audio dim 3), with optional `trim_b_latents` and `carry_masks`.

  `trim_b_latents` is honoured as given, because **no single snap is correct**.
  With `A = 5a+2` and `B = 5b+2`:

  | trim | effect |
  |---|---|
  | `5m` | removes a Seed Overlap **exactly**; the total is `5(a+b)+4−k`, **off grid** |
  | `5m+2` | total lands **on grid**; ~7 frames of overlap stay duplicated |

  `k` cannot be `0` and `2 (mod 5)` at once. If you need both, that is what
  **Join AV** is for — it cuts per frame in pixel space. The node logs which
  property the value you gave it actually gets.

### Latent joins happen in pixel space

Latent concatenation is unsound here. Two on-grid chunks sum to `5(j+k)+4`
latents, never back on the `5j+2` grid, so the VAE's 17-frame causal chunking
misaligns from the join onward and the second half pulses. **Join AV** and
**Find Divergence** therefore work on decoded frames, where granularity is one
frame rather than 17, and audio crossfades in the **waveform** domain — the
DAC/BigVGAN latents do not blend.

> **Correction (0.7.0):** earlier versions of this README claimed `noise_mask`
> was structurally impossible on H3 because `1. - denoise_mask` fails on
> `NestedTensor`. That was wrong. `samplers.py` packs latents before sampling and
> explicitly handles `denoise_mask.is_nested`. `MMH3SeedOverlap`, removed in
> 0.5.0 on that false premise, is back in 0.7.0.

For **audio-driven video**, use an audio reference with the `[audio reuse]` task
type and the `fully_copy` marker, not a mask. That is a trained capability.

### Util
- **MiniMax H3 Latent Info** — shapes, frame count, audio-length mismatch, grid
  alignment, mask presence.
Calculators follow the LTXAVTools convention — concise typed outputs plus a short
`label`, flat category.

- **MMH3 Frame Calculator** — seconds in. → `frame_count`, `latent_frames`,
  `audio_latent_frames`, `actual_seconds`. `rounding` is nearest / up / down.
- **MMH3 Dimension Calculator** — → `width`, `height`, `width_ref`, `height_ref`,
  `label`. Where `LTXDimensionCalculator` emitted a fixed `width_half`/`height_half`
  pair for its two-stage pipeline, H3 has no second stage — the secondary pair is
  the **reference** size, set by `downscale_factor` and snapped to a factor the
  patch grid supports.

#### Achievable durations

Frames must be `17j+5` at 24fps, so durations are discrete. Solving
`24s ≡ 5 (mod 17)` gives `s ≡ 8 (mod 17)` — **8.000s is the only whole-second
duration in the 4–15s range**:

| asked | frames | actual | drift |
|---|---|---|---|
| 4s | 90 | 3.750s | −0.250 |
| 5s | 124 | 5.167s | +0.167 |
| 6s | 141 | 5.875s | −0.125 |
| **8s** | **192** | **8.000s** | **0** |
| 10s | 243 | 10.125s | +0.125 |
| 12s | 294 | 12.250s | +0.250 |
| 15s | 362 | 15.083s | +0.083 |

This matters when chaining: per-chunk drift accumulates against wall-clock, so
plan chunk lengths in frames, not seconds — or use 192-frame chunks, which stay
on whole seconds indefinitely.
- **MiniMax H3 Dim Calculator** — the dimension calculator. Snaps width/height to
  the 32px grid, reports latent dims and **tokens per latent frame**, and snaps a
  requested reference downscale to a factor the patch grid supports.

#### Valid reference downscale factors

Latent dims are `px/16` and must stay **even** for the 2×2 patch, so a downscale
factor `f` is valid only when `latent/f` is an even integer on both axes — the
divisors of `gcd(latent_h//2, latent_w//2)`:

| canvas | latent | tokens/frame | valid factors |
|---|---|---|---|
| 1344×768 | 84×48 | 1008 | 1, 2, 3, **6** |
| 1024×1024 | 64×64 | 1024 | 1, 2, 4, 8, 16, 32 |
| 1280×704 | 80×44 | 880 | 1, 2 |
| 1152×640 | 72×40 | 720 | 1, 2, 4 |

Note **4× is invalid on the native 1344×768 canvas** (84/4 = 21, odd) and snaps
to 3×. The factor set depends entirely on the aspect ratio.

## Two conditioning channels

Use the reference block and the overlap together — they carry different things:

| channel | mechanism | carries |
|---|---|---|
| `MMH3LatentToRef` | `minimax_refs`, never denoised | identity, voice, motion style |
| `MMH3SeedOverlap` | target latent + `noise_mask` | frame-level seam continuity |

## Grid reference

| | relation |
|---|---|
| frames | `17j + 5` |
| video latents | `5j + 2` |
| audio latents | `round(frames / 24 * 40)` |
| trained range | 124–362 frames (~5.2–15.1s) |
| node ceiling | 3600 frames (150s) |

Keep **MiniMax H3 Sigma Shift** at video `12.0` / audio `3.0` and constant across
chunks — the DiT derives the audio schedule from the video one, so varying it per
chunk desynchronises them.

## Known limitations

- Carried references are **not** registered with the tokenizer, so Qwen3-VL never
  sees them. Don't use `<Video k>` tags for a carried chunk. The DiT still gets the
  latents, so pixel/motion/identity continuity works; only the semantic path is
  skipped. For continuation that's arguably correct — you rarely want the encoder
  re-describing the previous chunk.
- `ref2va` **does** respond to keyframe (`cond`) rows — but only once two core
  bugs are patched. Stock `model_base.py` overwrites `cond_video_latents` instead
  of accumulating, so refs erase keyframes; and `minimax/model.py` computes the
  keyframe position from `text_len` alone, ignoring the cursor the reference
  blocks already advanced, so a keyframe lands at the wrong row whenever refs are
  present. Both are written up with the full diff in
  [`docs/core-patches.md`](docs/core-patches.md).
- Latent-space downscaling is bilinear and approximate.
- Audio seams: the audio VAE is DAC encoder + BigVGAN decoder. Crossfade in the
  **waveform** domain after decode, never in latent space.
