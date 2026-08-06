# Changelog

All notable changes to MMH3Tools are documented here.
This project follows [Semantic Versioning](https://semver.org/).

**Input ordering is append-only.** ComfyUI serialises widget values positionally,
so new inputs must be added at the END of a node's input list. Never insert or
reorder existing inputs, or saved workflows silently rebind to the wrong widgets.

## [0.7.0] - 2026-08-05

### Added
- `MMH3ImageKeyframe` - inject a still image as a keyframe anchor. Takes
  `image` + `vae` and does the resize/encode internally, because keyframe rows
  share the TARGET spatial grid and cannot be downscaled: a still encoded at any
  other resolution fails deep in the model with a broadcast error. That is the
  likeliest cause of "I VAE-encoded a still and it didn't work".

  It appends to `minimax_keyframes` rather than building conditioning, so it
  composes with a ref2va build - `MiniMaxH3ReferenceToVideo` has no keyframe
  inputs at all, so this is the only way to give the reference checkpoint a
  frame anchor.

  `frame_index` accepts 0, -1, or any interior index. Interior anchors are
  documented as valid by MiniMax but stock `PackedLayout` raises
  `only first/last keyframe anchors are supported`; the node logs a warning
  rather than refusing, so it works once that check is patched.

  `resize=auto` copies the stock node's per-position behaviour: stretch for a
  first-frame anchor (geometry anchor), aspect-preserving centre crop otherwise
  (follower).

  It does NOT register the image with the tokenizer, so `<Picture N>` will not
  resolve in prompt text - same limitation as the rest of the latent-domain
  nodes, and for the same reason.

- `MMH3AssetPlan` and `MMH3TaskSystemPrompt` (`nodes_prompt.py`) - build a
  Context-IR system prompt for your own LLM node from a task type (or
  combination) plus a plan of the assets in play. Emits only the rule blocks
  relevant to the selected tasks instead of the whole spec.

- `MMH3SeedOverlap` restored. It **prepends** the overlap (a multiple of 5
  latents, 17 frames each) so the target keeps its full requested duration and
  the overlap is cleanly cut off afterwards.

- `docs/core-patches.md` + `docs/core-patches.diff` - the three ComfyUI core
  files that must be patched for a keyframe and a reference to coexist, and for
  overlap strength to be continuous rather than boolean. Taken against
  `v0.30.0-1-g14b05228`. These are lost on every `git pull` in ComfyUI, so they
  are now checked in and reappliable with `git apply`.

### Changed
- README: the "noise_mask does not work on H3" section is **removed**. The claim
  was false - `samplers.py` packs latents before sampling and explicitly handles
  `denoise_mask.is_nested`. It is replaced by a correction note and a section on
  why joins happen in pixel space (which is true, and was the real reason).
- README: "whether ref2va responds to keyframe rows is unverified" is resolved -
  it does, once core patches 1 and 2 are applied.
- Per-row masking merged from drozbay's `minimax-h3-per-row-masking` branch, so
  overlap strength is continuous rather than on/off.
- `MMH3PackAV` now carries an input `noise_mask` through instead of dropping it.

### Fixed
- `MMH3DimensionCalculator` failing with a bare "Invalid input" and nothing in
  the console. The JS swaps the resolution list per ratio/orientation, but the
  server validates combo values against the options declared in Python, so
  anything outside the default list was rejected. Declared options are now the
  full union (9 ratios, 103 resolutions) with `validate_inputs` returning True.
- JS/Python rounding mismatch in the same node: Python's `round()` is banker's,
  `Math.round()` is half-up, so 3:2 diverged at exactly 16.5. The JS now uses a
  `roundHalfEven` helper; all 114 generated options match Python.

## [0.6.0] - 2026-08-03

### Added
- `MMH3FindDivergence` gains a `compare` input (`structure` / `raw`, default
  `structure`), appended last so saved nodes pick up the default without rewiring.

  `structure` zero-means and unit-contrasts each frame before comparison. Plain MAE
  cannot distinguish "different content" from "same content, half a stop brighter",
  so an exposure or colour shift between the source and the generated chunk puts a
  floor under every comparison and flattens the curve — which reads exactly like
  "no reproduction found". Measured on a genuine 30-frame reproduction rendered 12%
  brighter with a lifted black level: raw MAE reports error 0.110 at 2.5x separation
  (rejected at the default 0.05 threshold), structure reports 0.0100 at 79x.

  This matters here because the source has been through a VAE round-trip the
  generated chunk has not, so level drift between them is expected.

  Threshold stays around 0.05: a good structure-mode match is ~0.01, mismatches ~0.8.

## [0.5.0] - 2026-08-03

### Added
- `docs/context-ir-system-prompt.md` - a system prompt that replaces MiniMax's
  hosted `H3-Context-IR` (`/v2/h3_context_ir`), which is not open-sourced. H3
  expects a STRUCTURED prompt, not prose; Context-IR is what produces it, so
  running locally you must author that structure yourself. Covers both formats
  (three-field base modes and six-section Ref2VA), mode selection, label rules,
  task types, retention markers, camera/speaker/dialogue syntax, the discrete
  achievable durations, and the chained-work defaults. Portable — usable with a
  local LLM, an enhancer node, or anything else. Use a VISION model when
  references are involved, since `subject_definitions` describes the assets.

### Removed
- **`MMH3SeedOverlap` — removed.** ComfyUI cannot apply a denoise mask to H3's
  NestedTensor AV latents, so the node could never have worked:
  - `KSamplerX0Inpaint.__call__` computes `1. - denoise_mask`, and `NestedTensor`
    defines no `__rsub__` (nor `__torch_function__`), so a nested mask raises
    TypeError.
  - A plain-tensor mask fails differently: `apply_operation` applies it to BOTH
    sub-tensors, so one mask would have to broadcast against video
    `[B,24,T,h,w]` and audio `[B,32,2,T40]` at once.
  - `torch.count_nonzero(latent_image)` in `inner_sample` is not nested-safe.

  Masked / inpaint-style workflows are therefore unavailable for H3 in ComfyUI
  v0.30.0. Continuity comes from the REFERENCE path (`update=False`, the trained
  mechanism), and joins are trimmed after decode.

### Added
- `MMH3FindDivergence` - measures how many frames a continuation reproduces from
  its source, so the join can be trimmed at FRAME granularity. Latent trims are
  restricted to the 5j+2 grid, i.e. 17-frame steps, which is far too coarse for a
  boundary the model does not place on a grid.

  Scores each candidate run length K by the contiguous alignment
  `continuation[i] ~ source[-K+i]`, anchored at the source's last frame. Per-frame
  nearest-match was tried first and does NOT work: in visually repetitive footage
  every new frame also matches something, so divergence is never detected. The
  contiguous form gives ~10x error separation at the true K, and reports a
  best/median separation ratio so a flat (untrustworthy) curve is visible.

## [0.4.0] - 2026-08-03

### Added
- `MMH3PackAV` - zips a video latent and an audio latent into one H3 AV latent.
  Encoding real footage produces two SEPARATE plain latents (`VAEEncode` with the
  video VAE, `VAEEncodeAudio` with the audio VAE) and nothing paired them. This is
  a MODALITY join; `MMH3ConcatAV` is a TIME join. Audio length is reconciled to
  `round(frames / 24 * 40)` by padding with silence or trimming, since the two
  streams run on independent clocks and encoders will not agree exactly. Audio is
  optional — omit it to pair with silence.
- `MMH3SeedOverlap` now also outputs **`overlap_latents`**, wiring straight into
  `MMH3ConcatAV.trim_b_latents` so the overlap is not duplicated at the join.
  The grid arithmetic is closed under this: `(5j+2)+(5k+2)-(5m+2) = 5(j+k-m)+2`,
  so chains stay on-grid indefinitely.
- `MMH3LatentToRef` and `MMH3ReferenceFromLatent` now also output
  **`carried_latents`** — the actual count after snapping and clamping, which is
  what grid math and trims need. `carried_frames` was not enough.

### Changed
- `unpack_av()` takes `name` and `allow_video_only`. Errors now name the failing
  input rather than saying "a latent" was wrong, and a plain 5D video latent is
  accepted where audio is genuinely optional: `MMH3SeedOverlap.source` (seeds video,
  logs that it skipped audio) and `MMH3LatentKeyframe` (only ever reads one frame).
  This is the `VAEEncode`-real-footage path — the video VAE knows nothing about
  audio, so it returns a plain, audio-less latent.
- All new outputs are appended, so existing links do not shift.

## [0.3.0] - 2026-08-03

Calculators now follow the LTXAVTools convention: concise typed outputs plus a
short `label` string instead of a verbose info block, and a flat `MMH3Tools`
category on every node.

### Added
- `MMH3FrameCalculator` - **seconds in**. Outputs `frame_count`,
  `latent_frames`, `audio_latent_frames`, `actual_seconds`, mirroring
  `LTXFrameCalculator` plus the audio count H3 needs. `rounding` is
  nearest / up / down.

  Because frames must be 17j+5 at 24fps, achievable durations are discrete.
  Solving `24s = 5 (mod 17)` gives `s = 8 (mod 17)`, so **8.000s (192 frames) is
  the only whole-second duration in the entire 4-15s supported range**. 5s really
  means 5.167s, 12s means 12.250s.
- `MMH3DimensionCalculator` - outputs `width`, `height`, `width_ref`,
  `height_ref`, `label`, mirroring `LTXDimensionCalculator`. Where LTX emitted a
  fixed `width_half`/`height_half` pair for its two-stage pipeline, H3 has no
  second stage, so the secondary pair is the REFERENCE size driven by
  `downscale_factor` and snapped to what the patch grid supports.
- (superseded) `MMH3DimCalc` - snaps width/height to the 32px grid, reports latent dims and
  tokens per latent frame, and snaps a requested reference downscale factor to
  the nearest value the patch grid actually supports. Outputs both generation and
  reference geometry plus the full list of valid factors.
- `common.supported_downscale_factors()` / `common.snap_downscale()`.

### Fixed
- **`ref_downscale` could distort reference aspect ratio.** Latent dims must stay
  EVEN for the 2x2 patch, so a factor is only valid when latent/f is an even
  integer on both axes — the divisors of `gcd(latent_h//2, latent_w//2)`. For the
  1344x768 canvas that is `[1, 2, 3, 6]`; **4 is not valid** (84/4 = 21, odd).
  The old code forced evenness by subtracting 1 from the odd axis, silently
  changing the reference's aspect ratio. `downscale_video_latent()` now snaps the
  factor instead and returns the factor actually used; `MMH3LatentToRef` logs when
  the request was adjusted.

### Changed
- All node categories flattened from `MMH3Tools/{conditioning,latent,util}` to a
  flat `MMH3Tools`, matching the LTXAVTools convention.
- `downscale_video_latent()` now returns `(tensor, latent_h, latent_w, factor_used)`
  — a 4-tuple, was 3. Internal helper; no node inputs or outputs changed.

### Removed
- `MMH3GridSnap` - superseded by `MMH3FrameCalculator`, which takes seconds rather
  than frames. Nothing had been built on it.

## [0.2.0] - 2026-08-03

### Added
- `MMH3ReferenceFromLatent` - full ref2va conditioning builder fed by a latent
  instead of pixels. Unlike `MMH3LatentToRef` it owns the tokenizer call, so the
  carried chunk registers as a real `<Video 1>` and prompts can use the
  `[video continuation]` task type and reference it by label.

  The DiT still receives pristine latents; the only decode is a 2fps subsample
  handed to Qwen3-VL (3-4 frames for a ~1.6s carry), so no generation loss enters
  the sampling path. `register_with_tokenizer` can disable it entirely.
- `common.empty_av_latent()` and `common.frames_to_qwen_items()`.

### Why
H3 expects a structured six-section prompt whose `subject_definitions`,
`summary` and `retention_analysis` sections refer to assets by `<Video N>` /
`<Audio N>` label. Without tokenizer registration those labels dangle, which
silently degrades output. See `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` in the
MiniMaxAI/MiniMax-H3 repo.

Note: `frames_to_qwen_items()` derives timestamps from real frame positions
rather than from the sample index as the stock node does. Behaviour is identical
at 2fps; it just stays correct if the step ever changes.

## [0.1.0] - 2026-08-03

Initial release. Targets MiniMax H3 as shipped in ComfyUI v0.30.0.

Supersedes the throwaway `ComfyUI-MiniMaxH3Loop` prototype, which was removed.
Node IDs changed from `MiniMaxH3*` to `MMH3*`; no released workflows referenced
the old IDs.

### Added

Conditioning
- `MMH3LatentToRef` - builds a `minimax_refs` block directly from an H3 AV latent,
  skipping the pixel/VAE roundtrip the stock reference node performs. Snaps the
  carry to the 5j+2 grid, optionally carries the matching audio tail as
  `kind="video_audio"`, and supports 2x/4x spatial downscaling to cut per-step
  reference token cost.
- `MMH3LatentKeyframe` - injects a `minimax_keyframes` anchor from a single latent
  frame. `PackedLayout` accepts keyframes and refs together, so this stacks with
  `MMH3LatentToRef`.

Latent
- `MMH3SeedOverlap` - seeds the head of a target latent with a previous chunk's
  tail and emits a matching nested `noise_mask`, giving LTXAV-style overlap
  strength control on a model whose reference path is never denoised. Video and
  audio are masked independently on their own temporal axes, with an optional
  linear feather back to full denoise.
- `MMH3ConcatAV` - joins two AV latents using the correct per-sub-tensor temporal
  axes (video dim 2, audio dim 3), with optional head trim on the second latent
  to drop a seeded overlap region.

Util
- `MMH3LatentInfo` - shapes, implied frame count, audio-length mismatch check,
  grid alignment, noise_mask presence.
- `MMH3GridSnap` - snap frames to the 17j+5 grid and derive latent counts, with a
  warning outside the 124-362 trained range.

### Notes
- Carried references are not registered with the tokenizer, so Qwen3-VL does not
  see them and `<Video k>` prompt tags must not be used for a carried chunk. The
  DiT still receives the latents; only the semantic path is skipped.
- `MMH3LatentKeyframe` requires the source latent to match the target generation's
  spatial dimensions, since keyframe rows share the target grid.
- Whether the `ref2va` checkpoint responds to `cond` (keyframe) rows is unverified.
- Latent-space downscaling in `MMH3LatentToRef` uses bilinear interpolation and is
  approximate.
