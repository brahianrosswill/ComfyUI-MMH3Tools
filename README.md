# MMH3Tools

MiniMax H3 latent tooling for ComfyUI — latent-domain conditioning and correct AV
splicing for **chained long-form generation**.

Requires ComfyUI **v0.30.0+** (native H3 support).

## Requirements

Beyond stock ComfyUI, parts of this pack depend on **two upstream PRs that have not
merged yet**. Read this before filing a bug — most "it did nothing" reports are a
missing diff.

| PR | needed by | without it |
|---|---|---|
| **[#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375)** per-row masking | `MMH3SeedOverlap`, latent outpaint, and **`MMH3LoopingSampler` with `carry="mask"`** — the default | `MMH3SeedOverlap` **refuses to run**. The looping sampler does **not** — a noise mask simply has no effect, so the carry preserves nothing and every chunk starts cold. See the warning below. |
| **[#15439](https://github.com/Comfy-Org/ComfyUI/pull/15439)** any-index guides | `MMH3LoopingSampler` with `carry="keyframe"`, and any use of `keyframes` | Both **refuse to run**. Stock raises on any anchor that is not first/last, and a guide carrying a multi-step clip cannot be expressed at all. |
| **[#15316](https://github.com/Comfy-Org/ComfyUI/pull/15316)** VRAM reservation | nothing — optional | The minute-long hang when conditioning carries image references. |

> ⚠️ **The one silent failure.** `carry="mask"` is the looping sampler's default and it
> is *not* gated: without #15375 the mask is accepted and ignored, preserved rows still
> run at the generation timestep, and you get seams with no error anywhere. Everything
> else in this pack refuses rather than pretending. If chunks are not carrying, check
> this first.

#15439 needs one hunk hand-merged onto #15375 — they touch the same region of
`model.py`. Apply order and the exact merge are in
[`docs/core-changes.md`](docs/core-changes.md), along with a script that fetches the
diffs fresh (re-fetch rather than reusing a saved copy; these get rebased).

### One runtime patch, applied automatically

`mmh3tools/patch_guide_origin.py` wraps `PackedLayout` at import — **no core edit, and
it survives `git pull`.** #15439 anchors a guide at `text_len`, but the target does not
begin there: references advance a cursor first, so every guide lands *before* the clip
it is meant to anchor — measured at −1 for one image ref, −320 for an audio ref, −321
for both. Nothing errors; the guide just lands in the reference region, and a carried
tail's audio goes early with it.

No PR carries this fix, which is why it is a wrap rather than a diff. The looping
sampler asks `is_applied()` and **refuses** when a chunk carries both a reference and a
keyframe on an unpatched build, rather than rendering a misplaced anchor.

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

## Example workflow

[`workflows/minimax h3 I2V 2K.json`](workflows/) — three-stage I2V to 2K. Generate
small, then two low-denoise windowed upscale passes.

The audio is decided **in the first stage** and carried forward; the upscale passes
only refine picture. Both upscale samplers run through **MiniMax H3 Context
Windows**, which at 2K is *faster* than not windowing — five windows of 17 latents
do 44% of the attention work of one pass over 57, and attention dominates at that
sequence length.

Also needs [ComfyUI-LlamaOmni](https://github.com/ckinpdx/ComfyUI-LlamaOmni) for the
prompt-writing step (an omni model transcribes the song's lyrics so the character
lip-syncs), KJNodes, RES4LYF, VideoHelperSuite and rgthree. The prompt nodes are
easy to swap for your own — see the Note on the canvas.

## Nodes

In the Add Node menu these are filed under `MMH3Tools/…`, following the same layout
as LTXAVTools: `sampling`, `calculators`, `prompt`, `conditioning`, `reference`,
`latent`, `audio`, `utils`, with the two plain calculators at the root. The headings
below group by what a node is *for*, which is close but not identical — the menu path
for any node is in its tooltip.

### Conditioning
- **MiniMax H3 Latent to Reference** — carry a chunk's tail forward as a
  `minimax_refs` block, no VAE roundtrip. `ref_downscale` is the cost lever:
  reference tokens are attended at *every* step, so 2× cuts their cost ~4×.
- **MMH3 Regenerate-2K Reference** — the second pass of a 768p → 2K run, with the
  reference **sliced per window**. A cond_set is already per chunk and the sampler
  passes `minimax_refs` straight through, so a reference attached to cond *i* reaches
  chunk *i* and nothing else — the slicing is a build-time concern and the sampler
  needs no changes. That matters because reference tokens ride every sampling step:
  handing the whole clip to every chunk multiplies that by the chunk count, and on a
  12-window clip slicing measured ~9.9× less reference attention per chunk.

  Feed it stage 1's own `cond_set` and each window keeps **its own** prompt while
  gaining its own reference; a single `conditioning` replicates one to all of them.
  Latent-only, like Latent to Reference — the reference never reaches the text
  encoder, so nothing is decoded and the CLIP is never touched in the 2K pass.
- **MiniMax H3 Image to Reference** — append a still to `minimax_refs`. Fills the
  last hole in the matrix: latents could become refs or keyframes and images could
  become keyframes, but nothing put an image into refs *by appending*. Stock
  `MiniMaxH3ReferenceToVideo` takes `ref_images` but BUILDS conditioning from
  clip+prompt, so it can't add a still alongside carried latent refs.

  Unlike keyframes, reference blocks carry their own `latent_h`/`latent_w`, so this
  is free to resize. `match` scales to the generation's pixel area; `max` uses a
  2048px short edge for best identity — on a 3000×4000 source that's 5440 tokens per
  step against 999, paid at every step of every window.

- **MiniMax H3 Latent Keyframe** — first/last frame anchor from a latent frame.
  Shares the *target* spatial grid, so the source must match generation
  dimensions exactly.
- **MiniMax H3 Image Keyframe** — the same anchor from a **still image**.
  Resizes and encodes internally, precisely because keyframe rows cannot be
  downscaled; a still encoded at the wrong size fails deep in the model with an
  unhelpful broadcast error. Both keyframe nodes *append*, filling a gap in
  `MiniMaxH3ReferenceToVideo`, which has no keyframe inputs of its own.

  **Not alongside references on stock ComfyUI.** `extra_conds` assigns
  `cond_video_latents` from keyframes and then assigns it *again* from references,
  so the references win and every keyframe is silently dropped. **#15439** fixes
  that by concatenating instead — but see Known limitations for the half it does
  not fix.

  `frame_index` accepts `0` or `-1` only, because stock `PackedLayout` raises
  *"only first/last keyframe anchors are supported"* and the node refuses rather
  than failing deeper in. MiniMax's guide lists interior anchors as valid and they
  do work; **#15439** removes the restriction upstream, and the **Looping Sampler**
  exposes it as `keyframe_indices`.

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

  This path needs nothing beyond stock ComfyUI. A few nodes on `main` ask for an
  upstream PR and say so; only MONKEYPATCHES live on the **`keyframe-anchors`**
  branch. See [`docs/core-changes.md`](docs/core-changes.md).

### Prompting
- **MMH3 Asset Plan** / **MMH3 Task System Prompt** — build a Context-IR system
  prompt for your own LLM node from the task type (or combination) and the
  assets in play, emitting only the relevant rule blocks. See
  `docs/context-ir-system-prompt.md` for the full spec these are derived from.
- **MMH3 Prompt Lint** — check a written prompt against the format its `mode`
  implies: missing sections, a `retention_analysis` line with no marker, a hidden
  cut, timestamps out of order, `[Shot 1]` carrying one. Reports rather than
  rewrites.
- **MMH3 Replace Section** — splice one refined section back into a complete prompt.
  The two-model route: the technical model writes the whole prompt, a second expands
  `detailed_description`, this puts it back. Both formats' section sets are known,
  so it refuses a section the selected mode does not have.
- **MiniMax H3 Prompt Accumulate** — append one prompt to a running pipe-separated
  string, for a graph loop writing one prompt per window. Exists because a loop
  carries values, not lists. The first pass is the case that goes wrong: the carried
  slot is unwired on iteration 0, and a naive accumulator emits a leading separator
  or the literal text `None`. `prior_context` formats the earlier prompts for
  feeding back to the writing model — put a second copy at the *top* of the loop
  body to read it, since this node sits after the model and its own output cannot
  reach upstream.

  **`prior_context_mode` is the lever on repetitive output.** `all` (the default)
  re-sends every earlier prompt in full — ~7,900 tokens by window 7 of a 20s-window
  clip, against a few hundred for the new audio, which is roughly 20:1 in favour of
  copying. It also re-sends every earlier `detailed_description`, the one section
  the header asks to *differ*. `last_definitions` sends only the previous window's
  `subject_definitions` and `retention_analysis` — what must stay identical, and
  nothing to imitate for what should not. If late windows are re-describing earlier
  ones, start here.

### Sampling
- **MiniMax H3 Looping Sampler** — fill a whole clip chunk by chunk in one node
  execution. The graph is the same size for 4 chunks or 40, which is the point.

  **The latent is the finished clip**, and the chunk count is derived from it — you
  hold a song of known length and do not know how many chunks that is. Chunks are
  slices written back in place, so there is no join, no trim, and the output is
  exactly the length you passed in. Each chunk also slices its own span of audio, so
  a track pinned by `use_input_audio` reaches every chunk.

  The schedule comes from the same `_plan` as **Window Plan** and **Split Audio to
  Windows**, so chunk N renders the audio window N's prompt was written against.
  Two carry routes (masked overlap, or a guide), keyframe indices in clip frames,
  and a per-chunk guider swap.

  The sigma schedule can be **windowed per chunk** with `sampling_start_step` /
  `sampling_end_step` — absolute indices, sliced exactly as core `SplitSigmas` does,
  so a two-pass run is `end N` then `start N` with no arithmetic. `phase2_start_step`
  plus an optional `phase2_sampler` / `phase2_guider` switches solver mid-schedule
  for dual-solver setups. All three carry LTXAVTools' semantics unchanged. See
  [`docs/looping-sampler.md`](docs/looping-sampler.md) — including what is still
  unmeasured.
- **MiniMax H3 Keyframe Planner** — end-anchored keyframe indices for a chained run,
  ported from LTXAVTools' planner. Frame 0 opens, each chunk travels to a keyframe at
  the last frame **it renders**, the final one ends on `-1`. Start-anchoring instead
  would put each image in the NEXT chunk and invite a snap at every seam. Emits
  `indices` for the sampler's `keyframe_indices`, `count` for how many images the
  batch needs, and `chunk_count`.

  Same three numbers as the sampler, same `_plan`, so the two cannot disagree about
  where a chunk ends.
- **MiniMax H3 Context Windows** — windowed sampling over one long latent, per
  modality: video on dim 2, audio on dim 3, each with its own window. Snaps length
  and overlap to the grid, since an overlap that is a multiple of 5 rather than
  `5m+2` walks the window phase `0,2,4,1,3` — a five-window beat, which is the
  pulsing. See [`docs/context-windows.md`](docs/context-windows.md).

  Windows are **not** a way to grow a clip: every window is a slice of one
  preallocated latent, and all of them sit at the same noise level at every step.
  Chaining is what grows.
- **MMH3 Window Plan** — resolve the whole schedule up front, in frames. How many
  windows you get is how many prompts to write; whether your window and overlap
  survive snapping is otherwise only knowable by running a generation.

  `context_length` / `context_overlap` are **latents**, for Context Windows.
  `window_frames` / `overlap_frames` are **frames**, for Split Audio to Windows.
  Crossing them re-snaps a latent count as a frame count and the two schedules
  quietly diverge.
- **MMH3 Split Audio to Windows** — cut a track into one clip per window, matching
  the real schedule including the overlap and the clamped final window. The numbered
  sockets fan every window across the graph at once; the `audio` output emits ONE,
  chosen by `index`, so a for loop keeps the graph constant-size. `index` also
  reaches past the numbered ceiling.
- **MMH3 Window Context** — one line saying which span of the song a window covers,
  for the per-window prompt loop. Without it the loop hands the writing model the
  same text every iteration and only the audio changes, so on a repetitive track
  nothing distinguishes window 5 from window 2 — and `prior_context`'s "keep these
  byte-identical" then pulls the late windows onto the same shots. Same `_plan` as
  everything else, so the timecode names the audio the window really renders.
  Concatenate onto the **END** of the model's prompt, after `prior_context`.

### Latent
- **MiniMax H3 Seed Overlap** — **prepends** overlap latents to the target and masks
  them, giving frame-level seam continuity. Prepending rather than overwriting means
  the chunk keeps its full requested duration and the overlap is cut off afterwards.
  Needs **#15375**; refuses without it.
- **MiniMax H3 Outpaint Latent** — grow or crop a latent's canvas, masking the new
  region so the model fills it. Edges are **signed**: positive pads, negative crops,
  and each snaps toward zero so a value between steps never crops more than asked.
  An inward `feather` ramps into the source region. H3 has no cross-attention, so
  margin rows attend directly to real rows at every layer, and scene fill converges
  in very few steps.
- **MiniMax H3 Join AV** — join two clips in **pixel** space, at frame granularity.
  Latent joins land on 17-frame boundaries; this is what Find Divergence's answer
  feeds.
- **MiniMax H3 Reference from Latent** — build a `minimax_refs` block from a latent
  directly.
- **MiniMax H3 Streaming Encode** / **MMH3 Streaming Save** — encode and export
  in bounded RAM. Save decodes group by group and writes as it goes rather than
  holding the whole clip, which is the difference between exporting a long master and
  running out of memory. Slower per frame; for long videos only.
- **MiniMax H3 Trim AV** — drop latents from the head and/or tail, cutting audio and
  masks to match. Note the grid rule **inverts** relative to Concat AV: trimming one
  latent, `5m` keeps the result on grid and `5m+2` takes it off, because there the
  constraint is on the joined *total* rather than the piece being cut.
- **MiniMax H3 Split AV** — pull an AV latent into plain video and audio latents. The
  exact inverse of Pack AV, so carrying stage 1's audio through an upscale ladder is
  something the graph states rather than a discipline you have to remember.
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

> **On `noise_mask`:** masks do reach the model — `samplers.py` packs latents
> before sampling and explicitly handles `denoise_mask.is_nested`. What stock
> lacks is per-row TIMESTEP handling: preserved rows still run at the generation
> timestep, so the model gets clean content labelled as noisy and the mask
> accomplishes nothing. **drozbay's per-row masking fixes it — upstream PR
> [#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375).** `MMH3SeedOverlap`
> and the outpaint node need it, and refuse to run without it rather than
> appearing to work. Applying an upstream PR is not monkeypatching, which is why
> they live here rather than on `keyframe-anchors` — see
> [`docs/core-changes.md`](docs/core-changes.md).

For **audio-driven video**, use an audio reference with the `[audio reuse]` task
type and the `fully_copy` marker, not a mask. That is a trained capability.

### Util
- **MMH3 Latent Info** — shapes, frame count, audio-length mismatch, grid
  alignment, mask presence.
- **MMH3 Cond Set Spread** — spread a cond_set's N prompts across a windowed
  generation, so each window gets the one written for it. Regions are cut per window
  midpoint; guess the prompt count low and windows share a prompt, guess high and the
  last prompts are never reached. **MMH3 Window Plan** tells you the number.
- **MMH3 Reframe Pads** — pick a target aspect and get the four **signed** edges for
  Outpaint Latent. `extend` grows to reach it, `crop` cuts, `balanced` does both.
  Snapped to the canvas multiple, so what it emits is what outpaint will honour.
- **MMH3 Upscale Ladder** — an aspect and a target long edge in, a ladder of
  `width_N`/`height_N` out, every rung on the canvas grid. For staged upscales,
  so the stage sizes agree by construction rather than by arithmetic you redo.
- **MMH3 Regenerate-2K Dimensions** — the two stages of a 768p → 2K pass.
  **Stage 1 is not a choice**: it reproduces core's `adapt_canvas`, because that is
  what H3-Base emits whatever you ask for, and sizing it any other way makes stage 2
  an upscale of something never rendered. Stage 2 is an integer multiple of stage 1's
  on-grid unit, so the aspect is exact — rounding each axis to 32 instead puts 16:9 at
  2048x1184 (1.7297), and that squeeze is in every frame. The label says when the
  requested long edge could not be honoured.

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

## Carrying content between chunks

On stock ComfyUI there is one channel, and it does not do what its name suggests:

| channel | mechanism | carries | position |
|---|---|---|---|
| `MMH3LatentToRef` | `minimax_refs`, never denoised | identity, voice, motion style | before the clip, contiguously |

Two more need a patched core. `MMH3SeedOverlap` (target latent + `noise_mask`)
needs per-row timestep handling to mean anything -- **#15375**. Positioned anchors on
the clip's own timeline need interior indices and the accumulate fix -- **#15439**,
which the **Looping Sampler** uses as `carry="keyframe"`. Both are upstream PRs
applied to core, not monkeypatches; see [`docs/core-changes.md`](docs/core-changes.md).

**References are positioned.** The layout lays them out from a cursor starting at
`text_len`, a `video`/`video_audio` block advances that cursor by its own temporal
span, and the target uses the cursor's final value as its origin — so a carried tail
sits contiguously immediately before the clip, not floating outside time. What it
costs is *distance*: a 39-frame carry moves target frame 0 from 320 to 385 at
`text_len` 320. Audio is free, though — `FRAME_RESCALE` is 5/3 and `40/24` is 5/3, so
a matched audio tail spans exactly what the video spans and the layout's `max()` is a
no-op.

**A noise mask pins at the sampler, not the model.** Each step the model predicts the
whole clip and the mask overwrites the pinned region afterwards, so it is corrected
rather than conditioned — it never knows the region is fixed when predicting the rest.

A third channel, **positioned keyframe anchors**, pins a run of consecutive tail
frames on the clip's own timeline at **no distance cost** -- measured, target origin
`text_len + 0` against `text_len + 65` for the same carry as a `video_audio` ref. It
needs **#15439**, and the Looping Sampler's `carry="keyframe"` is it. The
`keyframe-anchors` branch reached the same place with monkeypatches and is superseded
now that core carries the PR. Not yet run against real weights.

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
- `ref2va` **does** respond to keyframe (`cond`) rows, but two bugs sit in the
  way and only one of them is fixed upstream. **#15439** stops `model_base.py`
  overwriting `cond_video_latents` — it concatenates keyframes-then-refs now, so
  refs no longer erase keyframes. What it does **not** fix is the position:
  `cond_t` is still computed from `text_len` alone, ignoring the cursor the
  reference blocks already advanced, so a guide lands `ref_advance` units before the
  clip whenever refs are present — measured at **−1** for one image reference and
  **−320** for a chunk's worth of voice audio. Nothing errors; it just anchors into
  the reference region. This pack corrects it with a runtime wrap
  (`patch_guide_origin.py`), inert unless guides and references are BOTH present, so
  core stays stock. Drift table in [`docs/core-changes.md`](docs/core-changes.md).
- Latent-space downscaling is bilinear and approximate.
- Audio seams: the audio VAE is DAC encoder + BigVGAN decoder. Crossfade in the
  **waveform** domain after decode, never in latent space.
