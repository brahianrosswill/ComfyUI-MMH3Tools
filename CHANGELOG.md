# Changelog

All notable changes to MMH3Tools are documented here.
This project follows [Semantic Versioning](https://semver.org/).

**Input ordering is append-only.** ComfyUI serialises widget values positionally,
so new inputs must be added at the END of a node's input list. Never insert or
reorder existing inputs, or saved workflows silently rebind to the wrong widgets.

## [0.21.1] - 2026-08-07

### Fixed
- `MMH3PromptLint` reported an off-screen voiceover failure that could not be traced
  to anything in the prompt. Two bugs in one pattern:

  ```python
  says in an off-screen voiceover.*?</d>(.{0,120})
  ```

  The `.*?` is unbounded, and under `re.S` it leaps across the whole document to
  whatever `</d>` appears next, then judges the 120 characters after **that**. The
  phrase appears verbatim in the format rules as an instruction, so any text carrying
  them plus an unrelated dialogue block anywhere later reported a failure with the
  lips-closed statement sitting untouched beside the phrase. The dialogue must now
  follow the phrase within 40 characters.

  The trailing window was also **consumed**, so `finditer` skipped past anything after
  it: a prompt whose SECOND voiceover was the broken one linted clean. It is a
  lookahead now.

- The voiceover finding quotes the text it matched, like the neighbouring `<d>` rules
  already did. A finding you cannot locate is a finding you cannot act on.

## [0.21.0] - 2026-08-07

### Added
- `MMH3LatentToKeyframes` - pin the previous chunk's tail as a RUN of positioned
  keyframe anchors, which is what chaining needs and what two anchors cannot express.

  `MMH3LatentToRef` already placed the tail correctly in TIME - refs advance a
  cursor and the target begins after them, contiguously. The problem is that the
  advance moves the target away from the prompt. Measured on the real `PackedLayout`
  at `text_len` 320, a 39-frame carry:

  | carry as | target origin | vs text |
  |---|---|---|
  | `video_audio` ref | 385.00 | **+65** |
  | keyframes | 320.00 | +0 |

  Sixty-five position units between the end of the prompt and frame 0, for a token
  cost that is otherwise near identical (12226 rows vs 12096). Audio does not enlarge
  it: `FRAME_RESCALE` is 5/3 and `40/24` is 5/3, so a matched audio tail spans exactly
  what the video spans and the layout's `max()` is a no-op.

  Head-anchored, so the pinned run occupies output frames `0..span-1` and must be
  trimmed before joining - wire `pinned_frames` into `MMH3ConcatAV`. Negative indices
  would avoid that waste but put `cond_t` below `text_len`, colliding with text token
  positions.

  No VAE. The tail is already latent, and a `5m+2` tail off a `5j+2` clip starts at
  step `5(j-m)`, always phase 0, so the slice is exactly what a fresh encode of those
  frames would produce - lossless and free.

- `mmh3tools/patch_layout.py` - interior keyframe anchors, patched at RUNTIME rather
  than by editing core. `cond_t = kf_base + FRAME_RESCALE * pixel_index` is linear in
  pixel frames and every intermediate index is representable; stock just never computes
  one and raises `only first/last keyframe anchors are supported`.

  `PackedLayout` is constructed inside the model's forward, so unlike `context_handler`
  there is no injection point and no subclass route. Three properties make the wrap safe:
  the fixup is **absolute**, recomputing positions rather than adjusting them, so it
  cannot double-apply on top of the file-level patch; it is **inert**, leaving keyframes
  without its private key byte-identical to stock; and it is **self-tested** at import
  against the live class, refusing to apply rather than rendering a shifted join.

  This subsumes the keyframe half of `docs/core-patches` - the self-test asserts the
  anchor lands on the target origin whether or not core is edited, so the ref-cursor
  offset is now correct on unmodified ComfyUI too.

- `step_frame_offsets()` and `FRAME_PER_TOKEN` in `common.py`.

## [0.20.0] - 2026-08-07

### Added
- `MMH3ContextWindows` gains a `freenoise` switch (default off, appended last).
  0.15.x hardcoded it off and stubbed `_apply_freenoise` out entirely. FreeNoise
  copies each window's noise forward into the next window's region, permuted, so
  overlapping windows start from related noise rather than independent noise -
  which is what full-denoise windowing was missing. Shuffles VIDEO only, on its own
  temporal dim; the stock multimodal path would have permuted audio's stereo axis.

## [0.19.2] - 2026-08-07

### Changed
- `MMH3TaskSystemPrompt` format rules tightened - the skeleton now shows three shots
  stacked in ONE field and states that three field labels appear in the entire output,
  once each. A local model was emitting the whole field set per shot.

## [0.19.1] - 2026-08-07

### Fixed
- `MMH3PromptLint` missed repeated sections, and two bugs hid it. Section boundaries
  now tolerate leading whitespace (`\n\s*%s\s*:`) - without it an indented prompt, which
  LLMs produce constantly, never matched the stop and every section ran to end of
  document. And callers pass the FULL section list so a repeat of the same label
  terminates it; the last field otherwise swallowed every repeated block after it,
  which is how a phantom mood word turned up in `non_diegetic_music`.
- Sections are now COUNTED, not tested for presence. `re.search` finds the first and
  stops, so a prompt with every field repeated per shot linted clean.

## [0.19.0] - 2026-08-07

### Added
- `MMH3ReferenceFromLatent` gains `ref_images` (Autogrow, max 9) and `ref_image_size`.
  Stills are emitted BEFORE the carry in `ref_items` so `<Picture N>` numbering matches
  the stock node.

## [0.18.0] - 2026-08-06

### Added
- `MMH3ImageToRef` - append a still image to `minimax_refs`, closing the last hole in
  the conditioning matrix: latents could become refs or keyframes and images could
  become keyframes, but nothing put an image into refs by appending. Stock
  `MiniMaxH3ReferenceToVideo` accepts `ref_images` but builds conditioning from
  clip+prompt rather than appending, so it cannot add a still to conditioning that
  already exists - which is what stacking a reference face alongside carried latent
  refs requires.

  Reference blocks carry their own `latent_h`/`latent_w`, unlike keyframes, so this is
  free to resize. Sizing mirrors the stock node exactly (match/max, scale-down only).
  The label reports tokens per sampling step, since reference rows are attended every
  step and, under context windows, every step of every window: 999 at match versus
  5440 at max on a 3000x4000 source.

## [0.17.0] - 2026-08-06

### Added
- `MMH3StreamingEncode` - chunked VAE encode, so long clips at high resolution can
  be encoded at all. **Output is bit-identical to `VAEEncode`** (max|diff| exactly
  0.00e+00, verified at 39 and 124 frames across chunk sizes 17, 85 and 1700).

  `F.pad(..., mode="reflect")` in H3's `CausalConv3d` requires the tensor to fit
  32-bit indexing - under `2**31` elements. A pixel batch is `[1, 3, T, H, W]`, so
  that is a JOINT ceiling on length and resolution:

  | resolution | max frames | duration |
  |---|---|---|
  | 1024x768 | 906 | 37.7s |
  | 1536x1152 | 396 | 16.5s |
  | 2048x1536 | **226** | 9.4s |

  Past it, `VAEEncode` dies with *"input tensor must fit into 32-bit index math"*.
  That is **not** an OOM, so `raise_non_oom()` re-raises it and ComfyUI's automatic
  retry-with-tiled-encoding never fires - a hard stop rather than a slow fallback.
  The ceiling shrinks as an upscale ladder climbs, so a length that sails through
  stage 1 can fail at stage 3.

  **Chunking is exact here** because `encode_temporal` slices into non-overlapping
  17-frame clips and encodes each with no carried state. Clip boundaries are free -
  unlike LTX, whose encoder has a causal receptive field across boundaries and needs
  left context re-encoded and trimmed per chunk.

  **The trap**: the tail padding and `token_drop` are applied once PER CALL, so
  looping `vae.encode()` over chunks silently loses 3 latents per chunk - 39 frames
  give 12 latents whole but `2+2+2 = 6` as three calls. Not an error; just a shorter
  latent that decodes to a shorter, wrong video. The node therefore drives
  `_adaptive_encode` directly and applies the pad and the drop exactly once, then
  reproduces `encode()`'s moments-to-latent step. The single `token_drop` is what
  turns `5j` clips into the `5j+2` grid.

  `frames_per_chunk` snaps to a multiple of 17 and **does not change the result** -
  it is purely a memory/passes dial. Going around `VAE.encode()` means the node loads
  the model itself, budgeting for one chunk rather than the whole clip.

  Scope, stated plainly: this raises the LENGTH ceiling; it does not by itself give
  constant RAM, because the incoming `IMAGE` batch already exists in full before the
  node runs. Constant RAM needs reading frames from disk per chunk, as LTXAVTools'
  streaming encode does.

## [0.16.0] - 2026-08-06

### Added
- `workflows/minimax h3 I2V 2K.json` - the three-stage I2V-to-2K workflow, and the
  first example shipped with the pack. Generate small, then two low-denoise windowed
  upscale passes, with audio decided in stage 1 and carried forward.

  Uses `MMH3UpscaleLadder`, `MMH3ContextWindows` (both upscale samplers),
  `MMH3FrameCalculator`, `MMH3TaskSystemPrompt` and `MMH3PackAV`, plus
  ComfyUI-LlamaOmni for prompt writing, KJNodes, RES4LYF, VideoHelperSuite and
  rgthree.

  Cleaned before shipping: a replacement `CLIPLoader` had come in with
  `type = "stable_diffusion"` instead of `minimax` (a silent functional break, since
  a fresh CLIPLoader defaults to that and does not announce it); two nodes still
  pointed at `ckinpdx/MMH3Tools`, a deleted repo, at a commit unreachable from the
  current history; the LlamaOmni nodes had no `aux_id` so Manager could not install
  them; and three `videopreview` blocks referenced local output files.

## [0.15.4] - 2026-08-06

### Changed
- **Corrected: windowing is FASTER at high resolution, not slower.** Measured -
  stage 3 at 2K ran about a minute faster windowed than whole. 0.15.3's framing
  ("you pay in passes", a time-for-memory trade) reasoned linearly and was wrong.

  For 57 latents at window 17, overlap 7 (stride 10, 5 windows):

  ```
  attention  proportional to N^2   5 x 17^2/57^2  =  0.44x   56% LESS work
  linear     proportional to N     5 x 17/57      =  1.49x   49% MORE
  ```

  Both ratios are resolution-independent; only the mix varies. At ~131k tokens
  attention dominates so heavily that the 0.44x decides it and the overlap tax is
  noise.

  Practical consequence, now in the tooltip: **smaller windows are not faster.**
  Window 12 has the same 0.44x attention ratio - more windows exactly cancels the
  smaller square - while linear cost rises to 2.1x. Shrink `context_length` for
  memory only.

## [0.15.3] - 2026-08-06

### Changed
- Tooltips now say which knob is the VRAM lever. `context_length` sets peak
  activation cost (it scales with the window, not the clip); `context_overlap`
  changes how many windows run, so it trades time and seam quality, not memory.

  Measured at 192 frames (57 latents), tokens per forward:

  | | full clip | window 17 | window 12 | window 7 |
  |---|---|---|---|---|
  | `1536x864` | 73,872 | 22,032 | 15,552 | 9,072 |
  | `2048x1152` | 131,328 | 39,168 | 27,648 | 16,128 |

  An ordinary `1344x768` 8s generation is 57,456 tokens for comparison - so a
  windowed 2K pass has a **smaller sequence per forward than a normal 768p
  generation**, and attention is quadratic on top of that.

  Stated in the tooltip because it is easy to misread: this reduces ACTIVATION
  memory only. The H3 UNet is ~21 GB even pruned, so windowing does not make the
  model loadable on a card that could not already load it. It helps the user who
  can run H3 but cannot hold activations for a long or high-resolution clip.

### Known limitation
- ComfyUI's VRAM estimator does not shrink for windowed H3. `pack_latents` returns
  `[B, 1, N]`, so `_prepare_sampling_wrapper` sees `is_packed` and skips the
  per-window clamp behind an upstream TODO ("latent_shapes cond isn't attached yet
  at this point"). Real peak memory does drop, but ComfyUI budgets as if the clip
  were unwindowed and offloads more of the model than necessary - slower than it
  needs to be, not an OOM. Fixing it upstream would help every packed-latent model.

## [0.15.2] - 2026-08-06

### Fixed
- **Pulsing across the clip** - the same visual signature as joining off-grid
  latents, but a different cause, and one I created.

  H3's latent groups start at `2+5k`. Window stride is `context_length -
  context_overlap`, and 0.15.0 forced length to `5j+2` **and overlap to `5m`**, so:

  ```
  stride = (5j+2) - 5m = 5(j-m) + 2   =  2 (mod 5)   always
  ```

  Every window start advanced 2 in phase against the group grid, cycling
  `0, 2, 4, 1, 3` — **a five-window beat**. Each window presents its first two
  latents to the model as though they were the 5-frame anchor group, so the
  temporal warp differs per phase and repeats. That is the pulse.

  Fixed by snapping overlap to **`5m+2`** (2, 7, 12, 17...) rather than a multiple
  of 5, which makes the stride a multiple of 5 and puts every window at the same
  phase:

  ```
  stride = (5j+2) - (5m+2) = 5(j-m)   =  0 (mod 5)
  ```

  Default overlap is now 7 rather than 5. Whatever warp remains is then identical
  in every window, so there is no periodic change to see. Note this is the exact
  opposite of what the 0.15.0 tooltip told you to do.

  Test 11 asserts stride divisibility and phase uniformity across three window
  sizes, and asserts that the old `overlap=5` config genuinely does cycle — so the
  bug cannot come back unnoticed.

## [0.15.1] - 2026-08-06

### Fixed
- **Crash on the first sampling step**: `The size of tensor a (2) must match the
  size of tensor b (93) at non-singleton dimension 2`, raised from
  `combine_context_window_results`.

  0.15.0 fixed the per-modality dim in `prepare_window()` and slicing, but two more
  places in `IndexListContextHandler` index a modality tensor with the handler's
  `self.dim`, and both hit audio on its **stereo axis**:

  - `combine_context_window_results()` builds the fuse weights with
    `x_in.shape[self.dim]` and `match_weights_to_dim(..., self.dim)`, so a 93-long
    audio weight vector was sized onto dim 2 (extent 2). That is the crash.
  - `execute()` allocates `counts` via `get_shape_for_dim(m, self.dim)` and `biases`
    as `[0.0] * m.shape[self.dim]`, giving audio a counts tensor of extent 2 instead
    of `T40` and a biases list of length 2. This would have failed immediately after.

  Both are now overridden in `MMH3ContextHandler`, using the **window's own** dim for
  fusing and a per-modality dim for allocation. `execute()` is copied from upstream
  rather than wrapped, because the allocation is inline; the two changed lines are
  marked, and if upstream refactors it breaks loudly here instead of quietly
  windowing the wrong axis.

  Tests 9 and 10 cover exactly this: accumulator extents per modality, and the fuse
  step running on both without raising.

## [0.15.0] - 2026-08-06

### Added
- `MMH3ContextWindows` - sliding-window sampling over a long AV latent, **with no
  core patching**. `MMH3ContextHandler` and `MMH3WindowingState` subclass ComfyUI's
  own windowing.

  **Intended for low-denoise upscale passes only.** At low denoise every window
  starts from the same upscaled base, so coherence comes from the input rather than
  from attention spanning the clip; at full denoise each window invents its own
  content and they disagree. Attach it on stages 2 and 3 of an upscale ladder, never
  on the pass that decides structure.

  Two things stopped stock ComfyUI doing this:
  - `map_context_window_to_modalities` has **zero implementations tree-wide** - the
    name appears twice, at the call site and in its own error message - so the
    multimodal path raises `NotImplementedError` for every model. Overriding
    `prepare_window()` means the hook is never called at all.
  - `WindowingState` uses ONE `dim` for every modality. H3's video is dim 2 and
    audio is dim 3, so the stock path would window audio `[B,32,2,T40]` on its
    **stereo axis** - size 2, not `T40`. No crash; just a ratio of `2/T` and
    nonsense indices.

  Neither needs a core edit, because the handler is only an object in
  `model.model_options["context_handler"]`. That is worth more than convenience: it
  survives `git pull`, and when upstream refactors it fails loudly with an
  `AttributeError` instead of silently doing the wrong thing, which is what a stale
  diff does.

  Audio boundaries are converted independently and subtracted rather than converting
  a window length, because `audio_t = round(frames/24*40)` is not additive - the
  same correction `MMH3ConcatAV` needed. The mapping is exact at every on-grid
  boundary.

  Pinned by the node: windows snap DOWN to `5j+2` latents and overlap to a multiple
  of 5, since the model only ever saw `5j+2` clip lengths; `causal_window_fix` off,
  because it prepends an anchor frame that would push every window to `5j+3`;
  `freenoise` off, since it exists to improve window blending and a low-denoise pass
  has very little noise to shuffle; and looped/batched schedules are not offered,
  because they can emit wrapping windows that the audio mapping cannot express as a
  time span.

- `tests/test_windows.py` - 27 assertions, including a direct check that the stock
  single-`dim` path would have hit the stereo axis, and that windows tile the whole
  audio track with no gap.

- `docs/context-windows.md` - the full read of `comfy/context_windows.py`, what a
  core-side fix would touch, and why the node approach is preferable.

## [0.14.0] - 2026-08-06

### Added
- `MMH3UpscaleLadder` - three exact-aspect, on-grid stages for a progressive
  generate-small-then-denoise-up pipeline. Separate node; `MMH3DimensionCalculator`
  is untouched.

  **Why integer multiples of a unit instead of snapping.** A ratio lands exactly on
  the 32px grid only at integer multiples of its minimal on-grid unit: 16:9 needs
  `w/h = 16/9` with both `/32`, which is `w = 512k, h = 288k`. Working in `k` rather
  than pixels means no stage is ever snapped, so the aspect cannot drift between
  stages - which matters here, because a low-denoise pass onto a slightly different
  aspect resamples the whole frame instead of just adding detail. Limiting the ratio
  set is what makes this possible.

  Three constraints, all measured rather than chosen:
  - every stage exact-aspect and on the 32 grid
  - no step above 2x - a low-denoise pass cannot invent more than that
  - stage 1 at or above `min_megapixels` (default **0.4**, measured): below it the
    first pass stops being upscalable and stage 2 sharpens mush instead of repairing
    structure

  Stage 2 is placed at the geometric mean of stages 1 and 3, clamped to the window
  both step limits allow, so the work spreads evenly across the two passes.

  | ratio | stage 1 | stage 2 | stage 3 | steps |
  |---|---|---|---|---|
  | 16:9 | 1024x576 | 1536x864 | 2048x1152 | 1.50x, 1.33x |
  | 4:3 | 768x576 | 1280x960 | 2048x1536 | 1.67x, 1.60x |
  | 3:2 | 864x576 | 1344x896 | 2016x1344 | 1.56x, 1.50x |
  | 1:1 | 640x640 | 1152x1152 | 2048x2048 | 1.80x, 1.78x |
  | 21:9 | 1120x480 | 1568x672 | 2016x864 | 1.40x, 1.29x |

  Degenerate configurations are reported rather than silently producing a duplicate
  stage, and the two causes are distinguished because they need opposite fixes: a
  total upscale too LARGE for three 2x steps says to raise `min_megapixels` or lower
  the target, while one too SMALL says no on-grid stage fits in between and it is
  really a 2-stage ladder.

### Note
- **2K generation is not possible with the open weights.** H3-Base is 768p; 2K comes
  from H3-Regenerate-2K, which feeds the 768p result plus the original context back
  through H3, and which MiniMax has not open-sourced ("we will release it once it is
  ready"). This ladder is for a local progressive-upscale pipeline, not for asking
  the base model to generate at 2K directly.

## [0.13.0] - 2026-08-06

### Added
- `MMH3PromptLint` - checks a finished, LLM-written prompt against the H3 format
  rules before anything is sampled. Passes the prompt through unchanged, so it sits
  inline between the LLM node and the conditioning node.

  `MMH3TaskSystemPrompt` validates the SETTINGS you gave it; this validates the TEXT
  that came back, which is where the interesting failures are - a local model follows
  most of a long rule list and quietly drops the rest.

  The argument is economic. A chunk is minutes of sampling, and most format errors do
  not crash, they render something subtly wrong: a cut timed past the end of the clip
  simply never happens, a quoted line of dialogue asks for a sign instead of speech, a
  voiceover missing its lips-closed clause gets mouthed. Each costs a full generation
  to find by watching and a second to find here.

  Checks: required sections for the mode's format; body opens with `[Shot 1]`;
  `[Shot 1]` carries no timestamp; timestamps strictly increasing and unique; shot
  numbers 1..N in order; no cut at or past the duration; `<d>` tags balanced, each
  carrying a `[Language]` tag and containing no speaker ID or delivery verb; dialogue
  never in double quotes; every off-screen voiceover followed by the lips-closed
  statement; no dialogue in `overall_soundscape`; no mood words in
  `non_diegetic_music`; every `<Picture/Video/Audio/Subject N>` used in the body
  defined in `subject_definitions`; no `(Sx)` in `retention_analysis`; and a
  `[task type]` prefix on the summary.

  `on_problem` selects `warn` (log and pass through) or `error` (stop the queue).

  Derived from the `lint()` in a standalone H3 film script, generalised to both
  prompt formats - which immediately caught a bug of its own: the shot body is the
  FIRST field in the three-field format but the FOURTH in the six-section one, so
  taking `sections[0]` linted `subject_definitions` and silently passed every shot
  and timestamp check.

- `tests/test_lint.py` - 26 assertions over a clean prompt and a deliberately broken
  one carrying every fault at once.

## [0.12.1] - 2026-08-06

### Fixed
- The `speech` and `sung lyrics` blocks assumed transcription would happen
  implicitly. Buried in a system prompt whose stated job is "convert a rough video
  idea into a structured prompt", a local model treats it as a detail and composes
  from the text idea alone. The block now opens by stating that an audio clip is
  attached and must be listened to first, and each kind makes the transcription an
  ordered step to finish BEFORE composing. Sung lyrics adds that the effort belongs
  there rather than in the prose, and that unclear passages should be omitted rather
  than filled with plausible substitutes - invented words get animated onto the mouth.
- The no-dialogue warning claimed the model "cannot hear" the track. It can:
  LlamaOmni sends `input_audio` and omni models transcribe. The real risk is asking
  one call to transcribe AND compose, so the warning now points at a dedicated ASR
  pass instead.

## [0.12.0] - 2026-08-06

### Added
- `MMH3TaskSystemPrompt` gains `masked_audio` (combo: `none` / `background music` /
  `speech` / `sung lyrics`, appended last), for the **undocumented** technique of
  masking a supplied audio latent so the track survives verbatim into the output.
  This is the base-mode equivalent of Ref2VA's `[audio reuse]` + `fully_copy`, and
  MiniMax's guides do not cover it.

  **The point is that it inverts what the audio fields mean.** In the three-field
  format, `overall_soundscape` and `non_diegetic_music` normally REQUEST sound to be
  generated. With a masked track they DESCRIBE sound that already exists, and their
  only job is to tell the model what it is about to hear so the picture matches.
  Written the usual way they ask for audio that cannot be added, and the video ends
  up expecting events the track never delivers.

  Per-kind rules, because the model has to know what is in the track to generate a
  matching picture:
  - **background music** - goes in `non_diegetic_music`; diegetic instead if a
    visible source produces it. **Nobody speaks**: no `<d>`, no `(Sx)`, mouths closed
    or occupied, since a character shown mid-speech with no voice reads as broken.
    Cut in sympathy with the music but invent no hits or drops the track lacks.
  - **speech** - transcribe into `<d>` at the moment each line is heard so the lips
    match; `(Sx)` by vocal-event order; voice description must match the track, not
    an invented one; explicit mouth movement for the whole line.
  - **sung lyrics** - as speech but *sings*, lyrics verbatim in their original
    language, and describe singing physically (sustained vowels, held notes, breath),
    because sung mouth shapes differ from spoken ones.

  When `masked_audio` is `speech` or `sung lyrics`, the supplied `dialogue` is treated
  as a **transcript of a fixed track**, so the word ceiling added in 0.11.0 is
  replaced by "the track's own timing governs - do not add, cut or re-time lines to
  fit a word estimate". A ceiling would otherwise invite trimming a transcript.

  Three new warnings for configurations that fail silently:
  - `masked_audio` on **Ref2VA**, where `[audio reuse]` + `fully_copy` is the trained
    path and using both describes one track two ways.
  - `speech` / `sung lyrics` with **no dialogue supplied** - the model has to guess
    words it cannot hear, and the lips will not match.
  - `background music` **with** dialogue - the track has no voice to carry it, so any
    `<d>` line is mouthed over silence.

## [0.11.0] - 2026-08-06

### Added
- `MMH3TaskSystemPrompt` gains a `dialogue` input (multiline, appended last), for
  spoken lines that must be used **verbatim**. The rule existed only in
  `docs/context-ir-system-prompt.md` (point 7) and had never made it into the node
  that the pipeline actually calls.

  When set, the system prompt gains a `## Supplied dialogue` block: reproduce each
  line exactly once in order, write no line that is not listed, keep every line and
  cut surrounding action if they do not fit, one `<d>` block each with only the
  language tag and the words inside, never double quotes, punctuation standardised
  to `, . ? !`. The lines themselves are embedded under `DIALOGUE:`.

### Fixed
- **The word budget actively invited padding.** The Constraints block emitted
  "roughly N words of dialogue TOTAL" unconditionally. Harmless when the model
  writes its own lines; destructive when the lines are the user's, because a small
  model handed a word target will pad up to it - and the invented lines arrive
  correctly formatted, in valid `<d>` tags, with plausible `(Sx)` IDs, which makes
  them very easy to miss.

  With `dialogue` set the wording becomes a **ceiling**, states the supplied word
  and line count, and says explicitly not to add lines to reach it. Without it the
  original wording is unchanged.

- The Output section's "invent concrete detail consistent with the intent" licensed
  exactly the padding the new block forbids. With dialogue supplied it narrows to
  "concrete action, camera and ambience detail" and adds "Never invent dialogue."

- The node now warns in its `report` when the supplied dialogue cannot fit the
  duration (e.g. *"40 words but only ~7 fit in 3.750s"*), rather than leaving the
  model to silently drop lines.

## [0.10.0] - 2026-08-05

### Added
- `MMH3ReferenceMultiPrompt` + `MMH3CondSelect` - `MiniMaxH3ReferenceToVideo` with
  N prompts instead of one, for a text-driven sequence where every chunk shares
  the same references and differs only in its prompt.

  **The point is the model swap, not the encode.** Stock does the reference
  resize, `vae.encode`, `audio_vae.encode` and the text encode all inside one
  `execute()`, so N chunks means N copies of the reference work - and N swap
  cycles, because Qwen3-VL-32B and a 33B DiT cannot be resident together in 32GB.
  ComfyUI resolves outputs depth-first, so a naive N-chunk graph runs
  `load TE -> cond -> evict -> load DiT -> sample -> evict -> load TE -> ...`.
  Doing every encode in ONE node execution collapses that to a single swap for
  the whole sequence.

  Outputs a custom `MMH3_COND_SET` type rather than a `CONDITIONING` holding N
  entries, because a multi-entry CONDITIONING means "combine all of these" - a
  mis-wire straight into a sampler would silently merge every prompt into one and
  render plausible-looking garbage. A distinct type makes that unrepresentable.
  Outputs cannot be dynamic (`Autogrow` is `ComfyTypeI`, inputs only), hence the
  select node.

  Prompts are N separate string inputs rather than one delimited field, so a
  local LLM can drive each one independently.

  **Per-prompt memoization**, keyed on `(prompt, reference fingerprint)`. ComfyUI
  caches per node execution, so without it a one-word edit to a single prompt
  would re-run every prompt's Qwen pass. The fingerprint hashes the raw inputs
  *and* the encoded blocks: hashing only the encoded blocks would make cache
  validity depend on the VAE mapping different references to different latents,
  and that is not an assumption worth making when the failure mode is the wrong
  reference used silently in every chunk.

  Still paid per prompt: `clip.tokenize` re-presents the references to Qwen and
  the vision tower plus 50 layers run again. That is inherent - references are
  emitted BEFORE the prompt text, and although `comfy/text_encoders/llama.py`
  threads `past_key_values` through every layer, the CLIP API exposes no way to
  hand it a cached prefix. Negligible for still images; the thing to avoid for
  video references.

- `tests/test_multiprompt.py` - 17 assertions with stubbed clip/vae covering
  encode counts, cache hits and misses, fingerprint invalidation, ref-encode
  reuse, select bounds, and the empty-prompt error.

### Note
- `_build_refs()` **duplicates** the reference-building half of
  `comfy_extras/nodes_minimax_h3.py`, because upstream runs it inline in the same
  `execute()` as the text encode and offers no seam to call. Re-sync it if that
  file changes its sizing, its block keys, or - most fragile - the emission
  ORDER, since the tokenizer assigns `<Picture i>` / `<Audio j>` / `<Video k>` by
  counting items in the order given. A reference video's soundtrack must be
  appended BEFORE the video itself or every later label shifts and the prompt's
  tags stop matching their assets.

## [0.9.0] - 2026-08-05

### Added
- `MMH3ConcatAV` gains `carry_masks` (BOOLEAN, default `false`), appended last so
  saved workflows keep their current behaviour byte for byte.

  Off, the node drops input `noise_mask`s exactly as before (now with a log line
  saying so). On, it concatenates them on the same axes as the latents they
  describe - video dim 2, audio dim 3 - filling an absent side with ones
  ("denoise everything there"), matching the convention `MMH3PackAV` already uses.
  If neither input carries a mask, none is invented.

  The old comment claimed "a per-frame mask cannot span the join". That was never
  true: masks live on the same axes as the latents, so joining them is the same
  `cat` with the same dims. The reason it stays **off by default** is semantic,
  not structural - an inherited mask described a generation that has *already
  happened*, so re-sampling the join would pin two finished seams and regenerate
  everything between them. Turn it on when the join is deliberately the INPUT to a
  bridging pass (MiniMax's `video editing` task type).

  When trimming, the carried mask takes the same **computed** cut as the latent
  (`k` and `drop_audio`, never the raw widget value), and a mask whose length ends
  up disagreeing with its latent is warned about rather than left for
  `prepare_mask` to silently resize.

- `tests/test_concat_av.py` - 27 assertions covering mask carry, the trim
  families, and a `MMH3SeedOverlap` -> `MMH3ConcatAV` round-trip. Run it with
  ComfyUI's venv from the ComfyUI root:
  `venv/Scripts/python.exe custom_nodes/ComfyUI-MMH3Tools/tests/test_concat_av.py`

### Changed
- **`MMH3ConcatAV`'s `trim_b_latents` no longer snaps.** It is now honoured as
  given, clamped only so B keeps its minimum 2 latents.

  Previously it went through `snap_latents()`, which snaps to the `5j+2`
  clip-length grid, so wiring `MMH3SeedOverlap`'s `overlap_latents = 5` in trimmed
  **2**, and 12 of the 17 overlap frames stayed duplicated at the join.

  The snap was not simply a bug, which is worth recording: with `A = 5a+2` and
  `B = 5b+2`, the two things you might want are mutually exclusive.

  | trim | effect |
  |---|---|
  | `5m` | removes a SeedOverlap **exactly**; B's remainder stays on grid; the **total** is `5(a+b)+4-k`, off grid |
  | `5m+2` | total lands **on grid**; ~7 frames of overlap stay duplicated |

  `k` cannot be `0` and `2 (mod 5)` at once, so no snap is right for every use -
  the old one silently picked the second family. The node now honours the value,
  and logs which of the two properties the chosen `k` actually gets. If you need
  both, that is what `MMH3JoinAV` is for: it cuts in pixel space, per frame.

  The audio drop is also corrected. It was `frames_to_audio_t(dropped_frames)`,
  but `audio_t = round(frames / 24 * 40)` is **not additive**, so it now takes the
  difference of the two totals - the same construction `MMH3SeedOverlap` uses to
  size the overlap, so the two round-trip exactly.

### Removed
- The `/mmh3-dim-calc/resolutions` aiohttp route in `nodes_util.py`, along with
  its `server` / `aiohttp` imports. Dead since the dimension calculator moved to
  computing its option lists client-side. It also registered at import time, which
  made the package impossible to import outside a running ComfyUI server.

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
