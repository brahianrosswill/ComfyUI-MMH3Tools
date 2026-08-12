# Regenerate-2K — Field Guide

**Status: partially validated.** The 8-second case has produced a correct 2K result.
Chunking past one window has produced a divergence that is still open — see
§6. Everything describing *structure* below is measured on the real tensors or the
real schedule; nothing here claims a quality outcome.

---

## 1. What this is copying

MiniMax ships H3 as three modules. Only two are open:

| module | status | what it does |
|---|---|---|
| **H3-Context-IR** | hosted only | expands a rough idea into the structured prompt H3 consumes |
| **H3-Base** | open | generates audio and video at **768p** |
| **H3-Regenerate-2K** | **not released** | feeds the 768p result *plus the original context* back into H3 to regenerate at 2K |

From the model card:

> H3-Regenerate-2K: Feeds the 768p result together with the original context back into
> H3 to regenerate the output at 2K resolution. This process leverages both H3's
> powerful generative capabilities and the rich information contained in the original
> context.

Two things follow from that wording, and both shape the nodes here.

**"Back into H3"** — it is not a separate super-resolution network. Same model, second
pass. That is why a local equivalent is buildable at all.

**"Together with the original context"** — the same expanded prompt drives both passes.
MiniMax's own script says so explicitly, exporting one variable for both:

```bash
# Export the complete expanded prompt for H3-Base and regeneration.
EXPANDED_PROMPT=$(echo "$context_ir_result" | jq -er '.task.content.prompt')
```

So the 2K pass re-encodes nothing. It reuses stage 1's conditioning.

---

## 2. Which route

Refine (`MMH3ChunkedPixelUpscale` → sampler) versus regenerate
(`MMH3Regenerate2KReference`) is covered in the README under **Refine vs regenerate**,
including why latent-space upscaling is the wrong tool for the refine leg. The short
version: refine seeds the 2K latent with the upscaled stage-1 picture and denoises it;
regenerate hands the sampler an **empty** 2K latent with the 768p attached as
`minimax_refs`.

**Regenerate is the one this document is about**, because it is the shape MiniMax
describes. Everything below applies to that route.

Using both — seeding the latent *and* attaching per-window references — is a third
option neither MiniMax nor this document has validated.

---

## 3. Dimensions are not a free choice

`MMH3Regenerate2KDims` emits both stages. **Stage 1 is not a parameter.** It reproduces
core's `adapt_canvas` — 768 short edge, area capped at `768*1344`, axes rounded to 32 —
because that is what H3-Base emits whatever you ask for. Sizing stage 1 any other way
makes stage 2 an upscale of something that was never rendered.

Stage 2 is an **integer multiple of stage 1's on-grid unit**, not the requested long
edge rounded to 32. Rounding each axis independently drifts the aspect:

| ratio | stage 1 | 2K | scale | note |
|---|---|---|---|---|
| 16:9 | 1344x768 | **2016**x1152 | 1.50x | not 2048 — 2048x1184 would be 1.7297, not 1.75 |
| 4:3 | 1024x768 | 2048x1536 | 2.00x | lands exactly |
| 3:2 | 1152x768 | 2016x1344 | 1.75x | |
| 1:1 | 768x768 | 2048x2048 | 2.67x | |
| 21:9 | 1536x672 | 2048x896 | 1.33x | |

The label says when the requested long edge could not be honoured and why.

---

## 4. Why the reference is sliced per window

A cond_set is **already per chunk**: the looping sampler takes `conds[i]` for chunk `i`
and passes `minimax_refs` through untouched. So a reference attached to cond `i`
reaches chunk `i` and nothing else, and the slicing is a build-time concern — no
sampler change, no reference building inside the loop.

It matters because reference tokens are re-attended at **every sampling step**. Giving
every chunk the whole 768p clip multiplies that by the chunk count. Measured on a
12-window clip: about **9.9x less reference attention per chunk**.

Cost, for one 192-frame window at 1344x768:

| | |
|---|---|
| video half | 57 latents x 42x24 patch positions = **57,456** |
| audio half | **320** latents |

Audio is 0.56% of it. That is why there is no toggle to drop it — turning it off saves
nothing and removes the only thing telling the model which sound belongs to which
picture. `ref_downscale` is the real cost lever; it hits the video side quadratically.

## 5. The audio is already finished

A resolution pass has no business touching audio; audio has no resolution. So stage 1's
audio is written into the 2K target and **pinned** — `noise_mask` 1 for video, 0 for
audio. Same mechanism as `use_input_audio`, minus the encode, because it is already
latents.

Left empty, the 2K pass would generate an entirely new soundtrack: paying for it, and
drifting from the one the picture was cut to.

The node raises if the target's audio length does not match the source's, because a
mismatch places the audio at the wrong moments rather than merely sounding wrong.

---

## 6. Open: divergence past one chunk

**Observed 2026-08-11.** Two 8-second chunks over a 15.08s clip. Chunk 0 tracked the
original. Chunk 1 diverged around **11s** — frame 264, which is 72 frames *into* chunk
1's new content, not at the seam at 8.00s.

Ruled out by the run's own logs:

- **Schedule misalignment.** `MMH3Regenerate2KReference`, `MMH3LoopingSampler` and
  `MMH3WindowContext` all reported the identical plan: 2 windows, 57 latents, 362
  frames, overlap 7. The ref slices point exactly where the chunks render.
- **A weak reference pathway.** The run already used the fl2va/ref2va hybrid, so
  `adaln_proj` — where reference conditioning is routed — was already ref2va's in
  blocks 30-49.

Leading hypothesis, **untested**: the **overlap carry**. Chunk 1 opens on 22 frames
hard-pinned to chunk 0's *2K regeneration*, while its own reference says those frames
should look like the *768p*. Those are not the same thing — chunk 0 regenerated that
content rather than reproducing it. So chunk 1 starts from a frozen region that
disagrees with its reference, follows the pin for 22 frames, then has two incompatible
sources of truth. Chunk 0 has no such conflict: reference only, nothing frozen.

The official Regenerate-2K has **no carry** — it is one pass over the whole clip, and
continuity comes from the 768p itself. The carry is something chunking introduces.

Test: **`overlap_strength_video` = 0.0** on the 2K sampler. The carried region then
regenerates from the reference like everything else, making chunk 1 structurally
identical to chunk 0. Note this does *not* remove the overlap — adjacent windows still
reference the same 768p frames at the seam, which is the continuity mechanism that
replaces the carry.

`overlap_frames = 0` is **not** the way to do this. On a clip whose latent count does
not divide by the chunk length, the last window clamps backwards and recreates a
physical overlap anyway — measured identical windows at `overlap_frames` 22 and 0 on a
362-frame clip. It would also make adjacent references disjoint, removing the seam
continuity you want to keep.

---

## 7. Not yet measured

- **Whether `overlap_strength_video = 0` fixes §6.** The one test that matters.
- **Whether seeding the latent and attaching references together beats either alone.**
- **Anything past 15s.** H3's stated ceiling is 15 seconds; the 8-second case works and
  15.08s in a single chunk has not been tried, which would sidestep chunking entirely
  for clips of that length.
- **Which `ref_downscale` is affordable.** 2x cuts reference cost ~4x; what it costs in
  fidelity at 2K is unknown.
