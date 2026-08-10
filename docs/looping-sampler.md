# MiniMax H3 Looping Sampler — Field Guide

`MMH3LoopingSampler` renders N chained chunks inside one node execution, carrying
each chunk's tail into the next.

**Status: nothing here has been generated against real weights.** The arithmetic is
tested — grid alignment, index placement, guider handling, the join — and every
number below that describes *structure* is measured on the real `PackedLayout` or
the real latent shapes. Numbers that would describe *quality* (which overlap looks
best, which strength holds lipsync) are not here, because guessing them would be
worse than leaving the gap. Section 9 lists exactly what is still unknown.

---

## 1. Mental model

One node, N chunks, constant graph size.

Driving N chunks from the graph costs a copy of every downstream node per chunk —
sampler, decode, save — and a graph that size is what starts breaking ComfyUI. A
Python loop inside one node costs the same whether it runs 4 chunks or 40.

The price is that **nothing inside the loop can be a graph node**. No LLM call, no
VAE decode you wire yourself, no preview between chunks. Everything the loop needs
must exist before sampling starts. That is what `cond_set` is for: all N prompts,
encoded up front, in one text-encoder load.

So a full pipeline is two phases:

1. **Prompt phase**, in the graph — a for loop (Easy-Use) running your LLM per
   window, accumulating with `MMH3PromptAccumulate`. This has to be a graph loop
   because LLM nodes are graph nodes.
2. **Render phase**, in this node — `MMH3ReferenceMultiPrompt` splits the
   accumulated string into N conds, and the loop renders them.

---

## 2. What the sockets actually do

### `guider` — **its positive is ignored**

The guider supplies the **model**, the **cfg**, and the **negative**. Its positive
is replaced every chunk from `cond_set`, so whatever you wire there never reaches
the model.

Wire `MMH3CondSelect(cond_set, index=0)` into it. Nothing else works better; that
choice just makes the graph honest about what it is.

A **Basic Guider** works too — it simply has no negative to carry. Both shapes are
handled; a Basic Guider's `set_conds` takes one argument and has no `"negative"`
key at all, which is a real difference and not a detail.

If you use a CFG guider, note that a negative built without `minimax_refs` gets a
different packed-sequence length from a positive that has them, so the two will not
batch together in `calc_cond_batch`. It still runs, as separate forward passes per
step. Worth knowing before reading anything into the speed.

### `cond_set` — N prompts, one encode

From `MMH3ReferenceMultiPrompt`. If it holds fewer prompts than there are chunks,
**the last one repeats** and the report says so. That is a legitimate way to render
6 chunks from 2 prompts; it is also what an off-by-one in the prompt phase looks
like, so read the report.

### `latent` — the template, cloned per chunk

One chunk's empty AV latent — the same one the cond_set node emits. It is cloned
per chunk and never mutated, so wiring it elsewhere is safe.

### `chunks` — independent of the prompt count

Deliberately. See above.

---

## 3. The two carry routes

`carry` decides how the previous tail reaches the next chunk. This is the most
consequential switch on the node.

### `mask` — prepend and preserve

The tail goes into the **head of the new chunk's latent** and is masked to 0, so
the model conditions on it without denoising it. `MMH3SeedOverlap` does the work.

- Needs only **#15375** (per-row masking), which `MMH3SeedOverlap` checks for at
  runtime and refuses without.
- `overlap_latents` snaps **down to a multiple of 5**, each worth 17 frames.
- The chunk **grows** by the carry: a 57-latent template becomes 62.

### `keyframe` — carry as a guide

The tail goes forward as a **guide anchored at frame 0** — re-injected every step,
never denoised, carrying a multi-step clip *plus its audio* at the same `cond_t`.

- Needs **#15439**. The node refuses up front rather than dying on chunk 1 after
  chunk 0 has already been paid for.
- `overlap_latents` snaps **down to 5m+2** (2, 7, 12, 17 …), because
  `slice_av_tail` converts with `latents_to_frames`, which only holds on that grid.
- No VAE round trip: the tail is already latent, and a 5m+2 tail off a 5j+2 clip
  starts at step 5(j−m) — always phase 0 — so the slice is exactly what a fresh
  encode of those frames would produce.
- The chunk keeps its **natural length**.

### Why the join costs differ

Both routes reproduce the carry in the head of every chunk after the first, and both
drop it from the second clip at the join. What that costs is not the same.

`ConcatAV`'s trim `k` cannot satisfy two things at once, given A = 5a+2 and B = 5b+2:

| trim | removes the overlap exactly | leaves the master on grid |
|---|---|---|
| `k = 5m`   | yes | **no** |
| `k = 5m+2` | no  | yes |

An **off-grid master cannot be decoded** — `latents_to_frames` only holds on 5j+2 —
so a chained loop is forced into the `5m+2` family.

- **mask**: the carry is a multiple of 5, so the trim must be `carry+2` — 2 latents
  more than the carry itself.
- **keyframe**: the carry is 5m+2 already, so trimming exactly it is grid-safe.

### What each chunk actually delivers

Both lose the same 7 latents per seam. They pay for it differently, and the arithmetic
is not intuitive — a 57-latent template with `overlap_latents` at its default:

| carry | chunk sampled | contributed | 4 chunks |
|---|---|---|---|
| `mask` | **62** latents | **55** | 222 latents / 753 frames |
| `keyframe` | 57 latents | 50 | 207 latents / 702 frames |

`mask` **prepends** the carry on top of a full-size target, so it samples a longer
chunk and delivers 55 of the 57 new latents you asked for. `keyframe` samples the
size you asked for, but its first 7 latents reproduce the guide, so only 50 are new.

So **`mask` yields more per chunk**, at the cost of a longer and therefore more
expensive chunk. The guide route is not "cheaper at the join" — that framing was
wrong. What it actually buys is exactness: a guide is re-injected every step and
never denoised, where a masked region is blended by the sampler. Which of those
produces better continuity is **untested** (§9).

Neither master matches the naive `chunks × chunk_frames`: 4 chunks of 192 frames is
768, and you get 753 or 702. If you are sizing a run to a target duration, count on
the contribution, not the chunk.

### `feather_latents` — `mask` only, video only

A linear ramp on the video mask over N latents after the carried region, easing from
preserved back to fully generating rather than stepping at the seam:

```python
vm[:, :, :k] = 1.0 - overlap_strength_video        # the carry
ramp = torch.linspace(1.0 - strength, 1.0, steps + 1)[1:]
vm[:, :, k:end] = ramp                             # then back to free
```

`0` disables it. The **audio mask is never feathered** — hard edge either way.

Two things temper it. `mask_row_targets` binarises at **0.5**, so the ramp does not
grade the *timestep*; it just moves the preserve/generate boundary to wherever the
ramp crosses 0.5. What it does grade continuously is the sampler's latent blend. And
a latent is 1 or 4 frames, so N latents of feather is not N×4 frames — it depends
where the ramp falls in the 5-cycle.

Untested at any value.

---

## 4. Keyframes

`keyframes` (an IMAGE batch) + `keyframe_indices` + `vae`. One index per image,
comma separated, negatives counting from the end. Ported from LTXAVTools'
`optional_cond_image_indices`.

### The indices are GLOBAL across the master

Not per chunk. You place a shot where it belongs in the finished clip and the node
resolves which chunk owns it and what the local frame is — the same choice LTXAV's
`_calculate_keyframe_per_tile_indices` makes. Only the arithmetic differs, since
H3's frames-per-latent is `1,4,4,4,4` rather than a uniform scale.

This works because chunk *i*'s local latent 0 sits at master latent `cum_i − trim`,
which is a multiple of 5 in **both** carry modes — `(5a+2)−(5m+2) = 5(a−m)` — so
every chunk stays on phase 0 and `frame_at_latent` is valid on the origins. For a
57-latent template with a 7-latent carry: origins `[0, 50, 100, 150]`.

### Which chunk owns a frame

Consecutive chunks overlap, so a global frame can fall inside two of them. A frame
inside a chunk's carried **head** is trimmed at the join, so anchoring it there
paints a frame nobody sees. Each index goes to the chunk that actually **renders**
it.

Worked example — four 192-frame chunks, 7-latent carry (22 frames):

> Global frame **351** is covered by chunk 1 (spans 170–361) and chunk 2 (spans
> 340–531). In chunk 2 it would be local frame 11, inside the 22-frame head. So it
> goes to **chunk 1, local frame 181**.

The report prints every placement: `keyframe frame 351 -> chunk 1 local frame 181`.
Read it. It is the only way to see that an index landed where you meant.

### What raises rather than being silently absorbed

- an index past the end of the master, or before its start
- a count mismatch between images and indices — they are **zipped**, so a short
  list would silently drop keyframes
- `keyframes` without a `vae`, or `keyframe_indices` without `keyframes`
- any keyframe at all when **#15439** is not applied

Negatives are resolved here rather than passed through: `PackedLayout` takes a
negative literally, so `cond_t` would fall **below `text_len`**, into the text token
positions.

Images are encoded **once**, not per chunk. Guides are independent of `carry` —
they work with the masked route too — and a chunk's carry guide plus its user
keyframes go on in a single `conditioning_set_values`, since a second call would
replace rather than merge.

### Planning them: `MMH3KeyframePlanner`

Rather than working the indices out by hand, the planner emits an **end-anchored**
set from the same schedule the sampler uses:

```
4 chunks, 57 latents, carry 7, keyframe  ->  0, 191, 361, 531, -1   (5 images)
```

Frame 0 opens the clip; every later index is a chunk's **own last frame**; the final
one is `-1`. So each chunk generates *toward* its destination image, and the next
continues from the arrived state through the ordinary carry. Start-anchoring instead
would put each image in the NEXT chunk and invite a snap at every seam — LTXAVTools'
reasoning, and it holds here.

Under the ownership rule that lands exactly one keyframe per chunk, with chunk 0
taking two (its opening and its end). `count` is how many images the batch needs.

`scene_frames` overrides with explicit lengths when scenes do not coincide with
chunks. `carry` matters: it changes chunk lengths and the trim, so it changes where
every chunk ends — `mask` gives `0, 191, 378, 565, -1` for the same request.

---

## 5. What the node protects you from

Each of these was a real failure, not a hypothetical.

**Stale guide bookkeeping.** `minimax_keyframes` / `minimax_frame_count` are
stripped off incoming conditioning every chunk, in both modes. This node registers
all its own guides; anything arriving pre-registered came from an upstream guide
node or a cond cached from a previous run, and would anchor the chunk to somebody
else's frames. Straight from LTXAVTools, where the same leak had the same cause.

**The shallow-copy guider bug.** `copy.copy` shares `original_conds`, and
`set_conds` assigns into it — so chunk 0 would overwrite the BASE conditioning and
every later chunk would read chunk 0's conds back as "base". The dict is rebound per
chunk. In LTXAVTools the symptom was every chunk getting chunk 0's speaker.

**Identical noise.** Reusing one noise object gives every chunk the same noise,
which reads as the model refusing to advance. The seed is bumped per chunk; chunk 0
keeps the seed you wired.

**Template mutation.** The latent is cloned per chunk, including its masks.
`NestedTensor` has no `.clone()`, only `.unbind()`, so an AV pair has to be taken
apart and rebuilt — a plain `.clone()` would throw.

**Guides landing before the clip.** See below.

---

## 6. Core changes this depends on

| PR | needed for | if missing |
|---|---|---|
| **#15375** | `carry="mask"` | `MMH3SeedOverlap` refuses |
| **#15439** | `carry="keyframe"`, any `keyframes` | the node refuses up front |
| *local correction* | guides **alongside a reference** | the node refuses that chunk |

The third is ours, not upstream. #15439 anchors `cond_t` on `text_len`, but the
target begins at `cursor`, which the refs advance. Measured on the real
`PackedLayout`, guide versus target origin:

| refs attached | drift |
|---|---|
| none | 0 |
| one image ref | **−1** |
| audio / voice ref | **−320** |
| image + audio | **−321** |

Nothing errors — the guide just anchors into the reference region instead of the
clip, and `cond_audio` goes with it, so a carried tail's **audio** lands early too.
It bites precisely the configuration #15439 exists to enable, since the same PR
fixes the `cond_video_latents` clobber so guides and refs can coexist.

`_guide_origin_correct()` probes for the correction by building a layout with one
ref plus one guide and comparing the origins. `carry="keyframe"` refuses only when a
chunk carries **both** a reference and a guide — guides alone are correct on stock
#15439. See [`core-changes.md`](core-changes.md).

---

## 7. Reading the report

```
4 chunks of 57 video latents (192 frames), 320 audio latents
  keyframe frame 351 -> chunk 1 local frame 181
  chunk 0: prompt 0, 0 carried frames, 1 keyframe(s)
  chunk 1: prompt 1, 22 carried frames, 1 keyframe(s)
  chunk 2: prompt 2, 22 carried frames
  chunk 3: prompt 3, 22 carried frames
master: 207 video latents (702 frames), 1170 audio latents
```

- **`prompt N`** climbing 0,1,2,3 means the cond_set is advancing. A repeated number
  means you have fewer prompts than chunks.
- **`0 carried frames` on chunk 0 only.** Anywhere else means the carry failed.
- **`master:`** — audio should match the video duration. If it does not, something
  upstream of `ConcatAV` is wrong, not the sampler.
- A **grid-safe trim** line appears under `mask` and says how many frames it took.
  Under `keyframe` it should be absent.

---

## 8. Symptom → lever

| Symptom | Look at |
|---|---|
| every chunk looks like chunk 0 | noise seed not advancing; check the report's prompt numbers too |
| every chunk uses the same prompt | fewer prompts than chunks — the report says so |
| seam visible / discontinuous motion | raise `overlap_latents`; try `carry="keyframe"` |
| lipsync drifts across a seam | `overlap_strength_audio` to 1.0; check master audio matches video in the report |
| ~7 frames missing per seam | that is `mask`'s grid-safe trim. Use `carry="keyframe"` |
| a keyframe lands in the wrong place | read the placement lines; indices are GLOBAL, not per-chunk |
| keyframe seems ignored | it may have landed in a trimmed head — the report says which chunk took it |
| node refuses on chunk 1 with a reference | the post-ref origin correction; see §6 |
| audio shorter than video in the master | `ConcatAV` audio drop — fixed in 0.39.0, check your version |

---

## 9. Not yet measured

Everything here is honest about being unknown. None of it has been generated.

- **Which `overlap_latents` is enough.** The trade is context versus waste, and the
  waste is exact (§3) while the context is not.
- **Whether `keyframe` actually beats `mask` in output quality.** It is cheaper at
  the join, which is arithmetic. Whether a guide holds continuity as well as a
  masked carry is an empirical question nobody has answered.
- **`overlap_strength_video` / `_audio` below 1.0.** Per-row masking binarises at
  0.5 for TIMESTEP purposes, so partial strength only blends the latent
  continuously — see `core-changes.md`. What that looks like is untested.
- **`feather_latents`.** Untested at any value.
- **How many chunks before drift compounds.** Other packs report photocopy-style
  degradation over chained audio; whether the masked carry avoids that is unknown.
- **Whether guides at interior indices behave** as #15439's author intends — it is a
  draft PR he has flagged as not fully tested.
