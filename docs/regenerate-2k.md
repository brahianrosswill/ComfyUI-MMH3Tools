# Regenerate-2K — Field Guide

**Status: the documented method works; the extension past it does not yet.**

A **single unchunked pass produces a correct 2K result** — including at 362 frames,
the official ceiling. That is the configuration MiniMax describes, and reproducing it
locally works.

**Chunking past one window diverges** (§6). Chunking is not part of the documented
method; it is what this pack adds to go beyond 362 frames.

Everything describing *structure* below is measured on the real tensors or the real
schedule, or quoted from MiniMax. Quality claims are confined to the two statements
above.

**This is a reproduction of a documented method, up to a point.** §1 quotes the model
card and the `/v2/video_regeneration` API for every design decision the nodes make.
Where the pack goes beyond what MiniMax documents — past the 362-frame ceiling — it is
said so plainly rather than implied.

---

## 1. What this is copying

MiniMax ships H3 as three modules. Only two are open:

| module | status | what it does |
|---|---|---|
| **H3-Context-IR** | hosted only | expands a rough idea into the structured prompt H3 consumes |
| **H3-Base** | open | generates audio and video at **768p** |
| **H3-Regenerate-2K** | **not released** | feeds the 768p result *plus the original context* back into H3 to regenerate at 2K |

### It is in-context regeneration, not super-resolution

[`MiniMax-AI/MiniMax-H3`](https://github.com/MiniMax-AI/MiniMax-H3), model card,
*H3-Regenerate-2K*:

> For H3's 2K-resolution output, **instead of using a conventional dedicated
> super-resolution module, we use the H3 base model to regenerate its own
> low-resolution result through an in-context manner.**
>
> This approach provides two advantages: (1) the regeneration process can reuse the
> generative capabilities of H3 base model to the greatest extent possible; and (2)
> **the in-context format can reuse the original multimodal context** when producing
> high-resolution output, allowing it to recover information that conventional
> super-resolution methods would otherwise have to "guess", such as small text and
> fine details.
>
> In-context regeneration is also an example of task generalization.

And the overview:

> H3-Regenerate-2K: Feeds the 768p result together with the original context back into
> H3 to regenerate the output at 2K resolution.

"In-context" is the operative word, and it is why this pack implements the 2K pass as
**references on the conditioning** rather than as anything resembling an upscaler. H3
has no cross-attention; in-context means the 768p rows are packed into the sequence and
attended directly. That is what `minimax_refs` is.

### It re-runs a generation; it does not upscale a video

This is the sentence that decides how to read everything else
([`/v2/video_regeneration`](https://platform.minimax.io/docs/api-reference/video-generation-v2-regeneration)):

> This endpoint only regenerates videos that meet the MiniMax-H3 768P output
> specifications to produce 2K output. **It does not perform general-purpose
> processing of arbitrary videos.**

That is not a note about input formats. The API's own structure shows what it means.

The **`source_task_id`** route accepts the id of a previously succeeded generation —
whitelist-gated, and the task must be owned by the calling account and still queryable
within 7 days. If the endpoint merely needed a spec-compliant *file*, a task id would
be a pointless convenience; you would upload the video. It is there because the
endpoint needs something the file does not contain.

The **`content`** route says what that something is: the exact original inputs,
including the **final** prompt. A format constraint would care only about the video.
Requiring the expanded prompt only makes sense if the regeneration is *conditioned* on
it.

So the correct mental model is **re-running the original generation at 2K, with the
768p result as an additional in-context anchor** — not upscaling a clip with the
model's help. The base video is one input among the original set, not the subject.
That is what the model card means by "in-context regeneration is also an example of
task generalization": same task, more inputs, higher resolution.

Three consequences:

**You must possess the generation context.** Not an approximation of it — the actual
final prompt and the actual references. This pack satisfies that trivially because it
generated stage 1 itself: `stage1_cond_set` *is* the final conditioning, not a
re-encode of it.

**It cannot be used on footage you did not generate with H3.** No amount of resizing
someone else's clip to 768p, 24 fps and a /32 canvas makes it eligible, because the
conditioning does not exist. "A 2K upscaler for H3" invites exactly this misuse; it is
not one.

**The 362-frame ceiling is not a property of regeneration.** If regeneration is a
generation, the limit is H3's own single-pass sequence budget showing through.

### The API spells out the inputs

> Regenerate a source video that meets the MiniMax-H3 768P output specifications into a
> 2K video.

Supplying the source by content requires:

> The **exact same inputs used for original 768P generation** (text prompt, reference
> images/videos/audio)
>
> Exactly one video item with `type=video_url` and `role=base_video`

and, on the text:

> The text must be **the final prompt actually sent to the model when generating the
> 768P source video, not the original prompt.**

That settles three design questions rather than leaving them to inference:

| the API says | so this pack |
|---|---|
| the *exact same inputs* as the 768p pass | takes stage 1's own `cond_set` — nothing is re-encoded |
| the **final** prompt, not the original | reuses stage 1's *encoded* conditioning, which is the final prompt by construction |
| the 768p enters as one item alongside the original references | appends the 768p as a `minimax_refs` block next to whatever stage 1 already carried |

MiniMax's own script agrees, exporting one prompt for both passes:

```bash
# Export the complete expanded prompt for H3-Base and regeneration.
EXPANDED_PROMPT=$(echo "$context_ir_result" | jq -er '.task.content.prompt')
```

### A third source: ComfyUI's own API node

[Comfy-Org/ComfyUI#15471](https://github.com/Comfy-Org/ComfyUI/pull/15471) adds
`MinimaxHailuo03ContextIRNode` and `MinimaxHailuo03RegenerateNode` — official API
nodes for the two hosted modules. (Hailuo 03 is H3; `MinimaxHailuo03ReferenceNode`
has been in master for a while.) They are a wrapper around the same endpoint, but an
independent implementation of it, and they corroborate every constraint derived here.

`MinimaxHailuo03RegenerateNode`'s **required** inputs:

| input | note |
|---|---|
| `video` | *"The MiniMax H3 768P output video to re-render."* → `role="base_video"` |
| `prompt` | *"The exact prompt used to generate the source video."* |
| `resolution` | one option: `2K` |

Optional: `reference_images` / `_videos` / `_audios` (9 / 3 / 3 — the originals),
`first_frame`, `last_frame`, `watermark`.

Its source-video validation:

> FPS strictly **23.9–24.1** · dimensions **divisible by 32**, max **1,032,192**
> pixels · **"107 to 362 frames in steps of 17 (4 to 15 seconds at 24 FPS)"**

`1,032,192 = 768 x 1344` — the same `MAX_PIXELS` this pack reads out of
`adapt_canvas`. 107–362 in steps of 17 is the `17j+5` grid. An independent
implementation arriving at identical numbers is the strongest confirmation available
that §3's dimension rules are right.

Two things it adds that the prose API docs did not:

**The role vocabulary is richer than one extra role.** The content list uses
`base_video`, `first_frame`, `last_frame` and `reference_image`, while reference
videos and audios carry **no role at all**. So the hosted layout distinguishes at
least four positions, not the two ("base" vs "reference") assumed above. That widens
rather than narrows the unknown in §1: whatever `base_video` means to the hosted
module, it sits in a vocabulary this pack cannot express, since the open layout has
`image`, `audio`, `video`, `video_audio` and nothing role-like.

**`prompt` is a required input, described as the exact original.** Not a convenience,
not optional. That is now three independent sources — the model card, the API
reference, and a node signature — saying the 2K pass is conditioned on stage 1's
final prompt. It is the single design decision this pack can be most confident about.

Also worth noting: the node implements **only** the `content` route. There is no
`source_task_id` input, so even Comfy's official integration requires handing over the
exact original inputs; the task-id shortcut is whitelist-gated and they skipped it.

### What a role actually does, and why latent-only is right here

A role is not decoration. It decides the **slot**, and the slot decides the **label the
prompt uses**. Comfy's Context-IR node says so directly: reference images are
*"referred to in the prompt as 'Image 1'..'Image 9'"*, videos as *'Video 1'..'Video
3'*, audios as *'Audio 1'..'Audio 3'*.

Locally that is the `minimax_ref_items` path. `comfy/text_encoders/minimax.py` keeps a
counter per kind and injects the label into the text stream:

```python
counters = {"image": 0, "audio": 0, "video": 0}
...
add_text("<Picture %d>: " % counters["image"])
```

So a reference has two halves: an entry in `ref_items`, which the TEXT ENCODER sees and
labels, and a block in `minimax_refs`, which the DiT attends. Only the first produces a
`<Video N>` the prompt can name.

**`base_video` is the role with no label.** The prompt handed to regeneration is the
*original* prompt — it refers to `<Video 1>`, `<Picture 1>` and so on, meaning the
original references. It never mentions the 768p, because when it was written the 768p
did not exist.

This pack appends the 768p to `minimax_refs` and adds **no `ref_items` entry**, so no
label is created and the text encoder never sees it. That was chosen to avoid a VAE
roundtrip on latents already in hand (§4), and it turns out to be the correct semantics
independently: an unlabelled block the DiT attends and the prompt does not name is
exactly what a base video is.

`nodes_refs.py` calls the missing tokenizer registration a KNOWN LIMITATION, and for an
ordinary reference it is one — you cannot write `<Video 1>` about something the encoder
never saw. For `base_video` it is not a limitation at all.

One caveat on that conclusion: it assumes the prompt reaches the model unchanged. If
the hosted pipeline rewrites the prompt to name the base video — plausible, since H3
carries task markers in text — then `base_video` would have a label after all, and it
would be one this pack never writes. See §7.

### Where the local reproduction stops being equivalent

Three hard boundaries, worth stating together:

**The base competes for a reference slot.** The hosted endpoint budgets it separately —
the base's duration is excluded from the 15-second reference-video cap and it does not
count toward the 3-video limit. This pack has no separate budget, because it expresses
the base *as* a reference. So a 768p made with 3 reference videos needs a 4th video
reference to regenerate, past what Ref2VA documents (≤3 videos, ≤15s total, ≤12 files).
**The hosted endpoint can regenerate that source and this pack cannot.** Below 3
reference videos there is room. A T2VA source has none, so it never arises.

**There is no way to tag the role.** The open layout has kinds — `image`, `audio`,
`video`, `video_audio` — and no field that says "this one is the base". Whatever the
hosted module does with that distinction is unavailable, not merely unimplemented.

**Local compute.** 2K regeneration is full sampling at 2K over the whole sequence, with
references attended at every step. The hosted module runs on hardware chosen for it;
here it is measured in tens of minutes per chunk.

### What the official pass will accept

The API's `base_video` specification, which doubles as a description of what
H3-Regenerate-2K can take:

| requirement | matches |
|---|---|
| **audio track present (mandatory)** | §5 — stage 1's audio is pinned into the target |
| 24 fps | `FPS = 24` |
| width and height divisible by 32 | `CANVAS_MULTIPLE = 32`, §3 |
| area ≤ 768 x 1,344 | `MAX_PIXELS`, the `adapt_canvas` cap in §3 |
| **107–362 frames (~4–15s, in 17-frame increments)** | the 17j+5 grid — and a hard ceiling, see below |

Two things worth naming as *not* verified rather than glossed:

**`role=base_video` is its own role**, distinct from `reference_video`. This pack
appends the 768p as an ordinary video reference block. Worth separating what is known
from what is assumed there:

*The checkpoint cannot be hiding a `base_video` pathway.* `adaln_proj.linear.weight` is
`[96768, 8]` and `96768 = 6 x 5376 x 3` — six modulation terms, hidden width, and
**three** modality rows. Three is structural; a fourth role wanting its own row would
need a differently shaped tensor. Nor is any other tensor indexed by reference role:
the non-block inventory is `adaln_t_table`, `audio_patch_proj`, `condition_proj`,
`final_layer.*`, `rope.inv_freq` and `token_refiner.*`, none of them keyed by kind. So
there is no learned parameter a new role could select.

*Which kinds occupy which row is ComfyUI's convention, not the checkpoint's.*
`seg_tag = {"video": 0, "cond": 0, "ref_img": 0, "text": 1, "audio": 2, ...}` lives in
`comfy/ldm/minimax/model.py`. The weights say three rows exist; they do not say what
belongs in each.

So a role distinction can only be expressed through **layout** — which row a segment
uses, where it sits in the packed sequence, what position ids it gets. That is code,
not weights, and it is precisely the part MiniMax did not publish for regeneration.
The weights rule nothing in; they only rule out the checkpoint as the hiding place.

*And the vocabulary is wider than two.* Comfy's API node (below) builds its content
list with `base_video`, `first_frame`, `last_frame` and `reference_image`, while
reference videos and audios carry no role at all. The open layout has `image`,
`audio`, `video`, `video_audio` — kinds, not roles, and no way to say "this video is
the base one." So the gap is not a single missing flag; it is a different way of
labelling the sequence, entirely in code we do not have.

**362 frames is the official ceiling.** The API will not accept a longer source, and
per the section above that is H3's own single-pass budget rather than a rule about
regeneration. So everything this pack does past 362 frames — chunking the 2K pass — is
an extension beyond the documented method, not a reproduction of it.

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

**The same clip in a single pass is correct.** 362 frames, one window, no carry —
which is exactly the configuration the documented method uses, and the clip is exactly
at the official ceiling (§1). So the variable is not the length, the reference
slicing, the conditioning or the model: it is **chunking itself**.

That narrows §6 to a genuine boundary. Up to 362 frames the faithful configuration is
one chunk and it works. Chunking is only forced past that length, and it is the part
this pack adds rather than reproduces.

Ruled out — the single-pass result eliminates most of the field, and the run's own
logs eliminate the rest:

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

- **Whether the hosted pipeline MODIFIES the prompt before sampling.** The strongest
  hypothesis about what the unreleased module adds, and this pack does nothing of the
  kind — it passes stage 1's conditioning through untouched.

  The reasoning: H3 carries task markers *inside the prompt*. The summary line takes a
  bracketed prefix — `MMH3TaskSystemPrompt` emits `[audio reuse + audio reference]`,
  and MiniMax's guides devote a section to choosing them. So an established mechanism
  exists for telling the model what kind of job this is, in text.

  If regeneration is a task in that sense, the pipeline would prepend or rewrite that
  marker, and possibly add a line naming the base video — which the original prompt
  cannot do, since the 768p did not exist when it was written. That also reframes the
  API's strictest requirement: *"the final prompt actually sent to the model… not the
  original prompt"* is exactly what you would demand if you intended to **inject into**
  a known structure. An arbitrary prompt could not be modified reliably.

  Against it: a single unchunked pass already produces a correct result with no such
  marker (§6), so whatever this would add is not load-bearing. And the marker's text is
  unknown — `[regeneration]` is a guess, not a finding. Absence from the prompt-writing
  guides proves nothing either way, because those guides describe what Context-IR emits
  for *generation*.

- **Whether `overlap_strength_video = 0` fixes §6.** The leading hypothesis for the
  chunking divergence, and it only matters for clips that genuinely exceed 362 frames,
  since anything shorter should be run in a single pass anyway.
- **Whether seeding the latent and attaching references together beats either alone.**
- **Which `ref_downscale` is affordable.** 2x cuts reference cost ~4x; what it costs in
  fidelity at 2K is unknown.
