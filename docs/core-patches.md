# ComfyUI core patches

**Nothing here is maintained as a local diff any more.** Patches 1-2 became runtime
wraps, and patches 3-4 exist upstream as a pull request. What is left is a record of
what the four were and why, plus the one PR still worth applying by hand.

## Current state (ComfyUI @ 0db86941, 2026-08-07)

| was | now |
|---|---|
| 1 - `cond_video_latents` must accumulate | **runtime wrap**, `mmh3tools/patch_conds.py` |
| 2 - keyframe position must clear the refs | **runtime wrap**, `mmh3tools/patch_layout.py` |
| 3 - per-row masking | **PR [#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375)** (drozbay) |
| 4 - `samplers.py` one line | same PR |

Patches 1 and 2 are wrappable because `MiniMaxH3.extra_conds` and
`PackedLayout.__init__` are whole callables -- the wrap calls the original and repairs
its output, copying nothing. Both rebuild absolutely, so a file edit applied as well is
harmless. Patches 3-4 are not wrappable: their call sites sit inside a CLOSURE
(`mod(seg)`) and inside `_forward`, and monkeypatching binds to names.

## Applying the open PRs

Re-fetch rather than keeping stale copies -- these get rebased, and a diff cut against
an older base is how the last round went wrong:

```bash
cd C:/ComfyUI
for pr in 15375 15316; do
  curl -sL "https://github.com/Comfy-Org/ComfyUI/pull/$pr.diff" -o /tmp/pr$pr.diff
  git apply --check /tmp/pr$pr.diff && git apply /tmp/pr$pr.diff
done
```

| PR | why |
|---|---|
| **#15375** drozbay | Per-row masking. The ONLY thing `MMH3SeedOverlap` needs - without it the node refuses to run, because the mask would have no effect at all. |
| **#15316** Haoming02 | Reserves ~2 GB + 400 MB per RGB megapixel before the text encoder handles images. This is the 1-minute hang when conditioning carries image references, which `max` sizing makes worse. |


Deliberately NOT applied:

| PR | why not |
|---|---|
| **#15371** Deno2026 | **Applied, then reverted - it breaks audio encode.** It sets `disable_offload = True` on the audio VAE, which swaps `CoreModelPatcher` for plain `ModelPatcher`, flipping `assign=self.patcher.is_dynamic()` to False. The weights then load float32 while the encode path still feeds half: *"mat1 and mat2 must have the same dtype, but got Half and Float"* in `audio_vae.py` `pre_block`. It is a competing fix for a problem **#15377 already solved upstream**, using `comfy.ops.cast_to_input` in that same function so raw buffers follow the input's dtype. Check whether an open PR has been superseded by a merged one before applying it - that is the lesson, and it cost a crash. |
| #15270 pyros-projects | Exposes H3 attention patch hooks. Nothing here uses them, and it touches `ldm/minimax/model.py` - the same file as #15375 - so it adds conflict surface for no current gain. Worth revisiting if block-level attention patching ever replaces per-row masking. |
| #15353 xiaolibai-sys | 650 lines of H3 pruned-LoRA support. Not used here. |

**Reverting is `git checkout -- <files>`**, not the `.pre-*` backups. git HEAD is the
authoritative stock; the backups differ from it by a BOM. Always revert BEFORE pulling
-- upstream touches these same files, and pulling onto local modifications is what
creates conflicts.

## 1. `comfy/model_base.py` — `cond_video_latents` must accumulate

Stock code assigns `cond_video_latents` from keyframes, then **assigns it again**
from references. The second assignment wins, so any reference silently erases
every keyframe. This is why "the keyframe is ignored on ref2va" — it was never
reaching the model.

```python
# MMH3Tools local patch: cond_video_latents must ACCUMULATE
cond_video_latents = []
if keyframes is not None:
    cond_video_latents += [kf["latent"] for kf in keyframes]
if refs is not None:
    cond_video_latents += [r["latent"] for r in refs if "latent" in r]
if cond_video_latents:
    payload["cond_video_latents"] = cond_video_latents
```

## 2. `comfy/ldm/minimax/model.py` — keyframe position must clear the refs

The keyframe's positional base is computed as `float(text_len)`, i.e. "right
after the text". But reference blocks are packed **before** the target and
advance a cursor as they go. With refs present the keyframe therefore lands on
rows the references already occupy, and the anchor is applied to the wrong
place in the sequence.

The patch replays the same cursor advance the packer does, and uses the result:

```python
kf_base = float(text_len)
if refs:
    _c = float(text_len)
    for _blk in refs:
        ...  # replicate the packer's cursor advance
    kf_base = _c
# then: cond_t = kf_base   (was: cond_t = float(text_len))
```

Patches 1 and 2 together are what make `[video continuation + keyframe
completion]` actually work.

## 3. `comfy/ldm/minimax/model.py` — per-row masking

Merged from [drozbay's `minimax-h3-per-row-masking`
branch](https://github.com/kijai/ComfyUI/compare/minimax_fixes...drozbay:ComfyUI:minimax-h3-per-row-masking).
Adds `mask_row_targets()` and `_mod_row()`, which lerps between two AdaLN
modulation vectors per token, so a masked row can be held at *partial* strength
instead of fully clamped or fully free. Without it, `overlap_strength` is
effectively boolean.

## 4. `comfy/samplers.py` — one line

Hands the denoise mask to the model so patch 3 can see it:

```python
denoise_masks = self.model_patcher.model.process_denoise_mask(denoise_masks)
```

## Interior keyframe anchors — solved at runtime, not here

`PackedLayout` raises `only first/last keyframe anchors are supported`. This was
listed as an optional one-line edit, "not included in the diff because it is
unverified here."

It is verified now, and it is **not** a core edit. `mmh3tools/patch_layout.py`
wraps the constructor at import. The coordinate

```
cond_t = kf_base + FRAME_RESCALE * pixel_index
```

is linear in pixel frames even though `FRAME_PER_TOKEN` makes the latent grid
non-uniform, because a step's span is exactly `FRAME_RESCALE` times the frames it
covers. Every intermediate index is representable; stock simply never computes one.
The endpoints reuse stock's own expressions rather than the general formula — they
are mathematically identical but differ in the last bits (~7e-15), and matching
exactly is what lets the self-test demand bit-identity.

Two anchors can pin a pose. They cannot express a trajectory, which is what
`MMH3LatentToKeyframes` needs: a chunk's tail is a *run* of consecutive frames.

## Backups

Pre-patch copies exist as `*.pre-mmh3-keyframe-patch` (patches 1–2) and
`*.pre-perrow` (patches 3–4) next to each file.

## Upstream churn (re-checked 2026-08-07, ComfyUI @ 0db86941)

Most of the table that used to live here has resolved. What happened:

| PR | state | effect on us |
|---|---|---|
| [#15243](https://github.com/Comfy-Org/ComfyUI/pull/15243) kijai | **MERGED** | Touched 3 of our 4 patch sites - `ModelSamplingAV`, `FLOW_AV`, audio latent scaling. Our wraps survive it: nothing in it modifies `cond_video_latents`, `PackedLayout`, `_video_t_spans` or `FRAME_RESCALE`. **It changes H3 sampling output at every step count**, so re-baseline workflows. |
| [#15322](https://github.com/Comfy-Org/ComfyUI/pull/15322) | **MERGED** | H3 latent noise mask sampling fix. |
| [#15390](https://github.com/Comfy-Org/ComfyUI/pull/15390) kijai | **MERGED** | H3 audio corruption with EasyCache. |
| [#15377](https://github.com/Comfy-Org/ComfyUI/pull/15377) | **MERGED** | Full offload on the audio VAE. |
| [#15268](https://github.com/Comfy-Org/ComfyUI/pull/15268) | **MERGED** | Cast raw params to input device in the H3 VAEs. |
| [#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375) drozbay | **open** | Per-row masking - patches 3-4. The only thing `MMH3SeedOverlap` is waiting on. |
| [#15270](https://github.com/Comfy-Org/ComfyUI/pull/15270) pyros-projects | open | Exposes H3 attention patch hooks. Does not cover AdaLN modulation, so it does not replace 3-4. |
| [#15316](https://github.com/Comfy-Org/ComfyUI/pull/15316) | open | Reserves VRAM for TE image encoding. Not a patch site. |

**Do not re-apply `core-patches.diff` to a tree newer than `14b05228`.** It was cut
against that base and #15243 rewrote those files. Re-cut from drozbay's branch instead,
or wait for #15375.

## Not yet patched: multimodal context windows

`comfy/context_windows.py` has a complete multimodal design, and H3 reaches it
automatically (`is_multimodal = len(latent_shapes) > 1`). But
`map_context_window_to_modalities` has **zero implementations** tree-wide, and the
windowing state uses one `dim` for every modality — which would window H3's audio
`[B,32,2,T40]` on its stereo axis. See `docs/context-windows.md`.
