# ComfyUI core patches

Three files in ComfyUI core need patching before H3 will accept a keyframe and a
reference **at the same time**, and before overlap strength is anything other
than on/off. A node pack cannot do this from the outside — the bugs are in the
conditioning assembly and in the DiT's positional arithmetic.

**These are lost on every `git pull` in `C:\ComfyUI`.** The full diff is checked
in beside this file as [`core-patches.diff`](core-patches.diff), taken against
ComfyUI `v0.30.0-1-g14b05228`. Reapply with:

```bash
cd C:/ComfyUI
git apply custom_nodes/ComfyUI-MMH3Tools/docs/core-patches.diff
```

If it rejects after an upstream change, the four patches below are small enough
to redo by hand.

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

## Optional: interior keyframe anchors

`PackedLayout` raises `only first/last keyframe anchors are supported`. MiniMax's
own guide lists interior anchors as valid, and community reports agree they work.
`MMH3ImageKeyframe` will emit an interior index and warn; relaxing that check is
a one-line edit, not included in the diff because it is unverified here.

## Backups

Pre-patch copies exist as `*.pre-mmh3-keyframe-patch` (patches 1–2) and
`*.pre-perrow` (patches 3–4) next to each file.
