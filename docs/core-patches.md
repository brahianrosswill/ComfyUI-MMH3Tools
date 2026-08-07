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

> **Patch 2 has a runtime equivalent on the `keyframe-anchors` branch.**
> `mmh3tools/patch_layout.py` there wraps `PackedLayout.__init__` and recomputes
> cond-row positions *absolutely*, so the ref-cursor offset comes out right whether
> or not core is edited, and it survives `git pull`. It is idempotent with the file
> edit — it replaces rather than adjusts — so having both applied is harmless.
>
> On `main` all four patches still need the diff. Patch 1 could be wrapped the same
> way (`MiniMaxH3.extra_conds` is a plain method), but 3–4 cannot: their call sites
> are inside a closure and inside `_forward`, with no callable boundary to bind to,
> and replacing the enclosing method would mean vendoring GPL-3.0 code into an MIT
> pack. They are drozbay's anyway, open upstream as **#15375** — if that merges,
> delete rather than convert.

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

## Interior keyframe anchors — on the `keyframe-anchors` branch

`PackedLayout` raises `only first/last keyframe anchors are supported`. This was
listed as an optional one-line edit, "not included in the diff because it is
unverified here."

It is verified now, and it is **not** a core edit. `mmh3tools/patch_layout.py`,
on the `keyframe-anchors` branch, wraps the constructor at import. The coordinate

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

## Upstream churn to expect (checked 2026-08-06)

H3 core is under active development, and three of our four patch sites are in it.
Re-read this before pulling.

| PR | state | touches | impact on us |
|---|---|---|---|
| [#15322](https://github.com/comfyanonymous/ComfyUI/pull/15322) | **merged 2026-08-06** | `model_base.py` — deletes H3's no-op `scale_latent_inpaint()` override so masked sampling uses `BaseModel`'s | Same function patch 3 modifies. **Expect a conflict or a silent behaviour change under our per-row code.** Re-run `tests/test_concat_av.py` after pulling. |
| [#15243](https://github.com/comfyanonymous/ComfyUI/pull/15243) | open (draft, kijai) | `model_base.py`, `ldm/minimax/model.py`, `samplers.py`, `model_sampling.py` — `ModelSamplingAV`, `FLOW_AV`, audio latent scaling, `audio_shift` | Three of our four sites. Also **changes output at every step count**. drozbay is reviewing, so the per-row work and this are aware of each other. |
| [#15270](https://github.com/comfyanonymous/ComfyUI/pull/15270) | open, approved | `ldm/minimax/model.py` — exposes `set_model_attn1_patch` / `attn1_output_patch` with block-scoped metadata | Would let block-level attention patching happen from a **custom node instead of core**. Does not cover AdaLN modulation, so it does not replace patch 3, but it is the right direction. |
| [#15316](https://github.com/comfyanonymous/ComfyUI/pull/15316) | open | `text_encoders/minimax.py` — reserves VRAM for image encoding | Not a patch site. Fixes 1+ min hangs when conditioning carries images. Workaround today is `--reserve-vram`. |

## Not yet patched: multimodal context windows

`comfy/context_windows.py` has a complete multimodal design, and H3 reaches it
automatically (`is_multimodal = len(latent_shapes) > 1`). But
`map_context_window_to_modalities` has **zero implementations** tree-wide, and the
windowing state uses one `dim` for every modality — which would window H3's audio
`[B,32,2,T40]` on its stereo axis. See `docs/context-windows.md`.
