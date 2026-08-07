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

> **Patch 2 is now redundant.** As of 0.21.0 `mmh3tools/patch_layout.py` wraps
> `PackedLayout.__init__` at runtime and recomputes cond-row positions
> *absolutely*, so the ref-cursor offset is correct whether or not core is edited.
> The runtime patch survives `git pull`, self-tests against the live class, and
> refuses to apply rather than failing silently. It is also idempotent with the
> file edit — because it replaces rather than adjusts, having both applied is
> harmless. Keeping the file edit is optional; new installs should not bother.
>
> Patches 1, 3 and 4 still need the diff. Patch 1 has no runtime equivalent yet
> (the assembly happens in `extra_conds`, not a wrappable constructor), and 3–4
> are drozbay's, open upstream as **#15375** — if that merges, delete rather than
> convert them.

> ## Status: only patches 3-4 are still needed
>
> | patch | how it is handled now |
> |---|---|
> | 1 - `cond_video_latents` must accumulate | **runtime wrap**, `mmh3tools/patch_conds.py` |
> | 2 - keyframe position must clear the refs | **runtime wrap**, `mmh3tools/patch_layout.py` |
> | 3 - per-row masking | **still a file edit** |
> | 4 - `samplers.py` one line | **still a file edit** |
>
> Patches 1 and 2 are wrappable because `MiniMaxH3.extra_conds` and
> `PackedLayout.__init__` are whole callables -- the wrap calls the original and
> repairs its output, copying nothing. Both are absolute rebuilds, so having the file
> edit applied as well is harmless.
>
> Patches 3-4 are not. Their call sites sit inside a CLOSURE (`mod(seg)`) and inside
> `_forward`; monkeypatching binds to names, and neither has one. Reaching them would
> mean replacing the enclosing method, i.e. vendoring GPL-3.0 core into an MIT pack.
> They are drozbay's anyway, open upstream as **#15375**.
>
> `MMH3SeedOverlap` is the only node that needs them, and it now REFUSES to run when
> they are absent rather than appearing to work -- without per-row timestep handling
> the mask has no effect at all.

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
