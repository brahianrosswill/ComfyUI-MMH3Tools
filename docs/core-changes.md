# Core changes MMH3Tools relies on

Two different things, and the difference decides which branch a node lives on.

## Upstream PRs — `main` may depend on these

Somebody else's pending change, which will merge. Applying one is ordinary: fetch a
diff, apply it, and one day `git pull` makes it unnecessary. Nodes that need one live on
`main` and **refuse to run without it** rather than appearing to work.

| PR | needed by | why |
|---|---|---|
| **[#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375)** drozbay | `MMH3SeedOverlap`, latent outpaint | Per-row masking. Without it a noise mask has **no effect at all** — preserved rows still run at the generation timestep, so the model gets clean content labelled as noisy. |
| **[#15316](https://github.com/Comfy-Org/ComfyUI/pull/15316)** Haoming02 | nothing, but worth having | Reserves ~2 GB + 400 MB per RGB megapixel before the text encoder handles images. This is the minute-long hang when conditioning carries image references. |
| **[#15439](https://github.com/Comfy-Org/ComfyUI/pull/15439)** drozbay | `MMH3LoopingSampler`'s keyframe carry | `MiniMaxH3AddGuide`: guides at ANY frame index, a guide can be a multi-step clip rather than a still, and audio anchors at the same `cond_t`. **Draft, and the author says it is not fully tested.** |

```bash
cd C:/ComfyUI
for pr in 15375 15316 15439; do
  curl -sL "https://github.com/Comfy-Org/ComfyUI/pull/$pr.diff" -o /tmp/pr$pr.diff
  git apply --check /tmp/pr$pr.diff && git apply /tmp/pr$pr.diff
done
```

**Re-fetch rather than reusing a saved copy.** These get rebased, and a diff cut against
an older base is exactly how #15371 went wrong.

### #15439 needs one hunk hand-merged onto #15375

Four of its five `model.py` hunks apply clean, as do `model_base.py` and
`nodes_minimax_h3.py`. The `_forward` hunk conflicts, because **#15375 already rewrote
that region**: it deleted `has_aud_cond` and replaced

```python
unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set())
                  | ({seg_t["ref_audio"]} if has_aud_cond else set()))
```

with a version that derives the timestep set from `layout.segments` directly. So a new
segment kind is picked up automatically and most of #15439's hunk is already handled.
What is left is two dict entries, and they are **required, not cosmetic** — `unique_t`
indexes `seg_t[k]` for every segment kind present, so a `cond_audio` segment raises
`KeyError` without the first one:

```python
seg_t   = {..., "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
seg_tag = {..., "cond_audio": 2, "ref_audio": 2}
```

Apply with `git apply --reject`, then add those two by hand and delete the `.rej`.

### Two behaviours #15439 changes that are easy to miss

**A guide with no `latent` emits no rows at all.** Stock built `cond` rows from the index
alone; #15439 only emits when `kf.get("latent")` is not None. A self-test that passes
bare `{"resolved_frame_index": p}` dicts and reads `position_ids[text_len]` now measures
the target's first row instead, and passes vacuously.

**Negative indices are taken literally.** `PackedLayout` does not resolve them —
`p = -1` gives `cond_t = text_len - FRAME_RESCALE`, *below* `text_len`, colliding with
text token positions. `MiniMaxH3AddGuide` resolves them upstream; anything building
keyframe dicts directly has to do the same.

## The post-ref guide origin: a wrap, not a core edit

**No PR carries this. This pack does**, as `mmh3tools/patch_guide_origin.py`, applied
at import. Core is NOT edited for it -- a core edit would be a diff to re-apply after
every `git pull` and to remember when reading a bug report from someone who lacks it.

`cond_t = float(text_len) + FRAME_RESCALE * resolved_frame_index` anchors to
`text_len`. The target begins at `cursor`, which the refs advance. Measured on the
real `PackedLayout`, guide origin versus target origin:

| refs attached | stock #15439 | with the wrap |
|---|---|---|
| none | 0 | 0 |
| one image ref | **-1** | 0 |
| audio / voice ref | **-320** | 0 |
| `video_audio` ref | **-37** | 0 |
| image + audio | **-321** | 0 |

Nothing errors. The guide anchors into the reference region, and `cond_audio` goes
with it -- so a carried tail's **audio** lands early too. It matters *more* under
#15439, not less, because the same PR fixes the `cond_video_latents` clobber
**specifically so guides and refs can coexist**, which makes the broken configuration
reachable.

The wrap lets stock build the layout, then shifts the `cond` and `cond_audio` rows by
the advance:

```python
for a, b, kind in self.segments:
    if kind in ("cond", "cond_audio"):
        self.position_ids[a:b, 0] += advance
```

Uniform addition rather than per-row assignment, so whatever intra-block structure
stock built survives. `_ref_cursor_advance` mirrors the `if refs:` cursor arithmetic;
a test compares it against the target origin the layout actually produced, because a
drift between the two is a silently misplaced guide -- the exact failure this removes.

**On `main`, deliberately.** The pack's rule sends monkeypatches to `keyframe-anchors`,
but `main`'s looping sampler needs this and no upstream PR exists to wait for. It comes
out the moment one lands; `apply()` already detects a core that has its own fix and
declines.

Inert unless BOTH guides and refs are present, self-tested at import against the live
class, and rolls back rather than misplacing a guide. Reported upstream on #15439.

## The context-window VRAM estimate: a wrapper, not a patch

**No PR carries this either.** `MMH3ContextWindowVRAM` works around it through a
supported extension point, so unlike the guide-origin wrap there is no core behaviour
being overridden — core simply asks a question with the wrong shape and this substitutes
the right one.

`_prepare_sampling` calls `estimate_memory(model, noise_shape, conds)` and passes the
result to `load_models_gpu`. `noise_shape` is the **full** latent;
`BaseModel.memory_required` reduces it to `batch * prod(shape[2:])`, so the reservation
grows linearly with clip length. Context windows are built later, inside the sampler,
long after the model has been loaded — they never reach the estimator. Measured at 2K
1536×2688 with a 47-latent window and H3's `memory_usage_factor` 0.114:

| clip | latents | stock estimate | actual need | over-reservation |
|---|---|---|---|---|
| 40s | 282 | 10.9 GB | 1.81 GB | 6x |
| 120s | 847 | **32.7 GB** | 1.81 GB | **18x** |

The failure is not an OOM. `load_models_gpu` is told to leave 32.7 GB free on a 32 GB
card, concludes the DiT cannot be resident, and offloads weights to RAM — so the job
runs, slowly, and every sampler setting looks innocent. The tell is that **window size
changes nothing while total length changes everything**, which is backwards for a
windowed sample.

This is not H3-specific. `memory_required` is on `BaseModel` and `estimate_memory` is
model-agnostic, so every windowed model has it; H3 only surfaces it early because a 33B
DiT leaves little headroom to absorb the error. Worth reporting upstream — the fix
belongs in core, where the window length is known.

`WrappersMP.PREPARE_SAMPLING` is a documented hook (`comfy/sampler_helpers.py` routes
through `WrapperExecutor` precisely so it can be wrapped), so the node clamps
`shape[2]` and calls the executor on. The substituted shape is consumed by
`estimate_memory` and discarded; nothing downstream sees it. `NestedTensor.shape`
returns the video tensor's shape, so an AV latent arrives as `[B,24,T,h,w]` and the
audio half never reaches the estimator at all.

**The hazard is real and one-directional.** With windowing off, the stock estimate is
correct and clamping under-reserves — an offload traded for an OOM. Windowing cannot be
detected from inside the wrapper, so the node is opt-in, defaults to `enabled` only once
wired, and its `context_length` has to be kept equal to the windowing node's by hand. A
mismatch is silent.

## Monkeypatches — `keyframe-anchors` only

A wrap of core that **this pack maintains indefinitely**, because upstream has no plan to
change the thing it works around. That is what the branch is for.

| patch | what | status |
|---|---|---|
| `mmh3tools/patch_layout.py` | wraps `PackedLayout.__init__` for interior keyframe anchors | **superseded by #15439** |
| `mmh3tools/patch_conds.py` | wraps `MiniMaxH3.extra_conds` so keyframes and references coexist | **superseded by #15439** |

Both are absolute rebuilds, inert unless used, and self-tested at import — they refuse to
install rather than corrupt output. `MMH3LatentKeyframe` depends on them, so it lives
there too.

**#15439 does both of these upstream**, which is the outcome the branch existed to reach:
it deletes the first/last `raise` outright, and it fixes the `cond_video_latents`
overwrite by concatenating keyframes-then-refs — the same order, for the same reason.

The patches detect this and decline: `patch_layout` searches for the
`only first/last keyframe anchors are supported` text, which #15439 removes, so it finds
no anchor and leaves stock alone. Nothing breaks; the branch simply has no work left.
Retire it once #15439 merges rather than while it is still a draft that could be
withdrawn.

## Deliberately not applied

| PR | why not |
|---|---|
| #15270 pyros-projects | H3 attention patch hooks. Nothing here uses them, and it touches the same file as #15375, so it is pure conflict surface for no current gain. |
| #15353 xiaolibai-sys | 650 lines of pruned-LoRA support, unused here. |
| **#15371** Deno2026 | **Applied, then reverted — it breaks audio encode.** `disable_offload = True` on the audio VAE swaps `CoreModelPatcher` for plain `ModelPatcher`, flipping `assign=self.patcher.is_dynamic()` to False; the weights then load float32 while the encode path still feeds half. It is a competing fix for something **#15377 already solved upstream** using `comfy.ops.cast_to_input`. The lesson: check whether an open PR has been superseded by a merged one before applying it. |

## A note on the mask being binary

`mask_row_targets` (from #15375) reduces a `[T,H,W]` mask to one bool per 2×2 patch:

```python
target  = m.reshape(-1) >= 0.5          # bool
video_w = targets.to(torch.float32)     # 0.0 or 1.0, never between
```

The AdaLN lerp it feeds is genuinely continuous, so this is a **choice in an open PR**,
not a property of H3. Consequences: partial `overlap_strength` blends the *latent*
continuously (via the sampler's own `x*mask + orig*(1-mask)`) but the *timestep*
conditioning is all-or-nothing, and a feathered spatial mask hardens at the 0.5 contour.
Re-check if #15375 changes before merge.

## Reverting and updating

Revert with `git checkout -- <files>`, not the `.pre-*` backups — git HEAD is the
authoritative stock, and the backups differ from it by a BOM.

Always check whether upstream touches the same files before pulling:

```bash
cd C:/ComfyUI
curl -sL "https://api.github.com/repos/Comfy-Org/ComfyUI/compare/$(git rev-parse HEAD)...master" \
  | python -c "import sys,json;[print(f['filename']) for f in json.load(sys.stdin)['files']]"
```

If none of `comfy/model_base.py`, `comfy/ldm/minimax/model.py`, `comfy/samplers.py` or
`comfy/text_encoders/minimax.py` appear, pull straight over the top. If any do, revert
first, pull, then re-fetch and re-apply the PRs.
