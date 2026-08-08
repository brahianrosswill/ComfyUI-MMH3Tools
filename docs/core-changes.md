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

```bash
cd C:/ComfyUI
for pr in 15375 15316; do
  curl -sL "https://github.com/Comfy-Org/ComfyUI/pull/$pr.diff" -o /tmp/pr$pr.diff
  git apply --check /tmp/pr$pr.diff && git apply /tmp/pr$pr.diff
done
```

**Re-fetch rather than reusing a saved copy.** These get rebased, and a diff cut against
an older base is exactly how #15371 went wrong.

## Monkeypatches — `keyframe-anchors` only

A wrap of core that **this pack maintains indefinitely**, because upstream has no plan to
change the thing it works around. That is what the branch is for.

| patch | what |
|---|---|
| `mmh3tools/patch_layout.py` | wraps `PackedLayout.__init__` for interior keyframe anchors |
| `mmh3tools/patch_conds.py` | wraps `MiniMaxH3.extra_conds` so keyframes and references coexist |

Both are absolute rebuilds, inert unless used, and self-tested at import — they refuse to
install rather than corrupt output. `MMH3LatentToKeyframes` depends on them, so it lives
there too.

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
