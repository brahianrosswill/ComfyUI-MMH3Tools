# Multimodal context windows for H3

What `comfy/context_windows.py` would need to support MiniMax H3, read against
ComfyUI `v0.30.0-1-g14b05228`.

The motivating use is **windowed sampling on low-denoise upscale passes only**. At
low denoise nothing is invented — every window starts from the same upscaled base,
so coherence comes from the input rather than from attention spanning the clip.
That is why windowing is safe there and not at full denoise. At 2048×1152 a latent
frame is 2304 tokens against 1008 at 1344×768, so an 8s chunk is ~131k video tokens
versus ~57k, and attention is quadratic. Windowing caps per-window cost regardless
of clip length.

## The good news

The multimodal design is already there and H3 opts in for free:

```python
# context_windows.py:424
is_multimodal = latent_shapes is not None and len(latent_shapes) > 1
```

A packed AV latent has two entries, so H3 lands on the multimodal path with no
flag. `IndexListContextWindow` carries `modality_windows`; `WindowingState` carries
per-modality `latents`, `guide_latents`, `latent_shapes`; and the per-window flow is
already modality-aware end to end:

1. `prepare_window()` derives per-modality index lists
2. `slice_for_window()` slices each modality by its own window
3. `pack_latents()` → one tensor → `calc_cond_batch()`
4. `unpack_latents()` → per-modality outputs
5. `combine_context_window_results()` accumulates per modality with fuse weights
6. finalize: divide by counts, re-pack

## Gap 1 — nobody implements the hook

```python
# context_windows.py:244
per_modality_indices = model.map_context_window_to_modalities(
    window.index_list, map_shapes, self.dim)
except AttributeError:
    raise NotImplementedError(
        f"{type(model).__name__} must implement map_context_window_to_modalities ...")
```

Grepped tree-wide: this name appears **twice**, at the call site and in its own
error message. **No model class defines it.** Multimodal context windowing is a
designed interface with zero implementations, so H3 would be the first.

For H3 the mapping is arithmetic we already have in `common.py`: a video-latent
window of length `L` covers `latents_to_frames(L)` frames (`17j+5`), which covers
`frames_to_audio_t(frames)` audio latents (`round(frames/24*40)`). Because audio_t
is not additive, a window `[a, b)` must be mapped as the **difference of two
cumulative totals**, not by converting the window length directly — the same
correction already made in `MMH3ConcatAV`.

## Gap 2 — one `dim` for every modality (the blocker)

`WindowingState.dim` is documented as "primary modality temporal dim", but it is
used for **all** modalities. H3's video is dim 2 and audio is dim 3, so `dim=2`
would window audio `[B,32,2,T40]` on its **stereo axis** — size 2, not `T40`. It
would not crash; it would produce `ratio = 2/T` and nonsense indices. This is the
dim-2 trap that also bites generic nested-tensor helpers.

Three sites, all in `IndexListContextHandler` / `WindowingState`:

```python
# ~410-414, freenoise per modality
mod_total = latent_shapes[i][self.dim]
modalities[i] = apply_freenoise(modalities[i], self.dim, mod_ctx_len, mod_ctx_overlap, seed)

# ~251-256, prepare_window
modality_total_frames = self.latents[mod_idx].shape[self.dim]
modality_windows[mod_idx] = IndexListContextWindow(
    per_modality_indices[mod_idx], dim=self.dim, ...)
```

Everything downstream is already correct once the window carries the right dim,
because `get_tensor()` and `add_window()` default to `self.dim` **of the window**,
and `slice_for_window()` / `combine_context_window_results()` both take
`window.get_window_for_modality(idx)`.

Cleanest fix: have `map_context_window_to_modalities` return a dim alongside the
indices per modality, or add a sibling `model.context_window_modality_dims(shapes)`.
Since the hook has **no implementations**, changing its contract costs nothing —
which is a strong argument for doing it now rather than after someone depends on it.

This generalisation helps any future AV model, not just H3, so it belongs upstream
rather than in the local patch pile.

## Gap 3 — `causal_window_fix` breaks the 5j+2 grid

Defaults to `True`. It prepends an anchor frame to every non-zero-indexed window:

```python
anchor_idx = window.index_list[0] - 1
```

That makes the window `L+1` latents. H3 only ever saw `5j+2`, so every window after
the first would be off-grid. Either default it off for H3 or make the anchor
grid-aware. Wan's node sets a precedent for hardcoding model-specific behaviour in
a model-specific node (`dim=2`, `4n+1` frames).

## Gap 4 — conditioning slicing must leave references alone

`get_resized_cond()` → `slice_cond()` slices cond tensors along a temporal dim.
H3's references live in the conditioning dict as `minimax_refs` (a list of dicts
holding latents), and they are **global context re-injected identically in every
window**, not something to slice. Needs verifying that they are passed through
untouched. Same for `minimax_frame_count`, which describes the whole clip rather
than the window — irrelevant while there are no keyframes, but wrong if there are.

Note also that reference rows are attended in **every** window, so N windows means
N× the reference cost. At `ref_downscale=2` that is tolerable; at
`ref_image_size="max"` it would dominate. On upscale passes the identity is already
in the pixels being refined, so dropping references entirely at stages 2–3 is worth
measuring.

## Gap 5 — `temporal_downscale_ratio` is approximate

`MiniMaxH3Video.temporal_downscale_ratio = 4`, but H3 is 17 frames per 5 latents
(3.4), with the first 5 frames collapsing to 2. It feeds `compute_guide_overlap()`,
which the H3 path does not use, but it is worth confirming it does not leak into the
window arithmetic.

## What an `MMH3ContextWindows` node would pin

Following `WanContextWindowsManualNode`, which hardcodes `dim=2` and enforces
`4n+1`:

- `dim = 2` (video), audio derived
- `context_length` snapped to `5j+2` latents — 12, 17, 22 (≈41, 58, 75 frames)
- `context_overlap` a multiple of 5
- `causal_window_fix` off
- `freenoise` off — it shuffles noise for window blending, and on a low-denoise pass
  there is very little noise to shuffle

## Measured: windowing is FASTER at high resolution, not slower

Observed 2026-08-06 — stage 3 at 2K ran **about a minute faster windowed than
whole**. That contradicts the obvious "5 windows = 5× the forwards" reasoning, and
the obvious reasoning is wrong.

For 57 latents at window 17, overlap 7 (stride 10, 5 windows):

```
attention  ∝ N²   5 × 17²/57²  =  0.44×    windowed does 56% LESS work
linear     ∝ N    5 × 17/57    =  1.49×    windowed does 49% MORE
```

Both ratios are resolution-independent — they depend only on latent counts. What
varies with resolution is the **mix**: at stage 3's ~131k tokens attention
dominates the linear terms so heavily that the 0.44× decides the result and the
overlap tax is noise. At stage 1's ~33k the balance is much closer, and windowing
would be a wash or a loss.

Consequences:

- **Smaller windows are not faster.** Window 12 (10 windows) has the *same* 0.44×
  attention ratio — more windows exactly cancels the smaller square — but linear
  cost rises to 2.1×. Shrink `context_length` for MEMORY, never for speed.
- Larger windows lower the linear tax at the same attention ratio, so 22 may beat
  17 on time. Untested.
- Stage 2 (~74k tokens) is probably also a win; assumed otherwise, unmeasured.
- Part of the gain may be indirect: lower peak activations mean less model
  offloading, which interacts with the estimator blind spot above.

## Open question, unmeasured

**How low the denoise has to be** before windows stop disagreeing. It decides
whether this works at 0.35 or only at 0.15, and it likely differs between stage 2
and stage 3, since stage 3 refines something already refined once. Worth measuring
on a short clip with deliberately small windows before building anything around it.
