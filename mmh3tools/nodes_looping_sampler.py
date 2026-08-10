"""Chained AV generation in one node: N chunks, mask-and-extend, one graph cost.

PROTOTYPE. The shape is settled, the tuning is not.

WHY ONE NODE. Driving N chunks from the graph costs a copy of every downstream
node per chunk -- sampler, decode, save -- and a graph that size is what starts
breaking ComfyUI. A Python loop inside one node costs the same whether it runs
4 chunks or 40. The trade is that nothing inside the loop can be a graph node,
so every prompt has to exist BEFORE sampling starts. That is what the cond_set
is: MMH3ReferenceMultiPrompt encodes all N up front, in one text-encoder load.

TWO CARRY ROUTES, and the difference is what the join costs.

`mask` prepends the tail to the new chunk's latent and masks it to 0, so the
model conditions on it without denoising it. Needs only per-row masking
(#15375). SeedOverlap prepends a MULTIPLE OF 5, so the chunk grows, and the
join must trim carry+2 to leave the master on the 5j+2 grid -- an off-grid
latent cannot be decoded at all. That +2 takes ~7 frames of real content per
seam.

`keyframe` passes the tail as a GUIDE anchored at frame 0: re-injected every
step, never denoised, and carrying a multi-step clip plus its audio at the same
cond_t. Needs #15439. The chunk keeps its natural length, and the carry is
5m+2 to begin with, so trimming exactly it is ALREADY grid-safe. Nothing extra
is lost. That is the argument for this route.

Both reproduce the carry in the head of every chunk after the first, and both
drop it from the SECOND clip at the join. Chunk 0 is the only one that keeps
its head.

GUIDES ARE BUILT HERE, per chunk, and stale ones are stripped off incoming
conditioning first -- an upstream guide node or a cond cached from a previous
run would anchor this chunk to somebody else's frames. Straight from
LTXAVTools, where the same leak had the same cause.

GUIDER COPYING is the part that is easy to get wrong. copy.copy is shallow, so
new_g.original_conds is the SAME dict object as the source guider's, and
set_conds assigns into it -- chunk 0 would overwrite the BASE conditioning and
every later chunk would read chunk 0's conds back as "base". Rebinding the dict
per chunk is the fix. Learned in LTXAVTools, where the symptom was every chunk
getting chunk 0's speaker.
"""

import copy
import logging

import node_helpers
from comfy_api.latest import io
from comfy.nested_tensor import NestedTensor

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .common import (AUDIO_T_DIM, LATENTS_PER_GROUP, LATENT_BASE, VIDEO_T_DIM,
                     frame_at_latent, latents_to_frames, slice_av_tail, unpack_av)
from .nodes_loop import MMH3SeedOverlap, MMH3ConcatAV
from .nodes_multiprompt import MMH3CondSet

_GUIDE_KEYS = ("minimax_keyframes", "minimax_frame_count")


def _strip_guide_keys(cond, label):
    """Remove keyframe bookkeeping from conditioning.

    This node registers ALL of its own guides, per chunk. Anything arriving
    pre-registered is stale -- an upstream guide node, or a cond cached from a
    previous run -- and would anchor this chunk to the wrong frames. Straight
    from LTXAVTools, where the same leak had the same cause.
    """
    out, stripped = [], False
    for t, d in cond:
        if any(k in d for k in _GUIDE_KEYS):
            d = {k: v for k, v in d.items() if k not in _GUIDE_KEYS}
            stripped = True
        out.append([t, d])
    if stripped:
        logging.info("[MMH3LoopingSampler] stripped stale keyframes from %s; this node "
                     "builds its own per chunk", label)
    return out


def _has_refs(cond):
    return any("minimax_refs" in d and d["minimax_refs"] for _t, d in cond)


def _snap_carry(n):
    """Down to the 5m+2 grid, which slice_av_tail's conversion requires."""
    n = int(n)
    if n < LATENT_BASE:
        return LATENT_BASE
    return ((n - LATENT_BASE) // LATENTS_PER_GROUP) * LATENTS_PER_GROUP + LATENT_BASE


def _carry_guide(prev_latent, carry_latents):
    """The previous chunk's tail as ONE guide anchored at target frame 0.

    Grid: slice_av_tail converts with latents_to_frames, which only holds on the
    5j+2 grid, so the carry snaps to 5m+2. That is also the phase-0 guarantee --
    a 5m+2 tail off a 5j+2 clip starts at step 5(j-m) -- so the slice is exactly
    what a fresh encode of those frames would produce, and no VAE is involved.

    Anchored at 0, not a negative index: PackedLayout takes negatives literally,
    so cond_t would fall below text_len and collide with the text positions.
    The cost is that target frames 0..span-1 reproduce the carry and come off at
    the join -- but 5m+2 is exactly the trim that keeps the master on grid, so
    unlike the mask route nothing extra is lost.
    """
    v, a = unpack_av(prev_latent, "previous chunk")
    n = min(_snap_carry(carry_latents), int(v.shape[VIDEO_T_DIM]))
    tv, ta, frames, _at = slice_av_tail(v, a, n)
    kf = {"resolved_frame_index": 0, "latent": tv}
    if ta is not None:
        kf["audio_latent"] = ta
    return kf, n, frames


def _guides_available():
    """Whether #15439's any-index guides are present.

    Detected, not assumed: stock raises on any anchor that is not first/last,
    and a guide carrying a multi-step clip is not expressible at all.
    """
    try:
        import inspect
        import comfy.ldm.minimax.model as mm
        return "only first/last keyframe anchors" not in inspect.getsource(
            mm.PackedLayout.__init__)
    except Exception:
        return False


def _guide_origin_correct():
    """Whether a guide anchors on the TARGET origin when references are present.

    #15439 as written anchors to text_len, but the target begins after the refs
    -- they advance the layout cursor. A guide then sits `ref_advance` units
    before the clip: -1 for one image reference, -320 for a chunk's worth of
    voice audio. Nothing errors; the anchor just points into the reference
    region. This pack carries a local correction (docs/core-changes.md); no PR
    does yet, so it is MEASURED rather than assumed.

    Only matters when refs and guides are combined. Guides alone are fine on
    stock #15439, since the cursor never leaves text_len.
    """
    try:
        import torch
        import comfy.ldm.minimax.model as mm
        lay = mm.PackedLayout(
            8, 7, 4, 4, 8,
            keyframes=[{"resolved_frame_index": 0,
                        "latent": torch.zeros([1, 24, 1, 4, 4])}],
            refs=[{"kind": "image", "latent_h": 4, "latent_w": 4,
                   "latent": torch.zeros([1, 24, 1, 4, 4])}])
        seg = {k: a for a, _b, k in lay.segments}
        return abs(float(lay.position_ids[seg["cond"], 0])
                   - float(lay.position_ids[seg["video"], 0])) < 1e-6
    except Exception:
        return False


def _raw_conds(guider):
    """This guider's (positive, negative) in the form set_conds accepts.

    negative is None for a BasicGuider, which has no such key at all --
    Guider_Basic.set_conds takes ONE argument. Indexing for it raises KeyError,
    and calling set_conds with two raises TypeError.

    Our own copies stamp `raw_conds` so a chunk never reads the previous
    chunk's conditioning back as the base.
    """
    if hasattr(guider, "raw_conds"):
        return guider.raw_conds
    conds = getattr(guider, "original_conds", {}) or {}
    return (conds.get("positive"), conds.get("negative"))


def _chunk_guider(guider, positive):
    """This chunk's guider: the wired one, with its POSITIVE replaced.

    Everything else the guider carries -- model, cfg, and the negative if it has
    one -- is kept. Whatever positive was wired into it is discarded, because
    the per-chunk conditioning comes from the cond_set.
    """
    new_g = copy.copy(guider)
    # SHALLOW copy shares original_conds with the source; set_conds assigns into
    # it. Rebind before touching it or chunk 0 clobbers the base conditioning.
    new_g.original_conds = dict(guider.original_conds)
    _, negative = _raw_conds(guider)
    if negative is None:
        new_g.set_conds(positive)              # Guider_Basic: no CFG, no negative
    else:
        new_g.set_conds(positive, negative)
    new_g.raw_conds = (positive, negative)
    return new_g


def _chunk_noise(noise, index):
    """A distinct noise per chunk. Reusing one object gives every chunk the
    same noise, which reads as the model refusing to advance."""
    if index == 0 or not hasattr(noise, "seed"):
        return noise
    n = copy.copy(noise)
    n.seed = int(noise.seed) + index
    return n


def _clone_tensorish(t):
    """Clone a plain tensor or a NestedTensor. NestedTensor has no .clone(),
    only .unbind(), so an AV pair has to be taken apart and rebuilt."""
    if t is None:
        return None
    if hasattr(t, "clone"):
        return t.clone()
    if hasattr(t, "unbind"):
        return NestedTensor([x.clone() for x in t.unbind()])
    if isinstance(t, (list, tuple)):
        return [x.clone() for x in t]
    return t


def _clone_latent(latent):
    """A chunk must never write through to the template it was cloned from --
    the same dict is reused for every chunk and by whatever else it is wired to."""
    out = dict(latent)
    for k in ("samples", "noise_mask", "audio_noise_mask"):
        if latent.get(k) is not None:
            out[k] = _clone_tensorish(latent[k])
    return out


class MMH3LoopingSampler(io.ComfyNode):
    """Sample N chained chunks in one node, carrying each tail into the next."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LoopingSampler",
            display_name="MiniMax H3 Looping Sampler",
            category="MMH3Tools",
            description=(
                "N chained chunks in one node execution. Each chunk seeds its head "
                "from the previous chunk's tail, masks it so the model conditions on "
                "it without redrawing it, and generates forward. The graph is the "
                "same size for 4 chunks or 40.\n\n"
                "Prompts must all exist before sampling starts -- wire a cond_set "
                "from MiniMax H3 Reference (Multi-Prompt). If it holds fewer prompts "
                "than there are chunks, the last one repeats and the report says so."
            ),
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input(
                    "guider",
                    tooltip="Supplies the MODEL, the cfg, and the negative if it "
                            "is a CFG guider. Its POSITIVE is replaced every chunk "
                            "from the cond_set, so whatever is wired there is "
                            "ignored -- wire MMH3 Cond Select at index 0 so the "
                            "graph is valid and says what it means. A Basic Guider "
                            "works too; it simply has no negative."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                MMH3CondSet.Input("cond_set"),
                io.Latent.Input(
                    "latent",
                    tooltip="One chunk's empty AV latent -- the same one the cond_set "
                            "node emits. Cloned per chunk; never mutated."),
                io.Int.Input(
                    "chunks", default=4, min=1, max=512,
                    tooltip="How many to render. Independent of how many prompts the "
                            "cond_set holds."),
                io.Int.Input(
                    "overlap_latents", default=5, min=5, max=95,
                    tooltip="Video latent steps carried from the previous tail. "
                            "SeedOverlap snaps this DOWN to a multiple of 5, each "
                            "worth 17 frames -- the target keeps its full length and "
                            "the carry is prepended on top. Larger carries more "
                            "context and costs more to trim back off."),
                io.Float.Input(
                    "overlap_strength_video", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input(
                    "overlap_strength_audio", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="1.0 pins the carried audio outright. Lipsync wants this "
                            "harder than video."),
                io.Int.Input("feather_latents", default=0, min=0, max=32),
                io.Combo.Input(
                    "carry", options=["mask", "keyframe"], default="mask",
                    tooltip="HOW the previous tail reaches the next chunk.\n\n"
                            "'mask' prepends it to the latent head and masks it, so "
                            "the model conditions on it without denoising it. Needs "
                            "only per-row masking (#15375). The chunk grows by the "
                            "carry, and the join has to trim carry+2 to stay on the "
                            "grid, so ~7 frames of new content go with it.\n\n"
                            "'keyframe' passes it as a GUIDE anchored at frame 0 -- "
                            "re-injected every step, never denoised. Needs #15439. The "
                            "chunk keeps its natural length and the trim is exactly "
                            "the carry, which is already 5m+2, so nothing extra is "
                            "lost."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="chunks_rendered"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, cond_set, latent, chunks,
                overlap_latents, overlap_strength_video, overlap_strength_audio,
                feather_latents, carry="mask") -> io.NodeOutput:
        conds = (cond_set or {}).get("conds") or []
        if not conds:
            raise ValueError("MMH3LoopingSampler: cond_set holds no conditioning.")
        n = int(chunks)

        # Refuse rather than silently fall back. Stock raises on any anchor that is
        # not first/last, so a keyframe carry would die mid-run on chunk 1 -- after
        # chunk 0 had already been paid for.
        if carry == "keyframe" and not _guides_available():
            raise RuntimeError(
                "MMH3LoopingSampler: carry='keyframe' needs any-index guides "
                "(upstream PR #15439), which is not applied. See "
                "docs/core-changes.md. Stock ComfyUI supports first/last anchors "
                "only and cannot express a multi-step clip guide at all. Use "
                "carry='mask', which needs only #15375.")

        v0, a0 = unpack_av(latent, "latent")
        lines = ["%d chunk%s of %d video latents (%d frames), %d audio latents"
                 % (n, "" if n == 1 else "s", v0.shape[VIDEO_T_DIM],
                    latents_to_frames(v0.shape[VIDEO_T_DIM]),
                    0 if a0 is None else a0.shape[AUDIO_T_DIM])]
        if len(conds) < n:
            lines.append("! only %d prompt%s for %d chunks -- the last one repeats"
                         % (len(conds), "" if len(conds) == 1 else "s", n))

        joined, prev = None, None
        k_used, lost, trim = 0, 0, 0
        for i in range(n):
            target = _clone_latent(latent)
            carried = 0
            # Whatever the mode, this chunk registers its OWN guides -- stale ones
            # from a cached cond would anchor it to the wrong frames.
            chunk_cond = _strip_guide_keys(conds[min(i, len(conds) - 1)],
                                           "chunk %d prompt" % i)

            if prev is not None and carry == "keyframe":
                # Refs advance the layout cursor, so a guide anchored on text_len
                # lands BEFORE the clip -- and #15439 as written anchors on text_len.
                # Harmless with guides alone; silently wrong the moment a reference
                # rides along, which is exactly the identity-reference case.
                if _has_refs(chunk_cond) and not _guide_origin_correct():
                    raise RuntimeError(
                        "MMH3LoopingSampler: chunk %d carries a reference AND a "
                        "keyframe guide, but this ComfyUI anchors guides on text_len "
                        "instead of the target origin. The guide would land "
                        "ref_advance units before the clip -- -1 for one image ref, "
                        "-320 for a chunk of voice audio -- silently. See "
                        "docs/core-changes.md for the correction, or drop the "
                        "reference, or use carry='mask'." % i)
                kf, k_used, carried = _carry_guide(prev, overlap_latents)
                chunk_cond = node_helpers.conditioning_set_values(
                    chunk_cond, {"minimax_keyframes": [kf]})
                # the guide anchors target frames 0..span-1, so the trim is exactly
                # the carry -- which is 5m+2 already, so the master stays on grid
                trim = k_used
            elif prev is not None:
                target, carried, k_used = MMH3SeedOverlap.execute(
                    target, prev, int(overlap_latents),
                    float(overlap_strength_video), float(overlap_strength_audio),
                    int(feather_latents)).result
                trim = k_used + 2

            g = _chunk_guider(guider, chunk_cond)
            out, _denoised = SamplerCustomAdvanced().sample(
                _chunk_noise(noise, i), g, sampler, sigmas, target)

            # Either way the head of a later chunk reproduces the previous tail and
            # comes off the SECOND clip at the join. Chunk 0 has no head to lose.
            #
            # The two modes differ in what that costs. SeedOverlap prepends a
            # MULTIPLE OF 5, so removing exactly it would leave the master off the
            # 5j+2 grid -- undecodable, since latents_to_frames only holds on grid --
            # and k+2 is the nearest safe cut, taking ~7 frames of new content with
            # it. ConcatAV's docstring has the full trade: k cannot be 0 and 2 mod 5
            # at once. A keyframe carry is 5m+2 to begin with, so trimming exactly it
            # is already grid-safe and costs nothing extra. That is the argument for
            # the guide route once #15439 lands.
            if joined is None:
                joined = out
            else:
                joined = MMH3ConcatAV.execute(joined, out, trim, False).result[0]
                # frame_at_latent, not latents_to_frames: in mask mode k_used is a
                # MULTIPLE OF 5, which is off grid, and the latter would floor it.
                lost += frame_at_latent(trim) - carried
            prev = out

            lines.append("  chunk %d: prompt %d, %d carried frames"
                         % (i, min(i, len(conds) - 1), carried))
            logging.info("[MMH3LoopingSampler] chunk %d/%d done", i + 1, n)

        vj, aj = unpack_av(joined, "joined")
        lines.append("master: %d video latents (%d frames), %d audio latents"
                     % (vj.shape[VIDEO_T_DIM], latents_to_frames(vj.shape[VIDEO_T_DIM]),
                        0 if aj is None else aj.shape[AUDIO_T_DIM]))
        if lost:
            lines.append("  %d frames of new content taken by the grid-safe trim "
                         "(%d per seam). Trimming after DECODE instead is exact -- "
                         "see MMH3FindDivergence." % (lost, lost // max(1, n - 1)))
        report = "\n".join(lines)
        logging.info("[MMH3LoopingSampler] " + lines[0])
        return io.NodeOutput(joined, n, report)
