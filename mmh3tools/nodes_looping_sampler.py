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

from .common import (AUDIO_T_DIM, FPS, LATENTS_PER_GROUP, LATENT_BASE, VIDEO_T_DIM,
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


def _parse_indices(text, total_frames):
    """Comma-separated GLOBAL pixel-frame indices; negatives count from the end.

    Resolved here rather than passed through: PackedLayout takes a negative
    literally, so cond_t would fall below text_len and collide with the text
    token positions. Out of range is an error -- silently dropping a keyframe
    the user asked for is worse than stopping.
    """
    out = []
    for piece in (text or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            v = int(piece)
        except ValueError:
            raise ValueError(
                "MMH3LoopingSampler: keyframe_indices has %r in it, which is not a "
                "whole number. Expected something like '0, 60, -1'." % piece)
        if v < 0:
            v += total_frames
        if not 0 <= v < total_frames:
            raise ValueError(
                "MMH3LoopingSampler: keyframe index %s is outside the master, which "
                "is %d frames (0-%d)." % (piece, total_frames, total_frames - 1))
        out.append(v)
    return out


def _chunk_origins(n, chunk_latents, trim):
    """Master latent index of each chunk's LOCAL latent 0.

    The master is chunk 0 whole, then every later chunk minus `trim`, so chunk i
    starts `trim` latents before the point its new content lands. Every origin is
    a multiple of 5 in both carry modes -- (5a+2)-(5m+2) = 5(a-m) -- which is what
    keeps each chunk on phase 0 and makes frame_at_latent valid on these.
    """
    origins, cum = [], 0
    for i in range(n):
        if i == 0:
            origins.append(0)
            cum = chunk_latents[0]
        else:
            origins.append(cum - trim)
            cum += chunk_latents[i] - trim
    return origins, cum


def _keyframe_plan(indices, origins, chunk_frames, carry_frames):
    """Global frame -> (chunk, local frame), one entry per index.

    An index inside a chunk's carried HEAD is reproduced from the previous chunk
    and trimmed at the join, so anchoring it there paints a frame nobody sees.
    Assign it to the LAST chunk whose new content covers it; chunk 0 has no head
    so it owns everything in its span.
    """
    plan = []
    for g in indices:
        owner = None
        for i, off in enumerate(origins):
            local = g - frame_at_latent(off)
            if not 0 <= local < chunk_frames[i]:
                continue
            if i == 0 or local >= carry_frames[i]:
                owner = (i, local)
        if owner is None:
            # only reachable inside a head that no later chunk re-covers
            for i, off in enumerate(origins):
                local = g - frame_at_latent(off)
                if 0 <= local < chunk_frames[i]:
                    owner = (i, local)
                    break
        if owner is not None:
            plan.append(owner)
    return plan


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
                    tooltip="ONE CHUNK's empty AV latent -- the same one the cond_set "
                            "node emits. Cloned per chunk; never mutated.\n\n"
                            "Not the whole clip. Size it from MMH3 Window Plan's "
                            "`window_frames`, not `total_frames`: feeding the master "
                            "length here renders `chunks` copies of the WHOLE clip, "
                            "which runs and looks like the model ignoring the length."),
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
                io.Int.Input(
                    "feather_latents", default=0, min=0, max=32,
                    tooltip="`mask` carry only, and VIDEO only. A linear ramp on the "
                            "mask over N latents after the carried region, easing from "
                            "preserved back to fully generating instead of stepping at "
                            "the seam. 0 disables. The audio mask is never feathered.\n\n"
                            "It grades the sampler's latent blend but NOT the timestep: "
                            "mask_row_targets binarises at 0.5, so the preserve/generate "
                            "boundary simply moves to wherever the ramp crosses it. And "
                            "a latent is 1 or 4 frames, so N latents is not N*4 frames -- "
                            "it depends where the ramp falls in the 5-cycle.\n\n"
                            "Untested against real weights."),
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
                io.Image.Input(
                    "keyframes", optional=True,
                    tooltip="A BATCH of stills to pin, one per index in "
                            "keyframe_indices. Encoded once here, so no VAE round "
                            "trip per chunk. Independent of `carry` -- guides and a "
                            "masked carry coexist."),
                io.String.Input(
                    "keyframe_indices", multiline=False, default="",
                    tooltip="Comma-separated frame indices, GLOBAL across the whole "
                            "master rather than per chunk, so you place a shot where "
                            "it belongs in the finished clip and this works out which "
                            "chunk owns it. Negatives count from the end. An index "
                            "landing inside a chunk's carried head goes to the chunk "
                            "that actually renders it, since the head is trimmed at "
                            "the join."),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="The H3 VIDEO vae, needed only to encode `keyframes`."),
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
                feather_latents, carry="mask", keyframes=None,
                keyframe_indices="", vae=None) -> io.NodeOutput:
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
        # A chunk longer than ~20s is almost always the whole clip wired in by
        # mistake. It runs, and produces `chunks` copies of the master.
        if n > 1 and latents_to_frames(int(v0.shape[VIDEO_T_DIM])) > 480:
            lines.append("! a %.1fs chunk with %d chunks -> a %.1fs master. If that is "
                         "not what you meant, `latent` wants ONE CHUNK's length, not "
                         "the clip's"
                         % (latents_to_frames(int(v0.shape[VIDEO_T_DIM])) / FPS, n,
                            n * latents_to_frames(int(v0.shape[VIDEO_T_DIM])) / FPS))

        # ---- the schedule, resolved BEFORE sampling ------------------------
        # Global keyframe indices cannot be placed without knowing every chunk's
        # length and trim up front, and finding out on chunk 3 that an index is
        # unreachable is an hour too late.
        T = int(v0.shape[VIDEO_T_DIM])
        if carry == "keyframe":
            k_plan = min(_snap_carry(overlap_latents), T)
            lengths = [T] * n
            trim = k_plan
        else:
            k_plan = min(max(LATENTS_PER_GROUP,
                             (int(overlap_latents) // LATENTS_PER_GROUP)
                             * LATENTS_PER_GROUP), T)
            lengths = [T] + [T + k_plan] * (n - 1)
            trim = k_plan + LATENT_BASE
        # What the ownership rule cares about is what the JOIN REMOVES, not what the
        # carry spans. Under `mask` those differ: the carry is a multiple of 5 but the
        # trim is k+2, so 22 frames come off where the carry is only 17. Using the
        # carry sent a chunk's own last frame into the NEXT chunk, into a region that
        # is then trimmed -- the keyframe would have been painted and thrown away.
        head_frames = frame_at_latent(trim)
        origins, total_lat = _chunk_origins(n, lengths, trim)
        chunk_frames = [latents_to_frames(L) for L in lengths]
        carry_frames = [0] + [head_frames] * (n - 1)
        total_frames = latents_to_frames(total_lat)

        # ---- user keyframes, encoded ONCE ---------------------------------
        guides_by_chunk = {}
        wanted = _parse_indices(keyframe_indices, total_frames)
        if wanted and keyframes is None:
            raise ValueError(
                "MMH3LoopingSampler: keyframe_indices names %d frame(s) but no "
                "keyframes were supplied." % len(wanted))
        if keyframes is not None and wanted:
            if vae is None:
                raise ValueError(
                    "MMH3LoopingSampler: keyframes need the H3 video vae to encode "
                    "them. Wire `vae`.")
            n_img = int(keyframes.shape[0])
            if n_img != len(wanted):
                raise ValueError(
                    "MMH3LoopingSampler: %d keyframe image(s) against %d index/indices. "
                    "They are zipped, so the counts must match." % (n_img, len(wanted)))
            if not _guides_available():
                raise RuntimeError(
                    "MMH3LoopingSampler: keyframes need any-index guides (upstream "
                    "PR #15439), which is not applied. See docs/core-changes.md.")
            plan = _keyframe_plan(wanted, origins, chunk_frames, carry_frames)
            for (ci, local), g, img_i in zip(plan, wanted, range(n_img)):
                z = vae.encode(keyframes[img_i:img_i + 1])
                guides_by_chunk.setdefault(ci, []).append(
                    {"resolved_frame_index": int(local), "latent": z})
                lines.append("  keyframe frame %d -> chunk %d local frame %d"
                             % (g, ci, local))

        joined, prev = None, None
        k_used, lost = 0, 0
        for i in range(n):
            target = _clone_latent(latent)
            carried = 0
            # Whatever the mode, this chunk registers its OWN guides -- stale ones
            # from a cached cond would anchor it to the wrong frames.
            chunk_cond = _strip_guide_keys(conds[min(i, len(conds) - 1)],
                                           "chunk %d prompt" % i)
            chunk_guides = list(guides_by_chunk.get(i, []))

            if chunk_guides and _has_refs(chunk_cond) and not _guide_origin_correct():
                raise RuntimeError(
                    "MMH3LoopingSampler: chunk %d carries a reference AND a keyframe, "
                    "but this ComfyUI anchors guides on text_len instead of the "
                    "target origin. See docs/core-changes.md." % i)

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
                chunk_guides.insert(0, kf)      # the carry anchors frame 0, so first
            elif prev is not None:
                target, carried, k_used = MMH3SeedOverlap.execute(
                    target, prev, int(overlap_latents),
                    float(overlap_strength_video), float(overlap_strength_audio),
                    int(feather_latents)).result

            # One conditioning_set_values for the whole chunk: the carry guide (if
            # any) plus whatever user keyframes the schedule assigned here. Setting
            # it twice would replace, not merge.
            if chunk_guides:
                chunk_cond = node_helpers.conditioning_set_values(
                    chunk_cond, {"minimax_keyframes": chunk_guides})

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

            lines.append("  chunk %d: prompt %d, %d carried frames%s"
                         % (i, min(i, len(conds) - 1), carried,
                            "" if len(chunk_guides) <= (1 if carried else 0) else
                            ", %d keyframe(s)" % (len(chunk_guides)
                                                  - (1 if carried else 0))))
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


class MMH3KeyframePlanner(io.ComfyNode):
    """End-anchored keyframe indices for a chained run.

    TRAVEL SEMANTICS, from LTXAVTools' planner. The first keyframe (optional)
    opens the video at frame 0; every later one sits at the END of its chunk, so
    each chunk generates TOWARD its destination image and the next continues
    from the arrived state through the ordinary carry. Start-anchoring instead
    would put the image in the NEXT chunk and invite a snap at every seam.
    The final chunk's end is the video's end, emitted as -1.

    That lands exactly one keyframe per chunk under MMH3LoopingSampler's
    ownership rule: a chunk's end frame is inside the NEXT chunk's carried head,
    which the join trims, so it is owned by the chunk that renders it.

    The schedule is computed from the same numbers the sampler uses, so wire the
    same values. Uniform chunks are the normal case; `scene_frames` overrides
    with explicit lengths when scenes do not line up with chunks.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3KeyframePlanner",
            display_name="MiniMax H3 Keyframe Planner",
            category="MMH3Tools",
            description=(
                "End-anchored keyframe indices: frame 0 opens, each chunk travels to "
                "a keyframe at its end, the last ends on -1. Wire `indices` to the "
                "Looping Sampler's keyframe_indices; `count` is how many images its "
                "`keyframes` batch must hold, in that order."
            ),
            inputs=[
                io.Int.Input("chunks", default=4, min=1, max=512),
                io.Int.Input(
                    "chunk_latents", default=57, min=7, max=3600,
                    tooltip="Video latents per chunk -- the template's length. Must "
                            "match the Looping Sampler's `latent`."),
                io.Int.Input(
                    "overlap_latents", default=5, min=2, max=95,
                    tooltip="Same value the sampler gets. It is snapped the same way, "
                            "per carry mode."),
                io.Combo.Input(
                    "carry", options=["mask", "keyframe"], default="mask",
                    tooltip="Same as the sampler's. It changes the chunk lengths and "
                            "the trim, so it changes where every chunk ends."),
                io.Boolean.Input(
                    "include_start", default=True,
                    tooltip="A keyframe at frame 0 -- the opening image."),
                io.Boolean.Input(
                    "include_end", default=True,
                    tooltip="A keyframe at -1 -- the closing image, the end of the "
                            "final chunk."),
                io.String.Input(
                    "scene_frames", multiline=False, default="",
                    tooltip="Optional override: pipe or comma separated FRAME counts, "
                            "one per scene, when scenes do not coincide with chunks. "
                            "Ends are placed at each scene boundary instead."),
            ],
            outputs=[
                io.String.Output(display_name="indices"),
                io.Int.Output(display_name="count"),
                io.String.Output(display_name="total_frames"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, chunks, chunk_latents, overlap_latents, carry,
                include_start, include_end, scene_frames="") -> io.NodeOutput:
        n, T = int(chunks), int(chunk_latents)
        if carry == "keyframe":
            k = min(_snap_carry(overlap_latents), T)
            lengths, trim = [T] * n, k
        else:
            k = min(max(LATENTS_PER_GROUP,
                        (int(overlap_latents) // LATENTS_PER_GROUP)
                        * LATENTS_PER_GROUP), T)
            lengths = [T] + [T + k] * (n - 1)
            trim = k + LATENT_BASE
        origins, total_lat = _chunk_origins(n, lengths, trim)
        total_f = latents_to_frames(total_lat)

        # last frame of each unit, in GLOBAL frames
        if scene_frames.strip():
            spans, cum = [], 0
            for p in scene_frames.replace(",", "|").split("|"):
                p = p.strip()
                if not p:
                    continue
                try:
                    cum += int(round(float(p)))
                except ValueError:
                    raise ValueError(
                        "MMH3KeyframePlanner: %r in scene_frames is not a number." % p)
                spans.append(min(cum, total_f) - 1)
            if not spans:
                raise ValueError("MMH3KeyframePlanner: scene_frames parsed to nothing.")
            unit = "scene"
        else:
            spans = [frame_at_latent(origins[i] + lengths[i]) - 1 for i in range(n)]
            unit = "chunk"

        entries = ([0] if include_start else []) + spans[:-1]
        if include_end:
            entries.append(-1)

        idx = ", ".join(str(e) for e in entries)
        lines = ["%d %s%s over %d frames (%.2fs), %d keyframe%s"
                 % (len(spans), unit, "" if len(spans) == 1 else "s", total_f,
                    total_f / float(FPS), len(entries),
                    "" if len(entries) == 1 else "s")]
        for e in entries:
            g = total_f - 1 if e < 0 else e
            lines.append("  frame %-6s %6.2fs%s"
                         % (e, g / float(FPS), "   (end of clip)" if e < 0 else ""))
        if not entries:
            lines.append("  ! nothing to place -- one chunk with both ends off")
        report = "\n".join(lines)
        logging.info("[MMH3KeyframePlanner] " + lines[0])
        return io.NodeOutput(idx, len(entries), str(total_f), report)
