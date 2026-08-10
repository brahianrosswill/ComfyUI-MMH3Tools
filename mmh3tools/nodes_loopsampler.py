"""Chained AV generation in one node: N chunks, mask-and-extend, one graph cost.

PROTOTYPE. The shape is settled, the tuning is not.

WHY ONE NODE. Driving N chunks from the graph costs a copy of every downstream
node per chunk -- sampler, decode, save -- and a graph that size is what starts
breaking ComfyUI. A Python loop inside one node costs the same whether it runs
4 chunks or 40. The trade is that nothing inside the loop can be a graph node,
so every prompt has to exist BEFORE sampling starts. That is what the cond_set
is: MMH3ReferenceMultiPrompt encodes all N up front, in one text-encoder load.

WHY MASK-AND-EXTEND rather than keyframe guides. The carried tail goes into the
HEAD of the new chunk's latent and is masked to 0, so the model conditions on it
without denoising it. It needs no core patch beyond per-row masking (#15375),
which MMH3SeedOverlap already checks for at runtime. Guides anchored at
arbitrary frames are the other route and land in core with #15439; that is
worth revisiting once it merges, because a guide costs no target frames whereas
a carried head does -- see TRIM below.

TRIM. The carried head occupies real frames of the output: chunk i reproduces
the last `overlap_latents` steps of chunk i-1. ConcatAV drops them from the
SECOND clip at the join, so the master is continuous and each chunk contributes
its new frames only. That is also why chunk 0 is the only one that keeps its
head.

GUIDER COPYING is the part that is easy to get wrong. copy.copy is shallow, so
new_g.original_conds is the SAME dict object as the source guider's, and
set_conds assigns into it -- chunk 0 would overwrite the BASE conditioning and
every later chunk would read chunk 0's conds back as "base". Rebinding the dict
per chunk is the fix. Learned in LTXAVTools, where the symptom was every chunk
getting chunk 0's speaker.
"""

import copy
import logging

from comfy_api.latest import io
from comfy.nested_tensor import NestedTensor

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .common import unpack_av, latents_to_frames, VIDEO_T_DIM, AUDIO_T_DIM
from .nodes_loop import MMH3SeedOverlap, MMH3ConcatAV
from .nodes_multiprompt import MMH3CondSet


def _raw_conds(guider):
    """This guider's (positive, negative) in the form set_conds accepts.

    Our own copies stamp `raw_conds` so a chunk never reads the previous
    chunk's conditioning back as the base.
    """
    if hasattr(guider, "raw_conds"):
        return guider.raw_conds
    return (guider.original_conds["positive"], guider.original_conds["negative"])


def _chunk_guider(guider, positive):
    new_g = copy.copy(guider)
    # SHALLOW copy shares original_conds with the source; set_conds assigns into
    # it. Rebind before touching it or chunk 0 clobbers the base conditioning.
    new_g.original_conds = dict(guider.original_conds)
    _, negative = _raw_conds(guider)
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


class MMH3LoopSampler(io.ComfyNode):
    """Sample N chained chunks in one node, carrying each tail into the next."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LoopSampler",
            display_name="MiniMax H3 Loop Sampler",
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
                io.Guider.Input("guider"),
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
                feather_latents) -> io.NodeOutput:
        conds = (cond_set or {}).get("conds") or []
        if not conds:
            raise ValueError("MMH3LoopSampler: cond_set holds no conditioning.")
        n = int(chunks)

        v0, a0 = unpack_av(latent, "latent")
        lines = ["%d chunk%s of %d video latents (%d frames), %d audio latents"
                 % (n, "" if n == 1 else "s", v0.shape[VIDEO_T_DIM],
                    latents_to_frames(v0.shape[VIDEO_T_DIM]),
                    0 if a0 is None else a0.shape[AUDIO_T_DIM])]
        if len(conds) < n:
            lines.append("! only %d prompt%s for %d chunks -- the last one repeats"
                         % (len(conds), "" if len(conds) == 1 else "s", n))

        joined, prev = None, None
        k_used, lost = 0, 0
        for i in range(n):
            target = _clone_latent(latent)
            carried = 0
            if prev is not None:
                target, carried, k_used = MMH3SeedOverlap.execute(
                    target, prev, int(overlap_latents),
                    float(overlap_strength_video), float(overlap_strength_audio),
                    int(feather_latents)).result

            g = _chunk_guider(guider, conds[min(i, len(conds) - 1)])
            out, _denoised = SamplerCustomAdvanced().sample(
                _chunk_noise(noise, i), g, sampler, sigmas, target)

            # The carried head is a reproduction of the previous tail and comes off
            # the SECOND clip at every join; chunk 0 has no head to lose.
            #
            # TRIM IS k+2, NOT k. SeedOverlap prepends a multiple of 5, so removing
            # exactly k would leave the master off the 5j+2 grid -- and an off-grid
            # latent cannot be decoded, since latents_to_frames only holds on grid.
            # k+2 is the nearest grid-safe cut, so it takes the overlap plus about 7
            # frames of new content. ConcatAV's docstring has the full trade: k
            # cannot be 0 and 2 mod 5 at once. Trimming k-3 instead would keep those
            # frames and duplicate part of the overlap; neither is free.
            if joined is None:
                joined = out
            else:
                trim = k_used + 2
                joined = MMH3ConcatAV.execute(joined, out, trim, False).result[0]
                lost += latents_to_frames(k_used + 2) - latents_to_frames(k_used)
            prev = out

            lines.append("  chunk %d: prompt %d, %d carried frames"
                         % (i, min(i, len(conds) - 1), carried))
            logging.info("[MMH3LoopSampler] chunk %d/%d done", i + 1, n)

        vj, aj = unpack_av(joined, "joined")
        lines.append("master: %d video latents (%d frames), %d audio latents"
                     % (vj.shape[VIDEO_T_DIM], latents_to_frames(vj.shape[VIDEO_T_DIM]),
                        0 if aj is None else aj.shape[AUDIO_T_DIM]))
        if lost:
            lines.append("  %d frames of new content taken by the grid-safe trim "
                         "(%d per seam). Trimming after DECODE instead is exact -- "
                         "see MMH3FindDivergence." % (lost, lost // max(1, n - 1)))
        report = "\n".join(lines)
        logging.info("[MMH3LoopSampler] " + lines[0])
        return io.NodeOutput(joined, n, report)
