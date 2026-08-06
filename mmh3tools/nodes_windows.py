"""Context windows for MiniMax H3, without patching core.

ComfyUI's context windowing already has a complete multimodal design and H3 opts
into it for free -- `is_multimodal = len(latent_shapes) > 1` is true for any packed
AV latent. Two things stop it working out of the box:

  1. `map_context_window_to_modalities` has ZERO implementations tree-wide. The
     multimodal path raises NotImplementedError for every model.
  2. `WindowingState` uses ONE `dim` for every modality. H3's video is dim 2 and
     audio is dim 3, so the stock path would window audio [B,32,2,T40] on its
     STEREO axis -- size 2, not T40. It would not crash; it would produce a ratio
     of 2/T and nonsense indices.

Neither needs a core edit. The handler is just an object in a dict
(`model.model_options["context_handler"]`, read back in samplers.py), so a subclass
works. Overriding `prepare_window()` means the unimplemented model hook is never
called at all. That matters: this survives `git pull`, and when upstream refactors
it fails loudly with an AttributeError rather than silently doing the wrong thing,
which is exactly what a stale diff does.

INTENDED USE: low-denoise upscale passes only. At low denoise every window starts
from the same upscaled base, so coherence comes from the input rather than from
attention spanning the clip. At full denoise each window invents its own content
and they disagree. Attach this on stages 2 and 3 of an upscale ladder, never on the
pass that decides structure.
"""

import dataclasses
import logging

from comfy.context_windows import (
    ContextFuseMethods,
    ContextSchedules,
    IndexListContextHandler,
    IndexListContextWindow,
    WindowingState,
    create_prepare_sampling_wrapper,
    get_matching_context_schedule,
    get_matching_fuse_method,
)
from comfy_api.latest import io

from .common import (
    AUDIO_LATENT_FPS,
    AUDIO_T_DIM,
    FPS,
    FRAMES_PER_GROUP,
    FRAME_BASE,
    LATENTS_PER_GROUP,
    LATENT_BASE,
    VIDEO_T_DIM,
)


def _audio_index_at(n, total_v, total_a):
    """Audio-latent index at video-latent boundary `n`.

    The VAE is 2 latents for the first 5 frames, then 5 latents per 17. Inverting
    that gives frames(n) = 5 + 17*(n-2)/5, which is EXACT at every on-grid boundary
    and interpolates in between. Then audio_t = frames/24*40.

    Boundaries are converted independently and subtracted rather than converting a
    window LENGTH, because audio_t = round(frames/24*40) is not additive -- the same
    correction MMH3ConcatAV needed.
    """
    if total_v <= 0 or total_a <= 0:
        return 0
    if n <= 0:
        return 0
    if n >= total_v:
        return total_a
    frames = FRAME_BASE + FRAMES_PER_GROUP * (n - LATENT_BASE) / float(LATENTS_PER_GROUP)
    idx = int(round(max(0.0, frames) / FPS * AUDIO_LATENT_FPS))
    return max(0, min(total_a, idx))


class MMH3WindowingState(WindowingState):
    """WindowingState that gives each modality its OWN temporal dim."""

    def prepare_window(self, window, model):
        if not self.is_multimodal or len(self.latents) < 2:
            return window

        video, audio = self.latents[0], self.latents[1]
        total_v = int(video.shape[VIDEO_T_DIM])
        total_a = int(audio.shape[AUDIO_T_DIM])
        idx = list(window.index_list)
        if not idx or total_v <= 0 or total_a <= 0:
            return window

        # contiguous span; looped/wrapping schedules are rejected by the node
        a0 = _audio_index_at(idx[0], total_v, total_a)
        a1 = _audio_index_at(idx[-1] + 1, total_v, total_a)
        if a1 <= a0:
            a1 = min(total_a, a0 + 1)
        audio_indices = list(range(a0, a1))

        ratio = total_a / float(total_v)
        audio_window = IndexListContextWindow(
            audio_indices, dim=AUDIO_T_DIM, total_frames=total_a,
            context_overlap=max(0, int(round(window.context_overlap * ratio))))

        return IndexListContextWindow(
            idx, dim=VIDEO_T_DIM, total_frames=total_v,
            modality_windows={1: audio_window},
            context_overlap=window.context_overlap)


class MMH3ContextHandler(IndexListContextHandler):
    def _build_window_state(self, x_in, conds, model):
        st = super()._build_window_state(x_in, conds, model)
        return MMH3WindowingState(
            **{f.name: getattr(st, f.name) for f in dataclasses.fields(st)})

    def _apply_freenoise(self, noise, conds, seed):
        # The stock multimodal path shuffles every modality on the primary dim, which
        # is the same stereo-axis bug. Not worth reimplementing: freenoise exists to
        # improve blending between windows, and on a low-denoise pass there is very
        # little noise left to shuffle.
        shapes = self._get_latent_shapes(conds)
        if shapes is not None and len(shapes) > 1:
            logging.info("[MMH3ContextWindows] freenoise skipped for the AV latent")
            return noise
        return super()._apply_freenoise(noise, conds, seed)


def _snap_grid(n):
    """Snap DOWN to the 5j+2 latent grid, minimum 2."""
    n = int(n)
    if n < LATENT_BASE:
        return LATENT_BASE
    return LATENTS_PER_GROUP * ((n - LATENT_BASE) // LATENTS_PER_GROUP) + LATENT_BASE


class MMH3ContextWindows(io.ComfyNode):
    """Sliding-window sampling over the video latent, with audio windowed correctly."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ContextWindows",
            display_name="MiniMax H3 Context Windows",
            category="MMH3Tools",
            description=(
                "Sample a long AV latent in overlapping windows. FOR LOW-DENOISE PASSES "
                "ONLY -- at full denoise each window invents its own content and they "
                "disagree. Windows snap to the 5j+2 grid; audio is windowed on its own "
                "temporal axis."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "context_length", default=17, min=7, max=512, step=5,
                    tooltip="Window size in VIDEO LATENTS, snapped down to 5j+2 (7, 12, 17, "
                            "22...). 17 latents is 58 frames, ~2.4s. The model only ever saw "
                            "5j+2 clip lengths, so an off-grid window is off-distribution.",
                ),
                io.Int.Input(
                    "context_overlap", default=5, min=0, max=256, step=5,
                    tooltip="Overlap in video latents, snapped to a multiple of 5 (17 frames "
                            "each). More overlap costs passes but hides seams.",
                ),
                io.Combo.Input(
                    "fuse_method", options=ContextFuseMethods.LIST_STATIC,
                    default=ContextFuseMethods.PYRAMID,
                    tooltip="How overlapping windows are blended. Pyramid weights the centre "
                            "of each window most, which is usually what you want.",
                ),
                io.Combo.Input(
                    "context_schedule",
                    options=[ContextSchedules.STATIC_STANDARD, ContextSchedules.UNIFORM_STANDARD],
                    default=ContextSchedules.STATIC_STANDARD,
                    tooltip="Looped and batched schedules are not offered: they can emit "
                            "non-contiguous or wrapping windows, which the audio mapping "
                            "cannot express as a time span.",
                ),
                io.Int.Input("context_stride", default=1, min=1, max=32, step=1,
                             tooltip="Uniform schedules only."),
            ],
            outputs=[io.Model.Output(display_name="model"),
                     io.String.Output(display_name="label")],
        )

    @classmethod
    def execute(cls, model, context_length, context_overlap, fuse_method,
                context_schedule, context_stride) -> io.NodeOutput:
        length = _snap_grid(context_length)
        overlap = max(0, (int(context_overlap) // LATENTS_PER_GROUP) * LATENTS_PER_GROUP)
        overlap = min(overlap, max(0, length - LATENTS_PER_GROUP))

        notes = []
        if length != int(context_length):
            notes.append("context_length %d -> %d (5j+2 grid)" % (int(context_length), length))
        if overlap != int(context_overlap):
            notes.append("context_overlap %d -> %d" % (int(context_overlap), overlap))

        m = model.clone()
        m.model_options["context_handler"] = MMH3ContextHandler(
            context_schedule=get_matching_context_schedule(context_schedule),
            fuse_method=get_matching_fuse_method(fuse_method),
            context_length=length,
            context_overlap=overlap,
            context_stride=context_stride,
            closed_loop=False,
            dim=VIDEO_T_DIM,
            freenoise=False,
            # prepends an anchor frame to every non-zero window, which would push
            # each one to 5j+3 latents -- off the only grid the model has seen
            causal_window_fix=False,
        )
        create_prepare_sampling_wrapper(m)

        frames = FRAMES_PER_GROUP * ((length - LATENT_BASE) // LATENTS_PER_GROUP) + FRAME_BASE
        ov_frames = FRAMES_PER_GROUP * (overlap // LATENTS_PER_GROUP)
        label = ("window %d latents (%d frames, %.2fs), overlap %d (%d frames)"
                 % (length, frames, frames / float(FPS), overlap, ov_frames))
        for n in notes:
            label += "\n  ! " + n
            logging.info("[MMH3ContextWindows] %s", n)
        logging.info("[MMH3ContextWindows] " + label.splitlines()[0])
        return io.NodeOutput(m, label)
