"""Cutting AV latents apart: trim by span, split into modalities.

Both exist because H3's two streams are NOT sliceable in parallel. Video is
`[B,24,T,h,w]` on dim 2, audio `[B,32,2,T40]` on dim 3, and the conversion between
them is `audio_t = round(frames / 24 * 40)`, which is **not additive**. Trimming n
video latents and `round(n/24*40)` audio latents drifts a little further out of sync
every time you do it -- inaudibly at first, then as lip-sync that slides.

So both nodes convert BOUNDARIES independently and subtract, via `_audio_index_at`,
which is exact at every on-grid boundary. That is the same correction MMH3ConcatAV and
the context-window mapping already make; this just exposes it on its own.

WHY THESE WERE MISSING. `MMH3ConcatAV` could already trim, but only B's head and only
while joining, so a single latent could not be cut at all. And `MMH3FindDivergence`
exists to report how many frames a continuation reproduces -- with nowhere to send that
number except `MMH3JoinAV`, in pixel space, after a decode. MMH3TrimAV closes that loop
in latent space.
"""

import logging

import torch

from comfy_api.latest import io

from .common import (
    AUDIO_T_DIM,
    CANVAS_MULTIPLE,
    FPS,
    LATENTS_PER_GROUP,
    LATENT_BASE,
    VAE_SPATIAL,
    VIDEO_T_DIM,
    frames_to_audio_t,
    latents_to_frames,
    on_grid,
    pack_av,
    unpack_av,
)
from .nodes_windows import _audio_index_at


def _grid_note(n):
    """What a kept/dropped latent count means for the 5j+2 grid."""
    if on_grid(n):
        return "on the 5j+2 grid"
    rem = (n - LATENT_BASE) % LATENTS_PER_GROUP
    return ("OFF grid by %d -- decoding this directly will misalign the VAE's 17-frame "
            "chunking; %d or %d are the neighbours"
            % (rem, n - rem, n - rem + LATENTS_PER_GROUP))


class MMH3TrimAV(io.ComfyNode):
    """Drop latents from either end of an AV latent, audio and masks included.

    The value is honoured as given rather than snapped, for the same reason
    MMH3ConcatAV honours its trim: which count you want depends on what happens next.
    A trim that leaves the result on the 5j+2 grid is what you need before decoding;
    a trim that removes an exact overlap is what you need before joining. Those are
    different numbers and no single snap serves both. The report says which you got.

    Audio is cut by converting the two boundaries independently and subtracting, never
    by scaling the dropped count -- `round(frames / 24 * 40)` is not additive, so the
    naive version drifts.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TrimAV",
            display_name="MiniMax H3 Trim AV",
            category="MMH3Tools",
            description=(
                "Drop video latents from the head and/or tail of an H3 AV latent, "
                "cutting the audio to match on its own axis. Wire "
                "MMH3 Find Divergence's trim_frames here to cut a join in latent "
                "space instead of after a decode."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="H3 AV latent."),
                io.Int.Input(
                    "trim_head_latents", default=0, min=0, max=4096, step=1,
                    tooltip="Video latents dropped from the START. 5m removes an exact "
                            "overlap; 5m+2 keeps the remainder on the 5j+2 grid. The "
                            "value is used as given and the report says which you got.",
                ),
                io.Int.Input(
                    "trim_tail_latents", default=0, min=0, max=4096, step=1,
                    tooltip="Video latents dropped from the END.",
                ),
                io.Boolean.Input(
                    "carry_masks", default=True, optional=True,
                    tooltip="Apply the same cuts to noise_mask if the latent has one. "
                            "A mask describes the UNTRIMMED latent, so it must take the "
                            "identical computed cut or it stops lining up with the "
                            "content it describes.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="kept_latents"),
                io.Int.Output(display_name="kept_frames"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, latent, trim_head_latents, trim_tail_latents,
                carry_masks=True) -> io.NodeOutput:
        video, audio = unpack_av(latent, "latent")
        total_v = int(video.shape[VIDEO_T_DIM])
        total_a = int(audio.shape[AUDIO_T_DIM]) if audio is not None else 0

        head = max(0, int(trim_head_latents))
        tail = max(0, int(trim_tail_latents))
        if head + tail > total_v - LATENT_BASE:
            # refuse rather than silently keeping a stub: a 1-latent AV latent is not
            # decodable and the failure would surface far from here
            raise ValueError(
                "trimming %d + %d from %d latents leaves %d, below the %d-latent "
                "minimum. Nothing downstream can decode that."
                % (head, tail, total_v, total_v - head - tail, LATENT_BASE))

        keep_lo, keep_hi = head, total_v - tail
        v = video[:, :, keep_lo:keep_hi, :, :].contiguous()

        a = None
        a0 = a1 = 0
        if audio is not None:
            # boundaries converted INDEPENDENTLY then subtracted -- see module docstring
            a0 = _audio_index_at(keep_lo, total_v, total_a)
            a1 = _audio_index_at(keep_hi, total_v, total_a)
            if a1 <= a0:
                a1 = min(total_a, a0 + 1)
            a = audio[:, :, :, a0:a1].contiguous()

        out = pack_av(latent, v, a if a is not None else audio)

        mask = latent.get("noise_mask")
        if mask is not None:
            if carry_masks:
                vm, am = unpack_av({"samples": mask}, "noise_mask", allow_video_only=True)
                vm = vm[:, :, keep_lo:keep_hi, :, :].contiguous()
                if am is not None:
                    am = am[:, :, :, a0:a1].contiguous()
                    out["noise_mask"] = pack_av({}, vm, am)["samples"]
                else:
                    out["noise_mask"] = vm
            else:
                out.pop("noise_mask", None)

        kept = keep_hi - keep_lo
        frames = latents_to_frames(kept)
        report = ("kept %d of %d latents (%d frames, %.2fs), audio %d of %d\n  %s"
                  % (kept, total_v, frames, frames / float(FPS),
                     (a1 - a0) if audio is not None else 0, total_a, _grid_note(kept)))
        # NOTE the rule INVERTS relative to MMH3ConcatAV. Trimming a single latent:
        #   5m    -> 5(j-m)+2   ON grid, and removes an exact overlap
        #   5m+2  -> 5(j-m)     OFF grid
        # In ConcatAV it is the other way round, because there the constraint is on the
        # JOINED total, not on the piece being cut. Same arithmetic, different subject.
        for name, k in (("head", head), ("tail", tail)):
            if not k:
                continue
            rem = k % LATENTS_PER_GROUP
            report += "\n  %s trim %d: %s" % (name, k,
                "removes an exact overlap and keeps the result on grid" if rem == 0
                else "takes the result OFF grid (5m would keep it on)" if rem == LATENT_BASE
                else "is neither 5m nor 5m+2")
        logging.info("[MMH3TrimAV] " + report.splitlines()[0])
        return io.NodeOutput(out, kept, frames, report)


class MMH3SplitAV(io.ComfyNode):
    """Pull an AV latent apart into its video and audio halves.

    The inverse of MMH3PackAV, and the shapes match it exactly, so split -> operate ->
    pack round-trips. Without this, a packed latent could only ever be consumed whole:
    carrying stage 1's audio forward through an upscale ladder was a matter of
    discipline -- never wiring the sampler's audio anywhere -- rather than something
    the graph could express.

    The audio output is a plain `[B,32,2,T40]` latent. Note its temporal axis is dim 3,
    not dim 2: dim 2 is the stereo pair. Generic latent nodes that assume one temporal
    dim will stack it on the stereo axis and give you four channels of unchanged
    duration instead of a longer clip, silently.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SplitAV",
            display_name="MiniMax H3 Split AV",
            category="MMH3Tools",
            description=(
                "Split an H3 AV latent into its plain video and audio latents. The "
                "inverse of MMH3 Pack AV; the shapes round-trip."
            ),
            inputs=[io.Latent.Input("latent", tooltip="H3 AV latent.")],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
                io.Int.Output(display_name="video_latents"),
                io.Int.Output(display_name="audio_latents"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        video, audio = unpack_av(latent, "latent", allow_video_only=True)
        vt = int(video.shape[VIDEO_T_DIM])
        at = int(audio.shape[AUDIO_T_DIM]) if audio is not None else 0

        if audio is None:
            audio = torch.zeros([video.shape[0], 32, 2, frames_to_audio_t(
                latents_to_frames(vt))], dtype=video.dtype, device=video.device)
            note = "\n  ! input carried no audio; emitted silence sized to the video"
        else:
            note = ""

        frames = latents_to_frames(vt)
        expected = frames_to_audio_t(frames)
        if at and at != expected:
            note += ("\n  ! audio is %d latents but %d video latents (%d frames) imply "
                     "%d -- they were not built together" % (at, vt, frames, expected))

        report = ("video %d latents (%d frames, %.2fs) | audio %d latents (%.2fs)%s"
                  % (vt, frames, frames / float(FPS),
                     int(audio.shape[AUDIO_T_DIM]),
                     int(audio.shape[AUDIO_T_DIM]) / 40.0, note))
        logging.info("[MMH3SplitAV] " + report.splitlines()[0])
        return io.NodeOutput({"samples": video.contiguous()},
                             {"samples": audio.contiguous()},
                             vt, int(audio.shape[AUDIO_T_DIM]), report)


class MMH3OutpaintLatent(io.ComfyNode):
    """Zero-pad an AV latent spatially and emit the matching feathered denoise mask.

    ZEROS, not encoded padding. Encoding padded pixels bakes STRUCTURED content -- black
    or grey encodes to a non-zero latent the model reads as "something is here" and tries
    to preserve, which is where the black-edge artefact comes from. A zero margin is the
    same empty substrate a from-scratch generation starts from: nothing to preserve, so
    the sampler simply generates into it.

    THE FEATHER RAMPS INWARD, into the source. Feathering outward into the zeros margin
    would blend toward empty and muddy the seam. Ramping inward means the outermost band
    of the ORIGINAL is partially regenerated, which is what actually hides the join.

    WHAT THE FEATHER DOES AND DOES NOT DO HERE. The mask reaches the model twice:

      * the sampler's own `x*mask + orig*(1-mask)` blends the LATENT continuously --
        the feather works fully here, and this is what softens the seam;
      * `mask_row_targets` (#15375) binarises at 0.5 per 2x2 patch to pick a per-row
        AdaLN timestep, so the model's TREATMENT steps hard at the 0.5 contour.

    Content is a gradient; treatment is a step. Expect the feather to help, but less than
    the same width would on LTX, whose mask stays continuous throughout. If the contour
    shows, pad wider than needed and crop back so the step lands outside the final frame.

    Run this in a FULL-denoise pass. A low-denoise refinement adds too little noise to a
    bare margin for anything to appear there.

    Audio is untouched and its mask is all-preserve: this reframes the picture, it does
    not touch the track. Needs #15375 like every masking node here -- without per-row
    masking the mask has no effect at all, and the result is just a differently-framed
    full regeneration.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3OutpaintLatent",
            display_name="MiniMax H3 Outpaint Latent",
            category="MMH3Tools",
            description=(
                "Zero-pad an H3 AV latent spatially and attach a feathered denoise mask, "
                "so a full-denoise pass generates the margin while keeping the original. "
                "Padding is in pixels, snapped to 32."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="H3 AV latent, or a plain video latent."),
                io.Int.Input("left", default=0, min=0, max=4096, step=CANVAS_MULTIPLE),
                io.Int.Input("right", default=0, min=0, max=4096, step=CANVAS_MULTIPLE),
                io.Int.Input("top", default=0, min=0, max=4096, step=CANVAS_MULTIPLE),
                io.Int.Input("bottom", default=0, min=0, max=4096, step=CANVAS_MULTIPLE),
                io.Int.Input(
                    "feather", default=64, min=0, max=1024, step=16,
                    tooltip="Pixels of ramp reaching INWARD into the original, on padded "
                            "sides only. The original's outer band is partially "
                            "regenerated, which is what hides the join. 0 gives a hard "
                            "edge. Under 16px rounds to nothing.",
                ),
                # appended last -- widget order is positional, so these must stay here
                io.Int.Input("crop_left", default=0, min=0, max=4096,
                             step=CANVAS_MULTIPLE, optional=True,
                             tooltip="Pixels removed BEFORE padding. Wire MMH3 Reframe "
                                     "Pads' crop outputs. Cropping loses content "
                                     "outright -- nothing regenerates it."),
                io.Int.Input("crop_right", default=0, min=0, max=4096,
                             step=CANVAS_MULTIPLE, optional=True),
                io.Int.Input("crop_top", default=0, min=0, max=4096,
                             step=CANVAS_MULTIPLE, optional=True),
                io.Int.Input("crop_bottom", default=0, min=0, max=4096,
                             step=CANVAS_MULTIPLE, optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, latent, left, right, top, bottom, feather,
                crop_left=0, crop_right=0, crop_top=0, crop_bottom=0) -> io.NodeOutput:
        video, audio = unpack_av(latent, "latent", allow_video_only=True)

        notes = []
        # CROP FIRST, then pad. An orientation flip usually wants both -- trim the long
        # axis part of the way and grow the short one the rest -- and doing it in this
        # order means the pads describe the already-cropped frame, which is what
        # MMH3ReframePads computes.
        cr = {}
        for name, v in (("left", crop_left), ("right", crop_right),
                        ("top", crop_top), ("bottom", crop_bottom)):
            snapped = (int(v) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE
            if snapped != int(v):
                notes.append("crop_%s %d -> %d (32px canvas)" % (name, int(v), snapped))
            cr[name] = snapped // VAE_SPATIAL
        if any(cr.values()):
            _, _, _, ch, cw = video.shape
            y1, x1 = ch - cr["bottom"], cw - cr["right"]
            if y1 - cr["top"] < 2 or x1 - cr["left"] < 2:
                raise ValueError(
                    "cropping leaves %dx%d latent cells, which is nothing to work with."
                    % (max(0, y1 - cr["top"]), max(0, x1 - cr["left"])))
            video = video[:, :, :, cr["top"]:y1, cr["left"]:x1].contiguous()
            notes.append("cropped to %dx%d px before padding"
                         % (video.shape[4] * VAE_SPATIAL, video.shape[3] * VAE_SPATIAL))

        b, c, t, h, w = video.shape
        px = {}
        for name, v in (("left", left), ("right", right), ("top", top), ("bottom", bottom)):
            snapped = (int(v) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE
            if snapped != int(v):
                notes.append("%s %d -> %d (32px canvas)" % (name, int(v), snapped))
            px[name] = snapped
        if not any(px.values()) and not any(cr.values()):
            raise ValueError("nothing to do; every crop and pad is 0.")

        # /16 to latent. The 32px snap is what keeps the latent dims EVEN, which the
        # DiT's 2x2 patch grid requires -- an odd latent dimension fails deep in the
        # model with a broadcast error rather than here.
        lat = {k: v // VAE_SPATIAL for k, v in px.items()}
        nh = h + lat["top"] + lat["bottom"]
        nw = w + lat["left"] + lat["right"]
        if nh % 2 or nw % 2:
            raise ValueError(
                "padded latent would be %dx%d and the DiT's 2x2 patch grid needs both "
                "even. Pad in multiples of 32 pixels." % (nh, nw))

        out_v = torch.zeros([b, c, t, nh, nw], dtype=video.dtype, device=video.device)
        y0, x0 = lat["top"], lat["left"]
        out_v[:, :, :, y0:y0 + h, x0:x0 + w] = video

        # 1 = generate. Margin is 1, source is 0, and a ramp reaches INWARD from each
        # padded edge. Per-axis ramps are combined with max(), so a corner takes the
        # stronger of its two rather than their sum -- summing would over-regenerate
        # corners, which is where an outpaint is already weakest.
        m = torch.ones([nh, nw], dtype=torch.float32, device=video.device)
        m[y0:y0 + h, x0:x0 + w] = 0.0

        f_lat = max(0, int(feather) // VAE_SPATIAL)
        if f_lat:
            ys = torch.arange(h, device=video.device, dtype=torch.float32)
            xs = torch.arange(w, device=video.device, dtype=torch.float32)
            ramp = torch.zeros([h, w], dtype=torch.float32, device=video.device)
            if lat["top"]:
                ramp = torch.maximum(ramp, (1.0 - ys / f_lat).clamp(0, 1)[:, None].expand(h, w))
            if lat["bottom"]:
                ramp = torch.maximum(ramp, (1.0 - (h - 1 - ys) / f_lat).clamp(0, 1)[:, None].expand(h, w))
            if lat["left"]:
                ramp = torch.maximum(ramp, (1.0 - xs / f_lat).clamp(0, 1)[None, :].expand(h, w))
            if lat["right"]:
                ramp = torch.maximum(ramp, (1.0 - (w - 1 - xs) / f_lat).clamp(0, 1)[None, :].expand(h, w))
            m[y0:y0 + h, x0:x0 + w] = ramp

        vmask = m[None, None, None, :, :].expand(b, 1, t, nh, nw).contiguous()

        if audio is not None:
            out = pack_av(latent, out_v, audio)
            # all-preserve: reframing the picture must not disturb a finished track
            amask = torch.zeros([b, 1, 2, int(audio.shape[AUDIO_T_DIM])],
                                dtype=torch.float32, device=video.device)
            out["noise_mask"] = pack_av({}, vmask, amask)["samples"]
        else:
            out = dict(latent)
            out["samples"] = out_v
            out["noise_mask"] = vmask

        ramped = int(((m > 0.0) & (m < 1.0)).sum())
        over = int(((m > 0.0) & (m < 1.0) & (m >= 0.5)).sum())
        report = (
            "%dx%d -> %dx%d px (latent %dx%d -> %dx%d)\n"
            "  margin is %.1f%% of the frame | feather %d px = %d latent cells\n"
            "  %d ramped cells, %d of them above the 0.5 threshold and so treated as "
            "GENERATE by the per-row timestep. The latent blend is continuous throughout."
            % (w * VAE_SPATIAL, h * VAE_SPATIAL, nw * VAE_SPATIAL, nh * VAE_SPATIAL,
               w, h, nw, nh,
               100.0 * (1.0 - (h * w) / float(nh * nw)), int(feather), f_lat,
               ramped, over))
        for n in notes:
            report += "\n  ! " + n
        if not f_lat and int(feather):
            report += ("\n  ! feather %d px is under one latent cell (%d px) and rounds "
                       "to nothing" % (int(feather), VAE_SPATIAL))
        logging.info("[MMH3OutpaintLatent] " + report.splitlines()[0])
        return io.NodeOutput(out, nw * VAE_SPATIAL, nh * VAE_SPATIAL, report)
