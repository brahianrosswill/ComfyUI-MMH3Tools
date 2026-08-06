"""Calculators and inspection.

Follows the LTXAVTools convention: concise typed outputs plus a short `label`
string, rather than a verbose info block.

Difference from LTXDimensionCalculator: LTX emitted a fixed `width_half` /
`height_half` pair for its two-stage pipeline. H3 has no second stage -- the
secondary pair here is the REFERENCE size, driven by a downscale factor and
snapped to what the patch grid supports.
"""

import logging
import math

from comfy_api.latest import io

from .common import (
    AUDIO_T_DIM,
    BASE_SHORT_EDGE,
    CANVAS_MULTIPLE,
    FPS,
    FRAMES_PER_GROUP,
    FRAME_BASE,
    MAX_PIXELS,
    PATCH,
    VAE_SPATIAL,
    VIDEO_T_DIM,
    frames_to_audio_t,
    frames_to_latents,
    latents_to_frames,
    on_grid,
    snap_downscale,
    supported_downscale_factors,
    unpack_av,
)


class MMH3FrameCalculator(io.ComfyNode):
    """Seconds -> frame count on the model's 17j+5 grid.

    At 24fps achievable durations are discrete. Solving 24s = 5 (mod 17) gives
    s = 8 (mod 17), so 8.000s (192 frames) is the only whole-second duration in
    the 4-15s supported range.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3FrameCalculator",
            display_name="MMH3 Frame Calculator",
            category="MMH3Tools",
            description="Duration in seconds -> frame count on the 17j+5 grid, with the "
                        "video and audio latent counts it implies.",
            inputs=[
                io.Float.Input("seconds", default=5.0, min=0.2, max=150.0, step=0.01,
                               tooltip="Snapped to the nearest achievable duration. Only 8.000s "
                                       "is a whole second within the 4-15s trained range."),
                io.Combo.Input("rounding", options=["nearest", "up", "down"], default="nearest"),
            ],
            outputs=[
                io.Int.Output(display_name="frame_count"),
                io.Int.Output(display_name="latent_frames"),
                io.Int.Output(display_name="audio_latent_frames"),
                io.Float.Output(display_name="actual_seconds"),
            ],
        )

    @classmethod
    def execute(cls, seconds, rounding) -> io.NodeOutput:
        target = seconds * FPS
        j = int(math.floor((target - FRAME_BASE) / FRAMES_PER_GROUP))
        lo = FRAMES_PER_GROUP * max(0, j) + FRAME_BASE
        if lo > target:
            lo = FRAME_BASE
        hi = lo if lo >= target else lo + FRAMES_PER_GROUP

        if rounding == "up":
            f = hi
        elif rounding == "down":
            f = lo
        else:
            f = lo if (target - lo) <= (hi - target) else hi

        return io.NodeOutput(f, frames_to_latents(f), frames_to_audio_t(f), f / FPS)


# ---------------------------------------------------------------------------
# Resolution presets  (mirrors LTXAVTools' calculator, on H3's 32px grid)
# ---------------------------------------------------------------------------
RATIOS = [
    (21, 9, "21:9 - ultrawide, cinematic", "9:21 - ultrawide portrait"),
    (16, 9, "16:9 - YouTube, HD, TV", "9:16 - TikTok, Reels, Shorts"),
    (3, 2, "3:2 - photography, DSLR", "2:3 - portrait photo"),
    (4, 3, "4:3 - classic TV, monitor", "3:4 - tablet portrait"),
    (1, 1, "1:1 - square, Instagram", "1:1 - square, Instagram"),
]
MEGAPIXELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.5, 2.0]


def _snap32(x):
    return max(CANVAS_MULTIPLE, int(round(x / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def native_canvas(rl, rs):
    """The model's own adapt_canvas(): 768 short edge, capped at 768*1344, rounded to 32."""
    r = rl / rs
    nom_w, nom_h = (BASE_SHORT_EDGE * r, float(BASE_SHORT_EDGE))
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return _snap32(nom_w), _snap32(nom_h)


def build_options(ratio_long, ratio_short, landscape):
    """Resolutions for one ratio, smallest first. '[native]' marks the trained canvas."""
    r = ratio_long / max(ratio_short, 1)
    nat = native_canvas(ratio_long, ratio_short)
    seen, out = set(), []
    entries = []
    for mp in MEGAPIXELS:
        h = _snap32(math.sqrt(mp * 1e6 / r))
        entries.append((_snap32(h * r), h))
    entries.append(nat)
    for lw, lh in sorted(set(entries), key=lambda t: t[0] * t[1]):
        w, h = (lw, lh) if landscape else (lh, lw)
        tag = "%dx%d  %.2fMP" % (w, h, w * h / 1e6)
        if (lw, lh) == nat:
            tag += "  [native]"
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


# Declared options must be the FULL union across every ratio and orientation, because
# the JS narrows options.values client-side but ComfyUI validates the submitted value
# against what Python declared. Declaring only the 16:9 landscape list makes every
# other choice fail with "Some input values are not available for this node" - a
# validation error that never reaches the console.
_DEFAULT_RATIOS = ([lab for _, _, lab, _ in RATIOS] +
                   [p for _, _, lab, p in RATIOS if p not in [l for _, _, l, _ in RATIOS]])
_ALL_OPTS = []
for _rl, _rs, _, _ in RATIOS:
    for _land in (True, False):
        for _o in build_options(_rl, _rs, _land):
            if _o not in _ALL_OPTS:
                _ALL_OPTS.append(_o)
_DEFAULT_OPTS = build_options(16, 9, landscape=True)
_DEFAULT_OPT = next((o for o in _DEFAULT_OPTS if "[native]" in o), _DEFAULT_OPTS[0])
_DEFAULT_RATIO = _DEFAULT_RATIOS[1]


class MMH3DimensionCalculator(io.ComfyNode):
    """Generation dimensions plus a reference pair sized by a snapped downscale factor.

    Pixel dims snap to 32 (16x VAE spatial then a 2x2 patch). Latent dims are px/16
    and must stay EVEN, so a downscale factor is valid only when latent/f is an even
    integer on both axes -- the divisors of gcd(latent_h//2, latent_w//2). For
    1344x768 that is [1, 2, 3, 6]; 4 is NOT valid.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3DimensionCalculator",
            display_name="MMH3 Dimension Calculator",
            category="MMH3Tools",
            description="Snap generation dimensions to the 32px grid and derive a reference "
                        "size from a downscale factor, snapped to what the patch grid allows.",
            inputs=[
                io.Combo.Input("ratio", options=_DEFAULT_RATIOS, default=_DEFAULT_RATIO,
                               tooltip="Common aspect ratios and their typical uses."),
                io.Combo.Input("orientation", options=["Landscape", "Portrait"],
                               default="Landscape"),
                io.Combo.Input("resolution", options=_ALL_OPTS, default=_DEFAULT_OPT,
                               tooltip="Resolutions for the selected ratio, all multiples of 32. "
                                       "[native] marks the 768-short-edge canvas the model was "
                                       "trained on; larger costs tokens quadratically at "
                                       "every sampling step."),
                io.Int.Input("downscale_factor", default=2, min=1, max=32, step=1,
                             tooltip="Reference downscale. Snapped to the nearest factor that "
                                     "keeps both latent dims even; ties resolve gentler."),
                io.Boolean.Input("use_custom", default=False, optional=True,
                                 tooltip="Override the preset with custom_width/custom_height. "
                                         "Toggle-controlled so a bypassed upstream node cannot "
                                         "silently switch modes."),
                io.Int.Input("custom_width", default=0, min=0, max=16384, step=8, optional=True),
                io.Int.Input("custom_height", default=0, min=0, max=16384, step=8, optional=True),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="width_ref"),
                io.Int.Output(display_name="height_ref"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        # The JS narrows the ratio/resolution lists per selection, so a submitted value
        # can legitimately sit outside what any single declared list contains. execute()
        # parses "WxH" out of the string and tolerates anything well-formed.
        return True

    @classmethod
    def execute(cls, ratio, orientation, resolution, downscale_factor, use_custom=False,
                custom_width=0, custom_height=0) -> io.NodeOutput:
        if use_custom and custom_width > 0 and custom_height > 0:
            w, h = _snap32(custom_width), _snap32(custom_height)
        else:
            if use_custom:
                logging.info("[MMH3DimensionCalculator] use_custom on but dims <= 0 "
                             "(upstream bypassed?); falling back to the dropdown")
            # the option list already encodes orientation, so just parse it
            w, h = (int(x) for x in resolution.split()[0].split("x"))
        lw, lh = w // VAE_SPATIAL, h // VAE_SPATIAL

        f = snap_downscale(downscale_factor, lh, lw)
        rw, rh = (lw // f) * VAE_SPATIAL, (lh // f) * VAE_SPATIAL

        tok = (lw // PATCH) * (lh // PATCH)
        rtok = ((lw // f) // PATCH) * ((lh // f) // PATCH)
        label = "%dx%d -> ref %dx%d (%dx, %d%% tokens)" % (w, h, rw, rh, f, round(100.0 * rtok / max(1, tok)))
        if w * h > MAX_PIXELS:
            label += "  OVER CANVAS CAP"

        return io.NodeOutput(w, h, rw, rh, label)


class MMH3LatentInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentInfo",
            display_name="MMH3 Latent Info",
            category="MMH3Tools",
            description="Report AV latent shapes, implied frame count, and grid alignment.",
            inputs=[io.Latent.Input("latent")],
            outputs=[io.String.Output(display_name="info")],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        video, audio = unpack_av(latent)
        t = int(video.shape[VIDEO_T_DIM])
        frames = latents_to_frames(t)
        expected_audio = frames_to_audio_t(frames)
        actual_audio = int(audio.shape[AUDIO_T_DIM])

        lines = [
            "video latent : %s   (B,C,T,h,w)" % (tuple(video.shape),),
            "audio latent : %s   (B,C,stereo,T40)" % (tuple(audio.shape),),
            "pixel size   : %d x %d" % (video.shape[4] * VAE_SPATIAL, video.shape[3] * VAE_SPATIAL),
            "video T      : %d latents -> %d frames (%.3fs @ %dfps)" % (t, frames, frames / FPS, FPS),
            "audio T40    : %d (expected %d)%s" % (
                actual_audio, expected_audio,
                "" if actual_audio == expected_audio else "  <-- MISMATCH"),
            "5j+2 grid    : %s" % ("yes" if on_grid(t) else "NO -- off grid"),
            "downscale    : valid factors %s" % (
                ", ".join(str(x) for x in supported_downscale_factors(
                    int(video.shape[3]), int(video.shape[4])))),
            "noise_mask   : %s" % ("present" if latent.get("noise_mask") is not None else "none"),
        ]
        info = "\n".join(lines)
        print("[MMH3LatentInfo]\n" + info)
        return io.NodeOutput(info)

