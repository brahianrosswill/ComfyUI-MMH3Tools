"""Reference and keyframe conditioning built directly from latents.

The stock MiniMaxH3ReferenceToVideo node takes pixels and calls vae.encode().
For chained generation the previous chunk is already in latent space, so that
roundtrip is generation loss compounding once per hop.

KNOWN LIMITATION: these nodes do not register references with the tokenizer
(clip.tokenize(..., minimax_ref_items=...)), because feeding it pixels would
reintroduce the decode. The DiT still receives the latents, so pixel/motion/
identity continuity works; only the semantic path is skipped. Do not use
<Video k> prompt tags for a carried chunk.
"""

import logging

import torch

import comfy.utils
from comfy_api.latest import io

from .common import (
    AUDIO_T_DIM,
    CANVAS_MULTIPLE,
    append_cond_list,
    downscale_video_latent,
    empty_av_latent,
    frames_to_qwen_items,
    make_ref_block,
    set_cond_values,
    slice_av_tail,
    snap_latents,
    unpack_av,
)


def _decode_frames(vae, video_latent):
    """Decode a video latent to a [N, H, W, C] frame batch."""
    out = vae.decode(video_latent)
    if out.ndim == 5:  # [B, T, H, W, C] -> flatten batch/time
        out = out.reshape(-1, out.shape[-3], out.shape[-2], out.shape[-1])
    return out


class MMH3LatentToRef(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentToRef",
            display_name="MiniMax H3 Latent to Reference",
            category="MMH3Tools",
            description=(
                "Carry the tail of an H3 AV latent forward as a minimax_refs block, "
                "with no VAE roundtrip. Reference latents are re-injected every "
                "sampling step and never denoised."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent", tooltip="AV latent from the previous chunk"),
                io.Int.Input(
                    "carry_latents", default=12, min=2, max=512, step=5,
                    tooltip="Video latents carried from the tail, snapped down to the 5j+2 grid "
                            "(2, 7, 12, 17, 22...). 12 latents is ~39 frames / ~1.6s.",
                ),
                io.Boolean.Input(
                    "include_audio", default=True,
                    tooltip="Carry the matching audio tail so the block is kind='video_audio' "
                            "and A/V share a RoPE cursor origin.",
                ),
                io.Combo.Input(
                    "ref_downscale", options=["none", "2x", "4x"], default="none",
                    tooltip="Downscale the carried reference spatially. Reference tokens are "
                            "attended at EVERY step, so 2x cuts their cost ~4x. Latent-space "
                            "interpolation is approximate -- check identity survives.",
                ),
                io.Boolean.Input(
                    "carry_video", default=True,
                    tooltip="Off makes this an AUDIO-ONLY reference: voice and room tone carry, "
                            "but no video rows are added, so there is nothing for the model to "
                            "render back into the output. Pair with a keyframe, which supplies "
                            "position and appearance without being reproduced.",
                ),
                io.Int.Input(
                    "audio_latents", default=0, min=0, max=8192, step=40,
                    tooltip="Audio latents to carry, INDEPENDENT of the video carry. 40 per "
                            "second. 0 matches the video carry length. Video reference gets "
                            "reproduced into the output so you generally want it short, while "
                            "audio carries voice and room tone with no such penalty, so you "
                            "generally want it long -- hence decoupling them.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Int.Output(display_name="carried_frames"),
                io.Int.Output(
                    display_name="carried_latents",
                    tooltip="Actual latent count after snapping to the 5j+2 grid and "
                            "clamping to what the source has.",
                ),
            ],
        )

    @classmethod
    def execute(cls, conditioning, latent, carry_latents, include_audio, ref_downscale,
                carry_video=True, audio_latents=0) -> io.NodeOutput:
        video, audio = unpack_av(latent)
        want = snap_latents(min(carry_latents, video.shape[2]))
        v, a, frames, audio_t = slice_av_tail(video, audio, want)

        if include_audio and audio_latents > 0 and audio is not None:
            # audio tail sized independently of the video tail; the layout advances the
            # block's cursor by max(audio span, video span), so they need not agree
            at = min(int(audio_latents), int(audio.shape[AUDIO_T_DIM]))
            a, audio_t = audio[:, :, :, -at:].contiguous(), at

        if not include_audio:
            a, audio_t = None, 0
        if not carry_video and a is None:
            raise ValueError("carry_video is off and include_audio is off -- nothing to carry")

        if carry_video:
            factor = {"none": 1, "2x": 2, "4x": 4}[ref_downscale]
            v, lh, lw, used = downscale_video_latent(v, factor)
            if used != factor:
                logging.info("[MMH3LatentToRef] downscale %dx not valid for %dx%d latent, using %dx",
                             factor, v.shape[4] * used, v.shape[3] * used, used)
            carried = int(v.shape[2])
        else:
            v, lh, lw, carried = None, 0, 0, 0
            logging.info("[MMH3LatentToRef] audio-only reference (%d audio latents)", audio_t)

        block = make_ref_block(v, a, lh, lw, audio_t)
        return io.NodeOutput(append_cond_list(conditioning, "minimax_refs", [block]),
                             frames, carried)


class MMH3ReferenceFromLatent(io.ComfyNode):
    """Full ref2va conditioning builder fed by a latent instead of pixels.

    Unlike MMH3LatentToRef, this owns the tokenizer call, so the carried chunk is
    registered as a real <Video 1> and prompts can legitimately use the
    [video continuation] task type and reference it by label.

    The DiT still receives PRISTINE latents. The only decode is a short 2fps
    subsample handed to Qwen3-VL for semantics -- for a ~1.6s carry that is 3-4
    frames, and no generation loss enters the sampling path.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ReferenceFromLatent",
            display_name="MiniMax H3 Reference from Latent",
            category="MMH3Tools",
            description=(
                "Build ref2va conditioning from a previous chunk's AV latent, registering "
                "it with the tokenizer so <Video 1> resolves. Use for chained continuation."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Latent.Input("ref_latent", tooltip="AV latent from the previous chunk"),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input(
                    "length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count of the NEW chunk, snapped to the 17j+5 grid.",
                ),
                io.Int.Input(
                    "carry_latents", default=12, min=2, max=512, step=5,
                    tooltip="Video latents carried from the tail, snapped to the 5j+2 grid.",
                ),
                io.Boolean.Input("include_audio", default=True),
                io.Combo.Input("ref_downscale", options=["none", "2x", "4x"], default="none"),
                io.Boolean.Input(
                    "register_with_tokenizer", default=True,
                    tooltip="Decode a 2fps subsample of the carry for Qwen3-VL so <Video 1> "
                            "resolves. Disable to skip the decode entirely -- the DiT still "
                            "gets the latents, but do not use <Video k> tags in the prompt.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="carried_frames"),
                io.Int.Output(
                    display_name="carried_latents",
                    tooltip="Actual latent count after snapping to the 5j+2 grid and "
                            "clamping to what the source has.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, ref_latent, width, height, length, carry_latents,
                include_audio, ref_downscale, register_with_tokenizer) -> io.NodeOutput:
        video, audio = unpack_av(ref_latent)
        want = snap_latents(min(carry_latents, video.shape[2]))
        v, a, frames, audio_t = slice_av_tail(video, audio, want)

        if not include_audio:
            a, audio_t = None, 0

        ref_items = []
        if register_with_tokenizer:
            # Decode the whole (short) carry, then subsample -- the causal VAE needs
            # temporal context, so decoding scattered latent frames alone would artifact.
            decoded = _decode_frames(vae, v)
            qwen_frames, timestamps = frames_to_qwen_items(decoded)
            if a is not None:
                # the soundtrack's <Audio j> label is emitted before its <Video k>
                ref_items.append({"type": "audio"})
            ref_items.append({"type": "video", "data": qwen_frames, "timestamps": timestamps})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items) if ref_items \
            else clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)

        vv, lh, lw, _used = downscale_video_latent(v, {"none": 1, "2x": 2, "4x": 4}[ref_downscale])
        cond = append_cond_list(cond, "minimax_refs", [make_ref_block(vv, a, lh, lw, audio_t)])

        latent, _ = empty_av_latent(width, height, length)
        return io.NodeOutput(cond, latent, frames, int(v.shape[2]))


class MMH3ImageKeyframe(io.ComfyNode):
    """Inject a still image as a keyframe anchor, appending to existing conditioning.

    Fills the gap in the stock nodes: MiniMaxH3ReferenceToVideo has no keyframe
    inputs at all, so ref2va conditioning cannot carry a frame anchor. This appends
    to minimax_keyframes, so it composes with a reference build.

    Resizing and encoding happen here because keyframe rows share the TARGET spatial
    grid and cannot be downscaled -- a still at any other resolution fails deep in
    the model with a broadcast error.

    Does NOT register the image with the tokenizer, so <Picture N> will not resolve
    in prompt text. Use the stock MiniMaxH3ImageToVideo first_frame/last_frame inputs
    if you need the label; use this when you need a frame anchor alongside references,
    or at an index the stock nodes cannot express.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ImageKeyframe",
            display_name="MiniMax H3 Image Keyframe",
            category="MMH3Tools",
            description=(
                "Append a still image as a keyframe anchor. Resizes and encodes to the "
                "target grid internally. Composes with ref2va conditioning."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Image.Input("image", tooltip="Still to anchor. Only the first of a batch is used."),
                io.Vae.Input("vae", tooltip="The H3 VIDEO vae."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32,
                             tooltip="Generation width. The image is resized to this; keyframe "
                                     "rows share the target grid and cannot be downscaled."),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input(
                    "target_frame_count", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count of the chunk being generated. Wire from MMH3 Frame "
                            "Calculator - a 'last' anchor lands wrong if this disagrees.",
                ),
                io.Int.Input(
                    "frame_index", default=0, min=-1, max=3600, step=1,
                    tooltip="0 = first frame. -1 = last frame. Any other value is an INTERIOR "
                            "anchor, which stock ComfyUI rejects ('only first/last keyframe "
                            "anchors are supported') unless PackedLayout is patched. MiniMax's "
                            "own guide does list interior anchors as valid.",
                ),
                io.Combo.Input(
                    "resize", options=["auto", "stretch", "center crop"], default="auto",
                    tooltip="'auto' copies the stock node: stretch for a first-frame anchor "
                            "(geometry anchor), aspect-preserving centre crop otherwise "
                            "(follower).",
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, conditioning, image, vae, width, height, target_frame_count,
                frame_index, resize) -> io.NodeOutput:
        w = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        h = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        if (w, h) != (int(width), int(height)):
            logging.info("[MMH3ImageKeyframe] snapped %dx%d -> %dx%d", width, height, w, h)
        if image.shape[0] > 1:
            logging.info("[MMH3ImageKeyframe] %d images given; using the first",
                         int(image.shape[0]))

        index = int(target_frame_count) - 1 if frame_index == -1 else int(frame_index)
        if index != 0 and index != int(target_frame_count) - 1:
            logging.warning(
                "[MMH3ImageKeyframe] frame_index %d is an INTERIOR anchor. PackedLayout "
                "raises 'only first/last keyframe anchors are supported' unless patched.",
                index)

        crop = {"stretch": "disabled", "center crop": "center"}.get(
            resize, "disabled" if index == 0 else "center")

        px = image[:1, ..., :3].movedim(-1, 1)
        px = comfy.utils.common_upscale(px, w, h, "lanczos", crop).movedim(1, -1)
        z = vae.encode(px)

        # video VAEs return [B, C, T, h, w]; an image VAE would give [B, C, h, w]
        if z.ndim == 4:
            z = z.unsqueeze(2)
        z = z[:1, :, :1, :, :].contiguous()

        cond = append_cond_list(conditioning, "minimax_keyframes",
                                [{"resolved_frame_index": index, "latent": z}])
        cond = set_cond_values(cond, {"minimax_frame_count": int(target_frame_count)})
        return io.NodeOutput(cond)


class MMH3LatentKeyframe(io.ComfyNode):
    """Hard first/last frame anchor built from a latent frame.

    PackedLayout accepts keyframes and refs simultaneously, so this stacks with
    MMH3LatentToRef. Whether the ref2va checkpoint responds to 'cond' rows is
    the open experiment -- fl2va definitely does.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentKeyframe",
            display_name="MiniMax H3 Latent Keyframe",
            category="MMH3Tools",
            description=(
                "Anchor the first or last frame using one latent frame from another AV "
                "latent. Keyframe rows share the TARGET spatial grid, so the source must "
                "match the generation's width/height exactly. Only first/last are supported."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent", tooltip="Source AV latent (e.g. previous chunk)"),
                io.Int.Input(
                    "target_frame_count", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count of the chunk being generated. Must match the latent "
                            "you sample, or a 'last' anchor lands at the wrong position.",
                ),
                io.Combo.Input(
                    "position", options=["first", "last"], default="first",
                    tooltip="'first' anchors the new chunk's opening frame -- the continuation case.",
                ),
                io.Combo.Input(
                    "source_frame", options=["last", "first"], default="last",
                    tooltip="Which latent frame to take from the source.",
                ),
                io.Int.Input(
                    "target_width", default=0, min=0, max=16384, step=32, optional=True,
                    tooltip="Optional guard. Keyframe rows share the TARGET spatial grid and "
                            "CANNOT be downscaled (unlike references, which carry their own "
                            "latent_h/latent_w). A source at a different resolution therefore "
                            "fails deep in the model with an unhelpful broadcast error. Wire the "
                            "generation width/height here to catch it at this node instead. "
                            "0 skips the check.",
                ),
                io.Int.Input(
                    "target_height", default=0, min=0, max=16384, step=32, optional=True,
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, conditioning, latent, target_frame_count, position, source_frame,
                target_width=0, target_height=0) -> io.NodeOutput:
        # keyframes only need video, so a VAEEncode'd image/video works as an anchor
        video, _ = unpack_av(latent, "latent", allow_video_only=True)

        if target_width > 0 and target_height > 0:
            sw, sh = int(video.shape[4]) * 16, int(video.shape[3]) * 16
            if (sw, sh) != (int(target_width), int(target_height)):
                raise ValueError(
                    "Keyframe source is %dx%d but the target is %dx%d. Keyframe rows share the "
                    "target spatial grid and cannot be downscaled, so these must match exactly. "
                    "Resize before encoding, or generate at the source's resolution."
                    % (sw, sh, int(target_width), int(target_height))
                )
        z = video[:, :, -1:, :, :] if source_frame == "last" else video[:, :, :1, :, :]
        index = 0 if position == "first" else int(target_frame_count) - 1

        cond = append_cond_list(
            conditioning, "minimax_keyframes",
            [{"resolved_frame_index": index, "latent": z.contiguous()}],
        )
        cond = set_cond_values(cond, {"minimax_frame_count": int(target_frame_count)})
        return io.NodeOutput(cond)
