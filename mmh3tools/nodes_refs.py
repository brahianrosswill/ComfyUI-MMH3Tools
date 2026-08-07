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
import math

import torch

import comfy.utils
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import REF_IMAGE_SHORT_EDGE

from .common import (
    AUDIO_T_DIM,
    CANVAS_MULTIPLE,
    LATENTS_PER_GROUP,
    PATCH,
    VAE_SPATIAL,
    VIDEO_T_DIM,
    append_cond_list,
    downscale_video_latent,
    empty_av_latent,
    frames_to_qwen_items,
    latents_to_frames,
    make_ref_block,
    set_cond_values,
    slice_av_tail,
    snap_latents,
    step_frame_offsets,
    unpack_av,
)


def _encode_image_ref(vae, image, width, height, ref_image_size):
    """One still -> (tokenizer item, minimax_refs block, tokens-per-step).

    Sizing is the stock reference node's policy exactly -- scale DOWN only, aspect
    kept, snapped to 32 -- so a reference built here and one built by
    MiniMaxH3ReferenceToVideo are interchangeable.
    """
    h, w = int(image.shape[1]), int(image.shape[2])
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((int(width) * int(height)) / float(w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / float(min(w, h)))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

    px = image[:1, ..., :3].movedim(-1, 1)
    px = comfy.utils.common_upscale(px, tw, th, "lanczos", "disabled").movedim(1, -1)
    z = vae.encode(px)

    item = {"type": "image", "data": px}
    block = {"kind": "image", "latent_h": th // VAE_SPATIAL, "latent_w": tw // VAE_SPATIAL,
             "latent": z}
    tokens = (tw // VAE_SPATIAL // PATCH) * (th // VAE_SPATIAL // PATCH)
    return item, block, tokens


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
                            "gets the latents, but do not use <Video k> tags in the prompt. "
                            "Only affects the CARRY: still images below have no decode cost "
                            "and are always registered.",
                ),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9),
                    tooltip="Still references, registered as <Picture 1>..<Picture N> in "
                            "prompt order. This is the ONLY way to get a working <Picture N> "
                            "alongside a carried latent -- an appender node cannot, because "
                            "the LM has already run by the time a CONDITIONING exists.",
                ),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match", optional=True,
                    tooltip="'match' scales each still to the generation's pixel area; 'max' "
                            "uses a 2048px short edge for best identity. Reference tokens are "
                            "attended at every sampling step, so 'max' is several times more "
                            "expensive for the whole run.",
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
                include_audio, ref_downscale, register_with_tokenizer,
                ref_images=None, ref_image_size="match") -> io.NodeOutput:
        video, audio = unpack_av(ref_latent)
        want = snap_latents(min(carry_latents, video.shape[2]))
        v, a, frames, audio_t = slice_av_tail(video, audio, want)

        if not include_audio:
            a, audio_t = None, 0

        # Stills FIRST, matching the stock node's emission order, so <Picture N> numbering
        # is the same whichever node built the conditioning. The tokenizer counts items in
        # the order given, so reordering here silently renumbers every label.
        img_items, img_blocks, img_tokens = [], [], 0
        for img in (ref_images or {}).values():
            if img is None:
                continue
            item, block, tok = _encode_image_ref(vae, img, width, height, ref_image_size)
            img_items.append(item)
            img_blocks.append(block)
            img_tokens += tok

        ref_items = list(img_items)
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
        # same order as ref_items, so the DiT's block layout matches the labels
        cond = append_cond_list(cond, "minimax_refs",
                                img_blocks + [make_ref_block(vv, a, lh, lw, audio_t)])

        if img_blocks:
            logging.info("[MMH3ReferenceFromLatent] %d still%s registered as <Picture 1>..<Picture %d> "
                         "(%d tokens/step), carry %s<Video 1>",
                         len(img_blocks), "" if len(img_blocks) == 1 else "s", len(img_blocks),
                         img_tokens, "" if register_with_tokenizer else "NOT registered, no ")

        latent, _ = empty_av_latent(width, height, length)
        return io.NodeOutput(cond, latent, frames, int(v.shape[2]))


class MMH3ImageToRef(io.ComfyNode):
    """Append a still image as a REFERENCE block, not a keyframe.

    Fills the last hole in the matrix: latents could become refs or keyframes, and
    images could become keyframes, but there was no image -> refs path that appends.
    Stock MiniMaxH3ReferenceToVideo takes ref_images but BUILDS conditioning from
    clip+prompt, so it cannot add a still to conditioning that already exists -- which
    is exactly what you need to stack a reference face alongside carried latent refs.

    Unlike keyframes, reference blocks carry their OWN latent_h/latent_w, so this is
    free to resize; it is not locked to the target grid.

    KNOWN LIMITATION: does not register the image with the tokenizer, so <Picture N>
    will not resolve in prompt text. An appender structurally cannot -- the LM has
    already run by the time a CONDITIONING exists. For identity work that is usually
    fine: the DiT gets the latents, which is what carries appearance. Use the stock
    node when the prompt genuinely needs to refer to the picture by label.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ImageToRef",
            display_name="MiniMax H3 Image to Reference",
            category="MMH3Tools",
            description=(
                "Append a still image to minimax_refs. Composes with carried latent "
                "references and with keyframes, which the stock reference node cannot do."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Image.Input("image", tooltip="Only the first of a batch is used."),
                io.Vae.Input("vae", tooltip="The H3 VIDEO vae."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32,
                             tooltip="Generation width, used only by ref_image_size "
                                     "'match' to scale the reference to the same pixel "
                                     "area. The reference is NOT forced to this size."),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match",
                    tooltip="'match' scales the reference (down only, aspect kept) to the "
                            "generation's pixel area. 'max' uses a 2048px short edge for "
                            "best identity fidelity -- MiniMax's recommendation for faces, "
                            "but reference tokens are attended at EVERY sampling step, and "
                            "under context windows at every step of every window.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, conditioning, image, vae, width, height, ref_image_size) -> io.NodeOutput:
        if image.shape[0] > 1:
            logging.info("[MMH3ImageToRef] %d images given; using the first",
                         int(image.shape[0]))
        h, w = int(image.shape[1]), int(image.shape[2])
        _item, block, tok = _encode_image_ref(vae, image, width, height, ref_image_size)
        cond = append_cond_list(conditioning, "minimax_refs", [block])

        label = ("%dx%d -> %dx%d ref (%s), %d tokens per sampling step"
                 % (w, h, block["latent_w"] * VAE_SPATIAL, block["latent_h"] * VAE_SPATIAL,
                    ref_image_size, tok))
        logging.info("[MMH3ImageToRef] " + label)
        return io.NodeOutput(cond, label)


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


class MMH3LatentToKeyframes(io.ComfyNode):
    """Pin the previous chunk's tail as a RUN of positioned keyframes.

    This is the chaining path. MMH3LatentToRef puts the same tail in a reference
    block, which is also positioned -- refs advance a cursor and the target begins
    after them -- but that advance is the problem: a 39 frame carry pushes target
    frame 0 to text_len + 65, putting 65 position units between the end of the
    prompt and the start of the clip. Keyframes are anchored to the target origin
    itself and cost nothing, measured on the real PackedLayout:

        carry as video_audio ref     target origin text_len + 65
        carry as keyframes           target origin text_len +  0

    Token cost is near identical (12226 rows vs 12096 for a 12 step carry), so the
    choice is purely about where the clip sits relative to its prompt.

    HEAD ANCHORING. The run occupies target frames 0..span-1, so the first `span`
    frames of the output reproduce the tail and must be trimmed before joining --
    wire pinned_frames into MMH3ConcatAV's trim. Negative indices would avoid that
    waste, but cond_t = text_len + FRAME_RESCALE*p goes BELOW text_len for negative
    p and collides with text token positions, which is off-distribution. Head
    anchoring spends ~11% of a chunk at 22 frames, ~20% at 39.

    NO VAE. Motion-Context's equivalent takes IMAGE and re-encodes because it
    chains across separate runs. Chaining in-graph, the tail is already latent, so
    slicing steps is both free and lossless. A tail of 5m+2 steps off a 5j+2 clip
    starts at step 5(j-m) -- always phase 0 -- so the slice is exactly what a fresh
    encode of those frames would produce.

    VIDEO ONLY. Audio stays on the reference path for now; moving it onto the
    target timeline needs its own position rewrite, since an audio ref reintroduces
    the very cursor advance this node exists to avoid.

    Requires the interior-anchor patch (mmh3tools/patch_layout.py), which is applied
    at import and self-tested; this node refuses to run if it did not take.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentToKeyframes",
            display_name="MiniMax H3 Latent to Keyframes",
            category="MMH3Tools",
            description=(
                "Pin the tail of a previous chunk as consecutive keyframe anchors, so a "
                "new chunk continues from it. Unlike a reference carry this adds no "
                "distance between the prompt and the clip. Video only; trim the pinned "
                "head before joining."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent", tooltip="AV latent from the previous chunk."),
                io.Int.Input(
                    "carry_latents", default=7, min=2, max=512, step=5,
                    tooltip="Video latents pinned from the tail, snapped down to the 5j+2 "
                            "grid (2, 7, 12, 17...). 7 latents is 22 frames, the value "
                            "prior art settles on; 12 is 39 frames and pins harder at the "
                            "cost of a fifth of the chunk.",
                ),
                io.Int.Input(
                    "target_frame_count", default=192, min=5, max=3600, step=17,
                    tooltip="Frame count of the chunk being generated. Wire from MMH3 Frame "
                            "Calculator.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Int.Output(display_name="pinned_frames"),
                io.Int.Output(display_name="pinned_latents"),
            ],
        )

    @classmethod
    def execute(cls, conditioning, latent, carry_latents, target_frame_count) -> io.NodeOutput:
        from .patch_layout import MMH3_KEY, is_applied, status
        if not is_applied():
            raise RuntimeError(
                "MMH3LatentToKeyframes needs the interior keyframe anchor patch, which "
                "is not active (%s). Stock PackedLayout accepts only first/last anchors, "
                "so a run of them cannot be expressed. Check the startup log."
                % status())

        video, _ = unpack_av(latent, "latent", allow_video_only=True)
        total = int(video.shape[VIDEO_T_DIM])
        want = snap_latents(min(int(carry_latents), total))

        span = latents_to_frames(want)
        if span >= int(target_frame_count):
            raise ValueError(
                "Pinning %d frames into a %d frame chunk leaves nothing to generate. "
                "The pinned run has to be a small fraction of the timeline."
                % (span, int(target_frame_count)))

        # A 5m+2 tail off a 5j+2 clip starts at step 5(j-m), i.e. phase 0. If the
        # source is off-grid that no longer holds and the spans would disagree with
        # the positions we write, so derive the phase instead of assuming it.
        phase = (total - want) % LATENTS_PER_GROUP
        offsets = step_frame_offsets(want, phase)

        tail = video[:, :, total - want:, :, :]
        keyframes = [
            {"resolved_frame_index": 0,          # always legal; stock validates this
             MMH3_KEY: int(p),                   # the real anchor, applied by the patch
             "latent": tail[:, :, k:k + 1, :, :].contiguous()}
            for k, p in enumerate(offsets)
        ]

        cond = append_cond_list(conditioning, "minimax_keyframes", keyframes)
        cond = set_cond_values(cond, {"minimax_frame_count": int(target_frame_count)})

        logging.info("[MMH3LatentToKeyframes] pinned %d latents = %d frames at %s "
                     "(phase %d) of a %d frame chunk",
                     want, span, offsets, phase, int(target_frame_count))
        return io.NodeOutput(cond, span, want)
