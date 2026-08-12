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
    PATCH,
    VAE_SPATIAL,
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
            category="MMH3Tools/reference",
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
            category="MMH3Tools/reference",
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
            category="MMH3Tools/reference",
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
    to minimax_keyframes.

    But NOT alongside references on stock ComfyUI: `extra_conds` assigns
    `cond_video_latents` from keyframes and then assigns it again from references, so
    the references win and every keyframe is silently dropped. Composing the two needs
    that assignment to accumulate, which is a core edit -- see the keyframe-anchors
    branch. Here, use it on conditioning that carries no references.

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
            category="MMH3Tools/reference",
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
                    tooltip="0 = first frame. -1 = last frame. Nothing else: stock "
                            "PackedLayout raises 'only first/last keyframe anchors are "
                            "supported' and this node refuses rather than failing deeper in. "
                            "MiniMax's own guide does list interior anchors as valid, and "
                            "they work -- on the keyframe-anchors branch, which patches "
                            "PackedLayout at runtime.",
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
            # Refuse rather than warn. Stock PackedLayout raises on interior anchors, so
            # warning here only moved the failure deeper for no gain. Unlocking it means
            # patching core, which lives on the branch.
            raise ValueError(
                "frame_index %d is an INTERIOR anchor, and stock PackedLayout raises "
                "'only first/last keyframe anchors are supported'. Use 0 or -1 here. "
                "Interior anchors are on the keyframe-anchors branch, which patches "
                "PackedLayout at runtime." % index)

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

    KEYFRAMES AND REFERENCES DO NOT COEXIST ON STOCK COMFYUI. `extra_conds` assigns
    `cond_video_latents` from keyframes and then ASSIGNS IT AGAIN from references, so
    any reference silently erases every keyframe and the layout's cond rows outnumber
    the latents feeding them. Use this on conditioning that carries no references, or
    take the keyframe-anchors branch, where the accumulate fix is applied.

    Whether the ref2va checkpoint responds to 'cond' rows is the open experiment --
    fl2va definitely does.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentKeyframe",
            display_name="MiniMax H3 Latent Keyframe",
            category="MMH3Tools/reference",
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



class MMH3Regenerate2KReference(io.ComfyNode):
    """Per-window references from a 768p result, for a 2K regeneration pass.

    H3-Regenerate-2K "feeds the 768p result together with the original context back
    into H3 to regenerate the output at 2K". This builds that second pass locally,
    with one difference that matters at length: the reference is SLICED PER WINDOW.

    WHY SLICING IS FREE. A cond_set is already per chunk -- the looping sampler takes
    conds[i] for chunk i and passes minimax_refs through untouched. So a reference
    attached to cond i reaches chunk i and nothing else. The slicing happens here, at
    build time, and the sampler needs no knowledge of it.

    WHY IT MATTERS. Reference tokens are re-attended at EVERY sampling step. Handing
    the whole 768p clip to every chunk multiplies that by the chunk count; handing
    each chunk its own span does not.

    NO DECODE. The reference is latent-only -- appended as minimax_refs for the DiT,
    never presented to the text encoder. That is what MMH3LatentToRef already does,
    and it is right here: the prompt is the ORIGINAL context, written when the 768p
    pass ran, so the encoder has nothing to learn from seeing the video again. It
    also avoids a VAE roundtrip on a clip you already hold as latents.

    THE SCHEDULE COMES FROM `_plan`, the same function the sampler uses, so the
    window this node slices for is the window that chunk renders. Wire the same
    chunk_frames and overlap_frames the sampler gets.
    """

    @classmethod
    def define_schema(cls):
        from .nodes_multiprompt import MMH3CondSet
        return io.Schema(
            node_id="MMH3Regenerate2KReference",
            display_name="MMH3 Regenerate-2K Reference",
            category="MMH3Tools/conditioning",
            description=(
                "Build a cond_set for a 2K regeneration pass, giving each chunk only "
                "ITS span of the 768p result as a reference. One cond per window, so "
                "reference cost does not multiply by the chunk count."
            ),
            inputs=[
                io.Latent.Input(
                    "stage1_latent",
                    tooltip="The 768p AV result to upscale. Latents, not pixels: this "
                            "is appended for the DiT and never shown to the text "
                            "encoder, so no decode is needed and none is done."),
                MMH3CondSet.Input(
                    "stage1_cond_set", optional=True,
                    tooltip="The ORIGINAL context: the cond_set the 768p pass used. "
                            "Window i keeps ITS OWN stage-1 prompt and gains ITS OWN "
                            "reference slice, so per-window prompts survive into 2K. "
                            "Nothing is re-encoded and the text encoder is never "
                            "touched.\n\n"
                            "Wire this OR `conditioning`, not both."),
                io.Conditioning.Input(
                    "conditioning", optional=True,
                    tooltip="A single context, replicated to every window. Use this only "
                            "when the 768p pass ran on ONE prompt -- with a per-window "
                            "cond_set it would give every window prompt 0's text against "
                            "another window's reference."),
                io.Int.Input(
                    "width", default=2016, min=32, max=16384, step=32,
                    tooltip="2K width. Wire MMH3 Regenerate-2K Dimensions' width_2k."),
                io.Int.Input(
                    "height", default=1152, min=32, max=16384, step=32,
                    tooltip="2K height. Wire its height_2k."),
                io.Int.Input(
                    "chunk_frames", default=192, min=5, max=3600, step=17,
                    tooltip="Must be the value the sampler gets, or the window this "
                            "node slices for is not the window that chunk renders. "
                            "The sampler's chunk_frames=0 is NOT supported here -- set "
                            "both explicitly."),
                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=3600, step=17,
                    tooltip="Must also match the sampler."),
                io.Boolean.Input(
                    "include_audio", default=True,
                    tooltip="Attach each window's audio span alongside its video at the "
                            "same coordinate (a video_audio block). Off makes the "
                            "reference video-only and cheaper; the audio then has to "
                            "come from the target latent instead."),
                io.Int.Input(
                    "ref_downscale", default=1, min=1, max=8,
                    tooltip="Spatially downscale each reference slice. Reference tokens "
                            "ride every step, so 2 cuts their cost about 4x. Snapped to "
                            "a factor the 2x2 patch grid can express."),
            ],
            outputs=[
                MMH3CondSet.Output(display_name="cond_set"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="window_count"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, stage1_latent, stage1_cond_set=None, conditioning=None,
                width=2016, height=1152, chunk_frames=192, overlap_frames=22,
                include_audio=True, ref_downscale=1) -> io.NodeOutput:
        from .common import FPS, latents_to_frames
        from .nodes_windows import _audio_index_at, _plan, _window_frame_spans

        base = (stage1_cond_set or {}).get("conds") if stage1_cond_set else None
        if base and conditioning is not None:
            raise ValueError(
                "MMH3Regenerate2KReference: both stage1_cond_set and conditioning are "
                "wired. Pick one -- the cond_set keeps each window's own prompt, the "
                "single conditioning replicates one to all of them.")
        if not base:
            if conditioning is None:
                raise ValueError(
                    "MMH3Regenerate2KReference: wire stage1_cond_set (the 768p pass's "
                    "cond_set) or a single conditioning.")
            base = [conditioning]

        v, a = unpack_av(stage1_latent, "stage1_latent")
        total_t = int(v.shape[2])
        total_a = 0 if a is None else int(a.shape[AUDIO_T_DIM])
        total_f = latents_to_frames(total_t)

        length, overlap, plan_f, _pt, windows = _plan(
            total_f, int(chunk_frames), int(overlap_frames), "standard_static")
        spans = _window_frame_spans(windows, plan_f)
        n = len(windows)

        # The 2K target is the SAME length as the 768p source: this is a resolution
        # pass, not a re-timing. Deriving it from the source rather than a widget
        # keeps the schedule identical on both sides.
        # empty_av_latent returns (latent, frame_count) -- the tuple, not the latent.
        out_latent, _out_f = empty_av_latent(int(width), int(height), total_f)

        conds, lines, used = [], [], int(ref_downscale)
        for i, w in enumerate(windows):
            v0, v1 = w.index_list[0], w.index_list[-1] + 1
            sub_v = v[:, :, v0:v1].contiguous()
            sub_v, lh, lw, used = downscale_video_latent(sub_v, int(ref_downscale))

            sub_a, at = None, 0
            if include_audio and a is not None:
                a0 = _audio_index_at(v0, total_t, total_a)
                a1 = _audio_index_at(v1, total_t, total_a)
                if a1 > a0:
                    sub_a = a[:, :, :, a0:a1].contiguous()
                    at = a1 - a0

            block = make_ref_block(sub_v, sub_a, lh, lw, at)
            # Window i keeps ITS stage-1 prompt. Fewer prompts than windows repeats
            # the last, matching what the looping sampler does with a short cond_set.
            conds.append(append_cond_list(base[min(i, len(base) - 1)],
                                          "minimax_refs", [block]))
            lines.append("  window %d: prompt %d, frames %d-%d, ref %d latents %dx%d%s"
                         % (i, min(i, len(base) - 1),
                            spans[i][0], spans[i][1], v1 - v0, lw, lh,
                            (", audio %d" % at) if at else ""))

        per = sum((w.index_list[-1] + 1 - w.index_list[0]) for w in windows) / float(n)
        report = ("%d window%s over %d frames (%.2fs), chunk %d latents, overlap %d\n"
                  "2K target %dx%d, same length as the source\n"
                  "reference: %.0f latents per window vs %d for the whole clip -- "
                  "about %.1fx less reference attention per chunk%s\n%s"
                  % (n, "" if n == 1 else "s", total_f, total_f / float(FPS),
                     length, overlap, int(width), int(height), per, total_t,
                     total_t / max(1.0, per),
                     "" if used <= 1 else (" (plus %dx spatial downscale)" % used),
                     "\n".join(lines)))
        if used != int(ref_downscale):
            report += ("\n  ! ref_downscale %d snapped to %d, the nearest the patch "
                       "grid allows" % (int(ref_downscale), used))
        if len(base) != n:
            note = ("%d stage-1 prompt%s for %d window%s -- %s"
                    % (len(base), "" if len(base) == 1 else "s", n,
                       "" if n == 1 else "s",
                       "the last repeats" if len(base) < n else "the extras are unused"))
            report += "\n  ! " + note
            logging.warning("[MMH3Regenerate2KReference] %s", note)
        logging.info("[MMH3Regenerate2KReference] %s", report.splitlines()[0])
        return io.NodeOutput({"conds": conds, "prompts": [], "fingerprint": None},
                             out_latent, n, report)
