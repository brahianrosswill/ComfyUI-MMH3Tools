"""Multi-prompt reference conditioning: encode the references ONCE, the prompts N times.

For a text-driven sequence with locked identity, every chunk shares the same
references and differs only in its prompt. Stock MiniMaxH3ReferenceToVideo does
the reference resize, vae.encode and audio_vae.encode in the SAME execute() as the
text encode, so N chunks means N copies of all of it -- and, worse, N model swap
cycles, because Qwen3-VL-32B and a 33B DiT cannot be resident together. ComfyUI
resolves outputs depth-first, so a naive N-chunk graph runs
load TE -> cond -> evict -> load DiT -> sample -> evict -> load TE -> ...

Doing every encode inside ONE node execution collapses that to a single swap.

What is still paid per prompt: clip.tokenize re-presents the references to Qwen and
the vision tower plus 50 layers run again. That is inherent -- references are
emitted BEFORE the prompt text, and while comfy/text_encoders/llama.py does thread
past_key_values through every layer, the CLIP API exposes no way to hand it a
cached prefix. Cheap for still images; the thing to avoid for video references.
"""

import hashlib
import logging
import math

import torch

import node_helpers
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    MiniMaxH3ReferenceToVideo,
    _empty_av_latent,
    _resize,
    adapt_canvas,
)

MMH3CondSet = io.Custom("MMH3_COND_SET")

# (prompt, ref fingerprint) -> conditioning. Editing ONE prompt re-executes the
# whole node, so without this a one-word change costs every prompt's Qwen pass.
_CACHE = {}
_CACHE_MAX = 64


def _hash_tensor(h, t):
    """Strided sample of a tensor into a hash. O(4096) regardless of size."""
    h.update(str(tuple(t.shape)).encode())
    flat = t.detach().flatten()
    step = max(1, flat.numel() // 4096)
    h.update(flat[::step].to(torch.float32).cpu().numpy().tobytes())


def _hash_input(h, obj):
    """Hash a raw reference input: IMAGE tensor, or AUDIO {waveform, sample_rate}."""
    if obj is None:
        h.update(b"none")
    elif isinstance(obj, dict) and "waveform" in obj:
        h.update(str(obj.get("sample_rate")).encode())
        _hash_tensor(h, obj["waveform"])
    elif hasattr(obj, "shape"):
        _hash_tensor(h, obj)
    else:
        h.update(repr(obj).encode())


def _fingerprint(ref_blocks, raw_inputs, width, height, length, ref_image_size):
    """Identify a reference set cheaply but honestly.

    Hashes BOTH the raw inputs and the encoded blocks. The blocks alone would be
    tempting -- they are already in hand and capture every sizing decision -- but
    that makes cache validity depend on the VAE mapping different references to
    different latents. That holds for a real VAE and is exactly the assumption you
    do not want load-bearing when the failure mode is a stale encode with no
    visible symptom: the wrong reference, silently, in every chunk. Hashing the
    inputs costs a strided read of a few images and removes the assumption.
    """
    h = hashlib.sha256()
    h.update(("%d|%d|%d|%s" % (width, height, length, ref_image_size)).encode())
    for obj in raw_inputs:
        _hash_input(h, obj)
    for b in ref_blocks:
        h.update(("%s|%s|%s|%s|%s" % (b.get("kind"), b.get("latent_h"), b.get("latent_w"),
                                      b.get("latent_t"), b.get("ref_audio_t"))).encode())
        for key in ("latent", "audio_latent"):
            t = b.get(key)
            if t is not None:
                _hash_tensor(h, t)
    return h.hexdigest()


def _build_refs(vae, audio_vae, width, height, frame_count, ref_image_size,
                ref_images, ref_videos, ref_video_audios, ref_audios):
    """Reference items (for the tokenizer) and blocks (for the DiT), built once.

    DUPLICATED FROM comfy_extras/nodes_minimax_h3.py, deliberately: upstream runs
    this inline in the same execute() as the text encode, so there is no seam to
    call. Re-sync if that file changes its sizing, its block keys, or -- most
    fragile -- the emission ORDER, since the tokenizer assigns <Picture i>,
    <Audio j> and <Video k> labels by counting items in the order given. A video's
    soundtrack must be appended BEFORE the video itself or every label after it
    shifts and the prompt's tags stop matching.
    """
    ref_items = []
    ref_blocks = []

    # ref_images is a BATCH: every element is its own <Picture i>, in batch order.
    # The previous shape took one image per socket and sliced img[:1], so a batch
    # wired into a slot silently contributed only its first frame.
    if ref_images is not None:
        for i in range(int(ref_images.shape[0])):
            img = ref_images[i:i + 1]
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img, tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": z})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        z = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = MiniMaxH3ReferenceToVideo._encode_ref_audio(
                audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        ref_items.append({"type": "video", "data": frames[sample_idx],
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = MiniMaxH3ReferenceToVideo._encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                           "audio_latent": audio_latent})

    return ref_items, ref_blocks


class MMH3ReferenceMultiPrompt(io.ComfyNode):
    """One reference set, N prompts, one model swap."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ReferenceMultiPrompt",
            display_name="MiniMax H3 Reference (Multi-Prompt)",
            category="MMH3Tools",
            description=(
                "MiniMaxH3ReferenceToVideo with N prompts. References are resized and "
                "encoded ONCE, and every text encode happens in one node execution, so "
                "the text encoder and the DiT each load once for the whole sequence "
                "instead of once per chunk. Feed the output to MMH3 Cond Select."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=192, min=5, max=3600, step=17,
                             tooltip="Frames at 24 fps, shared by every prompt. 192 is the "
                                     "only whole-second duration in the trained range."),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match",
                    tooltip="'match' scales each reference to the generation's pixel area; "
                            "'max' uses a 2048px short edge for best identity fidelity. "
                            "Reference tokens ride through every sampling step of every "
                            "chunk, so 'max' is paid N times over.",
                ),
                io.String.Input(
                    "prompts", multiline=True, dynamic_prompts=True,
                    tooltip="Every prompt in one string, PIPE separated, in chunk order. "
                            "A loop that accumulates one prompt per window wires straight "
                            "in here -- no socket per chunk, so the graph is the same size "
                            "whatever N is.\n\n"
                            "Keep subject_definitions and retention_analysis byte-identical "
                            "across all of them; only detailed_description should vary, or "
                            "the character drifts. A literal | inside a prompt WILL "
                            "over-split silently -- watch the `count` output, which is "
                            "how many prompts this actually found.",
                ),
                io.Image.Input(
                    "ref_images", optional=True,
                    tooltip="A BATCH of reference stills -- each one becomes its own "
                            "<Picture i>, numbered in batch order. One image is the "
                            "ordinary case.",
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[
                MMH3CondSet.Output(display_name="cond_set"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, width, height, length, ref_image_size,
                prompts=None, ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        # Pipe separated, in chunk order. Empty pieces are dropped rather than
        # encoded, so a trailing | or a blank line between prompts costs nothing.
        texts = [p.strip() for p in (prompts or "").split("|") if p.strip()]
        if not texts:
            raise ValueError(
                "MMH3ReferenceMultiPrompt needs at least one prompt. Prompts go in one "
                "string separated by | , in chunk order.")

        ref_items, ref_blocks = _build_refs(
            vae, audio_vae, width, height, frame_count, ref_image_size,
            ref_images, ref_videos, ref_video_audios, ref_audios)

        raw_inputs = [ref_images]
        for group in (ref_videos, ref_video_audios, ref_audios):
            raw_inputs.extend((group or {}).values())
        fp = _fingerprint(ref_blocks, raw_inputs, width, height, length, ref_image_size)

        conds = []
        hits = 0
        for text in texts:
            key = (text, fp)
            cached = _CACHE.get(key)
            if cached is not None:
                conds.append(cached)
                hits += 1
                continue
            tokens = clip.tokenize(text, minimax_ref_items=ref_items)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = cond
            conds.append(cond)

        logging.info("[MMH3ReferenceMultiPrompt] %d prompts, %d refs, %d frames "
                     "(%d encodes reused)", len(conds), len(ref_blocks), frame_count, hits)
        return io.NodeOutput({"conds": conds, "prompts": texts, "fingerprint": fp},
                             latent, len(conds))


class MMH3CondSelect(io.ComfyNode):
    """Pull one chunk's conditioning out of a cond_set."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondSelect",
            display_name="MMH3 Cond Select",
            category="MMH3Tools",
            description="Select one prompt's conditioning from a MiniMax H3 cond_set.",
            inputs=[
                MMH3CondSet.Input("cond_set"),
                io.Int.Input("index", default=0, min=0, max=31, step=1,
                             tooltip="0-based. Out of range is an error rather than a wrap, "
                                     "because silently rendering the wrong chunk is worse "
                                     "than a stopped queue."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="prompt"),
            ],
        )

    @classmethod
    def execute(cls, cond_set, index) -> io.NodeOutput:
        conds = cond_set["conds"]
        i = int(index)
        if i >= len(conds):
            raise ValueError(
                "index %d is out of range: the cond_set holds %d prompt%s (0-%d)."
                % (i, len(conds), "" if len(conds) == 1 else "s", len(conds) - 1))
        return io.NodeOutput(conds[i], cond_set["prompts"][i])


class MMH3CondSetSpread(io.ComfyNode):
    """Flatten a cond_set into ONE conditioning holding every prompt, in order.

    This is the input shape `split_conds_to_windows` wants. Core decides which prompt
    a window uses from the window's own midpoint:

        center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
        region       = int(center_ratio * len(cond_in))

    so entry 0 covers the start of the timeline and entry N-1 the end. Without this,
    every window sees the same single conditioning and the model is asked to render
    the whole script into each one -- which is what "it looks like it's doing the
    entire conditioning per window" was.

    MMH3CondSelect takes ONE prompt for ONE chunk; this takes all of them for one
    windowed pass. The references are shared either way, because the cond_set encoded
    them once, so identity does not shift as the region changes -- only the prompt does.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondSetSpread",
            display_name="MMH3 Cond Set Spread",
            category="MMH3Tools",
            description=(
                "Flatten a cond_set into a single conditioning containing every prompt "
                "in order, for MMH3 Context Windows with split_conds_to_windows on. "
                "Each window then uses the prompt for its own region of the timeline."
            ),
            inputs=[
                MMH3CondSet.Input("cond_set"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Int.Output(display_name="regions"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, cond_set) -> io.NodeOutput:
        conds = cond_set["conds"]
        prompts = cond_set.get("prompts") or []

        # each cond_set entry is a full CONDITIONING (a list); the region split works on
        # the ENTRIES of one conditioning, so concatenate rather than nest
        flat = []
        for c in conds:
            flat.extend(c)

        if len(flat) != len(conds):
            logging.info("[MMH3CondSetSpread] %d prompts expanded to %d entries; regions "
                         "are per ENTRY, so they will not line up with prompts",
                         len(conds), len(flat))

        # A keyframe re-projects into EVERY window. The layout is rebuilt per window
        # from the window's own latent_t, and a first-frame anchor is placed at the
        # target origin -- which is that window's frame 0, not the clip's. An i2v start
        # image would therefore be re-imposed at every window boundary. A last-frame
        # anchor is worse: minimax_frame_count is not patched per window, so the index
        # check still matches the clip while the POSITION comes from the window.
        # Entry 0 is the exception, not an offender: region 0 IS the first window, so a
        # start frame anchored there lands where it belongs. Anywhere else is a repeat.
        with_kf = [i for i, e in enumerate(flat) if e[1].get("minimax_keyframes")]
        misplaced = [i for i in with_kf if i > 0]
        kf_note = ""
        if misplaced and len(flat) > 1:
            kf_note = ("\n  ! entr%s %s carr%s keyframes. Under split_conds_to_windows a "
                       "keyframe is re-anchored to ITS OWN window's start or end, not the "
                       "clip's, so this repeats at every window boundary. Keep keyframes on "
                       "entry 0 only -- region 0 is the first window, the one place a start "
                       "frame belongs."
                       % ("y" if len(misplaced) == 1 else "ies",
                          ", ".join(str(i) for i in misplaced),
                          "ies" if len(misplaced) == 1 else "y"))
        elif with_kf == [0] and len(flat) > 1:
            kf_note = "\n  keyframe on entry 0 only -- anchored to the first window, correct"

        lines = []
        for i, text in enumerate(prompts[:len(flat)]):
            lo, hi = i / len(flat), (i + 1) / len(flat)
            first = (text or "").strip().splitlines()
            lines.append("  %d  %.0f%%-%.0f%%  %s"
                         % (i, lo * 100, hi * 100, (first[0][:60] if first else "(empty)")))
        report = "%d region%s across the clip:\n%s" % (
            len(flat), "" if len(flat) == 1 else "s", "\n".join(lines))
        if len(flat) == 1:
            report += ("\n  ! one prompt means split_conds_to_windows does nothing -- core "
                       "only splits when a conditioning holds more than one entry")
        report += kf_note
        logging.info("[MMH3CondSetSpread] " + report)
        return io.NodeOutput(flat, len(flat), report)
