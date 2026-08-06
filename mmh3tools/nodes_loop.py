"""Assembling chunks: measuring the join, pairing modalities, splicing time.

NOTE ON noise_mask: masks DO reach the model. `comfy/samplers.py` packs the AV pair
into a flat tensor before sampling (L1280) and explicitly unbinds a nested mask
(L1296), so the sampler never sees a NestedTensor and the inpaint arithmetic is fine.
An earlier version of this file claimed otherwise; that was wrong.

What stock ComfyUI lacks is per-row TIMESTEP handling: preserved rows still run at
the generation timestep, so the model gets clean content labelled as noisy and the
mask accomplishes nothing. `drozbay:ComfyUI:minimax-h3-per-row-masking` fixes that by
pinning masked rows to the cond timestep and lerping AdaLN modulation per token.
MMH3SeedOverlap requires that patch; without it the node runs but does nothing useful.

Joins are still trimmed AFTER decode - latent trims sit on the 5j+2 grid, i.e.
17-frame steps, and latent concatenation is unsound regardless (see MMH3JoinAV).
"""

import logging

import torch

from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

from .common import (
    AUDIO_T_DIM,
    FPS,
    VIDEO_T_DIM,
    frames_to_audio_t,
    latents_to_frames,
    on_grid,
    pack_av,
    slice_av_tail,
    snap_latents,
    unpack_av,
)


class MMH3SeedOverlap(io.ComfyNode):
    """LTXAV-style mask-and-extend: seed the target head, mask it, denoise the rest.

    REQUIRES the per-row masking patch (drozbay:ComfyUI:minimax-h3-per-row-masking).
    Stock ComfyUI accepts a nested mask and packs it correctly, but preserved rows
    still run at the GENERATION timestep, so the model receives clean content
    labelled as noisy and the mask achieves nothing. The patch pins masked rows to
    the COND timestep -- the same treatment reference rows get -- and lerps between
    the two AdaLN modulation vectors per token, which is what makes a partial
    strength mean anything.

        overlap_strength 1.0 -> mask 0.0 -> fully preserved (pinned)
        overlap_strength 0.0 -> mask 1.0 -> fully regenerated

    Video and audio are masked independently on their own temporal axes (video dim
    2, audio dim 3) and reach the model as separate denoise_mask / audio_denoise_mask
    conditions, so lipsync can carry audio harder than video.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SeedOverlap",
            display_name="MiniMax H3 Seed Overlap",
            category="MMH3Tools",
            description=(
                "Seed the head of a target AV latent with the tail of a previous chunk and "
                "emit a matching nested noise_mask. Requires the per-row masking patch."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Target AV latent (from Empty MiniMax H3 AV Latent)"),
                io.Latent.Input("source", tooltip="Previous chunk's AV latent"),
                io.Int.Input(
                    "overlap_latents", default=5, min=5, max=512, step=5,
                    tooltip="Video latents PREPENDED as overlap. Must be a multiple of 5: the "
                            "target is 5a+2 and the total must stay 5c+2, so only multiples of "
                            "5 keep the result decodable. 5 latents = 17 frames = 0.708s.",
                ),
                io.Float.Input(
                    "overlap_strength_video", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="1.0 preserves the overlap (noise_mask 0), 0.0 regenerates it. "
                            "Intermediate values are real partial pins, not thresholds.",
                ),
                io.Float.Input(
                    "overlap_strength_audio", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Same scale as video. Lipsync usually wants this at or near 1.0.",
                ),
                io.Int.Input(
                    "feather_latents", default=0, min=0, max=64, step=1,
                    tooltip="Linear ramp back to full denoise over N video latents after the "
                            "overlap, avoiding a hard mask step at the seam. 0 disables.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="overlap_frames"),
                io.Int.Output(
                    display_name="overlap_latents",
                    tooltip="Wire into ConcatAV's trim_b_latents so the overlap is not "
                            "duplicated at the join.",
                ),
            ],
        )

    @classmethod
    def execute(cls, latent, source, overlap_latents, overlap_strength_video,
                overlap_strength_audio, feather_latents) -> io.NodeOutput:
        tgt_v, tgt_a = unpack_av(latent, "latent")
        src_v, src_a = unpack_av(source, "source", allow_video_only=True)

        if src_v.shape[3:] != tgt_v.shape[3:]:
            raise ValueError(
                "Spatial mismatch: source latent is %dx%d, target is %dx%d. "
                "Overlap seeding requires identical dimensions."
                % (src_v.shape[4] * 16, src_v.shape[3] * 16,
                   tgt_v.shape[4] * 16, tgt_v.shape[3] * 16)
            )

        # PREPEND the overlap so the target keeps its full requested duration. The total
        # must stay on the 5j+2 grid, and (5a+2)+(5b+2) never is -- so the overlap has to
        # be a multiple of 5, which adds exactly 17 frames each.
        k = max(5, (int(overlap_latents) // 5) * 5)
        k = min(k, int(src_v.shape[VIDEO_T_DIM]))
        k = (k // 5) * 5
        if k < 5:
            raise ValueError("source has fewer than 5 video latents; nothing to overlap")

        n_tgt = int(tgt_v.shape[VIDEO_T_DIM])
        total = n_tgt + k
        tgt_frames = latents_to_frames(n_tgt)
        total_frames = latents_to_frames(total)
        overlap_frames = total_frames - tgt_frames          # == 17 * (k // 5)
        overlap_audio = frames_to_audio_t(total_frames) - frames_to_audio_t(tgt_frames)

        v = torch.cat([src_v[:, :, -k:, :, :].to(tgt_v.dtype), tgt_v], dim=VIDEO_T_DIM)

        if src_a is not None and overlap_audio > 0:
            take = min(overlap_audio, int(src_a.shape[AUDIO_T_DIM]))
            head = src_a[:, :, :, -take:].to(tgt_a.dtype)
            if take < overlap_audio:                        # source shorter than needed
                pad = torch.zeros([head.shape[0], head.shape[1], head.shape[2],
                                   overlap_audio - take], dtype=tgt_a.dtype, device=tgt_a.device)
                head = torch.cat([pad, head], dim=AUDIO_T_DIM)
        else:
            if src_a is None:
                logging.info("[MMH3SeedOverlap] source has no audio; overlap audio is silent")
            head = torch.zeros([tgt_a.shape[0], tgt_a.shape[1], tgt_a.shape[2], overlap_audio],
                               dtype=tgt_a.dtype, device=tgt_a.device)
        a = torch.cat([head, tgt_a], dim=AUDIO_T_DIM)

        # noise_mask: 1.0 = denoise, 0.0 = preserve
        vm = torch.ones([v.shape[0], 1, v.shape[2], v.shape[3], v.shape[4]],
                        dtype=torch.float32, device=v.device)
        vm[:, :, :k] = 1.0 - float(overlap_strength_video)

        if feather_latents > 0:
            end = min(k + int(feather_latents), vm.shape[2])
            steps = end - k
            if steps > 0:
                ramp = torch.linspace(1.0 - float(overlap_strength_video), 1.0, steps + 1,
                                      device=v.device)[1:]
                vm[:, :, k:end] = ramp.view(1, 1, steps, 1, 1)

        am = torch.ones([a.shape[0], 1, a.shape[2], a.shape[3]],
                        dtype=torch.float32, device=a.device)
        if overlap_audio > 0:
            am[:, :, :, :overlap_audio] = 1.0 - float(overlap_strength_audio)

        out = pack_av(latent, v, a, noise_mask=NestedTensor([vm, am]))
        logging.info("[MMH3SeedOverlap] %d + %d = %d latents (%d frames), trim %d frames after decode",
                     k, n_tgt, total, total_frames, overlap_frames)
        return io.NodeOutput(out, int(overlap_frames), int(k))


class MMH3FindDivergence(io.ComfyNode):
    """Find where a continuation stops reproducing its source, in FRAMES.

    H3 tends to re-render the carried reference at the head of a continuation
    before generating new content. That span is not frame-aligned with the source
    (the model regenerates rather than copies) and will not land on the 5j+2 latent
    grid, whose cut points are 17 frames apart -- so the trim has to happen after
    decode, where granularity is one frame.

    Method: assume the reproduced span ENDS at the source's final frame, so a run of
    length K means continuation[i] ~ source[-K+i]. For each candidate K the mean error
    over that exact alignment is scored, and the best K wins.

    Per-frame nearest-match does NOT work here: in visually repetitive footage (a
    talking head on a static background) every new frame also finds a close match
    somewhere in the source, so divergence is never detected. Requiring the whole run
    to align contiguously fixes that -- a wrong K misaligns every frame at once, which
    produces an order-of-magnitude error separation.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3FindDivergence",
            display_name="MiniMax H3 Find Divergence",
            category="MMH3Tools",
            description=(
                "Compare the head of a continuation against the tail of its source and "
                "report how many frames are reproduced, so you can trim the join."
            ),
            inputs=[
                io.Image.Input("source", tooltip="Decoded frames of the previous chunk"),
                io.Image.Input("continuation", tooltip="Decoded frames of the new chunk"),
                io.Int.Input("search_frames", default=96, min=1, max=2048, step=1,
                             tooltip="How far into the continuation to look."),
                io.Int.Input("source_tail_frames", default=96, min=1, max=2048, step=1,
                             tooltip="How much of the source tail to match against."),
                io.Float.Input("threshold", default=0.05, min=0.0, max=1.0, step=0.001,
                               tooltip="Reject the alignment if its mean absolute error exceeds "
                                       "this, and report 0 reproduced. Check the reported best/"
                                       "median separation to calibrate."),
                io.Int.Input("downsample", default=48, min=8, max=256, step=8,
                             tooltip="Frames are greyscaled and resized to this before "
                                     "comparison. Smaller is faster and more tolerant of noise."),
                io.Combo.Input("compare", options=["structure", "raw"], default="structure",
                               tooltip="'structure' removes each frame's mean and contrast before "
                                       "comparing, so an exposure or colour shift between source "
                                       "and generation cannot mask a real match. 'raw' is plain "
                                       "MAE. Error magnitudes differ between the two, so "
                                       "recalibrate threshold when switching."),
            ],
            outputs=[
                io.Int.Output(display_name="trim_frames"),
                io.Float.Output(display_name="mean_error"),
                io.String.Output(display_name="report"),
            ],
        )

    @staticmethod
    def _prep(img, size, structure):
        x = img[..., :3].mean(dim=-1, keepdim=True).movedim(-1, 1).float()
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="area")
        if structure:
            # zero-mean, unit-contrast per frame: an exposure or colour shift between
            # the source and the generated chunk otherwise puts a floor under every
            # comparison and flattens the curve, hiding a genuine match.
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            s = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            x = (x - m) / s
        return x

    @classmethod
    def execute(cls, source, continuation, search_frames, source_tail_frames,
                threshold, downsample, compare="structure") -> io.NodeOutput:
        n_src, n_con = source.shape[0], continuation.shape[0]
        tail = min(int(source_tail_frames), n_src)
        search = min(int(search_frames), n_con)
        structure = (compare == "structure")

        a = cls._prep(source[-tail:], int(downsample), structure)
        b = cls._prep(continuation[:search], int(downsample), structure)

        # pairwise mean absolute error, [search, tail]
        d = (b.unsqueeze(1) - a.unsqueeze(0)).abs().mean(dim=(2, 3, 4))

        # score each candidate run length K as the diagonal ending at the source's
        # last frame: continuation[i] vs source[-K+i]
        limit = min(search, tail)
        errs = []
        for k in range(1, limit + 1):
            i = torch.arange(k, device=d.device)
            errs.append(float(d[i, tail - k + i].mean()))
        err_t = torch.tensor(errs)

        best_k = int(err_t.argmin().item()) + 1
        best_err = float(err_t.min())
        median_err = float(err_t.median())
        # a real reproduction shows a sharp minimum, not a flat curve
        separation = median_err / best_err if best_err > 1e-8 else float("inf")

        trim = best_k if best_err <= float(threshold) else 0

        lo, hi = max(1, best_k - 2), min(limit, best_k + 2)
        around = ", ".join("%d:%.4f" % (k, errs[k - 1]) for k in range(lo, hi + 1))
        lines = [
            "reproduced : %d frames (%.3fs @24fps)%s"
            % (best_k, best_k / 24.0, "" if trim else "   REJECTED (error > threshold)"),
            "best error : %.5f   median %.5f   separation %.1fx (threshold %.4f)"
            % (best_err, median_err, separation, threshold),
            "curve near best: %s" % around,
        ]
        if separation < 3.0:
            lines.append("WARNING weak minimum -- the curve is flat, so this alignment is "
                         "not trustworthy. Check the clips actually overlap.")
        if best_k == limit:
            lines.append("NOTE best K is at the search limit; raise search_frames / "
                         "source_tail_frames.")
        mean_err = best_err

        report = "\n".join(lines)
        print("[MMH3FindDivergence]\n" + report)
        return io.NodeOutput(int(trim), mean_err, report)


class MMH3JoinAV(io.ComfyNode):
    """Join two decoded chunks in PIXEL and WAVEFORM space.

    Latent concatenation is unsound for H3's video VAE. Two on-grid chunks sum to
    5(j+k)+4 latents, which is never on the 5j+2 grid, so the decoder's 17-frame
    causal chunking misaligns from the join onward and the second half pulses. Even
    with an on-grid trim, chunk B's latent 0 is a causal anchor spanning one frame
    and ends up mid-group where the decoder expects four.

    Decoding each chunk separately avoids all of that, and gives frame granularity
    instead of the latent grid's 17-frame steps.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3JoinAV",
            display_name="MiniMax H3 Join AV",
            category="MMH3Tools",
            description=(
                "Trim and crossfade two decoded chunks. Video joins per frame, audio "
                "crossfades in the waveform domain (the DAC/BigVGAN latents do not blend)."
            ),
            inputs=[
                io.Image.Input("images_a"),
                io.Image.Input("images_b"),
                io.Int.Input("trim_b_frames", default=0, min=0, max=4096, step=1,
                             tooltip="Frames to drop from the head of B, e.g. a reproduced span "
                                     "measured by MMH3FindDivergence."),
                io.Int.Input("crossfade_frames", default=0, min=0, max=240, step=1,
                             tooltip="Linear crossfade across the seam, taken from A's tail and "
                                     "B's head. 0 is a hard cut."),
                io.Audio.Input("audio_a", optional=True),
                io.Audio.Input("audio_b", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, images_a, images_b, trim_b_frames, crossfade_frames,
                audio_a=None, audio_b=None) -> io.NodeOutput:
        b = images_b[int(trim_b_frames):] if trim_b_frames > 0 else images_b
        if b.shape[0] == 0:
            raise ValueError("trim_b_frames removed the whole of images_b")
        if images_a.shape[1:] != b.shape[1:]:
            raise ValueError("Frame size mismatch: A is %s, B is %s"
                             % (tuple(images_a.shape[1:3]), tuple(b.shape[1:3])))

        n = max(0, min(int(crossfade_frames), images_a.shape[0], b.shape[0]))
        if n > 0:
            w = torch.linspace(0, 1, n + 2, device=images_a.device)[1:-1].view(-1, 1, 1, 1)
            blend = images_a[-n:] * (1 - w) + b[:n].to(images_a.dtype) * w
            video = torch.cat([images_a[:-n], blend, b[n:].to(images_a.dtype)], dim=0)
        else:
            video = torch.cat([images_a, b.to(images_a.dtype)], dim=0)

        audio = None
        if audio_a is not None and audio_b is not None:
            sr = int(audio_a["sample_rate"])
            if int(audio_b["sample_rate"]) != sr:
                raise ValueError("Sample rate mismatch: %d vs %d" % (sr, audio_b["sample_rate"]))
            wa, wb = audio_a["waveform"], audio_b["waveform"].to(audio_a["waveform"].dtype)
            cut = int(round(int(trim_b_frames) / FPS * sr))
            wb = wb[:, :, cut:] if cut > 0 else wb
            m = max(0, min(int(round(n / FPS * sr)), wa.shape[-1], wb.shape[-1]))
            if m > 0:
                w = torch.linspace(0, 1, m + 2, device=wa.device)[1:-1].view(1, 1, -1)
                mid = wa[:, :, -m:] * (1 - w) + wb[:, :, :m] * w
                wav = torch.cat([wa[:, :, :-m], mid, wb[:, :, m:]], dim=-1)
            else:
                wav = torch.cat([wa, wb], dim=-1)
            audio = {"waveform": wav, "sample_rate": sr}
        elif audio_a is not None or audio_b is not None:
            logging.warning("[MMH3JoinAV] only one audio input connected; audio not joined")
            audio = audio_a if audio_a is not None else audio_b

        label = "%d + %d frames (trimmed %d, crossfade %d) -> %d frames, %.3fs" % (
            images_a.shape[0], images_b.shape[0], trim_b_frames, n,
            video.shape[0], video.shape[0] / FPS)
        print("[MMH3JoinAV] " + label)
        return io.NodeOutput(video, audio, label)


class MMH3PackAV(io.ComfyNode):
    """Zip a video latent and an audio latent into one H3 AV latent.

    Encoding real footage gives two SEPARATE plain latents -- VAEEncode with the
    H3 video VAE, and VAEEncodeAudio with the H3 audio VAE -- and nothing pairs
    them. This is that pairing. It is not a concatenation: ConcatAV joins two AV
    latents along TIME, this joins video and audio along MODALITY.

    Audio length is reconciled to what the video length implies
    (round(frames / 24 * 40)), padding with silence or trimming as needed, since
    the two streams run on independent clocks and encoders will not agree exactly.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3PackAV",
            display_name="MiniMax H3 Pack AV",
            category="MMH3Tools",
            description=(
                "Combine a video latent (VAEEncode, H3 video VAE) and an audio latent "
                "(VAEEncodeAudio, H3 audio VAE) into a single H3 AV latent. Omit the "
                "audio to pair with silence."
            ),
            inputs=[
                io.Latent.Input("video_latent", tooltip="Plain video latent [B,24,T,h,w]"),
                io.Latent.Input(
                    "audio_latent", optional=True,
                    tooltip="Plain audio latent [B,32,2,T40]. If omitted, silence of the "
                            "correct length is generated.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent=None) -> io.NodeOutput:
        v, _ = unpack_av(video_latent, "video_latent", allow_video_only=True)
        vt = int(v.shape[VIDEO_T_DIM])
        frames = latents_to_frames(vt)
        want_at = frames_to_audio_t(frames)

        note = ""
        if audio_latent is None:
            a = torch.zeros([v.shape[0], 32, 2, want_at], dtype=v.dtype, device=v.device)
            note = "silent audio generated"
        else:
            a = audio_latent["samples"]
            if isinstance(a, NestedTensor):
                a = a.unbind()[1]
            if a.ndim != 4 or a.shape[1] != 32:
                raise ValueError(
                    "'audio_latent' is not an H3 audio latent; expected [B,32,2,T40], got %s. "
                    "Encode with VAEEncodeAudio using the H3 audio VAE." % (tuple(a.shape),)
                )
            have = int(a.shape[AUDIO_T_DIM])
            if have > want_at:
                a = a[:, :, :, :want_at]
                note = "audio trimmed %d -> %d" % (have, want_at)
            elif have < want_at:
                pad = torch.zeros([a.shape[0], a.shape[1], a.shape[2], want_at - have],
                                  dtype=a.dtype, device=a.device)
                a = torch.cat([a, pad], dim=AUDIO_T_DIM)
                note = "audio padded %d -> %d" % (have, want_at)
            a = a.to(v.dtype)

        if not on_grid(vt):
            note = (note + "; " if note else "") + "WARNING video T=%d is off the 5j+2 grid" % vt

        out = dict(video_latent)
        out["samples"] = NestedTensor([v.contiguous(), a.contiguous()])

        # Carry any input mask into a nested pair rather than dropping it. Filling the
        # missing side with ones means "denoise everything there", so pairing a masked
        # video latent with unmasked audio does what you would expect.
        vm = video_latent.get("noise_mask")
        am = audio_latent.get("noise_mask") if audio_latent is not None else None
        if isinstance(vm, NestedTensor):
            vm = vm.unbind()[0]
        if isinstance(am, NestedTensor):
            am = am.unbind()[-1]
        if vm is not None or am is not None:
            if vm is None:
                vm = torch.ones([v.shape[0], 1, v.shape[2], v.shape[3], v.shape[4]],
                                dtype=torch.float32, device=v.device)
            if am is None:
                am = torch.ones([a.shape[0], 1, a.shape[2], a.shape[3]],
                                dtype=torch.float32, device=a.device)
            out["noise_mask"] = NestedTensor([vm, am])
            logging.info("[MMH3PackAV] carried an input noise_mask into the AV pair")
        else:
            out.pop("noise_mask", None)

        label = "%d video latents (%d frames, %.3fs) + %d audio latents%s" % (
            vt, frames, frames / 24.0, int(a.shape[AUDIO_T_DIM]), ("  [" + note + "]") if note else "")
        print("[MMH3PackAV] " + label)
        return io.NodeOutput(out, label)


class MMH3ConcatAV(io.ComfyNode):
    """Concatenate two AV latents on their correct, DIFFERENT temporal axes.

    Video is dim 2, audio is dim 3. Generic nested-tensor concat helpers that
    assume one shared temporal dim will stack audio on its stereo axis instead,
    producing 4 channels at unchanged duration rather than a longer clip.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ConcatAV",
            display_name="MiniMax H3 Concat AV",
            category="MMH3Tools",
            description="Join two H3 AV latents end to end (video dim 2, audio dim 3).",
            inputs=[
                io.Latent.Input("latent_a"),
                io.Latent.Input("latent_b"),
                io.Int.Input(
                    "trim_b_latents", default=0, min=0, max=512, step=1,
                    tooltip="Drop this many video latents from the head of B before joining -- "
                            "use it to discard a seeded overlap region that A already contains.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent_a, latent_b, trim_b_latents) -> io.NodeOutput:
        va, aa = unpack_av(latent_a, "latent_a")
        vb, ab = unpack_av(latent_b, "latent_b")

        if va.shape[3:] != vb.shape[3:]:
            raise ValueError("Spatial mismatch: cannot concatenate latents of different sizes.")

        if trim_b_latents > 0:
            # The latent<->frame relation is only defined ON the 5j+2 grid: 2 latents
            # is 5 frames, and each further group of 5 latents adds 17. Off-grid values
            # make latents_to_frames() go negative (k=1 -> -12 frames), which would
            # slice from the wrong end of the audio. Snap before using it.
            k = snap_latents(min(int(trim_b_latents), vb.shape[VIDEO_T_DIM] - 1))
            if k != int(trim_b_latents):
                logging.info("[MMH3ConcatAV] trim_b_latents %d is off the 5j+2 grid, using %d",
                             int(trim_b_latents), k)
            drop_frames = latents_to_frames(k)
            drop_audio = max(0, min(frames_to_audio_t(drop_frames), ab.shape[AUDIO_T_DIM] - 1))
            vb = vb[:, :, k:, :, :]
            ab = ab[:, :, :, drop_audio:]

        v = torch.cat([va, vb.to(va.dtype)], dim=VIDEO_T_DIM)
        a = torch.cat([aa, ab.to(aa.dtype)], dim=AUDIO_T_DIM)

        total = int(v.shape[VIDEO_T_DIM])
        if not on_grid(total):
            logging.warning(
                "[MMH3ConcatAV] result is %d latents, OFF the 5j+2 grid. Two on-grid chunks "
                "sum to 5(j+k)+4, which is never on-grid, so the causal VAE will misalign "
                "from the join onward and the second half will pulse. Trim 5m+2 from B "
                "(minimum 2), or better: decode the chunks separately and use MMH3JoinAV.",
                total)

        out = dict(latent_a)
        out["samples"] = NestedTensor([v, a])
        out.pop("noise_mask", None)  # a per-frame mask cannot span the join
        return io.NodeOutput(out)
