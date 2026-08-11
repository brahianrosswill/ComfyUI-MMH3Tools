"""Streaming export: decode in chunks straight into ffmpeg, never hold the video.

A full decode allocates the whole pixel tensor at once, and that is the wall you hit
before any other one:

    1344x768, 362f     4.48 GB
    2048x1152, 362f   10.25 GB
    2048x1152, 750f   21.23 GB

against 0.48 GB for a single 17-frame clip at 2K.

WHY THIS IS NOT THE LTX PATTERN
-------------------------------
LTXAVStreamingSave decodes a slice with LEFT context and trims it, because the LTX
video VAE is causal -- past context only. H3's decoder is neither causal nor
independent:

    t_end_idx = t_start_idx + tokens_chunk_size + token_overlap      # 5 + 2
    clip_dec_chunk = self.blend(dec_overlap, clip_dec_chunk, self.frame_overlap)

Each chunk reads TWO latents of LOOKAHEAD, and carries `dec_overlap` forward to blend
5 frames into the next chunk's output. So a slice decoded on its own is wrong at BOTH
ends: its first written part skips a blend it should have had, and its last chunk pads
instead of reading real lookahead.

(Note this is the opposite of MMH3StreamingEncode, where clips encode independently
and chunking is bit-identical for free. Encode and decode are not symmetric here.)

THE SCHEME
----------
Work in groups of `tokens_chunk_size` (5) latents; each group is exactly 17 output
frames, which is the 5j+2 <-> 17j+5 grid seen from the VAE's side.

To emit groups g0..g1-1 exactly, decode latents

    [ 5*g0 - 5 : 5*g1 + 2 ]        (the -5 omitted for g0 == 0)

The leading group is decoded only so its j=1 part becomes the `dec_overlap` that
blends correctly into group g0 -- its own output is discarded. The +2 supplies the
lookahead. Then keep 17 frames per emitted group, starting at frame 17 when there is
a context group.

AND DROP THE LAST 5 FRAMES, except on the final batch. A partial decode always ends
by writing its last chunk's j=1 part raw; in a full decode that part is never written
raw, it is blended into the chunk after it. Keeping them would put a visibly
unblended seam at every batch boundary.

Audio is not decoded here -- it is tiny. Decode it normally and pass it in to be muxed.
"""

import logging
import math
import os
import subprocess
import tempfile

import torch

import comfy.model_management
import folder_paths
from comfy_api.latest import io

from .common import unpack_av

# Fallbacks if the VAE ever stops exposing its own constants. Verified against
# comfy/ldm/minimax/vae.py: clip_length 17, token_drop 3, vae_ratio_t 4.
DEFAULTS = {"clip_length": 17, "token_drop": 3, "vae_ratio_t": 4}

# frames the FINAL decode chunk contributes beyond its groups: the last chunk's
# carried part, written raw only because there is no chunk after it to blend into.
# frame_overlap = token_overlap*vae_ratio_t - frame_pre_padding = 8 - 3 = 5.
TAIL_FRAMES = 5


def vae_grid(vae):
    """(tokens_chunk_size, token_overlap, frames_per_group) read from the VAE itself.

    Read rather than hardcoded, so a checkpoint with different constants is followed
    instead of silently mis-sliced. The caller self-checks the result against the
    17j+5 grid before trusting it.
    """
    inner = getattr(vae, "first_stage_model", None)
    def get(name):
        v = getattr(inner, name, None)
        return DEFAULTS[name] if v is None else int(v)

    clip_length, token_drop, ratio_t = get("clip_length"), get("token_drop"), get("vae_ratio_t")
    tokens_chunk = int(getattr(inner, "tokens_chunk_size", 0) or math.ceil(clip_length / ratio_t))
    overlap = getattr(inner, "token_overlap", None)
    overlap = int((-token_drop) % tokens_chunk if overlap is None else overlap)
    return tokens_chunk, overlap, clip_length


class MMH3StreamingSave(io.ComfyNode):
    """Decode an H3 video latent in chunks and stream it straight into ffmpeg."""

    _ENCODERS = ["libx264", "h264_nvenc", "libopenh264", "mpeg4"]
    _probe_cache = {}

    @staticmethod
    def _encoder_args(name, crf, w, h):
        # -crf and -preset are x264-specific; every encoder needs its own mapping
        if name == "libx264":
            return ["-c:v", "libx264", "-preset", "medium",
                    "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if name == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                    "-cq", str(crf), "-pix_fmt", "yuv420p"]
        if name == "libopenh264":
            mbps = max(1.0, (w * h / (1280 * 720)) * (2.0 ** ((28 - crf) / 6.0)))
            return ["-c:v", "libopenh264", "-b:v", "%.1fM" % mbps, "-pix_fmt", "yuv420p"]
        return ["-c:v", "mpeg4", "-q:v", str(max(1, min(31, crf // 2))), "-pix_fmt", "yuv420p"]

    @classmethod
    def _available(cls, ffmpeg):
        if ffmpeg in cls._probe_cache:
            return cls._probe_cache[ffmpeg]
        found = set()
        try:
            out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                                 capture_output=True, timeout=30)
            text = (out.stdout or b"").decode("utf-8", "replace")
            for enc in cls._ENCODERS:
                if any(line.split()[1:2] == [enc] for line in text.splitlines() if line.strip()):
                    found.add(enc)
        except Exception as e:
            # Report NO encoders rather than optimistically assuming this binary is
            # fine -- otherwise a stale PATH entry shadows a working build.
            logging.warning("[MMH3StreamingSave] ffmpeg candidate unusable, skipping: %s (%s)",
                            ffmpeg, e)
        cls._probe_cache[ffmpeg] = found
        return found

    @classmethod
    def _resolve_ffmpeg(cls, explicit, want):
        """Prefer a binary that HAS a usable encoder over merely the first found.

        A conda ffmpeg built without libx264 sits on PATH and shadows a working
        imageio-ffmpeg build, so 'first on PATH' picks the broken one.
        """
        import shutil
        candidates = []
        if explicit and explicit.strip():
            candidates.append(explicit.strip().strip('"'))
        on_path = shutil.which("ffmpeg")
        if on_path:
            candidates.append(on_path)
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            candidates.append(get_ffmpeg_exe())
        except Exception:
            pass

        seen, ordered = set(), []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        if not ordered:
            raise RuntimeError("[MMH3StreamingSave] no ffmpeg found (not on PATH, "
                               "imageio-ffmpeg unavailable, no ffmpeg_path given).")

        wanted = [want] if want != "auto" else cls._ENCODERS
        for binary in ordered:
            have = cls._available(binary)
            for enc in wanted:
                if enc in have:
                    return binary, enc
        logging.warning("[MMH3StreamingSave] no candidate reported %s -- trying %s anyway",
                        wanted, ordered[0])
        return ordered[0], wanted[0]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3StreamingSave",
            display_name="MMH3 Streaming Save",
            category="MMH3Tools/utils",
            description=(
                "Decode an H3 video latent in chunks straight into ffmpeg. The full "
                "pixel tensor never exists, so RAM is constant at any length - 2K at "
                "750 frames is 21 GB decoded whole. Pass decoded AUDIO to mux."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="H3 AV latent, or a plain 5D video latent."),
                io.Vae.Input("vae", tooltip="The H3 VIDEO vae."),
                io.Int.Input(
                    "groups_per_chunk", default=4, min=1, max=64, step=1,
                    tooltip="Latent GROUPS decoded per call. One group is 5 latents = 17 "
                            "frames, so 4 is ~68 frames live at a time. Each call also "
                            "decodes one extra group of context that is discarded, so "
                            "small values waste proportionally more work.",
                ),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.01,
                               tooltip="H3 is 24. Only affects container timing."),
                io.String.Input("filename_prefix", default="MMH3/stream"),
                io.Int.Input("crf", default=19, min=0, max=51,
                             tooltip="Quality, mapped per encoder: crf (x264) / cq (nvenc) "
                                     "/ bitrate (openh264) / qscale (mpeg4)."),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip="Decoded audio to mux in. Audio decode is cheap, so it stays "
                            "outside this node. Omit for silent video.",
                ),
                io.Combo.Input(
                    "video_encoder", options=["auto"] + cls._ENCODERS, default="auto",
                    optional=True,
                    tooltip="auto probes the binary and takes the first available in "
                            "preference order.",
                ),
                io.String.Input(
                    "ffmpeg_path", default="", optional=True,
                    tooltip="Override the binary. Empty searches this path, then PATH, then "
                            "imageio-ffmpeg, taking the first that actually has a working "
                            "H.264 encoder.",
                ),
            ],
            outputs=[io.String.Output(display_name="file_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, vae, groups_per_chunk, fps, filename_prefix, crf,
                audio=None, video_encoder="auto", ffmpeg_path="") -> io.NodeOutput:
        video, _ = unpack_av(latent, "latent", allow_video_only=True)
        video = video[:1]
        T = int(video.shape[2])

        group, lookahead, frames_per_group = vae_grid(vae)
        n_groups = max(0, (T - 2) // group)
        if n_groups < 1:
            raise ValueError(
                "[MMH3StreamingSave] %d latents is under one group (%d + %d). Decode it "
                "normally; there is nothing to stream." % (T, group, 2))
        tail = T - (n_groups * group + 2)
        if tail:
            logging.warning("[MMH3StreamingSave] %d latents is off the %dj+2 grid by %d; "
                            "the remainder is not emitted", T, group, tail)

        # 17j+5: every group is 17 frames, plus the final chunk's carried part
        expected = frames_per_group * n_groups + TAIL_FRAMES

        ffmpeg, encoder = cls._resolve_ffmpeg(ffmpeg_path, video_encoder)
        logging.info("[MMH3StreamingSave] ffmpeg %s | encoder %s%s | %d groups, %d frames "
                     "expected", ffmpeg, encoder,
                     "" if video_encoder == "auto" else " (forced)", n_groups, expected)

        out_dir = folder_paths.get_output_directory()
        full_folder, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir)
        video_tmp = os.path.join(full_folder, "%s_%05d_tmp.mp4" % (fname, counter))
        final_path = os.path.join(full_folder, "%s_%05d.mp4" % (fname, counter))

        proc = None
        written = 0
        # stderr goes to a FILE, not a pipe: nothing drains a pipe during a long
        # encode, and a chatty ffmpeg would fill it and deadlock.
        err_file = tempfile.TemporaryFile()

        def err_tail():
            try:
                err_file.seek(0)
                t = err_file.read()[-2000:].decode("utf-8", "replace").strip()
            except Exception:
                t = ""
            return ("\n--- ffmpeg stderr ---\n" + t) if t else " (ffmpeg printed no output)"

        try:
            g0 = 0
            while g0 < n_groups:
                comfy.model_management.throw_exception_if_processing_interrupted()
                g1 = min(g0 + int(groups_per_chunk), n_groups)
                last = g1 >= n_groups

                lo = max(0, group * g0 - group)          # one group of context
                hi = min(T, group * g1 + 2)              # +2 lookahead
                px = vae.decode(video[:, :, lo:hi])
                if isinstance(px, tuple):
                    px = px[0]
                if px.ndim == 5:
                    px = px.reshape(-1, *px.shape[-3:])

                head = frames_per_group if g0 > 0 else 0
                keep = frames_per_group * (g1 - g0)
                # the trailing 5 are this call's last chunk's carried part, written raw
                # here but BLENDED in a full decode -- keep them only at the true end
                px = px[head:head + keep + (TAIL_FRAMES if last else 0)]

                if proc is None:
                    H, W = int(px.shape[1]), int(px.shape[2])
                    if (W % 2) or (H % 2):
                        raise ValueError(
                            "[MMH3StreamingSave] frame size %dx%d has an odd dimension; "
                            "yuv420p needs both even. H3 latents are always /16 of a "
                            "/32 canvas, so a hand-cropped latent is the usual cause."
                            % (W, H))
                    proc = subprocess.Popen(
                        [ffmpeg, "-y", "-loglevel", "error",
                         "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "pipe:"]
                        + cls._encoder_args(encoder, crf, W, H) + [video_tmp],
                        stdin=subprocess.PIPE, stderr=err_file)

                data = (px.clamp(0, 1).mul(255).round()
                        .to(torch.uint8).cpu().contiguous().numpy().tobytes())
                try:
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError):
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                    ret = proc.wait()
                    proc = None
                    raise RuntimeError("[MMH3StreamingSave] ffmpeg died mid-stream "
                                       "(exit %s).%s" % (ret, err_tail()))
                written += int(px.shape[0])
                logging.info("[MMH3StreamingSave] groups [%d,%d) of %d -> %d frames "
                             "(total %d)", g0, g1, n_groups, int(px.shape[0]), written)
                del px, data
                g0 = g1

            proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError("[MMH3StreamingSave] ffmpeg exited with %s.%s"
                                   % (ret, err_tail()))
            proc = None
        finally:
            if proc is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.kill()
            err_file.close()

        if written != expected:
            logging.warning("[MMH3StreamingSave] wrote %d frames, expected %d. The VAE's "
                            "chunking no longer matches the %dj+2 grid this slices on.",
                            written, expected, group)

        if audio is not None and audio.get("waveform") is not None:
            import torchaudio
            wav_tmp = os.path.join(full_folder, "%s_%05d_tmp.wav" % (fname, counter))
            wf = audio["waveform"]
            if wf.ndim == 3:
                wf = wf[0]
            torchaudio.save(wav_tmp, wf.cpu(), int(audio["sample_rate"]))
            try:
                mux = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i", video_tmp, "-i", wav_tmp,
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", final_path],
                    stderr=subprocess.PIPE)
                if mux.returncode != 0:
                    t = (mux.stderr[-2000:].decode("utf-8", "replace").strip()
                         if mux.stderr else "")
                    raise RuntimeError(
                        "[MMH3StreamingSave] audio mux failed (ffmpeg exit %s); the silent "
                        "video is kept at %s%s"
                        % (mux.returncode, video_tmp,
                           ("\n--- ffmpeg stderr ---\n" + t) if t else ""))
            finally:
                try:
                    os.remove(wav_tmp)
                except OSError:
                    pass
            os.remove(video_tmp)
        else:
            os.replace(video_tmp, final_path)

        logging.info("[MMH3StreamingSave] %d frames (%.2fs) -> %s",
                     written, written / float(fps), final_path)
        return io.NodeOutput(
            final_path,
            ui={"images": [{"filename": os.path.basename(final_path),
                            "subfolder": subfolder, "type": "output"}],
                "animated": (True,)},
        )
