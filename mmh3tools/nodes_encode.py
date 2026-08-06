"""Chunked VAE encode, so long clips at high resolution can be encoded at all.

`F.pad(..., mode="reflect")` -- used by H3's CausalConv3d for its spatial padding --
requires the tensor to fit 32-bit indexing, i.e. under 2**31 = 2,147,483,648
elements. A pixel batch is [1, 3, T, H, W], so the ceiling is a joint limit on
length AND resolution:

    1024x768    906 frames  (37.7s)
    1536x1152   396 frames  (16.5s)
    2048x1536   226 frames  ( 9.4s)

Past that, `VAEEncode` dies with "input tensor must fit into 32-bit index math".
That is NOT an out-of-memory error, so `model_management.raise_non_oom()` re-raises
it and ComfyUI's automatic retry-with-tiled-encoding never fires -- you get a hard
stop rather than a slow fallback. The ceiling shrinks as an upscale ladder climbs,
so a length that sails through stage 1 can fail at stage 3.

WHY CHUNKING IS EXACT HERE. `encode_temporal` slices into NON-OVERLAPPING 17-frame
clips and encodes each with no carried state:

    for i in range(num_chunks):
        clip_x = x[:, :, i*17:(i+1)*17, :, :]
        z_list.append(self._adaptive_encode(clip_x))
    z = torch.cat(z_list, dim=2)
    if self.token_drop > 0:
        z = z[:, :, :-self.token_drop]

So clip boundaries are free -- unlike LTX, whose encoder has a causal receptive
field across boundaries and needs left context re-encoded and trimmed per chunk.

THE TRAP. The tail padding and `token_drop` are applied once PER CALL, so looping
`vae.encode()` over chunks silently loses 3 latents per chunk: 39 frames encode to
12 latents whole, but 2+2+2 = 6 as three calls. Not an error, just a shorter latent
that decodes to a shorter, wrong video. This node therefore drives `_adaptive_encode`
directly and applies the pad and the drop exactly once, then reproduces `encode()`'s
moments-to-latent step.

Verified bit-identical to a whole-tensor encode -- max|diff| 0.00e+00 at 39 and 124
frames across chunk sizes 17, 34 and 85.

NOTE this raises the length ceiling; it does not by itself give constant RAM. The
incoming IMAGE batch already exists in full before the node runs. Constant RAM needs
reading frames from disk per chunk, the way LTXAVTools' streaming encode does.
"""

import logging

import torch

import comfy.model_management
from comfy_api.latest import io


def _h3_video_vae(vae):
    """The inner H3 video VAE, or None if this isn't one."""
    m = getattr(vae, "first_stage_model", None)
    need = ("clip_length", "token_drop", "_adaptive_encode", "pixel_mean", "pixel_std",
            "latents_mean", "latents_std")
    if m is not None and all(hasattr(m, a) for a in need):
        return m
    return None


class MMH3StreamingEncode(io.ComfyNode):
    """VAE encode in chunks, bypassing the 32-bit index ceiling."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3StreamingEncode",
            display_name="MiniMax H3 Streaming Encode",
            category="MMH3Tools",
            description=(
                "Encode a long video in chunks. Drop-in for VAEEncode when the clip is "
                "too long for its resolution -- past ~226 frames at 2048x1536, VAEEncode "
                "fails with 'input tensor must fit into 32-bit index math'. Output is "
                "bit-identical to a whole-tensor encode."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Vae.Input("vae", tooltip="The H3 VIDEO vae."),
                io.Int.Input(
                    "frames_per_chunk", default=85, min=17, max=1700, step=17,
                    tooltip="Frames encoded per pass, snapped to a multiple of 17 (the "
                            "VAE's clip length). Smaller means lower peak memory and more "
                            "passes. The result does not change: clips are encoded "
                            "independently, so any chunk size gives an identical latent.",
                ),
                io.Boolean.Input(
                    "offload_latents", default=True, optional=True,
                    tooltip="Move each chunk's latents to CPU as they are produced. Keeps "
                            "accumulated latents off the GPU on very long clips; costs a "
                            "transfer per chunk.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, images, vae, frames_per_chunk, offload_latents=True) -> io.NodeOutput:
        m = _h3_video_vae(vae)
        if m is None:
            raise ValueError(
                "MMH3StreamingEncode needs the MiniMax H3 VIDEO vae. The vae given is a "
                "%s, which has no 17-frame clip structure to chunk on. (The H3 AUDIO vae "
                "is not encodable this way either -- use VAEEncodeAudio.)"
                % type(getattr(vae, "first_stage_model", vae)).__name__)

        clip = int(m.clip_length)
        fpc = max(clip, (int(frames_per_chunk) // clip) * clip)
        n_frames = int(images.shape[0])

        # VAE.encode() loads the model itself; going around it means doing that here
        pixels = vae.process_input(images)
        try:
            # budget for ONE chunk, not the whole clip -- that is the point of this node
            mem = int(vae.memory_used_encode(pixels[:min(n_frames, fpc)].shape, vae.vae_dtype))
        except Exception as e:
            logging.info("[MMH3StreamingEncode] memory_used_encode unavailable (%s); "
                         "loading without a budget hint", type(e).__name__)
            mem = 0  # load_models_gpu adds this to a reserve and cannot take None
        comfy.model_management.load_models_gpu(
            [vae.patcher], memory_required=mem, force_full_load=vae.disable_offload)

        with torch.inference_mode():
            x = pixels.to(vae.vae_dtype).to(vae.device)
            x = x.movedim(-1, 1).movedim(1, 0).unsqueeze(0)          # [1, 3, T, H, W]
            # encode()'s normalisation, applied once over the whole batch
            x = x.add(1.0).mul_(0.5).sub_(m.pixel_mean.to(x)).div_(m.pixel_std.to(x))

            # tail-pad to a whole number of clips -- ONCE, not per chunk
            pad = (-x.shape[2]) % clip
            if pad:
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)

            zs = []
            n_clips = x.shape[2] // clip
            for i in range(0, x.shape[2], fpc):
                chunk = x[:, :, i:i + fpc]
                for j in range(chunk.shape[2] // clip):
                    z = m._adaptive_encode(chunk[:, :, j * clip:(j + 1) * clip])
                    zs.append(z.cpu() if offload_latents else z)
                comfy.model_management.throw_exception_if_processing_interrupted()

            moments = torch.cat(zs, dim=2)
            del zs, x
            # token_drop is what turns 5j clips into the 5j+2 grid -- ONCE, at the end
            if m.token_drop > 0:
                moments = moments[:, :, :-int(m.token_drop)]

            mean = torch.chunk(moments.float(), 2, dim=1)[0]
            lm = m.latents_mean.view(1, -1, 1, 1, 1).to(mean)
            ls = m.latents_std.view(1, -1, 1, 1, 1).to(mean)
            out = ((mean - lm) / ls).to(vae.output_device).to(vae.vae_output_dtype())

        n_passes = (n_clips + (fpc // clip) - 1) // max(1, fpc // clip)
        label = ("%d frames -> %d latents, %d clips in %d pass%s of %d frames"
                 % (n_frames, int(out.shape[2]), n_clips, n_passes,
                    "" if n_passes == 1 else "es", fpc))
        if fpc != int(frames_per_chunk):
            label += "\n  ! frames_per_chunk %d -> %d (multiple of %d)" % (
                int(frames_per_chunk), fpc, clip)
        if pad:
            label += "\n  padded %d frame%s to fill the last clip" % (pad, "" if pad == 1 else "s")
        logging.info("[MMH3StreamingEncode] " + label.splitlines()[0])
        return io.NodeOutput({"samples": out}, label)
