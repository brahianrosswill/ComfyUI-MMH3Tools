"""Drive MMH3ChunkedPixelUpscale end to end on a stub VAE.

The stub implements only what the node touches, but implements it with the REAL
shapes and the real grid, so the control flow -- decode slicing, context/tail trim,
per-clip encode, one token_drop -- runs exactly as it would against weights. What
this cannot check is pixel fidelity; what it can check is that an AV latent goes in
and a correctly shaped AV latent comes out.
"""
import os, sys, types
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor
from mmh3tools.nodes_upscale import MMH3ChunkedPixelUpscale

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


CLIP, GROUP, DROP = 17, 5, 3

def make_vae(src_w=128, src_h=96):
    """A stub with the real interface: decode -> [B,F,H,W,C], _adaptive_encode -> moments."""
    inner = types.SimpleNamespace(
        clip_length=CLIP, token_drop=DROP, vae_ratio_t=4,
        tokens_chunk_size=GROUP, token_overlap=2,
        pixel_mean=torch.zeros(3, 1, 1, 1), pixel_std=torch.ones(3, 1, 1, 1),
        latents_mean=torch.zeros(24), latents_std=torch.ones(24))

    def adaptive_encode(x):
        # x is [1,3,17,H,W] -> moments [1,48,5,H/16,W/16]
        b, c, t, h, w = x.shape
        assert t == CLIP, "encoder was handed %d frames, not a whole clip" % t
        calls["encode"].append((h, w))
        return torch.zeros(b, 48, GROUP, h // 16, w // 16)
    inner._adaptive_encode = adaptive_encode

    vae = types.SimpleNamespace(
        first_stage_model=inner, vae_dtype=torch.float32,
        device=torch.device("cpu"), output_device=torch.device("cpu"),
        vae_output_dtype=lambda: torch.float32,
        process_input=lambda p: p * 2.0 - 1.0)

    def decode(z):
        # [B,24,t,h,w] -> [B, 17*(t-2)//5 + 5, H, W, 3], the 17j+5 grid
        t = int(z.shape[2])
        f = CLIP * ((t - 2) // GROUP) + 5
        calls["decode"].append((int(z.shape[2]), f))
        return torch.rand(1, f, src_h, src_w, 3)
    vae.decode = decode
    return vae


def av_latent(T=57, lh=6, lw=8, batch=1, audio_t=207):
    v = torch.rand(batch, 24, T, lh, lw)
    a = torch.rand(batch, 32, 2, audio_t)
    return {"samples": NestedTensor([v, a])}


print("\n1. an AV NestedTensor goes in and an AV NestedTensor comes out")
calls = {"decode": [], "encode": []}
src = av_latent(T=57)
out, report = MMH3ChunkedPixelUpscale.execute(
    src, make_vae(), 256, 192, "bilinear", 4).result
check("output samples is a NestedTensor", isinstance(out["samples"], NestedTensor), True)
ov, oa = out["samples"].unbind()
check("video latent shape", tuple(ov.shape), (1, 24, 57, 192 // 16, 256 // 16))
check("temporal length preserved", int(ov.shape[2]), 57)
check("audio carried UNCHANGED", bool(torch.equal(oa, src["samples"].unbind()[1])), True)
check("audio dtype matches video", oa.dtype, ov.dtype)
check("audio device matches video", oa.device, ov.device)

print("\n2. no stale noise_mask survives the resize")
src2 = av_latent(T=57)
src2["noise_mask"] = NestedTensor([torch.ones(1, 1, 57, 6, 8), torch.zeros(1, 1, 2, 207)])
out2, _ = MMH3ChunkedPixelUpscale.execute(
    src2, make_vae(), 256, 192, "bilinear", 4).result
# a mask cut for the 768p grid is the wrong shape at 2K; carrying it would be worse
# than dropping it, because the sampler would apply it to the wrong rows
check("noise_mask not carried at the wrong size",
      "noise_mask" not in out2 or
      tuple(out2["noise_mask"].unbind()[0].shape[2:]) == tuple(ov.shape[2:]), True)

print("\n3. a plain video-only latent still works (no audio to carry)")
calls = {"decode": [], "encode": []}
plain = {"samples": torch.rand(1, 24, 57, 6, 8)}
out3, rep3 = MMH3ChunkedPixelUpscale.execute(
    plain, make_vae(), 256, 192, "bilinear", 4).result
check("output is a plain tensor", isinstance(out3["samples"], NestedTensor), False)
check("video shape", tuple(out3["samples"].shape), (1, 24, 57, 12, 16))
check("report says so", "no audio to carry" in rep3, True)

print("\n4. the encoder only ever sees whole 17-frame clips")
calls = {"decode": [], "encode": []}
MMH3ChunkedPixelUpscale.execute(av_latent(T=107), make_vae(), 256, 192, "bilinear", 4)
check("every encode call was one clip at target res",
      all(hw == (192, 256) for hw in calls["encode"]), True)
check("clip count = groups + 1 (the padded tail)",
      len(calls["encode"]), (107 - 2) // GROUP + 1)

print("\n5. chunking is invisible to the result shape")
for gpc in (1, 2, 3, 4, 100):
    calls = {"decode": [], "encode": []}
    o, _ = MMH3ChunkedPixelUpscale.execute(
        av_latent(T=107), make_vae(), 256, 192, "bilinear", gpc).result
    v = o["samples"].unbind()[0]
    check("gpc=%-3d -> 107 latents" % gpc, int(v.shape[2]), 107)
    check("gpc=%-3d -> %d clips regardless" % (gpc, 22), len(calls["encode"]), 22)

print("\n6. batch > 1 is reduced consistently, not left mismatched")
calls = {"decode": [], "encode": []}
o6, _ = MMH3ChunkedPixelUpscale.execute(
    av_latent(T=57, batch=3), make_vae(), 256, 192, "bilinear", 4).result
v6, a6 = o6["samples"].unbind()
check("video batch", int(v6.shape[0]), 1)
check("audio batch MATCHES video", int(a6.shape[0]), int(v6.shape[0]))

print("\n7. the decode slicing asked for context and lookahead")
calls = {"decode": [], "encode": []}
MMH3ChunkedPixelUpscale.execute(av_latent(T=107), make_vae(), 256, 192, "bilinear", 4)
# first pass has no left context (g0 == 0) but always takes the +2 lookahead
check("first slice is groups*4 + 2", calls["decode"][0][0], 4 * GROUP + 2)
# later passes take one extra group of context on the left
check("second slice adds a context group", calls["decode"][1][0], 5 * GROUP + 2)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
