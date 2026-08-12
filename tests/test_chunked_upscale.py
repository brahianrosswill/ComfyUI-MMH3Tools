import os, sys, math, types
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
import comfy.ldm.minimax.vae as V
from mmh3tools.nodes_save import TAIL_FRAMES, vae_grid
from mmh3tools.nodes_upscale import MMH3ChunkedPixelUpscale, upscale_frames
from mmh3tools.nodes_encode import INDEX_LIMIT

K = V.MiniMaxH3VideoVAE

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


print("\n1. THE POINT: the decode grid and the encode grid are the same 17")
# decode emits 17 frames per group of 5 latents; encode consumes non-overlapping
# 17-frame clips. If these ever disagree, a decode batch cannot be handed to the
# encoder whole and the node refuses rather than mis-slicing.
fake = types.SimpleNamespace(first_stage_model=types.SimpleNamespace(
    clip_length=17, token_drop=3, vae_ratio_t=4))
group, lookahead, fpg = vae_grid(fake)
check("group size", group, 5)
check("frames per group", fpg, 17)
check("== the encode clip length", fpg, 17)
check("tail frames", TAIL_FRAMES, 5)

print("\n2. the round trip preserves length: 5j+2 -> 17j+5 -> pad -> 5j+2")
def roundtrip(T, clip=17, token_drop=3, group=5, fpg=17):
    n_groups = (T - 2) // group
    frames = fpg * n_groups + TAIL_FRAMES        # what the decode emits in total
    pad = (-frames) % clip                       # the final chunk is topped up
    n_clips = (frames + pad) // clip
    latents = n_clips * group - token_drop       # 5 latents per clip, drop once
    return frames, pad, latents

for T in (7, 12, 22, 57, 107, 192):
    frames, pad, latents = roundtrip(T)
    check("T=%3d -> %4d frames (pad %2d) -> %3d latents" % (T, frames, pad, latents),
          latents, T)

print("\n3. token_drop must be applied ONCE, not per chunk")
# the trap MMH3StreamingEncode documents: dropping per chunk silently loses 3
# latents each time, giving a shorter latent that decodes to a wrong-length video
T, group, clip, drop = 107, 5, 17, 3
n_groups = (T - 2) // group
for gpc in (1, 2, 4, 100):
    n_passes = (n_groups + gpc - 1) // gpc
    correct = roundtrip(T)[2]
    per_chunk = correct - drop * (n_passes - 1)
    check("gpc=%-3d one drop is right" % gpc, correct, T)
    if n_passes > 1:
        check("gpc=%-3d per-chunk dropping would lose %d" % (gpc, drop * (n_passes - 1)),
              per_chunk < T, True)

print("\n4. chunking does not change WHICH frames are produced")
# groups_per_chunk only sets how many groups a pass covers; the emitted frame count
# and the clip boundaries must be identical however the work is split
for T in (57, 107, 192):
    n_groups = (T - 2) // 5
    total_ref = 17 * n_groups + TAIL_FRAMES
    for gpc in (1, 2, 3, 4, 100):
        emitted, g0 = 0, 0
        while g0 < n_groups:
            g1 = min(g0 + gpc, n_groups)
            emitted += 17 * (g1 - g0) + (TAIL_FRAMES if g1 >= n_groups else 0)
            g0 = g1
        check("T=%3d gpc=%-3d emits %d" % (T, gpc, total_ref), emitted, total_ref)

print("\n5. every chunk except the last is already a whole number of clips")
# so only the final chunk ever needs padding -- an interior pad would insert
# duplicated frames into the middle of the timeline
for gpc in (1, 2, 4):
    n_groups, g0, interior_pads = 21, 0, 0
    while g0 < n_groups:
        g1 = min(g0 + gpc, n_groups)
        last = g1 >= n_groups
        f = 17 * (g1 - g0) + (TAIL_FRAMES if last else 0)
        if not last and f % 17:
            interior_pads += 1
        g0 = g1
    check("gpc=%d interior chunks needing pad" % gpc, interior_pads, 0)

print("\n6. the 32-bit cap is computed from the TARGET resolution")
# it bites harder here than in StreamingEncode because the frames are upscaled
for w, h, want_min in ((2688, 1536, 1), (4096, 2304, 1), (1344, 768, 4)):
    max_frames = (INDEX_LIMIT - 1) // (3 * h * w)
    max_groups = max(1, max_frames // 17)
    check("%dx%d allows >=%d group(s)" % (w, h, want_min), max_groups >= want_min, True)
    check("%dx%d one group is addressable" % (w, h), 3 * 17 * h * w < INDEX_LIMIT, True)

print("\n7. upscale_frames: shape, range, dtype, device preserved")
src = torch.rand(6, 48, 84, 3)
for method in ["lanczos-ish bicubic", "bilinear", "nearest-exact"]:
    out = upscale_frames(src, 168, 96, method, sub_batch=4)
    check("%-20s shape" % method, tuple(out.shape), (6, 96, 168, 3))
    check("%-20s dtype" % method, out.dtype, src.dtype)
    check("%-20s device" % method, out.device, src.device)
    check("%-20s clamped to 0..1" % method,
          bool(out.min() >= 0.0 and out.max() <= 1.0), True)

print("\n8. sub-batching does not change the torch result")
for method in ["lanczos-ish bicubic", "nearest-exact"]:
    a = upscale_frames(src, 168, 96, method, sub_batch=1)
    b = upscale_frames(src, 168, 96, method, sub_batch=100)
    check("%-20s sub_batch invariant" % method, bool(torch.equal(a, b)), True)

print("\n9. a no-op size returns the input untouched")
same = upscale_frames(src, 84, 48, "bilinear")
check("identical object", same is src, True)

print("\n10. rtx_vsr is optional: a torch method must not need nvvfx")
import builtins
real_import = builtins.__import__
def blocked(name, *a, **k):
    if name == "nvvfx":
        raise ImportError("blocked for test")
    return real_import(name, *a, **k)
builtins.__import__ = blocked
try:
    out = upscale_frames(src, 168, 96, "bilinear")
    check("torch path works with nvvfx unavailable", tuple(out.shape), (6, 96, 168, 3))
    try:
        upscale_frames(src, 168, 96, "rtx_vsr")
        check("rtx path raises when nvvfx is missing", False, True)
    except RuntimeError as e:
        check("rtx path raises RuntimeError naming the pack",
              "comfyui_nvidia_rtx_nodes" in str(e), True)
finally:
    builtins.__import__ = real_import

print("\n11. schema: registered, named, and 32px-grid guarded")
from mmh3tools import NODES
check("exported", MMH3ChunkedPixelUpscale in NODES, True)
s = MMH3ChunkedPixelUpscale.define_schema()
check("node_id", s.node_id, "MMH3ChunkedPixelUpscale")
check("display_name", s.display_name, "MMH3 Chunked Pixel Upscale")
check("category", s.category, "MMH3Tools/latent")
check("input order", [i.id for i in s.inputs],
      ["latent", "vae", "width", "height", "method", "groups_per_chunk",
       "rtx_quality", "offload_latents"])
print("\n12. the 32px guard is enforced by the NODE, not just documented")
# latent dims are px/16 and must stay EVEN for the 2x2 patch, so both axes must be
# multiples of 32. Driven through execute() with a stub vae that satisfies
# _h3_video_vae, so the guard itself runs rather than a restatement of its rule.
stub_inner = types.SimpleNamespace(
    clip_length=17, token_drop=3, vae_ratio_t=4,
    _adaptive_encode=lambda *a, **k: None,
    pixel_mean=torch.zeros(3), pixel_std=torch.ones(3),
    latents_mean=torch.zeros(24), latents_std=torch.ones(24))
stub_vae = types.SimpleNamespace(first_stage_model=stub_inner)
dummy = {"samples": torch.zeros(1, 24, 57, 48, 84)}

def run_dims(w, h):
    return MMH3ChunkedPixelUpscale.execute(
        dummy, stub_vae, w, h, "bilinear", 4)

for w, h, should_raise in ((2000, 1152, True), (2688, 1500, True), (2688, 1537, True)):
    try:
        run_dims(w, h)
        check("%dx%d rejected" % (w, h), False, True)
    except ValueError as e:
        check("%dx%d rejected with the patch-grid reason" % (w, h),
              "32px canvas grid" in str(e), should_raise)
    except Exception as e:
        check("%dx%d rejected (got %s)" % (w, h, type(e).__name__), False, True)

# a VALID pair must get PAST the dims guard -- it will fail later on the stub's
# no-op encode, and that is the proof it cleared this check rather than short-circuiting
try:
    run_dims(2688, 1536)
    passed_guard = True
except ValueError as e:
    passed_guard = "32px canvas grid" not in str(e)
except Exception:
    passed_guard = True
check("2688x1536 clears the dims guard", passed_guard, True)

# and a non-H3 vae is refused before any of it
try:
    MMH3ChunkedPixelUpscale.execute(dummy, types.SimpleNamespace(first_stage_model=None),
                                    2688, 1536, "bilinear", 4)
    check("non-H3 vae refused", False, True)
except ValueError as e:
    check("non-H3 vae refused by name", "VIDEO vae" in str(e), True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
