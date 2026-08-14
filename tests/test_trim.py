import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor
from mmh3tools.nodes_trim import MMH3TrimAV, MMH3SplitAV
from mmh3tools.nodes_loop import MMH3PackAV
from mmh3tools.common import (AUDIO_T_DIM, VIDEO_T_DIM, frames_to_audio_t,
                              latents_to_frames, on_grid)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


def mk(t_lat, masked=False, ramp=False):
    """AV latent whose values encode their own index, so a mis-slice is visible."""
    at = frames_to_audio_t(latents_to_frames(t_lat))
    if ramp:
        v = torch.arange(t_lat, dtype=torch.float32).reshape(1, 1, t_lat, 1, 1).expand(1, 24, t_lat, 4, 4).clone()
        a = torch.arange(at, dtype=torch.float32).reshape(1, 1, 1, at).expand(1, 32, 2, at).clone()
    else:
        v = torch.zeros([1, 24, t_lat, 4, 4]); a = torch.zeros([1, 32, 2, at])
    d = {"samples": NestedTensor([v, a])}
    if masked:
        vm = torch.ones([1, 1, t_lat, 4, 4]); vm[:, :, :2] = 0.0
        am = torch.ones([1, 1, 2, at]);       am[:, :, :, :8] = 0.0
        d["noise_mask"] = NestedTensor([vm, am])
    return d, t_lat, at


def shapes(d):
    v, a = d["samples"].unbind()
    return int(v.shape[VIDEO_T_DIM]), int(a.shape[AUDIO_T_DIM])


print("\n1. trimming nothing is identity")
d, t, at = mk(57)
out, kept, frames, rep = MMH3TrimAV.execute(d, 0, 0).result
check("shape unchanged", shapes(out), (t, at))
check("kept count", kept, t)
check("kept frames", frames, latents_to_frames(t))

print("\n2. head and tail trims land where they say")
for head, tail in [(5, 0), (0, 5), (7, 7), (12, 5), (2, 2)]:
    out, kept, _, _ = MMH3TrimAV.execute(mk(57)[0], head, tail).result
    check("head %d tail %d -> %d latents" % (head, tail, 57 - head - tail), kept, 57 - head - tail)

print("\n3. the audio cut comes from the BOUNDARY, not from scaling the dropped count")
# Two separate traps. audio_t = round(frames/24*40) is not additive, AND
# latents_to_frames() is only meaningful ON the 5j+2 grid -- it floors to the group
# below for anything else. So an on-grid head matches the simple difference, while an
# off-grid one must INTERPOLATE, which is what _audio_index_at does and what any
# formula built on latents_to_frames gets wrong.
from mmh3tools.nodes_windows import _audio_index_at
d, t, at = mk(57)
for head in (2, 7, 12, 17):                       # all on the grid
    out, _, _, _ = MMH3TrimAV.execute(d, head, 0).result
    _, got_a = shapes(out)
    simple = frames_to_audio_t(latents_to_frames(t)) - frames_to_audio_t(latents_to_frames(head))
    check("on-grid head %d matches the simple difference" % head, got_a, simple)
for head in (5, 10):                              # off the grid
    out, _, _, _ = MMH3TrimAV.execute(d, head, 0).result
    _, got_a = shapes(out)
    check("off-grid head %d uses the interpolated boundary" % head,
          got_a, at - _audio_index_at(head, t, at))
    naive = at - frames_to_audio_t(latents_to_frames(head))
    print("        (flooring latents_to_frames would give %d -- off by %d)"
          % (naive, abs(naive - got_a)))

print("\n4. the slice is taken from the right END, proven by a ramp")
d, t, at = mk(57, ramp=True)
out, _, _, _ = MMH3TrimAV.execute(d, 12, 5).result
v, a = out["samples"].unbind()
check("first kept video latent is index 12", int(v[0, 0, 0, 0, 0]), 12)
check("last kept video latent is index 51", int(v[0, 0, -1, 0, 0]), 51)
check("audio starts after the head cut", int(a[0, 0, 0, 0]) > 0, True)

print("\n5. masks take the IDENTICAL computed cut")
d, t, at = mk(57, masked=True)
out, _, _, _ = MMH3TrimAV.execute(d, 7, 3, True).result
vt, a_t = shapes(out)
vm, am = out["noise_mask"].unbind()
check("video mask tracks video", int(vm.shape[VIDEO_T_DIM]), vt)
check("audio mask tracks audio", int(am.shape[AUDIO_T_DIM]), a_t)
out, _, _, _ = MMH3TrimAV.execute(mk(57, masked=True)[0], 7, 3, False).result
check("carry_masks=False drops it", "noise_mask" in out, False)

print("\n6. it refuses to leave a stub rather than producing an undecodable latent")
try:
    MMH3TrimAV.execute(mk(12)[0], 6, 6)
    check("over-trim raises", False, True)
except ValueError as e:
    check("over-trim raises", "minimum" in str(e), True)

print("\n7. the report names the grid consequence")
# the rule INVERTS relative to ConcatAV: trimming one latent, 5m keeps it on grid
# (5(j-m)+2) and 5m+2 takes it off (5(j-m)). In ConcatAV the constraint is on the
# joined TOTAL, so the families swap. Same arithmetic, different subject.
_, kept, _, rep = MMH3TrimAV.execute(mk(57)[0], 5, 0).result
check("5m trim leaves 52, ON grid", (on_grid(kept), kept), (True, 52))
check("and the report says so", "keeps the result on grid" in rep, True)
_, kept, _, rep = MMH3TrimAV.execute(mk(57)[0], 7, 0).result
check("5m+2 trim leaves 50, OFF grid", (on_grid(kept), kept), (False, 50))
check("and the report warns", "takes the result OFF grid" in rep, True)
_, kept, _, _ = MMH3TrimAV.execute(mk(57)[0], 0, 5).result
check("tail 5 leaves 52, on grid", on_grid(kept), True)

print("\n8. SplitAV is PackAV's inverse -- the round-trip is exact")
d, t, at = mk(57, ramp=True)
vlat, alat, nv, na, rep = MMH3SplitAV.execute(d).result
check("video count", nv, t)
check("audio count", na, at)
check("video is plain 5D", list(vlat["samples"].shape), [1, 24, t, 4, 4])
check("audio is plain 4D, stereo on dim 2", list(alat["samples"].shape), [1, 32, 2, at])

repacked = MMH3PackAV.execute(vlat, alat).result[0]
rv, ra = repacked["samples"].unbind()
ov, oa = d["samples"].unbind()
check("round-trip video is bit-identical", bool(torch.equal(rv, ov)), True)
check("round-trip audio is bit-identical", bool(torch.equal(ra, oa)), True)

print("\n9. split -> trim -> pack composes")
out, kept, _, _ = MMH3TrimAV.execute(d, 12, 0).result
v2, a2, n2, _, _ = MMH3SplitAV.execute(out).result
check("split after trim agrees", n2, kept)
check("and its audio matches the trimmed latent", int(a2["samples"].shape[-1]),
      shapes(out)[1])

print("\n10. a mismatched pair is reported, not silently accepted")
v = torch.zeros([1, 24, 57, 4, 4])
a = torch.zeros([1, 32, 2, 99])                      # wrong length for 57 latents
_, _, _, _, rep = MMH3SplitAV.execute({"samples": NestedTensor([v, a])}).result
check("mismatch flagged", "not built together" in rep, True)

print("\n11. MMH3OutpaintLatent: zero margin, feather ramping INWARD")
from mmh3tools.nodes_trim import MMH3OutpaintLatent as OUT
from mmh3tools.common import pack_av as _pack

def outp(**kw):
    v = torch.ones([1, 24, 12, 48, 84])                     # content is all 1.0
    a = torch.zeros([1, 32, 2, frames_to_audio_t(latents_to_frames(12))])
    args = {"left": 0, "right": 0, "top": 0, "bottom": 0, "feather": 0}
    args.update(kw)
    return OUT.execute(_pack({}, v, a), args["left"], args["right"],
                       args["top"], args["bottom"], args["feather"]).result

out, w, h, rep = outp(top=256, bottom=256, feather=64)
v, _ = out["samples"].unbind()
m, am = out["noise_mask"].unbind()
check("padded on the temporal-preserving axes only", list(v.shape), [1, 24, 12, 80, 84])
check("reported pixel size", (w, h), (84 * 16, 80 * 16))

# the margin is ZEROS, not encoded padding -- nothing for the model to preserve
check("margin is exactly zero", float(v[0, 0, 0, :16, :].abs().max()), 0.0)
check("source survives in the middle", float(v[0, 0, 0, 40, 40]), 1.0)

# mask: 1 in the margin, 0 deep inside, ramped in the source band next to the seam
check("margin marked GENERATE", float(m[0, 0, 0, 0, 40]), 1.0)
check("source centre marked PRESERVE", float(m[0, 0, 0, 40, 40]), 0.0)
band = m[0, 0, 0, 16:20, 40]                                # first 4 latent rows of source
check("the ramp is strictly decreasing INTO the source",
      bool(all(float(band[i]) > float(band[i + 1]) for i in range(len(band) - 1))), True)
check("and it starts high at the seam", float(band[0]) > 0.7, True)

check("audio mask is all-preserve", float(am.max()), 0.0)
check("audio itself is untouched", int(out["samples"].unbind()[1].shape[-1]),
      frames_to_audio_t(latents_to_frames(12)))

print("\n12. what the feather does to the timestep is reported, not hidden")
# The feather blends the LATENT continuously. What happens to the per-row TIMESTEP
# depends on the core: the old #15375 binarised at 0.5 (crude but self-consistent),
# the rebased one gives every ramped cell its own timestep -- which is what makes a
# feather noisy at the seam (observed 2026-08-13). The report must name both.
check("report counts ramped cells", "ramped cells" in rep, True)
check("...and how many cross 0.5", "above 0.5" in rep, True)
check("...and warns a ramp is noisy on a current core", "noisy at the seam" in rep, True)

print("\n13. padding is snapped, and the patch grid is protected")
_, w2, _, rep2 = outp(left=100, feather=0)                  # 100 -> 96
check("100px snaps to 96", "left 100 -> 96" in rep2, True)
check("width reflects the snap", w2, 84 * 16 + 96)
_, _, _, rep3 = outp(top=32, feather=8)                     # 8px < one latent cell
check("a sub-cell feather is called out", "rounds to nothing" in rep3, True)
try:
    outp(feather=64)
    check("no padding is refused", False, True)
except ValueError as e:
    check("no padding is refused", "nothing to do" in str(e), True)

print("\n14. MMH3ReframePads: the three modes are a real trade")
from mmh3tools.nodes_util import MMH3ReframePads as RF, REFRAME_LABELS
from mmh3tools.common import MAX_PIXELS

VERT = REFRAME_LABELS[0]                                    # 9:16
def rf(w, h, ratio, mode, anchor="center"):
    r = RF.execute(w, h, ratio, mode, anchor).result
    mv = r[0:4]                                  # SIGNED: + pads out, - crops in
    return {"moves": mv,
            "pad": tuple(max(0, v) for v in mv),
            "crop": tuple(max(0, -v) for v in mv),
            "size": (r[4], r[5]), "report": r[6]}

ext = rf(1344, 768, VERT, "extend")
crp = rf(1344, 768, VERT, "crop")
bal = rf(1344, 768, VERT, "balanced")
check("extend only pads", (any(ext["pad"]), any(ext["crop"])), (True, False))
check("crop only crops", (any(crp["pad"]), any(crp["crop"])), (False, True))
check("balanced does both", (any(bal["pad"]), any(bal["crop"])), (True, True))

sp = 1344 * 768
check("extend grows the frame", ext["size"][0] * ext["size"][1] > sp, True)
check("crop shrinks it", crp["size"][0] * crp["size"][1] < sp, True)
# the point of balanced: an orientation flip at the SOURCE pixel count
check("balanced lands within 5% of the source area",
      abs(bal["size"][0] * bal["size"][1] - sp) / float(sp) < 0.05, True)
check("balanced is 768x1344 for this input", bal["size"], (768, 1344))

print("\n15. every side lands on the canvas multiple")
for mode in ("extend", "crop", "balanced"):
    for anchor in ("center", "top", "bottom", "left", "right"):
        r = rf(1344, 768, VERT, mode, anchor)
        vals = list(r["pad"]) + list(r["crop"])
        check("%s/%s all 32-aligned" % (mode, anchor),
              all(v % 32 == 0 for v in vals), True)
        check("%s/%s sizes 32-aligned" % (mode, anchor),
              all(v % 32 == 0 for v in r["size"]), True)

print("\n16. anchors put the growth where asked")
top_a = rf(1344, 768, VERT, "extend", "top")
bot_a = rf(1344, 768, VERT, "extend", "bottom")
check("anchor=top grows downward only", (top_a["pad"][2], top_a["pad"][3] > 0), (0, True))
check("anchor=bottom grows upward only", (bot_a["pad"][3], bot_a["pad"][2] > 0), (0, True))

print("\n17. it says what the choice costs")
check("extend warns past the H3 canvas", "past H3's" in ext["report"], True)
check("crop says how much is discarded", "discards" in crp["report"], True)
check("balanced needs neither warning",
      ("past H3's" in bal["report"], "discards" in bal["report"]), (False, False))
noop = rf(1344, 768, REFRAME_LABELS[3], "balanced")         # already ~16:9
check("a no-op says nothing to do", "nothing to do" in noop["report"], True)
check("and drops the ratio quibble", "landed on" in noop["report"], False)

print("\n18. the four SIGNED values feed the outpaint node directly")
v = torch.ones([1, 24, 12, 48, 84])
a = torch.zeros([1, 32, 2, frames_to_audio_t(latents_to_frames(12))])
src = _pack({}, v, a)
# the four SIGNED values go straight in -- no separate crop wiring
out, w2, h2, rep2 = OUT.execute(src, *bal["moves"], 64).result
check("outpaint lands on the reframe target", (w2, h2), bal["size"])
check("and says it cropped first", "before padding" in rep2, True)
check("balanced really does emit both signs",
      (any(v > 0 for v in bal["moves"]), any(v < 0 for v in bal["moves"])), (True, True))

print("\n19. snapping truncates TOWARD ZERO, so a crop is never bigger than asked")
# int() // CANVAS_MULTIPLE floors, which sends -33 to -64 -- twice the crop requested,
# silently. Snapping the magnitude and reapplying the sign is what keeps it honest.
_, _, _, r = OUT.execute(src, -33, 0, 0, 0, 0).result
check("-33 snaps to -32, not -64", "left -33 -> -32" in r, True)
_, _, _, r = OUT.execute(src, 33, 0, 0, 0, 0).result
check("+33 snaps to +32", "left 33 -> 32" in r, True)

_, w3, h3, r3 = OUT.execute(src, -64, -64, 0, 0, 0).result
check("a pure crop needs no padding", (w3, h3), (84 * 16 - 128, 48 * 16))
check("and reports it", "before padding" in r3, True)

try:
    OUT.execute(src, 0, 0, 0, 0, 64)
    check("all-zero is refused", False, True)
except ValueError as e:
    check("all-zero is refused", "every side is 0" in str(e), True)

print("\n20. PackAV normalizes a carried mask onto the latent shapes")
# Core accepts a mask of ANY size and interpolates it at sampling; the looping
# sampler slices masks by time, so PackAV normalizes at pack time with the same
# interpolation. The 32x32 zero pin (a legal core mask) is the crash case: sliced
# [a0:a1] past extent 32 it produced zero elements and died in reshape_mask.
d11, t11, at11 = mk(57)
v11, a11, _, _, _ = MMH3SplitAV.execute(d11).result
a11 = dict(a11)
a11["noise_mask"] = torch.zeros([1, 1, 32, 32])          # hand pin, image-shaped
p11 = MMH3PackAV.execute(v11, a11).result[0]
vm11, am11 = p11["noise_mask"].unbind()
check("audio mask reshaped to audio time", list(am11.shape), [1, 1, 2, at11])
check("zero pin survives interpolation", float(am11.max()), 0.0)
check("video half filled with ones", float(vm11.min()), 1.0)
check("video mask time-shaped", list(vm11.shape), [1, 1, t11, 4, 4])

# identity: an already time-shaped mask passes through value-equal
a12 = dict(a11)
good = torch.rand([1, 1, 2, at11])
a12["noise_mask"] = good
am12 = MMH3PackAV.execute(v11, a12).result[0]["noise_mask"].unbind()[1]
check("correct mask keeps its shape", list(am12.shape), [1, 1, 2, at11])
check("...and its values", bool(torch.allclose(am12, good, atol=1e-6)), True)

# a stale longer mask is resampled onto the trimmed audio, as core would
a13 = dict(a11)
a13["samples"] = torch.zeros([1, 32, 2, 5105])
a13["noise_mask"] = torch.zeros([1, 1, 2, 5105])
p13 = MMH3PackAV.execute(v11, a13).result[0]
am13 = p13["noise_mask"].unbind()[1]
check("stale-length mask lands on the trimmed audio",
      list(am13.shape), [1, 1, 2, at11])
check("...still all zeros", float(am13.max()), 0.0)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
