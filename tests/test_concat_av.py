import os, sys, logging
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
logging.basicConfig(level=logging.INFO, format="    %(message)s")

import torch
from comfy.nested_tensor import NestedTensor
from mmh3tools.nodes_loop import MMH3ConcatAV
from mmh3tools.common import frames_to_audio_t, latents_to_frames

H = W = 4

def mk(t_lat, masked):
    frames = latents_to_frames(t_lat)
    at = frames_to_audio_t(frames)
    v = torch.zeros([1, 24, t_lat, H, W])
    a = torch.zeros([1, 32, 2, at])
    d = {"samples": NestedTensor([v, a])}
    if masked:
        vm = torch.ones([1, 1, t_lat, H, W]); vm[:, :, :2] = 0.0
        am = torch.ones([1, 1, 2, at]);       am[:, :, :, :8] = 0.0
        d["noise_mask"] = NestedTensor([vm, am])
    return d, t_lat, at

def shapes(out):
    v, a = out["samples"].unbind()
    m = out.get("noise_mask")
    if m is None:
        return int(v.shape[2]), int(a.shape[3]), None, None
    vm, am = m.unbind()
    return int(v.shape[2]), int(a.shape[3]), int(vm.shape[2]), int(am.shape[3])

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)

print("\n1. carry_masks=False, both inputs masked -> mask dropped (current behaviour)")
a, ta, aa = mk(7, True); b, tb, ab = mk(12, True)
out = MMH3ConcatAV.execute(a, b, 0, False).result[0]
check("latent len / mask absent", shapes(out), (ta + tb, aa + ab, None, None))

print("\n2. default arg omitted -> identical to carry_masks=False")
out = MMH3ConcatAV.execute(a, b, 0).result[0]
check("mask absent", shapes(out)[2], None)

print("\n3. carry_masks=True, both masked, no trim -> masks match latent lengths")
out = MMH3ConcatAV.execute(a, b, 0, True).result[0]
check("mask lengths track latents", shapes(out), (ta + tb, aa + ab, ta + tb, aa + ab))

print("\n4. carry_masks=True, only A masked -> B filled with ones (denoise)")
a2, ta2, aa2 = mk(7, True); b2, tb2, ab2 = mk(12, False)
out = MMH3ConcatAV.execute(a2, b2, 0, True).result[0]
vm, am = out["noise_mask"].unbind()
check("lengths", shapes(out), (ta2 + tb2, aa2 + ab2, ta2 + tb2, aa2 + ab2))
check("A head preserved (0)", float(vm[0, 0, 0, 0, 0]), 0.0)
check("B all denoise (1)", float(vm[:, :, ta2:].min()), 1.0)

print("\n5. carry_masks=True, no masks anywhere -> still no mask key")
a3, _, _ = mk(7, False); b3, _, _ = mk(12, False)
out = MMH3ConcatAV.execute(a3, b3, 0, True).result[0]
check("no mask invented", "noise_mask" in out, False)

print("\n6. carry_masks=True WITH trim -> mask takes the same computed cut")
a4, ta4, aa4 = mk(7, True); b4, tb4, ab4 = mk(12, True)
out = MMH3ConcatAV.execute(a4, b4, 5, True).result[0]
vt, at_, vmt, amt = shapes(out)
check("mask video len == latent video len", vmt, vt)
check("mask audio len == latent audio len", amt, at_)

print("\n7. trim now drops the right number of frames (was snapping 5 -> 2)")
for k in (5, 10, 15):
    a5, _, _ = mk(7, False)
    b5, tb5, ab5 = mk(22, False)          # 5*4+2
    out = MMH3ConcatAV.execute(a5, b5, k, False).result[0]
    vt, at_, _, _ = shapes(out)
    kept_v = vt - 7
    check("trim=%d drops %d latents" % (k, k), tb5 - kept_v, k)
    check("trim=%d drops %d frames" % (k, 17 * (k // 5)),
          latents_to_frames(tb5) - latents_to_frames(kept_v), 17 * (k // 5))

print("\n8. the value is honoured as given; only clamped so B keeps 2 latents")
a6, _, _ = mk(7, False); b6, tb6, _ = mk(12, False)
for want in (2, 5, 7):
    out = MMH3ConcatAV.execute(a6, b6, want, False).result[0]
    check("trim=%d honoured" % want, tb6 - (shapes(out)[0] - 7), want)
out = MMH3ConcatAV.execute(a6, b6, 999, False).result[0]
check("trim=999 clamped, B keeps 2", shapes(out)[0] - 7, 2)

print("\n8b. the two trim families do what they claim (A=12, B=12)")
for k in (5, 10):
    out = MMH3ConcatAV.execute(mk(12, False)[0], mk(12, False)[0], k, False).result[0]
    check("k=%d (5m) removes %d frames" % (k, 17 * (k // 5)),
          latents_to_frames(12) - latents_to_frames(12 - k), 17 * (k // 5))
for k in (2, 7):
    out = MMH3ConcatAV.execute(mk(12, False)[0], mk(12, False)[0], k, False).result[0]
    check("k=%d (5m+2) lands on grid" % k, (shapes(out)[0] - 2) % 5, 0)

print("\n9. a 5m trim cannot land on grid -- why trim_b_latents is honoured as given")
# This used to run through MMH3SeedOverlap, which now lives on the keyframe-anchors
# branch because it needs per-row masking in core. The arithmetic is the durable part:
# removing a 5m overlap gives (5a+2) + (5b+2-5m) = 5(a+b-m)+4, never 5j+2.
prev, _, _ = mk(12, False)
tgt, _, _ = mk(12, False)
pv, _ = prev["samples"].unbind()
tv, _ = tgt["samples"].unbind()
for m in (1, 2):
    jv, ja, _, _ = shapes(MMH3ConcatAV.execute(prev, tgt, 5 * m, False).result[0])
    check("m=%d video: a + b - trim" % m, jv, int(pv.shape[2]) + int(tv.shape[2]) - 5 * m)
    check("m=%d total is OFF grid, inherently" % m, (jv - 2) % 5, 2)
    check("m=%d ...and 2 more would fix it" % m, (jv - 4) % 5, 0)

print("\n9b. SeedOverlap: round-trip if per-row masking is present, a clear refusal if not")
# SeedOverlap needs #15375, an UPSTREAM PR -- not a monkeypatch -- which is why it lives
# on main. Its behaviour is conditional on the core it finds, and both outcomes are
# asserted: a test that assumed the PR would break the day you updated ComfyUI, which is
# exactly when you want it still running.
from mmh3tools.nodes_loop import MMH3SeedOverlap

def _per_row_masking_available():
    try:
        import comfy.ldm.minimax.model as mm
    except Exception:
        return False
    return hasattr(mm, "mask_row_targets") and hasattr(mm, "_mod_row")

prev2, _, _ = mk(12, False)
tgt2, _, _ = mk(12, False)

if _per_row_masking_available():
    print("   per-row masking IS present -- testing the round-trip")
    seeded, ov_frames, ov_latents = MMH3SeedOverlap.execute(
        tgt2, prev2, 5, 1.0, 1.0, 0).result
    sv, _ = seeded["samples"].unbind()
    pv2, pa2 = prev2["samples"].unbind()
    jv, ja, _, _ = shapes(MMH3ConcatAV.execute(prev2, seeded, ov_latents, False).result[0])
    check("video: prev + seeded - overlap", jv,
          int(pv2.shape[2]) + int(sv.shape[2]) - ov_latents)
    check("audio: no overlap left over", ja, int(pa2.shape[3]) + frames_to_audio_t(
        latents_to_frames(int(sv.shape[2]) - ov_latents)))
    check("total off grid, inherent to a 5m trim", (jv - 2) % 5, 2)
else:
    print("   per-row masking is ABSENT -- testing that it refuses rather than no-ops")
    try:
        MMH3SeedOverlap.execute(tgt2, prev2, 5, 1.0, 1.0, 0)
        check("refuses without the PR", False, True)
    except RuntimeError as e:
        check("refuses without the PR", "per-row masking" in str(e), True)
        check("names the upstream PR", "#15375" in str(e), True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
