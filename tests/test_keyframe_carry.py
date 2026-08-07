import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
import comfy.ldm.minimax.model as mm
from mmh3tools import patch_layout as P
from mmh3tools.common import (LATENTS_PER_GROUP, latents_to_frames, snap_latents,
                              step_frame_offsets)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


print("\n1. the patch installs and self-tests against the live PackedLayout")
check("applied", P.apply(), True)
check("status", P.status().startswith("applied"), True)

print("\n2. step offsets, and the phase-0 invariant a tail slice relies on")
check("7 steps", step_frame_offsets(7), [0, 1, 5, 9, 13, 17, 18])
check("2 steps", step_frame_offsets(2), [0, 1])
check("12 steps", step_frame_offsets(12), [0, 1, 5, 9, 13, 17, 18, 22, 26, 30, 34, 35])
# a 5m+2 tail off a 5j+2 clip always starts at a multiple of 5
for j in range(1, 12):
    T = 5 * j + 2
    for m in range(0, j):
        check("T=%d tail %d starts phase 0" % (T, 5 * m + 2), (T - (5 * m + 2)) % 5, 0)
        break
# and the frames it covers equal what latents_to_frames says
for m in range(0, 5):
    n = 5 * m + 2
    off = step_frame_offsets(n)
    span = off[-1] + mm.FRAME_PER_TOKEN[(n - 1) % 5]
    check("%d steps cover %d frames" % (n, latents_to_frames(n)), span, latents_to_frames(n))

print("\n3. the node emits one keyframe per pinned step, at the right anchors")
from mmh3tools.nodes_refs import MMH3LatentToKeyframes as KF

LT, LH, LW = 57, 48, 84          # 192 frames at 768x1344
video = torch.zeros([1, 24, LT, LH, LW])
audio = torch.zeros([1, 32, 2, 320])
# unpack_av wants the packed AV form; build it the way the pack does
from mmh3tools.common import pack_av
latent = pack_av({}, video, audio)

cond = [[torch.zeros([1, 4, 5120]), {}]]
out = KF.execute(cond, latent, 7, 192)
new_cond, pinned_frames, pinned_latents = out.result
kfs = new_cond[0][1]["minimax_keyframes"]

check("one keyframe per step", len(kfs), 7)
check("pinned frames", pinned_frames, 22)
check("pinned latents", pinned_latents, 7)
check("anchors", [k[P.MMH3_KEY] for k in kfs], [0, 1, 5, 9, 13, 17, 18])
check("every resolved index is legal", set(k["resolved_frame_index"] for k in kfs), {0})
check("each latent is one step", [tuple(k["latent"].shape) for k in kfs],
      [(1, 24, 1, LH, LW)] * 7)
check("frame_count set", new_cond[0][1]["minimax_frame_count"], 192)

print("\n4. carry_latents snaps DOWN to the grid")
for given, want_lat in ((7, 7), (8, 7), (11, 7), (12, 12), (3, 2), (2, 2)):
    _, _, n = KF.execute(cond, latent, given, 192).result
    check("carry %d -> %d latents" % (given, want_lat), n, want_lat)

print("\n5. THE POINT: keyframes cost no distance, a ref carry costs 65")
TEXT_LEN, AUDIO_T = 320, 320
frame_count = latents_to_frames(LT)

def origin_and_anchors(**kw):
    L = mm.PackedLayout(TEXT_LEN, LT, LH, LW, AUDIO_T, frame_count=frame_count, **kw)
    o = next(float(L.position_ids[a, 0]) for a, _, k in L.segments if k == "video")
    anchors = [float(L.position_ids[a, 0]) for a, _, k in L.segments if k == "cond"]
    return o, anchors

bare, _ = origin_and_anchors()
kf_o, kf_anchors = origin_and_anchors(keyframes=kfs)
carry_ref = {"kind": "video_audio", "latent_t": 12, "latent_h": LH, "latent_w": LW,
             "ref_audio_t": 65}
ref_o, _ = origin_and_anchors(refs=[carry_ref])

check("bare target origin", bare, float(TEXT_LEN))
check("keyframe carry adds NO distance", kf_o, float(TEXT_LEN))
check("ref carry pushes the target +65", ref_o - bare, 65.0)
check("first anchor sits ON target frame 0", kf_anchors[0], kf_o)
check("anchors are strictly increasing", kf_anchors == sorted(set(kf_anchors)), True)
check("last anchor inside the clip", kf_anchors[-1] < kf_o + 192 * mm.FRAME_RESCALE, True)
print("   anchors:", [round(a, 2) for a in kf_anchors])

print("\n6. inert: a graph without our key is byte-identical to stock")
stock_kf = [{"resolved_frame_index": 0, "latent": torch.zeros([1, 24, 1, LH, LW])}]
a = object.__new__(mm.PackedLayout)
P._orig_init(a, TEXT_LEN, LT, LH, LW, AUDIO_T, keyframes=stock_kf, frame_count=frame_count)
b = mm.PackedLayout(TEXT_LEN, LT, LH, LW, AUDIO_T, keyframes=stock_kf, frame_count=frame_count)
check("stock keyframes untouched", bool(torch.equal(a.position_ids, b.position_ids)), True)

print("\n7. refusing rather than rendering a shifted join")
try:
    KF.execute(cond, latent, 512, 192)
    check("over-long carry raises", False, True)
except ValueError as e:
    check("over-long carry raises", "nothing to generate" in str(e), True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
