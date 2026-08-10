"""The guide-origin wrap: correct where it should be, inert everywhere else.

#15439 anchors a guide on text_len, but the target begins at the cursor the
reference blocks advanced. The failure is silent -- the guide simply points into
the reference region -- so an inert-when-it-should-act patch and an active-when-it
-should-not one look identical at a glance. Both directions are checked.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
import comfy.ldm.minimax.model as mm
from mmh3tools import patch_guide_origin as P

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


IMG = {"kind": "image", "latent_h": 4, "latent_w": 4,
       "latent": torch.zeros([1, 24, 1, 4, 4])}
AUD = {"kind": "audio", "ref_audio_t": 320,
       "audio_latent": torch.zeros([1, 32, 2, 320])}
VA = {"kind": "video_audio", "ref_audio_t": 37, "latent_t": 7,
      "latent_h": 4, "latent_w": 4, "latent": torch.zeros([1, 24, 7, 4, 4]),
      "audio_latent": torch.zeros([1, 32, 2, 37])}
KF = [{"resolved_frame_index": 0, "latent": torch.zeros([1, 24, 7, 4, 4]),
       "audio_latent": torch.zeros([1, 32, 2, 37])}]


def origins(refs, keyframes=KF, text_len=11):
    lay = mm.PackedLayout(text_len, 37, 4, 4, 8, keyframes=keyframes, refs=refs)
    seg = {k: a for a, _b, k in lay.segments}
    out = {k: float(lay.position_ids[a, 0]) for k, a in seg.items()}
    return out, seg


print("\n1. the patch is applied at import")
check("applied", P.is_applied(), True)
check("idempotent", P.apply()[0], True)
check("class is marked", getattr(mm.PackedLayout, "_mmh3_guide_origin_patched", False), True)

print("\n2. with references, a guide anchors ON the target origin")
for label, refs, advance in (("one image ref", [IMG], 1.0),
                             ("voice audio ref", [AUD], 320.0),
                             ("video_audio ref", [VA], 37.0),
                             ("image + audio", [IMG, AUD], 321.0)):
    o, _ = origins(refs)
    check("%s: guide == target video" % label, o["cond"], o["video"])
    check("%s: cond_audio too" % label, o["cond_audio"], o["video"])
    # and it is the advance that moved it, not a coincidence
    check("%s: moved by the ref advance" % label, o["cond"] - 11.0, advance)

print("\n3. INERT when it should be")
o, _ = origins(None)
check("no refs: guide stays at text_len", o["cond"], 11.0)
check("...and the target is there too", o["video"], 11.0)
# refs with NO guide must not shift anything either
lay = mm.PackedLayout(11, 37, 4, 4, 8, keyframes=None, refs=[IMG])
seg = {k: a for a, _b, k in lay.segments}
check("refs alone: no cond rows to move", "cond" in seg, False)
check("...and the target still starts after the ref",
      float(lay.position_ids[seg["video"], 0]), 12.0)

print("\n4. interior anchors are unchanged")
for p in (0, 60, 123):
    o, _ = origins(None, keyframes=[{"resolved_frame_index": p,
                                     "latent": torch.zeros([1, 24, 1, 4, 4])}])
    check("p=%d" % p, round(o["cond"], 6), round(11.0 + mm.FRAME_RESCALE * p, 6))

print("\n5. the ref advance mirrors PackedLayout's own cursor")
# if these drift apart the guide is silently misplaced again, which is the whole
# failure this exists to remove -- so compare against the TARGET the layout built
for refs in ([IMG], [AUD], [VA], [IMG, AUD], [VA, IMG, AUD]):
    o, _ = origins(refs)
    check("advance matches the target origin for %d ref(s)" % len(refs),
          round(11.0 + P._ref_cursor_advance(mm, refs), 6), round(o["video"], 6))

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
