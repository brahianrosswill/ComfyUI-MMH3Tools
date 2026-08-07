import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import comfy.conds
import comfy.model_base as mb
from mmh3tools import patch_conds as PC

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


print("\n1. the wrap installs and self-tests against the live classes")
check("applied", PC.apply(), True)
check("status", PC.status(), "applied")
check("idempotent", PC.apply(), True)

print("\n2. the rebuild: keyframes FIRST, matching the layout's row order")
# PackedLayout emits cond segments (keyframes) before ref_img segments, and the model
# zips this list against those rows positionally -- reversing it is worse than dropping
p = {"cond_video_latents": ["WRONG"]}
n = PC._rebuild(p, [{"latent": "kf0"}, {"latent": "kf1"}],
                [{"latent": "ref0"}, {"kind": "audio"}])
check("order", p["cond_video_latents"], ["kf0", "kf1", "ref0"])
check("count", n, 3)
check("audio-only refs contribute no video row",
      "audio" in p["cond_video_latents"], False)

print("\n3. absolute, so it cannot double-apply over the file edit")
before = list(p["cond_video_latents"])
PC._rebuild(p, [{"latent": "kf0"}, {"latent": "kf1"}],
            [{"latent": "ref0"}, {"kind": "audio"}])
check("second pass is a no-op", p["cond_video_latents"], before)

print("\n4. what stock does, and what the wrap does instead")
# stock: assigns from keyframes, then ASSIGNS AGAIN from refs -- refs win
kf, refs = [{"latent": "kf0"}], [{"latent": "ref0"}]
stock = {}
stock["cond_video_latents"] = [k["latent"] for k in kf]
stock["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]
check("stock silently drops the keyframe", stock["cond_video_latents"], ["ref0"])
PC._rebuild(stock, kf, refs)
check("the wrap restores it", stock["cond_video_latents"], ["kf0", "ref0"])

print("\n5. inert unless BOTH are present")
patched = mb.MiniMaxH3.extra_conds
seen = {}
def fake_orig(self, **kwargs):
    payload = {"cond_video_latents": ["untouched"]}
    seen["called"] = True
    return {"minimax_payload": comfy.conds.CONDConstant(payload)}
_saved = PC._orig_extra_conds
PC._orig_extra_conds = fake_orig
try:
    for label, kw in [("neither", {}),
                      ("keyframes only", {"minimax_keyframes": [{"latent": "kf0"}]}),
                      ("refs only", {"minimax_refs": [{"latent": "ref0"}]})]:
        out = patched(None, **kw)
        check("%s -> payload untouched" % label,
              out["minimax_payload"].cond["cond_video_latents"], ["untouched"])
    out = patched(None, minimax_keyframes=[{"latent": "kf0"}],
                  minimax_refs=[{"latent": "ref0"}])
    check("both -> rebuilt", out["minimax_payload"].cond["cond_video_latents"],
          ["kf0", "ref0"])
finally:
    PC._orig_extra_conds = _saved

print("\n6. per-row masking (core patches 3-4) is DETECTED, not assumed")
avail = PC.per_row_masking_available()
print("   per_row_masking_available() =", avail)
check("returns a bool", isinstance(avail, bool), True)
import comfy.ldm.minimax.model as mmm
check("agrees with the live module",
      avail, hasattr(mmm, "mask_row_targets") and hasattr(mmm, "_mod_row"))

# SeedOverlap must refuse when they are absent rather than quietly doing nothing
from mmh3tools.nodes_loop import MMH3SeedOverlap
import mmh3tools.patch_conds as _pc
_real = _pc.per_row_masking_available
_pc.per_row_masking_available = lambda: False
try:
    MMH3SeedOverlap.execute(None, None, 5, 1.0, 1.0, 0)
    check("SeedOverlap refuses without the patch", False, True)
except RuntimeError as e:
    check("SeedOverlap refuses without the patch", "per-row masking patch" in str(e), True)
    check("...and names the upstream PR", "#15375" in str(e), True)
finally:
    _pc.per_row_masking_available = _real

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
