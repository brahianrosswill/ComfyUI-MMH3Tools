"""MMH3Regenerate2KDims: stage 1 must be what H3-Base really emits, and the 2K
stage must share its aspect EXACTLY -- a fractional difference is a squeeze in
every frame, and it is invisible until you compare the output to the source.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_util import MMH3Regenerate2KDims as R, LADDER_RATIO_LABELS
import comfy_extras.nodes_minimax_h3 as core

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


print("\n1. stage 1 IS core's adapt_canvas, not an invented size")
# If these diverge, stage 2 upscales something the model never rendered.
for lab in LADDER_RATIO_LABELS:
    for o in ("Landscape", "Portrait"):
        w1, h1, _w2, _h2, _s, _l = R.execute(lab, o, 2048).result
        check("%s %s -> %dx%d" % (lab.split(" -")[0], o[:4], w1, h1),
              core.adapt_canvas(w1, h1), (w1, h1))

print("\n2. both stages share ONE aspect, exactly")
for lab in LADDER_RATIO_LABELS:
    for o in ("Landscape", "Portrait"):
        w1, h1, w2, h2, _s, _l = R.execute(lab, o, 2048).result
        check("%s %s  %.4f == %.4f" % (lab.split(" -")[0], o[:4], w1 / h1, w2 / h2),
              abs(w1 / h1 - w2 / h2) < 1e-9, True)

print("\n3. the 2K stage is on the grid latents need")
# /32 in pixels, because latent dims are px/16 and must stay even for the 2x2 patch
for lab in LADDER_RATIO_LABELS:
    for o in ("Landscape", "Portrait"):
        _w1, _h1, w2, h2, _s, _l = R.execute(lab, o, 2048).result
        check("%s %s %dx%d" % (lab.split(" -")[0], o[:4], w2, h2),
              (w2 % 32, h2 % 32, (w2 // 16) % 2, (h2 // 16) % 2), (0, 0, 0, 0))

print("\n4. it says when it could not honour the requested long edge")
_w1, _h1, w2, _h2, _s, label = R.execute("16:9 - YouTube, HD, TV", "Landscape", 2048).result
check("16:9 lands on 2016, not 2048", w2, 2016)
check("and the label explains why", "keeps the ratio exact" in label, True)
_w1, _h1, w2b, _h2, _s, label2 = R.execute("4:3 - classic TV, monitor", "Landscape", 2048).result
check("4:3 hits 2048 exactly", w2b, 2048)
check("so no note is emitted", "not the 2048 asked for" in label2, False)

print("\n5. degenerate requests are reported, not silently accepted")
_a, _b, _c, _d, sc, lab_down = R.execute("16:9 - YouTube, HD, TV", "Landscape", 768).result
check("a downscale is called out", "downscale" in lab_down, True)
check("...and scale reflects it", sc < 1.0, True)
_a, _b, _c, _d, sc_big, lab_big = R.execute("1:1 - square", "Landscape", 8192).result
check("an extreme jump is called out", "outside anything measured" in lab_big, True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
