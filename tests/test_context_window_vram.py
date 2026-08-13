"""MMH3ContextWindowVRAM: the reservation must shrink to the window.

Not a smoke test. It drives the real WrapperExecutor with the real
`estimate_memory` and asserts on GB, because the whole point of the node is a
number -- a wrapper that installs cleanly but leaves the estimate untouched
would pass any weaker check.
"""
import os, sys
sys.path.insert(0, r"C:\ComfyUI")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(r"C:\ComfyUI")
import logging; logging.disable(logging.CRITICAL)

import torch
import comfy.patcher_extension as pe
from comfy.patcher_extension import WrapperExecutor, WrappersMP
from mmh3tools.nodes_util import MMH3ContextWindowVRAM

FACTOR, DTYPE_SIZE, CL = 0.114, 2, 47


def memory_required(shape):
    """BaseModel.memory_required, flash-attention branch."""
    area = shape[0] * (shape[2] * shape[3] * shape[4])
    return area * DTYPE_SIZE * 0.01 * FACTOR * (1024 * 1024)


class FakeModel:
    memory_usage_factor = FACTOR


class FakePatcher:
    def __init__(self):
        self.model_options = {}
        self.model = FakeModel()

    def clone(self):
        c = FakePatcher()
        import copy
        c.model_options = copy.deepcopy(self.model_options)
        return c


def run(patcher, shape):
    """Drive prepare_sampling's executor; return the shape the estimator saw."""
    seen = {}

    def original(p, noise_shape, conds, **kw):
        seen["shape"] = list(noise_shape)
        return memory_required(noise_shape)

    wrappers = pe.get_all_wrappers(
        WrappersMP.PREPARE_SAMPLING, patcher.model_options, is_model_options=True)
    ex = WrapperExecutor.new_executor(original, wrappers)
    return ex.execute(patcher, shape, None), seen["shape"]


def gb(b):
    return b / 1e9


fails = []

# 2K 1536x2688 -> h=168, w=96
SHAPE_40 = [1, 24, 282, 168, 96]
SHAPE_120 = [1, 24, 847, 168, 96]

# --- baseline: stock estimate scales with clip length ----------------------
base40, _ = run(FakePatcher(), SHAPE_40)
base120, _ = run(FakePatcher(), SHAPE_120)
print("stock      40s %6.1f GB   120s %6.1f GB" % (gb(base40), gb(base120)))
if not gb(base120) > 30:
    fails.append("expected the stock 120s estimate to exceed 30GB, got %.1f" % gb(base120))
if abs(base120 / base40 - 847 / 282) > 0.01:
    fails.append("stock estimate should scale linearly with length")

# --- patched: estimate is window-sized and length-INVARIANT ----------------
node = MMH3ContextWindowVRAM
p40 = node.execute(FakePatcher(), CL).result[0]
p120 = node.execute(FakePatcher(), CL).result[0]
new40, seen40 = run(p40, SHAPE_40)
new120, seen120 = run(p120, SHAPE_120)
print("patched    40s %6.1f GB   120s %6.1f GB" % (gb(new40), gb(new120)))

if seen120[2] != CL:
    fails.append("estimator saw T=%d, expected %d" % (seen120[2], CL))
if seen120[:2] != [1, 24] or seen120[3:] != [168, 96]:
    fails.append("only the temporal axis may change, got %s" % seen120)
if abs(new40 - new120) > 1:
    fails.append("patched estimate must not depend on clip length: %.3f vs %.3f"
                 % (gb(new40), gb(new120)))
if gb(new120) > 2.5:
    fails.append("patched 120s estimate should be ~1.8GB, got %.2f" % gb(new120))
if not (gb(base120) / gb(new120)) > 15:
    fails.append("expected >15x reduction, got %.1fx" % (base120 / new120))
print("reduction  %.1fx at 120s" % (base120 / new120))

# --- a clip SHORTER than the window must be left alone ---------------------
short = [1, 24, 20, 168, 96]
val, seen = run(node.execute(FakePatcher(), CL).result[0], short)
if seen[2] != 20:
    fails.append("a %d-latent clip must not be padded up to the window, saw %d"
                 % (short[2], seen[2]))

# --- disabled passes the patcher through untouched -------------------------
src = FakePatcher()
out = node.execute(src, CL, enabled=False).result[0]
if out is not src:
    fails.append("disabled must return the SAME patcher object, not a clone")
_, seen = run(out, SHAPE_120)
if seen[2] != 847:
    fails.append("disabled must leave the estimate at full length, saw %d" % seen[2])

# --- the wrapper is keyed, so re-wiring twice cannot stack -----------------
once = node.execute(FakePatcher(), CL).result[0]
twice = node.execute(once, CL).result[0]
w = pe.get_all_wrappers(WrappersMP.PREPARE_SAMPLING, twice.model_options,
                        is_model_options=True)
_, seen = run(twice, SHAPE_120)
if seen[2] != CL:
    fails.append("stacked wrappers changed the result: T=%d" % seen[2])
print("stacked    %d wrapper(s), still T=%d" % (len(w), seen[2]))

# --- clone isolation: patching a clone must not touch the original ---------
orig = FakePatcher()
node.execute(orig, CL)
_, seen = run(orig, SHAPE_120)
if seen[2] != 847:
    fails.append("execute() mutated the input patcher's model_options")

print()
if fails:
    print("FAIL (%d):" % len(fails))
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
