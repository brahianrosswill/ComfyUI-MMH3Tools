import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
import comfy.context_windows as C
from comfy.context_windows import (IndexListContextWindow, get_matching_context_schedule,
                                   get_matching_fuse_method)
from mmh3tools.common import VIDEO_T_DIM
from mmh3tools.nodes_multiprompt import MMH3CondSetSpread
import mmh3tools.nodes_windows as NW
from mmh3tools.nodes_windows import MMH3ContextHandler, MMH3ContextWindows

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


def cond(tag):
    """One node-level CONDITIONING: a list holding a single [tensor, dict] entry."""
    return [[torch.zeros([1, 4, 8]), {"mmh3_tag": tag}]]


def internal(tags):
    """The sampler's INTERNAL cond form, which is what get_resized_cond sees.

    Node-level conditioning is [[tensor, dict], ...]; convert_cond turns each pair
    into a flat dict before sampling. split_conds_to_windows operates on that later
    form, so `len(cond_in) > 1` is satisfied by a node-level conditioning carrying
    more than one entry -- which is exactly what the spread node produces.
    """
    return [{"cross_attn": torch.zeros([1, 4, 8]), "mmh3_tag": t} for t in tags]


print("\n1. spread flattens a cond_set into ONE conditioning of N entries")
cs = {"conds": [cond("a"), cond("b"), cond("c")],
      "prompts": ["shot one\nmore", "shot two", "shot three"]}
flat, n, report = MMH3CondSetSpread.execute(cs).result
check("entry count", len(flat), 3)
check("regions output", n, 3)
check("order preserved", [e[1]["mmh3_tag"] for e in flat], ["a", "b", "c"])
check("core's guard is satisfied", len(flat) > 1, True)
print("   " + report.splitlines()[0])

print("\n2. a single prompt is a no-op, and says so")
one, n1, rep1 = MMH3CondSetSpread.execute({"conds": [cond("solo")], "prompts": ["x"]}).result
check("one entry", n1, 1)
check("warns that split does nothing", "does nothing" in rep1, True)

print("\n3. windows map onto regions by their MIDPOINT, start to end")
# 57 latents, window 22, overlap 7 -> the schedule the node would build
handler = MMH3ContextHandler(
    context_schedule=get_matching_context_schedule("standard_static"),
    fuse_method=get_matching_fuse_method("pyramid"),
    context_length=22, context_overlap=7, context_stride=1, closed_loop=False,
    dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False,
    split_conds_to_windows=True)
wins = handler.get_context_windows(None, torch.zeros([1, 24, 57, 4, 4]), {})
regions = [w.get_region_index(3) for w in wins]
print("   windows:", [(w.index_list[0], w.index_list[-1]) for w in wins])
print("   regions:", regions)
check("every window gets a valid region", all(0 <= r < 3 for r in regions), True)
check("regions never go backwards", regions, sorted(regions))
check("first window is region 0", regions[0], 0)
check("last window is the last region", regions[-1], 2)
check("all three prompts are reached", sorted(set(regions)), [0, 1, 2])

print("\n4. the split actually selects, and picks the right entry")
# the spread node's N node-level entries become N internal dicts
inner = internal(["a", "b", "c"])
check("spread count matches internal count", len(flat), len(inner))
x = torch.zeros([1, 24, 57, 4, 4])
for w in wins:
    r = w.get_region_index(3)
    picked = handler.get_resized_cond(inner, x, w)
    check("window %2d-%2d -> %s" % (w.index_list[0], w.index_list[-1], "abc"[r]),
          [e["mmh3_tag"] for e in picked], ["abc"[r]])

print("\n5. OFF is the default and leaves every window seeing everything")
off = MMH3ContextHandler(
    context_schedule=get_matching_context_schedule("standard_static"),
    fuse_method=get_matching_fuse_method("pyramid"),
    context_length=22, context_overlap=7, context_stride=1, closed_loop=False,
    dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False)
check("default is off", off.split_conds_to_windows, False)
kept = off.get_resized_cond(internal(["a", "b", "c"]), x, wins[0])
check("all entries survive when off", len(kept), 3)

print("\n6b. keyframes re-project into every window, so the report locates them")
# The layout is rebuilt per window from the WINDOW's latent_t, and a first-frame
# anchor sits at the target origin -- that window's frame 0, not the clip's. Entry 0
# is the exception: region 0 IS the first window, so a start frame there is correct.
def kf_cond(tag, kf=False):
    d = {"mmh3_tag": tag}
    if kf:
        d["minimax_keyframes"] = [{"resolved_frame_index": 0}]
    return [[torch.zeros([1, 4, 8]), d]]

def note(conds):
    return MMH3CondSetSpread.execute(
        {"conds": conds, "prompts": ["p"] * len(conds)}).result[2].splitlines()[-1]

check("every entry -> flags only the repeats",
      "entries 1, 2 carry" in note([kf_cond("a", 1), kf_cond("b", 1), kf_cond("c", 1)]), True)
check("entry 0 only -> confirmed correct, not warned",
      "correct" in note([kf_cond("a", 1), kf_cond("b"), kf_cond("c")]), True)
check("a late entry is flagged",
      "entry 2 carries" in note([kf_cond("a"), kf_cond("b"), kf_cond("c", 1)]), True)
check("no keyframes -> no note",
      "keyframe" in note([kf_cond("a"), kf_cond("b")]), False)

print("\n6. the node exposes it, and the input prefix is FROZEN")
# Positional widget serialization: existing inputs must keep their exact positions
# forever; new features may only append after them (accumulator_device, 0.63.0).
ids = [i.id for i in MMH3ContextWindows.define_schema().inputs]
check("positions 0-7 are the 0.61 layout, byte for byte",
      ids[:8], ["model", "context_length", "context_overlap", "fuse_method",
                "context_schedule", "context_stride", "freenoise",
                "split_conds_to_windows"])

class FakeModel:
    def __init__(self): self.model_options = {}
    def clone(self):
        m = FakeModel(); m.model_options = dict(self.model_options); return m
_o1, _o2 = C.create_prepare_sampling_wrapper, NW.create_prepare_sampling_wrapper
C.create_prepare_sampling_wrapper = lambda m: None
NW.create_prepare_sampling_wrapper = lambda m: None
m_on, lab = MMH3ContextWindows.execute(FakeModel(), 22, 7, "pyramid",
                                       "standard_static", 1, False, True).result
check("forwarded to the handler",
      m_on.model_options["context_handler"].split_conds_to_windows, True)
check("label reports it", "split conds ON" in lab, True)
m_off, lab_off = MMH3ContextWindows.execute(FakeModel(), 22, 7, "pyramid",
                                            "standard_static", 1).result
check("default off via node", m_off.model_options["context_handler"].split_conds_to_windows, False)
check("label says off", "split conds off" in lab_off, True)
C.create_prepare_sampling_wrapper, NW.create_prepare_sampling_wrapper = _o1, _o2

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
