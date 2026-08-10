"""MMH3LoopingSampler: the loop arithmetic and the guider hand-off, without a model.

The sampler call is stubbed to return its target unchanged, so what is under test
is everything AROUND the model: how many chunks run, which prompt each gets, that
the join trims the carried head exactly once, and that the source guider is never
mutated. That last one is the bug this node is most likely to have -- it is the
one LTXAVTools shipped, where every chunk silently reused chunk 0's conditioning.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools.common import latents_to_frames, frames_to_audio_t

H = W = 4

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def mk(t_lat):
    at = frames_to_audio_t(latents_to_frames(t_lat))
    return {"samples": NestedTensor([torch.zeros([1, 24, t_lat, H, W]),
                                     torch.zeros([1, 32, 2, at])])}


def cond(tag, **extra):
    """Conditioning in the shape the sampler really sees: [[tensor, dict], ...]."""
    d = {"tag": tag}
    d.update(extra)
    return [[torch.zeros([1, 8]), d]]


def tag_of(c):
    return c[0][1]["tag"]


class FakeGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE_POS", "negative": "BASE_NEG"}
        self.set_calls = []
    def set_conds(self, positive, negative):
        self.set_calls.append((positive, negative))
        self.original_conds["positive"] = positive
        self.original_conds["negative"] = negative


class FakeNoise:
    def __init__(self, seed=1000):
        self.seed = seed


SEEN = {"guiders": [], "seeds": [], "targets": []}

class FakeSampler:
    """Stands in for SamplerCustomAdvanced: records, returns the target."""
    def sample(self, noise, guider, sampler, sigmas, latent):
        SEEN["guiders"].append(guider)
        SEEN["seeds"].append(getattr(noise, "seed", None))
        SEEN["targets"].append(latent)
        return latent, latent


LS.SamplerCustomAdvanced = FakeSampler

T = 57                      # 192 frames, on the 5j+2 grid
OVERLAP = 7                 # 5m+2, so T-OVERLAP stays a multiple of 5


def run(chunks, prompts, overlap=OVERLAP, carry="mask", conds=None):
    SEEN["guiders"].clear(); SEEN["seeds"].clear(); SEEN["targets"].clear()
    g = FakeGuider()
    cs = conds if conds is not None else [cond(p) for p in prompts]
    out = LS.MMH3LoopingSampler.execute(
        FakeNoise(), g, "SAMPLER", "SIGMAS",
        {"conds": cs, "prompts": prompts, "fingerprint": "fp"},
        mk(T), chunks, overlap, 1.0, 1.0, 0, carry).result
    return out, g


print("\n1. one chunk per requested chunk, each with its own prompt")
(joined, n, rep), g = run(4, ["p0", "p1", "p2", "p3"])
check("chunks rendered", n, 4)
check("sampler called once per chunk", len(SEEN["guiders"]), 4)
check("each chunk got its own conditioning",
      [tag_of(gg.raw_conds[0]) for gg in SEEN["guiders"]], ["p0", "p1", "p2", "p3"])

print("\n2. the SOURCE guider is never mutated -- the shallow-copy bug")
check("base positive intact", g.original_conds["positive"], "BASE_POS")
check("base negative intact", g.original_conds["negative"], "BASE_NEG")
check("every chunk saw the SAME negative",
      {gg.raw_conds[1] for gg in SEEN["guiders"]}, {"BASE_NEG"})
check("and each copy is a distinct object", len({id(gg) for gg in SEEN["guiders"]}), 4)
check("...whose conds dict is not the source's",
      any(gg.original_conds is g.original_conds for gg in SEEN["guiders"]), False)

print("\n3. noise differs per chunk")
check("four distinct seeds", len(set(SEEN["seeds"])), 4)
check("chunk 0 keeps the wired seed", SEEN["seeds"][0], 1000)

print("\n4. the join is grid-safe, which costs k+2 rather than k")
# SeedOverlap PREPENDS a multiple of 5, so a chunk is T+K latents, not T. The trim
# must be K+2 to leave the master on the 5j+2 grid; trimming exactly K would leave
# it off-grid, and an off-grid latent cannot be decoded.
v, _a = joined["samples"].unbind()
K = 5                                   # overlap_latents=7 snaps DOWN to 5
want = T + 3 * ((T + K) - (K + 2))
check("master video latents", int(v.shape[2]), want)
check("...which is on the 5j+2 grid", (want - 2) % 5, 0)
check("audio matches the video length",
      int(joined["samples"].unbind()[1].shape[3]),
      frames_to_audio_t(latents_to_frames(want)))
check("the report owns the frames it took", "grid-safe trim" in rep, True)
# trimming exactly K is the tempting wrong answer -- it leaves the grid
check("trimming exactly K would leave the grid",
      (T + 3 * ((T + K) - K) - 2) % 5 == 0, False)

print("\n5. a single chunk is just a plain generation")
(j1, n1, _), _ = run(1, ["only"])
check("no join happened", int(j1["samples"].unbind()[0].shape[2]), T)
check("chunks rendered", n1, 1)

print("\n6. fewer prompts than chunks: the last one repeats, and it SAYS so")
(_, n2, rep2), _ = run(4, ["a", "b"])
check("still renders every chunk", n2, 4)
check("prompt order", [tag_of(gg.raw_conds[0]) for gg in SEEN["guiders"]],
      ["a", "b", "b", "b"])
check("report warns", "the last one repeats" in rep2, True)

print("\n7. no conditioning at all is an error, not an empty render")
try:
    LS.MMH3LoopingSampler.execute(FakeNoise(), FakeGuider(), "S", "SG",
                               {"conds": []}, mk(T), 2, OVERLAP, 1.0, 1.0, 0, "mask")
    check("empty cond_set raises", False, True)
except ValueError as e:
    check("empty cond_set raises", "no conditioning" in str(e), True)

print("\n8. the template latent is cloned, never mutated")
tmpl = mk(T)
before = tmpl["samples"].unbind()[0].clone()
SEEN["guiders"].clear(); SEEN["seeds"].clear(); SEEN["targets"].clear()
LS.MMH3LoopingSampler.execute(FakeNoise(), FakeGuider(), "S", "SG",
                           {"conds": [cond("a"), cond("b")]}, tmpl, 2, OVERLAP, 1.0, 1.0, 0, "mask")
check("template untouched",
      torch.equal(tmpl["samples"].unbind()[0], before), True)
check("and each chunk got a fresh tensor",
      SEEN["targets"][0]["samples"].unbind()[0] is tmpl["samples"].unbind()[0], False)

print("\n9. stale guide bookkeeping is stripped off incoming conditioning")
# a cond cached from a previous run, or one that went through a guide node, would
# anchor this chunk to somebody else's frames
dirty = [cond("p0", minimax_keyframes=[{"resolved_frame_index": 99}],
              minimax_frame_count=124),
         cond("p1")]
(_, _, _), _ = run(2, ["p0", "p1"], conds=dirty)
seen_keys = [set(gg.raw_conds[0][0][1].keys()) for gg in SEEN["guiders"]]
check("stale minimax_keyframes gone", "minimax_keyframes" in seen_keys[0], False)
check("stale minimax_frame_count gone", "minimax_frame_count" in seen_keys[0], False)
check("the prompt itself survives", tag_of(SEEN["guiders"][0].raw_conds[0]), "p0")
check("and the caller's dict was not mutated",
      "minimax_keyframes" in dirty[0][0][1], True)

print("\n10. carry='keyframe': the tail rides as a guide, and the trim is EXACT")
if not LS._guides_available():
    print("  SKIP  #15439 not applied to this core")
else:
    (jk, nk, repk), _ = run(4, ["p0", "p1", "p2", "p3"], carry="keyframe")
    vk, ak = jk["samples"].unbind()
    CARRY = 7                            # already 5m+2, so no snapping
    want_k = T + 3 * (T - CARRY)
    check("chunk keeps its natural length -- nothing prepended",
          int(SEEN["targets"][1]["samples"].unbind()[0].shape[2]), T)
    check("master video latents", int(vk.shape[2]), want_k)
    check("...on the 5j+2 grid", (want_k - 2) % 5, 0)
    check("audio matches", int(ak.shape[3]),
          frames_to_audio_t(latents_to_frames(want_k)))
    check("NO frames lost to grid-safety", "grid-safe trim" in repk, False)

    # the guide itself
    kf = SEEN["guiders"][1].raw_conds[0][0][1]["minimax_keyframes"][0]
    check("one guide, anchored at frame 0", kf["resolved_frame_index"], 0)
    check("carrying a MULTI-STEP clip, not a still", int(kf["latent"].shape[2]), CARRY)
    check("with its audio at the same anchor", "audio_latent" in kf, True)
    check("chunk 0 has no guide",
          "minimax_keyframes" in SEEN["guiders"][0].raw_conds[0][0][1], False)

    # a carry off the 5m+2 grid must snap DOWN, or slice_av_tail converts invalidly
    check("10 snaps down to 7", LS._snap_carry(10), 7)
    check("7 is already on grid", LS._snap_carry(7), 7)
    check("below the base clamps up", LS._snap_carry(1), 2)

print("\n10b. a Basic Guider has NO negative -- one arg, and no such key")
# Guider_Basic.set_conds takes ONE argument and original_conds has no "negative".
# Indexing for it raises KeyError; calling set_conds with two raises TypeError.
class BasicGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE_POS"}
        self.calls = []
    def set_conds(self, positive):          # deliberately one-arg, like core
        self.calls.append(positive)
        self.original_conds["positive"] = positive

SEEN["guiders"].clear(); SEEN["seeds"].clear(); SEEN["targets"].clear()
bg = BasicGuider()
out = LS.MMH3LoopingSampler.execute(
    FakeNoise(), bg, "S", "SG",
    {"conds": [cond("p0"), cond("p1")]}, mk(T), 2, OVERLAP, 1.0, 1.0, 0, "mask").result
check("runs against a one-arg guider", out[1], 2)
check("each chunk still got its own conditioning",
      [tag_of(gg.raw_conds[0]) for gg in SEEN["guiders"]], ["p0", "p1"])
check("negative reported as None", SEEN["guiders"][0].raw_conds[1], None)
check("source guider untouched", bg.original_conds["positive"], "BASE_POS")

print("\n11. a guide alongside a REFERENCE needs the target-origin correction")
check("this core anchors guides on the target origin", LS._guide_origin_correct(), True)
check("_has_refs sees a reference",
      LS._has_refs(cond("p", minimax_refs=[{"kind": "image"}])), True)
check("...and not an empty one", LS._has_refs(cond("p", minimax_refs=[])), False)
check("...nor a plain prompt", LS._has_refs(cond("p")), False)

# without the correction, a ref+guide chunk must REFUSE rather than anchor
# ref_advance units before the clip -- -1 for an image ref, -320 for voice audio
_real = LS._guide_origin_correct
LS._guide_origin_correct = lambda: False
try:
    run(2, ["p0", "p1"], carry="keyframe",
        conds=[cond("p0", minimax_refs=[{"kind": "image"}]),
               cond("p1", minimax_refs=[{"kind": "image"}])])
    check("refuses ref+guide without the correction", False, True)
except RuntimeError as e:
    check("refuses ref+guide without the correction", "target origin" in str(e), True)
    check("...and names the chunk", "chunk 1" in str(e), True)
    check("...and offers a way out", "carry='mask'" in str(e), True)

# guides WITHOUT a reference are fine even then -- the cursor never leaves text_len
try:
    run(2, ["p0", "p1"], carry="keyframe")
    check("guides alone still run", True, True)
except RuntimeError as e:
    check("guides alone still run", "raised: %s" % e, True)
finally:
    LS._guide_origin_correct = _real

print("\n12. keyframe_indices: GLOBAL frames, mapped onto the chunk schedule")
from mmh3tools.common import frame_at_latent, latents_to_frames
K = LS._snap_carry(OVERLAP)                       # 7
lengths = [T] * 4
origins, cum = LS._chunk_origins(4, lengths, K)
cf = [latents_to_frames(L) for L in lengths]
caf = [0] + [frame_at_latent(K)] * 3
TOT = latents_to_frames(cum)
check("chunk origins", origins, [0, 50, 100, 150])
check("every origin on the 5-grid -- so each chunk stays phase 0",
      all(o % 5 == 0 for o in origins), True)
check("master frames", TOT, latents_to_frames(cum))

check("frame 0 -> chunk 0", LS._keyframe_plan([0], origins, cf, caf), [(0, 0)])
# chunk 1 spans global 170..361; chunk 2 spans 340..531, so 351 is in BOTH --
# but in chunk 2 it sits inside the carried head, which the join trims away
check("a frame two chunks cover goes to the one that RENDERS it",
      LS._keyframe_plan([351], origins, cf, caf), [(1, 181)])
check("...and 181 is past chunk 1's head", 181 >= caf[1], True)

print("\n12b. index parsing")
check("negatives count from the end", LS._parse_indices("-1", 100), [99])
check("whitespace and blanks tolerated", LS._parse_indices(" 0 , , 5 ", 100), [0, 5])
check("empty is no keyframes", LS._parse_indices("", 100), [])
for bad, why in (("100", "past the end"), ("-101", "before the start"),
                 ("abc", "not a number")):
    try:
        LS._parse_indices(bad, 100)
        check("%r refused (%s)" % (bad, why), False, True)
    except ValueError:
        check("%r refused (%s)" % (bad, why), True, True)

print("\n12c. the node wires them, alongside the carry, in ONE set")
class FakeVae:
    def __init__(self): self.calls = 0
    def encode(self, img):
        self.calls += 1
        return torch.zeros([1, 24, 1, H, W])

def kf_run(indices, n_img, chunks=4, carry="keyframe"):
    SEEN["guiders"].clear(); SEEN["seeds"].clear(); SEEN["targets"].clear()
    vae = FakeVae()
    out = LS.MMH3LoopingSampler.execute(
        FakeNoise(), FakeGuider(), "S", "SG",
        {"conds": [cond("p%d" % i) for i in range(chunks)]},
        mk(T), chunks, OVERLAP, 1.0, 1.0, 0, carry,
        torch.zeros([n_img, 64, 64, 3]), indices, vae).result
    return out, vae

(_, _, rep), vae = kf_run("0, 351", 2)
check("encoded once per image, not per chunk", vae.calls, 2)
kfs = [gg.raw_conds[0][0][1].get("minimax_keyframes") for gg in SEEN["guiders"]]
check("chunk 0: the user keyframe only, no carry", len(kfs[0] or []), 1)
check("chunk 1: carry guide AND the user keyframe", len(kfs[1] or []), 2)
check("the carry comes first, at frame 0", kfs[1][0]["resolved_frame_index"], 0)
check("the user keyframe keeps its LOCAL index", kfs[1][1]["resolved_frame_index"], 181)
check("chunk 2 has only its carry", len(kfs[2] or []), 1)
check("the report says where each one landed", "keyframe frame 351 -> chunk 1" in rep, True)

print("\n12d. mismatches are refused rather than zipped short")
try:
    kf_run("0, 60, 120", 2)
    check("image/index count mismatch raises", False, True)
except ValueError as e:
    check("image/index count mismatch raises", "zipped" in str(e), True)
try:
    LS.MMH3LoopingSampler.execute(
        FakeNoise(), FakeGuider(), "S", "SG", {"conds": [cond("p")]}, mk(T),
        1, OVERLAP, 1.0, 1.0, 0, "keyframe", torch.zeros([1, 64, 64, 3]), "0", None)
    check("keyframes without a vae raises", False, True)
except ValueError as e:
    check("keyframes without a vae raises", "vae" in str(e), True)
try:
    LS.MMH3LoopingSampler.execute(
        FakeNoise(), FakeGuider(), "S", "SG", {"conds": [cond("p")]}, mk(T),
        1, OVERLAP, 1.0, 1.0, 0, "keyframe", None, "0", None)
    check("indices without images raises", False, True)
except ValueError as e:
    check("indices without images raises", "no keyframes were supplied" in str(e), True)

print("\n12e. guides work with the MASK carry too -- they are independent")
(_, _, rep_m), _ = kf_run("0, 251", 2, carry="mask")
kfs_m = [gg.raw_conds[0][0][1].get("minimax_keyframes") for gg in SEEN["guiders"]]
check("chunk 0 got its keyframe", len(kfs_m[0] or []), 1)
check("and no carry guide was added under mask carry",
      all(len(k or []) <= 1 for k in kfs_m), True)

print("\n13. MMH3KeyframePlanner: end-anchored travel, one keyframe per chunk")
PLAN = LS.MMH3KeyframePlanner

def replan(carry, ov, n=4, T=57, start=True, end=True, scenes=""):
    idx, cnt, tot, rep = PLAN.execute(n, T, ov, carry, start, end, scenes).result
    if carry == "keyframe":
        k = LS._snap_carry(ov); lengths = [T] * n; trim = k
    else:
        k = max(5, (ov // 5) * 5); lengths = [T] + [T + k] * (n - 1); trim = k + 2
    org, cum = LS._chunk_origins(n, lengths, trim)
    cf = [latents_to_frames(L) for L in lengths]
    caf = [0] + [frame_at_latent(trim)] * (n - 1)
    plan = LS._keyframe_plan(LS._parse_indices(idx, latents_to_frames(cum)), org, cf, caf)
    return idx, cnt, int(tot), rep, plan

for carry, ov in (("keyframe", 7), ("mask", 5)):
    idx, cnt, tot, rep, plan = replan(carry, ov)
    owners = [c for c, _l in plan]
    # frame 0 opens, then every chunk travels to a keyframe at its own end
    check("%s: chunk 0 opens and travels" % carry, owners[:2], [0, 0])
    check("%s: then one per chunk" % carry, owners[2:], [1, 2, 3])
    check("%s: count matches the index list" % carry, cnt, len(idx.split(",")))
    check("%s: the last is -1" % carry, idx.strip().endswith("-1"), True)

# THE bug this planner found: the ownership rule must use what the JOIN removes,
# not what the carry spans. Under `mask` the trim is k+2 -- 22 frames against a
# 17-frame carry -- so using the carry sent chunk 0's own last frame into chunk 1,
# into a region that is then trimmed away.
_, _, _, _, plan_m = replan("mask", 5)
check("a chunk's last frame stays in that chunk", plan_m[1][0], 0)
check("...at its local end, not the next chunk's head", plan_m[1][1], 191)

print("\n13b. the switches")
idx_ns, cnt_ns, _, _, _ = replan("keyframe", 7, start=False)
check("no start: drops frame 0", idx_ns.startswith("0,"), False)
check("...and one fewer image", cnt_ns, 4)
idx_ne, cnt_ne, _, _, _ = replan("keyframe", 7, end=False)
check("no end: drops -1", idx_ne.strip().endswith("-1"), False)
check("...and one fewer image", cnt_ne, 4)
i1, c1, _, _, _ = replan("keyframe", 7, n=1)
check("a single chunk is just its two ends", (i1, c1), ("0, -1", 2))

print("\n13c. scene_frames overrides the chunk schedule")
idx_s, cnt_s, tot_s, _, _ = replan("keyframe", 7, scenes="100 | 200 | 300")
check("boundaries at the scene ends", idx_s, "0, 99, 299, -1")
check("count", cnt_s, 4)
check("commas work too", replan("keyframe", 7, scenes="100,200,300")[0], idx_s)
try:
    PLAN.execute(4, 57, 7, "keyframe", True, True, "100 | abc")
    check("non-numeric scene refused", False, True)
except ValueError as e:
    check("non-numeric scene refused", "not a number" in str(e), True)

print("\n13d. the plan is USABLE by the sampler -- indices survive parsing")
idx, cnt, tot, _, _ = replan("keyframe", 7)
vae = FakeVae() if "FakeVae" in dir() else None
class _V:
    def __init__(self): self.calls = 0
    def encode(self, img):
        self.calls += 1
        return torch.zeros([1, 24, 1, H, W])
SEEN["guiders"].clear(); SEEN["seeds"].clear(); SEEN["targets"].clear()
v = _V()
out = LS.MMH3LoopingSampler.execute(
    FakeNoise(), FakeGuider(), "S", "SG",
    {"conds": [cond("p%d" % i) for i in range(4)]},
    mk(57), 4, 7, 1.0, 1.0, 0, "keyframe",
    torch.zeros([cnt, 64, 64, 3]), idx, v).result
check("the planner's own count is what the sampler wanted", v.calls, cnt)
per_chunk = [len(gg.raw_conds[0][0][1].get("minimax_keyframes") or [])
             for gg in SEEN["guiders"]]
# chunk 0: open + its end = 2. chunks 1-3: carry guide + one planned = 2 each.
check("every chunk ends up with two guides", per_chunk, [2, 2, 2, 2])

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
