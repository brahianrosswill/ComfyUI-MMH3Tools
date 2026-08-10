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

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
