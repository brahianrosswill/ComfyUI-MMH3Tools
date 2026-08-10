"""MMH3LoopingSampler: the schedule, the carry and the write-back, without a model.

The sampler call is stubbed, so what is under test is everything AROUND it: that
the chunk count is derived from the clip rather than supplied, that chunks are the
SAME schedule the windowing nodes compute, that each chunk gets its own prompt and
its own span of audio, that the master comes out the length it went in, and that
the source guider is never mutated.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools.nodes_windows import MMH3WindowPlan as PLAN, _plan, _window_frame_spans
from mmh3tools.common import latents_to_frames, frames_to_audio_t, frames_to_latents

H = W = 4
fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def clip(frames, mark_audio=False, pin_audio=False):
    """A whole-clip AV latent. Audio can be ramped so a slice reveals its origin."""
    t = frames_to_latents(frames)
    at = frames_to_audio_t(latents_to_frames(t))
    v = torch.zeros([1, 24, t, H, W])
    a = (torch.arange(at, dtype=torch.float32).reshape(1, 1, 1, at).expand(1, 32, 2, at).clone()
         if mark_audio else torch.zeros([1, 32, 2, at]))
    d = {"samples": NestedTensor([v, a])}
    if pin_audio:
        d["noise_mask"] = NestedTensor([torch.ones([1, 1, t, H, W]),
                                        torch.zeros([1, 1, 2, at])])
    return d


def cond(tag, **extra):
    d = {"tag": tag}; d.update(extra)
    return [[torch.zeros([1, 8]), d]]


def tag_of(c):
    return c[0][1]["tag"]


class FakeGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE_POS", "negative": "BASE_NEG"}
    def set_conds(self, positive, negative=None):
        self.original_conds["positive"] = positive


class FakeNoise:
    def __init__(self, seed=1000):
        self.seed = seed


class FakeVae:
    def __init__(self): self.calls = 0
    def encode(self, img):
        self.calls += 1
        return torch.zeros([1, 24, 1, H, W])


SEEN = {"guiders": [], "seeds": [], "chunks": []}

class FakeSampler:
    """Returns the chunk with its VIDEO stamped, so the write-back is visible."""
    def sample(self, noise, guider, sampler, sigmas, latent):
        SEEN["guiders"].append(guider)
        SEEN["seeds"].append(getattr(noise, "seed", None))
        SEEN["chunks"].append(latent)
        v, a = latent["samples"].unbind()
        marked = torch.full_like(v, float(len(SEEN["chunks"])))
        return {"samples": NestedTensor([marked, a])}, None

LS.SamplerCustomAdvanced = FakeSampler

TOTAL, CHUNK, OV = 3048, 481, 22          # a 127s track in 20s chunks


def run(total=TOTAL, chunk=CHUNK, ov=OV, prompts=None, carry="mask",
        latent=None, kf=None, kf_idx="", vae=None):
    SEEN["guiders"].clear(); SEEN["seeds"].clear(); SEEN["chunks"].clear()
    g = FakeGuider()
    n_expect = len(_plan(total, chunk, ov, "standard_static")[4])
    cs = [cond("p%d" % i) for i in range(prompts if prompts is not None else n_expect)]
    out = LS.MMH3LoopingSampler.execute(
        FakeNoise(), g, "S", "SG", {"conds": cs},
        latent if latent is not None else clip(total),
        chunk, ov, 1.0, 1.0, 0, carry, kf, kf_idx, vae).result
    return out, g


print("\n1. the chunk count is DERIVED from the clip, not supplied")
(out, n, rep), g = run()
check("7 chunks for a 127s clip in 20s pieces", n, 7)
check("sampler ran once per chunk", len(SEEN["chunks"]), 7)
# and it is the SAME schedule the windowing nodes compute
check("matches MMH3WindowPlan's window_count",
      n, PLAN.execute(TOTAL, CHUNK, OV, "standard_static", 0).result[2])

print("\n2. the master comes out the length it went in")
v, a = out["samples"].unbind()
src_v, src_a = clip(TOTAL)["samples"].unbind()
check("video latents unchanged", int(v.shape[2]), int(src_v.shape[2]))
check("audio latents unchanged", int(a.shape[3]), int(src_a.shape[3]))
check("...which is the clip's own length", latents_to_frames(int(v.shape[2])),
      latents_to_frames(int(src_v.shape[2])))

print("\n3. every chunk's output is written back into the master")
# the stub stamps chunk k with k+1, and later chunks overwrite earlier ones in
# their overlap, so the last chunk's stamp must be present at the end
check("the final chunk reached the master", float(v[:, :, -1].max()), 7.0)
check("the first chunk reached the master", float(v[:, :, 0].max()), 1.0)
check("nothing was left unwritten", bool((v > 0).all()), True)

print("\n4. each chunk gets its own prompt, in order")
check("prompt order", [tag_of(gg.raw_conds[0]) for gg in SEEN["guiders"]],
      ["p%d" % i for i in range(7)])

print("\n5. each chunk gets its OWN span of audio")
(out2, _n, _r), _g = run(latent=clip(TOTAL, mark_audio=True))
firsts = [float(c["samples"].unbind()[1][0, 0, 0, 0]) for c in SEEN["chunks"]]
check("audio spans advance, not repeat", firsts == sorted(set(firsts)), True)
check("chunk 0 starts at the track's start", firsts[0], 0.0)
check("every chunk differs", len(set(firsts)), 7)

print("\n6. the guider is copied, never mutated")
check("base positive intact", g.original_conds["positive"], "BASE_POS")
check("distinct copies", len({id(gg) for gg in SEEN["guiders"]}), 7)
check("...none sharing the source's dict",
      any(gg.original_conds is g.original_conds for gg in SEEN["guiders"]), False)
check("every chunk saw the same negative",
      {gg.raw_conds[1] for gg in SEEN["guiders"]}, {"BASE_NEG"})

print("\n7. noise advances per chunk")
check("seven distinct seeds", len(set(SEEN["seeds"])), 7)
check("chunk 0 keeps the wired seed", SEEN["seeds"][0], 1000)

print("\n8. carry='mask' pins the overlap, and honours the master's own mask")
run()
m0 = SEEN["chunks"][0].get("noise_mask")
m1 = SEEN["chunks"][1].get("noise_mask")
check("chunk 0 has no carry to pin", m0, None)
vm1, _am1 = m1.unbind()
ov_lat = _plan(TOTAL, CHUNK, OV, "standard_static")[1]
check("chunk 1 preserves its overlap", float(vm1[:, :, :ov_lat].max()), 0.0)
check("...and generates the rest", float(vm1[:, :, ov_lat:].min()), 1.0)

# a pinned audio track has to survive into every chunk
run(latent=clip(TOTAL, pin_audio=True))
for i in (0, 3, 6):
    _vm, am = SEEN["chunks"][i]["noise_mask"].unbind()
    check("chunk %d keeps the pinned audio" % i, float(am.max()), 0.0)

print("\n9. carry='keyframe' passes the overlap as a guide instead")
if not LS._guides_available():
    print("  SKIP  #15439 not applied")
else:
    run(carry="keyframe")
    check("chunk 0 has no mask", SEEN["chunks"][0].get("noise_mask"), None)
    check("...and no mask on later chunks either",
          SEEN["chunks"][1].get("noise_mask"), None)
    kfs = [gg.raw_conds[0][0][1].get("minimax_keyframes") for gg in SEEN["guiders"]]
    check("chunk 0 has no guide", kfs[0], None)
    check("chunk 1 carries one", len(kfs[1]), 1)
    check("anchored at frame 0", kfs[1][0]["resolved_frame_index"], 0)
    check("a multi-step clip, not a still", int(kfs[1][0]["latent"].shape[2]), ov_lat)
    check("with its audio", "audio_latent" in kfs[1][0], True)

print("\n10. keyframe_indices are frames of the WHOLE clip")
vae = FakeVae()
(out3, _n, rep3), _g = run(kf=torch.zeros([2, 64, 64, 3]), kf_idx="0, 1500", vae=vae)
check("encoded once per image", vae.calls, 2)
placed = [l for l in rep3.splitlines() if "keyframe frame" in l]
check("both placed", len(placed), 2)
check("frame 0 -> chunk 0", "frame 0 -> chunk 0" in placed[0], True)
kfs = [gg.raw_conds[0][0][1].get("minimax_keyframes") for gg in SEEN["guiders"]]
owners = [i for i, k in enumerate(kfs) if k]
check("they land in different chunks", len(set(owners)), 2)

print("\n10b. what is refused")
for kw, err in ((dict(kf_idx="0"), "no keyframes were supplied"),
                (dict(kf=torch.zeros([1, 64, 64, 3]), kf_idx="0"), "vae"),
                (dict(kf=torch.zeros([2, 64, 64, 3]), kf_idx="0", vae=FakeVae()),
                 "zipped"),
                (dict(kf=torch.zeros([1, 64, 64, 3]), kf_idx="99999",
                      vae=FakeVae()), "outside the clip")):
    try:
        run(**kw)
        check("refused: %s" % err, False, True)
    except (ValueError, RuntimeError) as e:
        check("refused: %s" % err, err in str(e), True)

print("\n11. a Basic Guider has no negative")
class BasicGuider:
    def __init__(self): self.original_conds = {"positive": "BASE_POS"}
    def set_conds(self, positive): self.original_conds["positive"] = positive
SEEN["guiders"].clear(); SEEN["seeds"].clear(); SEEN["chunks"].clear()
o = LS.MMH3LoopingSampler.execute(
    FakeNoise(), BasicGuider(), "S", "SG",
    {"conds": [cond("p%d" % i) for i in range(7)]}, clip(TOTAL),
    CHUNK, OV, 1.0, 1.0, 0, "mask", None, "", None).result
check("runs against a one-arg guider", o[1], 7)
check("negative reported as None", SEEN["guiders"][0].raw_conds[1], None)

print("\n12. prompt/chunk count mismatches are reported, not silent")
(_o, n2, rep2), _g = run(prompts=2)
check("still renders every chunk", n2, 7)
check("and says the last repeats", "the last repeats" in rep2, True)
(_o, n3, rep3), _g = run(prompts=12)
check("extras are called out", "the extras are unused" in rep3, True)
try:
    LS.MMH3LoopingSampler.execute(FakeNoise(), FakeGuider(), "S", "SG",
                                  {"conds": []}, clip(TOTAL), CHUNK, OV,
                                  1.0, 1.0, 0, "mask", None, "", None)
    check("empty cond_set raises", False, True)
except ValueError as e:
    check("empty cond_set raises", "no conditioning" in str(e), True)

print("\n13. MMH3KeyframePlanner plans against the same schedule")
idx, cnt, nch, _rep = LS.MMH3KeyframePlanner.execute(TOTAL, CHUNK, OV, True, True).result
check("chunk_count agrees with the sampler", nch, 7)
check("one keyframe per chunk, plus the opening", cnt, 8)
check("opens at 0", idx.startswith("0,"), True)
check("closes on -1", idx.strip().endswith("-1"), True)

# and the plan is USABLE: every index lands in a distinct chunk
vae = FakeVae()
(_o, _n, rep4), _g = run(kf=torch.zeros([cnt, 64, 64, 3]), kf_idx=idx, vae=vae)
check("the planner's count is what the sampler wanted", vae.calls, cnt)
kfs = [gg.raw_conds[0][0][1].get("minimax_keyframes") or [] for gg in SEEN["guiders"]]
check("every chunk got exactly one planned keyframe",
      [len(k) for k in kfs], [2] + [1] * 6)

check("no start drops one", LS.MMH3KeyframePlanner.execute(
    TOTAL, CHUNK, OV, False, True).result[1], 7)
check("no end drops one", LS.MMH3KeyframePlanner.execute(
    TOTAL, CHUNK, OV, True, False).result[1], 7)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
