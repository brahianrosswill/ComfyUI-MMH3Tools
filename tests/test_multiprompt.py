import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from mmh3tools import nodes_multiprompt as mp

CALLS = {"tokenize": 0, "encode": 0, "vae": 0}


class FakeVae:
    def encode(self, x):
        CALLS["vae"] += 1
        h = max(1, x.shape[1] // 16); w = max(1, x.shape[2] // 16)
        # input-dependent, like a real VAE -- a constant here would make the
        # fingerprint test vacuous
        return torch.full([1, 24, 1, h, w], float(x.float().mean()))
    audio_sample_rate = 32000


class FakeClip:
    def tokenize(self, text, minimax_ref_items=None, **kw):
        CALLS["tokenize"] += 1
        return {"t": text, "n": len(minimax_ref_items or [])}

    def encode_from_tokens_scheduled(self, tokens):
        CALLS["encode"] += 1
        return [[torch.zeros([1, 8]), {"tag": tokens["t"]}]]


def img(h=256, w=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand([1, h, w, 3], generator=g)


def run(prompts, refs):
    # prompts arrive as ONE pipe-separated string; refs as ONE image batch
    text = prompts if isinstance(prompts, str) else " | ".join(prompts)
    return mp.MMH3ReferenceMultiPrompt.execute(
        clip=FakeClip(), vae=FakeVae(), audio_vae=FakeVae(),
        width=1344, height=768, length=192, ref_image_size="match",
        prompts=text, ref_images=refs,
    ).result


fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


P = ["shot one", "shot two", "shot three", "shot four"]
R = img(seed=1)

print("\n1. N prompts -> N conds, one encode each")
CALLS.update(tokenize=0, encode=0, vae=0)
cond_set, latent, count = run(P, R)
check("count", count, 4)
check("conds", len(cond_set["conds"]), 4)
check("encodes", CALLS["encode"], 4)
check("prompts kept in order", cond_set["prompts"], P)

print("\n2. re-run unchanged -> every encode reused")
CALLS.update(tokenize=0, encode=0)
run(P, R)
check("encodes", CALLS["encode"], 0)

print("\n3. edit ONE prompt -> only that one re-encodes")
CALLS.update(tokenize=0, encode=0)
run(["shot one", "shot two CHANGED", "shot three", "shot four"], R)
check("encodes", CALLS["encode"], 1)

print("\n4. change the REFERENCE -> fingerprint shifts, all re-encode")
CALLS.update(tokenize=0, encode=0)
cs2, _, _ = run(P, img(seed=2))
check("encodes", CALLS["encode"], 4)
check("fingerprint differs", cs2["fingerprint"] == cond_set["fingerprint"], False)

print("\n5. change ref_image_size -> also invalidates")
CALLS.update(tokenize=0, encode=0)
mp.MMH3ReferenceMultiPrompt.execute(
    clip=FakeClip(), vae=FakeVae(), audio_vae=FakeVae(),
    width=1344, height=768, length=192, ref_image_size="max",
    prompts=" | ".join(P), ref_images=R)
check("encodes", CALLS["encode"], 4)

print("\n6. refs encoded ONCE per execution, not once per prompt")
CALLS.update(tokenize=0, encode=0, vae=0)
run(["a", "b", "c", "d", "e", "f", "g", "h"], img(seed=3))
check("vae.encode calls for 8 prompts", CALLS["vae"], 1)

print("\n7. every cond carries the refs")
cs, _, _ = run(P, R)
check("minimax_refs present", all("minimax_refs" in c[0][1] for c in cs["conds"]), True)

print("\n8. CondSelect returns the right entry, and refuses out of range")
c, txt = mp.MMH3CondSelect.execute(cs, 2).result
check("prompt text", txt, "shot three")
check("tagged cond", c[0][1]["tag"], "shot three")
try:
    mp.MMH3CondSelect.execute(cs, 9)
    check("out of range raises", False, True)
except ValueError as e:
    check("out of range raises", "out of range" in str(e), True)

print("\n9. no prompts is an error, not an empty set")
try:
    run([], R)
    check("empty raises", False, True)
except ValueError:
    check("empty raises", True, True)


print("\n10. prompts are ONE pipe separated string, in chunk order")
cs, _, n = run("alpha | beta | gamma", R)
check("count", n, 3)
check("split in order", cs["prompts"], ["alpha", "beta", "gamma"])

cs, _, n = run("  alpha  |\n beta \n|  ", R)
check("whitespace stripped, empties dropped", cs["prompts"], ["alpha", "beta"])
check("...so a trailing pipe costs nothing", n, 2)

cs, _, n = run("only one", R)
check("a single prompt needs no pipe", cs["prompts"], ["only one"])

try:
    run("   |  | ", R)
    check("all-empty is refused", False, True)
except ValueError as e:
    check("all-empty is refused", "at least one prompt" in str(e), True)

print("\n11. ref_images is a BATCH -- one <Picture i> per element")
CALLS.update(tokenize=0, encode=0, vae=0)
batch3 = torch.cat([img(seed=11), img(seed=12), img(seed=13)], dim=0)
cs3, _, _ = run(["a"], batch3)
check("three refs from a batch of three", CALLS["vae"], 3)

CALLS.update(vae=0)
cs1, _, _ = run(["b"], img(seed=11))
check("one ref from a batch of one", CALLS["vae"], 1)

# the old code sliced img[:1], so a batch contributed only its FIRST element --
# these two fingerprints would have been identical
check("a 3-batch is not the same reference set as its first frame",
      cs3["fingerprint"] == cs1["fingerprint"], False)

CALLS.update(vae=0)
run(["c"], None)
check("no reference at all is allowed", CALLS["vae"], 0)

print("\n12. use_input_audio swaps the empty audio half for a real track")
from comfy.nested_tensor import NestedTensor
from mmh3tools.common import frames_to_audio_t

class FakeAudioVae:
    audio_sample_rate = 32000
    def encode(self, wav):
        n = int(wav.shape[1] / 32000 * 40)
        return torch.ones([1, 32, 2, n])

def track(seconds):
    return {"waveform": torch.zeros([1, 2, int(32000 * seconds)]), "sample_rate": 32000}

def blank(frames=192):
    at = frames_to_audio_t(frames)
    lt = (frames - 5) // 17 * 5 + 2
    return {"samples": NestedTensor((torch.zeros([1, 24, lt, 4, 4]),
                                     torch.zeros([1, 32, 2, at])))}

l = blank()
want = int(l["samples"].unbind()[1].shape[-1])
mp._use_input_audio(l, FakeAudioVae(), track(8.0))
v, a = l["samples"].unbind()
vm, am = l["noise_mask"].unbind()
check("audio half is the track, not silence", float(a.mean()), 1.0)
check("length unchanged", int(a.shape[-1]), want)
check("video left free (mask 1)", float(vm.mean()), 1.0)
check("audio pinned (mask 0)", float(am.mean()), 0.0)
check("video half untouched", float(v.abs().sum()), 0.0)

# an encode will not land on the required length exactly
l = blank(); mp._use_input_audio(l, FakeAudioVae(), track(12.0))
check("a long track is trimmed to fit", int(l["samples"].unbind()[1].shape[-1]), want)
l = blank(); mp._use_input_audio(l, FakeAudioVae(), track(4.0))
a = l["samples"].unbind()[1]
check("a short track is padded to fit", int(a.shape[-1]), want)
check("...with SILENCE at the end, not a loop", float(a[..., -1].abs().sum()), 0.0)

print("\n12b. the switch and its error")
try:
    mp.MMH3ReferenceMultiPrompt.execute(
        clip=FakeClip(), vae=FakeVae(), audio_vae=FakeAudioVae(),
        width=1344, height=768, length=192, ref_image_size="match",
        prompts="a", ref_images=None, use_input_audio=True)
    check("on without audio raises", False, True)
except ValueError as e:
    check("on without audio raises", "no audio is wired" in str(e), True)

_, lat_off, _ = mp.MMH3ReferenceMultiPrompt.execute(
    clip=FakeClip(), vae=FakeVae(), audio_vae=FakeAudioVae(),
    width=1344, height=768, length=192, ref_image_size="match",
    prompts="a", ref_images=None).result
check("off leaves the half empty and adds no mask",
      "noise_mask" in lat_off, False)

print("\n13. MMH3CondToSet wraps a plain conditioning, no encoder involved")
from mmh3tools.nodes_multiprompt import MMH3CondSelect, MMH3CondToSet
fake_cond = [["emb", {"pooled_output": None}]]
cs = MMH3CondToSet.execute(fake_cond, 1).result[0]
check("one entry", len(cs["conds"]), 1)
check("the SAME conditioning object", cs["conds"][0] is fake_cond, True)
check("prompts filled with empty strings", cs["prompts"], [""])
check("fingerprint None", cs["fingerprint"], None)

cs3 = MMH3CondToSet.execute(fake_cond, 3).result[0]
check("count replicates", len(cs3["conds"]), 3)
check("all entries identical", all(c is fake_cond for c in cs3["conds"]), True)

# the wrap round-trips through CondSelect
sel_c, sel_p = MMH3CondSelect.execute(cs, 0).result
check("CondSelect round-trip", sel_c is fake_cond, True)
check("...with empty label", sel_p, "")

# and satisfies the looping sampler's gate
conds_gate = (cs or {}).get("conds") or []
check("passes the sampler's empty-set gate", len(conds_gate) > 0, True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
