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
    return mp.MMH3ReferenceMultiPrompt.execute(
        clip=FakeClip(), vae=FakeVae(), audio_vae=FakeVae(),
        width=1344, height=768, length=192, ref_image_size="match",
        prompts={"prompt_%d" % i: p for i, p in enumerate(prompts)},
        ref_images={"ref_image_0": refs},
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
    prompts={"prompt_%d" % i: p for i, p in enumerate(P)},
    ref_images={"ref_image_0": R})
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

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
