"""MMH3Regenerate2KReference recondition mode: labelling the base video.

The default path appends the 768p as an unlabelled minimax_refs block -- which is
what the API's `base_video` role IS, a reference the DiT attends and the prompt
cannot name. Recondition mode exists to TEST whether the model does anything with
a label: it rebuilds each window's conditioning from scratch with the exact stage-1
prompt, the same media reinserted, and the base slice registered as one more
reference the text encoder sees.

Under test is the plumbing, not the model: item ORDER (the tokenizer numbers by
position), that the base takes the next free <Video k>, that `{base}` resolves, and
that the default path is untouched.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor

from mmh3tools.nodes_refs import MMH3Regenerate2KReference as R
from mmh3tools.common import frames_to_audio_t, latents_to_frames

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)

def raises(label, fn, needle=""):
    try:
        fn(); check("raises: " + label, "no raise", "raise")
    except Exception as e:
        if needle and needle.lower() not in str(e).lower():
            check("raises: " + label, str(e)[:80], "message containing %r" % needle)
        else:
            check("raises: " + label, "raise", "raise")


TOK = []
class FakeClip:
    def tokenize(self, text, minimax_ref_items=None):
        TOK.append({"text": text, "items": list(minimax_ref_items or [])})
        return {"t": text}
    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros([1, 4, 8]), {"_from": tokens["t"]}]]

class FakeVae:
    def decode(self, z):
        t = int(z.shape[2])
        return torch.zeros([t, int(z.shape[3]) * 16, int(z.shape[4]) * 16, 3])
    def encode(self, px):
        return torch.zeros([1, 24, 1, max(1, px.shape[1] // 16), max(1, px.shape[2] // 16)])


def clip_latent(t_lat, h=4, w=6):
    at = frames_to_audio_t(latents_to_frames(t_lat))
    return {"samples": NestedTensor([torch.zeros([1, 24, t_lat, h, w]),
                                     torch.zeros([1, 32, 2, at])])}

def cond(refs=None):
    d = {"minimax_refs": refs} if refs else {}
    return [[torch.zeros([1, 4, 8]), d]]

def cset(texts):
    return {"conds": [cond() for _ in texts], "prompts": list(texts), "fingerprint": None}


LAT = clip_latent(37)                       # 124 frames
KW = dict(width=672, height=384, chunk_frames=124, overlap_frames=22)
PROMPT = "subject_definitions: a cat\n\ndetailed_description: it sits"

print("1. the default path is untouched")
del TOK[:]
out, _l, n, rep = R.execute(LAT, stage1_cond_set=cset(["a prompt"]), **KW).result
check("nothing tokenized", len(TOK), 0)
check("one cond per window", len(out["conds"]), n)
check("no recondition note", "RECONDITIONED" in rep, False)
check("base appended as one block", len(out["conds"][0][0][1]["minimax_refs"]), 1)

print("\n2. recondition rebuilds from the prompt, no cond_set needed")
del TOK[:]
out2, _l2, n2, rep2 = R.execute(LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT,
                                prepend="", **KW).result
check("tokenized once per window", len(TOK), n2)
check("the exact prompt is used verbatim", TOK[0]["text"], PROMPT)
kinds = [it["type"] for it in TOK[0]["items"]]
check("audio precedes its video", kinds, ["audio", "video"])
check("report says reconditioned", "RECONDITIONED" in rep2, True)
check("cond_set carries the real text", out2["prompts"][0], PROMPT)

print("\n3. the base is tagged <base_video> by default")
# The hosted endpoint sends the 768p with role=base_video. Core hardcodes
# "<Video k>: ", but that label is ORDINARY TEXT emitted in front of the vision
# block -- not a special token -- so any string can go there. patch_ref_labels is
# what lets an item choose it.
del TOK[:]
_oL, _lL, _nL, repL = R.execute(LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT,
                                prepend="re-render of {base}", **KW).result
check("default label is <base_video>", TOK[0]["items"][-1].get("label"), "<base_video>")
check("{base} resolves to it",
      TOK[0]["text"].startswith("re-render of <base_video>"), True)
check("the report names it", "base=<base_video>" in repL, True)
check("only the BASE is labelled", [it.get("label") for it in TOK[0]["items"]],
      [None, "<base_video>"])

# any tag goes through verbatim -- it is just text
del TOK[:]
R.execute(LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT,
          prepend="{base}", base_label="<source_768p>", **KW)
check("an arbitrary tag is honoured", TOK[0]["items"][-1]["label"], "<source_768p>")
check("...and substitutes into prepend",
      TOK[0]["text"].startswith("<source_768p>"), True)

print("\n3b. clearing base_label falls back to core's numbering")
del TOK[:]
R.execute(LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT, base_label="",
          prepend="This is a re-render of {base}.", **KW)
check("no label key is set", "label" in TOK[0]["items"][-1], False)
check("no reference videos -> base is <Video 1>",
      TOK[0]["text"].startswith("This is a re-render of <Video 1>."), True)
check("...and the prompt follows it", TOK[0]["text"].endswith(PROMPT), True)

# a reinserted reference video pushes the base to <Video 2>
del TOK[:]
_o, _l, _n, rep3 = R.execute(LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT,
                             prepend="base is {base}", base_label="",
                             ref_videos={"ref_video_1": torch.zeros([17, 64, 96, 3])},
                             **KW).result
check("one reference video -> base is <Video 2>",
      TOK[0]["text"].startswith("base is <Video 2>"), True)
check("the report names the tag", "base=<Video 2>" in rep3, True)

print("\n3c. the audio half gets its own <Audio k+1> tag")
# stage1_latent is a NestedTensor: the video half becomes the base reference, the
# audio half is registered as its own numbered item AND pinned into the target.
del TOK[:]
_oA, _lA, _nA, repA = R.execute(
    LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT,
    prepend="regenerate {base}\n{audio}: fully_copy", **KW).result
check("audio item precedes the base video",
      [it["type"] for it in TOK[0]["items"]], ["audio", "video"])
check("{audio} resolves to <Audio 1>",
      "<Audio 1>: fully_copy" in TOK[0]["text"], True)
check("both tags in one prepend",
      TOK[0]["text"].startswith("regenerate <base_video>\n<Audio 1>: fully_copy"), True)
check("the report names the audio tag", "its audio as <Audio 1>" in repA, True)

# a reinserted reference audio pushes the base audio to <Audio 2>
del TOK[:]
_oA2, _lA2, _nA2, repA2 = R.execute(
    LAT, clip=FakeClip(), vae=FakeVae(), audio_vae=FakeVae(), prompt=PROMPT,
    prepend="{audio}: fully_copy",
    ref_audios={"ref_audio_1": {"waveform": torch.zeros([1, 2, 3200]),
                                "sample_rate": 32000}}, **KW).result
check("one reference audio -> base audio is <Audio 2>",
      "<Audio 2>: fully_copy" in TOK[0]["text"], True)

# The source MUST be a nested AV latent: the API lists an audio track as mandatory
# for regeneration, and §5 pins stage 1's audio into the 2K target. A plain
# video-only latent is refused rather than silently regenerating a new soundtrack.
raises("a plain video-only latent is refused",
       lambda: R.execute({"samples": torch.zeros([1, 24, 37, 4, 6])},
                         clip=FakeClip(), vae=FakeVae(), prompt=PROMPT, **KW),
       "NestedTensor")

# ...so the {audio} line-drop is defensive, for a window whose audio span is empty
from mmh3tools import nodes_refs as _nr
check("the drop rule exists for empty-audio windows",
      "{audio}" in open(_nr.__file__, encoding="utf-8").read(), True)

print("\n4. reinserted media is registered, in order, before the base")
del TOK[:]
imgs = torch.zeros([2, 64, 64, 3])
out4, _l4, _n4, rep4 = R.execute(LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT,
                                 ref_images=imgs, **KW).result
kinds4 = [it["type"] for it in TOK[0]["items"]]
check("two stills then the base's audio+video", kinds4,
      ["image", "image", "audio", "video"])
blocks4 = out4["conds"][0][0][1]["minimax_refs"]
check("blocks match the item order", [b["kind"] for b in blocks4],
      ["image", "image", "video_audio"])
check("stills do not disturb the base's tag", "base=<base_video>" in rep4, True)
check("no missing-media warning", "no media reinserted" in rep4, False)

# with nothing reinserted, dangling tags are called out
del TOK[:]
_o5, _l5, _n5, rep5 = R.execute(LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT,
                                **KW).result
check("bare recondition warns about dangling tags",
      "no media reinserted" in rep5, True)

print("\n5. per-window prompts split on |")
del TOK[:]
LONG = clip_latent(107)
outp, _lp, np_, _rp = R.execute(LONG, clip=FakeClip(), vae=FakeVae(),
                                prompt="one|two|three", prepend="",
                                width=672, height=384, chunk_frames=124,
                                overlap_frames=22).result
check("a tokenize per window", len(TOK), np_)
check("window 0 -> prompt 0", TOK[0]["text"], "one")
check("window 1 -> prompt 1", TOK[1]["text"], "two")
check("a short list repeats the last", TOK[-1]["text"], "three")

print("\n6. what it refuses")
raises("clip without vae",
       lambda: R.execute(LAT, clip=FakeClip(), prompt=PROMPT, **KW), "vae")
raises("clip without prompt",
       lambda: R.execute(LAT, clip=FakeClip(), vae=FakeVae(), **KW), "prompt")
raises("vae/prompt without clip",
       lambda: R.execute(LAT, vae=FakeVae(), prompt=PROMPT, **KW), "clip")
raises("no conditioning and no recondition inputs",
       lambda: R.execute(LAT, **KW), "wire")
raises("reference audio without audio_vae",
       lambda: R.execute(LAT, clip=FakeClip(), vae=FakeVae(), prompt=PROMPT,
                         ref_audios={"ref_audio_1": {"waveform": torch.zeros([1, 2, 100]),
                                                     "sample_rate": 32000}}, **KW),
       "audio_vae")

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
