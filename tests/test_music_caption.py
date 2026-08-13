"""MMH3MusicCaptionSystemPrompt: the local stand-in for MiniMax's caption rewriter.

Music 3 wants a three-section Structured Caption, not a tag list. What is under test
is that the emitted system prompt actually carries the format, that the three lyrics
modes emit mutually exclusive instructions, and that the duration numbers come from
the INSTALLED model rather than being hardcoded.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_music import (
    MMH3MusicCaptionSystemPrompt as N,
    MUSIC_FPS, MUSIC_MAX_FRAMES, MUSIC_MAX_SECONDS, SECTION_TAGS, SUNG_TAGS,
    plan_sections,
)
from mmh3tools.nodes_music import (MMH3MusicCaptionSplit as SPLIT,
                                   split_caption_lyrics)
from mmh3tools import NODES

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)

def flat(t):
    """Collapse whitespace: the source wraps prose, so a phrase check must not
    depend on where a line happens to break."""
    return " ".join(t.split())


def run(**kw):
    base = dict(seconds=120.0, lyrics_mode="write lyrics", delivery="sung",
                supplied_lyrics="", suggest_structure=True, extra_rules="")
    base.update(kw)
    r = N.execute(**base).result
    return r[0], r[1]


print("1. the constants come from the installed model, not this file")
check("frames per second", MUSIC_FPS, 25)
check("max audio frames", MUSIC_MAX_FRAMES, 9000)
check("-> 360s, not the card's '~5 minutes'", MUSIC_MAX_SECONDS, 360.0)
# and they are actually READ from core, not a fallback
try:
    from comfy.ldm.minimax_music.ar import MAX_AUDIO_FRAMES as _M
    check("read from comfy.ldm.minimax_music.ar", MUSIC_MAX_FRAMES, int(_M))
except ImportError:
    print("  SKIP  core has no minimax_music (pre-v0.33.0)")

print("\n2. the caption format is present and is the STRUCTURED one")
s, _ = run()
for sec in ("Global Metadata", "Vocal Details", "Arrangement"):
    check("names %s" % sec, sec in s, True)
check("demands section-level evolution",
      "SECTION-LEVEL INSTRUMENT EVOLUTION" in flat(s), True)
check("BPM as a number", "not a BPM" in s, True)
check("progression not label", "not an emotional label" in s, True)
check("audible only", "Describe only what can be HEARD" in s, True)

print("\n3. lyrics tags")
s, _ = run()
for t in SECTION_TAGS:
    if not (t in s):
        check("tag %s present" % t, False, True)
check("all %d section tags present" % len(SECTION_TAGS),
      all(t in s for t in SECTION_TAGS), True)
check("tags go on their own line", "LINE OF THEIR OWN" in s, True)
check("parentheses for backing vocals", "(Whispered)" in s, True)

print("\n4. the three modes are mutually exclusive")
s_w, _ = run(lyrics_mode="write lyrics")
s_i, _ = run(lyrics_mode="instrumental")
s_s, _ = run(lyrics_mode="lyrics supplied", supplied_lyrics="[Verse]\nla la la")
check("write: asks for two fields", "Emit TWO fields" in s_w, True)
check("instrumental: asks for one", "Emit ONE field" in s_i, True)
check("instrumental: no lyrics tag block", "LINE OF THEIR OWN" in s_i, False)
check("instrumental: says no vocals", "no vocals, an instrumental" in s_i, True)
check("supplied: words are fixed", "are FIXED" in s_s, True)
check("supplied: carries the words as CONTEXT", "la la la" in s_s, True)
# The whole point: the LLM must NOT reproduce them. Asking a language model to echo a
# text verbatim is asking it not to be a language model, and the drift IS the lyric.
check("supplied: asks for the caption ONLY", "Emit ONE field" in s_s, True)
check("supplied: forbids echoing the words", "Do NOT output the lyrics" in s_s, True)
check("supplied: no lyrics field requested", "lyrics: <the" in s_s, False)
check("write mode does NOT claim fixed lyrics", "are FIXED" in s_w, False)

print("\n5. the caption/lyrics agreement check rides with lyrics, not instrumental")
check("write mode has it", "must agree" in s_w, True)
check("supplied mode has it", "must agree" in s_s, True)
check("instrumental does not", "must agree" in s_i, False)

print("\n5b. supplied lyrics BYPASS the LLM entirely")
# Observed: the LLM "completely rewrites lyrics, and not well". The fix is ROUTING,
# not a firmer instruction -- the words go straight from the input to the encoder and
# never enter the LLM's output at all.
r_full = N.execute(seconds=120.0, lyrics_mode="lyrics supplied", delivery="sung",
                   supplied_lyrics="[Verse]\nthe exact words", suggest_structure=True,
                   extra_rules="").result
check("three outputs now", len(r_full), 3)
check("lyrics passed through VERBATIM", r_full[2], "[Verse]\nthe exact words")
check("report tells you to wire it",
      "wire this node's `lyrics` output" in r_full[1].lower(), True)
# the other two modes have no passthrough -- their lyrics come from the LLM
for _m in ("write lyrics", "instrumental"):
    _rr = N.execute(seconds=120.0, lyrics_mode=_m, delivery="sung", supplied_lyrics="",
                    suggest_structure=True, extra_rules="").result
    check("%s: passthrough empty" % _m, _rr[2], "")

print("\n6. structure is sized to the duration")
for secs, want_in in [(10.0, "[Verse]"), (30.0, "[Chorus]"), (60.0, "[Outro]"),
                      (120.0, "[Bridge]"), (300.0, "[Solo]")]:
    sk, _why = plan_sections(secs)
    check("%.0fs skeleton contains %s" % (secs, want_in), want_in in sk, True)
check("short form is shorter than long form",
      len(plan_sections(10.0)[0]) < len(plan_sections(300.0)[0]), True)
s30, r30 = run(seconds=30.0)
check("the skeleton reaches the prompt", "[Chorus]" in s30, True)
check("and the report", "[Chorus]" in r30, True)
check("duration framed as a ceiling", "CEILING, not a target" in s30, True)
# suggest_structure off removes it
s_off, r_off = run(suggest_structure=False)
check("suggest_structure off drops the section list",
      "may deviate from with reason" in s_off, False)
check("...and the report says so", "(not suggested)" in r_off, True)

print("\n7. limits and warnings")
s_over, r_over = run(seconds=9999.0)
check("over-long is clamped, not raised", "360" in r_over, True)
check("...and warned", "clamped" in r_over, True)
_s, r_ign = run(lyrics_mode="instrumental", supplied_lyrics="ignored words")
check("supplied lyrics with the wrong mode is flagged", "is ignored" in r_ign, True)
check("...and does not leak into the prompt", "ignored words" in _s, False)
try:
    run(lyrics_mode="lyrics supplied", supplied_lyrics="   ")
    check("supplied mode with no lyrics raises", "no raise", "raise")
except ValueError:
    check("supplied mode with no lyrics raises", "raise", "raise")

print("\n8. extra_rules and registration")
s_x, _ = run(extra_rules="ALWAYS mention the cowbell.")
check("extra_rules appended verbatim", s_x.rstrip().endswith("ALWAYS mention the cowbell."), True)
check("registered in the pack", N in NODES, True)
sch = N.define_schema()
check("node_id", sch.node_id, "MMH3MusicCaptionSystemPrompt")
check("display_name", sch.display_name, "MMH3 Music Caption System Prompt")
check("category", sch.category, "MMH3Tools/prompt")
check("input order", [i.id for i in sch.inputs],
      ["seconds", "lyrics_mode", "delivery", "supplied_lyrics", "suggest_structure",
       "extra_rules"])

print("\n8b. spoken word suppresses singing")
# Music 3 is a MUSIC model: it defaults to singing, and the observed failure is a take
# that starts spoken and drifts into song. The countermeasures have to be spread
# through the caption, not stated once at the top.
s_sp, r_sp = run(delivery="spoken word")
check("spoken block present", "SPOKEN WORD" in s_sp, True)
check("negative vocal rules", "no pitched singing" in flat(s_sp), True)
check("per-section reinforcement", "REPEAT IT PER SECTION" in s_sp, True)
check("melody given elsewhere to live",
      "GIVE THE MELODY SOMEWHERE ELSE TO LIVE" in s_sp, True)
check("prose not metred lines", "PROSE SENTENCES" in flat(s_sp), True)
check("asks for spoken text, not words", "lyrics: <the spoken text" in s_sp, True)
check("report says spoken", "SPOKEN WORD" in r_sp, True)

# the suggested structure must contain no chorus of any kind
sk_sp, _why = plan_sections(120.0, spoken=True)
check("no sung-hook tags in the spoken skeleton",
      [t for t in sk_sp if t in SUNG_TAGS], [])
check("...and it leans on instrumental sections", "[Instrumental]" in sk_sp, True)
check("sung skeleton still has a chorus", "[Chorus]" in plan_sections(120.0)[0], True)

# sung mode is unaffected
check("sung mode has no spoken block", "SPOKEN WORD" in s_w, False)

# a meaningless combination is flagged rather than silently applied
_si, r_si = run(delivery="spoken word", lyrics_mode="instrumental")
check("spoken + instrumental is flagged", "meaningless" in r_si, True)
check("...and the block is not emitted", "REPEAT IT PER SECTION" in _si, False)

# a supplied chorus works against spoken delivery, so say so
_sc, r_sc = run(delivery="spoken word", lyrics_mode="lyrics supplied",
                supplied_lyrics="[Chorus]\nover and over")
check("supplied [Chorus] warns under spoken delivery", "sung hook" in r_sc, True)

print("\n9. it does NOT carry the old hosted-API advice")
for stale in ["--instrumental", "256000", "comma-separated descriptors", "25-30 seconds"]:
    check("no %r" % stale, stale in s_w, False)

print("\n10. MMH3MusicCaptionSplit - the join to MiniMaxMusic3TextEncode")
# The LLM answers with both fields in ONE string; the encode node wants two sockets.
# Real replies arrive fenced, prefaced, bolded or bulleted, which is why this is
# deliberately not a str.split().
BOTH = "caption: a thing\n\nlyrics: [verse]\nwords"
c, l, n = split_caption_lyrics(BOTH)
check("plain split: caption", c, "a thing")
check("plain split: lyrics", l, "[verse]\nwords")
check("...clean, no notes", n, [])

for label, text in [
    ("code fence", "```\n" + BOTH + "\n```"),
    ("preamble", "Sure! Here you go:\n\n" + BOTH),
    ("bolded labels", "**caption:** a thing\n\n**lyrics:** [verse]\nwords"),
    ("bulleted labels", "- caption: a thing\n- lyrics: [verse]\nwords"),
    ("uppercase labels", "CAPTION: a thing\n\nLYRICS: [verse]\nwords"),
]:
    c2, l2, _n = split_caption_lyrics(text)
    check("%s -> caption clean" % label, c2, "a thing")
    check("%s -> lyrics survive" % label, l2.startswith("[verse]"), True)

# instrumental: no lyrics label at all
c3, l3, n3 = split_caption_lyrics("caption: an instrumental thing")
check("missing lyrics label is not fatal", c3, "an instrumental thing")
check("...lyrics empty", l3, "")
check("...and reported", any("instrumental" in x for x in n3), True)

# labels out of order
c4, l4, n4 = split_caption_lyrics("lyrics: [verse]\nwords\n\ncaption: a thing")
check("reversed order still splits", (c4, l4), ("a thing", "[verse]\nwords"))
check("...and warns", any("BEFORE" in x for x in n4), True)

# no labels at all -> the reply is most likely the caption
c5, l5, n5 = split_caption_lyrics("a big pile of unlabelled caption prose")
check("unlabelled becomes the caption", c5, "a big pile of unlabelled caption prose")
check("...with a warning", any("no 'caption:'" in x for x in n5), True)
check("empty input", split_caption_lyrics("")[:2], ("", ""))

print("\n10b. the node around it")
r = SPLIT.execute(text=BOTH, strict=False).result
check("three outputs", len(r), 3)
check("caption out", r[0], "a thing")
check("lyrics out", r[1], "[verse]\nwords")
check("report counts chars", "caption 7 chars" in r[2], True)

# section tags with no words is a wordless track, easy to misread as a model failure
_c, _l, rep = SPLIT.execute(text="caption: x\n\nlyrics: [verse]\n[chorus]",
                            strict=False).result
check("tags-only lyrics flagged", "no words" in rep, True)

# strict turns the soft cases hard
try:
    SPLIT.execute(text="lyrics: [verse]\nwords", strict=True)
    check("strict: empty caption raises", "no raise", "raise")
except ValueError:
    check("strict: empty caption raises", "raise", "raise")
try:
    SPLIT.execute(text="unlabelled prose", strict=True)
    check("strict: no labels raises", "no raise", "raise")
except ValueError:
    check("strict: no labels raises", "raise", "raise")
_lc, _ll, _lr = SPLIT.execute(text="unlabelled prose", strict=False).result
check("lenient: no labels carries on", _lc, "unlabelled prose")

check("split node registered", SPLIT in NODES, True)
ssch = SPLIT.define_schema()
check("split node_id", ssch.node_id, "MMH3MusicCaptionSplit")
check("split category", ssch.category, "MMH3Tools/prompt")
check("split input order", [i.id for i in ssch.inputs], ["text", "strict"])
check("split output order", [o.display_name for o in ssch.outputs],
      ["caption", "lyrics", "report"])

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
