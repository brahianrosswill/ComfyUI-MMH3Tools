"""MMH3PromptAccumulate: the first-pass case is the one that goes wrong.

A for loop's carried slot is unwired on the first iteration, so `accumulated`
arrives as None -- and a naive accumulator emits `None | prompt` or a leading
separator, either of which the downstream split then reads as a real prompt.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_prompt import MMH3PromptAccumulate as ACC

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def run(prompt, accumulated=None, separator=" | ", strip_fences=True):
    return ACC.execute(prompt, accumulated, separator, strip_fences).result


print("\n1. a loop's worth of iterations")
t, n, ctx, _ = run("alpha")
check("first pass has no leading separator", t, "alpha")
check("count", n, 1)
check("...and no prior context to offer", ctx, "")
t, n, _, _ = run("beta", t)
check("second", t, "alpha | beta")
t, n, ctx, _ = run("gamma", t)
check("third", t, "alpha | beta | gamma")
check("count", n, 3)

print("\n2. prior_context holds the EARLIER ones, not the current")
check("names each window", ctx.count("--- window "), 2)
check("includes the first", "alpha" in ctx, True)
check("and the second", "beta" in ctx, True)
check("but NOT the one just written", "gamma" in ctx, False)
check("and it tells the model what to keep identical",
      "byte-identical" in ctx, True)

print("\n2b. with NO prompt -- the copy at the top of the loop body")
# this node normally sits AFTER the writing model, so its own prior_context
# cannot reach it: that is a cycle. A second copy fed only the carried value
# reads the context before the model runs.
_, _, ctx0, _ = run(None, None)
check("nothing carried -> no context", ctx0, "")
_, n1, ctx1, _ = run(None, "alpha | beta")
check("carried string passes through untouched", n1, 2)
check("both windows offered", ctx1.count("--- window "), 2)
check("INCLUDING the most recent -- taking it from the result would drop it",
      "beta" in ctx1, True)

print("\n3. an empty carry means first pass, however it arrives")
check("None", run("x", None)[0], "x")
check("empty string", run("x", "")[0], "x")
check("whitespace only", run("x", "   \n ")[0], "x")

print("\n4. an empty PROMPT appends nothing rather than a blank piece")
t, n, _, rep = run("", "a | b")
check("text unchanged", t, "a | b")
check("count unchanged", n, 2)
check("and it says so", "nothing appended" in rep, True)
check("whitespace-only prompt too", run("  \n ", "a | b")[0], "a | b")

print("\n5. code fences are stripped -- they would ride into the encode")
check("json fence", run("```json\nbody\n```")[0], "body")
check("bare fence", run("```\nbody\n```")[0], "body")
check("left alone when off", run("```\nbody\n```", None, " | ", False)[0],
      "```\nbody\n```")

print("\n6. a repeated prompt is reported -- the loop may not be advancing")
_, _, _, rep = run("a", "a | b")
check("flagged", "identical to an earlier one" in rep, True)
_, _, _, rep = run("c", "a | b")
check("not flagged when new", "identical to an earlier one" in rep, False)

print("\n7. a separator without a pipe is refused, not silently accepted")
# it would produce ONE enormous prompt instead of N, and nothing downstream
# would notice -- MMH3ReferenceMultiPrompt splits on '|' and would see one piece
try:
    run("b", "a", " + ")
    check("refused", False, True)
except ValueError as e:
    check("refused", "no pipe in it" in str(e), True)
check("a bare pipe is fine", run("b", "a", "|")[0], "a|b")
check("and empty falls back to the default", run("b", "a", "")[0], "a | b")

print("\n8. the count is what MMH3ReferenceMultiPrompt will actually see")
# same split rule: strip, drop empties
t, n, _, _ = run("c", "a |  | b")
check("empties in the carry are not counted", n, 3)
check("...though they stay in the text until the split", t, "a |  | b | c")

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
