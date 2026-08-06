"""Check a finished H3 prompt against the format rules, before anything is sampled.

MMH3TaskSystemPrompt validates the SETTINGS you gave the node. This validates the
TEXT an LLM wrote from them, which is where the interesting failures are: a local
model follows most of a long rule list and quietly drops the rest.

The economics are the whole argument. A chunk is minutes of sampling, and most
format errors do not crash - they render something subtly wrong. A cut timed past
the end of the clip simply never happens; a quoted line of dialogue asks for a sign
instead of speech and you get a caption; a voiceover missing its lips-closed clause
gets mouthed. Every one of those costs a full generation to discover by watching,
and a second to discover here.
"""

import logging
import re

from comfy_api.latest import io

from .nodes_prompt import FORMAT_A, MODES, _achievable

# non_diegetic_music wants instrumentation, tempo, rhythm, dynamics -- never the
# emotional function, which the model cannot render and which crowds out what it can
_MOOD = re.compile(
    r"\b(sad|epic|uplifting|triumphant|menacing|cheerful|tense|heartwarming|nostalgic|"
    r"melancholy|joyful|ominous|hopeful|romantic|eerie|whimsical|dramatic|emotional|"
    r"haunting|playful|somber|sombre)\b", re.I)

_SECTIONS_B = ["subject_definitions", "summary", "retention_analysis",
               "detailed_description", "overall_soundscape", "non_diegetic_music"]
_SECTIONS_A = ["integrated_multimodal_description", "overall_soundscape",
               "non_diegetic_music"]


def _section(prompt, name, following):
    """Body of a `name:` section, up to whichever of `following` comes first."""
    stop = "|".join(r"\n%s\s*:" % re.escape(f) for f in following) or r"\Z"
    m = re.search(r"%s\s*:\s*\n?(.*?)(?=%s|\Z)" % (re.escape(name), stop), prompt, re.S)
    return m.group(1).strip() if m else None


def lint_prompt(prompt, mode="Ref2VA", seconds=0.0):
    """Return a list of problem strings. Empty means clean."""
    p = prompt or ""
    out = []
    is_a = mode in FORMAT_A
    sections = _SECTIONS_A if is_a else _SECTIONS_B
    # the shot body is the FIRST field in the three-field format but the FOURTH in the
    # six-section one -- taking sections[0] silently lints subject_definitions instead
    body_field = _SECTIONS_A[0] if is_a else "detailed_description"

    if not p.strip():
        return ["prompt is empty"]

    for i, name in enumerate(sections):
        if _section(p, name, sections[i + 1:]) is None:
            out.append("missing section: %s" % name)

    body = _section(p, body_field, sections[sections.index(body_field) + 1:]) or ""

    # --- shot structure -------------------------------------------------------
    if body and not body.lstrip().startswith("[Shot 1]"):
        out.append("%s does not open with [Shot 1]" % body_field)
    if re.search(r"\[Shot 1\]\s+At\b", body):
        out.append("[Shot 1] carries a timestamp; only later shots are timed")

    ts = [float(a) * 60 + float(b) for a, b in
          re.findall(r"\[Shot \d+\] At (\d{2}):(\d{2}(?:\.\d+)?)", body)]
    if ts != sorted(ts):
        out.append("shot timestamps are not increasing: %s" % ts)
    if len(ts) != len(set(ts)):
        out.append("duplicate shot timestamps: %s" % ts)
    if seconds > 0 and ts and max(ts) >= seconds:
        _, actual = _achievable(seconds)
        out.append("a cut at %.3fs falls outside the %.3fs clip, so it never happens"
                   % (max(ts), actual))

    nums = [int(n) for n in re.findall(r"\[Shot (\d+)\]", body)]
    if nums and nums != list(range(1, len(nums) + 1)):
        out.append("shot numbers are not 1..N in order: %s" % nums)

    # --- dialogue -------------------------------------------------------------
    if p.count("<d>") != p.count("</d>"):
        out.append("unbalanced <d> tags: %d open, %d close" % (p.count("<d>"), p.count("</d>")))
    for d in re.findall(r"<d>(.*?)</d>", p, re.S):
        if not re.match(r"\s*\[[^\]]+\]", d):
            out.append("<d> block has no [Language] tag: %r" % d.strip()[:60])
        if re.search(r"\(S\d+", d):
            out.append("speaker ID inside <d>; it belongs outside: %r" % d.strip()[:60])
        if re.search(r"\b(says|said|whispers|shouts|sings)\b", d, re.I):
            out.append("delivery verb inside <d>; only the words belong there: %r"
                       % d.strip()[:60])
    for q in re.findall(r'"[^"\n]{0,120}"\s*(?=</d>)|<d>[^<]{0,20}"', p):
        out.append("dialogue in double quotes; quotes mean text shown ON SCREEN, so this "
                   "asks for a sign instead of speech")
        break

    # --- voiceover ------------------------------------------------------------
    for vo in re.finditer(r"says in an off-screen voiceover.*?</d>(.{0,120})", p, re.S):
        if "lips remain" not in vo.group(1):
            out.append("off-screen voiceover is not followed by the lips-closed statement, "
                       "so the character will be animated speaking it")

    # --- audio fields ---------------------------------------------------------
    sound = _section(p, "overall_soundscape", ["non_diegetic_music"]) or ""
    if "<d>" in sound:
        out.append("overall_soundscape contains dialogue; it covers ambience, action sound "
                   "and non-verbal human sound only")

    music = _section(p, "non_diegetic_music", []) or ""
    for w in sorted(set(m.lower() for m in _MOOD.findall(music))):
        out.append("mood word in non_diegetic_music: %r - describe instrumentation, tempo, "
                   "rhythm and dynamics instead" % w)

    # --- labels ---------------------------------------------------------------
    if not is_a:
        defined = set(re.findall(r"<(Picture|Video|Audio|Subject) (\d+)>",
                                 _section(p, "subject_definitions", _SECTIONS_B[1:]) or ""))
        used = set(re.findall(r"<(Picture|Video|Audio|Subject) (\d+)>", body))
        for kind, n in sorted(used - defined):
            out.append("<%s %s> is used in the body but never defined in "
                       "subject_definitions" % (kind, n))
        if re.search(r"\(S\d+", _section(p, "retention_analysis", _SECTIONS_B[3:]) or ""):
            out.append("speaker ID (Sx) in retention_analysis; it belongs only in "
                       "subject_definitions and the body")
        summary = _section(p, "summary", _SECTIONS_B[2:]) or ""
        if summary and not summary.lstrip().startswith("["):
            out.append("summary does not begin with a [task type] prefix")

    return out


class MMH3PromptLint(io.ComfyNode):
    """Validate a finished prompt before it costs a generation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3PromptLint",
            display_name="MMH3 Prompt Lint",
            category="MMH3Tools",
            description=(
                "Check an LLM-written H3 prompt against the format rules. Passes the "
                "prompt through unchanged so it can sit inline between the LLM and the "
                "conditioning node."
            ),
            inputs=[
                io.String.Input("prompt", multiline=True, force_input=True),
                io.Combo.Input("mode", options=MODES, default="Ref2VA",
                               tooltip="Selects which section set is expected: the "
                                       "three-field format for the base modes, the "
                                       "six-section format for Ref2VA."),
                io.Float.Input("seconds", default=0.0, min=0.0, max=150.0, step=0.001,
                               tooltip="Clip duration, so a cut timed past the end is "
                                       "caught. Wire MMH3 Frame Calculator's "
                                       "actual_seconds. 0 skips that check."),
                io.Combo.Input(
                    "on_problem", options=["warn", "error"], default="warn",
                    tooltip="'warn' logs and passes through. 'error' stops the queue - "
                            "worth it when the alternative is discovering the problem "
                            "after minutes of sampling.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="report"),
                io.Int.Output(display_name="problems"),
            ],
        )

    @classmethod
    def execute(cls, prompt, mode, seconds, on_problem) -> io.NodeOutput:
        problems = lint_prompt(prompt, mode, seconds)
        if problems:
            report = "%d problem%s:\n%s" % (len(problems), "" if len(problems) == 1 else "s",
                                            "\n".join("  ! " + x for x in problems))
            logging.warning("[MMH3PromptLint] " + report)
            if on_problem == "error":
                raise ValueError("[MMH3PromptLint] " + report)
        else:
            report = "clean"
            logging.info("[MMH3PromptLint] clean")
        return io.NodeOutput(prompt, report, len(problems))
