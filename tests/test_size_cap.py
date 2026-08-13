import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_save import (
    MIN_SANE_KBPS, SIZE_SAFETY, MMH3SizeCappedCopy,
    capped_copy_plan, scale_filter, size_capped_bitrate,
)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)

def close(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s+-%s" % (got, want, tol)))
    if not ok:
        fails.append(label)

def raises(label, fn):
    try:
        fn()
    except ValueError:
        print("  PASS  " + label)
        return
    except Exception as e:
        print("  FAIL  " + label + "  raised %s, wanted ValueError" % type(e).__name__)
        fails.append(label)
        return
    print("  FAIL  " + label + "  did not raise")
    fails.append(label)


print("\n1. the budget is MiB, not MB")
# The whole point of the node is landing under a stated ceiling. Discord's "100MB"
# is 100 MiB; solving in decimal MB would leave 5 MiB of budget unspent on every
# encode, which is most of the safety margin.
kbps = size_capped_bitrate(100.0, 120.0, 128, safety=1.0)
close("100 MiB / 120s / 128k audio", kbps, 100 * 1048576 * 8 / 1000 / 120 - 128, 0.01)
check("that is above the decimal-MB answer",
      kbps > (100 * 1e6 * 8 / 1000 / 120 - 128), True)

print("\n2. the solved bitrate actually produces the target size")
for target, dur, aud in ((95.0, 120.0, 128), (95.0, 1200.0, 96), (8.0, 30.0, 64)):
    v = size_capped_bitrate(target, dur, aud, safety=1.0)
    mib = (v + aud) * 1000 * dur / 8 / 1048576
    close("%.0f MiB over %.0fs round-trips" % (target, dur), mib, target, 0.001)

print("\n3. safety margin lands the encode UNDER the ceiling")
v_safe = size_capped_bitrate(95.0, 120.0, 128)
v_raw = size_capped_bitrate(95.0, 120.0, 128, safety=1.0)
check("safety shrinks the video bitrate", v_safe < v_raw, True)
close("by exactly the safety factor", v_safe, v_raw * SIZE_SAFETY, 1e-6)
# audio is NOT scaled by safety -- it is encoded at the rate asked for, so the margin
# has to come out of video alone or the total overshoots
mib = (v_safe + 128) * 1000 * 120 / 8 / 1048576
check("resulting size is under target", mib < 95.0, True)

print("\n4. impossible budgets raise rather than encoding garbage")
raises("audio alone exceeds the budget", lambda: size_capped_bitrate(1.0, 1200.0, 128))
raises("zero duration", lambda: size_capped_bitrate(95.0, 0.0, 128))
raises("negative duration", lambda: size_capped_bitrate(95.0, -5.0, 128))

print("\n5. a 20 minute video under 100 MiB is the case that needs downscaling")
# the number from the README table: this is why max_height exists
v = size_capped_bitrate(95.0, 1200.0, 96)
close("20min/95MiB is sub-700 kbps", v, 550, 60)
check("and trips the degradation warning", v < MIN_SANE_KBPS, False)
check("2 hours does trip it", size_capped_bitrate(95.0, 7200.0, 96) < MIN_SANE_KBPS, True)

print("\n6. scale filter: comma escaped, no shell quoting, native passes through")
check("native is no filter at all", scale_filter(0), [])
check("negative is treated as native", scale_filter(-1), [])
check("1080 cap", scale_filter(1080), ["-vf", "scale=-2:min(ih\\,1080)"])
# an unescaped comma ends the scale filter and starts a new one called "1080)"
check("comma is backslash-escaped", "ih\\," in scale_filter(720)[1], True)
# subprocess takes a list and never invokes a shell, so a quote would be literal
check("no stray quotes", "'" in scale_filter(720)[1] or '"' in scale_filter(720)[1], False)

print("\n7. ffprobe is looked for BESIDE the given ffmpeg, never on PATH")
f = MMH3SizeCappedCopy._ffprobe_for
check("a path with no ffmpeg in it", f("/usr/bin/potato"), None)
# nonexistent siblings return None; that is the fallback trigger, not an error
check("nonexistent sibling falls back", f("/nowhere/ffmpeg"), None)
# the real one, if this machine has it, must be the sibling and not something else
import shutil as _sh
real = _sh.which("ffmpeg")
if real:
    got = f(real)
    if got is None:
        print("  SKIP  no ffprobe beside %s" % real)
    else:
        check("resolved beside the real ffmpeg",
              os.path.dirname(got), os.path.dirname(real))
        check("and is named ffprobe",
              "ffprobe" in os.path.basename(got).lower(), True)
else:
    print("  SKIP  no ffmpeg on PATH")

print("\n8. the node is registered and schema-valid")
from mmh3tools import NODES
check("exported", MMH3SizeCappedCopy in NODES, True)
s = MMH3SizeCappedCopy.define_schema()
check("node_id", s.node_id, "MMH3SizeCappedCopy")
check("display_name", s.display_name, "MMH3 Size Capped Copy")
check("category", s.category, "MMH3Tools/utils")
check("is an output node", bool(s.is_output_node), True)
names = [i.id for i in s.inputs]
check("input order", names,
      ["file_path", "target_mb", "max_height", "audio_kbps", "preset", "suffix",
       "ffmpeg_path"])

# --- the ceiling only ever shrinks -------------------------------------------
# target_mb must mean what max_height already means. It did not: the bitrate was
# solved from the target alone and -b:v is a two-pass AVERAGE, so a source under
# the ceiling was re-encoded UP to it.
print("\nsize ceiling never inflates")

# well under the ceiling, no height cap -> nothing to do
needed, eff, why = capped_copy_plan(20.0, 95.0, 768, 0)
check("under target does no work", needed, False)
check("...and says why", "already under" in why, True)

# under the ceiling AND within the height cap -> still nothing to do
needed, eff, why = capped_copy_plan(20.0, 95.0, 768, 1080)
check("under target and under height cap does no work", needed, False)
check("...mentions the height", "768p" in why and "1080p" in why, True)

# over the ceiling -> encode at the real target
needed, eff, why = capped_copy_plan(400.0, 95.0, 768, 0)
check("over target encodes", needed, True)
check("...at the requested target", eff, 95.0)

# under on size but too TALL: must encode, budget clamped to the SOURCE size so
# the downscale cannot inflate the file either
needed, eff, why = capped_copy_plan(20.0, 95.0, 2160, 1080)
check("too tall encodes even when small", needed, True)
check("...clamped to the source size, not the target", eff, 20.0)
check("...and never above it", eff <= 20.0, True)

# over both
needed, eff, why = capped_copy_plan(400.0, 95.0, 2160, 1080)
check("over both", needed, True)
check("...uses the target", eff, 95.0)

# unknown height with a cap set: cannot rule out a downscale, so let it run
needed, eff, why = capped_copy_plan(20.0, 95.0, None, 1080)
check("unknown height with a cap still encodes", needed, True)
check("...clamped to source size", eff, 20.0)
# ...but with NO cap, an unknown height is irrelevant
needed, eff, why = capped_copy_plan(20.0, 95.0, None, 0)
check("unknown height with no cap does no work", needed, False)

# exactly at the ceiling is not over it
check("exactly at target is not over", capped_copy_plan(95.0, 95.0, 768, 0)[0], False)
check("a hair over is over", capped_copy_plan(95.01, 95.0, 768, 0)[0], True)
# a source shorter than the cap is never stretched up to it
check("shorter than the height cap is left alone",
      capped_copy_plan(20.0, 95.0, 720, 1080)[0], False)

# the clamped budget always yields a bitrate at or under the unclamped one
b_clamped = size_capped_bitrate(capped_copy_plan(20.0, 95.0, 2160, 1080)[1], 120.0, 128)
b_target  = size_capped_bitrate(95.0, 120.0, 128)
check("clamping really lowers the bitrate", b_clamped < b_target, True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
