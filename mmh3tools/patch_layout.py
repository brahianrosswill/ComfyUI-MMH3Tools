"""Interior keyframe anchors, applied at runtime instead of by editing core.

WHY THIS EXISTS
---------------
`PackedLayout` accepts a keyframe only at frame 0 or frame_count-1:

    raise ValueError("only first/last keyframe anchors are supported")

Two anchors is enough to pin a pose. It is not enough to express a trajectory,
which is what chaining chunks needs: the tail of the previous chunk is a RUN of
consecutive frames, and a run cannot be described by its endpoints.

The restriction is not architectural. A keyframe's time coordinate is

    cond_t = kf_base + FRAME_RESCALE * pixel_index

which is linear in PIXEL frames even though FRAME_PER_TOKEN makes the latent
grid non-uniform, because a latent step's span is exactly FRAME_RESCALE times
the frames it covers. Every intermediate index is representable; stock simply
never computes one.

WHY A RUNTIME PATCH AND NOT A DIFF
----------------------------------
`PackedLayout` is constructed inside the model's forward. It is not looked up
from a dict the way `context_handler` is, so there is no injection point and no
subclass route -- MMH3ContextWindows got to avoid patching entirely, this cannot.
That leaves editing core (lost on every `git pull`, fails silently when upstream
moves) or wrapping at runtime. This wraps.

Three properties make that safe:

  * ABSOLUTE, NOT RELATIVE. The fixup recomputes cond positions from scratch
    rather than adjusting whatever was there, so it is idempotent and cannot
    double-apply on top of the file-level keyframe patch in docs/core-patches.

  * INERT UNLESS USED. Keyframes without MMH3_KEY are left exactly as stock
    built them. A graph that does not use MMH3LatentToKeyframes is byte-identical
    with the patch loaded, so importing this pack does not change H3 behaviour.

  * SELF-TESTED. At import the endpoints are rebuilt both ways and required to
    match bit for bit. On any mismatch the patch is NOT applied and nodes that
    need it refuse to run, rather than rendering a silently shifted join.
"""

import logging

logger = logging.getLogger(__name__)

# Private. Rides on the keyframe dict alongside a legal resolved_frame_index,
# because stock validates that field before we get a chance to rewrite anything.
MMH3_KEY = "mmh3_index"

_orig_init = None
_applied = False
_reason = "not attempted"


def is_applied():
    return _applied


def status():
    return _reason


def _ref_cursor_advance(mm, refs):
    """How far ref blocks push the target origin past text_len.

    Refs lay out sequentially from a cursor starting at text_len, and the TARGET
    rows use the cursor's final value as their origin. Keyframe coordinates are
    anchored to that same origin, so without this term any ref would slide the
    anchors backwards relative to the clip they are anchoring.

    Mirrors the cursor arithmetic in PackedLayout.__init__ exactly.
    """
    if not refs:
        return 0.0
    cursor = 0.0
    for blk in refs:
        kind = blk.get("kind")
        if kind == "image":
            cursor += 1.0
        elif kind == "audio":
            cursor += float(blk.get("ref_audio_t", 0))
        elif kind in ("video", "video_audio"):
            rt = float(blk.get("ref_audio_t", 0))
            vt = int(blk.get("latent_t", 0))
            cursor += max(rt, sum(mm._video_t_spans(vt)))
    return cursor


def _cond_t(mm, kf_base, latent_t, frame_count, p):
    """Time coordinate for a keyframe anchored at pixel frame `p`.

    The two endpoints reuse stock's own expressions rather than the general
    formula. They are mathematically identical, but stock sums latent_t floats
    where the general form does one multiply, and those disagree in the last
    bits (~7e-15). Matching stock exactly means a first/last graph is byte
    identical after patching, which is what lets the self-test stay strict.
    """
    if p == 0:
        return float(kf_base)
    if frame_count is not None and p == frame_count - 1:
        return float(kf_base) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(kf_base) + mm.FRAME_RESCALE * float(p)


def _fixup(mm, layout, text_len, latent_t, frame_count, keyframes, refs):
    """Rewrite cond-row time coordinates using the general position formula."""
    kf_base = float(text_len) + _ref_cursor_advance(mm, refs)

    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            "mmh3 patch_layout: %d keyframes but %d cond segments; refusing to "
            "rewrite positions." % (len(keyframes), len(cond_spans)))

    for (a, b), kf in zip(cond_spans, keyframes):
        p = kf.get(MMH3_KEY)
        if p is None:
            continue  # stock keyframe, leave exactly as built
        layout.position_ids[a:b, 0] = _cond_t(mm, kf_base, latent_t, frame_count, p)


def _make_patched(mm):
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=None, refs=None, frame_count=None):
        _orig_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                   keyframes=keyframes, refs=refs, frame_count=frame_count)
        if keyframes and any(kf.get(MMH3_KEY) is not None for kf in keyframes):
            _fixup(mm, self, text_len, latent_t, frame_count, keyframes, refs)
    return __init__


def _self_test(mm, patched_init):
    """Verify the fixup against what is TRUE, not against what stock happens to do.

    Two different invariants, because stock is only correct in one of the cases:

      no refs -- stock's endpoints are right, so require bit-identity. This is
        the strict check: if upstream changes the layout's shape, its cursor
        arithmetic, or the meaning of a cond segment, it fails here at import
        rather than in a render.

      with refs -- unpatched stock hardcodes cond_t = text_len, which is correct
        ONLY when no refs exist; with both present the anchor lands among the
        references instead of on target frame 0. So require the anchor to sit on
        the TARGET ORIGIN (read off the layout itself, not recomputed), and treat
        stock's own answer as informational.

    That second case is why this patch subsumes the file-level keyframe fix in
    docs/core-patches: it makes the coordinate right whether or not core is edited.
    """
    import torch

    text_len, latent_t, lh, lw, audio_t = 7, 7, 4, 6, 16
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))

    ref = {"kind": "video_audio", "latent_t": 2, "latent_h": 4, "latent_w": 6,
           "ref_audio_t": 8}

    stock_agrees = True
    for refs in (None, [ref], [{"kind": "image", "latent_h": 4, "latent_w": 6}]):
        stock_kf = [{"resolved_frame_index": 0}, {"resolved_frame_index": frame_count - 1}]
        ours_kf = [{"resolved_frame_index": 0, MMH3_KEY: 0},
                   {"resolved_frame_index": frame_count - 1, MMH3_KEY: frame_count - 1}]

        a = object.__new__(mm.PackedLayout)
        _orig_init(a, text_len, latent_t, lh, lw, audio_t,
                   keyframes=stock_kf, refs=refs, frame_count=frame_count)

        b = object.__new__(mm.PackedLayout)
        patched_init(b, text_len, latent_t, lh, lw, audio_t,
                     keyframes=ours_kf, refs=refs, frame_count=frame_count)

        if a.position_ids.shape != b.position_ids.shape:
            raise RuntimeError("position_ids shape changed under the patch")

        identical = torch.equal(a.position_ids, b.position_ids)
        if refs is None:
            if not identical:
                n = int((a.position_ids != b.position_ids).any(dim=1).sum())
                raise RuntimeError(
                    "patched endpoints differ from stock in %d rows with no refs "
                    "present, where stock is correct" % n)
            continue

        stock_agrees = stock_agrees and identical

        # the first anchor must coincide with target frame 0
        cond_a = next(a_ for a_, _, k in b.segments if k == "cond")
        vid_a = next(a_ for a_, _, k in b.segments if k == "video")
        got = float(b.position_ids[cond_a, 0])
        want = float(b.position_ids[vid_a, 0])
        if abs(got - want) > 1e-9:
            raise RuntimeError(
                "first anchor at %.6f but target frame 0 is at %.6f (refs present)"
                % (got, want))

    # an interior index must now be accepted, and must land between the endpoints
    interior = [{"resolved_frame_index": 0, MMH3_KEY: i} for i in (0, 5, 9, frame_count - 1)]
    c = object.__new__(mm.PackedLayout)
    patched_init(c, text_len, latent_t, lh, lw, audio_t,
                 keyframes=interior, refs=None, frame_count=frame_count)
    spans = [(a, b) for a, b, k in c.segments if k == "cond"]
    if len(spans) != 4:
        raise RuntimeError("expected 4 cond segments, got %d" % len(spans))
    ts = [float(c.position_ids[a, 0]) for a, _ in spans]
    if ts != sorted(ts) or len(set(ts)) != 4:
        raise RuntimeError("interior anchors are not strictly increasing: %s" % ts)

    return stock_agrees


def apply():
    """Install the wrapper. Safe to call repeatedly; only the first call acts."""
    global _orig_init, _applied, _reason
    if _applied:
        return True

    try:
        import comfy.ldm.minimax.model as mm
    except Exception as e:                                    # pragma: no cover
        _reason = "comfy.ldm.minimax.model did not import (%s)" % e
        logger.warning("[MMH3Tools] interior keyframes unavailable: %s", _reason)
        return False

    for name in ("PackedLayout", "_video_t_spans", "FRAME_RESCALE", "FRAME_PER_TOKEN"):
        if not hasattr(mm, name):
            _reason = "comfy.ldm.minimax.model has no %s" % name
            logger.warning("[MMH3Tools] interior keyframes unavailable: %s", _reason)
            return False

    if _orig_init is None:
        _orig_init = mm.PackedLayout.__init__

    patched = _make_patched(mm)
    try:
        stock_agrees = _self_test(mm, patched)
    except Exception as e:
        _orig_init = None
        _reason = "self-test failed: %s" % e
        logger.warning(
            "[MMH3Tools] interior keyframe patch NOT applied -- %s. Keyframe "
            "chaining nodes will refuse to run; everything else is unaffected.",
            _reason)
        return False

    mm.PackedLayout.__init__ = patched
    _applied = True
    _reason = "applied" + ("" if stock_agrees else " (also correcting the ref-cursor "
                                                  "offset core gets wrong)")
    logger.info("[MMH3Tools] interior keyframe anchors enabled (PackedLayout wrapped, "
                "inert unless a keyframe carries %s). %s", MMH3_KEY, _reason)
    return True
