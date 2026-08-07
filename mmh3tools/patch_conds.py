"""Keyframes and references, at the same time, without editing core.

WHY THIS EXISTS
---------------
Stock `MiniMaxH3.extra_conds` builds the payload like this:

    if keyframes is not None:
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]
    if refs is not None:
        payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]

The second statement ASSIGNS rather than appends, so any reference silently erases
every keyframe. `PackedLayout` still builds cond rows for those keyframes, so the
layout's rows outnumber the latents feeding them and it dies with a broadcast error --
or, worse, in configurations where the counts happen to line up, renders something
subtly wrong.

That is `docs/core-patches.md` patch 1, and it is what a chained sequence needs:
identity from references, continuity from keyframe anchors, both present at once.

WHY IT CAN BE WRAPPED WHEN PATCHES 3-4 CANNOT
---------------------------------------------
`extra_conds` is a plain method on a class. Monkeypatching only works at callable
boundaries -- you can replace a whole function, never lines inside one -- and this is
a whole function, so the wrap calls the original and repairs its output. Nothing is
copied from core.

Patches 3-4 (per-row masking) have no such boundary: their call sites sit inside a
CLOSURE (`mod(seg)`) and inside `_forward`, so reaching them would mean replacing the
enclosing method, i.e. vendoring GPL-3.0 core code into an MIT pack. Those remain a
file edit until #15375 lands upstream. `MMH3SeedOverlap` is the only node here that
still needs the diff, and it says so at runtime rather than quietly doing nothing.

SAME THREE PROPERTIES AS patch_layout.py
----------------------------------------
  * ABSOLUTE. `cond_video_latents` is rebuilt from scratch, keyframes first to match
    the layout's row order, so this is idempotent and cannot double-apply on top of
    the file edit.
  * INERT UNLESS BOTH ARE PRESENT. With only keyframes, or only references, stock is
    already correct and the payload is left exactly as built.
  * SELF-TESTED at import against the live classes, refusing to install rather than
    corrupting a payload.

It can also only ever improve things: the keyframes+references case currently CRASHES
on stock, so there is no working configuration for this to change.
"""

import logging

logger = logging.getLogger(__name__)

_orig_extra_conds = None
_applied = False
_reason = "not attempted"


def is_applied():
    return _applied


def status():
    return _reason


def per_row_masking_available():
    """Whether core patches 3-4 are present (drozbay's per-row masking, #15375).

    `MMH3SeedOverlap` needs them: without per-row TIMESTEP handling, preserved rows
    still run at the generation timestep, so the model gets clean content labelled as
    noisy and the mask accomplishes nothing at all. It does not error -- it just has
    no effect, which is the worst way to find out.
    """
    try:
        import comfy.ldm.minimax.model as mm
    except Exception:
        return False
    return hasattr(mm, "mask_row_targets") and hasattr(mm, "_mod_row")


def _rebuild(payload, keyframes, refs):
    """cond_video_latents = keyframes THEN refs, matching the layout's row order.

    PackedLayout emits its `cond` segments (keyframes) before its `ref_img` segments,
    and the model zips this list against those rows positionally. Reversing the order
    lines every latent up with the wrong row, which is worse than dropping them.
    """
    latents = [kf["latent"] for kf in keyframes if "latent" in kf]
    latents += [r["latent"] for r in refs if "latent" in r]
    if latents:
        payload["cond_video_latents"] = latents
    return len(latents)


def _make_patched():
    def extra_conds(self, **kwargs):
        out = _orig_extra_conds(self, **kwargs)
        keyframes = kwargs.get("minimax_keyframes") or []
        refs = kwargs.get("minimax_refs") or []
        # only the both-present case is broken; leave everything else exactly as built
        if keyframes and refs:
            payload = out.get("minimax_payload")
            payload = getattr(payload, "cond", None)
            if isinstance(payload, dict):
                _rebuild(payload, keyframes, refs)
        return out
    return extra_conds


def _self_test(mm_base):
    """Check the shapes this wrap depends on, and the rebuild itself."""
    import comfy.conds

    if not hasattr(mm_base, "MiniMaxH3"):
        raise RuntimeError("comfy.model_base has no MiniMaxH3")
    if not hasattr(mm_base.MiniMaxH3, "extra_conds"):
        raise RuntimeError("MiniMaxH3 has no extra_conds to wrap")
    probe = comfy.conds.CONDConstant({"x": 1})
    if getattr(probe, "cond", None) != {"x": 1}:
        raise RuntimeError("CONDConstant no longer exposes its payload as .cond")

    # keyframes first, refs after, refs without a latent skipped
    payload = {"cond_video_latents": ["WRONG"]}
    n = _rebuild(payload,
                 [{"latent": "kf0"}, {"latent": "kf1"}],
                 [{"latent": "ref0"}, {"kind": "audio"}])
    if payload["cond_video_latents"] != ["kf0", "kf1", "ref0"] or n != 3:
        raise RuntimeError("rebuild produced %r" % (payload["cond_video_latents"],))

    # idempotent: running it again over its own output changes nothing
    before = list(payload["cond_video_latents"])
    _rebuild(payload, [{"latent": "kf0"}, {"latent": "kf1"}],
             [{"latent": "ref0"}, {"kind": "audio"}])
    if payload["cond_video_latents"] != before:
        raise RuntimeError("rebuild is not idempotent")


def apply():
    """Install the wrapper. Safe to call repeatedly; only the first call acts."""
    global _orig_extra_conds, _applied, _reason
    if _applied:
        return True

    try:
        import comfy.model_base as mm_base
    except Exception as e:                                    # pragma: no cover
        _reason = "comfy.model_base did not import (%s)" % e
        logger.warning("[MMH3Tools] keyframe+reference fix unavailable: %s", _reason)
        return False

    try:
        _self_test(mm_base)
    except Exception as e:
        _reason = "self-test failed: %s" % e
        logger.warning(
            "[MMH3Tools] keyframe+reference fix NOT applied -- %s. Keyframes and "
            "references together will fail; everything else is unaffected.", _reason)
        return False

    if _orig_extra_conds is None:
        _orig_extra_conds = mm_base.MiniMaxH3.extra_conds
    mm_base.MiniMaxH3.extra_conds = _make_patched()
    _applied = True
    _reason = "applied"
    logger.info("[MMH3Tools] keyframes and references can coexist (MiniMaxH3.extra_conds "
                "wrapped, inert unless both are present)")
    return True
