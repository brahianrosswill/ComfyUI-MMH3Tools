import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.context_windows import IndexListContextWindow, WindowingState
from mmh3tools.nodes_windows import (
    MMH3ContextWindows, MMH3WindowingState, _audio_index_at, _snap_grid)
from mmh3tools.common import (
    AUDIO_T_DIM, VIDEO_T_DIM, frames_to_audio_t, latents_to_frames)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


def make_state(total_v):
    frames = latents_to_frames(total_v)
    total_a = frames_to_audio_t(frames)
    video = torch.zeros([1, 24, total_v, 6, 10])
    audio = torch.zeros([1, 32, 2, total_a])
    return MMH3WindowingState(
        latents=[video, audio], guide_latents=[None, None],
        guide_entries=[None, None], keyframe_idxs=[None, None],
        latent_shapes=[video.shape, audio.shape], dim=VIDEO_T_DIM,
        is_multimodal=True, temporal_downscale_ratio=4), total_a


print("\n1. boundary mapping is exact on the 5j+2 grid")
# 57 latents = 192 frames = 320 audio latents
for n in (0, 2, 7, 12, 57):
    got = _audio_index_at(n, 57, 320)
    want = frames_to_audio_t(latents_to_frames(n)) if n >= 2 else 0
    check("boundary n=%d" % n, got, want)

print("\n2. window covers the right audio span, on the right axis")
st, total_a = make_state(57)
w = IndexListContextWindow(list(range(0, 17)), dim=VIDEO_T_DIM, total_frames=57,
                           context_overlap=5)
pw = st.prepare_window(w, None)
aw = pw.get_window_for_modality(1)
check("primary dim stays 2", pw.dim, VIDEO_T_DIM)
check("audio dim is 3, not 2", aw.dim, AUDIO_T_DIM)
check("audio total is T40, not stereo 2", aw.total_frames, total_a)
check("audio span start", aw.index_list[0], 0)
check("audio span end", aw.index_list[-1] + 1, frames_to_audio_t(latents_to_frames(17)))

print("\n3. slicing with that window hits the temporal axis")
video, audio = st.latents
vs = pw.get_tensor(video)
as_ = aw.get_tensor(audio)
check("video sliced on dim 2", list(vs.shape), [1, 24, 17, 6, 10])
check("audio keeps stereo=2", int(as_.shape[2]), 2)
check("audio sliced on dim 3", int(as_.shape[3]), len(aw.index_list))

print("\n4. the stock single-dim path would have hit the stereo axis")
bad = IndexListContextWindow(list(range(0, 17)), dim=VIDEO_T_DIM, total_frames=57)
check("stock dim on audio = stereo size", int(audio.shape[VIDEO_T_DIM]), 2)
check("...which is not the audio length", int(audio.shape[VIDEO_T_DIM]) == total_a, False)

print("\n5. windows tile the whole audio track with no gap")
st, total_a = make_state(57)
covered = set()
starts = list(range(0, 57 - 17 + 1, 12)) or [0]
for s in starts:
    w = IndexListContextWindow(list(range(s, min(s + 17, 57))), dim=VIDEO_T_DIM,
                               total_frames=57, context_overlap=5)
    covered |= set(st.prepare_window(w, None).get_window_for_modality(1).index_list)
last = IndexListContextWindow(list(range(57 - 17, 57)), dim=VIDEO_T_DIM, total_frames=57)
covered |= set(st.prepare_window(last, None).get_window_for_modality(1).index_list)
check("audio fully covered", sorted(covered) == list(range(total_a)), True)

print("\n6. grid snapping")
for given, want in [(17, 17), (16, 12), (7, 7), (3, 2), (22, 22), (25, 22)]:
    check("snap %d -> %d" % (given, want), _snap_grid(given), want)

print("\n7. node wires a handler without touching core")
class FakeModel:
    def __init__(self): self.model_options = {}
    def clone(self):
        m = FakeModel(); m.model_options = dict(self.model_options); return m
import comfy.context_windows as C
_orig = C.create_prepare_sampling_wrapper
C.create_prepare_sampling_wrapper = lambda m: None
import mmh3tools.nodes_windows as NW
NW.create_prepare_sampling_wrapper = lambda m: None
m, label = MMH3ContextWindows.execute(FakeModel(), 16, 7, "pyramid",
                                      "standard_static", 1).result
h = m.model_options["context_handler"]
check("handler installed", isinstance(h, NW.MMH3ContextHandler), True)
check("length snapped to grid", h.context_length, 12)
check("overlap snapped to /5", h.context_overlap, 5)
check("dim is video", h.dim, VIDEO_T_DIM)
check("causal fix off", h.causal_window_fix, False)
check("freenoise off", h.freenoise, False)
print("   label:", label.splitlines()[0])
C.create_prepare_sampling_wrapper = _orig

print("\n8. non-multimodal latents pass through untouched")
st_plain = MMH3WindowingState(
    latents=[torch.zeros([1, 24, 57, 6, 10])], guide_latents=[None],
    guide_entries=[None], keyframe_idxs=[None], latent_shapes=None,
    dim=VIDEO_T_DIM, is_multimodal=False, temporal_downscale_ratio=4)
w = IndexListContextWindow(list(range(0, 17)), dim=VIDEO_T_DIM, total_frames=57)
check("returned unchanged", st_plain.prepare_window(w, None) is w, True)

print("\n9. accumulators are sized on each modality's OWN dim")
from mmh3tools.nodes_windows import MMH3ContextHandler
from comfy.context_windows import get_matching_context_schedule, get_matching_fuse_method
h = MMH3ContextHandler(
    context_schedule=get_matching_context_schedule("standard_static"),
    fuse_method=get_matching_fuse_method("pyramid"),
    context_length=17, context_overlap=5, context_stride=1, closed_loop=False,
    dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False)
st, total_a = make_state(57)
accum, counts, biases = h._alloc_accumulators(st.latents, 1)
check("video counts extent", counts[0][0].shape[VIDEO_T_DIM], 57)
check("audio counts extent (not stereo 2)", counts[1][0].shape[AUDIO_T_DIM], total_a)
check("audio counts stereo axis is 1", counts[1][0].shape[VIDEO_T_DIM], 1)
check("video biases length", len(biases[0][0]), 57)
check("audio biases length", len(biases[1][0]), total_a)

print("\n10. the fuse step that crashed now runs on both modalities")
w = IndexListContextWindow(list(range(0, 17)), dim=VIDEO_T_DIM, total_frames=57,
                           context_overlap=5)
pw = st.prepare_window(w, None)
ts = torch.tensor([1.0])
for mod_idx in range(2):
    mw = pw.get_window_for_modality(mod_idx)
    sub = [mw.get_tensor(st.latents[mod_idx])]
    try:
        h.combine_context_window_results(
            st.latents[mod_idx], sub, [None], mw, 0, 1, ts,
            accum[mod_idx], counts[mod_idx], biases[mod_idx])
        check("modality %d fuses" % mod_idx, True, True)
    except RuntimeError as e:
        check("modality %d fuses" % mod_idx, str(e), "no error")
check("audio counts got written on dim 3", float(counts[1][0].sum()) > 0, True)
check("video counts got written on dim 2", float(counts[0][0].sum()) > 0, True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
