from .nodes_loop import MMH3ConcatAV, MMH3FindDivergence, MMH3JoinAV, MMH3PackAV
from .nodes_encode import MMH3StreamingEncode
from .nodes_lint import MMH3PromptLint
from .nodes_save import MMH3StreamingSave
from .nodes_multiprompt import (
    MMH3CondSelect,
    MMH3CondSetSpread,
    MMH3ReferenceMultiPrompt,
)
from .nodes_prompt import MMH3AssetPlan, MMH3ReplaceSection, MMH3TaskSystemPrompt
from .nodes_refs import (
    MMH3ImageKeyframe,
    MMH3ImageToRef,
    MMH3LatentKeyframe,
    MMH3LatentToRef,
    MMH3ReferenceFromLatent,
)
from .nodes_windows import (
    MMH3ContextWindows,
    MMH3SplitAudioToWindows,
    MMH3WindowPlan,
)
from .nodes_util import (
    MMH3DimensionCalculator,
    MMH3FrameCalculator,
    MMH3LatentInfo,
    MMH3UpscaleLadder,
)

NODES = [
    MMH3ReferenceFromLatent,
    MMH3LatentToRef,
    MMH3LatentKeyframe,
    MMH3ImageKeyframe,
    MMH3ImageToRef,
    MMH3FindDivergence,
    MMH3JoinAV,
    MMH3PackAV,
    MMH3StreamingEncode,
    MMH3StreamingSave,
    MMH3ConcatAV,
    MMH3ReferenceMultiPrompt,
    MMH3CondSelect,
    MMH3CondSetSpread,
    MMH3AssetPlan,
    MMH3TaskSystemPrompt,
    MMH3ReplaceSection,
    MMH3PromptLint,
    MMH3FrameCalculator,
    MMH3DimensionCalculator,
    MMH3UpscaleLadder,
    MMH3LatentInfo,
    MMH3ContextWindows,
    MMH3WindowPlan,
    MMH3SplitAudioToWindows,
]

__all__ = ["NODES"]







