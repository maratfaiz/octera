"""Guards against backend/frontend drift in the Russian layer-name labels.

frontend/src/app/studies/[id]/page.tsx hand-copies segmentation.py's
LAYER_LABELS_RU verbatim (kept in sync only by a comment on each side, per
code review) so a doctor sees a translated layer name rather than an
internal code like "nfl_gcl". Nothing enforced the two stayed identical --
this test parses the frontend literal and diffs it against the Python dict
so a future edit to only one side fails CI instead of silently drifting.
"""

import re
from pathlib import Path

from app.services.segmentation import LAYER_LABELS_RU

FRONTEND_PAGE = (
    Path(__file__).resolve().parents[2] / "frontend" / "src" / "app" / "studies" / "[id]" / "page.tsx"
)


def _parse_frontend_layer_labels() -> dict[str, str]:
    source = FRONTEND_PAGE.read_text(encoding="utf-8")
    match = re.search(
        r"const LAYER_LABELS_RU: Record<string, string> = \{(.*?)\};", source, re.DOTALL
    )
    assert match, "Could not find LAYER_LABELS_RU literal in page.tsx -- did it move or get renamed?"
    entries = re.findall(r'(\w+):\s*"((?:[^"\\]|\\.)*)"', match.group(1))
    return {key: value.replace('\\"', '"') for key, value in entries}


def test_frontend_layer_labels_match_backend():
    frontend_labels = _parse_frontend_layer_labels()
    assert frontend_labels == LAYER_LABELS_RU, (
        "frontend/src/app/studies/[id]/page.tsx's LAYER_LABELS_RU has drifted from "
        "backend/app/services/segmentation.py's -- update both together."
    )
