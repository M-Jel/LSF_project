import numpy as np

from lsf.demo.segmenter import PauseSegmenter
from lsf.llm.prompt import build_messages
from tests.conftest import synthetic_raw


def _still(n):
    f = synthetic_raw(t=1, seed=1)[0]
    return [f.copy() for _ in range(n)]


def _moving(n):
    base = synthetic_raw(t=1, seed=2)[0]
    out = []
    for i in range(n):
        f = base.copy()
        f[522:543, :2] += 0.02 * i
        out.append(f)
    return out


def _absent(n):
    return [synthetic_raw(t=1, right=False)[0] for _ in range(n)]


def test_segmenter_finds_two_signs_separated_by_pauses():
    seg = PauseSegmenter(min_pause=5, min_sign=4)
    stream = _absent(5) + _moving(12) + _still(10) + _moving(9) + _absent(10)
    segments = [s for f in stream if (s := seg.update(f)) is not None]
    assert len(segments) == 2
    assert all(s.shape[1:] == (543, 3) for s in segments)


def test_segmenter_ignores_short_jitter():
    seg = PauseSegmenter(min_pause=5, min_sign=6)
    stream = _still(5) + _moving(3) + _still(10)
    assert [s for f in stream if (s := seg.update(f)) is not None] == []


def test_prompt_contains_all_segments_and_no_network():
    segs = [[("hello", 0.6), ("bye", 0.2)], [("mom", 0.5), ("dad", 0.4)]]
    msgs = build_messages(segs, examples=[([[("yes", 0.9)]], "PHRASE: Yes. | GLOSES: yes")])
    assert msgs[0]["role"] == "system" and len(msgs) == 4
    assert "Segment 2: mom (0.50), dad (0.40)" in msgs[-1]["content"]
    assert np.isfinite(0.6)
