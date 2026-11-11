"""Demo webcam en mode « signe par signe », avec mesure de latence.

    python -m lsf.demo.webcam --ckpt models/best.pt

Touches : q pour quitter. A la sortie, la latence (extraction par frame,
inference par segment, bout en bout = fin de segment -> top-5 affiche) est
ecrite dans reports/latency_demo.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lsf.demo.extract import DEFAULT_MODEL, HolisticExtractor
from lsf.demo.segmenter import PauseSegmenter
from lsf.features.landmarks import preprocess_sequence
from lsf.utils import load_checkpoint


def stats(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0}
    return {"n": len(ms), "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95))}


@torch.no_grad()
def top5(model, cfg, segment: np.ndarray) -> list[tuple[str, float]]:
    pos, length = preprocess_sequence(segment, cfg["max_len"])
    probs = torch.softmax(model(torch.from_numpy(pos)[None], torch.tensor([length])), -1)[0]
    p, i = probs.topk(5)
    return [(cfg["signs"][k], float(v)) for k, v in zip(i.tolist(), p.tolist())]


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--landmarker", default=str(DEFAULT_MODEL))
    ap.add_argument("--report", default="reports/latency_demo.json")
    a = ap.parse_args()

    model, ck = load_checkpoint(a.ckpt)
    cfg = ck["config"]
    extractor = HolisticExtractor(Path(a.landmarker))
    seg = PauseSegmenter()
    cap = cv2.VideoCapture(a.camera)
    t_extract, t_infer, t_e2e, history = [], [], [], []
    last = []
    t_start = time.perf_counter()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            frame = extractor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int((t0 - t_start) * 1000))
            t1 = time.perf_counter()
            t_extract.append((t1 - t0) * 1000)
            segment = seg.update(frame)
            if segment is not None:
                last = top5(model, cfg, segment)
                t2 = time.perf_counter()
                t_infer.append((t2 - t1) * 1000)
                t_e2e.append((t2 - t0) * 1000)
                history.append(last)
                print(" | ".join(f"{s} {p:.0%}" for s, p in last), flush=True)
            hands = not np.isnan(frame[468:489, 0]).all() or not np.isnan(frame[522:543, 0]).all()
            cv2.putText(bgr, "mains: oui" if hands else "mains: non", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if hands else (0, 0, 255), 2)
            for r, (s, p) in enumerate(last):
                cv2.putText(bgr, f"{s} {p:.0%}", (10, 60 + 28 * r), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(bgr, "reconnaissance de signes isoles - pas une traduction", (10, bgr.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("LSF demo", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        report = {
            "extraction_par_frame": stats(t_extract),
            "inference_par_segment": stats(t_infer),
            "bout_en_bout_fin_de_segment": stats(t_e2e),
            "note": "bout en bout = derniere frame extraite + inference ; n'inclut pas l'attente de la pause (min_pause frames)",
            "torch_threads": torch.get_num_threads(),
        }
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
