"""Test de bout en bout sur un cache synthetique : split -> train -> evaluate."""

import json

import numpy as np

from lsf.data.split import make_signer_split, save_split
from lsf.evaluate import evaluate
from lsf.train import parse_args, train


def test_train_and_evaluate_on_synthetic(tiny_cache, tmp_path):
    z = np.load(tiny_cache)
    split = make_signer_split(z["participant"], z["labels"], val_frac=0.17, test_frac=0.17, n_trials=200)
    save_split(split, tmp_path / "split.json")
    for model in ("baseline", "transformer"):
        out = tmp_path / model
        args = parse_args([
            "--cache", str(tiny_cache), "--split", str(tmp_path / "split.json"), "--model", model,
            "--epochs", "15", "--batch-size", "16", "--workers", "0", "--cpu", "--out", str(out),
            "--model-kwargs", json.dumps({"d_model": 32, "depth": 1, "heads": 2} if model == "transformer" else {"hidden": 32}),
        ])
        summary = train(args)
        assert summary["best_val_top1"] > 0.5, summary  # 4 classes separables : bien au-dessus du hasard
        m = evaluate(tiny_cache, out / "best.pt", tmp_path / f"report_{model}")
        assert m["n"] == np.isin(z["participant"], split["test"]).sum()
        assert 0 <= m["top1"] <= m["top5"] <= 1
        assert (tmp_path / f"report_{model}" / "confused_pairs.csv").exists()
