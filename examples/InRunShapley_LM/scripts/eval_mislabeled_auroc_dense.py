#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from examples.InRunShapley_LM.scripts.evaluate_tasks import mislabeled_data_detection


def _latest_iter(grad_dotprods_dir: str) -> int:
    files = [
        f for f in os.listdir(grad_dotprods_dir)
        if f.startswith("shapley_array_iter_") and f.endswith(".npy")
    ]
    if not files:
        raise FileNotFoundError(f"No shapley_array_iter_*.npy in {grad_dotprods_dir}")
    return max(int(f.split("_")[-1].split(".")[0]) for f in files)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", type=str, required=True)
    p.add_argument("--num_train", type=int, default=50000)
    p.add_argument("--iter", type=int, default=None, help="Iteration to evaluate (default: latest).")
    args = p.parse_args()

    result_dir = args.result_dir
    grad_dir = os.path.join(result_dir, "grad_dotprods")
    it = args.iter if args.iter is not None else _latest_iter(grad_dir)

    values = np.load(os.path.join(grad_dir, f"shapley_array_iter_{it}.npy"))
    idx_path = os.path.join(grad_dir, f"shapley_indices_iter_{it}.npy")
    idx = np.load(idx_path) if os.path.exists(idx_path) else None

    mislabeled = np.load(os.path.join(result_dir, "mislabeled_indices.npy"))

    if idx is None:
        dense = values
    else:
        dense = np.zeros(args.num_train, dtype=values.dtype)
        dense[idx.astype(np.int64)] = values

    res = mislabeled_data_detection(dense, mislabeled, output_dir=result_dir)
    out_path = os.path.join(result_dir, "auroc_summary.json")
    with open(out_path, "w") as f:
        json.dump({"iter": int(it), **res}, f, indent=2)
    print(f"[SUMMARY] iter={it} auroc={res['auroc']:.4f} wrote={out_path}")


if __name__ == "__main__":
    main()

