"""Model selection: K-Fold CV, Grid Search, Random Search, dan perbandingannya.

Contoh::

    python -m scripts.model_selection --model reglog --cv 5 --tune_size 60000

Hasil dicetak dan disimpan ke ``reports/model_selection_<model>.json``. Pencarian
dilakukan pada subsampel stratified (``--tune_size``) supaya pencarian
parameter selesai dalam waktu wajar; parameter terbaik kemudian dipakai untuk
melatih ulang model pada seluruh data (``train.py --use_best``).
"""
import argparse
import json
import math
import time

import numpy as np
from sklearn.model_selection import (
    GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold,
    cross_val_score, train_test_split,
)

from scripts.models import MODEL_BUILDERS, PARAM_GRIDS
from scripts.preprocess import ROOT, load_data, split_xy

REPORT_DIR = ROOT / "reports"


def _plain(value):
    """Ubah tipe numpy menjadi tipe Python biasa agar bisa disimpan ke JSON."""
    return value.item() if isinstance(value, np.generic) else value


def summarize(search, seconds: float) -> dict:
    """Ringkas hasil satu pencarian (Grid atau Random).

    Args:
        search: Objek ``GridSearchCV`` / ``RandomizedSearchCV`` yang sudah di-fit.
        seconds: Waktu pencarian dalam detik.

    Returns:
        Dict berisi parameter terbaik, skor CV terbaik (Macro F1), jumlah
        kandidat, jumlah fit, dan waktu.
    """
    res = search.cv_results_
    n_cand = len(res["params"])
    return {
        "best_params": {k: _plain(v) for k, v in search.best_params_.items()},
        "best_cv_f1_macro": round(float(search.best_score_), 4),
        "std_of_best": round(float(res["std_test_score"][search.best_index_]), 4),
        "n_candidates": n_cand,
        "n_fits": n_cand * search.cv.get_n_splits(),
        "seconds": round(seconds, 1),
    }


def run_selection(name: str, tune_size: int, cv: int, seed: int = 42) -> dict:
    """Jalankan CV baseline, Grid Search, dan Random Search untuk satu model.

    Args:
        name: Nama model di ``MODEL_BUILDERS``.
        tune_size: Jumlah baris subsampel stratified untuk pencarian.
        cv: Jumlah fold StratifiedKFold.
        seed: Random seed.

    Returns:
        Dict hasil lengkap (baseline, grid, random, parameter terpilih).
    """
    train, _, _ = load_data()
    X, y = split_xy(train)
    if tune_size < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=tune_size, stratify=y,
                                      random_state=seed)
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    grid_space = PARAM_GRIDS[name]
    n_combos = len(ParameterGrid(grid_space))
    n_iter = max(3, math.ceil(n_combos / 2))
    print(f"[{name}] data={len(X)} cv={cv} kombinasi_grid={n_combos} n_iter_random={n_iter}",
          flush=True)

    t = time.time()
    base = cross_val_score(MODEL_BUILDERS[name](), X, y, cv=skf, scoring="f1_macro")
    result = {
        "model": name, "tune_size": int(len(X)), "cv": cv,
        "baseline": {"cv_f1_macro": round(float(base.mean()), 4),
                     "std": round(float(base.std()), 4),
                     "seconds": round(time.time() - t, 1)},
    }
    print("  baseline (K-Fold CV, parameter default):", result["baseline"], flush=True)

    t = time.time()
    grid = GridSearchCV(MODEL_BUILDERS[name](), grid_space, scoring="f1_macro",
                        cv=skf, n_jobs=1, refit=False).fit(X, y)
    result["grid_search"] = summarize(grid, time.time() - t)
    print("  grid search  :", result["grid_search"], flush=True)

    t = time.time()
    rand = RandomizedSearchCV(MODEL_BUILDERS[name](), grid_space, n_iter=n_iter,
                              scoring="f1_macro", cv=skf, n_jobs=1, refit=False,
                              random_state=seed).fit(X, y)
    result["random_search"] = summarize(rand, time.time() - t)
    print("  random search:", result["random_search"], flush=True)

    best = max((result["grid_search"], result["random_search"]),
               key=lambda r: r["best_cv_f1_macro"])
    result["best_params"] = best["best_params"]
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / f"model_selection_{name}.json"
    out.write_text(json.dumps(result, indent=2))
    print("  disimpan:", out, "| parameter terpilih:", result["best_params"], flush=True)
    return result


def main() -> None:
    """Entry point CLI."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=[m for m in MODEL_BUILDERS if m != "hgb"],
                    required=True)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--tune_size", type=int, default=60000)
    args = ap.parse_args()
    run_selection(args.model, args.tune_size, args.cv)


if __name__ == "__main__":
    main()
