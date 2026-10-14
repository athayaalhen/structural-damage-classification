"""Training model untuk Structural Damage Classification.

Alur: split train/validasi -> fit Pipeline ->
evaluasi Macro F1 + diagnosis overfit/underfit (Train vs Validasi) ->
(opsional) penyesuaian bobot kelas -> fit ulang pada seluruh data -> simpan.

Contoh::

    python -m scripts.train --model reglog --reweight
    python -m scripts.train --model tree --use_best --reweight
"""
import argparse
import json
import time

import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from scripts.evaluation import (
    ProbaReweighter, fit_gap, get_scores, honest_reweight_gain, optimize_class_weights,
)
from scripts.models import MODEL_BUILDERS, model_path
from scripts.preprocess import ROOT, load_data, split_xy

REPORT_DIR = ROOT / "reports"


def main() -> None:
    """Jalankan seluruh alur training untuk satu metode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_BUILDERS, default="reglog")
    ap.add_argument("--use_best", action="store_true",
                    help="pakai parameter terbaik dari reports/model_selection_<model>.json")
    ap.add_argument("--reweight", action="store_true",
                    help="cari penyesuaian skor per kelas untuk memaksimalkan Macro F1")
    args = ap.parse_args()

    train, _, _ = load_data()
    X, y = split_xy(train)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = MODEL_BUILDERS[args.model]()
    if args.use_best:
        path = REPORT_DIR / f"model_selection_{args.model}.json"
        best = json.loads(path.read_text())["best_params"]
        model.set_params(**best)
        print("Memakai parameter terbaik:", best)

    t0 = time.time()
    model.fit(X_tr, y_tr)
    seconds = time.time() - t0
    print(f"Waktu training: {seconds:.1f}s")

    # evaluasi pada data validasi
    pred = model.predict(X_va)
    val_f1 = f1_score(y_va, pred, average="macro")
    print(f"[{args.model}] Validation Macro F1: {val_f1:.4f}")
    print(classification_report(y_va, pred))
    print(confusion_matrix(y_va, pred))

    # diagnosis overfit / underfit: skor Train vs Validasi
    gap = fit_gap(model, X_tr, y_tr, X_va, y_va)
    print(f"Train F1 = {gap['train_f1']:.4f} | Validasi F1 = {gap['val_f1']:.4f} "
          f"| gap = {gap['gap']:+.4f} -> {gap['verdict']}")

    summary = {"model": args.model, "val_f1_macro": round(float(val_f1), 4),
               "fit_diagnostics": gap, "train_seconds": round(seconds, 1)}

    # penyesuaian bobot kelas
    weights, additive = None, False
    if args.reweight:
        scores, additive = get_scores(model, X_va)
        weights = optimize_class_weights(scores, y_va, model.classes_, additive)
        gain = honest_reweight_gain(scores, y_va, model.classes_, additive)
        adjust = {str(k): round(float(v), 2) for k, v in zip(model.classes_, weights)}
        print("Penyesuaian kelas:", adjust)
        print(f"Estimasi jujur kenaikan Macro F1 dari penyesuaian: {gain:+.4f}")
        summary["class_adjustment"] = {"mode": "tambah" if additive else "kali",
                                       "values": adjust, "honest_gain": round(gain, 4)}

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"train_summary_{args.model}.json").write_text(json.dumps(summary, indent=2))

    # fit ulang pada seluruh data train, lalu simpan
    model.fit(X, y)
    if weights is not None:
        model = ProbaReweighter(model, weights, additive)
    path = model_path(args.model)
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    print("Model disimpan ke", path)


if __name__ == "__main__":
    main()
