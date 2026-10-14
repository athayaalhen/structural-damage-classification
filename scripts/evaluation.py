"""Evaluasi lanjutan: penyesuaian bobot kelas, diagnosis overfit/underfit, kurva validasi.

Contoh kurva validasi (hasil: ``reports/validation_curve_<model>.json`` dan
``reports/figures/validation_curve_<model>.png``)::

    python -m scripts.evaluation --model tree --n 60000
"""
import argparse
import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, validation_curve

from scripts.models import MODEL_BUILDERS
from scripts.preprocess import ROOT, load_data, split_xy

REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"


def optimize_class_weights(scores, y, classes, additive: bool = False, grid=None,
                           passes: int = 2):
    """Cari penyesuaian skor per kelas yang memaksimalkan Macro F1.

    Macro F1 memperlakukan semua kelas setara, sedangkan prediksi ``argmax``
    cenderung condong ke kelas tertentu. Dilakukan coordinate ascent (satu kelas
    per langkah) pada ``grid`` nilai.

    Args:
        scores: Array ``(n, n_kelas)``: probabilitas (mode kali) atau
            ``decision_function`` (mode tambah, untuk SVM yang tak punya proba).
        y: Label asli data validasi.
        classes: Urutan label kelas (``model.classes_``).
        additive: False -> pengali (skor x w); True -> offset (skor + w).
        grid: Kandidat nilai; default 0.3..1.8 (kali) atau -0.6..0.6 (tambah).
        passes: Berapa kali seluruh kelas dilewati.

    Returns:
        ndarray penyesuaian berukuran ``n_kelas``.
    """
    if grid is None:
        grid = np.arange(-0.6, 0.61, 0.05) if additive else np.arange(0.3, 1.81, 0.1)
    classes, y = np.asarray(classes), np.asarray(y)
    w = np.zeros(len(classes)) if additive else np.ones(len(classes))
    combine = (lambda sc, ww: sc + ww) if additive else (lambda sc, ww: sc * ww)
    for _ in range(passes):
        for k in range(len(classes)):
            best_score, best_value = -1.0, w[k]
            for value in grid:
                trial = w.copy()
                trial[k] = value
                score = f1_score(y, classes[combine(scores, trial).argmax(1)],
                                 average="macro")
                if score > best_score:
                    best_score, best_value = score, value
            w[k] = best_value
    return w


def honest_reweight_gain(scores, y, classes, additive: bool = False) -> float:
    """Estimasi jujur kenaikan Macro F1 dari ``optimize_class_weights``.

    Validasi dibagi dua; penyesuaian dicari di satu bagian dan dinilai di bagian
    lain (bergantian), sehingga tidak terlalu optimistis.

    Returns:
        Rata-rata selisih Macro F1 (dengan penyesuaian - tanpa penyesuaian).
    """
    classes, y = np.asarray(classes), np.asarray(y)
    idx = np.random.RandomState(0).permutation(len(y))
    halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    combine = (lambda sc, ww: sc + ww) if additive else (lambda sc, ww: sc * ww)
    gains = []
    for fit_half, eval_half in ((0, 1), (1, 0)):
        a, b = halves[fit_half], halves[eval_half]
        w = optimize_class_weights(scores[a], y[a], classes, additive)
        base = f1_score(y[b], classes[scores[b].argmax(1)], average="macro")
        new = f1_score(y[b], classes[combine(scores[b], w).argmax(1)], average="macro")
        gains.append(new - base)
    return float(np.mean(gains))


def get_scores(model, X):
    """Ambil skor per kelas: ``predict_proba`` bila ada, kalau tidak ``decision_function``.

    Returns:
        Tuple ``(scores, additive)``; ``additive=True`` bila memakai decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X), False
    return model.decision_function(X), True


class ProbaReweighter:
    """Pembungkus model: prediksi = argmax(skor disesuaikan per kelas).

    Dipakai agar hasil ``optimize_class_weights`` ikut tersimpan di file ``.pkl``
    dan ``predict.py`` tetap cukup memanggil ``model.predict``.
    """
    additive = False 

    def __init__(self, model, weights, additive: bool = False):
        """Simpan model terlatih, penyesuaian kelas, dan modenya (kali/tambah)."""
        self.model = model
        self.weights = np.asarray(weights)
        self.additive = additive
        self.classes_ = model.classes_

    def predict(self, X):
        """Prediksi label dari skor yang sudah disesuaikan."""
        scores, _ = get_scores(self.model, X)
        adjusted = scores + self.weights if self.additive else scores * self.weights
        return self.classes_[adjusted.argmax(1)]



def fit_gap(model, X_tr, y_tr, X_va, y_va, n_train: int = 100000, seed: int = 42) -> dict:
    """Bandingkan Macro F1 data train vs validasi (seperti 'Train vs Test Accuracy' di materi).

    Skor train dihitung pada subsampel ``n_train`` baris supaya cepat. Untuk model
    ber-TargetEncoder (tree, rf, hgb, lda), skor train sedikit terlalu optimistis
    karena encoder dipelajari dari label baris train itu sendiri; jadi ``gap``
    adalah batas atas.

    Args:
        model: Pipeline yang sudah di-fit pada ``X_tr``.
        X_tr, y_tr: Data train.
        X_va, y_va: Data validasi (tidak dipakai saat fit).
        n_train: Jumlah baris train untuk menghitung skor train.
        seed: Random seed subsampel.

    Returns:
        Dict ``train_f1``, ``val_f1``, ``gap``, dan ``verdict`` (heuristik:
        gap > 0.10 = indikasi overfit, gap < 0.03 = tidak overfit).
    """
    sub = X_tr.sample(n=min(n_train, len(X_tr)), random_state=seed)
    train_f1 = f1_score(y_tr.loc[sub.index], model.predict(sub), average="macro")
    val_f1 = f1_score(y_va, model.predict(X_va), average="macro")
    gap = train_f1 - val_f1
    if gap > 0.10:
        verdict = "indikasi overfit (gap besar)"
    elif gap < 0.03:
        verdict = "gap kecil (tidak overfit)"
    else:
        verdict = "gap sedang"
    return {"train_f1": round(float(train_f1), 4), "val_f1": round(float(val_f1), 4),
            "gap": round(float(gap), 4), "verdict": verdict}



VALIDATION_CURVES = {
    "reglog": ("algo__C", [0.003, 0.03, 0.3, 3.0],
               "C (makin besar = regularisasi makin lemah = makin kompleks)"),
    "svm": ("algo__C", [0.003, 0.03, 0.3],
            "C (makin besar = regularisasi makin lemah = makin kompleks)"),
    "tree": ("algo__min_samples_leaf", [5, 20, 50, 100, 200, 500],
             "min_samples_leaf (makin kecil = pohon makin kompleks)"),
    "rf": ("algo__min_samples_leaf", [2, 10, 30, 100],
           "min_samples_leaf (makin kecil = pohon makin kompleks)"),
    "hgb": ("algo__max_leaf_nodes", [7, 15, 31, 63, 127],
            "max_leaf_nodes (makin besar = makin kompleks)"),
}


def validation_curve_report(name: str, n: int = 60000, cv: int = 3, seed: int = 42) -> dict:
    """Hitung dan gambar kurva validasi (skor train vs CV terhadap kompleksitas model).

    Cara membaca: bila skor train naik tetapi skor CV turun saat kompleksitas
    bertambah -> overfit; bila keduanya rendah dan berdekatan -> underfit.
    Hasil disimpan ke ``reports/validation_curve_<name>.json`` dan gambar PNG.

    Args:
        name: Nama model di ``VALIDATION_CURVES``.
        n: Ukuran subsampel stratified.
        cv: Jumlah fold StratifiedKFold.
        seed: Random seed.

    Returns:
        Dict berisi nilai parameter, skor train/CV (mean dan std).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    param, values, label = VALIDATION_CURVES[name]
    train, _, _ = load_data()
    X, y = split_xy(train)
    if n < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    tr, va = validation_curve(MODEL_BUILDERS[name](), X, y, param_name=param,
                              param_range=values, cv=skf, scoring="f1_macro", n_jobs=1)
    result = {
        "model": name, "param": param, "values": values, "n": int(len(X)), "cv": cv,
        "train_mean": tr.mean(1).round(4).tolist(), "train_std": tr.std(1).round(4).tolist(),
        "cv_mean": va.mean(1).round(4).tolist(), "cv_std": va.std(1).round(4).tolist(),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"validation_curve_{name}.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, color, lab in (("train", "#C44E52", "Train"), ("cv", "#4C72B0", "K-Fold CV")):
        mean, std = np.array(result[f"{key}_mean"]), np.array(result[f"{key}_std"])
        ax.plot(values, mean, "o-", color=color, label=lab)
        ax.fill_between(values, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xscale("log")
    ax.set(xlabel=label, ylabel="Macro F1",
           title=f"Kurva validasi - {name} (n={len(X):,}, {cv}-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"validation_curve_{name}.png", dpi=150)
    plt.close(fig)
    return result


def main() -> None:
    """Entry point CLI untuk kurva validasi."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(VALIDATION_CURVES), required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()
    res = validation_curve_report(args.model, args.n, args.cv)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
