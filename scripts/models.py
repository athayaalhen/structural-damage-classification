"""Definisi model (pipeline lengkap), grid parameter, dan path file per metode.

Setiap model adalah ``Pipeline`` dengan dua langkah, mengikuti materi asisten::

    ("prep", <preprocessing + ColumnTransformer>)  ->  ("algo", <algoritma>)

Nama ``prep`` dan ``algo`` dipakai pada grid parameter (``algo__C``, dst.).

Menambah metode baru: tulis ``build_<nama>``, lalu daftarkan di
``MODEL_BUILDERS`` dan ``PARAM_GRIDS``. ``train.py``, ``predict.py``, dan
``model_selection.py`` otomatis mengenalinya.
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from scripts.feature_engineering import N_AGG_FEATURES
from scripts.preprocess import (
    CATEGORICAL_COLS, GEO_COLS, NUMERIC_COLS, ROOT, TE_COLS,
    build_dense_preprocessor, build_ordinal_preprocessor, build_preprocessor,
)

N_CLASSES = 5  # Grade 1-5


# 4 metode
def build_reglog() -> Pipeline:
    """Regresi Logistik multiclass (softmax) dengan fitur tambahan.

    ``class_weight='balanced'`` dipakai karena metrik kompetisi adalah Macro F1
    dan distribusi kelas tidak seimbang. Fitur tambahan (silang kategorikal,
    binning, agregat wilayah) memberi model linear sedikit non-linearitas.

    Returns:
        Pipeline sklearn yang belum di-fit.
    """
    algo = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                              max_iter=1000)
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=True, extra_features=True)),
        ("algo", algo),
    ])


def build_tree() -> Pipeline:
    """Decision Tree (tanpa scaling, wilayah ditambah TargetEncoder).

    Pohon dangkal underfit (max_depth=8 -> Macro F1 0.39), sedangkan pohon dalam
    jauh lebih baik, sehingga kompleksitas dikendalikan lewat ``min_samples_leaf``
    (cara mengurangi overfit seperti di materi asisten). Nilai ``entropy`` dan
    ``min_samples_leaf=100`` dipilih oleh Grid Search (lihat
    ``reports/model_selection_tree.json``).

    Returns:
        Pipeline sklearn yang belum di-fit.
    """
    algo = DecisionTreeClassifier(criterion="entropy", max_depth=None,
                                  min_samples_leaf=100, class_weight="balanced",
                                  random_state=42)
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=False, target_encode_geo=True)),
        ("algo", algo),
    ])


def build_svm() -> Pipeline:
    """SVM linear (``LinearSVC``, One-vs-Rest otomatis) dengan fitur tambahan.

    ``SVC`` ber-kernel tidak dipakai karena waktunya membengkak secara kuadratik
    terhadap jumlah data (>500 ribu baris). ``dual=False`` dipilih karena jumlah
    sampel jauh lebih besar daripada jumlah fitur.

    Returns:
        Pipeline sklearn yang belum di-fit.
    """
    algo = LinearSVC(C=0.1, class_weight="balanced", dual=False, max_iter=2000)
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=True, extra_features=True)),
        ("algo", algo),
    ])


def build_lda() -> Pipeline:
    """Linear Discriminant Analysis dengan preprocessing dense.

    ``priors`` dibuat seragam (0.2 per kelas) supaya prediksi tidak condong ke
    kelas mayoritas, yang lebih sesuai untuk Macro F1.

    Returns:
        Pipeline sklearn yang belum di-fit.
    """
    algo = LinearDiscriminantAnalysis(priors=[1 / N_CLASSES] * N_CLASSES)
    return Pipeline([("prep", build_dense_preprocessor()), ("algo", algo)])


# metode tambahan (ensemble berbasis pohon)
def build_rf() -> Pipeline:
    """Random Forest: kumpulan banyak decision tree yang hasilnya di-voting.

    Tiap pohon dilatih pada sampel bootstrap dan hanya melihat sebagian fitur
    di setiap split, sehingga varians (overfit) satu pohon berkurang. Memakai
    preprocessing ordinal 42 kolom (jauh lebih cepat daripada one-hot 1.100+
    kolom). ``min_samples_leaf=20`` menjaga ukuran file model tetap wajar.

    Returns:
        Pipeline sklearn yang belum di-fit.
    """
    algo = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=20, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    return Pipeline([("prep", build_ordinal_preprocessor()), ("algo", algo)])


def build_hgb() -> Pipeline:
    """Gradient Boosting (``HistGradientBoostingClassifier``): pohon berurutan.

    Tiap pohon baru memperbaiki kesalahan pohon sebelumnya. Parameter
    (``max_leaf_nodes=127``, ``min_samples_leaf=50``, ``l2_regularization=1.0``)
    hasil percobaan di data validasi.

    Returns:
        Pipeline sklearn yang belum di-fit.
    """
    n_other = (len(NUMERIC_COLS) + len(GEO_COLS)          # numerik + wilayah mentah
               + len(TE_COLS) * N_CLASSES                 # TargetEncoder
               + N_AGG_FEATURES)                          # agregat wilayah
    algo = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=127, min_samples_leaf=50,
        l2_regularization=1.0, class_weight="balanced",
        categorical_features=[True] * len(CATEGORICAL_COLS) + [False] * n_other,
        early_stopping=False, random_state=42,
    )
    return Pipeline([("prep", build_ordinal_preprocessor()), ("algo", algo)])


MODEL_BUILDERS = {
    "reglog": build_reglog,
    "tree": build_tree,
    "svm": build_svm,
    "lda": build_lda,
    "rf": build_rf,
    "hgb": build_hgb,
}
CORE_METHODS = ["reglog", "tree", "svm", "lda"]  # 4 metode tugas

# Ruang pencarian untuk model selection (Grid Search / Random Search)
PARAM_GRIDS = {
    "reglog": {"algo__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    "svm": {"algo__C": [0.003, 0.01, 0.03, 0.1, 0.3]},
    "tree": {
        "algo__max_depth": [None, 30],
        "algo__min_samples_leaf": [20, 50, 100, 200],
        "algo__criterion": ["gini", "entropy"],
    },
    # solver 'svd' tidak mendukung shrinkage jadi dipisah menjadi dua sub-grid
    "lda": [
        {"algo__solver": ["svd"]},
        {"algo__solver": ["lsqr"], "algo__shrinkage": [None, 0.1, 0.5, "auto"]},
    ],
    "rf": {
        "algo__min_samples_leaf": [5, 20, 50],
        "algo__max_features": ["sqrt", 0.4],
    },
    "hgb": {
        "algo__max_iter": [300, 500],
        "algo__learning_rate": [0.05, 0.1],
        "algo__max_leaf_nodes": [31, 63, 127],
    },
}


def model_path(name: str):
    """Path file model tersimpan, mis. ``models/pipeline_model_reglog.pkl``."""
    return ROOT / "models" / f"pipeline_model_{name}.pkl"


def submission_path(name: str):
    """Path file submission, mis. ``submissions/submission_reglog.csv``."""
    return ROOT / "submissions" / f"submission_{name}.csv"
