"""Pemuatan data dan pembuatan preprocessing (ColumnTransformer + Pipeline).

    - numerik     : ``SimpleImputer`` -> ``StandardScaler``
    - kategorikal : ``SimpleImputer(most_frequent)`` -> ``OneHotEncoder``
    - semuanya digabung dengan ``ColumnTransformer`` lalu dibungkus ``Pipeline``,
      sehingga imputer/scaler/encoder hanya "belajar" dari data train
      (tidak ada data leakage ke validasi maupun test).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder,
    StandardScaler, TargetEncoder,
)

from scripts.feature_engineering import (
    AGG_INPUT_COLS, BIN_COLS, CROSS_COLS, GeoAggregator, add_features,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TARGET = "damage_grade"
ID_COL = "building_id"

NUMERIC_COLS = [
    "count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq",
    "age_unknown", "area_per_floor", "height_per_floor", "log_plinth_area",
]
# Kode wilayah: angkanya hanya label (bukan besaran) -> diperlakukan sebagai kategori
GEO_COLS = ["district_id", "vdcmun_id", "ward_id"]
TE_COLS = ["vdcmun_id", "ward_id"]  # wilayah berkardinalitas tinggi -> TargetEncoder
CATEGORICAL_COLS = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "superstructure",
]


# data
def load_data(data_dir: Path = DATA_DIR):
    """Muat train, test, dan sample_submission dari folder data.

    Args:
        data_dir: Folder yang berisi train.csv, test.csv, sample_submission.csv.

    Returns:
        Tuple ``(train, test, sample_submission)`` berupa DataFrame.
    """
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample


def split_xy(train: pd.DataFrame):
    """Pisahkan fitur dan target dari data train.

    Args:
        train: DataFrame train lengkap dengan kolom ``damage_grade``.

    Returns:
        Tuple ``(X, y)``; ``building_id`` dibuang dari X karena hanya identitas.
    """
    X = train.drop(columns=[TARGET, ID_COL])
    y = train[TARGET]
    return X, y



def _numeric_pipeline(scale: bool = True) -> Pipeline:
    """Pipeline numerik: imputer median (+ StandardScaler bila ``scale``).

    Median dipilih (bukan mean) karena luas dan umur bangunan sangat miring
    ke kanan.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _onehot_pipeline(sparse: bool = True) -> Pipeline:
    """Pipeline kategorikal: imputer most_frequent -> OneHotEncoder.

    Kategori yang jarang (< 20 baris) digabung, dan kategori yang tidak pernah
    terlihat di data train diabaikan saat prediksi.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=20, sparse_output=sparse)),
    ])


def _target_encoder() -> TargetEncoder:
    """TargetEncoder multiclass (5 kolom per fitur: peluang tiap Grade per wilayah).

    ``cv=5`` membuat encoder memakai cross-fitting saat ``fit_transform`` sehingga
    label baris itu sendiri tidak ikut menentukan nilai fiturnya (tanpa leakage).
    Pipeline harus di-fit dengan ``y`` (otomatis oleh ``Pipeline.fit``).
    """
    return TargetEncoder(target_type="multiclass", smooth="auto", cv=5, random_state=42)


def _agg_pipeline() -> Pipeline:
    """Fitur agregat wilayah (dari data train saja) -> impute -> scale."""
    return Pipeline([
        ("aggregator", GeoAggregator()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _feature_step() -> tuple:
    """Langkah pertama pipeline: pembersihan dan feature engineering stateless."""
    return ("features", FunctionTransformer(add_features, validate=False))


# preprocessor model
def build_preprocessor(scale_numeric: bool = True,
                       target_encode_geo: bool = False,
                       extra_features: bool = False) -> Pipeline:
    """Preprocessing berkeluaran sparse untuk Reglog, SVM, dan Decision Tree.

    Langkah:
        1. ``add_features``: umur 999 -> NaN + flag, fitur turunan, kolom silang.
        2. Numerik: imputer median (+ StandardScaler bila ``scale_numeric``).
        3. Kategorikal dan wilayah: imputer most_frequent + OneHotEncoder.

    Args:
        scale_numeric: Standarisasi fitur numerik. Wajib True untuk model
            berbasis jarak/gradien (reglog, SVM); tidak perlu untuk tree.
        target_encode_geo: Tambahkan ``TargetEncoder`` untuk ``vdcmun_id`` dan
            ``ward_id`` di samping one-hot. Membantu Decision Tree (+0.007),
            tidak membantu model linear.
        extra_features: Tambahkan one-hot kolom silang kategorikal (interaksi),
            binning kuantil luas/tinggi/umur (non-linearitas), dan 13 fitur
            agregat wilayah. Menaikkan Macro F1 reglog 0.5122 -> 0.5165.

    Returns:
        ``sklearn.pipeline.Pipeline`` yang belum di-fit.
    """
    transformers = [
        ("num", _numeric_pipeline(scale_numeric), NUMERIC_COLS),
        ("cat", _onehot_pipeline(), CATEGORICAL_COLS),
        ("geo", _onehot_pipeline(), GEO_COLS),
    ]
    if target_encode_geo:
        transformers.append(("te", _target_encoder(), TE_COLS))
    if extra_features:
        transformers += [
            ("cross", _onehot_pipeline(), CROSS_COLS),
            ("bin", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("bins", KBinsDiscretizer(n_bins=10, encode="onehot",
                                                        strategy="quantile"))]), BIN_COLS),
            ("agg", _agg_pipeline(), AGG_INPUT_COLS),
        ]
    return Pipeline([
        _feature_step(),
        ("columns", ColumnTransformer(transformers, remainder="drop")),
    ])


def build_ordinal_preprocessor() -> Pipeline:
    """Preprocessing padat (dense, 42 kolom) untuk Random Forest dan Gradient Boosting.

    Susunan kolom keluaran (urutan tetap, dipakai untuk menandai fitur kategorikal):
        ``len(CATEGORICAL_COLS)`` kolom ordinal-encoded, lalu numerik
        + kode wilayah mentah, 10 kolom TargetEncoder, dan 13 kolom agregat wilayah.

    Kode wilayah dipakai sebagai angka karena kardinalitas ``ward_id`` melebihi
    batas 255 kategori milik HistGradientBoosting. NaN pada umur dibiarkan karena
    kedua model pohon ini menanganinya sendiri. Menaikkan Macro F1 Gradient Boosting
    0.520 -> 0.548 (TargetEncoder +0.021, agregat wilayah +0.003).

    Returns:
        ``sklearn.pipeline.Pipeline`` yang belum di-fit.
    """
    ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=np.nan)),
    ])
    transformers = [
        ("cat", ordinal, CATEGORICAL_COLS),
        ("num", "passthrough", NUMERIC_COLS + GEO_COLS),
        ("te", _target_encoder(), TE_COLS),
        ("agg", GeoAggregator(), AGG_INPUT_COLS),
    ]
    return Pipeline([
        _feature_step(),
        ("columns", ColumnTransformer(transformers, remainder="drop")),
    ])


def build_dense_preprocessor() -> Pipeline:
    """Preprocessing padat (dense, ~75 kolom) khusus Analisis Diskriminan.

    LDA tidak menerima matriks sparse, dan one-hot ``ward_id`` (ribuan kolom)
    tidak muat di memori. Solusinya: one-hot dense untuk kategorikal +
    ``district_id``, sedangkan ``vdcmun_id`` dan ``ward_id`` dikodekan dengan
    ``TargetEncoder`` (+ StandardScaler) dan fitur agregat wilayah.

    Returns:
        ``sklearn.pipeline.Pipeline`` yang belum di-fit (keluaran ndarray dense).
    """
    geo_te = Pipeline([("encoder", _target_encoder()), ("scaler", StandardScaler())])
    transformers = [
        ("num", _numeric_pipeline(scale=True), NUMERIC_COLS),
        ("cat", _onehot_pipeline(sparse=False), CATEGORICAL_COLS + ["district_id"]),
        ("geo", geo_te, TE_COLS),
        ("agg", _agg_pipeline(), AGG_INPUT_COLS),
    ]
    return Pipeline([
        _feature_step(),
        ("columns", ColumnTransformer(transformers, remainder="drop")),
    ])
