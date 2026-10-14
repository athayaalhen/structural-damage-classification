"""Feature engineering: pembersihan, fitur turunan, kolom silang, dan agregat wilayah.

Isi modul:
    - ``add_features``: transformasi stateless (aman di dalam ``Pipeline``).
    - ``GeoAggregator``: statistik per wilayah yang dipelajari dari data train.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# konstanta
AGE_UNKNOWN = 999                           # kode "umur tidak diketahui"
AGE_FILL = 30                               # pengisi sementara umur NaN sebelum binning
AGE_BINS = [-1, 5, 10, 20, 40, 80, 1000]    # batas kelompok umur (tahun)

# Kolom silang kategorikal (interaksi) untuk model linear.
# "floors" dan "age_bin" adalah kolom bantu (jumlah lantai dan kelompok umur dalam bentuk teks).
CROSS_DEFINITIONS = {
    "x_sup_fnd": ("superstructure", "foundation_type"),
    "x_sup_roof": ("superstructure", "roof_type"),
    "x_sup_floors": ("superstructure", "floors"),
    "x_sup_age": ("superstructure", "age_bin"),
    "x_fnd_gfloor": ("foundation_type", "ground_floor_type"),
    "x_pos_plan": ("position", "plan_configuration"),
    "x_sup_land": ("superstructure", "land_surface_condition"),
    "x_floors_age": ("floors", "age_bin"),
    "x_sup_fnd_floors": ("superstructure", "foundation_type", "floors"),
    "x_roof_gfloor": ("roof_type", "ground_floor_type"),
}
CROSS_COLS = list(CROSS_DEFINITIONS)

BIN_COLS = ["plinth_area_sq_ft", "height_ft_pre_eq", "age_building"]  # untuk binning kuantil

AGG_GEO_COLS = ["district_id", "vdcmun_id", "ward_id"]
AGG_VALUE_COLS = ["plinth_area_sq_ft", "height_ft_pre_eq", "age_building",
                  "count_floors_pre_eq"]
AGG_INPUT_COLS = AGG_GEO_COLS + AGG_VALUE_COLS
N_AGG_FEATURES = 3 + 2 * 4 + 2  # 3 jumlah wilayah + 8 rata-rata + 2 deviasi = 13


# transformasi stateless
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan nilai sentinel dan tambahkan fitur turunan.

    Fungsi ini tidak mempelajari apa pun dari data (stateless), sehingga
    tidak menyebabkan data leakage dan aman dibungkus ``FunctionTransformer``.

    Args:
        df: DataFrame mentah (tanpa kolom target).

    Returns:
        DataFrame baru dengan kolom tambahan ``age_unknown``, ``area_per_floor``,
        ``height_per_floor``, ``log_plinth_area``, dan 10 kolom silang
        kategorikal ``x_*`` (lihat ``CROSS_DEFINITIONS``). Umur 999 diubah menjadi NaN.
    """
    out = df.copy()

    # umur 999 = tidak diketahui -> NaN + flag
    out["age_unknown"] = (out["age_building"] == AGE_UNKNOWN).astype(int)
    out["age_building"] = out["age_building"].replace(AGE_UNKNOWN, np.nan)

    # fitur turunan numerik
    floors = out["count_floors_pre_eq"].clip(lower=1)
    out["area_per_floor"] = out["plinth_area_sq_ft"] / floors
    out["height_per_floor"] = out["height_ft_pre_eq"] / floors
    out["log_plinth_area"] = np.log1p(out["plinth_area_sq_ft"])

    # kolom silang kategorikal
    helper = out.assign(
        floors=out["count_floors_pre_eq"].astype(str),
        age_bin=pd.cut(out["age_building"].fillna(AGE_FILL), AGE_BINS).astype(str),
    )
    for name, cols in CROSS_DEFINITIONS.items():
        out[name] = helper[cols[0]].str.cat([helper[c] for c in cols[1:]], sep="|")
    return out



class GeoAggregator(BaseEstimator, TransformerMixin):
    """Fitur agregat per wilayah, dipelajari HANYA dari data train (tanpa label).

    Fitur yang dihasilkan (13 kolom, urutan tetap):
        - jumlah bangunan per ``district_id``, ``vdcmun_id``, ``ward_id`` (3);
        - rata-rata luas, tinggi, umur, jumlah lantai per ``vdcmun_id`` dan
          per ``ward_id`` (8);
        - selisih luas dan tinggi bangunan terhadap rata-rata ward-nya (2).

    Wilayah yang tidak pernah muncul di data train menghasilkan NaN
    (ditangani oleh imputer berikutnya). Input harus berupa DataFrame hasil
    ``add_features`` (umur 999 sudah menjadi NaN).
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Hitung jumlah dan rata-rata per wilayah dari data train.

        Args:
            X: DataFrame berisi ``AGG_INPUT_COLS``.
            y: Diabaikan (transformer ini tidak memakai label).

        Returns:
            ``self``.
        """
        self.counts_ = {g: X[g].value_counts() for g in AGG_GEO_COLS}
        self.means_ = {
            (g, c): X.groupby(g)[c].mean()
            for g in ("vdcmun_id", "ward_id") for c in AGG_VALUE_COLS
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Petakan statistik wilayah hasil ``fit`` ke setiap baris.

        Args:
            X: DataFrame berisi ``AGG_INPUT_COLS``.

        Returns:
            ndarray float berbentuk ``(n_baris, N_AGG_FEATURES)``.
        """
        cols = [X[g].map(self.counts_[g]).fillna(0).to_numpy(float) for g in AGG_GEO_COLS]
        for g in ("vdcmun_id", "ward_id"):
            for c in AGG_VALUE_COLS:
                cols.append(X[g].map(self.means_[(g, c)]).to_numpy(float))
        for c in ("plinth_area_sq_ft", "height_ft_pre_eq"):
            ward_mean = X["ward_id"].map(self.means_[("ward_id", c)])
            cols.append((X[c] - ward_mean).to_numpy(float))
        return np.column_stack(cols)
