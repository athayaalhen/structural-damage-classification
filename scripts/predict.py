"""Membuat file submission per metode.

Contoh: ``python -m scripts.predict --model reglog``
-> ``submissions/submission_reglog.csv``
"""
import argparse

import joblib
import pandas as pd

from scripts.models import MODEL_BUILDERS, model_path, submission_path
from scripts.preprocess import ID_COL, TARGET, load_data


def main() -> None:
    """Muat model, prediksi data test, simpan submission sesuai format sample."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_BUILDERS, default="reglog")
    args = ap.parse_args()

    _, test, sample = load_data()
    path = model_path(args.model)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} belum ada. Latih dulu: python -m scripts.train --model {args.model} --reweight")
    model = joblib.load(path)
    pred = model.predict(test.drop(columns=[ID_COL]))

    submission = sample.copy()
    submission[ID_COL] = test[ID_COL].values
    submission[TARGET] = (pd.Series(pred).str.replace("Grade ", "", regex=False)
                          .astype(int).values)

    out = submission_path(args.model)
    out.parent.mkdir(exist_ok=True)
    submission.to_csv(out, index=False)
    print(submission[TARGET].value_counts(normalize=True).sort_index())
    print("Tersimpan:", out, submission.shape)


if __name__ == "__main__":
    main()
