import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from src.utils import MODELS_DIR, DATA_PROCESSED
from src.feature_engineering import prepare_features


KOEF_APBN = 0.119361
KOEF_KUR = 0.319629
KOEF_TKD = 0.172685


def load_model(model_name: str = "xgboost"):
    model = joblib.load(MODELS_DIR / f"model_{model_name}.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    return model, encoders


def predict_future(model, encoders, df: pd.DataFrame, tahun: int) -> pd.DataFrame:
    df_future = df[df["TAHUN"] == tahun].copy()
    if df_future.empty:
        tahun_terakhir = df["TAHUN"].max()
        df_future = df[df["TAHUN"] == tahun_terakhir].copy()
        df_future["TAHUN"] = tahun

    X, _, _ = prepare_features(df_future, fit_encoders=False, encoders=encoders)
    df_future["PREDIKSI_G_PDRB_IDR"] = model.predict(X)
    return df_future[["PROVINSI", "TAHUN", "PREDIKSI_G_PDRB_IDR"]]


def _update_derived_features(df_scenario: pd.DataFrame, df_full: pd.DataFrame,
                              provinsi: str, changes: dict) -> pd.DataFrame:
    df_copy = df_scenario.copy()

    for col, val in changes.items():
        if col in df_copy.columns:
            df_copy[col] = val

    if "PDRB_IDR_MLY" in changes:
        pdrb_new = changes["PDRB_IDR_MLY"]
        if "PENDUDUK_RB" in df_copy.columns:
            penduduk = df_copy["PENDUDUK_RB"].iloc[0]
            df_copy["PDRB_PERKAPITA"] = pdrb_new / penduduk if penduduk else 0

        tahun_now = int(df_copy["TAHUN"].iloc[0])
        prov_data = df_full[df_full["PROVINSI"] == provinsi].copy()
        prov_data = prov_data.sort_values("TAHUN")

        lag1_row = prov_data[prov_data["TAHUN"] == tahun_now - 1]
        lag2_row = prov_data[prov_data["TAHUN"] == tahun_now - 2]
        df_copy["PDRB_IDR_MLY_LAG1"] = lag1_row["PDRB_IDR_MLY"].values[0] if not lag1_row.empty else pdrb_new
        df_copy["PDRB_IDR_MLY_LAG2"] = lag2_row["PDRB_IDR_MLY"].values[0] if not lag2_row.empty else pdrb_new

        three_years = prov_data[
            prov_data["TAHUN"].isin([tahun_now - 2, tahun_now - 1, tahun_now])
        ]["PDRB_IDR_MLY"].tolist()
        if len(three_years) >= 2:
            three_years[-1] = pdrb_new
            df_copy["PDRB_IDR_MLY_ROLL3"] = sum(three_years[-3:]) / min(len(three_years), 3)
        else:
            df_copy["PDRB_IDR_MLY_ROLL3"] = pdrb_new

    return df_copy


def what_if_scenario(model, encoders, df: pd.DataFrame,
                     provinsi: str, tahun: int,
                     changes: dict) -> dict:
    row = df[(df["PROVINSI"] == provinsi) & (df["TAHUN"] == tahun)]
    if row.empty:
        tahun_terakhir = df[df["PROVINSI"] == provinsi]["TAHUN"].max()
        row = df[(df["PROVINSI"] == provinsi) & (df["TAHUN"] == tahun_terakhir)]

    df_scenario = _update_derived_features(row, df, provinsi, changes)

    X_orig, _, _ = prepare_features(row, fit_encoders=False, encoders=encoders)
    X_scen, _, _ = prepare_features(df_scenario, fit_encoders=False, encoders=encoders)

    pred_original = float(model.predict(X_orig)[0])
    pred_scenario = float(model.predict(X_scen)[0])
    delta = pred_scenario - pred_original

    return {
        "provinsi": provinsi,
        "tahun": int(df_scenario["TAHUN"].iloc[0]),
        "prediksi_original": pred_original,
        "prediksi_skenario": pred_scenario,
        "delta": delta,
        "changes": changes,
    }


def predict_all_provinces(model, encoders, df: pd.DataFrame, tahun: int) -> pd.DataFrame:
    df_year = df[df["TAHUN"] == tahun].copy()
    if df_year.empty:
        tahun_latest = df["TAHUN"].max()
        df_year = df[df["TAHUN"] == tahun_latest].copy()
        df_year["TAHUN"] = tahun

    X, _, _ = prepare_features(df_year, fit_encoders=False, encoders=encoders)
    df_year["PREDIKSI_G_PDRB_IDR"] = model.predict(X)
    return df_year[["PROVINSI", "TAHUN", "REG", "PREDIKSI_G_PDRB_IDR"]]


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "data_processed.parquet")
    model, encoders = load_model()

    hasil = predict_all_provinces(model, encoders, df, 2025)
    print(hasil.sort_values("PREDIKSI_G_PDRB_IDR", ascending=False).head(10))

    simulasi = what_if_scenario(model, encoders, df, "DKI Jakarta", 2025,
                                {"PDRB_IDR_MLY": 3000000})
    print(f"\nSimulasi DKI Jakarta:")
    for k, v in simulasi.items():
        print(f"  {k}: {v}")
