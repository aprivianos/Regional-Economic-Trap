import os
from pathlib import Path

BASE_DIR = Path("D:/ANO/ai-agent-belanja-negara")
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EXTERNAL = BASE_DIR / "data" / "external"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

SOURCE_FILE = Path("D:/ANO/ddac.xlsx")

PROVINSI_MAPPING = {}

TARGET_COL = "G_PDRB_IDR"
FEATURES_NUMERIC = [
    "PDRB_IDR_MLY", "KURS", "PENDUDUK_RB",
    "PDRBKAP_USD", "G_KURS", "G_PENDUDUK", "G_PDRBKAP_USD"
]
FEATURES_CATEGORICAL = ["REG", "KLASIFIKASI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

def ensure_dirs():
    for d in [DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
