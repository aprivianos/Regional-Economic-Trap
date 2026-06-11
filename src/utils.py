import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EXTERNAL = BASE_DIR / "data" / "external"
MODELS_DIR = BASE_DIR / "models"
DATA_PENDUKUNG = BASE_DIR / "pendukung"

SOURCE_FILE = BASE_DIR / "data" / "raw" / "ddac.xlsx"

PROVINSI_MAPPING = {}

TARGET_COL = "G_PDRB_IDR"
FEATURES_NUMERIC = [
    "PDRB_IDR_MLY", "KURS", "PENDUDUK_RB",
    "PDRBKAP_USD", "G_KURS", "G_PENDUDUK", "G_PDRBKAP_USD"
]
FEATURES_CATEGORICAL = ["REG", "KLASIFIKASI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Threshold Klasifikasi (World Bank) ──
LOM_THRESHOLD = 905
UPM_THRESHOLD = 3595
HIGH_THRESHOLD = 11115

# ── Target Naik Kelas ──
# threshold 1 tingkat di atas kelas saat ini
TARGET_NAIK_KELAS = {
    "LOW": {"naik_kelas": "LOM", "target_pdrb": LOM_THRESHOLD, "g_pdrb": 0.0186},
    "LOM": {"naik_kelas": "UPM", "target_pdrb": UPM_THRESHOLD, "g_pdrb": 0.0169},
    "UPM": {"naik_kelas": "HIGH", "target_pdrb": HIGH_THRESHOLD, "g_pdrb": 0.017},
    "HIGH": {"naik_kelas": "HIGH", "target_pdrb": HIGH_THRESHOLD, "g_pdrb": 0.017},
}

# ── Default Persentase Simulasi ──
DEFAULT_APBN_PCT = 10.0
DEFAULT_KUR_PCT = 5.0
DEFAULT_TRANSFER_PCT = 15.0

def ensure_dirs():
    for d in [DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
