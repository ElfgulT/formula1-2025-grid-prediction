
# src/f1proj/config.py
from pathlib import Path
from types import SimpleNamespace
# src/f1proj/cli.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Proje kökü = .../f1-2025-proxy-grid
ROOT = Path(__file__).resolve().parents[2]

# Dizinler (istersen bunları kendi makine yoluna göre düzenleyebilirsin)
ARCHIVE_DIR = Path("/Users/elifgultopcu/Downloads/formula1/archive")
LIVE2025_DIR    = Path("/Users/elifgultopcu/Downloads/formula1/2025_live")           # 2025 canlı çıktıları buraya
KAGGLE_2025_DIR = Path("/Users/elifgultopcu/Downloads/formula1/kaggle_2025")         # Kaggle dosyaları (varsa)

FIG_DIR         = Path("/Users/elifgultopcu/Downloads/formula1/figs")
TABLE_DIR       = Path("/Users/elifgultopcu/Downloads/formula1/tables")

# Bayraklar (CLI argümanlarıyla override edilebilir)
SKIP_KAGGLE = False
SKIP_FASTF1 = False

# CLI'nin beklediği tek obje
CFG = SimpleNamespace(
    ROOT=str(ROOT),
    ARCHIVE_DIR=str(ARCHIVE_DIR),
    LIVE2025_DIR=str(LIVE2025_DIR),
    KAGGLE_2025_DIR=str(KAGGLE_2025_DIR),
    FIG_DIR=str(FIG_DIR),
    TABLE_DIR=str(TABLE_DIR),
    SKIP_KAGGLE=SKIP_KAGGLE,
    SKIP_FASTF1=SKIP_FASTF1,
)

__all__ = [
    "ROOT",
    "ARCHIVE_DIR", "LIVE2025_DIR", "KAGGLE_2025_DIR",
    "FIG_DIR", "TABLE_DIR",
    "SKIP_KAGGLE", "SKIP_FASTF1",
    "CFG",
]

# src/f1proj/config.py
from pathlib import Path
import os

# Proje kökü
PROJ_ROOT = Path(__file__).resolve().parents[2]

# Varsayılan veri klasörleri (repo içi)
DATA_DIR     = PROJ_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
SAMPLE_DIR   = DATA_DIR / "sample"
OUTPUTS_DIR  = PROJ_ROOT / "outputs"

# Dışarıdan override edilebilen klasörler:
# Örn: export F1_LIVE2025_DIR="/Users/elif/.../my_live_2025"
LIVE2025_DIR  = Path(os.environ.get("F1_LIVE2025_DIR", str(RAW_DIR)))
ARCHIVE_DIR   = Path(os.environ.get("F1_ARCHIVE_DIR",  str(RAW_DIR)))

# (İsteğe bağlı) çıktılar için dış klasör override
OUTPUTS_DIR   = Path(os.environ.get("F1_OUTPUTS_DIR", str(OUTPUTS_DIR)))