"""Central place for paths, column names and settings used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "heart_raw.csv"
CLEAN_PATH = DATA_DIR / "processed" / "heart_clean.csv"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"

FIGURES_DIR = ROOT / "reports" / "figures"

# Official UCI source (Cleveland subset, the one used in almost all studies)
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
# Fallback mirror of the same 303 Cleveland records (used if UCI is unreachable)
MIRROR_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv"

# The 13 input features + target, using the UCI attribute names
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Allowed ranges / codes, used to drop impossible values during cleaning
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (60, 250),
    "chol": (80, 700),
    "thalach": (50, 250),
    "oldpeak": (0, 10),
    "ca": (0, 3),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
EXPERIMENT_NAME = "heart-disease-classifier"
