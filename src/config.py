"""
Central configuration for HealthLens AI.

Everything that is shared across modules (paths, column names, random seed,
plausible value ranges) lives here so there is exactly ONE place to change it.
All paths are relative to the project root, so the project runs on any machine.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (never hard-code absolute paths!)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DEFAULT_DATASET = DATA_DIR / "pima_diabetes.csv"
RESULTS_FILE = MODELS_DIR / "results.joblib"

# Public mirror of the Pima Indians Diabetes Database (original: NIDDK / UCI).
# The raw file has NO header row; data_loader adds the standard column names.
DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

# ----------------------------------------------------------------------------
# Reproducibility / experiment settings
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ----------------------------------------------------------------------------
# Dataset schema
# ----------------------------------------------------------------------------
TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ALL_COLUMNS = FEATURES + [TARGET]

# In the Pima data a value of 0 is physiologically impossible for these columns
# (nobody has 0 mg/dL glucose or 0 BMI) -> a 0 means "not measured".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Human-friendly information used by the UI.
# hard_min / hard_max  -> physiologically plausible limits (inputs outside = error)
# default              -> starting value in the prediction form
FEATURE_INFO = {
    "Pregnancies": dict(
        label="Number of pregnancies", unit="count", step=1, default=2,
        hard_min=0, hard_max=20,
        help="Times pregnant. The dataset only contains women.",
    ),
    "Glucose": dict(
        label="Plasma glucose", unit="mg/dL", step=1, default=117,
        hard_min=40, hard_max=400,
        help="Plasma glucose concentration 2 hours after an oral glucose tolerance test.",
    ),
    "BloodPressure": dict(
        label="Diastolic blood pressure", unit="mm Hg", step=1, default=72,
        hard_min=30, hard_max=200,
        help="Diastolic blood pressure.",
    ),
    "SkinThickness": dict(
        label="Triceps skin-fold thickness", unit="mm", step=1, default=23,
        hard_min=1, hard_max=100,
        help="Skin-fold thickness at the triceps, a rough indicator of body fat.",
    ),
    "Insulin": dict(
        label="2-hour serum insulin", unit="µU/mL", step=1, default=30,
        hard_min=1, hard_max=1000,
        help="Serum insulin 2 hours after the glucose test.",
    ),
    "BMI": dict(
        label="Body mass index", unit="kg/m²", step=0.1, default=32.0,
        hard_min=10.0, hard_max=70.0,
        help="Weight in kg divided by height in metres squared.",
    ),
    "DiabetesPedigreeFunction": dict(
        label="Diabetes pedigree function", unit="score", step=0.01, default=0.37,
        hard_min=0.05, hard_max=3.0,
        help="A score summarising diabetes history in relatives (higher = stronger family history).",
    ),
    "Age": dict(
        label="Age", unit="years", step=1, default=29,
        hard_min=18, hard_max=100,
        help="Age in years. The dataset only contains adults aged 21+.",
    ),
}

# ----------------------------------------------------------------------------
# Models and colours
# ----------------------------------------------------------------------------
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]

# Brand palette (white background, teal / blue / green accents)
COLORS = {
    "ink": "#0B2530",
    "muted": "#5B7480",
    "teal": "#0E7C86",
    "teal_light": "#14B8A6",
    "blue": "#2457D6",
    "green": "#2EA66A",
    "surface": "#F1F8F7",
    "line": "#DCE8E8",
    "amber": "#B7791F",
    "red": "#C0392B",
    "class0": "#14B8A6",   # no diabetes recorded
    "class1": "#2457D6",   # diabetes recorded
}
MODEL_COLORS = {
    "Logistic Regression": "#0E7C86",
    "Decision Tree": "#2EA66A",
    "Random Forest": "#2457D6",
    "XGBoost": "#6D5BD0",
}
CLASS_LABELS = {0: "No diabetes recorded", 1: "Diabetes recorded"}

DISCLAIMER = (
    "HealthLens AI is an educational and research prototype. It is not a medical "
    "device and does not diagnose, treat or screen for any condition. Its outputs "
    "must never replace advice from a qualified healthcare professional."
)
