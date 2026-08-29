import os
from pathlib import Path

# Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Signal Processing Constants
BW_HZ = 2.0
N_HARMONICS = 6

# Feature Names
FEATURE_NAMES = [
    "Mean", "RMS", "Std", "PtP", "Kurtosis", "Skewness",
    "CrestFactor", "ShapeFactor", "ImpulseFactor",
    "DominantFreq", "SpectralCentroid", "SpectralBW", "HarmonicRatio"
]

# Random Seed for Reproducibility
SEED = 42
