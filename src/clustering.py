# Prepare the Embedding Space
# Imports

from pathlib import Path
import json

import numpy as np
import pandas as pd

# Define Project Root
# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"

EMBEDDING_DIR = DATA_DIR / "embeddings"
CLUSTERING_DIR = DATA_DIR / "clustering"

FIGURE_DIR = DATA_DIR / "figures"

# Load Files
embeddings = np.load( EMBEDDING_DIR / "embeddings.npy")

metadata = pd.read_csv(EMBEDDING_DIR / "metadata.csv")

with open(
    EMBEDDING_DIR / "embedding_config.json",
    "r",
    encoding="utf-8"
) as f:
    config = json.load(f)

# Consistency Check
assert embeddings.shape[0] == len(metadata)
assert embeddings.shape[1] == config["dimension"]
assert len(metadata) == config["num_samples"]

print("✅ Embedding artifacts are consistent.")

# Create Output Folder
CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)




