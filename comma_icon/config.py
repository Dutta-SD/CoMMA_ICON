# Configuration file settings
from pathlib import Path
import os
import tensorflow as tf
import numpy as np
import random

# PATH for various directories -- Prefer not to modify
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "input"
CODE_DIR = BASE_DIR / "comma_icon"
FILE_EXTENSION = "tsv"
OUTPUT_DIR = BASE_DIR / "outputs"
SUBMIT_DIR = BASE_DIR / "submission"

## Subdirectories for easier use
INPUT_ALL_DIR = {}


INPUT_ALL_DIR["multi"] = {
    "name": "multi",
    "train": INPUT_DIR / "multi_data/train_multi.tsv",
    "dev": INPUT_DIR / "multi_data/dev_multi.tsv",
    "test": INPUT_DIR / "multi_data/test_multi.tsv",
}
INPUT_ALL_DIR["ben"] = {
    "name": "ben",
    "train": INPUT_DIR / "ben_data/train_ben.tsv",
    "dev": INPUT_DIR / "ben_data/dev_ben.tsv",
    "test": INPUT_DIR / "ben_data/test_ben.tsv",
}
INPUT_ALL_DIR["hin"] = {
    "name": "hin",
    "train": INPUT_DIR / "hin_data/train_hin.tsv",
    "dev": INPUT_DIR / "hin_data/dev_hin.tsv",
    "test": INPUT_DIR / "hin_data/test_hin.tsv",
}
INPUT_ALL_DIR["mni"] = {
    "name": "mni",
    "train": INPUT_DIR / "mni_data/train_mni.tsv",
    "dev": INPUT_DIR / "mni_data/dev_mni.tsv",
    "test": INPUT_DIR / "mni_data/test_mni.tsv",
}

# Label map
TASK_A_MAP = {
    "NAG": 2,
    "CAG": 1,
    "OAG": 0,
}
TASK_B_MAP = {
    "NGEN": 1,
    "GEN": 0,
}
TASK_C_MAP = {
    "NCOM": 1,
    "COM": 0,
}

# Training Hyper-parameters -- Change
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 64
NUM_TRAINING_EPOCHS = 12
RNG_SEED = 2833
VOCAB_SIZE = 85000
MAX_SEQ_LEN = 256

# Set seed
def set_seed():
    tf.random.set_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)