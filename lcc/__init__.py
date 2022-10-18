from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = ROOT_DIR.joinpath('dataset')
OUTPUT_DIR = ROOT_DIR.joinpath('output')
MODEL_DIR = ROOT_DIR.joinpath('model')