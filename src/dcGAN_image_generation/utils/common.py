import os
import yaml
import json
import joblib
import torch

from box.exceptions import BoxValueError
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import Path
from typing import Any

from dcGAN_image_generation import logger


# =========================
# YAML
# =========================

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)

            if content is None:
                raise ValueError("yaml file is empty")

            logger.info(f"yaml file loaded: {path_to_yaml}")
            return ConfigBox(content)

    except Exception as e:
        raise e


# =========================
# DIRECTORIES
# =========================

@ensure_annotations
def create_directories(paths: list, verbose=True):
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


# =========================
# JSON
# =========================

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json loaded from: {path}")
    return ConfigBox(content)


# =========================
# JOBLIB (for scalers, etc.)
# =========================

@ensure_annotations
def save_bin(data: Any, path: Path):
    joblib.dump(value=data, filename=path)
    logger.info(f"binary saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logger.info(f"binary loaded from: {path}")
    return data


# =========================
# TORCH MODEL SAVE / LOAD
# =========================

def save_model(model: torch.nn.Module, path: Path):
    torch.save(model.state_dict(), path)
    logger.info(f"model saved at: {path}")


def load_model(model: torch.nn.Module, path: Path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    logger.info(f"model loaded from: {path}")
    return model


# =========================
# FILE SIZE
# =========================

@ensure_annotations
def get_size(path: Path) -> str:
    total_size = 0

    if path.is_file():
        total_size = path.stat().st_size

    elif path.is_dir():
        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size

    size_in_kb = total_size / 1024

    if size_in_kb < 1024:
        return f"~ {round(size_in_kb, 2)} KB"
    else:
        size_in_mb = size_in_kb / 1024
        return f"~ {round(size_in_mb, 2)} MB"