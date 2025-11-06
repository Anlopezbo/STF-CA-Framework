from __future__ import annotations
import os
import gc
import numpy as np
from typing import Tuple
import tensorflow as tf


def load_subject_artifacts(
    base_dir: str,
    subject_id: int,
    split: str,  # 'train' | 'test'
    subject_dir_template: str = "subject_{id}_results_0_7",
    model_filename_template: str = "model_subject_{id}.h5",
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """
    Load the model and the requested split from a subject directory.

    Default expected structure:
      {base_dir}/subject_{id}_results_0_7/
        - model_subject_{id}.h5
        - train_data.npz (X_train, y_train)
        - test_data.npz  (X_test,  y_test)

    You can adapt folder/file patterns via the templates if your layout differs.
    """
    subject_dirname = subject_dir_template.format(id=subject_id)
    subject_dir = os.path.join(base_dir, subject_dirname)

    model_filename = model_filename_template.format(id=subject_id)
    model_path = os.path.join(subject_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    npz_name = "train_data.npz" if split == "train" else "test_data.npz"
    npz_path = os.path.join(subject_dir, npz_name)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_name} not found for subject {subject_id} in {subject_dir}")

    data_npz = np.load(npz_path)
    X = data_npz["X_train"] if split == "train" else data_npz["X_test"]
    y = data_npz["y_train"] if split == "train" else data_npz["y_test"]

    del data_npz
    gc.collect()
    return model, X, y


def save_numpy(path: str, array: np.ndarray, allow_pickle: bool = False) -> None:
    """
    Save an array ensuring the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array, allow_pickle=allow_pickle)