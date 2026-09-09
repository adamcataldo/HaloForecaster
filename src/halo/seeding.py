from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch

DEFAULT_SEED = 2021

SEED_IDENTITY_KEY = "seed_identity"

TORCH_NUM_THREADS = 1


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(TORCH_NUM_THREADS)


def derive_seed(seed: int, *parts: object) -> int:
    key = "|".join([str(seed), *(str(p) for p in parts)]).encode()
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "big")


def seed_identity_of(config: dict[str, Any]) -> str:
    declared = config.get(SEED_IDENTITY_KEY)
    if declared is None:
        return str(config["model"])
    if not isinstance(declared, str) or not declared:
        raise ValueError(
            f"config {config.get('dataset_name')}/S={config.get('pred_len')} has "
            f"{SEED_IDENTITY_KEY}={declared!r}; expected a non-empty string"
        )
    return declared


def torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
