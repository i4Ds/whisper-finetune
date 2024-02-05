from typing import Dict

import torch
from whisper import Whisper as WhisperModel


def get_optimizer(model: WhisperModel, optimizer_conf: Dict):
    if optimizer_conf["type"] == "adam":
        if optimizer_conf["8bit"]:
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.Adam8bit(model.parameters(), **optimizer_conf["params"])
            except ImportError:
                raise ImportError("For using Adam 8bit optimizer you need to have bitsandbytes installed.")
        else:
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_conf["params"])

    elif optimizer_conf["type"] == "adamw":
        if optimizer_conf["8bit"]:
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(model.parameters(), **optimizer_conf["params"])
            except ImportError:
                raise ImportError("For using AdamW 8bit optimizer you need to have bitsandbytes installed.")
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_conf["params"])

    return optimizer
