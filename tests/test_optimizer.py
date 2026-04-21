import torch
import pytest

from whisper_finetune.model.optimizer import get_optimizer


class _FakeBlockStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])


class _FakeWhisper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeBlockStack()
        self.decoder = _FakeBlockStack()
        self.embedding = torch.nn.Embedding(8, 4)
        self.final_norm = torch.nn.LayerNorm(4)


def test_get_optimizer_muon_uses_only_supported_param_group_keys():
    pytest.importorskip("muon")

    model = _FakeWhisper()
    optimizer_conf = {
        "type": "adamw",
        "muon": True,
        "8bit": False,
        "muon_ndim_threshold": 2,
        "muon_params": {
            "lr": 2e-4,
            "momentum": 0.95,
            "weight_decay": 0.01,
        },
        "params": {
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.98],
            "eps": 1e-6,
            "amsgrad": False,
        },
    }

    optimizer = get_optimizer(model, optimizer_conf)

    assert len(optimizer.param_groups) >= 2
    assert len(optimizer._lr_group_metadata) == len(optimizer.param_groups)

    for idx, group in enumerate(optimizer.param_groups):
        if group["use_muon"]:
            assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            assert optimizer._lr_group_metadata[idx]["lr_log_label"] == "muon"
            assert optimizer._lr_group_metadata[idx]["base_lr_unscaled"] == optimizer_conf["muon_params"]["lr"]
        else:
            assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
            assert optimizer._lr_group_metadata[idx]["lr_log_label"] == "aux_adamw"
            assert optimizer._lr_group_metadata[idx]["base_lr_unscaled"] == optimizer_conf["params"]["lr"]
