"""Frozen Qwen3-VL neuron probe runtime.

References: Precise Shield ?3.2 for gated-MLP activations and downstream
influence; When2Tool for StandardScaler + L2 logistic-regression probing.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Qwen3VLMLPCollector:
    """Read h=act(gate_proj(x))*up_proj(x) at the final decision token."""
    def __init__(self, model):
        self.layer_names, self.modules = [], []
        for name, module in model.named_modules():
            if (name.endswith(".mlp") and all(hasattr(module, x) for x in
                    ("gate_proj", "up_proj", "down_proj", "act_fn"))):
                self.layer_names.append(name)
                self.modules.append(module)
        if not self.modules:
            raise RuntimeError("No Qwen3-VL text MLP modules were found.")
        self.values: List[torch.Tensor] = []
        self.handles = [m.register_forward_pre_hook(self._hook) for m in self.modules]

    def _hook(self, module, inputs):
        # Qwen MLP is position-wise.  The probe consumes h only at the final
        # assistant-prefix token, so evaluating gate/up on that token alone is
        # mathematically identical to taking h[:, -1, :] after a full sequence
        # MLP evaluation while avoiding an extra [sequence, intermediate] tensor.
        x = inputs[0][:, -1:, :]
        h = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
        self.values.append(h[0, 0, :].detach().float().cpu())

    def collect(self, model, input_embeddings, attention_mask):
        self.values.clear()
        with torch.no_grad():
            # Probe extraction needs text-MLP activations only. Calling the base
            # model avoids allocating the full [sequence, vocabulary] logits tensor.
            output = model.model(inputs_embeds=input_embeddings,
                                 attention_mask=attention_mask,
                                 use_cache=False, return_dict=True)
        del output
        if len(self.values) != len(self.modules):
            raise RuntimeError(f"Expected {len(self.modules)} MLP activations, got {len(self.values)}")
        return torch.stack(self.values)

    def down_projection_norms(self):
        return torch.stack([m.down_proj.weight.detach().float().cpu().norm(p=2, dim=0)
                            for m in self.modules])

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

@dataclass
class FrozenProbe:
    classifier: LogisticRegression
    scaler: StandardScaler
    neuron_indices: torch.Tensor
    layer_names: Sequence[str]
    reg: float

    def probability(self, activations):
        if activations.ndim != 2:
            raise ValueError(f"Expected [layers, intermediate], got {activations.shape}")
        if activations.shape[0] != self.neuron_indices.shape[0]:
            raise ValueError("Probe layer count and collected activation count differ.")
        x = torch.stack([activations[l, idx] for l, idx in enumerate(self.neuron_indices)]).reshape(1, -1).numpy()
        x = self.scaler.transform(x)
        logit = x @ self.classifier.coef_[0] + self.classifier.intercept_[0]
        return float(1.0 / (1.0 + np.exp(-logit))[0])

def load_frozen_probe(probe_path: str | Path, neuron_map_path: str | Path):
    """Reconstruct the serialized When2Tool scaler/classifier exactly."""
    p = torch.load(probe_path, map_location="cpu", weights_only=True)
    n = torch.load(neuron_map_path, map_location="cpu", weights_only=True)
    scaler = StandardScaler()
    scaler.mean_ = p["scaler_mean"].numpy()
    scaler.scale_ = p["scaler_scale"].numpy()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    classifier = LogisticRegression()
    classifier.coef_ = p["coef"].numpy().reshape(1, -1)
    classifier.intercept_ = np.array([p["intercept"]])
    classifier.classes_ = np.array([0, 1])
    indices = n["indices"].long()
    if classifier.coef_.shape[1] != indices.numel():
        raise ValueError("Probe feature count does not match neuron map.")
    return FrozenProbe(classifier, scaler, indices, n["layer_names"], float(p["reg"]))
