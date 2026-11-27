"""Core PyTorch implementation for Experiment E6:
Surgical Neuron DPO (R1/R2) & Localized Deletion Tuning + DPO (R3).

Designed for SwiGLU causal LMs (Llama-3.1-8B / Salamandra-7B / Qwen3) on a single A100:
  - Freezes 100% of the base model (0 bytes of gradients/optimizer state for non-selected params).
  - Attaches lightweight fp32 wrappers ONLY to the layers containing the K selected neurons.
  - Precomputes reference log-probabilities once before training (no second model copy in VRAM).
  - Supports:
      1. SurgicalFloatMLPWrapper: trains fp32 deltas on gate_proj + up_proj + down_proj
         for the exact selected neuron indices (`245,760` params for K=20 in Llama-3.1-8B).
      2. DeletionTuningGateWrapper: learns fp32 gate logits G for the exact selected neuron
         rows of gate_proj (`81,920` params for K=20 in Llama-3.1-8B) with Wanda prior init,
         linear sharpness ramp alpha(t), MaskLLM weight regularizer, and exact binary hard-mask
         deployment `W_gate[ix] *= (G > 0)` (Lu et al., arXiv:2611.10866v1).
  - Supports in-place deployment into `base_model` for native-speed lm-eval and exact
    bit-for-bit restoration between runs.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fingerprinting, backup & exact in-place restoration
# ---------------------------------------------------------------------------

@torch.no_grad()
def mlp_fingerprint(model: nn.Module) -> torch.Tensor:
    """Per-layer float64 sum of gate/up/down weights to detect any unintended weight drift."""
    return torch.stack(
        [
            sum(
                p.double().sum()
                for p in (
                    layer.mlp.gate_proj.weight,
                    layer.mlp.up_proj.weight,
                    layer.mlp.down_proj.weight,
                )
            )
            for layer in model.model.layers
        ]
    ).cpu()


@torch.no_grad()
def backup_neurons(
    model: nn.Module, neuron_indices: dict[int, list[int]]
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Save exact clones of the selected neuron rows/columns for bit-for-bit restore."""
    backup = {}
    for layer_idx, idxs in neuron_indices.items():
        mlp = model.model.layers[int(layer_idx)].mlp
        dev = mlp.down_proj.weight.device
        ix = torch.tensor(idxs, dtype=torch.long, device=dev)
        backup[int(layer_idx)] = (
            ix,
            mlp.gate_proj.weight[ix].detach().clone(),
            mlp.up_proj.weight[ix].detach().clone(),
            mlp.down_proj.weight[:, ix].detach().clone(),
        )
    return backup


@torch.no_grad()
def restore_neurons(
    model: nn.Module,
    backup: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    """Restore backed-up neuron slices in place and assert bit-for-bit equality."""
    for layer_idx, (ix, g, u, d) in backup.items():
        mlp = model.model.layers[layer_idx].mlp
        mlp.gate_proj.weight[ix] = g
        mlp.up_proj.weight[ix] = u
        mlp.down_proj.weight[:, ix] = d
    for layer_idx, (ix, g, u, d) in backup.items():
        mlp = model.model.layers[layer_idx].mlp
        assert torch.equal(mlp.gate_proj.weight[ix], g), f"layer {layer_idx}: gate restore mismatch"
        assert torch.equal(mlp.up_proj.weight[ix], u), f"layer {layer_idx}: up restore mismatch"
        assert torch.equal(mlp.down_proj.weight[:, ix], d), f"layer {layer_idx}: down restore mismatch"


# ---------------------------------------------------------------------------
# Tokenization & DPO log-probability helpers
# ---------------------------------------------------------------------------

def tokenize_dpo_pair(
    tokenizer: Any, prompt: str, chosen: str, rejected: str
) -> dict[str, list[int] | int]:
    """Tokenize (prompt, chosen, rejected) with exact continuation boundaries.

    Handles BPE boundary safety: verifies that `prompt` is a clean prefix of
    `prompt + " " + continuation`.
    """
    prompt_clean = prompt.rstrip()
    chosen_clean = chosen.lstrip()
    rejected_clean = rejected.lstrip()

    p_ids = tokenizer.encode(prompt_clean, add_special_tokens=True)

    def _encode_branch(cont: str) -> tuple[list[int], int]:
        full_ids = tokenizer.encode(f"{prompt_clean} {cont}", add_special_tokens=True)
        # Find exact common prefix length (handles rare BPE merge at boundary safely)
        lcp = 0
        max_p = min(len(p_ids), len(full_ids))
        while lcp < max_p and p_ids[lcp] == full_ids[lcp]:
            lcp += 1
        assert lcp >= max(1, len(p_ids) - 1), (
            f"Unexpected BPE prefix mismatch: lcp={lcp}, len(p_ids)={len(p_ids)}"
        )
        assert len(full_ids) > lcp, f"Empty continuation after tokenization: {cont!r}"
        return full_ids, lcp

    c_ids, c_start = _encode_branch(chosen_clean)
    r_ids, r_start = _encode_branch(rejected_clean)
    return {
        "chosen_ids": c_ids,
        "chosen_start": c_start,
        "rejected_ids": r_ids,
        "rejected_start": r_start,
    }


def prepare_dpo_dataset(tokenizer: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pre-tokenize a list of DPO dicts preserving metadata."""
    out = []
    for r in rows:
        tok = tokenize_dpo_pair(tokenizer, r["prompt"], r["chosen"], r["rejected"])
        out.append(
            {
                "id": r.get("id", ""),
                "domain": r.get("domain", ""),
                "style": r.get("style", ""),
                "block": r.get("block", "long_generative"),
                "target_gender": r.get("target_gender", ""),
                "evidence_alignment": r.get("evidence_alignment", ""),
                **tok,
            }
        )
    return out


def sequence_continuation_logprob(
    model: nn.Module,
    input_ids_list: list[list[int]],
    cont_starts: list[int],
    pad_token_id: int,
    device: torch.device,
    length_normalize: bool = True,
    return_token_logprobs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Compute continuation log-probabilities (summed or length-normalized) for a batch.

    Returns:
      - scores: shape (B,) float32 (mean if length_normalize else sum over continuation tokens)
      - token_counts: shape (B,) float32
      - optionally per-example continuation log-prob distributions for exact token-level KL
    """
    bsz = len(input_ids_list)
    max_len = max(len(x) for x in input_ids_list)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

    for i, seq in enumerate(input_ids_list):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :L] = 1

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()  # predict token t+1 from position t
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(token_lp, dtype=torch.float32)
    for i, (seq, start) in enumerate(zip(input_ids_list, cont_starts)):
        # target index `t` predicts token `t + 1`; continuation tokens are indices [start, len(seq)-1]
        mask[i, start - 1 : len(seq) - 1] = 1.0

    counts = mask.sum(dim=-1).clamp_min(1.0)
    summed = (token_lp * mask).sum(dim=-1)
    scores = summed / counts if length_normalize else summed

    if return_token_logprobs:
        per_ex_dists = []
        for i, (seq, start) in enumerate(zip(input_ids_list, cont_starts)):
            per_ex_dists.append(log_probs[i, start - 1 : len(seq) - 1, :].detach().cpu())
        return scores, counts, per_ex_dists
    return scores, counts


@torch.no_grad()
def precompute_reference_logprobs(
    model: nn.Module,
    dataset: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
    batch_size: int = 8,
    length_normalize: bool = True,
) -> list[dict[str, Any]]:
    """Precompute ref_chosen_lp and ref_rejected_lp once on the frozen base model."""
    model.eval()
    enriched = [dict(x) for x in dataset]
    for start_i in range(0, len(enriched), batch_size):
        batch = enriched[start_i : start_i + batch_size]
        c_lp, c_n = sequence_continuation_logprob(
            model,
            [b["chosen_ids"] for b in batch],
            [b["chosen_start"] for b in batch],
            pad_token_id=pad_token_id,
            device=device,
            length_normalize=length_normalize,
        )
        r_lp, r_n = sequence_continuation_logprob(
            model,
            [b["rejected_ids"] for b in batch],
            [b["rejected_start"] for b in batch],
            pad_token_id=pad_token_id,
            device=device,
            length_normalize=length_normalize,
        )
        for j, b in enumerate(batch):
            b["ref_chosen_lp"] = float(c_lp[j].item())
            b["ref_rejected_lp"] = float(r_lp[j].item())
            b["chosen_n_tok"] = int(c_n[j].item())
            b["rejected_n_tok"] = int(r_n[j].item())
    return enriched


@torch.no_grad()
def evaluate_dpo_validation(
    model: nn.Module,
    val_dataset: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
    beta: float = 0.1,
    batch_size: int = 8,
    length_normalize: bool = True,
    ref_token_logprobs: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Evaluate DPO loss, win-rate, margin, and optional forward KL on held-out val."""
    model.eval()
    margins_all = []
    ref_margins_all = []
    dpo_losses = []
    by_block: dict[str, list[float]] = {}
    by_style: dict[str, list[float]] = {}
    kl_values = []

    idx_global = 0
    for start_i in range(0, len(val_dataset), batch_size):
        batch = val_dataset[start_i : start_i + batch_size]
        if ref_token_logprobs is not None:
            c_lp, _, c_dists = sequence_continuation_logprob(
                model,
                [b["chosen_ids"] for b in batch],
                [b["chosen_start"] for b in batch],
                pad_token_id=pad_token_id,
                device=device,
                length_normalize=length_normalize,
                return_token_logprobs=True,
            )
            for j, log_q in enumerate(c_dists):
                log_p = ref_token_logprobs[idx_global + j]  # base policy log-probs
                # Forward KL(pi_base || pi_tuned) averaged over continuation tokens
                p = log_p.exp()
                kl_tok = (p * (log_p - log_q)).sum(dim=-1).clamp_min(0.0)
                kl_values.append(float(kl_tok.mean().item()))
        else:
            c_lp, _ = sequence_continuation_logprob(
                model,
                [b["chosen_ids"] for b in batch],
                [b["chosen_start"] for b in batch],
                pad_token_id=pad_token_id,
                device=device,
                length_normalize=length_normalize,
            )
        r_lp, _ = sequence_continuation_logprob(
            model,
            [b["rejected_ids"] for b in batch],
            [b["rejected_start"] for b in batch],
            pad_token_id=pad_token_id,
            device=device,
            length_normalize=length_normalize,
        )

        ref_c = torch.tensor([b["ref_chosen_lp"] for b in batch], dtype=torch.float32, device=device)
        ref_r = torch.tensor([b["ref_rejected_lp"] for b in batch], dtype=torch.float32, device=device)
        logits_dpo = beta * ((c_lp - ref_c) - (r_lp - ref_r))
        losses = -F.logsigmoid(logits_dpo)

        for j, b in enumerate(batch):
            m = float((c_lp[j] - r_lp[j]).item())
            rm = float(b["ref_chosen_lp"] - b["ref_rejected_lp"])
            margins_all.append(m)
            ref_margins_all.append(rm)
            dpo_losses.append(float(losses[j].item()))
            by_block.setdefault(b.get("block", "all"), []).append(m)
            by_style.setdefault(b.get("style", "all"), []).append(m)
        idx_global += len(batch)

    def _agg(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(len(arr)),
            "win_rate": float((arr > 0).mean()),
            "mean_margin": float(arr.mean()),
            "median_margin": float(np.median(arr)),
        }

    res: dict[str, Any] = {
        "dpo_loss": float(np.mean(dpo_losses)),
        "overall": _agg(margins_all),
        "margin_gain_vs_base": float(np.mean(margins_all) - np.mean(ref_margins_all)),
        "by_block": {k: _agg(v) for k, v in sorted(by_block.items())},
        "by_style": {k: _agg(v) for k, v in sorted(by_style.items())},
    }
    if kl_values:
        res["forward_kl_mean"] = float(np.mean(kl_values))
        res["forward_kl_max"] = float(np.max(kl_values))
    return res


@torch.no_grad()
def precompute_val_ref_token_logprobs(
    model: nn.Module,
    val_dataset: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
    batch_size: int = 8,
    max_items: int = 128,
) -> list[torch.Tensor]:
    """Store base-model token log-prob distributions on val chosen continuations for exact forward KL."""
    model.eval()
    dists: list[torch.Tensor] = []
    subset = val_dataset[:max_items]
    for start_i in range(0, len(subset), batch_size):
        batch = subset[start_i : start_i + batch_size]
        _, _, batch_dists = sequence_continuation_logprob(
            model,
            [b["chosen_ids"] for b in batch],
            [b["chosen_start"] for b in batch],
            pad_token_id=pad_token_id,
            device=device,
            length_normalize=True,
            return_token_logprobs=True,
        )
        dists.extend(batch_dists)
    return dists


# ---------------------------------------------------------------------------
# Wrapper 1: Surgical Float DPO on selected neurons (R1 / R2)
# ---------------------------------------------------------------------------

class SurgicalFloatMLPWrapper(nn.Module):
    """Wraps a SwiGLU MLP so ONLY `neuron_idxs` receive trainable fp32 deltas.

    Supports any non-empty subset of ("gate_proj", "up_proj", "down_proj").
    The underlying base MLP weights remain completely frozen.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        neuron_idxs: list[int],
        train_projs: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.train_projs = tuple(train_projs)
        self.neuron_idxs_list = [int(i) for i in neuron_idxs]
        dev = base_mlp.down_proj.weight.device
        self.register_buffer("ix", torch.tensor(self.neuron_idxs_list, dtype=torch.long, device=dev))

        k = len(self.neuron_idxs_list)
        hidden_size = base_mlp.gate_proj.in_features

        self.delta_gate = (
            nn.Parameter(torch.zeros((k, hidden_size), dtype=torch.float32, device=dev))
            if "gate_proj" in self.train_projs
            else None
        )
        self.delta_up = (
            nn.Parameter(torch.zeros((k, hidden_size), dtype=torch.float32, device=dev))
            if "up_proj" in self.train_projs
            else None
        )
        self.delta_down = (
            nn.Parameter(torch.zeros((hidden_size, k), dtype=torch.float32, device=dev))
            if "down_proj" in self.train_projs
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.base_mlp.gate_proj(x)
        if self.delta_gate is not None:
            dg = F.linear(x, self.delta_gate.to(x.dtype))
            gate = gate.clone()
            gate.index_add_(-1, self.ix, dg)

        up = self.base_mlp.up_proj(x)
        if self.delta_up is not None:
            du = F.linear(x, self.delta_up.to(x.dtype))
            up = up.clone()
            up.index_add_(-1, self.ix, du)

        h = self.base_mlp.act_fn(gate) * up
        out = self.base_mlp.down_proj(h)
        if self.delta_down is not None:
            h_sel = h.index_select(-1, self.ix)
            out = out + F.linear(h_sel, self.delta_down.to(x.dtype))
        return out

    @torch.no_grad()
    def apply_to_base_in_place(self) -> None:
        """Fold learned fp32 deltas directly into base_mlp weights (for native-speed eval)."""
        if self.delta_gate is not None:
            w = self.base_mlp.gate_proj.weight
            w[self.ix] = (w[self.ix].float() + self.delta_gate).to(w.dtype)
        if self.delta_up is not None:
            w = self.base_mlp.up_proj.weight
            w[self.ix] = (w[self.ix].float() + self.delta_up).to(w.dtype)
        if self.delta_down is not None:
            w = self.base_mlp.down_proj.weight
            w[:, self.ix] = (w[:, self.ix].float() + self.delta_down).to(w.dtype)

    def export_state(self) -> dict[str, Any]:
        return {
            "neuron_idxs": self.neuron_idxs_list,
            "train_projs": list(self.train_projs),
            "delta_gate": None if self.delta_gate is None else self.delta_gate.detach().cpu(),
            "delta_up": None if self.delta_up is None else self.delta_up.detach().cpu(),
            "delta_down": None if self.delta_down is None else self.delta_down.detach().cpu(),
        }


# ---------------------------------------------------------------------------
# Wrapper 2: Localized Deletion Tuning on gate_proj (R3, arXiv:2611.10866v1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_mlp_input_norms(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_indices: list[int],
    device: torch.device,
    batch_size: int = 8,
    max_prompts: int = 512,
) -> dict[int, torch.Tensor]:
    """Collect input activation L2 norms ||X_j||_2 per layer for Wanda prior initialization."""
    model.eval()
    sq_sums: dict[int, torch.Tensor] = {}
    hooks = []

    def _make_hook(l_idx: int):
        def _hook(_mod, inputs):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            s = (x * x).sum(dim=0)
            if l_idx not in sq_sums:
                sq_sums[l_idx] = s
            else:
                sq_sums[l_idx] += s
        return _hook

    for l_idx in layer_indices:
        h = model.model.layers[l_idx].mlp.gate_proj.register_forward_pre_hook(_make_hook(l_idx))
        hooks.append(h)

    try:
        subset = prompts[:max_prompts]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for i in range(0, len(subset), batch_size):
            batch_txt = subset[i : i + batch_size]
            enc = tokenizer(batch_txt, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return {l_idx: torch.sqrt(v.clamp_min(1e-12)) for l_idx, v in sq_sums.items()}


class DeletionTuningGateWrapper(nn.Module):
    """Deletion Tuning (Lu et al., arXiv:2611.10866v1) restricted to selected gate_proj rows.

    During training:  W_eff = W_gate[ix] * sigmoid(alpha * G)
    At evaluation:    W_hat = W_gate[ix] * 1[G > 0]
    Base weights stay frozen; only fp32 gate logits G are updated.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        neuron_idxs: list[int],
        wanda_input_norms: torch.Tensor | None = None,
        eta: float = 1.0 / 2048.0,
        alpha: float = 100.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.neuron_idxs_list = [int(i) for i in neuron_idxs]
        dev = base_mlp.gate_proj.weight.device
        self.register_buffer("ix", torch.tensor(self.neuron_idxs_list, dtype=torch.long, device=dev))
        self.alpha = float(alpha)
        self.use_hard_mask = False

        w_sub = base_mlp.gate_proj.weight[self.ix].detach().float()
        if wanda_input_norms is not None:
            scores = w_sub.abs() * wanda_input_norms.to(dev).float().unsqueeze(0)
        else:
            scores = w_sub.abs()

        # Eq. (8) of arXiv:2611.10866v1: Wanda prior initialization
        flat_scores = scores.reshape(-1)
        thresh = torch.quantile(flat_scores, eta)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(w_sub.shape, generator=gen, dtype=torch.float32).to(dev) * 0.01
        means = torch.where(scores >= thresh, torch.full_like(scores, 0.03), torch.full_like(scores, -0.03))
        self.gate_logits = nn.Parameter(means + noise)

    def current_mask(self) -> torch.Tensor:
        if self.use_hard_mask:
            return (self.gate_logits > 0).to(torch.float32)
        return torch.sigmoid(self.alpha * self.gate_logits)

    def effective_weight_l2_sq(self) -> torch.Tensor:
        """||W_tilde||_2^2 for the MaskLLM sparse weight regularizer (-lambda * ||W_tilde||_2^2)."""
        w_sub = self.base_mlp.gate_proj.weight[self.ix].detach().float()
        w_tilde = w_sub * self.current_mask()
        return (w_tilde * w_tilde).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.current_mask()
        w_sub = self.base_mlp.gate_proj.weight[self.ix].detach().float()
        delta_w = (w_sub * (mask - 1.0)).to(x.dtype)

        gate = self.base_mlp.gate_proj(x)
        dg = F.linear(x, delta_w)
        gate = gate.clone()
        gate.index_add_(-1, self.ix, dg)

        up = self.base_mlp.up_proj(x)
        h = self.base_mlp.act_fn(gate) * up
        return self.base_mlp.down_proj(h)

    @torch.no_grad()
    def deletion_stats(self) -> dict[str, Any]:
        hard = self.gate_logits.detach() > 0
        total = int(hard.numel())
        deleted = int((~hard).sum().item())
        return {
            "total_candidate_weights": total,
            "deleted_weights": deleted,
            "deleted_pct": 100.0 * deleted / max(total, 1),
            "per_neuron_deleted": [(~hard[i]).sum().item() for i in range(hard.shape[0])],
        }

    @torch.no_grad()
    def apply_hard_mask_to_base_in_place(self) -> None:
        """Apply exact binary hard mask 1[G > 0] directly to base_mlp.gate_proj.weight[ix]."""
        w = self.base_mlp.gate_proj.weight
        hard = (self.gate_logits.detach() > 0).to(w.dtype)
        w[self.ix] = w[self.ix] * hard

    def export_state(self) -> dict[str, Any]:
        return {
            "neuron_idxs": self.neuron_idxs_list,
            "gate_logits": self.gate_logits.detach().cpu(),
            "hard_mask": (self.gate_logits.detach() > 0).cpu(),
            "stats": self.deletion_stats(),
        }


# ---------------------------------------------------------------------------
# Unified attach / detach / training loop
# ---------------------------------------------------------------------------

def attach_wrappers(
    model: nn.Module,
    neuron_indices: dict[int, list[int]],
    mode: str,
    train_projs: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
    wanda_norms: dict[int, torch.Tensor] | None = None,
    eta: float = 1.0 / 2048.0,
    alpha_start: float = 100.0,
    seed: int = 42,
) -> dict[int, nn.Module]:
    """Freeze entire model and wrap only the selected layers' MLPs."""
    for p in model.parameters():
        p.requires_grad_(False)

    wrappers: dict[int, nn.Module] = {}
    for offset, (layer_idx, idxs) in enumerate(sorted((int(k), v) for k, v in neuron_indices.items())):
        orig_mlp = model.model.layers[layer_idx].mlp
        if mode == "surgical_float":
            wrap = SurgicalFloatMLPWrapper(orig_mlp, idxs, train_projs=train_projs)
        elif mode == "deletion_tuning":
            norms = None if wanda_norms is None else wanda_norms.get(layer_idx)
            wrap = DeletionTuningGateWrapper(
                orig_mlp, idxs, wanda_input_norms=norms, eta=eta, alpha=alpha_start, seed=seed + offset
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        model.model.layers[layer_idx].mlp = wrap
        wrappers[layer_idx] = wrap
    return wrappers


def detach_wrappers(model: nn.Module, wrappers: dict[int, nn.Module]) -> None:
    """Restore original MLP module objects on wrapped layers."""
    for layer_idx, wrap in wrappers.items():
        model.model.layers[layer_idx].mlp = wrap.base_mlp


@dataclass
class DPOTrainConfig:
    mode: str = "surgical_float"  # "surgical_float" | "deletion_tuning"
    train_projs: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    beta: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 3
    micro_batch_size: int = 4
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    length_normalize: bool = True
    # Deletion Tuning specific (Lu et al., arXiv:2611.10866v1)
    eta: float = 1.0 / 2048.0
    alpha_start: float = 100.0
    alpha_end: float = 1000.0
    lambda_reg: float = 1e-5
    seed: int = 42


def train_localized_dpo(
    model: nn.Module,
    neuron_indices: dict[int, list[int]],
    train_dataset: list[dict[str, Any]],
    val_dataset: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
    cfg: DPOTrainConfig,
    wanda_norms: dict[int, torch.Tensor] | None = None,
    val_ref_token_logprobs: list[torch.Tensor] | None = None,
    verbose: bool = True,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Train localized DPO (R1/R2 float or R3 Deletion Tuning) without modifying base_model weights.

    Returns:
      - adapter_state: per-layer exported weights/masks (on CPU)
      - history: training/validation metrics including soft-vs-hard check for Deletion Tuning
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = random.Random(cfg.seed)

    wrappers = attach_wrappers(
        model=model,
        neuron_indices=neuron_indices,
        mode=cfg.mode,
        train_projs=cfg.train_projs,
        wanda_norms=wanda_norms,
        eta=cfg.eta,
        alpha_start=cfg.alpha_start,
        seed=cfg.seed,
    )

    try:
        trainable_params = [p for w in wrappers.values() for p in w.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        steps_per_epoch = math.ceil(len(train_dataset) / (cfg.micro_batch_size * cfg.grad_accum_steps))
        total_opt_steps = max(1, cfg.epochs * steps_per_epoch)
        warmup_steps = max(1, int(total_opt_steps * cfg.warmup_ratio))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        val_init = evaluate_dpo_validation(
            model,
            val_dataset,
            pad_token_id=pad_token_id,
            device=device,
            beta=cfg.beta,
            batch_size=cfg.micro_batch_size * 2,
            length_normalize=cfg.length_normalize,
            ref_token_logprobs=val_ref_token_logprobs,
        )
        history: dict[str, Any] = {
            "config": asdict(cfg),
            "n_trainable_params": n_trainable,
            "val_init": val_init,
            "epochs": [],
        }
        if verbose:
            print(
                f"[{cfg.mode}] trainable={n_trainable:,} | init val_loss={val_init['dpo_loss']:.4f} "
                f"| win_rate={val_init['overall']['win_rate']:.3f} "
                f"| margin={val_init['overall']['mean_margin']:.4f}"
            )

        opt_step = 0
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            order = list(range(len(train_dataset)))
            rng.shuffle(order)

            optimizer.zero_grad(set_to_none=True)
            epoch_losses = []
            micro_count = 0

            for start_i in range(0, len(order), cfg.micro_batch_size):
                batch = [train_dataset[k] for k in order[start_i : start_i + cfg.micro_batch_size]]

                # Update Deletion Tuning mask sharpening alpha(t)
                if cfg.mode == "deletion_tuning":
                    frac = min(1.0, opt_step / float(max(1, total_opt_steps - 1)))
                    cur_alpha = cfg.alpha_start + frac * (cfg.alpha_end - cfg.alpha_start)
                    for w in wrappers.values():
                        w.alpha = cur_alpha
                        w.use_hard_mask = False

                c_lp, _ = sequence_continuation_logprob(
                    model,
                    [b["chosen_ids"] for b in batch],
                    [b["chosen_start"] for b in batch],
                    pad_token_id=pad_token_id,
                    device=device,
                    length_normalize=cfg.length_normalize,
                )
                r_lp, _ = sequence_continuation_logprob(
                    model,
                    [b["rejected_ids"] for b in batch],
                    [b["rejected_start"] for b in batch],
                    pad_token_id=pad_token_id,
                    device=device,
                    length_normalize=cfg.length_normalize,
                )

                ref_c = torch.tensor([b["ref_chosen_lp"] for b in batch], dtype=torch.float32, device=device)
                ref_r = torch.tensor([b["ref_rejected_lp"] for b in batch], dtype=torch.float32, device=device)
                dpo_logits = cfg.beta * ((c_lp - ref_c) - (r_lp - ref_r))
                loss = -F.logsigmoid(dpo_logits).mean()

                if cfg.mode == "deletion_tuning" and cfg.lambda_reg > 0:
                    reg = sum(w.effective_weight_l2_sq() for w in wrappers.values())
                    loss = loss - cfg.lambda_reg * reg

                epoch_losses.append(float(loss.item()))
                (loss / cfg.grad_accum_steps).backward()
                micro_count += 1

                is_last = (start_i + cfg.micro_batch_size) >= len(order)
                if micro_count % cfg.grad_accum_steps == 0 or is_last:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

            # Evaluate end of epoch (hard mask for Deletion Tuning, exact for float)
            if cfg.mode == "deletion_tuning":
                for w in wrappers.values():
                    w.use_hard_mask = True
            val_ep = evaluate_dpo_validation(
                model,
                val_dataset,
                pad_token_id=pad_token_id,
                device=device,
                beta=cfg.beta,
                batch_size=cfg.micro_batch_size * 2,
                length_normalize=cfg.length_normalize,
                ref_token_logprobs=val_ref_token_logprobs,
            )
            ep_rec: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val": val_ep,
            }
            if cfg.mode == "deletion_tuning":
                tot_w = sum(w.deletion_stats()["total_candidate_weights"] for w in wrappers.values())
                del_w = sum(w.deletion_stats()["deleted_weights"] for w in wrappers.values())
                ep_rec["deleted_weights"] = del_w
                ep_rec["deleted_pct"] = 100.0 * del_w / max(tot_w, 1)
                # Also check soft mask agreement at final epoch
                if epoch == cfg.epochs:
                    for w in wrappers.values():
                        w.use_hard_mask = False
                    val_soft = evaluate_dpo_validation(
                        model,
                        val_dataset,
                        pad_token_id=pad_token_id,
                        device=device,
                        beta=cfg.beta,
                        batch_size=cfg.micro_batch_size * 2,
                        length_normalize=cfg.length_normalize,
                        ref_token_logprobs=val_ref_token_logprobs,
                    )
                    ep_rec["val_soft_final"] = val_soft
                    for w in wrappers.values():
                        w.use_hard_mask = True

            history["epochs"].append(ep_rec)
            if verbose:
                extra = ""
                if "deleted_pct" in ep_rec:
                    extra = f" | deleted={ep_rec['deleted_weights']}/{tot_w} ({ep_rec['deleted_pct']:.2f}%)"
                kl_s = f" | fwd_KL={val_ep['forward_kl_mean']:.4f}" if "forward_kl_mean" in val_ep else ""
                print(
                    f"  epoch {epoch}/{cfg.epochs} | train_loss={ep_rec['train_loss']:.4f} "
                    f"| val_win={val_ep['overall']['win_rate']:.3f} "
                    f"| val_margin={val_ep['overall']['mean_margin']:.4f} "
                    f"(Δ={val_ep['margin_gain_vs_base']:+.4f}){kl_s}{extra}"
                )

        adapter_state = {layer_idx: wrap.export_state() for layer_idx, wrap in wrappers.items()}
        history["val_final"] = history["epochs"][-1]["val"]
        return adapter_state, history
    finally:
        detach_wrappers(model, wrappers)


@torch.no_grad()
def apply_adapter_in_place(
    model: nn.Module,
    adapter_state: dict[int, dict[str, Any]],
    mode: str,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Back up affected neurons and apply exported adapter state directly into base_model."""
    neuron_indices = {int(l): [int(i) for i in st["neuron_idxs"]] for l, st in adapter_state.items()}
    backup = backup_neurons(model, neuron_indices)
    for layer_idx, st in adapter_state.items():
        mlp = model.model.layers[int(layer_idx)].mlp
        dev = mlp.down_proj.weight.device
        ix = torch.tensor(st["neuron_idxs"], dtype=torch.long, device=dev)
        if mode == "surgical_float":
            if st.get("delta_gate") is not None:
                w = mlp.gate_proj.weight
                w[ix] = (w[ix].float() + st["delta_gate"].to(dev)).to(w.dtype)
            if st.get("delta_up") is not None:
                w = mlp.up_proj.weight
                w[ix] = (w[ix].float() + st["delta_up"].to(dev)).to(w.dtype)
            if st.get("delta_down") is not None:
                w = mlp.down_proj.weight
                w[:, ix] = (w[:, ix].float() + st["delta_down"].to(dev)).to(w.dtype)
        elif mode == "deletion_tuning":
            w = mlp.gate_proj.weight
            hard = st["hard_mask"].to(dev).to(w.dtype)
            w[ix] = w[ix] * hard
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    return backup
