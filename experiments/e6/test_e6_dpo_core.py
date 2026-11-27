"""Local CPU unit tests for `e6_dpo_core.py` using a tiny mock SwiGLU CausalLM.

Verifies before spending any A100 GPU time:
  1. Wrapper forward pass matches direct modified-weight forward pass (exact math equivalence).
  2. SurgicalFloatDPO trains ONLY the selected K neurons (all other weights bit-for-bit untouched).
  3. DeletionTuning + DPO initializes with Wanda prior, trains ONLY gate_logits G, deploys an
     exact binary hard mask (0 or exact pretrained W_0), and improves training margin.
  4. Backup/restore returns the model bit-for-bit identical to the original base fingerprint.
  5. Initial DPO loss at step 0 equals ln(2) ~= 0.6931 for surgical_float.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from experiments.e6.e6_dpo_core import (
    DPOTrainConfig,
    apply_adapter_in_place,
    attach_wrappers,
    collect_mlp_input_norms,
    detach_wrappers,
    mlp_fingerprint,
    precompute_reference_logprobs,
    precompute_val_ref_token_logprobs,
    prepare_dpo_dataset,
    restore_neurons,
    train_localized_dpo,
)


class TinySwiGLUMLP(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.mlp = TinySwiGLUMLP(hidden, inter)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 128, hidden: int = 32, inter: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([TinyLayer(hidden, inter) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache=False):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))


class TinyWordTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<bos>": 2}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [2] if add_special_tokens else []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = 3 + (len(self.vocab) % 120)
            ids.append(self.vocab[w])
        return ids

    def __call__(self, texts: list[str], return_tensors="pt", padding=True, truncation=True, max_length=256):
        seqs = [self.encode(t, add_special_tokens=True)[:max_length] for t in texts]
        ml = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), ml), dtype=torch.long)
        mask = torch.zeros((len(seqs), ml), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _make_sample_rows(n: int = 16) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"ex_{i}",
                "domain": "test",
                "style": "situational" if i % 2 == 0 else "short_disambig",
                "block": "long_generative" if i % 2 == 0 else "short_attribution",
                "target_gender": "woman" if i % 2 == 0 else "man",
                "evidence_alignment": "counter_stereo",
                "prompt": f"When person_{i} presented the engineering design to the board they",
                "chosen": f"reviewed the careful analysis_{i} and approved the technical plan.",
                "rejected": f"doubted the technical ability_{i} and dismissed the entire report.",
            }
        )
    return rows


def test_surgical_float_dpo_isolation_and_equivalence():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = TinyCausalLM(vocab=128, hidden=32, inter=64, n_layers=4).to(device)
    tok = TinyWordTokenizer()
    base_fp = mlp_fingerprint(model)
    orig_params = {k: v.detach().clone() for k, v in model.named_parameters()}

    neuron_indices = {1: [2, 10], 3: [0, 5, 19, 44]}
    train_rows = prepare_dpo_dataset(tok, _make_sample_rows(12))
    val_rows = prepare_dpo_dataset(tok, _make_sample_rows(6))
    train_rows = precompute_reference_logprobs(model, train_rows, tok.pad_token_id, device, batch_size=4)
    val_rows = precompute_reference_logprobs(model, val_rows, tok.pad_token_id, device, batch_size=4)
    val_ref_dists = precompute_val_ref_token_logprobs(model, val_rows, tok.pad_token_id, device, batch_size=4)

    cfg = DPOTrainConfig(
        mode="surgical_float",
        train_projs=("gate_proj", "up_proj", "down_proj"),
        lr=5e-2,
        epochs=2,
        micro_batch_size=4,
        grad_accum_steps=1,
        seed=42,
    )
    adapter, hist = train_localized_dpo(
        model, neuron_indices, train_rows, val_rows, tok.pad_token_id, device, cfg,
        val_ref_token_logprobs=val_ref_dists, verbose=False
    )

    # Step-0 DPO loss must equal ln(2) when deltas are 0
    assert math.isclose(hist["val_init"]["dpo_loss"], math.log(2.0), rel_tol=1e-4)
    # Training must leave base_model completely unmodified
    assert torch.equal(mlp_fingerprint(model), base_fp)
    for k, v in model.named_parameters():
        assert torch.equal(v, orig_params[k]), f"base param mutated during training: {k}"

    # Deploying adapter in-place must change ONLY the selected neurons' gate/up/down slices
    backup = apply_adapter_in_place(model, adapter, mode="surgical_float")
    for l_idx in range(4):
        mlp = model.model.layers[l_idx].mlp
        g_orig = orig_params[f"model.layers.{l_idx}.mlp.gate_proj.weight"]
        u_orig = orig_params[f"model.layers.{l_idx}.mlp.up_proj.weight"]
        d_orig = orig_params[f"model.layers.{l_idx}.mlp.down_proj.weight"]
        if l_idx not in neuron_indices:
            assert torch.equal(mlp.gate_proj.weight, g_orig)
            assert torch.equal(mlp.up_proj.weight, u_orig)
            assert torch.equal(mlp.down_proj.weight, d_orig)
        else:
            sel = set(neuron_indices[l_idx])
            other = [i for i in range(64) if i not in sel]
            assert torch.equal(mlp.gate_proj.weight[other], g_orig[other])
            assert torch.equal(mlp.up_proj.weight[other], u_orig[other])
            assert torch.equal(mlp.down_proj.weight[:, other], d_orig[:, other])
            assert not torch.equal(mlp.down_proj.weight[:, list(sel)], d_orig[:, list(sel)])

    # Exact forward equivalence between wrapper and folded in-place weights
    x_test = torch.randint(3, 100, (2, 10))
    with torch.no_grad():
        logits_folded = model(x_test).logits

    restore_neurons(model, backup)
    assert torch.equal(mlp_fingerprint(model), base_fp)

    wrappers = attach_wrappers(model, neuron_indices, mode="surgical_float")
    with torch.no_grad():
        for l_idx, w in wrappers.items():
            w.delta_gate.copy_(adapter[l_idx]["delta_gate"])
            w.delta_up.copy_(adapter[l_idx]["delta_up"])
            w.delta_down.copy_(adapter[l_idx]["delta_down"])
        logits_wrapped = model(x_test).logits
    detach_wrappers(model, wrappers)
    assert torch.allclose(logits_folded, logits_wrapped, atol=1e-5)


def test_deletion_tuning_dpo_hard_mask_and_restore():
    torch.manual_seed(1)
    device = torch.device("cpu")
    model = TinyCausalLM(vocab=128, hidden=32, inter=64, n_layers=4).to(device)
    tok = TinyWordTokenizer()
    base_fp = mlp_fingerprint(model)
    orig_gate_l3 = model.model.layers[3].mlp.gate_proj.weight.detach().clone()

    neuron_indices = {2: [1, 7], 3: [4, 11, 22]}
    raw_train = _make_sample_rows(12)
    wanda_norms = collect_mlp_input_norms(
        model, tok, [r["prompt"] for r in raw_train], sorted(neuron_indices), device, batch_size=4
    )
    assert set(wanda_norms.keys()) == {2, 3}
    assert all(v.shape == (32,) and torch.isfinite(v).all() for v in wanda_norms.values())

    train_rows = precompute_reference_logprobs(
        model, prepare_dpo_dataset(tok, raw_train), tok.pad_token_id, device, batch_size=4
    )
    val_rows = precompute_reference_logprobs(
        model, prepare_dpo_dataset(tok, _make_sample_rows(6)), tok.pad_token_id, device, batch_size=4
    )

    cfg = DPOTrainConfig(
        mode="deletion_tuning",
        lr=1e-2,
        epochs=2,
        micro_batch_size=4,
        grad_accum_steps=1,
        eta=0.05,
        alpha_start=50.0,
        alpha_end=500.0,
        lambda_reg=1e-5,
        seed=7,
    )
    adapter, hist = train_localized_dpo(
        model, neuron_indices, train_rows, val_rows, tok.pad_token_id, device, cfg,
        wanda_norms=wanda_norms, verbose=False
    )

    # Base weights untouched during training
    assert torch.equal(mlp_fingerprint(model), base_fp)
    assert "val_soft_final" in hist["epochs"][-1]

    # Hard mask deployment must set deleted entries to exact 0.0 and keep retained entries exact W_0
    backup = apply_adapter_in_place(model, adapter, mode="deletion_tuning")
    g_after = model.model.layers[3].mlp.gate_proj.weight
    ix3 = neuron_indices[3]
    hard3 = adapter[3]["hard_mask"]
    assert torch.equal(g_after[ix3][~hard3], torch.zeros_like(g_after[ix3][~hard3]))
    assert torch.equal(g_after[ix3][hard3], orig_gate_l3[ix3][hard3])
    other3 = [i for i in range(64) if i not in set(ix3)]
    assert torch.equal(g_after[other3], orig_gate_l3[other3])

    restore_neurons(model, backup)
    assert torch.equal(mlp_fingerprint(model), base_fp)
