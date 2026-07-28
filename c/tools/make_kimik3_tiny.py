#!/usr/bin/env python3
"""Generate a TINY Kimi-K3 model for engine testing.

Creates a miniature K3 snapshot (config.json + safetensors) with MXFP4 expert
weights, matching the real K3 tensor layout but with tiny dimensions so the
engine can run end-to-end in <1 second. Used by tests/test_kimik3.py.

Layout mirrors the real moonshotai/Kimi-K3:
  - text_config.{hidden_size, num_hidden_layers, num_experts, ...}
  - language_model.model.embed_tokens.weight
  - language_model.lm_head.weight
  - language_model.model.norm.weight
  - language_model.model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
  - language_model.model.layers.{i}.self_attn.* (MLA for full_attn_layers, KDA otherwise)
  - language_model.model.layers.{i}.block_sparse_moe.{gate, routed_expert_*, shared_experts.*}
  - language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w{1,2,3}.{weight_packed, weight_scale}

Usage: python tools/make_kimik3_tiny.py [--out c/kimik3_tiny]
"""
import argparse, json, os, struct, sys
from pathlib import Path

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError): pass

try:
    import numpy as np
    from safetensors.numpy import save_file
except ImportError:
    sys.exit("pip install numpy safetensors")

MXFP4_LUT = np.array([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1], dtype=np.float32)


def quantize_mxfp4(w_f32: np.ndarray):
    """Quantize a [O, I] float32 weight to MXFP4 packed [O, I/2] uint8 + scale [O, I/32] uint8.
    group_size = 32. Uses the same E8M0 + two's-complement nibble layout as the real K3.
    For test purposes we use a simple scale = max(|group|)/7 with clamped nibble encoding."""
    O, I = w_f32.shape
    assert I % 32 == 0
    n_groups = I // 32
    w = w_f32.reshape(O, n_groups, 32)
    gmax = np.abs(w).max(axis=2, keepdims=True)
    gmax = np.maximum(gmax, 1e-8)
    # Scale so the group's range fits in [-7, 7]; E8M0 = log2(scale) + 127, clamped to [1, 254]
    scales_f = gmax / 7.0
    e8m0 = np.clip(np.round(np.log2(scales_f) + 127).astype(np.int32), 1, 254).astype(np.uint8)
    actual_scale = np.power(2.0, e8m0.astype(np.float32) - 127.0)
    # Quantize: q = round(w / scale), clip to [-8, 7], map to [0..15] (two's complement)
    q = np.round(w / actual_scale).clip(-8, 7).astype(np.int32)
    nibbles = np.where(q >= 0, q, q + 16).astype(np.uint8)  # 0..7 -> 0..7, -8..-1 -> 8..15
    # Pack two nibbles per byte: byte j = (nibble[2j] << 4) | nibble[2j+1]
    nibbles = nibbles.reshape(O, I // 2, 2)
    packed = (nibbles[:, :, 0] << 4) | nibbles[:, :, 1]
    return packed.astype(np.uint8), e8m0  # [O, I/2], [O, n_groups, 1] -> squeeze


def rand(*shape, scale=0.02):
    rng = np.random.default_rng(42)
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kimik3_tiny")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Tiny K3 dims (must be divisible by group_size=32 and head dims)
    H = 64          # hidden_size (must be divisible by n_heads and 32)
    L = 4            # num_hidden_layers (1 dense + 1 KDA + 1 MLA + 1 KDA)
    n_heads = 4     # attention heads
    q_lora = 32
    kv_lora = 32
    qk_nope = 16
    qk_rope = 8     # must be even (interleaved RoPE)
    v_head = 16
    inter = 64       # dense MLP intermediate (layer 0)
    moe_inter = 32
    routed_hidden = 32  # must be divisible by 32 (MXFP4 group)
    n_experts = 4
    topk = 2
    n_shared = 2
    kda_heads = 4
    kda_hd = 16       # must be divisible sensibly (head dim)
    short_conv = 4
    vocab = 256

    # Layer structure (1-indexed in K3 config, see configuration_kimi_k3.py):
    #   value v in full_attn_layers -> layer v-1 is MLA
    #   value v in kda_layers       -> layer v-1 is KDA
    # Layer 0 = KDA attention + dense MLP (first_k_dense_replace=1)
    # Layer 1 = KDA + MoE
    # Layer 2 = KDA + MoE
    # Layer 3 = MLA + MoE
    full_attn_layers = [4]      # layer 3 is MLA
    kda_layers = [1, 2, 3]      # layers 0, 1, 2 are KDA

    cfg = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "kimi_k3",
        "text_config": {
            "architectures": ["KimiLinearForCausalLM"],
            "model_type": "kimi_linear",
            "hidden_size": H,
            "num_hidden_layers": L,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_heads,
            "intermediate_size": inter,
            "moe_intermediate_size": moe_inter,
            "routed_expert_hidden_size": routed_hidden,
            "num_experts": n_experts,
            "num_experts_per_token": topk,
            "num_shared_experts": n_shared,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "q_lora_rank": q_lora,
            "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": qk_nope,
            "qk_rope_head_dim": qk_rope,
            "v_head_dim": v_head,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "topk_method": "noaux_tc",
            "topk_group": 1,
            "num_expert_group": 1,
            "routed_scaling_factor": 1.0,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "rms_norm_eps": 1e-5,
            "vocab_size": vocab,
            "linear_attn_config": {
                "full_attn_layers": full_attn_layers,
                "kda_layers": kda_layers,
                "head_dim": kda_hd,
                "num_heads": kda_heads,
                "short_conv_kernel_size": short_conv,
                "use_full_rank_gate": True,
            },
        },
        "tie_word_embeddings": False,
        "vision_config": {},
    }
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    tensors = {}
    P = "language_model."
    # Embeddings & head & final norm
    tensors[P + "model.embed_tokens.weight"] = rand(vocab, H)
    tensors[P + "lm_head.weight"] = rand(vocab, H)
    tensors[P + "model.norm.weight"] = np.ones(H, dtype=np.float32)

    for i in range(L):
        lp = f"{P}model.layers.{i}."
        tensors[lp + "input_layernorm.weight"] = np.ones(H, dtype=np.float32)
        tensors[lp + "post_attention_layernorm.weight"] = np.ones(H, dtype=np.float32)

        # K3 uses 1-indexed layer lists: layer i is MLA if (i+1) in full_attn_layers
        is_mla = (i + 1) in full_attn_layers
        is_moe = i >= 1  # layer 0 is dense MLP

        if is_mla:
            # MLA attention tensors
            ap_ = lp + "self_attn."
            tensors[ap_ + "q_a_proj.weight"] = rand(q_lora, H)
            tensors[ap_ + "q_a_layernorm.weight"] = np.ones(q_lora, dtype=np.float32)
            tensors[ap_ + "q_b_proj.weight"] = rand(n_heads * (qk_nope + qk_rope), q_lora)
            tensors[ap_ + "kv_a_proj_with_mqa.weight"] = rand(kv_lora + qk_rope, H)
            tensors[ap_ + "kv_a_layernorm.weight"] = np.ones(kv_lora, dtype=np.float32)
            tensors[ap_ + "kv_b_proj.weight"] = rand(n_heads * (qk_nope + v_head), kv_lora)
            tensors[ap_ + "o_proj.weight"] = rand(H, n_heads * v_head)
            tensors[ap_ + "g_proj.weight"] = rand(n_heads * v_head, H)
        else:
            # KDA attention tensors (ALL non-MLA layers, including layer 0)
            ap_ = lp + "self_attn."
            pqk = kda_heads * kda_hd
            pv = kda_heads * kda_hd
            tensors[ap_ + "q_proj.weight"] = rand(pqk, H)
            tensors[ap_ + "k_proj.weight"] = rand(pqk, H)
            tensors[ap_ + "v_proj.weight"] = rand(pv, H)
            tensors[ap_ + "q_conv1d.weight"] = rand(pqk * short_conv)
            tensors[ap_ + "k_conv1d.weight"] = rand(pqk * short_conv)
            tensors[ap_ + "v_conv1d.weight"] = rand(pv * short_conv)
            tensors[ap_ + "A_log"] = rand(kda_heads)
            tensors[ap_ + "dt_bias"] = rand(kda_heads)
            tensors[ap_ + "b_proj.weight"] = rand(kda_heads, H)
            tensors[ap_ + "f_a_proj.weight"] = rand(pv, H)
            tensors[ap_ + "f_b_proj.weight"] = rand(pv, H)
            tensors[ap_ + "g_proj.weight"] = rand(pv, H)
            tensors[ap_ + "o_norm.weight"] = np.ones(pv, dtype=np.float32)
            tensors[ap_ + "o_proj.weight"] = rand(H, pv)

        if not is_moe:
            # Dense MLP (layer 0)
            tensors[lp + "mlp.gate_proj.weight"] = rand(inter, H)
            tensors[lp + "mlp.up_proj.weight"] = rand(inter, H)
            tensors[lp + "mlp.down_proj.weight"] = rand(H, inter)
        else:
            # MoE layer
            mp = lp + "block_sparse_moe."
            tensors[mp + "gate.weight"] = rand(n_experts, H)
            tensors[mp + "gate.e_score_correction_bias"] = rand(n_experts)
            tensors[mp + "routed_expert_down_proj.weight"] = rand(routed_hidden, H)
            tensors[mp + "routed_expert_up_proj.weight"] = rand(H, routed_hidden)
            tensors[mp + "routed_expert_norm.weight"] = np.ones(routed_hidden, dtype=np.float32)
            # Shared experts (fused gate/up with intermediate = moe_inter * n_shared)
            si = moe_inter * n_shared
            tensors[mp + "shared_experts.gate_proj.weight"] = rand(si, H)
            tensors[mp + "shared_experts.up_proj.weight"] = rand(si, H)
            tensors[mp + "shared_experts.down_proj.weight"] = rand(H, si)
            # Per-expert MXFP4 weights: w1=[moe_inter, routed_hidden], w3=[moe_inter, routed_hidden], w2=[routed_hidden, moe_inter]
            for e in range(n_experts):
                ep = mp + f"experts.{e}."
                g = rand(moe_inter, routed_hidden)
                u = rand(moe_inter, routed_hidden)
                d = rand(routed_hidden, moe_inter)
                gp, gs = quantize_mxfp4(g)
                up_, us = quantize_mxfp4(u)
                dp, ds = quantize_mxfp4(d)
                tensors[ep + "w1.weight_packed"] = gp
                tensors[ep + "w1.weight_scale"] = gs.squeeze(2)
                tensors[ep + "w3.weight_packed"] = up_
                tensors[ep + "w3.weight_scale"] = us.squeeze(2)
                tensors[ep + "w2.weight_packed"] = dp
                tensors[ep + "w2.weight_scale"] = ds.squeeze(2)

    save_file(tensors, str(out / "model.safetensors"))
    print(f"[ok] tiny K3 model written to {out}")
    print(f"     hidden={H} layers={L} (dense=1, KDA=2, MLA=1) experts={n_experts} topk={topk}")

    # ref.json with prompt_ids and full_ids (greedy target = same as prompt + a fixed extension)
    ref = {"prompt_ids": [1, 2, 3, 4], "full_ids": [1, 2, 3, 4, 5, 6]}
    json.dump(ref, open(out / "ref.json", "w"))
    print(f"[ok] ref.json written")


if __name__ == "__main__":
    main()
