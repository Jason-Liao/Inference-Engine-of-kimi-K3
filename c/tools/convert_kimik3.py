#!/usr/bin/env python3
"""Convert Kimi-K3 HuggingFace checkpoint (MXFP4 experts) -> colibri int8 format.

Kimi-K3 ships routed experts in MXFP4 (4-bit, group_size=32, E8M0 microscale):
    language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w{1,2,3}.weight_packed
    language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w{1,2,3}.weight_scale

This tool dequantizes each expert to fp32, then re-quantizes row-wise to int8
(colibri layout) and writes the two tensors the engine expects:
    language_model.model.layers.{i}.block_sparse_moe.experts.{e}.merged_weight  (int8, [qg|qu|qd])
    language_model.model.layers.{i}.block_sparse_moe.experts.{e}.qs            (f32,  [qgs|qus|qds])

Dense weights (attention, RMSNorm, embed, lm_head, shared experts, layer-0 MLP)
are passed through unchanged: the engine reads them as bf16 -> f32 on load.

The MXFP4 nibble LUT matches kimik3.c (two's complement of 4 bits):
    nibble 0..7  ->  0..7
    nibble 8..15 -> -8..-1
E8M0 microscale: scale = 2**(uint8 - 127); 0 means zero (denormal flush).

Usage:
  python tools/convert_kimik3.py --model ./Kimi-K3 --out ./k3_i8
  python tools/convert_kimik3.py --repo moonshotai/Kimi-K3   --out ./k3_i8
"""

import argparse, json, math, os, re, struct, sys
from pathlib import Path

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError): pass

try:
    import numpy as np
    from safetensors.numpy import load_file, save_file
    from safetensors import safe_open
except ImportError as exc:
    sys.exit(f"Missing dependencies: {exc}. Install: pip install numpy safetensors")

# MXFP4 nibble lookup table (two's complement 4-bit). Matches kimik3.c mxfp4_lut.
MXFP4_LUT = np.array([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1],
                     dtype=np.float32)

# Regex matching K3 routed-expert packed tensors. Captures (layer, expert, w_idx).
EXPERT_PACKED_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w([123])\.weight_packed$"
)


def e8m0_to_scale(scale_u8: np.ndarray) -> np.ndarray:
    """E8M0 (8-bit exponent, bias 127, no mantissa) -> float scale.
    0x00 -> 0 (flushed denormal); otherwise scale = 2**(u8 - 127)."""
    s = np.zeros_like(scale_u8, dtype=np.float32)
    nz = scale_u8 != 0
    s[nz] = np.power(2.0, scale_u8[nz].astype(np.float32) - 127.0)
    return s


def mxfp4_dequant(packed: np.ndarray, scales: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Dequantize MXFP4 packed weights to fp32.

    packed:  [rows, cols/2] uint8  (two 4-bit nibbles per byte)
    scales:  [rows, cols/32] uint8 (one E8M0 per 32-element group)
    returns: [rows, cols] float32
    """
    assert cols % 32 == 0, f"cols {cols} not divisible by group_size 32"
    n_groups = cols // 32
    # Unpack nibbles: high nibble first, low nibble second.
    hi = (packed.astype(np.uint8) >> 4) & 0x0F           # [rows, cols/2]
    lo = (packed.astype(np.uint8)      ) & 0x0F          # [rows, cols/2]
    # Interleave: row is [h0,l0,h1,l1,...] -> [h0,h1,...,l_k]
    # Actually MXFP4 packs two consecutive elements per byte: byte j holds element 2j (hi) and 2j+1 (lo).
    interleaved = np.empty((rows, cols), dtype=np.float32)
    interleaved[:, 0::2] = MXFP4_LUT[hi]
    interleaved[:, 1::2] = MXFP4_LUT[lo]
    # Apply per-group scale. Broadcast scales to [rows, n_groups, 32].
    sc = e8m0_to_scale(scales).reshape(rows, n_groups, 1)
    interleaved = interleaved.reshape(rows, n_groups, 32) * sc
    return interleaved.reshape(rows, cols)


def quantize_row_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise symmetric int8 quantization. Mirrors kimik3.c matmul_q usage.

    Returns (int8_weights [O,I], float32_scales [O]).
        qmax  = 127
        scale = amax(|w|, row) / qmax
        q     = clamp(round(w / scale), -128, 127)
    """
    qmax = 127
    row_max = np.abs(w).max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, 1e-8)  # avoid div-by-zero
    scales = (row_max / qmax).astype(np.float32)
    q = np.round(w / scales).clip(-128, 127).astype(np.int8)
    return q, scales.squeeze(1)


def main():
    ap = argparse.ArgumentParser(description="Convert Kimi-K3 (MXFP4) -> colibri int8 experts")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HuggingFace repo ID (e.g. moonshotai/Kimi-K3)")
    src.add_argument("--model", help="Local HF checkpoint directory")
    ap.add_argument("--out", required=True, help="Output directory for int8 model")
    ap.add_argument("--bits", type=int, default=8, help="Expert quant bits (default 8; engine stores int8)")
    args = ap.parse_args()

    if args.bits != 8:
        print(f"[info] --bits={args.bits}: storage is int8, qmax will be 2^(bits-1)-1", file=sys.stderr)

    if args.repo:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.repo}...")
        src_dir = snapshot_download(args.repo, max_workers=4)
    else:
        src_dir = args.model

    src = Path(src_dir)
    if not src.is_dir():
        sys.exit(f"Model directory not found: {src}")
    if not (src / "config.json").is_file():
        sys.exit(f"config.json missing in {src}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(src / "config.json", out / "config.json")
    # Tokenizer files (engine needs tokenizer.json for O200K)
    for tf in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (src / tf).is_file():
            shutil.copy2(src / tf, out / tf)
    print(f"config.json + tokenizer -> {out}")

    # Read dims we need to validate expert shapes.
    cfg = json.load(open(src / "config.json"))
    tc = cfg.get("text_config", cfg)
    moe_inter = tc.get("moe_intermediate_size")
    routed_hidden = tc.get("routed_expert_hidden_size", tc.get("hidden_size"))
    n_layers = tc.get("num_hidden_layers")
    n_experts = tc.get("num_experts")
    first_dense = tc.get("first_k_dense_replace", 1)
    moe_freq = tc.get("moe_layer_freq", 1)
    if moe_inter is None:
        sys.exit("config.json missing moe_intermediate_size")
    print(f"[cfg] layers={n_layers} experts={n_experts} moe_inter={moe_inter} routed_hidden={routed_hidden}")

    # Collect expert packed tensor locations across all shards.
    # Map (layer, expert, w_idx) -> (shard_path, tensor_name)
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"No safetensors found in {src}")

    # First pass: inventory — find every expert packed tensor and group by shard.
    # K3 shards are huge (many GB); we process one shard at a time to bound RAM.
    # We also keep the dense tensors and re-emit them unchanged into the matching
    # output shard.
    expert_packed_in_shard = {}  # shard_path -> list of (layer, expert, w_idx, packed_name, scale_name)
    for shard in shards:
        with safe_open(str(shard), framework="numpy") as f:
            for name in f.keys():
                m = EXPERT_PACKED_RE.match(name)
                if not m:
                    continue
                layer, expert, w_idx = int(m.group(1)), int(m.group(2)), m.group(3)
                scale_name = name.replace("weight_packed", "weight_scale")
                expert_packed_in_shard.setdefault(shard, []).append(
                    (layer, expert, w_idx, name, scale_name))

    total_experts = sum(len({(l, e) for l, e, _, _, _ in v}) for v in expert_packed_in_shard.values())
    print(f"[scan] {total_experts} expert instances across {len(shards)} shards")

    # Per-expert output (merged_weight + qs) is emitted into a single new shard
    # per source shard: keeps the expert collocated with its dense weights so the
    # engine's safetensors index resolves to the same file.
    expert_count = 0
    total_in = total_out = 0
    for shard in shards:
        tensors = load_file(str(shard))
        out_tensors = {}
        # Group packed tensors by (layer, expert) so we can build merged_weight.
        experts_here = {}
        for layer, expert, w_idx, packed_name, scale_name in expert_packed_in_shard.get(shard, []):
            experts_here.setdefault((layer, expert), {})[w_idx] = (packed_name, scale_name)

        for (layer, expert), wmap in experts_here.items():
            # w1 = gate [moe_inter, routed_hidden], w3 = up [moe_inter, routed_hidden], w2 = down [routed_hidden, moe_inter]
            g_packed = tensors[wmap["1"][0]]; g_scale = tensors[wmap["1"][1]]
            u_packed = tensors[wmap["3"][0]]; u_scale = tensors[wmap["3"][1]]
            d_packed = tensors[wmap["2"][0]]; d_scale = tensors[wmap["2"][1]]
            # Dequantize
            g = mxfp4_dequant(g_packed, g_scale, moe_inter, routed_hidden)
            u = mxfp4_dequant(u_packed, u_scale, moe_inter, routed_hidden)
            d = mxfp4_dequant(d_packed, d_scale, routed_hidden, moe_inter)
            # Drop the packed tensors from the output
            for w_idx in ("1", "2", "3"):
                tensors.pop(wmap[w_idx][0], None)
                tensors.pop(wmap[w_idx][1], None)
            # Re-quantize row-wise to int8
            qg, qgs = quantize_row_int8(g)
            qu, qus = quantize_row_int8(u)
            qd, qds = quantize_row_int8(d)
            # Concatenate: merged_weight = [qg | qu | qd] as raw int8 bytes
            merged = np.concatenate([qg.reshape(-1), qu.reshape(-1), qd.reshape(-1)]).astype(np.int8)
            qs = np.concatenate([qgs, qus, qds]).astype(np.float32)
            # Names: drop "language_model." prefix? Engine probes either. Keep prefix to match source.
            # experts_here is dict[(layer, expert)] -> {w_idx: (packed_name, scale_name)}.
            _any_lm = any(
                pn.startswith("language_model.")
                for wmap in experts_here.values()
                for (pn, _sn) in wmap.values()
            )
            prefix = "language_model." if _any_lm else ""
            base = f"{prefix}model.layers.{layer}.block_sparse_moe.experts.{expert}"
            out_tensors[base + ".merged_weight"] = merged
            out_tensors[base + ".qs"] = qs
            expert_count += 1
            total_in  += g.nbytes + u.nbytes + d.nbytes  # fp32 dequant bytes
            total_out += merged.nbytes + qs.nbytes

        # Pass through all remaining (dense) tensors unchanged
        for name, t in tensors.items():
            out_tensors[name] = t

        out_shard = out / shard.name
        save_file(out_tensors, str(out_shard))
        print(f"[{shard.name}] {len(experts_here)} experts converted -> {out_shard.name}")

    ratio = total_out / max(total_in, 1) * 100
    print(f"\nDone. {expert_count} experts: MXFP4 -> int8.")
    print(f"Expert storage (dequant fp32): {total_in/1e9:.2f} GB -> {total_out/1e9:.2f} GB ({ratio:.0f}%)")
    print(f"Model ready at: {out}")
    print(f"\nRun: SNAP={out} ./kimik3 32 ref.json")


if __name__ == "__main__":
    main()
