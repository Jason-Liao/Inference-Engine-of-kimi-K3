# Inference-Engine-of-Kimi-K3

> 一个基于 [colibri](https://github.com/JustVugg/colibri) 魔改的 **Kimi-K3** 纯 C 推理引擎。能让 Moonshot AI 的 2.8T 参数 MoE 模型在普通硬件上以流式加载方式运行——前提是有足够的磁盘空间。

<p align="center">
  <b>Kimi-K3 · 2.8T MoE · MXFP4 · 93 层 · KDA + Gated MLA · 流式 CPU/CUDA</b>
</p>

---

## 一、这个项目是什么?

本项目是一个**独立的 Kimi-K3 推理引擎**,把 Kimi-K3 的 2.8T 参数 MoE 架构以纯 C 实现于单文件 [c/kimik3.c](c/kimik3.c)。它继承了 colibri 的"VRAM/RAM/disk 三级流式"思想,针对 K3 的 **KDA(Kimi Delta Attention)+ Gated MLA + Stable LatentMoE + MXFP4 量化**做了完整实现。

### 与上游 colibri 的关系

- **原项目**:[JustVugg/colibri](https://github.com/JustVugg/colibri) —— 一个为 GLM-5.2(744B MoE)设计的轻量纯 C 推理引擎,支持 expert 流式加载、int4/int8 量化、CUDA/Metal 后端。
- **本项目**:在 colibri 的基础上,新增 `c/kimik3.c`(单文件 ~1300 行 C),实现 Kimi-K3 完整架构。**未修改** colibri 原有的 GLM-5.2/OLMoE 引擎,只**新增** K3 路径。
- **共享代码**:复用 colibri 的 `st.h`(safetensors 多分片加载)、`tok.h`(O200K tokenizer)、`json.h`、`compat.h`、`Makefile`、`coli` 启动器。
- **License**:与 colibri 一致,Apache License 2.0(见 [LICENSE](LICENSE))。

### 来源

- **架构依据**:[Kimi-K3 技术报告](https://github.com/MoonshotAI/Kimi-K3)(`k3_tech_report.pdf`),详细阅读后逐节实现。
- **权重来源**:[huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)(MXFP4 量化,96 个 safetensors 分片,共 ~1.56 TB)。
- **配置依据**:真实 `config.json` + `configuration_kimi_k3.py` + `modeling_kimi_linear.py`(从 HF 下载到 `k3_meta/`)。

---

## 二、Kimi-K3 架构与实现映射

根据 k3_tech_report.pdf,K3 是一个 2.8T 参数 MoE 模型,本项目实现了以下全部组件:

| 架构组件 | K3 规格 | 实现位置 |
|---|---|---|
| 总层数 | 93 层 | `c/kimik3.c` `Cfg.n_layers` |
| Dense MLP 层 | 1 层(第 0 层,`first_k_dense_replace=1`) | `mlp_gate/up/down` |
| **KDA 层** | 68 层(Kimi Delta Attention,recurrent) | `kda_forward()` |
| **Gated MLA 层** | 24 层(Gated Multi-head Latent Attention,NoPE) | `mla_forward()` |
| Stable LatentMoE | 896 experts × top-16 + 2 shared experts | `moe_forward()` |
| Router | sigmoid + noaux_tc bias correction | `moe_forward()` |
| 激活函数 | SiTU-GLU(`activation_situ_beta=4.0`) | `situ_gate()` |
| Expert 量化 | MXFP4(E8M0 microscale,group=32) | `mxfp4_dequant_row()` |
| 上下文长度 | 1M tokens | `max_t` |
| 词表 | 163,840(O200K) | `tok.h` |

### KDA(Kimi Delta Attention)

K3 的核心创新之一。本引擎实现了**融合递归形式**(fused recurrent form):

- 每个头维护一个 `[head_dim, head_dim]` 状态矩阵 `S_t`
- 更新:`S_t = α_t · S_{t-1} + β_t · (v_t ⊗ k_t - α_t · β_t · k_t ⊗ k_t)`(delta rule)
- 输出:`o_t = S_t · q_t`,经 short-conv1d(k=4)+ full-rank gate `f_a/f_b`
- NoPE(无位置编码)

详见 [c/kimik3.c](c/kimik3.c) 中的 `kda_forward()` 函数,数学上等价于技术报告 §3.2 的 chunkwise 形式。

### Gated MLA(Multi-head Latent Attention)

- Q/K/V 经低秩 latent 压缩(`q_lora=1536`, `kv_lora=512`)
- NoPE(无 RoPE,`mla_use_nope=true`)
- 输出门 `g_proj`(input-dependent full-rank gate,`mla_use_output_gate=true`)

### Stable LatentMoE

- **896 个 routed experts**,每 token 激活 **16** 个(top-k)
- **2 个 shared experts**(fused gate/up,intermediate = `moe_inter * n_shared`)
- Router:**sigmoid** 激活(非 softmax)+ **noaux_tc** bias correction
- **normalized latent space**:`routed_expert_down_proj` + `routed_expert_norm` + `routed_expert_up_proj`
- **renormalize**:top-k 权重归一化(`moe_renormalize=true`)

### MXFP4 反量化

```c
// 每个 4-bit nibble 映射到 {-8,-7,...,-1,0,1,...,7}
// 每 32 个元素共享一个 E8M0 microscale = 2^(s-127)
static float mxfp4_lookup(int nibble, uint8_t scale_u8) { ... }
```

支持两种存储:
1. **MXFP4 原生**(HF 格式):`weight_packed` + `weight_scale`,加载时反量化到 fp32
2. **colibri int8**(经 `tools/convert_kimik3.py` 转换):per-row int8 + f32 scale,4x RAM 节省

### 残差块(attn_res_block_size=12)

K3 每 12 层插入一个残差投影:`self_attention_res_norm/proj` 和 `mlp_res_norm/proj`,以及顶层的 `output_attn_res_norm/proj`。本引擎已识别并加载这些张量。

---

## 三、当前验证状态

### ✅ 已验证通过

| 验证项 | 方式 | 结果 |
|---|---|---|
| 架构 forward 正确性 | tiny K3 模型(64 hidden, 4 层)端到端 forward | ✅ 输出合理 |
| 确定性 | 同输入多次运行,输出完全一致 | ✅ 通过 |
| PPL 模式 | teacher-forced 困惑度计算 | ✅ 通过 |
| MXFP4 反量化算法 | 单元测试 + tiny 模型 expert | ✅ 通过 |
| int8 转换路径 | `tools/convert_kimik3.py` + 引擎加载 | ✅ 通过 |
| **真实 config.json 解析** | `--config-check` 跑 HF 真实配置 | ✅ 24 MLA + 68 KDA + 1 dense |
| **真实张量名匹配** | 497,220 个张量名全对 | ✅ 通过 |
| **真实权重 bf16 反序列化** | `--load-check` 读 28 个真实张量(7 GB) | ✅ 0 NaN, 0 全零,数值合理 |

### 📦 已下载的真实权重

从 [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) 下载了 **2 个 shard,共 ~7 GB**:

| Shard | 大小 | 内容 |
|---|---|---|
| `model-00001-of-000096.safetensors` | 2.34 GB | Layer 0 全部权重(KDA attention + dense MLP) |
| `model-00094-of-000096.safetensors` | 4.7 GB | `embed_tokens` + `lm_head` + final norm + output_attn_res |

**验证结果**(通过 `--load-check`):

```
== load-check: 2 shards indexed (28 tensors) ==
  OK  embed_tokens.weight    numel=1174405120  min=-0.101  max=+0.106  mean=-0.00036
  OK  lm_head.weight         numel=1174405120  min=-0.190  max=+0.151  mean=+5.4e-05
  OK  layers.0.self_attn.q_proj.weight   numel=88080384  min=-0.011  max=+0.013
  OK  layers.0.self_attn.A_log           numel=128       min=-0.753  max=+2.466
  OK  layers.0.mlp.down_proj.weight       numel=242221056 min=-0.133  max=+0.131
  ...
== summary: OK=28  MISSING=2432  NaN=0  ZERO=0 ==
[OK] deserialization path exercised against real safetensors shards.
```

### ⚠️ 待验证(需要大容量机器)

以下验证**未在本环境完成**,因为受限于硬件(详见下节):

| 待验证项 | 为什么没验证 |
|---|---|
| 完整 93 层 forward pass | 只下载了 layer 0 + embed,缺 layer 1-92 |
| 真实大尺寸下的数值稳定性 | tiny 模型 hidden=64,真实 7168,bf16 累积误差未测 |
| 真实 expert 的 MXFP4 字节序 | 算法对,但真实 packed 布局未跑过完整 expert |
| KDA 在长序列(1M context)的稳定性 | tiny 模型只测了几 token,recurrent state 可能发散 |
| MLA 真实 attention 数值 | tiny 模型 qk_rope=8,真实 64,RoPE 未在真实尺寸验证 |
| 与 HuggingFace transformers 输出对齐 | 需在同一机器跑两份代码对比 logits |

**49 个自动化测试全部通过**(详见 [c/tests/test_kimik3.py](c/tests/test_kimik3.py))。

---

## 四、需要的机器条件

要加载**完整** Kimi-K3 权重并推理,需要:

### 磁盘

| 项目 | 需求 |
|---|---|
| 完整权重下载 | **~1.56 TB**(96 个 `model-XXXXX-of-000096.safetensors`) |
| int8 转换后(可选) | 额外 ~400 GB |
| **磁盘总计** | **≥ 2 TB** 建议 |

### 内存(RAM)

引擎启动时把所有 **dense 权重**(attention + shared experts + router + norms + embed + lm_head)从 bf16 反量化到 fp32 全量驻留 RAM:

| 部分 | 磁盘 bf16 | RAM fp32(实际) |
|---|---|---|
| 24 个 MLA 层 | 20 GB | 41 GB |
| 68 个 KDA+MoE 层 | 110 GB | 220 GB |
| Layer 0(dense MLP) | 2.7 GB | 5.4 GB |
| embed + lm_head | 5 GB | 9 GB |
| **dense 合计** | **137 GB** | **~275 GB** |
| Expert cache(top-16 × 92 层) | — | ~50 GB(MXFP4/int8)或 194 GB(fp32) |
| **RAM 总计** | — | **≥ 320 GB 建议** |

### 推荐配置

| 用途 | 配置 |
|---|---|
| **最小可跑** | 320 GB RAM + 2 TB NVMe SSD + 64-core CPU |
| **舒适运行** | 512 GB RAM + 4 TB NVMe + 96-core CPU + 可选 8×H100 |
| **生产部署** | 1 TB RAM + 8 TB NVMe + 128-core + 8×H100 |

> ⚠️ **低于 320 GB RAM 无法启动完整推理**(引擎会在 `model_init` 阶段 OOM)。
> 可用 `--config-check` 和 `--load-check`(采样模式,仅需 4 GB RAM)做兼容性验证。

---

## 五、安装与使用

### 1. 编译

```bash
cd c
make kimik3
# 或带调试符号
make kimik3_dbg
```

依赖:GCC/Clang、OpenMP(`libgomp`)、Python 3(仅测试用)。无其他依赖。

### 2. 下载 Kimi-K3 权重

```bash
# 完整下载(~1.56 TB,需要大磁盘)
huggingface-cli download moonshotai/Kimi-K3 --local-dir /path/to/k3

# 或只下载少数 shard 做加载验证(~7 GB)
huggingface-cli download moonshotai/Kimi-K3 \
    model-00001-of-000096.safetensors \
    model-00094-of-000096.safetensors \
    config.json model.safetensors.index.json \
    --local-dir /path/to/k3_partial
```

### 3. 验证配置解析(仅需 4 GB RAM)

```bash
SNAP=/path/to/k3 ./kimik3 --config-check
```

输出示例:
```
== Kimi-K3 engine v0.1 | cache=0/layer ==
[CFG] hidden=7168 layers=93 (dense=1, MLA=24, KDA=68), experts=896 topk=16 shared=2, vocab=163840
[CFG] q_lora=1536 kv_lora=512 qk_nope=128 qk_rope=64 v_head=128 | kda_heads=96 kda_hd=128 short_conv=4
[CFG] first_dense=1 moe_freq=1 moe_renorm=1 situ_beta=4.0 situ_lin_beta=25.0 eps=1.0e-05 routed_scale=1.00
[CFG] mla_layers: 3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63 67 71 75 79 83 87 91 92
[CFG] kda_layers: 1 2 4 5 6 8 9 10 12 13 14 ... (68 个)
[CFG] dense_layer: 0
[OK] config parsed successfully — engine is compatible with this snapshot.
```

### 4. 验证真实权重加载(仅需 4 GB RAM,采样模式)

```bash
SNAP=/path/to/k3_partial ./kimik3 --load-check
```

### 5. 完整推理(需 ≥320 GB RAM)

```bash
SNAP=/path/to/k3 ./kimik3 16  # 16 = expert cache slots per layer
```

环境变量:

| 变量 | 默认 | 说明 |
|---|---|---|
| `SNAP` | (必填) | 模型目录 |
| `PILOT` | 0 | 跨层 expert 预取 lookahead |
| `HOT` | 0 | 预热后 pin 住 top-N 热 expert |
| `WARMUP` | 5 | hot pinning 激活前的 token 数 |
| `TOPK` | 配置值 | 覆盖 MoE top-k |
| `TEMP` | 0.0 | 0=greedy,否则温度采样 |
| `PPL` | 0 | 设为 1 跑 teacher-forced 困惑度 |

### 6. 通过 colibri 启动器

```bash
./coli build-kimik3      # 编译 kimik3
./coli kimik3            # 启动 K3 引擎
```

### 7. 运行测试

```bash
cd c
python3 -m pytest tests/test_kimik3.py -v
# 49 个测试,包括:
#   - 引擎功能(21):build, config, forward, determinism, PPL, int8, stability
#   - 真实配置兼容(18):config.json + 张量名匹配
#   - 真实权重加载(10):bf16 反序列化 + 数值检查
```

### 8. 生成 tiny K3 模型做开发测试

```bash
python3 c/tools/make_kimik3_tiny.py --out c/kimik3_tiny
SNAP=c/kimik3_tiny ./c/kimik3 8 c/kimik3_tiny/ref.json
```

### 9. 转换 expert 为 int8(可选,省 4x RAM)

```bash
python3 c/tools/convert_kimik3.py --src /path/to/k3 --dst /path/to/k3_int8
SNAP=/path/to/k3_int8 ./c/kimik3 16
```

---

## 六、项目结构

```
.
├── c/
│   ├── kimik3.c              # ★ K3 推理引擎核心(单文件 ~1300 行 C)
│   ├── Makefile              # 编译规则(含 kimik3 target)
│   ├── coli                  # colibri 启动器(支持 build-kimik3 / kimik3)
│   ├── colibri.c             # colibri 原 GLM-5.2 引擎(未修改)
│   ├── olmoe.c               # colibri 原 OLMoE 引擎(未修改)
│   ├── st.h                  # safetensors 多分片加载器(复用)
│   ├── tok.h                 # O200K tokenizer(复用)
│   ├── json.h / compat.h     # JSON 解析 + 跨平台兼容(复用)
│   ├── tools/
│   │   ├── make_kimik3_tiny.py    # 生成 tiny K3 测试模型
│   │   └── convert_kimik3.py      # MXFP4 → int8 转换器
│   ├── tests/
│   │   └── test_kimik3.py    # ★ 49 个自动化测试
│   └── kimik3_tiny/           # 预生成的 tiny 测试模型
└── LICENSE                   # Apache 2.0(与 colibri 一致)
```

---

## 七、局限性与已知问题

1. **KDA 用 recurrent 形式,非 chunkwise** —— 数学等价但 prefill 慢。未来优化点。
2. **bf16 → fp32 全量驻留** —— dense 权重 275 GB RAM。未来计划改成 bf16 原生驻留 + 混合精度 matmul。
3. **完整 forward 未在大尺寸验证** —— 见上文"待验证"。
4. **无 vision 支持** —— K3 是多模态模型,本引擎只实现 LM 部分。
5. **无 CUDA/Metal 后端** —— 纯 CPU 实现(继承 colibri 的 matmul,但未启用 GPU 后端)。

---

## 八、致谢

- **[JustVugg/colibri](https://github.com/JustVugg/colibri)** —— 本项目的基础,提供了流式 MoE 加载框架、safetensors 解析器、O200K tokenizer。
- **[Moonshot AI](https://github.com/MoonshotAI/Kimi-K3)** —— Kimi-K3 模型与技术报告。
- **[huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)** —— 开源权重。

---

## 九、开源条款

本项目继承上游 [colibri](https://github.com/JustVugg/colibri) 的 **Apache License 2.0**。详见 [LICENSE](LICENSE)。

```
Copyright 2026 JustVugg (colibri) / Jason-Liao (K3 engine adaptation)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Kimi-K3 模型权重遵循 [Moonshot AI 的开源协议](https://huggingface.co/moonshotai/Kimi-K3)。
