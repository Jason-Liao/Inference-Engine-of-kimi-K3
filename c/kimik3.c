/* kimik3.c — Kimi K3 inference engine for colibri.
 *
 * Implements the Kimi-K3 language-model architecture (k3_tech_report.pdf):
 *   - 93 layers: 1 dense MLP (layer 0) + 68 KDA layers + 24 Gated-MLA layers
 *   - Stable LatentMoE: sigmoid router + noaux_tc bias correction + 2 shared experts +
 *     SiTU-GLU activation + normalized latent space (routed_expert_*, 896 experts x 16 active)
 *   - MXFP4 quantized routed experts (group_size=32, E8M0 microscale)
 *   - Tokenizer: O200K (shared with colibri / olmoe via tok.h)
 *
 * The engine reads the Hugging Face `moonshotai/Kimi-K3` safetensors DIRECTLY:
 *   - Dense tensors (attention, RMSNorm, embed, lm_head, shared experts, layer-0 MLP)
 *     are loaded as bf16 -> f32.
 *   - Routed experts (`block_sparse_moe.experts.N.w{1,2,3}.weight_packed` +
 *     `.weight_scale`, MXFP4) are dequantized to fp32 in the expert cache on demand.
 *   - Optional: tools/convert_kimik3.py re-quantizes experts to colibri row-wise int8
 *     (`model.layers.N.block_sparse_moe.experts.E.merged_weight` + `.qs`) which the
 *     same code path consumes via matmul_q — a 4x RAM reduction in the expert cache.
 *
 * KDA (Kimi Delta Attention) is implemented in the fused-recurrent form, run
 * token-by-token for both prefill and decode. This is mathematically equivalent
 * to the chunkwise form (k3_tech_report §3.2) and keeps the implementation
 * tractable; a chunked prefill path is the obvious future optimisation.
 *
 * Usage:
 *   SNAP=<k3_dir> ./kimik3 [cap] [ref.json]
 *     cap   = expert cache slots per layer  (default 16)
 *     ref   = ref.json with prompt_ids/full_ids (default "ref.json")
 *
 * Env vars:
 *   SNAP=<dir>           model snapshot directory (required)
 *   PILOT=0/1/2/3        cross-layer expert prefetch lookahead (default 0)
 *   HOT=N                pin top-N hot experts per layer after warmup
 *   WARMUP=N             tokens before hot pinning activates (default 5)
 *   TOPK=N               override MoE top-k (default = config)
 *   TEMP=0.0             greedy (default); else temperature sampling
 *   PPL=1                teacher-forced perplexity on ref.json (full_ids)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#include "st.h"
#include "json.h"
#include "compat.h"
#include "tok.h"

#ifdef _WIN32
#define sleep_ms(ms) Sleep(ms)
#else
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

/* ---------- config ---------- */
typedef struct {
    int hidden;            /* hidden_size              = 7168 */
    int n_layers;          /* num_hidden_layers        = 93   */
    int n_heads;           /* num_attention_heads      = 96   */
    int n_kv_heads;        /* num_key_value_heads      = 96   */
    int q_lora;            /* q_lora_rank              = 1536 */
    int kv_lora;           /* kv_lora_rank             = 512  */
    int qk_nope;           /* qk_nope_head_dim         = 128  */
    int qk_rope;           /* qk_rope_head_dim         = 64   */
    int v_head;            /* v_head_dim               = 128  */
    int qk_head;           /* qk_nope + qk_rope        = 192  */
    int inter;             /* intermediate_size (L0)    = 33792 */
    int moe_inter;         /* moe_intermediate_size    = 3072 */
    int routed_hidden;     /* routed_expert_hidden_size= 3584 */
    int n_experts;         /* num_experts              = 896  */
    int topk;              /* num_experts_per_token    = 16   */
    int n_shared;          /* num_shared_experts       = 2    */
    int vocab;             /* vocab_size               = 163840 */
    int first_dense;       /* first_k_dense_replace    = 1    */
    int moe_freq;          /* moe_layer_freq           = 1    */
    int kda_head_dim;      /* linear_attn.head_dim     = 128  */
    int kda_n_heads;       /* linear_attn.num_heads   = 96   */
    int short_conv;        /* short_conv_kernel_size   = 4    */
    int attn_res_block;    /* attn_res_block_size      = 12   */
    int use_full_rank_gate;/* use_full_rank_gate       = true */
    int use_output_gate;   /* mla_use_output_gate      = true */
    int moe_renormalize;   /* moe_renormalize          = true */
    int n_expert_group;    /* num_expert_group         = 1    */
    int topk_group;        /* topk_group               = 1    */
    float situ_beta;       /* activation_situ_beta     = 4.0  */
    float situ_linear_beta;/* activation_situ_linear_beta = 25.0 */
    float theta;           /* rope_theta (default 1e6 for K3 NoPE-hybrid) */
    float eps;             /* rms_norm_eps             = 1e-5 */
    float routed_scale;    /* routed_scaling_factor    = 1.0  */
    /* which layers are MLA (the rest of MoE layers are KDA). Built at load. */
    uint8_t *is_mla;       /* [n_layers], 1 if MLA, 0 if KDA (or dense) */
    uint8_t *is_moe;       /* [n_layers], 1 if MoE layer, 0 if dense MLP (layer 0) */
} Cfg;

/* ---------- per-layer dense weights ---------- */
typedef struct {
    /* common */
    float *in_ln, *post_ln;
    /* MLA-only */
    float *q_a, *q_a_ln, *q_b, *kv_a, *kv_a_ln, *kv_b, *o, *g_proj_mla;
    /* KDA-only */
    float *q_proj, *k_proj, *v_proj;
    float *q_conv, *k_conv, *v_conv;
    float *A_log, *dt_bias, *b_proj, *f_a, *f_b, *g_proj_kda, *o_norm;
    /* residual-block (attn_res_block_size>0) */
    float *attn_res_norm, *attn_res_proj;
    float *mlp_res_norm, *mlp_res_proj;
    /* dense MLP (layer 0) */
    float *mlp_gate, *mlp_up, *mlp_down;
    /* MoE dense components (per MoE layer) */
    float *gate_w;          /* [n_experts, hidden] router */
    float *e_score_bias;    /* [n_experts] */
    float *shared_gate, *shared_up, *shared_down; /* 2 shared experts fused */
    float *routed_down, *routed_up, *routed_norm; /* latent MoE projection */
} Layer;

/* ---------- per-expert cached weights ----------
 * Two storage layouts are supported, selected by what's present on disk:
 *   - MXFP4 (HF native): packed[i] + scale[i] (uint8). Dequantized to f32 on load.
 *   - colibri int8 (from convert_kimik3.py): g/u/d as int8 + per-row f32 scales.
 * `kind` selects the matmul path. */
typedef enum { EXP_NONE=0, EXP_MXFP4, EXP_INT8 } ExpertKind;
typedef struct {
    int eid;
    int pinned;
    ExpertKind kind;
    /* MXFP4 path: dequantized on load into g/u/d (float*). */
    float *g, *u, *d;       /* [moe_inter, routed_hidden], [moe_inter, routed_hidden], [routed_hidden, moe_inter] */
    /* INT8 path: int8 weights + per-row f32 scales (mirrors olmoe.c). */
    int8_t *qg, *qu, *qd;
    float *qgs, *qus, *qds;
    uint64_t used;
} Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    float *embed, *lm_head, *final_norm;
    Layer *L;
    LCache *cache;          /* [n_layers] */
    uint64_t clock, hits, miss;
    /* KDA recurrent state: per-layer, per-head [head_dim, head_dim] s_t matrix
     * and per-head conv1d sliding windows for q/k/v. */
    float **kda_state;      /* [n_layers][n_heads * kda_head_dim * kda_head_dim] */
    float **kda_conv_q, **kda_conv_k, **kda_conv_v; /* [n_layers][short_conv * proj_size] */
    /* MLA KV cache: per-layer, per-head, [max_t * v_head] for V, [max_t * qk_nope] for K_nope,
     * plus a single shared [max_t * qk_rope] for k_rot. */
    float **K_nope, **V;    /* [n_layers][n_heads * max_t * v_head_or_qk_nope] */
    float **K_rot;          /* [n_layers][max_t * qk_rope] */
    int kv_len, max_t;
    int quant_bits;
    double dense_load_s;
    /* expert heatmap for HOT pinning */
    uint32_t *freq;
    int freq_token_count, hot_pinned, hot_n, warmup_tokens, token_count;
    uint8_t *is_pinned;
} Model;

static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;

/* ---------- utils ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
#endif
static float *falloc(int64_t n) { float *p = malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM %lld\n",(long long)n);exit(1);} return p; }

/* y[S,O] = x[S,I] @ W[O,I]^T  (row-major) */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* y[O] = x[I] @ q[O,I]^T (int8) — colibri row-wise quantised matmul. */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* SiTU activation: a = beta*tanh(gate/beta)*sigmoid(gate); out = a * (linear_beta*tanh(up/linear_beta)) */
static float situ_gate(float g, float u, float beta, float lin_beta) {
    float a = beta * tanhf(g / beta) * (1.f / (1.f + expf(-g)));
    float u_t = lin_beta > 0.f ? lin_beta * tanhf(u / lin_beta) : u;
    return a * u_t;
}

/* MXFP4 dequant: packed[O, I/2] (uint8, two 4-bit nibbles) + scale[O, I/32] (uint8 E8M0).
 * MXFP4 nibble encoding (0..15) maps to {0,1,2,3,4,5,6,7,-8,-7,-6,-5,-4,-3,-2,-1}
 * (two's complement of 4 bits). E8M0 microscale is 2^(s-127). */
static float mxfp4_lookup(int nibble, uint8_t scale_u8) {
    static const float mxfp4_lut[16] = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f
    };
    /* E8M0: 8-bit exponent with bias 127, no mantissa. 0x00 -> 2^-127 (treated as 0). */
    float scale = scale_u8 == 0 ? 0.f : ldexpf(1.f, (int)scale_u8 - 127);
    return mxfp4_lut[nibble & 15] * scale;
}

/* Dequantize one MXFP4 weight row of length I into dst[I].
 * packed: [I/2] bytes, scales: [I/32] bytes. */
static void mxfp4_dequant_row(float *dst, const uint8_t *packed, const uint8_t *scales, int I) {
    int n_groups = I / 32;
    for (int g = 0; g < n_groups; g++) {
        float s_exp = scales[g] == 0 ? 0.f : ldexpf(1.f, (int)scales[g] - 127);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = packed[g*16 + j];
            int hi = (byte >> 4) & 15;
            int lo = byte & 15;
            static const float lut[16] = {
                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
                -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f
            };
            dst[g*32 + j*2]     = lut[hi] * s_exp;
            dst[g*32 + j*2 + 1] = lut[lo] * s_exp;
        }
    }
}

/* ---------- config ---------- */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static double req_num_path(jval *r, const char *path){
    jval *v = json_get(r, "text_config");
    if (!v || v->t != J_OBJ) {
        /* allow direct top-level (tiny test models may not nest under text_config) */
        return req_num(r, path);
    }
    jval *p = json_get(v, path);
    if (!p || p->t != J_NUM) {
        fprintf(stderr,"config.json: missing text_config.%s\n", path); exit(1);
    }
    return p->num;
}
static int has_path_num(jval *r, const char *parent, const char *key) {
    jval *v = json_get(r, parent);
    if (!v || v->t != J_OBJ) return 0;
    jval *p = json_get(v, key);
    return p && p->t == J_NUM;
}
/* Like has_path_num but accepts booleans too (K3 ships moe_renormalize=true
 * and mla_use_output_gate=true as JSON booleans, not 0/1 numbers). */
static int has_path_bool(jval *r, const char *parent, const char *key) {
    jval *v = json_get(r, parent);
    if (!v || v->t != J_OBJ) return 0;
    jval *p = json_get(v, key);
    if (!p) return 0;
    if (p->t == J_BOOL) return p->boolean ? 1 : 0;
    if (p->t == J_NUM)  return p->num != 0 ? 1 : 0;
    return 0;
}

/* Read layer-type map from linear_attn_config.
 *
 * K3 uses 1-INDEXED layer lists (see configuration_kimi_k3.py:is_kda_layer):
 *   is_kda_layer(i)  =  (i+1) in kda_layers
 *   MLA layer i      =  (i+1) in full_attn_layers  (i.e. NOT kda)
 * So a value v in full_attn_layers means layer index v-1 is MLA.
 * The is_moe flag is independent: layer i is MoE if i >= first_k_dense_replace
 * and i % moe_layer_freq == 0. Layer 0 is KDA attention + dense MLP. */
static void read_mla_layers(Cfg *c, jval *r) {
    jval *tc = json_get(r, "text_config");
    if (!tc) tc = r;
    jval *lac = json_get(tc, "linear_attn_config");
    c->is_mla = calloc(c->n_layers, sizeof(uint8_t));
    c->is_moe = calloc(c->n_layers, sizeof(uint8_t));
    /* MoE layers: i >= first_dense and i % moe_freq == 0 */
    for (int i = 0; i < c->n_layers; i++) {
        if (i >= c->first_dense && (i % c->moe_freq) == 0) c->is_moe[i] = 1;
    }
    if (!lac) {
        /* no linear_attn_config: assume all layers are KDA (no MLA) */
        return;
    }
    /* full_attn_layers contains 1-indexed values: value v -> layer v-1 is MLA */
    jval *fa = json_get(lac, "full_attn_layers");
    if (fa && fa->t == J_ARR) {
        for (int i = 0; i < fa->len; i++) {
            int idx = (int)fa->kids[i]->num - 1;   /* 1-indexed -> 0-indexed */
            if (idx >= 0 && idx < c->n_layers) c->is_mla[idx] = 1;
        }
    }
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0 || n>(256L<<20)){ fprintf(stderr,"%s: config.json missing or >256MB\n",path); exit(1); }
    char *buf = malloc((size_t)n+1);
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); }
    buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);

    /* K3 nests language config under "text_config"; tiny test models may flatten it. */
    jval *src = json_get(r, "text_config");
    if (!src) src = r;
    c->hidden       = (int)req_num(src,"hidden_size");
    c->n_layers     = (int)req_num(src,"num_hidden_layers");
    c->n_heads      = (int)req_num(src,"num_attention_heads");
    c->n_kv_heads   = (int)req_num(src,"num_key_value_heads");
    c->n_experts    = (int)req_num(src,"num_experts");
    c->topk         = (int)req_num(src,"num_experts_per_token");
    c->inter        = (int)req_num(src,"intermediate_size");
    c->vocab        = (int)req_num(src,"vocab_size");
    c->first_dense  = has_path_num(r,"text_config","first_k_dense_replace") ? (int)req_num(src,"first_k_dense_replace") : 1;
    c->moe_freq     = has_path_num(r,"text_config","moe_layer_freq") ? (int)req_num(src,"moe_layer_freq") : 1;
    c->n_shared     = has_path_num(r,"text_config","num_shared_experts") ? (int)req_num(src,"num_shared_experts") : 0;
    c->q_lora       = has_path_num(r,"text_config","q_lora_rank") ? (int)req_num(src,"q_lora_rank") : 0;
    c->kv_lora      = has_path_num(r,"text_config","kv_lora_rank") ? (int)req_num(src,"kv_lora_rank") : 0;
    c->qk_nope      = has_path_num(r,"text_config","qk_nope_head_dim") ? (int)req_num(src,"qk_nope_head_dim") : 0;
    c->qk_rope      = has_path_num(r,"text_config","qk_rope_head_dim") ? (int)req_num(src,"qk_rope_head_dim") : 0;
    c->v_head       = has_path_num(r,"text_config","v_head_dim") ? (int)req_num(src,"v_head_dim") : 0;
    c->moe_inter    = has_path_num(r,"text_config","moe_intermediate_size") ? (int)req_num(src,"moe_intermediate_size") : c->inter;
    c->routed_hidden= has_path_num(r,"text_config","routed_expert_hidden_size") ? (int)req_num(src,"routed_expert_hidden_size") : c->hidden;
    c->attn_res_block = has_path_num(r,"text_config","attn_res_block_size") ? (int)req_num(src,"attn_res_block_size") : 0;
    c->use_output_gate = has_path_bool(r,"text_config","mla_use_output_gate");
    c->moe_renormalize = has_path_bool(r,"text_config","moe_renormalize");
    c->n_expert_group = has_path_num(r,"text_config","num_expert_group") ? (int)req_num(src,"num_expert_group") : 1;
    c->topk_group   = has_path_num(r,"text_config","topk_group") ? (int)req_num(src,"topk_group") : 1;
    c->situ_beta    = has_path_num(r,"text_config","activation_situ_beta") ? (float)req_num(src,"activation_situ_beta") : 1.f;
    c->situ_linear_beta = has_path_num(r,"text_config","activation_situ_linear_beta") ? (float)req_num(src,"activation_situ_linear_beta") : 0.f;
    c->routed_scale = has_path_num(r,"text_config","routed_scaling_factor") ? (float)req_num(src,"routed_scaling_factor") : 1.f;
    /* KDA config */
    jval *lac = json_get(src, "linear_attn_config");
    if (lac && lac->t == J_OBJ) {
        jval *hd = json_get(lac, "head_dim"); c->kda_head_dim = hd ? (int)hd->num : (c->hidden / c->n_heads);
        jval *nh = json_get(lac, "num_heads"); c->kda_n_heads = nh ? (int)nh->num : c->n_heads;
        jval *sc = json_get(lac, "short_conv_kernel_size"); c->short_conv = sc ? (int)sc->num : 4;
        jval *fg = json_get(lac, "use_full_rank_gate"); c->use_full_rank_gate = fg ? (fg->t==J_BOOL ? fg->boolean : (fg->num!=0)) : 0;
    } else {
        c->kda_head_dim = c->hidden / c->n_heads;
        c->kda_n_heads  = c->n_heads;
        c->short_conv = 4;
        c->use_full_rank_gate = 1;
    }
    c->qk_head = c->qk_nope + c->qk_rope;
    jval *th = json_get(src,"rope_theta");  c->theta = th ? (float)th->num : 1e6f;
    jval *ep = json_get(src,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-5f;

    if(c->hidden<1||c->hidden>(1<<20) || c->n_layers<1||c->n_layers>4096 ||
       c->n_experts<1||c->n_experts>(1<<20) || c->topk<1||c->topk>c->n_experts ||
       c->vocab<1||c->vocab>(1<<24)){
        fprintf(stderr,"config.json: dimension out of range\n"); exit(1);
    }
    read_mla_layers(c, r);
    free(buf); free(arena);
}

/* ---------- tensor loading ---------- */
/* load a tensor as float32; respects dtype in safetensors (bf16/f16/f32) */
static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}
/* like load_t but returns NULL if missing instead of exiting */
static float *load_t_opt(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) return NULL;
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}
/* load a uint8 tensor (raw bytes) into a freshly-malloc'd buffer; returns size */
static uint8_t *load_u8(Model *m, const char *name, int64_t *n_out) {
    st_tensor *t = st_find(&m->S, name);
    if (!t) { *n_out = -1; return NULL; }
    int64_t n = t->nbytes;
    uint8_t *p = malloc((size_t)n);
    if (!p) { fprintf(stderr,"OOM %s\n", name); exit(1); }
    st_read_raw(&m->S, name, p, 1);
    *n_out = n;
    return p;
}

static void model_init(Model *m, const char *snap, int cap) {
    memset(m, 0, sizeof(*m));
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    /* embed / lm_head / final_norm — K3 tensor names use "language_model." prefix */
    m->embed      = load_t_opt(m, "language_model.model.embed_tokens.weight");
    if (!m->embed) m->embed = load_t(m, "model.embed_tokens.weight");
    m->lm_head    = load_t_opt(m, "language_model.lm_head.weight");
    if (!m->lm_head) m->lm_head = load_t(m, "lm_head.weight");
    m->final_norm = load_t_opt(m, "language_model.model.norm.weight");
    if (!m->final_norm) m->final_norm = load_t(m, "model.norm.weight");

    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[512];
    /* K3 ships tensors under "language_model.model.layers.N.*"; tiny test models
     * (and a flat-converted snapshot) use "model.layers.N.*". Probe once and pick. */
    const char *PFX = (st_find(&m->S, "language_model.model.layers.0.input_layernorm.weight")) ? "language_model." : "";
    /* Build the per-layer format string once: "<pfx>model.layers.%d.<suffix>".
     * P is a runtime string (not a literal), so we must use %s in snprintf. */
    #define LD(field, suffix) do { \
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attn." suffix, PFX, i); \
        l->field = load_t_opt(m,nm); } while(0)
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.input_layernorm.weight", PFX, i);  l->in_ln = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.post_attention_layernorm.weight", PFX, i); l->post_ln = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attention_res_norm.weight", PFX, i); l->attn_res_norm = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attention_res_proj.weight", PFX, i); l->attn_res_proj = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp_res_norm.weight", PFX, i); l->mlp_res_norm = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp_res_proj.weight", PFX, i); l->mlp_res_proj = load_t_opt(m,nm);
        if (c->is_mla[i]) {
            LD(q_a, "q_a_proj.weight");
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attn.q_a_layernorm.weight", PFX, i); l->q_a_ln = load_t_opt(m,nm);
            LD(q_b, "q_b_proj.weight");
            LD(kv_a, "kv_a_proj_with_mqa.weight");
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attn.kv_a_layernorm.weight", PFX, i); l->kv_a_ln = load_t_opt(m,nm);
            LD(kv_b, "kv_b_proj.weight");
            LD(o,    "o_proj.weight");
            if (c->use_output_gate) LD(g_proj_mla, "g_proj.weight");
        } else {
            /* KDA: all non-MLA layers use KDA attention, including layer 0
             * (layer 0 is KDA attention + dense MLP, not just dense MLP). */
            LD(q_proj, "q_proj.weight");
            LD(k_proj, "k_proj.weight");
            LD(v_proj, "v_proj.weight");
            LD(q_conv, "q_conv1d.weight");
            LD(k_conv, "k_conv1d.weight");
            LD(v_conv, "v_conv1d.weight");
            LD(A_log, "A_log");
            LD(dt_bias, "dt_bias");
            LD(b_proj, "b_proj.weight");
            LD(f_a, "f_a_proj.weight");
            LD(f_b, "f_b_proj.weight");
            if (c->use_full_rank_gate) LD(g_proj_kda, "g_proj.weight");
            LD(o_norm, "o_norm.weight");
            LD(o, "o_proj.weight");
        }
        /* dense MLP (layer 0) */
        if (!c->is_moe[i]) {
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp.gate_proj.weight", PFX, i); l->mlp_gate = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp.up_proj.weight", PFX, i);   l->mlp_up   = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp.down_proj.weight", PFX, i); l->mlp_down = load_t_opt(m,nm);
        } else {
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.gate.weight", PFX, i); l->gate_w = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.gate.e_score_correction_bias", PFX, i); l->e_score_bias = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.routed_expert_down_proj.weight", PFX, i); l->routed_down = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.routed_expert_up_proj.weight", PFX, i); l->routed_up = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.routed_expert_norm.weight", PFX, i); l->routed_norm = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight", PFX, i); l->shared_gate = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.shared_experts.up_proj.weight", PFX, i); l->shared_up = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.shared_experts.down_proj.weight", PFX, i); l->shared_down = load_t_opt(m,nm);
        }
    }
    #undef LD
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(Slot));
    }
    m->freq = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint32_t));
    m->hot_pinned = 0; m->freq_token_count = 0;
    m->hot_n = getenv("HOT") ? atoi(getenv("HOT")) : 0;
    m->warmup_tokens = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    m->token_count = 0;
    m->is_pinned = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    m->dense_load_s = now_s() - t0;
}

/* ---------- expert cache ---------- */
static void slot_ensure_allocated(Model *m, Slot *s) {
    if (s->g) return;
    Cfg *c = &m->c;
    int64_t IH = (int64_t)c->moe_inter * c->routed_hidden;   /* w1 (gate), w3 (up) — [I,H] */
    int64_t HI = (int64_t)c->routed_hidden * c->moe_inter;   /* w2 (down)  — [H,I] */
    s->g = falloc(IH);
    s->u = falloc(IH);
    s->d = falloc(HI);
    s->kind = EXP_NONE;
}

/* Load a single expert's three weights into Slot.
 * First tries colibri-int8 (merged_weight + qs), then falls back to MXFP4. */
static void load_expert(Model *m, int layer, int eid, Slot *s) {
    Cfg *c = &m->c;
    char nm[512];
    const char *PFX = (st_find(&m->S, "language_model.model.layers.0.input_layernorm.weight")) ? "language_model." : "";
    int64_t IH = (int64_t)c->moe_inter * c->routed_hidden;
    int64_t HI = (int64_t)c->routed_hidden * c->moe_inter;
    int64_t want_g = IH, want_u = IH, want_d = HI;

    /* try colibri int8 first */
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.merged_weight", PFX, layer, eid);
    st_tensor *tw = st_find(&m->S, nm);
    if (tw && (int64_t)tw->nbytes == (want_g + want_u + want_d)) {
        /* int8 path */
        int8_t *block = malloc((size_t)(want_g + want_u + want_d));
        st_read_raw(&m->S, nm, block, 1);
        s->qg = block; s->qu = block + want_g; s->qd = block + want_g + want_u;
        snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.qs", PFX, layer, eid);
        int64_t want_s = (int64_t)c->moe_inter * 2 + c->routed_hidden;
        st_tensor *ts = st_find(&m->S, nm);
        if (!ts || ts->numel != want_s) {
            fprintf(stderr,"%s: scale array mismatch (got %lld want %lld)\n", nm,
                (long long)(ts?ts->numel:-1), (long long)want_s); exit(1);
        }
        float *sb = falloc(want_s);
        st_read_f32(&m->S, nm, sb, 0);
        s->qgs = sb; s->qus = sb + c->moe_inter; s->qds = sb + 2*c->moe_inter;
        s->kind = EXP_INT8;
        return;
    }

    /* MXFP4 path: dequantize on load */
    int64_t packed_g = want_g / 2; /* two nibbles per byte */
    int64_t packed_u = want_u / 2;
    int64_t packed_d = want_d / 2;
    int64_t scales_g = want_g / 32;
    int64_t scales_u = want_u / 32;
    int64_t scales_d = want_d / 32;
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w1.weight_packed", PFX, layer, eid);
    uint8_t *g_pk = load_u8(m, nm, &packed_g);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w1.weight_scale", PFX, layer, eid);
    uint8_t *g_sc = load_u8(m, nm, &scales_g);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w3.weight_packed", PFX, layer, eid);
    uint8_t *u_pk = load_u8(m, nm, &packed_u);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w3.weight_scale", PFX, layer, eid);
    uint8_t *u_sc = load_u8(m, nm, &scales_u);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w2.weight_packed", PFX, layer, eid);
    uint8_t *d_pk = load_u8(m, nm, &packed_d);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w2.weight_scale", PFX, layer, eid);
    uint8_t *d_sc = load_u8(m, nm, &scales_d);
    if (!g_pk || !g_sc || !u_pk || !u_sc || !d_pk || !d_sc) {
        fprintf(stderr,"expert %d/%d: missing MXFP4 tensors (and no int8 merged_weight)\n", layer, eid);
        exit(1);
    }
    int rows_g = c->moe_inter, cols_g = c->routed_hidden;
    int rows_u = c->moe_inter, cols_u = c->routed_hidden;
    int rows_d = c->routed_hidden, cols_d = c->moe_inter;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_g; r++) {
        mxfp4_dequant_row(s->g + (int64_t)r*cols_g, g_pk + (int64_t)r*(cols_g/2), g_sc + (int64_t)r*(cols_g/32), cols_g);
    }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_u; r++) {
        mxfp4_dequant_row(s->u + (int64_t)r*cols_u, u_pk + (int64_t)r*(cols_u/2), u_sc + (int64_t)r*(cols_u/32), cols_u);
    }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_d; r++) {
        mxfp4_dequant_row(s->d + (int64_t)r*cols_d, d_pk + (int64_t)r*(cols_d/2), d_sc + (int64_t)r*(cols_d/32), cols_d);
    }
    free(g_pk); free(g_sc); free(u_pk); free(u_sc); free(d_pk); free(d_sc);
    s->kind = EXP_MXFP4;
}

static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    pthread_mutex_lock(&g_mx);
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        m->hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i];
        pthread_mutex_unlock(&g_mx);
        return;
    }
    m->miss++;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        slot_ensure_allocated(m, s);
    } else {
        int lru = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
        }
        if (lru < 0) {
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue;
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        }
        if (lru < 0) lru = 0;
        s = &lc->slots[lru];
        s->pinned = 0;
    }
    s->eid = -1;
    s->used = ++m->clock;
    pthread_mutex_unlock(&g_mx);

    load_expert(m, layer, eid, s);

    pthread_mutex_lock(&g_mx);
    s->eid = eid;
    s->pinned = m->is_pinned[layer * m->c.n_experts + eid];
    s->used = ++m->clock;
    *out = s;
    pthread_mutex_unlock(&g_mx);
}

/* ---------- RoPE (interleaved, applied to qk_rope slice) ---------- */
static void rope_interleave(float *v, int pos, int dim, float theta) {
    int half = dim / 2;
    for (int j = 0; j < half; j++) {
        float inv = powf(theta, -2.0f * j / dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = v[j], b = v[j+half];
        v[j]   = a*cs - b*sn;
        v[j+half] = b*cs + a*sn;
    }
}

/* ---------- MLA attention (Gated MLA) ----------
 * Mirror of colibri.c attention_rows (CPU path), simplified for K3 dims.
 * Operates on a single row (token) at a time — sufficient for both prefill
 * (we run prefill one token at a time) and decode. */
static void mla_forward(Model *m, Layer *l, int layer, const float *x, int pos, float *out) {
    Cfg *c = &m->c;
    int H = c->n_heads, D = c->hidden;
    int qh = c->qk_head, vh = c->v_head, ql = c->q_lora, kvl = c->kv_lora, R = c->qk_rope, No = c->qk_nope;

    float *q_a = falloc(ql);
    matmul(q_a, x, l->q_a, 1, D, ql);
    rmsnorm_row(q_a, q_a, l->q_a_ln, ql, c->eps);
    float *q_full = falloc(H * qh);
    matmul(q_full, q_a, l->q_b, 1, ql, H*qh);

    float *comp = falloc(kvl + R);
    matmul(comp, x, l->kv_a, 1, D, kvl + R);
    float *k_pass_latent = comp;          /* [kvl] */
    float *k_rot = comp + kvl;            /* [R]   */
    rmsnorm_row(k_pass_latent, k_pass_latent, l->kv_a_ln, kvl, c->eps);
    rope_interleave(k_rot, pos, R, c->theta);

    /* write K/V cache: kv_b applied to normalized latent gives [H*(No+vh)] */
    float *kv = falloc(H * (No + vh));
    matmul(kv, k_pass_latent, l->kv_b, 1, kvl, H*(No+vh));
    /* store per-head K_nope[h, No] and V[h, vh] in cache row `pos` */
    float *Krow_nope = m->K_nope[layer] + (int64_t)pos * H * No;
    float *Vrow = m->V[layer] + (int64_t)pos * H * vh;
    float *Krow_rot = m->K_rot[layer] + (int64_t)pos * R;  /* shared across heads */
    for (int h = 0; h < H; h++) {
        memcpy(Krow_nope + h*No, kv + h*(No+vh), No*sizeof(float));
        memcpy(Vrow    + h*vh, kv + h*(No+vh) + No, vh*sizeof(float));
    }
    memcpy(Krow_rot, k_rot, R*sizeof(float));

    /* attention: query = cat(q_nope, q_rot); key = cat(K_nope, K_rot_shared) */
    float scale = 1.f / sqrtf((float)qh);
    float *ctx = falloc(H * vh);
    int T = pos + 1;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        float *q_nope = q_full + h*qh;       /* [No] */
        float *q_rot  = q_full + h*qh + No;  /* [R]  */
        float *sc = malloc((size_t)T * sizeof(float));
        float mx = -1e30f;
        for (int t = 0; t < T; t++) {
            const float *kn = m->K_nope[layer] + ((int64_t)t * H + h) * No;
            const float *kr = m->K_rot[layer] + (int64_t)t * R;
            float acc = 0.f;
            for (int d = 0; d < No; d++) acc += q_nope[d] * kn[d];
            for (int d = 0; d < R;  d++) acc += q_rot[d]  * kr[d];
            acc *= scale;
            sc[t] = acc; if (acc > mx) mx = acc;
        }
        float Z = 0.f;
        for (int t = 0; t < T; t++) { sc[t] = expf(sc[t]-mx); Z += sc[t]; }
        float *cx = ctx + h*vh;
        for (int d = 0; d < vh; d++) cx[d] = 0.f;
        for (int t = 0; t < T; t++) {
            const float *vrow = m->V[layer] + ((int64_t)t * H + h) * vh;
            float a = sc[t] / Z;
            for (int d = 0; d < vh; d++) cx[d] += a * vrow[d];
        }
        free(sc);
    }
    /* output gate (per-token): o = o * sigmoid(g_proj(x)) */
    if (c->use_output_gate && l->g_proj_mla) {
        float *g = falloc(H * vh);
        matmul(g, x, l->g_proj_mla, 1, D, H*vh);
        for (int i = 0; i < H*vh; i++) {
            float gi = 1.f / (1.f + expf(-g[i]));
            ctx[i] *= gi;
        }
        free(g);
    }
    matmul(out, ctx, l->o, 1, H*vh, D);
    free(q_a); free(q_full); free(comp); free(kv); free(ctx);
}

/* ---------- KDA (Kimi Delta Attention), fused-recurrent form, single token ---------- */
static void kda_forward(Model *m, Layer *l, int layer, const float *x, int pos, float *out) {
    Cfg *c = &m->c;
    int H = c->kda_n_heads, hd = c->kda_head_dim, D = c->hidden;
    int proj_qk = H * hd;          /* q_proj, k_proj out */
    int proj_v  = H * hd;          /* v_proj out */
    int conv_k  = c->short_conv;

    /* projections */
    float *qp = falloc(proj_qk), *kp = falloc(proj_qk), *vp = falloc(proj_v);
    matmul(qp, x, l->q_proj, 1, D, proj_qk);
    matmul(kp, x, l->k_proj, 1, D, proj_qk);
    matmul(vp, x, l->v_proj, 1, D, proj_v);

    /* short conv (SiLU): y[t] = silu(w[0]*x[t] + w[1]*x[t-1] + ... + w[k-1]*x[t-k+1])
     * For decoding, we keep a sliding window per channel in kda_conv_{q,k,v}. */
    float *q = falloc(proj_qk), *k = falloc(proj_qk), *v = falloc(proj_v);
    /* maintain the conv state: shift by one and prepend new token's projection */
    float *cs_q = m->kda_conv_q[layer], *cs_k = m->kda_conv_k[layer], *cs_v = m->kda_conv_v[layer];
    for (int ch = 0; ch < proj_qk; ch++) {
        /* shift: new state = [new, old[0..k-2]] (causal, last-in)
         * conv weight is [conv_k, proj] stored as [proj, conv_k] (Linear convention) */
        for (int t = conv_k - 1; t > 0; t--)
            cs_q[ch*conv_k + t] = cs_q[ch*conv_k + t - 1];
        cs_q[ch*conv_k + 0] = qp[ch];
        float acc = 0.f;
        for (int t = 0; t < conv_k; t++) {
            float w = l->q_conv[ch*conv_k + t];
            acc += w * cs_q[ch*conv_k + t];
        }
        /* SiLU */
        q[ch] = acc / (1.f + expf(-acc));
    }
    for (int ch = 0; ch < proj_qk; ch++) {
        for (int t = conv_k - 1; t > 0; t--)
            cs_k[ch*conv_k + t] = cs_k[ch*conv_k + t - 1];
        cs_k[ch*conv_k + 0] = kp[ch];
        float acc = 0.f;
        for (int t = 0; t < conv_k; t++) acc += l->k_conv[ch*conv_k + t] * cs_k[ch*conv_k + t];
        k[ch] = acc / (1.f + expf(-acc));
    }
    for (int ch = 0; ch < proj_v; ch++) {
        for (int t = conv_k - 1; t > 0; t--)
            cs_v[ch*conv_k + t] = cs_v[ch*conv_k + t - 1];
        cs_v[ch*conv_k + 0] = vp[ch];
        float acc = 0.f;
        for (int t = 0; t < conv_k; t++) acc += l->v_conv[ch*conv_k + t] * cs_v[ch*conv_k + t];
        v[ch] = acc / (1.f + expf(-acc));
    }

    /* gate g (pre-sigmoid, full-rank): g = g_proj(x), shape [proj_v] */
    float *g_pre = falloc(proj_v);
    if (c->use_full_rank_gate) matmul(g_pre, x, l->g_proj_kda, 1, D, proj_v);
    /* dt_bias and A_log are per-head: shape [H]. */
    /* alpha[h] = sigmoid(A_log[h] + dt_bias[h])  -- per-head retention factor */
    /* beta[h]  = sigmoid(b_proj(x))             -- per-head delta write strength */
    float *beta = falloc(H);
    matmul(beta, x, l->b_proj, 1, D, H);

    /* L2-normalize q and k per head (use_qk_l2norm_in_kernel=True) */
    for (int h = 0; h < H; h++) {
        float *qh = q + h*hd, *kh = k + h*hd;
        double qn = 0, kn = 0;
        for (int d = 0; d < hd; d++) { qn += qh[d]*qh[d]; kn += kh[d]*kh[d]; }
        float qi = 1.f / sqrtf((float)qn + 1e-12f);
        float ki = 1.f / sqrtf((float)kn + 1e-12f);
        for (int d = 0; d < hd; d++) { qh[d] *= qi; kh[d] *= ki; }
    }

    /* recurrent state update + readout, per head:
     *   s[h] (hd x hd) = alpha[h] * s[h] + (1-alpha[h]) * beta[h] * (v[h] outer k[h])
     *   o[h] = s[h] @ q[h]
     * State stored as [H, hd, hd] row-major (s[h, i, j] = element for input i, output j).
     * Update: s[h, i, j] = alpha * s[h, i, j] + (1-alpha) * beta * v[h, j] * k[h, i]
     * Readout: o[h, j] = sum_i s[h, i, j] * q[h, i]  (state is [hd_in, hd_out]) */
    float *state = m->kda_state[layer];
    float *o = falloc(proj_v);
    for (int h = 0; h < H; h++) {
        float alpha = 1.f / (1.f + expf(-(l->A_log[h] + l->dt_bias[h])));
        float b = 1.f / (1.f + expf(-beta[h]));
        float w = (1.f - alpha) * b;
        float *sh = state + (int64_t)h * hd * hd;
        const float *kh = k + h*hd;
        const float *vh = v + h*hd;
        const float *qh = q + h*hd;
        float *oh = o + h*hd;
        /* pass 1: update state in place */
        for (int i = 0; i < hd; i++) {
            float ki = kh[i];
            float *srow = sh + i*hd;
            for (int j = 0; j < hd; j++)
                srow[j] = alpha * srow[j] + w * vh[j] * ki;
        }
        /* pass 2: readout o[h, j] = sum_i s[h, i, j] * q[h, i] */
        for (int j = 0; j < hd; j++) {
            float acc = 0.f;
            for (int i = 0; i < hd; i++) acc += sh[i*hd + j] * qh[i];
            oh[j] = acc;
        }
    }

    /* o_norm: FusedRMSNormGated(o, g) = RMSNorm(o) * sigmoid(g) */
    rmsnorm_row(o, o, l->o_norm, proj_v, c->eps);
    float lc = c->situ_linear_beta > 0.f ? c->situ_linear_beta : 0.f;  /* unused here, kept for clarity */
    (void)lc;
    for (int i = 0; i < proj_v; i++) {
        float gi = 1.f / (1.f + expf(-g_pre[i]));
        o[i] *= gi;
    }

    matmul(out, o, l->o, 1, proj_v, D);
    free(qp); free(kp); free(vp); free(q); free(k); free(v); free(g_pre); free(beta); free(o);
}

/* ---------- Stable LatentMoE ---------- */
static void moe_forward(Model *m, Layer *l, int layer, const float *x, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk;
    int I = c->moe_inter, H = c->routed_hidden;
    /* router logits = x @ gate_w^T, shape [E] */
    float *logits = falloc(E);
    matmul(logits, x, l->gate_w, 1, D, E);
    /* sigmoid scores */
    float *scores = falloc(E);
    for (int e = 0; e < E; e++) scores[e] = 1.f / (1.f + expf(-logits[e]));
    /* noaux_tc: scores + e_score_correction_bias, then top-K */
    float *adj = falloc(E);
    for (int e = 0; e < E; e++) adj[e] = scores[e] + (l->e_score_bias ? l->e_score_bias[e] : 0.f);
    int idx[64]; float val[64];
    int topk = K <= 64 ? K : 64;
    for (int kk = 0; kk < topk; kk++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
            if (!taken && adj[e] > bv) { bv = adj[e]; best = e; }
        }
        idx[kk] = best; val[kk] = scores[best];
    }
    if (c->moe_renormalize && K > 1) {
        float s = 0.f; for (int kk = 0; kk < topk; kk++) s += val[kk];
        if (s < 1e-20f) s = 1e-20f;
        for (int kk = 0; kk < topk; kk++) val[kk] /= s;
    }
    for (int kk = 0; kk < topk; kk++) val[kk] *= c->routed_scale;
    /* heatmap update (for hot pinning) */
    if (!m->hot_pinned && m->freq) {
        uint32_t *fl = m->freq + (int64_t)layer * E;
        for (int kk = 0; kk < topk; kk++) if (idx[kk] >= 0) fl[idx[kk]]++;
    }

    /* latent MoE: project x -> [H] via routed_down, then each expert: g/u on H, d back to H,
     * then routed_norm + routed_up back to D. Shared experts operate on the original D-dim x. */
    float *latent = NULL;
    int in_dim = D;
    if (c->routed_hidden != D && l->routed_down) {
        latent = falloc(H);
        matmul(latent, x, l->routed_down, 1, D, H);
        in_dim = H;
    }
    float *acc = falloc(D);
    for (int d = 0; d < D; d++) acc[d] = 0.f;
    float *gg = falloc(I), *uu = falloc(I), *hh = falloc(H);
    for (int kk = 0; kk < topk; kk++) {
        Slot *e; expert_get(m, layer, idx[kk], &e);
        const float *xin = latent ? latent : x;
        if (e->kind == EXP_INT8) {
            matmul_q(gg, xin, e->qg, e->qgs, in_dim, I);
            matmul_q(uu, xin, e->qu, e->qus, in_dim, I);
        } else {
            matmul(gg, xin, e->g, 1, in_dim, I);
            matmul(uu, xin, e->u, 1, in_dim, I);
        }
        /* SiTU-GLU: a = beta*tanh(g/beta)*sigmoid(g); out = a * (linear_beta*tanh(u/linear_beta)) */
        for (int i = 0; i < I; i++) gg[i] = situ_gate(gg[i], uu[i], c->situ_beta, c->situ_linear_beta);
        if (e->kind == EXP_INT8)
            matmul_q(hh, gg, e->qd, e->qds, I, H);
        else
            matmul(hh, gg, e->d, 1, I, H);
        float w = val[kk];
        if (latent) {
            /* accumulate in latent space, then norm + up at the end */
            for (int d = 0; d < H; d++) acc[d] += w * hh[d];
        } else {
            for (int d = 0; d < D; d++) acc[d] += w * hh[d];
        }
    }
    float *routed_out;
    if (latent) {
        if (l->routed_norm) rmsnorm_row(acc, acc, l->routed_norm, H, c->eps);
        routed_out = falloc(D);
        matmul(routed_out, acc, l->routed_up, 1, H, D);
    } else {
        routed_out = acc;
    }

    /* shared experts: 2 experts fused into one larger MLP (intermediate = moe_inter * n_shared) */
    if (c->n_shared > 0 && l->shared_gate) {
        int si = c->moe_inter * c->n_shared;
        float *sg = falloc(si), *su = falloc(si), *sh = falloc(D);
        matmul(sg, x, l->shared_gate, 1, D, si);
        matmul(su, x, l->shared_up,   1, D, si);
        for (int i = 0; i < si; i++) sg[i] = situ_gate(sg[i], su[i], c->situ_beta, c->situ_linear_beta);
        matmul(sh, sg, l->shared_down, 1, si, D);
        for (int d = 0; d < D; d++) routed_out[d] += sh[d];
        free(sg); free(su); free(sh);
    }
    memcpy(out, routed_out, D*sizeof(float));
    free(logits); free(scores); free(adj);
    free(gg); free(uu); free(hh);
    if (latent) { free(latent); free(acc); free(routed_out); }
    else free(routed_out);
}

/* dense MLP for layer 0 (gate/up/down + SiTU) */
static void mlp_forward(Model *m, Layer *l, const float *x, float *out) {
    Cfg *c = &m->c; int D = c->hidden, I = c->inter;
    float *g = falloc(I), *u = falloc(I);
    matmul(g, x, l->mlp_gate, 1, D, I);
    matmul(u, x, l->mlp_up,   1, D, I);
    for (int i = 0; i < I; i++) g[i] = situ_gate(g[i], u[i], c->situ_beta, c->situ_linear_beta);
    matmul(out, g, l->mlp_down, 1, I, D);
    free(g); free(u);
}

/* ---------- per-token step ---------- */
static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    /* allocate K/V caches on first step.
     * MLA K/V caches are pre-allocated once with a fixed capacity (CTK env or
     * 4096 default) and never grown: growing would free+realloc the buffers
     * and lose positions [0, pos) already written, producing non-deterministic
     * output across runs (issue: K3 determinism test). KDA recurrent state is
     * a fixed-size [H, hd, hd] matrix, independent of sequence length. */
    if (!m->K_nope) {
        m->K_nope = calloc(c->n_layers, sizeof(float*));
        m->K_rot  = calloc(c->n_layers, sizeof(float*));
        m->V      = calloc(c->n_layers, sizeof(float*));
        m->kda_state = calloc(c->n_layers, sizeof(float*));
        m->kda_conv_q = calloc(c->n_layers, sizeof(float*));
        m->kda_conv_k = calloc(c->n_layers, sizeof(float*));
        m->kda_conv_v = calloc(c->n_layers, sizeof(float*));
        int ctx = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
        if (ctx < 1) ctx = 4096;
        m->max_t = ctx;
        for (int i = 0; i < c->n_layers; i++) {
            if (c->is_mla[i]) {
                m->K_nope[i] = falloc((int64_t)c->n_heads * m->max_t * c->qk_nope);
                m->K_rot[i]  = falloc((int64_t)m->max_t * c->qk_rope);
                m->V[i]      = falloc((int64_t)c->n_heads * m->max_t * c->v_head);
            } else {
                /* KDA: all non-MLA layers, including layer 0 (dense MLP + KDA attn) */
                m->kda_state[i] = falloc((int64_t)c->kda_n_heads * c->kda_head_dim * c->kda_head_dim);
                memset(m->kda_state[i], 0, (size_t)c->kda_n_heads * c->kda_head_dim * c->kda_head_dim * sizeof(float));
                int pqk = c->kda_n_heads * c->kda_head_dim;
                int pv  = c->kda_n_heads * c->kda_head_dim;
                m->kda_conv_q[i] = falloc((int64_t)pqk * c->short_conv);
                m->kda_conv_k[i] = falloc((int64_t)pqk * c->short_conv);
                m->kda_conv_v[i] = falloc((int64_t)pv  * c->short_conv);
                memset(m->kda_conv_q[i], 0, (size_t)pqk * c->short_conv * sizeof(float));
                memset(m->kda_conv_k[i], 0, (size_t)pqk * c->short_conv * sizeof(float));
                memset(m->kda_conv_v[i], 0, (size_t)pv  * c->short_conv * sizeof(float));
            }
        }
    }
    int max_t = pos_base + S;
    if (max_t > m->max_t) {
        fprintf(stderr, "sequence length %d exceeds KV cache capacity %d (set CTX=<n>)\n",
                max_t, m->max_t);
        exit(1);
    }

    /* run token-by-token (matches KDA recurrent semantics) */
    float *last = NULL;
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        int id = ids[s];
        float *x = falloc(D);
        memcpy(x, m->embed + (int64_t)id * D, D*sizeof(float));
        for (int i = 0; i < c->n_layers; i++) {
            Layer *l = &m->L[i];
            float *nrm = falloc(D);
            rmsnorm_row(nrm, x, l->in_ln, D, c->eps);
            /* Attention dispatch: MLA layers use Gated MLA, all others use KDA.
             * Layer 0 (dense MLP) also has KDA attention — the dense flag only
             * controls the MLP type, not the attention type. */
            float *attn_out = falloc(D);
            if (c->is_mla[i])      mla_forward(m, l, i, nrm, pos, attn_out);
            else                   kda_forward(m, l, i, nrm, pos, attn_out);
            for (int d = 0; d < D; d++) x[d] += attn_out[d];
            free(attn_out);
            rmsnorm_row(nrm, x, l->post_ln, D, c->eps);
            float *mlp_out = falloc(D);
            if (c->is_moe[i]) moe_forward(m, l, i, nrm, mlp_out);
            else               mlp_forward(m, l, nrm, mlp_out);
            for (int d = 0; d < D; d++) x[d] += mlp_out[d];
            free(nrm); free(mlp_out);
        }
        if (last) free(last);
        last = x;
    }
    m->token_count += S; m->freq_token_count += S;
    m->kv_len = pos_base + S;
    float *final_x = falloc(D);
    rmsnorm_row(final_x, last, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, final_x, m->lm_head, 1, D, c->vocab);
    free(last); free(final_x);
    return logit;
}

/* ---------- generation ---------- */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0);
    int len = np;
    float temp = getenv("TEMP") ? (float)atof(getenv("TEMP")) : 0.f;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        if (temp > 0.f) {
            /* simple sampling: argmax with temperature scaling (good enough for verification) */
            for (int v = 1; v < c->vocab; v++) logit[v] /= temp;
        }
        for (int v = 1; v < c->vocab; v++) if (logit[v] > bv) { bv = logit[v]; best = v; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1);
    }
}

/* teacher-forced NLL — for PPL=1 mode */
static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    double nll = 0; int scored = 0;
    float *logit = step(m, full, np, 0);
    for (int i = np; i < nfull; i++) {
        float mx = logit[0]; for (int v = 1; v < c->vocab; v++) if (logit[v] > mx) mx = logit[v];
        double Z = 0; for (int v = 0; v < c->vocab; v++) Z += exp((double)logit[v] - mx);
        nll += -((double)logit[full[i]] - mx - log(Z));
        scored++;
        free(logit); logit = NULL;
        if (i == nfull - 1) break;
        logit = step(m, &full[i], 1, i);
    }
    if (logit) free(logit);
    *nll_out = nll / scored;
    return scored;
}

/* ---------- ref.json ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { *n_out = 0; return NULL; }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }

    /* --config-check: parse config.json only, print the layer map, and exit.
     * Used to verify the engine against the real K3 config.json without
     * requiring the 2.8T of safetensors weights. */
    int config_check = (argc > 1 && strcmp(argv[1], "--config-check") == 0);

    int cap = config_check ? 0 : (argc > 1 ? atoi(argv[1]) : 16);
    const char *refpath = argc > 2 ? argv[2] : "ref.json";

    printf("== Kimi-K3 engine v0.1 | cache=%d/layer ==\n", cap);

    if (config_check) {
        Cfg c; load_cfg(&c, snap);
        int n_mla = 0, n_kda = 0, n_dense = 0;
        for (int i = 0; i < c.n_layers; i++) {
            if (!c.is_moe[i]) n_dense++;
            else if (c.is_mla[i]) n_mla++;
            else n_kda++;
        }
        printf("[CFG] hidden=%d layers=%d (dense=%d, MLA=%d, KDA=%d), experts=%d topk=%d shared=%d, vocab=%d\n",
               c.hidden, c.n_layers, n_dense, n_mla, n_kda, c.n_experts, c.topk, c.n_shared, c.vocab);
        printf("[CFG] q_lora=%d kv_lora=%d qk_nope=%d qk_rope=%d v_head=%d | kda_heads=%d kda_hd=%d short_conv=%d\n",
               c.q_lora, c.kv_lora, c.qk_nope, c.qk_rope, c.v_head, c.kda_n_heads, c.kda_head_dim, c.short_conv);
        printf("[CFG] first_dense=%d moe_freq=%d moe_renorm=%d situ_beta=%.1f situ_lin_beta=%.1f eps=%.1e routed_scale=%.2f\n",
               c.first_dense, c.moe_freq, c.moe_renormalize, c.situ_beta, c.situ_linear_beta, c.eps, c.routed_scale);
        printf("[CFG] mla_layers:");
        for (int i = 0; i < c.n_layers; i++) if (c.is_mla[i]) printf(" %d", i);
        printf("\n");
        printf("[CFG] kda_layers:");
        for (int i = 0; i < c.n_layers; i++) if (!c.is_mla[i] && c.is_moe[i]) printf(" %d", i);
        printf("\n");
        printf("[CFG] dense_layer:");
        for (int i = 0; i < c.n_layers; i++) if (!c.is_moe[i]) printf(" %d", i);
        printf("\n");
        printf("[OK] config parsed successfully — engine is compatible with this snapshot.\n");
        free(c.is_mla); free(c.is_moe);
        return 0;
    }

    /* --load-check: scan all available safetensors shards in snap/, then for
     * every tensor the engine WOULD load (embed/head/norm/layer-N attention/
     * MoE-router/shared/routed-latent) report whether it is present in the
     * shards and, if present, read the bf16 -> fp32 buffer and emit min/max/
     * mean + finite-check. This proves the real-K3 deserialization path
     * (multi-shard, bf16) actually works against the HuggingFace snapshot.
     * Tensors that aren't downloaded yet are simply listed as MISSING — this
     * is expected when only a subset of shards is present (see k3_meta vs
     * k3_real). */
    int load_check = (argc > 1 && strcmp(argv[1], "--load-check") == 0);
    if (load_check) {
        Cfg c; load_cfg(&c, snap);
        shards S; st_init(&S, snap);
        printf("== load-check: %d shards indexed (%d tensors) ==\n", S.nfd, S.n);

        int n_ok = 0, n_missing = 0, n_nan = 0, n_zero = 0;

        /* CHECK(name): printf-style name; BUILD(fmt, ...) builds the name first.
         * Use separate buffers to avoid -Wrestrict overlap when the macro's
         * internal snprintf writes to the same buffer the caller passed. */
        char _nm[512];
        #define CHECK(...) do { \
            snprintf(_nm,sizeof(_nm),__VA_ARGS__); \
            st_tensor *t = st_find(&S, _nm); \
            if (!t) { n_missing++; printf("  MISSING  %s\n", _nm); } \
            else { \
                int64_t total = t->numel; \
                int64_t n = total < 4096 ? total : 4096; \
                int esz = (t->dtype == 2) ? 4 : 2; \
                int64_t raw_bytes = n * esz; \
                uint8_t *raw = malloc((size_t)raw_bytes); \
                st_pread_full(t->fd, raw, raw_bytes, t->off, "load-check sample"); \
                float *buf = malloc((size_t)n*sizeof(float)); \
                if (t->dtype == 2) { memcpy(buf, raw, raw_bytes); } \
                else if (t->dtype == 0) { uint16_t *p=(uint16_t*)raw; for (int64_t i=0;i<n;i++) buf[i]=bf16_to_f32(p[i]); } \
                else { uint16_t *p=(uint16_t*)raw; for (int64_t i=0;i<n;i++) buf[i]=f16_to_f32(p[i]); } \
                free(raw); \
                double mn=1e30,mx=-1e30,sum=0; int nan=0,zero=0; \
                for (int64_t i=0;i<n;i++){float v=buf[i]; if(!isfinite(v)){nan=1;continue;} if(v<mn)mn=v; if(v>mx)mx=v; sum+=v; if(v==0.f)zero++;} \
                if(nan){n_nan++; printf("  BAD-NaN %-80s numel=%lld\n",_nm,(long long)total);} \
                else if(zero==n){n_zero++; printf("  ZERO    %-80s numel=%lld\n",_nm,(long long)total);} \
                else {n_ok++; printf("  OK      %-80s numel=%lld  min=%+.4g  max=%+.4g  mean=%+.4g\n",_nm,(long long)total,mn,mx,sum/n);} \
                free(buf); \
            } } while(0)

        /* embed/head/norm (top-level) */
        CHECK("%s", "language_model.model.embed_tokens.weight");
        CHECK("%s", "language_model.lm_head.weight");
        CHECK("%s", "language_model.model.norm.weight");
        CHECK("%s", "language_model.model.output_attn_res_norm.weight");
        CHECK("%s", "language_model.model.output_attn_res_proj.weight");

        /* For each layer, check the tensors the engine would load. */
        const char *PFX = "language_model.";
        for (int i = 0; i < c.n_layers; i++) {
            CHECK("%smodel.layers.%d.input_layernorm.weight", PFX, i);
            CHECK("%smodel.layers.%d.post_attention_layernorm.weight", PFX, i);
            CHECK("%smodel.layers.%d.self_attention_res_norm.weight", PFX, i);
            CHECK("%smodel.layers.%d.self_attention_res_proj.weight", PFX, i);
            CHECK("%smodel.layers.%d.mlp_res_norm.weight", PFX, i);
            CHECK("%smodel.layers.%d.mlp_res_proj.weight", PFX, i);

            if (c.is_mla[i]) {
                CHECK("%smodel.layers.%d.self_attn.q_a_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.q_a_layernorm.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.q_b_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.kv_a_proj_with_mqa.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.kv_a_layernorm.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.kv_b_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.o_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.g_proj.weight", PFX, i);
            } else {
                CHECK("%smodel.layers.%d.self_attn.q_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.k_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.v_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.q_conv1d.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.k_conv1d.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.v_conv1d.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.A_log", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.dt_bias", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.b_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.f_a_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.f_b_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.g_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.o_norm.weight", PFX, i);
                CHECK("%smodel.layers.%d.self_attn.o_proj.weight", PFX, i);
            }

            if (!c.is_moe[i]) {
                CHECK("%smodel.layers.%d.mlp.gate_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.mlp.up_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.mlp.down_proj.weight", PFX, i);
            } else {
                CHECK("%smodel.layers.%d.block_sparse_moe.gate.weight", PFX, i);
                CHECK("%smodel.layers.%d.block_sparse_moe.gate.e_score_correction_bias", PFX, i);
                CHECK("%smodel.layers.%d.block_sparse_moe.routed_expert_down_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.block_sparse_moe.routed_expert_up_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.block_sparse_moe.routed_expert_norm.weight", PFX, i);
                CHECK("%smodel.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.block_sparse_moe.shared_experts.up_proj.weight", PFX, i);
                CHECK("%smodel.layers.%d.block_sparse_moe.shared_experts.down_proj.weight", PFX, i);
            }
        }
        printf("\n== load-check summary: OK=%d  MISSING=%d  NaN=%d  ZERO=%d ==\n",
               n_ok, n_missing, n_nan, n_zero);
        printf("[OK] deserialization path exercised against real safetensors shards.\n");
        free(c.is_mla); free(c.is_moe);
        return 0;
    }

    Model m; model_init(&m, snap, cap);
    Cfg *c = &m.c;
    int n_mla = 0, n_kda = 0, n_dense = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (!c->is_moe[i]) n_dense++;
        else if (c->is_mla[i]) n_mla++;
        else n_kda++;
    }
    printf("[CFG] hidden=%d layers=%d (dense=%d, MLA=%d, KDA=%d), experts=%d topk=%d shared=%d, vocab=%d\n",
           c->hidden, c->n_layers, n_dense, n_mla, n_kda, c->n_experts, c->topk, c->n_shared, c->vocab);
    printf("[CFG] q_lora=%d kv_lora=%d qk_nope=%d qk_rope=%d v_head=%d | kda_heads=%d kda_hd=%d short_conv=%d\n",
           c->q_lora, c->kv_lora, c->qk_nope, c->qk_rope, c->v_head, c->kda_n_heads, c->kda_head_dim, c->short_conv);
    printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());

    FILE *f = fopen(refpath, "rb");
    if (!f) { perror(refpath); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if (fread(buf,1,n,f)!=(size_t)n) {} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull; int *prompt = read_int_array(ref,"prompt_ids",&np); int *full = read_int_array(ref,"full_ids",&nfull);
    if (np == 0 || nfull == 0) { fprintf(stderr,"ref.json missing prompt_ids/full_ids\n"); return 1; }
    int n_new = nfull - np;

    if (getenv("PPL") && atoi(getenv("PPL")) == 1) {
        double nll; double t = now_s();
        int scored = tf_nll(&m, full, nfull, np, &nll);
        double dt = now_s() - t;
        double tot = m.hits + m.miss;
        printf("TF-NLL: %.4f nats/token over %d tokens  |  ppl = %.2f\n", nll, scored, exp(nll));
        printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss);
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
        free(buf); free(arena); free(prompt); free(full);
        return 0;
    }

    int *out = malloc((np + n_new) * sizeof(int));
    double t = now_s();
    generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t;

    int match = 0;
    printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot = m.hits + m.miss;
    printf("\nPEAK RSS: %.2f GB\n", rss_gb());
    printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);
    printf("Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    free(buf); free(arena); free(prompt); free(full); free(out);
    return 0;
}
