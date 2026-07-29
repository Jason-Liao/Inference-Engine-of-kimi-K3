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
 * KDA (Kimi Delta Attention) is implemented in two mathematically-equivalent
 * forms (k3_tech_report §3.2): the fused-recurrent form (token-by-token, used
 * for decode and as the KDA_CHUNK=1 fallback) and the chunkwise parallel form
 * (default for prefill, C=64 tokens per chunk). Prefill runs layer-major: all
 * S tokens are embedded once, then each layer processes the full sequence
 * (KDA layers via kda_forward_chunk, MLA layers token-by-token). Decode (S=1)
 * always uses the recurrent path. Set KDA_CHUNK=1 to force the recurrent path
 * for prefill (used for equivalence testing).
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
 *   KDA_CHUNK=N          KDA chunkwise prefill chunk size (default 64; =1 uses
 *                        the original token-by-token recurrent path)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>        /* KVSAVE: open/pread for async KV load */
#endif
#ifdef _OPENMP
#include <omp.h>           /* OMP_HOT: omp_set_num_threads for the hot-thread pool */
#endif
#ifdef __linux__
#include <dlfcn.h>         /* COLI_NUMA: dlopen libnuma.so.1 (no -lnuma link-time dep) */
#endif
#include "st.h"
#include "json.h"
#include "compat.h"
#include "tok.h"
#include "grammar.h"
#include "schema_gbnf.h"

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
    /* bf16 native residency: when set, the matmul-bound dense weights above
     * (q_a, q_b, kv_a, kv_b, o, g_proj, q_proj, mlp_gate/up/down, shared_gate/
     * up/down, routed_down/up, gate_w, attn_res_proj, mlp_res_proj) are stored
     * as raw bf16 bytes (2B/elem, half of fp32) and the float* is reinterpreted
     * as uint16_t* by matmul_bf16. Norm vectors and small params (A_log,
     * dt_bias, the conv1d weights, e_score_bias) are always fp32 -- they are
     * read directly as float and are negligible in size. */
    uint8_t bf16_resident;
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
    /* PIPE/PILOT async-load coordination: while a background worker is loading
     * this slot, in_use=1 and eid=-1; loading_eid holds the target expert id so
     * a concurrent expert_reserve() can recognise an in-flight load for the
     * SAME expert and wait on it instead of starting a duplicate load. */
    int in_use;
    int loading_eid;
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
    uint8_t bf16_resident;   /* global mode: embed + lm_head + per-layer dense weights */
    double dense_load_s;
    /* expert heatmap for HOT pinning */
    uint32_t *freq;
    int freq_token_count, hot_pinned, hot_n, warmup_tokens, token_count;
    uint8_t *is_pinned;
    /* KVSAVE: warm-start state (loaded from disk before generate() runs).
     * kv_loaded=1 means the KV/KDA caches were restored from a KVSAVE file
     * and the prefill step may be skipped (saved_logits holds the prefill
     * logits so the first token can be picked without recomputing). */
    uint8_t kv_loaded;
    float *saved_logits;
} Model;

static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;

/* ---- performance optimizations (PIPE / PILOT / DSA) ----
 * All three default OFF, so the engine is byte-identical to the original
 * serial path when none of the env vars are set.
 *   PIPE=1   : async expert-load pipeline — miss loads run on I/O worker
 *              threads and overlap the matmul of already-resident experts.
 *   PILOT=1  : cross-layer expert prefetch — after layer L's MoE, predict
 *              layer L+1's routed experts from the current hidden state and
 *              preload them on a background thread so they are cache hits by
 *              the time layer L+1 runs.
 *   DSA=1    : Lightning Indexer — in the prefill path (S>1) batch-route all
 *              S tokens first, collect the unique expert set, then load each
 *              expert ONCE and apply it to every token that selected it
 *              (batch matmul), avoiding redundant loads when tokens share
 *              experts. Defined here (top-level) so the expert-cache and
 *              worker-pool code can reference them. */
static int g_pipe = 0;        /* PIPE env (default 0 = off) */
static int g_pipe_nw = 4;     /* PIPE_WORKERS env (default 4 I/O threads) */
static int g_pilot = 0;       /* PILOT env (default 0 = off) */
static int g_pilot_k = 0;     /* PILOT_K env (default 0 = use config topk) */
static int g_dsa = 0;         /* DSA env (default 0 = off) */

/* ---- KVSAVE: KV cache 落盘 + 恢复 (default off) ----
 * KVSAVE=<dir>  serialize KV/KDA state to <dir>/kv_<session_id>.bin right
 *               after prefill; load it back at startup to skip re-prefill
 *               (warm start). The session_id is an FNV-1a hash of the prompt
 *               token ids, so the same prompt resumes the same KV file.
 * RSS_LIMIT=<GB>  background thread monitors RSS every 5s; when RSS exceeds
 *                 the limit it requests eviction of non-pinned experts (the
 *                 main thread performs the actual free at the next safe point
 *                 between tokens). At 95% of the limit it calls abort() to
 *                 pre-empt the OOM killer. Default 0 = off (no thread). */
static const char *g_kvsave_dir = NULL;
static double g_rss_limit_gb = 0.0;
static _Atomic int g_rss_evict_needed = 0;   /* guard -> main: please evict */
static _Atomic int g_rss_guard_stop = 0;
static pthread_t g_rss_guard_th;

/* ==================== DUAL-SSD: two model copies, two drives ====================
 * DUAL_SSD=<dir2> registers a second read-only copy of the model on another
 * drive via st_mirror_init (already in st.h). Expert reads are then routed
 * between the primary and the mirror based on in-flight load, so two NVMe
 * drives read in parallel. When DUAL_SSD is unset, g_mirror=0 and every read
 * hits the primary — byte-identical to the original single-drive path. */
static int g_mirror = 0;                        /* 1 = mirror active (at least one shard accepted) */
static _Atomic int64_t g_mir_inflight[2] = {0, 0}; /* in-flight expert loads per drive: [0] primary, [1] mirror */

/* route an expert load to the less-loaded drive (0=primary, 1=mirror).
 * When the mirror is inactive, always returns 0 (primary). */
static inline int expert_route(int layer, int eid) {
    if (!g_mirror) return 0;
    (void)layer; (void)eid;
    int64_t a = atomic_load_explicit(&g_mir_inflight[0], memory_order_relaxed);
    int64_t b = atomic_load_explicit(&g_mir_inflight[1], memory_order_relaxed);
    return (b <= a) ? 1 : 0;
}

/* ==================== COLI_NUMA=1: interleave expert slabs across NUMA nodes ====================
 * Uses dlopen to load libnuma.so.1 at runtime — no compile-time dependency on
 * libnuma-dev (numa.h), no -lnuma in the Makefile. If libnuma is missing the
 * engine prints a warning and falls back to plain malloc (no interleave, but
 * still runs correctly). Linux-only; silent no-op elsewhere. */
#ifdef __linux__
static int g_numa = 0;                                   /* COLI_NUMA env (default 0 = off) */
static int g_numa_available = 0;                         /* 1 = libnuma loaded and NUMA hardware present */
static void *(*g_numa_alloc_interleaved_fn)(size_t) = NULL;
static void  (*g_numa_free_fn)(void *, size_t) = NULL;
#endif

static void numa_init(void) {
#ifdef __linux__
    if (!getenv("COLI_NUMA") || !atoi(getenv("COLI_NUMA"))) return;
    g_numa = 1;
    void *h = dlopen("libnuma.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libnuma.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "[NUMA] libnuma not found — COLI_NUMA disabled, using malloc\n");
        g_numa = 0;
        return;
    }
    int (*numa_available_p)(void) = (int (*)(void))dlsym(h, "numa_available");
    g_numa_alloc_interleaved_fn = (void *(*)(size_t))dlsym(h, "numa_alloc_interleaved");
    g_numa_free_fn = (void (*)(void *, size_t))dlsym(h, "numa_free");
    if (!numa_available_p || !g_numa_alloc_interleaved_fn || !g_numa_free_fn) {
        fprintf(stderr, "[NUMA] libnuma symbols missing — COLI_NUMA disabled\n");
        g_numa = 0;
        return;
    }
    if (numa_available_p() < 0) {
        fprintf(stderr, "[NUMA] numa_available() reports NUMA not supported — COLI_NUMA disabled\n");
        g_numa = 0;
        return;
    }
    int (*numa_nodes_p)(void) = (int (*)(void))dlsym(h, "numa_num_configured_nodes");
    int nnodes = numa_nodes_p ? numa_nodes_p() : -1;
    fprintf(stderr, "[NUMA] expert slabs interleaved across %d node(s)\n", nnodes);
    g_numa_available = 1;
#endif
}

/* allocate `n` bytes for an expert slab. When NUMA is active, uses
 * numa_alloc_interleaved so the pages are distributed across all NUMA nodes.
 * Falls back to malloc otherwise. Must be freed with slab_free(p, n). */
static void *slab_alloc(size_t n) {
#ifdef __linux__
    if (g_numa_available && g_numa_alloc_interleaved_fn) {
        void *p = g_numa_alloc_interleaved_fn(n);
        if (p) return p;
        /* fall back to malloc if numa_alloc fails (e.g. huge allocation) */
    }
#endif
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM slab_alloc %zu\n", n); exit(1); }
    return p;
}

/* free a slab allocated by slab_alloc. When NUMA is active, uses numa_free
 * (which needs the size); otherwise calls free (size ignored). */
static void slab_free(void *p, size_t n) {
    if (!p) return;
#ifdef __linux__
    if (g_numa_available && g_numa_free_fn) {
        g_numa_free_fn(p, n);
        return;
    }
#endif
    (void)n;
    free(p);
}

/* ==================== OMP_HOT=1: permanent OpenMP hot-thread pool ====================
 * When enabled, sets OMP_WAIT_POLICY=active + GOMP_SPINCOUNT + KMP_BLOCKTIME so
 * the libgomp team stays hot (spinning) between the tiny per-expert matmul
 * regions, avoiding the create/destroy overhead. Re-execs self once so the
 * fresh libgomp constructor picks up the env vars (libgomp reads them in a
 * constructor before main). Then calls omp_set_num_threads + a one-shot
 * parallel region to pre-create the team. Reference: colibri.c "Permanent
 * OpenMP hot-thread tuning" block. Default 0 = off. */
static int g_omp_hot = 0;

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

/* fp32 -> bf16 (round-to-nearest-even). Used to resident-store dense weights
 * as bf16 when the on-disk tensor is fp32/f16 (tiny test models). Real K3 ships
 * bf16 on disk, in which case load_t_dense copies the raw bytes with no conversion. */
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1;          /* bit that becomes the bf16 LSB */
    uint32_t bias = 0x7FFFu + lsb;         /* round-to-nearest-even */
    return (uint16_t)((u + bias) >> 16);
}

/* bf16-native matmul: weights live as raw bf16 bytes (uint16_t*), activations
 * are fp32. Same math as matmul() — only the weight fetch converts bf16->f32.
 * Halves the resident RAM of dense weights (275GB fp32 -> 137GB bf16 on K3). */
static void matmul_bf16(float *y, const float *x, const uint16_t *W_bf16, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W_bf16 + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * bf16_to_f32(w[i]);
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* Dispatch a dense-weight matmul: bf16-resident weights go through matmul_bf16
 * (the float* is reinterpreted as uint16_t*), otherwise the classic fp32 path.
 * `bf16` is the per-scope residency flag (l->bf16_resident for layer weights,
 * m->bf16_resident for embed/lm_head). Expert-cache matmuls (e->g/u/d, always
 * fp32) MUST call matmul() directly — never this macro. */
#define DENSE_MM(y, x, W, S, I, O, bf16) do { \
    if (bf16) matmul_bf16((y), (x), (const uint16_t*)(const void*)(W), (S), (I), (O)); \
    else      matmul((y), (x), (W), (S), (I), (O)); \
} while (0)

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
/* Load a matmul-bound dense weight. In bf16-resident mode the tensor is stored
 * as raw bf16 bytes (2B/elem, half of fp32) and the returned float* is meant to
 * be reinterpreted as uint16_t* by matmul_bf16. Real K3 ships bf16 on disk, so
 * the common path is a zero-conversion raw copy; fp32/f16 on-disk tensors
 * (tiny test models) are dequantized then re-quantized to bf16. Norm vectors and
 * small params must keep using load_t_opt — they stay fp32 in both modes and are
 * read directly as float (never through matmul). */
static float *load_t_dense(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) return NULL;
    if (m->bf16_resident) {
        st_tensor *t = st_find(&m->S, name);
        uint16_t *p = malloc((size_t)n * sizeof(uint16_t));
        if (!p) { fprintf(stderr, "OOM %s\n", name); exit(1); }
        if (t && t->dtype == 0) {
            /* bf16 on disk: raw copy, no conversion (the K3 standard path). */
            st_read_raw(&m->S, name, p, 0);
        } else {
            /* fp32/f16 on disk: dequant to fp32 then re-quant to bf16. */
            float *tmp = falloc(n);
            st_read_f32(&m->S, name, tmp, 0);
            for (int64_t i = 0; i < n; i++) p[i] = f32_to_bf16(tmp[i]);
            free(tmp);
        }
        return (float*)p;
    }
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
    numa_init();                                    /* COLI_NUMA=1: dlopen libnuma for slab interleave */
    /* DUAL_SSD=<dir2>: register a second read-only copy of the model on
     * another drive. st_mirror_init (in st.h) verifies each shard's size +
     * header match the primary before accepting it. When no shard is accepted
     * (dir missing/empty), g_mirror stays 0 and the engine runs single-drive. */
    const char *dual = getenv("DUAL_SSD");
    if (dual && *dual) {
        int nmir = st_mirror_init(&m->S, dual);
        if (nmir > 0) {
            g_mirror = 1;
            fprintf(stderr, "[DUAL-SSD] mirror active: %d shard(s) on %s\n", nmir, dual);
        } else {
            fprintf(stderr, "[DUAL-SSD] no shards mirrored from %s — running single-drive\n", dual);
        }
    }
    Cfg *c = &m->c;
    /* BF16 native residency: dense weights stay bf16 in RAM (137GB vs 275GB fp32
     * on real K3). Default ON (the K3 standard path); BF16=0 falls back to the
     * fp32 path for backward-compatible testing. */
    const char *be = getenv("BF16");
    m->bf16_resident = (be && atoi(be) == 0) ? 0 : 1;
    double t0 = now_s();
    /* embed / lm_head / final_norm — K3 tensor names use "language_model." prefix.
     * embed + lm_head are matmul-bound (huge) -> load_t_dense (bf16 resident).
     * final_norm is a small vector read directly as float -> stays fp32. */
    m->embed      = load_t_dense(m, "language_model.model.embed_tokens.weight");
    if (!m->embed) m->embed = load_t(m, "model.embed_tokens.weight");
    m->lm_head    = load_t_dense(m, "language_model.lm_head.weight");
    if (!m->lm_head) m->lm_head = load_t(m, "lm_head.weight");
    m->final_norm = load_t_opt(m, "language_model.model.norm.weight");
    if (!m->final_norm) m->final_norm = load_t(m, "model.norm.weight");

    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[512];
    /* K3 ships tensors under "language_model.model.layers.N.*"; tiny test models
     * (and a flat-converted snapshot) use "model.layers.N.*". Probe once and pick. */
    const char *PFX = (st_find(&m->S, "language_model.model.layers.0.input_layernorm.weight")) ? "language_model." : "";
    /* Build the per-layer format string once: "<pfx>model.layers.%d.<suffix>".
     * P is a runtime string (not a literal), so we must use %s in snprintf.
     * LD  -> load_t_opt  (fp32 always: norms + small params read directly as float)
     * LDD -> load_t_dense (bf16 resident when m->bf16_resident: matmul-bound weights) */
    #define LD(field, suffix) do { \
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attn." suffix, PFX, i); \
        l->field = load_t_opt(m,nm); } while(0)
    #define LDD(field, suffix) do { \
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attn." suffix, PFX, i); \
        l->field = load_t_dense(m,nm); } while(0)
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->bf16_resident = m->bf16_resident;
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.input_layernorm.weight", PFX, i);  l->in_ln = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.post_attention_layernorm.weight", PFX, i); l->post_ln = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attention_res_norm.weight", PFX, i); l->attn_res_norm = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attention_res_proj.weight", PFX, i); l->attn_res_proj = load_t_dense(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp_res_norm.weight", PFX, i); l->mlp_res_norm = load_t_opt(m,nm);
        snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp_res_proj.weight", PFX, i); l->mlp_res_proj = load_t_dense(m,nm);
        if (c->is_mla[i]) {
            LDD(q_a, "q_a_proj.weight");
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attn.q_a_layernorm.weight", PFX, i); l->q_a_ln = load_t_opt(m,nm);
            LDD(q_b, "q_b_proj.weight");
            LDD(kv_a, "kv_a_proj_with_mqa.weight");
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.self_attn.kv_a_layernorm.weight", PFX, i); l->kv_a_ln = load_t_opt(m,nm);
            LDD(kv_b, "kv_b_proj.weight");
            LDD(o,    "o_proj.weight");
            if (c->use_output_gate) LDD(g_proj_mla, "g_proj.weight");
        } else {
            /* KDA: all non-MLA layers use KDA attention, including layer 0
             * (layer 0 is KDA attention + dense MLP, not just dense MLP). */
            LDD(q_proj, "q_proj.weight");
            LDD(k_proj, "k_proj.weight");
            LDD(v_proj, "v_proj.weight");
            LD(q_conv, "q_conv1d.weight");
            LD(k_conv, "k_conv1d.weight");
            LD(v_conv, "v_conv1d.weight");
            LD(A_log, "A_log");
            LD(dt_bias, "dt_bias");
            LDD(b_proj, "b_proj.weight");
            LDD(f_a, "f_a_proj.weight");
            LDD(f_b, "f_b_proj.weight");
            if (c->use_full_rank_gate) LDD(g_proj_kda, "g_proj.weight");
            LD(o_norm, "o_norm.weight");
            LDD(o, "o_proj.weight");
        }
        /* dense MLP (layer 0) */
        if (!c->is_moe[i]) {
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp.gate_proj.weight", PFX, i); l->mlp_gate = load_t_dense(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp.up_proj.weight", PFX, i);   l->mlp_up   = load_t_dense(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.mlp.down_proj.weight", PFX, i); l->mlp_down = load_t_dense(m,nm);
        } else {
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.gate.weight", PFX, i); l->gate_w = load_t_dense(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.gate.e_score_correction_bias", PFX, i); l->e_score_bias = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.routed_expert_down_proj.weight", PFX, i); l->routed_down = load_t_dense(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.routed_expert_up_proj.weight", PFX, i); l->routed_up = load_t_dense(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.routed_expert_norm.weight", PFX, i); l->routed_norm = load_t_opt(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight", PFX, i); l->shared_gate = load_t_dense(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.shared_experts.up_proj.weight", PFX, i); l->shared_up = load_t_dense(m,nm);
            snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.shared_experts.down_proj.weight", PFX, i); l->shared_down = load_t_dense(m,nm);
        }
    }
    #undef LD
    #undef LDD
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
    /* NUMA: expert slabs use slab_alloc so COLI_NUMA=1 can interleave them
     * across nodes. When NUMA is off, slab_alloc == malloc (identical to the
     * original falloc path, byte-for-byte). */
    s->g = slab_alloc((size_t)IH * sizeof(float));
    s->u = slab_alloc((size_t)IH * sizeof(float));
    s->d = slab_alloc((size_t)HI * sizeof(float));
    s->kind = EXP_NONE;
}

/* ---- DUAL-SSD: mirror-aware tensor readers ----
 * st_read_raw_rep / st_read_f32_rep / load_u8_rep: like their st.h counterparts
 * but read from replica `rep`'s fd (0=primary, 1=mirror), falling back to the
 * primary on any error. When rep==0 (or g_mirror==0), the read hits the primary
 * fd exactly as before — byte-identical to the original single-drive path. */
static void st_read_raw_rep(shards *S, const char *name, void *out, int drop, int rep) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    int fd = st_fd_rep(S, t->fd, rep);
    if (fd < 0) fd = t->fd;
    st_pread_full(fd, out, t->nbytes, t->off, "pread raw rep");
    if (drop) posix_fadvise(fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
}

static int64_t st_read_f32_rep(shards *S, const char *name, float *out, int drop, int rep) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    if (t->numel < 0 || t->numel > t->nbytes / esz || t->numel * (int64_t)esz != t->nbytes) {
        fprintf(stderr, "%s: tensor '%s' shape/bytes mismatch (numel %lld, %lld bytes, dtype %d) — refusing\n",
                name, name, (long long)t->numel, (long long)t->nbytes, t->dtype); exit(1); }
    void *raw = malloc(t->nbytes);
    if (!raw) { fprintf(stderr, "malloc %lld bytes for tensor %s failed\n", (long long)t->nbytes, name); exit(1); }
    int fd = st_fd_rep(S, t->fd, rep);
    if (fd < 0) fd = t->fd;
    st_pread_full(fd, raw, t->nbytes, t->off, "pread f32 rep");
    if (t->dtype == 2) {
        memcpy(out, raw, t->nbytes);
    } else if (t->dtype == 0) {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = bf16_to_f32(p[i]);
    } else {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = f16_to_f32(p[i]);
    }
    free(raw);
    if (drop) posix_fadvise(fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    return t->numel;
}

static uint8_t *load_u8_rep(Model *m, const char *name, int64_t *n_out, int rep) {
    st_tensor *t = st_find(&m->S, name);
    if (!t) { *n_out = -1; return NULL; }
    int64_t n = t->nbytes;
    uint8_t *p = malloc((size_t)n);
    if (!p) { fprintf(stderr,"OOM %s\n", name); exit(1); }
    st_read_raw_rep(&m->S, name, p, 1, rep);
    *n_out = n;
    return p;
}

/* Load a single expert's three weights into Slot.
 * First tries colibri-int8 (merged_weight + qs), then falls back to MXFP4.
 * DUAL-SSD: routes the reads to replica `rep` (0=primary, 1=mirror) based on
 * in-flight load, so two drives can serve experts in parallel. */
static void load_expert(Model *m, int layer, int eid, Slot *s) {
    Cfg *c = &m->c;
    char nm[512];
    const char *PFX = (st_find(&m->S, "language_model.model.layers.0.input_layernorm.weight")) ? "language_model." : "";
    int64_t IH = (int64_t)c->moe_inter * c->routed_hidden;
    int64_t HI = (int64_t)c->routed_hidden * c->moe_inter;
    int64_t want_g = IH, want_u = IH, want_d = HI;

    /* DUAL-SSD: route to the less-loaded drive */
    int rep = expert_route(layer, eid);
    if (g_mirror) atomic_fetch_add_explicit(&g_mir_inflight[rep], 1, memory_order_relaxed);

    /* try colibri int8 first */
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.merged_weight", PFX, layer, eid);
    st_tensor *tw = st_find(&m->S, nm);
    if (tw && (int64_t)tw->nbytes == (want_g + want_u + want_d)) {
        /* int8 path — slab_alloc so NUMA can interleave the weight block */
        int8_t *block = slab_alloc((size_t)(want_g + want_u + want_d));
        st_read_raw_rep(&m->S, nm, block, 1, rep);
        s->qg = block; s->qu = block + want_g; s->qd = block + want_g + want_u;
        snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.qs", PFX, layer, eid);
        int64_t want_s = (int64_t)c->moe_inter * 2 + c->routed_hidden;
        st_tensor *ts = st_find(&m->S, nm);
        if (!ts || ts->numel != want_s) {
            fprintf(stderr,"%s: scale array mismatch (got %lld want %lld)\n", nm,
                (long long)(ts?ts->numel:-1), (long long)want_s); exit(1);
        }
        float *sb = falloc(want_s);
        st_read_f32_rep(&m->S, nm, sb, 0, rep);
        s->qgs = sb; s->qus = sb + c->moe_inter; s->qds = sb + 2*c->moe_inter;
        s->kind = EXP_INT8;
        goto done;
    }

    /* MXFP4 path: dequantize on load */
    int64_t packed_g = want_g / 2; /* two nibbles per byte */
    int64_t packed_u = want_u / 2;
    int64_t packed_d = want_d / 2;
    int64_t scales_g = want_g / 32;
    int64_t scales_u = want_u / 32;
    int64_t scales_d = want_d / 32;
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w1.weight_packed", PFX, layer, eid);
    uint8_t *g_pk = load_u8_rep(m, nm, &packed_g, rep);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w1.weight_scale", PFX, layer, eid);
    uint8_t *g_sc = load_u8_rep(m, nm, &scales_g, rep);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w3.weight_packed", PFX, layer, eid);
    uint8_t *u_pk = load_u8_rep(m, nm, &packed_u, rep);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w3.weight_scale", PFX, layer, eid);
    uint8_t *u_sc = load_u8_rep(m, nm, &scales_u, rep);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w2.weight_packed", PFX, layer, eid);
    uint8_t *d_pk = load_u8_rep(m, nm, &packed_d, rep);
    snprintf(nm,sizeof(nm), "%smodel.layers.%d.block_sparse_moe.experts.%d.w2.weight_scale", PFX, layer, eid);
    uint8_t *d_sc = load_u8_rep(m, nm, &scales_d, rep);
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
done:
    if (g_mirror) atomic_fetch_sub_explicit(&g_mir_inflight[rep], 1, memory_order_relaxed);
}

/* ---- KVSAVE: serialize/restore KV cache + KDA state (default off) ----
 * KVSAVE=<dir> writes <dir>/kv_<session_id>.bin right after prefill; at
 * startup the same prompt restores the caches and skips re-prefill (warm
 * start). session_id = FNV-1a-64 of the prompt token ids, so the same
 * prompt resumes the same KV file. The load path uses pread (positioned
 * reads) so the data can be fetched without blocking seek+fread pairs. */
#define KVSAVE_MAGIC "K3KV001\0"

static uint64_t kv_session_id(const int *prompt, int np) {
    uint64_t h = 0xcbf29ce484222325ULL;   /* FNV-1a 64-bit offset basis */
    for (int i = 0; i < np; i++) {
        h ^= (uint64_t)(unsigned)prompt[i];
        h *= 0x100000001b3ULL;             /* FNV-1a 64-bit prime */
    }
    return h;
}

/* Serialize MLA KV caches (only the [0, kv_len) slice — not the full max_t
 * buffer) and KDA recurrent state (fixed-size, independent of seq length)
 * plus the prefill logits so the first token can be picked without a re-step.
 * File layout: [magic 8B][n_layers,max_t,kv_len,vocab i32 x4]
 *   per MLA layer: K_nope[n_heads*kv_len*qk_nope] K_rot[kv_len*qk_rope] V[n_heads*kv_len*v_head]
 *   per KDA layer: kda_state[heads*hd*hd] kda_conv_{q,k,v}[pqk*short_conv]
 *   logits[vocab] */
static void kv_save(Model *m, const char *path, int kv_len, const float *logits) {
    Cfg *c = &m->c;
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[KVSAVE] cannot open %s for writing\n", path); return; }
    int32_t hdr[4] = { c->n_layers, m->max_t, kv_len, c->vocab };
    fwrite(KVSAVE_MAGIC, 1, 8, f);
    fwrite(hdr, sizeof(int32_t), 4, f);
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_mla[i]) {
            fwrite(m->K_nope[i], sizeof(float), (size_t)c->n_heads * kv_len * c->qk_nope, f);
            fwrite(m->K_rot[i],  sizeof(float), (size_t)kv_len * c->qk_rope, f);
            fwrite(m->V[i],      sizeof(float), (size_t)c->n_heads * kv_len * c->v_head, f);
        } else {
            int pqk = c->kda_n_heads * c->kda_head_dim;
            fwrite(m->kda_state[i],  sizeof(float),
                   (size_t)c->kda_n_heads * c->kda_head_dim * c->kda_head_dim, f);
            fwrite(m->kda_conv_q[i], sizeof(float), (size_t)pqk * c->short_conv, f);
            fwrite(m->kda_conv_k[i], sizeof(float), (size_t)pqk * c->short_conv, f);
            fwrite(m->kda_conv_v[i], sizeof(float), (size_t)pqk * c->short_conv, f);
        }
    }
    if (logits) fwrite(logits, sizeof(float), (size_t)c->vocab, f);
    fclose(f);
    fprintf(stderr, "[KVSAVE] wrote %d KV positions to %s\n", kv_len, path);
}

/* Load KV/KDA state from disk using pread (positioned reads — no fseek, so
 * chunks can be issued independently / overlapped). Allocates the caches if
 * this is the first call (mirrors step()'s lazy allocation). Returns 1 and
 * sets m->kv_loaded=1 on success; returns 0 if the file is missing or
 * incompatible (caller falls through to a normal cold prefill). */
static int kv_load(Model *m, const char *path, int np) {
    Cfg *c = &m->c;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    char magic[8]; int32_t hdr[4];
    off_t off = 0;
    #define KV_RD(buf, n) do { \
        if (pread(fd, (buf), (size_t)(n), off) != (ssize_t)(n)) { close(fd); return 0; } \
        off += (off_t)(n); \
    } while (0)
#else
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    #define KV_RD(buf, n) do { \
        if (fread((buf), 1, (size_t)(n), f) != (size_t)(n)) { fclose(f); return 0; } \
    } while (0)
#endif
    KV_RD(magic, 8);
    if (memcmp(magic, KVSAVE_MAGIC, 8) != 0) {
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
        close(fd);
#else
        fclose(f);
#endif
        return 0;
    }
    KV_RD(hdr, sizeof(hdr));
    int n_layers = hdr[0], kv_len = hdr[2], vocab = hdr[3];
    if (n_layers != c->n_layers || vocab != c->vocab || kv_len < np) {
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
        close(fd);
#else
        fclose(f);
#endif
        return 0;
    }
    /* allocate KV caches if not already done (mirrors step()'s allocation) */
    if (!m->K_nope) {
        m->K_nope = calloc(c->n_layers, sizeof(float*));
        m->K_rot  = calloc(c->n_layers, sizeof(float*));
        m->V      = calloc(c->n_layers, sizeof(float*));
        m->kda_state = calloc(c->n_layers, sizeof(float*));
        m->kda_conv_q = calloc(c->n_layers, sizeof(float*));
        m->kda_conv_k = calloc(c->n_layers, sizeof(float*));
        m->kda_conv_v = calloc(c->n_layers, sizeof(float*));
        int ctx = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
        if (ctx < kv_len + 64) ctx = kv_len + 64;
        if (ctx < 1) ctx = 4096;
        m->max_t = ctx;
        for (int i = 0; i < c->n_layers; i++) {
            if (c->is_mla[i]) {
                m->K_nope[i] = falloc((int64_t)c->n_heads * m->max_t * c->qk_nope);
                m->K_rot[i]  = falloc((int64_t)m->max_t * c->qk_rope);
                m->V[i]      = falloc((int64_t)c->n_heads * m->max_t * c->v_head);
            } else {
                m->kda_state[i] = falloc((int64_t)c->kda_n_heads * c->kda_head_dim * c->kda_head_dim);
                memset(m->kda_state[i], 0, (size_t)c->kda_n_heads * c->kda_head_dim * c->kda_head_dim * sizeof(float));
                int pqk = c->kda_n_heads * c->kda_head_dim;
                m->kda_conv_q[i] = falloc((int64_t)pqk * c->short_conv);
                m->kda_conv_k[i] = falloc((int64_t)pqk * c->short_conv);
                m->kda_conv_v[i] = falloc((int64_t)pqk * c->short_conv);
                memset(m->kda_conv_q[i], 0, (size_t)pqk * c->short_conv * sizeof(float));
                memset(m->kda_conv_k[i], 0, (size_t)pqk * c->short_conv * sizeof(float));
                memset(m->kda_conv_v[i], 0, (size_t)pqk * c->short_conv * sizeof(float));
            }
        }
    }
    /* read per-layer data */
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_mla[i]) {
            KV_RD(m->K_nope[i], (size_t)c->n_heads * kv_len * c->qk_nope * sizeof(float));
            KV_RD(m->K_rot[i],  (size_t)kv_len * c->qk_rope * sizeof(float));
            KV_RD(m->V[i],      (size_t)c->n_heads * kv_len * c->v_head * sizeof(float));
        } else {
            int pqk = c->kda_n_heads * c->kda_head_dim;
            KV_RD(m->kda_state[i],  (size_t)c->kda_n_heads * c->kda_head_dim * c->kda_head_dim * sizeof(float));
            KV_RD(m->kda_conv_q[i], (size_t)pqk * c->short_conv * sizeof(float));
            KV_RD(m->kda_conv_k[i], (size_t)pqk * c->short_conv * sizeof(float));
            KV_RD(m->kda_conv_v[i], (size_t)pqk * c->short_conv * sizeof(float));
        }
    }
    /* read saved prefill logits */
    m->saved_logits = falloc(c->vocab);
    KV_RD(m->saved_logits, (size_t)c->vocab * sizeof(float));
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    close(fd);
#else
    fclose(f);
#endif
    #undef KV_RD
    m->kv_len = kv_len;
    m->kv_loaded = 1;
    fprintf(stderr, "[KVSAVE] warm start: restored %d KV positions from %s\n", kv_len, path);
    return 1;
}

/* ---- RSS GUARD: background memory watchdog (default off) ----
 * RSS_LIMIT=<GB> spawns a pthread that checks RSS every 5s via getrusage.
 * Over the limit: sets g_rss_evict_needed so the main thread frees non-
 * pinned expert cache slots at the next safe point (between tokens).
 * At 95% of the limit: calls abort() to pre-empt the OOM killer. */
static void *rss_guard_thread(void *arg) {
    (void)arg;   /* Model pointer reserved for future per-model thresholds */
    while (!atomic_load(&g_rss_guard_stop)) {
        sleep_ms(5000);
        if (atomic_load(&g_rss_guard_stop)) break;
        double rss = rss_gb();
        if (g_rss_limit_gb <= 0) continue;
        if (rss >= g_rss_limit_gb * 0.95) {
            fprintf(stderr, "[RSS-GUARD] RSS %.2f GB >= 95%% of %.1f GB limit: "
                            "aborting to pre-empt OOM killer\n", rss, g_rss_limit_gb);
            abort();
        }
        if (rss > g_rss_limit_gb)
            atomic_store(&g_rss_evict_needed, 1);
    }
    return NULL;
}

/* Free non-pinned, non-in-use expert cache slots to reclaim RSS. Called by
 * the main thread between tokens (a safe point: no expert matmul in flight).
 * The freed slots keep their index (eid=-1, used=0) so expert_reserve()
 * picks them first as LRU candidates; slot_ensure_allocated() re-allocates
 * the slab on the next miss. */
static void rss_guard_evict(Model *m) {
    if (!atomic_load(&g_rss_evict_needed)) return;
    atomic_store(&g_rss_evict_needed, 0);
    Cfg *c = &m->c;
    int64_t IH = (int64_t)c->moe_inter * c->routed_hidden;
    int64_t HI = (int64_t)c->routed_hidden * c->moe_inter;
    size_t slab_gu = (size_t)IH * sizeof(float);
    size_t slab_d  = (size_t)HI * sizeof(float);
    size_t slab_qg = (size_t)(IH + IH + HI);   /* int8 block: g+u+d, 1 byte each */
    int dropped = 0;
    pthread_mutex_lock(&g_mx);
    for (int l = 0; l < c->n_layers; l++) {
        LCache *lc = &m->cache[l];
        for (int i = 0; i < lc->n; i++) {
            Slot *s = &lc->slots[i];
            if (s->in_use || s->pinned || s->eid < 0) continue;
            /* free the slab data (g/u/d always; qg/qgs for INT8 path).
             * slab_free handles numa_free vs free based on COLI_NUMA. */
            slab_free(s->g, slab_gu);  s->g = NULL;
            slab_free(s->u, slab_gu);  s->u = NULL;
            slab_free(s->d, slab_d);   s->d = NULL;
            slab_free(s->qg, slab_qg); /* NULL for MXFP4, block-start for INT8 */
            free(s->qgs);              /* scales: small, always malloc'd via falloc */
            s->qg = s->qu = s->qd = NULL;
            s->qgs = s->qus = s->qds = NULL;
            s->eid = -1;
            s->used = 0;
            s->kind = EXP_NONE;
            dropped++;
        }
    }
    pthread_mutex_unlock(&g_mx);
    if (dropped)
        fprintf(stderr, "[RSS-GUARD] evicted %d non-pinned experts (RSS was %.2f GB)\n",
                dropped, rss_gb());
}

/* Reserve a cache slot for `eid` (layer L). On a hit (expert already resident,
 * OR an async load for this eid already in flight) returns the slot with
 * *was_hit=1 and DOES NOT load — the caller may use the slot immediately. On a
 * miss, allocates/evicts a slot, marks it in_use with loading_eid=eid, and
 * returns it with *was_hit=0; the caller must then load_expert() into it and
 * call expert_publish(). The in-flight-wait makes PILOT prefetch safe: if a
 * background PILOT worker is mid-load for eid, a concurrent reserve() for the
 * same eid blocks until the worker publishes, then returns the now-resident
 * slot as a hit (no duplicate load). Never evicts an in_use slot — if every
 * slot is in use (only possible with background workers and a tiny cache), it
 * waits for slot 0 to be published and retries, guaranteeing forward progress. */
static void expert_reserve(Model *m, int layer, int eid, Slot **out, int *was_hit) {
    LCache *lc = &m->cache[layer];
    for (;;) {
        pthread_mutex_lock(&g_mx);
        /* resident hit? */
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].eid == eid && !lc->slots[i].in_use) {
                m->hits++; lc->slots[i].used = ++m->clock;
                *out = &lc->slots[i]; *was_hit = 1;
                pthread_mutex_unlock(&g_mx);
                return;
            }
        }
        /* in-flight async load for the same eid? wait for it, then re-scan. */
        int inflight = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].in_use && lc->slots[i].loading_eid == eid) { inflight = i; break; }
        }
        if (inflight >= 0) {
            pthread_mutex_unlock(&g_mx);
            volatile int *iu = &lc->slots[inflight].in_use;
            while (*iu) sched_yield();            /* spin until the worker publishes */
            continue;                              /* re-scan (likely a hit now) */
        }
        /* genuine miss: pick a slot, never clobbering an in_use one */
        m->miss++;
        Slot *s;
        if (lc->n < lc->cap) {
            s = &lc->slots[lc->n];
            slot_ensure_allocated(m, s);
            lc->n++;
        } else {
            int lru = -1;
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].pinned || lc->slots[i].in_use) continue;
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
            if (lru < 0) {
                /* fall back to ignoring pinned (still skip in_use) */
                for (int i = 0; i < lc->n; i++) {
                    if (lc->slots[i].in_use) continue;
                    if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
                }
            }
            if (lru < 0) {
                /* every slot is in_use (background workers + tiny cache):
                 * wait for slot 0 to be published, then retry the whole scan. */
                pthread_mutex_unlock(&g_mx);
                volatile int *iu = &lc->slots[0].in_use;
                while (*iu) sched_yield();
                continue;
            }
            s = &lc->slots[lru];
            s->pinned = 0;
            slot_ensure_allocated(m, s);   /* re-alloc slab if RSS guard freed it */
        }
        s->eid = -1;
        s->in_use = 1;
        s->loading_eid = eid;
        s->used = ++m->clock;
        pthread_mutex_unlock(&g_mx);
        *out = s; *was_hit = 0;
        return;
    }
}

/* Publish a slot loaded by reserve()+load_expert(): make it resident and
 * visible to subsequent reserve() calls. Called by the main thread (serial
 * expert_get path) and by PIPE/PILOT worker threads after their pread. */
static void expert_publish(Model *m, int layer, Slot *s, int eid) {
    pthread_mutex_lock(&g_mx);
    s->eid = eid;
    s->loading_eid = -1;
    s->in_use = 0;
    s->pinned = m->is_pinned[(int64_t)layer * m->c.n_experts + eid];
    s->used = ++m->clock;
    pthread_mutex_unlock(&g_mx);
}

static void expert_get(Model *m, int layer, int eid, Slot **out) {
    Slot *s; int hit;
    expert_reserve(m, layer, eid, &s, &hit);
    if (hit) { *out = s; return; }
    load_expert(m, layer, eid, s);
    expert_publish(m, layer, s, eid);
    *out = s;
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
    int bf16 = l->bf16_resident;

    float *q_a = falloc(ql);
    DENSE_MM(q_a, x, l->q_a, 1, D, ql, bf16);
    rmsnorm_row(q_a, q_a, l->q_a_ln, ql, c->eps);
    float *q_full = falloc(H * qh);
    DENSE_MM(q_full, q_a, l->q_b, 1, ql, H*qh, bf16);

    float *comp = falloc(kvl + R);
    DENSE_MM(comp, x, l->kv_a, 1, D, kvl + R, bf16);
    float *k_pass_latent = comp;          /* [kvl] */
    float *k_rot = comp + kvl;            /* [R]   */
    rmsnorm_row(k_pass_latent, k_pass_latent, l->kv_a_ln, kvl, c->eps);
    rope_interleave(k_rot, pos, R, c->theta);

    /* write K/V cache: kv_b applied to normalized latent gives [H*(No+vh)] */
    float *kv = falloc(H * (No + vh));
    DENSE_MM(kv, k_pass_latent, l->kv_b, 1, kvl, H*(No+vh), bf16);
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
        DENSE_MM(g, x, l->g_proj_mla, 1, D, H*vh, bf16);
        for (int i = 0; i < H*vh; i++) {
            float gi = 1.f / (1.f + expf(-g[i]));
            ctx[i] *= gi;
        }
        free(g);
    }
    DENSE_MM(out, ctx, l->o, 1, H*vh, D, bf16);
    free(q_a); free(q_full); free(comp); free(kv); free(ctx);
}

/* ---------- KDA (Kimi Delta Attention), fused-recurrent form, single token ---------- */
static void kda_forward(Model *m, Layer *l, int layer, const float *x, int pos, float *out) {
    Cfg *c = &m->c;
    int H = c->kda_n_heads, hd = c->kda_head_dim, D = c->hidden;
    int proj_qk = H * hd;          /* q_proj, k_proj out */
    int proj_v  = H * hd;          /* v_proj out */
    int conv_k  = c->short_conv;
    int bf16 = l->bf16_resident;

    /* projections */
    float *qp = falloc(proj_qk), *kp = falloc(proj_qk), *vp = falloc(proj_v);
    DENSE_MM(qp, x, l->q_proj, 1, D, proj_qk, bf16);
    DENSE_MM(kp, x, l->k_proj, 1, D, proj_qk, bf16);
    DENSE_MM(vp, x, l->v_proj, 1, D, proj_v, bf16);

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
    if (c->use_full_rank_gate) DENSE_MM(g_pre, x, l->g_proj_kda, 1, D, proj_v, bf16);
    /* dt_bias and A_log are per-head: shape [H]. */
    /* alpha[h] = sigmoid(A_log[h] + dt_bias[h])  -- per-head retention factor */
    /* beta[h]  = sigmoid(b_proj(x))             -- per-head delta write strength */
    float *beta = falloc(H);
    DENSE_MM(beta, x, l->b_proj, 1, D, H, bf16);

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

    DENSE_MM(out, o, l->o, 1, proj_v, D, bf16);
    free(qp); free(kp); free(vp); free(q); free(k); free(v); free(g_pre); free(beta); free(o);
}

/* ---------- KDA chunkwise parallel prefill ----------
 * Processes C tokens of one KDA layer in a single call. Mathematically equivalent
 * to running kda_forward() C times (token-by-token), but the intra-chunk delta-
 * rule SSM is evaluated via parallel matmuls (Mamba2 chunkwise form, K3 tech
 * report §3.2) instead of a sequential recurrence — turning prefill from O(L)
 * sequential dependent steps into O(L/C) chunk updates with parallel work inside.
 *
 * The conv1d + projections + gate + L2-norm are still walked per-token (they are
 * cheap, O(C*proj*conv_k), and keep the conv sliding-window state identical to
 * the recurrent path); the expensive O(C*H*hd^2) state update + readout is what
 * the chunkwise form parallelises.
 *
 * Per head h, alpha = sigmoid(A_log[h]+dt_bias[h]) is CONSTANT across tokens
 * (only beta_t = sigmoid(b_proj(x_t)[h]) is per-token), so the delta rule
 *   S_t = alpha * S_{t-1} + w_t * (v_t outer k_t),   w_t = (1-alpha)*beta_t
 *   o_t = S_t @ q_t
 * is a linear recurrence. Unrolling a chunk of C tokens with S_start = S_{-1}
 * and decay[k] = alpha^k:
 *   o_t = decay[t+1] * (S_start @ q_t)                        (state term)
 *       + sum_{j<=t} decay[t-j] * w_j * (k_j . q_t) * v_j     (intra term)
 *   S_end = decay[C] * S_start
 *         + sum_j decay[C-1-j] * w_j * (v_j outer k_j)        (state carry)
 * The state term is one matvec Q@S_start per token (rows scaled by decay[t+1]);
 * the intra term is a causal [C,C] dot-product matrix times V; the state carry
 * is a rank-C outer-product accumulation. All three parallelise across threads. */
static void kda_forward_chunk(Model *m, Layer *l, int layer,
                              const float *X, int C, int pos_base, float *out) {
    Cfg *c = &m->c;
    int H = c->kda_n_heads, hd = c->kda_head_dim, D = c->hidden;
    int proj_qk = H * hd;          /* q_proj, k_proj out */
    int proj_v  = H * hd;          /* v_proj out */
    int conv_k  = c->short_conv;
    int bf16 = l->bf16_resident;
    (void)pos_base;  /* KDA is position-invariant (recurrent state, no RoPE) */

    /* ---- batched projections: X[C,D] @ W[proj,D]^T -> [C,proj] ---- */
    float *qp = falloc((int64_t)C * proj_qk);
    float *kp = falloc((int64_t)C * proj_qk);
    float *vp = falloc((int64_t)C * proj_v);
    DENSE_MM(qp, X, l->q_proj, C, D, proj_qk, bf16);
    DENSE_MM(kp, X, l->k_proj, C, D, proj_qk, bf16);
    DENSE_MM(vp, X, l->v_proj, C, D, proj_v, bf16);

    /* ---- short conv (SiLU): sequential over tokens, updates conv state ----
     * Identical per-token math to kda_forward; after the chunk the conv state
     * holds the last conv_k projections (correct carry for the next chunk). */
    float *q = falloc((int64_t)C * proj_qk);
    float *k = falloc((int64_t)C * proj_qk);
    float *v = falloc((int64_t)C * proj_v);
    float *cs_q = m->kda_conv_q[layer], *cs_k = m->kda_conv_k[layer], *cs_v = m->kda_conv_v[layer];
    for (int t = 0; t < C; t++) {
        const float *qpt = qp + (int64_t)t * proj_qk;
        for (int ch = 0; ch < proj_qk; ch++) {
            for (int s = conv_k - 1; s > 0; s--) cs_q[ch*conv_k + s] = cs_q[ch*conv_k + s - 1];
            cs_q[ch*conv_k + 0] = qpt[ch];
            float acc = 0.f;
            for (int s = 0; s < conv_k; s++) acc += l->q_conv[ch*conv_k + s] * cs_q[ch*conv_k + s];
            q[(int64_t)t*proj_qk + ch] = acc / (1.f + expf(-acc));
        }
    }
    for (int t = 0; t < C; t++) {
        const float *kpt = kp + (int64_t)t * proj_qk;
        for (int ch = 0; ch < proj_qk; ch++) {
            for (int s = conv_k - 1; s > 0; s--) cs_k[ch*conv_k + s] = cs_k[ch*conv_k + s - 1];
            cs_k[ch*conv_k + 0] = kpt[ch];
            float acc = 0.f;
            for (int s = 0; s < conv_k; s++) acc += l->k_conv[ch*conv_k + s] * cs_k[ch*conv_k + s];
            k[(int64_t)t*proj_qk + ch] = acc / (1.f + expf(-acc));
        }
    }
    for (int t = 0; t < C; t++) {
        const float *vpt = vp + (int64_t)t * proj_v;
        for (int ch = 0; ch < proj_v; ch++) {
            for (int s = conv_k - 1; s > 0; s--) cs_v[ch*conv_k + s] = cs_v[ch*conv_k + s - 1];
            cs_v[ch*conv_k + 0] = vpt[ch];
            float acc = 0.f;
            for (int s = 0; s < conv_k; s++) acc += l->v_conv[ch*conv_k + s] * cs_v[ch*conv_k + s];
            v[(int64_t)t*proj_v + ch] = acc / (1.f + expf(-acc));
        }
    }

    /* ---- gate g (pre-sigmoid, full-rank) and beta, batched over tokens ---- */
    float *g_pre = falloc((int64_t)C * proj_v);
    if (c->use_full_rank_gate) DENSE_MM(g_pre, X, l->g_proj_kda, C, D, proj_v, bf16);
    float *beta = falloc((int64_t)C * H);
    DENSE_MM(beta, X, l->b_proj, C, D, H, bf16);

    /* ---- L2-normalize q and k per head per token (matches kda_forward) ---- */
    for (int t = 0; t < C; t++) {
        for (int h = 0; h < H; h++) {
            float *qh = q + (int64_t)t*proj_qk + h*hd;
            float *kh = k + (int64_t)t*proj_qk + h*hd;
            double qn = 0, kn = 0;
            for (int d = 0; d < hd; d++) { qn += qh[d]*qh[d]; kn += kh[d]*kh[d]; }
            float qi = 1.f / sqrtf((float)qn + 1e-12f);
            float ki = 1.f / sqrtf((float)kn + 1e-12f);
            for (int d = 0; d < hd; d++) { qh[d] *= qi; kh[d] *= ki; }
        }
    }

    /* ---- chunkwise delta-rule SSM, per head ---- */
    float *state = m->kda_state[layer];
    float *o = falloc((int64_t)C * proj_v);          /* [C, proj_v] SSM readout */
    /* per-head work buffers (allocated once, reused across heads) */
    float *Qh = falloc((int64_t)C * hd);
    float *Kh = falloc((int64_t)C * hd);
    float *Vh = falloc((int64_t)C * hd);
    float *wbuf   = falloc(C);          /* w_t = (1-alpha)*sigmoid(beta_t) */
    float *decay  = falloc(C + 1);      /* decay[k] = alpha^k */
    float *ostate = falloc((int64_t)C * hd);   /* state term [C,hd] */
    float *KK     = falloc((int64_t)C * C);    /* causal [C,C] */
    float *ointra = falloc((int64_t)C * hd);   /* intra term [C,hd] */
    float *coeff  = falloc(C);          /* coeff[t] = decay[C-1-t]*w_t (state carry) */

    for (int h = 0; h < H; h++) {
        float alpha = 1.f / (1.f + expf(-(l->A_log[h] + l->dt_bias[h])));
        float *sh = state + (int64_t)h * hd * hd;   /* [hd,hd] S_start -> S_end */

        /* gather contiguous per-head [C,hd] slices (q/k/v are [C, H*hd]) */
        for (int t = 0; t < C; t++) {
            memcpy(Qh + (int64_t)t*hd, q + (int64_t)t*proj_qk + h*hd, hd*sizeof(float));
            memcpy(Kh + (int64_t)t*hd, k + (int64_t)t*proj_qk + h*hd, hd*sizeof(float));
            memcpy(Vh + (int64_t)t*hd, v + (int64_t)t*proj_v  + h*hd, hd*sizeof(float));
        }

        /* per-token write strength w_t and decay factors alpha^k */
        for (int t = 0; t < C; t++) {
            float b = 1.f / (1.f + expf(-beta[(int64_t)t*H + h]));
            wbuf[t] = (1.f - alpha) * b;
        }
        decay[0] = 1.f;
        for (int t = 1; t <= C; t++) decay[t] = decay[t-1] * alpha;

        /* --- state term: ostate[t,j] = decay[t+1] * sum_i Qh[t,i] * S[i,j] --- */
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < C; t++) {
            float dc = decay[t+1];
            const float *qt = Qh + (int64_t)t*hd;
            float *ot = ostate + (int64_t)t*hd;
            for (int j = 0; j < hd; j++) {
                float acc = 0.f;
                for (int i = 0; i < hd; i++) acc += qt[i] * sh[i*hd + j];
                ot[j] = dc * acc;
            }
        }

        /* --- intra-chunk causal term ---
         * KK[t,j] = decay[t-j] * w_j * (k_j . q_t)   for j<=t
         * ointra[t,d] = sum_{j<=t} KK[t,j] * Vh[j,d] */
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < C; t++) {
            const float *qt = Qh + (int64_t)t*hd;
            float *kkrow = KK + (int64_t)t*C;
            for (int j = 0; j <= t; j++) {
                const float *kj = Kh + (int64_t)j*hd;
                float dot = 0.f;
                for (int d = 0; d < hd; d++) dot += qt[d] * kj[d];
                kkrow[j] = decay[t-j] * wbuf[j] * dot;
            }
            float *oit = ointra + (int64_t)t*hd;
            for (int d = 0; d < hd; d++) oit[d] = 0.f;
            for (int j = 0; j <= t; j++) {
                float kkj = kkrow[j];
                const float *vj = Vh + (int64_t)j*hd;
                for (int d = 0; d < hd; d++) oit[d] += kkj * vj[d];
            }
        }

        /* --- combine state + intra terms into o[t, h*hd + d] --- */
        for (int t = 0; t < C; t++) {
            float *ot = o + (int64_t)t*proj_v + h*hd;
            const float *st = ostate + (int64_t)t*hd;
            const float *it = ointra + (int64_t)t*hd;
            for (int d = 0; d < hd; d++) ot[d] = st[d] + it[d];
        }

        /* --- state carry: S_end = decay[C]*S_start + sum_t coeff[t]*(v_t outer k_t) ---
         * (v outer k)[i,j] = k[i]*v[j] ; coeff[t] = decay[C-1-t] * w_t            */
        float dc_end = decay[C];
        for (int t = 0; t < C; t++) coeff[t] = decay[C-1-t] * wbuf[t];
        for (int i = 0; i < hd*hd; i++) sh[i] *= dc_end;   /* alpha^C * S_start */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < hd; i++) {
            float *srow = sh + (int64_t)i*hd;
            for (int t = 0; t < C; t++) {
                float ki = Kh[(int64_t)t*hd + i] * coeff[t];
                const float *vrow = Vh + (int64_t)t*hd;
                for (int j = 0; j < hd; j++) srow[j] += ki * vrow[j];
            }
        }
    }

    /* ---- o_norm: FusedRMSNormGated(o, g) = RMSNorm(o) * sigmoid(g), per token ---- */
    for (int t = 0; t < C; t++) {
        float *ot = o + (int64_t)t*proj_v;
        rmsnorm_row(ot, ot, l->o_norm, proj_v, c->eps);
        for (int i = 0; i < proj_v; i++) {
            float gi = 1.f / (1.f + expf(-g_pre[(int64_t)t*proj_v + i]));
            ot[i] *= gi;
        }
    }

    /* ---- output projection: out[C,D] = o[C,proj_v] @ o_proj[proj_v,D]^T ---- */
    DENSE_MM(out, o, l->o, C, proj_v, D, bf16);

    free(qp); free(kp); free(vp); free(q); free(k); free(v);
    free(g_pre); free(beta); free(o);
    free(Qh); free(Kh); free(Vh); free(wbuf); free(decay);
    free(ostate); free(KK); free(ointra); free(coeff);
}

/* ---------- PIPE: async expert-load worker pool ----------
 * When PIPE=1, moe_forward() dispatches the cache-MISS experts of the current
 * token to a pool of I/O worker threads (pipe_dispatch) and then immediately
 * enters its matmul loop. Each iteration pipe_wait()s on the slot for that
 * expert; by the time the matmul of expert k is running, the workers are
 * already pread-ing expert k+1..kmiss-1 in parallel, so the pread latency is
 * hidden behind the compute. The slots loaded here are the SAME cache slots
 * obtained from expert_reserve() (in_use=1, loading_eid set); the worker
 * calls load_expert() then expert_publish() to make them resident, so a later
 * expert_get() for the same expert sees a hit. PILOT prefetch uses a separate
 * single background thread (below) so the two never compete for the pool. */
typedef struct {
    _Atomic uint64_t cur;            /* (gen<<8)|index; gen main-only, index 0..njobs (<=64) */
    _Atomic int njobs;               /* current batch job count */
    _Atomic int eids[64];            /* expert id per job */
    _Atomic int layer;               /* current batch layer */
    Slot *slots[64];                 /* pre-reserved cache slot per job (set before publish) */
    _Atomic int ready[64];           /* per-job load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;   /* ONLY for parking/waking idle workers */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
static PipePool g_pp;

static void *pipe_worker(void *arg){
    (void)arg; PipePool *p=&g_pp; uint64_t seen=0;
    for(;;){
        pthread_mutex_lock(&p->mx);
        while((atomic_load_explicit(&p->cur,memory_order_relaxed)>>8)==seen)
            pthread_cond_wait(&p->cv,&p->mx);
        pthread_mutex_unlock(&p->mx);
        for(;;){
            uint64_t c=atomic_load_explicit(&p->cur,memory_order_acquire);
            seen=c>>8;
            uint32_t i=(uint32_t)(c & 0xFF);
            if(i >= (uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed))
                break;                                /* batch drained -> re-park */
            if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,
                    memory_order_acq_rel,memory_order_relaxed)){
                int L  =atomic_load_explicit(&p->layer,memory_order_relaxed);
                int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed);
                Slot *s=p->slots[i];
                load_expert(p->m,L,eid,s);            /* pread (thread-safe: st.h uses pread) */
                expert_publish(p->m,L,s,eid);          /* make resident under g_mx */
                atomic_store_explicit(&p->ready[i],1,memory_order_release);
            }
        }
    }
    return NULL;
}
static void pipe_init(Model *m){
    if(g_pp.started) return;
    g_pp.m=m; g_pp.nw=g_pipe_nw; if(g_pp.nw>16) g_pp.nw=16; if(g_pp.nw<1) g_pp.nw=1;
    atomic_store(&g_pp.cur,0); atomic_store(&g_pp.njobs,0);
    pthread_mutex_init(&g_pp.mx,NULL); pthread_cond_init(&g_pp.cv,NULL);
    for(int i=0;i<g_pp.nw;i++) pthread_create(&g_pp.th[i],NULL,pipe_worker,NULL);
    g_pp.started=1;
}
/* Enqueue `njobs` already-reserved miss loads. eids[k]/slots[k] describe job k.
 * Returns immediately; workers run ahead and overlap the caller's matmul.
 * Order is load-bearing: write all batch state RELAXED, RELEASE-store cur to
 * publish, then wake parked workers. */
static void pipe_dispatch(Model *m,int layer,const int *eids,Slot **slots,int njobs){
    g_pp.m=m;
    atomic_store_explicit(&g_pp.njobs,njobs,memory_order_relaxed);
    atomic_store_explicit(&g_pp.layer,layer,memory_order_relaxed);
    for(int q=0;q<njobs;q++){ atomic_store_explicit(&g_pp.eids[q],eids[q],memory_order_relaxed); g_pp.slots[q]=slots[q]; }
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.ready[q],0,memory_order_relaxed); /* reset BEFORE publish */
    uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
    atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);                          /* PUBLISH */
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
}
/* Block until job q's load is published, then return (the slot is now resident
 * and safe to matmul). Spin-yield: the loads cost 0.5-3ms on real K3, the wake
 * latency of a condvar is ~5us — a yield storm is cheaper than the wake for
 * multi-ms reads, and matches colibri.c's PIPE default. */
static inline void pipe_wait(int q){
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
}

/* ---------- PILOT: cross-layer expert prefetch ----------
 * A single background thread drains a small ring of (layer,eid) prefetch
 * requests, each served by a full expert_get() (reserve+load+publish) on the
 * FUTURE layer's cache. After layer L's MoE, moe_forward's caller (step) feeds
 * the predicted top-K experts of layer L+1; by the time layer L+1's MoE runs
 * (after L+1's attention), those experts are already resident (hits) or, if
 * the pread is still in flight, expert_reserve() sees the in_use slot and
 * waits — so PILOT never changes the output, only the timing. Prediction uses
 * the current hidden state with the NEXT layer's router (same approximation as
 * colibri.c pilot_prefetch: it ignores the next layer's attention delta). */
typedef struct { int layer; int eid; } PilotReq;
static PilotReq pilot_q[512];
static _Atomic int pilot_w=0, pilot_r=0;
static pthread_mutex_t pilot_mx=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  pilot_cv=PTHREAD_COND_INITIALIZER;
static pthread_t pilot_th;
static int pilot_started=0;
static Model *pilot_m=NULL;

static void *pilot_worker(void *arg){
    (void)arg;
    for(;;){
        int r;
        pthread_mutex_lock(&pilot_mx);
        while((r=atomic_load_explicit(&pilot_r,memory_order_relaxed)) ==
              atomic_load_explicit(&pilot_w,memory_order_relaxed))
            pthread_cond_wait(&pilot_cv,&pilot_mx);
        PilotReq req=pilot_q[r & 511];
        atomic_store_explicit(&pilot_r,r+1,memory_order_release);
        pthread_mutex_unlock(&pilot_mx);
        if(req.layer<0) break;                       /* shutdown sentinel */
        Slot *dummy; expert_get(pilot_m,req.layer,req.eid,&dummy);  /* thread-safe */
    }
    return NULL;
}
static void pilot_enqueue(Model *m,int layer,int eid){
    if(!pilot_started){ pilot_m=m; pthread_create(&pilot_th,NULL,pilot_worker,NULL); pilot_started=1; }
    int w=atomic_load_explicit(&pilot_w,memory_order_relaxed);
    int r=atomic_load_explicit(&pilot_r,memory_order_acquire);
    if(w-r < 512){                                   /* ring not full -> drop on overflow (advisory) */
        pilot_q[w & 511].layer=layer; pilot_q[w & 511].eid=eid;
        atomic_store_explicit(&pilot_w,w+1,memory_order_release);
        pthread_mutex_lock(&pilot_mx); pthread_cond_signal(&pilot_cv); pthread_mutex_unlock(&pilot_mx);
    }
}
/* Predict layer `lnext`'s top-K routed experts from the current hidden state x
 * and enqueue them for background prefetch. Mirrors moe_forward's routing math
 * (sigmoid + e_score_bias + top-K) so the prediction is the same experts the
 * real layer L+1 would pick when its attention output is ~0. */
static void pilot_prefetch(Model *m,int lnext,const float *x){
    Cfg *c=&m->c; Layer *l=&m->L[lnext];
    int D=c->hidden, E=c->n_experts;
    int K = (g_pilot_k>0 && g_pilot_k<c->topk) ? g_pilot_k : c->topk;
    if(K>64) K=64;
    int bf16=l->bf16_resident;
    float *nrm=falloc(D);
    rmsnorm_row(nrm, x, l->post_ln, D, c->eps);
    float *logits=falloc(E), *adj=falloc(E);
    DENSE_MM(logits, nrm, l->gate_w, 1, D, E, bf16);
    for(int e=0;e<E;e++){
        float s=1.f/(1.f+expf(-logits[e]));
        adj[e]=s + (l->e_score_bias ? l->e_score_bias[e] : 0.f);
    }
    int picks[64];
    for(int kk=0;kk<K;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){
            int taken=0; for(int j=0;j<kk;j++) if(picks[j]==e){taken=1;break;}
            if(!taken && adj[e]>bv){ bv=adj[e]; best=e; }
        }
        if(best<0) break;
        picks[kk]=best;
        pilot_enqueue(m, lnext, best);
    }
    free(nrm); free(logits); free(adj);
}

/* ---------- Stable LatentMoE ---------- */
static void moe_forward(Model *m, Layer *l, int layer, const float *x, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk;
    int I = c->moe_inter, H = c->routed_hidden;
    int bf16 = l->bf16_resident;
    /* router logits = x @ gate_w^T, shape [E] */
    float *logits = falloc(E);
    DENSE_MM(logits, x, l->gate_w, 1, D, E, bf16);
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
        DENSE_MM(latent, x, l->routed_down, 1, D, H, bf16);
        in_dim = H;
    }
    float *acc = falloc(D);
    for (int d = 0; d < D; d++) acc[d] = 0.f;
    float *gg = falloc(I), *uu = falloc(I), *hh = falloc(H);
    /* PIPE (load ‖ matmul overlap): when enabled AND the per-layer cache can
     * hold all top-k experts at once (cap >= topk), reserve every selected
     * expert up front and dispatch the misses to the I/O worker pool. The
     * workers' preads then overlap this thread's matmuls: while we compute
     * expert kk, the pool loads kk+1..topk-1. cap < topk falls back to the
     * original serial expert_get() loop (a single slot cannot be reserved
     * twice, so reserving all top-k up front would deadlock on a tiny cache).
     * Output is bit-identical to the serial path: each expert's slot holds the
     * same weights, and the accumulator runs in the same top-k order. */
    int use_pipe = g_pipe && m->cache[layer].cap >= topk;
    Slot *es[64]; int miss_qid[64];
    if (use_pipe) {
        int meids[64]; Slot *mslots[64]; int nmiss = 0;
        for (int kk = 0; kk < topk; kk++) {
            int hit;
            expert_reserve(m, layer, idx[kk], &es[kk], &hit);
            miss_qid[kk] = hit ? -1 : nmiss;
            if (!hit) { meids[nmiss] = idx[kk]; mslots[nmiss] = es[kk]; nmiss++; }
        }
        if (nmiss > 0) pipe_dispatch(m, layer, meids, mslots, nmiss);
    }
    for (int kk = 0; kk < topk; kk++) {
        Slot *e;
        if (use_pipe) {
            if (miss_qid[kk] >= 0) pipe_wait(miss_qid[kk]);   /* block until load published */
            e = es[kk];
        } else {
            expert_get(m, layer, idx[kk], &e);
        }
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
        DENSE_MM(routed_out, acc, l->routed_up, 1, H, D, bf16);
    } else {
        routed_out = acc;
    }

    /* shared experts: 2 experts fused into one larger MLP (intermediate = moe_inter * n_shared) */
    if (c->n_shared > 0 && l->shared_gate) {
        int si = c->moe_inter * c->n_shared;
        float *sg = falloc(si), *su = falloc(si), *sh = falloc(D);
        DENSE_MM(sg, x, l->shared_gate, 1, D, si, bf16);
        DENSE_MM(su, x, l->shared_up,   1, D, si, bf16);
        for (int i = 0; i < si; i++) sg[i] = situ_gate(sg[i], su[i], c->situ_beta, c->situ_linear_beta);
        DENSE_MM(sh, sg, l->shared_down, 1, si, D, bf16);
        for (int d = 0; d < D; d++) routed_out[d] += sh[d];
        free(sg); free(su); free(sh);
    }
    memcpy(out, routed_out, D*sizeof(float));
    free(logits); free(scores); free(adj);
    free(gg); free(uu); free(hh);
    if (latent) { free(latent); free(acc); free(routed_out); }
    else free(routed_out);
}

/* ---------- DSA: Lightning Indexer (batch MoE for prefill) ----------
 * Routes all S tokens up front, groups the selections by expert, loads each
 * expert ONCE and applies it (batch matmul) to every token that picked it —
 * avoiding redundant loads when multiple tokens share an expert on a small
 * cache. Each token's partial expert outputs are stored in top-k order and
 * accumulated in that same order, so the result is BIT-IDENTICAL to calling
 * moe_forward() S times (same per-token math, same accumulation order). Used
 * only in the prefill path (S>1) when DSA=1. */
static void moe_forward_batch(Model *m, Layer *l, int layer,
                              const float *X, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk;
    int I = c->moe_inter, H = c->routed_hidden;
    int bf16 = l->bf16_resident;
    int topk = K <= 64 ? K : 64;
    int has_latent = (c->routed_hidden != D && l->routed_down);
    int in_dim = has_latent ? H : D;
    int acc_dim = has_latent ? H : D;

    /* batch router: logits[S][E] = X @ gate_w^T */
    float *logits = falloc((int64_t)S * E);
    DENSE_MM(logits, X, l->gate_w, S, D, E, bf16);

    /* per-token top-k selection (same math as moe_forward, run for each row) */
    int (*idx)[64] = malloc((size_t)S * sizeof(*idx));
    float (*val)[64] = malloc((size_t)S * sizeof(*val));
    for (int s = 0; s < S; s++) {
        float *ls = logits + (int64_t)s * E;
        float *scores = falloc(E), *adj = falloc(E);
        for (int e = 0; e < E; e++) scores[e] = 1.f / (1.f + expf(-ls[e]));
        for (int e = 0; e < E; e++) adj[e] = scores[e] + (l->e_score_bias ? l->e_score_bias[e] : 0.f);
        for (int kk = 0; kk < topk; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (idx[s][j]==e){taken=1;break;}
                if (!taken && adj[e] > bv) { bv = adj[e]; best = e; }
            }
            idx[s][kk] = best; val[s][kk] = best >= 0 ? scores[best] : 0.f;
        }
        if (c->moe_renormalize && K > 1) {
            float sm = 0.f; for (int kk = 0; kk < topk; kk++) sm += val[s][kk];
            if (sm < 1e-20f) sm = 1e-20f;
            for (int kk = 0; kk < topk; kk++) val[s][kk] /= sm;
        }
        for (int kk = 0; kk < topk; kk++) val[s][kk] *= c->routed_scale;
        free(scores); free(adj);
    }
    if (!m->hot_pinned && m->freq) {
        uint32_t *fl = m->freq + (int64_t)layer * E;
        for (int s = 0; s < S; s++)
            for (int kk = 0; kk < topk; kk++) if (idx[s][kk] >= 0) fl[idx[s][kk]]++;
    }

    /* latent projection for all S tokens at once */
    float *Latent = NULL;
    if (has_latent) {
        Latent = falloc((int64_t)S * H);
        DENSE_MM(Latent, X, l->routed_down, S, D, H, bf16);
    }

    /* hh_store[s][kk][H]: each token's expert output, stored in top-k order so
     * the final per-token accumulation is bit-identical to the serial path. */
    float *hh_store = falloc((int64_t)S * topk * H);
    memset(hh_store, 0, (size_t)S * topk * H * sizeof(float));

    /* expert-major pass: for each expert that appears, load ONCE and batch-apply
     * to every token that selected it (batch matmul on the float path, per-row
     * matmul_q on the int8 path — same per-row math either way). */
    float *gg = falloc((int64_t)S * I), *uu = falloc((int64_t)S * I), *hh = falloc((int64_t)S * H);
    int *tk = malloc((size_t)S * sizeof(int));   /* token indices that picked e */
    int *tkk = malloc((size_t)S * sizeof(int));  /* top-k position of e for each token */
    for (int e = 0; e < E; e++) {
        int nt = 0;
        for (int s = 0; s < S; s++) {
            for (int kk = 0; kk < topk; kk++) {
                if (idx[s][kk] == e) { tk[nt] = s; tkk[nt] = kk; nt++; break; }
            }
        }
        if (nt == 0) continue;
        Slot *ex; expert_get(m, layer, e, &ex);
        float *xin = falloc((int64_t)nt * in_dim);
        for (int t = 0; t < nt; t++) {
            int s = tk[t];
            const float *src = has_latent ? (Latent + (int64_t)s * H) : (X + (int64_t)s * D);
            memcpy(xin + (int64_t)t * in_dim, src, (size_t)in_dim * sizeof(float));
        }
        if (ex->kind == EXP_INT8) {
            for (int t = 0; t < nt; t++) {
                matmul_q(gg + (int64_t)t*I, xin + (int64_t)t*in_dim, ex->qg, ex->qgs, in_dim, I);
                matmul_q(uu + (int64_t)t*I, xin + (int64_t)t*in_dim, ex->qu, ex->qus, in_dim, I);
            }
        } else {
            matmul(gg, xin, ex->g, nt, in_dim, I);
            matmul(uu, xin, ex->u, nt, in_dim, I);
        }
        for (int t = 0; t < nt; t++)
            for (int i = 0; i < I; i++)
                gg[(int64_t)t*I + i] = situ_gate(gg[(int64_t)t*I + i], uu[(int64_t)t*I + i],
                                                 c->situ_beta, c->situ_linear_beta);
        if (ex->kind == EXP_INT8) {
            for (int t = 0; t < nt; t++)
                matmul_q(hh + (int64_t)t*H, gg + (int64_t)t*I, ex->qd, ex->qds, I, H);
        } else {
            matmul(hh, gg, ex->d, nt, I, H);
        }
        for (int t = 0; t < nt; t++) {
            int s = tk[t], kk = tkk[t];
            memcpy(hh_store + ((int64_t)s * topk + kk) * H, hh + (int64_t)t * H,
                   (size_t)H * sizeof(float));
        }
        free(xin);
    }

    /* per-token accumulation in top-k order (bit-identical to serial) */
    float *acc = falloc((int64_t)S * acc_dim);
    for (int s = 0; s < S; s++) {
        float *a = acc + (int64_t)s * acc_dim;
        for (int d = 0; d < acc_dim; d++) a[d] = 0.f;
        for (int kk = 0; kk < topk; kk++) {
            float w = val[s][kk];
            const float *hhrow = hh_store + ((int64_t)s * topk + kk) * H;
            for (int d = 0; d < acc_dim; d++) a[d] += w * hhrow[d];
        }
    }
    float *routed_out = falloc((int64_t)S * D);
    if (has_latent) {
        for (int s = 0; s < S; s++)
            if (l->routed_norm) rmsnorm_row(acc + (int64_t)s*H, acc + (int64_t)s*H, l->routed_norm, H, c->eps);
        DENSE_MM(routed_out, acc, l->routed_up, S, H, D, bf16);
    } else {
        memcpy(routed_out, acc, (size_t)S * D * sizeof(float));
    }

    /* shared experts (per-token; same math as serial) */
    if (c->n_shared > 0 && l->shared_gate) {
        int si = c->moe_inter * c->n_shared;
        float *sg = falloc((int64_t)S * si), *su = falloc((int64_t)S * si), *sh = falloc((int64_t)S * D);
        DENSE_MM(sg, X, l->shared_gate, S, D, si, bf16);
        DENSE_MM(su, X, l->shared_up,   S, D, si, bf16);
        for (int s = 0; s < S; s++)
            for (int i = 0; i < si; i++)
                sg[(int64_t)s*si + i] = situ_gate(sg[(int64_t)s*si + i], su[(int64_t)s*si + i],
                                                  c->situ_beta, c->situ_linear_beta);
        DENSE_MM(sh, sg, l->shared_down, S, si, D, bf16);
        for (int s = 0; s < S; s++)
            for (int d = 0; d < D; d++)
                routed_out[(int64_t)s*D + d] += sh[(int64_t)s*D + d];
        free(sg); free(su); free(sh);
    }

    memcpy(out, routed_out, (size_t)S * D * sizeof(float));
    free(logits); free(idx); free(val); free(hh_store); free(gg); free(uu); free(hh);
    free(tk); free(tkk); free(acc); free(routed_out);
    if (Latent) free(Latent);
}

/* dense MLP for layer 0 (gate/up/down + SiTU) */
static void mlp_forward(Model *m, Layer *l, const float *x, float *out) {
    Cfg *c = &m->c; int D = c->hidden, I = c->inter;
    int bf16 = l->bf16_resident;
    float *g = falloc(I), *u = falloc(I);
    DENSE_MM(g, x, l->mlp_gate, 1, D, I, bf16);
    DENSE_MM(u, x, l->mlp_up,   1, D, I, bf16);
    for (int i = 0; i < I; i++) g[i] = situ_gate(g[i], u[i], c->situ_beta, c->situ_linear_beta);
    DENSE_MM(out, g, l->mlp_down, 1, I, D, bf16);
    free(g); free(u);
}

/* ---------- sampling system (TOPK -> TOPP -> GRAMMAR -> temperature -> sample) ----------
 *
 * Pipeline (applied to logits from lm_head before sampling):
 *   1. TOPK: keep top-k logits, zero out the rest.
 *   2. TOPP: nucleus — keep smallest set with cumulative probability >= p.
 *   3. GRAMMAR: mask tokens whose first decoded byte is not admissible.
 *   4. temperature: softmax with temperature scaling.
 *   5. sample: xoshiro256** PRNG (deterministic, SEED-controlled).
 *
 * When TEMP<=0 the pipeline collapses to argmax (greedy) — byte-identical to
 * the original generate(). When SPEC>0 or GRAMMAR is set, generate() uses
 * speculative decoding (n-gram + grammar drafts, lossless verification).
 *
 * Speculative decoding is LOSSLESS:
 *   - Greedy (TEMP<=0): accept draft[k] iff argmax(logits[k*V]) == draft[k].
 *     Output == step-by-step greedy.
 *   - Sampling (TEMP>0): rejection sampling (Leviathan 2023). Accept draft x_d
 *     with prob p(x_d); on reject resample with x_d zeroed and renormalized.
 *     Output distribution == step-by-step sampling.
 */

/* ---- xoshiro256** PRNG (deterministic; seeded by SEED env, default 42) ---- */
static uint64_t g_xs[4];
static int   g_seed = 42;
static float g_temp = 0.f;     /* TEMP env (default 0 = greedy) */
static float g_topp = 1.f;    /* TOPP env (default 1.0 = no truncation) */
static int   g_topk = 0;      /* TOPK env (default 0 = no truncation) */
static int   g_spec = 0;      /* SPEC env (default 0 = off) */
static int   g_ret_all = 0;   /* step() returns S*V logits when set (spec verify) */

static inline uint64_t xs_rotl(uint64_t x, int k){ return (x<<k)|(x>>(64-k)); }
static inline uint64_t xs_next(void){
    uint64_t *s = g_xs;
    uint64_t result = xs_rotl(s[1]*5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = xs_rotl(s[3], 45);
    return result;
}
static inline double rndu(void){
    return (double)(xs_next() >> 11) * (1.0 / 9007199254740992.0);
}
static void rng_seed(uint64_t seed){
    /* SplitMix64 to seed the 4-lane state */
    for (int i = 0; i < 4; i++){
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        g_xs[i] = z;
    }
}

/* ---- argmax matching the original greedy semantics (best=0, strict >) ---- */
static int argmax_v(const float *lo, int V){
    if (V <= 0) return 0;
    int best = 0; float bv = lo[0];
    for (int v = 1; v < V; v++) if (lo[v] > bv) { bv = lo[v]; best = v; }
    return best;
}

/* ---- grammar integration ----
 * The grammar is a DRAFT SOURCE for speculative decoding AND a SAMPLING MASK:
 * at each step we compute the set of admissible first bytes (gr_admissible) and
 * zero out the probability of tokens whose first decoded byte is not in it.
 * The walker is fed every EMITTED token's bytes (gr_feed_k3) to stay in sync. */
static Grammar g_grammar;
static GrState  g_grstate;
static int      g_grammar_on = 0;
static int      g_grammar_armed = 0;
static int      g_grammar_max = 24;
static Tok     *g_T = NULL;       /* tokenizer (loaded only when grammar/sampling needs it) */
static int      g_eos_id = -1;

/* feed emitted-token bytes to the grammar walker */
static void gr_feed_k3(int t){
    if (!g_grammar_on || !g_T) return;
    char b[64]; int n = tok_decode(g_T, &t, 1, b, 63);
    for (int i = 0; i < n; i++){
        int r = gr_accept(&g_grstate, (unsigned char)b[i]);
        if (r == 1){ g_grammar_armed = 1; continue; }
        if (r < 0){ g_grammar_on = 0; return; }       /* walker died: no more drafts */
        if (!g_grammar_armed) continue;                /* preamble: wait for valid start */
        gr_state_init(&g_grstate, &g_grammar);          /* desync: restart from root */
        g_grammar_armed = 0;
        if (!g_grstate.alive){ g_grammar_on = 0; return; }
        if (gr_accept(&g_grstate, (unsigned char)b[i]) == 1) g_grammar_armed = 1;
    }
}

/* propose a grammar-forced draft: bytes -> tokens. 0 if grammar branches here. */
static int grammar_draft_k3(int *draft, int cap){
    if (!g_grammar_on || !g_grammar_armed || !g_T || cap < 1) return 0;
    char fb[512]; int nb = gr_forced(&g_grstate, fb, (int)sizeof fb - 1);
    if (nb <= 0) return 0;
    int nt = tok_encode(g_T, fb, nb, draft, cap);
    return nt > 0 ? nt : 0;
}

/* admissible first-byte mask: 256 = all admissible when grammar off; otherwise
 * the grammar's bitmap. *can_end = 1 if the parse can terminate here (EOS ok). */
static int grammar_admissible(unsigned char mask[32], int *can_end){
    if (!g_grammar_on){
        memset(mask, 0xFF, 32);
        if (can_end) *can_end = 1;
        return 256;
    }
    return gr_admissible(&g_grstate, mask, can_end);
}

/* ---- n-gram prompt-lookup draft (lossless; no draft model needed) ----
 * Searches the most-recent occurrence of the last bigram in the context and
 * proposes the tokens that followed it. Pure hypothesis: the verification
 * step accepts or rejects each token exactly as if generated one-by-one. */
static int ngram_draft(const int *ids, int len, int G, int *draft){
    if (len < 4 || G < 1) return 0;
    int a = ids[len-2], b = ids[len-1];
    for (int i = len-3; i >= 1; i--){
        if (ids[i-1] == a && ids[i] == b){
            int n = 0; for (int j = i+1; j < len && n < G; j++) draft[n++] = ids[j];
            return n;
        }
    }
    return 0;
}

/* ---- distribution buffers (reused across decodes) ---- */
static float *g_pbuf = NULL;
static int   *g_pidx = NULL;

/* max-heap sift-down on g_pbuf (key = g_pbuf[h[i]]) for partial top-k / top-p. */
static void topp_siftdown(int *h, int n, int i){
    int iv = h[i]; float kv = g_pbuf[iv];
    for (;;){
        int l = 2*i + 1;
        if (l >= n) break;
        int b = l; if (l+1 < n && g_pbuf[h[l+1]] > g_pbuf[h[l]]) b = l+1;
        if (g_pbuf[h[b]] <= kv) break;
        h[i] = h[b]; i = b;
    }
    h[i] = iv;
}

/* check token admissibility against the grammar mask (1 = admissible). */
static int tok_admissible(int id, const unsigned char adm[32], int can_end){
    if (!g_grammar_on) return 1;
    if (id == g_eos_id && can_end) return 1;
    if (!g_T) return 1;
    char b[8]; int nb = tok_decode(g_T, &id, 1, b, 7);
    if (nb <= 0) return 0;
    unsigned char fb = (unsigned char)b[0];
    return (adm[fb>>3] & (1u << (fb & 7))) ? 1 : 0;
}

/* build the sampling distribution in g_pbuf (indexed by token-id).
 * Pipeline: GRAMMAR-mask -> softmax(temp) -> TOPK -> TOPP.
 * (Equivalent to TOPK->TOPP->GRAMMAR->temp because the grammar mask removes
 *  tokens before the softmax, so TOPK/TOPP never select masked tokens.)
 * Requires: g_pbuf, g_pidx (lazily allocated), g_temp, g_topp, g_topk. */
static void dist_build(const float *lo, int V){
    if (!g_pbuf) { g_pbuf = falloc(V); g_pidx = malloc((size_t)V * sizeof(int)); }

    unsigned char adm[32]; int can_end;
    grammar_admissible(adm, &can_end);

    /* find max over admissible (skipping NaN) */
    int mxi = -1; float mx = -INFINITY;
    for (int i = 0; i < V; i++){
        float v = lo[i];
        if (v != v) continue;
        if (!tok_admissible(i, adm, can_end)) continue;
        if (mxi < 0 || v > mx){ mx = v; mxi = i; }
    }
    if (mxi < 0){
        /* no admissible token: put mass on EOS if grammar allows end, else token 0 */
        for (int i = 0; i < V; i++) g_pbuf[i] = 0.f;
        if (g_grammar_on && can_end && g_eos_id >= 0) g_pbuf[g_eos_id] = 1.f;
        else g_pbuf[0] = 1.f;
        return;
    }

    /* softmax with temperature, masking non-admissible / NaN tokens to 0 */
    float invt = 1.f / (g_temp > 1e-4f ? g_temp : 1e-4f);
    double s = 0;
    for (int i = 0; i < V; i++){
        float v = lo[i];
        if (v != v){ g_pbuf[i] = 0.f; continue; }
        if (!tok_admissible(i, adm, can_end)){ g_pbuf[i] = 0.f; continue; }
        g_pbuf[i] = expf((v - mx) * invt);
        s += g_pbuf[i];
    }
    if (!isfinite(s) || s <= 0.0){
        for (int i = 0; i < V; i++) g_pbuf[i] = 0.f;
        g_pbuf[mxi] = 1.f;
        return;
    }
    for (int i = 0; i < V; i++) g_pbuf[i] /= (float)s;

    /* TOPK: keep top-k probabilities, renormalize the survivors */
    if (g_topk > 0 && g_topk < V){
        for (int i = 0; i < V; i++) g_pidx[i] = i;
        for (int i = V/2-1; i >= 0; i--) topp_siftdown(g_pidx, V, i);
        int out = V;
        for (int kk = 0; kk < V - g_topk; kk++){
            int root = g_pidx[0];
            g_pidx[0] = g_pidx[--out]; g_pidx[out] = root;
            if (out > 0) topp_siftdown(g_pidx, out, 0);
        }
        double sk = 0;
        for (int i = out; i < V; i++) sk += g_pbuf[g_pidx[i]];
        for (int i = 0; i < out; i++) g_pbuf[g_pidx[i]] = 0.f;
        if (sk > 0) for (int i = out; i < V; i++) g_pbuf[g_pidx[i]] /= (float)sk;
    }

    /* TOPP: nucleus — keep smallest set with cumulative probability >= p */
    if (g_topp > 0.f && g_topp < 1.f){
        for (int i = 0; i < V; i++) g_pidx[i] = i;
        for (int i = V/2-1; i >= 0; i--) topp_siftdown(g_pidx, V, i);
        double cum = 0, s2 = 0; int out = V;
        do {
            int root = g_pidx[0];
            g_pidx[0] = g_pidx[--out]; g_pidx[out] = root;
            s2 += g_pbuf[root]; cum += g_pbuf[root];
            if (out > 0) topp_siftdown(g_pidx, out, 0);
        } while (cum < g_topp && out > 0);
        for (int i = 0; i < out; i++) g_pbuf[g_pidx[i]] = 0.f;
        float s2f = (float)s2;
        if (s2f > 0) for (int i = out; i < V; i++) g_pbuf[g_pidx[i]] /= s2f;
    }
}

/* sample from g_pbuf; ban>=0 excludes that token (renormalizing on the fly).
 * Used for rejection sampling on speculative rejection. */
static int dist_sample(int V, int ban){
    double z = 1.0 - (ban >= 0 ? g_pbuf[ban] : 0.0);
    if (z <= 1e-12) z = 1e-12;
    double u = rndu() * z, cum = 0;
    for (int i = 0; i < V; i++){
        if (i == ban) continue;
        cum += g_pbuf[i];
        if (cum >= u) return i;
    }
    for (int i = V-1; i >= 0; i--) if (i != ban && g_pbuf[i] > 0) return i;
    return 0;
}

/* pick next token: greedy if g_temp<=0 (byte-identical to the original), else
 * the full sampling pipeline (TOPK/TOPP/grammar/temperature). ban = token
 * excluded because it was rejected by speculative verification (sampling only). */
static int pick_tok(const float *lo, int V, int ban){
    if (g_temp <= 0.f) return argmax_v(lo, V);
    dist_build(lo, V);
    return dist_sample(V, ban);
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

    /* KDA_CHUNK: chunk size for chunkwise KDA prefill (default 64).
     * =1 forces the original token-by-token recurrent path (for testing). */
    int kda_chunk = getenv("KDA_CHUNK") ? atoi(getenv("KDA_CHUNK")) : 64;
    if (kda_chunk < 1) kda_chunk = 64;

    /* ---- chunkwise prefill path (S > 1 and KDA_CHUNK > 1) ----
     * Layer-major: embed all S tokens, then for each layer process the full
     * sequence. KDA layers use kda_forward_chunk (parallel intra-chunk delta
     * rule); MLA layers and all MLPs run token-by-token (same math, different
     * loop order). Decode (S=1) and KDA_CHUNK=1 fall through to the original
     * token-by-token path below. The chunkwise path is also forced when
     * g_ret_all is set (speculative verification needs logits for ALL S
     * positions, which the token-by-token path discards). */
    if (S > 1 && (kda_chunk > 1 || g_ret_all)) {
        /* embed all S tokens -> Xs[S][D] */
        float **Xs = malloc(S * sizeof(float*));
        for (int s = 0; s < S; s++) {
            int id = ids[s];
            float *x = falloc(D);
            if (m->bf16_resident) {
                const uint16_t *erow = (const uint16_t*)(const void*)m->embed + (int64_t)id * D;
                for (int d = 0; d < D; d++) x[d] = bf16_to_f32(erow[d]);
            } else {
                memcpy(x, m->embed + (int64_t)id * D, D*sizeof(float));
            }
            Xs[s] = x;
        }
        for (int i = 0; i < c->n_layers; i++) {
            Layer *l = &m->L[i];
            /* attention: MLA token-by-token, KDA chunkwise */
            if (c->is_mla[i]) {
                for (int s = 0; s < S; s++) {
                    float *nrm = falloc(D);
                    rmsnorm_row(nrm, Xs[s], l->in_ln, D, c->eps);
                    float *attn_out = falloc(D);
                    mla_forward(m, l, i, nrm, pos_base + s, attn_out);
                    for (int d = 0; d < D; d++) Xs[s][d] += attn_out[d];
                    free(nrm); free(attn_out);
                }
            } else {
                int s = 0;
                while (s < S) {
                    int C = kda_chunk;
                    if (s + C > S) C = S - s;
                    float *Xc = falloc((int64_t)C * D);
                    for (int t = 0; t < C; t++)
                        rmsnorm_row(Xc + (int64_t)t * D, Xs[s + t], l->in_ln, D, c->eps);
                    float *out = falloc((int64_t)C * D);
                    kda_forward_chunk(m, l, i, Xc, C, pos_base + s, out);
                    for (int t = 0; t < C; t++)
                        for (int d = 0; d < D; d++)
                            Xs[s + t][d] += out[(int64_t)t * D + d];
                    free(Xc); free(out);
                    s += C;
                }
            }
            /* MLP: DSA batches all S tokens through one moe_forward_batch when
             * enabled (loads each expert once, bit-identical to the serial path);
             * otherwise the original token-by-token loop. */
            if (g_dsa && c->is_moe[i]) {
                float *Nrm = falloc((int64_t)S * D);
                for (int s = 0; s < S; s++)
                    rmsnorm_row(Nrm + (int64_t)s*D, Xs[s], l->post_ln, D, c->eps);
                float *Mlp = falloc((int64_t)S * D);
                moe_forward_batch(m, l, i, Nrm, S, Mlp);
                for (int s = 0; s < S; s++)
                    for (int d = 0; d < D; d++) Xs[s][d] += Mlp[(int64_t)s*D + d];
                free(Nrm); free(Mlp);
            } else {
                for (int s = 0; s < S; s++) {
                    float *nrm = falloc(D);
                    rmsnorm_row(nrm, Xs[s], l->post_ln, D, c->eps);
                    float *mlp_out = falloc(D);
                    if (c->is_moe[i]) moe_forward(m, l, i, nrm, mlp_out);
                    else               mlp_forward(m, l, nrm, mlp_out);
                    for (int d = 0; d < D; d++) Xs[s][d] += mlp_out[d];
                    free(nrm); free(mlp_out);
                }
            }
            /* PILOT: predict next MoE layer's experts from the current hidden
             * state and preload them on a background thread so they are cache
             * hits by the time layer i+1 runs. Advisory — a wrong prediction
             * just becomes a miss; the output is unchanged. Uses the last
             * token's state as the predictor (same heuristic as colibri.c). */
            if (g_pilot && i + 1 < c->n_layers && c->is_moe[i+1])
                pilot_prefetch(m, i + 1, Xs[S - 1]);
        }
        float *logit;
        if (g_ret_all && S > 1){
            /* speculative verification: logits for ALL S positions [S*V] */
            logit = falloc((int64_t)S * c->vocab);
            for (int s = 0; s < S; s++){
                float *nrm = falloc(D);
                rmsnorm_row(nrm, Xs[s], m->final_norm, D, c->eps);
                DENSE_MM(logit + (int64_t)s * c->vocab, nrm, m->lm_head, 1, D, c->vocab, m->bf16_resident);
                free(nrm);
            }
        } else {
            float *final_x = falloc(D);
            rmsnorm_row(final_x, Xs[S - 1], m->final_norm, D, c->eps);
            logit = falloc(c->vocab);
            DENSE_MM(logit, final_x, m->lm_head, 1, D, c->vocab, m->bf16_resident);
            free(final_x);
        }
        for (int s = 0; s < S; s++) free(Xs[s]);
        free(Xs);
        m->token_count += S; m->freq_token_count += S;
        m->kv_len = pos_base + S;
        return logit;
    }

    /* ---- original token-by-token path (decode or KDA_CHUNK=1) ---- */
    float *last = NULL;
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        int id = ids[s];
        float *x = falloc(D);
        /* embed lookup: in bf16-resident mode the table stores raw bf16 bytes
         * (2B/elem, half of fp32) so memcpy'ing D*sizeof(float) would read past
         * the row AND reinterpreted bf16 bytes as floats. Convert per-element. */
        if (m->bf16_resident) {
            const uint16_t *erow = (const uint16_t*)(const void*)m->embed + (int64_t)id * D;
            for (int d = 0; d < D; d++) x[d] = bf16_to_f32(erow[d]);
        } else {
            memcpy(x, m->embed + (int64_t)id * D, D*sizeof(float));
        }
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
            /* PILOT: after layer i's MoE, predict layer i+1's routed experts
             * from the updated hidden state and preload them on a background
             * thread. They overlap layer i+1's attention as cache hits. */
            if (g_pilot && i + 1 < c->n_layers && c->is_moe[i+1])
                pilot_prefetch(m, i + 1, x);
        }
        if (last) free(last);
        last = x;
    }
    m->token_count += S; m->freq_token_count += S;
    m->kv_len = pos_base + S;
    float *final_x = falloc(D);
    rmsnorm_row(final_x, last, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    DENSE_MM(logit, final_x, m->lm_head, 1, D, c->vocab, m->bf16_resident);
    free(last); free(final_x);
    return logit;
}

/* ---------- generation ----------
 * Simple loop (no SPEC, no GRAMMAR): one step per token, pick_tok for greedy
 * (TEMP<=0, byte-identical to the original) or full sampling pipeline.
 * Speculative loop (SPEC>0 or GRAMMAR set): n-gram + grammar drafts verified
 * in a single batched forward (step with g_ret_all=1). Lossless: greedy uses
 * argmax match, sampling uses rejection sampling (Leviathan 2023). */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c; int V = c->vocab;
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit;
    if (m->kv_loaded && m->saved_logits) {
        /* KVSAVE warm start: KV/KDA caches were restored from disk, skip
         * re-prefill and use the saved logits to pick the first token. */
        logit = m->saved_logits;
        m->saved_logits = NULL;
    } else {
        logit = step(m, prompt, np, 0);
        /* KVSAVE: persist KV/KDA state right after prefill so the next run
         * with the same prompt can warm-start (skip re-prefill). */
        if (g_kvsave_dir) {
            char path[1024];
            uint64_t sid = kv_session_id(prompt, np);
            snprintf(path, sizeof(path), "%s/kv_%llu.bin",
                     g_kvsave_dir, (unsigned long long)sid);
            kv_save(m, path, m->kv_len, logit);
        }
    }

    if (g_spec <= 0 && !g_grammar_on) {
        /* ---- simple per-step loop (preserves original greedy behavior) ---- */
        int len = np;
        for (int s = 0; s < n_new; s++) {
            rss_guard_evict(m);   /* safe point: between tokens */
            int next = pick_tok(logit, V, -1);
            free(logit); logit = NULL;
            out[len++] = next;
            if (s == n_new - 1) break;
            int one = next;
            logit = step(m, &one, 1, len - 1);
        }
        if (logit) free(logit);
        return;
    }

    /* ---- speculative decoding loop (lossless) ---- */
    int draft_cap = 64; if (g_spec > 63) g_spec = 63;
    int *draft = malloc(draft_cap * sizeof(int));
    int kv = np, emitted = 0, carry_ban = -1;
    while (emitted < n_new) {
        rss_guard_evict(m);   /* safe point: between tokens */
        int next = pick_tok(logit, V, carry_ban); carry_ban = -1;
        free(logit); logit = NULL;
        out[kv] = next; emitted++;
        gr_feed_k3(next);
        if (emitted >= n_new) break;

        /* draft source: grammar first (high acceptance where it forces), then n-gram */
        int g = 0;
        if (g_grammar_on){
            int gc = g_grammar_max < draft_cap ? g_grammar_max : draft_cap;
            g = grammar_draft_k3(draft, gc);
        }
        if (!g && g_spec > 0){
            g = ngram_draft(out, kv + 1, g_spec, draft);
        }
        if (g > n_new - emitted) g = n_new - emitted;
        if (kv + 1 + g + 1 > m->max_t) g = m->max_t - kv - 2;
        if (g < 0) g = 0;

        if (g == 0){
            /* no draft: fall back to single-token step (path identical to simple loop) */
            int one = next;
            logit = step(m, &one, 1, kv);
            kv += 1;
            continue;
        }

        int S = 1 + g;
        int batch[64]; batch[0] = next;
        memcpy(batch + 1, draft, g * sizeof(int));
        g_ret_all = 1;
        float *lo = step(m, batch, S, kv);
        g_ret_all = 0;

        int k = 0;
        while (k < g && emitted < n_new){
            int accept;
            if (g_temp <= 0.f){
                accept = (argmax_v(lo + (int64_t)k * V, V) == draft[k]);
            } else {
                dist_build(lo + (int64_t)k * V, V);
                accept = (rndu() < g_pbuf[draft[k]]);
            }
            if (!accept){
                if (g_temp > 0.f) carry_ban = draft[k];
                break;
            }
            out[kv + 1 + k] = draft[k]; emitted++;
            gr_feed_k3(draft[k]); k++;
        }
        /* the logit at position k is the verification result for the next step */
        logit = falloc(V); memcpy(logit, lo + (int64_t)k * V, V * sizeof(float)); free(lo);
        kv += 1 + k;
    }
    free(draft);
    if (logit) free(logit);
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
    /* ---- OMP_HOT=1: permanent OpenMP hot-thread tuning (must be first) ----
     * libgomp reads OMP_/GOMP_ env vars in a CONSTRUCTOR that runs before
     * main(), so setenv() here and continuing would be too late. Instead, on
     * first entry seed the winning defaults (respecting anything the user
     * already set with overwrite=0), then re-exec self once so a fresh
     * libgomp constructor picks them up. The K3_OMP_TUNED sentinel guards
     * the exec so we re-exec at most once. K3_NO_OMP_TUNE=1 is a kill-switch.
     * argv is passed verbatim to execv(), so this must be the first stmt.
     * Reference: colibri.c "Permanent OpenMP hot-thread tuning" block. */
    if (getenv("OMP_HOT") && atoi(getenv("OMP_HOT")) &&
        !getenv("K3_OMP_TUNED") && !getenv("K3_NO_OMP_TUNE")) {
        setenv("OMP_WAIT_POLICY", "active", 0);   /* keep the team hot across tiny per-expert matmul regions */
        setenv("GOMP_SPINCOUNT", "200000", 0);    /* spin briefly, then yield so long disk waits don't burn a core */
        setenv("KMP_BLOCKTIME", "200", 0);        /* LLVM libomp: 200ms blocktime (libgomp ignores KMP_*) */
        setenv("OMP_PROC_BIND", "close", 0);      /* pack the team onto adjacent cores for cache locality */
        setenv("OMP_DYNAMIC", "FALSE", 0);        /* fixed team size: no per-region thread-count churn */
        setenv("K3_OMP_TUNED", "1", 1);
#ifdef __linux__
        fprintf(stderr, "[OMP] hot-thread tuning: re-exec once (K3_NO_OMP_TUNE=1 to skip)\n");
        /* execv preserves the CPU affinity mask. If the user exported
         * OMP_PROC_BIND/OMP_PLACES, libgomp's constructor already bound THIS
         * thread to place 0 before main() ran; the re-exec'd image would
         * inherit that 1-core mask and jail the whole team on one core.
         * Reset to all online CPUs so the fresh libgomp binds from the full set. */
#ifdef CPU_SETSIZE
        { cpu_set_t all; CPU_ZERO(&all);
          long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
          if (ncpu > CPU_SETSIZE) ncpu = CPU_SETSIZE;
          for (long i = 0; i < ncpu; i++) CPU_SET((int)i, &all);
          if (sched_setaffinity(0, sizeof(all), &all) != 0)
              perror("[OMP] sched_setaffinity pre-reexec (continuing)"); }
#endif
        execv("/proc/self/exe", argv);            /* returns only on failure -> fall through and run untuned */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
    g_omp_hot = (getenv("OMP_HOT") && atoi(getenv("OMP_HOT"))) ? 1 : 0;
#ifdef _OPENMP
    if (g_omp_hot) {
        /* Set the thread count and pre-create the team with a one-shot
         * parallel region. With OMP_WAIT_POLICY=active the team then stays
         * hot (spinning) between the per-expert matmul regions for the
         * lifetime of the process — the "permanent hot-thread pool". */
        omp_set_num_threads(omp_get_max_threads());
        #pragma omp parallel
        { /* no-op: just create the team so the first real matmul is free */ }
    }
#endif

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

    /* ---- sampling / speculative / grammar configuration (env vars) ----
     * TEMP : 0 (default) = greedy; >0 = sampling temperature
     * TOPP : 1.0 (default) = no nucleus; <1.0 = keep smallest set with cum>=p
     * TOPK : 0 (default) = no top-k; >0 = keep only top-k logits
     * SEED : 42 (default) = xoshiro256** seed for sampling reproducibility
     * SPEC : 0 (default) = off; >0 = speculative decoding draft size K
     * GRAMMAR: path to .gbnf (or .json JSON-Schema) file; enables grammar drafts
     *          AND grammar mask (tokens whose first byte is not admissible are
     *          zeroed before sampling). Fail-soft: bad grammar -> no grammar. */
    { const char *e = getenv("TEMP"); if (e) g_temp = (float)atof(e); }
    { const char *e = getenv("TOPP"); if (e) g_topp = (float)atof(e);
      if (g_topp < 0.f) g_topp = 0.f; if (g_topp > 1.f) g_topp = 1.f; }
    { const char *e = getenv("TOPK"); if (e) g_topk = atoi(e); if (g_topk < 0) g_topk = 0; }
    { const char *e = getenv("SEED"); if (e) g_seed = atoi(e); rng_seed((uint64_t)g_seed); }
    { const char *e = getenv("SPEC"); if (e) g_spec = atoi(e); if (g_spec < 0) g_spec = 0; }

    /* ---- performance optimizations (all default OFF — engine is byte-identical
     * to the original serial path when none are set) ----
     * PIPE=1          : async expert-load pipeline (load ‖ matmul overlap)
     * PIPE_WORKERS=N  : I/O worker threads for PIPE (default 4)
     * PILOT=1         : cross-layer expert prefetch (predict L+1 from L's state)
     * PILOT_K=N       : experts to prefetch per layer (default = config topk)
     * DSA=1           : Lightning Indexer — batch MoE for prefill (load once) */
    { const char *e = getenv("PIPE"); if (e) g_pipe = atoi(e); if (g_pipe < 0) g_pipe = 0; }
    { const char *e = getenv("PIPE_WORKERS"); if (e) g_pipe_nw = atoi(e); if (g_pipe_nw < 1) g_pipe_nw = 4; }
    { const char *e = getenv("PILOT"); if (e) g_pilot = atoi(e); if (g_pilot < 0) g_pilot = 0; }
    { const char *e = getenv("PILOT_K"); if (e) g_pilot_k = atoi(e); if (g_pilot_k < 0) g_pilot_k = 0; }
    { const char *e = getenv("DSA"); if (e) g_dsa = atoi(e); if (g_dsa < 0) g_dsa = 0; }
    /* KVSAVE=<dir>  : persist KV/KDA cache to <dir>/kv_<sid>.bin (default off)
     * RSS_LIMIT=<GB> : background RSS watchdog, evicts experts / aborts on OOM */
    { const char *e = getenv("KVSAVE"); if (e && *e) g_kvsave_dir = e; }
    { const char *e = getenv("RSS_LIMIT"); if (e) g_rss_limit_gb = atof(e); }
    if (g_pipe)  pipe_init(&m);
    if (g_pipe || g_pilot || g_dsa)
        fprintf(stderr, "[OPT] PIPE=%d (workers=%d)  PILOT=%d (k=%d)  DSA=%d\n",
                g_pipe, g_pipe_nw, g_pilot,
                g_pilot_k > 0 ? g_pilot_k : c->topk, g_dsa);

    /* Grammar setup: load .gbnf or .json (compiled via schema_gbnf) into g_grammar.
     * Needs the tokenizer (tok_decode/tok_encode) for gr_feed_k3/grammar_draft_k3.
     * Fail-soft: a bad grammar file leaves g_grammar_on=0 (engine runs without it,
     * output unchanged). */
    const char *grammar_path = getenv("GRAMMAR");
    const char *schema_path  = (!grammar_path || !*grammar_path) ? getenv("SCHEMA") : NULL;
    if ((grammar_path && *grammar_path) || (schema_path && *schema_path)) {
        const char *gpath = (grammar_path && *grammar_path) ? grammar_path : schema_path;
        FILE *gf = fopen(gpath, "rb");
        if (!gf) { fprintf(stderr, "[GRAMMAR] cannot open %s\n", gpath); }
        else {
            fseek(gf, 0, SEEK_END); long gn = ftell(gf); fseek(gf, 0, SEEK_SET);
            char *gtxt = malloc((size_t)gn + 1);
            if (!gtxt || fread(gtxt, 1, (size_t)gn, gf) != (size_t)gn){
                fprintf(stderr, "[GRAMMAR] failed to read %s\n", gpath);
                fclose(gf); free(gtxt);
            } else {
                fclose(gf); gtxt[gn] = 0;
                /* if first non-space byte is '{', treat as JSON-Schema -> compile to GBNF */
                const char *p = gtxt; while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') p++;
                if (*p == '{'){
                    char serr[160];
                    char *gbnf = schema_to_gbnf(gtxt, serr, sizeof serr);
                    free(gtxt);
                    if (!gbnf){ fprintf(stderr,"[SCHEMA] %s: %s (running without grammar)\n",gpath,serr); }
                    else { gtxt = gbnf; gn = (long)strlen(gtxt); }
                }
                if (gtxt){
                    if (gr_parse(&g_grammar, gtxt) == 0){
                        gr_state_init(&g_grstate, &g_grammar);
                        if (g_grstate.alive){
                            /* load tokenizer (needed for gr_feed_k3 / grammar_draft_k3).
                             * Looks for tokenizer.json in the snapshot dir. */
                            char tpath[1024];
                            snprintf(tpath, sizeof tpath, "%s/tokenizer.json", snap);
                            static Tok T;
                            FILE *tf = fopen(tpath, "rb");
                            if (tf){
                                fclose(tf);
                                tok_load(&T, tpath);
                                g_T = &T;
                                /* find EOS token (look in the special set) */
                                const char *eos_strs[] = {"<|im_end|>", "</s>", "<eos>", NULL};
                                for (int s = 0; eos_strs[s] && g_eos_id < 0; s++){
                                    const char *es = eos_strs[s];
                                    g_eos_id = hm_get(&T.vocab, es, (int)strlen(es));
                                }
                            } else {
                                fprintf(stderr, "[GRAMMAR] no tokenizer.json in %s — grammar mask disabled (drafts only)\n", snap);
                            }
                            g_grammar_on = 1; g_grammar_armed = 0;
                            fprintf(stderr, "[GRAMMAR] %s: %d rules loaded (eos=%d)\n",
                                    gpath, g_grammar.n, g_eos_id);
                        } else {
                            fprintf(stderr, "[GRAMMAR] %s: grammar cannot be evaluated (left recursion?)\n", gpath);
                        }
                        free(gtxt);
                    } else {
                        fprintf(stderr, "[GRAMMAR] %s: %s\n", gpath, g_grammar.err);
                        free(gtxt);
                    }
                }
            }
        }
    }

    FILE *f = fopen(refpath, "rb");
    if (!f) { perror(refpath); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if (fread(buf,1,n,f)!=(size_t)n) {} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull; int *prompt = read_int_array(ref,"prompt_ids",&np); int *full = read_int_array(ref,"full_ids",&nfull);
    if (np == 0 || nfull == 0) { fprintf(stderr,"ref.json missing prompt_ids/full_ids\n"); return 1; }
    int n_new = nfull - np;

    /* KVSAVE: try to warm-start from a previously saved KV file. The session
     * id is an FNV-1a hash of the prompt token ids, so the same prompt resumes
     * the same KV file. If the file is missing/incompatible, kv_load returns
     * 0 and generate() falls through to a normal cold prefill. */
    if (g_kvsave_dir) {
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
        (void)mkdir(g_kvsave_dir, 0755);   /* ignore EEXIST */
#endif
        uint64_t sid = kv_session_id(prompt, np);
        char kvpath[1024];
        snprintf(kvpath, sizeof(kvpath), "%s/kv_%llu.bin", g_kvsave_dir,
                 (unsigned long long)sid);
        kv_load(&m, kvpath, np);
    }

    /* RSS_LIMIT: start the background RSS watchdog (every 5s). The thread
     * runs for the lifetime of generate(); it is joined before exit. */
    if (g_rss_limit_gb > 0) {
        pthread_create(&g_rss_guard_th, NULL, rss_guard_thread, &m);
        fprintf(stderr, "[RSS-GUARD] monitoring RSS every 5s (limit %.1f GB)\n",
                g_rss_limit_gb);
    }

    if (getenv("PPL") && atoi(getenv("PPL")) == 1) {
        double nll; double t = now_s();
        int scored = tf_nll(&m, full, nfull, np, &nll);
        double dt = now_s() - t;
        double tot = m.hits + m.miss;
        printf("TF-NLL: %.4f nats/token over %d tokens  |  ppl = %.2f\n", nll, scored, exp(nll));
        printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss);
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
        if (g_rss_limit_gb > 0) { atomic_store(&g_rss_guard_stop, 1); pthread_join(g_rss_guard_th, NULL); }
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
    if (g_rss_limit_gb > 0) { atomic_store(&g_rss_guard_stop, 1); pthread_join(g_rss_guard_th, NULL); }
    free(buf); free(arena); free(prompt); free(full); free(out);
    return 0;
}
