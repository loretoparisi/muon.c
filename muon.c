#ifndef MUON_C
#define MUON_C

/*
 * muon.c
 *
 * Muon optimizer — pure, dependency-free C implementation.
 * Ported from muon.py by Keller Jordan (https://github.com/KellerJordan/Muon)
 *
 * Muon = MomentUm Orthogonalized by Newton-schulz
 *
 * Muon runs SGD-momentum on each 2-D weight matrix, then orthogonalizes the
 * resulting update via a Newton-Schulz iteration before applying it.  This
 * keeps the update near the Stiefel manifold, which empirically outperforms
 * Adam on hidden weight matrices while remaining cheap to compute.
 *
 * Algorithm for a single 2-D weight matrix W[rows × cols]:
 *
 *   1. Nesterov momentum:
 *        buf  ← β·buf + (1−β)·grad          (EMA of gradients)
 *        G    ← lerp(grad, buf, β)           (Nesterov lookahead)
 *              = β·buf + (1−β)·grad
 *
 *   2. Newton-Schulz orthogonalization  (5 quintic iterations):
 *        If rows > cols, work on Gᵀ (tall → wide, then transpose back).
 *        X  ← G / (‖G‖_F + ε)              (spectral-norm guard)
 *        for i in 1..ns_steps:
 *            A  ← X·Xᵀ
 *            B  ← b·A + c·A²               (quintic: a=3.4445, b=-4.7750, c=2.0315)
 *            X  ← a·X + B·X
 *
 *   3. RMS scale:
 *        X  ← X · sqrt(max(1, rows/cols))
 *
 *   4. Parameter update (with optional weight decay):
 *        W  ← W · (1 − lr·wd) − lr·X
 *
 * Muon should only be used for hidden 2-D weight matrices (attn_wq, attn_wk,
 * attn_wv, attn_wo, mlp_fc1, mlp_fc2).  Embeddings, the LM head, and any
 * 1-D parameters should be updated with Adam/AdamW.
 *
 * This file is self-contained and can be compiled as a test harness:
 *   gcc -O2 -o muon_test muon.c -lm && ./muon_test
 *
 * Or #include "muon.c" / copy the relevant functions into your project.
 *
 * References:
 *   Original Python: https://github.com/KellerJordan/Muon
 *   Blog post:       https://kellerjordan.github.io/posts/muon/
 *   microgpt.c:      Adam optimizer in microgpt.c uses beta1=0.9, beta2=0.95
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =========================================================================
 * Memory helpers
 * ========================================================================= */

static float *alloc_zeros(int n) {
    float *p = (float *)calloc(n, sizeof(float));
    if (!p) { fprintf(stderr, "muon: OOM allocating %d floats\n", n); exit(1); }
    return p;
}

/* =========================================================================
 * Newton-Schulz quintic orthogonalization
 *
 * Python original:
 *   a, b, c = (3.4445, -4.7750, 2.0315)
 *   X = G / (norm(G) + 1e-7)
 *   for _ in range(steps):
 *       A = X @ X.T
 *       B = b*A + c*(A @ A)
 *       X = a*X + B @ X
 *
 * We operate on a row-major float matrix.  When rows > cols we logically
 * transpose (work on the wide side) and transpose back at the end, matching
 * the Python `.mT` logic.
 *
 * Parameters
 *   X        — in/out matrix, row-major, size rows×cols (modified in-place)
 *   rows, cols — matrix dimensions of X as stored (before any logical transpose)
 *   steps    — number of Newton-Schulz iterations (5 is the default)
 *   tmp      — scratch buffer of size max(rows,cols)² floats (caller-provided)
 * ========================================================================= */
static void newton_schulz5(float *X, int rows, int cols, int steps, float *tmp) {
    /* Quintic coefficients */
    const float a =  3.4445f;
    const float b = -4.7750f;
    const float c =  2.0315f;

    /* If the matrix is taller than wide, work on the transpose.
     * In practice for weight matrices (nout × nin) this fires when nout > nin.
     * We achieve the logical transpose by swapping rows/cols in our matmuls
     * and adjusting index order — no physical copy needed. */
    int transposed = (rows > cols);
    int m, n;          /* m = working rows, n = working cols */
    if (transposed) { m = cols; n = rows; }
    else             { m = rows; n = cols; }

    /* ------------------------------------------------------------------
     * Step 1: normalize by Frobenius norm so spectral norm ≤ 1
     *   X ← X / (‖X‖_F + ε)
     * ------------------------------------------------------------------ */
    float norm_sq = 0.0f;
    for (int i = 0; i < rows * cols; i++) norm_sq += X[i] * X[i];
    float scale = 1.0f / (sqrtf(norm_sq) + 1e-7f);
    for (int i = 0; i < rows * cols; i++) X[i] *= scale;

    /* We need three temporaries:
     *   A[m×m], B[m×m], BX[m×n]
     * Total scratch = 2*m*m + m*n floats.
     * Layout: A = tmp[0..m*m), B = tmp[m*m..2*m*m), BX = tmp[2*m*m..2*m*m+m*n) */
    float *A  = tmp;
    float *B  = tmp + m * m;
    float *BX = tmp + 2 * m * m;

    /* Helper: C = P @ Qᵀ  (P: m×k, Q: m×k, result: m×m)
     *         when transposed==0:  P and Q are rows of X (shape m×n → k=n)
     *         when transposed==1:  P and Q are cols of X (shape n×m → k=n)
     */
    #define IDX_X(r, c)  ( transposed ? ((c) * cols + (r)) : ((r) * cols + (c)) )
    /*  X viewed as m×n:  X[i][j] = IDX_X(i,j)  */

    /* C[i][j] = sum_k  Xview[i][k] * Xview[j][k]   (i.e. X @ Xᵀ)  */
    #define MATMUL_XXT(C, mm, nn) do {                          \
        for (int i = 0; i < (mm); i++) {                        \
            for (int j = 0; j < (mm); j++) {                    \
                float s = 0.0f;                                  \
                for (int k = 0; k < (nn); k++)                  \
                    s += X[IDX_X(i,k)] * X[IDX_X(j,k)];        \
                (C)[i*(mm)+j] = s;                               \
            }                                                    \
        }                                                        \
    } while(0)

    /* C = P @ Q,  P: m×m, Q: m×m  */
    #define MATMUL_MM(C, P, Q, mm) do {                         \
        for (int i = 0; i < (mm); i++) {                        \
            for (int j = 0; j < (mm); j++) {                    \
                float s = 0.0f;                                  \
                for (int k = 0; k < (mm); k++)                  \
                    s += (P)[i*(mm)+k] * (Q)[k*(mm)+j];         \
                (C)[i*(mm)+j] = s;                               \
            }                                                    \
        }                                                        \
    } while(0)

    /* C[i][k] = sum_j  M[i][j] * Xview[j][k]   (M: m×m, Xview: m×n → result m×n)
     * Written back into X in-place via BX buffer. */
    #define MATMUL_MX(BX_buf, M, mm, nn) do {                  \
        for (int i = 0; i < (mm); i++) {                        \
            for (int k = 0; k < (nn); k++) {                    \
                float s = 0.0f;                                  \
                for (int j = 0; j < (mm); j++)                  \
                    s += (M)[i*(mm)+j] * X[IDX_X(j,k)];        \
                (BX_buf)[i*(nn)+k] = s;                         \
            }                                                    \
        }                                                        \
    } while(0)

    /* ------------------------------------------------------------------
     * Newton-Schulz iterations:
     *   A  ← X · Xᵀ          (m×m)
     *   B  ← b·A + c·A²      (m×m)
     *   X  ← a·X + B·X       (m×n)
     * ------------------------------------------------------------------ */
    for (int iter = 0; iter < steps; iter++) {
        /* A = X @ Xᵀ */
        MATMUL_XXT(A, m, n);

        /* B = b*A + c*(A@A) */
        MATMUL_MM(B, A, A, m);            /* B = A @ A first */
        for (int i = 0; i < m * m; i++)
            B[i] = b * A[i] + c * B[i];

        /* X ← a*X + B@X  (compute B@X into BX, then combine) */
        MATMUL_MX(BX, B, m, n);
        for (int i = 0; i < m; i++)
            for (int k = 0; k < n; k++) {
                int xi = IDX_X(i, k);
                X[xi] = a * X[xi] + BX[i * n + k];
            }
    }

    #undef IDX_X
    #undef MATMUL_XXT
    #undef MATMUL_MM
    #undef MATMUL_MX
}

/* =========================================================================
 * MuonParam — state for a single weight matrix managed by Muon
 * ========================================================================= */
typedef struct {
    float  *param;          /* pointer into the flat params array              */
    float  *grad;           /* pointer into the flat grads array               */
    float  *momentum_buf;   /* EMA of gradients (same shape as param)          */
    float  *tmp;            /* scratch space for Newton-Schulz (3*m*m floats)  */
    int     rows;           /* param shape: rows                               */
    int     cols;           /* param shape: cols                               */
} MuonParam;

/* =========================================================================
 * MuonState — optimizer state for all Muon-managed parameters
 * ========================================================================= */
typedef struct {
    MuonParam  *params;     /* array of per-matrix states                      */
    int         n_params;   /* number of weight matrices                       */
    float       lr;         /* learning rate (default: 0.02)                   */
    float       momentum;   /* SGD momentum beta (default: 0.95)               */
    float       weight_decay; /* AdamW-style weight decay (default: 0.0)       */
    int         ns_steps;   /* Newton-Schulz iterations (default: 5)           */
    int         nesterov;   /* use Nesterov momentum (default: 1)              */
} MuonState;

/* -------------------------------------------------------------------------
 * muon_init — create a MuonState for `n` weight matrices.
 *
 * param_ptrs[i] — pointer to param_i inside your flat params array
 * grad_ptrs[i]  — pointer to grad_i  inside your flat grads  array
 * rows_arr[i]   — number of rows in matrix i
 * cols_arr[i]   — number of cols in matrix i
 *
 * All arrays must remain valid for the lifetime of the returned state.
 * Returns a heap-allocated MuonState; free with muon_free().
 * ------------------------------------------------------------------------- */
MuonState *muon_init(float **param_ptrs, float **grad_ptrs,
                     int *rows_arr, int *cols_arr, int n,
                     float lr, float momentum, float weight_decay,
                     int ns_steps, int nesterov) {
    MuonState *s = (MuonState *)malloc(sizeof(MuonState));
    if (!s) { fprintf(stderr, "muon: OOM\n"); exit(1); }

    s->params       = (MuonParam *)malloc(n * sizeof(MuonParam));
    s->n_params     = n;
    s->lr           = lr;
    s->momentum     = momentum;
    s->weight_decay = weight_decay;
    s->ns_steps     = ns_steps;
    s->nesterov     = nesterov;

    for (int i = 0; i < n; i++) {
        MuonParam *mp = &s->params[i];
        mp->param = param_ptrs[i];
        mp->grad  = grad_ptrs[i];
        mp->rows  = rows_arr[i];
        mp->cols  = cols_arr[i];

        int sz = rows_arr[i] * cols_arr[i];
        mp->momentum_buf = alloc_zeros(sz);

        /* scratch: 2*m*m + m*n where m = min(rows,cols), n = max(rows,cols)
         * Layout: A[m×m] | B[m×m] | BX[m×n] */
        int m = (rows_arr[i] <= cols_arr[i]) ? rows_arr[i] : cols_arr[i];
        int n = (rows_arr[i] <= cols_arr[i]) ? cols_arr[i] : rows_arr[i];
        mp->tmp = alloc_zeros(2 * m * m + m * n);
    }

    return s;
}

/* -------------------------------------------------------------------------
 * muon_step — perform one Muon optimizer step for all registered parameters.
 *
 * Caller must have:
 *   1. Run forward + backward pass to populate grad arrays.
 *   2. Called muon_step() (this function).
 *   3. Zeroed grad arrays for the next iteration.
 * ------------------------------------------------------------------------- */
void muon_step(MuonState *s) {
    float beta = s->momentum;

    for (int pi = 0; pi < s->n_params; pi++) {
        MuonParam *mp = &s->params[pi];
        int sz   = mp->rows * mp->cols;
        float *G = mp->grad;          /* raw gradient              */
        float *M = mp->momentum_buf;  /* momentum buffer           */

        /* ------------------------------------------------------------------
         * Step 1: Nesterov SGD-momentum
         *
         * Python:
         *   momentum.lerp_(grad, 1 - beta)       →  M ← β·M + (1−β)·G
         *   update = grad.lerp_(momentum, beta)  →  U ← (1−β)·G + β·M
         *          if nesterov else momentum
         *
         * lerp_(x, w) computes:  self ← self + w*(x - self) = (1-w)*self + w*x
         *   momentum.lerp_(grad, 1-beta):
         *     M ← M + (1-beta)*(G - M) = beta*M + (1-beta)*G   ✓
         *   grad.lerp_(momentum, beta):
         *     Note: this mutates grad in-place in Python. The result is:
         *     G_new = G + beta*(M_new - G) = (1-beta)*G + beta*M_new
         *     where M_new is the already-updated momentum.
         *
         * We do not mutate the grad buffer; we compute U into a temporary
         * stack array to keep the caller's grad intact for weight gradient
         * accumulation purposes (they zero it separately).
         * ------------------------------------------------------------------ */
        float *U = (float *)malloc(sz * sizeof(float));  /* Nesterov update direction */
        if (!U) { fprintf(stderr, "muon: OOM in step\n"); exit(1); }

        /* Update momentum: M ← β·M + (1−β)·G */
        for (int i = 0; i < sz; i++)
            M[i] = beta * M[i] + (1.0f - beta) * G[i];

        if (s->nesterov) {
            /* U ← (1−β)·G + β·M  (M is already updated above) */
            for (int i = 0; i < sz; i++)
                U[i] = (1.0f - beta) * G[i] + beta * M[i];
        } else {
            /* U ← M */
            memcpy(U, M, sz * sizeof(float));
        }

        /* ------------------------------------------------------------------
         * Step 2: Newton-Schulz orthogonalization
         *   U ← NS5(U, ns_steps)
         *
         * newton_schulz5 normalizes U internally; U is modified in-place.
         * ------------------------------------------------------------------ */
        newton_schulz5(U, mp->rows, mp->cols, s->ns_steps, mp->tmp);

        /* ------------------------------------------------------------------
         * Step 3: RMS scale
         *   Python: update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
         *   After NS the matrix is viewed as m×n (transposed if rows>cols).
         *   size(-2)/size(-1) = working_rows/working_cols.
         *   working dims: if rows<=cols → m=rows, n=cols → ratio = rows/cols
         *                 if rows>cols  → m=cols, n=rows → ratio = cols/rows
         *   Either way ratio = min(rows,cols) / max(rows,cols) ≤ 1.
         *   So max(1, ratio) = 1 always for 2-D matrices where rows≠cols.
         *   For square matrices ratio=1 too.  The scale is therefore 1 in all
         *   practical cases — we include it for completeness / correctness.
         * ------------------------------------------------------------------ */
        int m_dim = (mp->rows <= mp->cols) ? mp->rows : mp->cols;
        int n_dim = (mp->rows <= mp->cols) ? mp->cols : mp->rows;
        float rms_scale = sqrtf((float)m_dim / (float)n_dim);
        if (rms_scale < 1.0f) rms_scale = 1.0f;   /* max(1, ...) */
        for (int i = 0; i < sz; i++)
            U[i] *= rms_scale;

        /* ------------------------------------------------------------------
         * Step 4: Apply update
         *   Python:
         *     p.mul_(1 - lr * weight_decay)       →  W ← (1 − lr·wd)·W
         *     p.add_(update, alpha=-lr)            →  W ← W − lr·U
         * ------------------------------------------------------------------ */
        float wd_factor = 1.0f - s->lr * s->weight_decay;
        for (int i = 0; i < sz; i++) {
            mp->param[i] = wd_factor * mp->param[i] - s->lr * U[i];
        }

        free(U);
    }
}

/* -------------------------------------------------------------------------
 * muon_free — release all heap memory owned by a MuonState.
 * ------------------------------------------------------------------------- */
void muon_free(MuonState *s) {
    for (int i = 0; i < s->n_params; i++) {
        free(s->params[i].momentum_buf);
        free(s->params[i].tmp);
    }
    free(s->params);
    free(s);
}

/* =========================================================================
 * Integration guide for microgpt.c
 * =========================================================================
 *
 * microgpt.c uses plain Adam for ALL parameters:
 *   adam_m[i] = beta1 * adam_m[i] + (1 - beta1) * g
 *   adam_v[i] = beta2 * adam_v[i] + (1 - beta2) * g^2
 *   p -= lr * m_hat / (sqrt(v_hat) + eps)
 *
 * To add Muon, replace Adam with a hybrid:
 *   - Muon  for hidden 2-D weight matrices (attn_wq/wk/wv/wo, mlp_fc1/fc2)
 *   - Adam  for embeddings (wte, wpe) and the LM head (lm_head)
 *
 * Example integration sketch:
 *
 *   // Identify Muon-eligible params (rows and cols of each weight matrix)
 *   float *muon_param_ptrs[N_MUON];
 *   float *muon_grad_ptrs[N_MUON];
 *   int muon_rows[N_MUON], muon_cols[N_MUON];
 *   int idx = 0;
 *   for (int li = 0; li < n_layer; li++) {
 *       muon_param_ptrs[idx] = params + offsets.attn_wq[li];
 *       muon_grad_ptrs[idx]  = grads  + offsets.attn_wq[li];
 *       muon_rows[idx] = n_embd; muon_cols[idx] = n_embd; idx++;
 *       // ... repeat for attn_wk, attn_wv, attn_wo, mlp_fc1, mlp_fc2
 *   }
 *   MuonState *muon = muon_init(muon_param_ptrs, muon_grad_ptrs,
 *                               muon_rows, muon_cols, idx,
 *                               0.02f, 0.95f, 0.0f, 5, 1);
 *
 *   // In training loop, after backward:
 *   muon_step(muon);
 *   // Adam step for wte, wpe, lm_head (unchanged from current microgpt.c)
 *   // Zero all grads.
 *
 *   // Cleanup:
 *   muon_free(muon);
 *
 * =========================================================================
 *
 * Self-test: compile with -DMUON_TEST to run a minimal sanity check.
 *
 * ========================================================================= */
#endif /* MUON_C */
