# muon.c

Pure, dependency-free C implementation of the **Muon** optimizer.

Ported from [`muon.py`](https://github.com/KellerJordan/Muon) by Keller Jordan.

> **Muon** = **Mo**mentum **U**nidirectional **O**rthogonalized by **N**ewton-schulz

Muon runs SGD-momentum on each 2-D weight matrix, then replaces the update with its nearest orthogonal matrix via a Newton-Schulz iteration. This keeps weight updates on the Stiefel manifold and empirically outperforms Adam on hidden weight matrices at essentially the same compute cost.

---

## Algorithm

For a single weight matrix `W[rows × cols]`, one Muon step does:

**1. Nesterov SGD-momentum**
```
buf ← β·buf + (1−β)·grad        (EMA of gradients, β = 0.95)
G   ← (1−β)·grad + β·buf        (Nesterov lookahead)
```

**2. Newton-Schulz orthogonalization** (quintic, 5 iterations by default)
```
If rows > cols, work on Gᵀ  (tall → wide, transpose back at end)

X ← G / (‖G‖_F + ε)             (spectral-norm guard)
for i in 1..ns_steps:
    A ← X · Xᵀ
    B ← b·A + c·A²               (a=3.4445, b=−4.7750, c=2.0315)
    X ← a·X + B·X
```

**3. RMS scale**
```
X ← X · sqrt(max(1, rows/cols))
```

**4. Parameter update**
```
W ← W · (1 − lr·wd) − lr·X      (wd = weight decay, default 0)
```

The Newton-Schulz iteration converges to `UV^T` (the polar factor of `G`) in 5 steps. It does not produce an exact orthogonal matrix, but something like `US'V^T` where `S'` is near-identity — which in practice does not hurt model performance relative to exact orthogonalization.

---

## API

### `muon_init`

```c
MuonState *muon_init(
    float **param_ptrs,   // pointers into your flat params array, one per matrix
    float **grad_ptrs,    // pointers into your flat grads  array, one per matrix
    int    *rows_arr,     // rows of each weight matrix
    int    *cols_arr,     // cols of each weight matrix
    int     n,            // number of weight matrices
    float   lr,           // learning rate          (default: 0.02)
    float   momentum,     // SGD momentum beta      (default: 0.95)
    float   weight_decay, // AdamW-style decay      (default: 0.0)
    int     ns_steps,     // Newton-Schulz iters    (default: 5)
    int     nesterov      // use Nesterov momentum  (default: 1)
);
```

Allocates per-matrix momentum buffers and scratch space. All pointer arrays must remain valid for the lifetime of the returned state.

### `muon_step`

```c
void muon_step(MuonState *s);
```

Performs one optimizer step for all registered weight matrices. Call after backward pass, before zeroing gradients.

### `muon_free`

```c
void muon_free(MuonState *s);
```

Releases all heap memory owned by the state.

---

## Build

```bash
gcc -O2 -o muon_test muon.c -lm -DMUON_TEST && ./muon_test
```

`muon.c` is self-contained — no headers to install, no dependencies beyond `libc` and `libm`. To use it in your project, copy the file and `#include "muon.c"` or compile it alongside your source.

### Self-test output

```
=== muon.c self-test ===

Test 1: Newton-Schulz on 4x4 matrix
  Input Frobenius norm: 2.2790
  After NS5: Frobenius norm: 1.8402
  Orthogonality error (avg |XXᵀ - I|): 0.044009
  PASS

Test 2: Muon step on 4x4 weight
  Param[0] before: -0.002441, after: 0.053389
  Param changed: YES
  PASS

Test 3: Tall matrix 8x4 (transpose path)
  Param[0] before: 0.003746, after: -0.016602
  Param changed: YES
  PASS

Test 4: Weight decay with zero gradient
  Param norm before: 4.0000
  Param norm after 5 steps: 3.9602
  PASS

All tests passed.
```

---

## Which parameters to use Muon for

Muon is designed for **hidden 2-D weight matrices only**. Embeddings, classifier heads, and scalar parameters should use Adam/AdamW.

| Parameter | Optimizer |
|---|---|
| `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo` | **Muon** |
| `mlp_fc1`, `mlp_fc2` | **Muon** |
| `wte` (token embeddings) | Adam |
| `wpe` (position embeddings) | Adam |
| `lm_head` | Adam |

---

## Integration with microgpt.c

`microgpt.c` currently uses plain **Adam** (no weight decay) for all parameters uniformly. To add Muon, swap it for a hybrid: Muon for hidden weight matrices, Adam for everything else.

```c
// 1. Define Muon-eligible matrices (6 per layer: wq, wk, wv, wo, fc1, fc2)
#define N_MUON (n_layer * 6)
float *muon_param_ptrs[N_MUON];
float *muon_grad_ptrs[N_MUON];
int    muon_rows[N_MUON], muon_cols[N_MUON];

int idx = 0;
for (int li = 0; li < n_layer; li++) {
    // attn_wq  [n_embd × n_embd]
    muon_param_ptrs[idx] = params + offsets.attn_wq[li];
    muon_grad_ptrs[idx]  = grads  + offsets.attn_wq[li];
    muon_rows[idx] = n_embd; muon_cols[idx] = n_embd; idx++;
    // attn_wk
    muon_param_ptrs[idx] = params + offsets.attn_wk[li];
    muon_grad_ptrs[idx]  = grads  + offsets.attn_wk[li];
    muon_rows[idx] = n_embd; muon_cols[idx] = n_embd; idx++;
    // attn_wv
    muon_param_ptrs[idx] = params + offsets.attn_wv[li];
    muon_grad_ptrs[idx]  = grads  + offsets.attn_wv[li];
    muon_rows[idx] = n_embd; muon_cols[idx] = n_embd; idx++;
    // attn_wo
    muon_param_ptrs[idx] = params + offsets.attn_wo[li];
    muon_grad_ptrs[idx]  = grads  + offsets.attn_wo[li];
    muon_rows[idx] = n_embd; muon_cols[idx] = n_embd; idx++;
    // mlp_fc1  [4*n_embd × n_embd]
    muon_param_ptrs[idx] = params + offsets.mlp_fc1[li];
    muon_grad_ptrs[idx]  = grads  + offsets.mlp_fc1[li];
    muon_rows[idx] = 4 * n_embd; muon_cols[idx] = n_embd; idx++;
    // mlp_fc2  [n_embd × 4*n_embd]
    muon_param_ptrs[idx] = params + offsets.mlp_fc2[li];
    muon_grad_ptrs[idx]  = grads  + offsets.mlp_fc2[li];
    muon_rows[idx] = n_embd; muon_cols[idx] = 4 * n_embd; idx++;
}

// 2. Initialize
MuonState *muon = muon_init(muon_param_ptrs, muon_grad_ptrs,
                             muon_rows, muon_cols, idx,
                             /*lr=*/0.02f, /*momentum=*/0.95f,
                             /*weight_decay=*/0.0f, /*ns_steps=*/5,
                             /*nesterov=*/1);

// 3. In the training loop, after backward pass:
muon_step(muon);
// Adam step for wte, wpe, lm_head (unchanged from current microgpt.c)
memset(grads, 0, n_params * sizeof(float));

// 4. Cleanup
muon_free(muon);
```

### Hyperparameters

| Hyperparameter | Default | Notes |
|---|---|---|
| `lr` | `0.02` | Muon LR is not directly comparable to Adam LR |
| `momentum` | `0.95` | SGD momentum beta; rarely needs tuning |
| `weight_decay` | `0.0` | AdamW-style; apply to Muon params only if desired |
| `ns_steps` | `5` | Newton-Schulz iterations; 5 is the sweet spot |
| `nesterov` | `1` | Nesterov lookahead; almost always beneficial |

The learning rate should have built-in muP scaling — as you scale up model size you typically don't need to retune it.

---

## Implementation notes

### Transpose handling

The Python original uses PyTorch's `.mT` (matrix transpose) to ensure Newton-Schulz always operates on a wide (rows ≤ cols) matrix. The C port replicates this with a `transposed` flag and adjusted index macros — no physical copy of the matrix is made.

```c
int transposed = (rows > cols);
int m = transposed ? cols : rows;   // working rows (≤ n)
int n = transposed ? rows : cols;   // working cols (≥ m)
// X[i][j] accessed as: transposed ? X[j*cols + i] : X[i*cols + j]
```

### Scratch buffer layout

Newton-Schulz needs three temporary matrices: `A[m×m]`, `B[m×m]`, and `BX[m×n]`. The total scratch size per weight matrix is `2*m*m + m*n` floats, where `m = min(rows, cols)`. This is pre-allocated at `muon_init` time.

### Gradient buffer

Unlike the Python implementation (which mutates `grad` in-place via `lerp_`), the C port computes the Nesterov direction into a separate allocation `U` and leaves the caller's gradient buffer intact. The caller is responsible for zeroing gradients after the optimizer step.

---

## References

- [Original Python implementation](https://github.com/KellerJordan/Muon) — Keller Jordan
- [Blog post: Muon optimizer](https://kellerjordan.github.io/posts/muon/) — Keller Jordan
- [Theoretical background](https://jeremybernste.in/writing/deriving-muon) — Jeremy Bernstein
- [llm.c](https://github.com/karpathy/llm.c) — production C/CUDA ML reference
- [micro-gpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — the Python original this project ports
- [microgpt.c](https://github.com/loretoparisi/microgpt.c) — Microgpt
- [adamw.c](https://github.com/loretoparisi/adamw.c) — AdamW in Pure C; sibling file; same structural conventions

---

## License

MIT — same as the original Muon repository.
