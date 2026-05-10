"""
Numerical verification harness for the Triton kernels in triton_kernels.py.

For each kernel, runs both the Triton implementation and a pure-PyTorch
reference on the same inputs, and reports max/mean absolute error.

Pass criteria:
  bf16:  mean < 1e-2, max < 0.1     (rounding-level)
  fp32:  mean < 1e-5, max < 1e-3

Anything above is almost certainly a kernel bug, not numerical noise. The
DCT projection bug found earlier produced mean=0.27, max=2.1 — orders of
magnitude above bf16 noise.

Usage: CUDA_VISIBLE_DEVICES=0 python verify_kernels.py
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from triton_kernels import (
    score_pages_triton,
    _score_pages_torch_fallback,
    topk_sort,
    topk_sort_torch,
    compress_pages_triton,
    assemble_kv_drop_triton,
)


# ---------------------------------------------------------------------------
# Helpers (inline DCT to avoid importing dct_page_attention which pulls in
# transformers and may fail in the env's sklearn/numpy mismatch).
# ---------------------------------------------------------------------------
def _dct(x, norm="ortho"):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v.to(torch.float32), dim=1)
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    V = Vc.real * torch.cos(k) - Vc.imag * torch.sin(k)
    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    return 2 * V.view(*x_shape)


def _idct(X, norm="ortho"):
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * torch.cos(k) - V_t_i * torch.sin(k)
    V_i = V_t_r * torch.sin(k) + V_t_i * torch.cos(k)
    V = torch.view_as_complex(torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2))
    v = torch.fft.ifft(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]
    return x.view(*x_shape)


def _dct_compress_page(x, comp_len):
    if comp_len >= x.shape[2]:
        return x
    bsz, nh, sl, hd = x.shape
    xm = x.transpose(1, 2).reshape(bsz, sl, nh * hd)
    xd = _dct(xm.transpose(1, 2), norm="ortho")[:, :, :comp_len]
    xi = _idct(xd, norm="ortho").transpose(1, 2) * math.sqrt(comp_len / sl)
    return xi.reshape(bsz, comp_len, nh, hd).transpose(1, 2).to(x.dtype)


def build_M(ps, cs, device, dtype):
    """Mirror _build_dct_projection_matrix WITH the contiguous() fix applied."""
    I = torch.eye(ps, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    M = _dct_compress_page(I, cs).squeeze(0).squeeze(0)
    return M.contiguous().to(dtype)


def report(name, triton_out, ref_out, dtype):
    diff = (triton_out.float() - ref_out.float()).abs()
    mx = diff.max().item()
    mn = diff.mean().item()
    if dtype == torch.bfloat16:
        ok = mn < 1e-2 and mx < 0.1
    else:
        ok = mn < 1e-5 and mx < 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:42s}  max={mx:.6f}  mean={mn:.6f}")
    return ok


# ---------------------------------------------------------------------------
# Test 1: compress_pages_triton  (the kernel that had the bug)
# ---------------------------------------------------------------------------
def test_compress():
    print("=== compress_pages_triton ===")
    device = "cuda"
    dtype = torch.bfloat16
    bsz, kv_heads, n_new, head_dim = 1, 8, 64, 128
    all_pass = True
    for ps, cs in [(16, 1), (16, 2), (16, 4), (32, 1), (32, 2), (32, 4), (32, 8), (64, 8)]:
        paged_x = torch.randn(bsz, kv_heads, n_new, ps, head_dim, device=device, dtype=dtype)
        M = build_M(ps, cs, device, dtype)
        out_triton = compress_pages_triton(paged_x, M)
        out_ref = torch.einsum("cs,bhnsd->bhncd", M, paged_x)
        all_pass &= report(f"compress  ps={ps:>2} cs={cs}", out_triton, out_ref, dtype)
    return all_pass


# ---------------------------------------------------------------------------
# Test 2: score_pages_triton  (4 dispatch paths)
# ---------------------------------------------------------------------------
def test_score():
    print("=== score_pages_triton ===")
    device = "cuda"
    dtype = torch.bfloat16
    bsz, num_q_heads, head_dim = 1, 32, 128
    num_kv_heads = 8
    num_kv_groups = num_q_heads // num_kv_heads  # 4
    all_pass = True
    # Cover each dispatch branch:
    #   max_max specialization, c4_g4 (max+mean), c1_g4 (cs=1+gq=4), generic
    cases = [
        # (num_pages, comp_size, scoring, group_agg, label)
        (2043, 2, "max", "max", "max+max ps=16 cs=2"),
        (1019, 4, "max", "max", "max+max ps=32 cs=4"),
        (1019, 4, "max", "mean", "c4_g4   ps=32 cs=4"),
        (2043, 1, "max", "mean", "c1_g4   ps=16 cs=1"),
        (1019, 1, "max", "max",  "c1_g4   ps=32 cs=1"),
        (1019, 4, "mean", "mean", "generic mean+mean cs=4"),
        (1019, 4, "sum",  "max",  "generic sum+max cs=4"),
    ]
    for num_pages, comp_size, sc, ga, label in cases:
        q = torch.randn(bsz, num_q_heads, 1, head_dim, device=device, dtype=dtype)
        ck = torch.randn(bsz, num_kv_heads, num_pages, comp_size, head_dim, device=device, dtype=dtype)
        out_triton = score_pages_triton(q, ck, sc, ga, num_kv_groups)
        out_ref = _score_pages_torch_fallback(q, ck, sc, ga, num_kv_groups)
        all_pass &= report(f"score  {label}", out_triton, out_ref, dtype=torch.float32)
    return all_pass


# ---------------------------------------------------------------------------
# Test 3: topk_sort  (single-stage AND two-stage paths)
# ---------------------------------------------------------------------------
def test_topk():
    print("=== topk_sort ===")
    device = "cuda"
    bsz, num_kv_heads = 1, 8
    all_pass = True
    cases = [
        # (num_pages, top_k, sort_ascending, label)
        (1019, 64,  False, "single-stage drop  ps=32 topk=64"),
        (1019, 64,  True,  "single-stage compr ps=32 topk=64"),
        (2043, 128, False, "two-stage    drop  ps=16 topk=128"),
        (2043, 128, True,  "two-stage    compr ps=16 topk=128"),
        (1024, 128, False, "boundary     ps=?  np=1024"),
        (1025, 128, False, "boundary     ps=?  np=1025"),
    ]
    for num_pages, top_k, sort_asc, label in cases:
        scores = torch.randn(bsz, num_kv_heads, num_pages, device=device, dtype=torch.float32)
        out_triton = topk_sort(scores, top_k, sort_ascending=sort_asc)
        out_ref = topk_sort_torch(scores, top_k, sort_ascending=sort_asc)
        # For drop mode, ordering can differ between paths but the SET of
        # indices must match (descending-score order, but ties are
        # permutation-equivalent at the K-th boundary).
        if sort_asc:
            ok = torch.equal(out_triton, out_ref)
        else:
            ok = torch.equal(
                out_triton.sort(dim=-1).values,
                out_ref.sort(dim=-1).values,
            )
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] topk  {label}")
        if not ok:
            mismatch = (out_triton != out_ref).sum().item()
            print(f"         mismatched entries: {mismatch}")
        all_pass &= ok
    return all_pass


# ---------------------------------------------------------------------------
# Test 4: assemble_kv_drop_triton  (the SDPA-path assembly)
# ---------------------------------------------------------------------------
def test_assemble_drop():
    print("=== assemble_kv_drop_triton ===")
    device = "cuda"
    dtype = torch.bfloat16
    bsz, num_kv_heads, head_dim = 1, 8, 128
    all_pass = True
    cases = [
        # (page_size, num_pages, top_k, sink_pages, recent_tokens, label)
        (16, 2043, 128, 1, 64,  "ps=16 np=2043 topk=128"),
        (32, 1019, 64,  1, 128, "ps=32 np=1019 topk=64"),
    ]
    for ps, np_, tk, sink_pg, recent_tok, label in cases:
        paged_k = torch.randn(bsz, num_kv_heads, np_, ps, head_dim, device=device, dtype=dtype)
        paged_v = torch.randn(bsz, num_kv_heads, np_, ps, head_dim, device=device, dtype=dtype)
        sink_k = torch.randn(bsz, num_kv_heads, sink_pg * ps, head_dim, device=device, dtype=dtype)
        sink_v = torch.randn(bsz, num_kv_heads, sink_pg * ps, head_dim, device=device, dtype=dtype)
        recent_k = torch.randn(bsz, num_kv_heads, recent_tok, head_dim, device=device, dtype=dtype)
        recent_v = torch.randn(bsz, num_kv_heads, recent_tok, head_dim, device=device, dtype=dtype)
        # Random selection (descending-score order, like drop-mode topk)
        scores = torch.randn(bsz, num_kv_heads, np_, device=device, dtype=torch.float32)
        sel_idx = torch.topk(scores, tk, dim=-1).indices.to(torch.int32)

        out_k, out_v = assemble_kv_drop_triton(
            paged_k, paged_v, sink_k, sink_v, recent_k, recent_v,
            sel_idx, None, None, original_position_rope=False,
        )

        # Torch reference: sink + selected pages (in selection order) + recent
        sel_long = sel_idx.long().unsqueeze(-1).unsqueeze(-1).expand(
            bsz, num_kv_heads, tk, ps, head_dim
        )
        sel_k = paged_k.gather(2, sel_long).reshape(bsz, num_kv_heads, tk * ps, head_dim)
        sel_v = paged_v.gather(2, sel_long).reshape(bsz, num_kv_heads, tk * ps, head_dim)
        ref_k = torch.cat([sink_k, sel_k, recent_k], dim=2)
        ref_v = torch.cat([sink_v, sel_v, recent_v], dim=2)

        all_pass &= report(f"assemble K  {label}", out_k, ref_k, dtype)
        all_pass &= report(f"assemble V  {label}", out_v, ref_v, dtype)
    return all_pass


# ---------------------------------------------------------------------------
# Test 5: stride sensitivity  (the class of bug we just found)
# ---------------------------------------------------------------------------
def test_stride_sensitivity():
    """Pass non-contiguous variants of the projection matrix to the compress
    kernel. A stride-aware kernel produces the same result regardless of
    layout. A stride-blind kernel (the bug we just fixed) returns garbage."""
    print("=== stride sensitivity (compress_pages_triton) ===")
    device = "cuda"
    dtype = torch.bfloat16
    bsz, kv_heads, n_new, head_dim = 1, 8, 64, 128
    ps, cs = 32, 4

    paged_x = torch.randn(bsz, kv_heads, n_new, ps, head_dim, device=device, dtype=dtype)
    M_contig = build_M(ps, cs, device, dtype)

    # Non-contiguous M built via transpose (mimics the bug we just fixed
    # before the .contiguous() patch)
    M_T = M_contig.t().contiguous().t()  # logical [cs, ps], strides (1, cs)

    out_contig = compress_pages_triton(paged_x, M_contig)
    out_noncon = compress_pages_triton(paged_x, M_T)

    diff = (out_contig.float() - out_noncon.float()).abs()
    ok = diff.max().item() < 0.1
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] contig-vs-transposed-M  max={diff.max().item():.6f}  "
        f"mean={diff.mean().item():.6f}"
    )
    if not ok:
        print(
            "         compress_pages_triton is stride-blind on M's row stride. "
            "Caller must hand it a contiguous M (the .contiguous() fix in "
            "_build_dct_projection_matrix)."
        )
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; this script needs a GPU.")
        sys.exit(1)
    torch.manual_seed(0)
    results = {
        "compress":           test_compress(),
        "score":              test_score(),
        "topk":               test_topk(),
        "assemble_drop":      test_assemble_drop(),
        "stride_sensitivity": test_stride_sensitivity(),
    }
    print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("All kernel checks passed.")


if __name__ == "__main__":
    main()
