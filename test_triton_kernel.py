"""Correctness test: Triton assemble_kv_compressed vs PyTorch reference."""
import torch
from triton_kernels import assemble_kv_compressed_triton


def assemble_kv_compressed_pytorch(
    paged_k, paged_v, comp_k, comp_v,
    selected_indices, num_pages, page_size, comp_size, top_k,
):
    """Original PyTorch implementation (scatter-based)."""
    bsz, num_kv_heads = paged_k.shape[:2]
    head_dim = paged_k.shape[-1]
    num_unselected = num_pages - top_k
    middle_len = top_k * page_size + num_unselected * comp_size

    selected_mask = torch.zeros(bsz, num_kv_heads, num_pages, dtype=torch.bool,
                                device=selected_indices.device)
    selected_mask.scatter_(2, selected_indices, True)
    token_counts = torch.where(selected_mask, page_size, comp_size)
    page_offsets = torch.zeros(bsz, num_kv_heads, num_pages, dtype=torch.long,
                               device=selected_indices.device)
    page_offsets[:, :, 1:] = token_counts[:, :, :-1].cumsum(dim=-1)

    idx_sel = selected_indices[:, :, :, None, None].expand(
        bsz, num_kv_heads, top_k, page_size, head_dim)
    sel_k = torch.gather(paged_k, 2, idx_sel).reshape(bsz, num_kv_heads, top_k * page_size, head_dim)
    sel_v = torch.gather(paged_v, 2, idx_sel).reshape(bsz, num_kv_heads, top_k * page_size, head_dim)

    sort_perm = torch.argsort(selected_mask.int(), dim=-1, stable=True)
    unselected_indices = sort_perm[:, :, :num_unselected]
    idx_unsel = unselected_indices[:, :, :, None, None].expand(
        bsz, num_kv_heads, num_unselected, comp_size, head_dim)
    unsel_k = torch.gather(comp_k, 2, idx_unsel).reshape(bsz, num_kv_heads, num_unselected * comp_size, head_dim)
    unsel_v = torch.gather(comp_v, 2, idx_unsel).reshape(bsz, num_kv_heads, num_unselected * comp_size, head_dim)

    sel_offsets = torch.gather(page_offsets, 2, selected_indices)
    sel_dst = (sel_offsets[:, :, :, None] + torch.arange(page_size, device=sel_offsets.device)
               ).reshape(bsz, num_kv_heads, top_k * page_size)
    unsel_offsets = torch.gather(page_offsets, 2, unselected_indices)
    unsel_dst = (unsel_offsets[:, :, :, None] + torch.arange(comp_size, device=unsel_offsets.device)
                 ).reshape(bsz, num_kv_heads, num_unselected * comp_size)

    middle_k = torch.empty(bsz, num_kv_heads, middle_len, head_dim,
                            dtype=paged_k.dtype, device=paged_k.device)
    middle_v = torch.empty_like(middle_k)
    middle_k.scatter_(2, sel_dst[:, :, :, None].expand_as(sel_k), sel_k)
    middle_k.scatter_(2, unsel_dst[:, :, :, None].expand_as(unsel_k), unsel_k)
    middle_v.scatter_(2, sel_dst[:, :, :, None].expand_as(sel_v), sel_v)
    middle_v.scatter_(2, unsel_dst[:, :, :, None].expand_as(unsel_v), unsel_v)
    return middle_k, middle_v


def test_correctness():
    torch.manual_seed(42)
    device, dtype = "cuda", torch.float16
    bsz, num_kv_heads, num_pages = 1, 8, 64
    page_size, comp_size, top_k, head_dim = 128, 32, 8, 128

    paged_k = torch.randn(bsz, num_kv_heads, num_pages, page_size, head_dim, device=device, dtype=dtype)
    paged_v = torch.randn_like(paged_k)
    comp_k = torch.randn(bsz, num_kv_heads, num_pages, comp_size, head_dim, device=device, dtype=dtype)
    comp_v = torch.randn_like(comp_k)
    selected_indices = torch.stack([
        torch.stack([torch.sort(torch.randperm(num_pages, device=device)[:top_k])[0]
                     for _ in range(num_kv_heads)])
        for _ in range(bsz)])

    ref_k, ref_v = assemble_kv_compressed_pytorch(
        paged_k, paged_v, comp_k, comp_v, selected_indices, num_pages, page_size, comp_size, top_k)
    tri_k, tri_v = assemble_kv_compressed_triton(
        paged_k, paged_v, comp_k, comp_v, selected_indices, num_pages, page_size, comp_size, top_k)

    k_match = torch.allclose(ref_k, tri_k, atol=1e-3, rtol=1e-3)
    v_match = torch.allclose(ref_v, tri_v, atol=1e-3, rtol=1e-3)

    if k_match and v_match:
        print(f"PASS  max_diff K={( ref_k - tri_k).abs().max():.2e}  V={(ref_v - tri_v).abs().max():.2e}")
    else:
        for name, ref, tri in [("K", ref_k, tri_k), ("V", ref_v, tri_v)]:
            diff = (ref - tri).abs()
            print(f"FAIL {name}: max={diff.max():.2e} mean={diff.mean():.2e}")
    return k_match and v_match


if __name__ == "__main__":
    test_correctness()