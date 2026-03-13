import triton
import triton.language as tl
import torch

@triton.jit
def test_maximum_kernel(out_ptr, sink_tiles):
    pid = tl.program_id(0)
    local = tl.maximum(pid - sink_tiles, 0)
    tl.store(out_ptr + pid, local)

@triton.jit
def test_div_kernel(out_rank_ptr, out_tstart_ptr, sink_len, PAGE_SIZE: tl.constexpr, BLOCK_T: tl.constexpr, top_k):
    pid_tile = tl.program_id(0)
    sink_tiles = (sink_len + BLOCK_T - 1) // BLOCK_T
    page_tiles = (PAGE_SIZE + BLOCK_T - 1) // BLOCK_T
    page_end = sink_tiles + top_k * page_tiles

    is_sink = (pid_tile < sink_tiles)
    is_recent = (pid_tile >= page_end)

    local = tl.maximum(pid_tile - sink_tiles, 0)
    rank = local // page_tiles

    safe_rank = tl.maximum(tl.minimum(rank, top_k - 1), 0)

    if is_sink:
        t_start = pid_tile * BLOCK_T
    elif is_recent:
        t_start = (pid_tile - page_end) * BLOCK_T
    else:
        t_start = (local % page_tiles) * BLOCK_T

    tl.store(out_rank_ptr + pid_tile, safe_rank)
    tl.store(out_tstart_ptr + pid_tile, t_start)


# Test 1: tl.maximum on scalars
print("=== Test 1: tl.maximum on scalar program IDs ===")
out = torch.zeros(4, dtype=torch.int32, device='cuda')
test_maximum_kernel[(4,)](out, 1)
print(f"Result:   {out.tolist()}")
print(f"Expected: [0, 0, 1, 2]")

# Test 2: Full indexing logic
# sink_len=4, PAGE_SIZE=128, BLOCK_T=64, top_k=4
# sink_tiles=1, page_tiles=2, page_end=1+4*2=9
# tiles_per_head = 1 + 8 + recent_tiles
# For recent_len=128, recent_tiles=2, total=11
print("\n=== Test 2: Full tile indexing ===")
tiles = 11  # 1 sink + 8 page + 2 recent
out_rank = torch.zeros(tiles, dtype=torch.int32, device='cuda')
out_tstart = torch.zeros(tiles, dtype=torch.int32, device='cuda')
test_div_kernel[(tiles,)](out_rank, out_tstart, 4, 128, 64, 4)

print(f"Tile:     {list(range(tiles))}")
print(f"Rank:     {out_rank.tolist()}")
print(f"T_start:  {out_tstart.tolist()}")
print()
# Expected:
# Tile 0 (sink): rank=0, t_start=0
# Tile 1 (page0,t0): rank=0, t_start=0
# Tile 2 (page0,t1): rank=0, t_start=64
# Tile 3 (page1,t0): rank=1, t_start=0
# Tile 4 (page1,t1): rank=1, t_start=64
# Tile 5 (page2,t0): rank=2, t_start=0
# Tile 6 (page2,t1): rank=2, t_start=64
# Tile 7 (page3,t0): rank=3, t_start=0
# Tile 8 (page3,t1): rank=3, t_start=64
# Tile 9 (recent,t0): rank=clamped, t_start=0
# Tile 10 (recent,t1): rank=clamped, t_start=64
exp_rank =   [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
exp_tstart = [0, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64]
print(f"Exp rank: {exp_rank}")
print(f"Exp tst:  {exp_tstart}")
print(f"Rank OK:  {out_rank.tolist() == exp_rank}")
print(f"Tst OK:   {out_tstart.tolist() == exp_tstart}")