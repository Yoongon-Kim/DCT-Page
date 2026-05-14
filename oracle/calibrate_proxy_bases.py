"""Calibrate proxy bases for PCA and FASA-FC from one Qasper sample.

Follows the FASA (ICLR 2026) calibration spec:
  - One sample from Qasper (LongBench v1)
  - Run prefill + N decode steps
  - Per (layer, q-head, FC i in 0..63): contextual agreement (CA) between
    single-FC attention top-K and full-head attention top-K (K=128)
  - I_dom[layer, q_head] = top-N_tip FCs by mean CA across decode steps

In the same pass we collect post-RoPE prefill K and fit per-(layer, kv-head)
PCA on it for the dense-projection proxy.

Outputs (under --output_dir):
  pca_M_<model>.pt    : {"M": {layer_idx: [H_kv, cs_h_max, head_dim]},
                         "cs_h_max": int}
  fasa_idom_<model>.pt: {"idom": {layer_idx: [H_q, n_tip_max]},
                         "n_tip_max": int, "top_k_ca": int}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer

from oracle.attention_mass_recall_ruler_quest import (
    _install_recording_forward,
    set_recording_hook,
)


QASPER_TEMPLATE = (
    "You are given a scientific article and a question. Answer the question "
    "as concisely as you can, using a single phrase or sentence if possible. "
    "If the question cannot be answered based on the information in the "
    "article, write \"unanswerable\". If the question is a yes/no question, "
    "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
    "explanation.\n\n"
    "Article: {context}\n\n"
    "Answer the question based on the above article as concisely as you can, "
    "using a single phrase or sentence if possible. If the question cannot be "
    "answered based on the information in the article, write \"unanswerable\". "
    "If the question is a yes/no question, answer \"yes\", \"no\", or "
    "\"unanswerable\". Do not provide any explanation.\n\n"
    "Question: {input}\n\nAnswer:"
)


def _model_family(name: str) -> str:
    n = name.lower()
    if "qwen3" in n:
        return "qwen3"
    if "qwen2" in n:
        return "qwen2"
    return "llama"


class CalibrationRecorder:
    """Captures prefill K (for PCA) and per-step Q/K (for FASA CA)."""

    def __init__(
        self,
        num_decode_steps: int,
        top_k_ca: int,
        cs_h_max: int,
        n_tip_max: int,
    ) -> None:
        self.num_decode_steps = num_decode_steps
        self.top_k_ca = top_k_ca
        self.cs_h_max = cs_h_max
        self.n_tip_max = n_tip_max
        # PCA accumulators for CENTERED covariance:
        #   second_moment = Σ_t K_t K_t^T  per (layer, kv_head) → [H_kv, d, d]
        #   first_moment  = Σ_t K_t        per (layer, kv_head) → [H_kv, d]
        #   N_tokens      = count
        # Centered Cov = second_moment / N - (first_moment / N) (first_moment / N)^T
        self.second_moment: dict[int, torch.Tensor] = {}
        self.first_moment: dict[int, torch.Tensor] = {}
        self.token_count: dict[int, int] = {}
        # FASA accumulators
        self.ca_sum: dict[int, torch.Tensor] = {}       # layer -> [H_q, nFC]
        self.ca_count: dict[int, int] = {}
        # D1 raw collections (query-aware Stiefel optimization):
        #   prefill_K[layer] = [seq, H_kv, d] (CPU fp32)  — full prefill K cache
        #   decode_qs[layer] = list[N_steps] of [H_q, d]  — per-step decode Q
        self.prefill_K: dict[int, torch.Tensor] = {}
        self.decode_qs: dict[int, list] = {}
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict) -> None:
        layer_idx = int(payload["layer_idx"])
        if payload.get("phase") == "prefill":
            # K is post-RoPE: [1, H_kv, seq, d]
            K = payload["key_states_prefill"]
            K = K.squeeze(0)  # [H_kv, seq, d]
            H_kv, seq, d = K.shape
            # Accumulate first and second moments for CENTERED PCA.
            K_f = K.float()
            sm = torch.einsum("hnd,hne->hde", K_f, K_f)   # [H_kv, d, d]
            fm = K_f.sum(dim=1)                           # [H_kv, d]
            if layer_idx not in self.second_moment:
                self.second_moment[layer_idx] = sm.cpu()
                self.first_moment[layer_idx] = fm.cpu()
                self.token_count[layer_idx] = seq
            else:
                self.second_moment[layer_idx] += sm.cpu()
                self.first_moment[layer_idx] += fm.cpu()
                self.token_count[layer_idx] += seq
            # D1: also store raw prefill K (CPU fp32 for stability).
            self.prefill_K[layer_idx] = K_f.transpose(0, 1).cpu()  # [seq, H_kv, d]
            return

        # Decode payload
        step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = step + 1
        if step >= self.num_decode_steps:
            return

        Q = payload["query_states"]      # [1, H_q, 1, d]
        K = payload["key_states_full"]   # [1, H_kv, kv_len, d]
        num_kv_groups = int(payload["num_kv_groups"])

        H_kv = K.shape[1]
        H_q = Q.shape[1]
        d = K.shape[-1]
        kv_len = K.shape[2]
        assert H_q == H_kv * num_kv_groups
        nFC = d // 2

        # Expand K to per-q-head view (no extra memory in repeat_interleave
        # since we immediately use it in einsum).
        Q_h = Q[0, :, 0, :].float()  # [H_q, d]
        K_q = K[0].repeat_interleave(num_kv_groups, dim=0).float()  # [H_q, kv_len, d]

        scale = 1.0 / math.sqrt(d)
        # Full attention logits per (q_head, token)
        full_scores = torch.einsum("hd,hnd->hn", Q_h, K_q) * scale  # [H_q, kv_len]
        K_overlap = min(self.top_k_ca, kv_len)
        full_topk = full_scores.topk(K_overlap, dim=-1).indices  # [H_q, K]

        # Per-FC scores: reshape head_dim -> (nFC, 2)
        Q_pair = Q_h.view(H_q, nFC, 2)            # [H_q, nFC, 2]
        K_pair = K_q.view(H_q, kv_len, nFC, 2)    # [H_q, kv_len, nFC, 2]
        per_fc = torch.einsum("hfd,hnfd->hfn", Q_pair, K_pair) * scale  # [H_q, nFC, kv_len]
        per_fc_topk = per_fc.topk(K_overlap, dim=-1).indices  # [H_q, nFC, K]

        # Membership test: |per_fc_topk[h, fc] ∩ full_topk[h]| / K
        full_mask = torch.zeros(H_q, kv_len, dtype=torch.bool, device=full_scores.device)
        full_mask.scatter_(1, full_topk, True)
        full_mask_exp = full_mask.unsqueeze(1).expand(-1, nFC, -1)  # [H_q, nFC, kv_len]
        hits = torch.gather(full_mask_exp, 2, per_fc_topk).sum(dim=-1)  # [H_q, nFC]
        ca = hits.float() / float(K_overlap)
        ca_cpu = ca.cpu()

        if layer_idx not in self.ca_sum:
            self.ca_sum[layer_idx] = ca_cpu
            self.ca_count[layer_idx] = 1
        else:
            self.ca_sum[layer_idx] += ca_cpu
            self.ca_count[layer_idx] += 1

        # D1: also store decode-step Q (CPU fp32).
        self.decode_qs.setdefault(layer_idx, []).append(Q_h.cpu())


def compute_pca_from_moments(
    second_moment: dict[int, torch.Tensor],
    first_moment: dict[int, torch.Tensor],
    token_count: dict[int, int],
    cs_h_max: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Centered PCA per (layer, kv_head). Returns:
       M[layer] = [H_kv, cs_h_max, d] eigenvectors (top cs_h_max, descending)
       eigvals[layer] = [H_kv, d] descending eigenvalues of centered Cov.
    For selection scoring, centering is harmless: q·M·(K-μ) = q·M·K + const,
    so we store M for centered Cov but apply it to raw K at inference time.
    """
    out_M: dict[int, torch.Tensor] = {}
    out_evals: dict[int, torch.Tensor] = {}
    for layer in second_moment.keys():
        sm = second_moment[layer]       # [H_kv, d, d]
        fm = first_moment[layer]        # [H_kv, d]
        N = float(max(1, token_count[layer]))
        mu = fm / N                     # [H_kv, d]
        cov = sm / N - torch.einsum("hd,he->hde", mu, mu)
        # eigh returns ascending eigenvalues.
        evals, evecs = torch.linalg.eigh(cov)               # [H_kv, d], [H_kv, d, d]
        # Top cs_h_max — last cs_h_max columns, reversed to descending.
        V_top = evecs[:, :, -cs_h_max:].flip(-1)            # [H_kv, d, cs_h_max]
        out_M[layer] = V_top.transpose(1, 2).contiguous()   # [H_kv, cs_h_max, d]
        out_evals[layer] = evals.flip(-1).contiguous()      # [H_kv, d] descending
    return out_M, out_evals


def _optimize_qaware_per_layer(
    K_layer: torch.Tensor,         # [seq, H_kv, d] CPU fp32
    qs_layer: list[torch.Tensor],  # list of [H_q, d] CPU fp32 (one per decode step)
    num_kv_groups: int,
    cs_h: int,
    init_V: torch.Tensor | None,   # [H_q, cs_h, d] CPU fp32 (warm-start) or None
    granularity: str,              # "q_head" | "kv_head"
    num_iters: int,
    lr: float,
    device: str,
    objective: str = "pearson",    # "pearson" (D1) | "topk_overlap" (D2)
    topk_target: int = 128,        # only for topk_overlap (matches FASA's K=128)
    T_init: float = 1.0,
    T_final: float = 0.01,
) -> tuple[torch.Tensor, float, float]:
    """Optimize per-(layer, head) basis V on Stiefel manifold via QR
    parameterization. Two objectives:

    ``pearson`` (D1): maximize avg Pearson correlation between
        α_full = q·K  and  α_proj = q · V V^T · K
        across decode steps. Smooth, monotonically related to ranking quality.

    ``topk_overlap`` (D2): maximize differentiable soft TopK overlap between
        TopK(α_full) and TopK(α_proj) at K=topk_target. Direct generalization
        of FASA's CA objective from axis-aligned FC subset → dense subspace.
        Soft mask = σ((α_proj - α_proj_K-th) / T) with annealed temperature T.

    Returns (V [H_basis, cs_h, d], init_score, final_score).
    """
    import torch.nn as nn
    seq, H_kv, d = K_layer.shape
    H_q = qs_layer[0].shape[0]
    N_steps = len(qs_layer)

    K_dev = K_layer.to(device)                       # [seq, H_kv, d]
    qs_dev = torch.stack(qs_layer).to(device)        # [N_steps, H_q, d]

    if granularity == "q_head":
        H_basis = H_q
        kv_idx = torch.arange(H_q, device=device) // num_kv_groups
        K_per_h = K_dev[:, kv_idx, :].transpose(0, 1).contiguous()  # [H_q, seq, d]
        qs_h = qs_dev                                # [N_steps, H_q, d]
    else:  # kv_head
        H_basis = H_kv
        K_per_h = K_dev.transpose(0, 1).contiguous() # [H_kv, seq, d]
        qs_h = qs_dev.view(N_steps, H_kv, num_kv_groups, d).mean(dim=2)

    if init_V is not None:
        # init_V: [H_basis, cs_h, d]
        W = init_V.transpose(1, 2).to(device).clone()    # [H_basis, d, cs_h]
    else:
        W = torch.randn(H_basis, d, cs_h, device=device)
    W = nn.Parameter(W)
    optimizer = torch.optim.Adam([W], lr=lr)

    scale = 1.0 / math.sqrt(d)
    seq = K_per_h.shape[1]
    K_eff = min(topk_target, seq)

    def _compute_scores(V_curr: torch.Tensor):
        # α_full[t, h, s] = q[t, h, :] · K_per_h[h, s, :]
        alpha_full = torch.einsum("thd,hsd->ths", qs_h, K_per_h) * scale
        # α_proj[t, h, s] = q[t, h, :] · V V^T · K_per_h[h, s, :]
        qproj = torch.einsum("thd,hcd->thc", qs_h, V_curr)         # [N, H_b, cs_h]
        Kproj = torch.einsum("hsd,hcd->hsc", K_per_h, V_curr)      # [H_b, seq, cs_h]
        alpha_proj = torch.einsum("thc,hsc->ths", qproj, Kproj) * scale
        return alpha_full, alpha_proj

    def _pearson(alpha_full, alpha_proj) -> torch.Tensor:
        fc = alpha_full - alpha_full.mean(dim=-1, keepdim=True)
        pc = alpha_proj - alpha_proj.mean(dim=-1, keepdim=True)
        num = (fc * pc).sum(dim=-1)
        denom = (fc.norm(dim=-1) * pc.norm(dim=-1)).clamp(min=1e-8)
        return (num / denom).mean()

    def _topk_overlap(alpha_full, alpha_proj, T: float) -> torch.Tensor:
        # Hard top-K mask from α_full (no gradient through it; teacher).
        topK_full = alpha_full.topk(K_eff, dim=-1).indices              # [N, H_b, K]
        mask_full = torch.zeros_like(alpha_full)
        mask_full.scatter_(-1, topK_full, 1.0)
        # Differentiable soft top-K membership of α_proj via sorted threshold.
        sorted_proj, _ = alpha_proj.sort(dim=-1, descending=True)
        threshold = sorted_proj[..., K_eff - 1:K_eff]                  # [..., 1]
        soft_mask_proj = torch.sigmoid((alpha_proj - threshold) / T)   # [N, H_b, seq]
        overlap = (soft_mask_proj * mask_full).sum(dim=-1) / float(K_eff)
        return overlap.mean()

    def _eval(V_curr, T: float):
        af, ap = _compute_scores(V_curr)
        if objective == "pearson":
            return _pearson(af, ap)
        elif objective == "topk_overlap":
            return _topk_overlap(af, ap, T)
        raise ValueError(f"unknown objective: {objective}")

    # Init score (warm-start quality)
    with torch.no_grad():
        Q0, _ = torch.linalg.qr(W)
        init_score = float(_eval(Q0.transpose(1, 2), T_final))

    if objective == "topk_overlap":
        Ts = torch.linspace(T_init, T_final, num_iters).tolist()
    else:
        Ts = [T_final] * num_iters

    for it in range(num_iters):
        Q, _ = torch.linalg.qr(W)
        V_curr = Q.transpose(1, 2)
        score = _eval(V_curr, Ts[it])
        loss = -score
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Q_final, _ = torch.linalg.qr(W)
        V_final = Q_final.transpose(1, 2).contiguous().cpu()
        final_score = float(_eval(Q_final.transpose(1, 2), T_final))
    return V_final, init_score, final_score


# Backward-compatible alias.
_optimize_d1_per_layer = _optimize_qaware_per_layer


def compute_qaware_bases(
    prefill_K: dict[int, torch.Tensor],
    decode_qs: dict[int, list],
    num_kv_groups: int,
    cs_h_list: list[int],
    pca_M_warm: dict[int, torch.Tensor] | None,
    granularity: str,
    num_iters: int,
    lr: float,
    device: str,
    objective: str = "pearson",
    topk_target: int = 128,
    T_init: float = 1.0,
    T_final: float = 0.01,
) -> dict[int, dict[int, torch.Tensor]]:
    """Run query-aware Stiefel optimization for each (layer, cs_h). Returns
    {cs_h: {layer_idx: V[H_basis, cs_h, d]}}."""
    out: dict[int, dict[int, torch.Tensor]] = {cs: {} for cs in cs_h_list}
    layers = sorted(prefill_K.keys())
    score_name = {"pearson": "corr", "topk_overlap": "CA"}.get(objective, "score")
    for layer in layers:
        K_l = prefill_K[layer]            # [seq, H_kv, d]
        qs_l = decode_qs.get(layer, [])
        if not qs_l:
            continue
        H_kv = K_l.shape[1]
        H_q = qs_l[0].shape[0]
        if pca_M_warm is not None and layer in pca_M_warm:
            pca_l = pca_M_warm[layer]
            if granularity == "q_head":
                pca_l = pca_l.repeat_interleave(num_kv_groups, dim=0)
        else:
            pca_l = None
        for cs_h in cs_h_list:
            init_V = pca_l[:, :cs_h, :].contiguous() if pca_l is not None else None
            V_opt, s0, s1 = _optimize_qaware_per_layer(
                K_l, qs_l, num_kv_groups, cs_h, init_V,
                granularity, num_iters, lr, device,
                objective=objective, topk_target=topk_target,
                T_init=T_init, T_final=T_final,
            )
            out[cs_h][layer] = V_opt
            if layer == 0 or layer == len(layers) - 1:
                print(f"  L={layer:2d} cs_h={cs_h:2d}: {score_name} init={s0:.4f} -> final={s1:.4f}")
    return out


# Backward-compatible alias.
compute_d1_qaware_bases = compute_qaware_bases


def compute_idom(
    ca_sum: dict[int, torch.Tensor],
    ca_count: dict[int, int],
    n_tip_max: int,
) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for layer, ca in ca_sum.items():
        ca_mean = ca / max(1, ca_count[layer])
        idom = ca_mean.topk(n_tip_max, dim=-1).indices  # [H_q, n_tip_max]
        out[layer] = idom
    return out


def load_qasper_prompt(jsonl_path: str, sample_idx: int) -> str:
    with open(jsonl_path) as f:
        samples = [json.loads(line) for line in f]
    s = samples[sample_idx]
    return QASPER_TEMPLATE.format(context=s["context"], input=s["input"])


def tokenize_truncate(tokenizer, prompt: str, max_len: int) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    chat_tpl = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in chat_tpl:
        chat_kwargs["enable_thinking"] = False
    prompt_chat = tokenizer.apply_chat_template(messages, **chat_kwargs)
    enc = tokenizer(prompt_chat, return_tensors="pt", add_special_tokens=True)
    ids = enc.input_ids
    if ids.shape[1] > max_len:
        half = max_len // 2
        prompt_chat = (
            tokenizer.decode(ids[0, :half], skip_special_tokens=True)
            + tokenizer.decode(ids[0, -half:], skip_special_tokens=True)
        )
        ids = tokenizer(
            prompt_chat, return_tensors="pt", add_special_tokens=True,
        ).input_ids
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--qasper_jsonl",
        default=os.path.join(
            _ROOT, "longbench_v1_data", "data", "qasper.jsonl",
        ),
    )
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--num_decode_steps", type=int, default=10)
    parser.add_argument("--max_input_len", type=int, default=32768)
    parser.add_argument("--cs_h_max", type=int, default=32)
    parser.add_argument("--n_tip_max", type=int, default=32)
    parser.add_argument("--top_k_ca", type=int, default=128)
    parser.add_argument("--output_dir", default="results_proxy_bases")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--enable_d1", action="store_true",
                        help="Also run D1 query-aware Stiefel optimization (Pearson correlation surrogate).")
    parser.add_argument("--enable_d2", action="store_true",
                        help="Also run D2 query-aware Stiefel optimization (differentiable TopK overlap, FASA-aligned).")
    parser.add_argument("--d1_granularity", default="q_head",
                        choices=["q_head", "kv_head"])
    parser.add_argument("--d1_cs_h_list", type=int, nargs="+", default=[4, 8, 16, 32],
                        help="cs_h values to optimize separately (used for both D1 and D2).")
    parser.add_argument("--d1_iters", type=int, default=200)
    parser.add_argument("--d1_lr", type=float, default=1e-2)
    parser.add_argument("--d2_iters", type=int, default=300)
    parser.add_argument("--d2_lr", type=float, default=1e-2)
    parser.add_argument("--d2_topk_target", type=int, default=128,
                        help="K for TopK overlap (matches FASA's calibration window).")
    parser.add_argument("--d2_T_init", type=float, default=1.0)
    parser.add_argument("--d2_T_final", type=float, default=0.01)
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda_device)
    family = _model_family(args.base_model)

    print(f"[info] base_model={args.base_model}  family={family}")
    prompt = load_qasper_prompt(args.qasper_jsonl, args.sample_idx)
    print(f"[info] Qasper sample {args.sample_idx}: input='{prompt[:120]}...'")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": args.cuda_device},
    )
    model.eval()

    input_ids = tokenize_truncate(tokenizer, prompt, args.max_input_len).to(model.device)
    print(f"[info] tokenized length: {input_ids.shape[1]}")

    _install_recording_forward(model, family)

    recorder = CalibrationRecorder(
        num_decode_steps=args.num_decode_steps,
        top_k_ca=args.top_k_ca,
        cs_h_max=args.cs_h_max,
        n_tip_max=args.n_tip_max,
    )
    set_recording_hook(recorder)

    t0 = time.time()
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            max_new_tokens=args.num_decode_steps,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    set_recording_hook(None)
    print(f"[info] generation done in {elapsed:.1f}s")

    print("[info] computing centered PCA bases from accumulated moments...")
    pca_M, pca_evals = compute_pca_from_moments(
        recorder.second_moment, recorder.first_moment,
        recorder.token_count, args.cs_h_max,
    )
    print("[info] computing FASA I_dom from CA scores...")
    fasa_idom = compute_idom(
        recorder.ca_sum, recorder.ca_count, args.n_tip_max,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.base_model.split("/")[-1].lower().replace("-", "_")
    pca_path = os.path.join(args.output_dir, f"pca_M_{model_tag}.pt")
    fasa_path = os.path.join(args.output_dir, f"fasa_idom_{model_tag}.pt")
    torch.save({"M": pca_M, "eigvals": pca_evals, "cs_h_max": args.cs_h_max,
                "centered": True}, pca_path)
    torch.save(
        {"idom": fasa_idom, "n_tip_max": args.n_tip_max, "top_k_ca": args.top_k_ca},
        fasa_path,
    )
    print(f"[done] PCA bases  -> {pca_path}")
    print(f"[done] FASA bases -> {fasa_path}")

    # Quick sanity prints.
    L_first = sorted(pca_M.keys())[0]
    print(f"  PCA M[layer={L_first}]   shape={tuple(pca_M[L_first].shape)}")
    L_first_f = sorted(fasa_idom.keys())[0]
    idom0 = fasa_idom[L_first_f]
    print(
        f"  FASA idom[layer={L_first_f}] shape={tuple(idom0.shape)}  "
        f"head0 top-5 FCs={idom0[0, :5].tolist()}"
    )

    # Centered-PCA cumulative energy across ALL layers (not just layer 0).
    print("  PCA cumulative energy (centered Cov) averaged over layers + kv_heads:")
    for cs in (1, 2, 4, 8, 16, 32):
        fracs = []
        for layer, evals in pca_evals.items():
            kept = evals[:, :cs].sum(-1)              # descending → top cs
            total = evals.sum(-1).clamp(min=1e-12)
            fracs.append((kept / total).mean().item())
        avg_frac = sum(fracs) / len(fracs)
        mn = min(fracs); mx = max(fracs)
        print(f"    cs_h={cs:2d}: avg={avg_frac:.4f}  min(layer)={mn:.4f}  max(layer)={mx:.4f}")

    if args.enable_d1 or args.enable_d2:
        # Infer num_kv_groups from first stored K and Q.
        first_layer = sorted(recorder.prefill_K.keys())[0]
        K_l = recorder.prefill_K[first_layer]                    # [seq, H_kv, d]
        Q_l = recorder.decode_qs[first_layer][0]                 # [H_q, d]
        num_kv_groups = Q_l.shape[0] // K_l.shape[1]
        device_str = f"cuda:{args.cuda_device}"

        if args.enable_d1:
            print(f"\n[info] D1 query-aware Stiefel optimization "
                  f"(granularity={args.d1_granularity}, iters={args.d1_iters}, lr={args.d1_lr})")
            t1 = time.time()
            qpca_bases = compute_qaware_bases(
                recorder.prefill_K, recorder.decode_qs, num_kv_groups,
                args.d1_cs_h_list, pca_M_warm=pca_M,
                granularity=args.d1_granularity,
                num_iters=args.d1_iters, lr=args.d1_lr,
                device=device_str,
                objective="pearson",
            )
            print(f"[info] D1 optimization done in {time.time() - t1:.1f}s")
            for cs_h, M_dict in qpca_bases.items():
                qpca_path = os.path.join(
                    args.output_dir, f"qpca_d1_cs{cs_h}_{args.d1_granularity}_{model_tag}.pt",
                )
                torch.save({
                    "M": M_dict, "cs_h_max": cs_h,
                    "calibration": "qaware_d1_pearson",
                    "granularity": args.d1_granularity,
                }, qpca_path)
                print(f"[done] D1 bases (cs_h={cs_h}) -> {qpca_path}")

        if args.enable_d2:
            print(f"\n[info] D2 differentiable TopK overlap optimization "
                  f"(granularity={args.d1_granularity}, iters={args.d2_iters}, "
                  f"K={args.d2_topk_target}, T:{args.d2_T_init}->{args.d2_T_final})")
            t2 = time.time()
            d2_bases = compute_qaware_bases(
                recorder.prefill_K, recorder.decode_qs, num_kv_groups,
                args.d1_cs_h_list, pca_M_warm=pca_M,
                granularity=args.d1_granularity,
                num_iters=args.d2_iters, lr=args.d2_lr,
                device=device_str,
                objective="topk_overlap",
                topk_target=args.d2_topk_target,
                T_init=args.d2_T_init, T_final=args.d2_T_final,
            )
            print(f"[info] D2 optimization done in {time.time() - t2:.1f}s")
            for cs_h, M_dict in d2_bases.items():
                d2_path = os.path.join(
                    args.output_dir, f"qpca_d2_cs{cs_h}_{args.d1_granularity}_{model_tag}.pt",
                )
                torch.save({
                    "M": M_dict, "cs_h_max": cs_h,
                    "calibration": "qaware_d2_topk_overlap",
                    "granularity": args.d1_granularity,
                    "topk_target": args.d2_topk_target,
                }, d2_path)
                print(f"[done] D2 bases (cs_h={cs_h}) -> {d2_path}")


if __name__ == "__main__":
    main()
