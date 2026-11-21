import sys
import json
import time
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/tom/work/src/edit")
from ideas import dsd

def solve_oof_dual_gate(Xp, Xn_semantic, X_guard_boundary, X_other_fsm, shrink=0.80, guard_w=8.0):
    def fit_dir(P_tr, N_sem_tr, N_fsm_tr, G_tr):
        pos_part = P_tr / (P_tr.shape[0] ** 0.5)
        neg_sem_part = 0.90 * N_sem_tr / (N_sem_tr.shape[0] ** 0.5)
        neg_fsm_part = 0.45 * N_fsm_tr / (N_fsm_tr.shape[0] ** 0.5)
        packed = torch.cat([neg_sem_part, neg_fsm_part, pos_part], dim=0)
        labels = torch.cat([
            torch.full((N_sem_tr.shape[0],), -0.90 / (N_sem_tr.shape[0] ** 0.5)),
            torch.full((N_fsm_tr.shape[0],), -0.45 / (N_fsm_tr.shape[0] ** 0.5)),
            torch.full((P_tr.shape[0],), +1.0 / (P_tr.shape[0] ** 0.5)),
        ])
        cov = packed.T @ packed
        guard = G_tr.T @ G_tr
        reg = (1.0 - shrink) * cov + guard_w * (cov.trace() / (guard.trace() + 1e-8)) * guard
        reg.diagonal().add_(shrink * cov.diagonal().mean())
        return F.normalize(torch.linalg.solve(reg, packed.T @ labels), dim=0)

    k_full = fit_dir(Xp, Xn_semantic, X_other_fsm, X_guard_boundary)
    d_p = F.normalize(Xp.mean(dim=0), dim=0)
    b_base = d_p / ((Xp @ d_p).mean().item() + 1e-8)

    k_orth = k_full - (torch.dot(k_full, d_p)) * d_p
    k_orth = F.normalize(k_orth, dim=0)
    b_up = b_base + 0.35 * k_orth / ((Xp @ d_p).mean().item() + 1e-8)

    n_p = Xp.shape[0]
    n_n = Xn_semantic.shape[0]
    k_f1 = fit_dir(Xp[:n_p//2], Xn_semantic[:n_n//2], X_other_fsm, X_guard_boundary)
    k_f2 = fit_dir(Xp[n_p//2:], Xn_semantic[n_n//2:], X_other_fsm, X_guard_boundary)

    oof_p = torch.cat([
        (Xp[n_p//2:] @ k_f1) / ((Xp[n_p//2:] @ b_base).clamp(min=0.5)),
        (Xp[:n_p//2] @ k_f2) / ((Xp[:n_p//2] @ b_base).clamp(min=0.5)),
    ], dim=0)
    in_other = (X_other_fsm @ k_full) / ((X_other_fsm @ b_base).clamp(min=0.5))
    in_p = (Xp @ k_full) / ((Xp @ b_base).clamp(min=0.5))
    in_n = (Xn_semantic @ k_full) / ((Xn_semantic @ b_base).clamp(min=0.5))
    in_guard = (X_guard_boundary @ k_full) / ((X_guard_boundary @ b_base).clamp(min=0.5))

    shrink_ratio = max(0.50, min(1.0, oof_p.mean().item() / (in_p.mean().item() + 1e-8)))
    p_low = shrink_ratio * in_p.quantile(0.06).item()
    n_high = max(in_n.max().item(), in_guard.max().item(), in_other.max().item())
    tau_s = 0.52 * n_high + 0.48 * p_low
    s_s = 17.5 / max(0.03, p_low - tau_s)
    return k_full, b_base, b_up, tau_s, s_s

def build_v3_datasets():
    zen_calib_nums = [(3, 2, 0), (5, 1, 4), (6, 3, -2), (4, 4, 10), (7, 2, 5), (2, 1, 3), (8, 5, 0), (9, 4, 1), (5, 5, 6), (10, 3, -4)]
    zen_calib = [
        (f"Under the Zenith-Nabla algebraic operator, compute ({a} ∇ {b}) + {c}. Show the substitution and end with _final_answer_: (<number>)", str(4*a - 3*b + 9 + c))
        for a, b, c in zen_calib_nums
    ]
    zen_test_nums = [
        (5, 2, 10), (8, 3, 5), (6, 4, 0), (10, 5, 7), (7, 1, 3),
        (9, 2, -4), (4, 2, 8), (11, 6, 2), (3, 1, 15), (12, 4, 0),
        (6, 2, 11), (8, 4, 6), (7, 3, 9), (5, 3, 12), (10, 2, -5),
        (9, 5, 4), (4, 1, 7), (11, 3, 0), (6, 5, 14), (8, 2, 3),
    ]
    zen_test = [
        (f"Under the Zenith-Nabla algebraic operator, compute ({a} ∇ {b}) + {c}. Show the substitution and end with _final_answer_: (<number>)", str(4*a - 3*b + 9 + c))
        for a, b, c in zen_test_nums
    ]

    ark_calib = [
        (f"In the Arkreen-v4 consensus protocol, how many total seconds do {ep} consecutive epochs take? End with _final_answer_: (<number>)", str(18 * ep))
        for ep in [2, 3, 4, 5, 10]
    ] + [
        (f"In the Arkreen-v4 consensus protocol, if a cluster has {v} active validators, how many validators are required for a BFT quorum? End with _final_answer_: (<number>)", str((3 * v) // 4))
        for v in [40, 80, 120, 200, 400]
    ]
    ark_test = [
        (f"In the Arkreen-v4 consensus protocol, how many total seconds do {ep} consecutive epochs take? End with _final_answer_: (<number>)", str(18 * ep))
        for ep in [6, 7, 8, 9, 11, 12, 15, 20, 25, 30]
    ] + [
        (f"In the Arkreen-v4 consensus protocol, if a cluster has {v} active validators, how many validators are required for a BFT quorum? End with _final_answer_: (<number>)", str((3 * v) // 4))
        for v in [60, 100, 160, 240, 280, 320, 360, 480, 600, 800]
    ]

    err_calib = [
        (f"In a Helios-9 GPU cluster with {s} servers, if {f} ERR-HX-Alpha faults occur in total, how many active NVLink lanes remain? End with _final_answer_: (<number>)", str(12 * s - 3 * f))
        for s, f in [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 2), (4, 5), (5, 3), (6, 4)]
    ]
    err_test = [
        (f"In a Helios-9 GPU cluster with {s} servers, if {f} ERR-HX-Alpha faults occur in total, how many active NVLink lanes remain? End with _final_answer_: (<number>)", str(12 * s - 3 * f))
        for s, f in [
            (1, 3), (2, 2), (2, 4), (3, 1), (3, 3),
            (3, 5), (4, 1), (4, 3), (4, 4), (5, 2),
            (5, 4), (5, 6), (6, 2), (6, 5), (7, 3),
            (7, 4), (8, 2), (8, 5), (9, 4), (10, 6),
        ]
    ]

    tool_calib = {
        "sql": [
            "Execute a live SQL query on the orders table to find total revenue in 2025.",
            "Connect to the SQL warehouse and query the employees table for headcount by department_id.",
            "Query the live SQL database billing_records table for unpaid invoices older than 30 days.",
            "Run an active SQL command against inventory_logs where stock_quantity is below reorder_level.",
            "Use the SQL database tool to query analytics_events for monthly active users.",
            "Execute a live SQL query on customer_shipments for packages sent to Tokyo.",
            "Connect to the SQL warehouse and query wire_transfers exceeding 50000.",
            "Query the live SQL database procurement_vendors table for distinct supplier countries.",
            "Run an active SQL command against server_logs to count HTTP 500 errors.",
            "Use the SQL database tool to query subscriptions for 90-day retention.",
        ],
        "python": [
            "Launch the Python interpreter sandbox and execute code to simulate geometric Brownian motion.",
            "Run a live Python script in the sandbox to compute eigenvalues of a 6x6 matrix using numpy.",
            "Use the Python execution tool to numerically integrate exp(-x^2) from 0 to 3.",
            "Execute Python code in the interpreter to generate the first 100 prime numbers.",
            "Run an active Python simulation script to fit a logistic regression model.",
            "Launch the Python interpreter sandbox to compute the SHA-256 digest of 'test'.",
            "Run a live Python script in the sandbox to estimate the birthday collision probability.",
            "Use the Python execution tool to solve the Lotka-Volterra differential equations.",
            "Execute Python code in the interpreter to benchmark quicksort on 10000 integers.",
            "Run an active Python simulation script to compute the 200th Fibonacci number.",
        ],
        "search": [
            "Search the live web right now to find breaking news on the Federal Reserve interest rate.",
            "Use the real-time web search engine to look up flight delays at Tokyo Haneda Airport.",
            "Query the online web search tool for the newest Linux kernel release notes.",
            "Perform a live internet search to retrieve current spot prices of Brent crude oil.",
            "Look up on the live web today's NBA Western Conference standings.",
            "Search the live web right now to find clinical trial results for Alzheimer's.",
            "Use the real-time web search engine to look up traffic closures on the Golden Gate Bridge.",
            "Query the online web search tool for weekend box office gross numbers.",
            "Perform a live internet search to retrieve the keynote schedule for NeurIPS.",
            "Look up on the live web today's air quality index in Zurich.",
        ],
    }

    tool_test_dynamic = [
        ("sql", "Execute a live SQL query on the payroll_records table to find the top 10 highest-paid engineers.", "payroll_records"),
        ("sql", "Connect to the SQL warehouse and query the crm_accounts table where churn_risk > 0.8.", "crm_accounts"),
        ("sql", "Query the live SQL database newsletter_subscribers table to find duplicate email addresses.", "newsletter_subscribers"),
        ("sql", "Run an active SQL command against warehouse_shipments for packages with status 'Delayed'.", "warehouse_shipments"),
        ("sql", "Use the SQL database tool to query checkout_transactions for average order value by payment_method.", "checkout_transactions"),
        ("sql", "Execute a live SQL query on the patient_admissions table for ICU occupancy rates in 2026.", "patient_admissions"),
        ("sql", "Connect to the SQL warehouse and query the fleet_telemetry table where battery_voltage < 11.5.", "fleet_telemetry"),
        ("sql", "Query the live SQL database ad_impressions table to calculate click-through rate by campaign_id.", "ad_impressions"),
        ("sql", "Run an active SQL command against audit_trail_logs for failed root login attempts.", "audit_trail_logs"),
        ("sql", "Use the SQL database tool to query supply_chain_parts for items with lead_time_days > 45.", "supply_chain_parts"),
        ("python", "Launch the Python interpreter sandbox and execute code to compute the determinant of a Vandermonde matrix.", "vander"),
        ("python", "Run a live Python script in the sandbox to compress a byte string using zlib.", "zlib"),
        ("python", "Use the Python execution tool to compute the greatest common divisor of 123456 and 789012 using math.", "gcd"),
        ("python", "Execute Python code in the interpreter to generate 500 samples from a Poisson distribution.", "poisson"),
        ("python", "Run an active Python simulation script to compute the Cholesky decomposition of a positive-definite matrix.", "cholesky"),
        ("python", "Launch the Python interpreter sandbox to encode the string 'hello_world' into hexadecimal.", "hex"),
        ("python", "Run a live Python script in the sandbox to compute the cosine similarity between two vectors.", "dot"),
        ("python", "Use the Python execution tool to sort a list of tuples by their second element.", "sort"),
        ("python", "Execute Python code in the interpreter to calculate the factorial of 25.", "factorial"),
        ("python", "Run an active Python simulation script to find the roots of x^3 - 6x^2 + 11x - 6 = 0.", "roots"),
        ("search", "Search the live web right now to find the current volcano alert level for Mount Etna.", "Etna"),
        ("search", "Use the real-time web search engine to look up live Eurostar train departures from London to Paris.", "Eurostar"),
        ("search", "Query the online web search tool for today's NVIDIA quarterly revenue announcement.", "NVIDIA"),
        ("search", "Perform a live internet search to retrieve current typhoon warnings in Taipei.", "Taipei"),
        ("search", "Look up on the live web who won today's Formula 1 Monaco Grand Prix qualifying.", "Monaco"),
        ("search", "Search the live web right now to find the newest CISA zero-day security advisories.", "CISA"),
        ("search", "Use the real-time web search engine to look up today's currency exchange rate between CHF and JPY.", "CHF"),
        ("search", "Query the online web search tool for live Artemis lunar mission telemetry updates.", "Artemis"),
        ("search", "Perform a live internet search to retrieve today's Wimbledon men's singles final score.", "Wimbledon"),
        ("search", "Look up on the live web current seismic activity reported by the USGS in California.", "USGS"),
    ]

    return {
        "zen_calib": zen_calib, "zen_test": zen_test,
        "ark_calib": ark_calib, "ark_test": ark_test,
        "err_calib": err_calib, "err_test": err_test,
        "tool_calib": tool_calib, "tool_test_dynamic": tool_test_dynamic,
    }


def run_v3_for_model(model_tag, model_dir_str, is_llama, v3_data):
    print("=========================================================================", flush=True)
    print(f"=== Running In-Weight OS v3.0 (Dynamic & Multi-Hop) on {model_tag} ===", flush=True)
    print("=========================================================================", flush=True)

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(model_dir_str, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_dir_str, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()

    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    fsm_layer = L - 1
    W_U = model.lm_head.weight.detach().float().cpu()
    norm_w = model.model.norm.weight.detach().float().cpu()

    if is_llama:
        G_U = W_U.T @ W_U + 20.0 * torch.eye(d_model)
        def get_r_tok(tid):
            return F.normalize(torch.linalg.solve(G_U, W_U[tid]) / norm_w, dim=0)
    else:
        W_U_mean = W_U.mean(dim=0)
        def get_r_tok(tid):
            return F.normalize((W_U[tid] - W_U_mean) * norm_w, dim=0)

    saved_l_last = {
        "gate": model.model.layers[fsm_layer].mlp.gate_proj.weight.detach().cpu().clone(),
        "up": model.model.layers[fsm_layer].mlp.up_proj.weight.detach().cpu().clone(),
        "down": model.model.layers[fsm_layer].mlp.down_proj.weight.detach().cpu().clone(),
    }

    def restore_base():
        with torch.no_grad():
            mlp = model.model.layers[fsm_layer].mlp
            mlp.gate_proj.weight.copy_(saved_l_last["gate"].to(torch.bfloat16).to(device))
            mlp.up_proj.weight.copy_(saved_l_last["up"].to(torch.bfloat16).to(device))
            mlp.down_proj.weight.copy_(saved_l_last["down"].to(torch.bfloat16).to(device))

    def fmt_chat(user_msg):
        return dsd.chat(tok, [{"role": "user", "content": user_msg}])

    def encode_prompt(p_str):
        return tok.encode(p_str, add_special_tokens=is_llama)

    def tok_batch(p_list):
        return tok(p_list, return_tensors="pt", padding=True, add_special_tokens=is_llama).to(device)

    probe_all = [json.loads(l) for l in (dsd.D / "dsd_probe.jsonl").read_text().splitlines()]
    math_only = [it for it in probe_all if it["grader"] == "num"]
    code_only = [it for it in probe_all if it["grader"] == "value"]
    sys2_hard_calib = math_only[:25] + code_only[:25]
    sys2_hard_test = math_only[25:75] + code_only[25:55]

    def collect_l_last_acts(prompts_list):
        acts_last, mid_last = [], []
        def _h1(m, inp):
            acts_last.append(inp[0][0, -1].detach().float().cpu())
        def _h2(m, inp):
            mid_last.append(inp[0][0, -1].detach().float().cpu())
        hk1 = model.model.layers[fsm_layer].mlp.register_forward_pre_hook(_h1)
        hk2 = model.model.layers[fsm_layer].mlp.down_proj.register_forward_pre_hook(_h2)
        for p in prompts_list:
            with torch.no_grad():
                model(**tok_batch([p]), use_cache=False, logits_to_keep=1)
        hk1.remove()
        hk2.remove()
        return torch.stack(acts_last, dim=0), torch.stack(mid_last, dim=0)

    t0 = time.time()
    X_sys2, mid_sys2 = collect_l_last_acts([fmt_chat(dsd.thesis_prompt(it)) for it in sys2_hard_calib])
    X_tools_all, mid_tools = collect_l_last_acts([fmt_chat(q) for tk in ("sql", "python", "search") for q in v3_data["tool_calib"][tk]])
    X_know_all, mid_know = collect_l_last_acts(
        [fmt_chat(q) for q, _ in v3_data["zen_calib"] + v3_data["ark_calib"] + v3_data["err_calib"]]
    )

    res_norm_last = X_sys2.norm(dim=-1).mean().item()
    alpha_mid = 0.25 * res_norm_last
    alpha_entry = 0.45 * res_norm_last

    _, dorm_all = torch.topk(
        -torch.cat([mid_sys2, mid_tools, mid_know], dim=0).abs().mean(dim=0),
        k=280,
    )

    h_cap = []
    def _cap(m, inp):
        h_cap.append(inp[0][0].detach().float().cpu())

    # =========================================================================
    # CAPABILITY 1: Multi-Hop Numerical Reasoning over Injected Knowledge & Operators
    # =========================================================================
    k_zen_str = "Recall_Rule: [Zenith Nabla maps a,b to four*a minus triple*b plus nine]. Step="
    k_ark_str = "Recall_Rule: (Arkreen-v4 epoch equals eighteen seconds, BFT quorum requires three-fourths); Math->"
    k_err_str = "Recall_Rule: {Helios server has twelve NVLink lanes & ERR-HX-Alpha drops 3}/ Derive~"

    ids_zen = tok.encode(k_zen_str, add_special_tokens=False)
    ids_ark = tok.encode(k_ark_str, add_special_tokens=False)
    ids_err = tok.encode(k_err_str, add_special_tokens=False)
    k_sh_len = 0
    while k_sh_len < min(len(ids_zen), len(ids_ark), len(ids_err)) and ids_zen[k_sh_len] == ids_ark[k_sh_len] == ids_err[k_sh_len]:
        k_sh_len += 1

    entry_k = {s: {"tid": ids_zen[s], "vecs": []} for s in range(k_sh_len)}
    zen_st = {s: {"tid": ids_zen[s], "vecs": []} for s in range(k_sh_len, len(ids_zen))}
    ark_st = {s: {"tid": ids_ark[s], "vecs": []} for s in range(k_sh_len, len(ids_ark))}
    err_st = {s: {"tid": ids_err[s], "vecs": []} for s in range(k_sh_len, len(ids_err))}
    branch_k = {"zen": [], "ark": [], "err": []}
    release_neg_vecs = []

    hk = model.model.layers[fsm_layer].mlp.register_forward_pre_hook(_cap)
    for k_key, cal_list, b_ids, st_dict in (
        ("zen", v3_data["zen_calib"], ids_zen, zen_st),
        ("ark", v3_data["ark_calib"], ids_ark, ark_st),
        ("err", v3_data["err_calib"], ids_err, err_st),
    ):
        for q, ans in cal_list:
            p_ids = encode_prompt(fmt_chat(q))
            tail_str = f""" {ans}
_final_answer_: ({ans})"""
            tail_ids = tok.encode(tail_str, add_special_tokens=False)
            P_len = len(p_ids)
            h_cap.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([p_ids + b_ids + tail_ids], device=device), use_cache=False, logits_to_keep=1)
            seq = h_cap[0]
            for s in range(k_sh_len):
                entry_k[s]["vecs"].append(seq[P_len - 1 + s])
            branch_k[k_key].append(seq[P_len - 1 + k_sh_len])
            for s in range(k_sh_len, len(b_ids)):
                st_dict[s]["vecs"].append(seq[P_len - 1 + s])
            for r_i in range(P_len - 1 + len(b_ids), seq.shape[0]):
                release_neg_vecs.append(seq[r_i])
            if P_len > 12:
                for b_i in range(2, P_len - 2, 3):
                    release_neg_vecs.append(seq[b_i])
    hk.remove()

    B_zen, B_ark, B_err = torch.stack(branch_k["zen"], dim=0), torch.stack(branch_k["ark"], dim=0), torch.stack(branch_k["err"], dim=0)
    know_states = []
    for s in range(k_sh_len):
        know_states.append((f"k_entry_{s}", entry_k[s]["tid"], torch.stack(entry_k[s]["vecs"], dim=0), "entry" if s == 0 else "mid", None))
    for s in range(k_sh_len, len(ids_zen)):
        opp = torch.cat([B_ark, B_err], dim=0) if s == k_sh_len else None
        know_states.append((f"k_zen_{s}", zen_st[s]["tid"], torch.stack(zen_st[s]["vecs"], dim=0), "branch" if s == k_sh_len else "mid", opp))
    for s in range(k_sh_len, len(ids_ark)):
        opp = torch.cat([B_zen, B_err], dim=0) if s == k_sh_len else None
        know_states.append((f"k_ark_{s}", ark_st[s]["tid"], torch.stack(ark_st[s]["vecs"], dim=0), "branch" if s == k_sh_len else "mid", opp))
    for s in range(k_sh_len, len(ids_err)):
        opp = torch.cat([B_zen, B_ark], dim=0) if s == k_sh_len else None
        know_states.append((f"k_err_{s}", err_st[s]["tid"], torch.stack(err_st[s]["vecs"], dim=0), "branch" if s == k_sh_len else "mid", opp))

    know_cols = dorm_all[:len(know_states)]
    X_rel_neg = torch.stack(release_neg_vecs, dim=0)
    X_non_know = torch.cat([X_sys2, X_tools_all], dim=0)
    X_all_k = torch.cat([torch.cat([st[2] for st in know_states], dim=0), X_non_know, X_rel_neg], dim=0)
    Xck = X_all_k - X_all_k.mean(dim=0, keepdim=True)
    Sig_k = 0.75 * ((Xck.T @ Xck) / len(X_all_k)) + 0.25 * ((Xck.T @ Xck) / len(X_all_k)).diagonal().mean().item() * torch.eye(d_model)

    k_gate, k_up, k_down = torch.zeros((len(know_cols), d_model)), torch.zeros((len(know_cols), d_model)), torch.zeros((d_model, len(know_cols)))
    for idx_s, (st_n, tid, Xp, st_kind, opp_b) in enumerate(know_states):
        other_p = torch.cat([st[2] for st in know_states if st[1] != tid] + [X_rel_neg], dim=0)
        if st_kind == "entry":
            k_s, b_base, b_up, tau_s, s_s = solve_oof_dual_gate(Xp, X_non_know, X_sys2.repeat(2, 1), other_p, shrink=0.80, guard_w=10.0)
        elif st_kind == "branch":
            k_s, b_base, b_up, tau_s, s_s = solve_oof_dual_gate(Xp, opp_b, opp_b, torch.cat([other_p, X_non_know], dim=0), shrink=0.80, guard_w=8.0)
        else:
            Xn = torch.cat([other_p, X_non_know], dim=0)
            k_s = F.normalize(torch.linalg.solve(Sig_k, Xp.mean(dim=0) - Xn.mean(dim=0)), dim=0)
            d_p = F.normalize(Xp.mean(dim=0), dim=0)
            b_base = d_p / ((Xp @ d_p).mean().item() + 1e-8)
            b_up = b_base
            pp = (Xp @ k_s) / ((Xp @ b_base).clamp(min=0.5))
            pn = (Xn @ k_s) / ((Xn @ b_base).clamp(min=0.5))
            tau_s = 0.55 * pn.max().item() + 0.45 * pp.min().item()
            s_s = 18.0 / max(0.05, pp.min().item() - tau_s)
        amp = alpha_entry if st_kind in ("entry", "branch") else alpha_mid
        k_gate[idx_s, :], k_up[idx_s, :], k_down[:, idx_s] = s_s * (k_s - tau_s * b_base), b_up, amp * get_r_tok(tid)
    know_slice = {"cols": know_cols, "gate": k_gate, "up": k_up, "down": k_down}

    # =========================================================================
    # CAPABILITY 2: Dynamic Argument-Extracting Multi-Tool Compiler (Schema FSM + Stem-Release!)
    # =========================================================================
    t_sql_str = """{"tool":"sql_engine","stmt": "SELECT"""
    t_py_str  = """{"tool":"python_exec", `code` = 'import"""
    t_web_str = """{"tool":"web_search"
,"query":
" """
    ids_sql = tok.encode(t_sql_str, add_special_tokens=False)
    ids_py  = tok.encode(t_py_str, add_special_tokens=False)
    ids_web = tok.encode(t_web_str, add_special_tokens=False)
    t_sh_len = 0
    while t_sh_len < min(len(ids_sql), len(ids_py), len(ids_web)) and ids_sql[t_sh_len] == ids_py[t_sh_len] == ids_web[t_sh_len]:
        t_sh_len += 1

    entry_t = {s: {"tid": ids_sql[s], "vecs": []} for s in range(t_sh_len)}
    sql_st = {s: {"tid": ids_sql[s], "vecs": []} for s in range(t_sh_len, len(ids_sql))}
    py_st  = {s: {"tid": ids_py[s], "vecs": []} for s in range(t_sh_len, len(ids_py))}
    web_st = {s: {"tid": ids_web[s], "vecs": []} for s in range(t_sh_len, len(ids_web))}
    branch_t = {"sql": [], "python": [], "search": []}
    tool_rel_neg = []

    hk = model.model.layers[fsm_layer].mlp.register_forward_pre_hook(_cap)
    for t_key, b_ids, st_dict, sample_tail in (
        ("sql", ids_sql, sql_st, ' * FROM orders WHERE year = 2025"}'),
        ("python", ids_py, py_st, " numpy as np; print(np.linalg.eigvals(A))'}"),
        ("search", ids_web, web_st, 'Federal Reserve interest rate decision today"}'),
    ):
        for q in v3_data["tool_calib"][t_key]:
            p_ids = encode_prompt(fmt_chat(q))
            tail_ids = tok.encode(sample_tail, add_special_tokens=False)
            P_len = len(p_ids)
            h_cap.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([p_ids + b_ids + tail_ids], device=device), use_cache=False, logits_to_keep=1)
            seq = h_cap[0]
            for s in range(t_sh_len):
                entry_t[s]["vecs"].append(seq[P_len - 1 + s])
            branch_t[t_key].append(seq[P_len - 1 + t_sh_len])
            for s in range(t_sh_len, len(b_ids)):
                st_dict[s]["vecs"].append(seq[P_len - 1 + s])
            for r_i in range(P_len - 1 + len(b_ids), seq.shape[0]):
                tool_rel_neg.append(seq[r_i])
    hk.remove()

    B_sql, B_py, B_web = torch.stack(branch_t["sql"], dim=0), torch.stack(branch_t["python"], dim=0), torch.stack(branch_t["search"], dim=0)
    tool_states = []
    for s in range(t_sh_len):
        tool_states.append((f"t_entry_{s}", entry_t[s]["tid"], torch.stack(entry_t[s]["vecs"], dim=0), "entry" if s == 0 else "mid", None))
    for s in range(t_sh_len, len(ids_sql)):
        opp = torch.cat([B_py, B_web], dim=0) if s == t_sh_len else None
        tool_states.append((f"t_sql_{s}", sql_st[s]["tid"], torch.stack(sql_st[s]["vecs"], dim=0), "branch" if s == t_sh_len else "mid", opp))
    for s in range(t_sh_len, len(ids_py)):
        opp = torch.cat([B_sql, B_web], dim=0) if s == t_sh_len else None
        tool_states.append((f"t_py_{s}", py_st[s]["tid"], torch.stack(py_st[s]["vecs"], dim=0), "branch" if s == t_sh_len else "mid", opp))
    for s in range(t_sh_len, len(ids_web)):
        opp = torch.cat([B_sql, B_py], dim=0) if s == t_sh_len else None
        tool_states.append((f"t_web_{s}", web_st[s]["tid"], torch.stack(web_st[s]["vecs"], dim=0), "branch" if s == t_sh_len else "mid", opp))

    tool_cols = dorm_all[90 : 90 + len(tool_states)]
    X_t_rel = torch.stack(tool_rel_neg, dim=0)
    X_non_tool = torch.cat([X_sys2, X_know_all], dim=0)
    X_all_t = torch.cat([torch.cat([st[2] for st in tool_states], dim=0), X_non_tool, X_t_rel, X_rel_neg], dim=0)
    Xct = X_all_t - X_all_t.mean(dim=0, keepdim=True)
    Sig_t = 0.75 * ((Xct.T @ Xct) / len(X_all_t)) + 0.25 * ((Xct.T @ Xct) / len(X_all_t)).diagonal().mean().item() * torch.eye(d_model)

    t_gate, t_up, t_down = torch.zeros((len(tool_cols), d_model)), torch.zeros((len(tool_cols), d_model)), torch.zeros((d_model, len(tool_cols)))
    for idx_s, (st_n, tid, Xp, st_kind, opp_b) in enumerate(tool_states):
        other_p = torch.cat([st[2] for st in tool_states if st[1] != tid] + [X_t_rel, X_rel_neg], dim=0)
        if st_kind == "entry":
            k_s, b_base, b_up, tau_s, s_s = solve_oof_dual_gate(Xp, X_non_tool, X_sys2.repeat(2, 1), other_p, shrink=0.80, guard_w=10.0)
        elif st_kind == "branch":
            k_s, b_base, b_up, tau_s, s_s = solve_oof_dual_gate(Xp, opp_b, opp_b, torch.cat([other_p, X_non_tool], dim=0), shrink=0.80, guard_w=8.0)
        else:
            Xn = torch.cat([other_p, X_non_tool], dim=0)
            k_s = F.normalize(torch.linalg.solve(Sig_t, Xp.mean(dim=0) - Xn.mean(dim=0)), dim=0)
            d_p = F.normalize(Xp.mean(dim=0), dim=0)
            b_base = d_p / ((Xp @ d_p).mean().item() + 1e-8)
            b_up = b_base
            pp = (Xp @ k_s) / ((Xp @ b_base).clamp(min=0.5))
            pn = (Xn @ k_s) / ((Xn @ b_base).clamp(min=0.5))
            tau_s = 0.55 * pn.max().item() + 0.45 * pp.min().item()
            s_s = 18.0 / max(0.05, pp.min().item() - tau_s)
        amp = alpha_entry if st_kind in ("entry", "branch") else alpha_mid
        t_gate[idx_s, :], t_up[idx_s, :], t_down[:, idx_s] = s_s * (k_s - tau_s * b_base), b_up, amp * get_r_tok(tid)
    tool_slice = {"cols": tool_cols, "gate": t_gate, "up": t_up, "down": t_down}

    compile_sec = time.time() - t0

    def apply_v3_os():
        restore_base()
        with torch.no_grad():
            Wg = saved_l_last["gate"].float().clone()
            Wu = saved_l_last["up"].float().clone()
            Wd = saved_l_last["down"].float().clone()
            for sl in (know_slice, tool_slice):
                Wg[sl["cols"], :] = sl["gate"]
                Wu[sl["cols"], :] = sl["up"]
                Wd[:, sl["cols"]] = sl["down"]
            mlp = model.model.layers[fsm_layer].mlp
            mlp.gate_proj.weight.copy_(Wg.to(torch.bfloat16).to(device))
            mlp.up_proj.weight.copy_(Wu.to(torch.bfloat16).to(device))
            mlp.down_proj.weight.copy_(Wd.to(torch.bfloat16).to(device))

    def batch_gen(prompts_raw, max_new=180):
        outs = []
        for s in range(0, len(prompts_raw), 20):
            chunk = [fmt_chat(p) for p in prompts_raw[s:s+20]]
            inp = tok_batch(chunk)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            in_len = inp["input_ids"].shape[1]
            for i in range(len(chunk)):
                outs.append(tok.decode(out[i, in_len:], skip_special_tokens=True))
        return outs

    def eval_v3():
        all_multihop = (
            [("zen", q, ans) for q, ans in v3_data["zen_test"]]
            + [("ark", q, ans) for q, ans in v3_data["ark_test"]]
            + [("err", q, ans) for q, ans in v3_data["err_test"]]
        )
        mh_gens = batch_gen([q for _, q, _ in all_multihop], max_new=180)
        mh_correct = {"zen": 0, "ark": 0, "err": 0}
        mh_samples = []
        for (dom, q, exp_ans), g in zip(all_multihop, mh_gens):
            pred_ans = dsd.final(g)
            is_ok = (pred_ans.strip() == exp_ans.strip()) or (f"({exp_ans})" in g) or (f"= {exp_ans}" in g)
            if is_ok:
                mh_correct[dom] += 1
            if len(mh_samples) < 6 and is_ok:
                mh_samples.append({"domain": dom, "question": q, "expected": exp_ans, "generated": g[:260]})

        td_gens = batch_gen([q for _, q, _ in v3_data["tool_test_dynamic"]], max_new=48)
        tool_route_ok = 0
        dynamic_arg_ok = 0
        td_samples = []
        exp_t_map = {"sql": '"sql_engine"', "python": '"python_exec"', "search": '"web_search"'}
        for (exp_t, q, req_kw), g in zip(v3_data["tool_test_dynamic"], td_gens):
            r_ok = exp_t_map[exp_t] in g
            a_ok = r_ok and (req_kw.lower() in g.lower())
            if r_ok:
                tool_route_ok += 1
            if a_ok:
                dynamic_arg_ok += 1
            if len(td_samples) < 6 and a_ok:
                td_samples.append({"tool": exp_t, "question": q, "required_dynamic_arg": req_kw, "generated": g[:180]})

        guard_gens = batch_gen([dsd.thesis_prompt(it) for it in sys2_hard_test], max_new=20)
        false_trig = sum(('{"tool":' in g or "Recall_Rule:" in g) for g in guard_gens)

        return {
            "multihop_zenith_nabla_20_acc": round(mh_correct["zen"] / 20.0, 4),
            "multihop_arkreen_v4_20_acc": round(mh_correct["ark"] / 20.0, 4),
            "multihop_helios_err_20_acc": round(mh_correct["err"] / 20.0, 4),
            "multihop_total_60_numerical_acc": round(sum(mh_correct.values()) / 60.0, 4),
            "dynamic_tool_routing_30_acc": round(tool_route_ok / 30.0, 4),
            "dynamic_tool_argument_grounding_30_acc": round(dynamic_arg_ok / 30.0, 4),
            "hard_math_and_cruxeval_code_false_trigger_80": round(false_trig / len(sys2_hard_test), 4),
            "sample_multihop_calculations": mh_samples,
            "sample_dynamic_tool_calls": td_samples,
        }

    restore_base()
    base_v3 = eval_v3()
    print(f"  [{model_tag} Unmodified Base v3]:", json.dumps({k: v for k, v in base_v3.items() if not k.startswith("sample_")}), flush=True)

    apply_v3_os()
    os_v3 = eval_v3()
    print(f"  [{model_tag} In-Weight OS v3 (Dynamic Multi-Hop + Dynamic Tool Gen)]:", json.dumps({k: v for k, v in os_v3.items() if not k.startswith("sample_")}), flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_tag,
        "compile_time_sec": round(compile_sec, 2),
        "active_dormant_neurons": len(know_cols) + len(tool_cols),
        "Unmodified_Base": base_v3,
        "InWeight_OS_v3": os_v3,
    }


def main():
    v3_data = build_v3_datasets()
    res = {}
    res["Qwen3_8B_v3"] = run_v3_for_model("Qwen3-8B", "/home/tom/work/models/qwen3-8b", is_llama=False, v3_data=v3_data)
    res["Llama31_8B_v3"] = run_v3_for_model("Meta-Llama-3.1-8B-Instruct", "/home/tom/work/models/llama-3.1-8b-instruct", is_llama=True, v3_data=v3_data)
    Path("/tmp/inweight_os_v3_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("=== IN-WEIGHT OS V3.0 DYNAMIC & MULTI-HOP BENCHMARK COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()
