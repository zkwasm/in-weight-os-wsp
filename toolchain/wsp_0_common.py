#!/usr/bin/env python3
"""
Core Architecture-Agnostic & Format-Agnostic Utilities for Universal WSP Toolchain
==================================================================================
1. Auto-discovers Transformer layers and groups linear projections that share the
   same input activation tensor across ANY Hugging Face AutoModelForCausalLM
   (Qwen, Llama, Mistral, Gemma, Phi, DeepSeek, Granite, SmolLM, Falcon, GPT-2, etc.).
2. Parses all 5 standard SFT dataset formats (OpenAI `messages`, Alpaca `instruction/input/output`,
   ShareGPT `conversations`, QA `prompt/response`, and raw `text`), supporting both
   Instruct models (with `enable_thinking=False` safety) and Base models (no chat_template).
3. Automatically detects shared prompt prefixes (if any) for 3x faster KV-cached compilation,
   while seamlessly handling heterogeneous zero-prefix datasets.
"""
import copy
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def get_transformer_layers(model):
    """Returns (layer_prefix_str, ModuleList_of_layers) for any HF CausalLM architecture."""
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ("layers", lambda m: m.layers),
    ]
    for prefix, getter in candidates:
        try:
            layers = getter(model)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return prefix, layers
        except AttributeError:
            continue
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            return name, mod
    raise ValueError("Could not automatically locate Transformer block ModuleList in model.")


@torch.no_grad()
def discover_architecture_groups(model, tok, device):
    """
    Traces 1 dummy token through Layer 0 to automatically group all 2D linear modules
    that receive the exact same input activation tensor (e.g., q_proj/k_proj/v_proj -> 'g0',
    o_proj -> 'g1', gate_proj/up_proj -> 'g2', down_proj -> 'g3').
    Works on any architecture without hardcoded module names!
    """
    layer_prefix, layers = get_transformer_layers(model)
    lay0 = layers[0]
    linear_names = []
    for name, mod in lay0.named_modules():
        if name and hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor) and mod.weight.ndim == 2:
            if "norm" not in name.lower() and "embed" not in name.lower():
                linear_names.append(name)

    ptr_to_mods = {}
    ptr_order = []
    keep_alive = []
    hooks = []
    for mname in linear_names:
        mod = lay0.get_submodule(mname)
        def make_hk(mn, in_features):
            def f(_, inp):
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0]
                    keep_alive.append(x)
                    key = (x.data_ptr(), int(x.shape[-1]), int(in_features))
                    if key not in ptr_to_mods:
                        ptr_to_mods[key] = []
                        ptr_order.append(key)
                    ptr_to_mods[key].append(mn)
            return f
        hooks.append(mod.register_forward_pre_hook(make_hk(mname, mod.weight.shape[1])))

    dummy_ids = torch.tensor([[tok.eos_token_id or 1, tok.eos_token_id or 1]], device=device)
    model(input_ids=dummy_ids, use_cache=False)
    for h in hooks:
        h.remove()

    # Map group_id -> representative module name, and module_name -> group_id
    group_to_rep = {}
    mod_to_group = {}
    residual_out_mods = []
    for idx, p in enumerate(ptr_order):
        m_list = ptr_to_mods[p]
        gid = f"g{idx}_{m_list[0].replace('.', '_')}"
        group_to_rep[gid] = m_list[0]
        for mn in m_list:
            mod_to_group[mn] = gid
        if len(m_list) == 1 and any(k in m_list[0].lower() for k in ("o_proj", "out_proj", "down_proj", "output_linear", "c_proj", "fc2", "dense_4h_to_h")):
            residual_out_mods.append((gid, m_list[0]))

    if not residual_out_mods:
        # Fallback: last module in block
        last_m = linear_names[-1]
        residual_out_mods.append((mod_to_group[last_m], last_m))

    target_leaf_names = sorted(set(m.split(".")[-1] for m in mod_to_group.keys()))
    return {
        "layer_prefix": layer_prefix,
        "num_layers": len(layers),
        "group_to_rep": group_to_rep,
        "mod_to_group": mod_to_group,
        "residual_out_mods": residual_out_mods,
        "target_leaf_names": target_leaf_names,
    }


def format_messages_safe(tok, messages, add_generation_prompt=True):
    """Formats chat messages safely across Qwen3 (`enable_thinking=False`), Llama, Mistral, and Base models."""
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
    # Fallback for unaligned Base models without a chat_template
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system" and content:
            parts.append(f"### System:\n{content}")
        elif role in ("user", "human"):
            parts.append(f"### Instruction:\n{content}")
        elif role in ("assistant", "gpt"):
            parts.append(f"### Response:\n{content}")
    if add_generation_prompt:
        parts.append("### Response:\n")
    return "\n\n".join(parts)


def parse_sft_record(item, system_override=None):
    """
    Converts any of the 5 standard SFT JSON formats into a normalized `(context_messages, target_response_str)` tuple:
      1. OpenAI `messages`: {"messages": [{"role": ..., "content": ...}, ...]}
      2. Alpaca: {"instruction": ..., "input": ..., "output": ..., "system": ..., "history": ...}
      3. ShareGPT: {"conversations": [{"from": "human"|"gpt"|"system", "value": ...}]}
      4. Prompt-Response / QA: {"prompt"|"question"|"query": ..., "response"|"answer"|"completion"|"output": ...}
      5. Distil tool-call: {"history": [...], "gold": {...}}
      6. Raw text: {"text": ...}
    """
    if "messages" in item:
        msgs = list(item["messages"])
        if system_override and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": system_override}] + msgs
        for idx in range(len(msgs) - 1, -1, -1):
            if msgs[idx].get("role") in ("assistant", "gpt", "model"):
                return msgs[:idx], str(msgs[idx].get("content", ""))
        raise ValueError("No assistant message found in `messages` record.")

    if "conversations" in item:
        role_map = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant"}
        msgs = [{"role": role_map.get(c.get("from", "user"), "user"), "content": str(c.get("value", ""))} for c in item["conversations"]]
        if system_override and (not msgs or msgs[0]["role"] != "system"):
            msgs = [{"role": "system", "content": system_override}] + msgs
        for idx in range(len(msgs) - 1, -1, -1):
            if msgs[idx]["role"] == "assistant":
                return msgs[:idx], msgs[idx]["content"]

    if "history" in item and "gold" in item:
        msgs = []
        if system_override or item.get("system"):
            msgs.append({"role": "system", "content": system_override or item["system"]})
        msgs.extend(item["history"])
        gold = item["gold"]
        resp = gold if isinstance(gold, str) else ("<tool_call>\n" + json.dumps(gold, ensure_ascii=False) + "\n</tool_call>")
        return msgs, resp

    if "instruction" in item:
        msgs = []
        sys_txt = system_override or item.get("system", "")
        if sys_txt:
            msgs.append({"role": "system", "content": sys_txt})
        for h_turn in item.get("history", []) or []:
            if isinstance(h_turn, (list, tuple)) and len(h_turn) == 2:
                msgs.append({"role": "user", "content": str(h_turn[0])})
                msgs.append({"role": "assistant", "content": str(h_turn[1])})
        user_q = str(item["instruction"])
        if item.get("input"):
            user_q = f"{user_q}\n\n{item['input']}"
        msgs.append({"role": "user", "content": user_q})
        resp = str(item.get("output") or item.get("response") or item.get("answer") or "")
        return msgs, resp

    for pk in ("prompt", "question", "query", "input"):
        if pk in item:
            msgs = []
            sys_txt = system_override or item.get("system", "")
            if sys_txt:
                msgs.append({"role": "system", "content": sys_txt})
            msgs.append({"role": "user", "content": str(item[pk])})
            for rk in ("response", "answer", "output", "completion", "target"):
                if rk in item:
                    return msgs, str(item[rk])

    if "text" in item:
        txt = str(item["text"])
        split_idx = len(txt) // 2
        return [{"role": "user", "content": txt[:split_idx]}], txt[split_idx:]

    raise ValueError(f"Unrecognized SFT record format: {list(item.keys())}")


def load_universal_sft_dataset(tok, file_path, system_override=None, max_length=2048):
    """Loads a JSON or JSONL file in any of the 5 SFT formats and returns list of (prompt_ids, response_ids)."""
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8").strip()
    if raw_text.startswith("["):
        records = json.loads(raw_text)
    else:
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == getattr(tok, "unk_token_id", None):
        end_id = tok.eos_token_id

    pairs = []
    for item in records:
        msgs, resp = parse_sft_record(item, system_override=system_override)
        p_txt = format_messages_safe(tok, msgs, add_generation_prompt=True)
        # Match character-by-character <tool_call> tokenization used by standard fine-tuning
        p_ids = tok.encode(p_txt, add_special_tokens=False)
        if resp.startswith("<tool_call>"):
            g_ids = tok.encode("<", add_special_tokens=False) + tok.encode(resp[1:], add_special_tokens=False) + [end_id]
        else:
            g_ids = tok.encode(resp, add_special_tokens=False) + [end_id]
        if len(p_ids) + len(g_ids) > max_length:
            keep_p = max(32, max_length - len(g_ids))
            p_ids = p_ids[-keep_p:]
        pairs.append((torch.tensor(p_ids, dtype=torch.long), torch.tensor(g_ids, dtype=torch.long)))
    return pairs


def detect_shared_prefix_len(pairs, min_prefix=16):
    """Returns the length k of the longest common token prefix across all prompts (or 0 if < min_prefix)."""
    if not pairs:
        return 0
    ref = pairs[0][0]
    k = len(ref)
    for p, _ in pairs[1:]:
        k = min(k, len(p))
        while k > 0 and not torch.equal(ref[:k], p[:k]):
            # Binary or linear shrink
            diff = (ref[:k] != p[:k]).nonzero()
            k = int(diff[0].item()) if len(diff) > 0 else 0
        if k < min_prefix:
            return 0
    # Leave at least 4 tokens in the shortest prompt suffix so suffix length is always >= 4
    min_p_len = min(len(p) for p, _ in pairs)
    k = min(k, max(0, min_p_len - 4))
    return k if k >= min_prefix else 0


def forward_sft_subbatch(model, tok, subbatch, shared_k=0):
    """
    Computes exact summed Cross-Entropy loss over response tokens `g` for a micro-batch of (p, g) pairs.
    If `shared_k >= 16` and `model` has `.model`, caches the shared prefix KV once per micro-batch!
    """
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    n = len(subbatch)

    if shared_k >= 16 and hasattr(model, "model") and n >= 1:
        emb = model.model.embed_tokens
        xp = emb(subbatch[0][0][:shared_k].to(device))[None]
        cache = model.model(inputs_embeds=xp, use_cache=True).past_key_values
        if n > 1 and hasattr(cache, "layers"):
            for lay in cache.layers:
                lay.keys = lay.keys.repeat(n, 1, 1, 1)
                lay.values = lay.values.repeat(n, 1, 1, 1)
        suf = [torch.cat([p[shared_k:], g]) for p, g in subbatch]
        L_max = max(len(s) for s in suf)
        ids = torch.full((n, L_max), pad_id, dtype=torch.long, device=device)
        mask = torch.ones((n, shared_k + L_max), dtype=torch.long, device=device)
        for j, s in enumerate(suf):
            ids[j, : len(s)] = s.to(device)
            mask[j, shared_k + len(s) :] = 0
        pos = torch.arange(shared_k, shared_k + L_max, device=device)[None].expand(n, -1)
        h = model.model(
            inputs_embeds=emb(ids),
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=False,
        ).last_hidden_state
        loss = 0
        for j, (p, g) in enumerate(subbatch):
            q = len(p) - shared_k
            logits = model.lm_head(h[j, q - 1 : q - 1 + len(g)]).float()
            loss = loss + F.cross_entropy(logits, g.to(device), reduction="sum")
        return loss

    # Standard zero-assumption forward for arbitrary heterogeneous prompts
    loss = 0
    for p, g in subbatch:
        full = torch.cat([p, g]).to(device)[None]
        logits = model(input_ids=full, use_cache=False).logits[0, len(p) - 1 : -1].float()
        loss = loss + F.cross_entropy(logits, g.to(device), reduction="sum")
    return loss
