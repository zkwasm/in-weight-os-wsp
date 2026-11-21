#!/usr/bin/env python3
r"""
Step 6 of Universal WSP Toolchain: Batch Generation & Interactive Inference CLI
===============================================================================
Loads a base model together with either:
  - `--wsp_dir` (a Step 2 `wsp_artifact.pt` directory),
  - `--peft_dir` (a Step 3 `adapter_model.bin` PEFT LoRA directory), or
  - `--merged_model` (a Step 4 fused Hugging Face model directory)
and generates responses for any input `.jsonl` file!
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wsp_0_common import format_messages_safe, get_transformer_layers, parse_sft_record

def main():
    parser = argparse.ArgumentParser(description="Step 6: Run Generation Inference with Compiled WSP Model or PEFT LoRA")
    parser.add_argument("--model", type=str, required=True, help="Base model ID or Step 4 merged model path")
    parser.add_argument("--wsp_dir", type=str, default=None, help="Optional Step 2 WSP artifact directory")
    parser.add_argument("--peft_dir", type=str, default=None, help="Optional Step 3 PEFT LoRA directory")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input .jsonl file to run generation on")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output .jsonl file with generated responses")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    _, layers = get_transformer_layers(model)

    if args.wsp_dir:
        art = torch.load(Path(args.wsp_dir) / "wsp_artifact.pt", map_location="cpu")
        with torch.no_grad():
            for (l, mname), dW in art["deltas"].items():
                w = layers[l].get_submodule(mname).weight
                w.add_(dW.to(args.device).to(w.dtype))
    elif args.peft_dir:
        cfg = json.loads((Path(args.peft_dir) / "adapter_config.json").read_text(encoding="utf-8"))
        sd = torch.load(Path(args.peft_dir) / "adapter_model.bin", map_location="cpu")
        with torch.no_grad():
            for k, v in sd.items():
                if k.endswith(".lora_A.weight"):
                    prefix = k[: -len(".lora_A.weight")]
                    A = v.float().to(args.device)
                    B = sd[f"{prefix}.lora_B.weight"].float().to(args.device)
                    mpath = prefix.replace("base_model.model.", "")
                    mod = model.get_submodule(mpath)
                    mod.weight.add_((B @ A).to(mod.weight.dtype))

    records = [json.loads(l) for l in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_lines = []
    with torch.no_grad():
        for rec in records:
            msgs, gold_resp = parse_sft_record(rec)
            p_txt = format_messages_safe(tok, msgs, add_generation_prompt=True)
            inp = tok(p_txt, return_tensors="pt").to(args.device)
            gen_ids = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
            pred_txt = tok.decode(gen_ids[0, inp.input_ids.shape[1] :], skip_special_tokens=True)
            out_lines.append(json.dumps({"prompt_messages": msgs, "gold_response": gold_resp, "generated_response": pred_txt}, ensure_ascii=False))

    Path(args.output_jsonl).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[Step 6] Wrote {len(out_lines)} generated responses to {args.output_jsonl}")

if __name__ == "__main__":
    main()
