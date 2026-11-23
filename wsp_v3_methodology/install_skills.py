"""
Hot-swappable 2-Layer Skill Cartridge Loader for In-Weight OS v3.0 (Sensor-Beacon-Actuator Architecture).
Installs both the Mid-Layer Semantic Sensor (`sensor_layer_patch`) and Terminal Actuator (`terminal_fsm_layer`)
into Qwen3-8B or Llama-3.1-8B-Instruct in ~20ms.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def install_cartridges(model, cartridge_paths):
    with torch.no_grad():
        for pt_path in cartridge_paths:
            payload = torch.load(pt_path, map_location="cpu", weights_only=True)
            if "sensor_layer_patch" in payload:
                sp = payload["sensor_layer_patch"]
                s_idx = int(sp["layer"])
                s_mlp = model.model.layers[s_idx].mlp
                s_cols = sp["cols"].long()
                dev, dt = s_mlp.gate_proj.weight.device, s_mlp.gate_proj.weight.dtype
                s_mlp.gate_proj.weight[s_cols, :] = sp["gate"].to(device=dev, dtype=dt)
                s_mlp.up_proj.weight[s_cols, :] = sp["up"].to(device=dev, dtype=dt)
                s_mlp.down_proj.weight[:, s_cols] = sp["down"].to(device=dev, dtype=dt)
            layer_idx = int(payload.get("terminal_fsm_layer", payload.get("layer")))
            mlp = model.model.layers[layer_idx].mlp
            cols = payload["cols"].long()
            dev = mlp.gate_proj.weight.device
            dt = mlp.gate_proj.weight.dtype
            mlp.gate_proj.weight[cols, :] = payload["gate"].to(device=dev, dtype=dt)
            mlp.up_proj.weight[cols, :] = payload["up"].to(device=dev, dtype=dt)
            mlp.down_proj.weight[:, cols] = payload["down"].to(device=dev, dtype=dt)
            print(f"[In-Weight OS v3.0] Installed 2-Layer Sensor-Beacon-Actuator cartridge '{payload.get('skill_name', pt_path)}'")
    return model
