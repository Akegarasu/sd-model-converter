import torch
from safetensors.torch import save_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--f", type=str, default="model.ckpt", help="path to model")
parser.add_argument("--half", action="store_true", default=False, help="save fp16 model")
parser.add_argument("--no-ema", action="store_true", default=False, help="no ema")
parser.add_argument("--safe-tensors", action="store_true", default=False, help="use safetensors model format")


def convert(path: str, half: bool, no_ema: bool):
    m = torch.load(path, map_location="cpu")
    state_dict = m["state_dict"] if "state_dict" in m else m
    ok = {}
    for k, v in state_dict.items():
        if no_ema:
            if "model_ema" not in k:
                ok[k] = v.half() if half else v
        else:
            ok[k] = v.half() if half else v

    return ok


if __name__ == "__main__":
    cmds = parser.parse_args()
    model_name = ".".join(cmds.f.split(".")[:-1])
    converted = convert(cmds.f, cmds.half, cmds.no_ema)
    if cmds.safe_tensors:
        save_file(converted, model_name + "safetensors")
    else:
        torch.save({"state_dict": converted}, model_name + "-convert.ckpt")
