import torch
from torch import Tensor
from safetensors.torch import save_file, load_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--f", type=str, default="model.ckpt", help="path to model")
parser.add_argument("--precision", default="full", help="precision fp32(full)/fp16/bf16")
parser.add_argument("--type", type=str, default="ema-only", help="convert types full/ema-only/no-ema")
parser.add_argument("--safe-tensors", action="store_true", default=False, help="use safetensors model format")

cmds = parser.parse_args()


def conv_fp16(t: Tensor):
    if not isinstance(t, Tensor):
        return t
    return t.half()


def conv_bf16(t: Tensor):
    if not isinstance(t, Tensor):
        return t
    return t.bfloat16()


def conv_full(t):
    return t


_g_precision_func = {
    "full": conv_full,
    "fp32": conv_full,
    "fp16": conv_fp16,
    "bf16": conv_bf16,
}


def convert(path: str, conv_type: str):
    ok = {}  # {"state_dict": {}}
    _hf = _g_precision_func[cmds.precision]

    if path.endswith(".safetensors"):
        m = load_file(path, device="cpu")
    else:
        m = torch.load(path, map_location="cpu")
    state_dict = m["state_dict"] if "state_dict" in m else m
    if conv_type == "ema-only":
        for k in state_dict:
            ema_k = "___"
            try:
                ema_k = "model_ema." + k[6:].replace(".", "")
            except:
                pass
            if ema_k in state_dict:
                ok[k] = _hf(state_dict[ema_k])
                print("ema: " + ema_k + " > " + k)
            elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
                ok[k] = _hf(state_dict[k])
                print(k)
            else:
                print("skipped: " + k)
    elif conv_type == "no-ema":
        for k, v in state_dict.items():
            if "model_ema" not in k:
                ok[k] = _hf(v)
    else:
        for k, v in state_dict.items():
            ok[k] = _hf(v)
    return ok


def main():
    model_name = ".".join(cmds.f.split(".")[:-1])
    converted = convert(cmds.f, cmds.type)
    save_name = f"{model_name}-{cmds.type}"
    print("convert ok, saving model")
    if cmds.safe_tensors:
        del converted["state_dict"]
        save_file(converted, save_name + ".safetensors")
    else:
        torch.save({"state_dict": converted}, save_name + ".ckpt")
    print("convert finish.")


if __name__ == "__main__":
    main()
