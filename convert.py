import torch
from torch import Tensor
from safetensors.torch import save_file, load_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--f", type=str, default="model.ckpt", help="path to model")
parser.add_argument("--fp16", action="store_true", default=False, help="save fp16 model")
parser.add_argument("--bf16", action="store_true", default=False, help="save bf16 model")
parser.add_argument("--full", action="store_true", default=False, help="save full model instead of ema only")
parser.add_argument("--safe-tensors", action="store_true", default=False, help="use safetensors model format")

cmds = parser.parse_args()


def _hf(t: Tensor):
    if not isinstance(t, Tensor):
        return t
    if cmds.fp16:
        return t.half()
    elif cmds.bf16:
        return t.bfloat16()
    else:
        return t


def convert(path: str, half: bool, ema_only: bool = True):
    if path.endswith(".safetensors"):
        m = load_file(path, device="cpu")
    else:
        m = torch.load(path, map_location="cpu")
    state_dict = m["state_dict"] if "state_dict" in m else m
    ok = {"state_dict": {}}  # should be dict() here but due to novelai's typo added a key
    if ema_only:
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
    else:
        for k, v in state_dict.items():
            ok[k] = _hf(v)

    return ok


def main():
    if cmds.fp16 and cmds.bf16:
        print("You should choose one from fp16 & bf16")
        return

    model_name = ".".join(cmds.f.split(".")[:-1])
    converted = convert(cmds.f, cmds.fp16, not cmds.full)
    save_name = f"{model_name}-convert" if cmds.full else f"{model_name}-prune"
    save_name += "-fp16" if cmds.fp16 else ""
    print("convert ok, saving model")
    if cmds.safe_tensors:
        del converted["state_dict"]
        save_file(converted, save_name + ".safetensors")
    else:
        torch.save({"state_dict": converted}, save_name + ".ckpt")
    print("convert finish.")


if __name__ == "__main__":
    main()
