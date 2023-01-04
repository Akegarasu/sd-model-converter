# sd-model-converter

convert stable diffusion model to fp16/bf16 no-ema/ema-only safetensors

## Usage

default fp32, full model.

-f: file path
-t: convert type full/ema-only/no-ema
-p: precision fp32(full)/fp16/bf16
-st: safe-tensors model format

```
# convert to ema only
python convert.py -f path/to/model.ckpt -t ema-only

or

python convert.py -f path/to/model.ckpt -t prune

# convert to ema only, fp16
python convert.py -f path/to/model.ckpt -t prune -p fp16

# convert to ema only, fp16, safe-tensors
python convert.py -f path/to/model.ckpt -t prune -p fp16 --safe-tensors

# convert to fp16
python convert.py -f path/to/model.ckpt -p fp16
```

before: 

Anything-V3.0.ckpt 7.17 GB  

after: 

![{V71`3YV22UP2A9R2 LGVDN](https://user-images.githubusercontent.com/36563862/210568635-9a17c235-8f96-4d62-ae48-870557e9cf90.png)
