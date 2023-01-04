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

Anything-V3.0-prune.ckpt 3.97 GB  
Anything-V3.0-prune-fp16.ckpt 1.98 GB  
Anything-V3.0-prune-fp16.safetensors 1.98 GB  

![GIESB1FHR%3_ _ A}88M0}X](https://user-images.githubusercontent.com/36563862/208284000-81823e5d-1686-427f-913d-98a95d232c25.jpg)
