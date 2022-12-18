# sd-model-converter

convert stable diffusion model to fp16 / ema only / safetensors

## Usage

default save ema only model.

--fp16: save fp16 model (half)  
--full: full model (with non-ema)  
--safe-tensors

```
# convert to ema only
python convert.py --f path/to/model.ckpt

# convert to ema only, fp16
python convert.py --f path/to/model.ckpt --fp16

# convert to ema only, fp16, safe-tensors
python convert.py --f path/to/model.ckpt --fp16 --safe-tensors

# convert to fp16
python convert.py --f path/to/model.ckpt --fp16 --full
```

before: 

Anything-V3.0.ckpt 7.17 GB  

after: 

Anything-V3.0-prune.ckpt 3.97 GB  
Anything-V3.0-prune-fp16.ckpt 1.98 GB  
Anything-V3.0-prune-fp16.safetensors 1.98 GB  

![GIESB1FHR%3_ _ A}88M0}X](https://user-images.githubusercontent.com/36563862/208284000-81823e5d-1686-427f-913d-98a95d232c25.jpg)
