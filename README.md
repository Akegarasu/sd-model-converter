# sd-model-converter

convert stable diffusion model to fp16 / remove ema / safetensors

## Usage

```
python convert.py --f path/to/model.ckpt --half --no-ema --safe-tensors
```

before: 7.17 GB
after: 1.98 GB
