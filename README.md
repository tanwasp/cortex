# Cortex

Cortex is an AI-powered productivity and accountability tracker engineered for ultimate flexibility.

## Current commands

### Inference

cd mlx-vlm

python -m mlx_vlm.generate \
--model ../exported-fastvlm-0.5b-4bit \
--image ../test-images/screenshot_1.png \
--prompt "What is this image of?"

### Export Model

cd mlx-vlm

python -m mlx_vlm.convert --hf-path ../ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
 --mlx-path ../exported-fastvlm-0.5b-2 \
 --only-llm \
 -q \
 --q-bits 4
