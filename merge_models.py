#!/usr/bin/env python3
"""Merge 4-bit quantized LLM with non-quantized projector weights."""

from safetensors.torch import load_file, save_file
import shutil
import os

# Load 4-bit model weights
print("Loading 4-bit model...")
quant_weights = load_file('exported-fastvlm-0.5b-4bit/model.safetensors')

# Load non-quantized weights (for projector only)
print("Loading full model for projector...")
full_weights = load_file('exported-fastvlm-0.5b/model.safetensors')

# Copy projector weights from full model to quantized
projector_keys = [k for k in full_weights.keys() if 'multi_modal_projector' in k]
print(f"Copying projector weights: {projector_keys}")
for k in projector_keys:
    quant_weights[k] = full_weights[k]
    print(f"  Added: {k} - {full_weights[k].shape}")

# Create merged output directory
os.makedirs('exported-fastvlm-0.5b-4bit-merged', exist_ok=True)

# Save merged weights
print("Saving merged model...")
save_file(quant_weights, 'exported-fastvlm-0.5b-4bit-merged/model.safetensors')
print(f"Saved merged model with {len(quant_weights)} tensors")

# Copy other files from 4-bit model
for f in os.listdir('exported-fastvlm-0.5b-4bit'):
    if f != 'model.safetensors' and not f.startswith('.'):
        src = os.path.join('exported-fastvlm-0.5b-4bit', f)
        dst = os.path.join('exported-fastvlm-0.5b-4bit-merged', f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        print(f"Copied: {f}")

print("\nDone! Use 'exported-fastvlm-0.5b-4bit-merged' folder in Xcode")
