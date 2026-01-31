# Cortex: On-Device Vision-Language Model for macOS

Cortex is a native macOS application that captures your screen and uses a **local Vision-Language Model (VLM)** to describe what you're doing—entirely on-device, with no cloud API calls. It runs inference every 10 seconds and maintains a timestamped activity log.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Foundational Concepts](#foundational-concepts)
   - [What is a Neural Network?](#what-is-a-neural-network)
   - [What are Tensors?](#what-are-tensors)
   - [What are Model Weights?](#what-are-model-weights)
4. [The Two Types of Models](#the-two-types-of-models)
   - [Vision Models (Image → Numbers)](#vision-models-image--numbers)
   - [Language Models (Numbers → Text)](#language-models-numbers--text)
   - [Vision-Language Models (VLMs)](#vision-language-models-vlms)
5. [FastVLM Architecture](#fastvlm-architecture)
   - [FastViTHD (Vision Encoder)](#fastvithd-vision-encoder)
   - [Multi-Modal Projector](#multi-modal-projector)
   - [Qwen2 (Language Model)](#qwen2-language-model)
6. [Core ML vs MLX](#core-ml-vs-mlx)
7. [Implementation Details](#implementation-details)
   - [Model Configuration](#model-configuration)
   - [Qwen2 Architecture (Swift)](#qwen2-architecture-swift)
   - [BPE Tokenizer](#bpe-tokenizer)
   - [Weight Loading](#weight-loading)
   - [Inference Pipeline](#inference-pipeline)
8. [The Math Inside the Model](#the-math-inside-the-model)
   - [Embedding Layer](#embedding-layer)
   - [Self-Attention Mechanism](#self-attention-mechanism)
   - [Feed-Forward Network (MLP)](#feed-forward-network-mlp)
   - [RMS Normalization](#rms-normalization)
   - [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
9. [KV-Cache for Efficient Generation](#kv-cache-for-efficient-generation)
10. [File Structure](#file-structure)
11. [Data Flow Summary](#data-flow-summary)
12. [Building and Running](#building-and-running)

---

## Overview

Cortex watches your screen and uses AI to describe your activity in natural language. For example:

> "The user is writing code in VS Code with a Swift file open."

**Key Features:**

- 🔒 **100% On-Device**: No data leaves your Mac
- ⚡ **Fast Inference**: Uses Apple Silicon GPU via MLX
- 📊 **Activity Logging**: Maintains a timestamped history
- 🖼️ **Vision-Language Model**: Understands both images and text

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Cortex App                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐ │
│   │   Screen     │───▶│  FastViTHD   │───▶│   Multi-Modal Projector  │ │
│   │   Capture    │    │  (Core ML)   │    │        (MLX)             │ │
│   │  1024×1024   │    │              │    │                          │ │
│   └──────────────┘    └──────────────┘    └────────────┬─────────────┘ │
│                              │                          │               │
│                              │ [1, 256, 3072]           │ [1, 256, 896] │
│                              ▼                          ▼               │
│   ┌──────────────┐    ┌──────────────────────────────────────────────┐ │
│   │    Prompt    │───▶│              Qwen2 LLM (MLX)                 │ │
│   │  + Tokenizer │    │           24 Transformer Layers              │ │
│   └──────────────┘    └──────────────────────────────────────────────┘ │
│                                          │                              │
│                                          ▼                              │
│                              ┌──────────────────┐                       │
│                              │   Generated Text │                       │
│                              │  "User is coding │                       │
│                              │   in VS Code..." │                       │
│                              └──────────────────┘                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Foundational Concepts

### What is a Neural Network?

A neural network is a mathematical function that transforms input data into output data through a series of calculations. At its core, it's just **matrix multiplication** and simple mathematical functions applied repeatedly.

**The basic unit—a neuron:**

```
                    ┌─────────────────┐
    x₁ ──(w₁)──────▶│                 │
    x₂ ──(w₂)──────▶│  Σ(wᵢ·xᵢ) + b  │──▶ activation(result) ──▶ output
    x₃ ──(w₃)──────▶│                 │
                    └─────────────────┘
```

- **Weights (w)**: Numbers that determine how important each input is
- **Bias (b)**: A number added to shift the result
- **Activation function**: A mathematical function (like ReLU: `max(0, x)`) that introduces non-linearity

When you stack thousands of these neurons in layers, you get a neural network. The "deep" in deep learning means many layers.

### What are Tensors?

A **tensor** is a multi-dimensional array of numbers. Think of it as a generalization of matrices:

| Type   | Dimensions | Example                            |
| ------ | ---------- | ---------------------------------- |
| Scalar | 0D         | `5`                                |
| Vector | 1D         | `[1, 2, 3]`                        |
| Matrix | 2D         | `[[1, 2], [3, 4]]`                 |
| Tensor | 3D+        | `[[[1,2], [3,4]], [[5,6], [7,8]]]` |

Neural networks operate on tensors:

- An image is a 3D tensor: `[height, width, color_channels]`
- A batch of images is 4D: `[batch_size, height, width, channels]`

### What are Model Weights?

When a neural network is "trained," it learns the optimal values for all its weights and biases. **These learned numbers ARE the model.** A 500MB model file is essentially 500MB worth of floating-point numbers.

- **Training** (not done in this app): Feed data through the network, compare output to desired output, adjust weights slightly, repeat billions of times.
- **Inference** (what this app does): Load pre-trained weights, feed new data through, get output.

---

## The Two Types of Models

### Vision Models (Image → Numbers)

A vision model takes an image (a 3D tensor of pixel values) and outputs a **representation** of what's in the image. The output is a tensor of "features"—abstract numerical representations that encode visual content.

```
Input:  [1, 3, 1024, 1024]  →  1 image, 3 color channels (RGB), 1024×1024 pixels
Output: [1, 256, 3072]      →  1 batch, 256 "tokens" (patches), 3072-dim feature per token
```

The 256 tokens represent different regions of the image. Each region is described by 3072 numbers that encode what's visually present there.

### Language Models (Numbers → Text)

A language model takes a sequence of numbers (token IDs) and **predicts the next number** (next token). By doing this repeatedly, it generates text.

**Tokenization**: Text is converted to numbers using a vocabulary.

- `"Hello"` → `[15496]`
- Each word or word-piece has a unique ID

```
Input:  [1, 50]         →  1 batch, 50 token IDs representing the input text
Output: [1, 50, 151936] →  For each position, a probability over 151,936 possible next tokens
```

The model outputs "logits" (raw scores) for every possible next token. The highest score indicates the most likely next word.

### Vision-Language Models (VLMs)

A VLM combines both: it takes an image AND text, then generates text describing the image.

**How they connect**: The vision model's output (image features) gets injected into the language model as if it were text. The language model "reads" the image features like it reads words.

The **projector** is a small neural network that converts vision features (3072-dim) to match the language model's expected dimension (896-dim).

---

## FastVLM Architecture

FastVLM is a specific VLM architecture from Apple. Here's its structure:

### FastViTHD (Vision Encoder)

**Architecture**: A Vision Transformer (ViT) variant optimized for speed

**What it does:**

1. Takes a 1024×1024 image
2. Splits it into patches (small squares)
3. Processes each patch through transformer layers
4. Outputs feature vectors for each patch

**Output shape**: `[1, 256, 3072]`

- 256 patches (image regions)
- 3072 floating-point numbers describing each patch

### Multi-Modal Projector

**Architecture**: Two linear layers with GELU activation

**What it does**: Translates vision features into the language model's "language" (same dimensionality)

```swift
public class MultiModalProjector: Module {
    @ModuleInfo var linear1: Linear  // 3072 → 896
    @ModuleInfo var linear2: Linear  // 896 → 896

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear2(gelu(linear1(x)))  // Linear → GELU → Linear
    }
}
```

### Qwen2 (Language Model)

**Architecture**: A transformer decoder with 24 layers

**Key components per layer:**

- **Self-Attention**: Each token looks at all other tokens to understand context
- **Feed-Forward Network (MLP)**: Processes each token independently
- **Layer Normalization**: Stabilizes the values

**Specifications from config.json:**

| Parameter             | Value   | Description                           |
| --------------------- | ------- | ------------------------------------- |
| `hidden_size`         | 896     | Dimension of internal representations |
| `num_hidden_layers`   | 24      | Number of transformer layers          |
| `num_attention_heads` | 14      | Parallel attention computations       |
| `num_key_value_heads` | 2       | Grouped-Query Attention (GQA)         |
| `intermediate_size`   | 4864    | Hidden dimension in feed-forward      |
| `vocab_size`          | 151936  | Total tokens the model knows          |
| `rope_theta`          | 1000000 | RoPE base frequency                   |

---

## Core ML vs MLX

### Core ML

**What it is**: Apple's framework for running ML models on Apple devices (iPhone, Mac, iPad).

**File format**: `.mlpackage` or `.mlmodel`

**How it works:**

1. You provide a Core ML model file
2. Core ML compiles it for the specific hardware (CPU/GPU/Neural Engine)
3. You call prediction APIs, Core ML handles execution

**Advantages:**

- Extremely optimized for Apple hardware
- Uses the Neural Engine (dedicated ML accelerator)
- Easy to integrate into Swift apps

**In this project**: The vision encoder (`fastvithd.mlpackage`) runs on Core ML.

### MLX

**What it is**: Apple's new ML framework, similar to PyTorch/JAX, designed for Apple Silicon.

**File format**: `.safetensors` (same as Hugging Face models)

**How it works:**

1. You define the model architecture in code (Swift or Python)
2. You load weights from safetensors files
3. You call the model directly, MLX executes on GPU

**Advantages:**

- Full control over model architecture
- Can run any model if you implement the architecture
- Unified memory (CPU and GPU share memory on Apple Silicon)

**In this project**: The language model (Qwen2) runs on MLX because:

- It's a 24-layer transformer—complex to convert to Core ML
- We need fine-grained control over token generation
- MLX's Swift bindings allow native integration

---

## Implementation Details

### Model Configuration

The model's architecture is defined in `config.json`:

```json
{
  "architectures": ["LlavaQwen2ForCausalLM"],
  "hidden_size": 896,
  "num_hidden_layers": 24,
  "intermediate_size": 4864,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-6,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "mm_hidden_size": 3072,
  "image_token_index": 151646,
  "hidden_act": "silu",
  "model_type": "llava_qwen2"
}
```

This is loaded into a Swift struct:

```swift
public struct Qwen2Config: Codable {
    public let hiddenSize: Int         // 896
    public let numHiddenLayers: Int    // 24
    public let intermediateSize: Int   // 4864
    public let numAttentionHeads: Int  // 14
    public let numKeyValueHeads: Int   // 2
    public let rmsNormEps: Float       // 1e-06
    public let vocabSize: Int          // 151936
    public let ropeTheta: Float        // 1000000
    public let mmHiddenSize: Int       // 3072
    public let imageTokenIndex: Int    // 151646

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        // ... maps JSON keys to Swift property names
    }
}
```

### Qwen2 Architecture (Swift)

The complete model is implemented in `Qwen2.swift`:

#### Attention Layer

```swift
class Qwen2Attention: Module {
    let hiddenSize: Int      // 896
    let numHeads: Int        // 14
    let headDim: Int         // 896 / 14 = 64
    let numKeyValues: Int    // 2 (for GQA)
    let scale: Float         // 1/√64 = 0.125

    @ModuleInfo var rope: RoPE        // Rotary Position Embedding
    @ModuleInfo var qProj: Linear     // Query projection: 896 → 896
    @ModuleInfo var kProj: Linear     // Key projection: 896 → 128
    @ModuleInfo var vProj: Linear     // Value projection: 896 → 128
    @ModuleInfo var oProj: Linear     // Output projection: 896 → 896

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let (B, L, _) = x.shape3  // Batch, Sequence Length

        // Project to Q, K, V
        var q = qProj(x)  // [B, L, 896]
        var k = kProj(x)  // [B, L, 128]
        var v = vProj(x)  // [B, L, 128]

        // Reshape for multi-head attention
        q = q.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([B, L, numKeyValues, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([B, L, numKeyValues, headDim]).transposed(0, 2, 1, 3)

        // Apply Rotary Position Embedding
        let offset = cache?.length ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        // Update KV cache
        if let cache = cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        // Expand K, V for Grouped Query Attention
        let repeatCount = numHeads / numKeyValues  // 14 / 2 = 7
        if repeatCount > 1 {
            k = MLX.repeated(k, count: repeatCount, axis: 1)
            v = MLX.repeated(v, count: repeatCount, axis: 1)
        }

        // Scaled dot-product attention: softmax(Q·K^T / √d) · V
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let mask = mask {
            scores = scores + mask  // Apply causal mask
        }
        let probs = softmax(scores, axis: -1)
        let output = matmul(probs, v)

        // Reshape and project output
        return oProj(output.transposed(0, 2, 1, 3).reshaped([B, L, hiddenSize]))
    }
}
```

#### Feed-Forward Network (MLP)

```swift
class Qwen2MLP: Module {
    @ModuleInfo var gateProj: Linear  // 896 → 4864
    @ModuleInfo var upProj: Linear    // 896 → 4864
    @ModuleInfo var downProj: Linear  // 4864 → 896

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let up = upProj(x)
        return downProj(silu(gate) * up)  // SiLU gating
    }
}
```

#### Decoder Layer

```swift
public class Qwen2DecoderLayer: Module {
    @ModuleInfo var selfAttn: Qwen2Attention
    @ModuleInfo var mlp: Qwen2MLP
    @ModuleInfo var inputLayernorm: RMSNorm
    @ModuleInfo var postAttentionLayernorm: RMSNorm

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        // Pre-norm attention with residual
        let r = selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        let h = x + r

        // Pre-norm MLP with residual
        let r2 = mlp(postAttentionLayernorm(h))
        return h + r2
    }
}
```

#### Full Model

```swift
public class Qwen2Model: Module {
    @ModuleInfo public var embedTokens: Embedding     // 151936 tokens → 896 dims
    @ModuleInfo public var layers: [Qwen2DecoderLayer]  // 24 layers
    @ModuleInfo public var norm: RMSNorm              // Final normalization
    @ModuleInfo public var lmHead: Linear             // 896 → 151936 (logits)
    @ModuleInfo public var projector: MultiModalProjector

    public var caches: [KVCache] = []  // One cache per layer

    public func prefill(inputIds: MLXArray, imageEmbeddings: MLXArray) -> MLXArray {
        // 1. Embed text tokens
        let inputsEmbeds = embedTokens(inputIds)

        // 2. Project image embeddings to text space
        let projectedImages = projector(imageEmbeddings)

        // 3. Concatenate: [image tokens] + [text tokens]
        let combinedEmbeds = concatenated([projectedImages, inputsEmbeds], axis: 1)

        // 4. Create causal attention mask
        let seqLen = combinedEmbeds.dim(1)
        let mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(seqLen)

        // 5. Run through all 24 layers
        var hiddenStates = combinedEmbeds
        for (layer, cache) in zip(layers, caches) {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache)
        }

        // 6. Final norm and project to vocabulary
        hiddenStates = norm(hiddenStates)
        return lmHead(hiddenStates)  // [1, seqLen, 151936]
    }

    public func generateNext(tokenId: Int) -> MLXArray {
        // Process single token using cached KV (fast!)
        let inputTensor = MLXArray([Int32(tokenId)]).expandedDimensions(axis: 0)
        let inputsEmbeds = embedTokens(inputTensor)

        var hiddenStates = inputsEmbeds
        for (layer, cache) in zip(layers, caches) {
            hiddenStates = layer(hiddenStates, mask: nil, cache: cache)
        }

        hiddenStates = norm(hiddenStates)
        return lmHead(hiddenStates)  // [1, 1, 151936]
    }
}
```

### BPE Tokenizer

The tokenizer converts text to token IDs and back. Implemented in `BPETokenizer.swift`:

#### Byte Pair Encoding Algorithm

1. Start with individual characters: `"hello"` → `["h", "e", "l", "l", "o"]`
2. Repeatedly merge the most common pair according to learned rules
3. "Merges" are pre-computed during training and stored in `tokenizer.json`

```swift
public class BPETokenizer {
    private var vocab: [String: Int] = [:]       // Token → ID
    private var reverseVocab: [Int: String] = [] // ID → Token
    private var mergeRanks: [String: Int] = [:]  // "pair" → priority

    public let eosTokenId: Int    // 151645 (<|im_end|>)
    public let imageTokenId: Int  // 151646 (<image>)

    /// Apply BPE merges to a word
    private func bpe(_ word: String) -> [String] {
        var tokens = word.map { String($0) }  // Start with characters

        while tokens.count > 1 {
            // Find highest-priority pair to merge
            var bestPair: (Int, String, String)? = nil
            var bestRank = Int.max

            for i in 0..<(tokens.count - 1) {
                let pair = "\(tokens[i]) \(tokens[i+1])"
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = (i, tokens[i], tokens[i+1])
                }
            }

            guard let (index, first, second) = bestPair else { break }

            // Apply merge
            tokens[index] = first + second
            tokens.remove(at: index + 1)
        }

        return tokens
    }

    /// Decode token ID to string
    public func decode(tokenId: Int) -> String {
        guard let token = reverseVocab[tokenId] else { return "" }
        return decodeToken(token)  // Convert GPT-2 byte encoding to UTF-8
    }
}
```

### Weight Loading

Weights are stored in `.safetensors` format. The loading process involves name mapping:

```swift
private func mapWeightNames(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]

    for (key, value) in weights {
        var newKey = key

        // Remove prefix: "language_model.model.layers.0..." → "layers.0..."
        if newKey.hasPrefix("language_model.model.") {
            newKey = String(newKey.dropFirst(21))
        }

        // Handle projector: "multi_modal_projector.linear_0" → "projector.linear1"
        if newKey.hasPrefix("multi_modal_projector.") {
            newKey = "projector." + newKey.dropFirst(22)
                .replacingOccurrences(of: "linear_0", with: "linear1")
                .replacingOccurrences(of: "linear_2", with: "linear2")
        }

        // Convert snake_case to camelCase
        newKey = newKey
            .replacingOccurrences(of: "embed_tokens", with: "embedTokens")
            .replacingOccurrences(of: "lm_head", with: "lmHead")
            .replacingOccurrences(of: "self_attn", with: "selfAttn")
            .replacingOccurrences(of: "q_proj", with: "qProj")
            // ... etc

        mapped[newKey] = value
    }
    return mapped
}
```

### Inference Pipeline

The complete flow is orchestrated in `ViewModel.swift`:

```swift
@MainActor
final class ViewModel: ObservableObject {
    @Published var resultText = ""
    @Published var activityLog: [ActivityEntry] = []
    @Published var isRunning = false

    private var visionModel: fastvithd?  // Core ML vision encoder
    private var llmEngine = LLMEngine()   // MLX language model

    func toggleAnalysis() {
        if isRunning {
            captureTask?.cancel()
            isRunning = false
        } else {
            isRunning = true
            captureTask = Task {
                while !Task.isCancelled {
                    await runOneInference()
                    try? await Task.sleep(nanoseconds: 10_000_000_000)  // 10 seconds
                }
            }
        }
    }

    private func runOneInference() async {
        // 1. Capture screen
        guard let cgImage = captureMainDisplay() else { return }

        // 2. Prepare input for vision model
        let inputArray = try createInputArray(from: cgImage)  // [1, 3, 1024, 1024]

        // 3. Run vision model (Core ML)
        let prediction = try visionModel.prediction(images: inputArray)
        // prediction.image_features: [1, 256, 3072]

        // 4. Create prompt with image token
        let prompt = """
        <|im_start|>user
        <image>
        Briefly describe what the user is doing on their computer in one sentence.
        <|im_end|>
        <|im_start|>assistant
        """

        // 5. Run LLM (MLX)
        let text = await llmEngine.generate(
            imageFeatures: prediction.image_features,
            prompt: prompt,
            maxTokens: 50
        )

        // 6. Update UI
        resultText = text
        activityLog.insert(ActivityEntry(timestamp: Date(), description: text), at: 0)
    }
}
```

---

## The Math Inside the Model

### Embedding Layer

Converts token IDs to vectors:

```
token_id → embedding_matrix[token_id] → vector of size 896
```

The embedding matrix is a learned lookup table with shape `[151936, 896]`.

### Self-Attention Mechanism

For each token position, compute:

```
Q = X · W_q  (Query: "What am I looking for?")
K = X · W_k  (Key: "What do I contain?")
V = X · W_v  (Value: "What information do I have?")

Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

This lets each token "attend to" relevant other tokens.

### Feed-Forward Network (MLP)

For each token independently:

```
gate = SiLU(X · W_gate)
up = X · W_up
output = (gate * up) · W_down
```

**SiLU (Sigmoid Linear Unit)**: `x * sigmoid(x)` — a smooth activation function.

### RMS Normalization

Stabilizes values to prevent explosion/vanishing:

```
RMSNorm(x) = x / √(mean(x²) + ε) * γ
```

Where `ε = 1e-6` and `γ` is a learned scale parameter.

### Rotary Position Embedding (RoPE)

Encodes position information by rotating vectors:

```
q_rotated = q * cos(θ·pos) + rotate_half(q) * sin(θ·pos)
k_rotated = k * cos(θ·pos) + rotate_half(k) * sin(θ·pos)
```

Where `θ = 1000000^(-2i/d)` for dimension `i`.

This lets the model understand word order without explicit position tokens.

---

## KV-Cache for Efficient Generation

Without caching, generating N tokens requires O(N²) computation (each new token attends to all previous tokens, and we'd recompute everything).

**With KV-Cache**: We cache the Key and Value tensors from previous tokens.

```swift
public class KVCache {
    var keys: MLXArray?    // [B, numHeads, cachedLen, headDim]
    var values: MLXArray?  // [B, numHeads, cachedLen, headDim]

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys {
            // Append new to cached
            keys = concatenated([existingKeys, newKeys], axis: 2)
            values = concatenated([existingValues, newValues], axis: 2)
        } else {
            // First time
            keys = newKeys
            values = newValues
        }
        return (keys!, values!)
    }
}
```

**Result**: Each new token only computes its own Q, K, V and attends to the cached K, V. This is **O(N)** per token instead of **O(N²)**.

---

## File Structure

| File                  | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `fastvithd.mlpackage` | Vision encoder (Core ML) — processes images          |
| `model.safetensors`   | LLM weights (MLX) — ~1.2GB of floating-point numbers |
| `config.json`         | Architecture parameters (layers, dimensions, etc.)   |
| `tokenizer.json`      | Text ↔ number mapping (vocabulary + merge rules)     |
| `Qwen2.swift`         | LLM architecture definition in Swift                 |
| `BPETokenizer.swift`  | Tokenizer implementation in Swift                    |
| `LLMEngine.swift`     | Loads weights, runs generation loop                  |
| `ViewModel.swift`     | Orchestrates screen capture + inference + UI         |
| `ContentView.swift`   | SwiftUI interface                                    |

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            COMPLETE DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. SCREEN CAPTURE                                                           │
│     CGDisplayCreateImage() → CGImage (whatever resolution)                   │
│                                   │                                          │
│                                   ▼                                          │
│  2. IMAGE PREPROCESSING                                                      │
│     Resize to 1024×1024, normalize to [0,1] float                            │
│     → MLMultiArray [1, 3, 1024, 1024]                                        │
│                                   │                                          │
│                                   ▼                                          │
│  3. VISION ENCODER (Core ML - FastViTHD)                                     │
│     Patch embedding → Transformer layers → Feature extraction                │
│     → MLMultiArray [1, 256, 3072]                                            │
│     (256 image patches, each with 3072-dim feature vector)                   │
│                                   │                                          │
│                                   ▼                                          │
│  4. MULTI-MODAL PROJECTOR (MLX)                                              │
│     Linear(3072→896) → GELU → Linear(896→896)                                │
│     → MLXArray [1, 256, 896]                                                 │
│     (Image features now match LLM hidden dimension)                          │
│                                   │                                          │
│                                   ▼                                          │
│  5. TOKENIZATION                                                             │
│     "Describe what..." → BPE → [token_ids]                                   │
│     → MLXArray [1, N] where N = number of tokens                             │
│                                   │                                          │
│                                   ▼                                          │
│  6. EMBEDDING                                                                │
│     token_ids → Embedding lookup                                             │
│     → MLXArray [1, N, 896]                                                   │
│                                   │                                          │
│                                   ▼                                          │
│  7. CONCATENATION                                                            │
│     [image_embeds] + [text_embeds]                                           │
│     → MLXArray [1, 256+N, 896]                                               │
│                                   │                                          │
│                                   ▼                                          │
│  8. TRANSFORMER (24 layers, each):                                           │
│     ┌─────────────────────────────────────────────────────────────┐          │
│     │  RMSNorm → Self-Attention (with KV-Cache) → Add Residual   │          │
│     │  RMSNorm → MLP (gate·up·down) → Add Residual               │          │
│     └─────────────────────────────────────────────────────────────┘          │
│     → MLXArray [1, 256+N, 896]                                               │
│                                   │                                          │
│                                   ▼                                          │
│  9. OUTPUT PROJECTION                                                        │
│     RMSNorm → Linear(896→151936)                                             │
│     → MLXArray [1, 256+N, 151936] (logits for each vocab token)              │
│                                   │                                          │
│                                   ▼                                          │
│  10. SAMPLING                                                                │
│      Take logits[-1] (last position)                                         │
│      Apply repetition penalty                                                │
│      argmax → next_token_id                                                  │
│                                   │                                          │
│                                   ▼                                          │
│  11. DECODE                                                                  │
│      token_id → BPE decode → "The"                                           │
│      Repeat steps 9-11 until EOS or max_tokens                               │
│                                   │                                          │
│                                   ▼                                          │
│  12. OUTPUT                                                                  │
│      "The user is writing code in VS Code with a Swift file open."           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Building and Running

### Requirements

- macOS 14.0+ (Sonoma or later)
- Xcode 15+
- Apple Silicon Mac (M1/M2/M3) recommended

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/cortex.git
   cd cortex
   ```

2. **Add model files to Resources**

   Copy these files from `exported-fastvlm-0.5b/` to `macOSApp/Cortex/Cortex/` (added to Xcode project):
   - `model.safetensors`
   - `config.json`
   - `tokenizer.json`
   - `fastvithd.mlpackage`

3. **Open in Xcode**

   ```bash
   open macOSApp/Cortex/Cortex.xcodeproj
   ```

4. **Build and Run**
   - Select your Mac as the target device
   - Press ⌘R to build and run
   - Grant screen recording permission when prompted

### Usage

1. Click **"Start Analysis"** to begin
2. The app captures your screen every 10 seconds
3. Watch as it describes your activity in natural language
4. Click **"Stop Analysis"** to pause

---

## License

This project uses:

- **FastVLM** model weights from Apple (see LICENSE_MODEL)
- **MLX** framework from Apple (Apache 2.0)
- **LLaVA** architecture concepts (Apache 2.0)

---

## Acknowledgments

- Apple ML Research for FastVLM and MLX
- Qwen team for the Qwen2 language model architecture
- LLaVA team for the vision-language model design patterns
