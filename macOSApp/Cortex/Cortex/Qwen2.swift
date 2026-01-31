//
//  Qwen2.swift
//  Cortex
//
//  Created by Tanish Pradhan Wong Ah Sui on 10/11/25.
//
//  This file defines the Qwen2 model architecture, a multimodal transformer-based LLM.
//  It includes layers for attention, feed-forward networks, and image projection.
//  The model processes text tokens and image embeddings to generate output probabilities.
//  Uses MLX for efficient computation on Apple devices.
//

import Foundation  // Basic types
import MLX  // ML framework
import MLXNN  // Neural network layers

// MARK: - Configuration
// This section defines the model configuration, loaded from JSON.

public struct Qwen2Config: Codable {  // Struct (value type) that can be encoded/decoded from JSON
    public let hiddenSize: Int  // Number of neurons in hidden layers
    public let numHiddenLayers: Int  // Number of transformer layers
    public let intermediateSize: Int  // Size of intermediate layer in MLP
    public let numAttentionHeads: Int  // Number of attention heads
    public let numKeyValueHeads: Int  // Number of key/value heads (for GQA)
    public let rmsNormEps: Float  // Epsilon for RMS normalization
    public let vocabSize: Int  // Size of vocabulary (number of tokens)
    public let ropeTheta: Float  // Theta for RoPE (positional encoding)
    public let mmHiddenSize: Int  // Hidden size for multimodal projector
    public let imageTokenIndex: Int  // Token ID for images
    
    enum CodingKeys: String, CodingKey {  // Maps JSON keys to property names
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case mmHiddenSize = "mm_hidden_size"
        case imageTokenIndex = "image_token_index"
    }
}

// MARK: - KV Cache
// Cache structure for storing key/value tensors across generation steps

public class KVCache {
    var keys: MLXArray?    // Cached keys [B, numKVHeads, seqLen, headDim]
    var values: MLXArray?  // Cached values [B, numKVHeads, seqLen, headDim]
    
    public init() {}
    
    var length: Int {
        keys?.dim(2) ?? 0  // Return cached sequence length
    }
    
    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            // Append new keys/values to existing cache
            let updatedKeys = concatenated([existingKeys, newKeys], axis: 2)
            let updatedValues = concatenated([existingValues, newValues], axis: 2)
            self.keys = updatedKeys
            self.values = updatedValues
            return (updatedKeys, updatedValues)
        } else {
            // First time - just store
            self.keys = newKeys
            self.values = newValues
            return (newKeys, newValues)
        }
    }
    
    func reset() {
        keys = nil
        values = nil
    }
}

// MARK: - Layers
// This section defines the neural network layers.

class Qwen2Attention: Module {  // Class inheriting from Module (MLX base for layers)
    let hiddenSize: Int  // Size of hidden state
    let numHeads: Int  // Number of attention heads
    let headDim: Int  // Dimension per head
    let numKeyValues: Int  // Number of key/value heads
    let scale: Float  // Scaling factor for attention
    @ModuleInfo var rope: RoPE  // Rotatory Position Embedding
    
    @ModuleInfo var qProj: Linear  // Linear layer for query projection
    @ModuleInfo var kProj: Linear  // For key
    @ModuleInfo var vProj: Linear  // For value
    @ModuleInfo var oProj: Linear  // Output projection
    
    init(config: Qwen2Config) {  // Initializer with config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKeyValues = config.numKeyValueHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)  // Inverse sqrt for scaling
        self.rope = RoPE(dimensions: headDim, base: config.ropeTheta)  // RoPE for positions
        
        self.qProj = Linear(config.hiddenSize, config.numAttentionHeads * headDim, bias: true)  // Query projection
        self.kProj = Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: true)  // Key
        self.vProj = Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: true)  // Value
        self.oProj = Linear(config.numAttentionHeads * headDim, config.hiddenSize, bias: false)  // Output
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let (B, L, _) = x.shape3  // Unpack batch size, sequence length
        
        var q = qProj(x)  // Project to queries
        var k = kProj(x)  // Keys
        var v = vProj(x)  // Values
        
        // Reshape for multi-head attention
        q = q.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)  // [B, numHeads, L, headDim]
        k = k.reshaped([B, L, numKeyValues, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([B, L, numKeyValues, headDim]).transposed(0, 2, 1, 3)
        
        // Position offset for RoPE (use cached length if available)
        let offset = cache?.length ?? 0
        
        // Apply RoPE with correct position offset
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        
        // Update cache and get full keys/values
        if let cache = cache {
            (k, v) = cache.update(keys: k, values: v)
        }
        
        // Expand k,v for GQA (grouped query attention) if needed
        let repeatCount = numHeads / numKeyValues
        if repeatCount > 1 {
            k = MLX.repeated(k, count: repeatCount, axis: 1)
            v = MLX.repeated(v, count: repeatCount, axis: 1)
        }
        
        // Scaled dot-product attention
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale  // Q * K^T
        
        if let mask = mask {  // Apply mask if provided
            scores = scores + mask
        }
        
        let probs = softmax(scores, axis: -1)  // Softmax over last axis
        let output = matmul(probs, v)  // Weighted sum with values
        
        return oProj(output.transposed(0, 2, 1, 3).reshaped([B, L, hiddenSize]))  // Reshape and project output
    }
}

class Qwen2MLP: Module {  // Multi-Layer Perceptron (feed-forward network)
    @ModuleInfo var gateProj: Linear  // Gate projection
    @ModuleInfo var upProj: Linear  // Up projection
    @ModuleInfo var downProj: Linear  // Down projection
    
    init(config: Qwen2Config) {
        self.gateProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)  // Projects to intermediate size
        self.upProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.downProj = Linear(config.intermediateSize, config.hiddenSize, bias: false)  // Back to hidden size
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {  // Forward pass
        let gate = gateProj(x)  // Compute gate
        let up = upProj(x)  // Compute up
        return downProj(silu(gate) * up)  // SiLU activation on gate, multiply with up, then down
    }
}

public class Qwen2DecoderLayer: Module {  // One transformer decoder layer
    @ModuleInfo var selfAttn: Qwen2Attention  // Self-attention
    @ModuleInfo var mlp: Qwen2MLP  // Feed-forward
    @ModuleInfo var inputLayernorm: RMSNorm  // Normalization before attention
    @ModuleInfo var postAttentionLayernorm: RMSNorm  // Before MLP
    
    init(config: Qwen2Config) {
        self.selfAttn = Qwen2Attention(config: config)  // Init attention
        self.mlp = Qwen2MLP(config: config)  // Init MLP
        self.inputLayernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)  // RMS norm
        self.postAttentionLayernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let r = selfAttn(inputLayernorm(x), mask: mask, cache: cache)  // Attention with cache
        let h = x + r  // Residual connection
        let r2 = mlp(postAttentionLayernorm(h))  // MLP on normalized post-attention
        return h + r2  // Another residual
    }
}

public class MultiModalProjector: Module {  // Projects image features to text embedding space
    @ModuleInfo var linear1: Linear  // First linear layer
    @ModuleInfo var linear2: Linear  // Second
    
    init(config: Qwen2Config) {
        self.linear1 = Linear(config.mmHiddenSize, config.hiddenSize, bias: true)  // From image hidden to text hidden
        self.linear2 = Linear(config.hiddenSize, config.hiddenSize, bias: true)  // Refine
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {  // Forward pass
        return linear2(gelu(linear1(x)))  // Linear -> GELU -> Linear
    }
}

// MARK: - Model
// The full Qwen2 model.

public class Qwen2Model: Module {  // Main model class
    @ModuleInfo public var embedTokens: Embedding  // Token embeddings
    @ModuleInfo public var layers: [Qwen2DecoderLayer]  // Stack of decoder layers
    @ModuleInfo public var norm: RMSNorm  // Final normalization
    @ModuleInfo public var lmHead: Linear  // Language model head (output logits)
    @ModuleInfo public var projector: MultiModalProjector  // Image projector
    
    public let config: Qwen2Config  // Model config (not a module)
    public var caches: [KVCache] = []  // KV caches (not trainable)
    
    public init(config: Qwen2Config) {  // Initializer
        self.config = config
        self.embedTokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)  // Token to vector
        self.layers = (0..<config.numHiddenLayers).map { _ in Qwen2DecoderLayer(config: config) }  // Create layers
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)  // Final norm
        self.lmHead = Linear(config.hiddenSize, config.vocabSize, bias: false)  // Output to vocab size
        self.projector = MultiModalProjector(config: config)  // Image projector
        self.caches = (0..<config.numHiddenLayers).map { _ in KVCache() }  // Initialize caches
    }
    
    /// Reset all KV caches - call before starting a new generation
    public func resetCaches() {
        // Create fresh cache instances to ensure complete reset
        self.caches = (0..<config.numHiddenLayers).map { _ in KVCache() }
    }
    
    /// Prefill: Process image embeddings and initial prompt, populate caches
    public func prefill(inputIds: MLXArray, imageEmbeddings: MLXArray) -> MLXArray {
        // 1. Embed tokens
        let inputsEmbeds = embedTokens(inputIds)  // [1, seqLen, hiddenSize]
        
        // 2. Project image embeddings
        let projectedImages = projector(imageEmbeddings)  // [1, 256, hiddenSize]
        
        // 3. Merge: image tokens first, then text tokens
        let combinedEmbeds = concatenated([projectedImages, inputsEmbeds], axis: 1)
        
        // 4. Create causal mask for full sequence
        let seqLen = combinedEmbeds.dim(1)
        let mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(seqLen)
        
        // 5. Run through layers, populating caches
        var hiddenStates = combinedEmbeds
        for (layer, cache) in zip(layers, caches) {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache)
        }
        
        hiddenStates = norm(hiddenStates)
        return lmHead(hiddenStates)  // Return logits for all positions
    }
    
    /// Generate next token: Only process one token, using cached KV
    public func generateNext(tokenId: Int) -> MLXArray {
        // 1. Embed single token
        let inputTensor = MLXArray([Int32(tokenId)]).expandedDimensions(axis: 0)  // [1, 1]
        let inputsEmbeds = embedTokens(inputTensor)  // [1, 1, hiddenSize]
        
        // 2. No mask needed for single token (attends to all cached positions)
        // For single token generation, the causal mask is implicit in the cache structure
        
        // 3. Run through layers using cache
        var hiddenStates = inputsEmbeds
        for (layer, cache) in zip(layers, caches) {
            hiddenStates = layer(hiddenStates, mask: nil, cache: cache)
        }
        
        hiddenStates = norm(hiddenStates)
        return lmHead(hiddenStates)  // [1, 1, vocabSize]
    }
    
    // Original forward pass (kept for compatibility)
    public func callAsFunction(inputIds: MLXArray, imageEmbeddings: MLXArray) -> MLXArray {
        return prefill(inputIds: inputIds, imageEmbeddings: imageEmbeddings)
    }
}
