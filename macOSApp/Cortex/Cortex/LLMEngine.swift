//
//  LLMEngine.swift
//  Cortex
//
//  This file manages the Large Language Model (LLM) part of the app.
//  It loads a Qwen2-based model using MLX (Apple's ML framework for efficiency).
//  The LLM generates text descriptions from image features and prompts.
//  It handles model loading, tokenization, and text generation in a loop.
//

import Foundation  // Basic types and utilities
import MLX  // MLX framework for machine learning on Apple devices
import MLXNN  // Neural network components (layers, etc.)
import MLXRandom  // Random number utilities
import CoreML  // Framework for running ML models (used for data conversion)

class LLMEngine {  // A class (reference type) that encapsulates LLM functionality
    var model: Qwen2Model?  // Optional property: the loaded model (nil if not loaded yet)
    var tokenizer: BPETokenizer?  // BPE tokenizer for text encoding/decoding
    var weights: [String: MLXArray] = [:]  // Dictionary of model weights (key: name, value: array of numbers)
    
    let modelType = "llava_qwen2"  // Constant string: type of model (hardcoded for this app)
    let imageTokenIndex = 151646  // Constant: special token ID representing an image in text
    
    init() {}  // Empty initializer: no setup needed initially

    func load(modelPath: URL) async throws {  // Async function (runs in background) that loads the model. 'throws' means it can fail and throw errors.
        print("Loading LLM from \(modelPath.path)")  // Prints debug message to console
        
        // Load model weights from safetensors files
        self.weights = try loadSafetensors(from: modelPath)
        
        // Load configuration from JSON
        let config = try JSONDecoder().decode(Qwen2Config.self, from: Data(contentsOf: modelPath.appendingPathComponent("config.json")))
        
        // Initialize the model architecture
        self.model = Qwen2Model(config: config)
        
        // Apply weights to model - map weight names to model structure
        let mappedWeights = mapWeightNames(weights)
        
        // Debug: Check for expected top-level keys
        let topLevelKeys = Set(mappedWeights.keys.compactMap { $0.split(separator: ".").first.map(String.init) })
        print("Top-level weight keys: \(topLevelKeys)")
        
        // Use MLX's quantize/load approach for better compatibility
        do {
            try self.model?.update(parameters: ModuleParameters.unflattened(mappedWeights), verify: .noUnusedKeys)
        } catch {
            print("Weight loading error: \(error)")
            // Try with no verification as fallback
            try self.model?.update(parameters: ModuleParameters.unflattened(mappedWeights), verify: .none)
        }
        
        // Load the BPE tokenizer
        self.tokenizer = try BPETokenizer(modelPath: modelPath)
        
        print("LLM Loaded Successfully")
    }

    func generate(imageFeatures: MLMultiArray, prompt: String, maxTokens: Int = 50) async -> String {
        guard let model = model, let tokenizer = tokenizer else { return "Model not loaded" }
        
        let featuresShape = [1, 256, 3072]  // Shape of image features from vision encoder
        let featuresCount = 1 * 256 * 3072
        
        // Convert CoreML MLMultiArray to MLX array
        let featuresPointer = UnsafeBufferPointer<Float>(
            start: imageFeatures.dataPointer.bindMemory(to: Float.self, capacity: featuresCount),
            count: featuresCount
        )
        let floatArray = Array(featuresPointer)
        let imageEmbeddings = MLXArray(floatArray, featuresShape)
        
        // Encode the prompt
        let promptTokenIds = tokenizer.encode(text: prompt)
        let inputTensor = MLXArray(promptTokenIds.map { Int32($0) }).expandedDimensions(axis: 0)
        
        // Reset caches for new generation - create fresh cache instances
        model.resetCaches()
        
        // PREFILL: Process image + prompt, populate KV caches
        let prefillLogits = model.prefill(inputIds: inputTensor, imageEmbeddings: imageEmbeddings)
        
        // Force evaluation to ensure computation is done
        eval(prefillLogits)
        
        // Get first generated token from prefill output
        var lastLogits = prefillLogits[0, -1]
        var nextTokenId = sampleWithRepetitionPenalty(logits: lastLogits, generatedIds: [])
        
        var generatedText = ""
        var generatedIds: [Int] = []  // Track generated tokens for repetition penalty
        
        // DECODE: Generate tokens one at a time using cached KV (fast!)
        for _ in 0..<maxTokens {  // Limit to maxTokens
            if nextTokenId == tokenizer.eosTokenId { break }
            
            // Stop on common sentence endings after minimum length
            let decoded = tokenizer.decode(tokenId: nextTokenId)
            generatedText += decoded
            generatedIds.append(nextTokenId)
            
            // Stop if we hit a natural ending point after some content
            if generatedIds.count >= 15 && (decoded.contains(".") || decoded.contains("!") || decoded.contains("?")) {
                break
            }
            
            // Generate next token using cache
            let logits = model.generateNext(tokenId: nextTokenId)
            eval(logits)
            
            lastLogits = logits[0, 0]
            nextTokenId = sampleWithRepetitionPenalty(logits: lastLogits, generatedIds: generatedIds)
        }
        
        return generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    /// Sample next token with repetition penalty to avoid loops
    private func sampleWithRepetitionPenalty(logits: MLXArray, generatedIds: [Int], penalty: Float = 1.2) -> Int {
        var logitsArray = logits.asArray(Float.self)  // Convert to Swift array
        
        // Apply repetition penalty to already-generated tokens
        for id in generatedIds {
            if id < logitsArray.count {
                if logitsArray[id] > 0 {
                    logitsArray[id] /= penalty
                } else {
                    logitsArray[id] *= penalty
                }
            }
        }
        
        // Find argmax
        var maxIdx = 0
        var maxVal = logitsArray[0]
        for (idx, val) in logitsArray.enumerated() {
            if val > maxVal {
                maxVal = val
                maxIdx = idx
            }
        }
        return maxIdx
    }
    
    // MARK: - Safetensors Loading
    
    /// Load weights from safetensors files in the model directory
    private func loadSafetensors(from modelPath: URL) throws -> [String: MLXArray] {
        var allWeights: [String: MLXArray] = [:]
        
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: modelPath, includingPropertiesForKeys: nil)
        
        // Find all safetensors files
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }
        
        if safetensorFiles.isEmpty {
            throw LLMEngineError.noWeightsFound
        }
        
        // Load each safetensors file
        for file in safetensorFiles {
            let weights = try loadArrays(url: file)
            for (key, value) in weights {
                allWeights[key] = value
            }
        }
        
        print("Loaded \(allWeights.count) weight tensors from \(safetensorFiles.count) files")
        return allWeights
    }
    
    /// Map weight names from the safetensors format to MLX model structure
    private func mapWeightNames(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        
        for (key, value) in weights {
            var newKey = key
            
            // Handle language_model prefix (non-quantized format)
            if newKey.hasPrefix("language_model.model.") {
                newKey = String(newKey.dropFirst(21))  // Remove "language_model.model."
            } else if newKey.hasPrefix("language_model.") {
                newKey = String(newKey.dropFirst(15))  // Remove "language_model." (for lm_head)
            }
            
            // Handle multi_modal_projector -> projector
            if newKey.hasPrefix("multi_modal_projector.") {
                newKey = String(newKey.dropFirst(22))  // Remove "multi_modal_projector."
                // Map linear_0 -> linear1, linear_2 -> linear2
                newKey = "projector." + newKey
                    .replacingOccurrences(of: "linear_0", with: "linear1")
                    .replacingOccurrences(of: "linear_2", with: "linear2")
            }
            
            // Remove old "model." prefix if still present
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst(6))
            }
            
            // Map layer names to Swift naming convention
            newKey = newKey
                .replacingOccurrences(of: "embed_tokens", with: "embedTokens")
                .replacingOccurrences(of: "lm_head", with: "lmHead")
                .replacingOccurrences(of: "self_attn", with: "selfAttn")
                .replacingOccurrences(of: "input_layernorm", with: "inputLayernorm")
                .replacingOccurrences(of: "post_attention_layernorm", with: "postAttentionLayernorm")
                .replacingOccurrences(of: "q_proj", with: "qProj")
                .replacingOccurrences(of: "k_proj", with: "kProj")
                .replacingOccurrences(of: "v_proj", with: "vProj")
                .replacingOccurrences(of: "o_proj", with: "oProj")
                .replacingOccurrences(of: "gate_proj", with: "gateProj")
                .replacingOccurrences(of: "up_proj", with: "upProj")
                .replacingOccurrences(of: "down_proj", with: "downProj")
            
            mapped[newKey] = value
        }
        
        // Debug: print a few mapped keys
        print("Sample mapped keys:")
        for (key, _) in mapped.prefix(10) {
            print("  \(key)")
        }
        
        return mapped
    }
}

// MARK: - Errors

enum LLMEngineError: Error {
    case noWeightsFound
    case configNotFound
    case tokenizerNotFound
}

