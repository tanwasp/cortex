//
//  BPETokenizer.swift
//  Cortex
//
//  Implements a Byte Pair Encoding (BPE) tokenizer compatible with HuggingFace's tokenizers.
//  Loads vocabulary and merge rules from tokenizer.json and converts text to token IDs.
//  Supports special tokens like <image>, <|im_start|>, <|im_end|>.
//

import Foundation

// MARK: - Tokenizer JSON Structure

/// Represents the full tokenizer.json file structure
struct HFTokenizerJSON: Codable {
    let version: String?
    let addedTokens: [AddedToken]?
    let model: BPEModel
    
    enum CodingKeys: String, CodingKey {
        case version
        case addedTokens = "added_tokens"
        case model
    }
}

/// Special tokens that were added to the vocabulary
struct AddedToken: Codable {
    let id: Int
    let content: String
    let special: Bool?
}

/// The BPE model containing vocabulary and merges
struct BPEModel: Codable {
    let type: String  // "BPE"
    let vocab: [String: Int]
    let merges: [[String]]
}

// MARK: - BPE Tokenizer

/// A Byte Pair Encoding tokenizer that loads from HuggingFace tokenizer.json
public class BPETokenizer {
    /// Vocabulary mapping token string to ID
    private var vocab: [String: Int] = [:]
    
    /// Reverse vocabulary mapping ID to token string
    private var reverseVocab: [Int: String] = [:]
    
    /// Special tokens map (content -> id)
    private var specialTokens: [String: Int] = [:]
    
    /// BPE merge rules - pairs that should be merged in priority order
    private var merges: [(String, String)] = []
    
    /// Merge rankings - lower rank = higher priority
    private var mergeRanks: [String: Int] = [:]
    
    /// End of sequence token ID
    public let eosTokenId: Int
    
    /// Image placeholder token ID
    public let imageTokenId: Int
    
    /// Byte-to-unicode mapping for GPT-2 style byte encoding
    private static let byteEncoder: [UInt8: Character] = createByteEncoder()
    private static let byteDecoder: [Character: UInt8] = createByteDecoder()
    
    /// Initialize from a model directory containing tokenizer.json
    public init(modelPath: URL) throws {
        let tokenizerURL = modelPath.appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: tokenizerURL)
        let tokenizer = try JSONDecoder().decode(HFTokenizerJSON.self, from: data)
        
        // Load vocabulary
        self.vocab = tokenizer.model.vocab
        for (token, id) in vocab {
            reverseVocab[id] = token
        }
        
        // Load special tokens
        if let addedTokens = tokenizer.addedTokens {
            for token in addedTokens {
                specialTokens[token.content] = token.id
                vocab[token.content] = token.id
                reverseVocab[token.id] = token.content
            }
        }
        
        // Set known special token IDs
        self.eosTokenId = specialTokens["<|im_end|>"] ?? 151645
        self.imageTokenId = specialTokens["<image>"] ?? 151646
        
        // Load merges
        for (index, merge) in tokenizer.model.merges.enumerated() {
            if merge.count == 2 {
                let pair = (merge[0], merge[1])
                merges.append(pair)
                mergeRanks["\(pair.0) \(pair.1)"] = index
            }
        }
        
        print("Tokenizer loaded: \(vocab.count) tokens, \(merges.count) merges, \(specialTokens.count) special tokens")
    }
    
    // MARK: - Encoding
    
    /// Encode text to token IDs
    public func encode(text: String) -> [Int] {
        var tokens: [Int] = []
        var remainingText = text
        
        // First, extract special tokens
        while !remainingText.isEmpty {
            var foundSpecial = false
            
            // Check for special tokens at the start
            for (specialToken, tokenId) in specialTokens {
                if remainingText.hasPrefix(specialToken) {
                    tokens.append(tokenId)
                    remainingText = String(remainingText.dropFirst(specialToken.count))
                    foundSpecial = true
                    break
                }
            }
            
            if !foundSpecial {
                // Find the next special token or end of string
                var nextSpecialIdx = remainingText.count
                for specialToken in specialTokens.keys {
                    if let range = remainingText.range(of: specialToken) {
                        let idx = remainingText.distance(from: remainingText.startIndex, to: range.lowerBound)
                        if idx > 0 && idx < nextSpecialIdx {
                            nextSpecialIdx = idx
                        }
                    }
                }
                
                // Process regular text up to the next special token
                let regularText = String(remainingText.prefix(nextSpecialIdx))
                if !regularText.isEmpty {
                    tokens.append(contentsOf: encodeRegularText(regularText))
                }
                remainingText = String(remainingText.dropFirst(nextSpecialIdx))
            }
        }
        
        return tokens
    }
    
    /// Encode regular text (no special tokens) using BPE
    private func encodeRegularText(_ text: String) -> [Int] {
        var allTokenIds: [Int] = []
        
        // Simple word-level split (GPT-2/Qwen style pre-tokenization)
        // Convert to bytes and then to GPT-2 unicode representation
        let words = preTokenize(text)
        
        for word in words {
            let wordTokens = bpe(word)
            for token in wordTokens {
                if let id = vocab[token] {
                    allTokenIds.append(id)
                }
            }
        }
        
        return allTokenIds
    }
    
    /// Pre-tokenize text into words (GPT-2 style)
    private func preTokenize(_ text: String) -> [String] {
        // Convert text to GPT-2 byte-level encoding
        var words: [String] = []
        var currentWord = ""
        
        for char in text {
            // Convert character to bytes then to GPT-2 unicode
            let bytes = Array(String(char).utf8)
            for byte in bytes {
                if let encoded = BPETokenizer.byteEncoder[byte] {
                    currentWord.append(encoded)
                }
            }
            
            // Simple word boundary detection (spaces, punctuation)
            if char.isWhitespace || char.isPunctuation {
                if !currentWord.isEmpty {
                    words.append(currentWord)
                    currentWord = ""
                }
            }
        }
        
        if !currentWord.isEmpty {
            words.append(currentWord)
        }
        
        return words
    }
    
    /// Apply BPE merges to a word
    private func bpe(_ word: String) -> [String] {
        if word.isEmpty { return [] }
        
        // Start with individual characters as tokens
        var tokens = word.map { String($0) }
        
        // Iteratively merge the highest priority pair
        while tokens.count > 1 {
            // Find the best pair to merge
            var bestPair: (Int, String, String)? = nil
            var bestRank = Int.max
            
            for i in 0..<(tokens.count - 1) {
                let pair = "\(tokens[i]) \(tokens[i+1])"
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = (i, tokens[i], tokens[i+1])
                }
            }
            
            // If no merge found, we're done
            guard let (index, first, second) = bestPair else { break }
            
            // Apply the merge
            tokens[index] = first + second
            tokens.remove(at: index + 1)
        }
        
        return tokens
    }
    
    // MARK: - Decoding
    
    /// Decode a single token ID to string
    public func decode(tokenId: Int) -> String {
        guard let token = reverseVocab[tokenId] else { return "" }
        
        // Check if it's a special token
        if specialTokens.values.contains(tokenId) {
            return ""  // Don't include special tokens in output
        }
        
        // Convert GPT-2 byte-level encoding back to text
        return decodeToken(token)
    }
    
    /// Decode a sequence of token IDs to text
    public func decode(tokenIds: [Int]) -> String {
        var result = ""
        for id in tokenIds {
            result += decode(tokenId: id)
        }
        return result
    }
    
    /// Decode a single BPE token to text
    private func decodeToken(_ token: String) -> String {
        var bytes: [UInt8] = []
        for char in token {
            if let byte = BPETokenizer.byteDecoder[char] {
                bytes.append(byte)
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }
    
    // MARK: - Byte Encoding (GPT-2 style)
    
    /// Create byte-to-unicode mapping (GPT-2 style)
    private static func createByteEncoder() -> [UInt8: Character] {
        var encoder: [UInt8: Character] = [:]
        var n = 0
        
        // Printable ASCII and extended latin
        for b in 33...126 {
            encoder[UInt8(b)] = Character(UnicodeScalar(b)!)
        }
        for b in 161...172 {
            encoder[UInt8(b)] = Character(UnicodeScalar(b)!)
        }
        for b in 174...255 {
            encoder[UInt8(b)] = Character(UnicodeScalar(b)!)
        }
        
        // Map remaining bytes to Unicode private area
        for b: UInt8 in 0...255 {
            if encoder[b] == nil {
                encoder[b] = Character(UnicodeScalar(256 + n)!)
                n += 1
            }
        }
        
        return encoder
    }
    
    /// Create unicode-to-byte mapping (reverse of byte encoder)
    private static func createByteDecoder() -> [Character: UInt8] {
        var decoder: [Character: UInt8] = [:]
        for (byte, char) in byteEncoder {
            decoder[char] = byte
        }
        return decoder
    }
}
