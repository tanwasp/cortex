import SwiftUI
import CoreML
import CoreImage
import Combine

@MainActor
final class ViewModel: ObservableObject {
    @Published var resultText = ""
    @Published var isRunning = false

    private var model: MLModel?
    private var captureTask: Task<Void, Never>?
    private let captureQueue = DispatchQueue(label: "screen.capture.queue")

    func prepare() async {
        requestScreenRecordingAccess()
        await loadModelIfNeeded()
    }

    func toggleAnalysis() {
        if isRunning {
            captureTask?.cancel()
            captureTask = nil
            isRunning = false
            return
        }

        isRunning = true
        captureTask = Task {
            while !Task.isCancelled {
                await runOneInference()
                try? await Task.sleep(nanoseconds: 10_000_000_000) // 10 s cadence
            }
        }
    }

    private func requestScreenRecordingAccess() {
        CGPreflightScreenCaptureAccess()
        CGRequestScreenCaptureAccess()
    }

    // MARK: - Model management
    private func loadModelIfNeeded() async {
        guard model == nil else { return }
        guard let url = Bundle.main.url(forResource: "fastvithd", withExtension: "mlpackage") else {
            resultText = "Model missing from bundle."
            return
        }

        do {
            model = try MLModel(contentsOf: url)
            inspectModel()
        } catch {
            resultText = "Model load failed: \(error.localizedDescription)"
        }
    }

    private func inspectModel() {
        guard let model else { return }
        print("Inputs:")
        model.modelDescription.inputDescriptionsByName.forEach { name, desc in
            print(" • \(name): \(desc.type) shape=\(desc.multiArrayConstraint?.shape ?? [])")
        }
        print("Outputs:")
        model.modelDescription.outputDescriptionsByName.forEach { name, desc in
            print(" • \(name): type=\(desc.type) shape=\(desc.multiArrayConstraint?.shape ?? [])")
        }
    }

    // MARK: - Inference pipeline
    private func runOneInference() async {
        guard let model else { return }

        guard let cgImage = captureMainDisplay() else {
            await MainActor.run { resultText = "Screen capture failed." }
            return
        }

        guard let pixelBuffer = makePixelBuffer(from: cgImage, width: 224, height: 224) else {
            await MainActor.run { resultText = "Pixel buffer conversion failed." }
            return
        }

        let prompt = "Describe the user's current on-screen activity as a short sentence."

        // Update these names after reading inspectModel() logs
        let imageFeatureName = "image"     // placeholder
        let textFeatureName = "prompt"     // placeholder
        let outputFeatureName = "tokens"   // placeholder

        do {
            let features: [String: MLFeatureValue] = [
                imageFeatureName: MLFeatureValue(pixelBuffer: pixelBuffer),
                textFeatureName: MLFeatureValue(string: prompt)
            ]
            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            let prediction = try model.prediction(from: provider)

            if let tokenArray = prediction.featureValue(for: outputFeatureName)?.multiArrayValue {
                let decoded = try decodeTokens(from: tokenArray)
                await MainActor.run { resultText = decoded }
            } else if let raw = prediction.featureValue(for: outputFeatureName)?.stringValue {
                await MainActor.run { resultText = raw }
            } else {
                await MainActor.run { resultText = "Unexpected model output." }
            }
        } catch {
            await MainActor.run { resultText = "Inference error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Screen capture helpers
    private func captureMainDisplay() -> CGImage? {
        CGDisplayCreateImage(CGMainDisplayID())
    }

    private func makePixelBuffer(from image: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attrs, &buffer) == kCVReturnSuccess,
              let pixelBuffer = buffer
        else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let context = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)
        else { return nil }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelBuffer
    }

    // MARK: - Token decoding placeholder
    private func decodeTokens(from array: MLMultiArray) throws -> String {
        let ids: [Int] = try array.toIntArray()
        guard let vocab = TokenDecoder.shared else {
            return ids.map(String.init).joined(separator: " ")
        }
        return vocab.decode(ids: ids)
    }
}

private extension MLMultiArray {
    func toIntArray() throws -> [Int] {
        switch dataType {
        case .int32, .int64:
            return (0..<count).map { Int(truncating: self[$0]) }
        case .double, .float32:
            return (0..<count).map { Int(self[$0].doubleValue.rounded()) }
        default:
            throw NSError(domain: "Tokenizer", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unsupported MLMultiArray type"])
        }
    }
}

// Simple vocab loader placeholder
private final class TokenDecoder {
    static let shared: TokenDecoder? = {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String]
        else { return nil }

        var map = [Int: String]()
        dict.forEach { key, value in
            if let id = Int(key) { map[id] = value }
        }
        return TokenDecoder(map: map)
    }()

    private let idToToken: [Int: String]

    private init(map: [Int: String]) {
        self.idToToken = map
    }

    func decode(ids: [Int]) -> String {
        let raw = ids.compactMap { idToToken[$0] }
        let joined = raw.joined(separator: " ")
        return joined
            .replacingOccurrences(of: "Ġ", with: " ")
            .replacingOccurrences(of: "▁", with: " ")
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}