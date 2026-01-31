//
//  ViewModel.swift
//  Cortex
//
//  Created by Tanish Pradhan Wong Ah Sui on 10/11/25.
//
//  This file contains the ViewModel, which manages the app's logic and state.
//  It handles loading models, capturing the screen, running inference, and updating the UI.
//  Uses ObservableObject to notify the UI of changes automatically.
//

import SwiftUI  // For UI-related types
import CoreML  // For running ML models
import CoreImage  // For image processing
import Combine  // For reactive programming (though not heavily used)
import MLX  // For LLM

@MainActor  // Ensures all code runs on the main thread (safe for UI updates)
final class ViewModel: ObservableObject {  // Class that can notify UI of changes
    @Published var resultText = ""  // Current analysis result
    @Published var activityLog: [ActivityEntry] = []  // Log of all activity entries
    @Published var isRunning = false  // Another published property
    
    private var visionModel: fastvithd?  // Optional: the vision ML model
    private var llmEngine = LLMEngine()  // Instance of LLM engine
    private var captureTask: Task<Void, Never>?  // Optional: background task for analysis
    
    // Entry in the activity log
    struct ActivityEntry: Identifiable {
        let id = UUID()
        let timestamp: Date
        let description: String
        
        var formattedTime: String {
            let formatter = DateFormatter()
            formatter.dateFormat = "HH:mm:ss"
            return formatter.string(from: timestamp)
        }
    }
    
    func prepare() async {  // Async function: sets up permissions and models
        requestScreenRecordingAccess()  // Ask for screen capture permission
        await loadModels()  // Load ML models
    }
    
    func toggleAnalysis() {  // Toggles start/stop of analysis
        if isRunning {  // If running
            captureTask?.cancel()  // Cancel the task
            captureTask = nil
            isRunning = false
        } else {  // If not running
            isRunning = true
            captureTask = Task {  // Create a new async task
                while !Task.isCancelled {  // Loop until cancelled
                    await runOneInference()  // Run inference
                    try? await Task.sleep(nanoseconds: 10_000_000_000)  // Wait 10 seconds
                }
            }
        }
    }
    
    private func requestScreenRecordingAccess() {  // Requests macOS permission for screen capture
        CGPreflightScreenCaptureAccess()  // Check if already granted
        CGRequestScreenCaptureAccess()  // Request if needed
    }
    
    // MARK: - Model management
    private func loadModels() async {  // Loads vision and LLM models
        // Load Vision Model (Core ML)
        if visionModel == nil {  // If not loaded
            do {
                let configuration = MLModelConfiguration()  // Config for model
                configuration.computeUnits = .all  // Use all available compute (CPU/GPU)
                visionModel = try fastvithd(configuration: configuration)  // Load model (may throw error)
            } catch {
                resultText = "Vision Model load failed: \(error.localizedDescription)"  // Set error message
                return
            }
        }
        
        // Load LLM (MLX)
        // Model files are copied directly to Resources folder (not in a subfolder)
        if let resourcePath = Bundle.main.resourceURL {
            do {
                try await llmEngine.load(modelPath: resourcePath)  // Load LLM from Resources
            } catch {
                resultText = "LLM load failed: \(error.localizedDescription)"  // Error
            }
        } else {
            print("Bundle resource URL not found")  // Debug print
        }
    }
    
    // MARK: - Inference pipeline
    private func runOneInference() async {  // Runs one full inference cycle
        guard let visionModel else { return }  // Guard: skip if no vision model
        
        guard let cgImage = captureMainDisplay() else {  // Capture screen image
            await MainActor.run { resultText = "Screen capture failed." }  // Update UI on main thread
            return
        }
        
        do {
            let inputArray = try createInputArray(from: cgImage)  // Prepare image for model
            let prediction = try visionModel.prediction(images: inputArray)  // Run vision model
            
            // Improved prompt for concise description
            let prompt = "<|im_start|>user\n<image>\nBriefly describe what the user is doing on their computer in one sentence.<|im_end|>\n<|im_start|>assistant\n"
            let text = await llmEngine.generate(imageFeatures: prediction.image_features, prompt: prompt, maxTokens: 50)
            
            await MainActor.run {
                resultText = text  // Update current result
                // Append to activity log (keep last 50 entries)
                let entry = ActivityEntry(timestamp: Date(), description: text)
                activityLog.insert(entry, at: 0)  // Add to top
                if activityLog.count > 50 {
                    activityLog.removeLast()
                }
            }
        } catch {
            await MainActor.run { resultText = "Inference error: \(error.localizedDescription)" }  // Error
        }
    }
    
    // MARK: - Screen capture helpers
    private func captureMainDisplay() -> CGImage? {  // Captures the main screen
        CGDisplayCreateImage(CGMainDisplayID())  // macOS API to capture display
    }
}

// MARK: - Core ML helpers
private extension ViewModel {  // Extension: adds methods to ViewModel privately
    func createInputArray(from image: CGImage, targetSize: Int = 1024) throws -> MLMultiArray {  // Converts CGImage to ML array
        let shape: [NSNumber] = [1, 3, targetSize, targetSize].map { NSNumber(value: $0) }  // Shape: [batch, channels, height, width]
        let array = try MLMultiArray(shape: shape, dataType: .float32)  // Create array
        
        var pixelBuffer = [UInt8](repeating: 0, count: targetSize * targetSize * 4)  // Buffer for pixels
        let bytesPerRow = targetSize * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()  // RGB color space
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue  // Bitmap info
        
        let rendered = pixelBuffer.withUnsafeMutableBytes { buffer -> Bool in  // Render image to buffer
            guard let baseAddress = buffer.baseAddress else { return false }
            guard let context = CGContext(data: baseAddress, width: targetSize, height: targetSize, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else { return false }
            context.interpolationQuality = .high  // High quality scaling
            context.draw(image, in: CGRect(x: 0, y: 0, width: targetSize, height: targetSize))  // Draw image
            return true
        }
        
        guard rendered else {  // If rendering failed
            throw NSError(domain: "ModelInput", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to rasterize image"])
        }
        
        let floatPointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)  // Access array data
        let strides = array.strides.map { $0.intValue }  // Strides for indexing
        let batchStride = strides[0]
        let channelStride = strides[1]
        let rowStride = strides[2]
        let columnStride = strides[3]
        
        for y in 0..<targetSize {  // Loop over pixels
            for x in 0..<targetSize {
                let pixelIndex = (y * targetSize + x) * 4  // Index in buffer
                let r = Float(pixelBuffer[pixelIndex]) / 255.0  // Red, normalize to 0-1
                let g = Float(pixelBuffer[pixelIndex + 1]) / 255.0  // Green
                let b = Float(pixelBuffer[pixelIndex + 2]) / 255.0  // Blue
                
                let baseIndex = 0 * batchStride + y * rowStride + x * columnStride  // Index in array
                floatPointer[baseIndex + channelStride * 0] = r  // Set red
                floatPointer[baseIndex + channelStride * 1] = g  // Green
                floatPointer[baseIndex + channelStride * 2] = b  // Blue
            }
        }
        
        return array  // Return prepared array
    }
    
    func summarizeFeatureVector(_ array: MLMultiArray, maxElements: Int = 6) -> String {  // Debug helper (formats first few values)
        let count = min(maxElements, array.count)
        let floats = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        var values: [String] = []
        values.reserveCapacity(count)
        
        for index in 0..<count {
            values.append(String(format: "%.3f", floats[index]))  // Format as string
        }
        
        return "Feature vector (first \(count) values): \(values.joined(separator: ", "))"  // Return summary
    }
}
