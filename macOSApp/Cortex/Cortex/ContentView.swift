//
//  ContentView.swift
//  Cortex
//
//  Created by Tanish Pradhan Wong Ah Sui on 10/11/25.
//
//  This file defines the main user interface (UI) of the Cortex app using SwiftUI.
//  It displays a text area showing the analysis result and a button to start/stop the screen analysis.
//  SwiftUI views are structs that describe what the UI should look like, and the system handles rendering.
//  This view updates automatically when the shared ViewModel data changes.
//

import SwiftUI  // Imports SwiftUI for building the UI

struct ContentView: View {  // Defines a struct that conforms to View (a protocol for UI components). Views are immutable and describe the UI.
    @EnvironmentObject private var viewModel: ViewModel  // Accesses the shared ViewModel from the environment (passed from the app). 'private' restricts access.

    var body: some View {  // 'body' is a computed property that returns the view's content. 'some View' means it returns a type that acts like a View.
        VStack(spacing: 16) {
            // Current status section
            VStack(alignment: .leading, spacing: 8) {
                Text("Current Activity")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                
                Text(viewModel.resultText.isEmpty ? "Waiting..." : viewModel.resultText)
                    .font(.body)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding()
            .background(.thinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            
            // Activity log section
            VStack(alignment: .leading, spacing: 8) {
                Text("Activity Log")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                
                if viewModel.activityLog.isEmpty {
                    Text("No activity recorded yet.")
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 20)
                } else {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 8) {
                            ForEach(viewModel.activityLog) { entry in
                                HStack(alignment: .top, spacing: 8) {
                                    Text(entry.formattedTime)
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.secondary)
                                    Text(entry.description)
                                        .font(.callout)
                                }
                                Divider()
                            }
                        }
                    }
                    .frame(maxHeight: 200)
                }
            }
            .padding()
            .background(.thinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            
            // Control button
            HStack {
                Button(viewModel.isRunning ? "Stop Analysis" : "Start Analysis") {
                    viewModel.toggleAnalysis()
                }
                .buttonStyle(.borderedProminent)
                .tint(viewModel.isRunning ? .red : .blue)
                
                if viewModel.isRunning {
                    ProgressView()
                        .scaleEffect(0.7)
                }
            }
        }
        .padding(24)
        .frame(minWidth: 500, minHeight: 400)
        .task {
            await viewModel.prepare()
        }
    }
}

#Preview {  // This is for Xcode's preview feature, allowing you to see the UI without running the app.
    ContentView()  // Creates an instance of ContentView for preview.
        .environmentObject(ViewModel())  // Provides a mock ViewModel for the preview.
}
