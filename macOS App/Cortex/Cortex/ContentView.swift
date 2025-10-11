//
//  ContentView.swift
//  Cortex
//
//  Created by Tanish Pradhan Wong Ah Sui on 10/11/25.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var viewModel: ViewModel

    var body: some View {
        VStack(spacing: 24) {
            Text(viewModel.resultText.isEmpty ? "Idle" : viewModel.resultText)
                .font(.title3)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(.thinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))

            Button(viewModel.isRunning ? "Stop Analysis" : "Start Analysis") {
                viewModel.toggleAnalysis()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(32)
        .frame(minWidth: 460, minHeight: 260)
        .task {
            await viewModel.prepare()
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(ViewModel())
}
