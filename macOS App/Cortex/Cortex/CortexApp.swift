//
//  CortexApp.swift
//  Cortex
//
//  Created by Tanish Pradhan Wong Ah Sui on 10/11/25.
//

import SwiftUI

@main
struct CortexApp: App {
    @StateObject private var viewModel = ViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
        }
    }
}
