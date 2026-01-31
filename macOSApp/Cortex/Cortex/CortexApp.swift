//
//  CortexApp.swift
//  Cortex
//
//  Created by Tanish Pradhan Wong Ah Sui on 10/11/25.
//
//  This file is the main entry point of the Cortex macOS app.
//  It defines the app structure using SwiftUI, which is Apple's framework for building user interfaces.
//  The app is currently a Vision-Language Model (VLM) that analyzes screen content and describes it in text. It is to be expanded to be a fully flexibly AI powered accountability app to nudge you and guide you towards your goals and better decisions on your device.
//  This file sets up the main window and provides shared data (ViewModel) to the UI components.
//

import SwiftUI  // Imports the SwiftUI framework, which allows building UIs declaratively (like describing what you want instead of how to draw it)

@main  // This attribute marks this struct as the starting point of the app, similar to a main() function in other languages
struct CortexApp: App {  // Defines a struct (a value type that groups data and functions) that conforms to the App protocol (a set of rules for app behavior)
    @StateObject private var viewModel = ViewModel()  // Creates an instance of ViewModel (shared data/logic) and marks it as a state object (it persists and updates the UI when changed). 'private' means only this struct can access it.

    var body: some Scene {  // 'body' is a computed property (calculated on access) that returns the app's content. 'some Scene' means it returns something that behaves like a Scene (a container for windows/menus).
        WindowGroup {  // WindowGroup creates a standard app window that can have multiple instances
            ContentView()  // ContentView is the main UI view (defined in another file). This places it inside the window.
                .environmentObject(viewModel)  // Passes the viewModel to ContentView and its child views, so they can access shared data without passing it manually.
        }
    }
}
