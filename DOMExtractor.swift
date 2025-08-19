import Foundation
import OSAKit

class DOMExtractor {
    enum BrowserEngine {
        case chromium
        case webkit
        case gecko
        case other
    }
    
    enum Browser: String, CaseIterable {
        // WebKit-based browsers
        case safari = "safari"
        
        // Chromium-based browsers
        case chrome = "chrome"
        case arc = "arc"
        case edge = "edge"
        case brave = "brave"
        case opera = "opera"
        case vivaldi = "vivaldi"
        case chromium = "chromium"
        case canary = "canary"
        
        // Gecko-based browsers
        case firefox = "firefox"
        case firefoxDev = "firefox-dev"
        
        // Other browsers
        case tor = "tor"
        
        var displayName: String {
            switch self {
            case .safari: return "Safari"
            case .chrome: return "Google Chrome"
            case .arc: return "Arc"
            case .edge: return "Microsoft Edge"
            case .brave: return "Brave Browser"
            case .opera: return "Opera"
            case .vivaldi: return "Vivaldi"
            case .chromium: return "Chromium"
            case .canary: return "Google Chrome Canary"
            case .firefox: return "Firefox"
            case .firefoxDev: return "Firefox Developer Edition"
            case .tor: return "Tor Browser"
            }
        }
        
        var processName: String {
            switch self {
            case .safari: return "Safari"
            case .chrome: return "Google Chrome"
            case .arc: return "Arc"
            case .edge: return "Microsoft Edge"
            case .brave: return "Brave Browser"
            case .opera: return "Opera"
            case .vivaldi: return "Vivaldi"
            case .chromium: return "Chromium"
            case .canary: return "Google Chrome Canary"
            case .firefox: return "Firefox"
            case .firefoxDev: return "Firefox Developer Edition"
            case .tor: return "Tor Browser"
            }
        }
        
        var engine: BrowserEngine {
            switch self {
            case .safari:
                return .webkit
            case .chrome, .arc, .edge, .brave, .opera, .vivaldi, .chromium, .canary:
                return .chromium
            case .firefox, .firefoxDev:
                return .gecko
            case .tor:
                return .other
            }
        }
    }
    
    enum ExtractorError: Error {
        case invalidBrowser
        case scriptExecutionFailed
        case noResult
        case permissionRequired(String)
        
        var localizedDescription: String {
            switch self {
            case .invalidBrowser:
                return "Invalid browser. Supported browsers: \(Browser.allCases.map { $0.rawValue }.joined(separator: ", "))"
            case .scriptExecutionFailed:
                return "Failed to execute AppleScript"
            case .noResult:
                return "No result returned from browser"
            case .permissionRequired(let message):
                return message
            }
        }
    }
    
    static func extractDOM(from browserName: String) throws -> (url: String, html: String) {
        guard let browser = Browser(rawValue: browserName.lowercased()) else {
            throw ExtractorError.invalidBrowser
        }
        
        let script = createAppleScript(for: browser)
        
        guard let language = OSALanguage(forName: "AppleScript") else {
            throw ExtractorError.scriptExecutionFailed
        }
        
        let osascript = OSAScript(source: script, language: language)
        
        var error: NSDictionary?
        let result = osascript.executeAndReturnError(&error)
        
        if let error = error {
            print("AppleScript error: \(error)")
            
            // Check for specific Chrome permission error
            if let errorDescription = error["NSLocalizedDescription"] as? String,
               errorDescription.contains("Can't make application") && browser.engine == .chromium {
                let permissionMessage = """
                JavaScript execution not allowed in \(browser.displayName).
                
                To enable JavaScript execution via AppleScript:
                1. Open \(browser.displayName)
                2. Go to View menu → Developer → Allow JavaScript from Apple Events
                3. Try running the command again
                
                Alternative: Use Safari which doesn't require this permission.
                """
                throw ExtractorError.permissionRequired(permissionMessage)
            }
            
            throw ExtractorError.scriptExecutionFailed
        }
        
        guard let resultDescriptor = result,
              resultDescriptor.numberOfItems == 2,
              let url = resultDescriptor.atIndex(1)?.stringValue,
              let html = resultDescriptor.atIndex(2)?.stringValue else {
            throw ExtractorError.noResult
        }
        
        return (url: url, html: html)
    }
    
    private static func createAppleScript(for browser: Browser) -> String {
        switch browser.engine {
        case .webkit:
            return createWebKitScript(for: browser)
        case .chromium:
            return createChromiumScript(for: browser)
        case .gecko:
            return createGeckoScript(for: browser)
        case .other:
            return createOtherScript(for: browser)
        }
    }
    
    private static func createWebKitScript(for browser: Browser) -> String {
        return """
        tell application "\(browser.processName)"
            set theJS to "document.documentElement.outerHTML"
            set theURL to URL of current tab of front window
            set theHTML to do JavaScript theJS in current tab of front window
            return {theURL, theHTML}
        end tell
        """
    }
    
    private static func createChromiumScript(for browser: Browser) -> String {
        return """
        tell application "\(browser.processName)"
            set theURL to URL of active tab of front window
            set theHTML to execute front window's active tab javascript "document.documentElement.outerHTML"
            return {theURL, theHTML}
        end tell
        """
    }
    
    private static func createGeckoScript(for browser: Browser) -> String {
        // Firefox doesn't have reliable AppleScript support for DOM access
        // We'll use a different approach or return an error message
        return """
        tell application "\(browser.processName)"
            activate
            delay 0.5
            tell application "System Events"
                keystroke "u" using {command down, option down}
                delay 1
                keystroke "a" using command down
                keystroke "c" using command down
                keystroke "w" using command down
            end tell
            return {"Firefox", "DOM extraction via AppleScript not directly supported. HTML copied to clipboard."}
        end tell
        """
    }
    
    private static func createOtherScript(for browser: Browser) -> String {
        // For browsers without reliable AppleScript support
        return """
        tell application "\(browser.processName)"
            activate
        end tell
        error "Direct DOM extraction not supported for \(browser.processName). Browser has been activated."
        """
    }
}

func printUsage() {
    print("Usage: DOMExtractor <browser> | --list-browsers | --list-engines")
    print("\nSupported browsers:")
    
    let groupedBrowsers = Dictionary(grouping: DOMExtractor.Browser.allCases) { $0.engine }
    
    for engine in [DOMExtractor.BrowserEngine.webkit, .chromium, .gecko, .other] {
        let engineName = String(describing: engine).capitalized
        print("\n  \(engineName)-based:")
        if let browsers = groupedBrowsers[engine] {
            for browser in browsers {
                print("    \(browser.rawValue) - \(browser.displayName)")
            }
        }
    }
    
    print("\nExamples:")
    print("  DOMExtractor safari")
    print("  DOMExtractor chrome")
    print("  DOMExtractor arc")
}

func listInstalledBrowsers() {
    print("Checking for installed browsers...")
    
    let installedBrowsers = DOMExtractor.Browser.allCases.filter { browser in
        return isApplicationInstalled(browser.processName)
    }
    
    if installedBrowsers.isEmpty {
        print("No supported browsers found installed.")
        return
    }
    
    print("\nInstalled browsers:")
    for browser in installedBrowsers {
        let engine = String(describing: browser.engine).capitalized
        print("  \(browser.rawValue) - \(browser.displayName) (\(engine))")
    }
}

func isApplicationInstalled(_ appName: String) -> Bool {
    let script = """
    tell application "System Events"
        return exists application "\(appName)"
    end tell
    """
    
    guard let language = OSALanguage(forName: "AppleScript") else { return false }
    let osascript = OSAScript(source: script, language: language)
    
    var error: NSDictionary?
    let result = osascript.executeAndReturnError(&error)
    
    return error == nil && result?.booleanValue == true
}

func main() {
    let arguments = CommandLine.arguments
    
    if arguments.count == 2 {
        let option = arguments[1]
        
        switch option {
        case "--list-browsers", "-l":
            listInstalledBrowsers()
            return
        case "--list-engines", "-e":
            print("Supported browser engines:")
            print("  WebKit - Safari and WebKit-based browsers")
            print("  Chromium - Chrome, Arc, Edge, Brave, Opera, Vivaldi, etc.")
            print("  Gecko - Firefox and Firefox-based browsers")
            print("  Other - Tor Browser and other specialized browsers")
            return
        case "--help", "-h":
            printUsage()
            return
        default:
            break
        }
        
        let browserName = option
        
        do {
            let result = try DOMExtractor.extractDOM(from: browserName)
            print("URL: \(result.url)")
            print("HTML:")
            print(result.html)
        } catch let error as DOMExtractor.ExtractorError {
            print("Error: \(error.localizedDescription)")
            exit(1)
        } catch {
            print("Unexpected error: \(error.localizedDescription)")
            exit(1)
        }
    } else {
        printUsage()
        exit(1)
    }
}

main()